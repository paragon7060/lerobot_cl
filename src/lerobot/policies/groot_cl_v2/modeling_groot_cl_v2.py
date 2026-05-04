import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor

from lerobot.policies.groot_cl.groot_n1 import GR00TN15
from lerobot.policies.groot_cl.modeling_groot import GrootPolicy
from lerobot.policies.groot_cl_v2.configuration_groot_cl_v2 import GrootCLv2Config

logger = logging.getLogger(__name__)

_GROOT_MODEL_PREFIX = "_groot_model."
_SAFETENSORS_FILENAME = "model.safetensors"

# GR00TN15 backbone (Eagle2.5-VL, Qwen3-2.3B LLM) output hidden size.
# Confirmed at runtime: backbone_features last dim = 2048
BACKBONE_FEAT_DIM = 2048


class VLMProjector(nn.Module):
    """VLM backbone features → L2-normalized RKD latent.

    Masked mean pool over token sequence → 2-layer MLP → L2 normalize.
    Used in Phase 2 RKD to compute the student similarity matrix S_v.
    """

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x:    (B, T_seq, D_vlm) backbone features
            mask: (B, T_seq) attention mask — 1 for valid tokens, 0 for padding
        Returns:
            (B, latent_dim) L2-normalized latent
        """
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()                            # (B, T_seq, 1)
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)  # (B, D_vlm)
        else:
            pooled = x.mean(dim=1)                                         # (B, D_vlm)
        return F.normalize(self.net(pooled), dim=-1)                       # (B, latent_dim)


class GrootCLv2Policy(GrootPolicy):
    """GR00T-CL v2: Action-guided Relational Knowledge Distillation for VLM finetuning.

    Phase 1: VLM backbone + vision tower frozen.
             Action expert (MultiEmbodimentActionEncoder + DiT) trained via
             per-joint weighted flow matching loss (wrist joint index 6 gets 5x weight).

    Phase 2: Action expert frozen (teacher).
             VLM backbone + VLMProjector trained via RKD loss (student).

             RKD Loss (CVPR 2019):
               z_a = L2_norm( pool( ActionEncoder(action, t=999) ) )   ← teacher
               z_v = VLMProjector( pool( VLMBackbone(obs) ) )          ← student
               S_a = z_a @ z_a.T,  P_a = softmax(S_a / τ_act)
               S_v = z_v @ z_v.T
               L_RKD = KL( P_a || softmax(S_v / τ_vlm) )

             → VLM embedding space learns to mirror action space's pairwise structure.
             → Same visual state with different action directions → different VLM embeddings.
    """

    name = "groot_cl_v2"
    config_class = GrootCLv2Config

    def __init__(self, config: GrootCLv2Config, **kwargs):
        # GrootPolicy.__init__ calls _create_groot_model() → sets self._groot_model
        super().__init__(config, **kwargs)

        # ── Joint weights for Phase 1 weighted FM loss ─────────────────────────────
        # Padding dims (index >= len(joint_fm_weights)) get weight=0.0
        w = list(config.joint_fm_weights)
        max_dim = config.max_action_dim
        if len(w) < max_dim:
            w = w + [0.0] * (max_dim - len(w))
        elif len(w) > max_dim:
            logger.warning(
                "joint_fm_weights has %d entries but max_action_dim=%d. Truncating.",
                len(w), max_dim,
            )
            w = w[:max_dim]
        self.register_buffer("_joint_weights", torch.tensor(w, dtype=torch.float32), persistent=False)

        # ── Phase 2: VLM Projector ──────────────────────────────────────────────────
        self.vlm_projector = VLMProjector(
            in_dim=BACKBONE_FEAT_DIM,
            hidden_dim=config.cl_v2_hidden_dim,   # 512
            latent_dim=config.cl_v2_latent_dim,   # 256
        )

        # Apply phase configuration (freeze/unfreeze appropriate components)
        self.set_phase(config.cl_v2_phase)

    def _create_groot_model(self):
        """Create GR00TN15; optionally load weights from a GrootPolicy checkpoint."""
        self._handle_flash_attention_compatibility()

        model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target=self.config.lora_target,
        )
        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        if self.config.groot_pretrained_path:
            safetensors_path = os.path.join(self.config.groot_pretrained_path, _SAFETENSORS_FILENAME)
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(
                    f"groot_pretrained_path 설정됨, 그러나 파일 없음: {safetensors_path}"
                )
            full_state = safetensors_load_file(safetensors_path)
            groot_state = {
                k[len(_GROOT_MODEL_PREFIX):]: v
                for k, v in full_state.items()
                if k.startswith(_GROOT_MODEL_PREFIX)
            }
            if not groot_state:
                raise ValueError(
                    f"{safetensors_path} 에서 '{_GROOT_MODEL_PREFIX}*' 키 없음. "
                    "GrootPolicy로 저장된 체크포인트인지 확인하세요."
                )
            missing, unexpected = model.load_state_dict(groot_state, strict=False)
            if missing:
                logger.warning("groot_pretrained_path 로드 후 missing keys (%d): %s ...",
                               len(missing), missing[:3])
            if unexpected:
                logger.warning("groot_pretrained_path 로드 후 unexpected keys (%d): %s ...",
                               len(unexpected), unexpected[:3])
            logger.info(
                "groot_pretrained_path '%s' 에서 %d keys 로드 완료.",
                self.config.groot_pretrained_path, len(groot_state),
            )

        return model

    def set_phase(self, phase: str) -> None:
        """Freeze/unfreeze model components for the given training phase."""
        self.config.cl_v2_phase = phase

        if phase == "phase1":
            # Freeze backbone (vision tower + LLM)
            self._groot_model.backbone.set_trainable_parameters(
                tune_visual=False,
                tune_llm=False,
            )
            # Unfreeze action expert
            self._groot_model.action_head.set_trainable_parameters(
                tune_projector=True,
                tune_diffusion_model=True,
            )
            # Freeze VLMProjector (not used in phase 1)
            for param in self.vlm_projector.parameters():
                param.requires_grad_(False)
            logger.info(
                "[Phase 1] VLM backbone frozen. Action expert (encoder + DiT) trainable. "
                "Joint FM weights: %s", list(self._joint_weights.cpu().numpy())
            )

        elif phase == "phase2":
            # Freeze action expert (teacher)
            self._groot_model.action_head.set_trainable_parameters(
                tune_projector=False,
                tune_diffusion_model=False,
            )
            # Unfreeze VLM backbone (student)
            self._groot_model.backbone.set_trainable_parameters(
                tune_visual=True,
                tune_llm=True,
            )
            # Unfreeze VLMProjector (student projector)
            for param in self.vlm_projector.parameters():
                param.requires_grad_(True)
            logger.info("[Phase 2] Action expert frozen (teacher). VLM backbone + VLMProjector trainable (student).")

        else:
            raise ValueError(f"Unknown phase: {phase!r}. Must be 'phase1' or 'phase2'.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.cl_v2_phase == "phase1":
            return self._forward_phase1(batch)
        elif self.config.cl_v2_phase == "phase2":
            return self._forward_phase2(batch)
        else:
            raise ValueError(f"Unknown phase: {self.config.cl_v2_phase!r}")

    def _forward_phase1(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Phase 1: weighted flow matching loss on action expert."""
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_"))
            and not (k.startswith("next.") or k == "info")
        }

        device = next(self.parameters()).device

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(
                groot_inputs,
                return_intermediate=False,
                joint_weights=self._joint_weights,
            )

        loss = outputs.get("loss")
        return loss, {"loss": loss.item(), "flow_matching_loss": loss.item()}

    @torch.no_grad()
    def _compute_action_z(self, action_traj: Tensor, B: int, device: torch.device) -> Tensor:
        """Teacher action representation → L2-normalized vector.

        두 가지 전략을 config(cl_v2_action_repr)로 선택:

        "mean_pool" (기본):
            ActionEncoder(action, t=999) → (B, T, D_a) → mean(dim=T) → L2 norm → (B, D_a=1536)
            - 장점: 학습된 action encoder feature 활용
            - 단점: per-step MLP 특성상 timestep 간 interaction 없음.
                    Mean pool로 temporal ordering 소실.

        "raw_flatten":
            raw action trajectory → flatten → (B, T*action_dim) → L2 norm
            - 장점: temporal 구조 완전 보존. 추가 파라미터 없음.
                    Wrist joint의 timestep별 회전 방향이 직접 벡터에 인코딩.
            - 단점: action_dim 중 불필요한 joint도 동등하게 포함.
                    Scale 차이가 있을 수 있어 joint별 normalization 고려 필요.

        Args:
            action_traj: (B, T, action_dim) — raw action trajectory (float32)
            B: batch size
            device: target device

        Returns:
            action_z: (B, D) L2-normalized teacher latent
        """
        repr_mode = self.config.cl_v2_action_repr

        if repr_mode == "mean_pool":
            emb_id = torch.zeros(B, dtype=torch.long, device=device)
            t_clean = torch.full((B,), 999, dtype=torch.long, device=device)
            # per-step MLP: (B, T, action_dim) → (B, T, D_a=1536)
            action_enc_out = self._groot_model.action_head.action_encoder(
                action_traj, t_clean, emb_id,
            )
            # mean pool → temporal ordering 소실, 각 timestep 동등 가중
            return F.normalize(action_enc_out.mean(dim=1), dim=-1)          # (B, 1536)

        elif repr_mode == "raw_flatten":
            # ActionEncoder를 거치지 않고 raw trajectory를 직접 flatten.
            # action_traj: (B, T=16, action_dim=32 or 8 etc.)
            # → flatten → (B, T*action_dim)
            # Temporal 순서가 vector 내에 그대로 보존됨.
            # e.g. wrist rotation 방향 차이:
            #   left  trajectory: [..., +0.1, +0.2, +0.3, ...]  @ joint 6 positions
            #   right trajectory: [..., -0.1, -0.2, -0.3, ...]  @ joint 6 positions
            # → flatten 후 cosine similarity에서 명확히 분리됨.
            flat = action_traj.flatten(1)                                    # (B, T*action_dim)
            return F.normalize(flat, dim=-1)                                 # (B, T*action_dim)

        else:
            raise ValueError(
                f"Unknown cl_v2_action_repr: {repr_mode!r}. "
                "Must be 'mean_pool' or 'raw_flatten'."
            )

    def _forward_phase2(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Phase 2: RKD loss — VLM embedding space mirrors action encoder structure.

        Teacher (frozen): ActionEncoder(gt_action, t=999) → action similarity matrix S_a → P_a
        Student (trainable): VLMBackbone(obs) → VLMProjector → VLM similarity matrix S_v
        Loss: KL( P_a || softmax(S_v / τ_vlm) )
        """
        B = batch["action"].shape[0]
        device = next(self.parameters()).device

        # ── Build groot_inputs ──────────────────────────────────────────────────────
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_"))
            and not (k.startswith("next.") or k == "info")
        }

        # ── Full forward: VLM trainable, action expert frozen ──────────────────────
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs, return_intermediate=True)
        fm_loss = outputs["loss"]

        # ── Student: VLM embedding → project → L2 normalize ───────────────────────
        backbone_features = outputs["backbone_features"]         # (B, T_seq, D_vlm=1536)
        backbone_mask = outputs.get("backbone_attention_mask")   # (B, T_seq) or None
        # Cast to float32 for projector stability
        vlm_z = self.vlm_projector(
            backbone_features.float(), backbone_mask
        )  # (B, latent_dim=256), L2 norm'd

        # ── Teacher: action representation → L2-normalized latent ────────────────
        with torch.no_grad():
            action_traj = batch["action"].to(device=device, dtype=torch.float32)  # (B, T, action_dim)
            action_z = self._compute_action_z(action_traj, B, device)

            # Teacher similarity matrix → soft target distribution
            S_a = action_z @ action_z.T                                          # (B, B)
            P_a = F.softmax(S_a / self.config.cl_v2_action_temp, dim=-1)         # (B, B)

        # ── Student similarity matrix ───────────────────────────────────────────────
        S_v = vlm_z @ vlm_z.T                                                    # (B, B)

        # ── RKD Loss: KL( P_a || P_v )  ────────────────────────────────────────────
        # F.kl_div(log_Q, P) computes Σ P*(log P - log Q) = KL(P||Q)
        # Here: log Q = log P_v, P = P_a  →  KL(P_a || P_v)
        rkd_loss = F.kl_div(
            F.log_softmax(S_v / self.config.cl_v2_vlm_temp, dim=-1),  # log P_v (student)
            P_a,                                                        # P_a (teacher, no grad)
            reduction="batchmean",
        )

        total_loss = (
            fm_loss * self.config.cl_v2_fm_loss_weight   # default 0.01 — FM quality 유지
            + rkd_loss * self.config.cl_v2_loss_weight   # default 0.1  — main RKD signal
        )
        return total_loss, {
            "loss": total_loss.item(),
            "flow_matching_loss": fm_loss.item(),
            "rkd_loss": rkd_loss.item(),
        }
