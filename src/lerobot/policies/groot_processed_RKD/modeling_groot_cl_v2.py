import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, HfHubHTTPError, RepositoryNotFoundError
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor

from lerobot.policies.groot_cl.groot_n1 import GR00TN15
from lerobot.policies.groot_cl.modeling_groot import GrootPolicy
from lerobot.policies.groot_processed_RKD.configuration_groot_cl_v2 import GrootCLv2Config

logger = logging.getLogger(__name__)

_GROOT_MODEL_PREFIX = "_groot_model."
_SAFETENSORS_FILENAME = "model.safetensors"

# GR00TN15 backbone (Eagle2.5-VL, Qwen3-2.3B LLM) output hidden size.
# Confirmed at runtime: backbone_features last dim = 2048.
BACKBONE_FEAT_DIM = 2048


def _build_token_mask(mask: Tensor | None, x: Tensor) -> Tensor:
    """Return a float mask of shape (B, T, 1)."""
    if mask is None:
        return torch.ones(
            (x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype
        )
    return mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)


def _masked_mean_pool(x: Tensor, mask: Tensor | None) -> Tensor:
    """Masked mean over token axis: (B, T, D) -> (B, D)."""
    m = _build_token_mask(mask, x)
    denom = m.sum(dim=1).clamp_min(1e-6)
    return (x * m).sum(dim=1) / denom


def _attention_pool(x: Tensor, mask: Tensor | None) -> Tensor:
    """Masked attention-style pooling without extra head: (B, T, D) -> (B, D)."""
    # Token saliency score: ||x_t||^2 / sqrt(D)
    d = x.shape[-1]
    logits = (x * x).sum(dim=-1) / (d ** 0.5)  # (B, T)
    if mask is not None:
        valid = mask.to(device=x.device, dtype=torch.bool)
        logits = logits.masked_fill(~valid, -1e9)
    attn = F.softmax(logits, dim=-1).unsqueeze(-1)  # (B, T, 1)
    return (x * attn).sum(dim=1)


class GrootCLv2Policy(GrootPolicy):
    """GR00T-CL v2: Action-guided Relational Knowledge Distillation for VLM finetuning.

    Phase 1: VLM backbone + vision tower frozen.
             Action expert (MultiEmbodimentActionEncoder + DiT) trained via
             per-joint weighted flow matching loss (wrist joint index 6 gets 5x weight).

    Phase 2: Action expert frozen (teacher).
             Processed backbone token sequence is pooled directly into the RKD
             student latent without an extra projection head.

             RKD Loss (CVPR 2019):
               z_a = L2_norm( pool( ActionEncoder(action, t=999) ) )   ← teacher
               z_v = L2_norm( pool( processed VLMBackbone(obs) ) )     ← student
               S_a = z_a @ z_a.T,  P_a = softmax(S_a / τ_act)
               S_v = z_v @ z_v.T
               L_RKD = KL( P_a || softmax(S_v / τ_vlm) )

             → VLM embedding space learns to mirror action space's pairwise structure.
             → Same visual state with different action directions → different VLM embeddings.
    """

    name = "groot_processed_rkd"
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
            pretrained_dir = self._resolve_groot_pretrained_dir()
            state_dict = self._load_groot_state_dict(pretrained_dir)
            first_key = next(iter(state_dict), "")
            if not first_key.startswith(_GROOT_MODEL_PREFIX) and not any(
                k.startswith("backbone.") or k.startswith("action_head.") for k in state_dict
            ):
                raise ValueError(
                    f"No compatible keys found in {pretrained_dir}. "
                    f"Expected '{_GROOT_MODEL_PREFIX}*' or GR00T bare keys."
                )
            groot_state = self._normalize_groot_state_dict(state_dict)
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

    def _resolve_groot_pretrained_dir(self) -> Path:
        pretrained_path = self.config.groot_pretrained_path
        if not pretrained_path:
            raise ValueError("groot_pretrained_path is empty.")

        candidate = Path(pretrained_path)
        if candidate.is_file():
            return candidate.parent
        if candidate.is_dir():
            return candidate

        try:
            downloaded = snapshot_download(
                repo_id=pretrained_path,
                repo_type="model",
                allow_patterns=[
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "model-*.safetensors",
                ],
            )
            logger.info(
                "Downloaded groot_pretrained_path from HuggingFace repo '%s' to %s",
                pretrained_path,
                downloaded,
            )
            return Path(downloaded)
        except (HFValidationError, RepositoryNotFoundError, HfHubHTTPError) as e:
            raise FileNotFoundError(
                f"groot_pretrained_path is not a local file/dir and HF download failed for repo id: {pretrained_path}"
            ) from e

    def _resolve_groot_pretrained_files(self, pretrained_dir: Path) -> list[Path]:
        direct_file = pretrained_dir / _SAFETENSORS_FILENAME
        if direct_file.exists():
            return [direct_file]

        index_file = pretrained_dir / "model.safetensors.index.json"
        if index_file.exists():
            import json

            index_data = json.loads(index_file.read_text())
            weight_map = index_data.get("weight_map", {})
            shard_names = sorted(set(weight_map.values()))
            if not shard_names:
                raise FileNotFoundError(
                    f"Checkpoint index exists but weight_map is empty: {index_file}"
                )
            shard_files = [pretrained_dir / name for name in shard_names]
            missing = [path for path in shard_files if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Checkpoint shards referenced by {index_file} are missing: {missing}"
                )
            return shard_files

        shard_files = sorted(pretrained_dir.glob("*.safetensors"))
        if shard_files:
            return shard_files

        raise FileNotFoundError(
            f"No safetensors checkpoint found in {pretrained_dir}. "
            "Expected model.safetensors, model.safetensors.index.json, or sharded *.safetensors files."
        )

    def _load_groot_state_dict(self, pretrained_dir: Path) -> dict[str, Tensor]:
        state_dict: dict[str, Tensor] = {}
        for shard_file in self._resolve_groot_pretrained_files(pretrained_dir):
            shard_state = safetensors_load_file(str(shard_file))
            state_dict.update(shard_state)
        return state_dict

    @staticmethod
    def _normalize_groot_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        groot_state = {
            k[len(_GROOT_MODEL_PREFIX):]: v
            for k, v in state_dict.items()
            if k.startswith(_GROOT_MODEL_PREFIX)
        }
        if groot_state:
            return groot_state
        if any(k.startswith("backbone.") or k.startswith("action_head.") for k in state_dict):
            return state_dict
        return {}

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
            logger.info(
                "[Phase 1] VLM backbone frozen. Action expert (encoder + DiT) trainable. "
                "Joint FM weights: %s", list(self._joint_weights.cpu().numpy())
            )

        elif phase == "phase2":
            mode = self.config.cl_v2_trainable_mode
            if mode == "default":
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
                logger.info(
                    "[Phase 2][default] Action expert frozen (teacher). "
                    "VLM backbone trainable (student)."
                )
            elif mode == "processed_only":
                # Match groot_processed_mgd processed_only semantics (without extra heads):
                # freeze all, then train only processed feature modules.
                for p in self.parameters():
                    p.requires_grad_(False)
                self._groot_model.action_head.vlln.requires_grad_(True)
                self._groot_model.action_head.vl_self_attention.requires_grad_(True)
                logger.info(
                    "[Phase 2][processed_only] Freeze all, train only "
                    "action_head.vlln + action_head.vl_self_attention."
                )
            elif mode == "dit_core_only":
                # Match groot_processed_mgd behavior:
                # freeze everything, then train only DiT core.
                for p in self.parameters():
                    p.requires_grad_(False)
                self._groot_model.action_head.model.requires_grad_(True)
                logger.info(
                    "[Phase 2][dit_core_only] Freeze all, train only action_head.model (DiT core)."
                )
            else:
                raise ValueError(
                    f"Unknown cl_v2_trainable_mode: {mode!r}. "
                    "Must be 'default', 'processed_only', or 'dit_core_only'."
                )
            self._log_trainable_setup()

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
            ActionEncoder(action, t=999) → (B, T, D_a) → flatten → L2 norm
            - 장점: action encoder가 만든 per-step latent 전체를 보존.
            - 단점: temporal ordering은 flatten에 남지만, step 간 관계는
                    similarity matrix에서만 간접적으로 반영됨.

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
            emb_id = torch.zeros(B, dtype=torch.long, device=device)
            t_clean = torch.full((B,), 999, dtype=torch.long, device=device)
            action_enc_out = self._groot_model.action_head.action_encoder(
                action_traj, t_clean, emb_id,
            )
            flat = action_enc_out.flatten(1)                                  # (B, T*D_a)
            return F.normalize(flat, dim=-1)                                 # (B, T*D_a)

        else:
            raise ValueError(
                f"Unknown cl_v2_action_repr: {repr_mode!r}. "
                "Must be 'mean_pool' or 'raw_flatten'."
            )

    def _forward_phase2(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Phase 2: RKD loss — VLM embedding space mirrors action encoder structure.

        Teacher (frozen): ActionEncoder(gt_action, t=999) → action similarity matrix S_a → P_a
        Student (trainable): processed backbone_features aggregated by
                             cl_v2_student_repr → VLM similarity matrix S_v
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

        # ── Student: processed backbone tokens → selected aggregation → L2 norm ───
        backbone_features = outputs["backbone_features"]         # (B, T_seq, D_vlm=2048)
        backbone_mask = outputs.get("backbone_attention_mask")   # (B, T_seq) or None
        student_mode = self.config.cl_v2_student_repr
        x = backbone_features.float()
        if student_mode == "flatten":
            student_vec = x.flatten(1)                           # (B, T_seq*D_vlm)
        elif student_mode == "mask_mean":
            student_vec = _masked_mean_pool(x, backbone_mask)    # (B, D_vlm)
        elif student_mode == "attention_pooling":
            student_vec = _attention_pool(x, backbone_mask)      # (B, D_vlm)
        else:
            raise ValueError(
                f"Unknown cl_v2_student_repr: {student_mode!r}. "
                "Must be 'flatten', 'mask_mean', or 'attention_pooling'."
            )
        vlm_z = F.normalize(student_vec, dim=-1)

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

    def _log_trainable_setup(self) -> None:
        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        trainable_params = sum(p.numel() for _, p in self.named_parameters() if p.requires_grad)
        logger.info(
            "RKD trainable parameters: tensors=%d params=%d mode=%s",
            len(trainable_names),
            trainable_params,
            self.config.cl_v2_trainable_mode,
        )
        if trainable_names:
            preview_n = min(10, len(trainable_names))
            logger.info(
                "RKD trainable parameter sample (%d/%d): %s",
                preview_n,
                len(trainable_names),
                ", ".join(trainable_names[:preview_n]),
            )
