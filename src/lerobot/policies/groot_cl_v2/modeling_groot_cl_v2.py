import logging
import os

import torch
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor

from lerobot.policies.groot_cl.groot_n1 import GR00TN15
from lerobot.policies.groot_cl.modeling_groot import GrootPolicy
from lerobot.policies.groot_cl_v2.configuration_groot_cl_v2 import GrootCLv2Config

logger = logging.getLogger(__name__)

_GROOT_MODEL_PREFIX = "_groot_model."
_SAFETENSORS_FILENAME = "model.safetensors"


class GrootCLv2Policy(GrootPolicy):
    """GR00T-CL v2: Action-guided contrastive VLM finetuning.

    Phase 1: VLM backbone + vision tower frozen.
             Action expert (MultiEmbodimentActionEncoder + DiT) trained via
             per-joint weighted flow matching loss. Wrist joint (index 6) gets 3x weight.

    Phase 2 (future): Action expert frozen.
             VLM finetuned via weighted InfoNCE using clean action features as
             soft positive/negative signal.
    """

    name = "groot_cl_v2"
    config_class = GrootCLv2Config

    def __init__(self, config: GrootCLv2Config, **kwargs):
        # GrootPolicy.__init__ calls _create_groot_model() → sets self._groot_model
        super().__init__(config, **kwargs)

        # Build joint_weights tensor: user-specified weights padded with 0.0 to max_action_dim.
        # Padding dims (index >= len(joint_fm_weights)) get weight=0.0, which completely
        # excludes them from the loss — no separate action_mask application needed.
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
        # Register as non-persistent buffer: moves with model.to(device) but not saved in checkpoint.
        self.register_buffer("_joint_weights", torch.tensor(w, dtype=torch.float32), persistent=False)

        # Apply phase configuration (freeze/unfreeze appropriate components).
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
        """Freeze/unfreeze model components for the given training phase.

        Phase 1: Freeze VLM backbone (vision tower + LLM + eagle_linear).
                 Unfreeze action expert (action_encoder + DiT decoder).
        Phase 2: Freeze action expert.
                 Unfreeze VLM backbone (for contrastive finetuning).
        """
        self.config.cl_v2_phase = phase

        if phase == "phase1":
            # Freeze backbone (vision tower + LLM). eagle_linear is inside backbone.
            self._groot_model.backbone.set_trainable_parameters(
                tune_visual=False,
                tune_llm=False,
            )
            # Unfreeze action expert: projectors (action/state encoder) + diffusion model (DiT)
            self._groot_model.action_head.set_trainable_parameters(
                tune_projector=True,
                tune_diffusion_model=True,
            )
            logger.info(
                "[Phase 1] VLM backbone frozen. Action expert (encoder + DiT) trainable. "
                "Joint FM weights: %s", list(self._joint_weights.cpu().numpy())
            )

        elif phase == "phase2":
            # Freeze action expert
            self._groot_model.action_head.set_trainable_parameters(
                tune_projector=False,
                tune_diffusion_model=False,
            )
            # Unfreeze VLM backbone
            self._groot_model.backbone.set_trainable_parameters(
                tune_visual=True,
                tune_llm=True,
            )
            logger.info("[Phase 2] Action expert frozen. VLM backbone trainable.")

        else:
            raise ValueError(f"Unknown phase: {phase!r}. Must be 'phase1' or 'phase2'.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass with per-joint weighted FM loss."""
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
        loss_dict = {
            "loss": loss.item(),
            "flow_matching_loss": loss.item(),
        }
        return loss, loss_dict
