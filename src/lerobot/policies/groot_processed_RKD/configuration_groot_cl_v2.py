from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot_cl.configuration_groot import GrootConfig


@PreTrainedConfig.register_subclass("groot_processed_rkd")
@dataclass
class GrootCLv2Config(GrootConfig):
    # Match NVIDIA pretrained GR00T-N1.5-3B model's action_head_cfg.max_state_dim=64
    max_state_dim: int = 64
    """Configuration for GR00T-Processed-RKD: MGD-style processed-token RKD finetuning.

    Phase 1: Freeze VLM backbone + vision tower. Train action expert (FM loss)
             with per-joint weighting to emphasize specific joints (e.g. wrist).
    Phase 2: Freeze action expert. Finetune VLM using RKD on the processed
             backbone token sequence, matching the student granularity used by
             the MGD branch.
    """

    # Phase control
    cl_v2_phase: str = "phase1"  # "phase1" | "phase2"

    # Pretrained GrootPolicy checkpoint path (optional).
    # If set, loads _groot_model.* weights from this checkpoint instead of base_model_path.
    groot_pretrained_path: str | None = None

    # Per-joint FM loss weights (Phase 1).
    # List of floats for the actual action_dim — padding dims are auto-zeroed.
    # paragon7060/INSIGHTfixposV3 기준: 8차원, index 6 (wrist) = 3.0배 강조.
    joint_fm_weights: list = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]
    )

    # Phase 2 direct-pooled student settings.
    # `cl_v2_hidden_dim` / `cl_v2_latent_dim` are kept for compatibility with
    # the earlier projector variant, but the current student path uses the
    # processed backbone features directly without an extra head.
    cl_v2_latent_dim: int = 256
    cl_v2_hidden_dim: int = 512

    # Phase 2 loss weights
    cl_v2_loss_weight: float = 0.1       # weighted InfoNCE weight
    cl_v2_fm_loss_weight: float = 0.01   # flow matching monitoring weight in phase 2

    # Weighted InfoNCE temperatures
    cl_v2_action_temp: float = 0.1   # action feature similarity → soft label sharpness
    cl_v2_vlm_temp: float = 0.07     # VLM logit temperature

    # Phase 2 teacher action representation strategy.
    #
    # "mean_pool"    : mean( ActionEncoder(action, t=999), dim=T ) → (B, 1536)
    #                  현재 기본 방식. 단순하지만 temporal ordering 소실.
    #
    # "raw_flatten"  : ActionEncoder(action, t=999) 출력 직접 flatten → (B, T*D_a)
    #                  action encoder의 per-step latent를 그대로 펼친다.
    #                  No extra parameters required.
    cl_v2_action_repr: str = "mean_pool"  # "mean_pool" | "raw_flatten"

    # Phase 2 trainable mode.
    # "default"       : action expert frozen, VLM backbone trainable (existing RKD behavior)
    # "processed_only": freeze all, train only processed feature modules
    #                   (action_head.vlln + action_head.vl_self_attention)
    # "dit_core_only" : freeze all, train only action_head.model (DiT core)
    cl_v2_trainable_mode: str = "default"  # "default" | "processed_only" | "dit_core_only"

    # Phase 2 student token aggregation strategy.
    #
    # "flatten"           : (B, T_seq, D_vlm) -> (B, T_seq*D_vlm)
    # "mask_mean"         : masked mean over token axis -> (B, D_vlm)
    # "attention_pooling" : masked softmax-weighted token sum -> (B, D_vlm)
    cl_v2_student_repr: str = "flatten"  # "flatten" | "mask_mean" | "attention_pooling"

    def __post_init__(self):
        super().__post_init__()
        if self.cl_v2_phase not in {"phase1", "phase2"}:
            raise ValueError(
                f"cl_v2_phase must be 'phase1' or 'phase2', got {self.cl_v2_phase!r}"
            )
        if self.cl_v2_action_repr not in {"mean_pool", "raw_flatten"}:
            raise ValueError(
                f"cl_v2_action_repr must be 'mean_pool' or 'raw_flatten', got {self.cl_v2_action_repr!r}"
            )
        if self.cl_v2_student_repr not in {"flatten", "mask_mean", "attention_pooling"}:
            raise ValueError(
                "cl_v2_student_repr must be 'flatten', 'mask_mean', or "
                f"'attention_pooling', got {self.cl_v2_student_repr!r}"
            )
        if self.cl_v2_trainable_mode not in {"default", "processed_only", "dit_core_only"}:
            raise ValueError(
                "cl_v2_trainable_mode must be 'default', 'processed_only', or "
                "'dit_core_only', "
                f"got {self.cl_v2_trainable_mode!r}"
            )
