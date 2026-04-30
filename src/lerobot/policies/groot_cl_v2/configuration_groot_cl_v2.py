from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot_cl.configuration_groot import GrootConfig


@PreTrainedConfig.register_subclass("groot_cl_v2")
@dataclass
class GrootCLv2Config(GrootConfig):
    # Match NVIDIA pretrained GR00T-N1.5-3B model's action_head_cfg.max_state_dim=64
    max_state_dim: int = 64
    """Configuration for GR00T-CL v2: action-guided contrastive VLM finetuning.

    Phase 1: Freeze VLM backbone + vision tower. Train action expert (FM loss)
             with per-joint weighting to emphasize specific joints (e.g. wrist).
    Phase 2: Freeze action expert. Finetune VLM via weighted InfoNCE using
             clean action features (noise_timestep=0) as soft positive/negative signal.
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

    # Phase 2 projector settings (for future use)
    cl_v2_latent_dim: int = 256
    cl_v2_hidden_dim: int = 512

    # Phase 2 loss weights
    cl_v2_loss_weight: float = 0.1       # weighted InfoNCE weight
    cl_v2_fm_loss_weight: float = 0.01   # flow matching monitoring weight in phase 2

    # Weighted InfoNCE temperatures
    cl_v2_action_temp: float = 0.1   # action feature similarity → soft label sharpness
    cl_v2_vlm_temp: float = 0.07     # VLM logit temperature

    def __post_init__(self):
        super().__post_init__()
        if self.cl_v2_phase not in {"phase1", "phase2"}:
            raise ValueError(
                f"cl_v2_phase must be 'phase1' or 'phase2', got {self.cl_v2_phase!r}"
            )
