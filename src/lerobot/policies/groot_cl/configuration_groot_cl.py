from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot_cl.configuration_groot import GrootConfig


@PreTrainedConfig.register_subclass("groot_cl")
@dataclass
class GrootCLConfig(GrootConfig):
    use_contrastive: bool = True
    contrastive_latent_dim: int = 256
    contrastive_cnn_hidden_dim: int = 128
    contrastive_proj_hidden_dim: int = 512
    contrastive_triplet_margin: float = 0.5
    contrastive_loss_weight: float = 0.1
    contrastive_phase: str = "phase1"
    contrastive_backprop_backbone: bool = True
    contrastive_fallback_to_in_batch: bool = False

    # 데이터셋별로 사전학습된 GrootPolicy 체크포인트 경로.
    # 설정 시 base_model_path(NVIDIA 원본) 대신 해당 checkpoint의
    # _groot_model.* weights를 로드한다. contrastive heads는 랜덤 초기화.
    # None이면 base_model_path에서 직접 로드 (기존 동작 유지).
    groot_pretrained_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.contrastive_phase not in {"phase1", "phase2a", "phase2b"}:
            raise ValueError(
                f"contrastive_phase must be 'phase1'|'phase2a'|'phase2b', "
                f"got {self.contrastive_phase!r}"
            )
