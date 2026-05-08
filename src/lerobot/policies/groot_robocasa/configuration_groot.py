#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PolicyFeature, PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.utils.constants import ACTION

from ..groot.configuration_groot import GrootConfig


@PreTrainedConfig.register_subclass("groot_robocasa")
@dataclass
class GrootRobocasaConfig(GrootConfig):
    """Robocasa-specialized defaults for GR00T policy."""

    base_model_path: str = "paragon7060/Robocasa_baseline"
    embodiment_tag: str = "new_embodiment"
    max_action_dim: int = 32
    chunk_size: int = 16
    n_action_steps: int = 16
    image_size: tuple[int, int] = (224, 224)
    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.robot0_agentview_left": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.robot0_eye_in_hand": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.robot0_agentview_right": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
        }
    )
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,)),
        }
    )


# Backward-compatible alias for copied groot_robocasa internals.
GrootConfig = GrootRobocasaConfig
