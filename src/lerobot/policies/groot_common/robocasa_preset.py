"""Robocasa preset shared by all GR00T-family policies.

The preset captures the schema that Isaac-GR00T's official `gr00t_finetune.py`
expects when training on the Robocasa dataset (PandaOmron embodiment): which
data-config to use, action / state padding sizes, embodiment tag, the three
camera streams, and the LeRobot-side feature shapes.

`apply_to_policy_config` mutates a `GrootConfig`-derived config in place so the
same preset can drive `groot_robocasa`, `groot_cl`, `groot_mgd`, `groot_cl_v2`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

if TYPE_CHECKING:
    from lerobot.policies.groot.configuration_groot import GrootConfig


# LeRobot side: keys that appear in `input_features`. The `observation.images.*`
# namespace is the LeRobot convention; the corresponding modality.json keys on
# the Isaac-GR00T side are `video.robot0_*` (without the prefix).
ROBOCASA_VIDEO_FEATURE_KEYS: tuple[str, ...] = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_right",
)


@dataclass
class RobocasaPreset:
    """Shared schema for GR00T official-style training on Robocasa pretrain."""

    # Pretrained backbone — usable for both `paragon7060/Robocasa_baseline`
    # and the upstream `nvidia/GR00T-N1.5-3B`.
    base_model_path: str = "paragon7060/Robocasa_baseline"

    # Embodiment tag override; `new_embodiment` is the projector index reserved
    # for "any new robot", which is what the Robocasa pretraining run uses.
    embodiment_tag: str = "new_embodiment"

    # Action / state padding budgets. Must match the model's action head config.
    chunk_size: int = 16
    n_action_steps: int = 16
    max_action_dim: int = 32
    max_state_dim: int = 64

    image_size: tuple[int, int] = (224, 224)

    # Key into `gr00t.experiment.data_config.DATA_CONFIG_MAP`. `panda_omron`
    # matches the modality.json layout under `groot_robocasa/.../pretrain/*/`.
    data_config_name: str = "panda_omron"

    # Used to pick the video decoder in LeRobotSingleDataset.
    video_backend: str = "opencv"

    # Raw LeRobot-side state / action dims (before GR00TTransform padding).
    # 16 = base_pos[3] + base_quat[4] + ee_pos[3] + ee_quat[4] + gripper[2].
    # 12 = base_motion[4] + control_mode[1] + ee_pos[3] + ee_rot[3] + gripper[1].
    state_shape: tuple[int, ...] = (16,)
    action_shape: tuple[int, ...] = (12,)

    video_feature_keys: tuple[str, ...] = field(
        default_factory=lambda: ROBOCASA_VIDEO_FEATURE_KEYS
    )


def _build_input_features(preset: RobocasaPreset) -> dict[str, PolicyFeature]:
    features: dict[str, PolicyFeature] = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=preset.state_shape),
    }
    for key in preset.video_feature_keys:
        features[key] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, preset.image_size[0], preset.image_size[1]),
        )
    return features


def _build_output_features(preset: RobocasaPreset) -> dict[str, PolicyFeature]:
    return {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=preset.action_shape),
    }


def apply_to_policy_config(cfg: "GrootConfig", preset: RobocasaPreset) -> None:
    """Mutate a GR00T-family config in place so it matches the Robocasa preset.

    Only fields the preset is authoritative for are overwritten; experiment-
    specific fields (e.g. contrastive loss weights on `GrootCLConfig`, MGD knobs
    on `GrootMGDConfig`) are left untouched.

    Model checkpoint paths are intentionally not overwritten here. They must
    come from YAML/CLI or the policy's own default fallback so that
    `base_model_path` / `groot_pretrained_path` obey user intent.
    """
    cfg.embodiment_tag = preset.embodiment_tag
    cfg.chunk_size = preset.chunk_size
    cfg.n_action_steps = preset.n_action_steps
    cfg.max_action_dim = preset.max_action_dim
    cfg.max_state_dim = preset.max_state_dim
    cfg.image_size = preset.image_size

    # `video_backend` is a per-policy field (defaults to "decord" on GrootConfig
    # but "opencv" on GrootRobocasaConfig). Force it to the preset value so
    # parity with `gr00t_finetune.py --video_backend` is unambiguous.
    if hasattr(cfg, "video_backend"):
        cfg.video_backend = preset.video_backend

    cfg.input_features = _build_input_features(preset)
    cfg.output_features = _build_output_features(preset)
