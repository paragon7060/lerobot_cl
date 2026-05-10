"""Batch schema primitives for GR00T Robocasa training/parity flows."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_REQUIRED_CORE_KEYS: tuple[str, ...] = (
    "state",
    "state_mask",
    "action",
    "action_mask",
    "embodiment_id",
    "has_real_action",
    "segmentation_target",
    "segmentation_target_mask",
)

DEFAULT_EXCLUDED_EAGLE_KEYS: tuple[str, ...] = (
    "eagle_input_ids",
    "eagle_attention_mask",
    "eagle_pixel_values",
    "eagle_image_sizes",
    "eagle_image_grid_thw",
)


@dataclass(frozen=True)
class BatchSpec:
    """Parity-oriented schema for the official-equivalent Robocasa batch."""

    camera_order: tuple[str, ...]
    state_order: tuple[str, ...]
    action_order: tuple[str, ...]
    action_horizon: int
    chunk_size: int
    padded_state_dim: int
    padded_action_dim: int
    required_core_keys: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_REQUIRED_CORE_KEYS
    )
    excluded_eagle_keys: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_EXCLUDED_EAGLE_KEYS
    )

