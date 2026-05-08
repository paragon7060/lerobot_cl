"""Shared helpers for GR00T-family policies running official-style training.

This module is consumed by `scripts/train_groot_robocasa_official.py` and any
other script that wants to drive `groot_robocasa` / `groot_cl` / `groot_mgd` /
`groot_cl_v2` through the official Isaac-GR00T data + collate pipeline while
preserving each policy's custom forward (e.g. contrastive loss).
"""

from .official_data import (
    build_official_collate,
    build_official_dataset,
    ensure_isaac_gr00t_on_path,
    extract_eagle_processor,
)
from .parity import (
    ParityReport,
    check_batch_parity,
    run_parity_smoke_test,
)
from .robocasa_preset import RobocasaPreset, apply_to_policy_config
from .training_adapter import (
    GROOT_POLICY_TYPES,
    assert_groot_compatible,
    assert_official_config_match,
    forward_with_groot_batch,
    unwrap_groot_model,
)

__all__ = [
    "GROOT_POLICY_TYPES",
    "ParityReport",
    "RobocasaPreset",
    "apply_to_policy_config",
    "assert_groot_compatible",
    "assert_official_config_match",
    "build_official_collate",
    "build_official_dataset",
    "check_batch_parity",
    "ensure_isaac_gr00t_on_path",
    "extract_eagle_processor",
    "forward_with_groot_batch",
    "run_parity_smoke_test",
    "unwrap_groot_model",
]
