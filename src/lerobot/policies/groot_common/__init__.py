"""Minimal shared helpers consumed by current GR00T training scripts."""

from .official_data import (
    build_official_collate,
    build_official_dataset,
    ensure_isaac_gr00t_on_path,
)
from .batch_builder import LeRobotNativeBatchBuilder, make_robocasa_preset, tensor_report
from .parity import (
    compare_tensor_reports,
    run_parity_smoke_test,
)
from .robocasa_preset import RobocasaPreset, apply_to_policy_config
from .robocasa_official_runtime import (
    RoboCasaOfficialRuntimeDiscovery,
    discover_robocasa_official_runtime_repos,
)
from .training_adapter import (
    assert_groot_compatible,
    assert_official_config_match,
    forward_with_mgd_custom_loss_hook,
    forward_with_groot_batch,
    unwrap_groot_model,
)

__all__ = [
    "LeRobotNativeBatchBuilder",
    "RobocasaPreset",
    "RoboCasaOfficialRuntimeDiscovery",
    "apply_to_policy_config",
    "assert_groot_compatible",
    "assert_official_config_match",
    "build_official_collate",
    "build_official_dataset",
    "compare_tensor_reports",
    "ensure_isaac_gr00t_on_path",
    "discover_robocasa_official_runtime_repos",
    "forward_with_mgd_custom_loss_hook",
    "forward_with_groot_batch",
    "make_robocasa_preset",
    "run_parity_smoke_test",
    "tensor_report",
    "unwrap_groot_model",
]
