"""Minimal shared helpers consumed by current GR00T training scripts."""

from .official_data import (
    build_official_collate,
    build_official_dataset,
    ensure_isaac_gr00t_on_path,
    resolve_dataset_soup,
)
from .batch_builder import LeRobotNativeBatchBuilder, make_robocasa_preset, tensor_report
from .filter_key_subset import (
    load_episode_ids,
    parse_num_demos_from_filter_key,
    sample_episode_ids_from_filter_key,
)
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
    "load_episode_ids",
    "make_robocasa_preset",
    "parse_num_demos_from_filter_key",
    "resolve_dataset_soup",
    "run_parity_smoke_test",
    "sample_episode_ids_from_filter_key",
    "tensor_report",
    "unwrap_groot_model",
]
