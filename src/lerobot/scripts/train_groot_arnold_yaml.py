#!/usr/bin/env python
"""Preliminary GR00T phase-training script for the converted ARNOLD LeRobot dataset.

This is intentionally conservative: it keeps the GR00T phase/trainability and
checkpoint flow, but uses the official LeRobot training batch path
raw_batch -> pre(raw_batch) -> policy(batch) on a single ARNOLD LeRobotDataset.
"""

import json
import logging
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
import lerobot
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)

LAST_CHECKPOINT_LINK = "last"
VALID_METHODS = {"auto", "mgd", "rkd", "base"}
VALID_PHASES = {"phase2", "phase3"}
VALID_PIPELINE_MODES = {"both", "phase2_only", "phase3_only"}
PIPELINE_ONLY_KEYS = {
    "recipe_yaml",
    "pipeline_yaml",
    "pipeline_mode",
    "dry_print_commands",
    "launcher",
    "phase2_output_dir",
    "phase3_output_dir",
    "wandb_auto_name",
    "run_name",
    "wandb_project_prefix",
    "wandb_group",
    "wandb_tags",
}
UNRESOLVED_ENV_VAR_RE = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ArnoldGrootPreset:
    """Schema bridge for the converted ARNOLD LeRobot dataset.

    Converted dataset facts checked on 2026-05-15:
      - observation.state: 9D Franka joint/gripper state
      - action: 9D Franka joint/gripper target
      - videos: front, base, left, wrist, wrist_bottom, each 480x480 RGB
    """

    embodiment_tag: str = "new_embodiment"
    chunk_size: int = 16
    n_action_steps: int = 16
    max_action_dim: int = 32
    max_state_dim: int = 64
    image_size: tuple[int, int] = (224, 224)
    state_shape: tuple[int, ...] = (9,)
    action_shape: tuple[int, ...] = (9,)
    video_backend: str = "pyav"
    video_feature_keys: tuple[str, ...] = (
        "observation.images.front",
        "observation.images.base",
        "observation.images.left",
        "observation.images.wrist",
        "observation.images.wrist_bottom",
    )


@dataclass
class ArnoldWandBConfig(WandBConfig):
    name: str = ""
    group: str = ""
    tags: str = ""


def apply_arnold_preset_to_policy_config(cfg: PreTrainedConfig, preset: ArnoldGrootPreset) -> None:
    cfg.embodiment_tag = preset.embodiment_tag
    cfg.chunk_size = preset.chunk_size
    cfg.n_action_steps = preset.n_action_steps
    cfg.max_action_dim = preset.max_action_dim
    cfg.max_state_dim = preset.max_state_dim
    cfg.image_size = preset.image_size
    if hasattr(cfg, "video_backend"):
        cfg.video_backend = preset.video_backend

    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=preset.state_shape),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, preset.image_size[0], preset.image_size[1]))
            for key in preset.video_feature_keys
        },
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=preset.action_shape)}


@dataclass
class ArnoldGrootPhaseTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="arnold_dataset",
            root="/home/ext_minje/arnold/arnold_dataset",
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    method: str = "auto"
    phase: str = "phase2"
    steps: int = 100_000
    batch_size: int = 64
    num_workers: int = 8
    log_freq: int = 100
    save_freq: int = 10_000
    seed: int = 42
    pretrained_path: str = ""
    resume: bool = False
    gradient_checkpointing: bool = False
    smoke_test: bool = False
    dry_run: bool = False
    grad_clip_norm: float = 10.0
    write_reports: bool = True
    fail_on_language_fallback: bool = True
    wandb: ArnoldWandBConfig = field(default_factory=ArnoldWandBConfig)

    recipe_yaml: str = ""
    pipeline_yaml: str = ""
    pipeline_mode: str = "both"
    dry_print_commands: bool = False
    launcher: str = "python"
    phase2_output_dir: str = ""
    phase3_output_dir: str = ""
    wandb_auto_name: bool = True
    run_name: str = ""
    wandb_project_prefix: str = "groot_processed"
    wandb_group: str = ""
    wandb_tags: str = ""

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set, e.g. --policy.type=groot_mgd or --policy.type=groot_cl_v2")
        if self.dataset.root is None:
            raise ValueError("dataset.root must be set")
        if self.method not in VALID_METHODS:
            raise ValueError(f"Unknown method={self.method!r}; expected one of {sorted(VALID_METHODS)}")
        if self.phase not in VALID_PHASES:
            raise ValueError(f"Unknown phase={self.phase!r}; expected one of {sorted(VALID_PHASES)}")
        if self.pipeline_mode not in VALID_PIPELINE_MODES:
            raise ValueError(
                f"Unknown pipeline_mode={self.pipeline_mode!r}; expected one of {sorted(VALID_PIPELINE_MODES)}"
            )
        if not self.job_name:
            self.job_name = "groot_arnold_phase_prelim"
        self.output_dir = Path(self.output_dir or "/home/ext_minje/arnold/outputs/groot_arnold_phase_prelim")
        self.output_dir.mkdir(parents=True, exist_ok=True)


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def infer_method(policy_cfg: PreTrainedConfig, method: str) -> str:
    if method != "auto":
        return method
    policy_type = str(safe_getattr(policy_cfg, "type", "")).lower()
    if "mgd" in policy_type:
        return "mgd"
    if "rkd" in policy_type or "cl_v2" in policy_type:
        return "rkd"
    return "base"


def apply_phase_and_trainable_overrides(cfg: ArnoldGrootPhaseTrainConfig) -> None:
    policy_cfg = cfg.policy
    if policy_cfg is None:
        raise ValueError("policy must be set before phase overrides")

    cfg.method = infer_method(policy_cfg, cfg.method)

    if cfg.method == "rkd" and hasattr(policy_cfg, "cl_v2_phase"):
        if safe_getattr(policy_cfg, "cl_v2_phase") != "phase2":
            logger.info(
                "Forcing policy.cl_v2_phase='phase2' for RKD unified %s path. "
                "phase3 disables auxiliary RKD/CL losses below but still uses the phase2 forward branch.",
                cfg.phase,
            )
        policy_cfg.cl_v2_phase = "phase2"

    if cfg.phase == "phase3":
        if hasattr(policy_cfg, "mgd_trainable_mode"):
            policy_cfg.mgd_trainable_mode = "dit_core_only"
        if hasattr(policy_cfg, "cl_v2_trainable_mode"):
            policy_cfg.cl_v2_trainable_mode = "dit_core_only"

    if safe_getattr(policy_cfg, "mgd_trainable_mode") == "dit_core_only":
        if hasattr(policy_cfg, "mgd_enabled"):
            policy_cfg.mgd_enabled = False
        if hasattr(policy_cfg, "mgd_loss_weight"):
            policy_cfg.mgd_loss_weight = 0.0

    if safe_getattr(policy_cfg, "cl_v2_trainable_mode") == "dit_core_only":
        if hasattr(policy_cfg, "cl_v2_fm_loss_weight"):
            if safe_getattr(policy_cfg, "cl_v2_fm_loss_weight") != 1.0:
                logger.info(
                    "phase3 RKD dit_core_only detected. Forcing policy.cl_v2_fm_loss_weight=1.0 "
                    "so the phase2 forward branch becomes FM-only at normal scale."
                )
            policy_cfg.cl_v2_fm_loss_weight = 1.0
        for weight_name in ("cl_v2_loss_weight", "rkd_loss_weight"):
            if hasattr(policy_cfg, weight_name):
                setattr(policy_cfg, weight_name, 0.0)


def apply_smoke_test_overrides(cfg: ArnoldGrootPhaseTrainConfig) -> None:
    if not cfg.smoke_test:
        return
    cfg.steps = min(cfg.steps, 100)
    cfg.batch_size = min(cfg.batch_size, 2)
    cfg.num_workers = 0
    cfg.save_freq = cfg.steps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def scalar_to_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def scalar_loss_dict(loss_dict: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in loss_dict.items():
        scalar = scalar_to_float(value)
        if scalar is not None:
            result[key] = scalar
    return result


def has_method(obj: Any, method_name: str) -> bool:
    return callable(getattr(obj, method_name, None))


def tensor_report(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            report[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    return report


def arnold_batch_spec_payload(preset: ArnoldGrootPreset) -> dict[str, Any]:
    return {
        "camera_order": list(preset.video_feature_keys),
        "state_order": ["observation.state"],
        "action_order": ["action"],
        "action_horizon": preset.n_action_steps,
        "chunk_size": preset.chunk_size,
        "padded_state_dim": preset.max_state_dim,
        "padded_action_dim": preset.max_action_dim,
    }


def normalize_policy_output(outputs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(outputs, tuple):
        if len(outputs) != 2:
            raise ValueError(f"Expected (loss, loss_dict), got tuple length={len(outputs)}")
        loss, loss_dict = outputs
        return loss, loss_dict or {}
    if isinstance(outputs, dict):
        if "loss" not in outputs:
            raise KeyError("policy output dict must contain 'loss'")
        return outputs["loss"], {k: v for k, v in outputs.items() if k != "loss"}
    if isinstance(outputs, torch.Tensor):
        return outputs, {"loss": outputs}
    raise TypeError(f"Unsupported policy output type: {type(outputs).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def parse_tags(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        tags: list[str] = []
        for item in value:
            tags.extend(parse_tags(item))
        return tags
    return [str(value)]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def expand_yaml_string(value: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(value))
    if UNRESOLVED_ENV_VAR_RE.search(expanded):
        raise RuntimeError(f"Unresolved environment variable in YAML value: {value}")
    return expanded


def expand_yaml_values(value: Any) -> Any:
    if isinstance(value, str):
        return expand_yaml_string(value)
    if isinstance(value, dict):
        return {key: expand_yaml_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_yaml_values(item) for item in value]
    return value


def apply_config_path_expansions(cfg: ArnoldGrootPhaseTrainConfig) -> None:
    if cfg.dataset.root is not None:
        cfg.dataset.root = expand_yaml_string(str(cfg.dataset.root))
    if cfg.pretrained_path:
        cfg.pretrained_path = expand_yaml_string(str(cfg.pretrained_path))
    if cfg.recipe_yaml:
        cfg.recipe_yaml = expand_yaml_string(str(cfg.recipe_yaml))
    if cfg.pipeline_yaml:
        cfg.pipeline_yaml = expand_yaml_string(str(cfg.pipeline_yaml))
    if cfg.phase2_output_dir:
        cfg.phase2_output_dir = expand_yaml_string(str(cfg.phase2_output_dir))
    if cfg.phase3_output_dir:
        cfg.phase3_output_dir = expand_yaml_string(str(cfg.phase3_output_dir))
    if cfg.policy is not None:
        for attr in ("groot_pretrained_path", "base_model_path"):
            value = safe_getattr(cfg.policy, attr)
            if isinstance(value, str) and ("$" in value or "~" in value):
                setattr(cfg.policy, attr, expand_yaml_string(value))


def load_pipeline_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML pipeline mode requires PyYAML. Please install pyyaml.") from exc

    yaml_path = Path(path)
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Pipeline YAML must contain a mapping at top level: {yaml_path}")
    return expand_yaml_values(payload)


def flatten_args(mapping: dict[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    if not mapping:
        return {}
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_args(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def merge_args(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if value is not None:
                merged[key] = value
    return merged


def cli_arg_was_provided(name: str) -> bool:
    return parser.parse_arg(name) is not None


def cli_override(cfg_value: Any, yaml_value: Any, cli_name: str) -> Any:
    return cfg_value if cli_arg_was_provided(cli_name) else yaml_value


def collect_explicit_cli_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    args = sys.argv[1:]
    idx = 0
    while idx < len(args):
        arg = args[idx]
        idx += 1
        if not arg.startswith("--"):
            continue
        key_value = arg[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
        else:
            key = key_value
            value = "true"
            if idx < len(args) and not args[idx].startswith("--"):
                value = args[idx]
                idx += 1
        if key in PIPELINE_ONLY_KEYS:
            continue
        overrides[key] = value
    return overrides


def command_value_to_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def args_to_cli_list(args: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key in sorted(args):
        if key in PIPELINE_ONLY_KEYS:
            continue
        value = command_value_to_string(args[key])
        if value is not None:
            cli_args.append(f"--{key}={value}")
    return cli_args


def infer_method_from_policy_type(policy_type: str | None) -> str:
    policy_type = str(policy_type or "").lower()
    if "mgd" in policy_type:
        return "mgd"
    if "rkd" in policy_type or "cl_v2" in policy_type:
        return "rkd"
    return "base"


def build_auto_wandb_tags(args: dict[str, Any], method: str, phase: str) -> list[str]:
    return unique_preserve_order(
        [
            method,
            phase,
            str(args.get("policy.type", "")),
            "arnold_native",
            *parse_tags(args.get("wandb.tags")),
        ]
    )


def build_wandb_resolved_from_cfg(cfg: ArnoldGrootPhaseTrainConfig) -> dict[str, Any]:
    return {
        "enable": bool(cfg.wandb.enable),
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "name": cfg.wandb.name or cfg.job_name,
        "group": cfg.wandb.group,
        "tags": parse_tags(cfg.wandb.tags),
        "method": cfg.method,
        "phase": cfg.phase,
    }


def apply_single_phase_wandb_auto_name(cfg: ArnoldGrootPhaseTrainConfig) -> None:
    if not cfg.wandb_auto_name:
        return
    policy_type = safe_getattr(cfg.policy, "type", "")
    method = infer_method_from_policy_type(policy_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not cli_arg_was_provided("wandb.project") and cfg.wandb.project == "lerobot":
        cfg.wandb.project = f"{cfg.wandb_project_prefix}_{method}"
    if not cfg.wandb.name:
        if cfg.run_name:
            cfg.wandb.name = f"{cfg.run_name}-{method}-{cfg.phase}"
        else:
            cfg.wandb.name = f"{policy_type}-{method}-{cfg.phase}-{timestamp}"
    if not cfg.wandb.group and cfg.wandb_group:
        cfg.wandb.group = cfg.wandb_group
    if not cfg.wandb.tags:
        tags = [
            method,
            cfg.phase,
            str(policy_type),
            "arnold_native",
            *parse_tags(cfg.wandb_tags),
        ]
        cfg.wandb.tags = ",".join(unique_preserve_order(tags))


def resolve_pipeline_wandb_args(
    phase_args: dict[str, Any],
    run_cfg: dict[str, Any],
    method: str,
    phase: str,
    timestamp: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = dict(phase_args)
    auto_name = parse_bool_value(run_cfg.get("wandb_auto_name", True))
    if not auto_name:
        tags = parse_tags(resolved.get("wandb.tags")) + parse_tags(run_cfg.get("wandb_tags"))
        if tags:
            resolved["wandb.tags"] = ",".join(unique_preserve_order(tags))
        return resolved, {
            "enable": parse_bool_value(resolved.get("wandb.enable", False)),
            "project": resolved.get("wandb.project"),
            "entity": resolved.get("wandb.entity"),
            "name": resolved.get("wandb.name", ""),
            "group": resolved.get("wandb.group", ""),
            "tags": parse_tags(resolved.get("wandb.tags")),
            "method": method,
            "phase": phase,
        }

    policy_type = str(resolved.get("policy.type", ""))
    run_name = str(run_cfg.get("run_name", "") or "")
    project_prefix = str(run_cfg.get("wandb_project_prefix", "groot_processed") or "groot_processed")
    group = str(resolved.get("wandb.group") or run_cfg.get("wandb_group") or "")
    if "wandb.project" not in resolved or not resolved.get("wandb.project"):
        resolved["wandb.project"] = f"{project_prefix}_{method}"
    if "wandb.name" not in resolved or not resolved.get("wandb.name"):
        if run_name:
            resolved["wandb.name"] = f"{run_name}-{method}-{phase}"
        else:
            resolved["wandb.name"] = f"{policy_type}-{method}-{phase}-{timestamp}"
    if not group:
        group_base = run_name or f"{policy_type}-{method}"
        group = f"{group_base}-{timestamp}"
    resolved["wandb.group"] = group
    tags = [
        *build_auto_wandb_tags(resolved, method, phase),
        *parse_tags(run_cfg.get("wandb_tags")),
    ]
    resolved["wandb.tags"] = ",".join(unique_preserve_order(tags))
    return resolved, {
        "enable": parse_bool_value(resolved.get("wandb.enable", False)),
        "project": resolved.get("wandb.project"),
        "entity": resolved.get("wandb.entity"),
        "name": resolved.get("wandb.name"),
        "group": resolved.get("wandb.group"),
        "tags": parse_tags(resolved.get("wandb.tags")),
        "method": method,
        "phase": phase,
    }


def resolve_phase2_checkpoint(phase2_output_dir: Path) -> Path:
    candidates = [
        phase2_output_dir / "checkpoints" / "last" / "pretrained_model",
        phase2_output_dir / "checkpoints" / "last",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError(
        "Could not resolve phase2 checkpoint. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def build_phase_command(launcher: str, script_path: Path, args: dict[str, Any]) -> list[str]:
    return [*shlex.split(launcher), str(script_path), *args_to_cli_list(args)]


def get_git_commit() -> str | None:
    repo_root = Path(lerobot.__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root}", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def save_pipeline_files(
    output_root: Path,
    commands_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
) -> None:
    write_json(output_root / "pipeline_commands.json", commands_payload)
    write_json(output_root / "pipeline_manifest.json", manifest_payload)


def run_pipeline_from_yaml(cfg: ArnoldGrootPhaseTrainConfig) -> None:
    recipe_path = Path(cfg.recipe_yaml or cfg.pipeline_yaml)
    recipe = load_pipeline_yaml(recipe_path)
    execution_cfg = recipe.get("execution", {}) or {}
    run_cfg = recipe.get("run", {}) or {}
    if cli_arg_was_provided("run_name"):
        run_cfg["run_name"] = cfg.run_name
    if cli_arg_was_provided("wandb_auto_name"):
        run_cfg["wandb_auto_name"] = cfg.wandb_auto_name
    if cli_arg_was_provided("wandb_project_prefix"):
        run_cfg["wandb_project_prefix"] = cfg.wandb_project_prefix
    if cli_arg_was_provided("wandb_group"):
        run_cfg["wandb_group"] = cfg.wandb_group
    if cli_arg_was_provided("wandb_tags"):
        run_cfg["wandb_tags"] = cfg.wandb_tags
    common_args = flatten_args(recipe.get("common", {}) or {})
    phase2_cfg = recipe.get("phase2", {}) or {}
    phase3_cfg = recipe.get("phase3", {}) or {}
    phase2_args = flatten_args(phase2_cfg.get("args", {}) or {})
    phase3_args = flatten_args(phase3_cfg.get("args", {}) or {})
    cli_overrides = collect_explicit_cli_overrides()

    pipeline_mode = cli_override(cfg.pipeline_mode, execution_cfg.get("pipeline_mode", "both"), "pipeline_mode")
    launcher = cli_override(cfg.launcher, execution_cfg.get("launcher", "python"), "launcher")
    dry_print = parse_bool_value(
        cli_override(cfg.dry_print_commands, execution_cfg.get("dry_print_commands", False), "dry_print_commands")
    )
    if pipeline_mode not in VALID_PIPELINE_MODES:
        raise ValueError(f"Unknown pipeline_mode={pipeline_mode!r}; expected one of {sorted(VALID_PIPELINE_MODES)}")

    phase2_output_dir_raw = cli_override(
        cfg.phase2_output_dir,
        phase2_cfg.get("output_dir") or phase2_args.get("output_dir"),
        "phase2_output_dir",
    )
    phase3_output_dir_raw = cli_override(
        cfg.phase3_output_dir,
        phase3_cfg.get("output_dir") or phase3_args.get("output_dir"),
        "phase3_output_dir",
    )
    phase2_output_dir = Path(str(phase2_output_dir_raw)) if phase2_output_dir_raw else None
    phase3_output_dir = Path(str(phase3_output_dir_raw)) if phase3_output_dir_raw else None

    if pipeline_mode in {"both", "phase2_only"} and phase2_output_dir is None:
        raise ValueError("phase2 output_dir is required for pipeline_mode='both' or 'phase2_only'.")
    if pipeline_mode in {"both", "phase3_only"} and phase3_output_dir is None:
        raise ValueError("phase3 output_dir is required for pipeline_mode='both' or 'phase3_only'.")

    phase3_pretrained_override = phase3_cfg.get("pretrained_path") or phase3_args.get("pretrained_path")
    if pipeline_mode == "phase3_only" and phase2_output_dir is None and not phase3_pretrained_override:
        raise ValueError(
            "phase2_output_dir is required in phase3_only when phase3.pretrained_path and "
            "phase3.args.pretrained_path are absent."
        )

    script_path = Path(__file__).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase2_command: list[str] | None = None
    phase3_command: list[str] | None = None
    phase2_final_args: dict[str, Any] | None = None
    phase3_final_args: dict[str, Any] | None = None
    phase2_wandb: dict[str, Any] | None = None
    phase3_wandb: dict[str, Any] | None = None
    resolved_phase2_checkpoint: str | None = None

    if pipeline_mode in {"both", "phase2_only"}:
        forced_phase2 = {"phase": "phase2", "output_dir": str(phase2_output_dir)}
        phase2_merged = merge_args(common_args, phase2_args, cli_overrides, forced_phase2)
        phase2_method = infer_method_from_policy_type(str(phase2_merged.get("policy.type", "")))
        phase2_final_args, phase2_wandb = resolve_pipeline_wandb_args(
            phase2_merged, run_cfg, phase2_method, "phase2", timestamp
        )
        phase2_command = build_phase_command(str(launcher), script_path, phase2_final_args)

    if pipeline_mode in {"both", "phase3_only"}:
        inherited_policy_type = phase3_args.get("policy.type") or phase2_args.get("policy.type") or common_args.get("policy.type")
        phase3_seed_args = dict(phase3_args)
        if inherited_policy_type and not phase3_seed_args.get("policy.type"):
            phase3_seed_args["policy.type"] = inherited_policy_type
        if phase3_pretrained_override:
            pretrained_path = Path(str(phase3_pretrained_override))
            resolved_phase2_checkpoint = str(pretrained_path)
        elif pipeline_mode == "phase3_only":
            checkpoint_path = resolve_phase2_checkpoint(phase2_output_dir)
            pretrained_path = checkpoint_path
            resolved_phase2_checkpoint = str(checkpoint_path)
        else:
            pretrained_path = phase2_output_dir / "checkpoints" / "last" / "pretrained_model"
            resolved_phase2_checkpoint = str(pretrained_path)
        forced_phase3 = {
            "phase": "phase3",
            "output_dir": str(phase3_output_dir),
            "pretrained_path": str(pretrained_path),
        }
        phase3_merged = merge_args(common_args, phase3_seed_args, cli_overrides, forced_phase3)
        phase3_method = infer_method_from_policy_type(str(phase3_merged.get("policy.type", "")))
        phase3_final_args, phase3_wandb = resolve_pipeline_wandb_args(
            phase3_merged, run_cfg, phase3_method, "phase3", timestamp
        )
        phase3_command = build_phase_command(str(launcher), script_path, phase3_final_args)

    output_root = phase3_output_dir if pipeline_mode != "phase2_only" else phase2_output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    commands_payload = {
        "phase2_command": phase2_command,
        "phase3_command": phase3_command,
        "phase2_args": phase2_final_args,
        "phase3_args": phase3_final_args,
    }
    manifest_payload = {
        "recipe_yaml": str(recipe_path),
        "pipeline_mode": pipeline_mode,
        "launcher": launcher,
        "phase2_command": phase2_command,
        "phase3_command": phase3_command,
        "phase2_output_dir": str(phase2_output_dir) if phase2_output_dir else None,
        "phase3_output_dir": str(phase3_output_dir) if phase3_output_dir else None,
        "resolved_phase2_checkpoint": resolved_phase2_checkpoint,
        "start_timestamp": utc_timestamp(),
        "end_timestamp": None,
        "return_status": "dry_print" if dry_print else "running",
        "git_commit": get_git_commit(),
        "wandb_resolved": {
            "project": (phase3_wandb or phase2_wandb or {}).get("project"),
            "group": (phase3_wandb or phase2_wandb or {}).get("group"),
            "phase2_name": phase2_wandb.get("name") if phase2_wandb else None,
            "phase3_name": phase3_wandb.get("name") if phase3_wandb else None,
            "tags": (phase3_wandb or phase2_wandb or {}).get("tags"),
            "phase2": phase2_wandb,
            "phase3": phase3_wandb,
        },
    }
    save_pipeline_files(output_root, commands_payload, manifest_payload)

    if dry_print:
        print(json.dumps(commands_payload, indent=2, sort_keys=True, default=str))
        manifest_payload["end_timestamp"] = utc_timestamp()
        manifest_payload["return_status"] = "dry_print"
        save_pipeline_files(output_root, commands_payload, manifest_payload)
        return

    try:
        if phase2_command is not None:
            logger.info("Running phase2 command: %s", phase2_command)
            subprocess.run(phase2_command, check=True)
        if phase3_command is not None:
            if phase3_final_args is not None and not phase3_pretrained_override:
                checkpoint_path = resolve_phase2_checkpoint(phase2_output_dir)
                resolved_phase2_checkpoint = str(checkpoint_path)
                phase3_final_args["pretrained_path"] = str(checkpoint_path)
                phase3_command = build_phase_command(str(launcher), script_path, phase3_final_args)
                commands_payload["phase3_command"] = phase3_command
                commands_payload["phase3_args"] = phase3_final_args
                manifest_payload["phase3_command"] = phase3_command
                manifest_payload["resolved_phase2_checkpoint"] = resolved_phase2_checkpoint
                save_pipeline_files(output_root, commands_payload, manifest_payload)
            logger.info("Running phase3 command: %s", phase3_command)
            subprocess.run(phase3_command, check=True)
    except subprocess.CalledProcessError:
        manifest_payload["end_timestamp"] = utc_timestamp()
        manifest_payload["return_status"] = "failed"
        save_pipeline_files(output_root, commands_payload, manifest_payload)
        raise

    manifest_payload["end_timestamp"] = utc_timestamp()
    manifest_payload["return_status"] = "success"
    save_pipeline_files(output_root, commands_payload, manifest_payload)


def get_trainable_mode(policy_cfg: PreTrainedConfig) -> str | None:
    return safe_getattr(policy_cfg, "mgd_trainable_mode") or safe_getattr(policy_cfg, "cl_v2_trainable_mode")


def build_trainable_param_report(policy: torch.nn.Module, policy_cfg: PreTrainedConfig) -> dict[str, Any]:
    named_params = list(policy.named_parameters())
    total = sum(p.numel() for _, p in named_params)
    trainable = sum(p.numel() for _, p in named_params if p.requires_grad)
    return {
        "policy_type": safe_getattr(policy_cfg, "type"),
        "trainable_mode": get_trainable_mode(policy_cfg),
        "num_total_params": int(total),
        "num_learnable_params": int(trainable),
        "trainable_ratio": float(trainable / max(total, 1)),
    }


def maybe_enable_gradient_checkpointing(policy: torch.nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    groot_model = getattr(policy, "_groot_model", None)
    if groot_model is not None and hasattr(groot_model, "gradient_checkpointing_enable"):
        groot_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
    else:
        logger.warning("gradient_checkpointing=true but policy._groot_model does not support it.")


def load_pretrained_if_requested(policy: torch.nn.Module, pretrained_path: str) -> Path | None:
    if not pretrained_path:
        return None
    from safetensors.torch import load_model as safetensors_load_model

    root = Path(pretrained_path)
    candidates = (root / "model.safetensors", root / "pretrained_model" / "model.safetensors")
    for model_path in candidates:
        if model_path.exists():
            safetensors_load_model(policy, str(model_path))
            logger.info("Loaded pretrained weights from %s", model_path)
            return model_path
    raise FileNotFoundError("Could not find model.safetensors. Checked: " + ", ".join(map(str, candidates)))


def build_dataloader(cfg: ArnoldGrootPhaseTrainConfig, dataset: LeRobotDataset) -> DataLoader:
    sampler = None
    shuffle = True
    drop_n_last_frames = safe_getattr(cfg.policy, "drop_n_last_frames")
    if drop_n_last_frames is not None:
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
        logger.warning(
            "policy.drop_n_last_frames=%s detected; using EpisodeAwareSampler like lerobot_train.py.",
            drop_n_last_frames,
        )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


def init_wandb_if_needed(
    cfg: ArnoldGrootPhaseTrainConfig,
    accelerator: Accelerator,
    dataset: LeRobotDataset,
) -> bool:
    use_wandb = cfg.wandb.enable and accelerator.is_main_process
    if not use_wandb:
        return False

    policy_cfg = cfg.policy
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name or cfg.job_name,
        group=cfg.wandb.group or None,
        tags=parse_tags(cfg.wandb.tags) or None,
        notes=cfg.wandb.notes,
        id=cfg.wandb.run_id,
        mode=cfg.wandb.mode,
        resume="allow" if cfg.resume else None,
        dir=str(cfg.output_dir),
        config={
            "repo_id": cfg.dataset.repo_id,
            "dataset_root": str(cfg.dataset.root),
            "method": cfg.method,
            "phase": cfg.phase,
            "steps": cfg.steps,
            "lr": safe_getattr(policy_cfg, "optimizer_lr"),
            "batch_size": cfg.batch_size,
            "num_processes": accelerator.num_processes,
            "effective_batch_size": cfg.batch_size * accelerator.num_processes,
            "tune_llm": safe_getattr(policy_cfg, "tune_llm"),
            "tune_visual": safe_getattr(policy_cfg, "tune_visual"),
            "tune_projector": safe_getattr(policy_cfg, "tune_projector"),
            "tune_diffusion_model": safe_getattr(policy_cfg, "tune_diffusion_model"),
            "lora_rank": safe_getattr(policy_cfg, "lora_rank"),
            "lora_alpha": safe_getattr(policy_cfg, "lora_alpha"),
            "lora_dropout": safe_getattr(policy_cfg, "lora_dropout"),
            "lora_target": safe_getattr(policy_cfg, "lora_target"),
            "mgd_enabled": safe_getattr(policy_cfg, "mgd_enabled"),
            "mgd_trainable_mode": safe_getattr(policy_cfg, "mgd_trainable_mode"),
            "mgd_loss_weight": safe_getattr(policy_cfg, "mgd_loss_weight"),
            "mgd_backprop_backbone": safe_getattr(policy_cfg, "mgd_backprop_backbone"),
            "cl_v2_phase": safe_getattr(policy_cfg, "cl_v2_phase"),
            "cl_v2_trainable_mode": safe_getattr(policy_cfg, "cl_v2_trainable_mode"),
            "cl_v2_loss_weight": safe_getattr(policy_cfg, "cl_v2_loss_weight"),
            "cl_v2_fm_loss_weight": safe_getattr(policy_cfg, "cl_v2_fm_loss_weight"),
            "cl_v2_student_repr": safe_getattr(policy_cfg, "cl_v2_student_repr"),
            "cl_v2_action_repr": safe_getattr(policy_cfg, "cl_v2_action_repr"),
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "seed": cfg.seed,
            "total_frames": dataset.num_frames,
            "total_episodes": dataset.num_episodes,
            "total_tasks": len(dataset.meta.tasks),
        },
        save_code=False,
    )
    logger.info("WandB initialized: project=%s, run=%s", cfg.wandb.project, wandb.run.name)
    return True


def save_training_checkpoint(
    cfg: ArnoldGrootPhaseTrainConfig,
    accelerator: Accelerator,
    policy: torch.nn.Module,
    optimizer: AdamW,
    scheduler,
    pre,
    post,
    step: int,
) -> Path | None:
    if not accelerator.is_main_process:
        return None
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step,
        cfg=cfg,
        policy=accelerator.unwrap_model(policy),
        optimizer=optimizer,
        scheduler=scheduler,
        preprocessor=pre,
        postprocessor=post,
    )
    update_last_checkpoint(checkpoint_dir)
    logger.info("checkpoint saved: %s", checkpoint_dir)
    return checkpoint_dir


def infinite_dataloader(dataloader: DataLoader):
    while True:
        yield from dataloader


def maybe_call_policy_update(accelerator: Accelerator, policy: torch.nn.Module) -> bool:
    try:
        unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    except TypeError:
        unwrapped = accelerator.unwrap_model(policy)
    if has_method(unwrapped, "update"):
        unwrapped.update()
        return True
    return False


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None


def _feature_shape(feature: Any) -> tuple[int, ...] | None:
    if isinstance(feature, dict) and "shape" in feature:
        return tuple(feature["shape"])
    shape = getattr(feature, "shape", None)
    return tuple(shape) if shape is not None else None


def _preview_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return {"shape": list(value.shape), "preview": []}
        flat = value.detach().reshape(-1).cpu()
        return {"shape": list(value.shape), "preview": flat[:5].tolist()}
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return {"shape": list(value.shape), "preview": []}
        return {"shape": list(value.shape), "preview": value.reshape(-1)[:5].tolist()}
    if isinstance(value, (list, tuple)):
        return value[:3]
    if isinstance(value, str):
        return value[:200]
    return value


def _describe_value(value: Any) -> dict[str, Any]:
    shape = _shape_tuple(value)
    description: dict[str, Any] = {"type": type(value).__name__}
    if shape is not None:
        description["shape"] = list(shape)
    if isinstance(value, torch.Tensor):
        description["dtype"] = str(value.dtype)
    elif isinstance(value, np.ndarray):
        description["dtype"] = str(value.dtype)
    description["preview"] = _preview_value(value)
    return description


def _extract_language_related_preview(raw_batch: dict[str, Any], processed_batch: dict[str, Any]) -> dict[str, Any]:
    previews: dict[str, Any] = {}
    candidate_keys = set(raw_batch.keys()) | set(processed_batch.keys())
    for key in sorted(candidate_keys):
        lowered = key.lower()
        if any(token in lowered for token in ("lang", "task", "instruction", "prompt", "text")):
            if key in raw_batch:
                previews[f"raw.{key}"] = _preview_value(raw_batch[key])
            if key in processed_batch:
                previews[f"processed.{key}"] = _preview_value(processed_batch[key])
    return previews


def _extract_mask_shapes(processed_batch: dict[str, Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for key, value in processed_batch.items():
        if "mask" in key.lower() or "is_pad" in key.lower():
            shape = _shape_tuple(value)
            if shape is not None:
                result[key] = list(shape)
    return result


def find_fallback_occurrences(value: Any, fallback: str = "Perform the task.", path: str = "value") -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        if value.strip() == fallback:
            found.append(path)
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(find_fallback_occurrences(item, fallback=fallback, path=f"{path}.{key}"))
        return found
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            found.extend(find_fallback_occurrences(item, fallback=fallback, path=f"{path}[{idx}]"))
        return found
    return found


def fallback_scan_report(raw_batch: dict[str, Any], processed_batch: dict[str, Any]) -> dict[str, Any]:
    raw_paths = find_fallback_occurrences(raw_batch, path="raw")
    processed_paths = find_fallback_occurrences(processed_batch, path="processed")
    return {
        "raw_paths": raw_paths,
        "processed_paths": processed_paths,
        "all_paths": raw_paths + processed_paths,
        "detected": bool(raw_paths or processed_paths),
    }


def first_batch_report_payload(
    cfg: ArnoldGrootPhaseTrainConfig,
    preset: ArnoldGrootPreset,
    dataset: LeRobotDataset,
    raw_batch: dict[str, Any],
    batch: dict[str, Any],
    fallback_report: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    language_preview = batch.get("language")
    payload = {
        "script_name": Path(__file__).name,
        "phase": cfg.phase,
        "method": cfg.method,
        "policy_type": safe_getattr(cfg.policy, "type"),
        "mgd_trainable_mode": safe_getattr(cfg.policy, "mgd_trainable_mode"),
        "mgd_enabled": safe_getattr(cfg.policy, "mgd_enabled"),
        "mgd_loss_weight": safe_getattr(cfg.policy, "mgd_loss_weight"),
        "mgd_fm_loss_weight": safe_getattr(cfg.policy, "mgd_fm_loss_weight"),
        "cl_v2_phase": safe_getattr(cfg.policy, "cl_v2_phase"),
        "cl_v2_trainable_mode": safe_getattr(cfg.policy, "cl_v2_trainable_mode"),
        "cl_v2_loss_weight": safe_getattr(cfg.policy, "cl_v2_loss_weight"),
        "cl_v2_fm_loss_weight": safe_getattr(cfg.policy, "cl_v2_fm_loss_weight"),
        "rkd_loss_weight": safe_getattr(cfg.policy, "rkd_loss_weight"),
        "camera_order": list(preset.video_feature_keys),
        "raw_keys": sorted(raw_batch.keys()),
        "dataset_feature_observation_state_shape": list(_feature_shape(dataset.meta.features.get(OBS_STATE)) or []),
        "dataset_feature_action_shape": list(_feature_shape(dataset.meta.features.get(ACTION)) or []),
        "raw_observation_state_shape": list(_shape_tuple(raw_batch.get(OBS_STATE)) or []),
        "raw_action_shape": list(_shape_tuple(raw_batch.get(ACTION)) or []),
        "raw_task_preview": _preview_value(raw_batch.get("task")),
        "language_related_preview": _extract_language_related_preview(raw_batch, batch),
        "train_batch_tensor_report": tensor_report(batch),
        "processed_state_shape": list(_shape_tuple(batch.get("state")) or []),
        "processed_action_shape": list(_shape_tuple(batch.get("action")) or []),
        "processed_mask_shapes": _extract_mask_shapes(batch),
        "processed_language_preview": language_preview if isinstance(language_preview, str) else _preview_value(language_preview),
        "language_fallback_scan": fallback_report,
        "language_fallback_detected": bool(fallback_report.get("detected")),
    }
    if extra:
        payload.update(extra)
    return payload


def hard_assert_arnold_dataset_schema(
    dataset: LeRobotDataset, raw_batch: dict[str, Any], preset: ArnoldGrootPreset
) -> None:
    if len(dataset.meta.tasks) <= 0:
        raise AssertionError("dataset.meta.tasks must contain at least one task string.")

    features = dataset.meta.features
    state_feature_shape = _feature_shape(features.get(OBS_STATE))
    action_feature_shape = _feature_shape(features.get(ACTION))
    if state_feature_shape != preset.state_shape:
        raise AssertionError(f"Expected dataset feature '{OBS_STATE}' shape {preset.state_shape}, got {state_feature_shape}")
    if action_feature_shape != preset.action_shape:
        raise AssertionError(f"Expected dataset feature '{ACTION}' shape {preset.action_shape}, got {action_feature_shape}")

    for key in preset.video_feature_keys:
        if key not in features:
            raise AssertionError(f"Missing required camera feature in dataset.meta.features: {key}")
        if key not in raw_batch:
            raise AssertionError(f"Missing required camera key in raw batch: {key}")

    raw_state = raw_batch.get(OBS_STATE)
    raw_action = raw_batch.get(ACTION)
    state_shape = _shape_tuple(raw_state)
    action_shape = _shape_tuple(raw_action)
    if state_shape is None or state_shape[-1] != preset.state_shape[0]:
        raise AssertionError(
            f"Expected raw '{OBS_STATE}' trailing dim {preset.state_shape[0]}, got shape={state_shape}"
        )
    if action_shape is None or action_shape[-1] != preset.action_shape[0]:
        raise AssertionError(
            f"Expected raw '{ACTION}' trailing dim {preset.action_shape[0]}, got shape={action_shape}"
        )


def assert_no_language_fallback(fallback_report: dict[str, Any], context: str, fail: bool) -> None:
    if not fallback_report.get("detected"):
        return
    message = (
        f"Detected fallback language 'Perform the task.' during {context}. "
        f"Occurrences: {fallback_report.get('all_paths')}"
    )
    if fail:
        raise RuntimeError(message)
    logger.warning("!!! %s !!!", message)


def assert_phase3_auxiliary_losses_disabled(cfg: ArnoldGrootPhaseTrainConfig, loss_dict: dict[str, Any]) -> None:
    if cfg.phase != "phase3" or cfg.method != "rkd":
        return
    cl_loss_weight = safe_getattr(cfg.policy, "cl_v2_loss_weight")
    rkd_loss_weight = safe_getattr(cfg.policy, "rkd_loss_weight")
    fm_loss_weight = safe_getattr(cfg.policy, "cl_v2_fm_loss_weight")
    if cl_loss_weight not in (None, 0.0):
        raise AssertionError(f"phase3 RKD expected cl_v2_loss_weight=0.0, got {cl_loss_weight}")
    if rkd_loss_weight not in (None, 0.0):
        raise AssertionError(f"phase3 RKD expected rkd_loss_weight=0.0/None, got {rkd_loss_weight}")
    if fm_loss_weight is not None and abs(float(fm_loss_weight) - 1.0) > 1e-8:
        raise AssertionError(f"phase3 RKD expected cl_v2_fm_loss_weight=1.0, got {fm_loss_weight}")

    loss = scalar_to_float(loss_dict.get("loss"))
    fm_loss = scalar_to_float(loss_dict.get("flow_matching_loss"))
    if loss is not None and fm_loss is not None and not np.isclose(loss, fm_loss, rtol=1e-4, atol=1e-5):
        raise AssertionError(f"phase3 RKD expected total loss == flow_matching_loss, got loss={loss}, fm={fm_loss}")


def dataset_sample_report(dataset: LeRobotDataset, preset: ArnoldGrootPreset, index: int = 0) -> dict[str, Any]:
    sample = dataset[index]
    return {
        "sample_index": index,
        "keys": sorted(sample.keys()),
        "task": sample.get("task"),
        "task_index": _preview_value(sample.get("task_index")),
        "camera_order": list(preset.video_feature_keys),
        "camera_keys_present": {key: key in sample for key in preset.video_feature_keys},
        "values": {key: _describe_value(value) for key, value in sorted(sample.items())},
    }


@parser.wrap()
def main(cfg: ArnoldGrootPhaseTrainConfig) -> None:
    apply_config_path_expansions(cfg)
    recipe_path = cfg.recipe_yaml or cfg.pipeline_yaml
    if recipe_path:
        run_pipeline_from_yaml(cfg)
        return

    cfg.validate()
    apply_smoke_test_overrides(cfg)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="no", step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])
    seed_everything(cfg.seed)

    preset = ArnoldGrootPreset(video_backend=cfg.dataset.video_backend or "pyav")
    apply_arnold_preset_to_policy_config(cfg.policy, preset)
    apply_phase_and_trainable_overrides(cfg)
    apply_single_phase_wandb_auto_name(cfg)

    dataset_root = Path(str(cfg.dataset.root)).expanduser()
    ds_meta = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.dataset.video_backend,
    )
    dataloader = build_dataloader(cfg, dataset)

    pre, post = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
    policy = make_policy(cfg.policy, ds_meta=ds_meta)
    loaded_pretrained_model_path = load_pretrained_if_requested(policy, cfg.pretrained_path)
    maybe_enable_gradient_checkpointing(policy, cfg.gradient_checkpointing)

    trainable_report = build_trainable_param_report(policy, cfg.policy)
    if accelerator.is_main_process and cfg.write_reports:
        write_json(cfg.output_dir / "trainable_param_report.json", trainable_report)
        write_json(cfg.output_dir / "dataset_sample_report.json", dataset_sample_report(dataset, preset, index=0))
        write_json(
            cfg.output_dir / "run_manifest.json",
            {
                "script_name": Path(__file__).name,
                "created_at_utc": utc_timestamp(),
                "dataset_root": str(dataset_root),
                "repo_id": cfg.dataset.repo_id,
                "total_frames": dataset.num_frames,
                "total_episodes": dataset.num_episodes,
                "total_tasks": len(dataset.meta.tasks),
                "fps": dataset.fps,
                "phase": cfg.phase,
                "method": cfg.method,
                "policy_type": safe_getattr(cfg.policy, "type"),
                "mgd_trainable_mode": safe_getattr(cfg.policy, "mgd_trainable_mode"),
                "mgd_enabled": safe_getattr(cfg.policy, "mgd_enabled"),
                "mgd_loss_weight": safe_getattr(cfg.policy, "mgd_loss_weight"),
                "mgd_fm_loss_weight": safe_getattr(cfg.policy, "mgd_fm_loss_weight"),
                "cl_v2_phase": safe_getattr(cfg.policy, "cl_v2_phase"),
                "cl_v2_trainable_mode": safe_getattr(cfg.policy, "cl_v2_trainable_mode"),
                "cl_v2_loss_weight": safe_getattr(cfg.policy, "cl_v2_loss_weight"),
                "cl_v2_fm_loss_weight": safe_getattr(cfg.policy, "cl_v2_fm_loss_weight"),
                "rkd_loss_weight": safe_getattr(cfg.policy, "rkd_loss_weight"),
                "batch_spec": arnold_batch_spec_payload(preset),
                "pretrained_path": cfg.pretrained_path,
                "loaded_pretrained_model_path": str(loaded_pretrained_model_path) if loaded_pretrained_model_path else None,
                "preprocessor_stats_source": "current ARNOLD dataset.meta.stats",
                "postprocessor_stats_source": "current ARNOLD dataset.meta.stats",
                "checkpoint_processor_state_loaded": False,
                "checkpoint_processor_state_note": (
                    "phase3 loads model.safetensors weights only; pre/post processors are rebuilt from "
                    "the current ARNOLD dataset stats for normalization consistency."
                ),
                "drop_n_last_frames": safe_getattr(cfg.policy, "drop_n_last_frames"),
                "uses_episode_aware_sampler": safe_getattr(cfg.policy, "drop_n_last_frames") is not None,
                "trainable_report": trainable_report,
                "wandb_resolved": build_wandb_resolved_from_cfg(cfg),
            },
        )
        logger.info("dataset=%s frames=%d episodes=%d tasks=%d", dataset_root, dataset.num_frames, dataset.num_episodes, len(dataset.meta.tasks))
        logger.info("trainable params: %s / %s", f"{trainable_report['num_learnable_params']:,}", f"{trainable_report['num_total_params']:,}")

    use_wandb = init_wandb_if_needed(cfg, accelerator, dataset)
    if use_wandb:
        wandb.summary["params/whole_total"] = int(trainable_report["num_total_params"])
        wandb.summary["params/whole_trainable"] = int(trainable_report["num_learnable_params"])

    if cfg.dry_run:
        raw_batch = next(iter(dataloader))
        hard_assert_arnold_dataset_schema(dataset, raw_batch, preset)
        batch = pre(raw_batch)
        fallback_report = fallback_scan_report(raw_batch, batch)
        assert_no_language_fallback(fallback_report, "dry_run", cfg.fail_on_language_fallback)

        if accelerator.is_main_process and cfg.write_reports:
            write_json(
                cfg.output_dir / "first_batch_report.json",
                first_batch_report_payload(
                    cfg,
                    preset,
                    dataset,
                    raw_batch,
                    batch,
                    fallback_report,
                ),
            )
            logger.info("dry_run=true: built dataset/dataloader/policy/first batch; no optimizer step.")
        if use_wandb:
            wandb.finish()
        return

    warmup_steps = int(cfg.steps * cfg.policy.warmup_ratio)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.policy.optimizer_lr,
        betas=cfg.policy.optimizer_betas,
        eps=cfg.policy.optimizer_eps,
        weight_decay=cfg.policy.optimizer_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=cfg.steps)

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, scheduler = accelerator.prepare(policy, optimizer, dataloader, scheduler)

    start_step = 0
    if cfg.resume:
        last_link = Path(cfg.output_dir) / "checkpoints" / LAST_CHECKPOINT_LINK
        if last_link.exists():
            resume_dir = last_link.resolve()
            from safetensors.torch import load_model as safetensors_load_model

            safetensors_load_model(accelerator.unwrap_model(policy), str(resume_dir / "pretrained_model" / "model.safetensors"))
            start_step, optimizer, scheduler = load_training_state(resume_dir, optimizer, scheduler)
            logger.info("resumed from %s at step %d", resume_dir, start_step)
        else:
            logger.warning("resume=true but no checkpoint exists at %s; starting from scratch", last_link)

    policy.train()
    data_stream = infinite_dataloader(dataloader)
    first_report_written = False
    steps_completed = start_step
    last_scalars: dict[str, float] = {}
    last_grad_norm: float | None = None
    last_lr: float | None = None
    last_policy_update_called = False

    for step in range(start_step + 1, cfg.steps + 1):
        t0 = time.perf_counter()
        raw_batch = next(data_stream)
        if step == start_step + 1:
            hard_assert_arnold_dataset_schema(dataset, raw_batch, preset)
        batch = pre(raw_batch)
        fallback_report = fallback_scan_report(raw_batch, batch)
        if step == start_step + 1:
            assert_no_language_fallback(fallback_report, "first full-train batch", cfg.fail_on_language_fallback)

        if accelerator.is_main_process and cfg.write_reports and not first_report_written:
            write_json(
                cfg.output_dir / "first_batch_report.json",
                first_batch_report_payload(
                    cfg,
                    preset,
                    dataset,
                    raw_batch,
                    batch,
                    fallback_report,
                    {"full_train_first_batch": True},
                ),
            )
            first_report_written = True
        t2 = time.perf_counter()

        outputs = policy(batch)
        loss, loss_dict = normalize_policy_output(outputs)
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"policy(batch) must return a Tensor loss, got {type(loss).__name__}")
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {scalar_to_float(loss)}")
        assert_phase3_auxiliary_losses_disabled(cfg, loss_dict)
        t3 = time.perf_counter()

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        policy_update_called = maybe_call_policy_update(accelerator, policy)
        t4 = time.perf_counter()
        steps_completed = step
        last_policy_update_called = policy_update_called
        last_grad_norm = scalar_to_float(grad_norm)
        last_lr = scheduler.get_last_lr()[0]
        last_scalars = scalar_loss_dict(loss_dict)
        last_scalars.setdefault("loss", float(loss.detach().float().cpu().item()))

        data_s = t2 - t0
        update_s = t4 - t2
        total_step_s = t4 - t0
        step_s = total_step_s
        step_per_s = (1.0 / step_s) if step_s > 0 else 0.0

        if accelerator.is_main_process and step % cfg.log_freq == 0:
            logger.info(
                "step=%d/%d | lr=%.2e | grad_norm=%.3f | step_s=%.3f | step/s=%.2f | %s",
                step,
                cfg.steps,
                last_lr,
                last_grad_norm or float("nan"),
                step_s,
                step_per_s,
                " | ".join(f"{k}={v:.4f}" for k, v in sorted(last_scalars.items()))
                + f" | policy_update_called={policy_update_called}",
            )
            if use_wandb:
                wandb_log = {
                    "train/loss": last_scalars.get("loss", float(loss.detach().float().cpu().item())),
                    "train/flow_matching_loss": last_scalars.get(
                        "flow_matching_loss",
                        last_scalars.get("loss", float(loss.detach().float().cpu().item())),
                    ),
                    "train/lr": last_lr,
                    "train/grad_norm": last_grad_norm,
                    "train/step_s": step_s,
                    "train/step_per_s": step_per_s,
                    "train/data_s": data_s,
                    "train/update_s": update_s,
                    "train/total_step_s": total_step_s,
                    "train/policy_update_called": 1.0 if policy_update_called else 0.0,
                }
                for key, value in last_scalars.items():
                    wandb_log[f"train/{key}"] = value
                wandb.log(wandb_log, step=step)

        if step % cfg.save_freq == 0:
            accelerator.wait_for_everyone()
            checkpoint_dir = save_training_checkpoint(cfg, accelerator, policy, optimizer, scheduler, pre, post, step)
            if use_wandb and checkpoint_dir is not None and not cfg.wandb.disable_artifact:
                artifact = wandb.Artifact(
                    name=f"{cfg.job_name}-step{step:06d}",
                    type="model",
                    description=f"checkpoint at step {step}",
                )
                artifact.add_dir(str(checkpoint_dir))
                wandb.log_artifact(artifact)

    accelerator.wait_for_everyone()
    final_checkpoint = save_training_checkpoint(cfg, accelerator, policy, optimizer, scheduler, pre, post, cfg.steps)
    if accelerator.is_main_process:
        if cfg.write_reports:
            write_json(
                cfg.output_dir / "train_summary_report.json",
                {
                    "script_name": Path(__file__).name,
                    "phase": cfg.phase,
                    "method": cfg.method,
                    "policy_type": safe_getattr(cfg.policy, "type"),
                    "steps_requested": cfg.steps,
                    "steps_completed": steps_completed,
                    "last_scalars": last_scalars,
                    "last_loss_finite": bool(np.isfinite(last_scalars.get("loss", float("nan")))),
                    "last_grad_norm": last_grad_norm,
                    "last_lr": last_lr,
                    "last_policy_update_called": last_policy_update_called,
                    "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
                    "loaded_pretrained_model_path": str(loaded_pretrained_model_path)
                    if loaded_pretrained_model_path
                    else None,
                },
            )
        logger.info("training complete. final checkpoint: %s", final_checkpoint)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
