#!/usr/bin/env python
"""LeRobot-native official-equivalent GR00T training for RoboCasa presliced datasets.

This script intentionally does not import the Isaac-GR00T dataset/transform/collate
stack. It trains GR00T-family policies on already-materialized RoboCasa v3
official-equivalent roots through MultiLeRobotDataset and LeRobotNativeBatchBuilder.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
from dataclasses import asdict, dataclass, field
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
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.groot_common.batch_builder import (
    LeRobotNativeBatchBuilder,
    make_robocasa_preset,
    tensor_report,
)
from lerobot.policies.groot_common.robocasa_official_runtime import (
    RoboCasaOfficialRuntimeDiscovery,
    discover_robocasa_official_runtime_repos,
)
from lerobot.policies.groot_common.robocasa_preset import apply_to_policy_config
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)

LAST_CHECKPOINT_LINK = "last"
VALID_METHODS = {"auto", "mgd", "rkd", "base"}
VALID_PHASES = {"phase2", "phase3"}
VALID_DATA_SPLITS = {"pretrain", "target", "real"}
VALID_SAMPLER_MODES = {"shuffle", "official_equiv"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GrootOfficialEquivTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="robocasa_human_v3",
            root="/home/seonho/groot_robocasa/robocasa_dataset/robocasa_v3_official_presliced",
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    method: str = "auto"
    phase: str = "phase2"
    data_split: str = "pretrain"
    steps: int = 100_000
    batch_size: int = 64
    num_workers: int = 8
    log_freq: int = 100
    save_freq: int = 10_000
    seed: int = 42
    pretrained_path: str = ""
    resume: bool = False
    gradient_checkpointing: bool = False
    sampler_mode: str = "shuffle"
    smoke_test: bool = False
    dry_run: bool = False
    grad_clip_norm: float = 10.0
    write_parity_report: bool = True
    write_manifest: bool = True
    dataset_family: str = "robocasa"
    use_policy_training_preset: bool = False

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if self.dataset.root is None:
            raise ValueError("dataset.root must be set")
        if self.method not in VALID_METHODS:
            raise ValueError(f"Unknown method: {self.method!r}. Expected one of {sorted(VALID_METHODS)}")
        if self.phase not in VALID_PHASES:
            raise ValueError(f"Unknown phase: {self.phase!r}. Expected one of {sorted(VALID_PHASES)}")
        if self.data_split not in VALID_DATA_SPLITS:
            raise ValueError(
                f"Unknown data_split: {self.data_split!r}. Expected one of {sorted(VALID_DATA_SPLITS)}"
            )
        if self.sampler_mode not in VALID_SAMPLER_MODES:
            raise ValueError(
                f"Unknown sampler_mode: {self.sampler_mode!r}. Expected one of {sorted(VALID_SAMPLER_MODES)}"
            )
        if self.dataset_family != "robocasa":
            raise NotImplementedError(
                f"dataset_family={self.dataset_family!r} is not implemented yet; only 'robocasa' is supported."
            )
        if not self.job_name:
            self.job_name = "groot_official_equiv"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_official_equiv")
        else:
            self.output_dir = Path(self.output_dir)
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


def apply_phase_and_trainable_overrides(cfg: GrootOfficialEquivTrainConfig) -> None:
    policy_cfg = cfg.policy
    if policy_cfg is None:
        raise ValueError("policy must be set before applying phase overrides")

    method = infer_method(policy_cfg, cfg.method)
    if cfg.method == "auto":
        cfg.method = method

    if method == "rkd" and hasattr(policy_cfg, "cl_v2_phase"):
        # groot_processed_rkd uses set_phase():
        # - phase1: FM-only action-expert path
        # - phase2: applies cl_v2_trainable_mode (including dit_core_only) and RKD/FM mixed forward
        # Therefore, unified phase2/phase3 should both resolve to cl_v2_phase="phase2".
        # In unified phase3 we disable auxiliary RKD/CL loss weights (below), resulting in FM-only
        # optimization under phase2's dit_core_only trainability routing.
        if safe_getattr(policy_cfg, "cl_v2_phase") != "phase2":
            logger.info(
                "Forcing policy.cl_v2_phase='phase2' for RKD unified %s path.",
                cfg.phase,
            )
        policy_cfg.cl_v2_phase = "phase2"

    if cfg.phase == "phase3":
        if hasattr(policy_cfg, "mgd_trainable_mode"):
            policy_cfg.mgd_trainable_mode = "dit_core_only"
        if hasattr(policy_cfg, "cl_v2_trainable_mode"):
            policy_cfg.cl_v2_trainable_mode = "dit_core_only"

    if safe_getattr(policy_cfg, "mgd_trainable_mode") == "dit_core_only":
        if safe_getattr(policy_cfg, "mgd_enabled") or safe_getattr(policy_cfg, "mgd_loss_weight", 0.0) != 0.0:
            logger.info(
                "mgd_trainable_mode=dit_core_only detected. Forcing policy.mgd_enabled=false and "
                "policy.mgd_loss_weight=0.0 (flow-matching-only stage)."
            )
        if hasattr(policy_cfg, "mgd_enabled"):
            policy_cfg.mgd_enabled = False
        if hasattr(policy_cfg, "mgd_loss_weight"):
            policy_cfg.mgd_loss_weight = 0.0

    if safe_getattr(policy_cfg, "cl_v2_trainable_mode") == "dit_core_only":
        logger.info(
            "cl_v2_trainable_mode=dit_core_only detected. Forcing RKD/CL auxiliary loss weights off; "
            "flow-matching weights are left unchanged."
        )
        for weight_name in ("cl_v2_loss_weight", "rkd_loss_weight"):
            if hasattr(policy_cfg, weight_name):
                setattr(policy_cfg, weight_name, 0.0)


def apply_smoke_test_overrides(cfg: GrootOfficialEquivTrainConfig) -> None:
    if not cfg.smoke_test:
        return
    cfg.steps = min(cfg.steps, 2)
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


def batch_spec_payload(batch_builder: LeRobotNativeBatchBuilder) -> dict[str, Any]:
    try:
        return asdict(batch_builder.batch_spec)
    except Exception:
        return dict(batch_builder.batch_spec.__dict__)


def scalar_to_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def scalar_loss_dict(loss_dict: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for key, value in loss_dict.items():
        scalar = scalar_to_float(value)
        if scalar is not None:
            scalars[key] = scalar
    return scalars


def normalize_policy_output(outputs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(outputs, tuple):
        if len(outputs) != 2:
            raise ValueError(f"Expected (loss, loss_dict), got tuple length={len(outputs)}")
        loss, loss_dict = outputs
        if loss_dict is None:
            loss_dict = {}
        return loss, loss_dict

    if isinstance(outputs, dict):
        if "loss" not in outputs:
            raise KeyError("policy output dict must contain 'loss'")
        loss = outputs["loss"]
        loss_dict = {k: v for k, v in outputs.items() if k != "loss"}
        return loss, loss_dict

    if isinstance(outputs, torch.Tensor):
        return outputs, {"loss": outputs}

    raise TypeError(f"Unsupported policy output type: {type(outputs).__name__}")


def get_trainable_mode(policy_cfg: PreTrainedConfig) -> str | None:
    return (
        safe_getattr(policy_cfg, "mgd_trainable_mode")
        or safe_getattr(policy_cfg, "cl_v2_trainable_mode")
    )


def build_trainable_param_report(policy: torch.nn.Module, policy_cfg: PreTrainedConfig) -> dict[str, Any]:
    named_params = list(policy.named_parameters())
    num_total_params = sum(p.numel() for _, p in named_params)
    num_learnable_params = sum(p.numel() for _, p in named_params if p.requires_grad)

    def group_count(prefix: str) -> dict[str, int]:
        total = sum(p.numel() for name, p in named_params if name.startswith(prefix))
        trainable = sum(
            p.numel()
            for name, p in named_params
            if name.startswith(prefix) and p.requires_grad
        )
        return {"total": int(total), "trainable": int(trainable)}

    groups = {
        "_groot_model.action_head.vlln": group_count("_groot_model.action_head.vlln"),
        "_groot_model.action_head.vl_self_attention": group_count(
            "_groot_model.action_head.vl_self_attention"
        ),
        "_groot_model.action_head.model": group_count("_groot_model.action_head.model"),
        "sequence_mgd_head": group_count("sequence_mgd_head"),
    }

    rkd_cl_keywords = ("rkd", "cl_v2", "teacher", "student", "contrastive")
    rkd_cl_total = sum(
        p.numel()
        for name, p in named_params
        if any(keyword in name.lower() for keyword in rkd_cl_keywords)
    )
    rkd_cl_trainable = sum(
        p.numel()
        for name, p in named_params
        if any(keyword in name.lower() for keyword in rkd_cl_keywords) and p.requires_grad
    )
    groups["rkd_cl_related"] = {"total": int(rkd_cl_total), "trainable": int(rkd_cl_trainable)}

    trainable_mode = get_trainable_mode(policy_cfg)
    if trainable_mode == "dit_core_only":
        selected_prefixes = ("_groot_model.action_head.model",)
    elif trainable_mode == "processed_only":
        selected_prefixes = (
            "_groot_model.action_head.vlln",
            "_groot_model.action_head.vl_self_attention",
            "sequence_mgd_head",
        )
    elif trainable_mode == "dit_only":
        selected_prefixes = ("_groot_model.action_head",)
    elif trainable_mode == "head_only":
        selected_prefixes = ("sequence_mgd_head",)
    else:
        selected_prefixes = tuple()

    if selected_prefixes:
        selected_total = sum(
            p.numel() for name, p in named_params if name.startswith(selected_prefixes)
        )
        selected_trainable = sum(
            p.numel()
            for name, p in named_params
            if name.startswith(selected_prefixes) and p.requires_grad
        )
        other_trainable = sum(
            p.numel()
            for name, p in named_params
            if p.requires_grad and not name.startswith(selected_prefixes)
        )
    else:
        selected_total = 0
        selected_trainable = 0
        other_trainable = num_learnable_params

    unexpected_dit_core_trainable = (
        int(other_trainable) if trainable_mode == "dit_core_only" else 0
    )

    return {
        "policy_type": safe_getattr(policy_cfg, "type"),
        "trainable_mode": trainable_mode,
        "num_total_params": int(num_total_params),
        "num_learnable_params": int(num_learnable_params),
        "trainable_ratio": float(num_learnable_params / max(num_total_params, 1)),
        "groups": groups,
        "selected_prefixes": list(selected_prefixes),
        "selected_total_params": int(selected_total),
        "selected_trainable_params": int(selected_trainable),
        "other_trainable_params": int(other_trainable),
        "dit_core_only_unexpected_trainable_params": unexpected_dit_core_trainable,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_first_batch_reports(
    cfg: GrootOfficialEquivTrainConfig,
    batch_builder: LeRobotNativeBatchBuilder,
    batch: dict[str, Any],
) -> tuple[Path, Path | None]:
    first_batch_path = cfg.output_dir / "first_batch_report.json"
    parity_path = cfg.output_dir / "parity_report.json"
    common_payload = {
        "script_name": Path(__file__).name,
        "policy_type": safe_getattr(cfg.policy, "type"),
        "phase": cfg.phase,
        "method": cfg.method,
        "batch_spec": batch_spec_payload(batch_builder),
    }
    write_json(
        first_batch_path,
        {
            **common_payload,
            "train_batch_tensor_report": tensor_report(batch),
        },
    )

    written_parity_path: Path | None = None
    if cfg.write_parity_report:
        parity_view = batch_builder.build_parity_view(batch)
        write_json(
            parity_path,
            {
                **common_payload,
                "parity_view_tensor_report": tensor_report(parity_view),
            },
        )
        written_parity_path = parity_path
    return first_batch_path, written_parity_path


def get_git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
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


def write_run_manifest(
    cfg: GrootOfficialEquivTrainConfig,
    discovery: RoboCasaOfficialRuntimeDiscovery,
    batch_builder: LeRobotNativeBatchBuilder,
    trainable_report_path: Path | None,
    first_batch_report_path: Path | None,
    parity_report_path: Path | None,
    final_checkpoint_path: Path | None,
) -> None:
    if not cfg.write_manifest:
        return
    manifest_path = cfg.output_dir / "run_manifest.json"
    policy_cfg = cfg.policy
    payload = {
        "script_name": Path(__file__).name,
        "policy_type": safe_getattr(cfg.policy, "type"),
        "method": cfg.method,
        "phase": cfg.phase,
        "trainable_mode": get_trainable_mode(cfg.policy),
        "dataset_root": str(cfg.dataset.root),
        "data_split": cfg.data_split,
        "repo_count": len(discovery.repo_ids),
        "atomic_count": discovery.atomic_count,
        "composite_count": discovery.composite_count,
        "v3_compatible_count": discovery.v3_compatible_count,
        "repo_ids_preview": discovery.repo_ids[:10],
        "batch_spec": batch_spec_payload(batch_builder),
        "sampler_mode": cfg.sampler_mode,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pretrained_path": cfg.pretrained_path,
        "resume": cfg.resume,
        "smoke_test": cfg.smoke_test,
        "dry_run": cfg.dry_run,
        "output_dir": str(cfg.output_dir),
        "first_batch_report_path": str(first_batch_report_path) if first_batch_report_path else None,
        "parity_report_path": str(parity_report_path) if parity_report_path else None,
        "trainable_param_report_path": str(trainable_report_path) if trainable_report_path else None,
        "final_checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path else None,
        "git_commit": get_git_commit(),
        "policy_resolved": {
            "policy_type": safe_getattr(policy_cfg, "type"),
            "method": cfg.method,
            "phase": cfg.phase,
            "mgd_enabled": safe_getattr(policy_cfg, "mgd_enabled"),
            "mgd_trainable_mode": safe_getattr(policy_cfg, "mgd_trainable_mode"),
            "mgd_loss_weight": safe_getattr(policy_cfg, "mgd_loss_weight"),
            "mgd_backprop_backbone": safe_getattr(policy_cfg, "mgd_backprop_backbone"),
            "cl_v2_phase": safe_getattr(policy_cfg, "cl_v2_phase"),
            "cl_v2_trainable_mode": safe_getattr(policy_cfg, "cl_v2_trainable_mode"),
            "cl_v2_loss_weight": safe_getattr(policy_cfg, "cl_v2_loss_weight"),
            "cl_v2_fm_loss_weight": safe_getattr(policy_cfg, "cl_v2_fm_loss_weight"),
            "rkd_loss_weight": safe_getattr(policy_cfg, "rkd_loss_weight"),
        },
    }
    write_json(manifest_path, payload)


def build_dataloader(cfg: GrootOfficialEquivTrainConfig, dataset: MultiLeRobotDataset) -> DataLoader:
    if cfg.sampler_mode == "official_equiv":
        raise NotImplementedError(
            "official_equiv sampler is not implemented yet; use shuffle or implement "
            "repo/episode/timestep sampler."
        )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )


def load_pretrained_if_requested(policy: torch.nn.Module, pretrained_path: str) -> None:
    if not pretrained_path:
        return
    from safetensors.torch import load_model as safetensors_load_model

    ckpt_root = Path(pretrained_path)
    candidates = (
        ckpt_root / "model.safetensors",
        ckpt_root / "pretrained_model" / "model.safetensors",
    )
    for model_path in candidates:
        if model_path.exists():
            safetensors_load_model(policy, str(model_path))
            logger.info("Loaded pretrained weights from %s", model_path)
            return
    raise FileNotFoundError(
        "Could not find model.safetensors in pretrained_path. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def maybe_enable_gradient_checkpointing(policy: torch.nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    groot_model = getattr(policy, "_groot_model", None)
    if groot_model is not None and hasattr(groot_model, "gradient_checkpointing_enable"):
        groot_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
    else:
        logger.warning("gradient_checkpointing=true but policy._groot_model does not support it.")


def infinite_dataloader(dataloader: DataLoader):
    while True:
        yield from dataloader


def init_wandb_if_needed(
    cfg: GrootOfficialEquivTrainConfig,
    accelerator: Accelerator,
    repo_ids: list[str],
    dataset: MultiLeRobotDataset,
) -> bool:
    use_wandb = cfg.wandb.enable and accelerator.is_main_process
    if not use_wandb:
        return False

    policy_cfg = cfg.policy
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.job_name,
        notes=cfg.wandb.notes,
        id=cfg.wandb.run_id,
        mode=cfg.wandb.mode,
        resume="allow" if cfg.resume else None,
        dir=str(cfg.output_dir),
        config={
            "repo_ids": repo_ids,
            "data_split": cfg.data_split,
            "dataset_root": str(cfg.dataset.root),
            "method": cfg.method,
            "phase": cfg.phase,
            "sampler_mode": cfg.sampler_mode,
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
        },
        save_code=False,
    )
    logger.info("WandB initialized: project=%s, run=%s", cfg.wandb.project, wandb.run.name)
    return True


def save_training_checkpoint(
    cfg: GrootOfficialEquivTrainConfig,
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
    logger.info("체크포인트 저장: %s", checkpoint_dir)
    return checkpoint_dir


@parser.wrap()
def main(cfg: GrootOfficialEquivTrainConfig) -> None:
    cfg.validate()
    apply_smoke_test_overrides(cfg)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="no",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    seed_everything(cfg.seed)

    dataset_root = Path(cfg.dataset.root)
    preset = make_robocasa_preset(video_backend=cfg.dataset.video_backend)
    apply_to_policy_config(cfg.policy, preset)
    apply_phase_and_trainable_overrides(cfg)
    batch_builder = LeRobotNativeBatchBuilder(preset=preset)

    discovery = discover_robocasa_official_runtime_repos(
        root=dataset_root,
        split=cfg.data_split,
    )
    repo_ids = discovery.repo_ids
    if not repo_ids:
        raise RuntimeError(f"No valid v3 RoboCasa task repos found. root={dataset_root}, split={cfg.data_split}")

    if accelerator.is_main_process:
        logger.info("Output dir: %s", cfg.output_dir)
        logger.info("Accelerator: num_processes=%d, device=%s", accelerator.num_processes, device)
        logger.info(
            "Dataset root: %s | split=%s | tasks=%d (atomic=%d, composite=%d, v3=%d)",
            dataset_root,
            cfg.data_split,
            len(repo_ids),
            discovery.atomic_count,
            discovery.composite_count,
            discovery.v3_compatible_count,
        )
        logger.info("BatchSpec: %s", batch_builder.batch_spec)
        if cfg.data_split == "pretrain":
            logger.info(
                "Using presliced dataset root; no extra episode slicing will be applied in train script."
            )

    first_ds_meta = LeRobotDatasetMetadata(
        repo_id=repo_ids[0],
        root=dataset_root / repo_ids[0],
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, first_ds_meta)

    if accelerator.is_main_process:
        dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            root=dataset_root,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            root=dataset_root,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )

    if accelerator.is_main_process:
        logger.info(
            "MultiLeRobotDataset: %d frames, %d episodes across %d tasks",
            dataset.num_frames,
            dataset.num_episodes,
            len(repo_ids),
        )
        if dataset.disabled_features:
            logger.warning("비활성화된 features (task 간 불일치): %s", dataset.disabled_features)

    dataloader = build_dataloader(cfg, dataset)

    pre, post = make_pre_post_processors(cfg.policy, dataset_stats=dataset.stats)
    policy = make_policy(cfg.policy, ds_meta=first_ds_meta)
    load_pretrained_if_requested(policy, cfg.pretrained_path)
    maybe_enable_gradient_checkpointing(policy, cfg.gradient_checkpointing)

    trainable_report = build_trainable_param_report(policy, cfg.policy)
    trainable_report_path = cfg.output_dir / "trainable_param_report.json"
    if accelerator.is_main_process:
        write_json(trainable_report_path, trainable_report)
        logger.info("num_learnable_params=%s", f"{trainable_report['num_learnable_params']:,}")
        logger.info("num_total_params=%s", f"{trainable_report['num_total_params']:,}")
        logger.info("trainable ratio=%.4f%%", 100.0 * trainable_report["trainable_ratio"])
        logger.info("trainable param report 저장: %s", trainable_report_path)
        if trainable_report["dit_core_only_unexpected_trainable_params"] > 0:
            logger.warning(
                "dit_core_only has non-DiT trainable params: %s",
                f"{trainable_report['dit_core_only_unexpected_trainable_params']:,}",
            )

    use_wandb = init_wandb_if_needed(cfg, accelerator, repo_ids, dataset)
    if use_wandb:
        wandb.summary["params/whole_total"] = int(trainable_report["num_total_params"])
        wandb.summary["params/whole_trainable"] = int(trainable_report["num_learnable_params"])
        wandb.summary["params/other_trainable"] = int(trainable_report["other_trainable_params"])

    if cfg.dry_run:
        raw_batch = next(iter(dataloader))
        batch = batch_builder.build_train_batch(raw_batch, pre)
        first_batch_report_path = None
        parity_report_path = None
        if accelerator.is_main_process:
            first_batch_report_path, parity_report_path = write_first_batch_reports(
                cfg, batch_builder, batch
            )
            write_run_manifest(
                cfg,
                discovery,
                batch_builder,
                trainable_report_path,
                first_batch_report_path,
                parity_report_path,
                final_checkpoint_path=None,
            )
            logger.info("dry_run=true: built dataset/dataloader/first batch/policy; exiting without checkpoint.")
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
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=cfg.steps,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, scheduler = accelerator.prepare(policy, optimizer, dataloader, scheduler)

    start_step = 0
    if cfg.resume:
        last_link = Path(cfg.output_dir) / "checkpoints" / LAST_CHECKPOINT_LINK
        if last_link.exists():
            resume_dir = last_link.resolve()
            logger.info("Resuming from checkpoint: %s", resume_dir)
            from safetensors.torch import load_model as safetensors_load_model

            model_path = resume_dir / "pretrained_model" / "model.safetensors"
            safetensors_load_model(accelerator.unwrap_model(policy), str(model_path))
            start_step, optimizer, scheduler = load_training_state(resume_dir, optimizer, scheduler)
            logger.info("Resumed at step %d", start_step)
        else:
            logger.warning("--resume=true but no checkpoint found at %s, training from scratch", last_link)

    if accelerator.is_main_process:
        write_run_manifest(
            cfg,
            discovery,
            batch_builder,
            trainable_report_path,
            first_batch_report_path=cfg.output_dir / "first_batch_report.json",
            parity_report_path=cfg.output_dir / "parity_report.json" if cfg.write_parity_report else None,
            final_checkpoint_path=None,
        )

    policy.train()
    data_stream = infinite_dataloader(dataloader)
    first_batch_report_written = False
    first_batch_report_path = cfg.output_dir / "first_batch_report.json"
    parity_report_path = cfg.output_dir / "parity_report.json" if cfg.write_parity_report else None

    for step in range(start_step + 1, cfg.steps + 1):
        raw_batch = next(data_stream)
        batch = batch_builder.build_train_batch(raw_batch, pre)

        if accelerator.is_main_process and not first_batch_report_written:
            first_batch_report_path, parity_report_path = write_first_batch_reports(cfg, batch_builder, batch)
            logger.info("first batch report 저장: %s", first_batch_report_path)
            if parity_report_path is not None:
                logger.info("parity report 저장: %s", parity_report_path)
            first_batch_report_written = True

        batch["compute_vlm_drift"] = (
            step % cfg.log_freq == 0 and bool(safe_getattr(cfg.policy, "vlm_drift_logging_enabled", False))
        )

        outputs = policy(batch)
        loss, loss_dict = normalize_policy_output(outputs)
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"policy(batch) must return a Tensor loss, got {type(loss).__name__}")
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {scalar_to_float(loss)}")

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if accelerator.is_main_process and step % cfg.log_freq == 0:
            lr_now = scheduler.get_last_lr()[0]
            loss_scalars = scalar_loss_dict(loss_dict)
            loss_scalars.setdefault("loss", float(loss.detach().float().cpu().item()))
            grad_norm_scalar = scalar_to_float(grad_norm)
            log_str = " | ".join(f"{key}={value:.4f}" for key, value in sorted(loss_scalars.items()))
            logger.info(
                "step=%d/%d | lr=%.2e | grad_norm=%.3f | %s",
                step,
                cfg.steps,
                lr_now,
                grad_norm_scalar if grad_norm_scalar is not None else float("nan"),
                log_str,
            )

            if use_wandb:
                wandb_log = {
                    "train/loss": loss_scalars.get("loss", float(loss.detach().float().cpu().item())),
                    "train/flow_matching_loss": loss_scalars.get(
                        "flow_matching_loss",
                        loss_scalars.get("loss", float(loss.detach().float().cpu().item())),
                    ),
                    "train/lr": lr_now,
                    "train/grad_norm": grad_norm_scalar,
                }
                for key, value in loss_scalars.items():
                    wandb_log[f"train/{key}"] = value
                wandb.log(wandb_log, step=step)

        if step % cfg.save_freq == 0:
            accelerator.wait_for_everyone()
            checkpoint_dir = save_training_checkpoint(
                cfg, accelerator, policy, optimizer, scheduler, pre, post, step
            )

            if use_wandb and checkpoint_dir is not None and not cfg.wandb.disable_artifact:
                artifact = wandb.Artifact(
                    name=f"{cfg.job_name}-step{step:06d}",
                    type="model",
                    description=f"checkpoint at step {step}",
                )
                artifact.add_dir(str(checkpoint_dir))
                wandb.log_artifact(artifact)

    accelerator.wait_for_everyone()
    final_checkpoint_path = save_training_checkpoint(
        cfg, accelerator, policy, optimizer, scheduler, pre, post, cfg.steps
    )

    if accelerator.is_main_process:
        logger.info("학습 완료. 최종 체크포인트: %s", final_checkpoint_path)
        write_run_manifest(
            cfg,
            discovery,
            batch_builder,
            trainable_report_path,
            first_batch_report_path,
            parity_report_path,
            final_checkpoint_path,
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
