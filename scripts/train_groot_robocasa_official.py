#!/usr/bin/env python
"""GR00T official-style training on Robocasa pretrain — works for any GR00T-family policy.

Usage (smoke test on a single atomic task)::

    cd /home/seonho/clvla/lerobot_cl
    python scripts/train_groot_robocasa_official.py \
        --policy.type=groot_robocasa \
        --dataset_paths=[/home/seonho/groot_robocasa/robocasa_dataset/v1.0/pretrain/atomic/CloseDrawer] \
        --output_dir=/tmp/groot_smoke \
        --batch_size=2 --max_steps=2 --smoke_test=true

Swap ``--policy.type`` to ``groot_cl`` / ``groot_mgd`` / ``groot_cl_v2`` to run
the same official-batch path against an experimental policy.

Design notes:
- The dataset / transform / collate stack is the official Isaac-GR00T one
  (`gr00t.data.dataset.LeRobotSingleDataset`, `DATA_CONFIG_MAP`,
  `gr00t.model.transforms.collate`). We bypass LeRobot's `processor_groot`
  on the training path; that processor is preserved for inference / eval.
- Forward goes through `groot_common.training_adapter.forward_with_groot_batch`,
  which prefers `policy.forward_official_batch(batch)` (custom-loss hook for
  CL / MGD / CLv2 variants) and falls back to ``policy._groot_model(batch)``.
- Hyperparameter defaults mirror `gr00t_finetune.py` exactly so the optimiser
  / scheduler / bf16 setup is byte-for-byte the same.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lerobot.configs import parser
from lerobot.configs.default import WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.groot_common import (
    RobocasaPreset,
    apply_to_policy_config,
    assert_groot_compatible,
    assert_official_config_match,
    build_official_collate,
    build_official_dataset,
    ensure_isaac_gr00t_on_path,
    forward_with_groot_batch,
    run_parity_smoke_test,
    unwrap_groot_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GrootOfficialTrainConfig:
    """Mirrors `ArgsConfig` in `gr00t_finetune.py` plus LeRobot policy plumbing."""

    # --- Policy selection (LeRobot-side) ---
    policy: Optional[PreTrainedConfig] = None

    # --- Dataset paths (Isaac-GR00T side) ---
    # Each entry is a directory containing meta/, data/, videos/. Multiple paths
    # → LeRobotMixtureDataset. Single path → LeRobotSingleDataset.
    dataset_paths: list[str] = field(default_factory=list)

    # --- Run identification ---
    output_dir: Path = Path("outputs/groot_robocasa_official")
    job_name: str = "groot_robocasa_official"

    # --- Official hyperparameters (gr00t_finetune.py defaults) ---
    batch_size: int = 128
    max_steps: int = 300_000
    learning_rate: float = 3e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    save_steps: int = 20_000
    log_freq: int = 10
    seed: int = 42
    dataloader_num_workers: int = 8

    # --- Tune flags (passed through to GrootConfig) ---
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # --- LoRA (passthrough to GrootConfig) ---
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # --- Mixture knobs ---
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True
    ds_weights_alpha: float = 0.4

    # --- Smoke test mode ---
    # True: forces max_steps=2 / batch_size=2 / num_workers=0 / first dataset
    # only, and runs `run_parity_smoke_test` before stepping.
    smoke_test: bool = False

    # --- Resume ---
    resume: bool = False
    resume_from: str | None = None  # path to checkpoint dir; defaults to last

    # --- Reporting ---
    report_to: str = "none"  # "wandb" | "none"
    wandb: WandBConfig = field(default_factory=WandBConfig)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_optimizer_param_groups(
    policy: torch.nn.Module, weight_decay: float
) -> list[dict]:
    """Match gr00t_finetune.py: bias / 1-D / LayerNorm get weight_decay=0."""
    decay_params = []
    no_decay_params = []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _print_batch_shapes(batch: dict) -> None:
    logger.info("=== batch shape report ===")
    for key in sorted(batch.keys()):
        v = batch[key]
        if isinstance(v, torch.Tensor):
            logger.info("  %-30s shape=%s dtype=%s", key, tuple(v.shape), v.dtype)
        else:
            logger.info("  %-30s type=%s", key, type(v).__name__)


def _save_checkpoint(
    policy,
    accelerator: Accelerator,
    optimizer,
    scheduler,
    step: int,
    output_dir: Path,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_policy = accelerator.unwrap_model(policy)

    # LeRobot-side checkpoint (preserves policy class + config).
    unwrapped_policy.save_pretrained(ckpt_dir)

    # Official Isaac-GR00T checkpoint, ready for `gr00t_finetune` resume / eval.
    groot_model = unwrap_groot_model(policy, accelerator=accelerator)
    groot_model.save_pretrained(ckpt_dir / "groot_model")

    # Optimizer / scheduler / step (for `--resume`).
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy_rng_state": np.random.get_state(),
    }
    torch.save(state, ckpt_dir / "trainer_state.pt")
    return ckpt_dir


def _resolve_resume_dir(cfg: GrootOfficialTrainConfig) -> Path | None:
    if not cfg.resume:
        return None
    if cfg.resume_from:
        return Path(cfg.resume_from)
    candidates = sorted(cfg.output_dir.glob("checkpoint-*"))
    if not candidates:
        logger.warning("--resume set but no checkpoint-* under %s; starting fresh", cfg.output_dir)
        return None
    return candidates[-1]


def _maybe_load_state(
    optimizer, scheduler, resume_dir: Path | None
) -> int:
    if resume_dir is None:
        return 0
    state_path = resume_dir / "trainer_state.pt"
    if not state_path.exists():
        logger.warning("No trainer_state.pt at %s; starting from step 0", resume_dir)
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if state.get("numpy_rng_state") is not None:
        np.random.set_state(state["numpy_rng_state"])
    if state.get("rng_state") is not None:
        torch.set_rng_state(state["rng_state"])
    if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state.get("step", 0))


def _propagate_tune_and_lora_flags(cfg: GrootOfficialTrainConfig) -> None:
    """Forward CLI tune / LoRA flags into the policy config before instantiation."""
    pol = cfg.policy
    if pol is None:
        return
    pol.tune_llm = cfg.tune_llm
    pol.tune_visual = cfg.tune_visual
    pol.tune_projector = cfg.tune_projector
    pol.tune_diffusion_model = cfg.tune_diffusion_model
    pol.lora_rank = cfg.lora_rank
    pol.lora_alpha = cfg.lora_alpha
    pol.lora_dropout = cfg.lora_dropout


@parser.wrap()
def main(cfg: GrootOfficialTrainConfig) -> None:
    if cfg.policy is None:
        raise ValueError("--policy.type must be set (groot_robocasa | groot_cl | groot_mgd | groot_cl_v2).")
    if not cfg.dataset_paths:
        raise ValueError("--dataset_paths must contain at least one entry.")

    if cfg.smoke_test:
        cfg.max_steps = 2
        cfg.batch_size = min(cfg.batch_size, 2)
        cfg.dataloader_num_workers = 0
        cfg.dataset_paths = cfg.dataset_paths[:1]
        cfg.save_steps = cfg.max_steps  # save at end to verify both formats
        logger.info("[smoke_test] forcing max_steps=2, batch_size<=2, num_workers=0, single dataset")

    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_isaac_gr00t_on_path()
    preset = RobocasaPreset()

    # 1) Apply preset BEFORE policy instantiation so chunk_size / dims / features
    #    match Robocasa even when the user picks groot_cl / groot_mgd / groot_cl_v2
    #    (which inherit GrootConfig defaults of chunk_size=50).
    apply_to_policy_config(cfg.policy, preset)
    _propagate_tune_and_lora_flags(cfg)

    # 2) Accelerator setup. mixed_precision="no" because the policies use
    #    `torch.autocast(bf16)` internally via `config.use_bf16`, matching
    #    what `gr00t_finetune.py` does through `TrainingArguments(bf16=True)`.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="no",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    _seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        logger.info("Output dir: %s", cfg.output_dir)
        logger.info("Policy: --policy.type=%s", cfg.policy.type)
        logger.info("Dataset paths (%d): %s", len(cfg.dataset_paths), cfg.dataset_paths)
        logger.info("Accelerator: num_processes=%d, device=%s", accelerator.num_processes, device)

    # 3) Parity smoke test BEFORE building the full training stack so failure
    #    is fast and obvious.
    if cfg.smoke_test and accelerator.is_main_process:
        logger.info("[smoke_test] running batch parity check (reference vs adapter) …")
        report = run_parity_smoke_test(cfg.dataset_paths[0], preset=preset)
        logger.info(report.summary())
        if not report.ok:
            raise RuntimeError("Batch parity check failed; aborting.")

    # 4) Build the official dataset + DataLoader.
    dataset = build_official_dataset(
        cfg.dataset_paths,
        preset=preset,
        ds_weights_alpha=cfg.ds_weights_alpha,
        balance_dataset_weights=cfg.balance_dataset_weights,
        balance_trajectory_weights=cfg.balance_trajectory_weights,
        seed=cfg.seed,
    )
    collate_fn = build_official_collate()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=cfg.dataloader_num_workers > 0,
    )

    # 5) Build the policy directly (no make_policy → no LeRobotDatasetMetadata
    #    requirement). Preset has already populated input/output features.
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls(config=cfg.policy)
    policy.to(device)

    assert_groot_compatible(policy)
    if accelerator.is_main_process:
        assert_official_config_match(policy, preset)

    # 6) Optimiser + scheduler with `gr00t_finetune.py` parity.
    param_groups = _build_optimizer_param_groups(policy, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )
    warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    # 7) Resume support — load optimizer / scheduler state before prepare.
    resume_dir = _resolve_resume_dir(cfg)
    start_step = _maybe_load_state(optimizer, scheduler, resume_dir)
    if start_step:
        logger.info("Resuming from step %d (%s)", start_step, resume_dir)

    policy, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, optimizer, dataloader, scheduler
    )

    # 8) WandB.
    use_wandb = cfg.report_to == "wandb" and accelerator.is_main_process
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.job_name,
            notes=cfg.wandb.notes,
            mode=cfg.wandb.mode,
            dir=str(cfg.output_dir),
            config={
                "policy_type": cfg.policy.type,
                "base_model_path": cfg.policy.base_model_path,
                "dataset_paths": cfg.dataset_paths,
                "batch_size": cfg.batch_size,
                "max_steps": cfg.max_steps,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "warmup_ratio": cfg.warmup_ratio,
                "preset": {
                    "chunk_size": preset.chunk_size,
                    "max_action_dim": preset.max_action_dim,
                    "max_state_dim": preset.max_state_dim,
                    "data_config_name": preset.data_config_name,
                },
                "smoke_test": cfg.smoke_test,
            },
            save_code=False,
        )

    # 9) Smoke-test batch shape print (one batch, before stepping).
    def _infinite():
        while True:
            yield from dataloader

    stream = _infinite()

    if cfg.smoke_test and accelerator.is_main_process:
        peek_batch = next(stream)
        _print_batch_shapes(peek_batch)
        # Re-create iterator so the peek batch is not consumed for training.
        stream = _infinite()

    # 10) Training loop.
    policy.train()
    use_bf16 = bool(getattr(cfg.policy, "use_bf16", True))
    autocast_kwargs = dict(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)

    for step in range(start_step + 1, cfg.max_steps + 1):
        batch = next(stream)

        with torch.autocast(**autocast_kwargs):
            loss, loss_dict = forward_with_groot_batch(policy, batch)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(
            policy.parameters(), max_norm=cfg.grad_clip_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if accelerator.is_main_process and (step % cfg.log_freq == 0 or step == 1 or step == cfg.max_steps):
            lr_now = scheduler.get_last_lr()[0]
            metrics_str = " | ".join(
                f"{k}={(v.item() if isinstance(v, torch.Tensor) else float(v)):.4f}"
                for k, v in loss_dict.items()
            )
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logger.info(
                "step=%d/%d | lr=%.2e | grad_norm=%.3f | %s",
                step, cfg.max_steps, lr_now,
                grad_norm_val if grad_norm_val is not None else float("nan"),
                metrics_str,
            )
            if use_wandb:
                import wandb

                log_payload = {
                    "train/lr": lr_now,
                    "train/grad_norm": grad_norm_val if grad_norm_val is not None else float("nan"),
                }
                for k, v in loss_dict.items():
                    log_payload[f"train/{k}"] = v.item() if isinstance(v, torch.Tensor) else float(v)
                wandb.log(log_payload, step=step)

        if step % cfg.save_steps == 0 or step == cfg.max_steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_dir = _save_checkpoint(
                    policy, accelerator, optimizer, scheduler, step, cfg.output_dir
                )
                logger.info("checkpoint saved: %s", ckpt_dir)

    if use_wandb:
        import wandb

        wandb.finish()

    if accelerator.is_main_process:
        logger.info("training complete (final step=%d)", cfg.max_steps)


if __name__ == "__main__":
    main()
