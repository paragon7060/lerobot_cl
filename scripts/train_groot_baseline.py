#!/usr/bin/env python
"""GR00T Baseline 학습 스크립트 (accelerate + lerobot parser CLI + wandb 지원).

학습 모드별 권장 설정:
  Vision LoRA (경량, 기본 권장, ~12 GB VRAM):
    --policy.type=groot --policy.lora_rank=16 --policy.lora_target=vision

  LLM LoRA (~22 GB VRAM, gradient checkpointing 권장):
    --policy.type=groot --policy.lora_rank=16 --policy.lora_target=llm --gradient_checkpointing=true

  Full LoRA (LLM + Vision, ~28 GB VRAM):
    --policy.type=groot --policy.lora_rank=16 --policy.lora_target=both --gradient_checkpointing=true

  Partial frozen (LoRA 없음, ~14 GB VRAM):
    --policy.type=groot --policy.lora_rank=0

  Full finetuning (~35 GB+ VRAM):
    --policy.type=groot --policy.tune_llm=true --policy.lora_rank=0 --gradient_checkpointing=true

단일 GPU 실행:
  python scripts/train_groot_baseline.py \
      --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
      --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
      --dataset.video_backend=pyav \
      --policy.type=groot \
      --output_dir=./outputs/groot_baseline \
      --job_name=groot_baseline_v1 \
      --steps=50000 \
      --batch_size=128 \
      --policy.lora_rank=16 \
      --policy.lora_target=vision \
      --wandb.enable=true \
      --wandb.project=groot_insight \
      --wandb.entity=RwHlabs

Multi-GPU 실행 (예: 4 GPU, effective BS = 4 x 32 = 128):
  accelerate launch --num_processes=4 --mixed_precision=no \
      scripts/train_groot_baseline.py \
      --policy.type=groot \
      --batch_size=32 [other args]
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GrootBaselineTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="paragon7060/INSIGHTfixposV3",
            root=None,
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    steps: int = 50_000
    batch_size: int = 128
    num_workers: int = 8
    log_freq: int = 100
    save_freq: int = 5_000
    seed: int = 42
    use_policy_training_preset: bool = False

    gradient_checkpointing: bool = False

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_baseline"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_baseline")


@parser.wrap()
def main(cfg: GrootBaselineTrainConfig) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="no",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: %s", cfg.output_dir)
        logger.info("Accelerator: num_processes=%d, device=%s", accelerator.num_processes, device)

    use_wandb = cfg.wandb.enable and accelerator.is_main_process
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.job_name,
            notes=cfg.wandb.notes,
            id=cfg.wandb.run_id,
            mode=cfg.wandb.mode,
            dir=str(cfg.output_dir),
            config={
                "repo_id": cfg.dataset.repo_id,
                "base_model_path": cfg.policy.base_model_path,
                "steps": cfg.steps,
                "lr": cfg.policy.optimizer_lr,
                "batch_size": cfg.batch_size,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": cfg.batch_size * accelerator.num_processes,
                "tune_llm": cfg.policy.tune_llm,
                "tune_visual": cfg.policy.tune_visual,
                "tune_projector": cfg.policy.tune_projector,
                "tune_diffusion_model": cfg.policy.tune_diffusion_model,
                "lora_rank": cfg.policy.lora_rank,
                "lora_alpha": cfg.policy.lora_alpha,
                "lora_dropout": cfg.policy.lora_dropout,
                "lora_target": cfg.policy.lora_target,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "seed": cfg.seed,
            },
            save_code=False,
        )
        logger.info("WandB initialized: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    ds_meta = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, root=cfg.dataset.root)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if accelerator.is_main_process:
        dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )

    if accelerator.is_main_process:
        logger.info("Dataset: %d frames, %d episodes", dataset.num_frames, dataset.num_episodes)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    pre, _ = make_pre_post_processors(cfg.policy, dataset_stats=ds_meta.stats)

    policy = make_policy(cfg.policy, ds_meta=ds_meta)

    if cfg.policy.lora_rank > 0:
        if cfg.policy.lora_target in ("llm", "both") and cfg.policy.tune_llm:
            logger.warning(
                "lora_target=%r + tune_llm=True: LLM base weights도 학습 대상이 됩니다. "
                "LoRA 표준 사용법은 tune_llm=False입니다.", cfg.policy.lora_target
            )
        if cfg.policy.lora_target in ("vision", "both") and cfg.policy.tune_visual:
            logger.warning(
                "lora_target=%r + tune_visual=True: Vision tower base weights도 학습 대상이 됩니다. "
                "LoRA 표준 사용법은 tune_visual=False입니다.", cfg.policy.lora_target
            )

    if cfg.gradient_checkpointing:
        policy._groot_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    accelerator.wait_for_everyone()

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

    if accelerator.is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info("num_learnable_params=%s", f"{num_learnable_params:,}")
        logger.info("num_total_params=%s", f"{num_total_params:,}")
        logger.info("trainable ratio=%.2f%%", 100 * num_learnable_params / max(num_total_params, 1))

        if cfg.policy.lora_rank > 0:
            lora_params = sum(
                p.numel() for n, p in policy.named_parameters()
                if p.requires_grad and "lora_" in n
            )
            logger.info(
                "LoRA [%s] adapter params: %s (trainable total: %s / %s)",
                cfg.policy.lora_target,
                f"{lora_params:,}",
                f"{num_learnable_params:,}",
                f"{num_total_params:,}",
            )

        logger.info("Policy: %s", policy.__class__.__name__)

    def infinite_dataloader():
        while True:
            yield from dataloader

    def _save(step: int) -> None:
        if accelerator.is_main_process:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=accelerator.unwrap_model(policy),
                optimizer=optimizer,
                scheduler=scheduler,
            )
            update_last_checkpoint(checkpoint_dir)
            logger.info("체크포인트 저장: %s", checkpoint_dir)

    policy.train()
    data_stream = infinite_dataloader()

    for step in range(1, cfg.steps + 1):
        raw_batch = next(data_stream)
        batch = pre(raw_batch)

        loss, loss_dict = policy(batch)

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if accelerator.is_main_process and step % cfg.log_freq == 0:
            lr_now = scheduler.get_last_lr()[0]
            log_str = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            logger.info(
                "step=%d/%d | lr=%.2e | grad_norm=%.3f | %s",
                step, cfg.steps, lr_now,
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                log_str,
            )

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss_dict.get("loss", loss.item()),
                        "train/flow_matching_loss": loss_dict.get("flow_matching_loss", loss_dict.get("loss", loss.item())),
                        "train/lr": lr_now,
                        "train/grad_norm": (
                            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                        ),
                    },
                    step=step,
                )

        if step % cfg.save_freq == 0:
            accelerator.wait_for_everyone()
            _save(step)

            if use_wandb and not cfg.wandb.disable_artifact:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                artifact = wandb.Artifact(
                    name=f"{cfg.job_name}-step{step:06d}",
                    type="model",
                    description=f"checkpoint at step {step}",
                )
                artifact.add_dir(str(checkpoint_dir))
                wandb.log_artifact(artifact)

    accelerator.wait_for_everyone()
    _save(cfg.steps)

    if accelerator.is_main_process:
        logger.info("학습 완료. 최종 체크포인트: %s", get_step_checkpoint_dir(cfg.output_dir, cfg.steps, cfg.steps))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
