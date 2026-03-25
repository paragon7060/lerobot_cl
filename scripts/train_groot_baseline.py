#!/usr/bin/env python
"""GR00T Baseline 학습 스크립트 (accelerate + draccus CLI + wandb 지원).

단일 GPU 실행:
  python scripts/train_groot_baseline.py \
      --repo_id=paragon7060/INSIGHTfixposV3 \
      --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
      --output_dir=./outputs/groot_baseline \
      --job_name=groot_baseline_v1 \
      --total_steps=50000 \
      --batch_size=128 \
      --lora_rank=16 \
      --wandb.enable=true \
      --wandb.project=groot_insight \
      --wandb.entity=RwHlabs

Multi-GPU 실행 (예: 4 GPU, effective BS = 4 x 32 = 128):
  accelerate launch --num_processes=4 --mixed_precision=no \
      scripts/train_groot_baseline.py \
      --batch_size=32 [other args]
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch
import wandb
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lerobot.configs.default import WandBConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GrootBaselineTrainConfig:
    repo_id: str = "paragon7060/INSIGHTfixposV3"
    root: str | None = None
    video_backend: str = "pyav"
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    output_dir: Path = Path("outputs/groot_baseline")
    job_name: str = "groot_baseline"
    total_steps: int = 50_000
    lr: float = 1e-4
    warmup_steps: int = 2_500
    batch_size: int = 128
    num_workers: int = 8
    seed: int = 42
    log_interval: int = 100
    save_interval: int = 5_000
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    wandb: WandBConfig = field(default_factory=WandBConfig)


@draccus.wrap()
def main(cfg: GrootBaselineTrainConfig) -> None:
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

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
                "repo_id": cfg.repo_id,
                "base_model_path": cfg.base_model_path,
                "total_steps": cfg.total_steps,
                "warmup_steps": cfg.warmup_steps,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": cfg.batch_size * accelerator.num_processes,
                "n_obs_steps": cfg.n_obs_steps,
                "chunk_size": cfg.chunk_size,
                "n_action_steps": cfg.n_action_steps,
                "tune_llm": cfg.tune_llm,
                "tune_visual": cfg.tune_visual,
                "tune_projector": cfg.tune_projector,
                "tune_diffusion_model": cfg.tune_diffusion_model,
                "lora_rank": cfg.lora_rank,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
                "seed": cfg.seed,
            },
            save_code=False,
        )
        logger.info("WandB initialized: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    policy_config = GrootConfig(
        base_model_path=cfg.base_model_path,
        n_obs_steps=cfg.n_obs_steps,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        tune_llm=cfg.tune_llm,
        tune_visual=cfg.tune_visual,
        tune_projector=cfg.tune_projector,
        tune_diffusion_model=cfg.tune_diffusion_model,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        video_backend=cfg.video_backend,
        use_bf16=True,
        batch_size=cfg.batch_size,
    )

    ds_meta = LeRobotDatasetMetadata(repo_id=cfg.repo_id, root=cfg.root)
    delta_timestamps = resolve_delta_timestamps(policy_config, ds_meta)

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.video_backend,
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
    )

    pre, _ = make_pre_post_processors(policy_config, dataset_stats=ds_meta.stats)

    policy = make_policy(policy_config, ds_meta=ds_meta)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.total_steps,
    )

    policy, dataloader, optimizer, scheduler = accelerator.prepare(policy, dataloader, optimizer, scheduler)

    if accelerator.is_main_process:
        logger.info("Policy: %s", policy.__class__.__name__)

    def to_device(batch: dict) -> dict:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def infinite_dataloader():
        while True:
            yield from dataloader

    def save_checkpoint(tag: str) -> None:
        if accelerator.is_main_process:
            path = cfg.output_dir / tag
            path.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(policy).save_pretrained(path)
            logger.info("체크포인트 저장: %s", path)

    policy.train()
    data_stream = infinite_dataloader()

    for step in range(1, cfg.total_steps + 1):
        raw_batch = next(data_stream)
        batch = pre(raw_batch)
        batch = to_device(batch)

        loss, loss_dict = policy(batch)

        optimizer.zero_grad()
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if accelerator.is_main_process and step % cfg.log_interval == 0:
            lr_now = scheduler.get_last_lr()[0]
            log_str = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            logger.info(
                "step=%d/%d | lr=%.2e | grad_norm=%.3f | %s",
                step, cfg.total_steps, lr_now,
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

        if step % cfg.save_interval == 0:
            accelerator.wait_for_everyone()
            save_checkpoint(f"step_{step:06d}")

            if use_wandb and not cfg.wandb.disable_artifact:
                ckpt_path = cfg.output_dir / f"step_{step:06d}"
                artifact = wandb.Artifact(
                    name=f"{cfg.job_name}-step{step:06d}",
                    type="model",
                    description=f"checkpoint at step {step}",
                )
                artifact.add_dir(str(ckpt_path))
                wandb.log_artifact(artifact)

    accelerator.wait_for_everyone()
    save_checkpoint("final")

    if accelerator.is_main_process:
        logger.info("학습 완료. 최종 체크포인트: %s/final", cfg.output_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
