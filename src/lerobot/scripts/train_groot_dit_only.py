#!/usr/bin/env python
"""DiT-only adaptation 학습 스크립트

lora_only로 학습된 체크포인트를 시작점으로,
LoRA(VLM backbone)는 frozen하고 action_head(DiT)만 학습한다.
action_encoder는 frozen 유지 (MGD target path).

실행 예:
  python scripts/train_groot_dit_only.py \
      --dataset.root=/home/seonho/slicing_robocasa_human_v3 \
      --policy.type=groot_mgd \
      --policy.groot_pretrained_path=/path/to/lora_only/checkpoints/040000 \
      --policy.mgd_trainable_mode=dit_only \
      --policy.mgd_enabled=false \
      --policy.lora_rank=8 \
      --policy.lora_alpha=16 \
      --policy.lora_dropout=0.05 \
      --policy.lora_target=llm \
      --policy.tune_visual=false \
      --policy.tune_llm=false \
      --output_dir=./outputs/groot_dit_only \
      --steps=100000 \
      --batch_size=64 \
      --log_freq=50 \
      --save_freq=25000 \
      --data_split=target \
      --wandb.enable=true \
      --wandb.project=groot_mgd \
      --wandb.entity=RwHlabs
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
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)

LAST_CHECKPOINT_LINK = "last"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PRETRAIN_DIRS = [
    "robocasa_pretrain_human_atomic",
    "robocasa_pretrain_human_composite",
]
TARGET_DIRS = [
    "robocasa_target_human_atomic",
    "robocasa_target_human_composite",
]
ALL_DIRS = PRETRAIN_DIRS + TARGET_DIRS


def get_task_repo_ids(root: Path, top_level_dirs: list[str]) -> list[str]:
    repo_ids = []
    for top_dir in top_level_dirs:
        top_path = root / top_dir
        if not top_path.exists():
            raise FileNotFoundError(f"데이터셋 경로 없음: {top_path}")
        tasks = sorted(
            p.name for p in top_path.iterdir()
            if p.is_dir() and p.name.startswith("task_")
            and (p / "meta" / "info.json").exists()
            and (p / "meta" / "episodes").exists()
        )
        skipped = sum(
            1 for p in top_path.iterdir()
            if p.is_dir() and p.name.startswith("task_")
            and not ((p / "meta" / "info.json").exists() and (p / "meta" / "episodes").exists())
        )
        if skipped:
            logger.warning("%s: %d개 task에 meta 없음 → 건너뜀", top_dir, skipped)
        if not tasks:
            raise RuntimeError(f"유효한 task 없음: {top_path}")
        repo_ids.extend(f"{top_dir}/{t}" for t in tasks)
    return repo_ids


@dataclass
class GrootDiTOnlyTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="",
            root="/home/seonho/slicing_robocasa_human_v3",
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    steps: int = 40_000
    batch_size: int = 64
    num_workers: int = 8
    log_freq: int = 100
    save_freq: int = 10_000
    seed: int = 42
    use_policy_training_preset: bool = False

    gradient_checkpointing: bool = False
    resume: bool = False
    data_split: str = "pretrain"  # "pretrain", "target", "all"

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_dit_only"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_dit_only")


@parser.wrap()
def main(cfg: GrootDiTOnlyTrainConfig) -> None:
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

    dataset_root = Path(cfg.dataset.root)

    if cfg.data_split == "pretrain":
        top_level_dirs = PRETRAIN_DIRS
    elif cfg.data_split == "target":
        top_level_dirs = TARGET_DIRS
    elif cfg.data_split == "all":
        top_level_dirs = ALL_DIRS
    else:
        raise ValueError(f"Unknown data_split: {cfg.data_split!r} (pretrain / target / all)")

    repo_ids = get_task_repo_ids(dataset_root, top_level_dirs)

    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: %s", cfg.output_dir)
        logger.info("Accelerator: num_processes=%d, device=%s", accelerator.num_processes, device)
        logger.info("Dataset root: %s | split=%s | tasks=%d", dataset_root, cfg.data_split, len(repo_ids))

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
            dataset.num_frames, dataset.num_episodes, len(repo_ids),
        )
        if dataset.disabled_features:
            logger.warning("비활성화된 features (task 간 불일치): %s", dataset.disabled_features)

    use_wandb = cfg.wandb.enable and accelerator.is_main_process
    if use_wandb:
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
                "top_level_dirs": top_level_dirs,
                "dataset_root": str(dataset_root),
                "steps": cfg.steps,
                "lr": cfg.policy.optimizer_lr,
                "batch_size": cfg.batch_size,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": cfg.batch_size * accelerator.num_processes,
                "mgd_trainable_mode": cfg.policy.mgd_trainable_mode,
                "mgd_enabled": cfg.policy.mgd_enabled,
                "lora_rank": cfg.policy.lora_rank,
                "lora_alpha": cfg.policy.lora_alpha,
                "lora_dropout": cfg.policy.lora_dropout,
                "lora_target": cfg.policy.lora_target,
                "groot_pretrained_path": cfg.policy.groot_pretrained_path,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "seed": cfg.seed,
                "total_frames": dataset.num_frames,
                "total_episodes": dataset.num_episodes,
            },
            save_code=False,
        )
        logger.info("WandB initialized: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    pre, post = make_pre_post_processors(cfg.policy, dataset_stats=dataset.stats)
    # make_policy 안에서 groot_pretrained_path가 GR00TN15에 직접 로드됨
    # (backbone LoRA weights + action_head weights 포함)
    policy = make_policy(cfg.policy, ds_meta=first_ds_meta)

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
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info("num_learnable_params=%s", f"{num_learnable_params:,}")
        logger.info("num_total_params=%s", f"{num_total_params:,}")
        logger.info("trainable ratio=%.4f%%", 100 * num_learnable_params / max(num_total_params, 1))

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
                preprocessor=pre,
                postprocessor=post,
            )
            update_last_checkpoint(checkpoint_dir)
            logger.info("체크포인트 저장: %s", checkpoint_dir)

    policy.train()
    data_stream = infinite_dataloader()

    for step in range(start_step + 1, cfg.steps + 1):
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
                wandb_log = {
                    "train/loss": loss_dict.get("loss", loss.item()),
                    "train/flow_matching_loss": loss_dict.get(
                        "flow_matching_loss", loss_dict.get("loss", loss.item())
                    ),
                    "train/lr": lr_now,
                    "train/grad_norm": (
                        grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                    ),
                }
                wandb.log(wandb_log, step=step)

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
        logger.info(
            "학습 완료. 최종 체크포인트: %s",
            get_step_checkpoint_dir(cfg.output_dir, cfg.steps, cfg.steps),
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
