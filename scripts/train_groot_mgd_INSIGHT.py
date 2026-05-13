#!/usr/bin/env python3
"""GROOT processed-MGD 학습 스크립트 (INSIGHT 단일 데이터셋 전용).

`Whalswp/INSIGHTfixposV3_EE_quat` 데이터셋 학습을 기본으로 하며,
policy.type은 `groot_processed_mgd`를 사용하도록 강제한다.

단일 GPU 실행 예:
  python scripts/train_groot_mgd_INSIGHT.py \
      --dataset.repo_id=Whalswp/INSIGHTfixposV3_EE_quat \
      --dataset.root=/home/seonho/workspace/data/Whalswp/INSIGHTfixposV3_EE_quat \
      --dataset.video_backend=pyav \
      --policy.type=groot_processed_mgd \
      --policy.groot_pretrained_path=/home/seonho/ws3/outputs/groot_inst/checkpoints/050000/pretrained_model \
      --policy.mgd_trainable_mode=processed_only \
      --policy.mgd_token_mask_ratio=0.1 \
      --policy.mgd_sequence_hidden_dim=512 \
      --policy.mgd_loss_weight=0.05 \
      --policy.mgd_backprop_backbone=true \
      --policy.lora_rank=0 \
      --policy.tune_visual=false \
      --policy.tune_llm=false \
      --output_dir=./outputs/groot_processed_mgd_insight \
      --steps=100000 \
      --batch_size=64 \
      --log_freq=50 \
      --save_freq=20000 \
      --wandb.enable=true \
      --wandb.project=groot_processed_mgd \
      --wandb.entity=RwHlabs
"""

import logging
import random
from copy import deepcopy
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


def slice_feature_dims(raw_batch: dict, action_dim: int, state_dim: int) -> dict:
    """Slice zero-padded tail and keep only effective action/state dims."""
    if "action" in raw_batch and action_dim is not None:
        raw_batch["action"] = raw_batch["action"][..., :action_dim]
    if "observation.state" in raw_batch and state_dim is not None:
        raw_batch["observation.state"] = raw_batch["observation.state"][..., :state_dim]
    return raw_batch


def slice_dataset_stats(stats: dict | None, action_dim: int, state_dim: int) -> dict | None:
    """Slice action/state stats to match effective dims."""
    if not stats:
        return stats
    stats = deepcopy(stats)

    def _trim(sub: dict, dim: int) -> dict:
        return {
            k: v[..., :dim].contiguous()
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[-1] > dim
            else v
            for k, v in sub.items()
        }

    if "action" in stats and action_dim is not None:
        stats["action"] = _trim(stats["action"], action_dim)
    if "observation.state" in stats and state_dim is not None:
        stats["observation.state"] = _trim(stats["observation.state"], state_dim)
    return stats


@dataclass
class GrootMGDTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="Whalswp/INSIGHTfixposV3_EE_quat",
            root=None,
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    steps: int = 100_000
    batch_size: int = 64
    num_workers: int = 8
    log_freq: int = 100
    save_freq: int = 10_000
    seed: int = 42
    use_policy_training_preset: bool = False

    gradient_checkpointing: bool = False
    resume: bool = False
    pretrained_path: str = ""
    action_dim: int = 8
    state_dim: int = 16

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_mgd_insight"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_mgd_insight")


@parser.wrap()
def main(cfg: GrootMGDTrainConfig) -> None:
    if cfg.policy.type != "groot_processed_mgd":
        raise ValueError(
            "train_groot_mgd_INSIGHT.py는 policy.type=groot_processed_mgd 전용입니다. "
            f"현재 값: {cfg.policy.type!r}"
        )

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
            resume="allow" if cfg.resume else None,
            dir=str(cfg.output_dir),
            config={
                "repo_id": cfg.dataset.repo_id,
                "dataset_root": str(cfg.dataset.root) if cfg.dataset.root is not None else None,
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
                "mgd_enabled": cfg.policy.mgd_enabled,
                "mgd_trainable_mode": cfg.policy.mgd_trainable_mode,
                "mgd_target_pooling": cfg.policy.mgd_target_pooling,
                "mgd_target_projection": cfg.policy.mgd_target_projection,
                "mgd_target_dim": cfg.policy.mgd_target_dim,
                "mgd_hidden_dim": cfg.policy.mgd_hidden_dim,
                "mgd_mask_ratio": cfg.policy.mgd_mask_ratio,
                "mgd_token_mask_ratio": cfg.policy.mgd_token_mask_ratio,
                "mgd_sequence_hidden_dim": cfg.policy.mgd_sequence_hidden_dim,
                "mgd_loss_weight": cfg.policy.mgd_loss_weight,
                "mgd_fm_loss_weight": cfg.policy.mgd_fm_loss_weight,
                "mgd_backprop_backbone": cfg.policy.mgd_backprop_backbone,
                "vlm_drift_logging_enabled": cfg.policy.vlm_drift_logging_enabled,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "seed": cfg.seed,
                "action_dim": cfg.action_dim,
                "state_dim": cfg.state_dim,
            },
            save_code=False,
        )
        logger.info("WandB initialized: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    ds_meta = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, root=cfg.dataset.root)
    if "action" in ds_meta.features:
        ds_meta.features["action"]["shape"] = (cfg.action_dim,)
    if "observation.state" in ds_meta.features:
        ds_meta.features["observation.state"]["shape"] = (cfg.state_dim,)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    sliced_stats = slice_dataset_stats(ds_meta.stats, cfg.action_dim, cfg.state_dim)

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

    pre, post = make_pre_post_processors(cfg.policy, dataset_stats=sliced_stats)
    policy = make_policy(cfg.policy, ds_meta=ds_meta)

    if cfg.pretrained_path:
        from safetensors.torch import load_model as safetensors_load_model

        ckpt_model = Path(cfg.pretrained_path) / "model.safetensors"
        if not ckpt_model.exists():
            ckpt_model = Path(cfg.pretrained_path) / "pretrained_model" / "model.safetensors"
        safetensors_load_model(policy, str(ckpt_model))
        logger.info("Loaded pretrained weights from %s", ckpt_model)

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

        if use_wandb:
            wandb.summary["params/whole_total"] = int(num_total_params)
            wandb.summary["params/whole_trainable"] = int(num_learnable_params)

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
        raw_batch = slice_feature_dims(raw_batch, cfg.action_dim, cfg.state_dim)
        if accelerator.is_main_process and step == start_step + 1:
            action_shape = tuple(raw_batch["action"].shape) if "action" in raw_batch else None
            state_shape = (
                tuple(raw_batch["observation.state"].shape)
                if "observation.state" in raw_batch
                else None
            )
            logger.info(
                "First sliced batch shapes: action=%s, observation.state=%s",
                action_shape,
                state_shape,
            )
        batch = pre(raw_batch)
        batch["compute_vlm_drift"] = step % cfg.log_freq == 0 and cfg.policy.vlm_drift_logging_enabled

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
                step,
                cfg.steps,
                lr_now,
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
                for k in (
                    "mgd_loss",
                    "mgd_cos_sim",
                    "mgd_mask_ratio",
                    "mgd_token_mask_ratio_cfg",
                    "mgd_target_norm_raw",
                    "mgd_pred_norm_raw",
                    "mgd_target_norm_post",
                    "mgd_pred_norm_post",
                    "valid_token_count",
                    "kept_token_count",
                    "actual_token_mask_ratio",
                    "vlm_drift_cos",
                    "vlm_drift_l2",
                ):
                    if k in loss_dict:
                        wandb_log[f"train/{k}"] = loss_dict[k]
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
