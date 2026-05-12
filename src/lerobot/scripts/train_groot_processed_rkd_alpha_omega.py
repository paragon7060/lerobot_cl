#!/usr/bin/env python
"""GROOT-Processed-RKD 학습 스크립트 (official util 경로, alpha-omega variant).

MGD 학습 스크립트와 동일한 실행/저장 흐름을 유지하고,
policy.type=groot_processed_rkd + cl_v2_* 옵션을 사용한다.
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
from lerobot.policies.groot_common import (
    LeRobotNativeBatchBuilder,
    discover_robocasa_official_runtime_repos,
    make_robocasa_preset,
    tensor_report,
)
from lerobot.policies.groot_common.filter_key_subset import load_or_create_subset_manifest
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


@dataclass
class GrootProcessedRKDTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="",
            root="/home/seonho/groot_robocasa/robocasa_dataset/robocasa_human_v3",
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
    data_split: str = "pretrain"  # "pretrain" or "target"
    filter_key: str | None = "100_demos"  # official-style subset size, e.g. "100_demos"
    subset_seed: int | None = 0
    pretrained_path: str = ""

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_processed_rkd"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_processed_rkd")


@parser.wrap()
def main(cfg: GrootProcessedRKDTrainConfig) -> None:
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

    if cfg.data_split not in ("pretrain", "target"):
        raise ValueError(f"Unknown data_split: {cfg.data_split!r} (pretrain / target)")
    discovery = discover_robocasa_official_runtime_repos(dataset_root, split=cfg.data_split)
    repo_ids = discovery.repo_ids

    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: %s", cfg.output_dir)
        logger.info("Accelerator: num_processes=%d, device=%s", accelerator.num_processes, device)
        logger.info("Dataset root: %s | split=%s | tasks=%d", dataset_root, cfg.data_split, len(repo_ids))
        logger.info(
            "Resolved subset args: filter_key=%s | subset_seed=%s | output_dir=%s",
            cfg.filter_key,
            cfg.subset_seed,
            cfg.output_dir,
        )

    episodes_by_repo = None
    subset_manifest = None
    subset_manifest_source = None
    if cfg.filter_key:
        if accelerator.is_main_process:
            episodes_by_repo, subset_manifest, subset_manifest_source = load_or_create_subset_manifest(
                dataset_root=dataset_root,
                repo_ids=repo_ids,
                split=cfg.data_split,
                filter_key=cfg.filter_key,
                seed=cfg.subset_seed,
                create_if_missing=True,
            )
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            episodes_by_repo, subset_manifest, subset_manifest_source = load_or_create_subset_manifest(
                dataset_root=dataset_root,
                repo_ids=repo_ids,
                split=cfg.data_split,
                filter_key=cfg.filter_key,
                seed=cfg.subset_seed,
                create_if_missing=False,
            )
        if accelerator.is_main_process:
            logger.info(
                "Subset manifest: %s (%s) | filter_key=%s | seed=%s",
                subset_manifest,
                subset_manifest_source,
                cfg.filter_key,
                cfg.subset_seed,
            )
            if subset_manifest_source == "cache":
                logger.info("Subset manifest 재사용됨(JSON cache hit): %s", subset_manifest)
            else:
                logger.info("Subset manifest 새로 생성됨(JSON cache miss): %s", subset_manifest)
            if episodes_by_repo:
                preview = {
                    repo_id: (None if episodes is None else len(episodes))
                    for repo_id, episodes in list(episodes_by_repo.items())[:5]
                }
                logger.info("Episode subset preview (first 5 repos): %s", preview)
                logger.info("Episode subset enabled: yes (official-style filtering active)")
            else:
                logger.info("Episode subset enabled: no (full repo loading)")

    first_ds_meta = LeRobotDatasetMetadata(
        repo_id=repo_ids[0],
        root=dataset_root / repo_ids[0],
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, first_ds_meta)

    if accelerator.is_main_process:
        dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            root=dataset_root,
            episodes=episodes_by_repo,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            root=dataset_root,
            episodes=episodes_by_repo,
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
                "filter_key": cfg.filter_key,
                "subset_seed": cfg.subset_seed,
                "subset_manifest": str(subset_manifest) if subset_manifest else None,
                "dataset_root": str(dataset_root),
                "steps": cfg.steps,
                "lr": cfg.policy.optimizer_lr,
                "batch_size": cfg.batch_size,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": cfg.batch_size * accelerator.num_processes,
                "tune_llm": cfg.policy.tune_llm,
                "tune_visual": cfg.policy.tune_visual,
                "tune_projector": cfg.policy.tune_projector,
                "tune_diffusion_model": cfg.policy.tune_diffusion_model,
                "cl_v2_phase": cfg.policy.cl_v2_phase,
                "cl_v2_trainable_mode": cfg.policy.cl_v2_trainable_mode,
                "cl_v2_student_repr": cfg.policy.cl_v2_student_repr,
                "cl_v2_action_repr": cfg.policy.cl_v2_action_repr,
                "cl_v2_loss_weight": cfg.policy.cl_v2_loss_weight,
                "cl_v2_fm_loss_weight": cfg.policy.cl_v2_fm_loss_weight,
                "cl_v2_action_temp": cfg.policy.cl_v2_action_temp,
                "cl_v2_vlm_temp": cfg.policy.cl_v2_vlm_temp,
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
    batch_builder = LeRobotNativeBatchBuilder(
        preset=make_robocasa_preset(video_backend=cfg.dataset.video_backend)
    )
    policy = make_policy(cfg.policy, ds_meta=first_ds_meta)

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
        batch = batch_builder.build_train_batch(raw_batch, pre)
        if step == start_step + 1 and accelerator.is_main_process:
            parity_view = batch_builder.build_parity_view(batch)
            logger.info("Batch parity-view keys: %s", sorted(parity_view.keys()))
            logger.info("Batch tensor report: %s", tensor_report(parity_view))

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
                    "rkd_loss",
                    "teacher_student_cos",
                    "teacher_entropy",
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
