#!/usr/bin/env python
"""GR00T Contrastive Learning 학습 스크립트 (accelerate + lerobot parser CLI + wandb 지원).

Phase 1: contrastive heads warm-up (_groot_model frozen)
Phase 2a: joint fine-tuning (LoRA + projector + diffusion + contrastive heads, warmup scheduler)

단일 GPU 실행:
  python scripts/train_groot_cl.py \
      --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
      --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
      --dataset.video_backend=pyav \
      --output_dir=./outputs/groot_cl \
      --job_name=groot_cl_run1 \
      --phase1_steps=500 \
      --phase2a_steps=5000 \
      --policy.lora_target=vision \
      --wandb.enable=true \
      --wandb.project=groot_cl \
      --wandb.entity=RwHlabs \
      --wandb.disable_artifact=true

Multi-GPU 실행:
  accelerate launch --num_processes 2 scripts/train_groot_cl.py [same args]

주의: negative_action은 preprocess() 내에서 NegativeActionNormalizeStep을 직접 호출하여
      수동으로 정규화한다. pipeline(pre)은 make_groot_pre_post_processors에서 생성한다.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import get_cosine_schedule_with_warmup

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.contrastive_dataset import ContrastiveLeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_policy
from lerobot.policies.groot_cl.configuration_groot_cl import GrootCLConfig
from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
from lerobot.policies.groot_cl.processor_groot import make_groot_pre_post_processors
from lerobot.policies.groot_cl.processor_groot_cl import NegativeActionNormalizeStep
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
class GrootCLTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="paragon7060/INSIGHTfixposV3",
            root=None,
            video_backend="pyav",
        )
    )
    policy: GrootCLConfig = field(default_factory=GrootCLConfig)

    steps: int = 17_000
    batch_size: int = 4
    num_workers: int = 4
    log_freq: int = 50
    save_freq: int = 500
    seed: int = 42
    use_policy_training_preset: bool = False

    neg_pairs_path: str = "/home/bluepot/cl_ws/negative_pairs.json"
    phase1_steps: int = 2_000
    phase2a_steps: int = 15_000
    phase1_lr: float = 1e-4
    phase2a_lr: float = 2e-5
    phase2a_warmup_steps: int = 500
    phase2a_loss_weight: float = 0.05

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_cl"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_cl")
        self.steps = self.phase1_steps + self.phase2a_steps


@parser.wrap()
def main(cfg: GrootCLTrainConfig) -> None:
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
                "groot_pretrained_path": cfg.policy.groot_pretrained_path,
                "phase1_steps": cfg.phase1_steps,
                "phase2a_steps": cfg.phase2a_steps,
                "phase2a_warmup_steps": cfg.phase2a_warmup_steps,
                "phase1_lr": cfg.phase1_lr,
                "phase2a_lr": cfg.phase2a_lr,
                "phase2a_loss_weight": cfg.phase2a_loss_weight,
                "batch_size": cfg.batch_size,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": cfg.batch_size * accelerator.num_processes,
                "contrastive_latent_dim": cfg.policy.contrastive_latent_dim,
                "contrastive_triplet_margin": cfg.policy.contrastive_triplet_margin,
                "tune_llm": cfg.policy.tune_llm,
                "tune_visual": cfg.policy.tune_visual,
                "tune_projector": cfg.policy.tune_projector,
                "tune_diffusion_model": cfg.policy.tune_diffusion_model,
                "lora_rank": cfg.policy.lora_rank,
                "lora_alpha": cfg.policy.lora_alpha,
                "lora_dropout": cfg.policy.lora_dropout,
                "lora_target": cfg.policy.lora_target,
                "seed": cfg.seed,
            },
            save_code=False,
        )
        logger.info("WandB initialized: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    cfg.policy.contrastive_phase = "phase1"

    ds_meta = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, root=cfg.dataset.root)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if accelerator.is_main_process:
        dataset = ContrastiveLeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            negative_pairs_path=cfg.neg_pairs_path,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = ContrastiveLeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            negative_pairs_path=cfg.neg_pairs_path,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
        )

    if accelerator.is_main_process:
        logger.info("Dataset: %d frames, %d episodes", dataset.num_frames, dataset.num_episodes)

    def collate_fn(batch: list[dict]) -> dict:
        neg_actions = [item.pop("negative_action", None) for item in batch]
        result = default_collate(batch)
        if all(isinstance(n, torch.Tensor) for n in neg_actions):
            result["negative_action"] = torch.stack(neg_actions)
        else:
            result["negative_action"] = None
        return result

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    pre, _ = make_groot_pre_post_processors(cfg.policy, dataset_stats=ds_meta.stats)
    neg_normalizer = NegativeActionNormalizeStep(stats=ds_meta.stats, normalize_min_max=True)

    policy: GrootCLPolicy = make_policy(cfg.policy, ds_meta=ds_meta)

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

    accelerator.wait_for_everyone()
    policy, dataloader = accelerator.prepare(policy, dataloader)

    if accelerator.is_main_process:
        logger.info("Policy: %s", policy.__class__.__name__)

    def preprocess(raw_batch: dict) -> dict:
        raw_neg = raw_batch.pop("negative_action", None)
        processed = pre(raw_batch)
        if isinstance(raw_neg, torch.Tensor):
            result = neg_normalizer({"negative_action": raw_neg})
            processed["negative_action"] = result["negative_action"].to(device)
        return processed

    def infinite_dataloader():
        while True:
            yield from dataloader

    def _save(step: int, optimizer, scheduler) -> None:
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

    def train_loop(
        phase: str,
        total_steps: int,
        lr: float,
        loss_weight: float,
        backprop_backbone: bool,
        warmup_steps: int = 0,
        global_step_offset: int = 0,
    ) -> int:
        if accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info(
                "Phase: %s | steps=%d | lr=%.2e | loss_weight=%.3f | backprop_backbone=%s | warmup_steps=%d",
                phase, total_steps, lr, loss_weight, backprop_backbone, warmup_steps,
            )
            logger.info("=" * 60)

        raw_policy = accelerator.unwrap_model(policy)
        raw_policy.set_contrastive_phase(phase)
        raw_policy.config.contrastive_loss_weight = loss_weight
        raw_policy.config.contrastive_backprop_backbone = backprop_backbone

        if accelerator.is_main_process:
            num_learnable = sum(p.numel() for p in raw_policy.parameters() if p.requires_grad)
            num_total = sum(p.numel() for p in raw_policy.parameters())
            logger.info(
                "trainable params: %s / %s (%.2f%%)",
                f"{num_learnable:,}", f"{num_total:,}", 100 * num_learnable / max(num_total, 1),
            )
            if cfg.policy.lora_rank > 0:
                lora_params = sum(
                    p.numel() for n, p in raw_policy.named_parameters()
                    if p.requires_grad and "lora_" in n
                )
                logger.info("LoRA [%s] adapter params: %s", cfg.policy.lora_target, f"{lora_params:,}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, raw_policy.parameters()),
            lr=lr,
            betas=cfg.policy.optimizer_betas,
            eps=cfg.policy.optimizer_eps,
            weight_decay=cfg.policy.optimizer_weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        policy.train()
        data_stream = infinite_dataloader()

        for step in range(1, total_steps + 1):
            raw_batch = next(data_stream)
            batch = preprocess(raw_batch)

            loss, loss_dict = policy(batch)

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step = global_step_offset + step

            if accelerator.is_main_process and step % cfg.log_freq == 0:
                lr_now = scheduler.get_last_lr()[0]
                log_str = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                logger.info(
                    "[%s] step=%d/%d (global=%d) | lr=%.2e | grad_norm=%.3f | %s",
                    phase, step, total_steps, global_step, lr_now,
                    grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    log_str,
                )

                if use_wandb:
                    wandb.log(
                        {
                            f"{phase}/loss": loss_dict.get("loss", loss.item()),
                            f"{phase}/flow_matching_loss": loss_dict.get("flow_matching_loss", 0.0),
                            f"{phase}/contrastive_loss": loss_dict.get("contrastive_loss", 0.0),
                            f"{phase}/lr": lr_now,
                            f"{phase}/grad_norm": (
                                grad_norm.item()
                                if isinstance(grad_norm, torch.Tensor)
                                else float(grad_norm)
                            ),
                        },
                        step=global_step,
                    )

            if step % cfg.save_freq == 0 or step == total_steps:
                accelerator.wait_for_everyone()
                _save(global_step, optimizer, scheduler)

                if use_wandb and not cfg.wandb.disable_artifact:
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, global_step)
                    artifact = wandb.Artifact(
                        name=f"{cfg.job_name}-{phase}-step{step:06d}",
                        type="model",
                        description=f"{phase} checkpoint at step {step}",
                    )
                    artifact.add_dir(str(checkpoint_dir))
                    wandb.log_artifact(artifact)

        if accelerator.is_main_process:
            logger.info("[%s] 완료", phase)

        return global_step_offset + total_steps

    global_step = train_loop(
        phase="phase1",
        total_steps=cfg.phase1_steps,
        lr=cfg.phase1_lr,
        loss_weight=1.0,
        backprop_backbone=False,
        warmup_steps=0,
        global_step_offset=0,
    )

    train_loop(
        phase="phase2a",
        total_steps=cfg.phase2a_steps,
        lr=cfg.phase2a_lr,
        loss_weight=cfg.phase2a_loss_weight,
        backprop_backbone=True,
        warmup_steps=cfg.phase2a_warmup_steps,
        global_step_offset=global_step,
    )

    if accelerator.is_main_process:
        logger.info("전체 학습 완료. 최종 체크포인트: %s", cfg.output_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
