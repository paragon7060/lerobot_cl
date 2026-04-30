#!/usr/bin/env python
"""GR00T-CL v2 Phase 1 학습 스크립트.

VLM backbone + vision tower를 완전히 freeze하고,
action expert (MultiEmbodimentActionEncoder + DiT)만 flow matching loss로 학습.
index 6 (wrist joint)에 3배 가중치를 주어 wrist 상태 구분 능력을 강화한다.

단일 GPU:
  python scripts/train_groot_cl_v2_phase1.py \
      --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
      --dataset.root=/home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
      --dataset.video_backend=pyav \
      --policy.type=groot_cl_v2 \
      --output_dir=./outputs/groot_cl_v2_phase1 \
      --job_name=groot_cl_v2_phase1 \
      --phase1_steps=3000 \
      --wandb.enable=true \
      --wandb.project=groot_cl_v2 \
      --wandb.entity=RwHlabs

Multi-GPU:
  accelerate launch --num_processes 2 scripts/train_groot_cl_v2_phase1.py [same args]
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.groot_cl_v2.configuration_groot_cl_v2 import GrootCLv2Config
from lerobot.policies.groot_cl_v2.modeling_groot_cl_v2 import GrootCLv2Policy
from lerobot.policies.groot_cl.processor_groot import make_groot_pre_post_processors
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)

import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ── Dataset-specific settings ──────────────────────────────────────────────────
# Only use these three cameras (exclude semantic cameras)
KEEP_CAMERAS = {
    "observation.images.wrist",
    "observation.images.guide",
    "observation.images.right_shoulder",
}

# Task prompt mapping: scene_name → instruction text
SCENE_TASK_PROMPT = {
    "1ext": "Open the cabinet following the guide.",
    "3a": "Open the door following the guide.",
    "3b": "Open the door following the guide.",
    "3c": "Open the door following the guide.",
    "3d": "Open the door following the guide.",
    "5a": "Open the bottle following the guide.",
    "5b": "Open the bottle following the guide.",
    "5c": "Open the bottle following the guide.",
    "5d": "Close the bottle following the guide.",
    "5e": "Close the bottle following the guide.",
    "5f": "Close the bottle following the guide.",
    "5g": "Open the bottle following the guide.",
    "5h": "Close the bottle cap following the guide.",
}


def build_task_index_to_prompt(dataset_root: str | Path) -> dict[int, str]:
    """tasks.parquet의 task_index(numeric) → prompt text 매핑 생성."""
    parquet_path = Path(dataset_root) / "meta" / "tasks.parquet"
    df = pd.read_parquet(parquet_path)
    # DataFrame index = scene_name (e.g. "1ext"), column "task_index" = numeric int
    mapping = {}
    for scene_name, row in df.iterrows():
        idx = int(row["task_index"])
        prompt = SCENE_TASK_PROMPT.get(str(scene_name), "Perform the task.")
        mapping[idx] = prompt
    return mapping


@dataclass
class GrootCLv2Phase1Config(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="paragon7060/INSIGHTfixposV3",
            root=None,
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    phase1_steps: int = 50_000
    batch_size: int = 4
    num_workers: int = 4
    log_freq: int = 50
    seed: int = 42
    use_policy_training_preset: bool = False

    lr: float = 1e-4
    warmup_steps: int = 0

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_cl_v2_phase1"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_cl_v2_phase1")
        self.steps = self.phase1_steps


@parser.wrap()
def main(cfg: GrootCLv2Phase1Config) -> None:
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
        logger.info("device=%s, num_processes=%d", device, accelerator.num_processes)

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
                "groot_pretrained_path": getattr(cfg.policy, "groot_pretrained_path", None),
                "phase1_steps": cfg.phase1_steps,
                "lr": cfg.lr,
                "warmup_steps": cfg.warmup_steps,
                "batch_size": cfg.batch_size,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": cfg.batch_size * accelerator.num_processes,
                "joint_fm_weights": cfg.policy.joint_fm_weights,
                "seed": cfg.seed,
            },
            save_code=False,
        )
        logger.info("WandB: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    # Force phase1
    cfg.policy.cl_v2_phase = "phase1"

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

    pre, _ = make_groot_pre_post_processors(cfg.policy, dataset_stats=ds_meta.stats)

    # Build task_index(int) → prompt text mapping from tasks.parquet
    task_index_to_prompt: dict[int, str] = {}
    if cfg.dataset.root is not None:
        task_index_to_prompt = build_task_index_to_prompt(cfg.dataset.root)
        if accelerator.is_main_process:
            logger.info("Task prompt mapping (%d tasks): %s", len(task_index_to_prompt), task_index_to_prompt)

    policy: GrootCLv2Policy = make_policy(cfg.policy, ds_meta=ds_meta)

    accelerator.wait_for_everyone()
    policy, dataloader = accelerator.prepare(policy, dataloader)

    if accelerator.is_main_process:
        raw_policy = accelerator.unwrap_model(policy)
        num_learnable = sum(p.numel() for p in raw_policy.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in raw_policy.parameters())
        logger.info(
            "Trainable params: %s / %s (%.2f%%)",
            f"{num_learnable:,}", f"{num_total:,}", 100 * num_learnable / max(num_total, 1),
        )
        logger.info(
            "Joint FM weights (index 6 = wrist ×3): %s",
            raw_policy._joint_weights.cpu().tolist()[:8],
        )
        logger.info(
            "Cameras used: %s", sorted(KEEP_CAMERAS)
        )

    def preprocess(raw_batch: dict) -> dict:
        # 1. Filter cameras: keep only KEEP_CAMERAS, drop the rest (including semantic)
        for key in list(raw_batch.keys()):
            if key.startswith("observation.images.") and key not in KEEP_CAMERAS:
                del raw_batch[key]

        # 2. Inject task prompt from task_index mapping (per-sample, list of strings)
        if task_index_to_prompt and "task_index" in raw_batch:
            indices = raw_batch["task_index"]
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            raw_batch["task"] = [
                task_index_to_prompt.get(int(idx), "Perform the task.")
                for idx in indices
            ]

        return pre(raw_batch)

    def infinite_dataloader():
        while True:
            yield from dataloader

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, accelerator.unwrap_model(policy).parameters()),
        lr=cfg.lr,
        betas=cfg.policy.optimizer_betas,
        eps=cfg.policy.optimizer_eps,
        weight_decay=cfg.policy.optimizer_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.phase1_steps,
    )
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    policy.train()
    data_stream = infinite_dataloader()

    for step in range(1, cfg.phase1_steps + 1):
        raw_batch = next(data_stream)
        batch = preprocess(raw_batch)

        loss, loss_dict = policy(batch)

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if accelerator.is_main_process and step % cfg.log_freq == 0:
            lr_now = scheduler.get_last_lr()[0]
            msg = (
                f"[phase1] step={step}/{cfg.phase1_steps} | lr={lr_now:.2e} | "
                f"grad_norm={grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm:.3f} | "
                f"fm_loss={loss_dict.get('flow_matching_loss', loss.item()):.4f}"
            )
            print(msg, flush=True)
            logger.info(msg)
            if use_wandb:
                wandb.log(
                    {
                        "phase1/loss": loss_dict.get("flow_matching_loss", loss.item()),
                        "phase1/lr": lr_now,
                        "phase1/grad_norm": (
                            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                        ),
                    },
                    step=step,
                )

        if step == cfg.phase1_steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.phase1_steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    preprocessor=pre,
                    postprocessor=None,
                )
                update_last_checkpoint(checkpoint_dir)
                logger.info("체크포인트 저장: %s", checkpoint_dir)

    if accelerator.is_main_process:
        logger.info("Phase 1 학습 완료. 최종 체크포인트: %s", cfg.output_dir)
        logger.info(
            "다음 단계: eval_action_features.py 로 action feature 품질 확인 후 Phase 2 진행."
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
