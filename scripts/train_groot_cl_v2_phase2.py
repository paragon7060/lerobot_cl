#!/usr/bin/env python
"""GR00T-CL v2 Phase 2 학습 스크립트 — Relational Knowledge Distillation (RKD).

Action Encoder (Teacher, frozen)의 pairwise 관계 구조를 VLM (Student)이 따라가도록 finetuning.
Occlusion으로 인해 같은 시각 상태에서 다른 방향(joint 7 좌/우)이 나오는 ambiguity를 해결.

RKD Loss (CVPR 2019):
  z_a = L2_norm( pool( ActionEncoder(action, t=999) ) )   ← Teacher (frozen)
  z_v = VLMProjector( pool( VLMBackbone(obs) ) )          ← Student (trainable)
  S_a = z_a @ z_a.T,  P_a = softmax(S_a / τ_act)
  S_v = z_v @ z_v.T
  L_RKD = KL( P_a ‖ softmax(S_v / τ_vlm) )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
빠른 시작 (Phase 1 완료 후):

  python scripts/train_groot_cl_v2_phase2.py \\
      --dataset.repo_id=paragon7060/INSIGHTfixposV3 \\
      --dataset.root=/path/to/data \\
      --dataset.video_backend=pyav \\
      --policy.type=groot_cl_v2 \\
      --policy.groot_pretrained_path=./outputs/groot_cl_v2_phase1/checkpoints/050000 \\
      --dataset_config=scripts/dataset_configs/INSIGHTfixposV3.json \\
      --output_dir=./outputs/groot_cl_v2_phase2 \\
      --job_name=groot_cl_v2_phase2_rkd \\
      --phase2_steps=10000 \\
      --batch_size=32 \\
      --wandb.enable=true \\
      --wandb.project=groot_cl_v2 \\
      --wandb.entity=YOUR_ENTITY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RKD 하이퍼파라미터 (policy config):
  --policy.cl_v2_action_temp=0.1   # Teacher 분포 sharpness (↓ = harder positive)
  --policy.cl_v2_vlm_temp=0.07     # Student logit temperature
  --policy.cl_v2_loss_weight=0.1   # RKD loss weight
  --policy.cl_v2_fm_loss_weight=0.01  # FM loss weight (quality 유지용)
"""

import json
import logging
import random
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# ── Dataset Config Helpers (phase1과 동일) ────────────────────────────────────

def load_dataset_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def build_task_index_to_prompt(
    dataset_root: str | Path,
    task_prompts: dict[str, str],
) -> dict[int, str]:
    """tasks.parquet 읽어서 task_index(int) → prompt text 매핑 생성.

    우선순위:
    1. task_prompts (dataset_config JSON) 가 있으면 항상 우선 사용
    2. tasks.parquet의 'task' column (표준 LeRobot 형식) 사용
    3. tasks.parquet의 'task_index' column만 있으면 default prompt 사용
    """
    parquet_path = Path(dataset_root) / "meta" / "tasks.parquet"
    if not parquet_path.exists():
        return {}

    df = pd.read_parquet(parquet_path)

    # 1순위: JSON task_prompts (항상 override)
    if task_prompts:
        mapping = {}
        for scene_name, row in df.iterrows():
            idx = int(row["task_index"]) if "task_index" in df.columns else int(scene_name)
            prompt = task_prompts.get(str(scene_name), "Perform the task.")
            mapping[idx] = prompt
        return mapping

    # 2순위: tasks.parquet의 'task' column (표준 LeRobot 형식)
    if "task" in df.columns:
        mapping = {}
        for idx, row in df.iterrows():
            task_idx = int(row.get("task_index", idx))
            mapping[task_idx] = str(row["task"])
        return mapping

    # 3순위: task_index만 있고 task text 없음 → default
    if "task_index" in df.columns:
        return {int(row["task_index"]): "Perform the task." for _, row in df.iterrows()}

    return {}


# ── Training Config ────────────────────────────────────────────────────────────

@dataclass
class GrootCLv2Phase2Config(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="",
            root=None,
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    # 학습 하이퍼파라미터
    phase2_steps: int = 10_000
    batch_size: int = 32
    num_workers: int = 4
    log_freq: int = 50
    seed: int = 42
    use_policy_training_preset: bool = False
    lr: float = 2e-5
    warmup_steps: int = 500

    # Dataset 설정
    dataset_config: str | None = None

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_cl_v2_phase2"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_cl_v2_phase2")
        self.steps = self.phase2_steps


# ── Main ───────────────────────────────────────────────────────────────────────

@parser.wrap()
def main(cfg: GrootCLv2Phase2Config) -> None:
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

    # ── Dataset config 로드 ────────────────────────────────────────────────────
    ds_cfg = load_dataset_config(cfg.dataset_config)
    keep_cameras: set[str] = set(ds_cfg.get("cameras", []))
    task_prompts: dict[str, str] = ds_cfg.get("task_prompts", {})

    if accelerator.is_main_process:
        logger.info("Dataset config: %s", cfg.dataset_config or "(none)")
        logger.info("Keep cameras: %s", sorted(keep_cameras) if keep_cameras else "ALL")

    # ── WandB ──────────────────────────────────────────────────────────────────
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
                "phase2_steps": cfg.phase2_steps,
                "lr": cfg.lr,
                "warmup_steps": cfg.warmup_steps,
                "batch_size": cfg.batch_size,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": cfg.batch_size * accelerator.num_processes,
                "cl_v2_action_temp": cfg.policy.cl_v2_action_temp,
                "cl_v2_vlm_temp": cfg.policy.cl_v2_vlm_temp,
                "cl_v2_loss_weight": cfg.policy.cl_v2_loss_weight,
                "cl_v2_fm_loss_weight": cfg.policy.cl_v2_fm_loss_weight,
                "keep_cameras": sorted(keep_cameras) if keep_cameras else "ALL",
                "seed": cfg.seed,
            },
            save_code=False,
        )
        logger.info("WandB: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    # Force phase2
    cfg.policy.cl_v2_phase = "phase2"

    # ── Dataset 로드 ───────────────────────────────────────────────────────────
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

    # ── Task prompt 매핑 ───────────────────────────────────────────────────────
    task_index_to_prompt: dict[int, str] = {}
    if cfg.dataset.root is not None:
        task_index_to_prompt = build_task_index_to_prompt(cfg.dataset.root, task_prompts)
        if accelerator.is_main_process:
            logger.info("Task prompts (%d tasks): %s", len(task_index_to_prompt), task_index_to_prompt)

    # ── Policy 생성 ────────────────────────────────────────────────────────────
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
            "RKD τ_act=%.3f, τ_vlm=%.3f, λ_rkd=%.3f, λ_fm=%.4f",
            cfg.policy.cl_v2_action_temp,
            cfg.policy.cl_v2_vlm_temp,
            cfg.policy.cl_v2_loss_weight,
            cfg.policy.cl_v2_fm_loss_weight,
        )

    # ── Preprocess 함수 ────────────────────────────────────────────────────────
    def preprocess(raw_batch: dict) -> dict:
        if keep_cameras:
            for key in list(raw_batch.keys()):
                if key.startswith("observation.images.") and key not in keep_cameras:
                    del raw_batch[key]

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

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
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
        num_training_steps=cfg.phase2_steps,
    )
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    # ── Training Loop ──────────────────────────────────────────────────────────
    policy.train()
    data_stream = infinite_dataloader()

    for step in range(1, cfg.phase2_steps + 1):
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
            rkd = loss_dict.get("rkd_loss", 0.0)
            fm = loss_dict.get("flow_matching_loss", 0.0)
            msg = (
                f"[phase2] step={step}/{cfg.phase2_steps} | lr={lr_now:.2e} | "
                f"grad_norm={grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm:.3f} | "
                f"rkd={rkd:.4f} | fm={fm:.4f} | total={loss_dict['loss']:.4f}"
            )
            print(msg, flush=True)
            logger.info(msg)
            if use_wandb:
                wandb.log(
                    {
                        "phase2/loss": loss_dict["loss"],
                        "phase2/rkd_loss": rkd,
                        "phase2/fm_loss": fm,
                        "phase2/lr": lr_now,
                        "phase2/grad_norm": (
                            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                        ),
                    },
                    step=step,
                )

        if step == cfg.phase2_steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.phase2_steps, step)
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
        logger.info("Phase 2 완료. 체크포인트: %s", cfg.output_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
