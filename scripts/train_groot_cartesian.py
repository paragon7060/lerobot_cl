#!/usr/bin/env python
"""GR00T Cartesian (pre-converted) 학습 스크립트.

convert_action_absolute.py로 미리 변환된 데이터셋 (axis-angle 7D action + 25D zero-padding)을
그대로 읽어서 학습. FK 변환 없음.

action: 32D 저장 중 앞 7D만 유효 [pos_x, pos_y, pos_z, aa_x, aa_y, aa_z, gripper]
state:  32D 저장 중 앞 16D만 유효 [ee7 + joint7 + gripper2]

단일 GPU 실행:
  python scripts/train_groot_cartesian.py \\
      --dataset.repo_id=paragon7060/INSIGHTfixposV3 \\
      --dataset.root=/path/to/converted_dataset \\
      --dataset.video_backend=pyav \\
      --policy.type=groot \\
      --output_dir=./outputs/groot_cartesian \\
      --job_name=groot_cartesian_v1 \\
      --steps=50000 \\
      --batch_size=128 \\
      --policy.lora_rank=16 \\
      --policy.lora_target=vision \\
      --task_prompt_mode=guide \\
      --wandb.enable=true \\
      --wandb.project=groot_insight \\
      --wandb.entity=RwHlabs

Multi-GPU 실행 (예: 4 GPU, effective BS = 4 x 32 = 128):
  accelerate launch --num_processes=4 --mixed_precision=no \\
      scripts/train_groot_cartesian.py \\
      --policy.type=groot \\
      --batch_size=32 [other args]
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
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

_SCRIPT_DIR = Path(__file__).parent
_KEYFRAME_PKG_DIR = _SCRIPT_DIR.parent / "src/lerobot/policies/groot_keyframe"
TASK_DESCRIPTIONS_PATHS: dict[str, Path] = {
    "guide":     _KEYFRAME_PKG_DIR / "task_descriptions.json",
    "non_guide": _KEYFRAME_PKG_DIR / "task_descriptions_non_guide.json",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Task description mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_task_desc_map(task_descriptions_path: Path, dataset_root: Path) -> dict[str, str]:
    with open(task_descriptions_path) as f:
        name_to_desc: dict[str, str] = json.load(f)
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet").reset_index()
    return {row["index"]: name_to_desc.get(row["index"], row["index"])
            for _, row in tasks_df.iterrows()}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Zero-padding slice
# ─────────────────────────────────────────────────────────────────────────────

def slice_feature_dims(raw_batch: dict, action_dim: int, state_dim: int) -> dict:
    """action/state의 zero-padded tail을 잘라 유효 dim만 남긴다."""
    if "action" in raw_batch and action_dim is not None:
        raw_batch["action"] = raw_batch["action"][..., :action_dim]
    if "observation.state" in raw_batch and state_dim is not None:
        raw_batch["observation.state"] = raw_batch["observation.state"][..., :state_dim]
    return raw_batch


def slice_dataset_stats(stats: dict | None, action_dim: int, state_dim: int) -> dict | None:
    """stats의 action/state를 유효 dim만큼 잘라낸다."""
    if not stats:
        return stats

    def _trim(sub: dict, d: int) -> dict:
        return {k: v[..., :d].contiguous()
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[-1] > d else v
                for k, v in sub.items()}

    if "action" in stats and action_dim is not None:
        stats["action"] = _trim(stats["action"], action_dim)
    if "observation.state" in stats and state_dim is not None:
        stats["observation.state"] = _trim(stats["observation.state"], state_dim)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GrootCartesianTrainConfig(TrainPipelineConfig):
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
    save_freq: int = 10_000
    seed: int = 42
    use_policy_training_preset: bool = False
    gradient_checkpointing: bool = False

    # 변환된 데이터셋: action 32→7 (pos3+quat4+gripper), state 32→16 (ee7+joint7+gripper2)
    action_dim: int = 8
    state_dim: int = 16

    # task description 모드: "guide" | "non_guide" | "raw"
    task_prompt_mode: str = "non_guide"
    task_descriptions_path: str | None = None  # None이면 task_prompt_mode로 자동 선택

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_cartesian"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_cartesian")
        valid_modes = list(TASK_DESCRIPTIONS_PATHS) + ["raw"]
        if self.task_prompt_mode not in valid_modes:
            raise ValueError(
                f"task_prompt_mode must be one of {valid_modes}, got {self.task_prompt_mode!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def main(cfg: GrootCartesianTrainConfig) -> None:
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
        logger.info("accelerator: num_processes=%d  device=%s", accelerator.num_processes, device)

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
                "action_space": "cartesian_7d_axis_angle",
                "action_dim": cfg.action_dim,
                "state_dim": cfg.state_dim,
                "task_prompt_mode": cfg.task_prompt_mode,
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
        logger.info("WandB initialized: project=%s  run=%s", cfg.wandb.project, wandb.run.name)

    # ── Dataset metadata ──────────────────────────────────────────────────────
    ds_meta = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, root=cfg.dataset.root)

    # info.json shape가 원본 32D로 남아 있으면 make_policy가 32D 출력 레이어를 구성해 shape mismatch 발생.
    # 유효 dim으로 덮어써서 모델 구성 기준을 맞춘다.
    if "action" in ds_meta.features:
        ds_meta.features["action"]["shape"] = (cfg.action_dim,)
    if "observation.state" in ds_meta.features:
        ds_meta.features["observation.state"]["shape"] = (cfg.state_dim,)

    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    # ── Stats: zero-padded tail 제거 ──────────────────────────────────────────
    sliced_stats = slice_dataset_stats(ds_meta.stats, cfg.action_dim, cfg.state_dim)
    if accelerator.is_main_process:
        act_min = sliced_stats.get("action", {}).get("min") if sliced_stats else None
        st_min  = sliced_stats.get("observation.state", {}).get("min") if sliced_stats else None
        logger.info(
            "Sliced stats shapes: action.min=%s, state.min=%s",
            tuple(act_min.shape) if isinstance(act_min, torch.Tensor) else None,
            tuple(st_min.shape)  if isinstance(st_min, torch.Tensor)  else None,
        )

    # ── Task description 매핑 ─────────────────────────────────────────────────
    if cfg.task_prompt_mode == "raw":
        task_name_to_desc = None
        if accelerator.is_main_process:
            logger.info("Task prompt mode: raw")
    else:
        desc_path = (
            Path(cfg.task_descriptions_path)
            if cfg.task_descriptions_path
            else TASK_DESCRIPTIONS_PATHS[cfg.task_prompt_mode]
        )
        task_name_to_desc = build_task_desc_map(desc_path, ds_meta.root)
        if accelerator.is_main_process:
            logger.info(
                "Task description 매핑 로드: mode=%s, path=%s, %d tasks",
                cfg.task_prompt_mode, desc_path, len(task_name_to_desc),
            )

    # ── Dataset 로드 ──────────────────────────────────────────────────────────
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
        logger.info(
            "Dataset: %d frames, %d episodes | action: cartesian axis-angle (%dD)",
            dataset.num_frames, dataset.num_episodes, cfg.action_dim,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # ── Policy & Preprocessors ────────────────────────────────────────────────
    pre, post = make_pre_post_processors(cfg.policy, dataset_stats=sliced_stats)
    policy = make_policy(cfg.policy, ds_meta=ds_meta)

    if cfg.policy.lora_rank > 0 and accelerator.is_main_process:
        if cfg.policy.lora_target in ("llm", "both") and cfg.policy.tune_llm:
            logger.warning(
                "lora_target=%r + tune_llm=True: LLM base weights도 학습 대상이 됩니다.",
                cfg.policy.lora_target,
            )
        if cfg.policy.lora_target in ("vision", "both") and cfg.policy.tune_visual:
            logger.warning(
                "lora_target=%r + tune_visual=True: Vision tower base weights도 학습 대상이 됩니다.",
                cfg.policy.lora_target,
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

    policy, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, optimizer, dataloader, scheduler
    )

    if accelerator.is_main_process:
        num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total     = sum(p.numel() for p in policy.parameters())
        logger.info("num_learnable_params=%s", f"{num_learnable:,}")
        logger.info("num_total_params=%s",     f"{num_total:,}")
        logger.info("trainable ratio=%.2f%%",  100 * num_learnable / max(num_total, 1))

        if cfg.policy.lora_rank > 0:
            lora_n = sum(
                p.numel() for n, p in policy.named_parameters()
                if p.requires_grad and "lora_" in n
            )
            logger.info(
                "LoRA [%s] adapter params: %s (trainable total: %s / %s)",
                cfg.policy.lora_target, f"{lora_n:,}",
                f"{num_learnable:,}", f"{num_total:,}",
            )
        logger.info("Policy: %s", policy.__class__.__name__)

    # ── Training Loop ─────────────────────────────────────────────────────────

    def infinite_dataloader():
        while True:
            yield from dataloader

    def _save(step: int) -> None:
        if accelerator.is_main_process:
            ckpt_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=ckpt_dir,
                step=step,
                cfg=cfg,
                policy=accelerator.unwrap_model(policy),
                optimizer=optimizer,
                scheduler=scheduler,
                preprocessor=pre,
                postprocessor=post,
            )
            update_last_checkpoint(ckpt_dir)
            logger.info("체크포인트 저장: %s", ckpt_dir)

    policy.train()
    data_stream = infinite_dataloader()

    for step in range(1, cfg.steps + 1):
        raw_batch = next(data_stream)

        # zero-padded tail 제거
        raw_batch = slice_feature_dims(raw_batch, cfg.action_dim, cfg.state_dim)

        # task name → description 교체
        if task_name_to_desc is not None:
            raw_batch["task"] = [task_name_to_desc.get(t, t) for t in raw_batch["task"]]

        if accelerator.is_main_process and step == 1:
            logger.info(
                "After slice: action.shape=%s, observation.state.shape=%s",
                tuple(raw_batch["action"].shape),
                tuple(raw_batch["observation.state"].shape),
            )

        batch = pre(raw_batch)
        loss, loss_dict = policy(batch)

        if accelerator.is_main_process and step in (1, 10, 100):
            with open("/tmp/groot_cart_dbg.txt", "a") as _f:
                a = batch["action"]
                m = batch.get("action_mask")
                s = batch.get("state")
                lines = []
                lines.append(f"==== step {step} ====")
                lines.append(f"norm action: shape={tuple(a.shape)} min={a.min().item():.4f} max={a.max().item():.4f} mean={a.mean().item():.4f} std={a.std().item():.4f}")
                if m is not None:
                    mb = m.bool() if m.dtype == torch.bool else (m > 0.5)
                    a_valid = a[mb]
                    lines.append(f"valid action (mask=True): min={a_valid.min().item():.4f} max={a_valid.max().item():.4f} mean={a_valid.mean().item():.4f} std={a_valid.std().item():.4f} n={a_valid.numel()}")
                    a_view = a.reshape(-1, a.shape[-1])
                    for d in range(min(8, a.shape[-1])):
                        col = a_view[:, d]
                        lines.append(f"  dim{d}: min={col.min().item():.4f} max={col.max().item():.4f} mean={col.mean().item():.4f} std={col.std().item():.4f}")
                if s is not None:
                    lines.append(f"norm state: shape={tuple(s.shape)} min={s.min().item():.4f} max={s.max().item():.4f} mean={s.mean().item():.4f} std={s.std().item():.4f}")
                lines.append(f"loss={loss.item():.4f} loss_dict={ {k: float(v) for k, v in loss_dict.items()} }")
                _f.write("\n".join(lines) + "\n")
                _f.flush()

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if accelerator.is_main_process and step % cfg.log_freq == 0:
            lr_now  = scheduler.get_last_lr()[0]
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
                        "train/flow_matching_loss": loss_dict.get(
                            "flow_matching_loss", loss_dict.get("loss", loss.item())
                        ),
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
                ckpt_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                artifact = wandb.Artifact(
                    name=f"{cfg.job_name}-step{step:06d}",
                    type="model",
                    description=f"cartesian checkpoint at step {step}",
                )
                artifact.add_dir(str(ckpt_dir))
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
