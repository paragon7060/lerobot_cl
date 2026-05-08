#!/usr/bin/env python
"""GR00T Multi-Dataset 학습 스크립트

로컬 경로의 4개 그룹 (각 그룹은 task_XXXX 서브디렉토리로 쪼개진 LeRobot 데이터셋)을
MultiLeRobotDataset으로 합쳐서 학습한다.

데이터셋 경로 구조:
  {dataset_root}/
    robocasa_pretrain_human_atomic/task_0001/{meta,data,videos}
    robocasa_pretrain_human_atomic/task_0002/...
    robocasa_pretrain_human_composite/task_0001/...
    robocasa_target_human_atomic/task_0001/...
    robocasa_target_human_composite/task_0001/...

단일 GPU 실행:
  python scripts/train_groot_multi.py \
      --dataset.root=/home/seonho/slicing_robocasa_human_v3 \
      --policy.type=groot \
      --output_dir=./outputs/groot_multi \
      --job_name=groot_multi_v1 \
      --steps=100000 \
      --batch_size=128 \
      --policy.tune_visual=false \
      --policy.tune_llm=false \
      --wandb.enable=true \
      --wandb.project=groot_robocasa \
      --wandb.entity=RwHlabs

Multi-GPU 실행 (예: 2 GPU, effective BS = 64x2 = 128):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \\
      --num_processes=2 \\
      scripts/train_groot_multi.py \\
      --dataset.root=/home/seonho/slicing_robocasa_human_v3 \\
      --policy.type=groot \\
      --output_dir=./outputs/groot_multi \\
      --batch_size=64 \\
      --steps=100000 \\
      --policy.tune_visual=false \\
      --policy.tune_llm=false \\
      --wandb.enable=true
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
    """root 아래 각 top-level dir의 task_XXXX 서브디렉토리를 열거한다.
    반환값: ["robocasa_pretrain_human_atomic/task_0001", ...] 형식
    MultiLeRobotDataset은 root / repo_id 경로로 로드하므로 이 형식이 맞다.
    """
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
class GrootMultiTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="",  # 사용 안 함 (multi-dataset이므로)
            root="/home/seonho/slicing_robocasa_human_v3",
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    steps: int = 100_000
    batch_size: int = 128
    num_workers: int = 8
    log_freq: int = 100
    save_freq: int = 10_000
    seed: int = 42
    use_policy_training_preset: bool = False

    gradient_checkpointing: bool = False
    resume: bool = False
    data_split: str = "all"  # "pretrain", "target", "all"
    pretrained_path: str = ""  # pretrain 체크포인트 경로 (weights만 로드, optimizer는 새로)

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_multi"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_multi")


@parser.wrap()
def main(cfg: GrootMultiTrainConfig) -> None:
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

    # data_split에 따라 사용할 디렉토리 선택
    if cfg.data_split == "pretrain":
        top_level_dirs = PRETRAIN_DIRS
    elif cfg.data_split == "target":
        top_level_dirs = TARGET_DIRS
    elif cfg.data_split == "all":
        top_level_dirs = ALL_DIRS
    else:
        raise ValueError(f"Unknown data_split: {cfg.data_split!r} (pretrain / target / all)")

    # task_XXXX 서브디렉토리 열거 (모든 프로세스에서 동일하게 실행)
    repo_ids = get_task_repo_ids(dataset_root, top_level_dirs)

    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: %s", cfg.output_dir)
        logger.info("Accelerator: num_processes=%d, device=%s", accelerator.num_processes, device)
        logger.info("Dataset root: %s", dataset_root)
        logger.info("총 task 수: %d", len(repo_ids))

    # delta_timestamps는 첫 번째 task 기준으로 계산 (fps/feature 구조 동일하다고 가정)
    first_ds_meta = LeRobotDatasetMetadata(
        repo_id=repo_ids[0],
        root=dataset_root / repo_ids[0],
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, first_ds_meta)

    # MultiLeRobotDataset: root/repo_id 경로로 각 task를 로드
    # accelerate 환경에서 main process 먼저 로드 후 나머지 동기화
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
                "top_level_dirs": top_level_dirs,
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
                "lora_rank": cfg.policy.lora_rank,
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

    # stats는 MultiLeRobotDataset의 aggregated stats 사용
    pre, post = make_pre_post_processors(cfg.policy, dataset_stats=dataset.stats)

    # make_policy에는 첫 번째 task ds_meta 전달 (feature shape 추론용)
    policy = make_policy(cfg.policy, ds_meta=first_ds_meta)

    # ── pretrained weights 로드 (optimizer/scheduler는 새로 생성) ──────
    if cfg.pretrained_path:
        from safetensors.torch import load_model as safetensors_load_model
        ckpt_model = Path(cfg.pretrained_path) / "model.safetensors"
        if not ckpt_model.exists():
            # pretrained_model 하위 폴더 자동 탐색
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

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_step = 0
    if cfg.resume:
        last_link = Path(cfg.output_dir) / "checkpoints" / LAST_CHECKPOINT_LINK
        if last_link.exists():
            resume_dir = last_link.resolve()
            logger.info("Resuming from checkpoint: %s", resume_dir)
            # Load model weights
            from safetensors.torch import load_model as safetensors_load_model
            model_path = resume_dir / "pretrained_model" / "model.safetensors"
            safetensors_load_model(accelerator.unwrap_model(policy), str(model_path))
            # Load optimizer, scheduler, rng state
            start_step, optimizer, scheduler = load_training_state(
                resume_dir, optimizer, scheduler
            )
            logger.info("Resumed at step %d", start_step)
        else:
            logger.warning("--resume=true but no checkpoint found at %s, training from scratch", last_link)

    if accelerator.is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info("num_learnable_params=%s", f"{num_learnable_params:,}")
        logger.info("num_total_params=%s", f"{num_total_params:,}")
        logger.info("trainable ratio=%.2f%%", 100 * num_learnable_params / max(num_total_params, 1))

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
        logger.info(
            "학습 완료. 최종 체크포인트: %s",
            get_step_checkpoint_dir(cfg.output_dir, cfg.steps, cfg.steps),
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
