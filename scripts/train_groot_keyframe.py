#!/usr/bin/env python
"""GR00T Keyframe 4-Camera Baseline 학습 스크립트.

기존 3-camera (guide, right_shoulder, wrist) baseline에 keyframe 4번째 카메라를 추가한 학습 코드.

Keyframe 카메라 로직:
  - keyframe 시점(frame_index) 이전: 4번째 슬롯 = wrist 카메라 복사본
  - keyframe 시점 이후: 4번째 슬롯 = file_path의 cropped keyframe 이미지
  cropped=true 항목이 없는 에피소드는 학습에서 완전히 제외됨.

카메라 순서 (알파벳 정렬, GrootPackInputsStep 자동 처리):
  guide | keyframe | right_shoulder | wrist

Action/state feature-dim 슬라이스:
  paragon7060/INSIGHTfixposV3는 action=[32]/state=[32]로 저장되지만 뒤쪽 dim이 zero-padded.
  실제 유효 dim은 action=8(j1-j7,gripper), state=16. processor 전에 슬라이싱해서 action_mask/
  state_mask가 유효 dim만 True가 되도록 하고, 정규화 stats도 유효 dim만 사용.
  필요 시 --action_dim / --state_dim 으로 조정.

Task description 모드:
  --task_prompt_mode=guide     (default) 구체적 회전방향 포함 task_descriptions.json
  --task_prompt_mode=non_guide "following the guide" 패턴 task_descriptions_non_guide.json

실행 방법:
  conda activate lerobot050_groot
  cd ~/clvla/lerobot_cl

  단일 GPU:
    python scripts/train_groot_keyframe.py \\
        --dataset.repo_id=paragon7060/INSIGHTfixposV3 \\
        --policy.type=groot \\
        --output_dir=./outputs/groot_keyframe \\
        --job_name=groot_keyframe_v1 \\
        --steps=50000 \\
        --batch_size=32 \\
        --policy.lora_rank=16 \\
        --policy.lora_target=vision \\
        --policy.tune_visual=true \\
        --task_prompt_mode=guide \\
        --wandb.enable=true \\
        --wandb.project=groot_insight \\
        --wandb.entity=RwHlabs

  Multi-GPU (4 GPU, effective BS = 4 x 32 = 128):
    accelerate launch --num_processes=4 --mixed_precision=no \\
        scripts/train_groot_keyframe.py \\
        --policy.type=groot \\
        --batch_size=32 \\
        --policy.tune_visual=true \\
        [other args]

주의: Vision tower 튜닝을 위해 --policy.tune_visual=true 를 명시적으로 지정하세요.
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

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

# 기본 경로 상수
_SCRIPT_DIR = Path(__file__).parent
DEFAULT_REGISTRY_PATH = Path(
    "~/clvla/memory_module/keyframe_output/frame_index_registry.json"
)
_KEYFRAME_PKG_DIR = _SCRIPT_DIR.parent / "src/lerobot/policies/groot_keyframe"
TASK_DESCRIPTIONS_PATHS: dict[str, Path] = {
    "guide": _KEYFRAME_PKG_DIR / "task_descriptions.json",
    "non_guide": _KEYFRAME_PKG_DIR / "task_descriptions_non_guide.json",
}


# ---------------------------------------------------------------------------
# Keyframe 유틸리티
# ---------------------------------------------------------------------------

def _remap_file_path(file_path: str, registry_path: Path) -> str:
    """JSON에 기록된 절대 경로를 실제 registry 위치 기준으로 치환한다.

    registry 파일은 .../keyframe_output/frame_index_registry.json 에 있고,
    file_path 도 .../keyframe_output/... 하위를 가리킨다.
    JSON이 다른 계정에서 생성됐을 때 발생하는 /home/<other_user>/... 경로 문제를 해결한다.
    """
    marker = "keyframe_output/"
    idx = file_path.find(marker)
    if idx == -1:
        return file_path
    rel = file_path[idx + len(marker):]          # e.g. "task_3a/episode_0/right_shoulder_crop.png"
    actual_base = registry_path.parent            # .../keyframe_output/
    return str(actual_base / rel)


def load_keyframe_registry(registry_path: Path) -> dict:
    """Keyframe registry를 로드하고 episode별 lookup dict를 반환.

    file_path는 registry 파일 위치를 기준으로 자동 재매핑된다.

    Returns:
        {episode_id: {"frame_index": int, "file_path": str}} - cropped=true 항목만 포함
    """
    with open(registry_path) as f:
        entries = json.load(f)

    keyframe_dict = {}
    for entry in entries:
        if entry["cropped"] and entry["episode_id"] not in keyframe_dict:
            m = re.search(r"task_(\w+)[/\\]", entry["file_path"])
            task_id = m.group(1) if m else ""
            source_cam = "right_shoulder" if task_id.startswith("3") else "wrist"
            keyframe_dict[entry["episode_id"]] = {
                "frame_index": entry["frame_index"],
                "file_path": _remap_file_path(entry["file_path"], registry_path),
                "source_cam": source_cam,
            }
    return keyframe_dict


def get_valid_episodes(registry_path: Path) -> list[int]:
    """cropped=true 항목이 존재하는 에피소드 ID 목록을 반환 (정렬됨)."""
    with open(registry_path) as f:
        entries = json.load(f)
    valid = {entry["episode_id"] for entry in entries if entry["cropped"]}
    return sorted(valid)


def build_task_desc_map(
    task_descriptions_path: Path,
    dataset_root: Path,
) -> dict[str, str]:
    """task_name (예: '3a') → 영어 description 문자열 매핑을 반환.

    tasks.parquet의 index 컬럼(task_name)을 task_descriptions.json의 값에 매핑.
    매핑에 없는 task_name은 원본 task_name을 그대로 사용.
    """
    with open(task_descriptions_path) as f:
        name_to_desc: dict[str, str] = json.load(f)

    tasks_parquet = dataset_root / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_parquet).reset_index()
    # columns: ['index' (task_name string), 'task_index' (int)]

    task_name_to_desc = {}
    for _, row in tasks_df.iterrows():
        task_name = row["index"]
        task_name_to_desc[task_name] = name_to_desc.get(task_name, task_name)

    return task_name_to_desc


# ---------------------------------------------------------------------------
# Feature-dim 슬라이스
# ---------------------------------------------------------------------------

def slice_feature_dims(raw_batch: dict, action_dim: int, state_dim: int) -> dict:
    """Dataset의 zero-padded tail을 잘라 실제 유효 dim만 남긴다.

    INSIGHTfixposV3는 action=[32](실제 8), state=[32](실제 16)로 저장되며 뒤쪽 dim은 항상 0.
    Processor의 GrootPackInputsStep은 max_action_dim=32 / max_state_dim=64에 맞춰 pad+mask를
    자동 계산하므로, 입력 단계에서 슬라이스하면 mask가 유효 dim만 True로 잡히고 min/max 정규화도
    유효 dim만 대상으로 한다.
    """
    if "action" in raw_batch and action_dim is not None:
        raw_batch["action"] = raw_batch["action"][..., :action_dim]
    if "observation.state" in raw_batch and state_dim is not None:
        raw_batch["observation.state"] = raw_batch["observation.state"][..., :state_dim]
    return raw_batch


def slice_dataset_stats(
    stats: dict | None,
    action_dim: int,
    state_dim: int,
) -> dict | None:
    """ds_meta.stats의 action/state min/max 통계를 유효 dim만큼 잘라낸다."""
    if not stats:
        return stats

    def _trim(sub: dict, d: int) -> dict:
        trimmed = {}
        for k, v in sub.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[-1] > d:
                trimmed[k] = v[..., :d].contiguous()
            else:
                trimmed[k] = v
        return trimmed

    if "action" in stats and action_dim is not None:
        stats["action"] = _trim(stats["action"], action_dim)
    if "observation.state" in stats and state_dim is not None:
        stats["observation.state"] = _trim(stats["observation.state"], state_dim)
    return stats


# ---------------------------------------------------------------------------
# Batch transform: keyframe 주입 + task description 교체
# ---------------------------------------------------------------------------

_kf_img_cache: dict[str, torch.Tensor] = {}


def _load_keyframe_tensor(file_path: str, h: int, w: int) -> torch.Tensor:
    """Cropped keyframe 이미지를 (C, H, W) float32 [0,1] 텐서로 로드 (캐시 사용)."""
    cache_key = f"{file_path}_{h}_{w}"
    if cache_key not in _kf_img_cache:
        img = Image.open(file_path).convert("RGB").resize((w, h), Image.BILINEAR)
        _kf_img_cache[cache_key] = transforms.ToTensor()(img)
    return _kf_img_cache[cache_key]


def inject_keyframe_and_task(
    raw_batch: dict,
    keyframe_dict: dict,
    task_name_to_desc: dict[str, str],
) -> dict:
    """배치에 keyframe 카메라 이미지를 추가하고 task 문자열을 description으로 교체.

    Args:
        raw_batch: DataLoader에서 가져온 flat batch dict.
        keyframe_dict: {episode_id: {"frame_index": int, "file_path": str}}
        task_name_to_desc: {task_name: description_string}

    Returns:
        수정된 raw_batch. 새 키 "observation.images.keyframe" 추가됨.
    """
    wrist_imgs = raw_batch["observation.images.wrist"]  # (B, C, H, W), float32 [0,1]
    B, C, H, W = wrist_imgs.shape

    episode_indices = raw_batch["episode_index"]  # (B,) int64
    frame_indices = raw_batch["frame_index"]       # (B,) int64 (에피소드 상대)

    kf_imgs = []
    for i in range(B):
        ep_id = episode_indices[i].item()
        fr_idx = frame_indices[i].item()
        kf = keyframe_dict.get(ep_id)

        if kf is None or fr_idx < kf["frame_index"]:
            # keyframe 이전: task별 지정 카메라 복사 (door=right_shoulder, others=wrist)
            src_key = f"observation.images.{kf['source_cam'] if kf else 'wrist'}"
            kf_imgs.append(raw_batch[src_key][i].clone())
        else:
            # keyframe 이후 → cropped keyframe 이미지 로드
            tensor = _load_keyframe_tensor(kf["file_path"], H, W).to(wrist_imgs.device)
            kf_imgs.append(tensor)

    raw_batch["observation.images.keyframe"] = torch.stack(kf_imgs)

    # task name → description 교체 (preprocessor가 language 입력으로 사용)
    raw_tasks = raw_batch.get("task", [])
    if isinstance(raw_tasks, (list, tuple)):
        raw_batch["task"] = [task_name_to_desc.get(t, t) for t in raw_tasks]

    return raw_batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GrootKeyframeTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            repo_id="paragon7060/INSIGHTfixposV3",
            root=None,
            video_backend="pyav",
        )
    )
    policy: PreTrainedConfig | None = None

    steps: int = 50_000
    batch_size: int = 32
    num_workers: int = 8
    log_freq: int = 100
    save_freq: int = 10_000
    seed: int = 42
    use_policy_training_preset: bool = False
    gradient_checkpointing: bool = False

    keyframe_registry_path: str = str(DEFAULT_REGISTRY_PATH)
    task_descriptions_path: str | None = None  # None이면 task_prompt_mode로 자동 선택
    task_prompt_mode: str = "guide"            # {"guide", "non_guide"}

    # INSIGHTfixposV3: action 32→8 (j1-j7,gripper), state 32→16 (ee7+joint7+gripper2)
    action_dim: int = 8
    state_dim: int = 16

    def validate(self) -> None:
        if self.policy is None:
            raise ValueError("policy must be set")
        if not self.job_name:
            self.job_name = "groot_keyframe"
        if not self.output_dir:
            self.output_dir = Path("outputs/groot_keyframe")

        if self.task_prompt_mode not in TASK_DESCRIPTIONS_PATHS:
            raise ValueError(
                f"task_prompt_mode must be one of {list(TASK_DESCRIPTIONS_PATHS)}, "
                f"got {self.task_prompt_mode!r}"
            )

        # Vision tower 튜닝 권장
        if hasattr(self.policy, "tune_visual") and not self.policy.tune_visual:
            logger.warning(
                "tune_visual=False 감지됨. groot_keyframe 학습에는 --policy.tune_visual=true 권장."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@parser.wrap()
def main(cfg: GrootKeyframeTrainConfig) -> None:
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

    # --------------------------------------------------------------------------
    # Keyframe registry 로드
    # --------------------------------------------------------------------------
    registry_path = Path(cfg.keyframe_registry_path)
    if accelerator.is_main_process:
        logger.info("Keyframe registry 로드: %s", registry_path)

    keyframe_dict = load_keyframe_registry(registry_path)
    valid_episodes = get_valid_episodes(registry_path)

    if accelerator.is_main_process:
        logger.info(
            "유효 에피소드 (cropped=true): %d개 / registry 전체: %d개",
            len(valid_episodes),
            len(keyframe_dict),
        )

    # --------------------------------------------------------------------------
    # Task description 매핑 로드 (prompt_mode 기준)
    # --------------------------------------------------------------------------
    if cfg.task_descriptions_path:
        descriptions_path = Path(cfg.task_descriptions_path)
    else:
        descriptions_path = TASK_DESCRIPTIONS_PATHS[cfg.task_prompt_mode]

    ds_meta = LeRobotDatasetMetadata(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
    )

    task_name_to_desc = build_task_desc_map(descriptions_path, ds_meta.root)
    if accelerator.is_main_process:
        logger.info(
            "Task description 매핑 로드 완료: mode=%s, path=%s, %d tasks (dataset root: %s)",
            cfg.task_prompt_mode,
            descriptions_path,
            len(task_name_to_desc),
            ds_meta.root,
        )

    # --------------------------------------------------------------------------
    # Output dir & WandB
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: %s", cfg.output_dir)
        logger.info(
            "Accelerator: num_processes=%d, device=%s",
            accelerator.num_processes,
            device,
        )

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
                "valid_episodes": len(valid_episodes),
                "num_cameras": 4,
                "cameras": ["guide", "keyframe", "right_shoulder", "wrist"],
                "task_prompt_mode": cfg.task_prompt_mode,
                "action_dim": cfg.action_dim,
                "state_dim": cfg.state_dim,
            },
            save_code=False,
        )
        logger.info("WandB 초기화: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    # --------------------------------------------------------------------------
    # Dataset (유효 에피소드만)
    # --------------------------------------------------------------------------
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if accelerator.is_main_process:
        dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
            episodes=valid_episodes,
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.dataset.video_backend,
            episodes=valid_episodes,
        )

    if accelerator.is_main_process:
        logger.info(
            "Dataset: %d frames, %d episodes (전체 %d 에피소드 중 필터링 후)",
            dataset.num_frames,
            dataset.num_episodes,
            ds_meta.total_episodes,
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

    # --------------------------------------------------------------------------
    # Policy & Preprocessor  (stats는 action/state 유효 dim에 맞춰 슬라이스)
    # --------------------------------------------------------------------------
    sliced_stats = slice_dataset_stats(ds_meta.stats, cfg.action_dim, cfg.state_dim)
    if accelerator.is_main_process:
        act_min = sliced_stats.get("action", {}).get("min") if sliced_stats else None
        st_min = sliced_stats.get("observation.state", {}).get("min") if sliced_stats else None
        logger.info(
            "Sliced stats shapes: action.min=%s, state.min=%s",
            tuple(act_min.shape) if isinstance(act_min, torch.Tensor) else None,
            tuple(st_min.shape) if isinstance(st_min, torch.Tensor) else None,
        )

    pre, _ = make_pre_post_processors(cfg.policy, dataset_stats=sliced_stats)
    policy = make_policy(cfg.policy, ds_meta=ds_meta)

    if cfg.policy.lora_rank > 0:
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
        logger.info("Gradient checkpointing 활성화.")

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
    policy, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, optimizer, dataloader, scheduler
    )

    if accelerator.is_main_process:
        num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in policy.parameters())
        logger.info("num_learnable_params=%s", f"{num_learnable:,}")
        logger.info("num_total_params=%s", f"{num_total:,}")
        logger.info("trainable ratio=%.2f%%", 100 * num_learnable / max(num_total, 1))

        if cfg.policy.lora_rank > 0:
            lora_params = sum(
                p.numel()
                for n, p in policy.named_parameters()
                if p.requires_grad and "lora_" in n
            )
            logger.info(
                "LoRA [%s] adapter params: %s (trainable total: %s / %s)",
                cfg.policy.lora_target,
                f"{lora_params:,}",
                f"{num_learnable:,}",
                f"{num_total:,}",
            )

        logger.info("Policy: %s", policy.__class__.__name__)

    # --------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------
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

        # keyframe 이미지 주입 + task description 교체
        raw_batch = inject_keyframe_and_task(raw_batch, keyframe_dict, task_name_to_desc)

        # action/state zero-padded tail 제거 → processor가 유효 dim만 mask=True로 처리
        raw_batch = slice_feature_dims(raw_batch, cfg.action_dim, cfg.state_dim)

        if accelerator.is_main_process and step == 1:
            logger.info(
                "After slice: action.shape=%s, observation.state.shape=%s",
                tuple(raw_batch["action"].shape),
                tuple(raw_batch["observation.state"].shape),
            )

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
                step,
                cfg.steps,
                lr_now,
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                log_str,
            )

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss_dict.get("loss", loss.item()),
                        "train/flow_matching_loss": loss_dict.get(
                            "flow_matching_loss",
                            loss_dict.get("loss", loss.item()),
                        ),
                        "train/lr": lr_now,
                        "train/grad_norm": (
                            grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else float(grad_norm)
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
