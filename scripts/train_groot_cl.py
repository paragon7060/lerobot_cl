#!/usr/bin/env python
"""GR00T Contrastive Learning 학습 스크립트 (accelerate 지원).

구조:
  Phase 1  — contrastive heads warm-up (_groot_model frozen, ~500 steps)
  Phase 2a — joint fine-tuning (contrastive gradient → VLM backbone, ~5000 steps)

단일 GPU 실행:
  conda activate lerobot_050_groot
  cd /home/bluepot/cl_ws/lerobot_cl
  python scripts/train_groot_cl.py

Multi-GPU 실행:
  accelerate launch --num_processes <N> scripts/train_groot_cl.py

주의: negative_action은 batch_to_transition()을 통과하면 사라지므로
      전처리 파이프라인(pre) 밖에서 NegativeActionNormalizeStep을 직접 호출한다.
"""

import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from lerobot.datasets.contrastive_dataset import ContrastiveLeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.groot_cl.configuration_groot_cl import GrootCLConfig
from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
from lerobot.policies.groot_cl.processor_groot_cl import NegativeActionNormalizeStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 사용자 설정 — 여기만 수정
# ─────────────────────────────────────────────────────────────────────────────

REPO_ID         = "paragon7060/INSIGHTfixposV3"
DATASET_ROOT    = "/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3"
NEG_PAIRS_PATH  = "/home/bluepot/cl_ws/negative_pairs.json"
BASE_MODEL_PATH = "nvidia/GR00T-N1.5-3B"
OUTPUT_DIR      = Path("outputs/groot_cl")

# 데이터셋별로 사전학습된 GrootPolicy 체크포인트 경로.
# None이면 BASE_MODEL_PATH(NVIDIA 원본)의 weights로 시작.
# 설정 시 해당 checkpoint의 _groot_model.* weights를 로드하고
# contrastive heads만 랜덤 초기화로 시작한다.
GROOT_PRETRAINED_PATH = None  # 예: "/home/bluepot/cl_ws/outputs/groot_my_task/step_010000"

# 학습 스텝 수
PHASE1_STEPS  = 500
PHASE2A_STEPS = 5000

# 로깅 / 저장 주기
LOG_INTERVAL  = 50
SAVE_INTERVAL = 500

# DataLoader
BATCH_SIZE   = 4
NUM_WORKERS  = 4


# ─────────────────────────────────────────────────────────────────────────────
# Accelerator 초기화
# mixed_precision="no": modeling_groot_cl.py forward()에 torch.autocast 이미 존재.
# accelerate와 중복 적용을 막기 위해 accelerate 측 autocast는 비활성화.
# ─────────────────────────────────────────────────────────────────────────────

accelerator = Accelerator(mixed_precision="no")
DEVICE = accelerator.device


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

config = GrootCLConfig(
    base_model_path=BASE_MODEL_PATH,
    contrastive_phase="phase1",
    contrastive_latent_dim=256,
    contrastive_cnn_hidden_dim=128,
    contrastive_proj_hidden_dim=512,
    contrastive_triplet_margin=0.5,
    contrastive_loss_weight=1.0,
    contrastive_backprop_backbone=True,
    contrastive_fallback_to_in_batch=False,
    groot_pretrained_path=GROOT_PRETRAINED_PATH,
    tune_llm=False,
    tune_visual=False,
    tune_projector=True,
    tune_diffusion_model=True,
    use_bf16=True,
    chunk_size=16,
    batch_size=BATCH_SIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

ds_meta = LeRobotDatasetMetadata(repo_id=REPO_ID, root=DATASET_ROOT)
delta_timestamps = resolve_delta_timestamps(config, ds_meta)

dataset = ContrastiveLeRobotDataset(
    repo_id=REPO_ID,
    root=DATASET_ROOT,
    negative_pairs_path=NEG_PAIRS_PATH,
    delta_timestamps=delta_timestamps,
)

if accelerator.is_main_process:
    logger.info("Dataset: %d frames, %d episodes", dataset.num_frames, dataset.num_episodes)


def collate_fn(batch: list[dict]) -> dict:
    """negative_action을 배치에서 분리해 collate한 뒤 다시 붙인다.
    배치 내 하나라도 None이면 배치 전체를 None으로 처리한다.
    → forward()에서 loss_cont = 0.0으로 안전하게 스킵된다.
    """
    neg_actions = [item.pop("negative_action", None) for item in batch]
    result = default_collate(batch)
    if all(isinstance(n, torch.Tensor) for n in neg_actions):
        result["negative_action"] = torch.stack(neg_actions)
    else:
        result["negative_action"] = None
    return result


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn,
)


# ─────────────────────────────────────────────────────────────────────────────
# 전처리 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

pre, post = make_pre_post_processors(config, dataset_stats=ds_meta.stats)

neg_normalizer = NegativeActionNormalizeStep(
    stats=ds_meta.stats,
    normalize_min_max=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────────────────────────────────────

policy: GrootCLPolicy = make_policy(config, ds_meta=ds_meta)

# policy와 dataloader를 함께 prepare → DDP 래핑 + DistributedSampler 삽입
# optimizer/scheduler는 train_loop에서 phase마다 별도 prepare
policy, dataloader = accelerator.prepare(policy, dataloader)

if accelerator.is_main_process:
    logger.info("Policy initialized: %s", policy.__class__.__name__)
    logger.info("Accelerator: num_processes=%d, device=%s", accelerator.num_processes, DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def preprocess(raw_batch: dict) -> dict:
    """전처리 파이프라인 실행 + negative_action 별도 정규화 후 병합."""
    raw_neg = raw_batch.pop("negative_action", None)
    processed = pre(raw_batch)
    if isinstance(raw_neg, torch.Tensor):
        result = neg_normalizer({"negative_action": raw_neg})
        processed["negative_action"] = result.get("negative_action")
    return processed


def infinite_dataloader():
    while True:
        yield from dataloader


def save_checkpoint(tag: str) -> None:
    if accelerator.is_main_process:
        path = OUTPUT_DIR / tag
        path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(policy).save_pretrained(path)
        logger.info("체크포인트 저장: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train_loop(
    phase: str,
    total_steps: int,
    lr: float,
    loss_weight: float,
    backprop_backbone: bool,
) -> None:
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info(
            "Phase: %s | steps=%d | lr=%.2e | loss_weight=%.3f | backprop_backbone=%s",
            phase, total_steps, lr, loss_weight, backprop_backbone,
        )
        logger.info("=" * 60)

    # DDP로 래핑된 policy의 실제 모듈에 접근
    raw_policy = accelerator.unwrap_model(policy)
    raw_policy.set_contrastive_phase(phase)
    raw_policy.config.contrastive_loss_weight = loss_weight
    raw_policy.config.contrastive_backprop_backbone = backprop_backbone

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, raw_policy.parameters()),
        lr=lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.1
    )

    # optimizer와 scheduler만 phase마다 prepare (policy/dataloader는 이미 prepare됨)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    policy.train()
    data_stream = infinite_dataloader()

    for step in range(1, total_steps + 1):
        raw_batch = next(data_stream)
        batch = preprocess(raw_batch)
        batch = to_device(batch, DEVICE)

        loss, loss_dict = policy(batch)

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if accelerator.is_main_process and step % LOG_INTERVAL == 0:
            log_str = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            lr_now = scheduler.get_last_lr()[0]
            logger.info("[%s] step=%d/%d | lr=%.2e | %s", phase, step, total_steps, lr_now, log_str)

        if step % SAVE_INTERVAL == 0 or step == total_steps:
            accelerator.wait_for_everyone()
            save_checkpoint(f"{phase}/step_{step:06d}")

    if accelerator.is_main_process:
        logger.info("[%s] 완료", phase)


# ─────────────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────────────

if accelerator.is_main_process:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 1: contrastive heads warm-up (_groot_model frozen)
train_loop(
    phase="phase1",
    total_steps=PHASE1_STEPS,
    lr=1e-4,
    loss_weight=1.0,
    backprop_backbone=False,
)

# Phase 2a: joint fine-tuning (contrastive gradient → VLM backbone)
train_loop(
    phase="phase2a",
    total_steps=PHASE2A_STEPS,
    lr=2e-5,
    loss_weight=0.05,
    backprop_backbone=True,
)

if accelerator.is_main_process:
    logger.info("전체 학습 완료. 최종 체크포인트: %s", OUTPUT_DIR)
