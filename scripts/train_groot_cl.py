#!/usr/bin/env python
"""GR00T Contrastive Learning 학습 스크립트.

구조:
  Phase 1  — contrastive heads warm-up (_groot_model frozen, ~500 steps)
  Phase 2a — joint fine-tuning (contrastive gradient → VLM backbone, ~5000 steps)

실행:
  conda activate lerobot_050_groot
  cd /home/bluepot/cl_ws/lerobot_cl
  python scripts/train_groot_cl.py

주의: negative_action은 batch_to_transition()을 통과하면 사라지므로
      전처리 파이프라인(pre) 밖에서 NegativeActionNormalizeStep을 직접 호출한다.
"""

import logging
from pathlib import Path

import torch
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

REPO_ID         = "paragon7060/my_task"               # HF 데이터셋 repo ID
DATASET_ROOT    = "/home/bluepot/cl_ws/dataset/my_task"  # 로컬 데이터셋 경로
NEG_PAIRS_PATH  = "/home/bluepot/cl_ws/negative_pairs.json"  # precompute 결과
BASE_MODEL_PATH = "nvidia/GR00T-N1.5-3B"              # 사전학습 모델 (또는 로컬 경로)
OUTPUT_DIR      = Path("outputs/groot_cl")             # 체크포인트 저장 위치
DEVICE          = "cuda"

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
# Config
# ─────────────────────────────────────────────────────────────────────────────

config = GrootCLConfig(
    base_model_path=BASE_MODEL_PATH,
    # contrastive 설정
    contrastive_phase="phase1",        # train_loop 진입 전 set_contrastive_phase()로 덮어씀
    contrastive_latent_dim=256,
    contrastive_cnn_hidden_dim=128,
    contrastive_proj_hidden_dim=512,
    contrastive_triplet_margin=0.5,
    contrastive_loss_weight=1.0,       # train_loop에서 phase별로 덮어씀
    contrastive_backprop_backbone=True,
    contrastive_fallback_to_in_batch=False,
    # GR00T 학습 대상
    tune_llm=False,
    tune_visual=False,
    tune_projector=True,
    tune_diffusion_model=True,
    # 기타
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
logger.info("Dataset: %d frames, %d episodes", dataset.num_frames, dataset.num_episodes)


def collate_fn(batch: list[dict]) -> dict:
    """negative_action=None인 샘플을 zero tensor로 교체하고 collate."""
    for item in batch:
        if item.get("negative_action") is None:
            item["negative_action"] = torch.zeros_like(item["action"])
    return default_collate(batch)


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
#
# 주의: pre(batch) 내부의 batch_to_transition()은 negative_action 키를
#       EnvTransition으로 옮기지 않는다. 따라서 negative_action은 파이프라인
#       밖에서 NegativeActionNormalizeStep을 직접 호출해 정규화한다.
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
policy.to(DEVICE)
logger.info("Policy initialized: %s", policy.__class__.__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def preprocess(raw_batch: dict) -> dict:
    """전처리 파이프라인 실행 + negative_action 별도 정규화 후 병합."""
    # 1) negative_action을 raw_batch에서 꺼낸다 (파이프라인은 이 키를 무시)
    raw_neg = raw_batch.pop("negative_action", None)

    # 2) 나머지 batch 전처리 (state, action, image 정규화/패딩 등)
    processed = pre(raw_batch)

    # 3) negative_action 정규화 (pre와 동일한 min-max 방식)
    if isinstance(raw_neg, torch.Tensor):
        result = neg_normalizer({"negative_action": raw_neg})
        processed["negative_action"] = result.get("negative_action")

    return processed


def infinite_dataloader():
    while True:
        yield from dataloader


def save_checkpoint(tag: str) -> None:
    path = OUTPUT_DIR / tag
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
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
    logger.info("=" * 60)
    logger.info("Phase: %s | steps=%d | lr=%.2e | loss_weight=%.3f | backprop_backbone=%s",
                phase, total_steps, lr, loss_weight, backprop_backbone)
    logger.info("=" * 60)

    policy.set_contrastive_phase(phase)
    policy.config.contrastive_loss_weight = loss_weight
    policy.config.contrastive_backprop_backbone = backprop_backbone

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.1
    )

    policy.train()
    data_stream = infinite_dataloader()

    for step in range(1, total_steps + 1):
        raw_batch = next(data_stream)
        batch = preprocess(raw_batch)
        batch = to_device(batch, DEVICE)

        loss, loss_dict = policy(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if step % LOG_INTERVAL == 0:
            log_str = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            lr_now = scheduler.get_last_lr()[0]
            logger.info("[%s] step=%d/%d | lr=%.2e | %s", phase, step, total_steps, lr_now, log_str)

        if step % SAVE_INTERVAL == 0 or step == total_steps:
            save_checkpoint(f"{phase}/step_{step:06d}")

    logger.info("[%s] 완료", phase)


# ─────────────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 1: contrastive heads warm-up (_groot_model frozen)
train_loop(
    phase="phase1",
    total_steps=PHASE1_STEPS,
    lr=1e-4,
    loss_weight=1.0,      # flow matching 없음 (groot frozen) — contrastive만 의미 있음
    backprop_backbone=False,
)

# Phase 2a: joint fine-tuning (contrastive gradient → VLM backbone)
train_loop(
    phase="phase2a",
    total_steps=PHASE2A_STEPS,
    lr=2e-5,              # backbone 포함 fine-tuning이므로 낮은 LR
    loss_weight=0.05,     # flow_matching_loss 스케일에 맞게 조정
    backprop_backbone=True,   # 핵심: backbone까지 contrastive gradient 전달
)

logger.info("전체 학습 완료. 최종 체크포인트: %s", OUTPUT_DIR)
