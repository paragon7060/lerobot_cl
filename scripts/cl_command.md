# GR00T Contrastive Learning 학습 가이드

## 전제 조건

- 환경: `conda activate lerobot_050_groot`
- 작업 디렉터리: `/home/bluepot/cl_ws/lerobot_cl`
- GPU: CUDA 지원 필수 (bf16 학습 권장)
- 데이터셋: LeRobot 포맷 (v3.0) 로컬 또는 HuggingFace Hub

## setting lerobot050_groot
conda create -y -n lerobot_050_groot python=3.12
conda install ffmpeg=7.1.1 -c conda-forge
conda install -c nvidia cuda-toolkit=12.1 -y
pip install psutil
pip install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0" # --index-url https://download.pytorch.org/whl/cu1XX
pip install ninja "packaging>=24.2,<26.0" # flash attention dependencies
pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')"
pip install lerobot[groot]

---

## 전체 흐름

```
Step 0. 패키지 설치
Step 1. Negative Pairs 사전 계산
Step 2. 학습 실행 (Phase 1 → Phase 2a 자동)
```

---

## Step 0. 패키지 설치

```bash
cd /home/bluepot/cl_ws/lerobot_cl
conda activate lerobot_050_groot
pip install -e .
```

---

## Step 1. Negative Pairs 사전 계산

Hard Negative 매핑 파일을 생성합니다. 학습 전 **1회만** 실행합니다.

```bash
python scripts/precompute_negative_pairs.py \
    --repo_id YOUR_HF_REPO_ID \
    --root /path/to/local/dataset \
    --output_path /home/bluepot/cl_ws/negative_pairs.json \
    --seed 42
```

**인자 설명:**
| 인자 | 설명 | 예시 |
|------|------|------|
| `--repo_id` | HuggingFace 데이터셋 repo ID | `paragon7060/my_task` |
| `--root` | 로컬 데이터셋 경로. 없으면 HF Hub에서 자동 다운로드 | `/home/bluepot/cl_ws/dataset/my_task` |
| `--output_path` | 출력 JSON 경로 | `/home/bluepot/cl_ws/negative_pairs.json` |
| `--seed` | 랜덤 시드 (재현성) | `42` |

**출력 형식 (`negative_pairs.json`):**
```json
{
  "paragon7060/my_task": {
    "0_0": {"neg_episode_idx": 3, "neg_frame_idx": 0},
    "0_1": {"neg_episode_idx": 3, "neg_frame_idx": 1},
    ...
  }
}
```

> **주의**: 데이터셋에 task_index가 1종류뿐이면 경고 메시지가 출력됩니다.
> 이 경우 Step 2에서 `contrastive_fallback_to_in_batch=True`로 설정하세요.

---

## Step 2. 학습 스크립트 작성

아래 내용을 `scripts/train_groot_cl.py`로 저장합니다.

```python
#!/usr/bin/env python
"""GR00T Contrastive Learning 학습 스크립트."""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 경로 / 하이퍼파라미터 설정
# ─────────────────────────────────────────────────────────────────────────────

REPO_ID          = "paragon7060/my_task"                # 데이터셋 HF repo ID
DATASET_ROOT     = "/home/bluepot/cl_ws/dataset/my_task"
NEG_PAIRS_PATH   = "/home/bluepot/cl_ws/negative_pairs.json"
BASE_MODEL_PATH  = "nvidia/GR00T-N1.5-3B"              # 또는 로컬 체크포인트
OUTPUT_DIR       = Path("outputs/groot_cl")
DEVICE           = "cuda"

PHASE1_STEPS     = 500
PHASE2A_STEPS    = 5000
LOG_INTERVAL     = 50
SAVE_INTERVAL    = 1000

# ─────────────────────────────────────────────────────────────────────────────
# 설정 (Phase1 기준으로 초기화)
# ─────────────────────────────────────────────────────────────────────────────

config = GrootCLConfig(
    base_model_path=BASE_MODEL_PATH,
    # contrastive 설정
    contrastive_phase="phase1",
    contrastive_loss_weight=1.0,         # phase1: contrastive loss만 사용
    contrastive_backprop_backbone=False, # phase1: groot frozen이므로 무관
    contrastive_fallback_to_in_batch=False,
    contrastive_latent_dim=256,
    contrastive_triplet_margin=0.5,
    # 학습 대상 (phase1에서는 무관, phase2a에서 활성화)
    tune_llm=False,
    tune_visual=False,
    tune_projector=True,
    tune_diffusion_model=True,
    # 기타
    use_bf16=True,
    batch_size=4,
    chunk_size=16,
)

# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────────────────────────────────────

ds_meta = LeRobotDatasetMetadata(repo_id=REPO_ID, root=DATASET_ROOT)
delta_timestamps = resolve_delta_timestamps(config, ds_meta)

dataset = ContrastiveLeRobotDataset(
    repo_id=REPO_ID,
    root=DATASET_ROOT,
    negative_pairs_path=NEG_PAIRS_PATH,
    delta_timestamps=delta_timestamps,
)

def collate_fn(batch: list[dict]) -> dict:
    """negative_action=None인 샘플을 zero tensor로 대체."""
    action_shape = batch[0]["action"].shape  # (chunk_size, action_dim)
    for item in batch:
        if item.get("negative_action") is None:
            item["negative_action"] = torch.zeros_like(
                torch.tensor(action_shape, dtype=torch.float32)
            ).expand(action_shape)
    return default_collate(batch)

dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)

# ─────────────────────────────────────────────────────────────────────────────
# 전처리 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

pre, post = make_pre_post_processors(config, dataset_stats=ds_meta.stats)

# ─────────────────────────────────────────────────────────────────────────────
# 모델 초기화  (make_policy가 ds_meta로부터 input/output features를 자동 설정)
# ─────────────────────────────────────────────────────────────────────────────

policy = make_policy(config, ds_meta=ds_meta)   # GrootCLPolicy 반환
policy.to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def move_to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

def train_loop(
    phase: str,
    total_steps: int,
    lr: float,
    loss_weight: float,
    backprop_backbone: bool,
    save_dir: Path,
):
    logger.info(f"=== {phase} 시작 | steps={total_steps} | lr={lr} ===")
    policy.set_contrastive_phase(phase)
    policy.config.contrastive_loss_weight = loss_weight
    policy.config.contrastive_backprop_backbone = backprop_backbone

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=lr,
        betas=(0.95, 0.999),
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    policy.train()
    data_iter = iter(dataloader)
    step = 0

    while step < total_steps:
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            raw_batch = next(data_iter)

        batch = pre(raw_batch)
        batch = move_to_device(batch, DEVICE)

        loss, loss_dict = policy(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % LOG_INTERVAL == 0:
            log_parts = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            logger.info(f"[{phase}] step={step}/{total_steps} | {log_parts}")

        if step % SAVE_INTERVAL == 0 or step == total_steps:
            ckpt_path = save_dir / f"step_{step:06d}"
            policy.save_pretrained(ckpt_path)
            logger.info(f"체크포인트 저장: {ckpt_path}")

    logger.info(f"=== {phase} 완료 ===")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: contrastive heads warm-up (groot frozen)
# ─────────────────────────────────────────────────────────────────────────────

train_loop(
    phase="phase1",
    total_steps=PHASE1_STEPS,
    lr=1e-4,
    loss_weight=1.0,
    backprop_backbone=False,
    save_dir=OUTPUT_DIR / "phase1",
)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2a: joint fine-tuning (contrastive gradient → VLM backbone)
# ─────────────────────────────────────────────────────────────────────────────

train_loop(
    phase="phase2a",
    total_steps=PHASE2A_STEPS,
    lr=2e-5,
    loss_weight=0.05,
    backprop_backbone=True,
    save_dir=OUTPUT_DIR / "phase2a",
)

logger.info("학습 완료.")
```

---

## Step 2. 학습 실행

### 기본 실행 (단일 GPU)

```bash
conda activate lerobot_050_groot
cd /home/bluepot/cl_ws/lerobot_cl

python scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_cl \
    --job_name=groot_cl_run1 \
    --phase1_steps=500 \
    --phase2a_steps=5000 \
    --batch_size=4 \
    --wandb.enable=true \
    --wandb.project=groot_cl \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true
```

### 로그 파일 저장 포함

```bash
python scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    [기타 인자...] \
    2>&1 | tee outputs/train_log.txt
```

### tmux 세션에서 실행 (서버 권장)

```bash
tmux new-session -s groot_cl
conda activate lerobot_050_groot
cd /home/bluepot/cl_ws/lerobot_cl

python scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_cl \
    --job_name=groot_cl_run1 \
    --wandb.enable=true \
    --wandb.project=groot_cl \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/train_log.txt

# 세션 detach: Ctrl+B, D
# 세션 복귀:   tmux attach -t groot_cl
```

### 사전학습된 GrootPolicy 체크포인트에서 시작

```bash
python scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --groot_pretrained_path=/home/bluepot/cl_ws/outputs/groot_my_task/step_010000 \
    --output_dir=./outputs/groot_cl_finetune \
    --job_name=groot_cl_finetune \
    --wandb.enable=true \
    --wandb.project=groot_cl
```

---

## CLI 인자 전체 목록

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--repo_id` | `paragon7060/INSIGHTfixposV3` | HuggingFace 데이터셋 repo ID |
| `--root` | `None` | 로컬 데이터셋 경로 (없으면 HF Hub) |
| `--neg_pairs_path` | `/home/bluepot/cl_ws/negative_pairs.json` | precompute 결과 JSON |
| `--base_model_path` | `nvidia/GR00T-N1.5-3B` | GR00T 기본 모델 경로 |
| `--groot_pretrained_path` | `None` | 사전학습된 GrootPolicy 체크포인트 경로 |
| `--output_dir` | `outputs/groot_cl` | 체크포인트 저장 위치 |
| `--job_name` | `groot_cl` | WandB run 이름 및 식별자 |
| `--phase1_steps` | `500` | Phase 1 학습 스텝 수 |
| `--phase2a_steps` | `5000` | Phase 2a 학습 스텝 수 |
| `--phase1_lr` | `1e-4` | Phase 1 learning rate |
| `--phase2a_lr` | `2e-5` | Phase 2a learning rate |
| `--phase2a_loss_weight` | `0.05` | Phase 2a contrastive loss 가중치 |
| `--batch_size` | `4` | GPU당 배치 크기 |
| `--num_workers` | `4` | DataLoader 워커 수 |
| `--seed` | `42` | 랜덤 시드 |
| `--log_interval` | `50` | 로그 출력 주기 (steps) |
| `--save_interval` | `500` | 체크포인트 저장 주기 (steps) |
| `--contrastive_latent_dim` | `256` | Contrastive latent 차원 |
| `--contrastive_triplet_margin` | `0.5` | Triplet Loss margin |
| `--tune_llm` | `false` | LLM backbone 학습 여부 |
| `--tune_visual` | `false` | Visual encoder 학습 여부 |
| `--tune_projector` | `true` | Projector 학습 여부 |
| `--tune_diffusion_model` | `true` | Diffusion head 학습 여부 |
| `--wandb.enable` | `false` | WandB 로깅 활성화 |
| `--wandb.project` | `lerobot` | WandB 프로젝트 이름 |
| `--wandb.entity` | `None` | WandB 팀/사용자 |
| `--wandb.disable_artifact` | `false` | 체크포인트 artifact 업로드 비활성화 |
| `--wandb.notes` | `None` | WandB run 메모 |
| `--wandb.mode` | `None` | `online` / `offline` / `disabled` |

---

## Multi-GPU 실행 (accelerate)

### 최초 1회: accelerate 설정

```bash
conda activate lerobot_050_groot
accelerate config
```

대화형 프롬프트에서 아래처럼 선택:
```
- This machine / multi-GPU
- How many GPUs: 2  (또는 서버의 실제 GPU 수)
- Do you want to use DeepSpeed? No
- Do you want to use FullyShardedDataParallel? No
- What GPU ids: 0,1  (사용할 GPU 번호)
- mixed precision: no  (스크립트 내부에서 bf16 처리)
```

설정 파일은 `~/.cache/huggingface/accelerate/default_config.yaml`에 저장됨.

### config 파일 없이 명령줄로 직접 실행

```bash
# GPU 2장
accelerate launch \
    --num_processes 2 \
    --mixed_precision no \
    scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_cl_multi \
    --job_name=groot_cl_multi \
    --batch_size=4 \
    --wandb.enable=true \
    --wandb.project=groot_cl \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true
```

### tmux + accelerate (서버 권장)

```bash
tmux new-session -s groot_cl_multi
conda activate lerobot_050_groot
cd /home/bluepot/cl_ws/lerobot_cl

accelerate launch --num_processes 2 \
    scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_cl_multi \
    --job_name=groot_cl_multi \
    --wandb.enable=true \
    --wandb.project=groot_cl \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/train_log_multigpu.txt
```

### 특정 GPU 지정 실행

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --wandb.enable=true \
    --wandb.project=groot_cl
```

### Multi-GPU 동작 원리

| 항목 | 동작 |
|------|------|
| 모델 | DDP로 각 GPU에 복제, gradient all-reduce |
| DataLoader | DistributedSampler 자동 삽입 (accelerator.prepare 시) |
| Batch size | `BATCH_SIZE`는 **per-GPU** 크기. 총 effective batch = `BATCH_SIZE × num_processes` |
| Checkpoint | main process(rank 0)에서만 저장 (`accelerator.is_main_process`) |
| Loss | 각 GPU에서 forward/backward 후 gradient 동기화 (DDP 표준) |
| Triplet Loss | GPU 간 negative 공유 없이도 동작 (명시적 hard negative 사용) |

> **Batch size 주의**: GPU 4장에서 `BATCH_SIZE=4`이면 실제 effective batch = 16.
> Triplet Loss는 배치 크기에 민감하지 않으나 (hard negative 고정), flow matching은 클수록 안정적.

---

## 체크포인트에서 이어서 학습

Phase 1이 완료된 체크포인트에서 Phase 2a를 이어서 시작하려면:

```python
# train_groot_cl.py에서 Phase1 학습 블록을 건너뛰고 아래로 대체
from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy

policy = GrootCLPolicy.from_pretrained(
    "outputs/groot_cl/phase1/step_000500",
    config=config,
).to(DEVICE)

# Phase 2a 바로 실행
train_loop(
    phase="phase2a",
    ...
)
```

---

## 주요 파라미터 요약

| 파라미터 | Phase 1 | Phase 2a | Phase 2b |
|---------|---------|---------|---------|
| `contrastive_phase` | `"phase1"` | `"phase2a"` | `"phase2b"` |
| `contrastive_loss_weight` | `1.0` | `0.05` | `0.0` |
| `contrastive_backprop_backbone` | `False` | **`True`** | `False` |
| `tune_llm` | `False` | 선택 | `False` |
| learning rate | `1e-4` | `2e-5` | `1e-5` |
| 권장 steps | 500 | 5000 | 2000 |

> **Phase 2a 핵심**: `contrastive_backprop_backbone=True`가 반드시 설정되어야 VLM backbone까지 contrastive gradient가 흐릅니다. 이것이 실질적인 성능 향상의 핵심입니다.

---

## loss 모니터링 포인트

Phase 2a에서 아래 세 가지 loss를 함께 확인하세요:

```
flow_matching_loss: 0.XXXX  ← 낮을수록 좋음 (action 예측 품질)
contrastive_loss:   0.XXXX  ← 낮을수록 좋음 (VLM-action alignment)
loss:               0.XXXX  ← flow_matching + weight * contrastive
```

- `contrastive_loss`가 `flow_matching_loss`보다 10배 이상 크면 `contrastive_loss_weight`를 낮추세요 (`0.05` → `0.01`).
- `contrastive_loss`가 수렴하지 않으면 `contrastive_triplet_margin`을 조정하세요 (`0.5` → `0.2`).

---

## 파일 구조 참고

```
lerobot_cl/
├── scripts/
│   ├── precompute_negative_pairs.py   # Step 1
│   ├── train_groot_cl.py              # Step 2 (직접 작성)
│   └── cl_command.md                  # 이 파일
├── src/lerobot/
│   ├── datasets/
│   │   └── contrastive_dataset.py     # negative_action 포함 데이터셋
│   └── policies/groot_cl/
│       ├── configuration_groot_cl.py  # GrootCLConfig
│       ├── modeling_groot_cl.py       # GrootCLPolicy (phase 관리 + forward)
│       ├── processor_groot_cl.py      # negative_action 정규화
│       └── action_head/
│           └── contrastive_heads.py   # VLMContrastiveHead, ActionContrastiveHead
└── outputs/                           # 학습 결과 저장 위치
    ├── groot_cl/
    │   ├── phase1/
    │   │   └── step_000500/           # 체크포인트
    │   └── phase2a/
    │       └── step_005000/
    └── train_log.txt
```
