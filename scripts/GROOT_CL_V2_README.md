# GR00T-CL v2: Action-Guided Contrastive VLM Finetuning

## 개요

GR00T-N1.5의 VLM backbone을 action space에 align하는 2단계 학습 파이프라인.

**해결하는 문제**: Occlusion ambiguity
- 손목이 가려진 상황에서 joint 7 (wrist)이 왼쪽 또는 오른쪽으로 돌아가는 경우가 시각적으로 구분되지 않음
- 기본 VLM은 동일한 이미지 → 동일한 embedding → action head가 방향 구분 불가

**해결 방법**:
1. **Phase 1**: Action expert를 weighted FM loss로 학습 (wrist joint 강조)
2. **Phase 2**: RKD (Relational Knowledge Distillation)로 VLM을 finetuning
   - Action encoder (teacher, frozen)의 pairwise 관계 구조를 VLM (student)이 따라가도록 학습
   - 같은 방향 회전 샘플 → VLM embedding이 가깝게, 다른 방향 → 멀게

---

## 학습 파이프라인

```
NVIDIA GR00T-N1.5-3B (pretrained)
         │
    ┌────▼────┐
    │ Phase 1 │  Action Expert만 학습 (VLM frozen)
    │ FM Loss │  joint 7 (wrist) ×5 가중치
    └────┬────┘
         │  checkpoint
    ┌────▼────┐
    │ Phase 2 │  VLM finetuning (Action Expert frozen)
    │ RKD Loss│  Teacher: Action Encoder
    └────┬────┘  Student: VLM + VLMProjector
         │  checkpoint
    (Evaluation / Deployment)
```

---

## 환경 설정

```bash
# Conda 환경 활성화
conda activate lerobot050_groot

# lerobot 설치 (editable)
cd lerobot_cl
pip install -e .
```

---

## Dataset Config 준비

새로운 데이터셋에서 학습하려면 dataset config JSON을 만들어야 합니다.

### Config 파일 구조

```json
{
    "cameras": [
        "observation.images.CAMERA_NAME_1",
        "observation.images.CAMERA_NAME_2"
    ],
    "task_prompts": {
        "TASK_OR_SCENE_ID": "Task instruction text.",
        "ANOTHER_TASK": "Another instruction."
    }
}
```

- **cameras**: 학습에 사용할 카메라만 지정 (나머지 필터링). 빈 배열 `[]`이면 모든 카메라 사용.
- **task_prompts**: task_index에 대응하는 자연어 지시문.
  - **표준 LeRobot 형식** (`tasks.parquet`에 `task` column 존재): 자동으로 읽으므로 생략 가능
  - **비표준 형식** (INSIGHTfixposV3 등): scene_name → prompt 매핑 직접 작성

### 기존 Config 예시

```
scripts/dataset_configs/
├── INSIGHTfixposV3.json   # paragon7060/INSIGHTfixposV3
└── template.json          # 새 dataset 작성용 템플릿
```

### 새 Dataset Config 작성법

```bash
# 1. dataset의 카메라 목록 확인
python -c "
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata(repo_id='YOUR/DATASET', root='/path/to/data')
print([k for k in meta.features.keys() if k.startswith('observation.images')])
"

# 2. tasks.parquet 구조 확인
python -c "
import pandas as pd
df = pd.read_parquet('/path/to/data/meta/tasks.parquet')
print(df)
"

# 3. template 복사 후 수정
cp scripts/dataset_configs/template.json scripts/dataset_configs/YOUR_DATASET.json
```

---

## Phase 1 학습

VLM backbone을 freeze하고 action expert만 학습.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/train_groot_cl_v2_phase1.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/path/to/data/paragon7060/INSIGHTfixposV3 \
    --dataset.video_backend=pyav \
    --policy.type=groot_cl_v2 \
    --dataset_config=scripts/dataset_configs/INSIGHTfixposV3.json \
    --output_dir=./outputs/groot_cl_v2_phase1 \
    --job_name=groot_cl_v2_phase1 \
    --phase1_steps=50000 \
    --batch_size=32 \
    --lr=1e-4 \
    --wandb.enable=true \
    --wandb.project=groot_cl_v2 \
    --wandb.entity=YOUR_ENTITY \
    > outputs/phase1.log 2>&1 &
```

### 주요 하이퍼파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--phase1_steps` | 50000 | 학습 step 수 |
| `--batch_size` | 4 | 배치 크기 (A100 80GB 기준 32 가능) |
| `--lr` | 1e-4 | 학습률 |
| `--policy.joint_fm_weights` | `[1,1,1,1,1,1,5,1]` | 각 joint FM loss 가중치 (index 6 = wrist) |

---

## Phase 2 학습 (RKD)

Phase 1 체크포인트를 로드해서 VLM을 RKD로 finetuning.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/train_groot_cl_v2_phase2.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/path/to/data/paragon7060/INSIGHTfixposV3 \
    --dataset.video_backend=pyav \
    --policy.type=groot_cl_v2 \
    --policy.groot_pretrained_path=./outputs/groot_cl_v2_phase1/checkpoints/050000 \
    --dataset_config=scripts/dataset_configs/INSIGHTfixposV3.json \
    --output_dir=./outputs/groot_cl_v2_phase2 \
    --job_name=groot_cl_v2_phase2_rkd \
    --phase2_steps=10000 \
    --batch_size=32 \
    --lr=2e-5 \
    --warmup_steps=500 \
    --wandb.enable=true \
    --wandb.project=groot_cl_v2 \
    --wandb.entity=YOUR_ENTITY \
    > outputs/phase2.log 2>&1 &
```

### 주요 하이퍼파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--phase2_steps` | 10000 | 학습 step 수 |
| `--batch_size` | 32 | 배치 크기 (클수록 diverse negative 확보) |
| `--lr` | 2e-5 | 학습률 (Phase 1보다 낮게) |
| `--warmup_steps` | 500 | LR warmup |
| `--policy.cl_v2_action_temp` | 0.1 | Teacher 분포 sharpness (↓ = harder positive) |
| `--policy.cl_v2_vlm_temp` | 0.07 | Student logit temperature |
| `--policy.cl_v2_loss_weight` | 0.1 | RKD loss weight |
| `--policy.cl_v2_fm_loss_weight` | 0.01 | FM loss weight (VLM 품질 유지용) |

---

## Multi-GPU 학습

```bash
# Phase 1: 2-GPU
accelerate launch --num_processes 2 \
    scripts/train_groot_cl_v2_phase1.py \
    [same args as above]

# Phase 2: 2-GPU
accelerate launch --num_processes 2 \
    scripts/train_groot_cl_v2_phase2.py \
    [same args as above]
```

---

## 체크포인트 구조

```
outputs/groot_cl_v2_phase1/
└── checkpoints/
    └── 050000/           ← phase1_steps
        ├── model.safetensors
        ├── config.json
        └── ...

outputs/groot_cl_v2_phase2/
└── checkpoints/
    └── 010000/           ← phase2_steps
        ├── model.safetensors
        ├── config.json
        └── ...
```

Phase 2는 Phase 1 체크포인트의 `model.safetensors`에서 `_groot_model.*` 키를 로드.

---

## 모델 구조 (groot_cl_v2)

```
GrootCLv2Policy
├── _groot_model: GR00TN15
│   ├── backbone: Eagle2.5-VL (SiglipViT + Qwen3 LLM)
│   └── action_head: FlowMatchingActionHead
│       ├── action_encoder: MultiEmbodimentActionEncoder (MLP, per-step)
│       └── model: DiT
└── vlm_projector: VLMProjector   ← Phase 2 추가
    └── Linear(1536→512) → GELU → Linear(512→256) → L2 norm
```

**Phase별 freeze/unfreeze:**

| | Phase 1 | Phase 2 |
|--|---------|---------|
| VLM backbone | ❄️ Frozen | 🔥 Trainable |
| Action encoder | 🔥 Trainable | ❄️ Frozen |
| DiT | 🔥 Trainable | ❄️ Frozen |
| VLMProjector | ❄️ Frozen | 🔥 Trainable |

---

## RKD 수식

```
Teacher: z_a = L2_norm( mean_t[ ActionEncoder(action, t=999) ] )  ∈ ℝ^1536
Student: z_v = VLMProjector( masked_mean[ VLMBackbone(obs) ] )    ∈ ℝ^256

S_a = z_a @ z_a.T    P_a = softmax(S_a / τ_act)   [Teacher, no grad]
S_v = z_v @ z_v.T

L_RKD = KL( P_a ‖ softmax(S_v / τ_vlm) )
```

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `src/lerobot/policies/groot_cl_v2/modeling_groot_cl_v2.py` | 모델 구현 (Phase 1/2, RKD loss, VLMProjector) |
| `src/lerobot/policies/groot_cl_v2/configuration_groot_cl_v2.py` | 설정 (joint_fm_weights, RKD 하이퍼파라미터 등) |
| `scripts/train_groot_cl_v2_phase1.py` | Phase 1 학습 스크립트 |
| `scripts/train_groot_cl_v2_phase2.py` | Phase 2 학습 스크립트 |
| `scripts/dataset_configs/INSIGHTfixposV3.json` | INSIGHTfixposV3 dataset config |
| `scripts/dataset_configs/template.json` | 새 dataset config 템플릿 |
| `RKD_loss_plan.md` | RKD loss 상세 설계 문서 |
