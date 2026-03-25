# GR00T INSIGHT 학습 가이드

## 전제 조건

- 환경: `conda activate lerobot_050_groot`
- 작업 디렉터리: `/home/bluepot/cl_ws/lerobot_cl`
- GPU: CUDA 지원 필수 (bf16 학습)
- 데이터셋: `paragon7060/INSIGHTfixposV3` (로컬 또는 HF Hub)

---

## 전체 파이프라인

```
Step 0. 패키지 설치
Step 1. Negative Pairs 사전 계산          ← CL 학습에만 필요
Step 2. Stage 1: Baseline 학습            ← train_groot_baseline.py
Step 3. Stage 2: Contrastive Learning 학습 ← train_groot_cl.py
```

Stage 1이 생성한 체크포인트를 Stage 2에서 초기 weight로 사용한다.

---

## Step 0. 패키지 설치

```bash
cd /home/bluepot/cl_ws/lerobot_cl
conda activate lerobot_050_groot
pip install -e .
```

---

## Step 1. Negative Pairs 사전 계산

CL 학습(Stage 2)에 필요한 Hard Negative 매핑 파일을 생성한다. **1회만** 실행.

```bash
python scripts/precompute_negative_pairs.py \
    --repo_id paragon7060/INSIGHTfixposV3 \
    --root /mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_path /home/seonho/clvla/dataset/negative_mapping/output/insight_negative_pairs.json \
    --seed 42
```

| 인자 | 설명 |
|------|------|
| `--repo_id` | HuggingFace 데이터셋 repo ID |
| `--root` | 로컬 데이터셋 경로. 없으면 HF Hub에서 다운로드 |
| `--output_path` | 출력 JSON 경로 |
| `--seed` | 랜덤 시드 |

---

## Step 2. Stage 1 — Baseline 학습

표준 GR00T policy를 INSIGHT 데이터로 fine-tuning한다.
LLM backbone에 LoRA(rank=16)를 적용하여 효율적으로 도메인 적응.

### 단일 GPU

```bash
conda activate lerobot_050_groot
cd /home/bluepot/cl_ws/lerobot_cl

python scripts/train_groot_baseline.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_baseline \
    --job_name=groot_baseline_v1 \
    --total_steps=50000 \
    --batch_size=128 \
    --lora_rank=16 \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/baseline_log.txt
```

### Multi-GPU (예: 4 GPU, effective BS = 4 × 32 = 128)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=no \
    scripts/train_groot_baseline.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_baseline_insight \
    --job_name=groot_baseline_insight_v1 \
    --total_steps=50000 \
    --batch_size=32 \
    --lora_rank=16 \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/baseline_log.txt
```

### tmux 세션 (서버 권장)

```bash
tmux new-session -s groot_baseline
conda activate lerobot_050_groot
cd /home/bluepot/cl_ws/lerobot_cl

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=no \
    scripts/train_groot_baseline.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_baseline \
    --job_name=groot_baseline_v1 \
    --total_steps=50000 \
    --batch_size=32 \
    --lora_rank=16 \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/baseline_log.txt

# detach: Ctrl+B, D  /  재접속: tmux attach -t groot_baseline
```

### Stage 1 CLI 인자 전체 목록

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--repo_id` | `paragon7060/INSIGHTfixposV3` | HuggingFace 데이터셋 repo ID |
| `--root` | `None` | 로컬 데이터셋 경로 |
| `--video_backend` | `pyav` | 비디오 디코더 (`pyav` 권장) |
| `--base_model_path` | `nvidia/GR00T-N1.5-3B` | GR00T 기반 모델 |
| `--output_dir` | `outputs/groot_baseline` | 체크포인트 저장 위치 |
| `--job_name` | `groot_baseline` | WandB run 이름 |
| `--total_steps` | `50000` | 총 학습 스텝 수 |
| `--lr` | `1e-4` | Learning rate (cosine decay) |
| `--warmup_steps` | `2500` | Warmup 스텝 수 (총의 5%) |
| `--batch_size` | `128` | GPU당 배치 크기 |
| `--num_workers` | `8` | DataLoader 워커 수 |
| `--seed` | `42` | 랜덤 시드 |
| `--log_interval` | `100` | 로그 출력 주기 (steps) |
| `--save_interval` | `5000` | 체크포인트 저장 주기 (steps) |
| `--n_obs_steps` | `1` | 관측 timestep 수 |
| `--chunk_size` | `50` | Action chunk 크기 |
| `--n_action_steps` | `50` | 한 번 추론 후 실행할 action 수 |
| `--tune_llm` | `false` | LLM 전체 fine-tuning 여부 (LoRA로 대체) |
| `--tune_visual` | `false` | Visual encoder 학습 여부 |
| `--tune_projector` | `true` | Projector 학습 여부 |
| `--tune_diffusion_model` | `true` | Diffusion head 학습 여부 |
| `--lora_rank` | `16` | LoRA rank (0이면 LoRA 비활성화) |
| `--lora_alpha` | `32` | LoRA alpha (보통 2× rank) |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--wandb.enable` | `false` | WandB 로깅 활성화 |
| `--wandb.project` | `lerobot` | WandB 프로젝트 이름 |
| `--wandb.entity` | `None` | WandB 팀/사용자 |
| `--wandb.disable_artifact` | `false` | 체크포인트 artifact 업로드 비활성화 |
| `--wandb.notes` | `None` | WandB run 메모 |
| `--wandb.mode` | `None` | `online` / `offline` / `disabled` |

### Stage 1 체크포인트 구조

```
outputs/groot_baseline/
├── step_005000/       ← save_interval마다 저장
├── step_010000/
├── ...
└── final/             ← 학습 완료 후 최종 저장
```

---

## Step 3. Stage 2 — Contrastive Learning 학습

Stage 1의 `final/` 체크포인트를 초기 weight로 사용하여 CL fine-tuning.
Phase 1(contrastive heads 워밍업) → Phase 2a(joint fine-tuning) 순으로 자동 진행.

### 단일 GPU

```bash
python scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --neg_pairs_path=/home/seonho/clvla/dataset/negative_mapping/output/insight_negative_pairs.json \
    --groot_pretrained_path=./outputs/groot_baseline/final \
    --output_dir=./outputs/groot_cl_v1 \
    --job_name=groot_cl_v1 \
    --phase1_steps=2000 \
    --phase2a_steps=15000 \
    --batch_size=16 \
    --lora_rank=16 \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    2>&1 | tee outputs/cl_log.txt
```

### Multi-GPU (예: 4 GPU, effective BS = 4 × 4 = 16)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=no \
    scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --neg_pairs_path=/home/seonho/clvla/dataset/negative_mapping/output/insight_negative_pairs.json \
    --groot_pretrained_path=./outputs/groot_baseline/final \
    --output_dir=./outputs/groot_cl_v1 \
    --job_name=groot_cl_v1 \
    --phase1_steps=2000 \
    --phase2a_steps=15000 \
    --batch_size=4 \
    --lora_rank=16 \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    2>&1 | tee outputs/cl_log.txt
```

### tmux 세션 (서버 권장)

```bash
tmux new-session -s groot_cl
conda activate lerobot_050_groot
cd /home/bluepot/cl_ws/lerobot_cl

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=no \
    scripts/train_groot_cl.py \
    --repo_id=paragon7060/INSIGHTfixposV3 \
    --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --neg_pairs_path=/home/seonho/clvla/dataset/negative_mapping/output/insight_negative_pairs.json \
    --groot_pretrained_path=./outputs/groot_baseline/final \
    --output_dir=./outputs/groot_cl_v1 \
    --job_name=groot_cl_v1 \
    --batch_size=4 \
    --lora_rank=16 \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/cl_log.txt

# detach: Ctrl+B, D  /  재접속: tmux attach -t groot_cl
```

### Stage 2 CLI 인자 전체 목록

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--repo_id` | `paragon7060/INSIGHTfixposV3` | HuggingFace 데이터셋 repo ID |
| `--root` | `None` | 로컬 데이터셋 경로 |
| `--neg_pairs_path` | `/home/bluepot/cl_ws/negative_pairs.json` | precompute 결과 JSON |
| `--video_backend` | `pyav` | 비디오 디코더 |
| `--base_model_path` | `nvidia/GR00T-N1.5-3B` | GR00T 기반 모델 |
| `--groot_pretrained_path` | `None` | Stage 1 체크포인트 경로 (`final/` 디렉터리) |
| `--output_dir` | `outputs/groot_cl` | 체크포인트 저장 위치 |
| `--job_name` | `groot_cl` | WandB run 이름 |
| `--phase1_steps` | `2000` | Phase 1 학습 스텝 수 |
| `--phase2a_steps` | `15000` | Phase 2a 학습 스텝 수 |
| `--phase1_lr` | `1e-4` | Phase 1 learning rate |
| `--phase2a_lr` | `2e-5` | Phase 2a learning rate |
| `--phase2a_loss_weight` | `0.05` | Phase 2a contrastive loss 가중치 |
| `--batch_size` | `4` | GPU당 배치 크기 |
| `--num_workers` | `4` | DataLoader 워커 수 |
| `--seed` | `42` | 랜덤 시드 |
| `--log_interval` | `50` | 로그 출력 주기 (steps) |
| `--save_interval` | `500` | 체크포인트 저장 주기 (steps) |
| `--chunk_size` | `50` | Action chunk 크기 (Stage 1과 반드시 일치) |
| `--n_action_steps` | `50` | 한 번 추론 후 실행할 action 수 |
| `--tune_llm` | `false` | LLM 전체 fine-tuning 여부 |
| `--tune_visual` | `false` | Visual encoder 학습 여부 |
| `--tune_projector` | `true` | Projector 학습 여부 |
| `--tune_diffusion_model` | `true` | Diffusion head 학습 여부 |
| `--lora_rank` | `16` | LoRA rank (Stage 1과 반드시 일치) |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--contrastive_latent_dim` | `256` | Contrastive latent 차원 |
| `--contrastive_triplet_margin` | `0.5` | Triplet Loss margin |
| `--wandb.enable` | `false` | WandB 로깅 활성화 |
| `--wandb.project` | `lerobot` | WandB 프로젝트 이름 |
| `--wandb.entity` | `None` | WandB 팀/사용자 |
| `--wandb.disable_artifact` | `false` | 체크포인트 artifact 업로드 비활성화 |
| `--wandb.notes` | `None` | WandB run 메모 |
| `--wandb.mode` | `None` | `online` / `offline` / `disabled` |

### Stage 2 학습 단계 설명

| Phase | 목적 | steps | lr | backbone |
|-------|------|-------|----|----------|
| Phase 1 | Contrastive heads 워밍업. `_groot_model` 전체 frozen | 2,000 | 1e-4 | Frozen |
| Phase 2a | Joint fine-tuning. LoRA + projector + diffusion head 함께 학습 | 15,000 | 2e-5 | LoRA 활성화 |

### Stage 2 체크포인트 구조

```
outputs/groot_cl_v1/
├── phase1/
│   ├── step_000500/
│   ├── ...
│   └── step_002000/
└── phase2a/
    ├── step_000500/
    ├── ...
    └── step_015000/
```

---

## 주의사항

### Stage 1 → Stage 2 체크포인트 연결

`--groot_pretrained_path`에는 **반드시 `final/` 디렉터리**를 지정한다.

```bash
# Stage 1이 생성하는 최종 체크포인트
outputs/groot_baseline/final/model.safetensors  ← 이 파일을 읽음

# 올바른 지정
--groot_pretrained_path=./outputs/groot_baseline/final
```

### LoRA rank 일치

Stage 1과 Stage 2의 `--lora_rank`가 **반드시 동일**해야 한다.
Stage 1에서 훈련된 LoRA adapter weights가 Stage 2 초기화에 그대로 복원된다.

### Multi-GPU batch_size

`--batch_size`는 **GPU당 크기**이다.

| GPU 수 | `--batch_size` | Effective BS |
|--------|----------------|--------------|
| 1 | 128 | 128 |
| 4 | 32 | 128 |
| 8 | 16 | 128 |

Stage 2 CL 학습의 경우:

| GPU 수 | `--batch_size` | Effective BS |
|--------|----------------|--------------|
| 1 | 16 | 16 |
| 4 | 4 | 16 |

### mixed_precision=no

`--mixed_precision=no` 는 필수. 스크립트 내부에서 `torch.autocast(bfloat16)`을 직접 사용하므로 accelerate의 autocast와 중복 적용을 방지한다.

### loss 모니터링 (Stage 2)

```
flow_matching_loss: 0.XXXX  ← action 예측 품질
contrastive_loss:   0.XXXX  ← VLM-action alignment
loss:               0.XXXX  ← flow_matching + weight * contrastive
```

- `contrastive_loss`가 `flow_matching_loss`보다 10배 이상 크면 `--phase2a_loss_weight`를 낮춰라 (`0.05` → `0.01`).
- Phase 1에서 `contrastive_loss`가 수렴하지 않으면 `--contrastive_triplet_margin`을 조정하라 (`0.5` → `0.2`).

---

## 파일 구조

```
lerobot_cl/
├── scripts/
│   ├── precompute_negative_pairs.py   ← Step 1
│   ├── train_groot_baseline.py        ← Step 2 (Stage 1)
│   ├── train_groot_cl.py              ← Step 3 (Stage 2)
│   └── cl_command.md                  ← 이 파일
├── src/lerobot/
│   ├── datasets/
│   │   ├── lerobot_dataset.py         ← Stage 1 표준 데이터셋
│   │   └── contrastive_dataset.py     ← Stage 2 negative_action 포함 데이터셋
│   └── policies/
│       ├── groot/
│       │   ├── configuration_groot.py ← GrootConfig (lora_rank 포함)
│       │   ├── modeling_groot.py      ← GrootPolicy
│       │   └── groot_n1.py            ← GR00TN15 (manual safetensors + LoRA)
│       └── groot_cl/
│           ├── configuration_groot_cl.py ← GrootCLConfig
│           ├── modeling_groot_cl.py      ← GrootCLPolicy (phase 관리 + LoRA 복원)
│           ├── groot_n1.py               ← GR00TN15 CL버전 (LoRA 지원)
│           └── action_head/
│               └── contrastive_heads.py  ← VLMContrastiveHead, ActionContrastiveHead
└── outputs/
    ├── groot_baseline/                ← Stage 1 결과
    │   ├── step_005000/
    │   └── final/                     ← Stage 2 입력으로 사용
    └── groot_cl_v1/                   ← Stage 2 결과
        ├── phase1/
        └── phase2a/
```
