# GR00T INSIGHT 학습 가이드

환경: `conda activate lerobot_050_groot` / 작업 디렉터리: `/home/bluepot/cl_ws/lerobot_cl`

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
    --output_path /home/bluepot/cl_ws/negative_pairs.json \
    --seed 42
```

| 인자 | 설명 |
|---|---|
| `--repo_id` | HuggingFace 데이터셋 repo ID |
| `--root` | 로컬 데이터셋 경로 |
| `--output_path` | 출력 JSON 경로 (`--neg_pairs_path`에 동일하게 지정) |
| `--seed` | 랜덤 시드 |

---

## Step 2. Stage 1 — Baseline 학습

→ `baseline_command.md` 참조.

학습 완료 후 최신 체크포인트 경로:
```
outputs/groot_baseline/checkpoints/last_checkpoint/pretrained_model/
```

---

## Step 3. Stage 2 — Contrastive Learning 학습

Stage 1의 체크포인트를 초기 weight로 사용.
Phase 1(contrastive heads 워밍업) → Phase 2a(joint fine-tuning) 순으로 자동 진행.

### CLI 인수 구조

| 그룹 | 접두사 | 예시 |
|---|---|---|
| 정책 타입 선택 | `--policy.type` | `--policy.type=groot_cl` (**필수**) |
| 데이터셋 | `--dataset.*` | `--dataset.repo_id=X` |
| 정책/모델 | `--policy.*` | `--policy.lora_rank=16` |
| WandB | `--wandb.*` | `--wandb.enable=true` |
| 학습 루프 / CL | (최상위) | `--phase1_steps=2000` |

> `--policy.type=groot_cl` 은 필수 인수야. 이 값이 없으면 `--policy.*` CLI override가 적용되지 않음.

### 단일 GPU

```bash
python scripts/train_groot_cl.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --dataset.video_backend=pyav \
    --neg_pairs_path=/home/bluepot/cl_ws/insight_negative.json \
    --policy.type=groot_cl \
    --policy.groot_pretrained_path=./outputs/groot_baseline/checkpoints/last_checkpoint/pretrained_model \
    --output_dir=./outputs/groot_cl_v1 \
    --job_name=groot_cl_v1 \
    --phase1_steps=2000 \
    --phase2a_steps=15000 \
    --batch_size=32 \
    --policy.lora_rank=16 \
    --policy.lora_target=vision \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    2>&1 | tee outputs/cl_log.txt
```

### Multi-GPU (4 GPU, effective BS = 4 × 4 = 16)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=no \
    scripts/train_groot_cl.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --neg_pairs_path=/home/bluepot/cl_ws/negative_pairs.json \
    --policy.type=groot_cl \
    --policy.groot_pretrained_path=./outputs/groot_baseline/checkpoints/last_checkpoint/pretrained_model \
    --output_dir=./outputs/groot_cl_v1 \
    --job_name=groot_cl_v1 \
    --phase1_steps=2000 \
    --phase2a_steps=15000 \
    --batch_size=4 \
    --policy.lora_rank=16 \
    --policy.lora_target=vision \
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
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --neg_pairs_path=/home/bluepot/cl_ws/negative_pairs.json \
    --policy.type=groot_cl \
    --policy.groot_pretrained_path=./outputs/groot_baseline/checkpoints/last_checkpoint/pretrained_model \
    --output_dir=./outputs/groot_cl_v1 \
    --job_name=groot_cl_v1 \
    --batch_size=4 \
    --policy.lora_rank=16 \
    --policy.lora_target=vision \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/cl_log.txt

# detach: Ctrl+B, D  /  재접속: tmux attach -t groot_cl
```

---

## 전체 인수 목록

### 데이터셋 (`--dataset.*`)

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--dataset.repo_id` | `paragon7060/INSIGHTfixposV3` | HuggingFace 데이터셋 repo ID |
| `--dataset.root` | `None` | 로컬 경로 |
| `--dataset.video_backend` | `pyav` | 비디오 디코더 |

### 정책/모델 (`--policy.*`)

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--policy.base_model_path` | `nvidia/GR00T-N1.5-3B` | 베이스 GR00T 모델 |
| `--policy.groot_pretrained_path` | `None` | Stage 1 체크포인트 (`pretrained_model/` 디렉터리) |
| `--policy.lora_rank` | `0` | LoRA rank (Stage 1과 반드시 일치) |
| `--policy.lora_alpha` | `16` | LoRA alpha |
| `--policy.lora_dropout` | `0.1` | LoRA dropout |
| `--policy.lora_target` | `llm` | LoRA 대상: `vision` / `llm` / `both` |
| `--policy.tune_llm` | `false` | LLM base weight 학습 여부 |
| `--policy.tune_visual` | `false` | Vision tower base weight 학습 여부 |
| `--policy.tune_projector` | `true` | Projector 학습 여부 |
| `--policy.tune_diffusion_model` | `true` | Diffusion head 학습 여부 |
| `--policy.contrastive_latent_dim` | `256` | Contrastive latent 차원 |
| `--policy.contrastive_triplet_margin` | `0.5` | Triplet Loss margin |
| `--policy.contrastive_cnn_hidden_dim` | `128` | CNN hidden 차원 |
| `--policy.contrastive_proj_hidden_dim` | `512` | Projection head hidden 차원 |
| `--policy.chunk_size` | `50` | Action chunk 크기 (Stage 1과 반드시 일치) |
| `--policy.n_action_steps` | `50` | Action prediction horizon |

### 학습 루프 / CL 설정

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--neg_pairs_path` | `/home/bluepot/cl_ws/negative_pairs.json` | Step 1 출력 JSON |
| `--phase1_steps` | `2000` | Phase 1 스텝 수 |
| `--phase2a_steps` | `15000` | Phase 2a 스텝 수 |
| `--phase1_lr` | `1e-4` | Phase 1 학습률 |
| `--phase2a_lr` | `2e-5` | Phase 2a 학습률 |
| `--phase2a_warmup_steps` | `500` | Phase 2a warmup 스텝 |
| `--phase2a_loss_weight` | `0.05` | Phase 2a contrastive loss 가중치 |
| `--batch_size` | `4` | GPU당 배치 크기 |
| `--num_workers` | `4` | DataLoader worker 수 |
| `--log_freq` | `50` | 로그 출력 간격 (스텝) |
| `--save_freq` | `500` | 체크포인트 저장 간격 (스텝) |
| `--seed` | `42` | 랜덤 시드 |
| `--output_dir` | `outputs/groot_cl` | 출력 디렉터리 |
| `--job_name` | `groot_cl` | 실험 이름 |

### WandB (`--wandb.*`)

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--wandb.enable` | `false` | WandB 로깅 활성화 |
| `--wandb.project` | `lerobot` | WandB 프로젝트 이름 |
| `--wandb.entity` | `None` | WandB 팀/엔티티 |
| `--wandb.disable_artifact` | `false` | artifact 업로드 비활성화 |
| `--wandb.run_id` | `None` | 기존 run 재개 시 run ID |
| `--wandb.mode` | `None` | `online` / `offline` / `disabled` |

---

## 학습 단계 설명

| Phase | 목적 | steps | lr | backbone |
|---|---|---|---|---|
| Phase 1 | Contrastive heads 워밍업. `_groot_model` 전체 frozen | 2,000 | 1e-4 | Frozen |
| Phase 2a | Joint fine-tuning. LoRA + projector + diffusion + contrastive heads 함께 학습 | 15,000 | 2e-5 | LoRA 활성화 |

---

## 체크포인트 구조

```
output_dir/
└── checkpoints/
    ├── 000500/                  ← Phase 1, step 500 (global step 기준)
    │   ├── pretrained_model/
    │   └── training_state/
    ├── 001000/
    │   └── ...
    ├── 002000/                  ← Phase 1 마지막 (global step 2000)
    ├── 002500/                  ← Phase 2a, save_freq 도달 첫 번째
    │   └── ...
    ├── 017000/                  ← Phase 2a 마지막 (global step 17000)
    └── last_checkpoint -> 017000/
```

---

## 주의사항

### Stage 1 → Stage 2 체크포인트 연결

`--policy.groot_pretrained_path`에는 반드시 `pretrained_model/` 디렉터리를 지정한다.

```bash
# last_checkpoint symlink 사용 (권장)
--policy.groot_pretrained_path=./outputs/groot_baseline/checkpoints/last_checkpoint/pretrained_model

# 특정 스텝 지정
--policy.groot_pretrained_path=./outputs/groot_baseline/checkpoints/050000/pretrained_model
```

### LoRA rank 일치

Stage 1과 Stage 2의 `--policy.lora_rank`가 반드시 동일해야 한다.
Stage 1 LoRA adapter weights가 Stage 2 초기화에 복원된다.

### loss 모니터링 (Phase 2a)

```
flow_matching_loss: 0.XXXX  ← action 예측 품질
contrastive_loss:   0.XXXX  ← VLM-action alignment
loss:               0.XXXX  ← flow_matching + weight × contrastive
```

- `contrastive_loss`가 `flow_matching_loss`보다 10배 이상 크면 `--phase2a_loss_weight`를 낮춰라 (`0.05` → `0.01`).
- Phase 1에서 `contrastive_loss`가 수렴하지 않으면 `--policy.contrastive_triplet_margin`을 조정하라 (`0.5` → `0.2`).

### Multi-GPU batch_size

| GPU 수 | `--batch_size` | Effective BS |
|---|---|---|
| 1 | 16 | 16 |
| 4 | 4 | 16 |
| 8 | 2 | 16 |

### mixed_precision=no

`--mixed_precision=no` 필수. 스크립트 내부에서 `torch.autocast(bfloat16)`을 직접 사용하므로 accelerate autocast와 중복 방지.

---

## 파일 구조

```
lerobot_cl/
├── scripts/
│   ├── precompute_negative_pairs.py   ← Step 1
│   ├── train_groot_baseline.py        ← Step 2 (Stage 1)
│   ├── train_groot_cl.py              ← Step 3 (Stage 2)
│   ├── baseline_command.md            ← Stage 1 명령어
│   └── cl_command.md                  ← 이 파일
├── src/lerobot/
│   ├── datasets/
│   │   ├── lerobot_dataset.py         ← Stage 1 표준 데이터셋
│   │   └── contrastive_dataset.py     ← Stage 2 negative_action 포함 데이터셋
│   └── policies/
│       ├── groot/
│       │   ├── configuration_groot.py ← GrootConfig
│       │   └── modeling_groot.py      ← GrootPolicy
│       └── groot_cl/
│           ├── configuration_groot_cl.py ← GrootCLConfig
│           ├── modeling_groot_cl.py      ← GrootCLPolicy (phase 관리 + LoRA 복원)
│           └── action_head/
│               └── contrastive_heads.py  ← VLMContrastiveHead, ActionContrastiveHead
└── outputs/
    ├── groot_baseline/checkpoints/    ← Stage 1 결과
    │   ├── 050000/pretrained_model/   ← Stage 2 입력으로 사용
    │   └── last_checkpoint -> 050000/
    └── groot_cl_v1/checkpoints/       ← Stage 2 결과
```
