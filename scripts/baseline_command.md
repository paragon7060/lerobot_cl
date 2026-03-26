# GR00T Baseline 학습 명령어

환경: `conda activate lerobot_050_groot` / 작업 디렉터리: `/home/bluepot/cl_ws/lerobot_cl`

## CLI 인수 구조

| 그룹 | 접두사 | 예시 |
|---|---|---|
| 데이터셋 | `--dataset.*` | `--dataset.repo_id=X` |
| 정책/모델 | `--policy.*` | `--policy.lora_rank=16` |
| WandB | `--wandb.*` | `--wandb.enable=true` |
| 학습 루프 | (최상위) | `--steps=50000` |

---

## 단일 GPU — Vision LoRA (권장, ~12 GB VRAM)

```bash
python scripts/train_groot_baseline.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --dataset.video_backend=pyav \
    --output_dir=./outputs/groot_baseline \
    --job_name=groot_baseline_v1 \
    --steps=50000 \
    --batch_size=128 \
    --policy.lora_rank=16 \
    --policy.lora_alpha=32 \
    --policy.lora_target=vision \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/baseline_log.txt
```

## 단일 GPU — LLM LoRA (~22 GB VRAM)

```bash
python scripts/train_groot_baseline.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_baseline_llm \
    --job_name=groot_baseline_llm \
    --steps=50000 \
    --batch_size=64 \
    --policy.lora_rank=16 \
    --policy.lora_target=llm \
    --gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/baseline_log.txt
```

## 단일 GPU — LoRA 없음 (Projector + Diffusion만, ~14 GB VRAM)

```bash
python scripts/train_groot_baseline.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_baseline_nolora \
    --job_name=groot_baseline_nolora \
    --steps=50000 \
    --batch_size=128 \
    --policy.lora_rank=0 \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/baseline_log.txt
```

## Multi-GPU (4 GPU, effective BS = 4 × 32 = 128)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=no \
    scripts/train_groot_baseline.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_baseline \
    --job_name=groot_baseline_v1 \
    --steps=50000 \
    --batch_size=32 \
    --policy.lora_rank=16 \
    --policy.lora_target=vision \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/baseline_log.txt
```

## tmux 세션 (서버 권장)

```bash
tmux new-session -s groot_baseline
conda activate lerobot_050_groot
cd /home/bluepot/cl_ws/lerobot_cl

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --mixed_precision=no \
    scripts/train_groot_baseline.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --output_dir=./outputs/groot_baseline \
    --job_name=groot_baseline_v1 \
    --steps=50000 \
    --batch_size=32 \
    --policy.lora_rank=16 \
    --policy.lora_target=vision \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    2>&1 | tee outputs/baseline_log.txt

# detach: Ctrl+B, D  /  재접속: tmux attach -t groot_baseline
```

---

## 전체 인수 목록

### 데이터셋 (`--dataset.*`)

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--dataset.repo_id` | `paragon7060/INSIGHTfixposV3` | HuggingFace 데이터셋 repo ID |
| `--dataset.root` | `None` | 로컬 경로 (없으면 HF Hub 스트리밍) |
| `--dataset.video_backend` | `pyav` | 비디오 디코더 (`pyav` 권장) |

### 정책/모델 (`--policy.*`)

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--policy.base_model_path` | `nvidia/GR00T-N1.5-3B` | 베이스 GR00T 모델 |
| `--policy.lora_rank` | `0` | LoRA rank (0이면 LoRA 비활성화) |
| `--policy.lora_alpha` | `16` | LoRA alpha |
| `--policy.lora_dropout` | `0.1` | LoRA dropout |
| `--policy.lora_target` | `llm` | LoRA 대상: `vision` / `llm` / `both` |
| `--policy.tune_llm` | `false` | LLM base weight 학습 여부 |
| `--policy.tune_visual` | `false` | Vision tower base weight 학습 여부 |
| `--policy.tune_projector` | `true` | Projector 학습 여부 |
| `--policy.tune_diffusion_model` | `true` | Diffusion head 학습 여부 |
| `--policy.optimizer_lr` | `1e-4` | 학습률 |
| `--policy.optimizer_betas` | `(0.95, 0.999)` | AdamW betas |
| `--policy.optimizer_weight_decay` | `1e-5` | Weight decay |
| `--policy.warmup_ratio` | `0.05` | Warmup 비율 (steps 대비) |
| `--policy.chunk_size` | `50` | Action chunk 크기 |
| `--policy.n_action_steps` | `50` | Action prediction horizon |

### 학습 루프

| 인수 | 기본값 | 설명 |
|---|---|---|
| `--steps` | `50000` | 총 학습 스텝 |
| `--batch_size` | `128` | GPU당 배치 크기 |
| `--num_workers` | `8` | DataLoader worker 수 |
| `--log_freq` | `100` | 로그 출력 간격 (스텝) |
| `--save_freq` | `5000` | 체크포인트 저장 간격 (스텝) |
| `--seed` | `42` | 랜덤 시드 |
| `--output_dir` | `outputs/groot_baseline` | 출력 디렉터리 |
| `--job_name` | `groot_baseline` | 실험 이름 |
| `--gradient_checkpointing` | `false` | Gradient checkpointing 활성화 |

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

## 체크포인트 구조

```
output_dir/
└── checkpoints/
    ├── 005000/
    │   ├── pretrained_model/
    │   │   ├── model.safetensors
    │   │   ├── config.json
    │   │   └── train_config.json
    │   └── training_state/
    │       ├── optimizer_state.safetensors
    │       ├── scheduler_state.json
    │       ├── rng_state.safetensors
    │       └── training_step.json
    ├── 010000/
    │   └── ...
    └── last_checkpoint -> 050000/   (symlink, 가장 최신 체크포인트)
```

Stage 2(CL 학습)의 `--policy.groot_pretrained_path`에는 `pretrained_model/` 디렉터리를 지정:

```bash
--policy.groot_pretrained_path=./outputs/groot_baseline/checkpoints/last_checkpoint/pretrained_model
```

---

## Multi-GPU batch_size 가이드

`--batch_size`는 GPU당 크기. effective BS = batch_size × GPU 수.

| GPU 수 | `--batch_size` | Effective BS |
|---|---|---|
| 1 | 128 | 128 |
| 4 | 32 | 128 |
| 8 | 16 | 128 |

`--mixed_precision=no` 필수: 스크립트 내부에서 `torch.autocast(bfloat16)`을 직접 사용하므로 accelerate autocast와 중복 방지.
