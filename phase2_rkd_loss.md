# Phase 2 RKD Loss — 설계 기록

## 문제 정의

**Occlusion Ambiguity**: 손목(wrist, joint 6)이 가려진 상황에서 VLM backbone이 동일한 이미지에 대해
동일한 embedding을 출력하여, action head가 wrist 회전 방향(좌/우)을 구분하지 못함.

```
같은 visual state (occlusion)
  → VLM embedding 동일
  → action head가 좌/우 방향 구분 불가
  → 잘못된 action 예측
```

---

## 해결 방법: Relational Knowledge Distillation (RKD)

**직접 embedding을 맞추는 것이 아닌, pairwise 관계 구조를 정렬**.

```
Teacher (frozen): ActionEncoder → action similarity matrix P_a (B×B)
Student (trainable): VLMBackbone + VLMProjector → VLM similarity matrix P_v (B×B)
Loss: KL( P_a ‖ P_v )
```

- Point-to-point matching 불필요 → Teacher/Student embedding dim이 달라도 됨
- Batch 내 샘플 간 관계 구조만 정렬
- "Action space에서 비슷한 샘플쌍은 VLM space에서도 비슷하게"

### 수식

```
# Teacher (no grad)
z_a = L2_norm( pool( ActionEncoder(action, t=999) ) )   ∈ ℝ^{D_a}
S_a = z_a @ z_a.T                                        ∈ ℝ^{B×B}
P_a = softmax(S_a / τ_act)                               [soft target]

# Student (trainable)
z_v = VLMProjector( masked_mean( VLMBackbone(obs) ) )   ∈ ℝ^{256}
S_v = z_v @ z_v.T                                        ∈ ℝ^{B×B}

# RKD Loss
L_RKD = KL( P_a ‖ softmax(S_v / τ_vlm) )
       = F.kl_div( log_softmax(S_v / τ_vlm), P_a, reduction="batchmean" )

# Total Loss
L = L_FM × λ_fm + L_RKD × λ_rkd
```

### 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `τ_act` (cl_v2_action_temp) | 0.1 | Teacher 분포 sharpness. ↓ = harder positive |
| `τ_vlm` (cl_v2_vlm_temp) | 0.07 | Student logit temperature |
| `λ_rkd` (cl_v2_loss_weight) | 0.1 | RKD loss weight |
| `λ_fm` (cl_v2_fm_loss_weight) | 0.01 | FM loss weight (VLM 품질 유지용) |

---

## 모델 구조

```
GrootCLv2Policy
├── _groot_model: GR00TN15 (pretrained)
│   ├── backbone: Eagle2.5-VL (SiglipViT + Qwen3-2.3B LLM)
│   │   └── output hidden dim: 2048
│   └── action_head: FlowMatchingActionHead
│       ├── action_encoder: MultiEmbodimentActionEncoder
│       │   └── per-step MLP (W1→W2→W3), output: 1536-dim
│       └── model: DiT
└── vlm_projector: VLMProjector (Phase 2 추가)
    └── Linear(2048→512) → GELU → Linear(512→256) → L2 norm
```

**Phase별 freeze/unfreeze:**

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| VLM backbone (Eagle LLM + ViT) | ❄️ Frozen | 🔥 Trainable |
| ActionEncoder | 🔥 Trainable | ❄️ Frozen (Teacher) |
| DiT | 🔥 Trainable | ❄️ Frozen |
| VLMProjector | ❄️ Frozen | 🔥 Trainable |

---

## 구현 과정에서 발견된 이슈 및 수정

### 1. BACKBONE_FEAT_DIM 오류 (1536 → 2048)

- **원인**: `groot_cl/contrastive_heads.py`의 `vlm_input_dim=1536`을 참고했으나, 이는 다른 head의 입력 dim
- **실제**: Eagle2.5-VL (Qwen3-2.3B) LLM hidden size = **2048**
- **발견**: 런타임 shape mismatch 에러 `(mat1 16x2048, mat2 1536x512)`
- **수정**: `BACKBONE_FEAT_DIM = 1536 → 2048`

### 2. Robocasa Multi-Dataset 지원

- **구조**: `/slicing_robocasa_human_v3/{group}/task_XXXX/` 형태의 분산 저장
- **문제**: 단일 LeRobotDataset으로 로드 불가
- **해결**: `MultiLeRobotDataset` + `dataset_index → task text` 매핑
  - 각 task subdir의 `tasks.parquet`에서 task text 읽어 `dataset_to_task[dataset_idx]` 구성
  - Batch의 `dataset_index`로 task text lookup

---

## Robocasa Phase 2 학습 현황

### 설정

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_groot_cl_v2_phase2.py \
    --dataset.root=/home/seonho/slicing_robocasa_human_v3 \
    --policy.groot_pretrained_path=.../groot_robocasa/outputs/target/checkpoints/060000/pretrained_model \
    --multi_dataset=true \
    --dataset_dirs='["robocasa_target_human_atomic","robocasa_target_human_composite"]' \
    --phase2_steps=100000 \
    --batch_size=24 \
    --lr=2e-5 \
    --warmup_steps=1000
```

- **데이터셋**: robocasa target (atomic 18 + composite 32 = 50 tasks, 14.8M frames)
- **GPU**: GPU 0 (A100 80GB), 73.6GB 사용
- **Teacher checkpoint**: `/groot_robocasa/outputs/target/checkpoints/060000/pretrained_model`
  - standard `groot` policy type으로 학습된 checkpoint
  - `_groot_model.*` 키 898개 → `groot_cl_v2`로 정상 로드됨
- **Trainable params**: 1,547,383,488 / 2,414,703,296 (64.08%)
- **체크포인트 저장**: step 10k, 30k, 50k, 100k (model only, optimizer 제외)
  - optimizer 제외 이유: 4개 × ~19GB = 76GB > 가용 디스크 86GB 위험

### 학습 추이 (cl_v2_action_repr="mean_pool")

| Step | rkd_loss | fm_loss | lr |
|------|----------|---------|-----|
| 50 | 0.0801 | 0.0559 | 2.0e-6 |
| 300 | 0.0122 | 0.0265 | 1.2e-5 |
| 1000 | ~0.02 | ~0.05 | 2.0e-5 |
| 10000 | - | - | (저장 완료) |
| 30000 | - | - | (저장 완료) |

---

## Teacher Action Representation 비교

현재 `mean_pool` 방식의 한계와 대안 구현 상태.

### 문제: Mean Pool의 Temporal Information Loss

Action Encoder는 **per-step MLP** — 각 timestep을 독립적으로 처리.
```
action_enc_out[:, t, :] = f(action[:, t, :])   # 순수 per-step, cross-timestep 없음
```

Mean pool하면:
- Temporal ordering 완전 소실
- 서로 다른 방향의 회전이 mean에서 상쇄될 수 있음
- Early/late timestep 동등 가중 → 회전이 집중된 구간의 신호 희석

### 구현된 대안: `raw_flatten`

> **파일**: `src/lerobot/policies/groot_cl_v2/modeling_groot_cl_v2.py` — `_compute_action_z()`
> **설정**: `--policy.cl_v2_action_repr=raw_flatten`

```python
# action_traj: (B, T=16, action_dim)
flat = action_traj.flatten(1)          # (B, T*action_dim) — temporal 구조 완전 보존
action_z = F.normalize(flat, dim=-1)
```

**장점**:
- Temporal 구조 완전 보존 (순서 정보 직접 인코딩)
- ActionEncoder 없이 raw trajectory 사용 → wrist joint의 timestep별 회전 방향이 벡터에 직접 반영
- 추가 파라미터 없음

**단점**:
- 32개 joint 전체가 동등 포함 → wrist 외 joint가 노이즈로 작용 가능
- Action dim 전체 scale 차이 고려 필요

**비교 실험 예정**:
```bash
# raw_flatten으로 Phase 2 재학습
--policy.cl_v2_action_repr=raw_flatten
```

### 검토된 기타 대안

| 방법 | 핵심 아이디어 | 비고 |
|------|------------|------|
| Last timestep | `action_enc_out[:, -1, :]` | 최종 상태가 목표 상태에 가장 가까움 |
| Max pool | `action_enc_out.max(dim=1).values` | Peak 신호 보존 |
| Velocity flatten | `(action[:, 1:] - action[:, :-1]).flatten(1)` | 회전 방향을 가장 직접적으로 인코딩 |
| Joint 7 weighted pool | timestep을 joint 7 크기로 가중 | 도메인 특화, wrist 집중 |
| 1D CNN | encoder output에 conv over T | teacher frozen이라 별도 학습 필요 |
| Statistics (pos+vel+acc) | mean/std 통계량 결합 | 경량, 해석 가능 |

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `src/lerobot/policies/groot_cl_v2/modeling_groot_cl_v2.py` | Phase 1/2 forward, VLMProjector, `_compute_action_z()` |
| `src/lerobot/policies/groot_cl_v2/configuration_groot_cl_v2.py` | `cl_v2_action_repr` 등 하이퍼파라미터 |
| `scripts/train_groot_cl_v2_phase1.py` | Phase 1 학습 스크립트 |
| `scripts/train_groot_cl_v2_phase2.py` | Phase 2 학습 스크립트 (multi-dataset 지원) |
| `scripts/dataset_configs/robocasa_target.json` | Robocasa target split 카메라 설정 |
| `scripts/dataset_configs/INSIGHTfixposV3.json` | INSIGHTfixposV3 카메라 + task 설정 |
| `RKD_loss_plan.md` | 초기 설계 문서 (수식 포함) |
