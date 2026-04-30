# GR00T-CL v2 Phase 2: Relational Knowledge Distillation (RKD) Loss

## 문제 정의

**Occlusion으로 인한 action ambiguity**:
- 같은 visual state(손목이 가려진 상황)에서 joint 7이 왼쪽 또는 오른쪽으로 돌아가는 경우가 혼재
- VLM은 동일한 이미지 → 동일한 embedding → action head가 방향을 구분 불가
- 목표: VLM embedding이 action space의 관계 구조를 따라가도록 finetuning

---

## 접근법: Relational Knowledge Distillation (CVPR 2019)

Point-to-point feature matching 대신 **pairwise relation structure** 를 전달.

| 역할 | 모듈 | 상태 |
|------|------|------|
| **Teacher** | Action Encoder (Phase 1에서 학습 완료) | Frozen |
| **Student** | VLM Backbone + VLMProjector | Trainable |

---

## 수식

Batch size $N$, 두 encoder의 feature:

$$z_a^i = \text{L2\_norm}\!\left(\text{mean}_t\!\left[\text{ActionEncoder}(\text{action}_i,\, t{=}999)\right]\right) \in \mathbb{R}^{D_a}$$

$$z_v^i = \text{L2\_norm}\!\left(\text{VLMProjector}\!\left(\text{pool}\!\left[\text{VLMBackbone}(\text{obs}_i)\right]\right)\right) \in \mathbb{R}^{D_{\text{proj}}}$$

**Similarity Matrices:**

$$S_a[i,j] = z_a^i \cdot z_a^j \in [-1, 1] \quad \text{(Teacher, }N \times N\text{)}$$

$$S_v[i,j] = z_v^i \cdot z_v^j \in [-1, 1] \quad \text{(Student, }N \times N\text{)}$$

**Distributions:**

$$P_a[i,:] = \text{Softmax}\!\left(S_a[i,:]\,/\,\tau_{\text{act}}\right) \quad \text{(Teacher, no grad)}$$

$$P_v[i,:] = \text{Softmax}\!\left(S_v[i,:]\,/\,\tau_{\text{vlm}}\right) \quad \text{(Student)}$$

**RKD Loss:**

$$\mathcal{L}_{\text{RKD}} = \text{KL}(P_a \,\|\, P_v) = \sum_{i,j} P_a[i,j]\,\log\frac{P_a[i,j]}{P_v[i,j]}$$

**Total Loss:**

$$\mathcal{L} = \lambda_{\text{FM}} \cdot \mathcal{L}_{\text{FM}} + \lambda_{\text{RKD}} \cdot \mathcal{L}_{\text{RKD}}$$

| Hyperparameter | Default | 설명 |
|---|---|---|
| $\tau_{\text{act}}$ | 0.1 | Teacher 분포 sharpness. ↓ → hard positive |
| $\tau_{\text{vlm}}$ | 0.07 | Student logit temperature |
| $\lambda_{\text{RKD}}$ | 0.1 | RKD loss weight |
| $\lambda_{\text{FM}}$ | 0.01 | FM loss weight (VLM quality 유지용 monitoring) |

---

## PyTorch 구현

```python
# ── Teacher: ActionEncoder at t=999 (near-clean) ─────────────────────────────
with torch.no_grad():
    t_clean = torch.full((B,), 999, dtype=torch.long, device=device)
    action_enc_out = action_encoder(action, t_clean, embodiment_id)  # (B, T=16, 1536)
    action_z = F.normalize(action_enc_out.mean(dim=1).float(), dim=-1)  # (B, 1536)

    S_a = action_z @ action_z.T                               # (B, B)
    P_a = F.softmax(S_a / tau_act, dim=-1)                    # (B, B), teacher

# ── Student: VLM → VLMProjector ──────────────────────────────────────────────
backbone_features = groot_model.forward(inputs, return_intermediate=True)["backbone_features"]
vlm_z = vlm_projector(backbone_features.float(), attention_mask)  # (B, 256)

S_v = vlm_z @ vlm_z.T                                         # (B, B)

# ── RKD Loss: KL( P_a || P_v ) ───────────────────────────────────────────────
# F.kl_div(log_Q, P) = Σ P*(log P - log Q) = KL(P||Q)
rkd_loss = F.kl_div(
    F.log_softmax(S_v / tau_vlm, dim=-1),   # log P_v (student)
    P_a,                                     # P_a     (teacher, detached)
    reduction="batchmean",
)
```

---

## VLMProjector 구조

```
backbone_features (B, T_seq, 1536)
    → masked mean pool  →  (B, 1536)
    → Linear(1536, 512) → GELU → Linear(512, 256)
    → L2 normalize
    →  z_v  (B, 256)
```

---

## 훈련 설정

| 항목 | 값 |
|------|-----|
| Phase 1 ckpt | `outputs/groot_cl_v2_phase1_w5/checkpoints/050000` |
| LR | `2e-5` |
| Warmup | 500 steps |
| Steps | 10,000 (시작값) |
| Batch size | 32 |
| Optimizer | AdamW |
| Scheduler | Cosine with warmup |

---

## 실행 명령

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/train_groot_cl_v2_phase2.py \
  --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
  --dataset.root=/home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
  --dataset.video_backend=pyav \
  --policy.type=groot_cl_v2 \
  --policy.groot_pretrained_path=./outputs/groot_cl_v2_phase1_w5/checkpoints/050000 \
  --output_dir=./outputs/groot_cl_v2_phase2 \
  --job_name=groot_cl_v2_phase2_rkd \
  --phase2_steps=10000 \
  --batch_size=32 \
  --wandb.enable=true \
  --wandb.project=groot_cl_v2 \
  --wandb.entity=RwHlabs \
  > outputs/train_groot_cl_v2_phase2.log 2>&1 &
```

---

## 구현 파일

| 파일 | 내용 |
|------|------|
| `src/lerobot/policies/groot_cl_v2/modeling_groot_cl_v2.py` | `VLMProjector` 클래스, `_forward_phase2()` RKD loss 구현, phase dispatch |
| `src/lerobot/policies/groot_cl_v2/configuration_groot_cl_v2.py` | `cl_v2_action_temp`, `cl_v2_vlm_temp`, `cl_v2_loss_weight`, `cl_v2_fm_loss_weight` 설정 |
| `scripts/train_groot_cl_v2_phase2.py` | Phase 2 학습 스크립트 |

---

## Verification

1. **Dry run**: `--phase2_steps=10` → `rkd_loss` 출력 확인
2. **Loss range**: 초기 RKD loss ≈ `log(B)` (B=32 → ≈3.47). 정상 범위 확인
3. **S_a 분포**: diagonal이 최대, 유사한 action trajectory 샘플끼리 off-diagonal이 높아야 함
4. **t-SNE (Phase 2 후)**: VLM embedding을 joint 7 방향(좌/우)으로 색상 구분 → 클러스터링 개선 확인

---

## 직관 요약

> Action space에서 비슷한 두 샘플은 VLM space에서도 비슷하게 embedding되어야 한다.
> Occlusion으로 가려진 손목이라도, guide camera 등의 subtle visual cue를 통해
> VLM이 action direction (joint 7 좌/우)을 구분하도록 학습된다.
