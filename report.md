# VLA Contrastive Learning Fine-tuning — 진행 보고

**날짜**: 2026년 3월 23일
**대상**: GR00T N1.5 기반 VLA Contrastive Learning 파인튜닝 구현

---

## 1. 연구 배경 및 문제 제기

### 기존 VLA의 한계

현재 GR00T N1.5(3B)와 같은 VLA(Vision-Language-Action) 모델은 크게 두 모듈로 구성됩니다.

- **VLM Backbone** (EagleBackbone): SigLIP 비전 인코더 + Qwen2/3 LLM → `backbone_features (B, T_seq, 1536)`
- **Action Expert** (FlowmatchingActionHead): DiT 기반 Flow Matching → `action prediction`

두 모듈은 Behavior Cloning(BC) loss(Flow Matching)만으로 학습되기 때문에, VLM backbone이 출력하는 시각-언어 특징과 Action Expert 내부의 action latent 표현 사이에 **명시적인 alignment 신호가 없습니다**.

> **핵심 가설**: VLM backbone이 "어떤 액션을 해야 하는가"에 민감한 특징을 출력하도록 유도하면, 같은 태스크 내 미세한 동작 차이를 더 잘 구분하고 fine-tuning 효율이 향상된다.

---

## 2. 제안 방법: Contrastive Learning 기반 VLA Fine-tuning

### 2.1 전체 구조

```
VLM Backbone
  backbone_features (B, T_seq, 1536)
    │ attention-mask weighted pooling
    │ Linear(1536→512) → LayerNorm → GELU → Linear(512→256)
    │ L2 normalize
    └──────────────────► vlm_z (B, 256)          ← Anchor
                                  │
                      Triplet Margin Loss
              d(vlm, pos) + margin < d(vlm, neg)
                                  │
clean action (B, T=16, D=32)     ← Positive (현재 관측의 GT action)
neg action   (B, T=16, D=32)     ← Hard Negative (같은 태스크, 다른 에피소드)
  │ transpose → (B, D, T)
  │ Conv1d(32→128,k=3) → BN → GELU
  │ Conv1d(128→128,k=3) → BN → GELU
  │ Global Average Pooling → (B, 128)
  │ Linear(128→256) → L2 normalize
  └──────────────────► action_z (B, 256)
```

**총 Loss**:
```
L_total = L_flow_matching + λ · L_triplet

L_triplet = mean( ReLU( d(vlm_z, pos_z) - d(vlm_z, neg_z) + margin ) )
d = cosine distance = 1 - cos_sim
```

### 2.2 Hard Negative 설계

단순 in-batch negative가 아닌 **Paired Hard Negative**를 사용합니다.

| 구분 | 설명 |
|------|------|
| Positive | 현재 관측에 대응하는 ground-truth action (T=16 horizon) |
| Hard Negative | **동일 태스크 언어, 다른 에피소드**의 action |
| 매칭 기준 | **Relative Timestep Ratio**: `ratio = frame_idx / episode_length` |

> 예: 전체의 30% 시점에서의 anchor에는, 같은 작업을 다른 방식으로 수행하는 에피소드의 동일 비율 시점 action이 Hard Negative로 매핑됩니다. 이 방식은 "시간적으로 유사한 상황의 다른 동작"을 negative로 사용하여 in-batch negative의 False Negative 문제를 원천적으로 제거합니다.

### 2.3 3단계 학습 Phase

| Phase | 학습 대상 | Frozen | 목적 |
|-------|----------|--------|------|
| **Phase 1** | Contrastive Heads만 | _groot_model 전체 | 헤드 warm-up (랜덤 초기화 헤드가 backbone 교란 방지) |
| **Phase 2a** | 전체 (tune_* + heads) | 없음 | **핵심**: VLM backbone에 contrastive gradient 전달 → action-aware 특징 학습 |
| **Phase 2b** | _groot_model (tune_* 준수) | Contrastive Heads | 선택적 후처리 fine-tuning |

Phase 2a의 `contrastive_backprop_backbone=True` 설정이 실질적 성능 향상의 핵심입니다.

---

## 3. 구현 현황

### 3.1 구현 원칙

- **원본 `groot/` 디렉터리 불변**: 기존 코드 일절 수정 없음
- **모든 신규 코드는 `groot_cl/`에**: 상속 및 composition으로 확장
- **LeRobot 프레임워크 규약 준수**: `PreTrainedConfig`, `PreTrainedPolicy`, `ProcessorStep`, `DataLoader` 등 기존 추상화 그대로 활용

### 3.2 구현 완료 목록

#### Policy 모듈 (`src/lerobot/policies/groot_cl/`)

| 파일 | 역할 | 상태 |
|------|------|------|
| `configuration_groot_cl.py` | `GrootCLConfig`: contrastive 하이퍼파라미터 + phase 설정 | ✅ 완료 |
| `modeling_groot_cl.py` | `GrootCLPolicy`: phase 관리 + forward (triplet + flow matching) | ✅ 완료 |
| `action_head/contrastive_heads.py` | `VLMContrastiveHead`, `ActionContrastiveHead`(1D CNN), `triplet_contrastive_loss`, `info_nce_fallback` | ✅ 완료 |
| `groot_n1.py` | `GR00TN15.forward()`에 `return_intermediate` 플래그 추가 | ✅ 완료 |
| `processor_groot_cl.py` | `NegativeActionNormalizeStep` (negative_action 정규화) | ✅ 완료 |

#### Dataset / Dataloader (`src/lerobot/datasets/`)

| 파일 | 역할 | 상태 |
|------|------|------|
| `contrastive_dataset.py` | `ContrastiveLeRobotDataset`: `__getitem__`에 `negative_action` 추가 | ✅ 완료 |

#### 사전 계산 스크립트 (`scripts/`)

| 파일 | 역할 | 상태 |
|------|------|------|
| `precompute_negative_pairs.py` | Relative Timestep Ratio 기반 Hard Negative 매핑 JSON 생성 | ✅ 완료 |
| `train_groot_cl.py` | Phase 1 → 2a 전체 학습 루프 (체크포인트 자동 저장) | ✅ 완료 |

#### 프레임워크 연동

| 위치 | 변경 내용 | 상태 |
|------|----------|------|
| `policies/factory.py` | `"groot_cl"` policy type 등록, processor factory 연결 | ✅ 완료 |
| `policies/__init__.py` | `GrootCLConfig` export 추가 | ✅ 완료 |

#### 테스트

| 파일 | 테스트 항목 | 결과 |
|------|------------|------|
| `tests/policies/groot_cl/test_contrastive_heads.py` | VLMContrastiveHead (output shape, L2 norm, attn mask), ActionContrastiveHead (1D CNN, GAP), triplet loss, InfoNCE fallback | **13/13 passed** |
| `tests/policies/groot_cl/test_groot_cl_forward.py` | forward (with/without negatives, train/eval mode), backbone detach, phase config validation | **8/8 passed** |

**총 21개 테스트 전부 통과**

### 3.3 코드 규모

| 모듈 | 파일 수 | 코드 라인 |
|------|---------|----------|
| `groot_cl/` (신규 정책) | 8 | ~740 |
| `groot_cl/action_head/` (신규 헤드) | 2 | ~92 |
| `datasets/contrastive_dataset.py` | 1 | 46 |
| `scripts/` (학습/사전계산) | 2 | ~360 |
| **합계** | **13** | **~1,240** |

---

## 4. 기술적 설계 결정 및 근거

### 4.1 ActionContrastiveHead — 1D CNN + Global Average Pooling

단순 mean pooling 대신 1D CNN을 선택한 이유:

- T=16 action horizon에서 **시간적 패턴(가속, 감속, 방향 전환)** 정보가 손실되지 않음
- Mean pool은 위치 정보 완전 소실 → "전반부 빠르게 + 후반부 느리게"와 "균일한 속도"를 구분 불가
- T=16 수준에서 Transformer는 과잉 설계. 1D CNN (~50K params)로 충분

### 4.2 Triplet Margin Loss (InfoNCE 대신)

- Hard Negative가 Dataloader에서 명시적으로 제공되므로 **False Negative가 없음**
- InfoNCE의 in-batch negative 방식은 배치 내 유사 action이 포함될 경우 loss 붕괴 위험
- Triplet: `L = mean( ReLU( d(anchor,pos) - d(anchor,neg) + margin ) )` — 직관적이며 안정적

### 4.3 Attention-mask Weighted Pooling (VLM 측)

- `backbone_features` (B, T_seq, 1536)는 패딩 토큰이 포함됨
- 단순 mean pooling은 패딩 토큰이 표현을 희석시킴
- `mask = attn_mask.unsqueeze(-1); pooled = (features * mask).sum(1) / mask.sum(1).clamp(min=1)` 로 실제 토큰만 반영

### 4.4 `return_intermediate` 플래그 (원본 코드 최소 수정)

```python
# groot_cl/groot_n1.py — GR00TN15.forward()에 추가된 유일한 변경
def forward(self, inputs, return_intermediate: bool = False):
    ...
    if return_intermediate:
        action_head_outputs["backbone_features"] = backbone_outputs["backbone_features"]
        action_head_outputs["backbone_attention_mask"] = backbone_outputs.get("backbone_attention_mask")
    return action_head_outputs
```

기본값 `False`로 기존 `get_action()` 등 모든 추론 경로에 영향 없음.

---

## 5. 발견된 엔지니어링 이슈 및 해결

### 이슈 1: `negative_action`이 전처리 파이프라인에서 소실

**원인**: LeRobot의 `batch_to_transition()` 함수가 `action`, `observation.*` 등 정해진 키만 `EnvTransition`으로 변환. `negative_action`은 이 과정에서 소실됨.

**해결**: 학습 스크립트의 `preprocess()` 함수에서 `negative_action`을 파이프라인 실행 **전에** 분리한 뒤, `NegativeActionNormalizeStep`을 직접 호출해 정규화 후 병합.

```python
raw_neg = raw_batch.pop("negative_action", None)    # 파이프라인 전 분리
processed = pre(raw_batch)                           # 나머지 정상 처리
result = neg_normalizer({"negative_action": raw_neg})  # 직접 정규화
processed["negative_action"] = result["negative_action"]
```

### 이슈 2: `groot_cl/` 내 프로세서 스텝 중복 등록 오류

**원인**: `groot_cl/processor_groot.py`가 `groot/processor_groot.py`의 전체 복사본이어서 `@ProcessorStepRegistry.register(name="groot_pack_inputs_v3")` 데코레이터가 두 번 실행됨.

**해결**: `groot_cl/processor_groot.py`를 `groot.processor_groot`로부터 re-export하는 얇은 래퍼로 교체. 단일 등록 보장.

---

## 6. 다음 단계

### 즉시 가능한 작업

1. **실 데이터셋 적용**: `scripts/precompute_negative_pairs.py` 실행 → `scripts/train_groot_cl.py` 실행
2. **하이퍼파라미터 탐색**: `contrastive_loss_weight` (0.01 ~ 0.1), `triplet_margin` (0.2 ~ 1.0)

### 추후 개선 사항

| 항목 | 설명 | 난이도 |
|------|------|--------|
| Multi-GPU DDP 지원 | `GatherLayer`로 전체 GPU의 negative를 공유해 유효 batch size 확대 | 중 |
| Negative 로드 비용 최적화 | action 컬럼 전용 mmap 캐시 구축으로 I/O 2× 비용 절감 | 중 |
| LoRA와의 통합 | Phase별 LoRA adapter와 contrastive heads 상호작용 정의 | 중 |
| 평가 지표 추가 | `cos_sim(vlm_z, pos_z)` vs `cos_sim(vlm_z, neg_z)` 분리 로깅 | 하 |
| Phase 1 자동 수렴 감지 | `contrastive_loss` plateau 감지 후 Phase 2a 자동 전환 | 중 |

---

## 7. 코드 저장소

- **lerobot_cl**: [https://github.com/paragon7060/lerobot_cl](https://github.com/paragon7060/lerobot_cl)
  - 최신 커밋: `feat(scripts): add contrastive learning training script and guide`
- **clvla** (상위 workspace): [https://github.com/paragon7060/clvla](https://github.com/paragon7060/clvla)
  - `lerobot_cl`을 git submodule로 관리

---

## 부록: 주요 설정값 (참고)

```python
GrootCLConfig(
    base_model_path    = "nvidia/GR00T-N1.5-3B",
    contrastive_latent_dim       = 256,
    contrastive_cnn_hidden_dim   = 128,
    contrastive_proj_hidden_dim  = 512,
    contrastive_triplet_margin   = 0.5,
    contrastive_loss_weight      = 0.05,   # Phase 2a 권장
    contrastive_backprop_backbone = True,  # Phase 2a 필수
    tune_llm             = False,
    tune_visual          = False,
    tune_projector       = True,
    tune_diffusion_model = True,
    use_bf16             = True,
)
```
