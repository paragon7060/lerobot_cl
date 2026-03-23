# GR00T Policy (groot) — Deep Research Report

## 1. 전체 역할 요약

`src/lerobot/policies/groot/`는 NVIDIA의 GR00T N1.5 (3B) foundation 모델을 LeRobot 프레임워크에 통합한 **VLA(Vision-Language-Action) 정책** 모듈이다.
핵심 설계 방침은 "포팅(porting)이지 외부 라이브러리 의존이 아님" — Isaac-GR00T 원본 레포에서 핵심 컴포넌트를 직접 포팅해 `self-contained`로 동작한다.

---

## 2. 디렉터리 구조 및 역할

```
groot/
├── __init__.py                          # 공개 API: GrootConfig, GrootPolicy, make_groot_pre_post_processors
├── configuration_groot.py               # GrootConfig (PreTrainedConfig 서브클래스)
├── modeling_groot.py                    # GrootPolicy (PreTrainedPolicy 래퍼)
├── groot_n1.py                          # GR00TN15, EagleBackbone (핵심 모델)
├── processor_groot.py                   # 전처리/후처리 파이프라인
├── utils.py                             # Eagle 캐시 준비 유틸
├── README.md                            # 논문/허브 링크
├── action_head/
│   ├── __init__.py
│   ├── action_encoder.py                # SinusoidalPositionalEncoding, swish
│   ├── cross_attention_dit.py           # DiT, SelfAttentionTransformer, AdaLayerNorm 등
│   └── flow_matching_action_head.py     # FlowmatchingActionHead (flow matching 기반 액션 예측)
└── eagle2_hg_model/
    ├── configuration_eagle2_5_vl.py     # Eagle2.5-VL 설정 (vision=SigLIP + text=Qwen2/3/Llama)
    ├── image_processing_eagle2_5_vl_fast.py  # 빠른 이미지 전처리 (동적 타일링)
    ├── modeling_eagle2_5_vl.py          # Eagle25VLForConditionalGeneration
    └── processing_eagle2_5_vl.py        # Eagle 멀티모달 프로세서
```

---

## 3. 아키텍처 — "Dual Brain"

```
입력: 이미지들 + 언어(task) + 상태(state)
         │
  ┌──────▼──────────────────────────────┐
  │    EagleBackbone (VLM)              │
  │  SigLIP vision_model                │
  │  + mlp1 (pixel shuffle + MLP)       │
  │  + Qwen2/3 language_model (일부 레이어)│
  │  + eagle_linear (→ 1536-dim)        │
  └──────────────────┬──────────────────┘
                     │ backbone_features (B, T_seq, 1536)
  ┌──────────────────▼──────────────────┐
  │    FlowmatchingActionHead           │
  │  state_encoder (CategorySpecificMLP)│
  │  action_encoder (MultiEmbodiment...)│
  │  DiT (Cross-Attention Diffusion)    │
  │  action_decoder (CategorySpecificMLP│
  └──────────────────┬──────────────────┘
                     │ action_pred (B, T_action, action_dim)
```

### 3.1 EagleBackbone (`groot_n1.py`)

| 속성 | 값 |
|------|-----|
| 기반 모델 | Eagle2.5-VL (SigLIP + Qwen2/3 LLM) |
| Vision → LLM 연결 | pixel_shuffle(0.5) + 2-layer MLP |
| select_layer | 기본 `-1` (마지막 레이어만 사용, 초과 레이어 pop) |
| 출력 | `BatchFeature(backbone_features, backbone_attention_mask)` |
| 프로젝션 | `eagle_linear: Linear(2048, 1536)` |
| DDP 핵 | `tune_visual=True`일 때 dummy_term += 0.0 * param.sum() 으로 미사용 파라미터 그래디언트 강제 참여 |

### 3.2 FlowmatchingActionHead (`action_head/flow_matching_action_head.py`)

- **Flow Matching** (Rectified Flow 계열): 노이즈 → 액션 방향 벡터를 MSE loss로 학습
- **시간 샘플링**: Beta(α=1.5, β=1.0) 분포, `t = (s - sample) / s` (s=0.999)
- **CategorySpecificLinear/MLP**: 각 embodiment별 별도 가중치 (`num_categories × D`)
- **MultiEmbodimentActionEncoder**: W1(d→w) → concat(tau_emb) → W2(2w→w) → W3(w→w)
- **DiT**: Cross-Attention Diffusion Transformer, `ada_norm` 타입 normalization
- **SelfAttentionTransformer (vlln)**: VL 특징을 LayerNorm + Self-Attention으로 후처리
- **future_tokens**: learnable embedding (32 tokens, 1536-dim)
- **추론**: Euler 적분 `x_{t+1} = x_t + dt * v_pred`로 num_inference_timesteps 스텝

### 3.3 DiT (`action_head/cross_attention_dit.py`)

- `ModelMixin, ConfigMixin` (diffusers) 상속 — `@register_to_config` 데코레이터 사용
- `BasicTransformerBlock`: Self/Cross Attention + FFN, `ada_norm` (AdaLayerNorm)
- `interleave_self_attention`: 짝수 레이어는 cross-attn, 홀수 레이어는 self-attn
- 출력: `proj_out_1` (scale/shift) → `norm_out` → `proj_out_2`
- **주의**: `encoder_attention_mask`는 시그니처에 있지만 실제 `attn1()` 호출 시 전달하지 않음 (주석 처리)

---

## 4. 설정 클래스 (`configuration_groot.py`)

### GrootConfig 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `n_obs_steps` | 1 | 관측 스텝 수 |
| `chunk_size` | 50 | 액션 청크 크기 |
| `n_action_steps` | 50 | 한 번 예측 후 실행할 액션 수 |
| `max_state_dim` | 64 | 상태 최대 차원 (제로 패딩) |
| `max_action_dim` | 32 | 액션 최대 차원 (제로 패딩) |
| `base_model_path` | `"nvidia/GR00T-N1.5-3B"` | HF 모델 ID 또는 로컬 경로 |
| `tokenizer_assets_repo` | `"lerobot/eagle2hg-processor-groot-n1p5"` | Eagle 토크나이저 자산 |
| `embodiment_tag` | `"new_embodiment"` | 로봇 embodiment 태그 |
| `tune_llm` | False | LLM 백본 학습 여부 |
| `tune_visual` | False | 비전 타워 학습 여부 |
| `tune_projector` | True | 프로젝터 학습 여부 |
| `tune_diffusion_model` | True | DiT 학습 여부 |
| `lora_rank` | 0 | LoRA rank (0=사용 안 함) |
| `use_bf16` | True | bfloat16 autocast 사용 |
| `action_horizon` 제한 | `min(chunk_size, 16)` | **pretrained 모델이 최대 16 고정** |

### 검증 규칙 (`__post_init__`, `validate_features`)

- `n_action_steps > chunk_size` → ValueError
- 비주얼 입력 없음 → ValueError
- `state_dim > max_state_dim` → ValueError
- `action_dim > max_action_dim` → ValueError
- visual feature 없으면 `OBS_STATE`, `ACTION`을 자동 추가

### `action_delta_indices`

- `list(range(min(chunk_size, 16)))` — 처음 최대 16 액션만 delta 인덱스

---

## 5. 정책 클래스 (`modeling_groot.py`)

### GrootPolicy

- `PreTrainedPolicy` (LeRobot) 상속
- `name = "groot"`, `config_class = GrootConfig`
- `_action_queue: deque(maxlen=n_action_steps)` — 액션 청킹

### `from_pretrained` 로직

두 가지 경로를 자동 분기:

1. **파인튜닝 체크포인트** (`model.safetensors` 존재): 부모 클래스 로딩 사용
2. **Base GR00T 모델** (`nvidia/GR00T-N1.5-3B` 등): `_create_groot_model()` → `GR00TN15.from_pretrained()`

- HF Hub에서 `model.safetensors` 존재 여부를 `hf_hub_download` 시도로 확인
- `HfHubHTTPError` → base 모델로 분기
- 범용 `Exception` → base 모델로 분기 (안전망)

### `forward` (학습)

- 허용 키 필터: `{state, state_mask, action, action_mask, embodiment_id, eagle_*}` — `next.*`, `info` 제외
- `torch.autocast(bfloat16)` 조건부 적용
- `outputs.get("loss")` → `(loss, {"loss": loss.item()})`

### `predict_action_chunk` (추론)

- 추론 시 `action`, `action_mask` **제외** (예측 대상이므로)
- `get_action()` 호출 → `outputs["action_pred"]`
- `actions[:, :, :original_action_dim]` 슬라이싱 (패딩 제거)

### `select_action`

- `_action_queue`가 비면 `predict_action_chunk` 호출 후 `.transpose(0,1)`로 배치→시간 순으로 재정렬
- 큐에서 `popleft()`

### Flash Attention 핸들링

- `_handle_flash_attention_compatibility()`: `undefined symbol` 오류 시 경고 출력 후 계속 진행
- 환경변수 `FLASH_ATTENTION_FORCE_BUILD=0`, `FLASH_ATTENTION_SKIP_CUDA_BUILD=0` 기본값 설정

---

## 6. 전처리/후처리 파이프라인 (`processor_groot.py`)

### `make_groot_pre_post_processors` 6단계 파이프라인

```
전처리 (입력 방향):
1. RenameObservationsProcessorStep   — 빈 rename map (필요 시 추가)
2. AddBatchDimensionProcessorStep    — 단일 샘플에 배치 차원 추가
3. GrootPackInputsStep               — 핵심: 비디오/상태/액션/언어 패킹 + min-max 정규화
4. GrootEagleEncodeStep              — Eagle VLM으로 멀티모달 인코딩 (eagle_content 생성)
5. GrootEagleCollateStep             — eagle_content → eagle_* 텐서 (배치 콜레이션)
6. DeviceProcessorStep               — GPU로 이동

후처리 (출력 방향):
1. GrootActionUnpackUnnormalizeStep  — 마지막 타임스텝 선택 + env dim 슬라이싱 + 역정규화
2. DeviceProcessorStep               — CPU로 이동
```

### GrootPackInputsStep 상세

| 처리 대상 | 변환 |
|-----------|------|
| 이미지 | `(B,C,H,W)` → uint8 numpy → `(B,1,V,C,H,W)` video |
| 언어 | `comp["task"]` → 없으면 `"Perform the task."` |
| 상태 | `(B,D)` → min-max norm → `(B,1,max_state_dim)` + state_mask |
| 액션 | `(B,D)` or `(B,T,D)` → norm → horizon 맞춤 pad/crop → `(B,T,max_action_dim)` + action_mask |
| embodiment_id | tag → int (매핑 테이블) → `LongTensor(B,)` |

**정규화 공식**: `y = 2 * (x - min) / (max - min) - 1` (분모 0이면 0으로 처리)

**embodiment_id 매핑**:
```python
"new_embodiment": 31, "oxe_droid": 17, "agibot_genie1": 26,
"gr1": 24, "so100": 2, "unitree_g1": 3
```

### GrootEagleEncodeStep

- Lazy 초기화 (`_proc`: 첫 호출 시 `_build_eagle_processor()`)
- 이미지 배열 → PIL Image 변환 → Eagle 채팅 템플릿 적용
- `str([lang])` 형식으로 언어 포맷팅 (원본 GR00T 호환)
- `apply_chat_template` + `process_vision_info` → eagle_content 목록

### GrootEagleCollateStep

- `collate()`: text_list + image_inputs → `eagle_processor(padding=True)` → `eagle_*` 텐서
- `min_dynamic_tiles=1, max_dynamic_tiles=1, use_thumbnail=False` 고정
- 처리 후 `video`, `eagle_content` 메모리 해제

### GrootActionUnpackUnnormalizeStep

- 역정규화: `x = (y + 1) / 2 * denom + min` (분모 0이면 min으로)
- `action.dim() == 3` → 마지막 타임스텝만 선택 (`[:, -1, :]`)

### state_dict / load_state_dict 패턴

- `GrootPackInputsStep`과 `GrootActionUnpackUnnormalizeStep` 모두 `state_dict()`/`load_state_dict()` 구현
- 플랫 키 형식: `"{feature_key}.{stat_name}"` (예: `"observation.state.min"`)
- safetensors 파일에 함께 저장/로드 가능

---

## 7. Eagle 캐시 준비 (`utils.py`)

`ensure_eagle_cache_ready(vendor_dir, cache_dir, assets_repo)`:

1. `vendor_dir`(`eagle2_hg_model/`) → `cache_dir`(HF_LEROBOT_HOME 하위) `copytree`
2. 누락된 토크나이저 자산 10개를 HF Hub에서 개별 다운로드
3. 캐시 준비 실패 시 경고만 출력하고 계속 진행 (`nosec: B110`)

**필수 자산 목록**: vocab.json, merges.txt, added_tokens.json, chat_template.json, special_tokens_map.json, config.json, generation_config.json, preprocessor_config.json, processor_config.json, tokenizer_config.json

---

## 8. Eagle2.5-VL 모델 (`eagle2_hg_model/`)

### Eagle25VLConfig

- vision: `SiglipVisionConfig` (고정)
- text: `LlamaConfig` / `Qwen2Config` / `Qwen3Config` 지원
- `_attn_implementation` 기본값: `"flash_attention_2"` (Qwen2에서 강제됨)

### Eagle25VLForConditionalGeneration

- `extract_feature()`: SigLIP → pixel_shuffle → mlp1 → visual tokens
- 이미지 토큰 삽입: `input_embeds[selected] = vit_embeds.reshape(-1, c)`
  - 크기 불일치 시 `n_token` 잘라서 대입 (예외 처리)
- LoRA 지원: `wrap_backbone_lora()`, `wrap_llm_lora()` (PEFT)
- **제약**: `forward`에서 `**kwargs` 사용 금지 (`check_forward_kwargs`로 어설트 검사)

---

## 9. factory 통합 (`policies/factory.py`)

- `get_policy_cls("groot")` → `GrootPolicy`
- `make_policy_config("groot", **kwargs)` → `GrootConfig`
- `make_pre_post_processors()`: `GrootConfig` 감지 시 **임시 패치** 적용
  - pretrained 로드 시 `groot_pack_inputs_v3`와 `groot_action_unpack_unnormalize_v1`에 dataset_stats + normalize_min_max=True 강제 주입
  - 코드 주석: `# TODO(Steven): Temporary patch, implement correctly the processors for Gr00t`

---

## 10. 예외 케이스 및 엣지 케이스

### 10.1 action_horizon 상한 강제

```python
action_horizon = min(config.chunk_size, 16)  # processor_groot.py:104
```
사전 학습된 GR00T 모델의 아키텍처가 action_horizon=16 하드코딩. `chunk_size=50`이어도 실제 예측은 16 스텝.
`config.n_action_steps`는 50이지만 `predict_action_chunk`가 반환하는 텐서의 T 차원은 16 → `select_action`에서 `deque(maxlen=50)` 초과 시 자동 드롭.

### 10.2 상태/액션 차원 패딩

- state: 실제 차원 < max_state_dim → zero pad; > max_state_dim → 잘라냄
- action: 실제 차원 < max_action_dim → zero pad; > max_action_dim → 잘라냄
- state_mask/action_mask: 실제 차원까지만 True, 패딩 부분은 False

### 10.3 Flash Attention 누락 처리

- `ImportError` / `undefined symbol` 모두 경고 출력 후 폴백 어텐션으로 진행
- Eagle 설정의 `_attn_implementation="flash_attention_2"`가 기본값이지만, flash_attn 없으면 HF Transformers가 자동 폴백

### 10.4 Eagle 캐시 미준비 상태에서 프로세서 빌드

`_build_eagle_processor()`: 캐시 디렉터리에 processor_config.json, preprocessor_config.json, image_processing_eagle2_5_vl_fast.py 중 하나라도 없으면 **`FileNotFoundError`** 발생 (경고가 아닌 하드 에러). 모델 초기화 전에 프로세서를 먼저 빌드하면 이 오류 발생.

### 10.5 Eagle 비디오 형상 확인

`GR00TN15.validate_inputs()`: `video.shape[3] == 3` (채널 축)을 체크. `GrootPackInputsStep`에서 `(B,1,V,H,W,C)` → `(B,1,V,C,H,W)` 재배열로 맞춤.

### 10.6 from_pretrained 분기 오탐

HF Hub 연결 불안정 시 `hf_hub_download` 실패 → `is_finetuned_checkpoint = False`로 폴백 → base 모델로 잘못 처리될 수 있음. `Exception` 폭 넓게 잡아서 방어.

### 10.7 tune_visual DDP 이슈

`EagleBackbone.forward()`: `tune_visual=True` + DDP 환경에서 vision_model 파라미터 일부가 forward에 사용 안 될 때 DDP 동기화 오류 방지를 위해 `dummy_term` 패턴 사용. 코드 주석에 `# YL (TODO HACK)`.

### 10.8 BasicTransformerBlock cross_attention_dim

`attn1` 생성 시 `cross_attention_dim` 전달하지만 `forward()` 내 `attn1()` 호출에서 `encoder_hidden_states`를 전달하지 않음. cross-attn 레이어가 self-attn처럼 동작하는 구조.

### 10.9 eagle_image_sizes 키 제거

`EagleBackbone.forward_eagle()`: `eagle_input.pop("image_sizes")` 후 eagle_model 호출. Eagle 프로세서가 image_sizes를 출력하지만 언어 모델에는 불필요해서 제거.

---

## 11. 학습 파라미터 제어

| 컴포넌트 | tune 파라미터 | 기본값 |
|----------|--------------|--------|
| LLM 백본 | `tune_llm` | False (frozen) |
| Vision 타워 | `tune_visual` | False (frozen) |
| MLP Projector | `tune_projector` | True |
| DiT (diffusion) | `tune_diffusion_model` | True |

학습 가능 파라미터 없음 경고: `set_trainable_parameters()` 내 `any(p.requires_grad)` 검사 후 경고 출력.

---

## 12. 의존성

### 핵심 의존

```
torch, transformers (AutoConfig, AutoModel, AutoProcessor, PreTrainedModel)
huggingface_hub (snapshot_download, hf_hub_download)
diffusers (ModelMixin, ConfigMixin, Attention, FeedForward, TimestepEmbedding 등)
peft (LoraConfig, get_peft_model)
einops (rearrange)
Pillow (Image)
numpy
dm-tree (tree.map_structure — 없으면 None으로 설정됨)
```

### 조건부/옵션

```
flash-attn     — 없어도 폴백, 버전 불일치 시 경고만
dm-tree        — try/except import, None이면 prepare_input에서 사용 불가
```

### `_transformers_available` 가드

`lerobot.utils.import_utils._transformers_available`이 False면 AutoConfig 등을 `None`/`object`로 대체. 테스트 환경 등 transformers 없는 환경 지원.

---

## 13. 컨벤션

- **타입 어노테이션**: `X | Y` 스타일 (Python ≥3.12)
- **반환 타입**: `BatchFeature` (transformers) 로 모델 출력 통일
- **ProcessorStep 등록**: `@ProcessorStepRegistry.register(name="...")` 데코레이터
- **설정 직렬화**: `get_config()` — stats 제외 / `state_dict()` — stats만
- **모델 로그**: `print(f"[GROOT] ...")` 패턴 (logger 아닌 print)
- **frozen 모듈 eval 강제**: `set_frozen_modules_to_eval_mode()` — HF Trainer가 `model.train()`을 매 스텝 호출하는 문제 대응
- **빈 배치 처리**: `transition.get(key, {}) or {}` — None 반환 시에도 빈 dict로 폴백

---

## 14. 테스트 커버리지

```
tests/policies/groot/
├── test_groot_lerobot.py       — LeRobot 단독 추론/학습 forward 테스트
└── test_groot_vs_original.py   — 원본 Isaac-GR00T vs LeRobot 포팅 동작 비교
```

- `MODEL_PATH = "aractingi/bimanual-handover-groot-10k"` — 파인튜닝 체크포인트
- 테스트는 실제 HF Hub 다운로드 필요 (offline 테스트 불가)

---

## 15. 알려진 임시 패치 / TODO

| 위치 | 내용 |
|------|------|
| `factory.py:246` | `# TODO(Steven): Temporary patch, implement correctly the processors for Gr00t` — pretrained 로드 시 stats 강제 주입 |
| `groot_n1.py:152` | `# YL (TODO HACK): to resolve DDP issue` — tune_visual DDP dummy term |
| `cross_attention_dit.py:133` | `# encoder_attention_mask=encoder_attention_mask,` 주석 처리 — attn1에 mask 전달 안 됨 |
| `modeling_groot.py:129` | LoRA 지원 코드가 config에 있지만 `_create_groot_model`에서 실제 적용하지 않음 (lora_rank 파라미터 존재하나 미사용) |
| `processor_groot.py:100-104` | action_horizon 16 하드 상한 — 주석으로 설명됨 |
