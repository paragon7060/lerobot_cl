# Contrastive Learning for GR00T VLA — 구현 계획 (v2)

## 핵심 아이디어 요약

VLM backbone이 출력하는 `backbone_features`(시각+언어 표현)와
FlowmatchingActionHead 내부의 clean action 표현 사이의 alignment 부족 문제를
**Triplet Margin Loss 기반 Contrastive Fine-tuning**으로 해소한다.

- **Positive**: 현재 관측에 대응하는 ground-truth action
- **Hard Negative**: 같은 task이지만 다른 variation의 action (동일 진행 비율 기준으로 사전 매칭)
- **목적**: VLM이 "액션 디테일이 반영된 시각 피처"를 출력하도록, contrastive gradient가 반드시 VLM backbone까지 흘러야 실질적 효과 발생

---

## 구현 원칙

- **원본 `groot/` 디렉터리는 일절 수정하지 않는다.**
- 모든 새 코드는 `src/lerobot/policies/groot_cl/` 에 작성한다.
  - 기존 클래스는 상속/composition으로 확장.
- Dataloader 변경은 `src/lerobot/datasets/` 내 기존 클래스를 서브클래싱하여 처리.

---

## TODO 체크리스트

### Dataset / Dataloader
- [x] `src/lerobot/datasets/contrastive_dataset.py` — `ContrastiveLeRobotDataset` 신규 작성
- [x] `scripts/precompute_negative_pairs.py` — Hard Negative 매핑 사전 계산 스크립트

### groot_cl 정책 모듈
- [x] `src/lerobot/policies/groot_cl/__init__.py`
- [x] `src/lerobot/policies/groot_cl/action_head/contrastive_heads.py` — VLMContrastiveHead, ActionContrastiveHead(1D CNN), triplet_contrastive_loss
- [x] `src/lerobot/policies/groot_cl/configuration_groot_cl.py` — GrootCLConfig
- [x] `src/lerobot/policies/groot_cl/groot_n1.py` — GR00TN15 (return_intermediate 추가, in-place 수정)
- [x] `src/lerobot/policies/groot_cl/modeling_groot_cl.py` — GrootCLPolicy (phase 관리 + forward)
- [x] `src/lerobot/policies/groot_cl/processor_groot_cl.py` — negative_action 처리 추가
- [x] `src/lerobot/policies/factory.py` — "groot_cl" 등록
- [x] `src/lerobot/policies/__init__.py` — GrootCLConfig 익스포트

### 테스트
- [x] `tests/policies/groot_cl/test_contrastive_heads.py` — 13/13 passed
- [x] `tests/policies/groot_cl/test_groot_cl_forward.py` — 8/8 passed

---

## 1. 아키텍처 결정

### 1.1 어디에 Encoder를 붙이나

**VLM 측** — `backbone_features` (B, T_seq, 1536):
- `process_backbone_output()` (vlln + vl_self_attention) 이후, 가장 정제된 표현
- Pooling: attention_mask 기반 **Weighted Mean Pooling** — 패딩 토큰 제외

**Action 측** — clean action (B, T=16, D=32):
- `action_input.action` — flow matching에 노이즈 주입 전 ground-truth
- Pooling: 단순 mean pool로는 T=16 궤적의 속도/변화율(dynamics) 정보가 소실됨
- → **1D CNN + Global Average Pooling** 사용, temporal feature 보존

```
VLM backbone_features (B, T_seq, 1536)
  │  attn_mask weighted mean → (B, 1536)
  │  VLMContrastiveHead(Linear → LN → GELU → Linear) → L2 norm
  └──────────────────────────────► vlm_z (B, latent_dim)
                                           │
                               Triplet Margin Loss
                       d(vlm, pos) + margin < d(vlm, neg)
                                           │
clean action (B, T=16, D=32)    ←──────── Positive
neg action   (B, T=16, D=32)    ←──────── Hard Negative (Dataloader 제공)
  │  (B, D, T) reshape for Conv1d
  │  Conv1d → BN → GELU → Conv1d → BN → GELU
  │  Global Average Pooling → (B, hidden)
  │  ActionContrastiveHead(Linear) → L2 norm
  └──────────────────────────────► action_z (B, latent_dim)
```

### 1.2 Positive / Negative 정의

**In-batch negatives의 문제점**:
- 같은 task 내 미세한 variation 차이를 구분하는 신호 제공 불가
- 배치 내 유사한 action이 우연히 들어오면 False Negative → loss 붕괴

**Paired Hard Negative** (Dataloader 사전 계산):
- Negative 조건: **Task 언어 동일, Episode 다름** (같은 작업의 다른 수행 방식)
- Matching 기준: **Relative Timestep Ratio** 일치
  - `ratio = frame_idx / episode_length`
  - Anchor (ep_i, frame_j)와 동일 ratio의 프레임을 neg_episode에서 추출
  - 이유: "부 붓기 10%" 시점의 앵커에는 "같은 시점의 다른 붓는 방식"이 Hard Negative

**Loss**: Triplet Margin Loss (InfoNCE 대신)
```
L = mean( ReLU( d(vlm_z, pos_z) - d(vlm_z, neg_z) + margin ) )
where d = cosine distance = 1 - cos_sim
```
- 직관적: "positive보다 negative를 margin만큼 더 멀리"
- Triplet이 Dataloader에서 명시적으로 제공되므로 False Negative 없음

### 1.3 Phase 구성 (수정된 목적)

**Phase1의 올바른 역할**:
`_groot_model`이 frozen이면 VLM features가 변하지 않으므로, contrastive gradient가 policy 성능에 직접 기여하지 못한다.
Phase1의 목적은 **contrastive heads의 초기화(warm-up)** 뿐이다 —
랜덤 초기화 상태의 헤드가 joint fine-tuning 초반에 VLM을 교란하지 않도록.

**실질적 개선은 Phase2a에서 발생**: `contrastive_backprop_backbone=True`로
contrastive gradient가 VLM backbone까지 흘러야 VLM이 "action-aware" feature를 학습한다.

| Phase | 학습 대상 | Frozen | contrastive_backprop_backbone | 목적 |
|-------|----------|--------|-------------------------------|------|
| phase1 | contrastive heads | _groot_model 전체 | — (frozen이므로 무관) | 헤드 warm-up |
| phase2a | 전체 (tune_* + contrastive heads) | 없음 | **True (필수)** | VLM alignment |
| phase2b | _groot_model (tune_* 준수) | contrastive heads | False | 선택적 후처리 |

---

## 2. Dataset / Dataloader

### 2.1 신규: `scripts/precompute_negative_pairs.py`

Hard Negative 매핑 파일(`negative_pairs.json`)을 사전 계산한다.

```python
# 출력 형식:
# {
#   "<dataset_repo_id>": {
#     "<episode_idx>_<frame_idx>": {
#       "neg_episode_idx": int,
#       "neg_frame_idx": int,
#     },
#     ...
#   }
# }

def build_negative_pairs(dataset: LeRobotDataset) -> dict:
    """
    알고리즘:
    1. 에피소드별로 task 언어 인덱스 추출 (hf_dataset["task_index"])
    2. 같은 task_index를 가진 에피소드 그룹화
    3. 각 (ep_i, frame_j)에 대해:
       a. 같은 task_index에서 ep_i 제외한 후보 에피소드 중 랜덤 선택 → neg_ep
       b. ratio = frame_j / len(ep_i)
       c. neg_frame = int(ratio * len(neg_ep))  # Relative Timestep Ratio 매칭
    4. 결과 dict 반환 후 JSON 저장
    """
```

**주의**: task_index가 없는 데이터셋(task 언어가 모두 동일)은 매핑 의미가 없음 → 경고 출력 후 in-batch negatives로 폴백.

### 2.2 신규: `src/lerobot/datasets/contrastive_dataset.py`

```python
class ContrastiveLeRobotDataset(LeRobotDataset):
    """LeRobotDataset을 상속해 negative_action을 __getitem__에 추가."""

    def __init__(self, *args, negative_pairs_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._negative_pairs: dict | None = None
        if negative_pairs_path:
            with open(negative_pairs_path) as f:
                self._negative_pairs = json.load(f)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)

        if self._negative_pairs is None:
            # 매핑 없으면 negative_action 필드를 None으로 채워 GrootCLPolicy가 폴백 처리
            item["negative_action"] = None
            return item

        ep_idx = int(item["episode_index"])
        fr_idx = int(item["frame_index"])
        key = f"{ep_idx}_{fr_idx}"

        pair = self._negative_pairs.get(key)
        if pair is None:
            item["negative_action"] = None
            return item

        neg_ep = pair["neg_episode_idx"]
        neg_fr  = pair["neg_frame_idx"]

        # negative 샘플의 action만 로드 (이미지/상태 불필요)
        neg_global_idx = self._ep_frame_to_global_idx(neg_ep, neg_fr)
        neg_item = super().__getitem__(neg_global_idx)
        item["negative_action"] = neg_item["action"]   # (chunk_size, action_dim)
        return item

    def _ep_frame_to_global_idx(self, ep_idx: int, frame_idx: int) -> int:
        """(episode_idx, episode-relative frame_idx) → dataset global idx."""
        ep_start = self.episode_data_index["from"][ep_idx].item()
        return int(ep_start) + frame_idx
```

**트레이드오프**:
- negative 샘플 로드 시 `super().__getitem__`을 한 번 더 호출 → I/O 비용 2배.
- action만 필요하므로 video 디코딩은 피하도록 향후 최적화 가능.

---

## 3. groot_cl 정책 모듈

### 3.1 신규: `src/lerobot/policies/groot_cl/action_head/contrastive_heads.py`

#### ContrastiveHeadConfig

```python
@dataclass
class ContrastiveHeadConfig:
    latent_dim: int = 256
    vlm_input_dim: int = 1536       # backbone_embedding_dim (N1.5 고정)
    action_input_dim: int = 32      # max_action_dim
    cnn_hidden_dim: int = 128       # 1D CNN channel dim
    proj_hidden_dim: int = 512      # MLP hidden dim
    triplet_margin: float = 0.5     # Triplet Margin Loss margin
```

#### VLMContrastiveHead

```python
class VLMContrastiveHead(nn.Module):
    """
    backbone_features (B, T_seq, 1536)
    → attention-mask weighted mean → (B, 1536)
    → Linear(1536, 512) → LayerNorm → GELU → Linear(512, latent_dim)
    → L2 normalize
    """
    def __init__(self, config: ContrastiveHeadConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.vlm_input_dim, config.proj_hidden_dim),
            nn.LayerNorm(config.proj_hidden_dim),
            nn.GELU(),
            nn.Linear(config.proj_hidden_dim, config.latent_dim),
        )

    def forward(self, backbone_features: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).float()           # (B, T_seq, 1)
            pooled = (backbone_features * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = backbone_features.mean(1)
        return F.normalize(self.proj(pooled), dim=-1)        # (B, latent_dim)
```

#### ActionContrastiveHead (1D CNN)

```python
class ActionContrastiveHead(nn.Module):
    """
    clean action (B, T=16, D=32)
    → transpose → (B, D=32, T=16)   [Conv1d input format]
    → Conv1d(32, 128, k=3) → BN → GELU
    → Conv1d(128, 128, k=3) → BN → GELU
    → Global Average Pooling → (B, 128)
    → Linear(128, latent_dim) → L2 normalize

    이유: T=16 궤적의 시간적 패턴(가속/감속, 방향 전환)을 보존해야
         "10% 붓기 vs 50% 붓기"처럼 dynamics가 다른 action을 구분 가능.
         Mean pool은 이 정보를 완전히 소실.
    """
    def __init__(self, config: ContrastiveHeadConfig):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(config.action_input_dim, config.cnn_hidden_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(config.cnn_hidden_dim),
            nn.GELU(),
            nn.Conv1d(config.cnn_hidden_dim, config.cnn_hidden_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(config.cnn_hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(config.cnn_hidden_dim, config.latent_dim)

    def forward(self, actions: Tensor) -> Tensor:
        # actions: (B, T, D) → Conv1d expects (B, C=D, L=T)
        x = actions.transpose(1, 2)   # (B, 32, 16)
        x = self.cnn(x)               # (B, 128, 16)
        x = x.mean(dim=-1)            # Global Average Pooling: (B, 128)
        return F.normalize(self.proj(x), dim=-1)  # (B, latent_dim)
```

#### triplet_contrastive_loss

```python
def triplet_contrastive_loss(
    vlm_z: Tensor,          # (B, latent_dim) — anchor
    pos_action_z: Tensor,   # (B, latent_dim) — positive
    neg_action_z: Tensor,   # (B, latent_dim) — hard negative
    margin: float = 0.5,
) -> Tensor:
    """Cosine distance 기반 Triplet Margin Loss.

    Loss = mean( ReLU( d(anchor, pos) - d(anchor, neg) + margin ) )
    d = 1 - cosine_similarity  (L2 normalized 입력이므로 dot product로 계산)
    """
    if vlm_z.shape[0] < 1:
        return vlm_z.new_tensor(0.0)

    d_pos = 1.0 - (vlm_z * pos_action_z).sum(dim=-1)  # (B,)
    d_neg = 1.0 - (vlm_z * neg_action_z).sum(dim=-1)  # (B,)
    loss = F.relu(d_pos - d_neg + margin).mean()
    return loss
```

---

### 3.2 신규: `src/lerobot/policies/groot_cl/configuration_groot_cl.py`

```python
@PreTrainedConfig.register_subclass("groot_cl")
@dataclass
class GrootCLConfig(GrootConfig):
    """GrootConfig를 상속하고 Contrastive Learning 파라미터를 추가."""

    # Contrastive 활성화
    use_contrastive: bool = True

    # 공유 latent space 차원
    contrastive_latent_dim: int = 256

    # 1D CNN hidden dim
    contrastive_cnn_hidden_dim: int = 128

    # MLP hidden dim
    contrastive_proj_hidden_dim: int = 512

    # Triplet Margin Loss margin
    contrastive_triplet_margin: float = 0.5

    # Flow matching loss 대비 contrastive loss 가중치
    contrastive_loss_weight: float = 0.1

    # 학습 단계: "phase1" | "phase2a" | "phase2b"
    contrastive_phase: str = "phase1"

    # Phase2a에서 backbone까지 contrastive gradient 전달 여부
    # True (필수): VLM이 action-aware feature 학습 → 실질적 성능 향상
    # False: projection heads만 학습 → VLM 미변경, 효과 미미
    contrastive_backprop_backbone: bool = True

    # negative_action 없을 때 in-batch negatives로 폴백 (디버깅/데이터 없을 때)
    contrastive_fallback_to_in_batch: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.contrastive_phase not in {"phase1", "phase2a", "phase2b"}:
            raise ValueError(
                f"contrastive_phase must be 'phase1'|'phase2a'|'phase2b', "
                f"got {self.contrastive_phase!r}"
            )
```

---

### 3.3 신규: `src/lerobot/policies/groot_cl/groot_n1_cl.py`

`GR00TN15`를 상속해 `return_intermediate` 기능만 추가. 원본 코드 미수정.

```python
class GR00TN15CL(GR00TN15):
    """GR00TN15에 return_intermediate 옵션 추가. 원본은 미수정."""

    def forward(
        self,
        inputs: dict,
        return_intermediate: bool = False,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        # action_head.forward() 내부 process_backbone_output() 호출로
        # backbone_outputs["backbone_features"]는 vlln+vl_self_attention 적용 후 상태.
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

        if return_intermediate:
            action_head_outputs["backbone_features"] = backbone_outputs["backbone_features"]
            action_head_outputs["backbone_attention_mask"] = backbone_outputs.get(
                "backbone_attention_mask"
            )
        return action_head_outputs
```

**`from_pretrained` 연결**: `GR00TN15CL`은 `GR00TN15`의 `config_class = GR00TN15Config`를 그대로 상속하므로, `GR00TN15.from_pretrained()`로 로드한 weight를 그대로 활용 가능.
단, `GR00TN15CL.from_pretrained()`를 직접 호출하면 HF `from_pretrained`가 `GR00TN15Config`를 config_class로 등록된 것을 찾으므로, `GrootCLPolicy._create_groot_model()`에서 수동으로 weight 로드 필요.

---

### 3.4 신규: `src/lerobot/policies/groot_cl/modeling_groot_cl.py`

#### 핵심 구조

```python
class GrootCLPolicy(GrootPolicy):
    name = "groot_cl"
    config_class = GrootCLConfig

    def __init__(self, config: GrootCLConfig, **kwargs):
        # GrootPolicy.__init__ 호출 → _groot_model 생성 (GR00TN15CL로 교체됨)
        super().__init__(config, **kwargs)

        # Contrastive heads
        contrastive_cfg = ContrastiveHeadConfig(
            latent_dim=config.contrastive_latent_dim,
            vlm_input_dim=1536,
            action_input_dim=config.max_action_dim,
            cnn_hidden_dim=config.contrastive_cnn_hidden_dim,
            proj_hidden_dim=config.contrastive_proj_hidden_dim,
            triplet_margin=config.contrastive_triplet_margin,
        )
        self.vlm_contrastive_head = VLMContrastiveHead(contrastive_cfg)
        self.action_contrastive_head = ActionContrastiveHead(contrastive_cfg)

        self.set_contrastive_phase(config.contrastive_phase)

    def _create_groot_model(self):
        """GR00TN15 대신 GR00TN15CL 사용."""
        self._handle_flash_attention_compatibility()
        # GR00TN15와 동일 로딩 흐름, 클래스만 교체
        model = GR00TN15CL.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
        )
        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype
        return model
```

#### `set_contrastive_phase()`

```python
def set_contrastive_phase(self, phase: str) -> None:
    if phase == "phase1":
        self._groot_model.requires_grad_(False)
        self.vlm_contrastive_head.requires_grad_(True)
        self.action_contrastive_head.requires_grad_(True)

    elif phase == "phase2a":
        self._restore_groot_trainability()
        self.vlm_contrastive_head.requires_grad_(True)
        self.action_contrastive_head.requires_grad_(True)

    elif phase == "phase2b":
        self._restore_groot_trainability()
        self.vlm_contrastive_head.requires_grad_(False)
        self.action_contrastive_head.requires_grad_(False)

    else:
        raise ValueError(f"Unknown contrastive_phase: {phase!r}")

    self.config.contrastive_phase = phase


def _restore_groot_trainability(self) -> None:
    cfg = self.config
    self._groot_model.backbone.set_trainable_parameters(
        tune_visual=cfg.tune_visual,
        tune_llm=cfg.tune_llm,
    )
    self._groot_model.action_head.set_trainable_parameters(
        tune_projector=cfg.tune_projector,
        tune_diffusion_model=cfg.tune_diffusion_model,
    )
```

#### `forward()`

```python
def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
    allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
    groot_inputs = {
        k: v
        for k, v in batch.items()
        if (k in allowed_base or k.startswith("eagle_"))
        and not (k.startswith("next.") or k == "info")
    }

    device = next(self.parameters()).device
    use_contrastive = self.training  # 추론 시 contrastive 비활성

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
        outputs = self._groot_model.forward(
            groot_inputs,
            return_intermediate=use_contrastive,
        )

    loss_fm = outputs.get("loss")
    loss_dict = {"flow_matching_loss": loss_fm.item()}

    if use_contrastive:
        backbone_features = outputs.get("backbone_features")  # (B, T_seq, 1536)
        attn_mask = outputs.get("backbone_attention_mask")
        actions = groot_inputs.get("action")                  # (B, T=16, D=32)
        negative_action = batch.get("negative_action")        # (B, T=16, D=32) or None

        # Phase2a: contrastive_backprop_backbone=True → backbone에 grad 허용
        # Phase1 / Phase2b: detach
        if not self.config.contrastive_backprop_backbone:
            backbone_features = backbone_features.detach()

        vlm_z = self.vlm_contrastive_head(backbone_features, attn_mask)
        pos_action_z = self.action_contrastive_head(actions)

        if negative_action is not None:
            neg_action_z = self.action_contrastive_head(negative_action)
            loss_cont = triplet_contrastive_loss(
                vlm_z, pos_action_z, neg_action_z,
                margin=self.config.contrastive_triplet_margin,
            )
        elif self.config.contrastive_fallback_to_in_batch:
            # negative 없을 때 in-batch negatives로 InfoNCE 폴백
            loss_cont = _info_nce_fallback(vlm_z, pos_action_z)
        else:
            loss_cont = vlm_z.new_tensor(0.0)

        loss_total = loss_fm + self.config.contrastive_loss_weight * loss_cont
        loss_dict["contrastive_loss"] = loss_cont.item()
        loss_dict["loss"] = loss_total.item()
        return loss_total, loss_dict

    loss_dict["loss"] = loss_fm.item()
    return loss_fm, loss_dict
```

---

### 3.5 `src/lerobot/policies/groot_cl/processor_groot_cl.py`

`make_groot_pre_post_processors`를 감싸고 `negative_action` 전처리 스텝 추가.

```python
def make_groot_cl_pre_post_processors(config: GrootCLConfig, dataset_stats=None):
    pre, post = make_groot_pre_post_processors(config, dataset_stats)

    # 전처리 파이프라인에 negative_action 정규화 스텝 추가
    # (negative_action도 state/action와 동일한 min-max norm 적용 필요)
    pre.steps.append(
        NegativeActionNormalizeStep(stats=dataset_stats or {})
    )
    return pre, post
```

`NegativeActionNormalizeStep`: batch의 `negative_action` 키가 있으면 action과 동일한 min-max norm 적용. None이면 pass-through.

---

### 3.6 `src/lerobot/policies/factory.py` 등록

```python
elif name == "groot_cl":
    from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
    return GrootCLPolicy

elif policy_type == "groot_cl":
    from lerobot.policies.groot_cl.configuration_groot_cl import GrootCLConfig
    return GrootCLConfig(**kwargs)
```

---

## 4. 학습 플로우

### Phase 1 (헤드 warm-up, ~500–1000 steps)

```python
config = GrootCLConfig(
    base_model_path="nvidia/GR00T-N1.5-3B",
    contrastive_phase="phase1",
    contrastive_loss_weight=1.0,    # flow matching loss 없음
    contrastive_backprop_backbone=False,  # phase1에서는 무관 (frozen)
    tune_llm=False, tune_visual=False,
    tune_projector=True, tune_diffusion_model=True,
)
policy = GrootCLPolicy(config)
# → _groot_model fully frozen, contrastive heads만 학습
# → 목적: heads가 의미있는 초기값을 갖도록 warm-up
```

### Phase 2a (핵심 Joint Fine-tuning)

```python
policy.set_contrastive_phase("phase2a")
policy.config.contrastive_backprop_backbone = True   # ← 필수
policy.config.contrastive_loss_weight = 0.05         # flow matching 스케일 맞춤 조정
# → VLM backbone까지 contrastive gradient 흐름
# → flow matching + contrastive 동시 최적화
# → VLM이 action-differentiated feature 출력하도록 학습
```

### Phase 2b (선택적 정리, 필요 시)

```python
policy.set_contrastive_phase("phase2b")
# → contrastive heads frozen, _groot_model만 tune_* config 기준 학습
# → contrastive loss = 0 (heads frozen이면 gradient 없음)
```

---

## 5. 트레이드오프 및 고려사항

### 5.1 1D CNN vs Transformer for Action

| | 1D CNN | Transformer |
|---|---|---|
| T=16 처리 | ✅ 충분한 temporal coverage (k=3, 2 layers) | 과잉 설계 |
| 구현 복잡도 | ✅ 낮음 | 높음 |
| 파라미터 수 | ✅ 적음 (~50K) | 많음 |
| 경계 정보 | `padding=1`로 양 끝 처리 | positional embedding 필요 |

→ **1D CNN 선택 확정**. T=16 수준에서 Transformer는 오버엔지니어링.

### 5.2 GR00TN15CL 로드 전략

`GR00TN15.from_pretrained()`는 내부적으로 `snapshot_download → super().from_pretrained()`를 사용한다.
`GR00TN15CL`도 같은 `config_class = GR00TN15Config`를 사용하므로 weight 로드 시 HF Hub가 등록된 config_class 기반으로 base 모델을 찾는다.

**안전한 로드 방법**: `GrootCLPolicy._create_groot_model()`에서 아래 순서로 진행.
1. `GR00TN15.from_pretrained()` 호출 → base 모델 로드
2. `GR00TN15CL(base_model.config)` 인스턴스 생성
3. `state_dict` 복사: `groot_cl_model.load_state_dict(base_model.state_dict())`

이렇게 하면 HF Hub의 `_auto_class` 등록 이슈 없이 안전하게 weight 이관 가능.

### 5.3 contrastive_backprop_backbone=True의 위험성

backbone에 두 가지 gradient 경로가 생긴다:
- flow matching loss → backbone (느리고 안정적)
- contrastive loss → backbone (빠르고 강함)

contrastive loss가 너무 크면 backbone이 flow matching과 무관한 방향으로 이동.
→ `contrastive_loss_weight=0.05` 이하로 시작, loss 로그 모니터링 후 조정.

### 5.4 negative 로드 비용

`ContrastiveLeRobotDataset.__getitem__`에서 매번 `super().__getitem__(neg_idx)` 호출.
Parquet action 컬럼 읽기는 빠르지만 video 디코딩이 포함될 경우 2x 비용.
→ 향후 최적화: action 컬럼만 별도 메모리맵(numpy mmap)으로 캐시.

### 5.5 task_index 없는 데이터셋

단일 task 데이터셋에서는 모든 에피소드가 같은 task → negative pair 의미 없음.
→ `precompute_negative_pairs.py`에서 경고 출력 + `contrastive_fallback_to_in_batch=True` 권장.

### 5.6 FlowmatchingActionHeadConfig / FlowmatchingActionHead 비수정 원칙 유지

모든 contrastive 로직은 `groot_cl/` 레이어에서 처리.
포팅 코드 업스트림 동기화 용이성 유지.

---

## 6. 미구현 항목 (향후)

- [ ] LoRA와 contrastive heads의 phase별 interaction (`lora_rank > 0` 시)
- [ ] 멀티 GPU DDP에서 negative pairs를 전체 GPU로 확장 (GatherLayer 패턴)
- [ ] action 컬럼 전용 mmap 캐시로 negative 로드 비용 절감
- [ ] Phase1 수렴 판단 자동화 (contrastive_loss plateau 감지)
- [ ] eval 지표: vlm_z ↔ pos_action_z cosine similarity vs vlm_z ↔ neg_action_z cosine similarity

---

## 7. Multi-GPU Accelerate 지원 — TODO

### 변경 파일
- [x] `scripts/train_groot_cl.py` — accelerate 통합 ✅
- [x] `scripts/cl_command.md` — multi-GPU 실행 방법 추가 ✅

### 구현 세부사항

#### prepare() 분리 전략
- `policy`, `dataloader` → 모듈 레벨에서 1회 `accelerator.prepare()`
- `optimizer`, `scheduler` → `train_loop` 내 phase마다 `accelerator.prepare()`
- 이유: DDP wrap은 1회만, optimizer는 phase마다 trainable params가 달라짐

#### 핵심 변경점
| 기존 | accelerate 버전 |
|------|----------------|
| `policy.to("cuda")` | `accelerator.prepare(policy)` |
| `DEVICE = "cuda"` | `DEVICE = accelerator.device` |
| `loss.backward()` | `accelerator.backward(loss)` |
| `clip_grad_norm_(policy.parameters(), ...)` | `accelerator.clip_grad_norm_(policy.parameters(), ...)` |
| `policy.set_contrastive_phase()` | `accelerator.unwrap_model(policy).set_contrastive_phase()` |
| `policy.save_pretrained()` | `accelerator.unwrap_model(policy).save_pretrained()` (main process만) |

#### mixed_precision 처리
- `modeling_groot_cl.py:forward()` 내부에 `torch.autocast(dtype=bfloat16)` 이미 존재
- accelerate `mixed_precision="no"` 설정 → autocast 중복 방지
- bf16 처리는 모델 내부에서만 담당
