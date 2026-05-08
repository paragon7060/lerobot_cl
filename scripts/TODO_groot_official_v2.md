# GR00T Robocasa Official-Style Training in LeRobot — Plan & TODO

## Context

목표는 공식 Isaac-GR00T `scripts/gr00t_finetune.py` 와 동일한 dataset / transform / collate / forward 흐름으로 LeRobot 쪽에서도 GR00T 계열 policy를 학습할 수 있게 하는 것. 단 사용자가 실제로 학습에 쓰고 싶은 policy는 `groot_robocasa`(baseline)에 더해 자기가 만든 `groot_cl`, `groot_mgd`, `groot_cl_v2` 같은 실험 policy들이라, 구현은 `groot_robocasa` 전용이 아니라 GR00T 계열 전체가 공유할 수 있는 **Robocasa preset + 공통 학습 어댑터** 형태로 가야 함. 학습은 우선 `pretrain` split 의 atomic task 1개로 smoke test 후 나중에 multi-task로 확장.

배경 사실:
- 모든 groot_* policy는 이미 `self._groot_model` (GR00TN15 인스턴스, official Isaac-GR00T 포팅) 을 갖고 있고 `forward(batch) -> (Tensor, dict)` 시그니처가 통일됨. → 공통 어댑터를 깔기 적합.
- 공식 Isaac-GR00T 코드는 48server에 이미 존재: `/home/minje/48server/clvla/benchmarks/robocasa365/Isaac-GR00T/` (= remote `/home/seonho/clvla/benchmarks/robocasa365/Isaac-GR00T/`). sys.path 주입으로 직접 import (사용자 결정).
- 학습 루프는 LeRobot 스타일 수동 loop (Accelerator) 으로 작성하되, hyperparameter / optimizer grouping / scheduler / batch 키는 공식과 동일하게 맞춤 (사용자 결정).
- **Batch parity check 필수** (사용자 명시): 비교 대상은 (1) 공식 Isaac-GR00T `gr00t_finetune.py` 경로가 만드는 batch 와 (2) 새 LeRobot official adapter 경로(이번에 만드는 `groot_common`)가 만드는 batch. 같은 raw sample 에 대해 키 집합 / shape / dtype / 값 범위 일치를 검증. **기존 LeRobot processor 와의 비교가 아님** — 기존 processor 는 eval/inference path 보존용이며, official-style training path 에서는 완전히 우회한다.

---

## 결정 사항 정리

| 항목 | 선택 |
|---|---|
| 공식 GR00T 코드 사용 방식 | 48server 내 Isaac-GR00T 를 sys.path 로 직접 import |
| Trainer | LeRobot 스타일 Accelerator 수동 loop, hyperparameter는 공식과 1:1 일치 |
| Dataset 스코프 | 1개 task smoke test 부터, CLI 로 N개 받게 설계 (Mixture 확장 여지 남김) |
| 학습 시 모델 | 각 policy 의 `policy._groot_model(batch)` 직접 호출 (LeRobot policy wrapper는 그대로 살아 있고, 추론 path 도 보존) |
| 지원 policy.type | `groot_robocasa`, `groot_cl`, `groot_mgd`, `groot_cl_v2` (factory.py 에 이미 모두 등록됨) |
| Parity check | 공식 Isaac-GR00T 경로 batch vs 새 LeRobot official adapter 경로 batch 비교 (smoke test 모드에서 fail-fast). 기존 LeRobot processor 와의 비교 아님 — 그건 eval/inference 전용으로 그대로 둠. |

---

## 구현 구조

### 1. 새 모듈: `src/lerobot/policies/groot_common/`

GR00T 계열 policy들이 공유할 Robocasa preset 과 학습용 헬퍼. 새로 만듦.

**파일들**:

- `__init__.py` — 아래 심볼 re-export.
- `robocasa_preset.py`
  - `@dataclass RobocasaPreset` 필드:
    - `base_model_path: str = "paragon7060/Robocasa_baseline"` (또는 official `nvidia/GR00T-N1.5-3B`)
    - `embodiment_tag: str = "new_embodiment"`
    - `chunk_size: int = 16`, `n_action_steps: int = 16`, `max_action_dim: int = 32`, `max_state_dim: int = 64`
    - `image_size: tuple[int, int] = (224, 224)`
    - `video_keys: list[str]` = robocasa 카메라 3개 (agentview_left/right + eye_in_hand)
    - `data_config_name: str = "panda_omron"` — official `gr00t.experiment.data_config.DATA_CONFIG_MAP` 키
    - `video_backend: str = "opencv"`
  - `def apply_to_policy_config(cfg, preset) -> None`: 각 groot_* config 에 preset 값 주입 (이미 robocasa 인 cfg는 no-op).
- `official_data.py` — Isaac-GR00T import 헬퍼
  - `_ensure_isaac_gr00t_on_path()` — 48server `Isaac-GR00T` 폴더를 sys.path 에 추가 (이미 있으면 skip).
  - `build_official_dataset(dataset_paths: list[str|Path], preset, embodiment_tag) -> LeRobotSingleDataset | LeRobotMixtureDataset`
    - 단일 path → `LeRobotSingleDataset(modality_configs=cfg.modality_config(), transforms=cfg.transform(), embodiment_tag=tag, video_backend=...)`
    - 다중 path → `LeRobotMixtureDataset` with `balance_dataset_weights=True`, `balance_trajectory_weights=True`, `ds_weights_alpha=0.4`
  - `build_official_collate(eagle_processor) -> Callable` — `gr00t.model.transforms.collate` 부분 적용.
  - `extract_eagle_processor(policy)` — `policy._groot_model.backbone.eagle_processor` 또는 동등한 경로 헬퍼.
- `training_adapter.py`
  - `def forward_with_groot_batch(policy, batch) -> tuple[Tensor, dict]` — **우선순위**:
    1. `policy` 에 `forward_official_batch(batch)` 메서드가 있으면 그걸 호출 (각 policy 의 custom loss 보존용 hook). 반환은 `(loss, metrics_dict)` 시그니처로 통일.
    2. 없으면 fallback: `outputs = policy._groot_model(batch); loss = outputs["loss"]` → `(loss, {"loss": loss.detach()})`.
    - 어느 쪽이든 LeRobot policy 의 processor 흐름은 우회 (공식 batch 가 이미 모델이 기대하는 모양).
    - `groot_cl`, `groot_mgd`, `groot_cl_v2` 의 본 실험 학습 시 custom loss 를 살리려면 각 policy 에 `forward_official_batch` 를 추후 추가. 1차 smoke test 단계에서는 4개 모두 fallback 으로 통과 OK.
  - `assert_groot_compatible(policy)` — `_groot_model` 존재/타입 확인, GrootRobocasaPolicy/GrootCLPolicy/GrootMGDPolicy/GrootCLv2Policy 만 허용.
  - `def assert_official_config_match(policy, preset) -> None` — policy 생성 직후 호출. `_groot_model.action_head.action_horizon`, action dim, state dim (각 policy config 의 값) 과 trainable parameter count 를 stdout 에 출력. preset 의 `chunk_size` / `n_action_steps` / `max_action_dim` / `max_state_dim` 과 다르면 **hard error** (raise). 학습 시작 전에 silent mismatch 를 막는다.
  - `def unwrap_groot_model(policy_or_wrapped, accelerator)` — DDP/Accelerator wrap 이후 `_groot_model` 접근 / 체크포인트 저장 시 사용. 내부적으로 `accelerator.unwrap_model(policy_or_wrapped)._groot_model` 반환.
- `parity.py`
  - **목적**: 공식 Isaac-GR00T `gr00t_finetune.py` 가 만드는 batch 와, 이번에 만드는 LeRobot official adapter (`groot_common.official_data` + 공식 collate) 가 만드는 batch 가 동일한지 검증. 기존 LeRobot `processor_groot` 는 비교 대상 아님 (그건 eval/inference 전용).
  - `def check_batch_parity(*, official_batch, adapter_batch, atol=1e-4) -> ParityReport`:
    - 키 집합 비교 (양쪽 키 차이를 정확히 출력)
    - 공통 키 shape/dtype 비교
    - 수치 비교: `state`, `action`, `state_mask`, `action_mask`, `eagle_input_ids`, `eagle_pixel_values` — `torch.allclose` (이미지/연속값) / `torch.equal` (마스크/토큰)
    - 차이 발견 시 어떤 키 / 어떤 위치 / 차이 크기 출력
  - `def run_parity_smoke_test(dataset_path, preset)` —
    - **Path A (공식 reference)**: 공식 `gr00t_finetune.py` 와 동일한 방식으로 1 sample 빌드. 즉 `LeRobotSingleDataset(...)` 직접 인스턴스화 + 공식 `DATA_CONFIG_MAP[name]` 의 transform + 공식 `collate` 로 1-batch.
    - **Path B (어댑터)**: 새 `groot_common.official_data.build_official_dataset(...)` + `build_official_collate(...)` 로 같은 sample 1-batch.
    - 같은 seed / 같은 episode index 사용하여 두 path 의 출력을 `check_batch_parity` 로 비교. 차이가 있으면 어댑터에 버그 있는 것이므로 fail-fast.

### 2. 새 학습 스크립트: `scripts/train_groot_robocasa_official.py`

LeRobot 의 dataclass-config + parser 패턴을 따르되, 데이터/transform/collate/forward 는 공식 Isaac-GR00T 흐름을 그대로 사용.

**핵심 흐름**:

1. CLI 파싱 (LeRobot `lerobot.configs.parser` 사용):
   - `--dataset_paths` (List[str], 1개 이상). 디렉토리 자동 스캔 옵션도 있음 (예: `--dataset_root` + glob).
   - `--policy.type` ∈ {`groot_robocasa`, `groot_cl`, `groot_mgd`, `groot_cl_v2`}.
   - 공식 hyperparameter 기본값:
     `batch_size=128`, `max_steps=300000`, `learning_rate=3e-5`, `weight_decay=1e-5`,
     `warmup_ratio=0.05`, `lr_scheduler="cosine"`, `bf16=True`, `tf32=True`,
     `adam_betas=(0.95, 0.999)`, `adam_eps=1e-8`, `dataloader_num_workers=8`,
     `save_steps=20000`, `save_total_limit=100`.
   - `--smoke_test` (bool): 1개 dataset, 작은 값으로 max_steps=2, parity check 강제.
   - `--tune_llm`, `--tune_visual`, `--tune_projector=True`, `--tune_diffusion_model=True`, `--lora_rank=0`.

2. Isaac-GR00T sys.path 주입 (`_ensure_isaac_gr00t_on_path()`).

3. **Dataset 구성** (공식): `build_official_dataset(dataset_paths, preset)` → official `LeRobotSingleDataset` / `LeRobotMixtureDataset`. transform 은 `DATA_CONFIG_MAP[preset.data_config_name].transform()` 사용 — `ComposedModalityTransform` (Video* + StateAction* + ConcatTransform + GR00TTransform).

4. **Policy 생성** (LeRobot factory):
   - `apply_to_policy_config(policy_cfg, preset)` 로 robocasa 기본값을 주입.
   - `make_policy(policy_cfg, dataset.meta=...)` 로 GrootRobocasaPolicy/GrootCLPolicy/GrootMGDPolicy/GrootCLv2Policy 인스턴스 생성.
   - `assert_groot_compatible(policy)` 호출.
   - `assert_official_config_match(policy, preset)` 호출 — action horizon / action dim / state dim / trainable parameter count 를 stdout 에 출력. preset 과 다르면 hard error.
   - `eagle_processor = extract_eagle_processor(policy)`.

5. **DataLoader**: `DataLoader(dataset, batch_size=..., collate_fn=build_official_collate(eagle_processor), num_workers=..., pin_memory=False)`.

6. **Optimizer / Scheduler**:
   - param grouping: bias / LayerNorm.weight / 1-D 파라미터 → weight_decay=0, 나머지 → weight_decay=1e-5.
   - `torch.optim.AdamW(param_groups, lr=3e-5, betas=(0.95, 0.999), eps=1e-8)`.
   - `transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(max_steps*0.05), num_training_steps=max_steps)`.

7. **Accelerator** (`accelerate.Accelerator(mixed_precision="bf16")`): `accelerator.prepare(policy, optimizer, dataloader, scheduler)`.

8. **Smoke test / parity step** (max_steps 진입 직전 1회):
   - `run_parity_smoke_test(dataset_paths[0], preset)` 호출 — 공식 reference path 와 어댑터 path 의 batch 가 동일한지 검증. 실패 시 학습 중단.
   - 어댑터 dataloader 에서 1 batch fetch 해서 키/shape/dtype 출력 (state, state_mask, action, action_mask, embodiment_id, eagle_input_ids, eagle_attention_mask, eagle_pixel_values, eagle_image_grid_thw 등).
   - `forward_with_groot_batch(policy, batch)` 1 step → `loss.backward()` → optimizer step → 정상 종료 확인.

9. **본 학습 loop**:
   - `for step in range(max_steps):` 안에서 dataloader iterator → batch → `loss, _ = forward_with_groot_batch(policy, batch)` → `accelerator.backward(loss)` → `optimizer.step()` / `scheduler.step()` / `optimizer.zero_grad()`.
   - `logging_steps` 마다 loss / lr / step time 로깅 (wandb / tensorboard, `--report_to`).
   - `save_steps` 마다 체크포인트 저장 (Accelerator/DDP 이후이므로 **반드시 `accelerator.unwrap_model(policy)` 사용**):
     - `unwrapped = accelerator.unwrap_model(policy)`
     - `unwrapped.save_pretrained(output_dir/checkpoint-{step})` (LeRobot 형식)
     - 추가로 `unwrapped._groot_model.save_pretrained(output_dir/checkpoint-{step}/groot_model)` (공식 HF 형식, eval / official inference 호환). DDP wrap 된 채 `_groot_model` 직접 접근 금지.
     - optimizer / scheduler / step state 저장 (resume 용).
   - `--resume` 시 위 저장 경로에서 복원.

### 3. 기존 코드 수정 (최소)

- `src/lerobot/policies/groot_common/__init__.py` 신설.
- `src/lerobot/policies/groot_robocasa/configuration_groot.py`, `groot_cl/configuration_groot_cl.py`, `groot_mgd/configuration_groot.py`, `groot_cl_v2/configuration_groot_cl_v2.py`:
  - `apply_to_policy_config` 가 셋해야 할 필드들이 모두 노출되어 있는지 점검. 누락된 필드만 추가 (대부분 이미 존재).
- `factory.py` 변경 없음 (이미 4개 policy 모두 등록).
- `processor_groot.py` 류는 변경 없음 — LeRobot 추론 path 보존.
- (선택) 추후 `groot_cl`, `groot_mgd`, `groot_cl_v2` 본 실험 학습에 들어갈 때 각 policy 의 `modeling_*.py` 에 `forward_official_batch(self, batch) -> (loss, metrics)` hook 추가. 1차 smoke test 단계에서는 추가 불필요 (fallback 으로 통과).

---

## 핵심 파일 경로 요약

수정/신설 (모두 48server `/home/seonho/clvla/lerobot_cl/` = local `/home/minje/48server/clvla/lerobot_cl/` 아래):

- 신설: `src/lerobot/policies/groot_common/__init__.py`
- 신설: `src/lerobot/policies/groot_common/robocasa_preset.py`
- 신설: `src/lerobot/policies/groot_common/official_data.py`
- 신설: `src/lerobot/policies/groot_common/training_adapter.py`
- 신설: `src/lerobot/policies/groot_common/parity.py`
- 신설: `scripts/train_groot_robocasa_official.py`

참조 (변경 없음, import 만):

- 공식: `/home/seonho/clvla/benchmarks/robocasa365/Isaac-GR00T/gr00t/{data,model,experiment}/...`
- LeRobot policy 등록: `src/lerobot/policies/factory.py:130-145, 200-210, 443-471`
- 각 policy `_groot_model` 노출:
  - `groot_robocasa/modeling_groot.py` (GrootRobocasaPolicy → GrootPolicy)
  - `groot_cl/modeling_groot_cl.py:25` (GrootCLPolicy)
  - `groot_mgd/modeling_groot.py:69` (GrootMGDPolicy)
  - `groot_cl_v2/modeling_groot_cl_v2.py:55` (GrootCLv2Policy)

데이터:

- `/home/seonho/groot_robocasa/robocasa_dataset/v1.0/pretrain/atomic/<task>/` — LeRobot v2.1 layout, modality.json 포함, 20 fps, robot_type=PandaOmron.

---

## 작업 항목 (TODO)

- [x] **1. Isaac-GR00T 핵심 모듈 시그니처 정독** — 완료
  - `gr00t_finetune.py` (`ArgsConfig` + main flow), `LeRobotSingleDataset` / `LeRobotMixtureDataset`, `DATA_CONFIG_MAP` (panda_omron 이 robocasa modality.json 과 정확히 일치), `GR00TTransform` / `collate` 모두 정독
  - 공식 collate 출력 키 확정: `state` `[B,1,64]`, `state_mask` `[B,1,64]`, `action` `[B,16,32]`, `action_mask` `[B,16,32]`, `embodiment_id`, `segmentation_target` / `segmentation_target_mask` / `has_real_action`, `eagle_input_ids` / `eagle_attention_mask` / `eagle_pixel_values` / `eagle_image_grid_thw`
  - GR00TTransform default: `state_horizon=1, action_horizon=16, max_state_dim=64, max_action_dim=32`

- [x] **2. groot_* policy forward 시그니처 / `_groot_model` 인터페이스 정독** — 완료
  - 4개 policy 모두 `self._groot_model` (= `GR00TN15`) 노출, `forward(batch) -> (Tensor, dict)` 통일
  - 기존 `GrootPolicy.forward` 는 batch keys 를 `state, state_mask, action, action_mask, embodiment_id, eagle_*` 로 필터링; 공식 학습 path 에선 full batch 를 그대로 모델에 넘겨야 함 (`segmentation_target`, `has_real_action` 포함). 어댑터는 직접 `policy._groot_model(batch)` 호출.

- [x] **3. LeRobot configs/parser 패턴 정독** — 완료
  - `@parser.wrap()` + `@dataclass` config (TrainPipelineConfig 상속하지 않아도 됨). 단, **`from __future__ import annotations` 쓰면 draccus 가 type 을 string 으로 받아 깨짐** — 새 스크립트에선 사용 금지.
  - `LeRobotDatasetMetadata` 는 우리 path 에선 불필요 (preset 이 input/output_features 채움). `make_policy` 우회하고 `get_policy_class(name)(config=cfg)` 직접 호출.

- [x] **4. `src/lerobot/policies/groot_common/` 모듈 신설** — 완료
  - 5 파일 작성: `__init__.py`, `robocasa_preset.py`, `official_data.py`, `training_adapter.py`, `parity.py`
  - **Policy type 등록 안 함** (factory.py 손 안 댐) — 그냥 utility 모듈
  - `apply_to_policy_config(cfg, preset)` 가 base_model_path / embodiment_tag / chunk_size=16 / n_action_steps=16 / max_action_dim=32 / max_state_dim=64 / image_size=(224,224) / video_backend / input_features (3 robocasa cameras + state) / output_features (action shape 12) 를 in-place 주입
  - `build_official_dataset(paths, preset, ...)` — 1개면 LeRobotSingleDataset, 2개+ 면 LeRobotMixtureDataset (`np.power(len, 0.4)` weight 정규화, `balance_*_weights`, `seed=42`, `metadata_config={percentile_mixing_method: weighted_average}`) — `gr00t_finetune.py` 와 1:1
  - `forward_with_groot_batch(policy, batch)` 우선순위: `forward_official_batch` 가 있으면 그것, 없으면 `_groot_model(batch)["loss"]` fallback
  - `assert_official_config_match(policy, preset)` — action_horizon / action_dim / state_dim / trainable param count 출력 + preset mismatch 시 hard error
  - `unwrap_groot_model(policy, accelerator)` — DDP/Accelerator wrap 후 `_groot_model` 안전 접근
  - `run_parity_smoke_test(dataset_path, preset)` — 같은 sample 을 (A) 공식 reference path 와 (B) 어댑터 path 로 빌드 후 키/shape/dtype/value 비교

- [x] **5. `scripts/train_groot_robocasa_official.py` 작성** — 완료
  - CLI 통과 확인 (`--help` 정상 출력); `--policy.type` 에 4개 + 다른 LeRobot policy 들 모두 노출됨
  - Hyperparameter 공식 1:1: `lr=3e-5, weight_decay=1e-5, warmup_ratio=0.05, cosine, betas=(0.95,0.999), eps=1e-8, save_steps=20000, max_steps=300000, batch_size=128`
  - bf16 은 `mixed_precision="no"` + `torch.autocast(bf16)` (policy `use_bf16=True` 기본값) — `gr00t_finetune.py` 의 `bf16=True` 와 동치
  - param grouping: bias / LayerNorm / 1-D 텐서 → weight_decay=0
  - `--smoke_test` 시 max_steps=2, batch_size<=2, num_workers=0, 첫 dataset 만, parity check 강제, 1 batch shape 출력
  - checkpoint 저장: `accelerator.unwrap_model(policy).save_pretrained` + `_groot_model.save_pretrained(.../groot_model)` + optimizer/scheduler/RNG state
  - `--resume[_from]` 지원

- [/] **6. Smoke test — 4개 policy.type 검증** — 진행중
  - 검증 기준 명령 (smoke test) — 실제 atomic LeRobot dataset 경로는 `<task>/<date>/lerobot/`:
    ```bash
    ssh seonho@166.104.35.48
    cd /home/seonho/clvla/lerobot_cl
    CUDA_VISIBLE_DEVICES=0 \
      /home/seonho/miniconda3/envs/public_groot/bin/python \
      scripts/train_groot_robocasa_official.py \
      --policy.type=groot_robocasa \
      --dataset_paths='[/home/seonho/groot_robocasa/robocasa_dataset/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot]' \
      --output_dir=/tmp/groot_smoke_robocasa \
      --batch_size=2 --max_steps=2 --smoke_test=true --report_to=none
    ```
  - 검증 항목:
    - `[parity] OK — reference and adapter batches match.` 출력, state/action/mask/eagle_* 키/shape/dtype 일치
    - `state: [B, 1, 64]`, `state_mask: [B, 1, 64]`, `action: [B, 16, 32]`, `action_mask: [B, 16, 32]`, `embodiment_id: [B]`, `eagle_input_ids` / `eagle_attention_mask` / `eagle_pixel_values` / `eagle_image_grid_thw` 존재
    - 2 step forward/backward 정상 종료
    - `<output_dir>/checkpoint-0000002/` 에 LeRobot 형식 + `groot_model/` 공식 형식 두 가지 다 존재
  - 4개 policy.type (`groot_robocasa`, `groot_cl`, `groot_mgd`, `groot_cl_v2`) 모두 통과해야 끝
  - `--resume` 으로 1 회 더 → 저장된 step 부터 1 step 진행되는지 확인

  **환경 셋업 노트** (실행 전 1회):
  - `lerobot050_groot` / `minje` env 둘 다 transformers 5.3.0 인 반면 공식 gr00t pyproject 는 `transformers==4.51.3` / `pydantic==2.10.6` 핀 → 충돌. 같은 env 에 numpydantic 만 추가하면 pydantic 2.12 ↔ numpydantic 1.6.7 schema generation 버그 발생.
  - 결론: 새 conda env `public_groot` 생성, `pip install -e Isaac-GR00T` 로 모든 deps 설치, `pip install -e lerobot_cl --upgrade-strategy only-if-needed` 로 lerobot 추가. (현재 단계: gr00t deps 설치 진행 중)
  - minje env 에 잠시 추가했던 numpydantic / albumentations / kornia / dm_tree 등은 모두 uninstall + dm_tree 0.1.9 복원 완료.

---

## 본 학습 (별도 단계, 본 plan 의 verification 범위 밖)

동일 스크립트에 `--dataset_paths` 를 atomic 65 개 또는 atomic+composite 300 개로 늘리고 `batch_size=128`, `max_steps=300000` 으로 띄움.
