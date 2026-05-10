# HowToUseGrootCommon

이 문서는 `train_groot_*.py` 계열 스크립트가 `groot_common`을 어떤 경계로 사용해야 하는지 정리한다.

기본 runtime dataset root:

- `/home/seonho/groot_robocasa/robocasa_dataset/robocasa_human_v3`

## What `groot_common` Is

`groot_common`은 여러 policy와 train script가 공통으로 사용할 official-compatible batch semantics / parity / training adapter utility layer다.

핵심 구성:

- `BatchSpec`
- `RobocasaPreset`
- `LeRobotNativeBatchBuilder`
- parity helper
- training adapter
- config drift assert
- `robocasa_official_runtime.py`
- `filter_key_subset.py`

역할 분리는 아래처럼 유지한다.

```text
groot_common
= BatchSpec / RobocasaPreset / LeRobotNativeBatchBuilder / parity helper / training adapter

각 policy
= 모델 구조 / policy-specific forward / policy-specific loss

train script
= config 조립 / dataloader / optimizer / backward / save / manifest 관리
```

`groot_common`이 담당하는 것:

- official-compatible batch semantics 정의
- raw batch -> train batch 변환 helper
- parity view / report helper
- optional training adapter helper
- config drift 검증
- v3 runtime repo discovery helper
- official-style `filter_key` subset helper

## What `groot_common` Is Not

`groot_common`은 아래 역할을 맡지 않는다.

- custom loss 구현체
- policy-specific head/module 구현체
- optimizer/backward/save orchestration
- official export 구현체

즉, script는 `groot_common`을 공통 입력 계층으로 사용하고, 각 policy는 자기 forward/loss를 유지한다.

## Policy Boundary

각 policy는 아래 책임을 가진다.

- model structure
- `forward()`
- intermediate feature handling
- policy-specific loss

`GrootPolicy`, `GrootMGDPolicy`, RKD/CL/LoRA 계열 후속 policy는 같은 batch semantics를 공유할 수 있지만, loss 구현은 각 policy 또는 plugin 내부에 둔다.

예를 들어 어떤 custom loss도 `groot_common`이 아니라 해당 policy 쪽 구현이 책임진다.

## Runtime Discovery

`robocasa_official_runtime.py`

- 변환된 `robocasa_human_v3` runtime 입력 탐색용
- `<root>/robocasa_<split>_human_atomic/task_XXXX`
- `<root>/robocasa_<split>_human_composite/task_XXXX`
- `MultiLeRobotDataset`에 넣을 `repo_ids`와 `dataset_paths`를 만드는 용도
- 공식 원본을 읽는 게 아니라, 변환된 v3 dataset을 train-ready 입력으로 정규화하는 용도
- 실제 train script는 이 파일만 사용해도 된다.

`robocasa_official_dataset.py`

- legacy validation helper only

`filter_key_subset.py`

- official과 같은 `filter_key` subset 규칙을 v3 runtime dataset에도 재사용하는 helper
- `subset_<split>_<filter_key>_seed<seed>.json` manifest를 dataset root에 캐시하고 재사용한다.
- 학습 런타임에서는 이미 materialize된 presliced dataset root를 그냥 쓰면 되며, 이 util은 주로 one-time subset 생성/검증에 사용한다.

## Recommended Train Script Flow

권장 흐름:

```text
1. BatchSpec / RobocasaPreset 생성
2. LeRobotNativeBatchBuilder 생성
3. runtime discovery로 repo_ids 생성
4. raw_batch -> build_train_batch(raw_batch, preprocessor)
5. build_parity_view(batch)
6. 기본은 policy.forward(batch) 호출
7. 필요하면 training_adapter helper 사용
8. loss.backward()
9. optimizer.step()
10. LeRobot native save
11. manifest 기록
```

## Native Save Contract

train script는 official format checkpoint를 직접 만들지 말고, LeRobot native checkpoint만 저장한다.

train script가 저장해야 할 것:

- native checkpoint dir
- `config.json`
- `model.safetensors` 또는 native model weights
- processor/state/stat 관련 native files가 있으면 같이 저장
- run manifest

manifest에 포함할 것:

- source checkpoint 또는 `model_path`
- `base_model_path`
- policy type
- dataset root
- `repo_id` 또는 task id
- batch spec 요약
- parity report path
- train loss
- `grad_norm`
- `changed_param_count`
- native checkpoint path
- output dir
- env name
- commit hash가 가능하면 포함

## Autoshim v2 Export Contract

official-format export는 train script 바깥의 별도 gate로 분리한다.

autoshim v2가 하는 일:

- LeRobot native checkpoint에서 official base model weight 추출
- `_groot_model.` prefix stripping
- official config/metadata를 `base_model_path`에서 복사
- tokenizer/processor/metadata 등 official loader가 기대하는 파일 배치
- 필요 시 `embed_tokens` 보정
- training-only key 제외
- official-format checkpoint를 fresh output dir에 저장
- export manifest 저장
- official dry-load 확인

중요 원칙:

- train script는 official format으로 직접 저장하지 않는다.
- native save와 official export를 분리한다.
- official export dir은 항상 fresh output dir을 사용한다.
- 기존 export output path를 덮어쓰지 않는다.
- custom policy의 auxiliary module, teacher, projection head, loss head 등 training-only key는 official export에서 제외 후보로 둔다.
- official export는 base model weight만 official eval 가능한 layout으로 복원하는 단계다.

## Script Rules

- batch key/shape/order를 train script마다 직접 하드코딩하지 않는다.
- parity 비교 로직을 각 script에 복붙하지 않는다.
- policy-specific loss를 `groot_common`에 넣지 않는다.
- official export를 train script에서 직접 하지 않는다.
- native save 이후 autoshim v2 export gate를 사용한다.
- 산출물은 fresh output dir에 저장한다.

## Example

현재 코드 기준 generic 예시는 아래와 같다.

```python
from lerobot.policies.groot_common.batch_spec import BatchSpec
from lerobot.policies.groot_common.robocasa_preset import RobocasaPreset
from lerobot.policies.groot_common.batch_builder import LeRobotNativeBatchBuilder
batch_spec = BatchSpec(
    camera_order=(
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_right",
    ),
    state_order=("robot_state",),
    action_order=("robot_action",),
    action_horizon=16,
    chunk_size=16,
    padded_state_dim=64,
    padded_action_dim=32,
)

preset = RobocasaPreset()
batch_builder = LeRobotNativeBatchBuilder(preset=preset)

raw_batch = next(iter(dataloader))
batch = batch_builder.build_train_batch(raw_batch, preprocessor)
parity_view = batch_builder.build_parity_view(batch)
effective_batch_spec = batch_builder.batch_spec

outputs = policy.forward(batch)
loss = outputs["loss"]

loss.backward()
optimizer.step()

# native save
# manifest 기록
```

현재 `LeRobotNativeBatchBuilder`는 `batch_spec` 인자를 직접 받지 않고 `preset` 기반으로 `batch_builder.batch_spec`을 계산한다.

`training_adapter`는 선택적 helper다. policy별 `forward()` signature 통일, preset mismatch assert, DDP/Accelerator unwrap helper가 필요할 때만 쓴다.

## Runtime Usage

`robocasa_official_runtime.py`는 train script에서 직접 사용하는 입력 discovery helper다.

권장 사용 흐름은 아래와 같다.

```python
from lerobot.policies.groot_common import (
    RobocasaPreset,
    LeRobotNativeBatchBuilder,
    discover_robocasa_official_runtime_repos,
)

preset = RobocasaPreset.robocasa365_pretrain()
batch_builder = LeRobotNativeBatchBuilder(preset=preset)

discovery = discover_robocasa_official_runtime_repos(
    root="/home/seonho/groot_robocasa/robocasa_dataset/robocasa_human_v3",
    split="pretrain",
)

repo_ids = discovery.repo_ids
dataset_paths = discovery.dataset_paths

raw_batch = next(iter(dataloader))
batch = batch_builder.build_train_batch(raw_batch, preprocessor)
parity_view = batch_builder.build_parity_view(batch)
outputs = policy.forward(batch)
loss = outputs["loss"]
```

주의할 점:

- `runtime.py`는 `v1.0` 원본 split을 읽는 용도가 아니다.
- `runtime.py`는 `robocasa_human_v3` 같은 변환된 v3 layout만 대상으로 한다.
- `runtime.py`의 결과를 `MultiLeRobotDataset`에 넘겨 train-ready repo list로 사용한다.

## Train Script Usage

`train_groot_mgd_robocasa.py`에서는 이제 `robocasa_human_v3_presliced_100` 같은 **official-equivalent presliced root**를 그대로 쓰면 된다.

즉 학습 시에는:

- `--dataset.root=/home/seonho/groot_robocasa/robocasa_dataset/robocasa_human_v3_presliced_100`
- `train script`는 별도의 `filter_key` 해석이나 episode slicing을 하지 않는다.
- `groot_common.filter_key_subset`은 필요한 경우 새 presliced root를 만들거나 검증할 때만 사용한다.

## Summary

`groot_common`은 custom loss 구현체가 아니라 여러 policy/train script가 같은 official-compatible batch semantics를 공유하도록 만드는 공통 입력·검증 계층이다. Train script는 LeRobot native checkpoint만 저장하고, official-format export는 autoshim v2 같은 별도 export gate에서 수행한다.
