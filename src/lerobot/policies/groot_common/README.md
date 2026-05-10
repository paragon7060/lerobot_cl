# `groot_common`

## What `groot_common` Is

`groot_common`은 여러 policy와 train script가 공통으로 사용할 official-compatible batch semantics / parity / training adapter utility layer다.

기본 Robocasa dataset root:

- `/home/minje/48server/groot_robocasa/robocasa_dataset/v1.0/pretrain`

핵심 구성:

- `BatchSpec`
- `RobocasaPreset`
- `LeRobotNativeBatchBuilder`
- parity helper
- training adapter
- config drift assert

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
- camera/state/action order와 padded dimension 공유
- Robocasa preset 공유
- raw batch -> train batch 변환 helper
- parity view / key-shape-dtype report helper
- optional policy forward 호출 helper
- preset 대비 config drift 조기 검출

## What `groot_common` Is Not

`groot_common`은 아래 역할을 맡지 않는다.

- custom loss 구현체
- policy-specific head/module 구현체
- optimizer/backward/save orchestration
- official export 구현체

즉, `groot_common`은 공통 입력·검증 계층이고, 각 policy는 자기 모델과 자기 loss를 가진다.

## Policy Boundary

각 policy는 아래 책임을 가진다.

- model structure
- `forward()`
- intermediate feature handling
- policy-specific loss

`GrootPolicy`, `GrootMGDPolicy`, RKD/CL/LoRA 계열 후속 policy는 같은 batch semantics를 공유할 수 있지만, loss 구현은 각 policy 또는 plugin 내부에 둔다.

예를 들어 MGD loss도 `groot_common`이 아니라 policy 내부 구현이 책임진다.

## Recommended Train Script Flow

train script에서의 권장 흐름은 아래와 같다.

```text
1. BatchSpec / RobocasaPreset 생성
2. LeRobotNativeBatchBuilder 생성
3. raw_batch -> build_train_batch(raw_batch, preprocessor)
4. build_parity_view(batch)
5. 기본은 policy.forward(batch) 호출
6. 필요하면 training_adapter helper 사용
7. loss.backward()
8. optimizer.step()
9. LeRobot native save
10. manifest 기록
```

train script는 orchestrator이고, batch semantics 자체를 새로 정의하지 않는다.

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

핵심은 train script가 official eval용 layout을 직접 책임지지 않는다는 점이다. train script는 학습 산출물을 LeRobot native 경계 안에서만 저장한다.

## Autoshim v2 Export Contract

autoshim v2는 native save 이후 별도 export gate다.

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

현재 `LeRobotNativeBatchBuilder`는 `batch_spec` 인자를 직접 받지 않고 `preset`으로부터 `batch_builder.batch_spec`을 계산한다. 따라서 train script는 직접 만든 `batch_spec`을 문서화/manifest에 남기고, 실제 실행 경로에서는 `batch_builder.batch_spec`과 함께 관리하는 방식이 안전하다.

`training_adapter`는 선택적 helper다. policy별 `forward()` signature 통일, preset mismatch assert, DDP/Accelerator unwrap helper가 필요할 때만 쓴다.

## Summary

`groot_common`은 custom loss 구현체가 아니라 여러 policy/train script가 같은 official-compatible batch semantics를 공유하도록 만드는 공통 입력·검증 계층이다. Train script는 LeRobot native checkpoint만 저장하고, official-format export는 autoshim v2 같은 별도 export gate에서 수행한다.
