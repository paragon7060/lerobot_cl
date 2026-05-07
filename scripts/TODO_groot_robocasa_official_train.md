# TODO: LeRobot에서 Isaac-GR00T 공식 finetune 흐름 맞추기

목표: `/home/minje/maxim/minje/clvla/lerobot_cl/src/lerobot/policies/groot_robocasa`가 `/home/minje/maxim/minje/clvla/benchmarks/robocasa365/Isaac-GR00T`의 공식 Isaac-GR00T 학습 흐름과 같은 방식으로 LeRobot에서 train되도록 맞춘다.

주요 공식 참고 파일:
- `Isaac-GR00T/scripts/gr00t_finetune.py`
- `Isaac-GR00T/gr00t/experiment/data_config.py`
- `Isaac-GR00T/gr00t/data/dataset.py`
- `Isaac-GR00T/gr00t/model/transforms.py`
- `Isaac-GR00T/gr00t/experiment/trainer.py`
- `robocasa/robocasa/utils/dataset_registry.py`
- `robocasa/robocasa/utils/dataset_registry_utils.py`

LeRobot 수정 대상:
- `lerobot_cl/src/lerobot/policies/groot_robocasa`
- `lerobot_cl/scripts` 아래 새 train script

## 0. 맞출 기준 결정

- [ ] 공식 `gr00t_finetune.py`를 기본값의 기준으로 둔다.
  - `data_config=panda_omron`
  - `embodiment_tag=new_embodiment`
  - `base_model_path=nvidia/GR00T-N1.5-3B`
  - `batch_size=128`
  - `max_steps=300000`
  - `save_steps=20000`
  - `lr=3e-5`
  - `weight_decay=1e-5`
  - `warmup_ratio=0.05`
  - `optim=adamw_torch`
  - `adam_beta1=0.95`, `adam_beta2=0.999`, `adam_epsilon=1e-8`
  - `lr_scheduler_type=cosine`
  - `bf16=True`, `tf32=True`
  - `gradient_accumulation_steps=1`
  - `dataloader_pin_memory=False`
  - `dataloader_persistent_workers=num_workers > 0`
  - `ddp_find_unused_parameters=False`

- [ ] 현재 LeRobot/Accelerate 스타일은 model input, optimizer grouping, sampler behavior, checkpoint 내용, resume 동작을 공식과 동일하게 재현할 수 있을 때만 유지한다.

## 1. Dataset soup 로딩 맞추기

- [ ] 공식 `--dataset-soup`에 대응되는 LeRobot train script 인자를 추가한다.
  - `DATASET_SOUP_REGISTRY[dataset_soup]`를 읽어야 한다.
  - 각 metadata dict의 `path`, `filter_key`, `task`, `split`, `source`, 선택적 `ds_weight`를 보존해야 한다.

- [ ] Robocasa registry 규칙을 손으로 다시 구현하지 않는다.
  - 공식 path 생성은 `dataset_registry_utils.get_ds_meta`에 있다.
  - 이 함수는 `robocasa.macros.DATASET_BASE_PATH` 또는 package-local `datasets` 폴더를 사용한다.
  - 새 script는 `--robocasa-dataset-base-path`를 제공하거나 필요한 macro/env 설정을 문서화한다.

- [ ] 공식 single-vs-mixture 동작을 맞춘다.
  - dataset 1개: `LeRobotSingleDataset` 하나 생성.
  - dataset 여러 개: 여러 `LeRobotSingleDataset` 생성 후 `LeRobotMixtureDataset` 생성.

- [ ] 공식 filtering을 지원한다.
  - 각 `ds_meta["filter_key"]`를 `LeRobotSingleDataset`에 넘긴다.
  - `10p`/`30p` soup 같은 subset 의미를 `filter_key`로 보존한다.

- [ ] 공식 mixture weight를 맞춘다.
  - 공식 script는 현재 `ds_weights = len(dataset) ** ds_weights_alpha`를 계산하고 `ds_weights[0]`로 normalize한다.
  - `add_cotraining_weights`에서 생기는 선택적 `ds_meta["ds_weight"]`는 쓰지 않는다. 이 동작을 그대로 둘지, 의도적으로 고칠지 결정해야 한다.
  - 인자는 유지한다: `balance_dataset_weights`, `balance_trajectory_weights`, `ds_weights_alpha`.

- [ ] 공식 sampler epoch 동작을 맞춘다.
  - 공식 `DualBrainTrainer`는 `dataset.set_epoch(epoch)`를 호출하는 custom `BaseSampler`를 쓴다.
  - Accelerate/DataLoader를 쓴다면 `LeRobotMixtureDataset.set_epoch`가 일관되게 호출되도록 sampler 또는 loop hook을 추가한다.

## 2. DataConfig와 transform 맞추기

- [ ] 최소한 `panda_omron`에 대해 공식 `DATA_CONFIG_MAP` 동작을 port하거나 재사용한다.
  - 공식 `PandaOmronDataConfig` key:
    - videos: `video.robot0_agentview_left`, `video.robot0_agentview_right`, `video.robot0_eye_in_hand`
    - states: `state.end_effector_position_relative`, `state.end_effector_rotation_relative`, `state.gripper_qpos`, `state.base_position`, `state.base_rotation`
    - actions: `action.end_effector_position`, `action.end_effector_rotation`, `action.gripper_close`, `action.base_motion`, `action.control_mode`
    - language: `annotation.human.task_description`
    - observation indices: `[0]`
    - action indices: `range(16)`

- [ ] official-style training에서는 현재 generic LeRobot processor 경로를 대체한다.
  - 현재 `GrootPackInputsStep`는 `observation.state`, `action`, 정렬된 `observation.images.*`에서 pack한다.
  - 공식 흐름은 이름이 정해진 Robocasa modality를 로드하고, modality-specific transform을 적용하고, concat한 뒤 `GR00TTransform`을 적용한다.
  - 정확히 맞추려면 새 script는 LeRobot `make_pre_post_processors`가 아니라 공식 `LeRobotSingleDataset` + 공식 transform output을 직접 써야 한다.

- [ ] 공식 video transform 순서를 맞춘다.
  - `VideoToTensor`
  - `VideoCrop(scale=0.95)`
  - `VideoResize(224, 224)`
  - `VideoColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)`
  - `VideoToNumpy`

- [ ] 공식 state/action transform 규칙을 맞춘다.
  - state position, gripper, base, rotation-relative/base-rotation field에 `min_max`.
  - `state.end_effector_rotation_relative`, `state.base_rotation`은 `rotation_6d`로 변환.
  - action position, rotation, base motion에 `min_max`.
  - `gripper_close`, `control_mode`에 `binary`.

- [ ] 공식 최종 model input key와 shape를 맞춘다.
  - `state`: `(B, 1, 64)`
  - `state_mask`: `(B, 1, 64)`
  - `action`: `(B, 16, 32)`
  - `action_mask`: `(B, 16, 32)`
  - `embodiment_id`: `(B,)`
  - `GR00TTransform.collate`에서 만든 `eagle_*` tokenizer/vision tensor.

- [ ] language formatting을 확인한다.
  - 공식 `GR00TTransform._apply_vlm_processing`은 raw language text를 text content item으로 넘긴다.
  - 현재 LeRobot `GrootEagleEncodeStep`는 language를 `str([lang])`로 감싼다.
  - 정확히 맞추려면 official-style training path에서는 list-string wrapping을 제거한다.

- [ ] Eagle processor 호출을 맞춘다.
  - 공식 collate는 `eagle_processor(text=text_list, images=image_inputs, return_tensors="pt", padding=True)`를 호출한다.
  - 현재 코드는 `images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False}`를 추가한다.
  - local port에 꼭 필요한지 확인하고, 필요 없으면 공식과 맞춘다.

## 3. Model loading과 action-head 맞추기

- [ ] `GR00TN15.from_pretrained(...)` 기본값을 공식과 맞춘다.
  - `tune_llm=False`
  - `tune_visual=False`
  - `tune_projector=True`
  - `tune_diffusion_model=True`

- [ ] 공식 action horizon 재생성 로직을 LeRobot model wrapper 또는 train script에 추가한다.
  - 공식은 `data_action_horizon = len(data_config_cls.action_indices)`를 쓴다.
  - `data_action_horizon != model.action_head.config.action_horizon`이면 `FlowmatchingActionHead`를 다시 만들고, 기존 weight를 `strict=False`로 로드한 뒤 아래를 업데이트한다.
    - `model.action_head`
    - `model.config.action_horizon`
    - `model.action_horizon`
    - `model.config.action_head_cfg["action_horizon"]`
    - 새 action head의 trainable parameter 설정.

- [ ] Robocasa에서는 `action_dim=32`, `max_action_dim=32`가 그대로 유지되는지 확인한다.

- [ ] compute dtype을 공식과 맞춘다.
  - `model.compute_dtype = "bfloat16"` 설정.
  - `model.config.compute_dtype = "bfloat16"` 설정.
  - bf16 autocast는 공식 tensor casting 의미를 바꾸지 않을 때만 사용한다.

- [ ] LoRA 지원 방식을 결정한다.
  - 공식 script는 `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_full_model`을 지원한다.
  - 현재 config에는 field가 있지만 LeRobot wrapper는 `get_lora_model`을 적용하지 않는다.
  - `gr00t.utils.peft.get_lora_model`을 port하거나, 구현 전까지는 `lora_rank > 0`일 때 명확히 error를 내도록 한다.

## 4. Trainer / optimizer / scheduler 맞추기

- [ ] 새 train script는 둘 중 하나의 경로를 선택한다.
  - 가장 정확한 경로: 공식 dataset class/transform을 import하고 공식 `TrainRunner`/`DualBrainTrainer`와 동등한 HF `Trainer`를 사용.
  - 허용 가능한 LeRobot 경로: Accelerate loop를 유지하되 아래 trainer 동작을 전부 맞춘다.

- [ ] loss 호출을 맞춘다.
  - 공식 `DualBrainTrainer.compute_loss`는 `outputs = model(inputs)`를 호출하고 `outputs["loss"]`를 사용한다.
  - 현재 `GrootPolicy.forward`는 batch를 filtering한 뒤 `_groot_model.forward`를 호출한다.
  - 새 script는 LeRobot policy wrapper mismatch를 피하려고 `_groot_model`을 직접 train해도 된다.

- [ ] optimizer parameter grouping을 맞춘다.
  - 공식은 layernorm/bias 기준으로 weight-decay group과 no-decay group을 나눈다.
  - 현재 script는 plain `AdamW(filter(...))`로 모든 trainable param에 weight decay를 적용한다.
  - Accelerate를 유지한다면 공식 grouping을 추가한다.

- [ ] scheduler를 맞춘다.
  - 공식 `TrainingArguments`는 `lr_scheduler_type="cosine"`과 `warmup_ratio`를 쓴다.
  - 현재 script는 `get_cosine_schedule_with_warmup`을 쓴다. 유지하되 warmup은 전체 `steps` 기준으로 계산한다.

- [ ] mixed precision을 맞춘다.
  - 공식은 `bf16=True`, `tf32=True`를 쓴다.
  - 현재 script는 `Accelerator(mixed_precision="no")`로 초기화하고 policy autocast에 의존한다.
  - Accelerate mixed precision을 `bf16`으로 설정하거나, autocast를 유지한다면 parity check를 명확히 둔다.

- [ ] gradient 동작을 맞춘다.
  - 공식 `gradient_checkpointing=False`.
  - 공식 `ddp_find_unused_parameters=False`.
  - 현재 script는 `find_unused_parameters=True`를 쓴다. local DDP 문제가 강제하지 않는 한 official-style training에서는 false로 바꾼다.

- [ ] dataloader 동작을 맞춘다.
  - `num_workers=config.dataloader_num_workers`
  - `pin_memory=False`
  - `persistent_workers=num_workers > 0`
  - 공식 dataset transform을 쓸 때는 LeRobot preprocessor stage를 넣지 않는다.

## 5. Checkpoint / resume 맞추기

- [ ] 공식 `DualBrainTrainer.save_model`처럼 `model.save_pretrained(output_dir, state_dict=state_dict)`로 tuned model을 저장한다.

- [ ] LeRobot checkpoint format을 유지할지, 공식 format과 bridge할지 결정한다.
  - 공식 checkpoint는 HF model folder다.
  - 기존 LeRobot checkpoint는 `pretrained_model/model.safetensors`, processor, optimizer, scheduler를 저장한다.
  - 정확한 공식 호환성을 위해서는 각 checkpoint를 HF-pretrained format으로 저장한다.
  - 필요하면 LeRobot wrapper metadata를 추가 artifact로 저장하되, 공식 resume에 필수로 만들지는 않는다.

- [ ] Resume 동작을 맞춘다.
  - 공식 `--resume=true`는 `output_dir` 안의 last checkpoint를 찾는다.
  - custom script를 쓴다면 둘 다 지원한다.
    - `--resume`: last checkpoint에서 resume.
    - `--resume-from-checkpoint PATH`: 명시 경로에서 resume.

## 6. 새 script 형태

- [ ] `lerobot_cl/scripts/train_groot_robocasa_official.py`를 추가한다.

- [ ] CLI 인자는 우선 공식과 맞춘다.
  - `--dataset-soup`
  - `--output-dir`
  - `--data-config`
  - `--batch-size`
  - `--max-steps`
  - `--num-gpus`
  - `--save-steps`
  - `--base-model-path`
  - `--tune-llm/--no-tune-llm`
  - `--tune-visual/--no-tune-visual`
  - `--tune-projector/--no-tune-projector`
  - `--tune-diffusion-model/--no-tune-diffusion-model`
  - `--resume`
  - `--learning-rate`
  - `--weight-decay`
  - `--warmup-ratio`
  - `--lora-rank`
  - `--lora-alpha`
  - `--lora-dropout`
  - `--lora-full-model`
  - `--dataloader-num-workers`
  - `--report-to`
  - `--embodiment-tag`
  - `--video-backend`
  - `--balance-dataset-weights`
  - `--balance-trajectory-weights`
  - `--ds-weights-alpha`

- [ ] local path bootstrapping을 추가한다.
  - 필요하면 Isaac-GR00T root를 `sys.path`에 넣는다.
  - 필요하면 Robocasa root를 `sys.path`에 넣는다.
  - remote file은 SSH로 수정하지 않는다. 이 repo는 `/home/minje/maxim` 아래에서 local edit한다.

- [ ] 공식과 같은 single-GPU / torchrun multi-GPU 동작을 추가한다.
  - `num_gpus > 1`이면 이미 `IS_TORCHRUN=1`이 아닌 경우 `torchrun --standalone --nproc_per_node={num_gpus}`로 재실행한다.

- [ ] 시작할 때 공식처럼 resolved config를 출력한다.

## 7. 실제 학습 전 검증 checklist

- [ ] 작은 soup 또는 task 1개로 dataset construction smoke test를 실행한다.
  - 첫 sample key와 shape를 출력한다.
  - batching 전 `state/action` shape가 `(1,64)`, `(16,32)`인지, collate 후 `(B,1,64)`, `(B,16,32)`인지 확인한다.

- [ ] remote maxim에서 forward/backward 1 step을 실행한다.
  - 실행은 반드시 `ssh maxim 'cd /home/seonho/minje/clvla/lerobot_cl && ...'` 형태로 한다.

- [ ] 공식 batch 1개와 LeRobot-script batch 1개를 비교한다.
  - 같은 `dataset_soup`, 같은 `data_config`, 같은 `filter_key`를 사용한다.
  - key, tensor shape, dtype, min/max range, `embodiment_id`를 비교한다.

- [ ] 같은 tune flag에서 trainable parameter count가 공식 script와 같은지 확인한다.

- [ ] checkpoint load를 확인한다.
  - `GR00TN15.from_pretrained(checkpoint_path)` 또는 local equivalent.
  - LeRobot inference compatibility가 필요하면 현재 `GrootPolicy.from_pretrained(checkpoint_path)`도 확인한다.

## 8. `groot_robocasa`에서 예상되는 code 변경

- [ ] `align_action_head_horizon(model, action_horizon, tune_projector, tune_diffusion_model)` 같은 helper를 추가한다.

- [ ] official-style optimizer parameter grouping을 추가하거나 외부에 노출한다.

- [ ] LoRA 적용을 추가하거나, 미지원 LoRA에 대해 명확한 hard error를 추가한다.

- [ ] `GrootPolicy.get_optim_params()`가 dict annotation mismatch가 아니라 iterable/list of parameters를 반환하도록 고친다.

- [ ] `GrootPolicy.forward()`가 LeRobot processor assumption 없이 이미 official format인 batch도 받을 수 있게 한다.

- [ ] `processor_groot.py`를 inference 전용으로 둘지 training에도 쓸지 결정한다.
  - training parity가 공식 dataset transform을 쓰면 새 train script에서는 이 processor를 제외한다.
  - LeRobot-native training도 계속 필요하면 빠진 공식 transform detail을 이 processor에 port한다.

## 9. 구현 전 물어볼 결정 사항

- [ ] 새 script가 공식 Isaac-GR00T module을 직접 import하게 할지, 아니면 최소 class만 `groot_robocasa`에 copy/port할지 결정한다.
  - 직접 import는 parity를 가장 빠르게 맞출 수 있다.
  - porting은 standalone LeRobot 통합은 깔끔하지만 유지보수할 code와 risk가 늘어난다.

- [ ] checkpoint를 공식 HF format만 저장할지, LeRobot format만 저장할지, 둘 다 저장할지 결정한다.

- [ ] 공식 `gr00t_finetune.py`가 현재 dataset length로 weight를 다시 계산하더라도, real/cotraining soup의 `ds_meta["ds_weight"]`를 존중할지 결정한다.

- [ ] 첫 target run에 사용할 `dataset_soup`를 결정한다.
  - registry의 흔한 option: `target50`, `pretrain_human50`, `pretrain_human100`, `pretrain_human300`, `lifelong_learning_phase1..4`, `real_multitask_realonly`, `real_multitask_cotrainDC`, `real_multitask_cotrain301`.

