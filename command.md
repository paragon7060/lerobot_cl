# GR00T (lerobot050_groot or groot)
## setting lerobot050_groot
conda install -c nvidia cuda-toolkit=12.1 -y
pip install psutil
pip install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0" # --index-url https://download.pytorch.org/whl/cu1XX
pip install ninja "packaging>=24.2,<26.0" # flash attention dependencies
pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')"
pip install lerobot[groot]
## Using a multi-GPU setup
accelerate launch \
  --multi_gpu \
  --num_processes=$NUM_GPUS \
  $(which lerobot-train) \
  --output_dir=$OUTPUT_DIR \
  --save_checkpoint=true \
  --batch_size=$BATCH_SIZE \
  --steps=$NUM_STEPS \
  --save_freq=$SAVE_FREQ \
  --log_freq=$LOG_FREQ \
  --policy.push_to_hub=true \
  --policy.type=groot \
  --policy.repo_id=$REPO_ID \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id=$DATASET_ID \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --job_name=$JOB_NAME
### RoboCasa
CUDA_VISIBLE_DEVICES=1,3 accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    $(which lerobot-train) \
    --dataset.repo_id=paragon7060/robocasa_mg_gr00t_300_refined_wiener_filter \
    --policy.type=groot \
    --output_dir=./outputs/groot_robocasa_baseline/ \
    --job_name=groot_robocasa_baseline \
    --policy.repo_id=paragon7060/groot_robocasa_baseline \
    --policy.push_to_hub=false \
    --steps=50000 \
    --policy.device=cuda \
    --policy.batch_size=128 \
    --wandb.enable=true \
    --wandb.project=groot_robocasa \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    --save_freq=10000

## Using a single-GPU setup
### LIBERO
lerobot-train \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --policy.type=groot \
    --output_dir=./outputs/groot_libero_baseline/ \
    --job_name=groot_libero_baseline \
    --policy.repo_id=paragon7060/groot_libero_baseline \
    --policy.push_to_hub=false \
    --steps=30000 \
    --policy.device=cuda \
    --policy.batch_size=128 \
    --wandb.enable=true \
    --wandb.project=groot_libero \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    --save_freq=10000

### robocasa
python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=paragon7060/robocasa_mg_gr00t_300_refined_wiener_filter \
    --policy.type=groot \
    --output_dir=./outputs/groot_robocasa_baseline/ \
    --job_name=groot_robocasa_baseline \
    --policy.repo_id=paragon7060/groot_robocasa_baseline \
    --policy.push_to_hub=false \
    --steps=50000 \
    --policy.device=cuda \
    --policy.batch_size=128 \
    --wandb.enable=true \
    --wandb.project=groot_robocasa \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    --save_freq=10000

### arnold
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=paragon7060/arnold_data_pickup_fixed \
    --policy.type=groot \
    --output_dir=./outputs/groot_arnold_baseline/ \
    --job_name=groot_arnold_baseline \
    --policy.repo_id=paragon7060/groot_arnold_baseline \
    --policy.push_to_hub=false \
    --steps=50000 \
    --policy.tune_visual=true \
    --policy.device=cuda \
    --policy.batch_size=128 \
    --wandb.enable=true \
    --wandb.project=groot_arnold \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    --save_freq=10000
    2>&1 | tee outputs/baseline_arnold_log.txt


# pi05 (lerobot050 / '~/lerobot_series/lerobot_050/lerobot' )
## Multi-GPU
### ARNOLD
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    $(which lerobot-train) \
    --dataset.repo_id=paragon7060/arnold_data_pickup_fixed \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.max_state_dim=16 \
    --policy.max_action_dim=16 \
    --policy.chunk_size=16 \
    --policy.n_action_steps=16 \
    --output_dir=./outputs/pi05_arnold_baseline \
    --job_name=pi05_arnold_baseline \
    --policy.repo_id=paragon7060/pi05_arnold_baseline \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=50000 \
    --policy.device=cuda \
    --batch_size=32 \
    --wandb.enable=true \
    --wandb.project=pi05_arnold \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    --save_freq=10000
