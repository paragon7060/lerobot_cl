# INSIGHT groot baseline
CUDA_VISIBLE_DEVICES=0,1 accelerate launch     --num_processes=2     --mixed_precision=no     scripts/train_groot_baseline.py     --dataset.repo_id=paragon7060/INSIGHTfixposV3     --dataset.root=/home/seonho/workspace/data/paragon7060/INSIGHTfixposV3     --dataset.video_backend=pyav     --output_dir=./outputs/insight_groot_baseline_vt     --job_name=insight_groot_baseline_vt     --steps=100000     --save_checkpoint=true     --save_freq=25000     --batch_size=32     --policy.tune_visual=true     --policy.lora_rank=0     --wandb.enable=true     --wandb.project=groot_insight     --wandb.entity=RwHlabs     2>&1 | tee outputs/baseline_log.txt

# INSIGHT GR00T cl
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes=2 \
    --mixed_precision=no \
    scripts/train_groot_cl.py \
    --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
    --dataset.root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
    --neg_pairs_path=/home/bluepot/cl_ws/negative_pairs.json \
    --policy.type=groot_cl \
    --policy.groot_pretrained_path=./outputs/groot_baseline/checkpoints/last_checkpoint/pretrained_model \
    --output_dir=./outputs/groot_cl_v1 \
    --job_name=groot_cl_v1 \
    --phase1_steps=2000 \
    --phase2a_steps=15000 \
    --batch_size=4 \
    --policy.lora_rank=16 \
    --policy.lora_target=vision \
    --wandb.enable=true \
    --wandb.project=groot_insight \
    --wandb.entity=RwHlabs \
    --wandb.disable_artifact=true \
    2>&1 | tee outputs/cl_log.txt