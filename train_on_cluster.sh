#!/usr/bin/env bash

#SBATCH --partition=# Partition to submit to
#SBATCH --job-name=#job name
#SBATCH --array=0
#SBATCH --gres=gpu:4
# Runtime in D-HH:MM, if needed
#SBATCH --time=666-23:59:00
#SBATCH --mem=70G #memory requirements
#SBATCH --output=./debug/geom_guidance_tokens-%A_%a.out # STDOUT
#SBATCH --ntasks-per-node=4 #tasks per node
#SBATCH --cpus-per-task=4 #cpus per task


echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
#export HF_HOME=#path to local HF folder

MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
PORT=$((29814 + SLURM_ARRAY_TASK_ID))
data_root=""
categories_path=$data_root/categories.json
textured_images_dir=$data_root/controlnet_images_offset_all/
prompts_path=$data_root/foreground.txt
pb_emb_path=$data_root/shapenet_pointbert_tokens/
output_dir=output_folder/
gg_depth=6
gg_heads=8
gg_drop_path_rate=0.1
exp_name=token_guidance_d_${gg_depth}_h_${gg_heads}_drop_${gg_drop_path_rate}_loss_weight_by_t/
split_path=stats/train_val.txt

accelerate launch --main_process_port $PORT train_geometry_guidance.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --categories_path=$categories_path \
  --textured_images_dir=$textured_images_dir \
  --pb_emb_path=$pb_emb_path \
  --prompts_path=$prompts_path \
  --train_batch_size=24 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --learning_rate=5.0e-05 \
  --lr_warmup_steps=1000 \
  --max_images=10000000 \
  --checkpointing_steps=1000 \
  --dataloader_num_workers=4 \
  --output_dir=$output_dir/$exp_name \
  --gg_depth=$gg_depth \
  --gg_heads=$gg_heads \
  --gg_drop_path_rate=$gg_drop_path_rate \
  --use_pb_tokens \
  --split_path=$split_path \
  --lr_scheduler=constant_with_warmup \
  --weight_loss_by_t \
  --loss_weight_m=500 \
  --loss_weight_s=250 \
  --use_transform \
  --min_crop_scale=0.8 \
#  --resume_from_checkpoint='latest' #use if you want to resume checkpoint


