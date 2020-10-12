#!/bin/bash

DOMAIN=carla098
TASK=highway

SAVEDIR=./save
mkdir -p ${SAVEDIR}

CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent 'deepmdp' \
    --init_steps 1000 \
    --num_train_steps 1000000 \
    --encoder_type pixelCarla098 \
    --decoder_type pixel \
    --img_source video \
    --resource_files 'distractors/*.mp4' \
    --action_repeat 4 \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --decoder_weight_lambda 0.0000001 \
    --hidden_dim 1024 \
    --total_frames 10000 \
    --num_filters 32 \
    --batch_size 128 \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK} \
    --seed 1 $@ \
    --frame_stack 3 \
    --image_size 84 \
    --eval_freq 10000 \
    --num_eval_episodes 25 \
    --render
