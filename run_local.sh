#!/bin/bash

DOMAIN=cartpole
TASK=swingup

SAVEDIR=./save

MUJOCO_GL="osmesa" LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=1 python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent 'bisim' \
    --init_steps 1000 \
    --num_train_steps 1000000 \
    --encoder_type pixel \
    --decoder_type pixel \
    --img_source video \
    --resource_files 'distractors/*.mp4' \
    --transition_model_type 'probabilistic' \
    --action_repeat 2 \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --decoder_latent_lambda 0.000001 \
    --decoder_weight_lambda 0.0000001 \
    --hidden_dim 1024 \
    --total_frames 10000 \
    --num_layers 4 \
    --num_filters 32 \
    --batch_size 128 \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK} \
    --seed 1 $@
