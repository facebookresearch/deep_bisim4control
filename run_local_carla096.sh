#!/bin/bash

DOMAIN=carla096
TASK=highway
AGENT=deepmdp
SEED=5
DECODER_TYPE=identity
TRANSITION_MODEL=deterministic

SAVEDIR=./save
#SAVEDIR=/checkpoint/${USER}/pixel-pets/carla/${AGENT}_${TRANSITION_MODEL}_currrew_${DECODER_TYPE}/seed_${SEED}
mkdir -p ${SAVEDIR}

CUDA_VISIBLE_DEVICES=1 python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent ${AGENT} \
    --init_steps 100 \
    --num_train_steps 100000 \
    --encoder_type pixelCarla096 \
    --decoder_type ${DECODER_TYPE} \
    --resource_files 'distractors/*.mp4' \
    --action_repeat 4 \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --encoder_stride 2 \
    --decoder_weight_lambda 0.0000001 \
    --hidden_dim 1024 \
    --replay_buffer_capacity 100000 \
    --total_frames 10000 \
    --num_layers 4 \
    --num_filters 32 \
    --batch_size 128 \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --work_dir ${SAVEDIR} \
    --transition_model_type ${TRANSITION_MODEL} \
    --seed ${SEED} $@ \
    --frame_stack 3 \
    --image_size 84 \
    --save_model >> ${SAVEDIR}/output.txt \
#    --render
