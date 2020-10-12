#!/bin/bash

CURDIR=`pwd`
CODEDIR=`mktemp -d -p ${CURDIR}/tmp`

cp ${CURDIR}/*.py ${CODEDIR}
cp -r ${CURDIR}/local_dm_control_suite ${CODEDIR}/
cp -r ${CURDIR}/dmc2gym ${CODEDIR}/
cp -r ${CURDIR}/agent ${CODEDIR}/

DOMAIN=${1:-walker}
TASK=${2:-walk}
ACTION_REPEAT=${3:-2}
NOW=${4:-$(date +"%m%d%H%M")}
ENCODER_TYPE=pixel

DECODER_TYPE=identity
NUM_LAYERS=4
NUM_FILTERS=32
IMG_SOURCE=video
AGENT=bisim
BATCH_SIZE=512
ENCODER_LR=0.001
NUM_FRAMES=100
BISIM_COEF=0.5
CDIR=/checkpoint/${USER}/DBC/${DOMAIN}_${TASK}
mkdir -p ${CDIR}

for NUM_FRAMES in 1000; do
for TRANSITION_MODEL_TYPE in 'ensemble'; do
for SEED in 1 2 3; do
  SUBDIR=${AGENT}_${BISIM_COEF}coef_${TRANSITION_MODEL_TYPE}_frames${NUM_FRAMES}_${IMG_SOURCE}kinetics/seed_${SEED}
  SAVEDIR=${CDIR}/${SUBDIR}
  mkdir -p ${SAVEDIR}
  JOBNAME=${NOW}_${DOMAIN}_${TASK}
  SCRIPT=${SAVEDIR}/run.sh
  SLURM=${SAVEDIR}/run.slrm
  CODEREF=${SAVEDIR}/code
  extra=""
  echo "#!/bin/sh" > ${SCRIPT}
  echo "#!/bin/sh" > ${SLURM}
  echo ${CODEDIR} > ${CODEREF}
  echo "#SBATCH --job-name=${JOBNAME}" >> ${SLURM}
  echo "#SBATCH --output=${SAVEDIR}/stdout" >> ${SLURM}
  echo "#SBATCH --error=${SAVEDIR}/stderr" >> ${SLURM}
  echo "#SBATCH --partition=learnfair" >> ${SLURM}
  echo "#SBATCH --nodes=1" >> ${SLURM}
  echo "#SBATCH --time=4000" >> ${SLURM}
  echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
  echo "#SBATCH --signal=USR1" >> ${SLURM}
  echo "#SBATCH --gres=gpu:volta:1" >> ${SLURM}
  echo "#SBATCH --mem=500000" >> ${SLURM}
  echo "#SBATCH -c 1" >> ${SLURM}
  echo "srun sh ${SCRIPT}" >> ${SLURM}
  echo "echo \$SLURM_JOB_ID >> ${SAVEDIR}/id" >> ${SCRIPT}
  echo "nvidia-smi" >> ${SCRIPT}
  echo "cd ${CODEDIR}" >> ${SCRIPT}
  echo MUJOCO_GL="osmesa" LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent ${AGENT} \
    --init_steps 1000 \
    --bisim_coef ${BISIM_COEF} \
    --num_train_steps 1000000 \
    --encoder_type ${ENCODER_TYPE} \
    --decoder_type ${DECODER_TYPE} \
    --encoder_lr ${ENCODER_LR} \
    --action_repeat ${ACTION_REPEAT} \
    --img_source ${IMG_SOURCE} \
    --num_layers ${NUM_LAYERS} \
    --num_filters ${NUM_FILTERS} \
    --resource_files \'/datasets01/kinetics/070618/400/train/driving_car/*.mp4\' \
    --eval_resource_files \'/datasets01/kinetics/070618/400/train/driving_car/*.mp4\' \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --total_frames ${NUM_FRAMES} \
    --decoder_weight_lambda 0.0000001 \
    --hidden_dim 1024 \
    --batch_size ${BATCH_SIZE} \
    --transition_model_type ${TRANSITION_MODEL_TYPE} \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5\
    --work_dir ${SAVEDIR} \
    --seed ${SEED} >> ${SCRIPT}
  sbatch ${SLURM}
done
done
done