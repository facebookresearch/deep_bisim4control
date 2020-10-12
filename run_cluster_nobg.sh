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

DECODER_TYPE=pixel
NUM_LAYERS=4
NUM_FILTERS=32
IMG_SOURCE=video
AGENT=bisim

CDIR=/checkpoint/${USER}/DBC/${DOMAIN}_${TASK}
mkdir -p ${CDIR}

for TRANSITION_MODEL_TYPE in 'probabilistic'; do
for DECODER_TYPE in 'identity'; do
for SEED in 1 2 3; do
  SUBDIR=${AGENT}_transition${TRANSITION_MODEL_TYPE}_nobg/seed_${SEED}
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
    --num_train_steps 1000000 \
    --encoder_type ${ENCODER_TYPE} \
    --decoder_type ${DECODER_TYPE} \
    --action_repeat ${ACTION_REPEAT} \
    --resource_files \'/datasets01/kinetics/070618/400/train/driving_car/*.mp4\' \
    --num_layers ${NUM_LAYERS} \
    --num_filters ${NUM_FILTERS} \
    --transition_model_type ${TRANSITION_MODEL_TYPE} \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --decoder_weight_lambda 0.0000001 \
    --hidden_dim 1024 \
    --batch_size 128 \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5\
    --save_model \
    --work_dir ${SAVEDIR} \
    --seed ${SEED} >> ${SCRIPT}
  sbatch ${SLURM}
done
done
done
