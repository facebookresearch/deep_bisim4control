# Learning Invariant Representations for Reinforcement Learning without Reconstruction

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with:
```
source activate dbc
```

## Instructions
To train a DBC agent on the `cheetah run` task from image-based observations  run:
```
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir ./log \
    --seed 1
```
This will produce 'log' folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir log
```
and opening up tensorboad in your browser.

The console output is also available in a form:
```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000
```
a training entry decodes as:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
RLOSS - average reconstruction loss (only if it is trained from pixels and decoder)
```
while an evaluation entry:
```
| eval | S: 0 | ER: 21.1676
```
which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 10).

## CARLA
Download CARLA from https://github.com/carla-simulator/carla/releases, e.g.:
1. https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.8.tar.gz
2. https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.8.tar.gz

Add to your python path:
```
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.8/PythonAPI
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.8/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.8/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg
```
and merge the directories.

Then pull altered carla branch files:
```
git fetch
git checkout carla
```

Install:
```
pip install pygame
pip install networkx
```

Terminal 1:
```
cd CARLA_0.9.6
bash CarlaUE4.sh -fps 20
```

Terminal 2:
```
cd CARLA_0.9.6
# can run expert autopilot (uses privileged game-state information):
python PythonAPI/carla/agents/navigation/carla_env.py
# or can run bisim:
./run_local_carla096.sh --agent bisim --transition_model_type probabilistic --domain_name carla
```

## License
This project is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
