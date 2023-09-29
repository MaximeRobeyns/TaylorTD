#!/bin/bash

#SBATCH --job-name TaylorRL
#SBATCH --partition cnu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus 1
#SBATCH --time 0-20:00:00
#SBATCH --mem 64G
#SBATCH --output taylor_rl_output

# Load any modules you need here, e.g.
module load lang/python/anaconda/3.9.7-2021.12-tensorflow.2.7.0

# Load a standard python virtual environment
source /user/work/$(whoami)/path/to/venv/bin/activate

# Alternatively load a conda env (need to load a lang/python/anconda module)
conda activate my_env

# run stuff here, e.g.
python main.py with env_name=GYMMB_HalfCheetah-v2 agent_alg=td3 tdg_error_weight=5. td_error_weight=1. record=True

