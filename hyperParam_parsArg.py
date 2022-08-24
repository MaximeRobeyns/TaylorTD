import numpy as np
import os
import sys
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--job-name', '-j',type=str, nargs='?')
parser.add_argument('--partition',     type=str, nargs='?', default='cnu')
parser.add_argument('--nodes',         type=int,nargs='?', default=1)
parser.add_argument('--ntasks-per-node',        type=int, nargs='?', default=1) 
parser.add_argument('--cpus-per-task', '-c',type=int,nargs='?',default=8) 
parser.add_argument('--gpus',    '-g', type=int, nargs='?',default=1)
parser.add_argument('--hours',   '-t', type=int, nargs='?', default=10, help="time in hours")
parser.add_argument('--mins',          type=int, nargs='?', default=0, help="time in mins")
parser.add_argument('--mem',  '-m', type=int, nargs='?',default=64,    help="memory in gp")
parser.add_argument('--output',         type=str, nargs='?',             help="queue")
parser.add_argument('--exclude',type=str,nargs='?', default='=bp1-gpu004,bp1-gpu006,bp1-gpu016,bp1-gpu017')
parser.add_argument('--cmd',           type=str, nargs='*',             help="job command --- must be last argument")
parser.add_argument('--env-name', '-e', type=str)
parser.add_argument('--agent_alg', '-a', type=str, default='td3_taylor')
parser.add_argument('--state_cov', '-sc', type=bool, default=False)
parser.add_argument('--action_cov', '-ac', type=bool, default=False)
parser.add_argument('--state_cov_training', '-tsc', type=bool, default=False)
parser.add_argument('--action_cov_training', '-tac', type=bool, default=False)
parser.add_argument('--n_steps',  type=int, default=20000)
parser.add_argument('--run_type', type=str, default='train')

# split input args on --cmd
cmd_idx = sys.argv.index('--cmd')
args = sys.argv[1:cmd_idx]
args = parser.parse_args(args)
cmd = ' '.join(sys.argv[(1+cmd_idx):])

print(f"#!/bin",)
print(f"#SBATCH --job-name {args.job_name}")
print(f"#SBATCH --partition {args.partition}")
print(f"#SBATCH --nodes {args.nodes}")
print(f"#SBATCH --ntasks-per-node {args.ntasks_per_node}")
print(f"#SBATCH --cpus-per-task {args.cpus_per_task}"  )
print(f"#SBATCH --gpus {args.gpus}"  )
print(f"#SBATCH --time=0-{args.hours}:{args.mins}:00")
print(f"#SBATCH --mem  {args.mem}G"  )
print(f"#SBATCH --output {args.output}"  )
print(f"#SBATCH --exclude={args.exclude}"  )

print('')
print("cd", "/user/work/px19783/code_repository/RL_project/TaylorRL")

if args.state_cov:
    # add state cov true variable
    if args.state_cov_training:
        state_cov_range = torch.linspace(0.000001,0.0001,5)
    else:
        state_cov_range = [0.00001]

if args.action_cov_training:
    action_cov_range = torch.linspace(0.000001,0.0001,5)
else:
    action_cov_range = [0.25]


counter = 0
seeds = [1] # Add random seeds    
# MISSING: need to add the directory    
for s in seeds:

    for sc in state_cov_range:

        for ac in action_cov_range:

            print(cmd,' with env_name=GYMMB_'+args.env_name,'agent_alg='+args.agent_alg,f'td3_action_cov={ac}',f'td3_state_cov={sc}',f'n_total_steps={args.n_steps}', f'seed={s}',f'run_type='+args.run_type,f'run_number={counter}')
            counter+=1
