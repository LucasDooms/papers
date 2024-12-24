#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time={{walltime_formatted}}
#SBATCH --job-name={{id}}

module purge
conda activate new-cg-idps

python simulate.py run --job {{id}}
