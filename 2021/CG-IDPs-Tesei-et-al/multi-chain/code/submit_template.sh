#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time={{walltime_formatted}}
#SBATCH --job-name={{name}}_{{temp}}

module purge
conda activate new-cg-idps

python simulate.py --name {{name}} --temp {{temp}} --walltime {{walltime_seconds}}
