#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=30
#SBATCH --gres=gpu:a100:4

export NCCL_DEBUG=INFO

set -x

srun python3 $*
