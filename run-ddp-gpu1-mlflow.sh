#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:1

module purge
module load python-pytorch

# New way with torchrun
srun torchrun --standalone --nnodes=1 --nproc_per_node=1 mnist_ddp_mlflow.py --epochs=100
