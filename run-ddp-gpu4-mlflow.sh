#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:4

module purge
module load python-pytorch

# New way with torchrun
srun torchrun --standalone --nnodes=1 --nproc_per_node=4 mnist_ddp_mlflow.py --epochs=100
