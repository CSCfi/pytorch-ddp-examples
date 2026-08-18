#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:4

module purge
module load python-pytorch

srun torchrun --standalone --nnodes=1 --nproc_per_node=4 mnist_ddp.py --epochs=100
