#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:4

module purge
module load python-pytorch

# Old way with torch.distributed.run
# srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 mnist_ddp.py --epochs=100

# New way with torchrun
srun torchrun --standalone --nnodes=1 --nproc_per_node=4 mnist_ddp.py --epochs=100
