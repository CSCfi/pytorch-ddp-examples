#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:1

module purge
module load pytorch

# Old way with torch.distributed.run
# srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 mnist_ddp.py --epochs=100

# New way with torchrun
srun torchrun --standalone --nnodes=1 --nproc_per_node=1 mnist_ddp_profiler.py --epochs=100
