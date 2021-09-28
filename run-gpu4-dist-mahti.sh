#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4

#module purge
#module load pytorch/1.9

srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 $*
