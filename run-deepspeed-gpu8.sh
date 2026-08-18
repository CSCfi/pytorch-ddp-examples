#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpularge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:4

module purge
module load python-pytorch

srun python3 mnist_deepspeed.py --epochs=100 \
     --deepspeed --deepspeed_config ds_config.json
