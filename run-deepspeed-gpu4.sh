#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:4

module purge
module load python-pytorch

srun deepspeed mnist_deepspeed.py --epochs=100 \
     --deepspeed --deepspeed_config ds_config.json
