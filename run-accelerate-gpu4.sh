#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:4

module purge
module load python-pytorch

accelerate launch --multi_gpu --num_processes=4 --num_machines=1 \
     --mixed_precision=bf16 --dynamo_backend=no \
     mnist_accelerate.py --epochs=100
