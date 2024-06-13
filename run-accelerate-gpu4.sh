#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4

module purge
module load pytorch

#pip install --user accelerate

srun apptainer_wrapper exec accelerate launch --multi_gpu --num_processes=4 --num_machines=1 \
     --mixed_precision=bf16 --dynamo_backend=no \
     mnist_accelerate.py --epochs=100
