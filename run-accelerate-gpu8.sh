#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4

module purge
module load pytorch

#pip install --user accelerate

MASTER_IP=$(ip -4 -brief addr show | grep -E 'hsn0|ib0' | grep -oP '([\d]+.[\d.]+)')
MASTER_PORT=29400

srun accelerate.sh --multi_gpu --num_processes=8 --num_machines=2 \
     --mixed_precision=no --dynamo_backend=no \
     --main_process_ip=$MASTER_IP --main_process_port=$MASTER_PORT \
     mnist_accelerate.py --epochs=100
