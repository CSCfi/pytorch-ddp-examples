#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4,nvme:200

IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

set -x

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH

date
srun python3 benchmark_deepspeed.py --deepspeed --deepspeed_config ds_config_benchmark.json --datadir ${LOCAL_SCRATCH}/ilsvrc2012-torch $*
date
