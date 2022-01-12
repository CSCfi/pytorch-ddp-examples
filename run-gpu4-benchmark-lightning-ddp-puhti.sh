#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=30
#SBATCH --gres=gpu:v100:4,nvme:200

IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

set -x

tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH

date
srun python3 benchmark_lightning_ddp.py --gpus=4 --epochs=1 --datadir ${LOCAL_SCRATCH}/ilsvrc2012-torch $*
date
