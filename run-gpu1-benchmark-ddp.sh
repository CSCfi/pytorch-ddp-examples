#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=0-1
#SBATCH --gres=gpu:a100:1,nvme:200

IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

set -x

tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH

date
srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 benchmark_ddp.py --datadir ${LOCAL_SCRATCH}/ilsvrc2012-torch $*
date
