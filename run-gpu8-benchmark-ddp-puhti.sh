#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4,nvme:200

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

set -x

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH

date
srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    benchmark_ddp.py --datadir ${LOCAL_SCRATCH}/ilsvrc2012-torch $*
date
