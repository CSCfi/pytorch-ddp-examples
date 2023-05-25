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

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400                   

# Old way with torch.distributed.run
# srun python3 -m torch.distributed.run \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --nproc_per_node=4 \
#     --rdzv_id=$SLURM_JOB_ID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
#     mnist_ddp.py --epochs=100

# New way with torchrun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    mnist_ddp.py --epochs=100
