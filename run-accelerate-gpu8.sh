#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpularge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=15
#SBATCH --gres=gpu:gh200:4

module purge
module load python-pytorch

MASTER_ADDR=$(hostname -i)
MASTER_PORT=12802
NUM_PROCS=$(expr ${SLURM_NNODES} \* ${SLURM_GPUS_ON_NODE})

# Note: --machine_rank must be evaluated on each node, hence the LAUNCH_CMD setup
export LAUNCH_CMD="
    accelerate launch \
        --multi_gpu --mixed_precision=bf16 --dynamo_backend=no \
        --num_machines=${SLURM_NNODES} \
        --num_processes=${NUM_PROCS} \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MASTER_ADDR} \
        --main_process_port=${MASTER_PORT} \
        mnist_accelerate.py --epochs=100 \
    "
echo ${LAUNCH_CMD}
srun bash -c "${LAUNCH_CMD}"
