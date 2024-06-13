#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4

module purge
module load pytorch

GPUS_PER_NODE=4
MASTER_ADDR=$(hostname -i)
MASTER_PORT=12802

# Note: --machine_rank must be evaluated on each node, hence the LAUNCH_CMD setup
export LAUNCH_CMD="
    accelerate launch \
        --multi_gpu --mixed_precision no \
        --num_machines=${SLURM_NNODES} \
        --num_processes=$(expr ${SLURM_NNODES} \* ${GPUS_PER_NODE}) \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MASTER_ADDR} \
        --main_process_port=${MASTER_PORT} \
        mnist_accelerate.py --epochs=100 \
    "
echo ${LAUNCH_CMD}
srun singularity_wrapper exec bash -c "${LAUNCH_CMD}"
