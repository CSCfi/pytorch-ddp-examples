#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4

#module purge
#module load pytorch/1.9

export NCCL_DEBUG=INFO
#export SING_IMAGE=/appl/soft/ai/singularity/images/pytorch_1.9.0_csc_custom.sif

set -x

srun python3 $*
