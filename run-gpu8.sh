#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4

module purge
module load pytorch/1.9

set -x

srun python3 $*
