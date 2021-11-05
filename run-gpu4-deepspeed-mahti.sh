#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4

srun singularity_wrapper exec deepspeed $*
#srun singularity_wrapper exec ~/.local/bin/deepspeed $*
