#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4,nvme:200

IMAGENET_DATA=/scratch/dac/data/ilsvrc2012-torch-resized-new.tar

set -x

tar -xf $IMAGENET_DATA -C $LOCAL_SCRATCH

date
srun singularity_wrapper exec deepspeed benchmark_deepspeed.py --deepspeed --deepspeed_config ds_config_benchmark.json --datadir ${LOCAL_SCRATCH}/ilsvrc2012-torch $*
date
