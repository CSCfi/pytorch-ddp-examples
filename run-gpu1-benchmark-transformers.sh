#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=0-1
#SBATCH --gres=gpu:a100:1,nvme:10

export HF_DATASETS_CACHE=/scratch/project_2001659/mvsjober/hf-cache/datasets/
export HF_METRICS_CACHE=/scratch/project_2001659/mvsjober/hf-cache/metrics/
export HF_MODULES_CACHE=/scratch/project_2001659/mvsjober/hf-cache/modules/
export OUTPUT_DIR=/scratch/project_2001659/mvsjober/hf-cache/output/${SLURM_JOB_ID}

echo $SING_IMAGE
python3 -c "import torch; print('PyTorch', torch.__version__); print(torch.__config__.show())" 2>/dev/null

HF_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
HF_DIR="transformers-$HF_VERSION"
test -d $HF_DIR || (echo "$HF_DIR not found"; exit 1)

RUN_CLM=$HF_DIR/examples/pytorch/language-modeling/run_clm.py

set -x

date
srun python3 $RUN_CLM \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --output_dir $OUTPUT_DIR
date
