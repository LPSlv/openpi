#!/bin/bash
#SBATCH --job-name=openpi_prefetch
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/space/projects/<myproject>/openpi_hpc/slurm_logs/prefetch-%j.out
#SBATCH --error=/gpfs/space/projects/<myproject>/openpi_hpc/slurm_logs/prefetch-%j.err

set -euo pipefail

module load singularity
module load squashfs

PROJECT_ROOT="/gpfs/space/projects/<myproject>/openpi_hpc"
REPO_DIR="$PROJECT_ROOT/repo"
CONTAINER="$PROJECT_ROOT/containers/openpi_train.sif"
CACHE_DIR="$PROJECT_ROOT/cache"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"

mkdir -p \
  "$CACHE_DIR/singularity_cache" \
  "$CACHE_DIR/openpi_data" \
  "$CACHE_DIR/huggingface" \
  "$CACHE_DIR/jax_cache" \
  "$CACHE_DIR/wandb" \
  "$OUTPUTS_DIR/assets" \
  "$OUTPUTS_DIR/checkpoints"

export SINGULARITY_CACHEDIR="$CACHE_DIR/singularity_cache"
export SINGULARITY_TMPDIR="/tmp"

export OPENPI_DATA_HOME="$CACHE_DIR/openpi_data"
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_DIR="$CACHE_DIR/wandb"
export WANDB_MODE=offline

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Keep JAX compilation cache on /gpfs to avoid $HOME quota usage.
export SINGULARITYENV_HOME="/openpi_home"

BIND_PATHS="$REPO_DIR:/app,$OUTPUTS_DIR:/outputs,$CACHE_DIR:/cache,$CACHE_DIR/jax_cache:/openpi_home,$OUTPUTS_DIR/assets:/app/assets"

# 1) Warm the dataset and assets caches via norm stats.
singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
  bash -lc 'cd /app && uv run scripts/compute_norm_stats.py --config-name pi05_ur5_low_mem_finetune --max-frames 128'

# 2) Warm model checkpoint cache with a tiny training run.
singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
  bash -lc 'cd /app && uv run scripts/train.py pi05_ur5_low_mem_finetune \
    --exp-name=cache_warmup \
    --overwrite \
    --num-train-steps=1 \
    --log-interval=1 \
    --save-interval=1 \
    --wandb-enabled=False \
    --checkpoint-base-dir=/outputs/checkpoints \
    --assets-base-dir=/outputs/assets'
