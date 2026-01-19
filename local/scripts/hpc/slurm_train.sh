#!/bin/bash
#SBATCH --job-name=openpi_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/space/projects/<myproject>/openpi_hpc/slurm_logs/train-%j.out
#SBATCH --error=/gpfs/space/projects/<myproject>/openpi_hpc/slurm_logs/train-%j.err

set -euo pipefail

module load singularity
module load squashfs

PROJECT_ROOT="/gpfs/space/projects/<myproject>/openpi_hpc"
REPO_DIR="$PROJECT_ROOT/repo"
CONTAINER="$PROJECT_ROOT/containers/openpi_train.sif"
CACHE_DIR="$PROJECT_ROOT/cache"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"

EXP_NAME="my_experiment"

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

# Toggle offline mode if compute nodes lack internet.
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export WANDB_MODE=offline

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Keep JAX compilation cache on /gpfs to avoid $HOME quota usage.
export SINGULARITYENV_HOME="/openpi_home"

BIND_PATHS="$REPO_DIR:/app,$OUTPUTS_DIR:/outputs,$CACHE_DIR:/cache,$CACHE_DIR/jax_cache:/openpi_home,$OUTPUTS_DIR/assets:/app/assets"

singularity exec --nv --bind "$BIND_PATHS" "$CONTAINER" \
  bash -lc "cd /app && uv run scripts/train.py pi05_ur5_low_mem_finetune \
    --exp-name=${EXP_NAME} \
    --overwrite \
    --checkpoint-base-dir=/outputs/checkpoints \
    --assets-base-dir=/outputs/assets"
