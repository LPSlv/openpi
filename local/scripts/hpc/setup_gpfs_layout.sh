#!/bin/bash
set -euo pipefail

# Usage:
#   PROJECT_ROOT=/gpfs/space/projects/<myproject>/openpi_hpc \
#     local/scripts/hpc/setup_gpfs_layout.sh

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "ERROR: PROJECT_ROOT is not set."
  echo "Example: PROJECT_ROOT=/gpfs/space/projects/<myproject>/openpi_hpc $0"
  exit 1
fi

mkdir -p \
  "$PROJECT_ROOT/repo" \
  "$PROJECT_ROOT/containers" \
  "$PROJECT_ROOT/cache/singularity_cache" \
  "$PROJECT_ROOT/cache/openpi_data" \
  "$PROJECT_ROOT/cache/huggingface" \
  "$PROJECT_ROOT/cache/jax_cache" \
  "$PROJECT_ROOT/cache/wandb" \
  "$PROJECT_ROOT/outputs/assets" \
  "$PROJECT_ROOT/outputs/checkpoints" \
  "$PROJECT_ROOT/slurm_logs"

echo "Created directory layout under: $PROJECT_ROOT"
