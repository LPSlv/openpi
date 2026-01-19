#!/bin/bash
set -euo pipefail

# Usage:
#   PROJECT_ROOT=/gpfs/space/projects/<myproject>/openpi_hpc \
#   DOCKER_TAR=/gpfs/space/projects/<myproject>/openpi_hpc/containers/openpi_train.tar \
#     local/scripts/hpc/build_sif_from_docker_archive.sh

if [[ -z "${PROJECT_ROOT:-}" || -z "${DOCKER_TAR:-}" ]]; then
  echo "ERROR: PROJECT_ROOT and DOCKER_TAR must be set."
  exit 1
fi

module load squashfs
module load singularity

CONTAINERS_DIR="$PROJECT_ROOT/containers"
SIF_PATH="$CONTAINERS_DIR/openpi_train.sif"

mkdir -p "$CONTAINERS_DIR"
singularity build "$SIF_PATH" "docker-archive://$DOCKER_TAR"

echo "Built: $SIF_PATH"
