# OpenPI HPC Fine-Tuning (Singularity + Slurm)

This guide implements the HPC plan for fine-tuning with Singularity. It mirrors the
repo's Docker training setup (Ubuntu 22.04 + CUDA + FFmpeg 7) and runs the same
commands from `local/scripts/README.md`.

## 1) Directory layout on /gpfs

Pick a project root on your HPC, for example:

`/gpfs/space/projects/<myproject>/openpi_hpc`

Recommended structure:

```
openpi_hpc/
  repo/                 # git checkout of this repo
  containers/           # openpi_train.sif (and/or openpi_train.tar)
  cache/
    singularity_cache/
    openpi_data/
    huggingface/
    jax_cache/
    wandb/
  outputs/
    assets/
    checkpoints/
  slurm_logs/
```

Use the setup helper in this directory:

```
local/scripts/hpc/setup_gpfs_layout.sh
```

## 2) Build the Singularity image

The training image is defined in `scripts/docker/train.Dockerfile` and includes
FFmpeg 7 built from source to avoid host FFmpeg mismatches.

**Option A (recommended): build with Docker locally, convert on HPC**

On a machine with Docker:

```
docker build -t openpi_train -f scripts/docker/train.Dockerfile .
docker save openpi_train -o openpi_train.tar
rsync -av openpi_train.tar <user>@<hpc>:/gpfs/space/projects/<myproject>/openpi_hpc/containers/
```

On the HPC (login node):

```
module load squashfs
module load singularity
cd /gpfs/space/projects/<myproject>/openpi_hpc/containers
singularity build openpi_train.sif docker-archive://openpi_train.tar
```

Or use the helper script:

```
PROJECT_ROOT=/gpfs/space/projects/<myproject>/openpi_hpc \
DOCKER_TAR=/gpfs/space/projects/<myproject>/openpi_hpc/containers/openpi_train.tar \
  local/scripts/hpc/build_sif_from_docker_archive.sh
```

**Option B: direct pull/build on HPC (if OCI access is allowed)**

```
module load squashfs
module load singularity
cd /gpfs/space/projects/<myproject>/openpi_hpc/containers
singularity build openpi_train.sif docker-daemon://openpi_train:latest
```

## 3) Slurm scripts (norm stats + train)

Use the scripts in this directory:

- `slurm_norm_stats.sh`
- `slurm_train.sh`

Both scripts:
- set Singularity cache/tmp to `/gpfs` to avoid `$HOME` quota issues
- bind repo, outputs, and caches into the container
- set `OPENPI_DATA_HOME` and Hugging Face cache paths
- keep JAX cache under `/gpfs` by setting `SINGULARITYENV_HOME=/openpi_home`

### Norm stats

```
sbatch local/scripts/hpc/slurm_norm_stats.sh
```

This runs:

```
uv run scripts/compute_norm_stats.py --config-name pi05_ur5_low_mem_finetune
```

### Training

```
sbatch local/scripts/hpc/slurm_train.sh
```

This runs:

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_ur5_low_mem_finetune --exp-name=<exp> --overwrite \
  --checkpoint-base-dir=/outputs/checkpoints --assets-base-dir=/outputs/assets
```

## 4) No internet on compute nodes (offline fallback)

If compute nodes cannot reach Hugging Face or GCS, pre-stage caches on a node
that has internet access. Use:

```
sbatch local/scripts/hpc/slurm_prefetch_cache.sh
```

Then re-run training with offline flags enabled in `slurm_train.sh`
(`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`).

## 5) Quick sanity checks

```
singularity exec --nv /gpfs/space/projects/<myproject>/openpi_hpc/containers/openpi_train.sif nvidia-smi
singularity exec --nv /gpfs/space/projects/<myproject>/openpi_hpc/containers/openpi_train.sif \
  bash -lc 'cd /app && uv run python -c "import jax; print(jax.devices())"'
```

