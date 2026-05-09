# Training Guide

## Available Training Configs

All configs are defined in `src/openpi/training/config.py` (search for `ur5`).

| Config | Model | Base Checkpoint | Notes |
|--------|-------|-----------------|-------|
| `pi0_ur5` | Pi0 | `pi0_base` | Full fine-tune, z-score normalization |
| `pi05_ur5` | Pi0.5 | `pi05_base` | Full fine-tune, quantile normalization |
| `pi05_ur5_droid` | Pi0.5 | `pi05_droid` | Uses DROID checkpoint, quantile norm |
| `pi05_ur5_low_mem_finetune` | Pi0.5 | `pi05_base` | LoRA, lower memory (~22 GB) |
| `pi05_ur5_busthetable` | Pi0.5 | `pi05_base` | Bus-the-table task dataset |

## Step 1: Compute Normalization Statistics

Before training, compute norm stats for your dataset:

```bash
uv run scripts/compute_norm_stats.py --config-name <config>
```

This writes `norm_stats.json` to `assets/<config>/<asset_id>/`.

**Normalization types:**
- **Z-score** (Pi0): `(x - mean) / (std + 1e-6)`, denorm: `x * (std + 1e-6) + mean`
- **Quantile** (Pi0.5): uses `q01` and `q99` percentiles for scaling

**Reloading pre-trained stats:** Some configs reload normalization stats from the
base model checkpoint (e.g., `gs://openpi-assets/checkpoints/pi0_base/assets`).
This helps when your dataset is small. See [norm_stats.md](../../docs/norm_stats.md)
for details.

## Step 2: Train

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> \
  --exp-name=<name> --overwrite
```

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` lets JAX use 90% of GPU memory
- `--overwrite` replaces existing checkpoints for the same experiment name
- Checkpoints are saved to `checkpoints/<config>/<exp-name>/<step>/`
- Training progress is logged to Weights & Biases

## Step 3: Monitor on W&B

Training automatically logs to [Weights & Biases](https://wandb.ai/). Key metrics:
- `train/loss` -- should decrease steadily
- Watch for overfitting if loss plateaus very early (try fewer steps)

## Step 4: Download Checkpoint (from HPC)

Only `params/` and `assets/` are needed for serving (skip optimizer/training state):

```bash
mkdir -p ./checkpoints/<config>/<exp_name>/<step>
rsync -avhP \
  --include='params' \
  --include='params/**' \
  --include='assets' \
  --include='assets/**' \
  --include='_CHECKPOINT_METADATA' \
  --exclude='*' \
  <user>@<hpc>:openpi/checkpoints/<config>/<exp_name>/<step>/ \
  ./checkpoints/<config>/<exp_name>/<step>/
```

Example (LoRA checkpoint from UT HPC):

```bash
mkdir -p ./checkpoints/pi05_ur5_lora/ur5_blueblock_box_10-2/260
rsync -avhP \
  --include='params' \
  --include='params/**' \
  --include='assets' \
  --include='assets/**' \
  --include='_CHECKPOINT_METADATA' \
  --exclude='*' \
  lenardspatriks@rocket.hpc.ut.ee:openpi/checkpoints/pi05_ur5_lora/ur5_blueblock_box_10-2/260/ \
  ./checkpoints/pi05_ur5_lora/ur5_blueblock_box_10-2/260/
```

## HPC Training (Slurm + Singularity)

For training on HPC clusters, see [`ur5/scripts/hpc/README_singularity_slurm.md`](../scripts/hpc/README_singularity_slurm.md).

Quick summary:
```bash
# Compute norm stats
sbatch ur5/scripts/hpc/slurm_norm_stats.sh

# Train
sbatch ur5/scripts/hpc/slurm_train.sh
```

## Building FFmpeg 7 on HPC

If your HPC system doesn't have FFmpeg 7 (required by PyAV), see the build
instructions in [`ur5/scripts/README.md`](../scripts/README.md#building-ffmpeg-7-on-hpc).

## Lessons Learned

- **Norm stats magnitude**: dataset delta actions have std ~0.003 while pretrained
  stats have std ~0.3 (50x difference). This can cause the model to output tiny or
  huge actions after denormalization. Check your `norm_stats.json` values.
- **Absolute vs delta actions**: absolute actions with `use_delta_action_transform=True`
  (converts to delta for training, back to absolute for inference) works better than
  raw delta datasets.
- **Training steps**: 300-500 steps works well for 10-episode datasets. More steps
  can lead to overfitting.
- **Pi0.5 vs Pi0**: Pi0.5 generalizes better to unseen positions and objects.

See [`ur5/docs/experiments.md`](experiments.md) for the full experiment log.
