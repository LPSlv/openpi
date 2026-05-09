# Training

UR5 training configs and the commands to run them. For end-to-end setup
(including data recording) see [quickstart.md](quickstart.md).

## Available configs

All defined in `src/openpi/training/config.py`.

| Config | Model | Base checkpoint | Notes |
|--------|-------|-----------------|-------|
| `pi0_ur5` | Pi0 | `pi0_base` | full fine-tune, z-score |
| `pi05_ur5` | Pi0.5 | `pi05_base` | full fine-tune, fresh stats |
| `pi05_ur5_blueblock10` | Pi0.5 | `pi05_base` | final paper config (500 steps, reuses pretrained ur5e stats) |
| `pi05_ur5_lora` | Pi0.5 | `pi05_base` | LoRA fine-tune, vision frozen |
| `pi0_fast_ur5` | Pi0-FAST | `pi0_fast_base` | discrete action tokens |

## 1. Compute norm stats

```bash
uv run scripts/compute_norm_stats.py --config-name <config>
```

Writes `norm_stats.json` to `assets/<config>/<asset_id>/`.

z-score is `(x - mean) / (std + 1e-6)`; quantile uses `q01`/`q99`. Pi0/Pi0.5 use
z-score by default, Pi0-FAST forces quantile (FAST tokenizer needs `[-1, 1]`).
For when to reuse pretrained stats vs recomputing, see
[`docs/norm_stats.md`](../../docs/norm_stats.md).

## 2. Train

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> \
  --exp-name=<name> \
  --overwrite
```

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` lets JAX use 90% of GPU memory
- `--overwrite` replaces existing checkpoints for the same `--exp-name`
- checkpoints land in `checkpoints/<config>/<exp-name>/<step>/`
- progress logs to Weights & Biases (`train/loss` is the headline metric)

For CLI overrides (dataset, batch size, num steps, etc.) see the
[CLI overrides table in quickstart](quickstart.md#cli-config-overrides).

## 3. Download a checkpoint from a remote machine

Only `params/` and `assets/` are needed for serving; the optimizer state can be
skipped:

```bash
mkdir -p ./checkpoints/<config>/<exp_name>/<step>
rsync -avhP \
  --include='params' \
  --include='params/**' \
  --include='assets' \
  --include='assets/**' \
  --include='_CHECKPOINT_METADATA' \
  --exclude='*' \
  <user>@<host>:openpi/checkpoints/<config>/<exp_name>/<step>/ \
  ./checkpoints/<config>/<exp_name>/<step>/
```

The step number is `num_train_steps - 1` (0-indexed): 500 steps -> step 499.

## Lessons learned

- **Norm-stats magnitude.** Dataset delta actions have std ~0.003 while the
  pretrained ur5e stats have std ~0.3 (~50x difference). This can collapse the
  model output after denormalization. Sanity-check your `norm_stats.json`.
- **Absolute vs delta actions.** Recording absolute targets and letting the
  pipeline convert to deltas (`use_delta_action_transform=True`) works better
  than recording raw deltas.
- **Training length.** 300-500 steps is a good range for 10-episode datasets;
  longer runs overfit fast.
- **Pi0.5 vs Pi0.** Pi0.5 generalizes better to unseen positions and objects.

See [experiments.md](experiments.md) for the full lab notebook.
