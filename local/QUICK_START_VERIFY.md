# Quick Start: Enable Norm Stats Verification in Bridge

The bridge code now includes built-in norm_stats verification. To enable it:

## Enable Verification

Set the environment variable before running the bridge:

```bash
export VERIFY_NORM_STATS=1
python local/pi0_bridge_ur5_headless.py
```

## What It Does

When enabled, the bridge will:
1. Load the same norm_stats that the server uses (from `gs://openpi-assets/checkpoints/pi05_base/assets/ur5e`)
2. Check if actions are in normalized range [-1, 1]
3. Manually apply unnormalization to see what the expected values should be
4. Compare expected vs actual values and warn if there's a mismatch

## Example Output

When verification is enabled, you'll see output like:

```
✅ Loaded norm_stats for verification from: /root/.cache/openpi/...
✅ Norm stats verification enabled. Model type: pi05, Using quantile normalization.

=== Action shapes and types ===
Action value range: [-0.8, 0.9], max_abs: 0.9
ℹ️  Actions appear NORMALIZED (in [-1, 1] range). Server should unnormalize these.

Action joints (deg): [13.23°, 1.86°, -19.46°, 77.81°, -10.56°, -5.11°]
  [NORM_STATS CHECK] Action appears normalized (max_abs=0.900)
  [NORM_STATS CHECK] Expected unnormalized range: [-8.59°, 8.59°]
  [NORM_STATS CHECK] This action would unnormalize to max: 8.59°
  ⚠️  [NORM_STATS CHECK] WARNING: Even after unnormalization, this would be 77.81°!
  [NORM_STATS CHECK] This suggests norm_stats may have very large q99 values.
```

## Customize Assets Directory

If you need to use a different assets directory or asset_id:

```bash
export VERIFY_NORM_STATS=1
export NORM_STATS_ASSETS_DIR="gs://openpi-assets/checkpoints/pi05_base/assets"
export NORM_STATS_ASSET_ID="ur5e"
python local/pi0_bridge_ur5_headless.py
```

## Disable Verification

To disable (default):

```bash
export VERIFY_NORM_STATS=0
# or simply don't set it
python local/pi0_bridge_ur5_headless.py
```

## Troubleshooting

If you see:
```
⚠️  Warning: Could not load norm_stats for verification: ...
```

This means:
- The norm_stats couldn't be downloaded/loaded
- The bridge will continue without verification
- Check your network connection and GCS access

The bridge will still work normally, just without the verification checks.
