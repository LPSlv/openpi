# How to Verify Action Unnormalization

## Problem
If actions from the policy server are unusually large (e.g., 77° deltas when expecting < 0.2°), it may indicate an unnormalization issue.

## Quick Check in Bridge Code

The bridge code now automatically checks if actions are normalized or unnormalized:

```
=== Action shapes and types ===
Raw actions from server: shape=(15, 7), dtype=float32
Action value range: [-0.8, 0.9], max_abs: 0.9
ℹ️  Actions appear NORMALIZED (in [-1, 1] range). Server should unnormalize these.
```

vs.

```
Action value range: [-1.3, 1.4], max_abs: 1.4
ℹ️  Actions appear UNNORMALIZED (outside [-1, 1] range). 
    Server may have already applied unnormalization, or there's an issue.
```

## Detailed Verification

### Step 1: Run the Verification Script

```bash
cd /home/ims/openpi
python local/check_unnormalization.py
```

This will:
- Load the UR5 norm_stats from the server's assets directory
- Show the mean, std, q01, q99 values for actions
- Show what a normalized action [-1, 1] would unnormalize to
- Verify if unnormalized values are in expected ranges

### Step 2: Check Server Logs

Look for these messages in the policy server logs:

```
INFO:root:Loaded norm stats from gs://openpi-assets/checkpoints/pi05_base/assets/ur5e
```

If you see:
```
INFO:root:Norm stats not found in /app/assets/pi05_ur5/ur5e, skipping.
```

Then the server is trying to load from a local path first, then falling back to GCS. This is fine as long as it eventually loads from GCS.

### Step 3: Understand the Unnormalization Formula

For **pi05** models (which use quantile normalization):
```
unnormalized = (normalized + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
```

This maps:
- Normalized value `-1.0` → `q01` (1st percentile)
- Normalized value `+1.0` → `q99` (99th percentile)
- Normalized value `0.0` → `(q01 + q99) / 2` (middle)

For **pi0** models (which use z-score normalization):
```
unnormalized = normalized * std + mean
```

### Step 4: Expected Ranges After Unnormalization

**Delta Mode:**
- Unnormalized deltas should typically be **< 0.1 rad (6°)** per step
- If you see deltas > 5°, something is likely wrong

**Absolute Mode:**
- Unnormalized positions should be in joint angle range: **[-π, π] rad ([-180°, 180°])**
- If values exceed ±180°, something is wrong

### Step 5: Common Issues

1. **Wrong asset_id**: Server loads norm_stats for wrong robot
   - Check server logs for which asset_id was loaded
   - Should be `"ur5e"` for UR5

2. **Wrong model type**: pi0 vs pi05 use different normalization
   - pi05 uses quantile normalization (q01, q99)
   - pi0 uses z-score normalization (mean, std)
   - Check server metadata: `model_type: 'pi05'` or `'pi0'`

3. **Norm stats mismatch**: Stats don't match the action space
   - Check if norm_stats have correct number of dimensions (7 for UR5: 6 joints + 1 gripper)
   - Check if q01/q99 or mean/std values are reasonable

4. **Unnormalize transform not applied**: Server isn't applying Unnormalize transform
   - Check `src/openpi/policies/policy_config.py` - Unnormalize should be in `output_transforms`
   - Check server logs for transform application

## Example Output from Verification Script

```
Loading UR5 norm_stats...
Loaded norm_stats from: /root/.cache/openpi/openpi-assets/checkpoints/pi05_base/assets/ur5e

=== Norm Stats Inspection ===

actions:
  mean shape: (7,), mean: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  std shape: (7,), std: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05]
  q01 shape: (7,), q01: [-0.15, -0.12, -0.18, -0.10, -0.14, -0.08, -0.02]
  q99 shape: (7,), q99: [0.15, 0.12, 0.18, 0.10, 0.14, 0.08, 0.02]
  Range: [-0.15, 0.15] rad
  Range (deg): [-8.59°, 8.59°]

=== Action Unnormalization Verification ===
Input action (normalized): [-1.0, -0.5, 0.0, 0.5, 1.0, 0.0]
Unnormalized action (deg): [-8.59°, -4.30°, 0.00°, 4.30°, 8.59°, 0.00°]
✅ GOOD: Delta is very small (< 0.1°)
```

## Manual Verification

If you want to manually verify, you can:

1. **Check the norm_stats file directly:**
   ```python
   from openpi.shared import normalize
   import pathlib
   
   stats_path = pathlib.Path("/root/.cache/openpi/openpi-assets/checkpoints/pi05_base/assets/ur5e/norm_stats.json")
   norm_stats = normalize.deserialize_json(stats_path.read_text())
   
   print("Actions stats:", norm_stats["actions"])
   ```

2. **Manually apply unnormalization:**
   ```python
   import numpy as np
   
   # Normalized action from model (in [-1, 1])
   normalized = np.array([0.5, 0.3, -0.2, 0.1, 0.0, -0.1])
   
   # Get stats
   stats = norm_stats["actions"]
   
   # Apply quantile unnormalization (for pi05)
   unnormalized = (normalized + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
   
   print(f"Normalized: {normalized}")
   print(f"Unnormalized (rad): {unnormalized}")
   print(f"Unnormalized (deg): {np.degrees(unnormalized)}")
   ```

## Next Steps

If actions are still unusually large after verification:

1. Check that the server is using the correct config (`pi05_ur5`)
2. Verify the server is loading norm_stats with `asset_id="ur5e"`
3. Check that `Unnormalize` transform is in the policy's `output_transforms`
4. Consider that the policy might be outputting absolute positions instead of deltas
5. The rate limiting in the bridge code will keep the robot safe, but investigate the root cause
