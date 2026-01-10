# Enabling Norm Stats Verification in Docker

Since you're running the bridge inside Docker, you need to pass the `VERIFY_NORM_STATS` environment variable to the container.

## Quick Add

Just add this flag to your `docker run` command:

```bash
-e VERIFY_NORM_STATS=1 \
```

## Complete Example

**With display:**
```bash
docker run --rm -it \
  --gpus=all \
  --network=host \
  --device=/dev/bus/usb:/dev/bus/usb \
  $RUN_DEVICES \
  --group-add video \
  -v "$PWD":/app \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY=$DISPLAY \
  --name openpi-robot \
  -e RS_BASE=137322074310 \
  -e RS_WRIST=137322075008 \
  -e PROMPT="your prompt here" \
  -e VERIFY_NORM_STATS=1 \
  openpi_robot
```

**Headless (no display):**
```bash
docker run --rm -it \
  --gpus=all \
  --network=host \
  --device=/dev/bus/usb:/dev/bus/usb \
  $RUN_DEVICES \
  --group-add video \
  -v "$PWD":/app \
  --name openpi-robot \
  -e RS_BASE=137322074310 \
  -e RS_WRIST=137322075008 \
  -e PROMPT="your prompt here" \
  -e SHOW_IMAGES=0 \
  -e VERIFY_NORM_STATS=1 \
  openpi_robot
```

## What You'll See

When `VERIFY_NORM_STATS=1` is enabled, you'll see output like:

```
✅ Loaded norm_stats for verification from: /root/.cache/openpi/...
✅ Norm stats verification enabled. Model type: pi05, Using quantile normalization.

[NORM_STATS CHECK] Action appears normalized (max_abs=0.236)
[NORM_STATS CHECK] Expected unnormalized range: [-X.XX°, Y.YY°]
[NORM_STATS CHECK] This action would unnormalize to max: Z.ZZ°
```

This will help you understand:
- What the actual q01 and q99 values are in the norm_stats
- Whether the large action values are expected based on norm_stats
- If there's a mismatch between what the server is doing and what norm_stats say

## Customize Assets Directory (if needed)

If you need to use different norm_stats:

```bash
-e VERIFY_NORM_STATS=1 \
-e NORM_STATS_ASSETS_DIR="gs://openpi-assets/checkpoints/pi05_base/assets" \
-e NORM_STATS_ASSET_ID="ur5e" \
```

## Disable Verification

To disable (default behavior), simply don't include the `-e VERIFY_NORM_STATS=1` flag, or set it to 0:

```bash
-e VERIFY_NORM_STATS=0 \
```

## Troubleshooting

If you see:
```
⚠️  Warning: Could not load norm_stats for verification: ...
```

This means:
- The norm_stats couldn't be downloaded/loaded inside the container
- The bridge will continue without verification
- Check that the container has network access to download from GCS

The bridge will still work normally, just without the verification checks.
