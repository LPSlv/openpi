# UR5 Quickstart Guide

## Prerequisites

**Hardware:**
- UR5e robot arm, powered on, in Remote Control mode
- 1-2 Intel RealSense D400-series cameras (USB 3.0)
- Linux workstation with NVIDIA GPU (RTX 4090+ for inference, A100/H100 for training)

**Software:**
- Docker + nvidia-container-toolkit
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## 1. Clone and Set Up

```bash
git clone --recurse-submodules <your-fork-url>
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 2. Verify Hardware

Activate the environment and install hardware-specific packages:

```bash
source .venv/bin/activate
uv pip install pyrealsense2 ur-rtde numpy opencv-python
```

List connected cameras:
```bash
python ur5/test/rs_list.py
```

Verify the cameras stream:
```bash
python ur5/test/camera_test.py
```

## 3. Record Your First Dataset

See [data_pipeline.md](data_pipeline.md) for the full recording workflow:

```bash
# Step 1: Record waypoints in freedrive
uv run python ur5/scripts/ur5_record_freedrive_waypoints.py \
  --ur_ip 192.10.0.11 --prompt "pick up the block" --out_dir raw_episodes

# Step 2: Replay with cameras
uv run python ur5/scripts/ur5_replay_and_record_raw.py \
  --ur_ip 192.10.0.11 \
  --waypoints_path raw_episodes/<episode_id>/waypoints.json \
  --rs_base_serial <SERIAL> --prompt "pick up the block" \
  --out_dir raw_episodes --fps 10

# Step 3: Convert to LeRobot
uv run python ur5/scripts/convert_ur5_raw_to_lerobot.py \
  --raw_dir raw_episodes --repo_id <hf_username>/ur5_dataset --fps 10
```

## 4. Train

See [training.md](training.md) for full details:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_ur5
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5 --exp-name=my_exp --overwrite
```

## 5. Deploy and Run

See [deployment.md](deployment.md) for Docker setup:

```bash
# Build image
docker build -t openpi_robot -f ur5/docker/serve_policy_robot.Dockerfile .

# Run (override env vars as needed)
docker run --rm -it --gpus=all --network=host \
  --device=/dev/bus/usb:/dev/bus/usb --group-add video \
  -v "$PWD":/app \
  -e RS_BASE=<camera_serial> \
  -e PROMPT="pick up the block" \
  openpi_robot
```

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UR_IP` | `192.10.0.11` | Robot IP address |
| `RS_BASE` | *(required)* | Base camera serial number |
| `RS_WRIST` | *(optional)* | Wrist camera serial number |
| `PROMPT` | `"pick up the grey shaker bottle"` | Language instruction |
| `ACTION_MODE` | `delta` | `"delta"` or `"absolute"` |

See [deployment.md](deployment.md) for the full environment variable reference.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera not found | Check USB 3.0 connection, run `ur5/test/rs_list.py` |
| Robot connection refused | Ensure Remote Control mode, `ping 192.10.0.11`, RTDE enabled |
| Missing Python modules | `uv pip install pyrealsense2 ur-rtde numpy opencv-python` |
| Gripper not responding | Check Robotiq URCap is running, port 63352 open |
| Docker camera access | Pass `--device=/dev/bus/usb:/dev/bus/usb --group-add video` |
| No display in Docker | Run `xhost +local:docker` on host, pass `-e DISPLAY=$DISPLAY` |
