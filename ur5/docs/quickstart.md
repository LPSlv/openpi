# UR5 Quickstart

End-to-end workflow: record a dataset, train, deploy on the robot.

## Prerequisites

- UR5e in Remote Control mode
- 1-2 Intel RealSense D400 cameras (USB 3.0)
- Linux workstation with NVIDIA GPU
- Docker + nvidia-container-toolkit
- [uv](https://docs.astral.sh/uv/)

## 1. Setup

```bash
git clone --recurse-submodules <your-fork-url>
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
uv pip install pyrealsense2 ur-rtde numpy opencv-python
```

For tasks that don't need the ML stack (recording, replay, hardware tests):

```bash
python3 -m venv .venv-robot
source .venv-robot/bin/activate
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

## 2. Verify hardware

```bash
python ur5/test/rs_list.py        # list connected cameras
python ur5/test/camera_test.py    # live preview
```

## 3. Record a dataset

```bash
# step 1: record sparse waypoints in freedrive
python ur5/scripts/ur5_record_freedrive_waypoints.py \
  --ur_ip 192.10.0.11 \
  --prompt "pick up the block" \
  --out_dir raw_episodes

# step 2: replay with cameras to capture the full episode
python ur5/scripts/ur5_replay_and_record_raw.py \
  --ur_ip 192.10.0.11 \
  --waypoints_path raw_episodes/<episode_id>/waypoints.json \
  --rs_base_serial <SERIAL_BASE> \
  --rs_wrist_serial <SERIAL_WRIST> \
  --prompt "pick up the block" \
  --out_dir raw_episodes \
  --fps 10
```

Repeat steps 1-2 for each episode. Convert and (optionally) push to HuggingFace:

```bash
export HF_TOKEN=your_token_here

uv run python ur5/scripts/convert_ur5_raw_to_lerobot.py \
  --raw_dir raw_episodes \
  --repo_id <hf_username>/ur5_dataset \
  --fps 10
# add --no-push-to-hub to keep it local
```

## 4. Train

```bash
# compute norm stats first
uv run scripts/compute_norm_stats.py --config-name pi05_ur5

# train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5 \
  --exp-name=my_exp --overwrite
```

Checkpoint output: `checkpoints/pi05_ur5/my_exp/<step>/`. See [training.md](training.md) for available configs and lessons learned.

## 5. Deploy

```bash
# build the inference image
docker build -t openpi_robot -f ur5/docker/serve_policy_robot.Dockerfile .

# run policy server + robot bridge
docker run --rm -it --gpus=all --network=host \
  --device=/dev/bus/usb:/dev/bus/usb --group-add video \
  -v "$PWD":/app \
  -e RS_BASE=<SERIAL_BASE> \
  -e RS_WRIST=<SERIAL_WRIST> \
  -e PROMPT="pick up the block" \
  openpi_robot
```

See [deployment.md](deployment.md) for the full Docker + bridge setup and the env-var reference.

## CLI config overrides

The training CLI overrides any field on the train config without editing `config.py`:

```bash
uv run scripts/train.py pi05_ur5 \
  --exp-name=my_exp \
  --data.repo-id=<hf_username>/ur5_new_dataset \
  --num-train-steps=300 \
  --overwrite
```

| Flag | Purpose |
|------|---------|
| `--data.repo-id=X` | use a different HF dataset |
| `--num-train-steps=N` | training length |
| `--exp-name=X` | experiment name (becomes part of the checkpoint path) |
| `--batch-size=N` | override batch size |
| `--save-interval=N` | checkpoint cadence |
| `--overwrite` | overwrite an existing checkpoint dir |
| `--resume` | resume from the last checkpoint |

## Defaults

Hardware defaults live in [`ur5/defaults.py`](../defaults.py) and override via env vars:

| Setting | Env var | Default |
|---------|---------|---------|
| Robot IP | `UR_IP` | `192.10.0.11` |
| Output dir | `OUT_DIR` | `raw_episodes` |
| Base camera | `RS_BASE` | `137322074310` |
| Wrist camera | `RS_WRIST` | `137322075008` |
| Gripper port | `ROBOTIQ_PORT` | `63352` |
| Start position | — | `(-90, -40, -140, -50, 90, 0)` deg |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera not found | check USB 3.0, run `ur5/test/rs_list.py` |
| Robot connection refused | Remote Control mode on, `ping 192.10.0.11`, RTDE enabled |
| Missing Python modules | `uv pip install pyrealsense2 ur-rtde numpy opencv-python` |
| Gripper not responding | Robotiq URCap running, port 63352 open |
| Docker camera access | pass `--device=/dev/bus/usb:/dev/bus/usb --group-add video` |
| No display in Docker | `xhost +local:docker` on host, pass `-e DISPLAY=$DISPLAY` |
| No cameras during recording | use `--fake_cam` |
