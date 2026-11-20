# UR5 Quickstart

Record a dataset, train a policy, and evaluate it on the robot. To use the study dataset and configuration, start with [the study run](training.md#study-run).

## Prerequisites

- Recording and evaluation: UR5e in Remote Control mode, RTDE connectivity, and a Robotiq Hand-E with its URCap socket available on port 63352. Use two USB 3.0 Intel RealSense D400 cameras for the study setup.
- Training and conversion: Linux, Git LFS, [uv](https://docs.astral.sh/uv/), and access to the base weights in `gs://openpi-assets`. Training requires an NVIDIA GPU.
- Docker deployment: Docker with NVIDIA Container Toolkit on the robot-connected host.

## 1. Setup

```bash
git clone --recurse-submodules https://github.com/LPSlv/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked
```

Use a separate environment for robot-side recording, replay, and camera tests. Run these commands from the repository root on the robot-connected host.

```bash
python3 -m venv .venv-robot
source .venv-robot/bin/activate
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

## 2. Verify hardware

```bash
python ur5/test/rs_list.py        # list connected cameras
# Configure SERIAL_BASE and SERIAL_WRIST in ur5/test/camera_test.py first.
python ur5/test/camera_test.py    # live preview
```

## 3. Record a dataset

Both recording scripts can command motion to a configured start pose. Verify the
start pose and replay path for your cell before running them. Set `UR_IP`,
`RS_BASE` (shoulder), and `RS_WRIST` (wrist) to your hardware values, and use a
new episode ID for each demonstration.

```bash
EPISODE=episode_001
PROMPT="Pick up the blue block and place it in the cardboard box"

python -m ur5.scripts.ur5_record_freedrive_waypoints \
  --ur-ip "${UR_IP:?Set the robot IP}" \
  --prompt "$PROMPT" --out-dir waypoints --episode-id "$EPISODE"

python -m ur5.scripts.ur5_replay_and_record_raw \
  --ur-ip "$UR_IP" --waypoints-path "waypoints/$EPISODE/waypoints.json" \
  --rs-base-serial "${RS_BASE:?Set the shoulder camera serial}" \
  --rs-wrist-serial "${RS_WRIST:?Set the wrist camera serial}" \
  --out-dir raw_episodes --episode-id "$EPISODE" --fps 10
```

Replay inherits the instruction from the waypoint file. Check that the gripper
initializes successfully during both recording and replay. Check that each replay
directory contains `meta.json`, `steps.jsonl`, and both camera views. The
converter expects 256 × 256 images and forward-looking absolute joint and gripper
targets from replay. Transfer the complete `raw_episodes/` directory if you train
on a separate host.

```bash
deactivate  # if the robot environment is active
DATASET_ID=local/ur5_dataset
uv run python ur5/scripts/convert_ur5_raw_to_lerobot.py \
  --raw-dir raw_episodes --repo-id "$DATASET_ID" --fps 10 --no-push-to-hub
```

Run conversion and training with the same LeRobot cache, or copy the converted
dataset to the same dataset ID in the training host's cache. The converter
replaces the existing local dataset under the specified ID. To publish it
publicly, choose a Hugging Face dataset ID you control, authenticate with a
write token, and replace `--no-push-to-hub` with `--push-to-hub`.

## 4. Train

In [`src/openpi/training/config.py`](../../src/openpi/training/config.py), set
both `repo_id` and `assets=AssetsConfig(asset_id=...)` in the `pi05_ur5` entry
to the dataset ID used in conversion. The preset points to the 51-episode
study dataset. The stats script takes a config name and reads its dataset
settings directly.

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_ur5
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5 \
  --exp-name=my_exp --no-wandb-enabled
```

Select a saved step under `checkpoints/pi05_ur5/my_exp/`. If moving it to
another machine, copy the checkpoint's `params/` and `assets/` directories
using the [checkpoint transfer instructions](training.md#3-download-a-checkpoint-from-a-remote-machine).

## 5. Deploy and evaluate

Follow [deployment.md](deployment.md) with the same config name and the saved
checkpoint directory. Use the [evaluation procedure](deployment.md#evaluate)
to compare checkpoints and execution horizons on your setup.

## CLI config overrides

The training CLI supports overrides for a single run. Keep the same dataset and stats settings across stats computation and training.

```bash
uv run scripts/train.py pi05_ur5 \
  --exp-name=my_longer_run \
  --num-train-steps=300 --no-wandb-enabled
```

| Flag | Purpose |
|------|---------|
| `--data.repo-id=X` | override the training dataset; pair it with the matching stats asset |
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
| Start position |   | `(-90, -40, -140, -50, 90, 0)` deg |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera not found | check USB 3.0, run `ur5/test/rs_list.py` |
| Robot connection refused | Remote Control mode on, `ping 192.10.0.11`, RTDE enabled |
| Missing Python modules in robot scripts | Activate `.venv-robot` and install the robot dependencies from step 1 |
| Gripper not responding | Robotiq URCap running, port 63352 open |
| Docker camera access | pass `--device=/dev/bus/usb:/dev/bus/usb --group-add video` |
| No display in Docker | `xhost +local:docker` on host, pass `-e DISPLAY=$DISPLAY` |
| No cameras during replay diagnostics | `--fake-cam` uses synthetic images. Keep these episodes out of training and evaluation. |
