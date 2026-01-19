# UR5 Scripts

## Setup

```bash
python3 -m venv venv_ur5
source venv_ur5/bin/activate
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

## Scripts

### Record Waypoints

```bash
python openpi/local/scripts/ur5_record_freedrive_waypoints.py
```

1. Robot moves to start position
2. Teach mode enables automatically
3. Guide the arm
4. Press Enter to finish
5. Waypoints saved to `raw_episodes/<episode_id>/waypoints.json`

### Replay Waypoints

```bash
python openpi/local/scripts/ur5_replay_and_record_raw.py \
  --waypoints_path raw_episodes/ur5_freedrive_20260113_140635/waypoints.json
```

### Convert to LeRobot Format

**Hugging Face Authentication:**

Get token from https://huggingface.co/settings/tokens

**Option 1: Environment Variable**
```bash
export HF_TOKEN=your_token_here
```

Or add to `~/.bashrc`:
```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
```

**Option 2: CLI**
```bash
pip install huggingface-hub[cli]
huggingface-cli login
```

**Convert:**
```bash
uv run python openpi/local/scripts/convert_ur5_raw_to_lerobot.py \
  --raw_dir raw_episodes \
  --repo_id your_hf_username/ur5_freedrive \
  --fps 10

# Skip pushing to hub:
uv run python openpi/local/scripts/convert_ur5_raw_to_lerobot.py \
  --raw_dir raw_episodes \
  --repo_id your_hf_username/ur5_freedrive \
  --fps 10 \
  --no-push-to-hub
```

## Fine-Tuning

```bash
# Compute normalization stats
uv run scripts/compute_norm_stats.py --config-name pi05_ur5_low_mem_finetune

# Run training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5_low_mem_finetune --exp-name=my_experiment --overwrite

# Run inference
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_ur5_low_mem_finetune \
  --policy.dir=checkpoints/pi05_ur5_low_mem_finetune/my_experiment/20000
```

## Defaults

- Robot IP: `192.10.0.11`
- Output: `raw_episodes`
- Base camera: `137322074310`
- Wrist camera: `137322075008`
- Start position: `(-90, -70, -120, -80, 90, 0)` degrees
- Replay speed: `0.2` rad/s

## Troubleshooting

**Missing modules:**
```bash
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

**No cameras:** Use `--fake_cam` flag

**Robot connection:** Ensure Remote Control mode, check `ping 192.10.0.11`, verify RTDE enabled

**Gripper:** Enabled by default. Disable with `--no-use-gripper`

Run scripts with `--help` for all options.
