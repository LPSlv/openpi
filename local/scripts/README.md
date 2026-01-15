# UR5 Scripts

Scripts for recording and replaying UR5 robot trajectories.

## Quick Setup

```bash
# Minimal setup (recommended)
python3 -m venv venv_ur5
source venv_ur5/bin/activate
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

## Scripts

### Record Waypoints

Records a human-guided trajectory in freedrive mode.

```bash
python local/scripts/ur5_record_freedrive_waypoints.py
```

**Usage:**
1. Robot moves to start position (-90, -70, -120, -80, 90, 0 degrees)
2. Teach mode enables automatically
3. Physically guide the arm
4. Press Enter to finish
5. Waypoints saved to `raw_episodes/<episode_id>/waypoints.json`

### Replay Waypoints

Replays a trajectory and records images + proprioceptive data.

```bash
python openpi/local/scripts/ur5_replay_and_record_raw.py \
  --waypoints_path raw_episodes/ur5_freedrive_20260113_140635/waypoints.json
```


## Default Values

- **Robot IP**: `192.10.0.11`
- **Output directory**: `raw_episodes`
- **Base camera**: `137322074310`
- **Wrist camera**: `137322075008`
- **Start position**: `(-90, -70, -120, -80, 90, 0)` degrees
- **Replay speed**: `0.2` rad/s (slow for safety)

## Troubleshooting

**Missing modules:**
```bash
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

**No cameras available:**
Use `--fake_cam` flag (generates black placeholder images)

**Robot connection:**
- Ensure robot is in Remote Control mode
- Check network: `ping 192.10.0.11`
- Verify RTDE is enabled on controller

**Gripper:**
Disabled by default. Enable with `--use_gripper` (may not work depending on RTDE version).

For all available options, run scripts with `--help`.
