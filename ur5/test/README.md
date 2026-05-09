# Test Setup Scripts

This directory contains scripts for testing and verifying the hardware setup for the openpi robot system.

## Scripts

- **`rs_list.py`** - List all connected RealSense cameras and display their serial numbers
- **`camera_test.py`** - Launch one or two RealSense RGB cameras with live preview
- **`ur_read_state.py`** - Read and display the current state of a UR robot arm
- **`ur_test_movement.py`** - Test UR robot movement with a safe base joint nudge

## Setup

### 1. Environment

**Full environment (if you also need inference/training):**

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
```

**Lightweight environment (hardware tests only):**

```bash
python3 -m venv .venv-robot
source .venv-robot/bin/activate
pip install numpy ur-rtde tyro opencv-python pyrealsense2
```

### 2. Configure Camera Serial Numbers

Before running camera scripts, get your camera serial numbers:

```bash
python ur5/test/rs_list.py
```

Then edit `camera_test.py` and update the serial numbers:

```python
SERIAL_BASE = "YOUR_BASE_CAMERA_SERIAL"    # over-the-shoulder (required)
SERIAL_WRIST = None                         # or set to wrist camera serial
```

### 3. Configure UR Robot IP

Set the `UR_IP` environment variable, or edit the default in `ur5/defaults.py`:

```bash
export UR_IP="192.10.0.11"
```

## Usage

### Test RealSense Cameras

```bash
# List connected cameras
python ur5/test/rs_list.py

# Test camera feed
python ur5/test/camera_test.py
# Press 'q' to quit
```

### Test UR Robot

```bash
# Read robot state (continuous monitoring)
python ur5/test/ur_read_state.py
# Press Ctrl+C to stop

# Test robot movement (safe base joint nudge)
python ur5/test/ur_test_movement.py
```

## Troubleshooting

- **Camera not found**: Make sure the camera is connected via USB and run `rs_list.py` to verify it's detected
- **Robot connection failed**: Check that the robot IP address is correct and the robot is powered on and in Remote Control mode
- **Import errors**: Make sure the virtual environment is activated and required packages are installed

