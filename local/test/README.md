# Test Setup Scripts

This directory contains scripts for testing and verifying the hardware setup for the openpi robot system.

## Scripts

- **`rs_list.py`** - List all connected RealSense cameras and display their serial numbers
- **`camera_test.py`** - Launch one or two RealSense RGB cameras with live preview
- **`ur_read_state.py`** - Read and display the current state of a UR robot arm
- **`ur_test_movement.py`** - Test UR robot movement with a safe base joint nudge

## Setup

### 1. Activate the Virtual Environment

First, make sure `uv` is in your PATH:

```bash
# Add to your ~/.bashrc for persistence, or run each time:
export PATH="$HOME/.local/bin:$PATH"

# Or source the env file:
source $HOME/.local/bin/env
```

Then activate the project's virtual environment:

```bash
cd openpi
source .venv/bin/activate
```

### 2. Install Required Packages

The test_setup scripts require additional hardware-specific packages:

```bash
# Install packages for RealSense cameras and UR robot
uv pip install pyrealsense2 ur-rtde
```

**Note:** If you haven't set up the main project environment yet, you may need to install Python development headers first:

```bash
sudo apt-get install python3-dev
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 3. Configure Camera Serial Numbers

Before running camera scripts, get your camera serial numbers:

```bash
python test_setup/rs_list.py
```

Then edit `camera_test.py` and update the serial numbers:

```python
SERIAL_BASE = "YOUR_BASE_CAMERA_SERIAL"    # over-the-shoulder (required)
SERIAL_WRIST = None                         # or set to wrist camera serial
```

### 4. Configure UR Robot IP

Edit `ur_read_state.py` and `ur_test_movement.py` to set your robot's IP address:

```python
UR_IP = "192.10.0.11"  # Change to your robot's IP
```

## Usage

### Test RealSense Cameras

```bash
# List connected cameras
python test_setup/rs_list.py

# Test camera feed
python test_setup/camera_test.py
# Press 'q' to quit
```

### Test UR Robot

```bash
# Read robot state (continuous monitoring)
python test_setup/ur_read_state.py
# Press Ctrl+C to stop

# Test robot movement (safe base joint nudge)
python test_setup/ur_test_movement.py
```

## Troubleshooting

- **Camera not found**: Make sure the camera is connected via USB and run `rs_list.py` to verify it's detected
- **Robot connection failed**: Check that the robot IP address is correct and the robot is powered on and in Remote Control mode
- **Import errors**: Make sure the virtual environment is activated and required packages are installed

