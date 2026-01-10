# Launch Guide: Running π₀ Policy on UR5 Robot

This guide explains how to run the π₀ policy on a UR5 robot arm using Docker. The setup uses a single container that runs both the policy server and the robot control bridge.

## Prerequisites

- UR5 robot arm (powered on, Remote Control mode, RTDE script running)
- RealSense camera connected via USB
- NVIDIA GPU with Docker support
- Network access to robot IP

## Step 1: Get Camera Serial Number

Create a separate virtual environment for test_setup scripts (to avoid installing hardware packages in the main project venv):

```bash
cd /home/ims/openpi
python3 -m venv test_setup/.venv
source test_setup/.venv/bin/activate
pip install pyrealsense2 ur-rtde numpy opencv-python
```

Then list connected cameras:

```bash
python test_setup/rs_list.py
```

Note the serial number for `RS_BASE`.

**Note:** You can deactivate the test_setup venv after getting the serial number:
```bash
deactivate
```

## Step 2: Build Docker Image

```bash
cd /home/ims/openpi
docker build -t openpi_robot -f scripts/docker/serve_policy_robot.Dockerfile .
```

## Step 3: Run Container

Detect video devices:

```bash
RUN_DEVICES="$(for d in /dev/video*; do [ -e "$d" ] && printf -- '--device=%s ' "$d"; done)"
```

Remove existing container:

```bash
docker rm -f openpi-robot 2>/dev/null || true
```

Run container (override RS_BASE with your camera serial):

**For display (to see camera feeds live):**

First, enable X11 access for Docker (run this once per terminal session, or add to your `~/.bashrc`):
```bash
xhost +local:docker
```

Then run the container:
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
  -e PROMPT="bus the table" \
  -e INFER_PERIOD=0.5 \
  -e HORIZON_STEPS=10 \
  -e HOLD_PER_STEP=0.05 \
  -e MAX_STEP_DEG=0.5 \
  -e DT=0.01 \
  -e VEL=0.08 \
  -e ACC=0.15 \
  -e LOOKAHEAD=0.12 \
  -e GAIN=300 \
  openpi_robot
```

**To enable norm_stats verification** (helps diagnose large action values), add:
```bash
  -e VERIFY_NORM_STATS=1 \
```

**Without display (headless mode):**
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
  -e PROMPT="pick up the bottle with orange cap" \
  -e INFER_PERIOD=0.15 \
  -e HORIZON_STEPS=1 \
  -e HOLD_PER_STEP=0.15 \
  -e SHOW_IMAGES=0 \
  -e VERIFY_NORM_STATS=1 \
  openpi_robot
```

**Camera configuration:**
- `RS_BASE`: Serial number for the over-the-shoulder (exterior) camera (required)
- `RS_WRIST`: Serial number for the wrist-mounted camera (optional, if not set, base camera will be used for both views)

**Display options:**
- `SHOW_IMAGES=1` (default): Show live camera feeds in a window. Requires X11 forwarding (see above).
- `SHOW_IMAGES=0`: Disable display (useful for headless operation).

**Enabling X11 Access:**

1. **One-time per terminal session:**
   ```bash
   xhost +local:docker
   ```

2. **Make it permanent (add to `~/.bashrc`):**
   ```bash
   echo 'xhost +local:docker > /dev/null 2>&1' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify X11 is working:**
   ```bash
   echo $DISPLAY  # Should show something like :0 or :1
   ```

**Note:** `xhost +local:docker` allows all local Docker containers to access your X server. For more security, you can use `xhost +SI:localuser:$(whoami)` instead, but this may require additional configuration.

Optional: Override other defaults (UR_IP, PROMPT, etc.) with additional `-e` flags.

## Example Launch Script

Save as `launch_robot.sh`:

```bash
#!/bin/bash
cd /home/ims/openpi

docker build -t openpi_robot -f scripts/docker/serve_policy_robot.Dockerfile .

RUN_DEVICES="$(for d in /dev/video*; do [ -e "$d" ] && printf -- '--device=%s ' "$d"; done)"

docker rm -f openpi-robot 2>/dev/null || true


```

Run with: `chmod +x launch_robot.sh && ./launch_robot.sh`
docker run --rm -it --gpus=all --network=host --device=/dev/bus/usb:/dev/bus/usb $RUN_DEVICES --group-add video -v "$PWD":/app --name openpi-robot -e RS_BASE=137322074310 -e RS_WRIST=137322075008 -e PROMPT="pick up the the red mug 10cm from the table" -e INFER_PERIOD=0.3 -e HORIZON_STEPS=3 -e HOLD_PER_STEP=0.1 openpi_robot