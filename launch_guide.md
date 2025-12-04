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

```bash
docker run --rm -it \
  --gpus=all \
  --network=host \
  --device=/dev/bus/usb:/dev/bus/usb \
  $RUN_DEVICES \
  --group-add video \
  -v "$PWD":/app \
  --name openpi-robot \
  -e RS_BASE=137322075008 \
  -e RS_WRIST=137322074310 \
  -e PROMPT="pick up the bottle with orange cap" \
  -e INFER_PERIOD=0.15 \
  -e HORIZON_STEPS=1 \
  -e HOLD_PER_STEP=0.15 \
  openpi_robot
```

**Camera configuration:**
- `RS_BASE`: Serial number for the over-the-shoulder (exterior) camera (required)
- `RS_WRIST`: Serial number for the wrist-mounted camera (optional, if not set, base camera will be used for both views)

Optional: Override other defaults (UR_IP, PROMPT, etc.) with additional `-e` flags.

## Example Launch Script

Save as `launch_robot.sh`:

```bash
#!/bin/bash
cd /home/ims/openpi

docker build -t openpi_robot -f scripts/docker/serve_policy_robot.Dockerfile .

RUN_DEVICES="$(for d in /dev/video*; do [ -e "$d" ] && printf -- '--device=%s ' "$d"; done)"

docker rm -f openpi-robot 2>/dev/null || true

docker run --rm -it \
  --gpus=all \
  --network=host \
  --device=/dev/bus/usb:/dev/bus/usb \
  $RUN_DEVICES \
  --group-add video \
  -v "$PWD":/app \
  --name openpi-robot \
  -e RS_BASE=137322075008 \
  -e RS_WRIST=137322074310 \
  -e PROMPT="pick up the bottle with orange cap" \
  -e INFER_PERIOD=0.15 \
  -e HORIZON_STEPS=1 \
  -e HOLD_PER_STEP=0.15 \
  openpi_robot
```

Run with: `chmod +x launch_robot.sh && ./launch_robot.sh`
