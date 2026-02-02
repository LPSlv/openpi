# Launch Guide: Running π₀ Policy on UR5 Robot

This guide explains how to run the π₀ policy on a UR5 robot arm using Docker. The setup uses a single container that runs both the policy server and the robot control bridge.

## Prerequisites

- UR5 robot arm (powered on, Remote Control mode, RTDE script running)
- RealSense camera connected via USB
- NVIDIA GPU with Docker support
- Network access to robot IP

## Step 1: Get Camera Serial Number

First, ensure the main project environment is set up with `uv`:

```bash
cd /home/ims/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install additional hardware-specific packages for camera and robot testing:

```bash
uv pip install pyrealsense2 ur-rtde numpy opencv-python
```

Then list connected cameras:

```bash
python local/test/rs_list.py
```

Note the serial number for `RS_BASE`.

**Note:** You can deactivate the venv after getting the serial number:
```bash
deactivate
```

## Step 2: Build Docker Image

```bash
cd /home/ims/openpi
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

First, run these **host** commands (once per terminal session) so the OpenCV preview window can open from inside Docker:
```bash
# If DISPLAY is empty, you won't get a window. On a local desktop session this is usually :0.
echo "DISPLAY=$DISPLAY"
export DISPLAY=${DISPLAY:-:0}

# Optional sanity check (should not error)
xdpyinfo -display "$DISPLAY" >/dev/null && echo "X11 OK on $DISPLAY"

# Allow local Docker containers to connect to your X server
xhost +local:docker

# Often needed because the container runs as root
xhost +SI:localuser:root
```

Then run the container:
```bash
docker run --rm -it \
  --gpus=all \
  --network=host \
  --ipc=host \
  --device=/dev/bus/usb:/dev/bus/usb \
  $RUN_DEVICES \
  --group-add video \
  -v "$PWD":/app \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e QT_QPA_PLATFORM=xcb \
  -e QT_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins \
  -e QT_QPA_PLATFORM_PLUGIN_PATH=/.venv/lib/python3.11/site-packages/cv2/qt/plugins/platforms \
  --name openpi-robot \
  -e RS_BASE=137322074310 \
  -e RS_WRIST=137322075008 \
  -e PROMPT="pick up the blue block and place it in the cardboard box" \
  -e INFER_PERIOD=0.6 \
  -e HORIZON_STEPS=6 \
  -e MAX_STEP_DEG=3.0 \
  -e DT=0.05 \
  -e VEL=0.08 \
  -e ACC=0.15 \
  -e LOOKAHEAD=0.15 \
  -e GAIN=200 \
  openpi_robot
```

**Tuning notes (UR5 smoothness + “short” motions):**
- **Chunk duration**: The bridge executes `HORIZON_STEPS` actions, holding each for `HOLD_PER_STEP` seconds. Total time per policy call is approximately \(HORIZON\_STEPS \times HOLD\_PER\_STEP\).
  - If you set **`INFER_PERIOD`** and do **not** set `HOLD_PER_STEP`, the bridge will derive `HOLD_PER_STEP = INFER_PERIOD / HORIZON_STEPS`.
- **Motion distance (“very short”)**: In default `ACTION_MODE=delta`, each step delta is clamped per-joint by **`MAX_STEP_DEG`**. If `MAX_STEP_DEG` is too small (e.g. `0.5`), motion will look tiny even if the policy is trying to move.
- **Jagged motion**: Usually improves by increasing chunk duration (larger `INFER_PERIOD` / `HOLD_PER_STEP`), using a less aggressive `GAIN`, and avoiding very small `DT` unless you’ve tuned `LOOKAHEAD/GAIN` for it.

docker run --rm -it \
  --gpus=all \
  --network=host \
  --ipc=host \
  --device=/dev/bus/usb:/dev/bus/usb \
  $RUN_DEVICES \
  --group-add video \
  -v "$PWD":/app \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --name openpi-robot \
  -e RS_BASE=137322074310 \
  -e RS_WRIST=137322075008 \
  -e PROMPT="bus the table" \
  -e INFER_PERIOD=0.6 \
  -e HORIZON_STEPS=8 \
  -e HOLD_PER_STEP=0.05 \
  -e MAX_STEP_DEG=0.10 \
  -e DT=0.05 \
  -e VEL=0.05 \
  -e ACC=0.10 \
  -e LOOKAHEAD=0.10 \
  -e GAIN=200 \
  openpi_robot


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

**If the camera window does not show up (common causes):**
- **X11 not permitted**: you must run `xhost +local:docker` on the host *in the same login session* before starting the container.
  - If it still fails, the container often runs as `root`, so allow root explicitly: `xhost +SI:localuser:root`
- **No GUI / SSH session**: if you’re SSH’d into the machine without X forwarding, there is no local X server for `cv2.imshow`. Either enable X forwarding (slower) or run headless with `-e SHOW_IMAGES=0`.
- **Wayland**: ensure XWayland is available and `$DISPLAY` is set (try `echo $DISPLAY` on the host). If needed, force an X11 display like `-e DISPLAY=:0`.
- **Headless OpenCV inside the container**: if you see `WARNING: OpenCV compiled without GUI support (no GTK/QT)`, your container is importing a headless OpenCV build. Rebuild the image after updating `local/docker/serve_policy_robot.Dockerfile` (it now removes `opencv-python-headless`).
- **Check container warnings**: the UR5 bridge prints warnings like “DISPLAY not set”, “OpenCV compiled without GUI support”, or “X11 connection test failed” to stderr at startup.

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

docker build -t openpi_robot -f local/docker/serve_policy_robot.Dockerfile .

RUN_DEVICES="$(for d in /dev/video*; do [ -e "$d" ] && printf -- '--device=%s ' "$d"; done)"

docker rm -f openpi-robot 2>/dev/null || true


```

Run with: `chmod +x launch_robot.sh && ./launch_robot.sh`
docker run --rm -it --gpus=all --network=host --ipc=host --device=/dev/bus/usb:/dev/bus/usb $RUN_DEVICES --group-add video -v "$PWD":/app -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 --name openpi-robot -e RS_BASE=137322074310 -e RS_WRIST=137322075008 -e PROMPT="pick up the the red mug 10cm from the table" -e INFER_PERIOD=0.3 -e HORIZON_STEPS=3 -e HOLD_PER_STEP=0.1 openpi_robot