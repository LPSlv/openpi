# Deployment

Run the pi0 policy server and the UR5 bridge in a single Docker container.
For end-to-end setup (recording, training, then deploying) see
[quickstart.md](quickstart.md).

## Prerequisites

- UR5e in Remote Control mode, RTDE script running
- RealSense camera connected via USB 3.0
- NVIDIA GPU with Docker + nvidia-container-toolkit
- Network access to the robot

## 1. Get camera serials

From the repo root, in the venv:

```bash
python ur5/test/rs_list.py
```

Note the serial(s) for `RS_BASE` (over-the-shoulder) and optionally
`RS_WRIST`. See also [`ur5/test/README.md`](../test/README.md).

## 2. Build the image

```bash
docker build -t openpi_robot -f ur5/docker/serve_policy_robot.Dockerfile .
```

## 3. Allow X11 in for the camera preview (optional)

Run these on the host once per login session if you want `cv2.imshow` to work
inside the container:

```bash
export DISPLAY=${DISPLAY:-:0}
xhost +local:docker
xhost +SI:localuser:root   # the container runs as root
```

Skip this section if you're running headless (set `SHOW_IMAGES=0` below).

## 4. Run the container

```bash
docker rm -f openpi-robot 2>/dev/null || true

RUN_DEVICES="$(for d in /dev/video*; do [ -e "$d" ] && printf -- '--device=%s ' "$d"; done)"

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
  --name openpi-robot \
  -e RS_BASE=<base_serial> \
  -e RS_WRIST=<wrist_serial> \
  -e PROMPT="pick up the blue block and place it in the cardboard box" \
  openpi_robot
```

Drop the `-v /tmp/.X11-unix...` and `-e DISPLAY=...` lines plus add
`-e SHOW_IMAGES=0` for headless operation.

The container starts the policy server, waits `SERVER_WAIT` seconds, then
launches the bridge. The bridge connects to the robot and the cameras and
starts driving.

## Tuning the motion

The bridge plays `HORIZON_STEPS` actions back-to-back, holding each for
`HOLD_PER_STEP` seconds, so each policy call covers roughly
`HORIZON_STEPS * HOLD_PER_STEP` seconds of wall-clock time.

- if you set `INFER_PERIOD` and leave `HOLD_PER_STEP` unset, the bridge
  derives `HOLD_PER_STEP = INFER_PERIOD / HORIZON_STEPS`
- with `ACTION_MODE=delta`, each step is clipped per-joint by `MAX_STEP_DEG`;
  too small a value (e.g. 0.5 deg) makes motion look frozen
- jagged motion usually improves with longer `INFER_PERIOD`/`HOLD_PER_STEP`,
  a softer `GAIN`, and not chasing very small `DT` unless `LOOKAHEAD`/`GAIN`
  are tuned for it

## X11 troubleshooting

The bridge prints warnings to stderr if the preview can't open: "DISPLAY not
set", "OpenCV compiled without GUI support", "X11 connection test failed".

Common causes:

- **xhost not permitted** — re-run `xhost +local:docker` (and
  `xhost +SI:localuser:root`) in the same login session
- **No X server reachable** — happens over plain SSH; either enable X
  forwarding (slow) or set `-e SHOW_IMAGES=0`
- **Wayland host** — make sure XWayland is running and `$DISPLAY` is set; if
  needed force `-e DISPLAY=:0`
- **Headless OpenCV inside the container** — if you see "OpenCV compiled
  without GUI support", an opencv-python-headless wheel beat the GUI build
  to the venv. Rebuild the image; the Dockerfile uninstalls the headless
  variant before pinning the GUI build.

## Environment variable reference

| Variable | Default | Description |
|----------|---------|-------------|
| `UR_IP` | `192.10.0.11` | robot IP address |
| `RS_BASE` | *(required)* | base camera serial |
| `RS_WRIST` | *(optional)* | wrist camera serial; falls back to base for both views |
| `PROMPT` | `"pick up the blue block and place it in the cardboard box"` | language instruction |
| `INFER_PERIOD` | unset | wall-clock seconds per policy call (derives HOLD_PER_STEP) |
| `HORIZON_STEPS` | `6` | actions executed per policy call |
| `HOLD_PER_STEP` | derived | seconds to hold each action step |
| `ACTION_MODE` | `absolute` | `"absolute"` or `"delta"` |
| `DT` | `0.02` | servoJ time step (s) |
| `VEL` | `0.5` | joint velocity limit (rad/s) |
| `ACC` | `0.5` | joint acceleration limit (rad/s^2) |
| `LOOKAHEAD` | `0.1` | servoJ lookahead time (s) |
| `GAIN` | `300` | servoJ proportional gain |
| `MAX_STEP_DEG` | `3.0` | max per-joint step size (deg), used in delta mode |
| `SHOW_IMAGES` | `1` | set `0` to disable the camera preview |
| `ROBOTIQ_PORT` | `63352` | Robotiq URCap socket port |
| `GRIPPER_DEBOUNCE` | `0.02` | min gripper-command change before resending |
| `RECORD_DIR` | empty | directory to dump per-step inference recordings |
| `DRY_RUN` | `0` | set `1` to skip sending commands to the robot |
| `FAKE_CAM` | `0` | set `1` to bypass cameras with synthetic images |
| `SERVER_ARGS` | *(see Dockerfile)* | arguments passed to `scripts/serve_policy.py` |
| `SERVER_WAIT` | `6` | seconds to wait for the policy server before starting the bridge |
