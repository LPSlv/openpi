# Deployment

Run the policy server and UR5 bridge from the same Docker image in two terminals.
For end-to-end setup (recording, training, then deploying) see
[quickstart.md](quickstart.md).

## Prerequisites

- UR5e in Remote Control mode, RTDE script running
- Two RealSense cameras connected via USB 3.0 for study evaluation
- Robotiq Hand-E with its URCap socket available on port 63352
- A saved checkpoint with `params/` and `assets/`, plus the config used to train it
- NVIDIA GPU with Docker + nvidia-container-toolkit
- Network access to the robot

## 1. Get camera serials

From the repo root, in the venv:

```bash
python ur5/test/rs_list.py
```

Set `RS_BASE` to the shoulder camera serial and `RS_WRIST` to the wrist camera serial. See also [`ur5/test/README.md`](../test/README.md).

## 2. Build the image

```bash
docker build -t openpi_robot -f ur5/docker/serve_policy_robot.Dockerfile .
```

## 3. Select and serve a checkpoint

From the repository root, set `CONFIG` to the training config and `CHECKPOINT`
to the saved step directory relative to this checkout. For the study these
are `pi05_ur5_blueblock10` and
`checkpoints/pi05_ur5_blueblock10/pi05_ur5_blueblock10-1/150`.
Keep the training config available in the checkout mounted at `/app`.

```bash
CONFIG=pi05_ur5
CHECKPOINT=checkpoints/pi05_ur5/my_exp/180   # replace 180 with your saved step
test -d "$CHECKPOINT/params" && test -d "$CHECKPOINT/assets" || exit 1
docker run --rm -it --gpus=all --network=host --ipc=host \
  -e PYTHONPATH=/app:/app/src \
  -v "$PWD":/app openpi_robot \
  python scripts/serve_policy.py policy:checkpoint \
  --policy.config="$CONFIG" --policy.dir="$CHECKPOINT"
```

Confirm the directories exist before you run Docker. This command starts only
the policy server. Wait for it to report that it is listening on port 8000
before starting the bridge. If the server cannot load normalization stats,
check that the checkpoint contains the matching `assets/<asset_id>/norm_stats.json`.

## 4. Start the bridge

The bridge commands a reset move using the policy metadata (or the default
pose) and can activate and open the gripper before the inference loop. This
startup runs even with `DRY_RUN=1`. Verify the reset pose and motion path for
your cell before starting the bridge.

In a second terminal on the same host, from the repository root, set `UR_IP`,
`RS_BASE`, and `RS_WRIST` to the hardware values and run:

```bash
VIDEO_DEVICES=()
for device in /dev/video*; do
  [ -e "$device" ] && VIDEO_DEVICES+=(--device="$device")
done

docker run --rm -it --network=host \
  --device=/dev/bus/usb:/dev/bus/usb "${VIDEO_DEVICES[@]}" --group-add video \
  -v "$PWD":/app \
  -e UR_IP="${UR_IP:?Set the robot IP}" \
  -e RS_BASE="${RS_BASE:?Set the shoulder camera serial}" \
  -e RS_WRIST="${RS_WRIST:?Set the wrist camera serial}" \
  -e PROMPT="Pick up the blue block and place it in the cardboard box" \
  -e SHOW_IMAGES=0 -e ACTION_MODE=absolute \
  -e HORIZON_STEPS=6 -e HOLD_PER_STEP=0.1 \
  openpi_robot python -m ur5.utils.pi0_bridge_ur5_headless
```

The supplied UR5 policy converts joint deltas back to absolute targets on
output. Use `ACTION_MODE=absolute` with that transform. For preview, set
`SHOW_IMAGES=1`, pass `-e DISPLAY="$DISPLAY"`, and mount
`-v /tmp/.X11-unix:/tmp/.X11-unix:ro`. See the X11 troubleshooting below.

## Tuning the motion

The bridge plays `HORIZON_STEPS` actions back-to-back, holding each for
`HOLD_PER_STEP` seconds, so each policy call covers roughly
`HORIZON_STEPS * HOLD_PER_STEP` seconds of action execution, plus observation and inference latency.

- The image sets `HOLD_PER_STEP=0.1`. Set it explicitly to change timing in
  Docker. Outside that image, with `HOLD_PER_STEP` unset, `INFER_PERIOD` sets
  the action-chunk duration by deriving `HOLD_PER_STEP = INFER_PERIOD / HORIZON_STEPS`.
- with `ACTION_MODE=delta`, each step is clipped per-joint by `MAX_STEP_DEG`;
  too small a value (e.g. 0.5 deg) makes motion look frozen
- jagged motion usually improves with longer `INFER_PERIOD`/`HOLD_PER_STEP`,
  a softer `GAIN`, and not chasing very small `DT` unless `LOOKAHEAD`/`GAIN`
  are tuned for it

## Evaluate

Start with the matched tabletop setup, task prompt, reset pose, and two camera
views used for training. The study selected checkpoint 150 and executed six
of the 15 predicted actions per query at 0.1 s per action. Treat these as
starting settings when camera placement, gripper, or task geometry changes.

Compare saved checkpoints with the execution horizon fixed, then compare
horizons with the checkpoint and hold time fixed. Reset the scene between
trials, change one factor per condition, and record which stages completed:
reach, grasp, transport, and release. Keep the checkpoint, horizon, timing,
prompt, camera placement and exposure, gripper, and initial object placement
with the trial record. Report stage progress and full releases separately.
The study used five trials per out-of-distribution condition. See the
[results](../../README.md#results) and [experiment notes](experiments.md).

## X11 troubleshooting

The bridge prints warnings to stderr if the preview can't open: "DISPLAY not
set", "OpenCV compiled without GUI support", "X11 connection test failed".

Common causes:

- **xhost not permitted.** Re-run `xhost +local:docker` (and
  `xhost +SI:localuser:root`) in the same login session.
- **No X server reachable.** Happens over plain SSH. Either enable X
  forwarding (slow) or set `-e SHOW_IMAGES=0`.
- **Wayland host.** Make sure XWayland is running and `$DISPLAY` is set; if
  needed force `-e DISPLAY=:0`.
- **Headless OpenCV inside the container.** If you see "OpenCV compiled
  without GUI support", an opencv-python-headless wheel beat the GUI build
  to the venv. Rebuild the image; the Dockerfile uninstalls the headless
  variant before pinning the GUI build.

## Environment variable reference

| Variable | Default | Description |
|----------|---------|-------------|
| `UR_IP` | `192.10.0.11` | robot IP address |
| `RS_BASE` | study serial baked into image | base camera serial |
| `RS_WRIST` | study serial baked into image | wrist camera serial; set empty explicitly for a single-camera diagnostic |
| `PROMPT` | `"pick up the blue block and place it in the cardboard box"` | language instruction |
| `INFER_PERIOD` | unset | action-chunk duration in seconds, used only when HOLD_PER_STEP is absent |
| `HORIZON_STEPS` | `6` | actions executed per policy call |
| `HOLD_PER_STEP` | `0.1` in image | seconds to hold each action step |
| `ACTION_MODE` | `absolute` | `"absolute"` or `"delta"` |
| `DT` | `0.02` | servoJ time step (s) |
| `VEL` | `0.5` | joint velocity limit (rad/s) |
| `ACC` | `0.5` | joint acceleration limit (rad/s^2) |
| `LOOKAHEAD` | `0.1` | servoJ lookahead time (s) |
| `GAIN` | `300` | servoJ proportional gain |
| `MAX_STEP_DEG` | `3.0` | max per-joint step size (deg), used in delta mode |
| `SHOW_IMAGES` | `1` | set `0` to disable the camera preview |
| `GRIPPER_PORT` | `63352` | Robotiq URCap socket port |
| `GRIPPER_DEBOUNCE` | `0.02` | min gripper-command change before resending |
| `RECORD_DIR` | empty | directory to dump per-step inference recordings |
| `DRY_RUN` | `0` | set `1` to skip policy commands in the inference loop; reset and gripper initialization still run |
| `FAKE_CAM` | `0` | set `1` to bypass cameras with synthetic images |
| `SERVER_ARGS` | *(see Dockerfile)* | arguments passed to `scripts/serve_policy.py` |
| `SERVER_WAIT` | `6` | seconds to wait for the policy server before starting the bridge |
