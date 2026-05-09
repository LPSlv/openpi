"""UR5 + RealSense bridge for the websocket policy server."""

import os
import sys
import time

# Qt MIT-SHM in Docker → X11 connection breaks; force XCB for container reliability.
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

import rtde_control
import rtde_receive
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from ur5 import defaults as _defaults
from ur5.utils.robotiq_gripper import RobotiqGripperHelper

FAKE_CAM = os.environ.get("FAKE_CAM", "0") == "1"
if not FAKE_CAM:
    import pyrealsense2 as rs


UR_IP = os.environ.get("UR_IP", _defaults.UR_IP)
PROMPT = os.environ.get("PROMPT", "do something")

POLICY_HOST = os.environ.get("POLICY_HOST", "localhost")
POLICY_PORT = int(os.environ.get("POLICY_PORT", "8000"))

RS_BASE = os.environ.get("RS_BASE", "")
RS_WRIST = os.environ.get("RS_WRIST", "")
RS_W = int(os.environ.get("RS_W", str(_defaults.RS_W)))
RS_H = int(os.environ.get("RS_H", str(_defaults.RS_H)))
RS_FPS = int(os.environ.get("RS_FPS", str(_defaults.RS_FPS)))
RS_TIMEOUT_MS = int(os.environ.get("RS_TIMEOUT_MS", str(_defaults.RS_TIMEOUT_MS)))

DT = float(os.environ.get("DT", "0.05"))
VEL = float(os.environ.get("VEL", "0.05"))
ACC = float(os.environ.get("ACC", "0.05"))
LOOKAHEAD = float(os.environ.get("LOOKAHEAD", "0.2"))
GAIN = float(os.environ.get("GAIN", "150"))

# Wall-clock per chunk; derives HOLD_PER_STEP = INFER_PERIOD / HORIZON_STEPS.
_infer_period_raw = os.environ.get("INFER_PERIOD")
INFER_PERIOD: float | None = None
if _infer_period_raw not in (None, ""):
    try:
        INFER_PERIOD = float(_infer_period_raw)
    except ValueError as e:
        raise ValueError(f"Invalid INFER_PERIOD={_infer_period_raw!r}, expected float seconds") from e

# Pi0 pretrained model: 20 Hz for UR5e → 0.05s/step.
HOLD_PER_STEP = float(os.environ.get("HOLD_PER_STEP", "0.05"))
HORIZON_STEPS = int(os.environ.get("HORIZON_STEPS", "10"))

if INFER_PERIOD is not None and "HOLD_PER_STEP" not in os.environ:
    if HORIZON_STEPS <= 0:
        raise ValueError(f"HORIZON_STEPS must be > 0 when using INFER_PERIOD, got {HORIZON_STEPS}")
    HOLD_PER_STEP = INFER_PERIOD / float(HORIZON_STEPS)

ACTION_MODE = os.environ.get("ACTION_MODE", "absolute").lower()
MAX_STEP_DEG = float(os.environ.get("MAX_STEP_DEG", "2.0"))

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

GRIPPER_PORT = int(os.environ.get("GRIPPER_PORT", str(_defaults.ROBOTIQ_PORT)))
USE_GRIPPER = os.environ.get("USE_GRIPPER", "1") == "1"
GRIPPER_DEBOUNCE = float(os.environ.get("GRIPPER_DEBOUNCE", "0.02"))
# Binarize gripper at this threshold; empty = continuous 0–1.
GRIPPER_THRESHOLD = os.environ.get("GRIPPER_THRESHOLD", "")

SHOW_IMAGES = os.environ.get("SHOW_IMAGES", "1") == "1"

# Camera exposure: RS_EXPOSURE base, RS_WRIST_EXPOSURE wrist; RS_AUTO_EXPOSURE=0 → manual.
RS_AUTO_EXPOSURE = os.environ.get("RS_AUTO_EXPOSURE", "")  # "" = don't touch, "0" disable, "1" enable
RS_EXPOSURE = os.environ.get("RS_EXPOSURE", "")  # "" = don't touch
RS_WRIST_EXPOSURE = os.environ.get("RS_WRIST_EXPOSURE", "")  # "" = use RS_EXPOSURE

# Inference recording dir; empty = off.
RECORD_DIR = os.environ.get("RECORD_DIR", "")

# Dual-policy override: arm from main policy, gripper (dim 6) from a second server.
GRIPPER_POLICY_HOST = os.environ.get("GRIPPER_POLICY_HOST", "")
GRIPPER_POLICY_PORT = int(os.environ.get("GRIPPER_POLICY_PORT", "8001"))

# OpenCV GUI capability probe.
_has_gui = False
if SHOW_IMAGES:
    display = os.environ.get("DISPLAY")
    if not display:
        print("WARNING: DISPLAY environment variable not set. Image preview disabled.", file=sys.stderr)
        print("WARNING: Make sure to pass -e DISPLAY=$DISPLAY to docker run", file=sys.stderr)
    else:
        print(f"INFO: DISPLAY={display}", file=sys.stderr)
        
        # OpenCV build must include GTK or QT for cv2.imshow.
        try:
            build_info = cv2.getBuildInformation()
            has_gtk = False
            if "GTK:" in build_info:
                gtk_line = [line for line in build_info.split("\n") if "GTK:" in line]
                if gtk_line:
                    has_gtk = "YES" in gtk_line[0] or "ON" in gtk_line[0]
            has_qt = False
            if "QT:" in build_info:
                qt_line = [line for line in build_info.split("\n") if "QT:" in line]
                if qt_line:
                    has_qt = "YES" in qt_line[0] or "ON" in qt_line[0]

            if not (has_gtk or has_qt):
                print("WARNING: OpenCV compiled without GUI support (no GTK/QT). Image preview disabled.", file=sys.stderr)
                print("WARNING: The opencv-python wheel may not have GUI support. Consider rebuilding with GUI support.", file=sys.stderr)
            else:
                # Probe with a throwaway window.
                try:
                    test_img = np.zeros((10, 10, 3), dtype=np.uint8)
                    cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
                    cv2.imshow("_test", test_img)
                    cv2.waitKey(1)
                    cv2.destroyWindow("_test")
                    _has_gui = True
                    print("INFO: OpenCV GUI support verified, image preview enabled", file=sys.stderr)
                except Exception as e:
                    _has_gui = False
                    error_str = str(e).lower()
                    if "x11" in error_str or "connection" in error_str or "display" in error_str:
                        print("WARNING: X11 connection test failed. Image preview disabled.", file=sys.stderr)
                        print("INFO: Robot control will work without image display.", file=sys.stderr)
                    else:
                        print(f"WARNING: OpenCV GUI test failed: {e}. Image preview disabled.", file=sys.stderr)
                        print("WARNING: Check X11 forwarding: xhost +local:docker and -v /tmp/.X11-unix:/tmp/.X11-unix:ro", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Could not check OpenCV build info: {e}", file=sys.stderr)
            # Probe directly.
            try:
                test_img = np.zeros((10, 10, 3), dtype=np.uint8)
                cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
                cv2.imshow("_test", test_img)
                cv2.waitKey(1)
                cv2.destroyWindow("_test")
                _has_gui = True
                print("INFO: OpenCV GUI test passed, image preview enabled", file=sys.stderr)
            except Exception as e2:
                _has_gui = False
                print(f"WARNING: OpenCV GUI test failed: {e2}. Image preview disabled.", file=sys.stderr)


def _process_bgr(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # 256→224 path matches training (recording saves 256x256, model ResizeImages 224).
    rgb = image_tools.resize_with_pad(rgb, 256, 256)
    return image_tools.convert_to_uint8(rgb)


def _start_rgb(serial: str, *, exposure_override: str = "") -> "rs.pipeline | None":
    if FAKE_CAM or not serial:
        return None
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
    prof = pipe.start(cfg)

    # Per-camera exposure override (wrist may differ from base).
    exposure_val = exposure_override if exposure_override else RS_EXPOSURE
    try:
        for s in prof.get_device().sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                if RS_AUTO_EXPOSURE != "":
                    s.set_option(rs.option.enable_auto_exposure, float(RS_AUTO_EXPOSURE))
                if exposure_val != "":
                    s.set_option(rs.option.exposure, float(exposure_val))

                ae = s.get_option(rs.option.enable_auto_exposure)
                exp = s.get_option(rs.option.exposure)
                gain = s.get_option(rs.option.gain)
                intrinsics = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                print(
                    f"Camera {serial}: {intrinsics.width}x{intrinsics.height} "
                    f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
                    f"auto_exposure={ae} exposure={exp} gain={gain} fps={RS_FPS}",
                    flush=True,
                )
                break
    except Exception as e:
        print(f"Camera {serial}: could not read/set settings: {e}", flush=True)

    return pipe


def _read_rgb(pipe: "rs.pipeline") -> np.ndarray | None:
    """Latest frame; drains stale buffered frames after long inference."""
    frame = None
    # Drain to latest.
    while True:
        try:
            frames = pipe.poll_for_frames()
            f = frames.get_color_frame()
            if f:
                frame = f
            else:
                break
        except Exception:
            break
    # Fallback: blocking read.
    if frame is None:
        frames = pipe.wait_for_frames(RS_TIMEOUT_MS)
        frame = frames.get_color_frame()
    if not frame:
        return None
    return _process_bgr(np.asanyarray(frame.get_data()))


def _move_to_reset_position(
    ctrl: rtde_control.RTDEControlInterface,
    rcv: rtde_receive.RTDEReceiveInterface,
    reset_q_rad: list[float],
    vel: float = 0.2,
    acc: float = 0.3,
    timeout_sec: float = 10.0,
) -> None:
    """Move robot to reset position via RTDE moveJ."""
    target_q = np.asarray(reset_q_rad, dtype=np.float64)

    success = ctrl.moveJ(reset_q_rad, vel, acc)
    if not success:
        print("Warning: moveJ command failed for reset position")
        return

    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        current_q = np.asarray(rcv.getActualQ(), dtype=np.float64)
        dist = float(np.linalg.norm(current_q - target_q))
        if dist < 0.05:  # ~3° tolerance.
            print(f"Moved to reset position (error: {np.degrees(dist):.2f} deg)")
            time.sleep(0.5)  # Settle.
            return
        time.sleep(0.1)

    final_q = np.asarray(rcv.getActualQ(), dtype=np.float64)
    final_error = np.degrees(np.linalg.norm(final_q - target_q))
    print(f"Warning: Timeout waiting for reset position (final error: {final_error:.2f} deg)")


def main() -> None:
    global _has_gui  # Disabled below if X11 drops mid-session.

    print("=" * 60)
    print("Initializing UR5 Robot Bridge")
    print("=" * 60)

    print("Initializing cameras...", end=" ", flush=True)
    base_cam = _start_rgb(RS_BASE)
    wrist_cam = _start_rgb(RS_WRIST, exposure_override=RS_WRIST_EXPOSURE)
    if base_cam is not None or wrist_cam is not None:
        print("OK")
        time.sleep(1.0)  # Stream warm-up.
    else:
        print("SKIPPED (FAKE_CAM or no cameras configured)")

    print("Connecting to robot via RTDE...", end=" ", flush=True)
    rcv = rtde_receive.RTDEReceiveInterface(UR_IP, frequency=125.0)
    ctrl = rtde_control.RTDEControlInterface(UR_IP)
    print("OK")
    time.sleep(0.5)

    print(f"Connecting to policy server at {POLICY_HOST}:{POLICY_PORT}...", end=" ", flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(host=POLICY_HOST, port=POLICY_PORT)
    print("OK")
    time.sleep(0.5)

    # Optional secondary server for dim-6 gripper predictions.
    gripper_client = None
    if GRIPPER_POLICY_HOST:
        print(f"Connecting to GRIPPER policy at {GRIPPER_POLICY_HOST}:{GRIPPER_POLICY_PORT}...", end=" ", flush=True)
        gripper_client = websocket_client_policy.WebsocketClientPolicy(
            host=GRIPPER_POLICY_HOST, port=GRIPPER_POLICY_PORT
        )
        print("OK")
        time.sleep(0.5)

    print("Fetching policy configuration...", end=" ", flush=True)
    metadata = client.get_server_metadata()
    reset_pose = metadata.get("reset_pose")
    cfg = metadata.get("train_config")
    ckpt = metadata.get("checkpoint_dir")
    ns = metadata.get("norm_stats_dir")
    if cfg or ckpt:
        print(f"\nPolicy server reports: train_config={cfg!r} checkpoint_dir={ckpt!r}", flush=True)
    if ns:
        print(f"Policy server reports: norm_stats_dir={ns!r}", flush=True)
    if reset_pose is None:
        reset_pose = [float(np.deg2rad(d)) for d in _defaults.START_POSITION_DEG]
        print("OK (using default reset position)")
    else:
        print(f"OK (reset_pose from metadata)")
    
    print("Moving robot to reset position...", end=" ", flush=True)
    _move_to_reset_position(ctrl, rcv, reset_pose)
    print("OK")
    print("Waiting for robot to settle...", end=" ", flush=True)
    time.sleep(5.0)
    print("OK")

    gripper: RobotiqGripperHelper | None = None
    if USE_GRIPPER and GRIPPER_PORT == _defaults.ROBOTIQ_PORT:
        print("Initializing Robotiq Gripper...", end=" ", flush=True)
        try:
            gripper = RobotiqGripperHelper(UR_IP)
            time.sleep(1.0)  # Socket connect.
            print("Activating gripper...", end=" ", flush=True)
            gripper.activate()
            time.sleep(5.0)
            print("OK. Opening gripper...", end=" ", flush=True)
            gripper.open()
            time.sleep(5.0)
            print("OK - Gripper ready!")
        except Exception as e:
            print(f"FAILED: {e}")
            print("Continuing without gripper control.")
            if gripper is not None:
                try:
                    gripper.disconnect()
                except Exception:
                    pass
                gripper = None
    elif GRIPPER_PORT > 0 and GRIPPER_PORT != _defaults.ROBOTIQ_PORT:
        print(f"Warning: GRIPPER_PORT={GRIPPER_PORT} is not the Robotiq port ({_defaults.ROBOTIQ_PORT}).")
        print(f"Gripper control disabled. Set GRIPPER_PORT={_defaults.ROBOTIQ_PORT} or USE_GRIPPER=0 to disable.")
    else:
        print("Gripper: DISABLED")
    
    print("Finalizing setup...", end=" ", flush=True)
    time.sleep(0.5)
    print("OK")
    
    print("=" * 60)
    print("Setup complete! Starting main control loop...")
    chunk_time_s = float(HORIZON_STEPS) * float(HOLD_PER_STEP)
    print(
        "Control params:",
        f"ACTION_MODE={ACTION_MODE}",
        f"MAX_STEP_DEG={MAX_STEP_DEG}",
        f"HORIZON_STEPS={HORIZON_STEPS}",
        f"HOLD_PER_STEP={HOLD_PER_STEP:.4f}s",
        f"(~{chunk_time_s:.3f}s/chunk)",
        f"DT={DT}",
        f"VEL={VEL}",
        f"ACC={ACC}",
        f"LOOKAHEAD={LOOKAHEAD}",
        f"GAIN={GAIN}",
        flush=True,
    )
    if INFER_PERIOD is not None:
        if "HOLD_PER_STEP" in os.environ:
            print(
                f"Note: INFER_PERIOD={INFER_PERIOD} is set but HOLD_PER_STEP was explicitly set; "
                "INFER_PERIOD will not change timing.",
                flush=True,
            )
        else:
            print(
                f"Note: Using INFER_PERIOD={INFER_PERIOD} to derive HOLD_PER_STEP={HOLD_PER_STEP:.4f}s.",
                flush=True,
            )
    print("=" * 60)
    time.sleep(0.5)

    max_step_rad = np.deg2rad(MAX_STEP_DEG)
    # state[6] = last commanded gripper (matches recorder's g_cmd; 0.0 = open).
    last_gripper: float = 0.0
    infer_step = 0

    try:
        while True:
            # Capture images + state together to minimize desync.
            t_capture = time.time()
            if FAKE_CAM:
                rgb_base = np.zeros((224, 224, 3), dtype=np.uint8)
                rgb_wrist = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                if base_cam is None:
                    raise RuntimeError("RS_BASE must be set (or set FAKE_CAM=1).")
                rgb_base = _read_rgb(base_cam)
                if rgb_base is None:
                    continue
                if wrist_cam is None:
                    rgb_wrist = rgb_base
                else:
                    rgb_wrist = _read_rgb(wrist_cam)
                    if rgb_wrist is None:
                        rgb_wrist = rgb_base

            q = np.asarray(rcv.getActualQ(), dtype=np.float32)
            state = np.concatenate([q, np.array([last_gripper], dtype=np.float32)], axis=0)

            obs = {
                "observation/image": rgb_base,
                "observation/wrist_image": rgb_wrist,
                "observation/state": state,
                "prompt": PROMPT,
            }
            
            if SHOW_IMAGES and _has_gui:
                try:
                    bgr_base = cv2.cvtColor(rgb_base, cv2.COLOR_RGB2BGR)
                    bgr_wrist = cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR)
                    vis = np.hstack([bgr_base, bgr_wrist])
                    cv2.imshow("Base | Wrist (224x224)", vis)
                    cv2.waitKey(1)  # Non-blocking.
                except Exception as e:
                    error_str = str(e).lower()
                    is_x11_error = (
                        "x11" in error_str or
                        "display" in error_str or
                        "connection" in error_str or
                        "connection broke" in error_str or
                        "badvalue" in error_str or
                        "badwindow" in error_str or
                        "x11 server" in error_str
                    )
                    
                    if is_x11_error:
                        # X11 dropped → disable GUI for session.
                        if _has_gui:  # Print once.
                            _has_gui = False
                            print(
                                "\n" + "=" * 60,
                                file=sys.stderr,
                                flush=True
                            )
                            print(
                                "WARNING: X11 connection failed. Disabling image preview.",
                                file=sys.stderr,
                                flush=True
                            )
                            print(
                                "INFO: Robot control will continue without image display.",
                                file=sys.stderr,
                                flush=True
                            )
                            print(
                                "=" * 60 + "\n",
                                file=sys.stderr,
                                flush=True
                            )
                    else:
                        # Non-X11 error; keep trying.
                        if _has_gui:
                            print(f"WARNING: Failed to display image: {e}", file=sys.stderr, flush=True)
            
            if infer_step == 0:
                print(
                    f"Image diagnostics (step 0):\n"
                    f"  base:  shape={rgb_base.shape} dtype={rgb_base.dtype} mean={rgb_base.mean():.1f} std={rgb_base.std():.1f}\n"
                    f"  wrist: shape={rgb_wrist.shape} dtype={rgb_wrist.dtype} mean={rgb_wrist.mean():.1f} std={rgb_wrist.std():.1f}\n"
                    f"  state: {state}",
                    flush=True,
                )

            t_infer_start = time.time()
            out = client.infer(obs)

            # Optional dual-policy gripper override.
            if gripper_client is not None:
                grip_out = gripper_client.infer(obs)
                grip_actions = np.asarray(grip_out["actions"], dtype=np.float32)
                if grip_actions.ndim == 1:
                    grip_actions = grip_actions[None, :]

            t_infer_end = time.time()
            actions = np.asarray(out["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            actions = actions[:, :7]

            if gripper_client is not None:
                actions[:, 6] = grip_actions[:actions.shape[0], 6]

            print(
                f"--- chunk: {actions.shape} joints_range=[{actions[:,:6].min():.4f}, {actions[:,:6].max():.4f}] "
                f"gripper_range=[{actions[:,6].min():.3f}, {actions[:,6].max():.3f}] "
                f"t_capture={t_infer_start - t_capture:.3f}s t_infer={t_infer_end - t_infer_start:.3f}s",
                flush=True,
            )

            if RECORD_DIR:
                rec_dir = os.path.join(RECORD_DIR)
                os.makedirs(rec_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(rec_dir, f"step_{infer_step:04d}.npz"),
                    image=rgb_base,
                    wrist_image=rgb_wrist,
                    state=state,
                    actions=actions,
                    prompt=np.array(PROMPT),
                )

            infer_step += 1

            for a in actions[:HORIZON_STEPS]:
                dq = np.asarray(a[:6], dtype=np.float32)
                if ACTION_MODE == "absolute":
                    q_tgt = dq
                else:
                    q_tgt = q + np.clip(dq, -max_step_rad, max_step_rad)

                g_raw = float(a[6])
                g = float(np.clip(g_raw, 0.0, 1.0))
                if GRIPPER_THRESHOLD != "":
                    g = 1.0 if g >= float(GRIPPER_THRESHOLD) else 0.0
                delta = dq - q if ACTION_MODE == "absolute" else dq
                print(f"joints delta={np.array2string(delta, precision=4, suppress_small=True)} gripper={g_raw:.3f}", flush=True)
                if abs(g - last_gripper) > GRIPPER_DEBOUNCE:
                    if not DRY_RUN and gripper is not None:
                        try:
                            # Non-blocking: arm continues while gripper moves.
                            gripper.send_normalized(g)
                        except Exception as e:
                            print(f"Warning: Gripper move failed: {e}", file=sys.stderr, flush=True)
                    last_gripper = g

                if DRY_RUN:
                    time.sleep(HOLD_PER_STEP)
                    q = np.asarray(rcv.getActualQ(), dtype=np.float32)
                    continue

                t0 = time.time()
                while time.time() - t0 < HOLD_PER_STEP:
                    ctrl.servoJ(q_tgt.tolist(), VEL, ACC, DT, LOOKAHEAD, GAIN)
                    time.sleep(DT)
                q = np.asarray(rcv.getActualQ(), dtype=np.float32)

    finally:
        try:
            ctrl.speedStop()
        except Exception:
            pass
        if gripper is not None:
            try:
                gripper.disconnect()
            except Exception:
                pass
        if not FAKE_CAM:
            try:
                if base_cam is not None:
                    base_cam.stop()
            except Exception:
                pass
            try:
                if wrist_cam is not None:
                    wrist_cam.stop()
            except Exception:
                pass
        if SHOW_IMAGES:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
