"""UR5 + RealSense bridge for the websocket policy server."""

import os
import socket
import sys
import time

import cv2
import numpy as np

import rtde_control
import rtde_receive
from openpi_client import image_tools
from openpi_client import websocket_client_policy

FAKE_CAM = os.environ.get("FAKE_CAM", "0") == "1"
if not FAKE_CAM:
    import pyrealsense2 as rs


UR_IP = os.environ.get("UR_IP", "192.10.0.11")
PROMPT = os.environ.get("PROMPT", "do something")

POLICY_HOST = os.environ.get("POLICY_HOST", "localhost")
POLICY_PORT = int(os.environ.get("POLICY_PORT", "8000"))

RS_BASE = os.environ.get("RS_BASE", "")
RS_WRIST = os.environ.get("RS_WRIST", "")
RS_W = int(os.environ.get("RS_W", "640"))
RS_H = int(os.environ.get("RS_H", "480"))
RS_FPS = int(os.environ.get("RS_FPS", "60"))
RS_TIMEOUT_MS = int(os.environ.get("RS_TIMEOUT_MS", "10000"))

DT = float(os.environ.get("DT", "0.05"))
VEL = float(os.environ.get("VEL", "0.05"))
ACC = float(os.environ.get("ACC", "0.05"))
LOOKAHEAD = float(os.environ.get("LOOKAHEAD", "0.2"))
GAIN = float(os.environ.get("GAIN", "150"))

HOLD_PER_STEP = float(os.environ.get("HOLD_PER_STEP", "0.15"))
HORIZON_STEPS = int(os.environ.get("HORIZON_STEPS", "10"))

ACTION_MODE = os.environ.get("ACTION_MODE", "delta").lower()
MAX_STEP_DEG = float(os.environ.get("MAX_STEP_DEG", "2.0"))

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

GRIPPER_PORT = int(os.environ.get("GRIPPER_PORT", "0"))
GRIPPER_DEBOUNCE = float(os.environ.get("GRIPPER_DEBOUNCE", "0.02"))

SHOW_IMAGES = os.environ.get("SHOW_IMAGES", "1") == "1"

# Check if OpenCV has GUI support
_has_gui = False
if SHOW_IMAGES:
    # Check DISPLAY environment variable
    display = os.environ.get("DISPLAY")
    if not display:
        print("WARNING: DISPLAY environment variable not set. Image preview disabled.", file=sys.stderr)
        print("WARNING: Make sure to pass -e DISPLAY=$DISPLAY to docker run", file=sys.stderr)
    else:
        print(f"INFO: DISPLAY={display}", file=sys.stderr)
        
        # Check OpenCV build info for GUI support
        try:
            build_info = cv2.getBuildInformation()
            # Check for GTK support
            has_gtk = False
            if "GTK:" in build_info:
                gtk_line = [line for line in build_info.split("\n") if "GTK:" in line]
                if gtk_line:
                    has_gtk = "YES" in gtk_line[0] or "ON" in gtk_line[0]
            # Check for QT support  
            has_qt = False
            if "QT:" in build_info:
                qt_line = [line for line in build_info.split("\n") if "QT:" in line]
                if qt_line:
                    has_qt = "YES" in qt_line[0] or "ON" in qt_line[0]
            
            if not (has_gtk or has_qt):
                print("WARNING: OpenCV compiled without GUI support (no GTK/QT). Image preview disabled.", file=sys.stderr)
                print("WARNING: The opencv-python wheel may not have GUI support. Consider rebuilding with GUI support.", file=sys.stderr)
            else:
                # Try to create a test window to verify it works
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
                    print(f"WARNING: OpenCV GUI test failed: {e}. Image preview disabled.", file=sys.stderr)
                    print("WARNING: Check X11 forwarding: xhost +local:docker and -v /tmp/.X11-unix:/tmp/.X11-unix:ro", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Could not check OpenCV build info: {e}", file=sys.stderr)
            # Try anyway
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
    rgb = image_tools.resize_with_pad(rgb, 224, 224)
    return image_tools.convert_to_uint8(rgb)


def _start_rgb(serial: str) -> "rs.pipeline | None":
    if FAKE_CAM or not serial:
        return None
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
    pipe.start(cfg)
    return pipe


def _read_rgb(pipe: "rs.pipeline") -> np.ndarray | None:
    frames = pipe.wait_for_frames(RS_TIMEOUT_MS)
    frame = frames.get_color_frame()
    if not frame:
        return None
    return _process_bgr(np.asanyarray(frame.get_data()))


def _set_gripper(pos01: float) -> None:
    if GRIPPER_PORT <= 0:
        return
    pos01 = float(np.clip(pos01, 0.0, 1.0))
    pos = int(round(pos01 * 255))
    with socket.create_connection((UR_IP, GRIPPER_PORT), timeout=1.0) as s:
        s.sendall(f"rq_set_pos({pos})\n".encode())


def _move_to_reset_position(
    ctrl: rtde_control.RTDEControlInterface,
    rcv: rtde_receive.RTDEReceiveInterface,
    reset_q_rad: list[float],
    vel: float = 0.2,
    acc: float = 0.3,
    timeout_sec: float = 10.0,
) -> None:
    """Move robot to reset position using RTDE moveJ."""
    target_q = np.asarray(reset_q_rad, dtype=np.float64)

    # Use RTDE moveJ
    success = ctrl.moveJ(reset_q_rad, vel, acc)
    if not success:
        print("Warning: moveJ command failed for reset position")
        return

    # Wait for movement to complete
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        current_q = np.asarray(rcv.getActualQ(), dtype=np.float64)
        dist = float(np.linalg.norm(current_q - target_q))
        if dist < 0.05:  # ~3 degrees tolerance
            print(f"Moved to reset position (error: {np.degrees(dist):.2f} deg)")
            time.sleep(0.2)
            return
        time.sleep(0.1)

    final_q = np.asarray(rcv.getActualQ(), dtype=np.float64)
    final_error = np.degrees(np.linalg.norm(final_q - target_q))
    print(f"Warning: Timeout waiting for reset position (final error: {final_error:.2f} deg)")


def main() -> None:
    base_cam = _start_rgb(RS_BASE)
    wrist_cam = _start_rgb(RS_WRIST)

    rcv = rtde_receive.RTDEReceiveInterface(UR_IP, frequency=125.0)
    ctrl = rtde_control.RTDEControlInterface(UR_IP)
    client = websocket_client_policy.WebsocketClientPolicy(host=POLICY_HOST, port=POLICY_PORT)

    # Get reset_pose from server metadata
    metadata = client.get_server_metadata()
    reset_pose = metadata.get("reset_pose")
    if reset_pose is None:
        # Default reset position: (-90, -45, -120, -75, 90, 0) degrees in radians
        reset_pose = [-1.5708, -0.7854, -2.0944, -1.3089, 1.5708, 0.0]
        print("No reset_pose in metadata, using default reset position")
    else:
        print(f"Using reset_pose from metadata: {reset_pose}")

    # Move to reset position at startup
    print("Moving robot to reset position...")
    _move_to_reset_position(ctrl, rcv, reset_pose)
    # Reset gripper to open position
    if GRIPPER_PORT > 0:
        _set_gripper(1.0)
        time.sleep(0.5)
    print("Reset complete, starting main control loop...")

    max_step_rad = np.deg2rad(MAX_STEP_DEG)
    last_gripper: float | None = None

    try:
        while True:
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
            state = np.concatenate([q, np.array([0.0], dtype=np.float32)], axis=0)

            obs = {
                "observation/image": rgb_base,
                "observation/wrist_image": rgb_wrist,
                "observation/state": state,
                "prompt": PROMPT,
            }
            
            # Display images if enabled
            if SHOW_IMAGES and _has_gui:
                try:
                    # Convert RGB to BGR for OpenCV display
                    bgr_base = cv2.cvtColor(rgb_base, cv2.COLOR_RGB2BGR)
                    bgr_wrist = cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR)
                    # Show side by side
                    vis = np.hstack([bgr_base, bgr_wrist])
                    cv2.imshow("Base | Wrist (224x224)", vis)
                    cv2.waitKey(1)  # Non-blocking wait
                except Exception as e:
                    # Log the error but don't crash
                    print(f"WARNING: Failed to display image: {e}", file=sys.stderr, flush=True)
            
            out = client.infer(obs)
            actions = np.asarray(out["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            actions = actions[:, :7]

            for a in actions[:HORIZON_STEPS]:
                dq = np.asarray(a[:6], dtype=np.float32)
                if ACTION_MODE == "absolute":
                    q_tgt = dq
                else:
                    q_tgt = q + np.clip(dq, -max_step_rad, max_step_rad)

                g = float(a[6])
                if last_gripper is None or abs(g - last_gripper) > GRIPPER_DEBOUNCE:
                    if not DRY_RUN:
                        _set_gripper(g)
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
