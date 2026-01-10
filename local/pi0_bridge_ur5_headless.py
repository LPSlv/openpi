"""
Robot bridge for running π₀ policy on UR5 robot arm with RealSense cameras.

Connects policy server to UR5 robot and RealSense camera. Handles control loop:
capture images, read robot state, call policy server, execute actions with safety checks.
"""

import os
import time
import socket
import numpy as np
import cv2

FAKE_CAM = os.environ.get("FAKE_CAM", "0") == "1"
if not FAKE_CAM:
    import pyrealsense2 as rs

import rtde_receive, rtde_control
from openpi_client import websocket_client_policy
from openpi_client import image_tools

def _ef(name, default):
    """Read environment variable with type conversion."""
    v = os.environ.get(name)
    try:
        return type(default)(v) if v is not None else default
    except Exception:
        return default

# Configuration
UR_IP = os.environ.get("UR_IP", "192.168.1.116")
PROMPT = os.environ.get("PROMPT", "pick up the grey shaker bottle")

SERIAL_BASE = os.environ.get("RS_BASE", "")
SERIAL_WRIST = os.environ.get("RS_WRIST", "")
W = int(os.environ.get("RS_W", "640"))
H = int(os.environ.get("RS_H", "480"))
FPS = int(os.environ.get("RS_FPS", "60"))
RS_TIMEOUT_MS = int(os.environ.get("RS_TIMEOUT_MS", "10000"))  # Frame wait timeout in milliseconds

DT = _ef("DT", 1.0/20.0)
VEL = _ef("VEL", 0.05)
ACC = _ef("ACC", 0.05)
LOOKAHEAD = min(max(_ef("LOOKAHEAD", 0.20), 0.03), 0.20)
GAIN = min(max(_ef("GAIN", 150), 100), 2000)

INFER_PERIOD = _ef("INFER_PERIOD", 0.6)
HORIZON_STEPS = int(_ef("HORIZON_STEPS", 10))
HOLD_PER_STEP = _ef("HOLD_PER_STEP", 0.15)

ACTION_MODE = os.environ.get("ACTION_MODE", "absolute").lower()
MAX_STEP_DEG = float(os.environ.get("MAX_STEP_DEG", "2.0"))

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
SHOW_IMAGES = os.environ.get("SHOW_IMAGES", "1") == "1"
_display_available = True

def process_image(bgr):
    """Process BGR image from RealSense to RGB uint8 format expected by policy.
    
    Matches the preprocessing pattern used in DROID and ALOHA examples:
    - Convert BGR to RGB (OpenCV uses BGR by default)
    - Resize with padding to 224x224 (maintains aspect ratio)
    - Convert to uint8 (required for network transmission and model input)
    
    Returns: (224, 224, 3) uint8 array in RGB format, HWC layout
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = image_tools.resize_with_pad(rgb, 224, 224)
    return image_tools.convert_to_uint8(resized)

def get_fake_frame():
    """Return a black 224x224 RGB frame for fake camera mode."""
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            np.zeros((224, 224, 3), dtype=np.uint8), 224, 224
        )
    )

def start_rgb(serial):
    """Start RealSense RGB camera with fixed exposure and white balance."""
    if not serial:
        return None
    if FAKE_CAM:
        return None
    p = rs.pipeline()
    c = rs.config()
    c.enable_device(serial)
    c.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    prof = p.start(c)
    try:
        for s in prof.get_device().sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                s.set_option(rs.option.enable_auto_exposure, 0)
                s.set_option(rs.option.exposure, 100.0)
                s.set_option(rs.option.enable_auto_white_balance, 0)
                s.set_option(rs.option.white_balance, 3600.0)
                break
    except Exception:
        pass
    return p

# Global socket for URScript commands (persistent connection)
_urscript_sock = None

def send_urscript(host, port, script):
    """Send URScript command to robot controller.
    
    Uses a persistent socket connection to avoid connection overhead.
    
    Args:
        host: UR robot IP address
        port: Port for URScript commands (default 30002)
        script: URScript code to execute
    """
    global _urscript_sock
    try:
        # Create or reuse socket connection
        if _urscript_sock is None:
            _urscript_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _urscript_sock.settimeout(2.0)
            _urscript_sock.connect((host, port))
        
        # Send the script
        _urscript_sock.sendall(script.encode())
    except (socket.timeout, ConnectionError, OSError) as e:
        # Connection lost or failed, reset socket and try once more
        if _urscript_sock is not None:
            try:
                _urscript_sock.close()
            except:
                pass
            _urscript_sock = None
        
        # Try one more time with a fresh connection
        try:
            _urscript_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _urscript_sock.settimeout(2.0)
            _urscript_sock.connect((host, port))
            _urscript_sock.sendall(script.encode())
        except Exception as e2:
            # Only warn on the second failure to avoid spam
            print(f"Warning: Failed to send URScript (retry also failed): {e2}", flush=True)
            _urscript_sock = None
    except Exception as e:
        print(f"Warning: Failed to send URScript: {e}", flush=True)

def assert_control_ready(ctrl, rcv):
    """Verify robot controller is ready to accept servo commands."""
    if not ctrl.isConnected():
        raise RuntimeError("RTDEControlInterface not connected")
    q_now = rcv.getActualQ()
    t0 = time.time()
    ok_once = False
    while time.time() - t0 < 3.0:
        ok = ctrl.servoJ(q_now, VEL, ACC, DT, LOOKAHEAD, GAIN)
        if ok:
            ok_once = True
            break
        time.sleep(DT)
    if not ok_once:
        raise RuntimeError("RTDE control script is not running on the controller. "
                           "Start External Control (or your control URScript) and press Play.")

def main():
    global _display_available
    if FAKE_CAM:
        print("*** FAKE_CAM=1: Using black 224x224 RGB frames for base and wrist inputs ***", flush=True)
        pb = None
        pw = None
    else:
        pb = start_rgb(SERIAL_BASE) if SERIAL_BASE else None
        if pb is None:
            raise RuntimeError("Base RealSense not started. Set RS_BASE to the serial number.")
        
        pw = start_rgb(SERIAL_WRIST) if SERIAL_WRIST else None
        if pw is None:
            print("Warning: Wrist camera not configured (RS_WRIST not set). Using base camera for both views.", flush=True)

    print(f"Connecting to robot at {UR_IP}...", flush=True)
    try:
        rcv = rtde_receive.RTDEReceiveInterface(UR_IP, frequency=125.0)
        ctrl = rtde_control.RTDEControlInterface(UR_IP)
        
        # Check if control interface is connected
        if not ctrl.isConnected():
            raise RuntimeError("RTDEControlInterface not connected. Make sure the robot is powered on and reachable.")
        
        # Try to read current joint positions to verify connection
        try:
            test_q = rcv.getActualQ()
            print(f"Robot connected! Current joint positions (deg): {np.degrees(test_q)}", flush=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read robot state: {e}. Check robot connection and RTDE script.")
            
    except Exception as e:
        raise RuntimeError(f"Failed to connect to robot at {UR_IP}: {e}\n"
                          "Make sure:\n"
                          "1. Robot is powered on and in Remote Control mode\n"
                          "2. RTDE control script is running on the controller\n"
                          "3. Robot IP address is correct (current: {UR_IP})")

    print("Connecting to policy server at localhost:8000...", flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
    server_metadata = client.get_server_metadata()
    print("Connected! Server metadata:", server_metadata, flush=True)
    
    
    # Auto-detect action mode from server metadata if available, otherwise use environment variable
    # This ensures the robot bridge matches what the policy server is outputting
    global ACTION_MODE
    server_action_mode = server_metadata.get("action_mode")
    if server_action_mode:
        ACTION_MODE = str(server_action_mode).lower()
        print(f"Auto-detected action_mode from server: {ACTION_MODE} (overriding env var)", flush=True)
    else:
        print(f"Using action_mode from environment: {ACTION_MODE}", flush=True)
    print(f"Cadence: INFER_PERIOD={INFER_PERIOD}s  HORIZON_STEPS={HORIZON_STEPS}  HOLD_PER_STEP={HOLD_PER_STEP}s", flush=True)
    print(f"servoJ: VEL={VEL} ACC={ACC} DT={DT} LOOKAHEAD={LOOKAHEAD} GAIN={GAIN}", flush=True)
    print(f"Camera capture: {W}x{H}@{FPS} -> 224x224 RGB; WB=3600", flush=True)
    if SERIAL_WRIST:
        print(f"Base camera: {SERIAL_BASE}, Wrist camera: {SERIAL_WRIST}", flush=True)
    else:
        print(f"Base camera: {SERIAL_BASE} (used for both views)", flush=True)
    print(f"ACTION_MODE = {ACTION_MODE}", flush=True)
    print(f"MAX_STEP_DEG = {MAX_STEP_DEG}", flush=True)
    print("DRY_RUN =", DRY_RUN, flush=True)
    print("SHOW_IMAGES =", SHOW_IMAGES, flush=True)

    if not DRY_RUN:
        assert_control_ready(ctrl, rcv)

    last_infer = 0.0
    actions_chunk = None
    chunk_idx = 0
    last_gripper_pos = None  # Track last gripper position for debouncing

    try:
        while True:
            if FAKE_CAM:
                rgb_base = get_fake_frame()
                rgb_wrist = get_fake_frame()
            else:
                fb = pb.wait_for_frames(RS_TIMEOUT_MS)
                cb = fb.get_color_frame()
                if not cb:
                    continue
                img_b = np.asanyarray(cb.get_data())
                rgb_base = process_image(img_b)
                
                # Get wrist camera image if available, otherwise use base camera
                if pw is not None:
                    fw = pw.wait_for_frames(RS_TIMEOUT_MS)
                    cw = fw.get_color_frame()
                    if cw:
                        img_w = np.asanyarray(cw.get_data())
                        rgb_wrist = process_image(img_w)
                    else:
                        rgb_wrist = rgb_base  # Fallback to base if wrist frame not available
                else:
                    rgb_wrist = rgb_base  # Use base camera for wrist view if no wrist camera

            q6 = np.array(rcv.getActualQ(), float)
            # Use last known gripper position for state, or default to 0.0
            gripper_pos = last_gripper_pos if last_gripper_pos is not None else 0.0
            gripper_state = np.array(gripper_pos, dtype=np.float32)
            state = np.concatenate([q6, gripper_state[None]]).astype(np.float32)

            now = time.time()
            need_new = (
                actions_chunk is None
                or (now - last_infer) >= INFER_PERIOD
                or chunk_idx >= actions_chunk.shape[0]
            )
            if need_new:
                # Build observation dict with proper keys for UR5 policy
                observation = {
                    "observation/image": rgb_base,
                    "observation/wrist_image": rgb_wrist,
                    "observation/state": state,
                    "prompt": PROMPT,
                }
                
                
                if SHOW_IMAGES and _display_available:
                    try:
                        combined = np.concatenate([rgb_base, rgb_wrist], axis=1)
                        display_img = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Policy Input: Base | Wrist (224x224 each)", display_img)
                        cv2.waitKey(1)
                    except cv2.error as e:
                        if "not implemented" in str(e).lower() or "gtk" in str(e).lower():
                            print("Warning: OpenCV GUI not available. Display disabled. Set SHOW_IMAGES=0 to suppress this warning.", flush=True)
                            _display_available = False
                        else:
                            raise
                
                out = client.infer(observation)
                
                arr = np.asarray(out.get("actions", []), dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                
                # Extract first 7 dims (6 joints + 1 gripper)
                assert arr.shape[1] >= 7, f"Actions must have at least 7 dims, got {arr.shape[1]}"
                arr = arr[:, :7]
                assert np.all(np.isfinite(arr)), f"Actions contain non-finite values: {arr}"
                
                # Calculate analysis for first step
                a0 = arr[0, :6]  # First step actions in rad
                q_now = q6  # Current joints in rad
                dq0 = a0 - q_now  # Difference (not wrapped)
                
                # Determine if absolute or delta
                dq0_abs = np.abs(dq0)
                a0_abs = np.abs(a0)
                is_absolute_near_current = np.all(dq0_abs < 0.05)  # < 0.05 rad = 2.9°
                is_delta = np.all(a0_abs < 0.05) and np.allclose(dq0_abs, a0_abs, atol=0.01)
                
                action_type = "absolute (near current)" if is_absolute_near_current else ("delta" if is_delta else "absolute (far from current)")
                
                # Print action chunk in table format (degrees only)
                print(f"\n{'='*100}", flush=True)
                print(f"Action chunk (shape={arr.shape}) - Type: {action_type}", flush=True)
                print(f"{'='*100}", flush=True)
                print(f"{'Step':<6} {'J1 (deg)':<10} {'J2 (deg)':<10} {'J3 (deg)':<10} {'J4 (deg)':<10} {'J5 (deg)':<10} {'J6 (deg)':<10} {'Gripper':<10}", flush=True)
                print(f"{'-'*100}", flush=True)
                for i in range(arr.shape[0]):
                    joints_deg = np.degrees(arr[i, :6])
                    gripper = arr[i, 6]
                    print(f"{i:<6} {joints_deg[0]:<10.2f} {joints_deg[1]:<10.2f} {joints_deg[2]:<10.2f} "
                          f"{joints_deg[3]:<10.2f} {joints_deg[4]:<10.2f} {joints_deg[5]:<10.2f} {gripper:<10.4f}", flush=True)
                print(f"{'='*100}\n", flush=True)
                
                actions_chunk = arr
                chunk_idx = 0
                last_infer = now

            steps = min(HORIZON_STEPS, actions_chunk.shape[0] - chunk_idx)
            if steps <= 0:
                time.sleep(0.01)
                continue

            for _ in range(steps):
                a = actions_chunk[chunk_idx]
                chunk_idx += 1

                # Extract first 6 joints (ignore gripper at index 6)
                # Actions should have shape (7,) from UR5Outputs transform: 6 joints + 1 gripper
                assert a.shape[0] >= 6, f"Action must have at least 6 dims for joints, got shape {a.shape}"
                assert np.all(np.isfinite(a[:6])), f"Action joints contain non-finite values: {a[:6]}"
                
                # Extract joint actions (first 6 dims)
                a6 = a[:6].astype(np.float32)

                if ACTION_MODE == "absolute":
                    # In absolute mode, actions are absolute joint positions
                    # Clip to reasonable range first, then rate-limit below
                    a6 = np.clip(a6, -2*np.pi, 2*np.pi)  # UR5 joint limits are roughly [-pi, pi]
                    q_tgt = a6
                else:
                    # In delta mode, actions are joint velocity deltas
                    # Clip delta to MAX_STEP_DEG per step for safety
                    a6 = np.clip(a6, -np.deg2rad(MAX_STEP_DEG), np.deg2rad(MAX_STEP_DEG))
                    q_tgt = q6 + a6

                # Rate limit: never move more than MAX_STEP_DEG per step per joint, regardless of mode
                # This ensures smooth, controlled motion even if policy outputs large absolute positions
                delta_q = q_tgt - q6
                delta_q_deg = np.abs(np.degrees(delta_q))
                max_delta_deg = MAX_STEP_DEG
                # Scale down each joint independently if it exceeds MAX_STEP_DEG
                scale = np.minimum(1.0, max_delta_deg / (delta_q_deg + 1e-9))
                # Apply scaling per-joint (not global minimum)
                delta_q_scaled = delta_q * scale
                q_tgt = q6 + delta_q_scaled
                
                # Final safety check: ensure target is finite
                assert np.all(np.isfinite(q_tgt)), f"Target joint positions contain non-finite values: {q_tgt}"
                
                # Extract and set gripper position with debouncing
                gripper_action = a[6]  # Gripper is at index 6
                # 1) Map policy gripper to [0..1] and clip
                g = np.clip(gripper_action, 0.0, 1.0)
                
                # 2) Debounce: only send if change > 2% (0.02)
                if last_gripper_pos is None or abs(g - last_gripper_pos) > 0.02:
                    # 3) Convert to Robotiq position (0..255)
                    pos = int(round(g * 255))
                    
                    # 4) Send URScript to port 30002
                    if not DRY_RUN:
                        urscript = f"rq_set_pos({pos})\n"
                        send_urscript(UR_IP, 30002, urscript)
                    
                    last_gripper_pos = g  # Update last position

                if DRY_RUN:
                    delta_deg = np.round(np.degrees(q_tgt - q6), 3)
                    print(f"[DRY_RUN] q_deg: {np.round(np.degrees(q6), 2)}, "
                          f"q_tgt_deg: {np.round(np.degrees(q_tgt), 2)}, "
                          f"delta_deg: {delta_deg}, "
                          f"a_deg: {np.round(np.degrees(a6), 3)}, "
                          f"gripper: {g:.4f}, "
                          f"mode: {ACTION_MODE}", flush=True)
                    time.sleep(HOLD_PER_STEP)
                    q6 = np.array(rcv.getActualQ(), float)
                    continue

                t0 = time.time()
                while time.time() - t0 < HOLD_PER_STEP:
                    ok = ctrl.servoJ(q_tgt.tolist(), VEL, ACC, DT, LOOKAHEAD, GAIN)
                    if not ok:
                        raise RuntimeError("servoJ rejected by controller. "
                                           "Control script may have stopped or a protective stop occurred.")
                    time.sleep(DT)

                q6 = np.array(rcv.getActualQ(), float)

    except KeyboardInterrupt:
        print("Stopping.", flush=True)
    finally:
        if SHOW_IMAGES and _display_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        try:
            ctrl.speedStop()
        except Exception:
            pass
        # Close URScript socket
        global _urscript_sock
        if _urscript_sock is not None:
            try:
                _urscript_sock.close()
            except Exception:
                pass
            _urscript_sock = None
        if not FAKE_CAM:
            try:
                if pb is not None:
                    pb.stop()
            except Exception:
                pass
            try:
                if pw is not None:
                    pw.stop()
            except Exception:
                pass

if __name__ == "__main__":
    main()
