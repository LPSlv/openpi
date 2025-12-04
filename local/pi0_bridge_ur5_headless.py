"""
Robot bridge for running π₀ policy on UR5 robot arm with RealSense cameras.

Connects policy server to UR5 robot and RealSense camera. Handles control loop:
capture images, read robot state, call policy server, execute actions with safety checks.
"""

import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import rtde_receive, rtde_control
from openpi_client import websocket_client_policy

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
FPS = int(os.environ.get("RS_FPS", "30"))

DT = _ef("DT", 1.0/20.0)
VEL = _ef("VEL", 0.05)
ACC = _ef("ACC", 0.05)
LOOKAHEAD = min(max(_ef("LOOKAHEAD", 0.20), 0.03), 0.20)
GAIN = min(max(_ef("GAIN", 150), 100), 2000)

INFER_PERIOD = _ef("INFER_PERIOD", 0.6)
HORIZON_STEPS = int(_ef("HORIZON_STEPS", 10))
HOLD_PER_STEP = _ef("HOLD_PER_STEP", 0.15)

ACTION_MODE = os.environ.get("ACTION_MODE", "delta").lower()
MAX_STEP_DEG = float(os.environ.get("MAX_STEP_DEG", "0.5"))

def _parse_deg_limits(name):
    """Parse comma-separated joint limits from environment variable."""
    s = os.environ.get(name)
    if not s:
        return None
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 6:
        return None
    return np.array(vals, dtype=float)

SOFT_MIN_DEG = _parse_deg_limits("SOFT_MIN_DEG")
SOFT_MAX_DEG = _parse_deg_limits("SOFT_MAX_DEG")

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

def to_uint8_224(bgr):
    """Resize and center-crop BGR image to 224x224 RGB."""
    h, w = bgr.shape[:2]
    s = 224 / max(h, w)
    nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(bgr, (nw, nh), cv2.INTER_AREA)
    canvas = np.zeros((224, 224, 3), np.uint8)
    y0 = (224 - nh) // 2
    x0 = (224 - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def start_rgb(serial):
    """Start RealSense RGB camera with fixed exposure and white balance."""
    if not serial:
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
                s.set_option(rs.option.exposure, 200.0)
                s.set_option(rs.option.enable_auto_white_balance, 0)
                s.set_option(rs.option.white_balance, 3600.0)
                break
    except Exception:
        pass
    return p

def clamp_joint_limits(q_rad):
    """Clamp joint angles to soft limits if configured."""
    if SOFT_MIN_DEG is None or SOFT_MAX_DEG is None:
        return q_rad
    q_deg = np.degrees(q_rad)
    q_deg = np.clip(q_deg, SOFT_MIN_DEG, SOFT_MAX_DEG)
    return np.radians(q_deg)

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
    pb = start_rgb(SERIAL_BASE) if SERIAL_BASE else None
    if pb is None:
        raise RuntimeError("Base RealSense not started. Set RS_BASE to the serial number.")
    
    pw = start_rgb(SERIAL_WRIST) if SERIAL_WRIST else None
    if pw is None:
        print("Warning: Wrist camera not configured (RS_WRIST not set). Using base camera for both views.", flush=True)

    rcv = rtde_receive.RTDEReceiveInterface(UR_IP, frequency=125.0)
    ctrl = rtde_control.RTDEControlInterface(UR_IP)

    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
    print("Server metadata:", client.get_server_metadata(), flush=True)
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

    if not DRY_RUN:
        assert_control_ready(ctrl, rcv)

    last_infer = 0.0
    actions_chunk = None
    chunk_idx = 0

    try:
        while True:
            fb = pb.wait_for_frames()
            cb = fb.get_color_frame()
            if not cb:
                continue
            img_b = np.asanyarray(cb.get_data())
            rgb_base = to_uint8_224(img_b)
            
            # Get wrist camera image if available, otherwise use base camera
            if pw is not None:
                fw = pw.wait_for_frames()
                cw = fw.get_color_frame()
                if cw:
                    img_w = np.asanyarray(cw.get_data())
                    rgb_wrist = to_uint8_224(img_w)
                else:
                    rgb_wrist = rgb_base  # Fallback to base if wrist frame not available
            else:
                rgb_wrist = rgb_base  # Use base camera for wrist view if no wrist camera

            q6 = np.array(rcv.getActualQ(), float)  # UR5 has 6 joints
            # Pad to 7 joints for DROID compatibility (DROID expects 7+1 inputs)
            q7 = np.concatenate([q6, [0.0]], axis=0)  # pad a fake 7th joint

            now = time.time()
            need_new = (
                actions_chunk is None
                or (now - last_infer) >= INFER_PERIOD
                or chunk_idx >= actions_chunk.shape[0]
            )
            if need_new:
                obs = {
                    "observation/exterior_image_1_left": rgb_base,
                    "observation/wrist_image_left": rgb_wrist,
                    "observation/joint_position": q7.astype(np.float32),  # length 7 now
                    "observation/gripper_position": 0.0,
                    "prompt": PROMPT,
                }
                out = client.infer(obs)
                arr = np.asarray(out.get("actions", []), dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                # Map actions back: take only first 6 joints, ignore the model's 7th joint and gripper
                if arr.shape[1] >= 6:
                    arr = arr[:, :6]  # Keep only first 6 joints
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

                # Extract first 6 joints (throw away model's 7th joint and gripper if present)
                if a.shape[0] < 6 or not np.all(np.isfinite(a[:6])):
                    continue

                a6 = np.clip(a[:6], -np.deg2rad(MAX_STEP_DEG), np.deg2rad(MAX_STEP_DEG))

                if ACTION_MODE == "absolute":
                    q_tgt = a6
                else:
                    q_tgt = q6 + a6  # Use q6 (6 joints) instead of q

                q_tgt = clamp_joint_limits(q_tgt)

                if DRY_RUN:
                    print("q_deg:", np.round(np.degrees(q6), 2),
                          "a_deg:", np.round(np.degrees(a6), 3),
                          "mode:", ACTION_MODE, flush=True)
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
        try:
            ctrl.speedStop()
        except Exception:
            pass
        try:
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
