"""UR5 + RealSense bridge for the websocket policy server."""

import os
import socket
import sys
import threading
import time
from collections import OrderedDict

# OpenCV GUI in Docker can break with:
#   "The X11 connection broke (error 1). Did the X11 server die?"
# A common cause is Qt using the MIT-SHM X11 extension from inside a container.
# Disable it by default (can be overridden by explicitly setting QT_X11_NO_MITSHM=0).
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
# Also force the Qt platform plugin to XCB for reliability inside containers.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

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

# Optional: desired wall-clock duration per policy "chunk" (seconds).
# If set and HOLD_PER_STEP is NOT explicitly set, we derive HOLD_PER_STEP as:
#   HOLD_PER_STEP = INFER_PERIOD / HORIZON_STEPS
_infer_period_raw = os.environ.get("INFER_PERIOD")
INFER_PERIOD: float | None = None
if _infer_period_raw not in (None, ""):
    try:
        INFER_PERIOD = float(_infer_period_raw)
    except ValueError as e:
        raise ValueError(f"Invalid INFER_PERIOD={_infer_period_raw!r}, expected float seconds") from e

HOLD_PER_STEP = float(os.environ.get("HOLD_PER_STEP", "0.15"))
HORIZON_STEPS = int(os.environ.get("HORIZON_STEPS", "10"))

if INFER_PERIOD is not None and "HOLD_PER_STEP" not in os.environ:
    if HORIZON_STEPS <= 0:
        raise ValueError(f"HORIZON_STEPS must be > 0 when using INFER_PERIOD, got {HORIZON_STEPS}")
    HOLD_PER_STEP = INFER_PERIOD / float(HORIZON_STEPS)

ACTION_MODE = os.environ.get("ACTION_MODE", "delta").lower()
MAX_STEP_DEG = float(os.environ.get("MAX_STEP_DEG", "2.0"))

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# Robotiq Hand-E gripper settings (URCap socket service)
ROBOTIQ_PORT = 63352
GRIPPER_PORT = int(os.environ.get("GRIPPER_PORT", str(ROBOTIQ_PORT)))
USE_GRIPPER = os.environ.get("USE_GRIPPER", "1") == "1"
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
                    error_str = str(e).lower()
                    if "x11" in error_str or "connection" in error_str or "display" in error_str:
                        print("WARNING: X11 connection test failed. Image preview disabled.", file=sys.stderr)
                        print("INFO: Robot control will work without image display.", file=sys.stderr)
                    else:
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


class RobotiqGripperSocket:
    """Robotiq gripper control via URCap socket service (port 63352)."""

    # WRITE VARIABLES (also readable)
    ACT = "ACT"
    GTO = "GTO"
    ATR = "ATR"
    ADR = "ADR"
    FOR = "FOR"
    SPE = "SPE"
    POS = "POS"

    # READ VARIABLES
    STA = "STA"  # 0 reset, 1 activating, 3 active
    PRE = "PRE"
    OBJ = "OBJ"
    FLT = "FLT"

    ENCODING = "UTF-8"

    def __init__(self, host: str, port: int = ROBOTIQ_PORT, *, socket_timeout: float = 8.0):
        self.host = host
        self.port = int(port)
        self.socket_timeout = float(socket_timeout)
        self.socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._rx_buf = bytearray()

    def connect(self) -> None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.socket_timeout)
            s.connect((self.host, self.port))
            self.socket = s
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Robotiq socket at {self.host}:{self.port}: {e}") from e

    def disconnect(self) -> None:
        if self.socket is None:
            return
        try:
            self.socket.close()
        finally:
            self.socket = None
            self._rx_buf.clear()

    def _recv_line(self) -> str:
        """Receive a single \\n-terminated line (stripped).
        
        Also handles responses without newline (e.g., "ack" without \\n).
        Uses short timeouts per recv() call to avoid blocking for full socket timeout.
        """
        if self.socket is None:
            raise ConnectionError("Robotiq socket not connected")
        
        # Check if we already have a complete line
        nl = self._rx_buf.find(b"\n")
        if nl != -1:
            line = bytes(self._rx_buf[:nl])
            del self._rx_buf[: nl + 1]
            return line.decode(self.ENCODING, errors="replace").strip()
        
        # Check if buffer contains "ack" without newline (common case)
        if len(self._rx_buf) > 0:
            buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
            if buf_str == "ack":
                self._rx_buf.clear()
                return "ack"
        
        # Use shorter timeout per recv() call to avoid blocking for full 8s
        recv_timeout = 0.5  # 500ms per recv() call
        original_timeout = self.socket.gettimeout()
        
        try:
            self.socket.settimeout(recv_timeout)
            t0 = time.time()
            grace_period = 0.1  # 100ms grace period for newline after receiving data
            
            while True:
                # Check for newline
                nl = self._rx_buf.find(b"\n")
                if nl != -1:
                    line = bytes(self._rx_buf[:nl])
                    del self._rx_buf[: nl + 1]
                    return line.decode(self.ENCODING, errors="replace").strip()
                
                # Check if buffer contains "ack" (with or without newline)
                if len(self._rx_buf) > 0:
                    buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
                    if "ack" in buf_str:
                        # If we have "ack" and grace period expired, return it
                        if time.time() - t0 > grace_period:
                            self._rx_buf.clear()
                            return "ack"
                
                # Try to receive more data with short timeout
                try:
                    chunk = self.socket.recv(1024)
                    if not chunk:
                        raise ConnectionError("Robotiq socket closed by peer")
                    self._rx_buf.extend(chunk)
                    t0 = time.time()  # Reset grace period when we receive data
                except socket.timeout:
                    # If we have "ack" in buffer and timeout, return it
                    if len(self._rx_buf) > 0:
                        buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
                        if "ack" in buf_str:
                            self._rx_buf.clear()
                            return "ack"
                    # Otherwise, raise timeout - caller can handle it
                    raise
        finally:
            # Restore original timeout
            self.socket.settimeout(original_timeout)

    def _send_and_recv_line(self, cmd: str) -> str:
        if self.socket is None:
            raise ConnectionError("Robotiq socket not connected")
        with self._lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            return self._recv_line()

    @staticmethod
    def _is_ack(line: str) -> bool:
        """Check if response contains 'ack' (handles variations with/without newline)."""
        return "ack" in line.strip().lower()

    def _set_vars(self, var_dict: OrderedDict[str, int]) -> None:
        # Edge-trigger fix: set GTO 0 first if GTO is in the command, then set all vars with GTO 1
        # Do NOT call _set_var() here to avoid infinite recursion - send directly
        if self.GTO in var_dict:
            line0 = self._send_and_recv_line("SET GTO 0\n")
            if not self._is_ack(line0):
                raise RuntimeError(f"Robotiq SET GTO 0 not acknowledged: {line0!r}")
            time.sleep(0.02)
        
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {int(value)}"
        cmd += "\n"
        line = self._send_and_recv_line(cmd)
        if not self._is_ack(line):
            raise RuntimeError(f"Robotiq SET not acknowledged: {line!r}")

    def _set_var(self, variable: str, value: int) -> None:
        self._set_vars(OrderedDict([(variable, int(value))]))

    def _get_var(self, variable: str) -> int:
        line = self._send_and_recv_line(f"GET {variable}\n")
        parts = line.split()
        if len(parts) != 2:
            raise RuntimeError(f"Unexpected GET response: {line!r}")
        var_name, value_str = parts
        if var_name != variable:
            raise RuntimeError(f"Unexpected GET response {line!r}: expected '{variable}'")
        return int(value_str)

    def _reset(self) -> None:
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        # Wait until ACT=0 and STA=0
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self._get_var(self.ACT) == 0 and self._get_var(self.STA) == 0:
                break
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
            time.sleep(0.1)  # Reduced polling frequency
        time.sleep(0.5)

    def is_active(self) -> bool:
        return self._get_var(self.STA) == 3

    def activate(self) -> None:
        if self.is_active():
            return
        self._reset()
        self._set_var(self.ACT, 1)
        # Wait until active (STA=3)
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if self._get_var(self.ACT) == 1 and self._get_var(self.STA) == 3:
                return
            time.sleep(0.1)  # Reduced polling frequency
        raise RuntimeError("Robotiq activation timed out (STA did not reach 3)")

    def move_and_wait(self, position: int, speed: int = 128, force: int = 64) -> None:
        position = int(np.clip(position, 0, 255))
        speed = int(np.clip(speed, 0, 255))
        force = int(np.clip(force, 0, 255))
        # Send command once
        self._set_vars(OrderedDict([(self.POS, position), (self.SPE, speed), (self.FOR, force), (self.GTO, 1)]))
        
        # Poll lightly with hard deadline to avoid hanging forever on blocking GETs
        deadline = time.time() + 12.0
        last_good_obj = None
        consecutive_failures = 0
        max_failures = 3  # Allow a few failures before giving up
        
        while time.time() < deadline:
            try:
                obj = self._get_var(self.OBJ)
                last_good_obj = obj
                consecutive_failures = 0  # Reset on success
                # Primary completion check: OBJ == 3 (at target) or OBJ in {1,2} (stopped due to contact)
                if obj in (1, 2, 3):
                    return
            except Exception:
                # URCap not responding right now
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    # Too many failures in a row - gripper might be stuck or URCap unresponsive
                    # If we've waited a reasonable time, assume success if no fault
                    elapsed = time.time() - (deadline - 12.0)
                    if elapsed > 3.0:  # After 3 seconds, if no response, assume it's working
                        # Try one more quick check, then give up
                        try:
                            flt = self._get_var(self.FLT)
                            if flt == 0:  # No fault - assume success
                                return
                        except Exception:
                            pass  # Even diagnostics failed, but we've waited long enough
            time.sleep(0.15)  # Poll every 150ms - reduced from 200ms for faster response
        
        # Timeout - try to get diagnostics once (but don't block forever if it fails)
        try:
            sta = self._get_var(self.STA)
            flt = self._get_var(self.FLT)
            obj = self._get_var(self.OBJ)
            pre = self._get_var(self.PRE)
            act = self._get_var(self.ACT)
            gto = self._get_var(self.GTO)
            raise RuntimeError(
                f"Robotiq move timed out after 12s\n"
                f"Diagnostics: STA={sta}, FLT={flt}, OBJ={obj}, PRE={pre} (target={position}), ACT={act}, GTO={gto}"
            )
        except Exception as diag_err:
            # If diagnostics also fail, just report the last known OBJ value
            raise RuntimeError(f"Robotiq move timed out. Last OBJ={last_good_obj} (diagnostics failed: {diag_err})")

    def open(self) -> None:
        self.move_and_wait(0)

    def close(self) -> None:
        self.move_and_wait(255)


class RobotiqGripperHelper:
    """Helper class for Robotiq gripper control via URCap socket (port 63352)."""
    
    def __init__(self, host: str):
        """Initialize gripper helper.
        
        Args:
            host: Robot IP address
        """
        self._gripper = RobotiqGripperSocket(host, ROBOTIQ_PORT)
        self._gripper.connect()
    
    def activate(self) -> None:
        """Activate the gripper.
        
        Raises:
            ConnectionError: If connection to robot fails
            TimeoutError: If connection times out
        """
        self._gripper.activate()
    
    def open(self) -> None:
        """Open the gripper fully.
        
        Raises:
            ConnectionError: If connection to robot fails
            TimeoutError: If connection times out
        """
        self._gripper.open()
    
    def close(self) -> None:
        """Close the gripper fully.
        
        Raises:
            ConnectionError: If connection to robot fails
            TimeoutError: If connection times out
        """
        self._gripper.close()
    
    def move(self, position: int) -> None:
        """Move gripper to specific position.
        
        Args:
            position: Gripper position (0-255, where 0 is fully open, 255 is fully closed)
        
        Raises:
            ConnectionError: If connection to robot fails
            TimeoutError: If connection times out
        """
        self._gripper.move_and_wait(position)

    def move_normalized(self, position_01: float) -> None:
        """Move gripper to normalized position (0.0 = open, 1.0 = closed).
        
        Args:
            position_01: Gripper position normalized to [0.0, 1.0]
        """
        position = int(np.clip(position_01 * 255, 0, 255))
        self.move(position)

    def get_position(self) -> int:
        """Get current gripper position (0-255, where 0 is fully open, 255 is fully closed).
        
        Returns:
            Current gripper position as integer (0-255)
        """
        return self._gripper._get_var(self._gripper.PRE)

    def get_position_normalized(self) -> float:
        """Get current gripper position normalized (0.0 = open, 1.0 = closed).
        
        Returns:
            Current gripper position normalized to [0.0, 1.0]
        """
        pos = self.get_position()
        return float(pos) / 255.0

    def disconnect(self) -> None:
        self._gripper.disconnect()


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
            time.sleep(0.5)  # Increased delay after reaching target position
            return
        time.sleep(0.1)

    final_q = np.asarray(rcv.getActualQ(), dtype=np.float64)
    final_error = np.degrees(np.linalg.norm(final_q - target_q))
    print(f"Warning: Timeout waiting for reset position (final error: {final_error:.2f} deg)")


def main() -> None:
    global _has_gui  # Allow disabling GUI if X11 connection fails
    
    print("=" * 60)
    print("Initializing UR5 Robot Bridge")
    print("=" * 60)
    
    # Initialize cameras
    print("Initializing cameras...", end=" ", flush=True)
    base_cam = _start_rgb(RS_BASE)
    wrist_cam = _start_rgb(RS_WRIST)
    if base_cam is not None or wrist_cam is not None:
        print("OK")
        # Give cameras time to start streaming
        time.sleep(1.0)
    else:
        print("SKIPPED (FAKE_CAM or no cameras configured)")
    
    # Initialize RTDE connections
    print("Connecting to robot via RTDE...", end=" ", flush=True)
    rcv = rtde_receive.RTDEReceiveInterface(UR_IP, frequency=125.0)
    ctrl = rtde_control.RTDEControlInterface(UR_IP)
    print("OK")
    # Give RTDE time to establish connection
    time.sleep(0.5)
    
    # Connect to policy server
    print(f"Connecting to policy server at {POLICY_HOST}:{POLICY_PORT}...", end=" ", flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(host=POLICY_HOST, port=POLICY_PORT)
    print("OK")
    # Give websocket time to establish connection
    time.sleep(0.5)

    # Get reset_pose from server metadata
    print("Fetching policy configuration...", end=" ", flush=True)
    metadata = client.get_server_metadata()
    reset_pose = metadata.get("reset_pose")
    # Print which checkpoint/config the server claims it is serving (added by scripts/serve_policy.py).
    cfg = metadata.get("train_config")
    ckpt = metadata.get("checkpoint_dir")
    ns = metadata.get("norm_stats_dir")
    if cfg or ckpt:
        print(f"\nPolicy server reports: train_config={cfg!r} checkpoint_dir={ckpt!r}", flush=True)
    if ns:
        print(f"Policy server reports: norm_stats_dir={ns!r}", flush=True)
    if reset_pose is None:
        # Default reset position: same as dataset gathering "0 position"
        # (-90.0, -70.0, -120.0, -80.0, 90.0, 0.0) degrees in radians
        reset_pose = [-1.5708, -1.2217, -2.0944, -1.3963, 1.5708, 0.0]
        print("OK (using default reset position)")
    else:
        print(f"OK (reset_pose from metadata)")
    
    # Move to reset position at startup
    print("Moving robot to reset position...", end=" ", flush=True)
    _move_to_reset_position(ctrl, rcv, reset_pose)
    print("OK")
    # Wait for robot to fully settle at reset position
    print("Waiting for robot to settle...", end=" ", flush=True)
    time.sleep(5.0)  # Extended delay to ensure robot is fully settled
    print("OK")
    
    # Initialize gripper if enabled
    gripper: RobotiqGripperHelper | None = None
    if USE_GRIPPER and GRIPPER_PORT == ROBOTIQ_PORT:
        print("Initializing Robotiq Gripper...", end=" ", flush=True)
        try:
            gripper = RobotiqGripperHelper(UR_IP)
            time.sleep(1.0)  # Give gripper socket time to connect
            print("Activating gripper...", end=" ", flush=True)
            gripper.activate()
            time.sleep(5.0)  # Extended delay to ensure gripper is fully activated
            print("OK. Opening gripper...", end=" ", flush=True)
            gripper.open()  # Open gripper at startup
            time.sleep(5.0)  # Extended delay to ensure gripper is fully open
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
    elif GRIPPER_PORT > 0 and GRIPPER_PORT != ROBOTIQ_PORT:
        print(f"Warning: GRIPPER_PORT={GRIPPER_PORT} is not the Robotiq port ({ROBOTIQ_PORT}).")
        print("Gripper control disabled. Set GRIPPER_PORT={ROBOTIQ_PORT} or USE_GRIPPER=0 to disable.")
    else:
        print("Gripper: DISABLED")
    
    # Final setup delay to ensure everything is ready
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
    time.sleep(0.5)  # Brief pause before starting control loop

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
                    # Check if this is an X11 connection error
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
                        # X11 connection broke - disable GUI for the rest of this session
                        if _has_gui:  # Only print message once
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
                        # Other error - log but continue trying (only if GUI still enabled)
                        if _has_gui:
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
                    if not DRY_RUN and gripper is not None:
                        try:
                            gripper.move_normalized(g)
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
