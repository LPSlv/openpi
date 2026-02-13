"""
Replay a freedrive waypoint trajectory on a UR5 (moveJ with blending) and record a raw dataset.

During replay, records at fixed FPS (default 20 Hz):
- external RGB image (256x256)
- optional wrist RGB image (256x256)
- robot proprio state (actual_q + gripper_cmd) => 7 dims
- action (absolute joints + absolute gripper_cmd) => 7 dims (forward-looking: action[i] = state[i+1])
- task string (language instruction / prompt)

Raw episode format (one folder per episode):
  <out_dir>/<episode_id>/
    meta.json
    waypoints.json   (copied in)
    steps.jsonl
    images/base/000000.jpg
    images/wrist/000000.jpg

Example:
  uv run python local/scripts/ur5_replay_and_record_raw.py \
    --ur_ip 192.10.0.11 \
    --waypoints_path raw_episodes/ur5_freedrive_.../waypoints.json \
    --rs_base_serial <SERIAL> --rs_wrist_serial <SERIAL> \
    --out_dir raw_episodes --prompt "pick up the block"
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import rtde_control
import rtde_receive
import tyro
import threading
from collections import OrderedDict

try:
    from openpi_client import image_tools
except ImportError:
    # Fallback implementation if openpi_client is not available
    def convert_to_uint8(img: np.ndarray) -> np.ndarray:
        """Converts an image to uint8 if it is a float image."""
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        return img.astype(np.uint8)

    def resize_with_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resizes an image to target size with padding to maintain aspect ratio."""
        if images.ndim == 3:
            h, w = images.shape[:2]
        else:
            h, w = images.shape[-3:-1]
        
        if h == height and w == width:
            return images
        
        # Calculate scaling to fit within target size
        scale = min(width / w, height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(images, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        if resized.ndim == 3:
            pad_h = (height - new_h) // 2
            pad_w = (width - new_w) // 2
            padded = np.zeros((height, width, resized.shape[2]), dtype=resized.dtype)
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        else:
            pad_h = (height - new_h) // 2
            pad_w = (width - new_w) // 2
            padded = np.zeros((*resized.shape[:-3], height, width, resized.shape[-1]), dtype=resized.dtype)
            padded[..., pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
        
        return padded
    
    class image_tools:
        convert_to_uint8 = staticmethod(convert_to_uint8)
        resize_with_pad = staticmethod(resize_with_pad)

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:  # pragma: no cover
    rs = None  # pyrealsense2 is optional until you use cameras

# Robotiq Hand-E gripper settings (URCap socket service)
ROBOTIQ_PORT = 63352


def _utcnow_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _safe_disconnect(obj) -> None:
    """Safely disconnect an RTDE interface."""
    try:
        obj.disconnect()
    except Exception:
        pass


def _teardown_rtde_control(ctrl: rtde_control.RTDEControlInterface | None) -> None:
    """Deterministic RTDEControl teardown (best-effort).

    Order:
    - speedStop() (ignore errors)
    - stopScript() if available (ignore errors)
    - disconnect() (ignore errors)
    """
    if ctrl is None:
        return
    try:
        if ctrl.isConnected():
            ctrl.speedStop()
    except Exception:
        pass
    try:
        stop_script = getattr(ctrl, "stopScript", None)
        if callable(stop_script):
            stop_script()
    except Exception:
        pass
    try:
        _safe_disconnect(ctrl)
    except Exception:
        pass


def ok_to_move(rcv: rtde_receive.RTDEReceiveInterface) -> bool:
    """Check if robot is ready to move (mode RUNNING, safety NORMAL)."""
    try:
        mode = rcv.getRobotMode()
        safety = rcv.getSafetyMode()
        return (mode == 7) and (safety == 1)
    except Exception:
        return False


def _create_rtde_receive(host: str, *, frequency: float = 125.0, retries: int = 2) -> rtde_receive.RTDEReceiveInterface:
    """Create an RTDEReceiveInterface with a simple health check."""
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            rcv = rtde_receive.RTDEReceiveInterface(host, frequency=frequency)
            # Give the internal thread a moment to start.
            time.sleep(0.2)
            _ = rcv.getActualQ()
            return rcv
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"RTDE receive connection failed: {last_err}")


def ensure_rcv(rcv: rtde_receive.RTDEReceiveInterface | None, host: str) -> rtde_receive.RTDEReceiveInterface:
    """Ensure RTDE receive is healthy, recreate if needed."""
    if rcv is None:
        return _create_rtde_receive(host, frequency=125.0, retries=2)
    try:
        _ = rcv.getActualQ()
        return rcv
    except Exception:
        try:
            _safe_disconnect(rcv)
        except Exception:
            pass
        return _create_rtde_receive(host, frequency=125.0, retries=2)


def _create_rtde_control(host: str, *, retries: int = 1) -> rtde_control.RTDEControlInterface:
    """Create an RTDEControlInterface, with clear errors for common robot-side conflicts."""
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            ctrl = rtde_control.RTDEControlInterface(host)
            # Give the control script a moment.
            time.sleep(0.2)
            if not ctrl.isConnected():
                raise RuntimeError("RTDE control script not running / not connected")
            return ctrl
        except Exception as e:
            last_err = e
            msg = str(e)
            if "RTDE input registers are already in use" in msg:
                raise RuntimeError(
                    "RTDE control cannot start because RTDE input registers are in use.\n"
                    "On the teach pendant disable Fieldbus adapters that reserve registers:\n"
                    "- Installation -> Fieldbus -> EtherNet/IP (disable)\n"
                    "- Installation -> Fieldbus -> PROFINET (disable)\n"
                    "- Installation -> Fieldbus -> MODBUS (disable any units)\n"
                    "Then fully reboot the robot controller and retry.\n"
                    "Also ensure no other RTDE client is connected."
                ) from e
            time.sleep(0.8)
    raise RuntimeError(f"RTDE control connection failed: {last_err}")


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

    def _set_vars(self, var_dict: "OrderedDict[str, int]") -> None:
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


def _send_urscript(host: str, script: str, *, port: int = 30002, timeout_sec: float = 3.0) -> None:
    with socket.create_connection((host, port), timeout=timeout_sec) as s:
        s.sendall(script.encode("utf-8"))


def _move_to_start_position(
    ur_ip: str,
    ctrl: rtde_control.RTDEControlInterface,
    rcv: rtde_receive.RTDEReceiveInterface,
    start_q_rad: list[float],
    vel: float = 0.2,
    acc: float = 0.3,
    timeout_sec: float = 10.0,
) -> None:
    """Move robot to starting position using RTDE moveJ."""
    target_q = np.asarray(start_q_rad, dtype=np.float64)
    
    # Use RTDE moveJ instead of URScript to avoid script conflicts
    success = ctrl.moveJ(start_q_rad, vel, acc)
    if not success:
        print("Warning: moveJ command failed, trying URScript method...")
        # Fallback to URScript if RTDE moveJ doesn't work
        script = f'movej({start_q_rad}, a={acc}, v={vel})\n'
        _send_urscript(ur_ip, script, timeout_sec=timeout_sec)
        # Wait a bit for script to start
        time.sleep(0.5)
    
    # Wait for movement to complete
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        current_q = np.asarray(rcv.getActualQ(), dtype=np.float64)
        dist = float(np.linalg.norm(current_q - target_q))
        if dist < 0.05:  # ~3 degrees tolerance
            print(f"Moved to start position (error: {np.degrees(dist):.2f} deg)")
            # Ensure any running script is stopped before proceeding
            time.sleep(0.2)
            return
        time.sleep(0.1)
    
    final_q = np.asarray(rcv.getActualQ(), dtype=np.float64)
    final_error = np.degrees(np.linalg.norm(final_q - target_q))
    print(f"Warning: Timeout waiting for start position (final error: {final_error:.2f} deg)")


def _process_bgr_to_rgb256(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = image_tools.resize_with_pad(rgb, 256, 256)
    return image_tools.convert_to_uint8(rgb)


def _start_rs_rgb(serial: str, *, w: int, h: int, fps: int) -> "rs.pipeline | None":
    if not serial:
        return None
    if rs is None:
        raise RuntimeError("pyrealsense2 is not available; install it or set empty serial(s).")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    pipe.start(cfg)
    return pipe


def _read_rs_rgb(pipe: "rs.pipeline", *, timeout_ms: int) -> np.ndarray | None:
    frames = pipe.wait_for_frames(timeout_ms)
    frame = frames.get_color_frame()
    if not frame:
        return None
    return _process_bgr_to_rgb256(np.asanyarray(frame.get_data()))


def _ensure_dirs(ep_dir: Path) -> tuple[Path, Path]:
    base_dir = ep_dir / "images" / "base"
    wrist_dir = ep_dir / "images" / "wrist"
    base_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, wrist_dir


def _write_jpg_rgb(path: Path, rgb: np.ndarray, *, quality: int = 95) -> None:
    # cv2 expects BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _build_movej_program(
    waypoints_q: list[list[float]],
    *,
    vel: float,
    acc: float,
    blend_radius: float,
    program_name: str = "openpi_replay",
    gripper_waypoints: list[float] | None = None,
    gripper_pause_sec: float = 1.5,
    gripper_stop_delay_sec: float = 0.3,
) -> str:
    if len(waypoints_q) < 2:
        raise ValueError("Need at least 2 waypoints to replay.")
    r = max(0.0, float(blend_radius))
    lines: list[str] = [f"def {program_name}():"]
    for i, q in enumerate(waypoints_q):
        # Remove blending for waypoint if gripper changes at next waypoint (ensures full stop)
        if gripper_waypoints is not None and i + 1 < len(gripper_waypoints):
            current_g = gripper_waypoints[i] if i < len(gripper_waypoints) else 0.0
            next_g = gripper_waypoints[i + 1]
            # If gripper changes significantly at next waypoint, remove blending to ensure full stop
            if abs(next_g - current_g) > 0.1:  # Significant gripper change (10% threshold)
                r_i = 0.0  # No blending - ensure full stop
            else:
                r_i = r if i < (len(waypoints_q) - 1) else 0.0
        else:
            r_i = r if i < (len(waypoints_q) - 1) else 0.0
        lines.append(f"  movej({q}, a={acc}, v={vel}, r={r_i})")
        # Add pause after waypoint if gripper changes at NEXT waypoint (offset by 1)
        # This gives time for gripper to complete before reaching the waypoint where gripper change occurs
        if gripper_waypoints is not None and i + 1 < len(gripper_waypoints):
            current_g = gripper_waypoints[i] if i < len(gripper_waypoints) else 0.0
            next_g = gripper_waypoints[i + 1]
            # If gripper changes significantly at next waypoint, add pause after current waypoint
            if abs(next_g - current_g) > 0.1:  # Significant gripper change (10% threshold)
                lines.append(f"  sleep({gripper_stop_delay_sec})  # Wait for movement to fully stop")
                lines.append(f"  sleep({gripper_pause_sec})  # Pause for gripper movement (gripper will change at waypoint {i+1}: {current_g:.2f} -> {next_g:.2f})")
    lines.append("end")
    lines.append(f"{program_name}()")
    lines.append("")
    return "\n".join(lines)


def _parse_gripper_waypoints(s: str) -> list[float] | None:
    s = s.strip()
    if not s:
        return None
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return [float(np.clip(v, 0.0, 1.0)) for v in vals]




@dataclasses.dataclass(frozen=True)
class Args:
    ur_ip: str = os.environ.get("UR_IP", "192.10.0.11")
    prompt: str = os.environ.get("PROMPT", "bus the table")

    # Input waypoints
    waypoints_path: Path = Path("raw_episodes/waypoints.json")

    # Output episode folder
    out_dir: Path = Path(os.environ.get("OUT_DIR", "raw_episodes"))
    episode_id: str = ""

    # Replay motion params (URScript moveJ)
    movej_vel: float = 0.1  # rad/s (reduced for slower, safer replay)
    movej_acc: float = 0.15  # rad/s^2 (reduced for slower, safer replay)
    blend_radius: float = 0.01  # meters in TCP space for moveL, but also accepted for moveJ blending on UR
    
    # Starting position (in degrees, will be converted to radians)
    move_to_start: bool = True
    start_position_deg: tuple[float, float, float, float, float, float] = (-90.0, -40.0, -140.0, -50.0, 90.0, 0.0)
    start_move_vel: float = 0.1  # rad/s
    start_move_acc: float = 0.15  # rad/s^2

    # RTDE streaming
    rtde_frequency_hz: float = 125.0

    # Cameras (RealSense)
    rs_base_serial: str = os.environ.get("RS_BASE", "137322074310")
    rs_wrist_serial: str = os.environ.get("RS_WRIST", "137322075008")
    rs_w: int = int(os.environ.get("RS_W", "640"))
    rs_h: int = int(os.environ.get("RS_H", "480"))
    rs_fps: int = int(os.environ.get("RS_FPS", "30"))
    rs_timeout_ms: int = int(os.environ.get("RS_TIMEOUT_MS", "10000"))
    fake_cam: bool = os.environ.get("FAKE_CAM", "0") == "1"

    # Dataset recording
    # NOTE: pi0 pretrained model expects 20 Hz for UR5e (see docs/norm_stats.md).
    fps: float = 20.0
    jpeg_quality: int = 95

    # Stop conditions
    final_q_tol: float = float(np.deg2rad(2.0))  # L2 rad threshold to consider "at goal"
    vel_norm_thresh: float = 0.03  # rad/s
    stop_settle_sec: float = 0.5
    max_seconds: float = 10.0 * 60.0
    
    # Gripper timing
    gripper_advance_distance: float = float(np.deg2rad(30.0))  # Send gripper command when this far from waypoint (rad) - gives time for gripper to complete
    gripper_pause_sec: float = 1.5  # Pause in URScript after waypoint where gripper changes (seconds)
    gripper_stop_delay_sec: float = 0.3  # Delay before pause to ensure movement fully stops (seconds)

    # Gripper (Robotiq URCap socket server)
    use_gripper: bool = True  # Enable gripper control by default
    robotiq_port: int = 63352
    gripper_default: float = 0.0
    gripper_waypoints: str = ""  # comma-separated list, length == number of waypoints (optional)
    gripper_debounce: float = 0.02


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episode_id = args.episode_id.strip() or _dt.datetime.now().strftime("ur5_replay_%Y%m%d_%H%M%S")
    ep_dir = args.out_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=False)
    base_dir, wrist_dir = _ensure_dirs(ep_dir)

    waypoints_obj = json.loads(args.waypoints_path.read_text())
    # Prioritize prompt from waypoints.json (if present), then use explicit --prompt/PROMPT env var, then fallback
    # This ensures the original waypoint prompt is preserved unless explicitly overridden
    waypoint_prompt = waypoints_obj.get("prompt")
    if waypoint_prompt:
        prompt = waypoint_prompt
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = "do something"
    waypoints = waypoints_obj["waypoints"]
    waypoints_q: list[list[float]] = [list(map(float, w["q"])) for w in waypoints]
    q_goal = np.asarray(waypoints_q[-1], dtype=np.float64)

    # Extract gripper positions from waypoints if available, otherwise use command line argument
    gripper_wp = None
    if waypoints and "gripper" in waypoints[0]:
        # Gripper positions are recorded in waypoints
        gripper_wp = [float(w.get("gripper", args.gripper_default)) for w in waypoints]
        print(f"Found gripper positions in waypoints ({len(gripper_wp)} waypoints)")
    else:
        # Fall back to command line argument
        gripper_wp = _parse_gripper_waypoints(args.gripper_waypoints)
        if gripper_wp is not None:
            print(f"Using gripper positions from command line ({len(gripper_wp)} waypoints)")
    
    if gripper_wp is not None and len(gripper_wp) != len(waypoints_q):
        raise ValueError(f"Gripper waypoints length ({len(gripper_wp)}) must match waypoints ({len(waypoints_q)})")

    # Start cameras
    base_pipe = None
    wrist_pipe = None
    last_base_rgb = None
    last_wrist_rgb = None
    if args.fake_cam:
        last_base_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        last_wrist_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    else:
        base_pipe = _start_rs_rgb(args.rs_base_serial, w=args.rs_w, h=args.rs_h, fps=args.rs_fps)
        wrist_pipe = _start_rs_rgb(args.rs_wrist_serial, w=args.rs_w, h=args.rs_h, fps=args.rs_fps) if args.rs_wrist_serial else None

    # Connect to robot via RTDE
    print("Connecting to robot via RTDE (receive)...", end=" ", flush=True)
    try:
        rcv = _create_rtde_receive(args.ur_ip, frequency=args.rtde_frequency_hz, retries=2)
        print("OK")
    except Exception as e:
        print("FAILED")
        print(f"ERROR: {e}")
        return
    
    # Check if robot is ready to move
    if not ok_to_move(rcv):
        print("ERROR: Robot not ready (mode or safety). Put in Remote Control, clear any stops, then retry.")
        try:
            _safe_disconnect(rcv)
        except Exception:
            pass
        return
    
    print("Connecting to robot via RTDE (control)...", end=" ", flush=True)
    try:
        ctrl = _create_rtde_control(args.ur_ip, retries=1)
        print("OK")
    except Exception as e:
        print("FAILED")
        print(f"ERROR: {e}")
        try:
            _safe_disconnect(rcv)
        except Exception:
            pass
        return
    
    # Move to starting position if requested
    if args.move_to_start:
        start_q_rad = [float(np.deg2rad(d)) for d in args.start_position_deg]
        print(f"Moving to start position: {args.start_position_deg} degrees")
        try:
            _move_to_start_position(
                args.ur_ip,
                ctrl,
                rcv,
                start_q_rad,
                vel=args.start_move_vel,
                acc=args.start_move_acc,
            )
            # Small delay to ensure movement is complete and controller is ready
            time.sleep(0.5)
        except Exception as e:
            print(f"Warning: Failed to move to start position: {e}")
            print("Continuing anyway...")

    # Initialize gripper
    g_cmd = float(np.clip(args.gripper_default, 0.0, 1.0))
    last_g_sent: float | None = None
    use_gripper = args.use_gripper  # Local variable that can be modified
    
    # Auto-enable gripper if gripper positions are found in waypoints
    if gripper_wp is not None and not args.use_gripper:
        print("Note: Gripper positions found in waypoints, but --use_gripper is False.")
        print("      Gripper will not be controlled during replay. Use --use_gripper to enable.")
    
    gripper: RobotiqGripperHelper | None = None
    if args.use_gripper:
        print("Initializing Robotiq Gripper...", end=" ", flush=True)
        try:
            gripper = RobotiqGripperHelper(args.ur_ip)
            gripper.activate()
            gripper.move_normalized(g_cmd)
            last_g_sent = g_cmd
            print("OK")
        except Exception as e:
            print("FAILED")
            print(f"Warning: Failed to init Robotiq gripper via URCap socket: {e}")
            print("Continuing without gripper control. Gripper is disabled by default.")
            use_gripper = False
            if gripper is not None:
                try:
                    gripper.disconnect()
                except Exception:
                    pass
                gripper = None

    meta = {
        "kind": "ur5_replay_raw_episode",
        "created_at": _utcnow_iso(),
        "prompt": prompt,
        "fps": args.fps,
        "ur_ip": args.ur_ip,
        "waypoints_path": str(args.waypoints_path),
        "movej_vel": args.movej_vel,
        "movej_acc": args.movej_acc,
        "blend_radius": args.blend_radius,
        "rs_base_serial": args.rs_base_serial,
        "rs_wrist_serial": args.rs_wrist_serial,
        "rs_w": args.rs_w,
        "rs_h": args.rs_h,
        "rs_fps": args.rs_fps,
        "jpeg_quality": args.jpeg_quality,
        "use_gripper": use_gripper,
        "robotiq_port": args.robotiq_port,
        "gripper_default": g_cmd,
        "gripper_waypoints": gripper_wp,
        "state_spec": {"dtype": "float32", "shape": [7], "desc": "actual_q(6) + gripper_cmd(1)"},
        "action_spec": {"dtype": "float32", "shape": [7], "desc": "absolute_q(6) + absolute gripper_cmd(1)"},
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (ep_dir / "waypoints.json").write_text(json.dumps(waypoints_obj, indent=2))

    # Ensure any running script is stopped before sending replay program
    try:
        _send_urscript(args.ur_ip, "stop\n")
        time.sleep(0.3)
    except Exception:
        pass  # Ignore if this fails
    
    # Send replay program (moveJ with blending)
    program = _build_movej_program(
        waypoints_q,
        vel=args.movej_vel,
        acc=args.movej_acc,
        blend_radius=args.blend_radius,
        gripper_waypoints=gripper_wp,
        gripper_pause_sec=args.gripper_pause_sec,
        gripper_stop_delay_sec=args.gripper_stop_delay_sec,
    )
    _send_urscript(args.ur_ip, program)

    steps_path = ep_dir / "steps.jsonl"
    f_steps = steps_path.open("w", encoding="utf-8")

    pending_step: dict | None = None  # buffered step; action filled in next tick
    i = 0
    t0 = time.time()
    next_tick = t0
    final_stable_since: float | None = None
    reached_wp_idx = -1
    gripper_sent_for_wp = -1  # Track which waypoint we've sent gripper command for

    try:
        while True:
            now = time.time()
            if now - t0 > args.max_seconds:
                print(f"Reached max_seconds={args.max_seconds:.1f}; stopping.")
                break

            # Fixed-FPS loop
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            tick_time = time.time()
            next_tick = next_tick + (1.0 / float(args.fps))

            # Read cameras (best-effort; reuse last frame if missing)
            if args.fake_cam:
                assert last_base_rgb is not None and last_wrist_rgb is not None
                base_rgb = last_base_rgb
                wrist_rgb = last_wrist_rgb
            else:
                base_rgb = None
                wrist_rgb = None
                if base_pipe is not None:
                    base_rgb = _read_rs_rgb(base_pipe, timeout_ms=args.rs_timeout_ms)
                if wrist_pipe is not None:
                    wrist_rgb = _read_rs_rgb(wrist_pipe, timeout_ms=args.rs_timeout_ms)
                if base_rgb is None:
                    base_rgb = last_base_rgb
                if wrist_rgb is None:
                    wrist_rgb = last_wrist_rgb
                if base_rgb is None and wrist_rgb is None:
                    # No images yet; skip this tick.
                    continue
                if base_rgb is None:
                    base_rgb = wrist_rgb
                if wrist_rgb is None:
                    wrist_rgb = base_rgb
                last_base_rgb = base_rgb
                last_wrist_rgb = wrist_rgb

            # Ensure RTDE receive is healthy
            rcv = ensure_rcv(rcv, args.ur_ip)
            
            # Read proprio
            q = np.asarray(rcv.getActualQ(), dtype=np.float64)
            qd = np.asarray(rcv.getActualQd(), dtype=np.float64)

            # Track waypoint index progression and (optionally) send gripper commands
            dists = [float(np.linalg.norm(q - np.asarray(qw, dtype=np.float64))) for qw in waypoints_q]
            nearest = int(np.argmin(dists))
            
            # Update reached waypoint index when actually at waypoint
            if nearest > reached_wp_idx and dists[nearest] < args.final_q_tol * 1.5:
                reached_wp_idx = nearest
                # Update g_cmd from waypoints if available (for accurate state recording)
                if gripper_wp is not None:
                    g_cmd = float(gripper_wp[nearest])
            
            # Send gripper command early based on waypoint progression (offset by 1)
            # Send gripper command for waypoint N+1 when we've reached waypoint N
            # This ensures the gripper command is sent early enough
            if gripper_wp is not None and use_gripper and gripper is not None:
                # Target waypoint for gripper command: reached_wp_idx + 1 (offset by 1)
                target_wp_idx = reached_wp_idx + 1
                if target_wp_idx < len(gripper_wp) and reached_wp_idx > gripper_sent_for_wp:
                    target_g_cmd = float(gripper_wp[target_wp_idx])
                    # Only send if different from last sent command
                    if last_g_sent is None or abs(target_g_cmd - last_g_sent) > args.gripper_debounce:
                        try:
                            print(f"Sending gripper command for waypoint {target_wp_idx} (reached waypoint {reached_wp_idx}): {target_g_cmd:.3f}")
                            gripper.move_normalized(target_g_cmd)
                            last_g_sent = target_g_cmd
                            gripper_sent_for_wp = reached_wp_idx  # Track that we sent command for target_wp_idx when at reached_wp_idx
                        except Exception as e:
                            print(f"Warning: Gripper move failed: {e}")

            # Build state (absolute joint positions + gripper)
            state = np.asarray([*q.tolist(), g_cmd], dtype=np.float32)  # (7,)

            # Save images
            base_rel = Path("images/base") / f"{i:06d}.jpg"
            wrist_rel = Path("images/wrist") / f"{i:06d}.jpg"
            _write_jpg_rgb(ep_dir / base_rel, base_rgb, quality=args.jpeg_quality)
            _write_jpg_rgb(ep_dir / wrist_rel, wrist_rgb, quality=args.jpeg_quality)

            # Forward-looking absolute actions: action[i] = state[i+1].
            # Write the previous step now that we know its forward action.
            if pending_step is not None:
                pending_step["actions"] = state.tolist()
                f_steps.write(json.dumps(pending_step) + "\n")
                f_steps.flush()

            # Buffer current step (action will be filled on next tick)
            pending_step = {
                "i": i,
                "t_wall": tick_time,
                "q_actual": q.tolist(),
                "qd_actual": qd.tolist(),
                "gripper_cmd": float(g_cmd),
                "state": state.tolist(),
                "image_path": str(base_rel),
                "wrist_image_path": str(wrist_rel),
                "task": prompt,
            }
            i += 1

            # Stop condition: near final waypoint and stable
            final_dist = float(np.linalg.norm(q - q_goal))
            vel_norm = float(np.linalg.norm(qd))
            at_goal = final_dist < args.final_q_tol
            stable = vel_norm < args.vel_norm_thresh
            if at_goal and stable:
                if final_stable_since is None:
                    final_stable_since = tick_time
                elif tick_time - final_stable_since >= args.stop_settle_sec:
                    break
            else:
                final_stable_since = None

    finally:
        # Write last buffered step (action = hold current position)
        if pending_step is not None:
            pending_step["actions"] = pending_step["state"]
            f_steps.write(json.dumps(pending_step) + "\n")
            f_steps.flush()
        f_steps.close()
        _teardown_rtde_control(ctrl)
        try:
            if rcv is not None:
                _safe_disconnect(rcv)
        except Exception:
            pass
        try:
            if gripper is not None:
                gripper.disconnect()
        except Exception:
            pass
        if not args.fake_cam:
            try:
                if base_pipe is not None:
                    base_pipe.stop()
            except Exception:
                pass
            try:
                if wrist_pipe is not None:
                    wrist_pipe.stop()
            except Exception:
                pass

    print(f"Wrote raw episode: {ep_dir} (frames={i})")


if __name__ == "__main__":
    main(tyro.cli(Args))

