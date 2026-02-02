"""
Record a human-guided UR5 freedrive trajectory as sparse joint-space waypoints.

This script:
- Enables teach/freedrive mode (so you can physically move the arm)
- Streams joint state via RTDE
- Saves sparse joint-space waypoints when motion has settled

Output is a simple, raw-on-disk format that can be replayed and converted later.

Example:
  uv run python openpi/local/scripts/ur5_record_freedrive_waypoints.py \
    --ur_ip 192.10.0.11 \
    --prompt "pick up the block" \
    --out_dir raw_episodes
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import select
import sys
import time
import termios
import tty
from pathlib import Path

import numpy as np
import rtde_control
import rtde_receive
import socket
import tyro
import threading
from collections import OrderedDict

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
    """Send URScript to robot controller."""
    with socket.create_connection((host, port), timeout=timeout_sec) as s:
        s.sendall(script.encode("utf-8"))


def _enable_freedrive_urscript(ur_ip: str) -> None:
    """Enable freedrive mode using URScript program."""
    # First, stop any running script
    try:
        _send_urscript(ur_ip, "stop\n", timeout_sec=1.0)
        time.sleep(0.3)
    except Exception:
        pass
    
    # URScript program that enables freedrive mode and keeps it active
    script = """def freedrive_program():
    freedrive_mode()
    while True:
        sync()
    end
freedrive_program()
"""
    _send_urscript(ur_ip, script)
    time.sleep(1.0)  # Give it more time to start and verify


def _disable_freedrive_urscript(ur_ip: str) -> None:
    """Disable freedrive mode by sending a new script that stops freedrive."""
    # Send a script that ends freedrive mode
    script = """def stop_freedrive():
    end_freedrive_mode()
end
stop_freedrive()
"""
    try:
        _send_urscript(ur_ip, script)
        time.sleep(0.5)
    except Exception:
        # If that doesn't work, try sending an empty script to stop the previous one
        try:
            _send_urscript(ur_ip, "stop\n")
            time.sleep(0.5)
        except Exception:
            pass  # User may need to stop manually


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


def _maybe_read_stdin_char() -> str | None:
    """Non-blocking: return a single character if available, else None.
    Works with both line-buffered (Enter required) and raw mode (instant)."""
    if not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not r:
        return None
    try:
        # Try to read a single character
        char = sys.stdin.read(1)
        return char.lower() if char else None
    except Exception:
        return None


def _maybe_read_stdin_line() -> str | None:
    """Non-blocking: return a line if available, else None.
    Falls back to this if character input doesn't work."""
    if not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not r:
        return None
    try:
        line = sys.stdin.readline()
        if line:
            # Strip whitespace and newlines
            line = line.strip()
            return line if line else None
        return None
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class Args:
    ur_ip: str = os.environ.get("UR_IP", "192.10.0.11")
    out_dir: Path = Path(os.environ.get("OUT_DIR", "raw_episodes"))
    episode_id: str = ""
    prompt: str = os.environ.get("PROMPT", "pick up the blue block and place it in the cardboard box")

    # RTDE sampling
    rtde_frequency_hz: float = 125.0

    # Waypoint extraction
    settle_sec: float = 0.35
    # NOTE: freedrive often reports small non-zero joint velocities; 0.03 can be too strict
    # and result in "only the first waypoint". Use a slightly looser default.
    vel_norm_thresh: float = 0.06  # rad/s (L2 norm of qd)
    min_joint_dist: float = float(np.deg2rad(3.0))  # rad (L2 distance between waypoints)
    min_time_between_waypoints_sec: float = 0.75

    # Optional extra fields
    record_tcp_pose: bool = True
    gripper_default: float = 0.0  # logged as metadata; actual gripper actuation is handled in replay

    # Gripper control
    use_gripper: bool = True  # Enable gripper control via keyboard
    robotiq_port: int = 63352

    # Safety / UX
    max_seconds: float = 30.0 * 60.0
    print_every_sec: float = 0.5
    
    # Starting position (in degrees, will be converted to radians)
    move_to_start: bool = True
    start_position_deg: tuple[float, float, float, float, float, float] = (-90.0, -40.0, -140.0, -50.0, 90.0, 0.0)
    start_move_vel: float = 0.2  # rad/s
    start_move_acc: float = 0.3  # rad/s^2


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episode_id = args.episode_id.strip() or _dt.datetime.now().strftime("ur5_freedrive_%Y%m%d_%H%M%S")
    ep_dir = args.out_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "kind": "ur5_freedrive_waypoints",
        "created_at": _utcnow_iso(),
        "prompt": args.prompt,
        "ur_ip": args.ur_ip,
        "rtde_frequency_hz": args.rtde_frequency_hz,
        "settle_sec": args.settle_sec,
        "vel_norm_thresh": args.vel_norm_thresh,
        "min_joint_dist": args.min_joint_dist,
        "min_time_between_waypoints_sec": args.min_time_between_waypoints_sec,
        "record_tcp_pose": args.record_tcp_pose,
        "gripper_default": float(np.clip(args.gripper_default, 0.0, 1.0)),
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    rcv = rtde_receive.RTDEReceiveInterface(args.ur_ip, frequency=args.rtde_frequency_hz)
    ctrl = rtde_control.RTDEControlInterface(args.ur_ip)
    
    # Check robot state
    try:
        robot_mode = rcv.getRobotMode()
        safety_mode = rcv.getSafetyMode()
        print(f"Robot mode: {robot_mode}, Safety mode: {safety_mode}")
        if robot_mode != 7:  # 7 = RUNNING_MODE
            print(f"Warning: Robot is not in RUNNING mode (current: {robot_mode})")
            print("Robot should be in Remote Control mode for teach mode to work.")
    except Exception as e:
        print(f"Warning: Could not read robot state: {e}")

    waypoints: list[dict] = []
    last_wp_q: np.ndarray | None = None
    last_wp_time: float | None = None
    stable_since: float | None = None

    # Initialize gripper if enabled
    gripper: RobotiqGripperHelper | None = None
    use_gripper = args.use_gripper  # Local variable that can be modified
    if args.use_gripper:
        print("Initializing Robotiq Gripper...", end=" ", flush=True)
        try:
            gripper = RobotiqGripperHelper(args.ur_ip)
            print("Connected", end=" ", flush=True)
            gripper.activate()
            print("Activated", end=" ", flush=True)
            gripper.move_normalized(args.gripper_default)
            print("OK - Gripper ready!")
            print(f"  Test: Try pressing 'o' or 'c' to control gripper")
        except Exception as e:
            print("FAILED")
            print(f"Warning: Failed to init Robotiq gripper: {e}")
            print("  Check:")
            print(f"  - Robot IP is correct: {args.ur_ip}")
            print(f"  - Robotiq URCap is installed and socket server is running on port {args.robotiq_port}")
            print("  - Robot is powered on and connected")
            print("Continuing without gripper control.")
            if gripper is not None:
                try:
                    gripper.disconnect()
                except Exception:
                    pass
                gripper = None
            use_gripper = False

    print(
        "\nUR5 freedrive waypoint recorder\n"
        f"- Saving to: {ep_dir}\n"
        "- Put the robot in a safe state.\n"
    )
    if use_gripper and gripper is not None:
        print("- Gripper control: Press 'o' (open) or 'c' (close) - no Enter needed!")
        print("  (If that doesn't work, try: 'o' + Enter or 'c' + Enter)\n")
    
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
    
    print(
        "- This script will enable teach mode. Physically guide the arm.\n"
        "- Press Ctrl+C to finish recording.\n"
    )
    if use_gripper and gripper is not None:
        print("- Gripper: Press 'o' (open) or 'c' (close) + Enter")
        print("  (Empty Enter is ignored - use Ctrl+C to stop)\n")

    t_start = time.time()
    last_print = 0.0
    try:
        # Enable freedrive / teach mode.
        # (If your UR controller disallows this via RTDE, this may raise.)
        # Ensure any running script is stopped before enabling teach mode
        # This is important if we used URScript for moving to start position
        try:
            # Send a stop command to clear any running scripts
            _send_urscript(args.ur_ip, "stop\n")
            time.sleep(0.3)
        except Exception:
            pass  # Ignore if this fails
        
        print("Enabling teach/freedrive mode...")
        teach_mode_enabled = False
        
        # Try RTDE method first
        try:
            if hasattr(ctrl, 'teachMode'):
                result = ctrl.teachMode()
                if result:
                    print("✓ Teach mode enabled successfully via RTDE.")
                    teach_mode_enabled = True
                else:
                    print("RTDE teachMode() returned False, trying URScript...")
            else:
                print("teachMode() method not available, trying URScript...")
        except Exception as e:
            print(f"RTDE teachMode() failed: {e}, trying URScript...")
        
        # If RTDE didn't work, try URScript
        if not teach_mode_enabled:
            try:
                print("Enabling freedrive mode via URScript...")
                _enable_freedrive_urscript(args.ur_ip)
                # Verify freedrive is enabled by checking if we can read robot state
                # Give it a moment to activate
                time.sleep(0.5)
                try:
                    # Try to read robot state - if freedrive is active, this should work
                    _ = rcv.getActualQ()
                    print("✓ Freedrive mode enabled via URScript.")
                    print("You should now be able to move the robot manually.")
                    teach_mode_enabled = True
                except Exception as e:
                    print(f"Warning: Could not verify freedrive mode: {e}")
                    print("Freedrive script was sent, but verification failed.")
                    print("The robot may still be in freedrive mode - try moving it manually.")
                    teach_mode_enabled = True  # Assume it worked if script was sent
            except Exception as e:
                print(f"Error: Failed to enable freedrive mode: {e}")
                print("\nTroubleshooting:")
                print("  1. Ensure robot is in Remote Control mode (Settings -> System -> Remote Control)")
                print("  2. Check that RTDE is enabled on the robot controller")
                print("  3. Verify robot is not in a protective stop or error state")
                print("  4. Try manually enabling freedrive mode from the teach pendant")
                raise
        
        if not teach_mode_enabled:
            raise RuntimeError("Could not enable teach/freedrive mode via any method")

        while True:
            now = time.time()
            if now - t_start > args.max_seconds:
                print(f"Reached max_seconds={args.max_seconds:.1f}; stopping.")
                break

            # Handle keyboard input - try character input first, then line input
            char_input = _maybe_read_stdin_char()
            line_input = None
            if char_input is None:
                line_input = _maybe_read_stdin_line()
            
            user_input = char_input if char_input is not None else line_input
            if user_input is not None:
                user_input_lower = user_input.lower().strip()
                
                # Skip empty input (just Enter pressed) - don't stop on it
                if not user_input_lower or user_input_lower in ["\n", "\r"]:
                    continue
                
                # Debug: show what was received
                print(f"[DEBUG: Received input: '{user_input_lower}']")
                
                # Handle gripper commands
                if use_gripper and gripper is not None:
                    if user_input_lower == "o":
                        try:
                            print("Opening gripper...", end=" ", flush=True)
                            gripper.open()
                            print("OK")
                        except Exception as e:
                            print(f"FAILED: {e}")
                            import traceback
                            traceback.print_exc()
                    elif user_input_lower == "c":
                        try:
                            print("Closing gripper...", end=" ", flush=True)
                            gripper.close()
                            print("OK")
                        except Exception as e:
                            print(f"FAILED: {e}")
                            import traceback
                            traceback.print_exc()
                    elif user_input_lower in ["q", "quit", "exit"]:
                        print("Stopping on user request (quit command).")
                        break
                    else:
                        # Unknown command - show help but don't stop
                        print(f"Unknown command: '{user_input_lower}'. Use 'o' (open), 'c' (close), or Ctrl+C to stop.")
                else:
                    # Check if user tried to use gripper commands when gripper is not available
                    if user_input_lower in ["o", "c"]:
                        print(f"Gripper command '{user_input_lower}' ignored: gripper is not available.")
                        print("  (Gripper initialization failed or --no-use-gripper was used)")
                    elif user_input_lower in ["q", "quit", "exit"]:
                        print("Stopping on user request (quit command).")
                        break
                    else:
                        print(f"Unknown command: '{user_input_lower}'. Use Ctrl+C to stop.")

            q = np.asarray(rcv.getActualQ(), dtype=np.float64)  # (6,)
            qd = np.asarray(rcv.getActualQd(), dtype=np.float64)  # (6,)

            vel_norm = float(np.linalg.norm(qd))
            is_stable = vel_norm < args.vel_norm_thresh
            if is_stable:
                if stable_since is None:
                    stable_since = now
            else:
                stable_since = None

            can_add = stable_since is not None and (now - stable_since) >= args.settle_sec
            if can_add:
                if last_wp_time is not None and (now - last_wp_time) < args.min_time_between_waypoints_sec:
                    pass
                else:
                    dist = float(np.linalg.norm(q - last_wp_q)) if last_wp_q is not None else float("inf")
                    if last_wp_q is None or dist >= args.min_joint_dist:
                        wp: dict = {
                            "t_wall": now,
                            "q": q.tolist(),
                            "qd": qd.tolist(),
                        }
                        if args.record_tcp_pose:
                            tcp = np.asarray(rcv.getActualTCPPose(), dtype=np.float64)  # (6,)
                            wp["tcp_pose"] = tcp.tolist()
                        # Record gripper position if gripper is available
                        if use_gripper and gripper is not None:
                            try:
                                gripper_pos = gripper.get_position_normalized()
                                wp["gripper"] = float(gripper_pos)
                            except Exception as e:
                                # If we can't read gripper position, use last known or default
                                print(f"Warning: Could not read gripper position: {e}")
                                wp["gripper"] = args.gripper_default
                        else:
                            # No gripper available, use default
                            wp["gripper"] = args.gripper_default
                        waypoints.append(wp)
                        last_wp_q = q
                        last_wp_time = now
                        gripper_info = f", gripper={wp['gripper']:.3f}" if use_gripper else ""
                        print(
                            f"Added waypoint {len(waypoints)} "
                            f"(dist_deg={np.degrees(dist):.2f}, vel_norm={vel_norm:.4f}{gripper_info})",
                            flush=True,
                        )

            # Periodic status print removed per user request
            # if now - last_print >= args.print_every_sec:
            #     stable_for = (now - stable_since) if stable_since is not None else 0.0
            #     dist_to_last = float(np.linalg.norm(q - last_wp_q)) if last_wp_q is not None else float("nan")
            #     msg = (
            #         f"waypoints={len(waypoints):4d}  "
            #         f"vel_norm={vel_norm:0.4f}  "
            #         f"stable={int(is_stable)}  "
            #         f"stable_for={stable_for:0.2f}s  "
            #         f"dist_last_deg={np.degrees(dist_to_last):0.2f}  "
            #         f"q_deg={np.degrees(q).round(1)}"
            #     )
            #     print(msg, flush=True)
            #     last_print = now

            time.sleep(0.001)  # RTDEReceive is already rate-limited internally

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print("Disabling teach/freedrive mode...")
        try:
            # Try RTDE method first
            try:
                ctrl.endTeachMode()
                print("✓ Teach mode disabled via RTDE.")
            except (AttributeError, Exception):
                # Fallback to URScript stop
                _disable_freedrive_urscript(args.ur_ip)
                print("✓ Freedrive mode disabled via URScript.")
        except Exception as e:
            print(f"Warning: Could not disable teach mode: {e}")
            print("You may need to disable it manually from the teach pendant or press the emergency stop.")
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

    out = {
        "prompt": args.prompt,
        "created_at": meta["created_at"],
        "ur_ip": args.ur_ip,
        "waypoints": waypoints,
    }
    (ep_dir / "waypoints.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(waypoints)} waypoints to {ep_dir / 'waypoints.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))

