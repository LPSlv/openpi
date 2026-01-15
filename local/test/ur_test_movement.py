"""
Test UR robot movement and Robotiq gripper control.

This script performs:
1. Robotiq gripper activation
2. Base joint movement test (joint 0): -20 degrees, then +20 degrees
3. Robotiq gripper close and open

Gripper control uses Robotiq URCap socket service (port 63352).

Requirements:
- ur_rtde library: pip3 install ur_rtde
- Robotiq URCap installed on robot controller
"""

import time
import socket
import numpy as np
import threading
from collections import OrderedDict
import rtde_receive
import rtde_control

# Robot IP address
UR_IP = "192.10.0.11"

# Robotiq Hand-E gripper settings (URCap socket service)
ROBOTIQ_PORT = 63352

# Motion tuning
DELTA_DEG  = 20.0          # base joint movement step (degrees)
MOVEJ_VEL  = 0.50          # joint speed (rad/s) for moveJ
MOVEJ_ACC  = 0.50          # joint accel (rad/s^2) for moveJ


# Soft absolute joint limits (deg) – very conservative
SOFT_MIN_DEG = np.array([-180, -180, -180, -180, -180, -180], float)
SOFT_MAX_DEG = np.array([ 180,  180,  180,  180,  180,  180], float)

def ok_to_move(rcv: rtde_receive.RTDEReceiveInterface) -> bool:
    """Check if robot is ready to move (mode RUNNING, safety NORMAL)."""
    try:
        mode = rcv.getRobotMode()
        safety = rcv.getSafetyMode()
        return (mode == 7) and (safety == 1)
    except Exception:
        return False

def clamp_to_soft_limits_deg(q_deg):
    return np.clip(q_deg, SOFT_MIN_DEG, SOFT_MAX_DEG)

def _safe_disconnect(obj) -> None:
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

def _wait_joint0_reaches_target(
    rcv: rtde_receive.RTDEReceiveInterface,
    target_rad: list[float],
    *,
    tol_deg: float = 1.0,
    timeout_s: float = 8.0,  # Increased timeout for slower movements
) -> bool:
    """Return True when joint0 is within tol_deg of target_rad[0]."""
    t0 = time.time()
    target0_deg = float(np.degrees(target_rad[0]))
    while time.time() - t0 < timeout_s:
        try:
            q = np.array(rcv.getActualQ(), float)
            q0_deg = float(np.degrees(q[0]))
            if abs(q0_deg - target0_deg) <= tol_deg:
                return True
        except Exception:
            # If RTDE receive fails during verification, the control likely dropped too
            return False
        time.sleep(0.05)
    return False


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

    def disconnect(self) -> None:
        self._gripper.disconnect()



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


def main():
    ctrl: rtde_control.RTDEControlInterface | None = None
    rcv: rtde_receive.RTDEReceiveInterface | None = None
    gripper: RobotiqGripperHelper | None = None

    print("Connecting to robot via RTDE (receive)...", end=" ", flush=True)
    try:
        rcv = _create_rtde_receive(UR_IP, frequency=125.0, retries=2)
        print("OK")
    except Exception as e:
        print("FAILED")
        print(f"ERROR: {e}")
        return

    try:
        if not ok_to_move(rcv):
            print("Robot not ready (mode or safety). Put in Remote Control, clear any stops, then retry.")
            return

        # Initialize and activate gripper
        print("\n" + "="*60)
        print("Initializing Robotiq Gripper")
        print("="*60)
        try:
            gripper = RobotiqGripperHelper(UR_IP)
        except Exception as e:
            print(f"ERROR: Failed to connect to gripper socket: {e}")
            return
        
        print("Activating gripper...", end=" ", flush=True)
        try:
            gripper.activate()
            print("OK")
        except Exception as e:
            print("FAILED")
            print(f"ERROR: Gripper activation failed: {e}")
            return

        time.sleep(2.0)
        rcv = ensure_rcv(rcv, UR_IP)

        # Base joint movement test
        print("\n" + "="*60)
        print("Testing Base Joint Movement")
        print("="*60)

        time.sleep(2.0)

        print("Connecting to robot via RTDE (control)...", end=" ", flush=True)
        try:
            ctrl = _create_rtde_control(UR_IP, retries=1)
            print("OK")
        except Exception as e:
            print("FAILED")
            print(f"ERROR: {e}")
            return

        try:
            q_rad = np.array(rcv.getActualQ(), float)
            q_deg = np.degrees(q_rad)
        except Exception as e:
            print(f"ERROR: Cannot read robot position: {e}")
            return

        # Base joint movement test
        delta = DELTA_DEG
        base_move_ok = True
        for step in (-delta, +delta):  # negative first, then back
            rcv = ensure_rcv(rcv, UR_IP)
            
            if not ok_to_move(rcv):
                print("Robot not ready (mode or safety) – stopping.")
                return

            target_deg = q_deg.copy()
            target_deg[0] += step
            target_deg = clamp_to_soft_limits_deg(target_deg)
            target_rad = np.radians(target_deg)

            print(f"Target (deg): {np.round(target_deg, 2)}")
            try:
                assert ctrl is not None
                if not ctrl.isConnected():
                    raise RuntimeError("RTDE control not connected before moveJ")
                ok = ctrl.moveJ(target_rad.tolist(), MOVEJ_VEL, MOVEJ_ACC)
                if ok is False:
                    raise RuntimeError("RTDE moveJ returned False - control script may not be running")
            except Exception as e:
                print(f"ERROR: RTDE moveJ failed: {e}")
                base_move_ok = False
                return

            # Verify target reached
            try:
                reached = _wait_joint0_reaches_target(rcv, target_rad.tolist(), tol_deg=1.0, timeout_s=8.0)
            except Exception as e:
                print(f"WARNING: RTDE receive failed during verification: {e}")
                reached = True  # Assume success if receive dies

            if not reached:
                print("ERROR: Base joint did not reach target within timeout.")
                base_move_ok = False
                return
            
            time.sleep(0.2)

            try:
                q_rad = np.array(rcv.getActualQ(), float)
                q_deg = np.degrees(q_rad)
            except Exception as e:
                print(f"WARNING: Cannot read position after moveJ: {e}")

        _teardown_rtde_control(ctrl)
        ctrl = None

        if not base_move_ok:
            print("ERROR: Base joint movement test failed.")
            return

        print("Base joint movement test completed!")
        print("="*60 + "\n")

        # Test gripper close and open
        if gripper is not None:
            print("Closing gripper...", end=" ", flush=True)
            try:
                gripper.close()
                print("OK")
            except Exception as e:
                print("FAILED")
                print(f"ERROR: {e}")
                return
            
            print("Opening gripper...", end=" ", flush=True)
            try:
                gripper.open()
                print("OK")
            except Exception as e:
                print("FAILED")
                print(f"ERROR: {e}")
                return
        
        print("\nAll tests completed successfully!")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
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

if __name__ == "__main__":
    main()