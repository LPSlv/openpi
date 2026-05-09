"""Robotiq Hand-E gripper control via URCap socket service (port 63352)."""

from __future__ import annotations

import socket
import threading
import time
from collections import OrderedDict

import numpy as np

from ur5.defaults import ROBOTIQ_PORT


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
        """Receive one \\n-terminated line.

        Some responses (notably plain "ack") arrive without a newline, so the
        loop also returns once an ack is buffered. Each recv() uses a short
        timeout instead of the full socket timeout to avoid long stalls.
        """
        if self.socket is None:
            raise ConnectionError("Robotiq socket not connected")

        nl = self._rx_buf.find(b"\n")
        if nl != -1:
            line = bytes(self._rx_buf[:nl])
            del self._rx_buf[: nl + 1]
            return line.decode(self.ENCODING, errors="replace").strip()

        # bare "ack" already in the buffer
        if len(self._rx_buf) > 0:
            buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
            if buf_str == "ack":
                self._rx_buf.clear()
                return "ack"

        recv_timeout = 0.5  # per recv() call, in seconds
        original_timeout = self.socket.gettimeout()

        try:
            self.socket.settimeout(recv_timeout)
            t0 = time.time()
            grace_period = 0.1  # wait this long after data for a trailing newline

            while True:
                nl = self._rx_buf.find(b"\n")
                if nl != -1:
                    line = bytes(self._rx_buf[:nl])
                    del self._rx_buf[: nl + 1]
                    return line.decode(self.ENCODING, errors="replace").strip()

                if len(self._rx_buf) > 0:
                    buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
                    if "ack" in buf_str and time.time() - t0 > grace_period:
                        self._rx_buf.clear()
                        return "ack"

                try:
                    chunk = self.socket.recv(1024)
                    if not chunk:
                        raise ConnectionError("Robotiq socket closed by peer")
                    self._rx_buf.extend(chunk)
                    t0 = time.time()  # reset the grace window on fresh data
                except socket.timeout:
                    if len(self._rx_buf) > 0:
                        buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
                        if "ack" in buf_str:
                            self._rx_buf.clear()
                            return "ack"
                    # let the caller decide what to do with the timeout
                    raise
        finally:
            self.socket.settimeout(original_timeout)

    def _send_and_recv_line(self, cmd: str) -> str:
        if self.socket is None:
            raise ConnectionError("Robotiq socket not connected")
        with self._lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            return self._recv_line()

    @staticmethod
    def _is_ack(line: str) -> bool:
        return "ack" in line.strip().lower()

    def _set_vars(self, var_dict: OrderedDict[str, int]) -> None:
        # GTO needs an edge trigger, so when it appears in the batch we drop it
        # to 0 first then send the full batch with GTO=1; we send directly here
        # rather than going through _set_var to avoid recursion
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
        # poll until both ACT and STA report 0
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self._get_var(self.ACT) == 0 and self._get_var(self.STA) == 0:
                break
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
            time.sleep(0.1)
        time.sleep(0.5)

    def is_active(self) -> bool:
        return self._get_var(self.STA) == 3

    def activate(self) -> None:
        if self.is_active():
            return
        self._reset()
        self._set_var(self.ACT, 1)
        # wait for STA=3 (active)
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if self._get_var(self.ACT) == 1 and self._get_var(self.STA) == 3:
                return
            time.sleep(0.1)
        raise RuntimeError("Robotiq activation timed out (STA did not reach 3)")

    def move_and_wait(self, position: int, speed: int = 128, force: int = 64) -> None:
        position = int(np.clip(position, 0, 255))
        speed = int(np.clip(speed, 0, 255))
        force = int(np.clip(force, 0, 255))
        self._set_vars(OrderedDict([(self.POS, position), (self.SPE, speed), (self.FOR, force), (self.GTO, 1)]))

        # poll with a hard deadline so we never hang on a blocking GET
        deadline = time.time() + 12.0
        last_good_obj = None
        consecutive_failures = 0
        max_failures = 3

        while time.time() < deadline:
            try:
                obj = self._get_var(self.OBJ)
                last_good_obj = obj
                consecutive_failures = 0
                if obj in (1, 2, 3):
                    return
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    elapsed = time.time() - (deadline - 12.0)
                    if elapsed > 3.0:
                        try:
                            flt = self._get_var(self.FLT)
                            if flt == 0:
                                return
                        except Exception:
                            pass
            time.sleep(0.15)

        # timeout, gather diagnostics for the error message
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
            raise RuntimeError(f"Robotiq move timed out. Last OBJ={last_good_obj} (diagnostics failed: {diag_err})")

    def open(self) -> None:
        self.move_and_wait(0)

    def close(self) -> None:
        self.move_and_wait(255)


class RobotiqGripperHelper:
    """Helper class for Robotiq gripper control via URCap socket (port 63352)."""

    def __init__(self, host: str, port: int = ROBOTIQ_PORT):
        self._gripper = RobotiqGripperSocket(host, port)
        self._gripper.connect()

    def activate(self) -> None:
        self._gripper.activate()

    def open(self) -> None:
        self._gripper.open()

    def close(self) -> None:
        self._gripper.close()

    def move(self, position: int) -> None:
        """Move gripper to position (0-255, 0=open, 255=closed)."""
        self._gripper.move_and_wait(position)

    def move_normalized(self, position_01: float) -> None:
        """Move gripper to normalized position (0.0=open, 1.0=closed). Blocking."""
        position = int(np.clip(position_01 * 255, 0, 255))
        self.move(position)

    def send_normalized(self, position_01: float, speed: int = 255, force: int = 128) -> None:
        """Send gripper target without blocking (fire-and-forget).

        Unlike move_normalized, this returns immediately so the arm can
        keep moving while the gripper travels.
        """
        position = int(np.clip(position_01 * 255, 0, 255))
        speed = int(np.clip(speed, 0, 255))
        force = int(np.clip(force, 0, 255))
        self._gripper._set_vars(OrderedDict([
            (self._gripper.POS, position),
            (self._gripper.SPE, speed),
            (self._gripper.FOR, force),
            (self._gripper.GTO, 1),
        ]))

    def get_position(self) -> int:
        """Get current gripper position (0-255, 0=open, 255=closed)."""
        return self._gripper._get_var(self._gripper.PRE)

    def get_position_normalized(self) -> float:
        """Get current gripper position normalized (0.0=open, 1.0=closed)."""
        return float(self.get_position()) / 255.0

    def disconnect(self) -> None:
        self._gripper.disconnect()
