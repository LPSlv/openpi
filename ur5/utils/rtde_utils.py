"""Shared RTDE utility functions for UR5 scripts."""

from __future__ import annotations

import time

import rtde_control
import rtde_receive


def safe_disconnect(obj) -> None:
    """Safely disconnect an RTDE interface."""
    try:
        obj.disconnect()
    except Exception:
        pass


def teardown_rtde_control(ctrl: rtde_control.RTDEControlInterface | None) -> None:
    """Deterministic RTDEControl teardown (best-effort).

    Order: speedStop → stopScript → disconnect (all ignore errors).
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
        safe_disconnect(ctrl)
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


def create_rtde_receive(host: str, *, frequency: float = 125.0, retries: int = 2) -> rtde_receive.RTDEReceiveInterface:
    """Create an RTDEReceiveInterface with a simple health check."""
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            rcv = rtde_receive.RTDEReceiveInterface(host, frequency=frequency)
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
        return create_rtde_receive(host, frequency=125.0, retries=2)
    try:
        _ = rcv.getActualQ()
        return rcv
    except Exception:
        try:
            safe_disconnect(rcv)
        except Exception:
            pass
        return create_rtde_receive(host, frequency=125.0, retries=2)


def create_rtde_control(host: str, *, retries: int = 1) -> rtde_control.RTDEControlInterface:
    """Create an RTDEControlInterface, with clear errors for common robot-side conflicts."""
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            ctrl = rtde_control.RTDEControlInterface(host)
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
