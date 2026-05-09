"""
Record a human-guided UR5 freedrive trajectory as sparse joint-space waypoints.

This script:
- Enables teach/freedrive mode (so you can physically move the arm)
- Streams joint state via RTDE
- Saves sparse joint-space waypoints when motion has settled

Output is a simple, raw-on-disk format that can be replayed and converted later.

Example:
  uv run python ur5/scripts/ur5_record_freedrive_waypoints.py \
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

# put the repo root on sys.path so `ur5` resolves when this is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import rtde_control
import rtde_receive
import socket
import tyro

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None

from ur5 import defaults as _defaults
from ur5.utils.robotiq_gripper import RobotiqGripperHelper
from ur5.utils.rtde_utils import (
    safe_disconnect as _safe_disconnect,
    teardown_rtde_control as _teardown_rtde_control,
)


def _utcnow_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _start_rs_rgb(serial: str, *, w: int, h: int, fps: int) -> "rs.pipeline | None":
    """Start a RealSense RGB pipeline with fixed exposure (matches camera_test.py)."""
    if not serial:
        return None
    if rs is None:
        raise RuntimeError("pyrealsense2 is not available; install it or set empty serial(s).")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    prof = pipe.start(cfg)
    try:
        for s in prof.get_device().sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                s.set_option(rs.option.enable_auto_exposure, 0)
                s.set_option(rs.option.exposure, 100.0)
                break
    except Exception:
        pass
    return pipe


def _read_rs_bgr(pipe: "rs.pipeline", *, timeout_ms: int = 100) -> np.ndarray | None:
    """Non-blocking read of a BGR frame from a RealSense pipeline."""
    try:
        frames = pipe.wait_for_frames(timeout_ms)
        frame = frames.get_color_frame()
        if not frame:
            return None
        return np.asanyarray(frame.get_data())
    except Exception:
        return None


def _send_urscript(host: str, script: str, *, port: int = 30002, timeout_sec: float = 3.0) -> None:
    """Send URScript to robot controller."""
    with socket.create_connection((host, port), timeout=timeout_sec) as s:
        s.sendall(script.encode("utf-8"))


def _enable_freedrive_urscript(ur_ip: str) -> None:
    """Enable freedrive via a long-running URScript."""
    try:
        _send_urscript(ur_ip, "stop\n", timeout_sec=1.0)
        time.sleep(0.3)
    except Exception:
        pass

    # the script must keep running to hold freedrive on
    script = """def freedrive_program():
    freedrive_mode()
    while True:
        sync()
    end
freedrive_program()
"""
    _send_urscript(ur_ip, script)
    time.sleep(1.0)  # give the controller a moment to pick it up


def _disable_freedrive_urscript(ur_ip: str) -> None:
    """End freedrive by pushing a short URScript on top of the holding one."""
    script = """def stop_freedrive():
    end_freedrive_mode()
end
stop_freedrive()
"""
    try:
        _send_urscript(ur_ip, script)
        time.sleep(0.5)
    except Exception:
        # last resort, just tell the controller to stop
        try:
            _send_urscript(ur_ip, "stop\n")
            time.sleep(0.5)
        except Exception:
            pass


def _move_to_start_position(
    ur_ip: str,
    ctrl: rtde_control.RTDEControlInterface,
    rcv: rtde_receive.RTDEReceiveInterface,
    start_q_rad: list[float],
    vel: float = 0.2,
    acc: float = 0.3,
    timeout_sec: float = 10.0,
) -> None:
    """Move robot to starting position via RTDE moveJ (URScript fallback)."""
    target_q = np.asarray(start_q_rad, dtype=np.float64)

    # prefer RTDE moveJ over URScript so we don't conflict with a running script
    success = ctrl.moveJ(start_q_rad, vel, acc)
    if not success:
        print("Warning: moveJ command failed, trying URScript method...")
        script = f"movej({start_q_rad}, a={acc}, v={vel})\n"
        _send_urscript(ur_ip, script, timeout_sec=timeout_sec)
        time.sleep(0.5)  # let the URScript come up

    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        current_q = np.asarray(rcv.getActualQ(), dtype=np.float64)
        dist = float(np.linalg.norm(current_q - target_q))
        if dist < 0.05:  # ~3 deg tolerance
            print(f"Moved to start position (error: {np.degrees(dist):.2f} deg)")
            time.sleep(0.2)  # let the controller quiesce
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
        char = sys.stdin.read(1)
        return char.lower() if char else None
    except Exception:
        return None


def _maybe_read_stdin_line() -> str | None:
    """Non-blocking line read; fallback when single-char input isn't usable."""
    if not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not r:
        return None
    try:
        line = sys.stdin.readline()
        if line:
            line = line.strip()
            return line if line else None
        return None
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class Args:
    ur_ip: str = _defaults.UR_IP
    out_dir: Path = Path(_defaults.OUT_DIR)
    episode_id: str = ""
    prompt: str = os.environ.get("PROMPT", "pick up the blue block and place it in the cardboard box")

    # RTDE sampling
    rtde_frequency_hz: float = 125.0

    # waypoint extraction
    settle_sec: float = 0.20
    # freedrive reports small non-zero joint velocities, so 0.03 is often too
    # strict and only the first waypoint gets through; 0.06 works in practice
    vel_norm_thresh: float = 0.06  # rad/s, L2 norm of qd
    min_joint_dist: float = float(np.deg2rad(1.0))  # rad, L2 distance between waypoints
    min_time_between_waypoints_sec: float = 0.40

    # optional extra fields
    record_tcp_pose: bool = True
    gripper_default: float = 0.0  # metadata only; actuation happens in replay

    # gripper control
    use_gripper: bool = True
    robotiq_port: int = _defaults.ROBOTIQ_PORT

    # realsense cameras for live preview
    rs_base_serial: str = _defaults.RS_BASE_SERIAL
    rs_wrist_serial: str = _defaults.RS_WRIST_SERIAL
    rs_w: int = _defaults.RS_W
    rs_h: int = _defaults.RS_H
    rs_fps: int = _defaults.RS_FPS

    # safety / UX
    max_seconds: float = 30.0 * 60.0
    print_every_sec: float = 0.5

    # starting position in degrees, converted to radians at use
    move_to_start: bool = True
    start_position_deg: tuple[float, float, float, float, float, float] = _defaults.START_POSITION_DEG
    start_move_vel: float = 0.2
    start_move_acc: float = 0.3


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

    # surface robot mode early so misconfiguration is obvious
    try:
        robot_mode = rcv.getRobotMode()
        safety_mode = rcv.getSafetyMode()
        print(f"Robot mode: {robot_mode}, Safety mode: {safety_mode}")
        if robot_mode != 7:  # 7 == RUNNING_MODE
            print(f"Warning: Robot is not in RUNNING mode (current: {robot_mode})")
            print("Robot should be in Remote Control mode for teach mode to work.")
    except Exception as e:
        print(f"Warning: Could not read robot state: {e}")

    waypoints: list[dict] = []
    last_wp_q: np.ndarray | None = None
    last_wp_time: float | None = None
    stable_since: float | None = None

    gripper: RobotiqGripperHelper | None = None
    use_gripper = args.use_gripper  # may flip to False if init fails
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

    base_pipe = None
    wrist_pipe = None
    if rs is not None:
        if args.rs_base_serial:
            print(f"Starting base camera: {args.rs_base_serial}...", end=" ", flush=True)
            try:
                base_pipe = _start_rs_rgb(args.rs_base_serial, w=args.rs_w, h=args.rs_h, fps=args.rs_fps)
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
        if args.rs_wrist_serial:
            print(f"Starting wrist camera: {args.rs_wrist_serial}...", end=" ", flush=True)
            try:
                wrist_pipe = _start_rs_rgb(args.rs_wrist_serial, w=args.rs_w, h=args.rs_h, fps=args.rs_fps)
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")

    print(
        "\nUR5 freedrive waypoint recorder\n"
        f"- Saving to: {ep_dir}\n"
        "- Put the robot in a safe state.\n"
    )
    if use_gripper and gripper is not None:
        print("- Gripper control: Press 'o' (open) or 'c' (close) - no Enter needed!")
        print("  (If that doesn't work, try: 'o' + Enter or 'c' + Enter)\n")
    
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
            time.sleep(0.5)  # let the controller settle before continuing
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
        # clear any running script before enabling teach mode, otherwise the
        # start-position URScript can keep ownership of the controller
        try:
            _send_urscript(args.ur_ip, "stop\n")
            time.sleep(0.3)
        except Exception:
            pass

        print("Enabling teach/freedrive mode...")
        teach_mode_enabled = False

        # try the RTDE API first
        try:
            if hasattr(ctrl, "teachMode"):
                result = ctrl.teachMode()
                if result:
                    print("Teach mode enabled via RTDE.")
                    teach_mode_enabled = True
                else:
                    print("RTDE teachMode() returned False, trying URScript...")
            else:
                print("teachMode() method not available, trying URScript...")
        except Exception as e:
            print(f"RTDE teachMode() failed: {e}, trying URScript...")

        # fall back to URScript
        if not teach_mode_enabled:
            try:
                print("Enabling freedrive mode via URScript...")
                _enable_freedrive_urscript(args.ur_ip)
                time.sleep(0.5)
                try:
                    # if the read works, freedrive is up
                    _ = rcv.getActualQ()
                    print("Freedrive mode enabled via URScript.")
                    print("You should now be able to move the robot manually.")
                    teach_mode_enabled = True
                except Exception as e:
                    print(f"Warning: Could not verify freedrive mode: {e}")
                    print("Freedrive script was sent, but verification failed.")
                    print("The robot may still be in freedrive mode - try moving it manually.")
                    teach_mode_enabled = True  # script was sent, assume it took
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

            # try single-char input first, fall back to line input
            char_input = _maybe_read_stdin_char()
            line_input = None
            if char_input is None:
                line_input = _maybe_read_stdin_line()

            user_input = char_input if char_input is not None else line_input
            if user_input is not None:
                user_input_lower = user_input.lower().strip()

                # ignore a bare Enter so the user doesn't accidentally stop
                if not user_input_lower or user_input_lower in ["\n", "\r"]:
                    continue

                print(f"[DEBUG: Received input: '{user_input_lower}']")

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
                        print(f"Unknown command: '{user_input_lower}'. Use 'o' (open), 'c' (close), or Ctrl+C to stop.")
                else:
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
                        if use_gripper and gripper is not None:
                            try:
                                gripper_pos = gripper.get_position_normalized()
                                wp["gripper"] = float(gripper_pos)
                            except Exception as e:
                                # if the read fails, keep going with the configured default
                                print(f"Warning: Could not read gripper position: {e}")
                                wp["gripper"] = args.gripper_default
                        else:
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

            # live camera preview
            if base_pipe is not None or wrist_pipe is not None:
                base_bgr = _read_rs_bgr(base_pipe) if base_pipe is not None else None
                wrist_bgr = _read_rs_bgr(wrist_pipe) if wrist_pipe is not None else None
                if base_bgr is not None and wrist_bgr is not None:
                    cv2.imshow("Base | Wrist", np.hstack([base_bgr, wrist_bgr]))
                elif base_bgr is not None:
                    cv2.imshow("Base", base_bgr)
                elif wrist_bgr is not None:
                    cv2.imshow("Wrist", wrist_bgr)
                cv2.waitKey(1)
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print("Disabling teach/freedrive mode...")
        try:
            try:
                ctrl.endTeachMode()
                print("Teach mode disabled via RTDE.")
            except (AttributeError, Exception):
                _disable_freedrive_urscript(args.ur_ip)
                print("Freedrive mode disabled via URScript.")
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
        cv2.destroyAllWindows()

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

