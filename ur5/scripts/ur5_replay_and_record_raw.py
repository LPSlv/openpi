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
  uv run python ur5/scripts/ur5_replay_and_record_raw.py \
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
import sys
import time
from pathlib import Path

# put the repo root on sys.path so `ur5` resolves when this is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import rtde_control
import rtde_receive
import tyro

from ur5 import defaults as _defaults
from ur5.utils.robotiq_gripper import RobotiqGripperHelper
from ur5.utils.rtde_utils import (
    safe_disconnect as _safe_disconnect,
    teardown_rtde_control as _teardown_rtde_control,
    ok_to_move,
    create_rtde_receive as _create_rtde_receive,
    ensure_rcv,
    create_rtde_control as _create_rtde_control,
)

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
        """Resize keeping aspect ratio, padding the rest to the target size."""
        if images.ndim == 3:
            h, w = images.shape[:2]
        else:
            h, w = images.shape[-3:-1]

        if h == height and w == width:
            return images

        scale = min(width / w, height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(images, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # center the resized image inside a zero-padded canvas
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
    rs = None  # pyrealsense2 is only required when cameras are actually used

def _utcnow_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


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
    """Move robot to starting position via RTDE moveJ (URScript fallback)."""
    target_q = np.asarray(start_q_rad, dtype=np.float64)

    # prefer RTDE moveJ over URScript so we don't conflict with a running script
    success = ctrl.moveJ(start_q_rad, vel, acc)
    if not success:
        print("Warning: moveJ command failed, trying URScript method...")
        script = f"movej({start_q_rad}, a={acc}, v={vel})\n"
        _send_urscript(ur_ip, script, timeout_sec=timeout_sec)
        time.sleep(0.5)  # give the URScript a moment to start

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
    # cv2 wants BGR
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
        # if the next waypoint changes the gripper, drop the blend radius here
        # so the arm fully stops before the gripper command fires
        if gripper_waypoints is not None and i + 1 < len(gripper_waypoints):
            current_g = gripper_waypoints[i] if i < len(gripper_waypoints) else 0.0
            next_g = gripper_waypoints[i + 1]
            if abs(next_g - current_g) > 0.1:  # >10% change counts as significant
                r_i = 0.0
            else:
                r_i = r if i < (len(waypoints_q) - 1) else 0.0
        else:
            r_i = r if i < (len(waypoints_q) - 1) else 0.0
        lines.append(f"  movej({q}, a={acc}, v={vel}, r={r_i})")
        # and pause after this waypoint so the gripper has time to finish
        # before we hit the next waypoint where it actually changes
        if gripper_waypoints is not None and i + 1 < len(gripper_waypoints):
            current_g = gripper_waypoints[i] if i < len(gripper_waypoints) else 0.0
            next_g = gripper_waypoints[i + 1]
            if abs(next_g - current_g) > 0.1:
                lines.append(f"  sleep({gripper_stop_delay_sec})  # let movement stop")
                lines.append(f"  sleep({gripper_pause_sec})  # gripper change at waypoint {i+1}: {current_g:.2f} -> {next_g:.2f}")
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
    ur_ip: str = _defaults.UR_IP
    prompt: str = os.environ.get("PROMPT", "bus the table")

    # input waypoints
    waypoints_path: Path = Path("raw_episodes/waypoints.json")

    # output episode folder
    out_dir: Path = Path(_defaults.OUT_DIR)
    episode_id: str = ""

    # replay motion params (URScript moveJ)
    movej_vel: float = 0.1  # rad/s, kept low for safer replay
    movej_acc: float = 0.15  # rad/s^2
    blend_radius: float = 0.01  # meters of TCP blending (UR also accepts this for moveJ)

    # starting position in degrees, converted to radians at use
    move_to_start: bool = True
    start_position_deg: tuple[float, float, float, float, float, float] = (-90.0, -40.0, -140.0, -50.0, 90.0, 0.0)
    start_move_vel: float = 0.1
    start_move_acc: float = 0.15

    # RTDE streaming
    rtde_frequency_hz: float = 125.0

    # realsense cameras
    rs_base_serial: str = _defaults.RS_BASE_SERIAL
    rs_wrist_serial: str = _defaults.RS_WRIST_SERIAL
    rs_w: int = int(os.environ.get("RS_W", "640"))
    rs_h: int = int(os.environ.get("RS_H", "480"))
    rs_fps: int = int(os.environ.get("RS_FPS", "30"))
    rs_timeout_ms: int = int(os.environ.get("RS_TIMEOUT_MS", "10000"))
    fake_cam: bool = os.environ.get("FAKE_CAM", "0") == "1"

    # dataset recording
    # pi0 pretrained ur5e expects 20 Hz, see docs/norm_stats.md
    fps: float = 10.0
    jpeg_quality: int = 95

    # stop conditions
    final_q_tol: float = float(np.deg2rad(2.0))  # L2 rad threshold for "at goal"
    vel_norm_thresh: float = 0.03  # rad/s
    stop_settle_sec: float = 0.5
    max_seconds: float = 10.0 * 60.0

    # gripper timing
    gripper_advance_distance: float = float(np.deg2rad(30.0))  # send gripper command this far ahead so it has time to finish
    gripper_pause_sec: float = 1.5  # URScript sleep after a waypoint that changes the gripper
    gripper_stop_delay_sec: float = 0.3  # short sleep before that pause so motion fully stops first

    # gripper (Robotiq URCap socket server)
    use_gripper: bool = True
    robotiq_port: int = _defaults.ROBOTIQ_PORT
    gripper_default: float = 0.0
    gripper_waypoints: str = ""  # optional comma-separated list, must match the number of waypoints
    gripper_debounce: float = 0.02


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episode_id = args.episode_id.strip() or _dt.datetime.now().strftime("ur5_replay_%Y%m%d_%H%M%S")
    ep_dir = args.out_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=False)
    base_dir, wrist_dir = _ensure_dirs(ep_dir)

    waypoints_obj = json.loads(args.waypoints_path.read_text())
    # prefer the prompt baked into waypoints.json so the original recording
    # context survives, fall back to --prompt/PROMPT, then a generic default
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

    # gripper positions can come from waypoints.json or the --gripper_waypoints flag
    gripper_wp = None
    if waypoints and "gripper" in waypoints[0]:
        gripper_wp = [float(w.get("gripper", args.gripper_default)) for w in waypoints]
        print(f"Found gripper positions in waypoints ({len(gripper_wp)} waypoints)")
    else:
        gripper_wp = _parse_gripper_waypoints(args.gripper_waypoints)
        if gripper_wp is not None:
            print(f"Using gripper positions from command line ({len(gripper_wp)} waypoints)")
    
    if gripper_wp is not None and len(gripper_wp) != len(waypoints_q):
        raise ValueError(f"Gripper waypoints length ({len(gripper_wp)}) must match waypoints ({len(waypoints_q)})")

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

    print("Connecting to robot via RTDE (receive)...", end=" ", flush=True)
    try:
        rcv = _create_rtde_receive(args.ur_ip, frequency=args.rtde_frequency_hz, retries=2)
        print("OK")
    except Exception as e:
        print("FAILED")
        print(f"ERROR: {e}")
        return

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

    g_cmd = float(np.clip(args.gripper_default, 0.0, 1.0))
    last_g_sent: float | None = None
    use_gripper = args.use_gripper  # may flip to False below if init fails

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

    # stop any running URScript before we push our replay program
    try:
        _send_urscript(args.ur_ip, "stop\n")
        time.sleep(0.3)
    except Exception:
        pass

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

    pending_step: dict | None = None  # buffered step, action gets filled on the next tick
    i = 0
    t0 = time.time()
    next_tick = t0
    final_stable_since: float | None = None
    reached_wp_idx = -1
    gripper_sent_for_wp = -1  # last waypoint index we issued a gripper command for

    try:
        while True:
            now = time.time()
            if now - t0 > args.max_seconds:
                print(f"Reached max_seconds={args.max_seconds:.1f}; stopping.")
                break

            # fixed-FPS pacing
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            tick_time = time.time()
            next_tick = next_tick + (1.0 / float(args.fps))

            # cameras are best-effort, reuse the last frame if a poll missed
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
                    # no images yet, skip the tick
                    continue
                if base_rgb is None:
                    base_rgb = wrist_rgb
                if wrist_rgb is None:
                    wrist_rgb = base_rgb
                last_base_rgb = base_rgb
                last_wrist_rgb = wrist_rgb

            rcv = ensure_rcv(rcv, args.ur_ip)

            q = np.asarray(rcv.getActualQ(), dtype=np.float64)
            qd = np.asarray(rcv.getActualQd(), dtype=np.float64)

            # track which waypoint we've passed and (optionally) push gripper commands ahead
            dists = [float(np.linalg.norm(q - np.asarray(qw, dtype=np.float64))) for qw in waypoints_q]
            nearest = int(np.argmin(dists))

            if nearest > reached_wp_idx and dists[nearest] < args.final_q_tol * 1.5:
                reached_wp_idx = nearest
                # mirror the waypoint's gripper into g_cmd so state[6] stays accurate
                if gripper_wp is not None:
                    g_cmd = float(gripper_wp[nearest])

            # send the gripper command for waypoint N+1 as soon as we hit
            # waypoint N, otherwise the gripper finishes moving too late
            if gripper_wp is not None and use_gripper and gripper is not None:
                target_wp_idx = reached_wp_idx + 1
                if target_wp_idx < len(gripper_wp) and reached_wp_idx > gripper_sent_for_wp:
                    target_g_cmd = float(gripper_wp[target_wp_idx])
                    # debounce against the last issued value
                    if last_g_sent is None or abs(target_g_cmd - last_g_sent) > args.gripper_debounce:
                        try:
                            print(f"Sending gripper command for waypoint {target_wp_idx} (reached waypoint {reached_wp_idx}): {target_g_cmd:.3f}")
                            gripper.move_normalized(target_g_cmd)
                            last_g_sent = target_g_cmd
                            gripper_sent_for_wp = reached_wp_idx
                        except Exception as e:
                            print(f"Warning: Gripper move failed: {e}")

            # state is 7D, joint positions plus the gripper command
            state = np.asarray([*q.tolist(), g_cmd], dtype=np.float32)

            base_rel = Path("images/base") / f"{i:06d}.jpg"
            wrist_rel = Path("images/wrist") / f"{i:06d}.jpg"
            _write_jpg_rgb(ep_dir / base_rel, base_rgb, quality=args.jpeg_quality)
            _write_jpg_rgb(ep_dir / wrist_rel, wrist_rgb, quality=args.jpeg_quality)

            # actions are forward-looking absolute: action[i] = state[i+1].
            # the previous step's action is only known once we capture this state,
            # so we flush it here.
            if pending_step is not None:
                pending_step["actions"] = state.tolist()
                f_steps.write(json.dumps(pending_step) + "\n")
                f_steps.flush()

            # buffer the current step, action gets filled on the next tick
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

            # stop once we're near the final waypoint and the joints are stable
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
        # last buffered step: hold the final position as its action
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

