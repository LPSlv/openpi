"""widowx ai bridge to the openpi websocket server.

forked from ur5/utils/pi0_bridge_ur5_headless.py. server is unchanged --
it serves a ur5-trained checkpoint and expects a ur5-shaped obs dict; we
synthesize that from widowx state and project the 7-vec action chunk back
onto the widowx via the lerobot trossen driver.
"""

import os
import sys
import time

# qt's MIT-SHM extension breaks X11 inside docker; xcb is the most reliable
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from ur5 import defaults as _defaults  # only RS_*/OUT_DIR
from widowx import defaults as _wx
from widowx.utils import remap as _remap

FAKE_CAM = os.environ.get("FAKE_CAM", "0") == "1"
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

if not FAKE_CAM:
    import pyrealsense2 as rs

# lazy driver import lets DRY_RUN run on machines without lerobot_robot_trossen
if not DRY_RUN:
    from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
    from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower


WX_IP = os.environ.get("WX_IP", _wx.WX_IP)
PROMPT = os.environ.get("PROMPT", "do something")

POLICY_HOST = os.environ.get("POLICY_HOST", "localhost")
POLICY_PORT = int(os.environ.get("POLICY_PORT", "8000"))

RS_BASE = os.environ.get("RS_BASE", "")
RS_WRIST = os.environ.get("RS_WRIST", "")
RS_W = int(os.environ.get("RS_W", str(_defaults.RS_W)))
RS_H = int(os.environ.get("RS_H", str(_defaults.RS_H)))
RS_FPS = int(os.environ.get("RS_FPS", str(_defaults.RS_FPS)))
RS_TIMEOUT_MS = int(os.environ.get("RS_TIMEOUT_MS", str(_defaults.RS_TIMEOUT_MS)))

# wall-clock seconds per chunk; if set, derives HOLD_PER_STEP from HORIZON_STEPS
_infer_period_raw = os.environ.get("INFER_PERIOD")
INFER_PERIOD: float | None = None
if _infer_period_raw not in (None, ""):
    INFER_PERIOD = float(_infer_period_raw)

HOLD_PER_STEP = float(os.environ.get("HOLD_PER_STEP", "0.1"))
HORIZON_STEPS = int(os.environ.get("HORIZON_STEPS", "10"))

if INFER_PERIOD is not None and "HOLD_PER_STEP" not in os.environ:
    if HORIZON_STEPS <= 0:
        raise ValueError(f"HORIZON_STEPS must be > 0, got {HORIZON_STEPS}")
    HOLD_PER_STEP = INFER_PERIOD / float(HORIZON_STEPS)

# the ur5 server emits absolute joint targets (rad) for [:, :6] and 0..1
# gripper for [:, 6]. ACTION_MODE=delta would treat [:, :6] as deltas instead.
ACTION_MODE = os.environ.get("ACTION_MODE", "absolute").lower()

MAX_REL_RAD = float(os.environ.get("WX_MAX_REL_RAD", str(_wx.WX_MAX_REL_RAD)))

GRIPPER_DEBOUNCE = float(os.environ.get("GRIPPER_DEBOUNCE", "0.02"))
GRIPPER_THRESHOLD = os.environ.get("GRIPPER_THRESHOLD", "")

SHOW_IMAGES = os.environ.get("SHOW_IMAGES", "1") == "1"

RS_AUTO_EXPOSURE = os.environ.get("RS_AUTO_EXPOSURE", "")
RS_EXPOSURE = os.environ.get("RS_EXPOSURE", "")
RS_WRIST_EXPOSURE = os.environ.get("RS_WRIST_EXPOSURE", "")

RECORD_DIR = os.environ.get("RECORD_DIR", "")

# probe whether cv2.imshow can actually open a window before we try mid-loop
_has_gui = False
if SHOW_IMAGES:
    if not os.environ.get("DISPLAY"):
        print("DISPLAY not set; preview disabled", file=sys.stderr)
    else:
        try:
            bi = cv2.getBuildInformation()
            has_gtk = any("YES" in l or "ON" in l for l in bi.split("\n") if "GTK:" in l)
            has_qt = any("YES" in l or "ON" in l for l in bi.split("\n") if "QT:" in l)
            if not (has_gtk or has_qt):
                print("opencv built without GTK/QT; preview disabled", file=sys.stderr)
            else:
                test_img = np.zeros((10, 10, 3), dtype=np.uint8)
                cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
                cv2.imshow("_test", test_img)
                cv2.waitKey(1)
                cv2.destroyWindow("_test")
                _has_gui = True
        except Exception as e:
            print(f"gui probe failed: {e}; preview disabled", file=sys.stderr)


def _process_bgr(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = image_tools.resize_with_pad(rgb, 256, 256)
    return image_tools.convert_to_uint8(rgb)


def _start_rgb(serial: str, *, exposure_override: str = "") -> "rs.pipeline | None":
    if FAKE_CAM or not serial:
        return None
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
    prof = pipe.start(cfg)

    exposure_val = exposure_override if exposure_override else RS_EXPOSURE
    try:
        for s in prof.get_device().sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                if RS_AUTO_EXPOSURE != "":
                    s.set_option(rs.option.enable_auto_exposure, float(RS_AUTO_EXPOSURE))
                if exposure_val != "":
                    s.set_option(rs.option.exposure, float(exposure_val))
                ae = s.get_option(rs.option.enable_auto_exposure)
                exp = s.get_option(rs.option.exposure)
                gain = s.get_option(rs.option.gain)
                intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                print(
                    f"cam {serial}: {intr.width}x{intr.height} "
                    f"fx={intr.fx:.1f} fy={intr.fy:.1f} ae={ae} exp={exp} gain={gain} fps={RS_FPS}",
                    flush=True,
                )
                break
    except Exception as e:
        print(f"cam {serial}: settings read/set failed: {e}", flush=True)

    return pipe


def _read_rgb(pipe: "rs.pipeline") -> np.ndarray | None:
    # drain any buffered frames so we keep only the latest
    frame = None
    while True:
        try:
            frames = pipe.poll_for_frames()
            f = frames.get_color_frame()
            if f:
                frame = f
            else:
                break
        except Exception:
            break
    if frame is None:
        frames = pipe.wait_for_frames(RS_TIMEOUT_MS)
        frame = frames.get_color_frame()
    if not frame:
        return None
    return _process_bgr(np.asanyarray(frame.get_data()))


def _read_wx_arm_q(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.asarray([obs[f"{n}.pos"] for n in _wx.WX_JOINT_NAMES[:6]], dtype=np.float64)


def _build_action_dict(q_wx_arm: np.ndarray, gripper_m: float) -> dict:
    d = {f"{n}.pos": float(q_wx_arm[i]) for i, n in enumerate(_wx.WX_JOINT_NAMES[:6])}
    d[_wx.WX_GRIPPER_KEY] = float(gripper_m)
    return d


def main() -> None:
    global _has_gui

    print("starting cameras...", end=" ", flush=True)
    base_cam = _start_rgb(RS_BASE)
    wrist_cam = _start_rgb(RS_WRIST, exposure_override=RS_WRIST_EXPOSURE)
    if base_cam is not None or wrist_cam is not None:
        print("ok")
        time.sleep(1.0)
    else:
        print("skipped")

    robot = None
    if not DRY_RUN:
        print(f"connecting to widowx at {WX_IP}...", end=" ", flush=True)
        cfg = WidowXAIFollowerConfig(
            ip_address=WX_IP,
            max_relative_target=MAX_REL_RAD,
            loop_rate=_wx.WX_LOOP_RATE,
            min_time_to_move_multiplier=_wx.WX_MIN_TIME_MULT,
            staged_positions=list(_wx.WX_STAGED_RAD),
            cameras={},
        )
        robot = WidowXAIFollower(cfg)
        robot.connect(calibrate=False)
        print("ok")
        time.sleep(0.5)

    print(f"connecting to policy server at {POLICY_HOST}:{POLICY_PORT}...", end=" ", flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(host=POLICY_HOST, port=POLICY_PORT)
    print("ok")

    metadata = client.get_server_metadata()
    print(
        f"server: train_config={metadata.get('train_config')!r} "
        f"checkpoint={metadata.get('checkpoint_dir')!r} "
        f"norm_stats={metadata.get('norm_stats_dir')!r}",
        flush=True,
    )
    # NB: ignore metadata['reset_pose'], it's ur5-frame and unsafe on widowx

    chunk_time_s = float(HORIZON_STEPS) * float(HOLD_PER_STEP)
    print(
        f"control: ACTION_MODE={ACTION_MODE} MAX_REL_RAD={MAX_REL_RAD} "
        f"HORIZON_STEPS={HORIZON_STEPS} HOLD_PER_STEP={HOLD_PER_STEP:.4f}s "
        f"(~{chunk_time_s:.3f}s/chunk) WX_LOOP_RATE={_wx.WX_LOOP_RATE} "
        f"WX_MIN_TIME_MULT={_wx.WX_MIN_TIME_MULT} PERM={_wx.UR5_TO_WX_PERM} SIGN={_wx.UR5_TO_WX_SIGN} "
        f"GRIP=[{_wx.WX_GRIPPER_OPEN_M},{_wx.WX_GRIPPER_CLOSED_M}]m",
        flush=True,
    )

    last_gripper: float = 0.0
    infer_step = 0
    n_clamped_total = 0

    try:
        while True:
            t_capture = time.time()
            if FAKE_CAM:
                rgb_base = np.zeros((224, 224, 3), dtype=np.uint8)
                rgb_wrist = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                if base_cam is None:
                    raise RuntimeError("RS_BASE must be set (or set FAKE_CAM=1)")
                rgb_base = _read_rgb(base_cam)
                if rgb_base is None:
                    continue
                if wrist_cam is None:
                    rgb_wrist = rgb_base
                else:
                    rgb_wrist = _read_rgb(wrist_cam)
                    if rgb_wrist is None:
                        rgb_wrist = rgb_base

            if DRY_RUN:
                q_wx_arm = np.array(_wx.WX_HOME_RAD, dtype=np.float64)
            else:
                q_wx_arm = _read_wx_arm_q(robot)
            q_ur5 = _remap.wx_to_ur5_q(q_wx_arm)
            state = np.concatenate([q_ur5.astype(np.float32), np.array([last_gripper], dtype=np.float32)])

            obs = {
                "observation/image": rgb_base,
                "observation/wrist_image": rgb_wrist,
                "observation/state": state,
                "prompt": PROMPT,
            }

            if SHOW_IMAGES and _has_gui:
                try:
                    bgr_base = cv2.cvtColor(rgb_base, cv2.COLOR_RGB2BGR)
                    bgr_wrist = cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR)
                    cv2.imshow("base | wrist", np.hstack([bgr_base, bgr_wrist]))
                    cv2.waitKey(1)
                except Exception as e:
                    err = str(e).lower()
                    if any(s in err for s in ("x11", "display", "connection", "badvalue", "badwindow")):
                        if _has_gui:
                            _has_gui = False
                            print("x11 dropped; preview off", file=sys.stderr, flush=True)
                    else:
                        if _has_gui:
                            print(f"imshow failed: {e}", file=sys.stderr, flush=True)

            if infer_step == 0:
                print(
                    f"step 0 diag: base mean={rgb_base.mean():.1f} "
                    f"wrist mean={rgb_wrist.mean():.1f} state(ur5)={state} q_wx={q_wx_arm}",
                    flush=True,
                )

            t_infer_start = time.time()
            out = client.infer(obs)
            t_infer_end = time.time()
            actions = np.asarray(out["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            actions = actions[:, :7]

            print(
                f"--- chunk: {actions.shape} "
                f"joints=[{actions[:, :6].min():.4f},{actions[:, :6].max():.4f}] "
                f"grip=[{actions[:, 6].min():.3f},{actions[:, 6].max():.3f}] "
                f"t_capture={t_infer_start - t_capture:.3f}s t_infer={t_infer_end - t_infer_start:.3f}s",
                flush=True,
            )

            if RECORD_DIR:
                os.makedirs(RECORD_DIR, exist_ok=True)
                np.savez_compressed(
                    os.path.join(RECORD_DIR, f"step_{infer_step:04d}.npz"),
                    image=rgb_base,
                    wrist_image=rgb_wrist,
                    state=state,
                    actions=actions,
                    prompt=np.array(PROMPT),
                    q_wx=q_wx_arm,
                )

            infer_step += 1

            chunk_clamped = 0
            for a in actions[:HORIZON_STEPS]:
                q_tgt_ur5 = np.asarray(a[:6], dtype=np.float64)
                q_tgt_wx = _remap.ur5_to_wx_q(q_tgt_ur5)

                q_tgt_wx, n_lim = _remap.clamp_to_limits(
                    q_tgt_wx, _wx.WX_JOINT_LIMITS_LOW, _wx.WX_JOINT_LIMITS_HIGH
                )
                if n_lim > 0:
                    chunk_clamped += n_lim

                q_now_wx = _read_wx_arm_q(robot) if not DRY_RUN else np.array(_wx.WX_HOME_RAD, dtype=np.float64)
                q_tgt_wx = _remap.clip_step(q_tgt_wx, q_now_wx, MAX_REL_RAD)

                g_raw = float(a[6])
                g = float(np.clip(g_raw, 0.0, 1.0))
                if GRIPPER_THRESHOLD != "":
                    g = 1.0 if g >= float(GRIPPER_THRESHOLD) else 0.0
                gripper_m = _remap.map_gripper_ur5_to_wx(g)

                print(
                    f"q_tgt_wx={np.array2string(q_tgt_wx, precision=4, suppress_small=True)} "
                    f"grip={g_raw:.3f}->{gripper_m:.4f}m clamped={n_lim}",
                    flush=True,
                )

                if abs(g - last_gripper) > GRIPPER_DEBOUNCE:
                    last_gripper = g

                action_dict = _build_action_dict(q_tgt_wx, gripper_m)

                if DRY_RUN:
                    time.sleep(HOLD_PER_STEP)
                    continue

                robot.send_action(action_dict)
                time.sleep(HOLD_PER_STEP)

            n_clamped_total += chunk_clamped
            if chunk_clamped >= _wx.WX_LIMIT_CLAMP_ABORT_N:
                print(
                    f"aborting: {chunk_clamped} joint-limit clamps in chunk "
                    f"(>= {_wx.WX_LIMIT_CLAMP_ABORT_N})",
                    file=sys.stderr,
                    flush=True,
                )
                break

    finally:
        if robot is not None:
            # if the arm faulted mid-loop, clear so disconnect's blocking moves
            # to staged + sleep can complete
            try:
                robot.driver.clear_error()
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception as e:
                print(f"disconnect failed: {e}", file=sys.stderr)
                try:
                    robot.driver.cleanup()
                except Exception:
                    pass
        if not FAKE_CAM:
            for cam in (base_cam, wrist_cam):
                if cam is not None:
                    try:
                        cam.stop()
                    except Exception:
                        pass
        if SHOW_IMAGES:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print(f"total joint-limit clamps: {n_clamped_total}", flush=True)


if __name__ == "__main__":
    main()
