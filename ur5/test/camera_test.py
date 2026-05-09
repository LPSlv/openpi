"""Live preview for one or two RealSense RGB cameras (full and 224x224 crops).

Workflow: run rs_list.py to find serials, fill SERIAL_BASE / SERIAL_WRIST below,
run this script, press 'q' to quit.
"""

import numpy as np
import cv2
import pyrealsense2 as rs

# put your serials here, at least one is required (use None to skip)
SERIAL_BASE = "137322074310"   # over-the-shoulder
SERIAL_WRIST = "137322075008"  # wrist

W, H, FPS = 640, 480, 30  # resolution and framerate


def to_uint8_224(bgr):
    """Resize-with-pad to 224x224, return RGB uint8."""
    h, w = bgr.shape[:2]
    s = 224 / max(h, w)
    nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    y0 = (224 - nh) // 2
    x0 = (224 - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def start_rgb_pipeline(serial):
    """Start a RealSense RGB pipeline with fixed manual exposure."""
    if not serial:
        return None

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    prof = pipe.start(cfg)

    # fixed exposure stops the auto-exposure jitter that hurts trained policies
    try:
        for s in prof.get_device().sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                s.set_option(rs.option.enable_auto_exposure, 0)
                s.set_option(rs.option.exposure, 100.0)
                break
    except Exception:
        pass

    return pipe


def main():
    if not SERIAL_BASE and not SERIAL_WRIST:
        raise RuntimeError("At least one camera must be configured (SERIAL_BASE or SERIAL_WRIST)")

    pb = None
    if SERIAL_BASE:
        print("Starting base camera:", SERIAL_BASE)
        pb = start_rgb_pipeline(SERIAL_BASE)
        if pb is None:
            raise RuntimeError("Base camera failed to start (check serial)")

    pw = None
    if SERIAL_WRIST:
        print("Starting wrist camera:", SERIAL_WRIST)
        pw = start_rgb_pipeline(SERIAL_WRIST)
        if pw is None:
            raise RuntimeError("Wrist camera failed to start (check serial)")

    if pb is None and pw is None:
        raise RuntimeError("No cameras started successfully")

    try:
        while True:
            img_bgr_base = None
            rgb224_base = None
            img_bgr_wrist = None
            rgb224_wrist = None

            if pb is not None:
                fb = pb.wait_for_frames()
                cb = fb.get_color_frame()
                if cb:
                    img_bgr_base = np.asanyarray(cb.get_data())
                    rgb224_base = to_uint8_224(img_bgr_base)

            if pw is not None:
                fw = pw.wait_for_frames()
                cw = fw.get_color_frame()
                if cw:
                    img_bgr_wrist = np.asanyarray(cw.get_data())
                    rgb224_wrist = to_uint8_224(img_bgr_wrist)

            if pb is not None and pw is not None:
                if img_bgr_base is not None and img_bgr_wrist is not None:
                    vis_full = np.hstack([img_bgr_base, img_bgr_wrist])
                    vis_crops = np.hstack([
                        cv2.cvtColor(rgb224_base, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(rgb224_wrist, cv2.COLOR_RGB2BGR),
                    ])
                    cv2.imshow("Base | Wrist", vis_full)
                    cv2.imshow("224x224 Base | Wrist", vis_crops)
            elif pb is not None:
                if img_bgr_base is not None:
                    cv2.imshow("Base", img_bgr_base)
                    cv2.imshow("224x224 Base", cv2.cvtColor(rgb224_base, cv2.COLOR_RGB2BGR))
            elif pw is not None:
                if img_bgr_wrist is not None:
                    cv2.imshow("Wrist", img_bgr_wrist)
                    cv2.imshow("224x224 Wrist", cv2.cvtColor(rgb224_wrist, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if pb is not None:
            try:
                pb.stop()
            except Exception:
                pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

