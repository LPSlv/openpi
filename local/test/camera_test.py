"""
Launch one or two RealSense RGB cameras with live preview.

This script allows you to:
- Launch one or two cameras by serial number
- View live camera feed and 224x224 cropped preview

Usage:
1. Run rs_list.py to get camera serial numbers
2. Edit SERIAL_BASE and/or SERIAL_WRIST below with your camera serials
   (at least one must be set)
3. Run this script
4. Press 'q' to quit
"""

import numpy as np
import cv2
import pyrealsense2 as rs

# Put your serials here (from rs_list.py)
# At least one camera must be configured
SERIAL_BASE = "137322074310"              # over-the-shoulder (set to None if not used)
SERIAL_WRIST = "137322075008"   # wrist (set to None if not used)
# Example for two cameras:
# SERIAL_BASE = "137322074310"
# SERIAL_WRIST = "137322075008"

W, H, FPS = 640, 480, 30  # resolution and framerate


def to_uint8_224(bgr):
    """Resize and center-crop BGR image to 224x224 RGB."""
    h, w = bgr.shape[:2]
    s = 224 / max(h, w)
    nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    y0 = (224 - nh) // 2
    x0 = (224 - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # BGR -> RGB


def start_rgb_pipeline(serial):
    """Start RealSense pipeline and return pipeline."""
    if not serial:
        return None

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    prof = pipe.start(cfg)
    
    # Configure exposure (lower value for darker images)
    try:
        for s in prof.get_device().sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                s.set_option(rs.option.enable_auto_exposure, 0)
                s.set_option(rs.option.exposure, 100.0)  # Lower exposure value
                break
    except Exception:
        pass  # Continue even if exposure setting fails

    return pipe


def main():
    # Check that at least one camera is configured
    if not SERIAL_BASE and not SERIAL_WRIST:
        raise RuntimeError("At least one camera must be configured (SERIAL_BASE or SERIAL_WRIST)")

    # Start base camera if configured
    pb = None
    if SERIAL_BASE:
        print("Starting base camera:", SERIAL_BASE)
        pb = start_rgb_pipeline(SERIAL_BASE)
        if pb is None:
            raise RuntimeError("Base camera failed to start (check serial)")

    # Start wrist camera if configured
    pw = None
    if SERIAL_WRIST:
        print("Starting wrist camera:", SERIAL_WRIST)
        pw = start_rgb_pipeline(SERIAL_WRIST)
        if pw is None:
            raise RuntimeError("Wrist camera failed to start (check serial)")

    # At least one camera should be running at this point
    if pb is None and pw is None:
        raise RuntimeError("No cameras started successfully")

    try:
        while True:
            img_bgr_base = None
            rgb224_base = None
            img_bgr_wrist = None
            rgb224_wrist = None

            # Grab base camera frame if available
            if pb is not None:
                fb = pb.wait_for_frames()
                cb = fb.get_color_frame()
                if cb:
                    img_bgr_base = np.asanyarray(cb.get_data())
                    rgb224_base = to_uint8_224(img_bgr_base)

            # Grab wrist camera frame if available
            if pw is not None:
                fw = pw.wait_for_frames()
                cw = fw.get_color_frame()
                if cw:
                    img_bgr_wrist = np.asanyarray(cw.get_data())
                    rgb224_wrist = to_uint8_224(img_bgr_wrist)

            # Display frames based on which cameras are available
            if pb is not None and pw is not None:
                # Both cameras: show side by side
                if img_bgr_base is not None and img_bgr_wrist is not None:
                    vis_full = np.hstack([img_bgr_base, img_bgr_wrist])
                    vis_crops = np.hstack([
                        cv2.cvtColor(rgb224_base, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(rgb224_wrist, cv2.COLOR_RGB2BGR),
                    ])
                    cv2.imshow("Base | Wrist", vis_full)
                    cv2.imshow("224x224 Base | Wrist", vis_crops)
            elif pb is not None:
                # Only base camera
                if img_bgr_base is not None:
                    cv2.imshow("Base", img_bgr_base)
                    cv2.imshow("224x224 Base", cv2.cvtColor(rgb224_base, cv2.COLOR_RGB2BGR))
            elif pw is not None:
                # Only wrist camera
                if img_bgr_wrist is not None:
                    cv2.imshow("Wrist", img_bgr_wrist)
                    cv2.imshow("224x224 Wrist", cv2.cvtColor(rgb224_wrist, cv2.COLOR_RGB2BGR))

            # Press q to quit
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

