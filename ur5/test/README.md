# Hardware tests

Standalone scripts to verify cameras and the UR5 are reachable. For full setup
see [`ur5/docs/quickstart.md`](../docs/quickstart.md).

## List cameras

```bash
python ur5/test/rs_list.py
```

Copy the serial numbers into `camera_test.py` (`SERIAL_BASE`, `SERIAL_WRIST`)
or set them via env vars used by the recording scripts (`RS_BASE`, `RS_WRIST`).

## Live camera preview

```bash
python ur5/test/camera_test.py
# 'q' to quit
```

## Troubleshooting

- **Camera not found** — check USB 3.0 connection, re-run `rs_list.py`
- **Import errors** — activate the env and `pip install pyrealsense2 ur-rtde numpy opencv-python`
