"""Shared defaults for UR5 hardware scripts. Override via env vars."""

import os

UR_IP = os.environ.get("UR_IP", "192.10.0.11")
ROBOTIQ_PORT = int(os.environ.get("ROBOTIQ_PORT", "63352"))

# RealSense cameras
RS_BASE_SERIAL = os.environ.get("RS_BASE", "137322074310")
RS_WRIST_SERIAL = os.environ.get("RS_WRIST", "137322075008")
RS_W = int(os.environ.get("RS_W", "640"))
RS_H = int(os.environ.get("RS_H", "480"))
RS_FPS = int(os.environ.get("RS_FPS", "30"))
RS_TIMEOUT_MS = int(os.environ.get("RS_TIMEOUT_MS", "10000"))

# Start / reset position (degrees)
START_POSITION_DEG = (-90.0, -40.0, -140.0, -50.0, 90.0, 0.0)

# Output directory for raw episodes
OUT_DIR = os.environ.get("OUT_DIR", "raw_episodes")
