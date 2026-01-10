"""
List all connected RealSense cameras and display their serial numbers.

This script scans for RealSense devices and prints information about each:
- Device name
- Serial number (needed for other scripts)
- USB type
- Available sensors

Run this first to get serial numbers before using rs_rgb.py.
"""

import pyrealsense2 as rs

ctx = rs.context()
devs = ctx.query_devices()
print(f"Found {len(devs)} RealSense device(s)")
for i, d in enumerate(devs):
    sn = d.get_info(rs.camera_info.serial_number)
    name = d.get_info(rs.camera_info.name)
    usb = d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else "n/a"
    print(f"[{i}] {name}  SN={sn}  USB={usb}")
    for s in d.sensors:
        print("   -", s.get_info(rs.camera_info.name))
