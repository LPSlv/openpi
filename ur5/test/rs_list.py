"""List connected RealSense cameras with their serial numbers and USB type.

Run this first to get serials for camera_test.py and the recording scripts.
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
