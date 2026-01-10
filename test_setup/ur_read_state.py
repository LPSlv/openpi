"""
Read and display the current state of a UR robot arm.

This script continuously reads and prints:
- Joint positions (degrees)
- Joint velocities (degrees/second)
- TCP pose (position and orientation)
- Robot mode and safety status
- Update frequency

Press Ctrl+C to stop.
"""

import time
import numpy as np
import rtde_receive

UR_IP = "192.168.1.116"
FREQ = 125.0        # Hz
PRINT_EVERY = 0.2   # seconds between prints
RAD2DEG = 180.0 / np.pi

rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP, frequency=FREQ)

last_print = 0.0
count = 0
t0 = time.time()

try:
    while True:
        q_rad  = np.array(rtde_r.getActualQ(),  float)       # [rad]
        qd_rad = np.array(rtde_r.getActualQd(), float)       # [rad/s]
        tcp    = np.array(rtde_r.getActualTCPPose(), float)  # [x y z rx ry rz] -> m, rad
        mode = rtde_r.getRobotMode()                         # 7 = RUNNING
        sfty = rtde_r.getSafetyMode()                        # 1 = NORMAL

        # convert joints to degrees and deg/s
        q_deg  = q_rad * RAD2DEG
        qd_deg = qd_rad * RAD2DEG

        now = time.time()
        count += 1

        if now - last_print >= PRINT_EVERY:
            hz = count / (now - t0)
            print(
                f"Hz={hz:5.1f}  "
                f"q_deg={q_deg.round(2)}  "
                f"qd_deg_s={qd_deg.round(2)}  "
                f"tcp(m,rad)={tcp.round(4)}  "
                f"mode={mode} safety={sfty}"
            )
            last_print = now
            count = 0
            t0 = now

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopped by user.")
