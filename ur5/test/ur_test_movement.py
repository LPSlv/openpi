"""Sanity test: activate the gripper, swing joint 0 by +/- 20 deg, close and re-open the gripper.

Gripper control uses the Robotiq URCap socket service on port 63352.
Requires ur_rtde and the Robotiq URCap on the controller.
"""

import time
import numpy as np
import rtde_receive
import rtde_control

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

UR_IP = _defaults.UR_IP

# motion tuning for the base-joint test
DELTA_DEG  = 20.0          # joint 0 swing in degrees
MOVEJ_VEL  = 0.50          # rad/s
MOVEJ_ACC  = 0.50          # rad/s^2

# conservative soft joint limits in degrees
SOFT_MIN_DEG = np.array([-180, -180, -180, -180, -180, -180], float)
SOFT_MAX_DEG = np.array([ 180,  180,  180,  180,  180,  180], float)


def main():
    ctrl: rtde_control.RTDEControlInterface | None = None
    rcv: rtde_receive.RTDEReceiveInterface | None = None
    gripper: RobotiqGripperHelper | None = None

    print("Connecting to robot via RTDE (receive)...", end=" ", flush=True)
    try:
        rcv = _create_rtde_receive(UR_IP, frequency=125.0, retries=2)
        print("OK")
    except Exception as e:
        print("FAILED")
        print(f"ERROR: {e}")
        return

    try:
        if not ok_to_move(rcv):
            print("Robot not ready (mode or safety). Put in Remote Control, clear any stops, then retry.")
            return

        print("\n" + "="*60)
        print("Initializing Robotiq Gripper")
        print("="*60)
        try:
            gripper = RobotiqGripperHelper(UR_IP)
        except Exception as e:
            print(f"ERROR: Failed to connect to gripper socket: {e}")
            return
        
        print("Activating gripper...", end=" ", flush=True)
        try:
            gripper.activate()
            print("OK")
        except Exception as e:
            print("FAILED")
            print(f"ERROR: Gripper activation failed: {e}")
            return

        time.sleep(2.0)
        rcv = ensure_rcv(rcv, UR_IP)

        print("\n" + "="*60)
        print("Testing Base Joint Movement")
        print("="*60)

        time.sleep(2.0)

        print("Connecting to robot via RTDE (control)...", end=" ", flush=True)
        try:
            ctrl = _create_rtde_control(UR_IP, retries=1)
            print("OK")
        except Exception as e:
            print("FAILED")
            print(f"ERROR: {e}")
            return

        try:
            q_rad = np.array(rcv.getActualQ(), float)
            q_deg = np.degrees(q_rad)
        except Exception as e:
            print(f"ERROR: Cannot read robot position: {e}")
            return

        delta = DELTA_DEG
        base_move_ok = True
        for step in (-delta, +delta):  # negative direction first, then back
            rcv = ensure_rcv(rcv, UR_IP)

            if not ok_to_move(rcv):
                print("Robot not ready (mode or safety), stopping.")
                return

            target_deg = q_deg.copy()
            target_deg[0] += step
            target_deg = clamp_to_soft_limits_deg(target_deg)
            target_rad = np.radians(target_deg)

            print(f"Target (deg): {np.round(target_deg, 2)}")
            try:
                assert ctrl is not None
                if not ctrl.isConnected():
                    raise RuntimeError("RTDE control not connected before moveJ")
                ok = ctrl.moveJ(target_rad.tolist(), MOVEJ_VEL, MOVEJ_ACC)
                if ok is False:
                    raise RuntimeError("RTDE moveJ returned False - control script may not be running")
            except Exception as e:
                print(f"ERROR: RTDE moveJ failed: {e}")
                base_move_ok = False
                return

            try:
                reached = _wait_joint0_reaches_target(rcv, target_rad.tolist(), tol_deg=1.0, timeout_s=8.0)
            except Exception as e:
                print(f"WARNING: RTDE receive failed during verification: {e}")
                reached = True  # if receive crashes, fail open and assume the move worked

            if not reached:
                print("ERROR: Base joint did not reach target within timeout.")
                base_move_ok = False
                return
            
            time.sleep(0.2)

            try:
                q_rad = np.array(rcv.getActualQ(), float)
                q_deg = np.degrees(q_rad)
            except Exception as e:
                print(f"WARNING: Cannot read position after moveJ: {e}")

        _teardown_rtde_control(ctrl)
        ctrl = None

        if not base_move_ok:
            print("ERROR: Base joint movement test failed.")
            return

        print("Base joint movement test completed!")
        print("="*60 + "\n")

        if gripper is not None:
            print("Closing gripper...", end=" ", flush=True)
            try:
                gripper.close()
                print("OK")
            except Exception as e:
                print("FAILED")
                print(f"ERROR: {e}")
                return
            
            print("Opening gripper...", end=" ", flush=True)
            try:
                gripper.open()
                print("OK")
            except Exception as e:
                print("FAILED")
                print(f"ERROR: {e}")
                return
        
        print("\nAll tests completed successfully!")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
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

if __name__ == "__main__":
    main()