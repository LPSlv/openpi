"""
Test UR robot movement with a safe base joint nudge and Robotiq gripper test.

This script performs:
1. A small test movement on the base joint (joint 0):
   - Moves the base joint +5 degrees, then back -5 degrees
2. Robotiq gripper test:
   - Opens gripper (fully open)
   - Closes gripper (fully closed)
   - Tests intermediate positions

Includes safety checks (robot mode and safety status) and uses conservative
motion parameters for safe testing. Automatically stops if safety conditions change.

Use this to verify robot connectivity and basic movement before running policies.

Requirements:
- ur_rtde library installed: pip3 install ur_rtde
  (or: sudo apt install librtde librtde-dev)
- Robotiq URCap installed on the robot controller (for socket interface on port 63352)
"""

import time
import socket
import numpy as np
import rtde_receive
import rtde_control
import dashboard_client

UR_IP = "192.168.1.116"

# Motion tuning – slow & smooth (conservative to avoid force limits)
DT         = 1.0 / 125.0
DELTA_DEG  = 5.0           # very small step to avoid force limits
VEL        = 0.02          # slower velocity (rad/s)
ACC        = 0.02          # slower acceleration (rad/s^2)
LOOKAHEAD  = 0.2
GAIN       = 100
STREAM_SEC = 3.0           # longer duration for smoother, gentler movement


# Soft absolute joint limits (deg) – very conservative
SOFT_MIN_DEG = np.array([-180, -180, -180, -180, -180, -180], float)
SOFT_MAX_DEG = np.array([ 180,  180,  180,  180,  180,  180], float)

def ok_to_move(dash: dashboard_client.DashboardClient, rcv: rtde_receive.RTDEReceiveInterface) -> bool:
    # Dashboard: should be RUNNING; RTDE: safety NORMAL (1), robot RUNNING (7)
    try:
        mode = rcv.getRobotMode()
        safety = rcv.getSafetyMode()
        # quick human-readable info (optional)
        state = dash.robotmode()
        # print("Dashboard:", state.strip(), "robotmode:", mode, "safety:", safety)
        return (mode == 7) and (safety == 1)
    except Exception:
        return False

def clamp_to_soft_limits_deg(q_deg):
    return np.clip(q_deg, SOFT_MIN_DEG, SOFT_MAX_DEG)

def send_urscript(host, port, script):
    """Send URScript command to robot controller.
    
    Args:
        host: UR robot IP address
        port: Port for URScript commands (default 30002)
        script: URScript code to execute
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)  # Increased timeout
        sock.connect((host, port))
        sock.sendall(script.encode())
        sock.close()
        return True
    except socket.timeout:
        print(f"Timeout connecting to {host}:{port}", flush=True)
        return False
    except ConnectionRefusedError:
        print(f"Connection refused to {host}:{port}. Is the robot controller running?", flush=True)
        return False
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", flush=True)
        return False

def send_robotiq_socket_command(host, port, command):
    """Send command to Robotiq gripper via socket interface (port 63352).
    
    This is an alternative to URScript if the URCap socket server is running.
    
    Args:
        host: UR robot IP address
        port: Port for Robotiq socket (default 63352)
        command: ASCII command string (e.g., "SET POS 100")
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((host, port))
        sock.sendall((command + "\n").encode())
        response = sock.recv(1024).decode().strip()
        sock.close()
        return True, response
    except Exception as e:
        return False, str(e)

def test_gripper(ur_ip, ctrl):
    """Test Robotiq gripper using ur_rtde's sendCustomScriptFunction.
    
    This method uses ur_rtde's RTDEControlInterface.sendCustomScriptFunction()
    to send Robotiq gripper commands via the socket interface (port 63352).
    Each command includes opening the socket, sending the command, and closing it.
    
    This approach works with the Robotiq URCap socket server and doesn't require
    pre-loaded URCap functions.
    
    Args:
        ur_ip: UR robot IP address
        ctrl: RTDEControlInterface instance
    """
    print("\n" + "="*60)
    print("Testing Robotiq Gripper (using ur_rtde)")
    print("="*60)
    
    if not ctrl.isConnected():
        print("ERROR: RTDE control interface not connected!")
        return False
    
    # Robotiq socket commands via URScript
    # These commands connect to the Robotiq URCap socket server on port 63352
    
    # Step 1: Activate the gripper
    print("Activating gripper...", end=" ", flush=True)
    activate_script = '''socket_open("127.0.0.1", 63352, "gripper_socket")
socket_send_string("ACT\\n", "gripper_socket")
socket_close("gripper_socket")
'''
    try:
        ctrl.sendCustomScriptFunction(activate_script)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Robotiq URCap is installed and running on the robot")
        print("  2. The Robotiq socket server should be listening on port 63352")
        print("  3. Make sure the robot is in Remote Control mode")
        print("  4. Check that ur_rtde is installed: pip3 install ur_rtde")
        return False
    
    time.sleep(3.0)  # Wait for activation
    
    # Step 2: Test positions: [0.0 (open), 0.5 (half), 1.0 (closed), 0.0 (open)]
    test_positions = [0.0, 0.5, 1.0, 0.0]
    position_names = ["fully open", "half closed", "fully closed", "fully open"]
    
    print("\nTesting gripper positions:")
    for pos, name in zip(test_positions, position_names):
        # Convert to Robotiq format: 0 = open, 255 = closed
        robotiq_pos = int(round(pos * 255))
        
        print(f"  Setting gripper to {name} (position {pos:.1f}, Robotiq value {robotiq_pos})...", end=" ", flush=True)
        
        # Create URScript to send position command via socket
        pos_script = f'''socket_open("127.0.0.1", 63352, "gripper_socket")
socket_send_string("SET POS {robotiq_pos}\\n", "gripper_socket")
socket_close("gripper_socket")
'''
        try:
            ctrl.sendCustomScriptFunction(pos_script)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
        
        # Wait for gripper to move
        time.sleep(2.5)
    
    print("\nGripper test completed!")
    print("="*60 + "\n")
    return True

def main():
    dash = dashboard_client.DashboardClient(UR_IP)
    dash.connect()  # optional but useful for quick state queries

    rcv = rtde_receive.RTDEReceiveInterface(UR_IP, frequency=125.0)
    ctrl = rtde_control.RTDEControlInterface(UR_IP)

    try:
        if not ok_to_move(dash, rcv):
            print("Robot not ready (mode or safety). Put in Remote Control, clear any stops, then retry.")
            return

        # Test 1: Base joint movement
        print("\n" + "="*60)
        print("Testing Base Joint Movement")
        print("="*60)
        
        q_rad = np.array(rcv.getActualQ(), float)
        q_deg = np.degrees(q_rad)

        # Build a tiny nudge on joint 0 (base)
        delta = DELTA_DEG
        for step in (+delta, -delta):  # out and back
            target_deg = q_deg.copy()
            target_deg[0] += step
            target_deg = clamp_to_soft_limits_deg(target_deg)
            target_rad = np.radians(target_deg)

            print(f"Target (deg): {np.round(target_deg, 2)}")
            t0 = time.time()
            while time.time() - t0 < STREAM_SEC:
                # Watchdog – bail immediately if safety or mode changes
                if not ok_to_move(dash, rcv):
                    print("Safety/mode changed – stopping.")
                    try:
                        if ctrl.isConnected():
                            ctrl.speedStop()
                    except Exception:
                        pass  # Control script may have stopped
                    return

                ctrl.servoJ(
                    target_rad.tolist(),
                    VEL, ACC, DT, LOOKAHEAD, GAIN
                )
                time.sleep(DT)

            # refresh current position before the return step
            q_rad = np.array(rcv.getActualQ(), float)
            q_deg = np.degrees(q_rad)

        print("Base joint movement test completed!")
        print("="*60 + "\n")
        
        # Test 2: Robotiq gripper (using ur_rtde)
        test_gripper(UR_IP, ctrl)
        
        print("All tests completed successfully!")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        try:
            if ctrl.isConnected():
                ctrl.speedStop()
        except Exception:
            pass  # Control script may have stopped
        try:
            dash.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()