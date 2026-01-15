"""UR5 real robot environment using RTDE for control."""

import os
import socket
import time
from typing import Optional, List
import numpy as np
import rtde_control
import rtde_receive

# Try to import cv2 for image processing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Try to import RealSense, but allow it to be optional
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

# This is the reset position that is used by the standard UR5 runtime.
# Default: (-90, -45, -120, -75, 90, 0) degrees in radians
DEFAULT_RESET_POSITION = [-1.5708, -0.7854, -2.0944, -1.3089, 1.5708, 0.0]


class RealEnv:
    """
    Environment for UR5 robot control via RTDE.

    Action space: [joint_deltas (6), gripper_position (1)]
                  # joint deltas in radians, gripper position normalized (0: close, 1: open)

    Observation space: {"qpos": [joints (6), gripper (1)],
                        "qvel": [joint_velocities (6), gripper_velocity (1)],
                        "images": {"base": (224x224x3), "wrist": (224x224x3)}}
    """

    def __init__(
        self,
        ur_ip: str,
        *,
        reset_position: Optional[List[float]] = None,
        rs_base_serial: Optional[str] = None,
        rs_wrist_serial: Optional[str] = None,
        gripper_port: int = 0,
        fake_cam: bool = False,
    ):
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION
        self._ur_ip = ur_ip
        self._gripper_port = gripper_port
        self._fake_cam = fake_cam

        # Initialize RTDE interfaces
        self._rcv = rtde_receive.RTDEReceiveInterface(ur_ip, frequency=125.0)
        self._ctrl = rtde_control.RTDEControlInterface(ur_ip)

        # Initialize cameras
        self._base_cam = None
        self._wrist_cam = None
        if not fake_cam and HAS_REALSENSE:
            if rs_base_serial:
                self._base_cam = self._start_rgb(rs_base_serial)
            if rs_wrist_serial:
                self._wrist_cam = self._start_rgb(rs_wrist_serial)

    def _start_rgb(self, serial: str) -> "rs.pipeline | None":
        """Start RealSense camera pipeline."""
        if self._fake_cam or not serial or not HAS_REALSENSE:
            return None
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        pipe.start(cfg)
        return pipe

    def _read_rgb(self, pipe: "rs.pipeline") -> np.ndarray | None:
        """Read RGB frame from RealSense camera."""
        if pipe is None:
            return None
        try:
            frames = pipe.wait_for_frames(timeout_ms=10000)
            frame = frames.get_color_frame()
            if not frame:
                return None
            bgr = np.asanyarray(frame.get_data())
            # Convert BGR to RGB
            if HAS_CV2:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = np.flip(bgr, axis=2)  # Fallback: flip channels
            # Resize to 224x224 with padding
            from openpi_client import image_tools
            rgb = image_tools.resize_with_pad(rgb, 224, 224)
            rgb = image_tools.convert_to_uint8(rgb)
            return rgb
        except Exception:
            return None

    def _set_gripper(self, pos01: float) -> None:
        """Set gripper position (0: close, 1: open)."""
        if self._gripper_port <= 0:
            return
        pos01 = float(np.clip(pos01, 0.0, 1.0))
        pos = int(round(pos01 * 255))
        try:
            with socket.create_connection((self._ur_ip, self._gripper_port), timeout=1.0) as s:
                s.sendall(f"rq_set_pos({pos})\n".encode())
        except Exception:
            pass  # Silently fail if gripper not available

    def _move_to_reset_position(self, vel: float = 0.2, acc: float = 0.3, timeout_sec: float = 10.0) -> None:
        """Move robot to reset position using RTDE moveJ."""
        target_q = np.asarray(self._reset_position, dtype=np.float64)

        # Use RTDE moveJ
        success = self._ctrl.moveJ(self._reset_position, vel, acc)
        if not success:
            print("Warning: moveJ command failed for reset position")
            return

        # Wait for movement to complete
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            current_q = np.asarray(self._rcv.getActualQ(), dtype=np.float64)
            dist = float(np.linalg.norm(current_q - target_q))
            if dist < 0.05:  # ~3 degrees tolerance
                print(f"Moved to reset position (error: {np.degrees(dist):.2f} deg)")
                time.sleep(0.2)
                return
            time.sleep(0.1)

        final_q = np.asarray(self._rcv.getActualQ(), dtype=np.float64)
        final_error = np.degrees(np.linalg.norm(final_q - target_q))
        print(f"Warning: Timeout waiting for reset position (final error: {final_error:.2f} deg)")

    def get_qpos(self) -> np.ndarray:
        """Get current joint positions and gripper state."""
        q = np.asarray(self._rcv.getActualQ(), dtype=np.float32)  # (6,)
        # For gripper, we use a placeholder value (0.0) since we don't have direct gripper feedback
        # The gripper state would need to be read from the gripper controller if available
        gripper_pos = np.array([0.0], dtype=np.float32)
        return np.concatenate([q, gripper_pos])

    def get_qvel(self) -> np.ndarray:
        """Get current joint velocities."""
        qd = np.asarray(self._rcv.getActualQd(), dtype=np.float32)  # (6,)
        gripper_vel = np.array([0.0], dtype=np.float32)
        return np.concatenate([qd, gripper_vel])

    def get_images(self) -> dict:
        """Get camera images."""
        images = {}
        if self._fake_cam:
            images["base"] = np.zeros((224, 224, 3), dtype=np.uint8)
            images["wrist"] = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            base_img = self._read_rgb(self._base_cam)
            if base_img is not None:
                images["base"] = base_img
            else:
                images["base"] = np.zeros((224, 224, 3), dtype=np.uint8)

            wrist_img = self._read_rgb(self._wrist_cam)
            if wrist_img is not None:
                images["wrist"] = wrist_img
            else:
                # Fallback to base image if wrist camera not available
                images["wrist"] = images["base"].copy()

        return images

    def get_observation(self) -> dict:
        """Get full observation."""
        obs = {}
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["images"] = self.get_images()
        return obs

    def reset(self, *, fake: bool = False) -> None:
        """Reset the environment by moving to reset position."""
        if not fake:
            print("Resetting UR5 to home position...")
            self._move_to_reset_position()
            # Reset gripper to open position
            self._set_gripper(1.0)
            time.sleep(0.5)

    def step(self, action: np.ndarray, dt: float = 0.05, vel: float = 0.05, acc: float = 0.05) -> None:
        """
        Apply action to the robot.

        Args:
            action: Array of shape (7,) with [joint_deltas (6), gripper (1)]
            dt: Control timestep
            vel: Velocity scaling
            acc: Acceleration scaling
        """
        if len(action) < 7:
            raise ValueError(f"Action must have 7 elements, got {len(action)}")

        # Get current joint positions
        current_q = np.asarray(self._rcv.getActualQ(), dtype=np.float32)

        # Apply joint deltas (first 6 elements)
        joint_deltas = action[:6]
        target_q = current_q + joint_deltas

        # Apply gripper command (last element)
        gripper_cmd = float(action[6])
        self._set_gripper(gripper_cmd)

        # Use servoJ for smooth control
        lookahead = 0.2
        gain = 150
        self._ctrl.servoJ(target_q.tolist(), vel, acc, dt, lookahead, gain)
        time.sleep(dt)

    def close(self) -> None:
        """Clean up resources."""
        try:
            self._ctrl.speedStop()
        except Exception:
            pass
        try:
            if self._base_cam is not None:
                self._base_cam.stop()
        except Exception:
            pass
        try:
            if self._wrist_cam is not None:
                self._wrist_cam.stop()
        except Exception:
            pass
        try:
            self._ctrl.disconnect()
        except Exception:
            pass
        try:
            self._rcv.disconnect()
        except Exception:
            pass


def make_real_env(
    ur_ip: str,
    *,
    reset_position: Optional[List[float]] = None,
    rs_base_serial: Optional[str] = None,
    rs_wrist_serial: Optional[str] = None,
    gripper_port: int = 0,
    fake_cam: bool = False,
) -> RealEnv:
    """Factory function to create a RealEnv instance."""
    return RealEnv(
        ur_ip,
        reset_position=reset_position,
        rs_base_serial=rs_base_serial,
        rs_wrist_serial=rs_wrist_serial,
        gripper_port=gripper_port,
        fake_cam=fake_cam,
    )
