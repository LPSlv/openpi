"""UR5 environment wrapper implementing the Environment interface."""

from typing import List, Optional  # noqa: UP035

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.ur5 import real_env as _real_env


class UR5RealEnvironment(_environment.Environment):
    """An environment for a UR5 robot on real hardware."""

    def __init__(
        self,
        ur_ip: str,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        rs_base_serial: Optional[str] = None,
        rs_wrist_serial: Optional[str] = None,
        gripper_port: int = 0,
        fake_cam: bool = False,
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        self._env = _real_env.make_real_env(
            ur_ip=ur_ip,
            reset_position=reset_position,
            rs_base_serial=rs_base_serial,
            rs_wrist_serial=rs_wrist_serial,
            gripper_port=gripper_port,
            fake_cam=fake_cam,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None

    @override
    def reset(self) -> None:
        """Reset the environment."""
        self._env.reset()
        # Get observation after reset and store as timestep
        obs = self._env.get_observation()
        self._ts = {
            "observation": obs,
            "step_type": "first",
        }

    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete."""
        return False  # UR5 episodes don't have a natural completion condition

    @override
    def get_observation(self) -> dict:
        """Get current observation."""
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts["observation"].copy()

        # Policy server expects the same key conventions used across environments:
        # - "observation/state": float32 (7,) = joints(6) + gripper(1)
        # - "observation/image": uint8 (H,W,3) base camera
        # - "observation/wrist_image": uint8 (H,W,3) wrist camera
        #
        # Note: Model-side UR5 transforms (`openpi.policies.ur5_policy.UR5Inputs`) handle HWC uint8.
        state = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)

        # Process images into uint8 HWC at the model resolution.
        base_img = obs["images"].get("base")
        wrist_img = obs["images"].get("wrist")

        def _proc(img):
            img = image_tools.resize_with_pad(img, self._render_height, self._render_width)
            return image_tools.convert_to_uint8(img)

        base_img = _proc(base_img) if base_img is not None else np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
        wrist_img = _proc(wrist_img) if wrist_img is not None else base_img.copy()

        return {
            "observation/state": state,
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
        }

    @override
    def apply_action(self, action: dict) -> None:
        """Apply action to the environment."""
        # Extract actions array
        actions = action["actions"]
        if actions.ndim == 1:
            actions = actions[None, :]

        # Apply first action in the chunk
        self._env.step(actions[0])

        # Update timestep with new observation
        obs = self._env.get_observation()
        self._ts = {
            "observation": obs,
            "step_type": "mid",
        }

    def close(self) -> None:
        """Clean up resources."""
        self._env.close()
