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
        rs_base_serial: str | None = None,
        rs_wrist_serial: str | None = None,
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
        self._env.reset()
        obs = self._env.get_observation()
        self._ts = {
            "observation": obs,
            "step_type": "first",
        }

    @override
    def is_episode_complete(self) -> bool:
        # UR5 episodes don't have a natural completion condition
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts["observation"].copy()

        # the policy server expects the standard keys:
        #   observation/state        float32 (7,)  = joints(6) + gripper(1)
        #   observation/image        uint8 (H,W,3) base camera
        #   observation/wrist_image  uint8 (H,W,3) wrist camera
        # UR5Inputs in src/openpi/policies/ur5_policy.py handles HWC uint8 from here
        state = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)

        base_img = obs["images"].get("base")
        wrist_img = obs["images"].get("wrist")

        def _proc(img):
            img = image_tools.resize_with_pad(img, self._render_height, self._render_width)
            return image_tools.convert_to_uint8(img)

        base_img = (
            _proc(base_img)
            if base_img is not None
            else np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
        )
        wrist_img = _proc(wrist_img) if wrist_img is not None else base_img.copy()

        return {
            "observation/state": state,
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
        }

    @override
    def apply_action(self, action: dict) -> None:
        actions = action["actions"]
        if actions.ndim == 1:
            actions = actions[None, :]

        # only the first action in the chunk gets applied; the rest are dropped
        self._env.step(actions[0])

        obs = self._env.get_observation()
        self._ts = {
            "observation": obs,
            "step_type": "mid",
        }

    def close(self) -> None:
        self._env.close()
