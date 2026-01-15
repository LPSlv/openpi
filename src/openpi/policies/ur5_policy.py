import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_ur5_example() -> dict:
    """Creates a random input example for the UR5 policy."""
    joints = np.random.randn(6).astype(np.float32)
    gripper = np.array(0.0, dtype=np.float32)
    state = np.concatenate([joints, gripper[None]]).astype(np.float32)  # (7,)
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": state,
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Support both training format (joints/gripper/base_rgb/wrist_rgb) and
        # inference format (observation/state/observation/image/observation/wrist_image)
        if "joints" in data and "gripper" in data:
            # Training format: concatenate joints and gripper into state vector
            joints = np.asarray(data["joints"]).reshape(-1)
            gripper = np.asarray(data["gripper"]).reshape(-1)  # Ensure 1D array
            state = np.concatenate([joints, gripper])
            base_image = _parse_image(data["base_rgb"])
            wrist_image = _parse_image(data["wrist_rgb"])
        elif "observation/state" in data:
            # Inference format: state is already concatenated (joints + gripper)
            state = np.asarray(data["observation/state"], dtype=np.float32).reshape(-1)
            if state.shape[0] != 7:
                raise ValueError(f"Expected observation/state shape (7,), got {state.shape}.")
            base_image = _parse_image(data["observation/image"])
            wrist_image = _parse_image(data["observation/wrist_image"])
        else:
            raise ValueError(
                "Expected either (joints, gripper, base_rgb, wrist_rgb) or "
                "(observation/state, observation/image, observation/wrist_image) keys"
            )

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}

