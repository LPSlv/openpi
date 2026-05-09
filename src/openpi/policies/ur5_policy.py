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
        # three accepted input shapes:
        #   1. training with separate joints (6D) + gripper (1D)
        #   2. training with a combined 7D state mapped to the "joints" key
        #      (this is how F-Fer's datasets land here, observation.state is 7D)
        #   3. inference, where observation/state is already 7D
        if "joints" in data and "gripper" in data:
            joints = np.asarray(data["joints"]).reshape(-1)
            gripper = np.asarray(data["gripper"]).reshape(-1)
            state = np.concatenate([joints, gripper])
            base_image = _parse_image(data["base_rgb"])
            wrist_image = _parse_image(data["wrist_rgb"])
        elif "joints" in data and "gripper" not in data:
            full = np.asarray(data["joints"]).reshape(-1)
            if full.shape[0] == 7:
                state = full
            elif full.shape[0] == 6:
                state = np.concatenate([full, np.zeros(1, dtype=full.dtype)])
            else:
                raise ValueError(f"Expected joints shape (6,) or (7,), got {full.shape}")
            base_image = _parse_image(data["base_rgb"])
            wrist_image = _parse_image(data["wrist_rgb"])
        elif "observation/state" in data:
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

        # image-key conventions differ between pi0/pi05 and pi0_fast
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                # pi0_fast wants (base_0_rgb, base_1_rgb, wrist_0_rgb) and doesn't mask the padding image
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # forward the language instruction
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
