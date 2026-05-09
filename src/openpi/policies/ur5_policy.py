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
        # Support three input formats:
        # 1. Training with separate joints/gripper (your datasets)
        # 2. Training with combined state (F-Fer datasets: observation.state is 7D)
        # 3. Inference (observation/state + observation/image + observation/wrist_image)
        if "joints" in data and "gripper" in data:
            # Format 1: separate joints (6D) + gripper (1D)
            joints = np.asarray(data["joints"]).reshape(-1)
            gripper = np.asarray(data["gripper"]).reshape(-1)
            state = np.concatenate([joints, gripper])
            base_image = _parse_image(data["base_rgb"])
            wrist_image = _parse_image(data["wrist_rgb"])
        elif "joints" in data and "gripper" not in data:
            # Format 2: combined state mapped to "joints" key (e.g. F-Fer's observation.state → joints)
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

        # Model image-key conventions differ slightly between PI0/PI05 vs PI0_FAST.
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                # PI0_FAST expects (base_0_rgb, base_1_rgb, wrist_0_rgb) and we do not mask padding images.
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

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}

