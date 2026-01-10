import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # allow CHW
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
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
        state = np.asarray(data["observation/state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != 7:
            raise ValueError(f"UR5 expects state of size 7 (6 joints + 1 gripper), got shape {state.shape}")
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        out = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            p = data["prompt"]
            if isinstance(p, bytes):
                p = p.decode("utf-8")
            out["prompt"] = p

        return out


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        a = np.asarray(data["actions"])
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[-1] < 7:
            raise ValueError(f"Model produced {a.shape[-1]} action dims, need at least 7 for UR5+gripper")
        return {"actions": a[:, :7]}

