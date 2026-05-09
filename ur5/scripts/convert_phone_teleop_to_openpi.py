"""
Convert phone-teleoperated UR5 dataset (from GauthierBassereau/lerobot fork)
to OpenPI-compatible LeRobot format.

NOTE: For new recordings, use the integrated flow in lerobot-phone instead:
  cd /home/ims/lerobot-phone && python examples/phone_to_ur5/record.py
This records and auto-converts to openpi format in one step.
This standalone script is kept as a fallback for datasets recorded with the old flow.

The phone teleop fork records EE-space actions and observations including joint
positions (with keep_joints=True). This script extracts joint positions, computes
joint-space actions (absolute, forward-looking), and reformats for openpi training.

Source dataset fields (recorded by lerobot phone_to_ur5/record.py):
  observation.state (13,) with names metadata for joints/gripper indices
  observation.images.front / observation.images.wrist  (camera images)

Target fields (openpi convention, see convert_ur5_raw_to_lerobot.py):
  joints (6,) float32   - radians
  gripper (1,) float32   - [0, 1]
  state (7,) float32    - concat(joints, gripper)
  actions (7,) float32   - state[t+1] (forward-looking absolute)
  image (256, 256, 3)    - front camera
  wrist_image (256, 256, 3) - wrist camera

Example:
  uv run python ur5/scripts/convert_phone_teleop_to_openpi.py \\
    --source-repo-id LPSlvlv/ur5_phone_teleop_1 \\
    --dest-repo-id LPSlvlv/ur5_phone_openpi_1 \\
    --fps 10
"""

from __future__ import annotations

import shutil

import cv2
import numpy as np
import torch
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

# Joint observation keys in the phone-teleop dataset (order matters: matches UR5 joint order).
SRC_JOINT_KEYS = [
    "observation.shoulder_pan.pos",
    "observation.shoulder_lift.pos",
    "observation.elbow.pos",
    "observation.wrist_1.pos",
    "observation.wrist_2.pos",
    "observation.wrist_3.pos",
]
SRC_GRIPPER_KEY = "observation.ee.gripper_pos"
SRC_FRONT_CAM_KEY = "observation.images.front"
SRC_WRIST_CAM_KEY = "observation.images.wrist"


def _to_numpy_image(img) -> np.ndarray:
    """Convert a LeRobot image (torch tensor or PIL) to numpy uint8 HWC RGB."""
    if isinstance(img, torch.Tensor):
        # LeRobot returns (C, H, W) float [0, 1] after hf_transform_to_torch
        arr = img.permute(1, 2, 0).numpy()
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr
    # PIL Image
    return np.asarray(img, dtype=np.uint8)


def _resize_256(img: np.ndarray) -> np.ndarray:
    """Resize HWC image to 256x256 if needed."""
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return img


def main(
    source_repo_id: str,
    dest_repo_id: str,
    *,
    fps: int = 10,
    downsample: int = 1,
    joints_in_degrees: bool = True,
    robot_type: str = "ur5e",
    push_to_hub: bool = True,
) -> None:
    """Convert a phone-teleop UR5 dataset to openpi format.

    Args:
        source_repo_id: HuggingFace repo ID of the phone-teleop dataset.
        dest_repo_id: HuggingFace repo ID for the converted dataset.
        fps: Target FPS for the output dataset.
        downsample: Take every Nth frame (e.g. 3 to go from 30Hz→10Hz).
        joints_in_degrees: If True, convert joint values from degrees to radians.
        robot_type: Robot type string for dataset metadata.
        push_to_hub: Push the converted dataset to HuggingFace Hub.
    """
    print(f"Loading source dataset: {source_repo_id}")
    src = LeRobotDataset(source_repo_id)

    # Validate expected keys exist
    sample = src[0]
    available_keys = list(sample.keys())
    for key in [*SRC_JOINT_KEYS, SRC_GRIPPER_KEY]:
        if key not in available_keys:
            raise KeyError(
                f"Expected key '{key}' not found in source dataset. "
                f"Available keys: {available_keys}"
            )
    print(f"Source dataset loaded: {len(src)} frames, keys: {available_keys}")

    # Determine camera keys (try expected names, fall back to discovery)
    front_key = SRC_FRONT_CAM_KEY
    wrist_key = SRC_WRIST_CAM_KEY
    if front_key not in available_keys:
        # Try without 'images.' prefix
        front_key = "observation.front"
        wrist_key = "observation.wrist"
    if front_key not in available_keys:
        raise KeyError(
            f"Cannot find camera keys. Tried '{SRC_FRONT_CAM_KEY}' and "
            f"'observation.front'. Available: {available_keys}"
        )
    print(f"Using camera keys: front='{front_key}', wrist='{wrist_key}'")

    # Clean destination
    output_path = HF_LEROBOT_HOME / dest_repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dest = LeRobotDataset.create(
        repo_id=dest_repo_id,
        robot_type=robot_type,
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "joints": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joints"],
            },
            "gripper": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Get episode boundaries
    num_episodes = src.meta.total_episodes
    print(f"Processing {num_episodes} episodes (downsample={downsample})...")

    for ep_idx in range(num_episodes):
        from_idx = src.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = src.meta.episodes["dataset_to_index"][ep_idx]

        # Select frame indices with downsampling
        frame_indices = list(range(from_idx, to_idx, downsample))
        if not frame_indices:
            print(f"  Episode {ep_idx}: empty after downsampling, skipping")
            continue

        # First pass: collect states for action computation
        states = []
        for i in frame_indices:
            frame = src[i]
            joints = np.array(
                [frame[k].item() for k in SRC_JOINT_KEYS], dtype=np.float32
            )
            if joints_in_degrees:
                joints = np.deg2rad(joints)
            gripper = np.array(
                [frame[SRC_GRIPPER_KEY].item() / 100.0], dtype=np.float32
            )
            states.append(np.concatenate([joints, gripper]))

        states = np.stack(states)

        # Compute forward-looking absolute actions: action[t] = state[t+1]
        actions = np.empty_like(states)
        actions[:-1] = states[1:]
        actions[-1] = states[-1]  # hold position for last frame

        # Second pass: write frames with images
        task = src[frame_indices[0]].get("task", "")
        for t, i in enumerate(frame_indices):
            frame = src[i]
            front_img = _resize_256(_to_numpy_image(frame[front_key]))
            wrist_img = _resize_256(_to_numpy_image(frame[wrist_key]))

            dest.add_frame(
                {
                    "image": front_img,
                    "wrist_image": wrist_img,
                    "state": states[t],
                    "joints": states[t][:6],
                    "gripper": states[t][6:7],
                    "actions": actions[t],
                    "task": task,
                }
            )

        dest.save_episode()
        print(f"  Episode {ep_idx}: {len(frame_indices)} frames")

    dest.finalize()
    print(f"Dataset finalized: {dest_repo_id}")

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub: {dest_repo_id}")
        dest.push_to_hub(
            tags=["ur5", "ur5e", "phone-teleop"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Pushed: {dest_repo_id}")
    else:
        print(f"Saved locally at {output_path}. Use --push-to-hub to upload.")


if __name__ == "__main__":
    tyro.cli(main)
