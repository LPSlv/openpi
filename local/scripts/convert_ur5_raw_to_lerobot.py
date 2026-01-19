"""
Convert UR5 raw episodes (recorded by openpi/local/scripts/ur5_replay_and_record_raw.py)
to LeRobot format.

This mirrors examples/libero/convert_libero_data_to_lerobot.py, but reads our simple
raw-on-disk episode folders instead of TFDS/RLDS.

Example:
  uv run python openpi/local/scripts/convert_ur5_raw_to_lerobot.py \
    --raw_dir raw_episodes \
    --repo_id your_hf_username/ur5_freedrive \
    --fps 10

Note: By default, the script will push the dataset to Hugging Face Hub.
      To skip pushing, use --no-push-to-hub flag.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (256, 256):
        raise ValueError(f"Expected 256x256 image at {path}, got {rgb.shape}")
    return rgb.astype(np.uint8, copy=False)


def _iter_episode_dirs(raw_dir: Path) -> list[Path]:
    eps = []
    for p in sorted(raw_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "steps.jsonl").exists() and (p / "meta.json").exists():
            eps.append(p)
    return eps


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    fps: int = 10,
    robot_type: str = "ur5e",
    push_to_hub: bool = True,
) -> None:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=int(fps),
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

    episode_dirs = _iter_episode_dirs(raw_dir)
    if not episode_dirs:
        raise RuntimeError(f"No episodes found in {raw_dir} (expected */steps.jsonl + */meta.json)")

    for ep_dir in episode_dirs:
        meta = json.loads((ep_dir / "meta.json").read_text())
        default_task = meta.get("prompt", "")

        with (ep_dir / "steps.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                step = json.loads(line)
                image_path = ep_dir / step["image_path"]
                wrist_path = ep_dir / step["wrist_image_path"]
                task = step.get("task", default_task)

                dataset.add_frame(
                    {
                        "image": _imread_rgb(image_path),
                        "wrist_image": _imread_rgb(wrist_path),
                        "state": np.asarray(step["state"], dtype=np.float32),
                        "joints": np.asarray(step["state"], dtype=np.float32)[:6],
                        "gripper": np.asarray(step["state"], dtype=np.float32)[6:7],
                        "actions": np.asarray(step["actions"], dtype=np.float32),
                        "task": task,
                    }
                )

        dataset.save_episode()

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub(
            tags=["ur5", "ur5e", "freedrive"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Successfully pushed dataset to Hugging Face Hub: {repo_id}")
    else:
        print(f"Dataset saved locally at {output_path}. Use --push-to-hub to push to Hugging Face.")


if __name__ == "__main__":
    tyro.cli(main)
