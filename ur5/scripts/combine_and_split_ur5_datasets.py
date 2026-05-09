"""
Combine UR5 raw episodes from multiple directories, shuffle, and create
nested LeRobot datasets of increasing size.

All episodes must already have absolute forward-looking actions
(action[i] = state[i+1]).  Run convert_raw_deltas_to_absolute.py first
if needed.

Episodes are shuffled once with a deterministic seed. Datasets are nested
prefixes: the 1-episode dataset is a subset of the 5-episode dataset, etc.

Example:
  uv run python ur5/scripts/combine_and_split_ur5_datasets.py \
    --raw-dirs raw_episodes_blueblock2 raw_episodes_blueblock_2 \
    --repo-id-prefix LPSlvlv/ur5_blueblock_box \
    --splits 1 5 10 15 20 \
    --dry-run
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# helpers, also live in convert_ur5_raw_to_lerobot.py
# ---------------------------------------------------------------------------

def _imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (256, 256):
        raise ValueError(f"Expected 256x256 image at {path}, got {rgb.shape}")
    return rgb.astype(np.uint8, copy=False)


def _iter_replay_dirs(raw_dir: Path) -> list[Path]:
    """Return sorted list of valid ur5_replay_* episode dirs."""
    eps = []
    for p in sorted(raw_dir.iterdir()):
        if not p.is_dir() or not p.name.startswith("ur5_replay_"):
            continue
        if (p / "steps.jsonl").exists() and (p / "meta.json").exists():
            eps.append(p)
    return eps


FEATURES = {
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
}


# ---------------------------------------------------------------------------
# Episode metadata
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EpisodeInfo:
    path: Path
    fps: float
    action_desc: str
    num_steps: int
    prompt: str


def _discover_episodes(raw_dirs: list[Path]) -> list[EpisodeInfo]:
    episodes: list[EpisodeInfo] = []
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        for ep_dir in _iter_replay_dirs(raw_dir):
            meta = json.loads((ep_dir / "meta.json").read_text())
            num_steps = sum(1 for line in (ep_dir / "steps.jsonl").open() if line.strip())
            episodes.append(EpisodeInfo(
                path=ep_dir,
                fps=float(meta.get("fps", 10.0)),
                action_desc=meta.get("action_spec", {}).get("desc", "unknown"),
                num_steps=num_steps,
                prompt=meta.get("prompt", ""),
            ))
    return episodes


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def _read_steps(ep_dir: Path) -> list[dict]:
    steps = []
    with (ep_dir / "steps.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def _downsample_steps(steps: list[dict], source_fps: float, target_fps: float) -> list[dict]:
    """Downsample steps and fix forward-looking actions."""
    ratio = int(round(source_fps / target_fps))
    if ratio <= 1:
        return steps

    kept_indices = list(range(0, len(steps), ratio))
    result = []
    for i, orig_idx in enumerate(kept_indices):
        step = copy.deepcopy(steps[orig_idx])
        # rewire the forward-looking action to the next kept frame
        if i + 1 < len(kept_indices):
            next_idx = kept_indices[i + 1]
            step["actions"] = steps[next_idx]["state"]
        # the last kept step keeps its original action (hold position)
        result.append(step)
    return result


# ---------------------------------------------------------------------------
# Dataset creation for one split
# ---------------------------------------------------------------------------

def _create_dataset(
    episodes: list[EpisodeInfo],
    repo_id: str,
    target_fps: int,
    robot_type: str,
    push_to_hub: bool,
) -> None:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=target_fps,
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_frames = 0
    for ep_info in episodes:
        steps = _read_steps(ep_info.path)

        if ep_info.fps != target_fps:
            steps = _downsample_steps(steps, ep_info.fps, target_fps)

        for step in steps:
            image_path = ep_info.path / step["image_path"]
            wrist_path = ep_info.path / step["wrist_image_path"]
            task = step.get("task", ep_info.prompt)

            state = np.asarray(step["state"], dtype=np.float32)
            dataset.add_frame(
                {
                    "image": _imread_rgb(image_path),
                    "wrist_image": _imread_rgb(wrist_path),
                    "state": state,
                    "joints": state[:6],
                    "gripper": state[6:7],
                    "actions": np.asarray(step["actions"], dtype=np.float32),
                    "task": task,
                }
            )
            total_frames += 1

        # lerobot maps shape (1,) features to a HF Value (scalar) but validation
        # writes them back as (1,) numpy arrays, so we squeeze before save
        for key, ft in FEATURES.items():
            if ft.get("shape") == (1,) and key in dataset.episode_buffer:
                for j in range(len(dataset.episode_buffer[key])):
                    dataset.episode_buffer[key][j] = dataset.episode_buffer[key][j].item()
        dataset.save_episode()

    dataset.finalize()
    print(f"  Created {repo_id}: {len(episodes)} episodes, {total_frames} frames")

    if push_to_hub:
        print(f"  Pushing {repo_id} to Hub...")
        dataset.push_to_hub(
            tags=["ur5", "ur5e", "pick-and-place", "blue-block"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"  Pushed {repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Args:
    raw_dirs: list[Path] = dataclasses.field(
        default_factory=lambda: [Path("raw_episodes_blueblock2"), Path("raw_episodes_blueblock_2")]
    )
    repo_id_prefix: str = "LPSlvlv/ur5_blueblock_box"
    splits: list[int] = dataclasses.field(default_factory=lambda: [1, 5, 10, 15, 20])
    target_fps: int = 10
    seed: int = 42
    robot_type: str = "ur5e"
    push_to_hub: bool = True
    dry_run: bool = False


def main(args: Args) -> None:
    episodes = _discover_episodes(args.raw_dirs)
    if not episodes:
        raise RuntimeError("No episodes found")

    for ep in episodes:
        if "absolute" not in ep.action_desc:
            raise ValueError(
                f"{ep.path.name}: expected absolute actions, got '{ep.action_desc}'. "
                f"Run convert_raw_deltas_to_absolute.py first."
            )

    rng = random.Random(args.seed)
    rng.shuffle(episodes)

    max_split = max(args.splits)
    if max_split > len(episodes):
        raise ValueError(f"Requested {max_split} episodes but only found {len(episodes)}")

    print(f"Discovered {len(episodes)} episodes (seed={args.seed} shuffle):\n")
    for i, ep in enumerate(episodes):
        fps_tag = f"{int(ep.fps)}Hz"
        ds_tag = f"->{args.target_fps}Hz" if ep.fps != args.target_fps else ""
        est_frames = ep.num_steps // int(round(ep.fps / args.target_fps)) if ep.fps != args.target_fps else ep.num_steps
        print(f"  {i:2d}: {ep.path.parent.name}/{ep.path.name}  ({fps_tag}{ds_tag}, {est_frames} frames)")

    print(f"\nDatasets to create:")
    for count in sorted(args.splits):
        subset = episodes[:count]
        total_est = sum(
            ep.num_steps // int(round(ep.fps / args.target_fps)) if ep.fps != args.target_fps else ep.num_steps
            for ep in subset
        )
        repo_id = f"{args.repo_id_prefix}_{count}"
        print(f"  {repo_id}  ({count} episodes, ~{total_est} frames)")

    if args.dry_run:
        print("\nDRY RUN, no datasets created.")
        return

    # build the smallest split first so we catch errors before the long ones
    print()
    for count in sorted(args.splits):
        subset = episodes[:count]
        repo_id = f"{args.repo_id_prefix}_{count}"
        _create_dataset(subset, repo_id, args.target_fps, args.robot_type, args.push_to_hub)

    print("\nDone!")


if __name__ == "__main__":
    main(tyro.cli(Args))
