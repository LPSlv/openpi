"""
Convert UR5 raw episodes to LeRobot format with smooth gripper transitions.

Instead of instant 0→1 / 1→0 gripper jumps, this creates a linear ramp over
RAMP_FRAMES frames. This gives the model more transition signal during training,
matching how DROID/ALOHA datasets record actual gripper feedback.

The ramp is applied to state[6], gripper, and actions[6], maintaining the
forward-looking property: action[t] = state[t+1].

Example:
  uv run python ur5/scripts/convert_ur5_smooth_gripper.py \
    --raw_dir raw_episodes_blueblock2 \
    --repo_id LPSlvlv/ur5_blueblock_box_20_smooth \
    --ramp_frames 5
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset


RAMP_FRAMES_DEFAULT = 5


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


def _smooth_gripper(gripper_values: np.ndarray, ramp_frames: int) -> np.ndarray:
    """Replace instant 0→1 / 1→0 transitions with linear ramps placed AFTER the command.

    The raw data records an instant jump at frame T because state[6]=g_cmd and g_cmd
    switched at frame T. Real physical gripper lag means the actual sensor reading
    would be: open at T-1, partially closed at T, fully closed at T+1 (~100ms lag for
    a Robotiq HAND-E at 10 Hz).

    For a close at frame T (where the raw data shows the jump 0→1):
      Before: ...0, 0, 0, 1, 1, 1...
      After (ramp_frames=2): ...0, 0, 0, 0.5, 1, 1...
      After (ramp_frames=3): ...0, 0, 0, 0.33, 0.67, 1...

    The ramp fills frames [T, T+ramp_frames-1] with linearly interpolated values
    from (old, new]. The frame BEFORE the command stays at the old value. The last
    frame of the ramp reaches the new value.

    Detection runs on the ORIGINAL data before any ramps are applied, so overlapping
    transitions (close followed by open within ramp_frames) don't corrupt each other.
    """
    original = gripper_values.copy().astype(np.float64)
    result = original.copy()
    n = len(result)

    # First pass: detect all transitions in the ORIGINAL (unmodified) data.
    transitions = []  # list of (frame, old_value, new_value)
    for i in range(1, n):
        diff = original[i] - original[i - 1]
        if abs(diff) >= 0.5:
            transitions.append((i, float(original[i - 1]), float(original[i])))

    # Second pass: apply ramp AFTER each transition.
    for t_frame, old_val, new_val in transitions:
        ramp_end = min(t_frame + ramp_frames, n)
        ramp_len = ramp_end - t_frame
        if ramp_len <= 0:
            continue
        for j in range(ramp_len):
            # j+1 so that the first ramp frame is NOT the old value (it already was)
            # and the last ramp frame IS the new value (full close/open reached).
            result[t_frame + j] = old_val + (new_val - old_val) * (j + 1) / ramp_len

    return result.astype(np.float32)


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    fps: int = 10,
    robot_type: str = "ur5e",
    ramp_frames: int = RAMP_FRAMES_DEFAULT,
    push_to_hub: bool = True,
) -> None:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=int(fps),
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "joints": {"dtype": "float32", "shape": (6,), "names": ["joints"]},
            "gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episode_dirs = _iter_episode_dirs(raw_dir)
    if not episode_dirs:
        raise RuntimeError(f"No episodes found in {raw_dir}")

    total_transitions_before = 0
    total_ramp_frames = 0

    for ep_idx, ep_dir in enumerate(episode_dirs):
        meta = json.loads((ep_dir / "meta.json").read_text())
        default_task = meta.get("prompt", "")

        # First pass: load all steps to get gripper timeline
        steps = []
        with (ep_dir / "steps.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                steps.append(json.loads(line))

        n = len(steps)
        if n == 0:
            continue

        # Extract original gripper state values
        gripper_states = np.array([s["state"][6] for s in steps], dtype=np.float32)

        # Count original transitions
        orig_transitions = sum(1 for i in range(1, n) if abs(gripper_states[i] - gripper_states[i-1]) > 0.5)
        total_transitions_before += orig_transitions

        # Apply smooth ramp to gripper states
        smooth_states = _smooth_gripper(gripper_states, ramp_frames)

        # Build forward-looking actions for gripper: action[t] = state[t+1]
        smooth_actions_gripper = np.zeros(n, dtype=np.float32)
        for i in range(n - 1):
            smooth_actions_gripper[i] = smooth_states[i + 1]
        smooth_actions_gripper[n - 1] = smooth_states[n - 1]  # Last frame: hold

        # Count ramp frames (where gripper is not 0 or 1)
        ramp_count = sum(1 for v in smooth_states if 0.01 < v < 0.99)
        total_ramp_frames += ramp_count

        # Print episode summary
        transitions_str = []
        for i in range(1, n):
            if abs(gripper_states[i] - gripper_states[i-1]) > 0.5:
                direction = "CLOSE" if gripper_states[i] > gripper_states[i-1] else "OPEN"
                transitions_str.append(f"{direction}@{i}")
        print(f"  Episode {ep_idx}: {n} frames, transitions: {', '.join(transitions_str)}, ramp frames: {ramp_count}")

        # Second pass: write frames with smoothed gripper
        for i, step in enumerate(steps):
            image_path = ep_dir / step["image_path"]
            wrist_path = ep_dir / step["wrist_image_path"]
            task = step.get("task", default_task)

            state = np.asarray(step["state"], dtype=np.float32)
            actions = np.asarray(step["actions"], dtype=np.float32)

            # Replace gripper values with smoothed versions
            state[6] = smooth_states[i]
            actions[6] = smooth_actions_gripper[i]

            dataset.add_frame({
                "image": _imread_rgb(image_path),
                "wrist_image": _imread_rgb(wrist_path),
                "state": state,
                "joints": state[:6],
                "gripper": np.array([smooth_states[i]], dtype=np.float32),
                "actions": actions,
                "task": task,
            })

        # Squeeze gripper scalars for HF validation
        if "gripper" in dataset.episode_buffer:
            for j in range(len(dataset.episode_buffer["gripper"])):
                v = dataset.episode_buffer["gripper"][j]
                if hasattr(v, 'item'):
                    dataset.episode_buffer["gripper"][j] = v.item()
        dataset.save_episode()

    dataset.finalize()

    print(f"\n=== SUMMARY ===")
    print(f"Original transitions: {total_transitions_before} (single-frame jumps)")
    print(f"New ramp frames (intermediate values): {total_ramp_frames}")
    print(f"Ramp width: {ramp_frames} frames per transition")

    if push_to_hub:
        print(f"\nPushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub(
            tags=["ur5", "ur5e", "smooth-gripper"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Successfully pushed to: {repo_id}")
    else:
        print(f"\nDataset saved locally at {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
