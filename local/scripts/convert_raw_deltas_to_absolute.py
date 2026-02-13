"""
Convert existing raw episodes (recorded with the old backward-looking delta script)
to forward-looking absolute actions.

Old format:  action[i] = q[i] - q[i-1]  (backward-looking delta, off-by-one bug)
New format:  action[i] = state[i+1]      (forward-looking absolute target)

The state values are already absolute and correct, so we just shift them:
  action[i] = state[i+1]   for i < N-1
  action[N-1] = state[N-1] (hold position, last step)

Also updates meta.json action_spec.

Usage:
  uv run python local/scripts/convert_raw_deltas_to_absolute.py --raw_dir raw_episodes
  uv run python local/scripts/convert_raw_deltas_to_absolute.py --raw_dir raw_episodes --dry_run
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import tyro


def _iter_episode_dirs(raw_dir: Path) -> list[Path]:
    eps = []
    for p in sorted(raw_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "steps.jsonl").exists() and (p / "meta.json").exists():
            eps.append(p)
    return eps


def convert_episode(ep_dir: Path, *, dry_run: bool = False) -> int:
    """Convert one episode. Returns number of steps converted."""
    steps_path = ep_dir / "steps.jsonl"

    # Read all steps
    steps: list[dict] = []
    with steps_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))

    if len(steps) == 0:
        print(f"  Skipping {ep_dir.name}: no steps")
        return 0

    # Convert: action[i] = state[i+1], action[N-1] = state[N-1]
    for i in range(len(steps)):
        if i + 1 < len(steps):
            steps[i]["actions"] = steps[i + 1]["state"]
        else:
            steps[i]["actions"] = steps[i]["state"]

    if dry_run:
        # Show a sample
        if len(steps) > 1:
            s0 = steps[0]
            print(f"  Step 0: state={s0['state'][:3]}...  action(new)={s0['actions'][:3]}...")
        return len(steps)

    # Backup original
    backup_path = ep_dir / "steps.jsonl.bak"
    if not backup_path.exists():
        shutil.copy2(steps_path, backup_path)

    # Write converted steps
    with steps_path.open("w", encoding="utf-8") as f:
        for step in steps:
            f.write(json.dumps(step) + "\n")

    # Update meta.json action_spec
    meta_path = ep_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    if "action_spec" in meta:
        meta["action_spec"]["desc"] = "absolute_q(6) + absolute gripper_cmd(1)"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    return len(steps)


def main(
    raw_dir: Path = Path("raw_episodes"),
    *,
    dry_run: bool = False,
) -> None:
    """Convert raw episodes from backward-looking deltas to forward-looking absolute actions.

    Args:
        raw_dir: Directory containing raw episode folders.
        dry_run: If True, show what would be changed without writing.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    episode_dirs = _iter_episode_dirs(raw_dir)
    if not episode_dirs:
        print(f"No episodes found in {raw_dir}")
        return

    print(f"Found {len(episode_dirs)} episodes in {raw_dir}")
    if dry_run:
        print("DRY RUN - no files will be modified\n")

    total_steps = 0
    for ep_dir in episode_dirs:
        n = convert_episode(ep_dir, dry_run=dry_run)
        print(f"  {ep_dir.name}: {n} steps {'(would convert)' if dry_run else 'converted'}")
        total_steps += n

    print(f"\nTotal: {total_steps} steps across {len(episode_dirs)} episodes")
    if not dry_run:
        print("Backups saved as steps.jsonl.bak in each episode directory")


if __name__ == "__main__":
    tyro.cli(main)
