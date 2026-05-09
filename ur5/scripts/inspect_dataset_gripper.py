#!/usr/bin/env python3
"""Inspect gripper values in the UR5 LeRobot dataset.

Verifies:
1. Gripper state vs action correlation per frame
2. Transition frame count (where action[6] != state[6])
3. Forward-looking action alignment (action[t] == state[t+1])
4. Gripper timeline per episode
"""

import sys
import numpy as np

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "LPSlvlv/ur5_blueblock_box_20"

print(f"Loading dataset: {REPO_ID}")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(REPO_ID)
n = len(dataset)
print(f"Total frames: {n}")
print(f"Features: {list(dataset[0].keys())}")

# Collect all gripper state and action values
gripper_states = []
gripper_actions = []
episode_indices = []

for i in range(n):
    sample = dataset[i]
    gs = float(sample["gripper"])
    ga = float(sample["actions"][6])
    ep = int(sample["episode_index"])
    gripper_states.append(gs)
    gripper_actions.append(ga)
    episode_indices.append(ep)

gripper_states = np.array(gripper_states)
gripper_actions = np.array(gripper_actions)
episode_indices = np.array(episode_indices)

# Overall stats
print(f"\n=== GRIPPER STATS ===")
print(f"State  - min: {gripper_states.min():.4f}, max: {gripper_states.max():.4f}, mean: {gripper_states.mean():.4f}")
print(f"Action - min: {gripper_actions.min():.4f}, max: {gripper_actions.max():.4f}, mean: {gripper_actions.mean():.4f}")

# Correlation
corr = np.corrcoef(gripper_states, gripper_actions)[0, 1]
print(f"Correlation(state, action): {corr:.4f}")

# Echo frames (state == action)
THRESHOLD = 0.01
echo_mask = np.abs(gripper_states - gripper_actions) < THRESHOLD
echo_count = echo_mask.sum()
transition_count = n - echo_count
print(f"\n=== TRANSITION ANALYSIS ===")
print(f"Echo frames (state ≈ action):     {echo_count} ({100*echo_count/n:.1f}%)")
print(f"Transition frames (state ≠ action): {transition_count} ({100*transition_count/n:.2f}%)")

# Classify transitions
close_transitions = ((gripper_states < 0.5) & (gripper_actions > 0.5)).sum()
open_transitions = ((gripper_states > 0.5) & (gripper_actions < 0.5)).sum()
print(f"  Close transitions (0→1): {close_transitions}")
print(f"  Open transitions (1→0):  {open_transitions}")

# Forward-looking verification: action[t] should == state[t+1]
print(f"\n=== FORWARD-LOOKING VERIFICATION ===")
mismatches = 0
checked = 0
for ep in np.unique(episode_indices):
    ep_mask = episode_indices == ep
    ep_states = gripper_states[ep_mask]
    ep_actions = gripper_actions[ep_mask]
    for t in range(len(ep_states) - 1):
        checked += 1
        if abs(ep_actions[t] - ep_states[t + 1]) > THRESHOLD:
            mismatches += 1
            if mismatches <= 5:
                print(f"  Episode {ep}, frame {t}: action[t]={ep_actions[t]:.3f} != state[t+1]={ep_states[t+1]:.3f}")

print(f"Checked {checked} frame pairs, {mismatches} mismatches ({100*mismatches/max(checked,1):.2f}%)")
if mismatches == 0:
    print("PASS: action[t] == state[t+1] for all frames (forward-looking confirmed)")

# Per-episode gripper timeline
print(f"\n=== PER-EPISODE GRIPPER TIMELINE ===")
for ep in np.unique(episode_indices):
    ep_mask = episode_indices == ep
    ep_states = gripper_states[ep_mask]
    ep_actions = gripper_actions[ep_mask]
    n_ep = len(ep_states)

    # Find transition points
    transitions = []
    for t in range(n_ep - 1):
        if abs(ep_states[t + 1] - ep_states[t]) > 0.5:
            direction = "CLOSE" if ep_states[t + 1] > ep_states[t] else "OPEN"
            transitions.append((t + 1, direction))

    trans_str = ", ".join([f"{d} at frame {t}" for t, d in transitions])
    print(f"  Episode {ep:2d}: {n_ep:4d} frames, transitions: {trans_str or 'NONE'}")
