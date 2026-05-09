#!/usr/bin/env python3
"""Replay recorded inference observations through a model and analyze gripper outputs.

Loads .npz files saved by the bridge (RECORD_DIR) and runs them through a checkpoint
for offline analysis. Optionally saves side-by-side image comparisons with dataset frames.

Usage:
    JAX_PLATFORMS=cpu uv run ur5/scripts/replay_inference_recording.py \
        --record-dir /tmp/inference_recording \
        --config pi0_ur5 \
        --checkpoint checkpoints/pi0_ur5/ur5_blueblock_box_v2_40_smooth-8/1999

    # With visual comparison to dataset:
    JAX_PLATFORMS=cpu uv run ur5/scripts/replay_inference_recording.py \
        --record-dir /tmp/inference_recording \
        --config pi0_ur5 \
        --checkpoint checkpoints/pi0_ur5/ur5_blueblock_box_v2_40_smooth-8/1999 \
        --save-images /tmp/replay_comparison
"""

import argparse
import copy
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record-dir", required=True, help="Directory with step_NNNN.npz files from bridge RECORD_DIR")
    parser.add_argument("--config", default="pi0_ur5", help="Training config name")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory path")
    parser.add_argument("--save-images", default="", help="If set, save side-by-side images to this directory")
    parser.add_argument("--dataset", default="", help="LeRobot dataset repo_id for visual comparison")
    parser.add_argument("--prompt", default="Pick up the blue block and place it in the cardboard box")
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps to replay (0=all)")
    args = parser.parse_args()

    rec_dir = pathlib.Path(args.record_dir)
    npz_files = sorted(rec_dir.glob("step_*.npz"))
    if not npz_files:
        print(f"No step_*.npz files found in {rec_dir}")
        sys.exit(1)
    print(f"Found {len(npz_files)} recorded steps in {rec_dir}")

    if args.max_steps > 0:
        npz_files = npz_files[:args.max_steps]

    print(f"Loading model: config={args.config} checkpoint={args.checkpoint}")
    from openpi.policies import policy_config as pc
    from openpi.training import config as cfg

    config = cfg.get_config(args.config)
    policy = pc.create_trained_policy(config, args.checkpoint, default_prompt=args.prompt)
    print(f"Model loaded. action_horizon={config.model.action_horizon}")

    # the dataset is only needed when --dataset is passed for side-by-side comparison
    dataset = None
    if args.dataset:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(args.dataset)
        print(f"Dataset loaded: {args.dataset} ({len(dataset)} frames)")

    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)

    print(f"\n{'Step':>4s}  {'Gripper(bridge)':>15s}  {'Gripper(replay)':>15s}  {'Match':>5s}  {'ImgMean':>7s}")
    print("-" * 70)

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        step_name = npz_path.stem

        image = data["image"]           # (H, W, 3) uint8
        wrist_image = data["wrist_image"]  # (H, W, 3) uint8
        state = data["state"]           # (7,) float32
        bridge_actions = data["actions"]  # (horizon, 7) float32
        prompt = str(data["prompt"]) if "prompt" in data else args.prompt

        obs = {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state.astype(np.float32),
            "prompt": prompt,
        }

        out = policy.infer(copy.deepcopy(obs))
        replay_actions = out["actions"]  # (horizon, 7)

        bridge_g = bridge_actions[:, 6]
        replay_g = replay_actions[:, 6]

        bridge_g_mean = bridge_g.mean()
        replay_g_mean = replay_g.mean()
        match = abs(bridge_g_mean - replay_g_mean) < 0.05

        step_num = int(step_name.split("_")[1])
        print(
            f"{step_num:4d}  "
            f"mean={bridge_g_mean:+.4f} [{bridge_g.min():+.3f},{bridge_g.max():+.3f}]  "
            f"mean={replay_g_mean:+.4f} [{replay_g.min():+.3f},{replay_g.max():+.3f}]  "
            f"{'OK' if match else 'DIFF':>5s}  "
            f"{image.mean():7.1f}"
        )

        if args.save_images:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            bgr_wrist = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
            vis = np.hstack([bgr, bgr_wrist])

            cv2.putText(vis, f"Step {step_num} | gripper={replay_g_mean:.3f}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(vis, f"state[6]={state[6]:.3f}", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.imwrite(os.path.join(args.save_images, f"{step_name}.jpg"), vis)

    print(f"\nReplay complete. {len(npz_files)} steps analyzed.")
    if args.save_images:
        print(f"Images saved to {args.save_images}")


if __name__ == "__main__":
    main()
