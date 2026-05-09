#!/usr/bin/env python3
"""Compare training dataset images with recorded inference images side-by-side.

Loads a recorded inference observation and matching dataset frames, saves
a visual comparison grid to identify camera angle, lighting, or framing differences.

Usage:
    uv run ur5/scripts/compare_training_vs_inference_images.py \
        --record-dir /tmp/inference_recording \
        --dataset LPSlvlv/ur5_blueblock_box_v2_40_smooth \
        --output /tmp/comparison.jpg \
        --step 5
"""

import argparse
import os
import pathlib
import sys

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record-dir", required=True, help="Directory with step_NNNN.npz files")
    parser.add_argument("--dataset", default="LPSlvlv/ur5_blueblock_box_v2_40_smooth")
    parser.add_argument("--output", default="/tmp/training_vs_inference.jpg")
    parser.add_argument("--step", type=int, default=5, help="Which recorded step to compare")
    parser.add_argument("--dataset-frame", type=int, default=-1,
                        help="Dataset frame to compare (-1 = auto-find closest joints)")
    args = parser.parse_args()

    # Load recorded inference step
    rec_dir = pathlib.Path(args.record_dir)
    npz_path = rec_dir / f"step_{args.step:04d}.npz"
    if not npz_path.exists():
        print(f"Not found: {npz_path}")
        npz_files = sorted(rec_dir.glob("step_*.npz"))
        if npz_files:
            print(f"Available: {[f.stem for f in npz_files[:10]]}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    infer_base = data["image"]         # (H, W, 3) RGB
    infer_wrist = data["wrist_image"]
    infer_state = data["state"]        # (7,)
    print(f"Inference step {args.step}:")
    print(f"  base: {infer_base.shape} mean={infer_base.mean():.1f} std={infer_base.std():.1f}")
    print(f"  wrist: {infer_wrist.shape} mean={infer_wrist.mean():.1f} std={infer_wrist.std():.1f}")
    print(f"  state: joints={infer_state[:6].round(4)}, gripper={infer_state[6]:.4f}")

    # Load dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import einops
    from openpi_client import image_tools

    ds = LeRobotDataset(args.dataset)
    print(f"\nDataset: {args.dataset} ({len(ds)} frames)")

    # Find closest frame by joint distance
    if args.dataset_frame >= 0:
        ds_idx = args.dataset_frame
    else:
        print("Finding closest dataset frame by joint positions...")
        infer_joints = infer_state[:6]
        best_dist = float("inf")
        ds_idx = 0
        for i in range(0, len(ds), max(1, len(ds) // 2000)):  # Sample every Nth frame
            sample = ds[i]
            ds_joints = np.array(sample["joints"], dtype=np.float32).reshape(-1)
            dist = np.linalg.norm(ds_joints - infer_joints)
            if dist < best_dist:
                best_dist = dist
                ds_idx = i
        print(f"  Closest frame: {ds_idx} (joint distance: {best_dist:.4f} rad)")

    ds_sample = ds[ds_idx]

    def to_rgb(img_tensor):
        img = np.asarray(img_tensor)
        if img.dtype != np.uint8:
            img = (img * 255 if img.max() <= 1.0 else img).astype(np.uint8)
        if img.shape[0] == 3:
            img = einops.rearrange(img, "c h w -> h w c")
        return img

    ds_base = to_rgb(ds_sample["image"])
    ds_wrist = to_rgb(ds_sample["wrist_image"])
    ds_joints = np.array(ds_sample["joints"], dtype=np.float32).reshape(-1)
    ds_gripper = float(ds_sample["gripper"])

    print(f"\nDataset frame {ds_idx}:")
    print(f"  base: {ds_base.shape} mean={ds_base.mean():.1f} std={ds_base.std():.1f}")
    print(f"  wrist: {ds_wrist.shape} mean={ds_wrist.mean():.1f} std={ds_wrist.std():.1f}")
    print(f"  state: joints={ds_joints.round(4)}, gripper={ds_gripper:.4f}")

    # Resize to same size for comparison
    target_h, target_w = 256, 256
    infer_base_r = image_tools.resize_with_pad(infer_base, target_h, target_w)
    infer_wrist_r = image_tools.resize_with_pad(infer_wrist, target_h, target_w)
    ds_base_r = image_tools.resize_with_pad(ds_base, target_h, target_w)
    ds_wrist_r = image_tools.resize_with_pad(ds_wrist, target_h, target_w)

    # Create comparison grid (2x2): top=inference, bottom=training
    # Convert to BGR for cv2
    top = np.hstack([
        cv2.cvtColor(infer_base_r, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(infer_wrist_r, cv2.COLOR_RGB2BGR),
    ])
    bot = np.hstack([
        cv2.cvtColor(ds_base_r, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(ds_wrist_r, cv2.COLOR_RGB2BGR),
    ])

    # Add labels
    cv2.putText(top, f"INFERENCE step={args.step} g={infer_state[6]:.3f}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(top, f"mean={infer_base.mean():.0f}", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(top, f"mean={infer_wrist.mean():.0f}", (target_w + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    cv2.putText(bot, f"TRAINING frame={ds_idx} g={ds_gripper:.3f}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(bot, f"mean={ds_base.mean():.0f}", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(bot, f"mean={ds_wrist.mean():.0f}", (target_w + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    grid = np.vstack([top, bot])

    # Pixel statistics comparison
    print(f"\nPixel statistics comparison:")
    print(f"  {'':20s} {'Inference':>12s} {'Training':>12s} {'Diff':>8s}")
    print(f"  {'Base mean':20s} {infer_base.mean():12.1f} {ds_base.mean():12.1f} {abs(infer_base.mean()-ds_base.mean()):8.1f}")
    print(f"  {'Base std':20s} {infer_base.std():12.1f} {ds_base.std():12.1f} {abs(infer_base.std()-ds_base.std()):8.1f}")
    print(f"  {'Wrist mean':20s} {infer_wrist.mean():12.1f} {ds_wrist.mean():12.1f} {abs(infer_wrist.mean()-ds_wrist.mean()):8.1f}")
    print(f"  {'Wrist std':20s} {infer_wrist.std():12.1f} {ds_wrist.std():12.1f} {abs(infer_wrist.std()-ds_wrist.std()):8.1f}")
    print(f"  {'Base shape':20s} {str(infer_base.shape):>12s} {str(ds_base.shape):>12s}")

    mean_diff = abs(infer_base.mean() - ds_base.mean())
    if mean_diff > 30:
        print(f"\n  WARNING: Large brightness difference ({mean_diff:.0f}) — check camera exposure settings!")
    elif mean_diff > 15:
        print(f"\n  CAUTION: Moderate brightness difference ({mean_diff:.0f})")
    else:
        print(f"\n  Brightness looks similar (diff={mean_diff:.0f})")

    cv2.imwrite(args.output, grid)
    print(f"\nComparison saved to {args.output}")


if __name__ == "__main__":
    main()
