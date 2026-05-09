#!/usr/bin/env python3
"""Merge fine-tuned checkpoint with pretrained base weights.

Interpolates between pretrained and fine-tuned weights:
  merged = alpha * pretrained + (1 - alpha) * finetuned

alpha=0.0 → fully fine-tuned (good arm, overfit gripper)
alpha=1.0 → fully pretrained (no task knowledge)
alpha=0.3 → mostly fine-tuned with some pretrained regularization

Usage:
    uv run ur5/scripts/merge_checkpoints.py \
        --finetuned checkpoints/pi05_ur5/ur5_blueblock_box_v2_40_smooth-15/499 \
        --pretrained gs://openpi-assets/checkpoints/pi05_base \
        --alpha 0.3 \
        --output checkpoints/pi05_ur5/ur5_blueblock_box_v2_40_smooth-15/499_merged_0.3

    # Try multiple alphas:
    for a in 0.1 0.2 0.3 0.5; do
        uv run ur5/scripts/merge_checkpoints.py \
            --finetuned checkpoints/pi05_ur5/ur5_blueblock_box_v2_40_smooth-15/499 \
            --pretrained gs://openpi-assets/checkpoints/pi05_base \
            --alpha $a \
            --output checkpoints/pi05_ur5/ur5_blueblock_box_v2_40_smooth-15/499_merged_$a
    done
"""

import argparse
import pathlib
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.download as _download


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finetuned", required=True, help="Fine-tuned checkpoint dir")
    parser.add_argument("--pretrained", required=True, help="Pretrained checkpoint dir (or gs:// path)")
    parser.add_argument("--alpha", type=float, required=True, help="Interpolation weight: 0=finetuned, 1=pretrained")
    parser.add_argument("--output", required=True, help="Output checkpoint dir")
    parser.add_argument("--vision-only", action="store_true",
                        help="Only merge vision (img) weights, keep action weights from finetuned")
    args = parser.parse_args()

    assert 0.0 <= args.alpha <= 1.0, f"alpha must be in [0, 1], got {args.alpha}"

    # Download pretrained if it's a GCS path
    pretrained_dir = pathlib.Path(_download.maybe_download(args.pretrained))
    finetuned_dir = pathlib.Path(args.finetuned)
    output_dir = pathlib.Path(args.output).resolve()

    print(f"Fine-tuned: {finetuned_dir}")
    print(f"Pretrained: {pretrained_dir}")
    print(f"Alpha: {args.alpha} (0=finetuned, 1=pretrained)")
    print(f"Vision-only: {args.vision_only}")
    print(f"Output: {output_dir}")

    # Load both checkpoints (use bfloat16 to fit in RAM — two checkpoints ~11GB each)
    print("\nLoading fine-tuned params...", flush=True)
    finetuned_params = _model.restore_params(finetuned_dir / "params", dtype=jnp.bfloat16)
    print(f"  Loaded ({len(jax.tree.leaves(finetuned_params))} arrays)", flush=True)
    print("Loading pretrained params...", flush=True)
    pretrained_params = _model.restore_params(pretrained_dir / "params", dtype=jnp.bfloat16)
    print(f"  Loaded ({len(jax.tree.leaves(pretrained_params))} arrays)", flush=True)

    # Merge weights
    print(f"\nMerging with alpha={args.alpha}...", flush=True)
    merged_count = 0
    kept_count = 0

    def merge_fn(path, ft, pt):
        nonlocal merged_count, kept_count
        path_str = str(path)

        if args.vision_only and "img" not in path_str:
            kept_count += 1
            return ft

        merged_count += 1
        return args.alpha * pt + (1.0 - args.alpha) * ft

    # Flatten, merge, unflatten
    from flax import traverse_util
    flat_ft = traverse_util.flatten_dict(finetuned_params)
    flat_pt = traverse_util.flatten_dict(pretrained_params)
    total_keys = len(flat_ft)
    print(f"  {total_keys} parameters to process", flush=True)

    flat_merged = {}
    for i, key in enumerate(flat_ft):
        path_str = "/".join(str(k) for k in key)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total_keys}] {path_str} {flat_ft[key].shape}", flush=True)
        if key in flat_pt and flat_ft[key].shape == flat_pt[key].shape:
            flat_merged[key] = merge_fn(path_str, flat_ft[key], flat_pt[key])
        else:
            flat_merged[key] = flat_ft[key]
            kept_count += 1
            if key not in flat_pt:
                print(f"  WARNING: {path_str} not in pretrained, keeping finetuned")
            elif flat_ft[key].shape != flat_pt[key].shape:
                print(f"  WARNING: {path_str} shape mismatch ft={flat_ft[key].shape} pt={flat_pt[key].shape}, keeping finetuned")

    merged_params = traverse_util.unflatten_dict(flat_merged)
    print(f"Merged {merged_count} params, kept {kept_count} params from finetuned")

    # Save merged checkpoint
    print(f"\nSaving to {output_dir}...", flush=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Copy only assets (norm stats) from finetuned checkpoint
    src_assets = finetuned_dir / "assets"
    if src_assets.exists():
        shutil.copytree(src_assets, output_dir / "assets")

    # Save merged params
    print(f"  Saving merged params...", flush=True)
    import orbax.checkpoint as ocp
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(output_dir / "params", {"params": merged_params})
    print(f"  Save complete.", flush=True)

    print(f"\nDone! Merged checkpoint saved to {output_dir}", flush=True)
    print(f"\nTo evaluate:")
    print(f"  uv run ur5/scripts/eval_generalization.py \\")
    print(f"    --config pi05_ur5 \\")
    print(f"    --checkpoint-dir {output_dir.parent} \\")
    print(f"    --eval-assets ur5/eval_assets")


if __name__ == "__main__":
    main()
