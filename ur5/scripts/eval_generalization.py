#!/usr/bin/env python3
"""Evaluate gripper generalization across checkpoints.

Runs inference on dataset pre-grip images, brightness-perturbed images, and
recorded robot images. Reports gripper activation for each checkpoint step.

Usage (after training with save_interval=30):
    uv run ur5/scripts/eval_generalization.py \
        --config pi0_ur5 \
        --checkpoint-dir checkpoints/pi0_ur5/<exp_name> \
        --eval-assets ur5/eval_assets

Output: table of gripper scores per checkpoint step, identifying the best
generalizing checkpoint (highest gripper on robot images).
"""

import argparse
import copy
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


PROMPT = "Pick up the blue block and place it in the cardboard box"


def load_eval_obs(npz_path: str) -> dict:
    """Load an observation from .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        "observation/image": d["image"],
        "observation/wrist_image": d["wrist_image"],
        "observation/state": d["state"].astype(np.float32),
        "prompt": str(d["prompt"]) if "prompt" in d else PROMPT,
    }


def perturb_brightness(obs: dict, factor: float) -> dict:
    """Apply brightness perturbation to images."""
    out = copy.deepcopy(obs)
    for key in ["observation/image", "observation/wrist_image"]:
        img = out[key].astype(np.float32) * factor
        out[key] = np.clip(img, 0, 255).astype(np.uint8)
    return out


def eval_checkpoint(policy, eval_observations: dict, n_runs: int = 3) -> dict:
    """Run inference on all eval observations and return gripper scores."""
    results = {}
    for name, obs in eval_observations.items():
        gripper_means = []
        for _ in range(n_runs):
            out = policy.infer(copy.deepcopy(obs))
            g = out["actions"][:, 6]
            gripper_means.append(float(g.mean()))
        results[name] = {
            "mean": np.mean(gripper_means),
            "std": np.std(gripper_means),
            "max": np.max(gripper_means),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="pi0_ur5", help="Training config name")
    parser.add_argument("--checkpoint-dir", required=True, help="Base checkpoint dir (contains step subdirs)")
    parser.add_argument("--eval-assets", default="ur5/eval_assets", help="Directory with eval .npz files")
    parser.add_argument("--n-runs", type=int, default=2, help="Inference runs per observation (for variance)")
    parser.add_argument("--filter", default="", help="Only include checkpoints whose name contains this string")
    args = parser.parse_args()

    # Find all checkpoint directories (numeric steps + merged checkpoints)
    ckpt_base = pathlib.Path(args.checkpoint_dir)
    step_dirs = []
    for d in ckpt_base.iterdir():
        if not d.is_dir():
            continue
        # Include numeric steps and merged checkpoints (e.g., 499_merged_0.3)
        if d.name.isdigit() or (d / "params").exists():
            if args.filter and args.filter not in d.name:
                continue
            step_dirs.append(d)
    # Sort: numeric first (by number), then non-numeric alphabetically
    step_dirs.sort(key=lambda d: (0, int(d.name)) if d.name.isdigit() else (1, d.name))
    if not step_dirs:
        print(f"No checkpoint directories found in {ckpt_base}")
        sys.exit(1)
    print(f"Found {len(step_dirs)} checkpoints: {[d.name for d in step_dirs]}")

    # Load eval assets
    eval_dir = pathlib.Path(args.eval_assets)
    dataset_files = sorted(eval_dir.glob("dataset_pregrip_*.npz"))
    robot_files = sorted(eval_dir.glob("robot_step_*.npz"))
    print(f"Eval assets: {len(dataset_files)} dataset, {len(robot_files)} robot")

    if not dataset_files and not robot_files:
        print(f"No eval assets found in {eval_dir}")
        sys.exit(1)

    # Build eval observation dict
    eval_observations = {}

    # Dataset pre-grip images (should activate gripper)
    for f in dataset_files[:2]:  # Use 2 to save time
        obs = load_eval_obs(str(f))
        name = f.stem
        eval_observations[f"ds_{name}"] = obs
        # Brightness perturbations
        eval_observations[f"ds_{name}_dark"] = perturb_brightness(obs, 0.6)
        eval_observations[f"ds_{name}_bright"] = perturb_brightness(obs, 1.5)

    # Robot images (the real test)
    for f in robot_files[:2]:  # Use 2 to save time
        eval_observations[f"robot_{f.stem}"] = load_eval_obs(str(f))

    obs_names = list(eval_observations.keys())
    print(f"Eval observations: {obs_names}")

    # Import here to avoid slow import if args are wrong
    from openpi.policies import policy_config as pc
    from openpi.training import config as cfg

    train_config = cfg.get_config(args.config)

    # Evaluate each checkpoint
    all_results = {}
    for step_dir in step_dirs:
        step = step_dir.name
        print(f"\n--- Checkpoint {step} ---")

        try:
            policy = pc.create_trained_policy(train_config, str(step_dir), default_prompt=PROMPT)
        except Exception as e:
            import traceback
            print(f"  Failed to load: {e}")
            # Debug: show what's in the checkpoint
            params_path = step_dir / "params"
            if params_path.exists():
                print(f"  params/ contents: {[f.name for f in params_path.iterdir()][:10]}")
                try:
                    import orbax.checkpoint as ocp
                    ckptr = ocp.PyTreeCheckpointer()
                    meta = ckptr.metadata(params_path.resolve())
                    print(f"  orbax metadata keys: {list(meta.keys())}")
                except Exception as me:
                    print(f"  orbax metadata read failed: {me}")
            else:
                print(f"  params/ directory does not exist!")
            traceback.print_exc()
            continue

        results = eval_checkpoint(policy, eval_observations, n_runs=args.n_runs)
        all_results[step] = results

        for name, r in results.items():
            tag = "GRIP" if r["mean"] > 0.3 else "open"
            print(f"  {name:45s}: gripper={r['mean']:+.4f} +/- {r['std']:.4f}  [{tag}]")

        # Cleanup to free memory
        del policy

    # Summary table
    print("\n" + "=" * 100)
    print("GENERALIZATION SUMMARY")
    print("=" * 100)

    # Categorize observations
    ds_orig = [n for n in obs_names if n.startswith("ds_") and "dark" not in n and "bright" not in n]
    ds_perturbed = [n for n in obs_names if "dark" in n or "bright" in n]
    robot = [n for n in obs_names if n.startswith("robot_")]

    header = f"{'Checkpoint':>20s}  {'Dataset(orig)':>13s}  {'Dataset(perturb)':>16s}  {'Robot':>8s}  {'Overfit':>8s}  {'Verdict':>10s}"
    print(header)
    print("-" * len(header))

    best_step = None
    best_robot_score = -1

    for step, results in sorted(all_results.items(), key=lambda x: (0, int(x[0])) if x[0].isdigit() else (1, x[0])):
        ds_score = np.mean([results[n]["mean"] for n in ds_orig]) if ds_orig else 0
        perturb_score = np.mean([results[n]["mean"] for n in ds_perturbed]) if ds_perturbed else 0
        robot_score = np.mean([results[n]["mean"] for n in robot]) if robot else 0
        overfit = ds_score - robot_score

        if ds_score > 0.3 and robot_score > 0.1:
            verdict = "GOOD"
        elif ds_score > 0.3 and robot_score < 0.1:
            verdict = "OVERFIT"
        elif ds_score < 0.1:
            verdict = "UNDERFIT"
        else:
            verdict = "partial"

        print(f"{step:>20s}  {ds_score:+13.4f}  {perturb_score:+16.4f}  {robot_score:+8.4f}  {overfit:+8.4f}  {verdict:>10s}")

        if robot_score > best_robot_score:
            best_robot_score = robot_score
            best_step = step

    print(f"\nBest checkpoint: {best_step} (robot gripper={best_robot_score:.4f})")
    if best_robot_score < 0.1:
        print("WARNING: No checkpoint shows gripper activation on robot images.")
        print("Consider: fewer episodes, stronger augmentation, or different training approach.")


if __name__ == "__main__":
    main()
