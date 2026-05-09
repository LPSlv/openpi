#!/usr/bin/env python3
"""Comprehensive gripper diagnostics — run locally on CPU.

Tests every component of the gripper pipeline: data transforms, norm stats,
inference output, ODE step sensitivity, action chunk timing, and model comparison.

Usage:
    JAX_PLATFORMS=cpu uv run ur5/scripts/test_gripper_diagnostics.py

Checkpoints tested:
    - og_smooth-3 (WORKING): checkpoints/pi05_ur5/ur5_blueblock_box_og_smooth-3/420
    - smooth-8 (FAILING):    checkpoints/pi0_ur5/ur5_blueblock_box_v2_40_smooth-8/1999
"""

import copy
import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

RESET_POSE = np.array([-1.5708, -0.6981, -2.4435, -0.8727, 1.5708, 0.0], dtype=np.float32)

SMOOTH8_CKPT = "checkpoints/pi0_ur5/ur5_blueblock_box_v2_40_smooth-8/1999"
SMOOTH8_CONFIG = "pi0_ur5"

OG_SMOOTH3_CKPT = "checkpoints/pi05_ur5/ur5_blueblock_box_og_smooth-3/420"
OG_SMOOTH3_CONFIG = "pi05_ur5"

PROMPT = "Pick up the blue block and place it in the cardboard box"

results = {}


def make_obs(gripper_state: float = 0.0, prompt: str = PROMPT) -> dict:
    """Create a deterministic test observation with reset pose."""
    state = np.concatenate([RESET_POSE, [gripper_state]]).astype(np.float32)
    return {
        "observation/image": np.random.RandomState(42).randint(0, 256, (224, 224, 3)).astype(np.uint8),
        "observation/wrist_image": np.random.RandomState(43).randint(0, 256, (224, 224, 3)).astype(np.uint8),
        "observation/state": state,
        "prompt": prompt,
    }


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Test A: Data pipeline round-trip
# ──────────────────────────────────────────────────────────────────────

def test_a_data_pipeline():
    section("TEST A: Data Pipeline Round-Trip")
    from openpi.training import config as _config
    import openpi.transforms as _transforms
    from openpi.policies import ur5_policy

    config = _config.get_config(SMOOTH8_CONFIG)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Create a sample with known gripper values
    sample = {
        "joints": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        "gripper": np.array([0.8], dtype=np.float32),
        "base_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "actions": np.tile(
            np.array([[0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.9]], dtype=np.float32),
            (50, 1),
        ),
        "prompt": "test",
    }
    # Set a transition at step 25
    sample["actions"][25:, 6] = 0.1  # gripper opens midway

    original_gripper = sample["actions"][:, 6].copy()

    # Apply input transforms
    s1 = copy.deepcopy(sample)
    for t in data_config.data_transforms.inputs:
        s1 = t(s1)
    print(f"After UR5Inputs + DeltaActions:")
    print(f"  state[6] (gripper): {s1['state'][6]:.4f}")
    print(f"  actions[:, 6] before transition: {s1['actions'][0, 6]:.4f}")
    print(f"  actions[:, 6] after transition:  {s1['actions'][30, 6]:.4f}")
    gripper_unchanged = np.allclose(s1["actions"][:, 6], original_gripper, atol=1e-6)
    print(f"  DeltaActions left gripper unchanged: {gripper_unchanged}")

    # Normalize
    norm_stats = data_config.norm_stats
    if norm_stats:
        normalize = _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        s2 = normalize(copy.deepcopy(s1))
        print(f"\nAfter Normalize:")
        print(f"  actions[:, 6] normalized (gripper=0.9): {s2['actions'][0, 6]:.4f}")
        print(f"  actions[:, 6] normalized (gripper=0.1): {s2['actions'][30, 6]:.4f}")
        print(f"  actions[:, 0] normalized (joint delta):  {s2['actions'][0, 0]:.4f}")

        # Unnormalize round-trip
        unnormalize = _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        s3 = unnormalize(copy.deepcopy(s2))
        roundtrip_err = np.max(np.abs(s3["actions"][:, 6] - s1["actions"][:, 6]))
        print(f"\nRound-trip (Normalize → Unnormalize):")
        print(f"  Max gripper error: {roundtrip_err:.8f}")
        rt_pass = roundtrip_err < 0.001
        print(f"  {'PASS' if rt_pass else 'FAIL'}")
        results["A_roundtrip"] = rt_pass
    else:
        print("  No norm stats available, skipping normalize test")
        results["A_roundtrip"] = None

    results["A_delta_unchanged"] = gripper_unchanged


# ──────────────────────────────────────────────────────────────────────
# Test B: Norm stats verification
# ──────────────────────────────────────────────────────────────────────

def test_b_norm_stats():
    section("TEST B: Norm Stats Verification")

    for name, ckpt_path in [("smooth-8", SMOOTH8_CKPT), ("og_smooth-3", OG_SMOOTH3_CKPT)]:
        stats_path = os.path.join(ckpt_path, "assets/ur5e/norm_stats.json")
        if not os.path.exists(stats_path):
            print(f"  {name}: norm_stats.json not found at {stats_path}")
            continue

        with open(stats_path) as f:
            data = json.load(f)

        ns = data["norm_stats"]
        print(f"\n--- {name} ({stats_path}) ---")

        for key in ["actions", "state"]:
            if key not in ns:
                continue
            mean = np.array(ns[key]["mean"])
            std = np.array(ns[key]["std"])
            real_dims = min(7, len(mean))
            print(f"\n  {key} (first {real_dims} dims):")
            dim_names = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]
            for i in range(real_dims):
                print(f"    [{i}] {dim_names[i]:>8s}: mean={mean[i]:+.6f}  std={std[i]:.6f}")

            # Check gripper stats specifically
            if real_dims >= 7:
                g_mean, g_std = mean[6], std[6]
                print(f"\n  Gripper {key} normalization:")
                print(f"    raw=0.0 (open)  → normalized: {(0.0 - g_mean) / (g_std + 1e-6):+.4f}")
                print(f"    raw=0.5 (half)  → normalized: {(0.5 - g_mean) / (g_std + 1e-6):+.4f}")
                print(f"    raw=1.0 (close) → normalized: {(1.0 - g_mean) / (g_std + 1e-6):+.4f}")

    # Compare the two checkpoints' gripper stats
    try:
        with open(os.path.join(SMOOTH8_CKPT, "assets/ur5e/norm_stats.json")) as f:
            s8 = json.load(f)["norm_stats"]["actions"]
        with open(os.path.join(OG_SMOOTH3_CKPT, "assets/ur5e/norm_stats.json")) as f:
            og = json.load(f)["norm_stats"]["actions"]
        s8_g = (s8["mean"][6], s8["std"][6])
        og_g = (og["mean"][6], og["std"][6])
        match = abs(s8_g[0] - og_g[0]) < 0.05 and abs(s8_g[1] - og_g[1]) < 0.05
        print(f"\n  Gripper action stats match between models: {match}")
        print(f"    smooth-8:    mean={s8_g[0]:.4f} std={s8_g[1]:.4f}")
        print(f"    og_smooth-3: mean={og_g[0]:.4f} std={og_g[1]:.4f}")
        results["B_stats_match"] = match
    except Exception as e:
        print(f"  Could not compare: {e}")
        results["B_stats_match"] = None


# ──────────────────────────────────────────────────────────────────────
# Test C: Full inference pipeline
# ──────────────────────────────────────────────────────────────────────

def test_c_inference(config_name, ckpt_path, label):
    section(f"TEST C: Inference Pipeline — {label}")

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        results[f"C_{label}_gripper_mean"] = None
        return

    config = _config.get_config(config_name)
    print(f"  Loading {label} from {ckpt_path}...")
    policy = _policy_config.create_trained_policy(config, ckpt_path, default_prompt=PROMPT)
    print(f"  Model loaded. action_horizon={config.model.action_horizon}")

    # Test with gripper=0 (open)
    obs_open = make_obs(gripper_state=0.0)
    gripper_vals_open = []
    print(f"\n  Running 5 inferences with gripper_state=0.0 (open):")
    for i in range(5):
        out = policy.infer(copy.deepcopy(obs_open))
        actions = out["actions"]  # (horizon, 7)
        g = actions[:, 6]
        gripper_vals_open.append(g)
        print(f"    Run {i}: gripper min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f}  first5={g[:5].round(4)}")

    all_g_open = np.concatenate(gripper_vals_open)
    print(f"  Overall open: mean={all_g_open.mean():.4f} std={all_g_open.std():.4f} range=[{all_g_open.min():.4f}, {all_g_open.max():.4f}]")

    # Test with gripper=1 (closed)
    obs_closed = make_obs(gripper_state=1.0)
    gripper_vals_closed = []
    print(f"\n  Running 5 inferences with gripper_state=1.0 (closed):")
    for i in range(5):
        out = policy.infer(copy.deepcopy(obs_closed))
        g = out["actions"][:, 6]
        gripper_vals_closed.append(g)
        print(f"    Run {i}: gripper min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f}  first5={g[:5].round(4)}")

    all_g_closed = np.concatenate(gripper_vals_closed)
    print(f"  Overall closed: mean={all_g_closed.mean():.4f} std={all_g_closed.std():.4f} range=[{all_g_closed.min():.4f}, {all_g_closed.max():.4f}]")

    # Key metric: does the model respond to gripper state?
    diff = abs(all_g_closed.mean() - all_g_open.mean())
    print(f"\n  Gripper response to state change: |closed_mean - open_mean| = {diff:.4f}")
    responds = diff > 0.05
    print(f"  Model responds to gripper state: {responds}")

    results[f"C_{label}_open_mean"] = float(all_g_open.mean())
    results[f"C_{label}_closed_mean"] = float(all_g_closed.mean())
    results[f"C_{label}_responds"] = responds

    return policy


# ──────────────────────────────────────────────────────────────────────
# Test D: ODE step sweep
# ──────────────────────────────────────────────────────────────────────

def test_d_ode_sweep(config_name, ckpt_path, label):
    section(f"TEST D: ODE Step Sweep — {label}")

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return

    obs = make_obs(gripper_state=0.0)

    for num_steps in [10, 20, 50]:
        config = _config.get_config(config_name)
        policy = _policy_config.create_trained_policy(
            config, ckpt_path, default_prompt=PROMPT,
            sample_kwargs={"num_steps": num_steps},
        )
        out = policy.infer(copy.deepcopy(obs))
        g = out["actions"][:, 6]
        print(f"  num_steps={num_steps:3d}: gripper mean={g.mean():.4f} range=[{g.min():.4f}, {g.max():.4f}]  first5={g[:5].round(4)}")
        results[f"D_{label}_steps{num_steps}"] = float(g.mean())


# ──────────────────────────────────────────────────────────────────────
# Test E: Action chunk temporal analysis
# ──────────────────────────────────────────────────────────────────────

def test_e_temporal(config_name, ckpt_path, label):
    section(f"TEST E: Action Chunk Temporal Analysis — {label}")

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return

    config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(config, ckpt_path, default_prompt=PROMPT)

    obs = make_obs(gripper_state=0.0)
    out = policy.infer(copy.deepcopy(obs))
    actions = out["actions"]  # (horizon, 7)
    horizon = actions.shape[0]

    print(f"  Action chunk shape: {actions.shape} (horizon={horizon})")
    print(f"\n  Gripper values across full {horizon}-step chunk:")
    print(f"  {'Step':>4s}  {'Gripper':>8s}  {'Joint0':>8s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}")

    for i in range(0, horizon, max(1, horizon // 20)):
        print(f"  {i:4d}  {actions[i, 6]:+8.4f}  {actions[i, 0]:+8.4f}")

    # Find max gripper value and its timestep
    max_g = actions[:, 6].max()
    max_g_step = actions[:, 6].argmax()
    min_g = actions[:, 6].min()
    print(f"\n  Max gripper: {max_g:.4f} at step {max_g_step}")
    print(f"  Min gripper: {min_g:.4f}")
    print(f"  Gripper range across chunk: {max_g - min_g:.4f}")

    # Check if gripper signal appears after HORIZON_STEPS=10
    first10_max = actions[:10, 6].max()
    rest_max = actions[10:, 6].max() if horizon > 10 else 0
    print(f"\n  First 10 steps gripper max: {first10_max:.4f}")
    if horizon > 10:
        print(f"  Steps 10+ gripper max:      {rest_max:.4f}")
        if rest_max > first10_max + 0.05:
            print(f"  WARNING: Gripper signal appears AFTER step 10 — HORIZON_STEPS=10 may miss it!")
            results[f"E_{label}_late_signal"] = True
        else:
            results[f"E_{label}_late_signal"] = False


# ──────────────────────────────────────────────────────────────────────
# Test F: Model comparison (og_smooth-3 vs smooth-8)
# ──────────────────────────────────────────────────────────────────────

def test_f_comparison():
    section("TEST F: Model Comparison — og_smooth-3 vs smooth-8")

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    for name, cfg, ckpt in [("og_smooth-3", OG_SMOOTH3_CONFIG, OG_SMOOTH3_CKPT),
                              ("smooth-8", SMOOTH8_CONFIG, SMOOTH8_CKPT)]:
        if not os.path.exists(ckpt):
            print(f"  {name}: checkpoint not found")
            continue

        config = _config.get_config(cfg)
        policy = _policy_config.create_trained_policy(config, ckpt, default_prompt=PROMPT)
        horizon = config.model.action_horizon

        obs = make_obs(gripper_state=0.0)
        out = policy.infer(copy.deepcopy(obs))
        g = out["actions"][:, 6]
        j = out["actions"][:, :6]

        print(f"\n  {name} (horizon={horizon}):")
        print(f"    Gripper: mean={g.mean():.4f} range=[{g.min():.4f}, {g.max():.4f}]")
        print(f"    Joints:  range=[{j.min():.4f}, {j.max():.4f}]")
        print(f"    First 5 gripper: {g[:5].round(4)}")
        print(f"    Last 5 gripper:  {g[-5:].round(4)}")

        results[f"F_{name}_gripper_mean"] = float(g.mean())
        results[f"F_{name}_gripper_max"] = float(g.max())


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  GRIPPER DIAGNOSTICS — Comprehensive Pipeline Test")
    print("=" * 70)
    print(f"  JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', 'not set')}")

    # Test A: Data pipeline (no model needed)
    try:
        test_a_data_pipeline()
    except Exception as e:
        print(f"  TEST A FAILED: {e}")
        import traceback; traceback.print_exc()

    # Test B: Norm stats (no model needed)
    try:
        test_b_norm_stats()
    except Exception as e:
        print(f"  TEST B FAILED: {e}")
        import traceback; traceback.print_exc()

    # Test C: Inference for both models
    for label, cfg, ckpt in [("og_smooth3", OG_SMOOTH3_CONFIG, OG_SMOOTH3_CKPT),
                               ("smooth8", SMOOTH8_CONFIG, SMOOTH8_CKPT)]:
        try:
            test_c_inference(cfg, ckpt, label)
        except Exception as e:
            print(f"  TEST C ({label}) FAILED: {e}")
            import traceback; traceback.print_exc()

    # Test D: ODE sweep (only on smooth-8 to save time)
    try:
        test_d_ode_sweep(SMOOTH8_CONFIG, SMOOTH8_CKPT, "smooth8")
    except Exception as e:
        print(f"  TEST D FAILED: {e}")
        import traceback; traceback.print_exc()

    # Test E: Temporal analysis for both
    for label, cfg, ckpt in [("og_smooth3", OG_SMOOTH3_CONFIG, OG_SMOOTH3_CKPT),
                               ("smooth8", SMOOTH8_CONFIG, SMOOTH8_CKPT)]:
        try:
            test_e_temporal(cfg, ckpt, label)
        except Exception as e:
            print(f"  TEST E ({label}) FAILED: {e}")
            import traceback; traceback.print_exc()

    # Test F: Direct comparison
    try:
        test_f_comparison()
    except Exception as e:
        print(f"  TEST F FAILED: {e}")
        import traceback; traceback.print_exc()

    # Summary
    section("SUMMARY")
    for k, v in sorted(results.items()):
        status = "PASS" if v is True else "FAIL" if v is False else str(v)
        print(f"  {k:40s}: {status}")


if __name__ == "__main__":
    main()
