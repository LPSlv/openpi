#!/usr/bin/env python3
"""Verify the UR5 training data pipeline by tracing gripper values through each transform.

Loads a sample with a gripper transition from the dataset and prints the gripper
dimension (dim 6) after each transform stage. This helps diagnose whether any
transform is corrupting, clipping, or collapsing the gripper signal.

Usage:
    uv run ur5/scripts/verify_data_pipeline.py [--config-name pi0_ur5]
"""

import copy
import dataclasses
import sys

import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


def main():
    config_name = "pi0_ur5"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--config-name" and i + 1 < len(sys.argv) - 1:
            config_name = sys.argv[i + 2]

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    print(f"Config: {config_name}")
    print(f"Dataset: {data_config.repo_id}")
    print(f"action_horizon: {config.model.action_horizon}")
    print(f"action_dim: {config.model.action_dim}")
    print(f"use_quantile_norm: {data_config.use_quantile_norm}")
    print()

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    print(f"Dataset size: {len(dataset)} frames")

    # find a sample whose action chunk contains a gripper transition
    transition_idx = None
    for i in range(len(dataset)):
        sample = dataset[i]
        actions = np.asarray(sample.get("actions", sample.get("action", [])))
        if actions.ndim >= 2 and actions.shape[-1] >= 7:
            gripper_vals = actions[..., 6]
            gripper_range = float(np.max(gripper_vals) - np.min(gripper_vals))
            if gripper_range > 0.3:
                transition_idx = i
                break

    if transition_idx is None:
        print("ERROR: No gripper transition found in dataset!")
        return

    raw_sample = dataset[transition_idx]
    print(f"Found transition at index {transition_idx}")
    print(f"Raw sample keys: {list(raw_sample.keys())}")
    print()

    raw_actions = np.asarray(raw_sample.get("actions", raw_sample.get("action")))
    raw_gripper_col = raw_actions[..., 6] if raw_actions.ndim >= 2 else raw_actions[6:]
    print("=" * 70)
    print("STAGE 0: Raw LeRobot sample")
    print(f"  actions shape: {raw_actions.shape}")
    if "gripper" in raw_sample:
        print(f"  gripper (state): {float(np.asarray(raw_sample['gripper'])):.4f}")
    print(f"  actions[:, 6] (gripper col): min={raw_gripper_col.min():.4f}, max={raw_gripper_col.max():.4f}")
    print(f"  First 10 gripper values: {raw_gripper_col[:10].round(4)}")
    print()

    # stage 1: repack
    repack_transforms = data_config.repack_transforms.inputs
    sample = copy.deepcopy(raw_sample)
    for t in repack_transforms:
        sample = t(sample)
    print("STAGE 1: After RepackTransform")
    if "state" in sample:
        state = np.asarray(sample["state"])
        print(f"  state shape: {state.shape}, state[6] (gripper): {state[6]:.4f}")
    if "actions" in sample:
        actions = np.asarray(sample["actions"])
        print(f"  actions shape: {actions.shape}")
        print(f"  actions[:, 6]: min={actions[:, 6].min():.4f}, max={actions[:, 6].max():.4f}")
    print()

    # stage 2: UR5Inputs + DeltaActions
    data_input_transforms = data_config.data_transforms.inputs
    sample_pre_delta = copy.deepcopy(sample)
    for t in data_input_transforms:
        sample_pre_delta = t(sample_pre_delta)

    print("STAGE 2: After data_transforms (UR5Inputs + DeltaActions)")
    if "state" in sample_pre_delta:
        state = np.asarray(sample_pre_delta["state"])
        print(f"  state shape: {state.shape}")
        if state.shape[0] >= 7:
            print(f"  state[6] (gripper): {state[6]:.4f}")
    if "actions" in sample_pre_delta:
        actions = np.asarray(sample_pre_delta["actions"])
        print(f"  actions shape: {actions.shape}")
        print(f"  actions[:, 6] (gripper, should be ABSOLUTE): min={actions[:, 6].min():.4f}, max={actions[:, 6].max():.4f}")
        print(f"  actions[:, 0] (joint 0, should be DELTA):    min={actions[:, 0].min():.4f}, max={actions[:, 0].max():.4f}")
        print(f"  First 10 gripper values: {actions[:10, 6].round(4)}")
    print()

    # stage 3: normalize
    norm_stats = data_config.norm_stats
    if norm_stats:
        print("STAGE 3: After Normalize")
        print(f"  Norm stats keys: {list(norm_stats.keys())}")
        if "actions" in norm_stats:
            ns = norm_stats["actions"]
            mean_dim = len(ns.mean) if hasattr(ns.mean, '__len__') else ns.mean.shape[-1]
            print(f"  actions norm_stats dims: {mean_dim}")
            print(f"  actions mean[6] (gripper): {ns.mean[6]:.4f}")
            print(f"  actions std[6]  (gripper): {ns.std[6]:.4f}")
        normalize_fn = _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        sample_normed = normalize_fn(copy.deepcopy(sample_pre_delta))
        if "actions" in sample_normed:
            actions = np.asarray(sample_normed["actions"])
            print(f"  Normalized actions shape: {actions.shape}")
            print(f"  actions[:, 6] (gripper, normalized): min={actions[:, 6].min():.4f}, max={actions[:, 6].max():.4f}")
            print(f"  actions[:, 0] (joint 0, normalized):  min={actions[:, 0].min():.4f}, max={actions[:, 0].max():.4f}")
            print(f"  First 10 normalized gripper: {actions[:10, 6].round(4)}")
    else:
        print("STAGE 3: SKIPPED (no norm_stats)")
        sample_normed = copy.deepcopy(sample_pre_delta)
    print()

    # stage 4: model transforms (PadStatesAndActions)
    model_transforms = data_config.model_transforms.inputs
    sample_padded = copy.deepcopy(sample_normed)
    for t in model_transforms:
        sample_padded = t(sample_padded)
    print("STAGE 4: After model_transforms (PadStatesAndActions)")
    if "state" in sample_padded:
        state = np.asarray(sample_padded["state"])
        print(f"  state shape: {state.shape}")
        if state.shape[-1] >= 7:
            print(f"  state[6] (gripper): {state[6]:.4f}")
            print(f"  state[7:] (padded): {state[7:12].round(6)}")
    if "actions" in sample_padded:
        actions = np.asarray(sample_padded["actions"])
        print(f"  actions shape: {actions.shape}")
        print(f"  actions[:, 6] (gripper): min={actions[:, 6].min():.4f}, max={actions[:, 6].max():.4f}")
        print(f"  actions[:, 7] (first pad): min={actions[:, 7].min():.6f}, max={actions[:, 7].max():.6f}")
        print(f"  actions[:, 31] (last pad): min={actions[:, 31].min():.6f}, max={actions[:, 31].max():.6f}")
    print()

    # stage 5: verify the inference inverse path
    print("=" * 70)
    print("INVERSE PIPELINE (inference path)")
    print()

    # pretend the model returned the padded normalized actions exactly
    model_output = copy.deepcopy(sample_padded)

    # inverse stage 1: unnormalize
    if norm_stats:
        unnormalize_fn = _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        inv_sample = unnormalize_fn(copy.deepcopy(model_output))
        print("INV STAGE 1: After Unnormalize")
        if "actions" in inv_sample:
            actions = np.asarray(inv_sample["actions"])
            print(f"  actions shape: {actions.shape}")
            print(f"  actions[:, 6] (gripper): min={actions[:, 6].min():.4f}, max={actions[:, 6].max():.4f}")
            print(f"  actions[:, 0] (joint 0): min={actions[:, 0].min():.4f}, max={actions[:, 0].max():.4f}")
    else:
        inv_sample = copy.deepcopy(model_output)
    print()

    # inverse stage 2: UR5Outputs (slice back to 7 dims) + AbsoluteActions
    output_transforms = data_config.data_transforms.outputs
    inv_sample2 = copy.deepcopy(inv_sample)
    for t in output_transforms:
        inv_sample2 = t(inv_sample2)
    print("INV STAGE 2: After output_transforms (UR5Outputs + AbsoluteActions)")
    if "actions" in inv_sample2:
        actions = np.asarray(inv_sample2["actions"])
        print(f"  actions shape: {actions.shape}")
        if actions.shape[-1] >= 7:
            print(f"  actions[:, 6] (gripper, final): min={actions[:, 6].min():.4f}, max={actions[:, 6].max():.4f}")
            print(f"  actions[:, 0] (joint 0, final): min={actions[:, 0].min():.4f}, max={actions[:, 0].max():.4f}")
            print(f"  First 10 final gripper: {actions[:10, 6].round(4)}")

    print()
    print("=" * 70)
    print("ROUND-TRIP COMPARISON")
    raw_gripper = np.asarray(sample_pre_delta["actions"])[:, 6]
    final_gripper = np.asarray(inv_sample2["actions"])[:, 6] if "actions" in inv_sample2 else None
    if final_gripper is not None:
        n = min(len(raw_gripper), len(final_gripper))
        max_err = np.max(np.abs(raw_gripper[:n] - final_gripper[:n]))
        print(f"  Max round-trip error (gripper): {max_err:.6f}")
        if max_err < 0.01:
            print("  PASS: Round-trip preserves gripper values")
        else:
            print("  FAIL: Round-trip introduces significant error!")


if __name__ == "__main__":
    main()
