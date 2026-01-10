"""
Helper script to verify action unnormalization for UR5.

This script helps diagnose if actions from the policy server are being
unnormalized correctly by:
1. Loading and inspecting norm_stats
2. Manually applying unnormalization to verify the formula
3. Checking if action values are in expected ranges
"""

import numpy as np
from etils import epath
from openpi.shared import normalize as _normalize
from openpi.shared import download as _download


def load_ur5_norm_stats(assets_dir: str = "gs://openpi-assets/checkpoints/pi05_base/assets", asset_id: str = "ur5e"):
    """Load UR5 norm_stats from the assets directory."""
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats_dir_downloaded = _download.maybe_download(str(norm_stats_dir))
    norm_stats = _normalize.load(norm_stats_dir_downloaded)
    print(f"Loaded norm_stats from: {norm_stats_dir_downloaded}")
    return norm_stats


def inspect_norm_stats(norm_stats: dict):
    """Print detailed information about norm_stats."""
    print("\n=== Norm Stats Inspection ===")
    for key, stats in norm_stats.items():
        print(f"\n{key}:")
        print(f"  mean shape: {stats.mean.shape}, mean: {stats.mean}")
        print(f"  std shape: {stats.std.shape}, std: {stats.std}")
        if stats.q01 is not None:
            print(f"  q01 shape: {stats.q01.shape}, q01: {stats.q01}")
            print(f"  q99 shape: {stats.q99.shape}, q99: {stats.q99}")
            print(f"  Range: [{stats.q01}, {stats.q99}]")
            print(f"  Range (deg): [{np.degrees(stats.q01)}, {np.degrees(stats.q99)}]")


def unnormalize_quantile(x: np.ndarray, stats) -> np.ndarray:
    """Manually apply quantile unnormalization (for pi05 models)."""
    assert stats.q01 is not None
    assert stats.q99 is not None
    q01, q99 = stats.q01, stats.q99
    if (dim := q01.shape[-1]) < x.shape[-1]:
        return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def unnormalize_zscore(x: np.ndarray, stats) -> np.ndarray:
    """Manually apply z-score unnormalization (for pi0 models)."""
    mean = stats.mean[..., :x.shape[-1]] if stats.mean.shape[-1] >= x.shape[-1] else np.pad(
        stats.mean, (0, x.shape[-1] - stats.mean.shape[-1]), mode='constant', constant_values=0.0
    )
    std = stats.std[..., :x.shape[-1]] if stats.std.shape[-1] >= x.shape[-1] else np.pad(
        stats.std, (0, x.shape[-1] - stats.std.shape[-1]), mode='constant', constant_values=1.0
    )
    return x * (std + 1e-6) + mean


def verify_action_unnormalization(
    normalized_action: np.ndarray,
    norm_stats: dict,
    use_quantiles: bool = True,
    action_mode: str = "delta"
):
    """
    Verify if an action is unnormalized correctly.
    
    Args:
        normalized_action: Action values as they come from the model (should be in [-1, 1] if normalized)
        norm_stats: Dictionary of norm_stats with "actions" key
        use_quantiles: Whether quantile normalization is used (True for pi05, False for pi0)
        action_mode: "delta" or "absolute"
    """
    print("\n=== Action Unnormalization Verification ===")
    print(f"Input action (normalized): {normalized_action}")
    print(f"Input action range: [{normalized_action.min():.4f}, {normalized_action.max():.4f}]")
    
    if "actions" not in norm_stats:
        print("⚠️  WARNING: 'actions' key not found in norm_stats!")
        print(f"Available keys: {list(norm_stats.keys())}")
        return
    
    stats = norm_stats["actions"]
    
    # Apply unnormalization manually
    if use_quantiles:
        unnormalized = unnormalize_quantile(normalized_action, stats)
        print(f"\nUsing quantile unnormalization (pi05):")
        print(f"  Formula: (x + 1.0) / 2.0 * (q99 - q01) + q01")
    else:
        unnormalized = unnormalize_zscore(normalized_action, stats)
        print(f"\nUsing z-score unnormalization (pi0):")
        print(f"  Formula: x * std + mean")
    
    print(f"\nUnnormalized action: {unnormalized}")
    print(f"Unnormalized action (deg): {np.degrees(unnormalized)}")
    print(f"Unnormalized range: [{unnormalized.min():.4f}, {unnormalized.max():.4f}]")
    print(f"Unnormalized range (deg): [{np.degrees(unnormalized.min()):.2f}°, {np.degrees(unnormalized.max()):.2f}°]")
    
    # Check if values are reasonable
    max_abs = np.max(np.abs(unnormalized))
    max_abs_deg = np.degrees(max_abs)
    
    print(f"\n=== Reasonableness Check ===")
    if action_mode == "delta":
        print(f"Max abs delta: {max_abs:.4f} rad ({max_abs_deg:.2f}°)")
        if max_abs_deg < 0.1:
            print("✅ GOOD: Delta is very small (< 0.1°)")
        elif max_abs_deg < 1.0:
            print("✅ OK: Delta is small (< 1°)")
        elif max_abs_deg < 5.0:
            print("⚠️  WARNING: Delta is moderate (1-5°). May be acceptable but check.")
        else:
            print(f"❌ ERROR: Delta is very large ({max_abs_deg:.2f}°). Unnormalization may be incorrect!")
            print("   Expected deltas should typically be < 0.1 rad (6°) per step.")
    else:  # absolute
        print(f"Max abs value: {max_abs:.4f} rad ({max_abs_deg:.2f}°)")
        if max_abs_deg < 180.0:
            print("✅ GOOD: Value is within joint limits (±180°)")
        else:
            print(f"❌ ERROR: Value exceeds joint limits ({max_abs_deg:.2f}° > 180°)")
    
    return unnormalized


def example_usage():
    """Example of how to use this script."""
    print("Loading UR5 norm_stats...")
    norm_stats = load_ur5_norm_stats()
    
    inspect_norm_stats(norm_stats)
    
    # Example: Check what a normalized action [-1, 1] would unnormalize to
    print("\n" + "="*60)
    print("Example 1: Normalized action = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.0] (6 joints)")
    normalized_example = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)
    verify_action_unnormalization(normalized_example, norm_stats, use_quantiles=True, action_mode="delta")
    
    print("\n" + "="*60)
    print("Example 2: Normalized action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (zero action)")
    zero_action = np.zeros(6, dtype=np.float32)
    verify_action_unnormalization(zero_action, norm_stats, use_quantiles=True, action_mode="delta")
    
    print("\n" + "="*60)
    print("Example 3: Large normalized action = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]")
    large_action = np.array([0.8] * 6, dtype=np.float32)
    verify_action_unnormalization(large_action, norm_stats, use_quantiles=True, action_mode="delta")


if __name__ == "__main__":
    example_usage()
