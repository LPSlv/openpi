"""
Check if norm_stats have a units mismatch (degrees vs radians).

The norm_stats show q99 = 67° and q01 = -69.47°.
If these values are stored in the file as degrees (but should be radians),
that would explain the large ranges.
"""

import numpy as np
from etils import epath
from openpi.shared import normalize as _normalize
from openpi.shared import download as _download


def check_units_mismatch():
    """Check if norm_stats values suggest a degrees/radians mismatch."""
    
    # Load norm_stats
    assets_dir = "gs://openpi-assets/checkpoints/pi05_base/assets"
    asset_id = "ur5e"
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats_dir_downloaded = _download.maybe_download(str(norm_stats_dir))
    norm_stats = _normalize.load(norm_stats_dir_downloaded)
    
    print("=== Checking for Units Mismatch ===\n")
    
    if "actions" not in norm_stats:
        print("ERROR: 'actions' key not found in norm_stats")
        return
    
    stats = norm_stats["actions"]
    
    # Get the first 6 dimensions (joints, ignoring gripper)
    q01_joints = stats.q01[:6] if stats.q01 is not None else None
    q99_joints = stats.q99[:6] if stats.q99 is not None else None
    
    if q01_joints is None or q99_joints is None:
        print("ERROR: q01 or q99 not available (not using quantile normalization?)")
        return
    
    print("Raw values from norm_stats file (should be in radians):")
    print(f"  q01 (first 6 joints): {q01_joints}")
    print(f"  q99 (first 6 joints): {q99_joints}")
    print()
    
    # Convert to degrees for display
    q01_deg = np.degrees(q01_joints)
    q99_deg = np.degrees(q99_joints)
    
    print("Converted to degrees (for display):")
    print(f"  q01: {q01_deg}")
    print(f"  q99: {q99_deg}")
    print()
    
    # Check if values look like they might be in degrees
    # If q99 is around 67, and that's stored as 67 (not 67 * π/180),
    # then it might be degrees stored as radians
    max_abs_rad = np.max(np.abs(q99_joints))
    max_abs_deg = np.degrees(max_abs_rad)
    
    print(f"Maximum absolute value: {max_abs_rad:.4f} rad = {max_abs_deg:.2f}°")
    print()
    
    # Hypothesis 1: Values are correctly in radians
    print("=== Hypothesis 1: Values are correctly in radians ===")
    print(f"  q99 = {max_abs_rad:.4f} rad = {max_abs_deg:.2f}°")
    if max_abs_deg > 10:
        print(f"  ❌ This is VERY LARGE for delta actions (expected < 6°)")
        print(f"  This suggests the training data had very large action ranges.")
    else:
        print(f"  ✅ This is reasonable for delta actions")
    print()
    
    # Hypothesis 2: Values are stored in degrees (but should be radians)
    # If the file has 67 stored, and that's meant to be 67 degrees,
    # then it should be 67 * π/180 ≈ 1.17 rad
    # But if it's stored as 67 (as a number), and we interpret it as radians,
    # then 67 rad = 3838°, which is clearly wrong
    # So if we see values around 1-2 rad that convert to 60-120°, 
    # they might have been stored as degrees
    
    # Check: if we assume the stored values are degrees (wrong),
    # what would they be in radians?
    if max_abs_deg > 50 and max_abs_rad < 2.0:
        print("=== Hypothesis 2: Values might be stored in degrees (WRONG) ===")
        print(f"  If q99 = {max_abs_deg:.2f} was stored as degrees (wrong),")
        print(f"  it should be {max_abs_deg * np.pi / 180:.4f} rad")
        print(f"  But file has: {max_abs_rad:.4f} rad")
        print(f"  ❌ This doesn't match - values are likely correctly in radians")
    elif max_abs_rad > 10:
        print("=== Hypothesis 2: Values might be stored in degrees (WRONG) ===")
        print(f"  If q99 = {max_abs_rad:.2f} was stored as degrees (wrong),")
        print(f"  it should be {max_abs_rad * np.pi / 180:.4f} rad")
        print(f"  This would be {np.degrees(max_abs_rad * np.pi / 180):.2f}° when converted")
        print(f"  ⚠️  This is a POSSIBLE units mismatch!")
        print(f"  If true, correct q99 should be: {max_abs_rad * np.pi / 180:.4f} rad = {max_abs_rad:.2f}°")
    else:
        print("=== Hypothesis 2: Values might be stored in degrees (WRONG) ===")
        print(f"  Values are small enough that this is unlikely")
    print()
    
    # Hypothesis 3: Training data was in degrees, converted incorrectly
    # If training data had actions in degrees (e.g., 67°),
    # and they were stored as 67 (not converted to rad),
    # then 67 would be interpreted as 67 rad = 3838°
    # But we see 67°, so this doesn't match
    
    # Check: what if the training data had absolute positions in degrees?
    # For absolute positions, ±180° is reasonable
    # But for delta actions, ±67° is still very large
    
    print("=== Expected Ranges ===")
    print("For delta actions (per step):")
    print("  Expected: < 0.1 rad (6°)")
    print("  Moderate: 0.1-0.5 rad (6-30°)")
    print("  Large: > 0.5 rad (30°)")
    print()
    print("For absolute positions:")
    print("  Expected: ±π rad (±180°)")
    print()
    print(f"Your values: ±{max_abs_rad:.4f} rad = ±{max_abs_deg:.2f}°")
    print()
    
    # Final assessment
    if max_abs_deg < 10:
        print("✅ Values are reasonable - likely no units mismatch")
    elif max_abs_deg < 180 and max_abs_rad < 3.14:
        print("⚠️  Values are large but within absolute position range")
        print("   This suggests training data had large action ranges, not a units issue")
    elif max_abs_rad > 10:
        print("❌ Values are extremely large - possible units mismatch!")
        print("   If q99 = 67 was meant to be 67°, it should be stored as 67 * π/180 ≈ 1.17 rad")
    else:
        print("⚠️  Values are large - check if training data had correct units")


if __name__ == "__main__":
    check_units_mismatch()
