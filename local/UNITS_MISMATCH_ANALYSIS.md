# Units Mismatch Analysis

## Your Question
Could the large norm_stats ranges (`[-69.47°, 67.00°]`) be due to a units mismatch (degrees vs radians) rather than actually large action ranges?

## Answer: Likely NOT a units mismatch, but let's verify

### What We Know

1. **Expected units**: According to OpenPI docs, joint angles should be in **radians**
2. **Your norm_stats**: Show q99 ≈ 67° when converted from stored values
3. **If stored correctly**: q99 = 67° means q99 ≈ 1.17 rad (67 * π/180)
4. **If stored incorrectly**: If q99 = 67 was meant to be 67°, it should be stored as 1.17 rad

### Analysis

From your output:
```
[NORM_STATS CHECK] Expected unnormalized range: [-69.47°, 67.00°]
```

This means:
- q99 ≈ 1.17 rad (which converts to 67°)
- q01 ≈ -1.21 rad (which converts to -69.47°)

**These are correctly stored in radians**, but they represent very large ranges.

### Possible Scenarios

#### Scenario 1: Correct Units, Large Training Data (Most Likely)
- Training data had actions with large ranges (±67° deltas)
- Stored correctly in radians (1.17 rad)
- This is the actual behavior of the training data
- **Solution**: Use different norm_stats or recompute with smaller-range data

#### Scenario 2: Training Data in Degrees, Stored as Degrees (Wrong)
- Training data had actions in degrees (e.g., 67°)
- Stored as 67 (not converted to radians)
- When unnormalizing: 67 is interpreted as 67 rad = 3838°
- **But**: You're seeing 67°, not 3838°, so this is NOT the case

#### Scenario 3: Training Data in Degrees, Converted Incorrectly
- Training data had actions in degrees
- Converted to radians incorrectly (e.g., stored as 67 instead of 1.17)
- **But**: Again, you'd see 3838°, not 67°

### How to Verify

Run the diagnostic script to see raw values:

```bash
# Inside Docker
python local/check_units_mismatch.py
```

Or check the raw radian values in the bridge output (now shown with VERIFY_NORM_STATS=1):
```
[NORM_STATS CHECK] Raw norm_stats values (rad): q01=-1.2123, q99=1.1684
```

### Expected Values

**For delta actions:**
- Expected: < 0.1 rad (6°)
- Your values: ±1.17 rad (67°) ❌ **VERY LARGE**

**For absolute positions:**
- Expected: ±π rad (±180°)
- Your values: ±1.17 rad (67°) ✅ Within range, but still large for deltas

### Conclusion

**Most likely**: The norm_stats are correctly stored in radians, but the training data had very large action ranges. The values represent actual behavior from training, not a units error.

**To fix**: You would need norm_stats computed from data with smaller action ranges, or the training data itself had these large ranges.

### Quick Check

If you see in the bridge output:
```
Raw norm_stats values (rad): q01=-1.21, q99=1.17
```

This confirms:
- ✅ Values are in radians (correct)
- ❌ But they represent ±67° ranges (very large for deltas)

If you saw:
```
Raw norm_stats values (rad): q01=-69.47, q99=67.00
```

This would indicate:
- ❌ Values stored as degrees (wrong)
- Would need conversion: 67° → 1.17 rad
