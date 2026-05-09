"""ur5 <-> widowx joint and gripper remap."""

from __future__ import annotations

import numpy as np

from widowx import defaults as _wx


def ur5_to_wx_q(q_ur5: np.ndarray) -> np.ndarray:
    q = np.asarray(q_ur5, dtype=np.float64).reshape(-1)
    if q.shape[0] != 6:
        raise ValueError(f"expected (6,), got {q.shape}")
    perm = np.asarray(_wx.UR5_TO_WX_PERM, dtype=np.int64)
    sign = np.asarray(_wx.UR5_TO_WX_SIGN, dtype=np.float64)
    return sign * q[perm]


def wx_to_ur5_q(q_wx: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wx, dtype=np.float64).reshape(-1)
    if q.shape[0] != 6:
        raise ValueError(f"expected (6,), got {q.shape}")
    perm = np.asarray(_wx.UR5_TO_WX_PERM, dtype=np.int64)
    sign = np.asarray(_wx.UR5_TO_WX_SIGN, dtype=np.float64)
    inv_perm = np.argsort(perm)
    return (q / sign)[inv_perm]


def map_gripper_ur5_to_wx(g_norm: float) -> float:
    # hard clamp to [MIN_M, MAX_M] so motor overshoot at slow goal_time can't
    # trip the firmware joint-6 fault
    g = float(np.clip(g_norm, 0.0, 1.0))
    m = (1.0 - g) * _wx.WX_GRIPPER_OPEN_M + g * _wx.WX_GRIPPER_CLOSED_M
    return float(np.clip(m, _wx.WX_GRIPPER_MIN_M, _wx.WX_GRIPPER_MAX_M))


def clip_step(q_target_wx: np.ndarray, q_current_wx: np.ndarray, max_rel_rad: float) -> np.ndarray:
    delta = np.asarray(q_target_wx, dtype=np.float64) - np.asarray(q_current_wx, dtype=np.float64)
    delta = np.clip(delta, -max_rel_rad, max_rel_rad)
    return np.asarray(q_current_wx, dtype=np.float64) + delta


def clamp_to_limits(
    q_wx: np.ndarray,
    low: tuple[float, ...],
    high: tuple[float, ...],
) -> tuple[np.ndarray, int]:
    q = np.asarray(q_wx, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    out = np.minimum(np.maximum(q, lo), hi)
    n_clamped = int(np.sum(out != q))
    return out, n_clamped
