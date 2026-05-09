"""widowx ai constants. override via env."""

import math
import os


WX_IP = os.environ.get("WX_IP", "192.168.1.4")

# 6 arm joints + gripper carriage, per the lerobot driver
WX_JOINT_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
)
WX_GRIPPER_KEY = "left_carriage_joint.pos"


def _parse_floats(env_val: str, default: tuple[float, ...], expected_len: int | None = None) -> tuple[float, ...]:
    if not env_val:
        return default
    parts = tuple(float(x) for x in env_val.split(","))
    if expected_len is not None and len(parts) != expected_len:
        raise ValueError(f"expected {expected_len} floats, got {len(parts)}")
    return parts


def _parse_ints(env_val: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if not env_val:
        return default
    return tuple(int(x) for x in env_val.split(","))


# 7-vec home pose: 6 arm joints (rad) + gripper (m). driver homes here on connect
# and returns here on disconnect. j0=pi/2 faces the arm sideways; j3=0.2 tilts
# the head a little down from trossen's default.
# joint roles on this arm: j0 base yaw, j1 shoulder pitch, j2 elbow pitch,
# j3 wrist pitch, j4 wrist yaw, j5 wrist roll.
WX_STAGED_RAD = _parse_floats(
    os.environ.get("WX_STAGED_RAD", ""),
    (math.pi / 2.0, math.pi / 3.0, math.pi / 6.0, 0.2, 0.0, 0.0, 0.0),
    expected_len=7,
)
WX_HOME_RAD = WX_STAGED_RAD[:6]

# ur5 -> widowx joint remap, identity by default. both arms list proximal->distal.
UR5_TO_WX_PERM = _parse_ints(os.environ.get("UR5_TO_WX_PERM", ""), (0, 1, 2, 3, 4, 5))
UR5_TO_WX_SIGN = _parse_floats(os.environ.get("UR5_TO_WX_SIGN", ""), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

# carriage firmware range is [-0.004, 0.044] m. stay 6 mm below the ceiling to
# absorb motor overshoot at slow goal_time. ur5 0=open, 1=closed.
WX_GRIPPER_OPEN_M = float(os.environ.get("WX_GRIPPER_OPEN_M", "0.035"))
WX_GRIPPER_CLOSED_M = float(os.environ.get("WX_GRIPPER_CLOSED_M", "0.002"))
WX_GRIPPER_MIN_M = float(os.environ.get("WX_GRIPPER_MIN_M", "0.0"))
WX_GRIPPER_MAX_M = float(os.environ.get("WX_GRIPPER_MAX_M", "0.038"))

# tighter than the driver's 5.0 rad default; the model is kinematically blind.
WX_MAX_REL_RAD = float(os.environ.get("WX_MAX_REL_RAD", "0.05"))

# higher multiplier = smoother and slower
WX_LOOP_RATE = int(os.environ.get("WX_LOOP_RATE", "30"))
WX_MIN_TIME_MULT = float(os.environ.get("WX_MIN_TIME_MULT", "6.0"))

# bail the run if a chunk hits the soft limits this many times
WX_LIMIT_CLAMP_ABORT_N = int(os.environ.get("WX_LIMIT_CLAMP_ABORT_N", "5"))

# soft per-joint clamp on the 6 arm joints (rad); tighten via env
WX_JOINT_LIMITS_LOW = _parse_floats(
    os.environ.get("WX_JOINT_LIMITS_LOW", ""),
    (-math.pi, -math.pi / 2.0, -math.pi / 2.0, -math.pi, -math.pi / 2.0, -math.pi),
)
WX_JOINT_LIMITS_HIGH = _parse_floats(
    os.environ.get("WX_JOINT_LIMITS_HIGH", ""),
    (math.pi, math.pi / 2.0, math.pi / 2.0, math.pi, math.pi / 2.0, math.pi),
)
