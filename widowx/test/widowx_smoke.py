"""smoke test: connect, dump features, hold, disconnect."""

import argparse
import os
import time

from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower

from widowx import defaults as _wx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=os.environ.get("WX_IP", _wx.WX_IP))
    ap.add_argument("--hold-s", type=float, default=5.0)
    ap.add_argument("--max-rel-rad", type=float, default=_wx.WX_MAX_REL_RAD)
    args = ap.parse_args()

    print(f"connecting to {args.ip}...", flush=True)
    cfg = WidowXAIFollowerConfig(
        ip_address=args.ip,
        max_relative_target=args.max_rel_rad,
        loop_rate=_wx.WX_LOOP_RATE,
        min_time_to_move_multiplier=_wx.WX_MIN_TIME_MULT,
        staged_positions=list(_wx.WX_STAGED_RAD),
        cameras={},
    )
    robot = WidowXAIFollower(cfg)
    robot.connect(calibrate=False)
    print("connected", flush=True)

    print("\nobservation_features:")
    for k, v in robot.observation_features.items():
        print(f"  {k}: {v}")
    print("\naction_features:")
    for k, v in robot.action_features.items():
        print(f"  {k}: {v}")

    obs = robot.get_observation()
    print("\nfirst observation:")
    for k in sorted(obs.keys()):
        if k.endswith(".pos"):
            print(f"  {k}: {obs[k]:.4f}")

    print(f"\nholding {args.hold_s:.1f}s...", flush=True)
    time.sleep(args.hold_s)

    print("disconnecting...", flush=True)
    robot.disconnect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
