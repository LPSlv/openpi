"""Main entry point for UR5 robot runtime."""

import dataclasses
import logging
import os

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.ur5 import env as _env


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # If None, will auto-use the action horizon reported by the policy server metadata (recommended).
    action_horizon: int | None = None

    num_episodes: int = 1
    max_episode_steps: int = 1000

    # UR5 specific configuration
    ur_ip: str = os.environ.get("UR_IP", "192.168.1.116")
    rs_base_serial: str = os.environ.get("RS_BASE", "")
    rs_wrist_serial: str = os.environ.get("RS_WRIST", "")
    gripper_port: int = int(os.environ.get("GRIPPER_PORT", "0"))
    fake_cam: bool = os.environ.get("FAKE_CAM", "0") == "1"


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    action_horizon = int(metadata.get("action_horizon", 15)) if args.action_horizon is None else int(args.action_horizon)
    runtime = _runtime.Runtime(
        environment=_env.UR5RealEnvironment(
            ur_ip=args.ur_ip,
            reset_position=metadata.get("reset_pose"),
            rs_base_serial=args.rs_base_serial if args.rs_base_serial else None,
            rs_wrist_serial=args.rs_wrist_serial if args.rs_wrist_serial else None,
            gripper_port=args.gripper_port,
            fake_cam=args.fake_cam,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=action_horizon,
            )
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        # Clean up environment resources
        runtime._environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
