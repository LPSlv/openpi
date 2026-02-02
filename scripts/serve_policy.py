import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    UR5 = "ur5"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Optional override: load normalization stats from a specific directory containing `norm_stats.json`.
    # If set, this replaces the norm stats that would otherwise be loaded from the checkpoint directory.
    #
    # Example (UR5 bus-the-table stats in this repo):
    #   --norm_stats_dir=assets/pi05_ur5_low_mem_finetune/ims/ur5_bus_the_table
    norm_stats_dir: str | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    #
    # Note: In tyro, union subcommands (like policy:checkpoint) effectively terminate parsing for subsequent
    # fields, so keep any global flags (like norm_stats_dir) *above* this field.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.UR5: Checkpoint(
        config="pi05_ur5_low_mem_finetune",
        dir="checkpoints/pi05_ur5_low_mem_finetune/ur5_second/499",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    norm_stats = _normalize.load(args.norm_stats_dir) if args.norm_stats_dir else None
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                norm_stats=norm_stats,
            )
        case Default():
            checkpoint = DEFAULT_CHECKPOINT[args.env]
            return _policy_config.create_trained_policy(
                _config.get_config(checkpoint.config),
                checkpoint.dir,
                default_prompt=args.default_prompt,
                norm_stats=norm_stats,
            )


def main(args: Args) -> None:
    policy = create_policy(args)

    # Resolve which checkpoint/config we actually loaded so we can surface it in metadata/logs.
    # This helps avoid the common footgun of thinking you're serving "the new checkpoint" while
    # actually still running the default one from SERVER_ARGS / --env.
    if isinstance(args.policy, Checkpoint):
        resolved_ckpt = args.policy
    else:
        resolved_ckpt = DEFAULT_CHECKPOINT[args.env]

    policy_metadata = dict(policy.metadata)
    policy_metadata["train_config"] = resolved_ckpt.config
    policy_metadata["checkpoint_dir"] = resolved_ckpt.dir
    if args.norm_stats_dir:
        policy_metadata["norm_stats_dir"] = args.norm_stats_dir

    logging.info("Serving policy: config=%s checkpoint_dir=%s", resolved_ckpt.config, resolved_ckpt.dir)
    if args.norm_stats_dir:
        logging.info("Using norm stats override from: %s", args.norm_stats_dir)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
