import dataclasses
import logging
import socket

import tyro

from openpi.configs.tasks import TASK_REGISTRY
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_b1k_server
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    robot: str
    task: str
    # Specifies how to load the policy.
    policy: Checkpoint

    # Normalization asset directory inside the checkpoint. This is independent
    # of `task`, which only selects the language prompt.
    asset_id: str | None = None
    # Alias for `asset_id`, retained for existing launch commands.
    repo_id: str | None = None

    control_mode: str = "receding_horizon"
    # Number of actions to execute before replanning.
    action_horizon: int = 15
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Optional fixed metric-depth bounds. If omitted, each live frame uses its
    # valid 1st/99th percentiles, the causal analogue of training-time per-video
    # normalization.
    depth_bounds: tuple[float, float] | None = None
    # Percentiles used when fixed metric bounds are not supplied.
    depth_percentiles: tuple[float, float] = (1.0, 99.0)
    # Log inverse-depth clipping range used by the depth-video builder.
    depth_inverse_clip: tuple[float, float] = (1.0, 100.0)


def main(args: Args) -> None:
    # Load task from registry
    task_bucket, task_name = args.task.split("/")
    task_prompt = TASK_REGISTRY[task_bucket][task_name]
    # log the prompt used
    logging.info(f"Using robot: {args.robot}, prompt: {task_prompt}")

    # Load training config and override request-specific fields. The prompt task
    # and normalization assets are deliberately independent: a task such as
    # s2rg/radio is not a dataset/asset identifier.
    config = _config.get_config(args.policy.config)
    if args.asset_id is not None and args.repo_id is not None:
        raise ValueError("--asset-id and --repo-id refer to the same setting; provide only one value")
    asset_id = args.asset_id or args.repo_id
    data_config = dataclasses.replace(config.data, robot_config_name=args.robot)
    if asset_id is not None:
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, asset_id=asset_id),
        )
    config = dataclasses.replace(config, data=data_config)
    resolved_asset_id = asset_id or config.data.assets.asset_id or config.data.repo_id
    logging.info("Using checkpoint norm stats asset: %s", resolved_asset_id)

    policy = _policy_config.create_trained_policy(
        config, args.policy.dir, default_prompt=task_prompt
    )
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    policy = B1KPolicyWrapper(
        policy=policy,
        robot=args.robot,
        text_prompt=task_prompt,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        max_len=config.model.action_horizon,
        depth_bounds=args.depth_bounds,
        depth_percentiles=args.depth_percentiles,
        depth_inverse_clip=args.depth_inverse_clip,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_b1k_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
