"""Small RSL-RL CLI helpers used by the navigation playback entrypoint."""

from __future__ import annotations

import argparse


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add the checkpoint and runner overrides needed for playback."""

    group = parser.add_argument_group("rsl_rl")
    group.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint path.")
    group.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Gym registry key for the runner config.")
    group.add_argument("--seed", type=int, default=42)


def update_rsl_rl_cfg(agent_cfg, args_cli):
    """Apply playback overrides to the registered runner config."""

    agent_cfg.seed = args_cli.seed
    agent_cfg.resume = True
    agent_cfg.load_checkpoint = args_cli.checkpoint
    return agent_cfg
