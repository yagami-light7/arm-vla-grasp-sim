# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class ArmJointPositionCommand(CommandTerm):
    """Command generator that samples target joint positions for arm joints."""

    cfg: ArmJointPositionCommandCfg  # type: ignore

    def __init__(self, cfg: ArmJointPositionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.asset_name]
        joint_names = cfg.joint_names
        if joint_names is None:
            raise ValueError("ArmJointPositionCommand requires joint_names to be specified.")
        if isinstance(joint_names, str):
            joint_names = [joint_names]
        self.joint_ids, _ = self.asset.find_joints(joint_names, preserve_order=cfg.preserve_order)
        if len(self.joint_ids) == 0:
            raise ValueError(f"No joints matched joint_names={joint_names} in asset '{cfg.asset_name}'.")
        self.command_buffer = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        joint_defaults = self.asset.data.default_joint_pos[env_ids][:, self.joint_ids]
        pos_range = self.cfg.position_range
        if isinstance(pos_range, (list, tuple)) and len(pos_range) > 0 and isinstance(pos_range[0], (list, tuple)):
            if len(pos_range) != len(self.joint_ids):
                raise ValueError("position_range list must match arm joint count.")
            min_range = torch.tensor([r[0] for r in pos_range], device=self.device)
            max_range = torch.tensor([r[1] for r in pos_range], device=self.device)
            offsets = torch.empty((len(env_ids), len(self.joint_ids)), device=self.device)
            offsets.uniform_(0.0, 1.0)
            offsets = min_range + offsets * (max_range - min_range)
        else:
            offsets = torch.empty((len(env_ids), len(self.joint_ids)), device=self.device).uniform_(*pos_range)
        if self.cfg.use_default_offset:
            target_pos = joint_defaults + offsets
        else:
            target_pos = offsets
        if self.cfg.clip_to_joint_limits:
            limits = self.asset.data.soft_joint_pos_limits[env_ids][:, self.joint_ids, :]
            min_pos = limits[..., 0]
            max_pos = limits[..., 1]
            target_pos = torch.max(torch.min(target_pos, max_pos), min_pos)
        self.command_buffer[env_ids] = target_pos

    def _update_command(self):
        pass


@configclass
class ArmJointPositionCommandCfg(CommandTermCfg):
    """Configuration for arm joint position commands."""

    class_type: type = ArmJointPositionCommand
    asset_name: str = "robot"
    joint_names: list[str] | str | None = None
    preserve_order: bool = True
    position_range: tuple[float, float] | list[tuple[float, float]] = (-0.5, 0.5)
    use_default_offset: bool = True
    clip_to_joint_limits: bool = True
