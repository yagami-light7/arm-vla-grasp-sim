# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def arm_joint_position_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    initial_position_range: Sequence[Sequence[float]],
    final_position_range: Sequence[Sequence[float]],
    curriculum_iterations: int = 2000,
) -> None:
    """Linearly expand arm joint command ranges over training.

    This is used for arm-unlock stages that resume from a checkpoint trained with the arm fixed at
    its default pose. The network interface stays unchanged while the arm command distribution is
    widened gradually.
    """

    del env_ids

    if len(initial_position_range) != len(final_position_range):
        raise ValueError("initial_position_range and final_position_range must have the same length.")

    current_iter = getattr(env, "common_step_counter", 0) // getattr(env, "max_episode_length", 1)
    state = getattr(env, "_arm_joint_range_curriculum_state", None)
    if state is None:
        state = {}
        env._arm_joint_range_curriculum_state = state

    if command_name not in state:
        state[command_name] = {
            "start_iter": current_iter,
            "initial_position_range": [tuple(bounds) for bounds in initial_position_range],
            "final_position_range": [tuple(bounds) for bounds in final_position_range],
            "total_iters": curriculum_iterations,
        }

    cfg = env.command_manager.get_term(command_name).cfg
    command_state = state[command_name]
    progress = min(
        (current_iter - command_state["start_iter"]) / max(command_state["total_iters"], 1),
        1.0,
    )

    current_position_range = []
    for init_bounds, final_bounds in zip(
        command_state["initial_position_range"], command_state["final_position_range"], strict=True
    ):
        lower = init_bounds[0] + (final_bounds[0] - init_bounds[0]) * progress
        upper = init_bounds[1] + (final_bounds[1] - init_bounds[1]) * progress
        current_position_range.append((float(lower), float(upper)))

    cfg.position_range = current_position_range
    return progress


def arm_joint_position_range_staged_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    position_ranges: Sequence[Sequence[Sequence[float]]],
    stage_iterations: Sequence[int],
) -> None:
    """Interpolate arm joint command ranges across multiple curriculum stages.

    Args:
        env: The learning environment.
        env_ids: Environment IDs (unused, kept for curriculum interface compatibility).
        command_name: Command term name.
        position_ranges: A sequence of per-joint position ranges. Each element defines one stage.
        stage_iterations: Cumulative iteration anchors for the corresponding stages. The first entry
            should typically be 0. For example, with 3 stages and ``[0, 48, 128]``, the curriculum
            interpolates stage 1 -> stage 2 over iterations [0, 48], then stage 2 -> stage 3 over
            iterations [48, 128].
    """

    del env_ids

    if len(position_ranges) != len(stage_iterations):
        raise ValueError("position_ranges and stage_iterations must have the same length.")
    if len(position_ranges) < 2:
        raise ValueError("At least two stages are required for staged arm position curriculum.")
    if any(stage_iterations[idx] > stage_iterations[idx + 1] for idx in range(len(stage_iterations) - 1)):
        raise ValueError("stage_iterations must be non-decreasing.")

    joint_count = len(position_ranges[0])
    if any(len(stage_range) != joint_count for stage_range in position_ranges):
        raise ValueError("All staged arm position ranges must have the same joint count.")

    current_iter = getattr(env, "common_step_counter", 0) // getattr(env, "max_episode_length", 1)
    cfg = env.command_manager.get_term(command_name).cfg

    if current_iter <= stage_iterations[0]:
        cfg.position_range = [tuple(bounds) for bounds in position_ranges[0]]
        return 0.0

    if current_iter >= stage_iterations[-1]:
        cfg.position_range = [tuple(bounds) for bounds in position_ranges[-1]]
        return float(len(position_ranges) - 1)

    stage_idx = 1
    while current_iter > stage_iterations[stage_idx]:
        stage_idx += 1

    prev_iter = stage_iterations[stage_idx - 1]
    next_iter = stage_iterations[stage_idx]
    denom = max(next_iter - prev_iter, 1)
    progress = (current_iter - prev_iter) / denom

    current_position_range = []
    for prev_bounds, next_bounds in zip(position_ranges[stage_idx - 1], position_ranges[stage_idx], strict=True):
        lower = prev_bounds[0] + (next_bounds[0] - prev_bounds[0]) * progress
        upper = prev_bounds[1] + (next_bounds[1] - prev_bounds[1]) * progress
        current_position_range.append((float(lower), float(upper)))

    cfg.position_range = current_position_range
    return float(stage_idx - 1) + progress


def reward_weights_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    p1_weights: dict,
    p2_weights: dict,
    curriculum_iterations: int = 2000,
) -> None:
    """Gradually transition reward weights from Phase 1 (Foundation Flat) to Phase 2 (Robust Rough) values.

    This curriculum function linearly interpolates reward weights from P1 values to P2 values
    over a specified number of training iterations. This helps smooth the transition when
    fine-tuning a model trained on flat terrain to rough terrain.

    Args:
        env: The learning environment.
        env_ids: Environment IDs (unused, kept for curriculum interface compatibility).
        p1_weights: Dictionary of reward weights from Phase 1 (Foundation Flat).
        p2_weights: Dictionary of target reward weights for Phase 2 (Robust Rough).
        curriculum_iterations: Number of training iterations to complete the transition.
    """
    # Calculate current progress based on iteration count (if available)
    current_iter = getattr(env, "common_step_counter", 0) // getattr(env, "max_episode_length", 1)

    # Initialize tracking variables on first call
    if not hasattr(env, "_reward_curriculum_initialized"):
        env._reward_curriculum_initialized = True
        env._reward_curriculum_start_iter = current_iter
        env._reward_curriculum_p1_weights = p1_weights
        env._reward_curriculum_p2_weights = p2_weights
        env._reward_curriculum_total_iters = curriculum_iterations

    # Calculate progress (0.0 = P1 weights, 1.0 = P2 weights)
    progress = min((current_iter - env._reward_curriculum_start_iter) / env._reward_curriculum_total_iters, 1.0)

    # Update reward weights based on current progress
    for attr_name, p1_weight in p1_weights.items():
        if attr_name not in p2_weights:
            continue
        p2_weight = p2_weights[attr_name]

        # Linear interpolation
        current_weight = p1_weight + (p2_weight - p1_weight) * progress

        # Update the reward weight through the public manager config API.
        if hasattr(env.reward_manager, "_term_names"):
            term_names = env.reward_manager._term_names
            if attr_name in term_names:
                reward_term_cfg = env.reward_manager.get_term_cfg(attr_name)
                reward_term_cfg.weight = current_weight
                env.reward_manager.set_term_cfg(attr_name, reward_term_cfg)

    return progress
