# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
import isaaclab.envs.mdp as core_mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def _delay_signal(env: ManagerBasedEnv, key: str, value: torch.Tensor, delay_steps: int) -> torch.Tensor:
    if delay_steps <= 0:
        return value
    state = getattr(env, "_sim2sim_delay_state", None)
    if state is None:
        state = {}
        setattr(env, "_sim2sim_delay_state", state)

    buffer_size = delay_steps + 1
    entry = state.get(key, None)
    if (
        entry is None
        or entry["buffer"].shape[0] != buffer_size
        or entry["buffer"].shape[1:] != value.shape
    ):
        buffer = torch.zeros((buffer_size,) + value.shape, device=value.device, dtype=value.dtype)
        buffer[:] = value
        entry = {"buffer": buffer, "idx": 0, "last_step": None}
        state[key] = entry

    step_counter = getattr(env, "common_step_counter", None)
    if step_counter is None:
        if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
            step_counter = int(env.episode_length_buf.max().item())
        else:
            step_counter = 0

    if entry["last_step"] != step_counter:
        write_idx = entry["idx"]
        entry["buffer"][write_idx] = value
        if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
            reset_ids = torch.where(env.episode_length_buf == 0)[0]
            if reset_ids.numel() > 0:
                entry["buffer"][:, reset_ids] = value[reset_ids].unsqueeze(0)
        entry["idx"] = (write_idx + 1) % buffer_size
        entry["last_step"] = step_counter

    read_idx = (entry["idx"] - 1 - delay_steps) % buffer_size
    return entry["buffer"][read_idx]


def delayed_base_lin_vel(env: ManagerBasedEnv, delay_steps: int = 1) -> torch.Tensor:
    value = core_mdp.base_lin_vel(env)
    return _delay_signal(env, "base_lin_vel", value, delay_steps)


def delayed_base_ang_vel(env: ManagerBasedEnv, delay_steps: int = 1) -> torch.Tensor:
    value = core_mdp.base_ang_vel(env)
    return _delay_signal(env, "base_ang_vel", value, delay_steps)


def delayed_projected_gravity(env: ManagerBasedEnv, delay_steps: int = 1) -> torch.Tensor:
    value = core_mdp.projected_gravity(env)
    return _delay_signal(env, "projected_gravity", value, delay_steps)


def delayed_joint_pos_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), delay_steps: int = 1
) -> torch.Tensor:
    value = core_mdp.joint_pos_rel(env, asset_cfg=asset_cfg)
    return _delay_signal(env, "joint_pos_rel", value, delay_steps)


def delayed_joint_vel_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), delay_steps: int = 1
) -> torch.Tensor:
    value = core_mdp.joint_vel_rel(env, asset_cfg=asset_cfg)
    return _delay_signal(env, "joint_vel_rel", value, delay_steps)


def delayed_generated_commands(
    env: ManagerBasedEnv, command_name: str, delay_steps: int = 1
) -> torch.Tensor:
    value = core_mdp.generated_commands(env, command_name=command_name)
    return _delay_signal(env, f"command_{command_name}", value, delay_steps)


def delayed_last_action(env: ManagerBasedEnv, delay_steps: int = 1) -> torch.Tensor:
    value = core_mdp.last_action(env)
    return _delay_signal(env, "last_action", value, delay_steps)


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor
