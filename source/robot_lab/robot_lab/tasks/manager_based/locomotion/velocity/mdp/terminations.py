# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def root_height_above_maximum(
    env: ManagerBasedRLEnv, maximum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is above the maximum height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height


def root_lin_vel_z_above_maximum(
    env: ManagerBasedRLEnv, maximum_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the absolute root vertical velocity is too large."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_lin_vel_b[:, 2]) > maximum_speed


def root_ang_vel_xy_above_maximum(
    env: ManagerBasedRLEnv, maximum_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the root roll/pitch angular speed is too large."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=1) > maximum_speed
