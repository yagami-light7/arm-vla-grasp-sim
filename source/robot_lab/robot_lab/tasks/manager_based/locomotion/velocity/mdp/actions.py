# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class ArmCommandPositionAction(ActionTerm):
    """从 command manager 读取机械臂目标，不占用 policy action 维度。"""

    cfg: "ArmCommandPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "ArmCommandPositionActionCfg", env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names,
            preserve_order=cfg.preserve_order,
        )
        if len(self._joint_ids) == 0:
            raise ValueError(f"资产 {cfg.asset_name!r} 中没有匹配 joint_names={cfg.joint_names} 的关节。")
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        if actions.numel() != 0:
            raise ValueError(
                f"{self.__class__.__name__} 只接受 0 维 policy action，实际收到 shape={tuple(actions.shape)}。"
            )

    def apply_actions(self):
        command_term = self._env.command_manager.get_term(self.cfg.command_name)
        target = command_term.command
        if target.shape[-1] != len(self._joint_ids):
            raise ValueError(
                f"命令 {self.cfg.command_name!r} 的维度 {target.shape[-1]} "
                f"与机械臂关节数 {len(self._joint_ids)} 不一致。"
            )
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        del env_ids


@configclass
class ArmCommandPositionActionCfg(ActionTermCfg):
    """机械臂命令 action term 配置；该 term 的 action_dim 为 0。"""

    class_type: type = ArmCommandPositionAction
    asset_name: str = "robot"
    joint_names: list[str] | str | None = None
    command_name: str = "arm_joint_pos"
    preserve_order: bool = True
