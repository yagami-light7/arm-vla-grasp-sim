# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5FoundationFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5FoundationFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmUnlock-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5ArmUnlockFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5ArmUnlockFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmLocomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5ArmLocomotionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5ArmLocomotionFlatPPORunnerCfg",
    },
)
