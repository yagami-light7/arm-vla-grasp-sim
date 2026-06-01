# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2X5RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "go2_x5_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2X5FlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.save_interval = 1000
        self.experiment_name = "go2_x5_flat"


@configclass
class Go2X5FoundationFlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 24
        self.max_iterations = 8000
        self.save_interval = 500
        self.experiment_name = "go2_x5_foundation_flat"
        self.algorithm.entropy_coef = 0.01


@configclass
class Go2X5ArmUnlockFlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 5000
        self.save_interval = 250
        self.experiment_name = "go2_x5_arm_unlock_flat"
        self.algorithm.entropy_coef = 0.003
        self.algorithm.learning_rate = 5.0e-4


@configclass
class Go2X5ArmLocomotionFlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 6000
        self.save_interval = 250
        self.experiment_name = "go2_x5_arm_locomotion_flat"
        self.algorithm.entropy_coef = 0.0025
        self.algorithm.learning_rate = 2.0e-4


@configclass
class Go2X5RobustRoughPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 6000
        self.save_interval = 250
        self.experiment_name = "go2_x5_robust_rough"
        self.algorithm.entropy_coef = 0.004


@configclass
class Go2X5ArmWarmupRoughPPORunnerCfg(Go2X5RobustRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 4000
        self.save_interval = 250
        self.experiment_name = "go2_x5_arm_warmup_rough"
        self.algorithm.entropy_coef = 0.003
