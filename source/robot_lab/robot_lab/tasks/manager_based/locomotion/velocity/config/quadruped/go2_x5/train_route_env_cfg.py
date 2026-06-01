# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.assets import GO2_X5_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


HEIGHT_SCAN_DIM = 187
ARM_LOCKED_DEFAULT_RANGE = [(0.0, 0.0)] * 6
ARM_ROUGH_WARMUP_RANGE = [
    (-0.10, 0.10),
    (-0.12, 0.12),
    (-0.12, 0.12),
    (-0.08, 0.08),
    (-0.08, 0.08),
    (-0.08, 0.08),
]
ARM_FLAT_UNLOCK_START_RANGE = [
    (-0.25, 0.25),
    (0.00, 0.45),
    (0.00, 0.45),
    (-0.25, 0.25),
    (-0.20, 0.20),
    (-0.20, 0.20),
]
ARM_FLAT_UNLOCK_FINAL_RANGE = [
    (-2.40, 3.00),
    (0.00, 3.00),
    (0.00, 3.00),
    (-1.45, 1.45),
    (-1.45, 1.45),
    (-1.45, 1.45),
]

FLAT_FOUNDATION_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(220.0, 220.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


def _zero_height_scan(env, sensor_cfg=None):
    buffer = getattr(env, "_go2_x5_flat_height_scan_zeros", None)
    if buffer is None or buffer.shape[0] != env.num_envs:
        buffer = torch.zeros((env.num_envs, HEIGHT_SCAN_DIM), device=env.device)
        env._go2_x5_flat_height_scan_zeros = buffer
    return buffer


@configclass
class _Go2X5LeggedBaseEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    dog_joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    arm_joint_names = [
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
    ]
    joint_names = dog_joint_names + arm_joint_names

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4096
        self.decimation = 8
        self.episode_length_s = 20.0
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation

        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # Keep a consistent 18-DoF policy/action interface across all route stages.
        # Foundation freezes the arm at the default pose by using a zero range.
        self.commands.arm_joint_pos = mdp.ArmJointPositionCommandCfg(
            asset_name="robot",
            joint_names=self.arm_joint_names,
            resampling_time_range=(6.0, 8.0),
            position_range=ARM_LOCKED_DEFAULT_RANGE,
            use_default_offset=True,
            clip_to_joint_limits=True,
            preserve_order=True,
        )
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0

        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.arm_joint_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "arm_joint_pos"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.arm_joint_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "arm_joint_pos"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
            "arm_joint.*": 0.10,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.1),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.25, 0.25),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!{self.base_link_name}$).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names

        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))

        base_body_cfg = SceneEntityCfg("robot", body_names=[self.base_link_name])
        arm_joint_cfg = SceneEntityCfg("robot", joint_names=self.arm_joint_names, preserve_order=True)
        self.rewards.arm_joint_vel_l2.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_joint_acc_l2.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_joint_torques_l2.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_action_rate_l2.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_joint_pos_limits.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_joint_deviation_l2.params["asset_cfg"].joint_names = self.arm_joint_names
        self.rewards.arm_joint_pos_tracking_l2 = RewTerm(
            func=mdp.arm_joint_pos_tracking_l2,
            weight=0.0,
            params={"command_name": "arm_joint_pos", "asset_cfg": arm_joint_cfg},
        )
        self.rewards.arm_motion_tilt_penalty = RewTerm(
            func=mdp.arm_motion_tilt_penalty,
            weight=0.0,
            params={
                "base_asset_cfg": base_body_cfg,
                "arm_asset_cfg": arm_joint_cfg,
                "tilt_clip": 1.0,
                "vel_clip": 6.0,
            },
        )
        self.rewards.arm_action_in_unstable_base = RewTerm(
            func=mdp.arm_action_in_unstable_base,
            weight=0.0,
            params={
                "arm_asset_cfg": arm_joint_cfg,
                "base_asset_cfg": base_body_cfg,
                "tilt_threshold": 0.18,
                "lin_vel_z_threshold": 0.4,
                "ang_vel_threshold": 1.5,
            },
        )
        self.rewards.arm_stable_track_bonus = RewTerm(
            func=mdp.arm_stable_track_exp,
            weight=0.0,
            params={
                "command_name": "arm_joint_pos",
                "arm_asset_cfg": arm_joint_cfg,
                "base_asset_cfg": base_body_cfg,
                "tracking_std": 0.12,
                "tilt_std": 0.18,
                "vel_z_std": 0.2,
                "command_scale": 0.15,
            },
        )

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_height_below_minimum = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.18, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_height_above_maximum = DoneTerm(
            func=mdp.root_height_above_maximum,
            params={"maximum_height": 0.65, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_lin_vel_z_above_maximum = DoneTerm(
            func=mdp.root_lin_vel_z_above_maximum,
            params={"maximum_speed": 3.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_ang_vel_xy_above_maximum = DoneTerm(
            func=mdp.root_ang_vel_xy_above_maximum,
            params={"maximum_speed": 8.0, "asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class Go2X5FoundationFlatEnvCfg(_Go2X5LeggedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_FOUNDATION_TERRAIN_CFG
        self.scene.terrain.use_terrain_origins = False
        self.scene.terrain.visual_material = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = ObsTerm(func=_zero_height_scan, clip=(-1.0, 1.0), scale=1.0)
        self.observations.critic.height_scan = ObsTerm(func=_zero_height_scan, clip=(-1.0, 1.0), scale=1.0)
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.arm_joint_pos.position_range = ARM_LOCKED_DEFAULT_RANGE
        self.commands.arm_joint_pos.resampling_time_range = (8.0, 10.0)

        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.45, 1.1)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.2)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.9, 1.1)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "scale"
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_push_robot.interval_range_s = (8.0, 14.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}

        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.03, n_max=0.03)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.02, n_max=0.02)

        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.08
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = -0.2
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.joint_torques_l2.weight = -1.5e-5
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -1.0e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -0.8
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -1.0e-4
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 1.8
        self.rewards.feet_air_time.weight = 0.15
        self.rewards.feet_air_time.params["threshold"] = 0.45
        self.rewards.feet_air_time_variance.weight = -0.5
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.15
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_slide.weight = -0.08
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_gait.weight = 0.25
        self.rewards.upward.weight = 1.0
        self.rewards.arm_joint_pos_tracking_l2.weight = -6.0
        self.rewards.arm_joint_vel_l2.weight = -0.001
        self.rewards.arm_joint_acc_l2.weight = -5.0e-7
        self.rewards.arm_joint_torques_l2.weight = -5.0e-5
        self.rewards.arm_action_rate_l2.weight = -0.005
        self.rewards.arm_joint_pos_limits.weight = -1.0
        self.rewards.arm_joint_deviation_l2.weight = -0.8
        self.rewards.arm_motion_tilt_penalty.weight = -0.1
        self.rewards.arm_action_in_unstable_base.weight = -0.02
        self.rewards.arm_stable_track_bonus.weight = 0.0

        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.7, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.7, 1.0)

        self.disable_zero_weight_rewards()


@configclass
class Go2X5ArmUnlockFlatEnvCfg(Go2X5FoundationFlatEnvCfg):
    reward_curriculum_iterations: int = 128
    arm_command_curriculum_iterations: int = 128
    reward_curriculum_enable: bool = True
    arm_command_curriculum_enable: bool = True

    def __post_init__(self):
        super().__post_init__()

        from isaaclab.managers import CurriculumTermCfg as CurrTerm

        # P4: resume from the flat foundation checkpoint and unlock arm motion
        # without changing the route-stage policy interface.
        self.scene.num_envs = 2048

        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
            "arm_joint1": 1.20,
            "arm_joint2": 1.20,
            "arm_joint3": 1.20,
            "arm_joint4": 0.80,
            "arm_joint5": 0.70,
            "arm_joint6": 0.70,
        }

        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (6.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.arm_joint_pos.position_range = ARM_FLAT_UNLOCK_START_RANGE
        self.commands.arm_joint_pos.resampling_time_range = (4.0, 6.0)

        # Keep the base command distribution fixed during arm unlock.
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self._p1_reward_weights = {
            "lin_vel_z_l2": -1.5,
            "ang_vel_xy_l2": -0.08,
            "flat_orientation_l2": -0.5,
            "base_height_l2": -0.2,
            "body_lin_acc_l2": -0.01,
            "joint_torques_l2": -1.5e-5,
            "joint_acc_l2": -1.0e-7,
            "joint_pos_limits": -2.0,
            "joint_power": -1.0e-5,
            "stand_still": -2.0,
            "joint_pos_penalty": -0.8,
            "action_rate_l2": -0.01,
            "undesired_contacts": -1.0,
            "contact_forces": -1.0e-4,
            "track_lin_vel_xy_exp": 4.0,
            "track_ang_vel_z_exp": 1.8,
            "feet_air_time": 0.15,
            "feet_air_time_variance": -0.5,
            "feet_contact_without_cmd": 0.15,
            "feet_slide": -0.08,
            "feet_gait": 0.25,
            "arm_joint_pos_tracking_l2": -6.0,
            "arm_joint_vel_l2": -0.001,
            "arm_joint_acc_l2": -5.0e-7,
            "arm_joint_torques_l2": -5.0e-5,
            "arm_action_rate_l2": -0.005,
            "arm_joint_pos_limits": -1.0,
            "arm_joint_deviation_l2": -0.8,
            "arm_motion_tilt_penalty": -0.1,
            "arm_action_in_unstable_base": -0.02,
            "arm_stable_track_bonus": 1.0e-6,
        }
        self._p4_reward_weights = {
            "lin_vel_z_l2": -1.8,
            "ang_vel_xy_l2": -0.12,
            "flat_orientation_l2": -0.60,
            "base_height_l2": -0.2,
            "body_lin_acc_l2": -0.020,
            "joint_torques_l2": -1.5e-5,
            "joint_acc_l2": -1.0e-7,
            "joint_pos_limits": -2.0,
            "joint_power": -1.0e-5,
            "stand_still": -4.0,
            "joint_pos_penalty": -1.2,
            "action_rate_l2": -0.01,
            "undesired_contacts": -1.0,
            "contact_forces": -1.0e-4,
            "track_lin_vel_xy_exp": 4.0,
            "track_ang_vel_z_exp": 2.0,
            "feet_air_time": 0.0,
            "feet_air_time_variance": 0.0,
            "feet_contact_without_cmd": 0.4,
            "feet_slide": -0.15,
            "feet_gait": 0.0,
            "arm_joint_pos_tracking_l2": -3.5,
            "arm_joint_vel_l2": -0.001,
            "arm_joint_acc_l2": -7.5e-7,
            "arm_joint_torques_l2": -6.0e-5,
            "arm_action_rate_l2": -0.012,
            "arm_joint_pos_limits": -2.0,
            "arm_joint_deviation_l2": 0.0,
            "arm_motion_tilt_penalty": -0.35,
            "arm_action_in_unstable_base": -0.12,
            # Keep this effectively disabled until the gating logic uses delta-from-default.
            "arm_stable_track_bonus": 1.0e-6,
        }

        if self.reward_curriculum_enable:
            self.curriculum.reward_weights = CurrTerm(
                func=mdp.reward_weights_curriculum,
                params={
                    "p1_weights": self._p1_reward_weights,
                    "p2_weights": self._p4_reward_weights,
                    "curriculum_iterations": self.reward_curriculum_iterations,
                },
            )
            for attr_name, p1_weight in self._p1_reward_weights.items():
                reward_term = getattr(self.rewards, attr_name, None)
                if reward_term is not None:
                    reward_term.weight = p1_weight
        else:
            self.curriculum.reward_weights = None
            for attr_name, p4_weight in self._p4_reward_weights.items():
                reward_term = getattr(self.rewards, attr_name, None)
                if reward_term is not None:
                    reward_term.weight = p4_weight

        if self.arm_command_curriculum_enable:
            self.curriculum.arm_command_range = CurrTerm(
                func=mdp.arm_joint_position_range_curriculum,
                params={
                    "command_name": "arm_joint_pos",
                    "initial_position_range": ARM_FLAT_UNLOCK_START_RANGE,
                    "final_position_range": ARM_FLAT_UNLOCK_FINAL_RANGE,
                    "curriculum_iterations": self.arm_command_curriculum_iterations,
                },
            )
        else:
            self.curriculum.arm_command_range = None
            self.commands.arm_joint_pos.position_range = ARM_FLAT_UNLOCK_FINAL_RANGE

        for reward_name in (
            "is_terminated",
            "joint_vel_l2",
            "joint_vel_limits",
            "joint_mirror",
            "feet_contact",
            "feet_stumble",
            "feet_height",
            "feet_height_body",
        ):
            reward_term = getattr(self.rewards, reward_name, None)
            if reward_term is not None:
                reward_term.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.feet_air_time.params["threshold"] = 0.45
        self.rewards.upward.weight = 1.0
        if self.rewards.arm_stable_track_bonus is not None:
            self.rewards.arm_stable_track_bonus.params["tracking_std"] = 0.20
            self.rewards.arm_stable_track_bonus.params["tilt_std"] = 0.22
            self.rewards.arm_stable_track_bonus.params["vel_z_std"] = 0.25
            self.rewards.arm_stable_track_bonus.params["command_scale"] = 0.12

        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.65, 1.00)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.55, 0.90)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.10)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.96, 1.04)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "scale"
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.98, 1.03)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
        }
        self.events.randomize_actuator_gains.mode = "interval"
        self.events.randomize_actuator_gains.interval_range_s = (3.0, 5.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.015, n_max=0.015)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.02, n_max=0.02)

        self.sim2sim_action_delay_range = (0, 0)
        self.sim2sim_action_hold_prob = 0.02
        self.sim2sim_action_noise_std = 0.003
        self.sim2sim_obs_delay_steps = 0
        delay_steps = int(self.sim2sim_obs_delay_steps)
        if delay_steps > 0:
            delayed_terms = [
                ("base_lin_vel", mdp.delayed_base_lin_vel),
                ("base_ang_vel", mdp.delayed_base_ang_vel),
                ("projected_gravity", mdp.delayed_projected_gravity),
                ("joint_pos", mdp.delayed_joint_pos_rel),
                ("joint_vel", mdp.delayed_joint_vel_rel),
                ("actions", mdp.delayed_last_action),
                ("velocity_commands", mdp.delayed_generated_commands),
                ("arm_joint_command", mdp.delayed_generated_commands),
            ]
            for term_name, func in delayed_terms:
                term = getattr(self.observations.policy, term_name, None)
                if term is None:
                    continue
                term.func = func
                if term.params is None:
                    term.params = {}
                term.params["delay_steps"] = delay_steps
            if self.observations.policy.velocity_commands is not None:
                self.observations.policy.velocity_commands.params["command_name"] = "base_velocity"
            if self.observations.policy.arm_joint_command is not None:
                self.observations.policy.arm_joint_command.params["command_name"] = "arm_joint_pos"

        self.disable_zero_weight_rewards()


@configclass
class Go2X5ArmLocomotionFlatEnvCfg(Go2X5ArmUnlockFlatEnvCfg):
    reward_curriculum_iterations: int = 1000

    def __post_init__(self):
        super().__post_init__()

        # P5: keep full-range arm control from P4, and re-introduce low-speed
        # base motion on flat terrain with 30% standing and 70% locomotion samples.
        self.scene.num_envs = 2048

        self.commands.base_velocity.rel_standing_envs = 0.30
        self.commands.base_velocity.resampling_time_range = (5.0, 7.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.15, 0.15)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.18, 0.18)
        self.commands.arm_joint_pos.position_range = ARM_FLAT_UNLOCK_FINAL_RANGE
        self.commands.arm_joint_pos.resampling_time_range = (4.0, 6.0)

        # P5 keeps a fixed 30/70 stand-vs-motion mixture and a fixed full-range
        # arm command support. We do not expand command ranges through curriculum.
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.arm_command_range = None

        p4_final_weights = dict(self._p4_reward_weights)
        p5_target_weights = dict(p4_final_weights)
        p5_target_weights.update(
            {
                "track_lin_vel_xy_exp": 4.5,
                "track_ang_vel_z_exp": 2.2,
                "lin_vel_z_l2": -1.6,
                "ang_vel_xy_l2": -0.10,
                "flat_orientation_l2": -0.50,
                "body_lin_acc_l2": -0.015,
                "stand_still": -2.0,
                "joint_pos_penalty": -0.9,
                "feet_air_time": 0.08,
                "feet_air_time_variance": -0.2,
                "feet_contact_without_cmd": 0.2,
                "feet_slide": -0.12,
                "feet_gait": 0.20,
                "arm_joint_pos_tracking_l2": -3.2,
                "arm_action_rate_l2": -0.010,
                "arm_motion_tilt_penalty": -0.30,
                "arm_action_in_unstable_base": -0.10,
            }
        )

        self._p1_reward_weights = p4_final_weights
        self._p2_reward_weights = p5_target_weights

        if self.reward_curriculum_enable and getattr(self.curriculum, "reward_weights", None) is not None:
            self.curriculum.reward_weights.params["p1_weights"] = self._p1_reward_weights
            self.curriculum.reward_weights.params["p2_weights"] = self._p2_reward_weights
            self.curriculum.reward_weights.params["curriculum_iterations"] = self.reward_curriculum_iterations
            for attr_name, p4_weight in self._p1_reward_weights.items():
                reward_term = getattr(self.rewards, attr_name, None)
                if reward_term is not None:
                    reward_term.weight = p4_weight
        else:
            for attr_name, p5_weight in self._p2_reward_weights.items():
                reward_term = getattr(self.rewards, attr_name, None)
                if reward_term is not None:
                    reward_term.weight = p5_weight

        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.feet_air_time.params["threshold"] = 0.45
        self.rewards.upward.weight = 1.0

        # Keep randomization conservative so instability primarily reflects
        # leg-arm coupling rather than external disturbances.
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.60, 1.05)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.50, 0.95)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.10)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "scale"
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.97, 1.04)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.015, 0.015),
            "y": (-0.015, 0.015),
            "z": (-0.015, 0.015),
        }
        self.events.randomize_actuator_gains.mode = "interval"
        self.events.randomize_actuator_gains.interval_range_s = (3.0, 5.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.93, 1.07)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.93, 1.07)
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.02, n_max=0.02)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.025, n_max=0.025)

        self.sim2sim_action_delay_range = (0, 1)
        self.sim2sim_action_hold_prob = 0.03
        self.sim2sim_action_noise_std = 0.004
        self.sim2sim_obs_delay_steps = 0

        self.disable_zero_weight_rewards()


@configclass
class Go2X5RobustRoughEnvCfg(_Go2X5LeggedBaseEnvCfg):
    # Reward weight curriculum settings
    reward_curriculum_iterations: int = 64  # Roughly ~2k PPO updates with the current env-step based schedule
    reward_curriculum_enable: bool = True  # Enable reward weight curriculum

    def __post_init__(self):
        super().__post_init__()

        # P2a: pure terrain transfer from flat to rough.
        # Keep command distribution and arm behavior close to P1 so the dominant
        # distribution shift comes from terrain, not from task semantics.
        self.scene.num_envs = 2048
        self.scene.terrain.max_init_terrain_level = 1

        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.arm_joint_pos.position_range = ARM_LOCKED_DEFAULT_RANGE
        self.commands.arm_joint_pos.resampling_time_range = (8.0, 10.0)

        # Base-height reward already uses terrain-relative sensing on rough terrain.
        # Disable world-frame root-height terminations during rough adaptation so
        # local terrain elevation changes do not create artificial failures.
        self.terminations.root_height_below_minimum = None
        self.terminations.root_height_above_maximum = None

        # Explicitly disable terrain levels curriculum for P2 (we use fixed level 1)
        self.curriculum.terrain_levels = None

        # Store Phase 1 (Foundation Flat) reward weights for curriculum transition
        # These will be used as starting weights, gradually transitioning to Phase 2 targets
        self._p1_reward_weights = {
            "lin_vel_z_l2": -1.5,
            "ang_vel_xy_l2": -0.08,
            "flat_orientation_l2": -0.5,
            "base_height_l2": -0.2,
            "body_lin_acc_l2": -0.01,
            "joint_torques_l2": -1.5e-5,
            "joint_acc_l2": -1.0e-7,
            "joint_pos_limits": -2.0,
            "joint_power": -1.0e-5,
            "stand_still": -2.0,
            "joint_pos_penalty": -0.8,
            "action_rate_l2": -0.01,
            "undesired_contacts": -1.0,
            "contact_forces": -1.0e-4,
            "track_lin_vel_xy_exp": 4.0,
            "track_ang_vel_z_exp": 1.8,
            "feet_air_time": 0.15,
            "feet_air_time_variance": -0.5,
            "feet_contact_without_cmd": 0.15,
            "feet_slide": -0.08,
            "feet_gait": 0.25,
            "arm_joint_pos_tracking_l2": -6.0,
            "arm_joint_vel_l2": -0.001,
            "arm_joint_acc_l2": -5.0e-7,
            "arm_joint_torques_l2": -5.0e-5,
            "arm_action_rate_l2": -0.005,
            "arm_joint_pos_limits": -1.0,
            "arm_joint_deviation_l2": -0.8,
            "arm_motion_tilt_penalty": -0.1,
            "arm_action_in_unstable_base": -0.02,
            "arm_stable_track_bonus": 1.0e-6,
        }

        # Store Phase 2 (Robust Rough) target reward weights
        self._p2_reward_weights = {
            "lin_vel_z_l2": -1.6,
            "ang_vel_xy_l2": -0.10,
            "flat_orientation_l2": -0.55,
            "base_height_l2": -0.24,
            "body_lin_acc_l2": -0.012,
            "joint_torques_l2": -1.6e-5,
            "joint_acc_l2": -1.1e-7,
            "joint_pos_limits": -2.2,
            "joint_power": -1.1e-5,
            "stand_still": -2.0,
            "joint_pos_penalty": -0.82,
            "action_rate_l2": -0.01,
            "undesired_contacts": -1.1,
            "contact_forces": -1.2e-4,
            "track_lin_vel_xy_exp": 4.0,
            "track_ang_vel_z_exp": 1.8,
            "feet_air_time": 0.13,
            "feet_air_time_variance": -0.6,
            "feet_contact_without_cmd": 0.15,
            "feet_slide": -0.12,
            "feet_gait": 0.30,
            "arm_joint_pos_tracking_l2": -6.0,
            "arm_joint_vel_l2": -0.001,
            "arm_joint_acc_l2": -5.0e-7,
            "arm_joint_torques_l2": -5.0e-5,
            "arm_action_rate_l2": -0.005,
            "arm_joint_pos_limits": -1.0,
            "arm_joint_deviation_l2": -0.8,
            "arm_motion_tilt_penalty": -0.1,
            "arm_action_in_unstable_base": -0.02,
            "arm_stable_track_bonus": 1.0e-6,
        }

        # Add reward weight curriculum term after the phase dictionaries exist.
        if self.reward_curriculum_enable:
            from isaaclab.managers import CurriculumTermCfg as CurrTerm

            self.curriculum.reward_weights = CurrTerm(
                func=mdp.reward_weights_curriculum,
                params={
                    "p1_weights": self._p1_reward_weights,
                    "p2_weights": self._p2_reward_weights,
                    "curriculum_iterations": self.reward_curriculum_iterations,
                },
            )

        # Initialize with Phase 1 weights (will be updated by curriculum during training)
        if self.reward_curriculum_enable:
            for attr_name, p1_weight in self._p1_reward_weights.items():
                if hasattr(self.rewards, attr_name):
                    getattr(self.rewards, attr_name).weight = p1_weight

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.45, 1.1)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.2)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.9, 1.1)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "scale"
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }
        self.events.randomize_actuator_gains.mode = "reset"
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.03, n_max=0.03)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.02, n_max=0.02)

        # Set reward weights based on curriculum setting
        # If curriculum is enabled, weights were already initialized to P1 values above
        # and will be gradually adjusted during training. Otherwise, use P2 values directly.
        if not self.reward_curriculum_enable:
            self.rewards.is_terminated.weight = 0.0
            self.rewards.lin_vel_z_l2.weight = -1.6
            self.rewards.ang_vel_xy_l2.weight = -0.10
            self.rewards.flat_orientation_l2.weight = -0.55
            self.rewards.base_height_l2.weight = -0.24
            self.rewards.base_height_l2.params["target_height"] = 0.33
            self.rewards.body_lin_acc_l2.weight = -0.012
            self.rewards.joint_torques_l2.weight = -1.6e-5
            self.rewards.joint_vel_l2.weight = 0.0
            self.rewards.joint_acc_l2.weight = -1.1e-7
            self.rewards.joint_pos_limits.weight = -2.2
            self.rewards.joint_vel_limits.weight = 0.0
            self.rewards.joint_power.weight = -1.1e-5
            self.rewards.stand_still.weight = -2.0
            self.rewards.joint_pos_penalty.weight = -0.82
            self.rewards.joint_mirror.weight = 0.0
            self.rewards.action_rate_l2.weight = -0.01
            self.rewards.undesired_contacts.weight = -1.1
            self.rewards.contact_forces.weight = -1.2e-4
            self.rewards.track_lin_vel_xy_exp.weight = 4.0
            self.rewards.track_ang_vel_z_exp.weight = 1.8
            self.rewards.feet_air_time.weight = 0.13
            self.rewards.feet_air_time.params["threshold"] = 0.45
            self.rewards.feet_air_time_variance.weight = -0.6
            self.rewards.feet_contact.weight = 0.0
            self.rewards.feet_contact_without_cmd.weight = 0.15
            self.rewards.feet_stumble.weight = 0.0
            self.rewards.feet_slide.weight = -0.12
            self.rewards.feet_height.weight = 0.0
            self.rewards.feet_height_body.weight = 0.0
            self.rewards.feet_height_body.params["target_height"] = -0.2
            self.rewards.feet_gait.weight = 0.30
            self.rewards.upward.weight = 1.0
            self.rewards.arm_joint_pos_tracking_l2.weight = -6.0
            self.rewards.arm_joint_vel_l2.weight = -0.001
            self.rewards.arm_joint_acc_l2.weight = -5.0e-7
            self.rewards.arm_joint_torques_l2.weight = -5.0e-5
            self.rewards.arm_action_rate_l2.weight = -0.005
            self.rewards.arm_joint_pos_limits.weight = -1.0
            self.rewards.arm_joint_deviation_l2.weight = -0.8
            self.rewards.arm_motion_tilt_penalty.weight = -0.1
            self.rewards.arm_action_in_unstable_base.weight = -0.02
            self.rewards.arm_stable_track_bonus.weight = 1.0e-6
        else:
            # For curriculum mode, set non-curriculum rewards to their final values
            self.rewards.is_terminated.weight = 0.0
            self.rewards.joint_vel_l2.weight = 0.0
            self.rewards.joint_mirror.weight = 0.0
            self.rewards.feet_contact.weight = 0.0
            self.rewards.feet_stumble.weight = 0.0
            self.rewards.feet_height.weight = 0.0
            self.rewards.feet_height_body.weight = 0.0
            self.rewards.feet_height_body.params["target_height"] = -0.2
            self.rewards.upward.weight = 1.0
            self.rewards.base_height_l2.params["target_height"] = 0.33
            self.rewards.feet_air_time.params["threshold"] = 0.45

        self.rewards.arm_stable_track_bonus.params["tracking_std"] = 0.18
        self.rewards.arm_stable_track_bonus.params["tilt_std"] = 0.22
        self.rewards.arm_stable_track_bonus.params["vel_z_std"] = 0.3
        self.rewards.arm_stable_track_bonus.params["command_scale"] = 0.08

        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)

        self.sim2sim_action_delay_range = (0, 0)
        self.sim2sim_action_hold_prob = 0.0
        self.sim2sim_action_noise_std = 0.0
        self.sim2sim_obs_delay_steps = 0
        delay_steps = int(self.sim2sim_obs_delay_steps)
        if delay_steps > 0:
            delayed_terms = [
                ("base_lin_vel", mdp.delayed_base_lin_vel),
                ("base_ang_vel", mdp.delayed_base_ang_vel),
                ("projected_gravity", mdp.delayed_projected_gravity),
                ("joint_pos", mdp.delayed_joint_pos_rel),
                ("joint_vel", mdp.delayed_joint_vel_rel),
                ("actions", mdp.delayed_last_action),
                ("velocity_commands", mdp.delayed_generated_commands),
                ("arm_joint_command", mdp.delayed_generated_commands),
            ]
            for term_name, func in delayed_terms:
                term = getattr(self.observations.policy, term_name, None)
                if term is None:
                    continue
                term.func = func
                if term.params is None:
                    term.params = {}
                term.params["delay_steps"] = delay_steps
            if self.observations.policy.velocity_commands is not None:
                self.observations.policy.velocity_commands.params["command_name"] = "base_velocity"
            if self.observations.policy.arm_joint_command is not None:
                self.observations.policy.arm_joint_command.params["command_name"] = "arm_joint_pos"

        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None

        self.disable_zero_weight_rewards()


@configclass
class Go2X5ArmWarmupRoughEnvCfg(Go2X5RobustRoughEnvCfg):
    reward_curriculum_iterations: int = 96

    def __post_init__(self):
        super().__post_init__()

        # P2b: keep the same route-stage policy interface as P2a, but unlock
        # small-amplitude arm motions on rough terrain.
        self.scene.num_envs = 2048
        self.scene.terrain.max_init_terrain_level = 1

        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.arm_joint_pos.position_range = ARM_ROUGH_WARMUP_RANGE
        self.commands.arm_joint_pos.resampling_time_range = (4.0, 6.0)

        # Widen mass domain randomization for the first arm-enabled rough stage.
        # This now includes arm_base_link as part of the non-base body set.
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "scale"
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.90, 1.10)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.03, 0.03),
        }

        p2a_final_weights = dict(self._p2_reward_weights)
        p2b_target_weights = dict(p2a_final_weights)
        p2b_target_weights.update(
            {
                "arm_joint_pos_tracking_l2": -4.5,
                "arm_joint_vel_l2": -0.0015,
                "arm_joint_acc_l2": -1.0e-6,
                "arm_joint_torques_l2": -7.5e-5,
                "arm_action_rate_l2": -0.008,
                "arm_joint_pos_limits": -1.5,
                "arm_joint_deviation_l2": -0.15,
                "arm_motion_tilt_penalty": -0.25,
                "arm_action_in_unstable_base": -0.05,
                # Keep this effectively disabled until the gating logic is revisited.
                "arm_stable_track_bonus": 1.0e-6,
            }
        )

        self._p1_reward_weights = p2a_final_weights
        self._p2_reward_weights = p2b_target_weights

        if self.reward_curriculum_enable and getattr(self.curriculum, "reward_weights", None) is not None:
            self.curriculum.reward_weights.params["p1_weights"] = self._p1_reward_weights
            self.curriculum.reward_weights.params["p2_weights"] = self._p2_reward_weights
            self.curriculum.reward_weights.params["curriculum_iterations"] = self.reward_curriculum_iterations
            for attr_name, p1_weight in self._p1_reward_weights.items():
                reward_term = getattr(self.rewards, attr_name, None)
                if reward_term is not None:
                    reward_term.weight = p1_weight
        else:
            for attr_name, p2_weight in self._p2_reward_weights.items():
                reward_term = getattr(self.rewards, attr_name, None)
                if reward_term is not None:
                    reward_term.weight = p2_weight

        self.rewards.arm_stable_track_bonus.params["tracking_std"] = 0.18
        self.rewards.arm_stable_track_bonus.params["tilt_std"] = 0.22
        self.rewards.arm_stable_track_bonus.params["vel_z_std"] = 0.3
        self.rewards.arm_stable_track_bonus.params["command_scale"] = 0.08
