# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ARM_VLA_ROOT = Path(os.environ.get("ARM_VLA_ROOT", Path(__file__).resolve().parents[4]))
GO2_X5_URDF = ARM_VLA_ROOT / "source/robot/go2_x5/urdf/go2_x5.urdf"

##
# Configuration
##

GO2_X5_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=str(GO2_X5_URDF),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            # 机器狗关节
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            # 机械臂关节（全零默认姿态）
            "arm_joint1": 0.0,
            "arm_joint2": 0.0,
            "arm_joint3": 0.0,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        # manipulation baseline 通过 Isaac Sim implicit drive 跟踪 ArticulationAction。
        # 这里不要用 DCMotorCfg 的速度-力矩曲线限幅机械臂，否则 arm_joint4 等腕部
        # 关节在抓取轨迹中会明显滞后，表现为 TCP 低头并扫到桌面。
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1000.0,
            damping=50.0,
            friction=0.0,
        ),
        "gripper": DCMotorCfg(
            joint_names_expr=["arm_joint[7-8]"],
            effort_limit=20.0,
            saturation_effort=20.0,
            velocity_limit=1.0,
            stiffness=1000.0,
            damping=50.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 using DC motor.
"""
