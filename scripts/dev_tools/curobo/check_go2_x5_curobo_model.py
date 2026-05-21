#!/usr/bin/env python3
"""
检查 Go2-X5 arm-only cuRobo 机器人模型配置。

用途：
    本脚本是 cuRobo 前期准备检查脚本。
    它不读取 Isaac Sim，不需要打开 stage，只验证：

        source/robot/go2_x5/curobo/go2_x5_arm.yml

    是否能被 cuRobo MotionPlanner 正确加载，并且是否符合当前项目约定：

        base_link   = arm_base_link
        tool_frame  = grasp_tcp_link
        joint_names = arm_joint1 ~ arm_joint6

检查内容：
    1. YAML 文件存在且关键字段正确。
    2. MotionPlannerCfg 可以创建。
    3. MotionPlanner 可以创建。
    4. cuRobo joint_names / tool_frames 与项目约定一致。
    5. default_joint_state 可以跑 FK。
    6. robot_spheres 存在，且 arm_link1 已使用 refit 后的最终版本。

运行：
    cd /home/light/workspace/arm_vla

    PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
    /data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
      scripts/dev_tools/curobo/check_go2_x5_curobo_model.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml


WORKSPACE = Path("/home/light/workspace/arm_vla")
CUROBO_SOURCE_ROOT = Path("/home/light/workspace/curobo")

ROBOT_YAML = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.yml"

EXPECTED_BASE_LINK = "arm_base_link"
EXPECTED_TOOL_FRAME = "grasp_tcp_link"
EXPECTED_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

# 这个值只用于提示模型版本差异，不作为硬性失败条件。
# 注意：如果你重新运行官方 build_robot_model，arm_link1 可能回到 12 个小球；
# 如果你重新运行 refit-link arm_link1，可能是 4 个较大的球。
PREFERRED_ARM_LINK1_SPHERE_COUNT = 4


if CUROBO_SOURCE_ROOT.exists():
    sys.path.insert(0, str(CUROBO_SOURCE_ROOT))

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState


def print_header(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def tensor_to_list(value) -> list:
    """把 torch Tensor 转成普通 list，便于打印。"""
    return value.detach().cpu().tolist()


def load_robot_yaml(path: Path) -> dict:
    """读取 cuRobo robot YAML。"""
    if not path.exists():
        raise FileNotFoundError(f"cuRobo robot YAML 不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_yaml_fields(data: dict) -> None:
    """检查 YAML 中的关键字段。"""
    print_header("1. 检查 YAML 关键字段")

    kinematics = data.get("kinematics")
    if not isinstance(kinematics, dict):
        raise RuntimeError("YAML 中没有 kinematics 字段。")

    base_link = kinematics.get("base_link")
    tool_frames = list(kinematics.get("tool_frames", []))
    cspace = kinematics.get("cspace", {})
    joint_names = list(cspace.get("joint_names", []))
    collision_spheres = kinematics.get("collision_spheres", {})

    print("base_link:", base_link)
    print("tool_frames:", tool_frames)
    print("joint_names:", joint_names)

    if base_link != EXPECTED_BASE_LINK:
        raise RuntimeError(f"base_link 不一致: {base_link} != {EXPECTED_BASE_LINK}")

    if EXPECTED_TOOL_FRAME not in tool_frames:
        raise RuntimeError(f"tool_frames 中缺少 {EXPECTED_TOOL_FRAME}: {tool_frames}")

    if joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"joint_names 不一致: {joint_names} != {EXPECTED_JOINT_NAMES}")

    arm_link1_spheres = collision_spheres.get("arm_link1", [])
    total_spheres = sum(len(spheres) for spheres in collision_spheres.values())
    radii = [float(sphere["radius"]) for sphere in arm_link1_spheres]

    print("total collision spheres:", total_spheres)
    print("arm_link1 sphere count:", len(arm_link1_spheres))
    if radii:
        print("arm_link1 radius min:", min(radii))
        print("arm_link1 radius max:", max(radii))
        print("arm_link1 radius avg:", sum(radii) / len(radii))

    if len(arm_link1_spheres) != PREFERRED_ARM_LINK1_SPHERE_COUNT:
        print(
            "[warning] arm_link1 sphere 数量不是之前记录的 refit 版本。"
            f"当前={len(arm_link1_spheres)}, "
            f"refit 参考值={PREFERRED_ARM_LINK1_SPHERE_COUNT}。"
            "这不阻止 MotionPlanner 加载；是否需要重新 refit 取决于后续碰撞效果。"
        )

    print("YAML 关键字段检查通过")


def create_motion_planner(robot_yaml: Path) -> MotionPlanner:
    """创建 cuRobo MotionPlanner。"""
    print_header("2. 创建 MotionPlanner")
    print("robot yaml:", robot_yaml)
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。cuRobo MotionPlanner 需要 CUDA。")

    print("cuda device:", torch.cuda.get_device_name(0))

    cfg = MotionPlannerCfg.create(
        robot=str(robot_yaml),
        scene_model=None,
        self_collision_check=True,
        use_cuda_graph=False,
        num_ik_seeds=4,
        num_trajopt_seeds=1,
    )
    print("MotionPlannerCfg 创建成功")

    planner = MotionPlanner(cfg)
    print("MotionPlanner 创建成功")
    return planner


def check_planner_metadata(planner: MotionPlanner) -> None:
    """检查 MotionPlanner 暴露的 joint_names / tool_frames。"""
    print_header("3. 检查 MotionPlanner metadata")

    joint_names = list(planner.joint_names)
    tool_frames = list(planner.tool_frames)

    print("planner.joint_names:", joint_names)
    print("planner.tool_frames:", tool_frames)

    if joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"planner.joint_names 不一致: {joint_names}")

    if EXPECTED_TOOL_FRAME not in tool_frames:
        raise RuntimeError(f"planner.tool_frames 中缺少 {EXPECTED_TOOL_FRAME}: {tool_frames}")

    print("MotionPlanner metadata 检查通过")


def run_default_fk(planner: MotionPlanner) -> None:
    """用 default_joint_state 跑一次 FK。"""
    print_header("4. 运行 default_joint_state FK")

    default_state = planner.default_joint_state
    q_default = default_state.position.detach().clone()
    print("raw default_joint_state.position shape:", tuple(q_default.shape))
    print("raw default_joint_state.position:", tensor_to_list(q_default))

    if q_default.ndim == 1:
        q_default = q_default.unsqueeze(0)

    joint_state = JointState.from_position(
        position=q_default,
        joint_names=list(planner.joint_names),
    )

    kin_state = planner.compute_kinematics(joint_state)
    tool_pose = kin_state.tool_poses.get_link_pose(EXPECTED_TOOL_FRAME, make_contiguous=True)

    position = tool_pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
    quaternion = tool_pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]

    print(f"{EXPECTED_TOOL_FRAME} position in {EXPECTED_BASE_LINK} frame:")
    print(position)
    print(f"{EXPECTED_TOOL_FRAME} quaternion wxyz:")
    print(quaternion)

    if kin_state.robot_spheres is None:
        raise RuntimeError("kin_state.robot_spheres is None，collision spheres 没有生效。")

    print("robot_spheres shape:", tuple(kin_state.robot_spheres.shape))
    print("default FK 检查通过")


def main() -> None:
    """脚本入口。"""
    print_header("Go2-X5 cuRobo Robot Model Check")

    data = load_robot_yaml(ROBOT_YAML)
    check_yaml_fields(data)

    planner = None
    try:
        planner = create_motion_planner(ROBOT_YAML)
        check_planner_metadata(planner)
        run_default_fk(planner)
        print_header("检查通过")
        print("Go2-X5 arm-only cuRobo robot model 可用于后续 FK / IK / Motion Planning。")
    finally:
        if planner is not None:
            planner.destroy()


if __name__ == "__main__":
    main()
