#!/usr/bin/env python3
"""
生成 Go2-X5 机械臂的 cuRobo 专用 arm-only URDF。

用途：
    原始 go2_x5.urdf 是整机模型，包含 Go2 四足底盘、X5 六轴机械臂和双指夹爪。
    后续导航 + 抓取任务仍然要保留整机 URDF 给 Isaac Sim 使用。

    但第一版 cuRobo 只负责机械臂 TCP 轨迹规划：
        q_arm_current + target_tcp_pose -> arm_joint1~6 trajectory

    因此本脚本从完整 go2_x5.urdf 中派生一个 arm-only URDF：
        - base link: arm_base_link
        - active joints: arm_joint1 ~ arm_joint6
        - tool frame: grasp_tcp_link
        - gripper links: 保留几何
        - gripper joints: arm_joint7 / arm_joint8 固定到给定开合位置，不参与规划

输入：
    source/robot/go2_x5/urdf/go2_x5.urdf

输出：
    source/robot/go2_x5/curobo/go2_x5_arm.urdf

注意：
    本脚本不会修改原始整机 URDF。
"""

from __future__ import annotations

import argparse
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path


WORKSPACE = Path("/home/light/workspace/arm_vla")

DEFAULT_SOURCE_URDF = WORKSPACE / "source/robot/go2_x5/urdf/go2_x5.urdf"
DEFAULT_OUTPUT_URDF = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.urdf"

# arm-only URDF 中保留的 link。
# 夹爪 link7/link8 被保留，是为了让 cuRobo 的碰撞球能覆盖夹爪几何。
ARM_LINK_NAMES = [
    "arm_base_link",
    "arm_link1",
    "arm_link2",
    "arm_link3",
    "arm_link4",
    "arm_link5",
    "arm_link6",
    "arm_link7",
    "arm_link8",
    # 保留原始末端参考帧，便于和旧配置/Isaac 导入结果对照。
    "arm_eef_link",
    # 指尖 TCP，位于两指指尖中心，作为 cuRobo tool frame。
    "grasp_tcp_link",
]

# cuRobo 第一版只把这 6 个关节当成 active joints。
ACTIVE_ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

# 夹爪关节保留在 URDF 中，但会被转成 fixed joint。
GRIPPER_JOINT_NAMES_TO_FIX = [
    "arm_joint7",
    "arm_joint8",
]

# 固定参考帧：
#   arm_gripper_fixed_joint: 原 URDF 自带 eef frame
#   grasp_tcp_fixed_joint:  本项目新增真实抓取 TCP frame
FIXED_FRAME_JOINT_NAMES = [
    "arm_gripper_fixed_joint",
    "grasp_tcp_fixed_joint",
]

ARM_JOINT_NAMES = ACTIVE_ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES_TO_FIX + FIXED_FRAME_JOINT_NAMES


def parse_xyz_attribute(text: str | None, default: tuple[float, float, float]) -> list[float]:
    """解析 URDF 中的 xyz/rpy 属性字符串。"""
    if text is None:
        return list(default)

    values = [float(item) for item in text.split()]
    if len(values) != 3:
        raise ValueError(f"期望 xyz/rpy 属性包含 3 个数，实际为: {text}")
    return values


def format_xyz_attribute(values: list[float]) -> str:
    """把 float list 写回 URDF 属性字符串。"""
    return " ".join(f"{value:.9g}" for value in values)


def normalize_vector(values: list[float]) -> list[float]:
    """归一化 joint axis。"""
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1.0e-12:
        raise ValueError(f"joint axis 范数过小，无法归一化: {values}")
    return [value / norm for value in values]


def index_elements_by_name(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    """按 name 属性索引 URDF 中的 link 或 joint。"""
    indexed: dict[str, ET.Element] = {}

    for element in root.findall(tag):
        name = element.get("name")
        if name:
            indexed[name] = element

    return indexed


def remove_child_if_present(parent: ET.Element, child_tag: str) -> None:
    """如果 XML 子节点存在，就从 parent 中移除。"""
    child = parent.find(child_tag)
    if child is not None:
        parent.remove(child)


def convert_prismatic_gripper_joint_to_fixed(
    joint: ET.Element,
    fixed_position: float,
) -> ET.Element:
    """
    将夹爪 prismatic joint 固定在 fixed_position。

    原 prismatic joint 的 child link 位姿等价于：
        T_parent_child(q) = T_origin * translation(axis * q)

    转为 fixed joint 后，需要把 axis * q 合并进 origin.xyz。
    """
    fixed_joint = copy.deepcopy(joint)
    fixed_joint.set("type", "fixed")

    origin = fixed_joint.find("origin")
    if origin is None:
        origin = ET.SubElement(fixed_joint, "origin")

    origin_xyz = parse_xyz_attribute(origin.get("xyz"), default=(0.0, 0.0, 0.0))

    axis_element = joint.find("axis")
    axis_xyz = parse_xyz_attribute(
        axis_element.get("xyz") if axis_element is not None else None,
        default=(0.0, 0.0, 1.0),
    )
    axis_xyz = normalize_vector(axis_xyz)

    fixed_xyz = [
        origin_xyz[index] + axis_xyz[index] * fixed_position
        for index in range(3)
    ]
    origin.set("xyz", format_xyz_attribute(fixed_xyz))

    # fixed joint 不再需要这些 prismatic-only 字段。
    for child_tag in [
        "axis",
        "limit",
        "dynamics",
        "mimic",
        "safety_controller",
        "calibration",
    ]:
        remove_child_if_present(fixed_joint, child_tag)

    return fixed_joint


def validate_required_elements(
    links_by_name: dict[str, ET.Element],
    joints_by_name: dict[str, ET.Element],
) -> None:
    """确认原始 URDF 中存在我们要提取的机械臂 link/joint。"""
    missing_links = [name for name in ARM_LINK_NAMES if name not in links_by_name]
    missing_joints = [name for name in ARM_JOINT_NAMES if name not in joints_by_name]

    if missing_links or missing_joints:
        raise RuntimeError(
            "原始 Go2-X5 URDF 缺少机械臂元素："
            f"missing_links={missing_links}, missing_joints={missing_joints}"
        )


def build_arm_only_urdf(
    source_urdf_path: Path,
    output_urdf_path: Path,
    gripper_fixed_position: float,
) -> None:
    """从完整 Go2-X5 URDF 生成 arm-only URDF。"""
    if not source_urdf_path.exists():
        raise FileNotFoundError(f"原始 URDF 不存在: {source_urdf_path}")

    source_tree = ET.parse(source_urdf_path)
    source_root = source_tree.getroot()

    links_by_name = index_elements_by_name(source_root, "link")
    joints_by_name = index_elements_by_name(source_root, "joint")
    validate_required_elements(links_by_name, joints_by_name)

    arm_robot = ET.Element("robot", {"name": "go2_x5_arm_curobo"})

    for link_name in ARM_LINK_NAMES:
        arm_robot.append(copy.deepcopy(links_by_name[link_name]))

    for joint_name in ARM_JOINT_NAMES:
        source_joint = joints_by_name[joint_name]
        if joint_name in GRIPPER_JOINT_NAMES_TO_FIX:
            arm_robot.append(
                convert_prismatic_gripper_joint_to_fixed(
                    source_joint,
                    fixed_position=gripper_fixed_position,
                )
            )
        else:
            arm_robot.append(copy.deepcopy(source_joint))

    ET.indent(arm_robot, space="  ")
    output_urdf_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(arm_robot).write(
        output_urdf_path,
        encoding="utf-8",
        xml_declaration=True,
    )


def validate_generated_urdf(output_urdf_path: Path) -> None:
    """对生成的 arm-only URDF 做最小结构检查。"""
    tree = ET.parse(output_urdf_path)
    root = tree.getroot()

    links_by_name = index_elements_by_name(root, "link")
    joints_by_name = index_elements_by_name(root, "joint")
    validate_required_elements(links_by_name, joints_by_name)

    for joint_name in GRIPPER_JOINT_NAMES_TO_FIX:
        joint_type = joints_by_name[joint_name].get("type")
        if joint_type != "fixed":
            raise RuntimeError(f"{joint_name} 应该被固定为 fixed joint，实际 type={joint_type}")

    active_joint_types = {
        joint_name: joints_by_name[joint_name].get("type")
        for joint_name in ACTIVE_ARM_JOINT_NAMES
    }
    non_revolute = {
        joint_name: joint_type
        for joint_name, joint_type in active_joint_types.items()
        if joint_type != "revolute"
    }
    if non_revolute:
        raise RuntimeError(f"active arm joints 必须都是 revolute，异常项: {non_revolute}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Generate arm-only URDF for Go2-X5 cuRobo planning.",
    )
    parser.add_argument(
        "--source-urdf",
        type=Path,
        default=DEFAULT_SOURCE_URDF,
        help="原始完整 Go2-X5 URDF 路径。",
    )
    parser.add_argument(
        "--output-urdf",
        type=Path,
        default=DEFAULT_OUTPUT_URDF,
        help="输出的 arm-only URDF 路径。",
    )
    parser.add_argument(
        "--gripper-fixed-position",
        type=float,
        default=0.044,
        help="夹爪固定位置，单位 m。0.044 表示打开，0.0 表示闭合。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()

    build_arm_only_urdf(
        source_urdf_path=args.source_urdf,
        output_urdf_path=args.output_urdf,
        gripper_fixed_position=args.gripper_fixed_position,
    )
    validate_generated_urdf(args.output_urdf)

    print("Go2-X5 arm-only URDF 生成完成")
    print(f"source URDF: {args.source_urdf}")
    print(f"output URDF: {args.output_urdf}")
    print(f"base link: arm_base_link")
    print(f"active joints: {ACTIVE_ARM_JOINT_NAMES}")
    print(f"fixed gripper joints: {GRIPPER_JOINT_NAMES_TO_FIX}")
    print(f"gripper fixed position: {args.gripper_fixed_position}")
    print("tool frame: grasp_tcp_link")


if __name__ == "__main__":
    main()
