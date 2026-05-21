#!/usr/bin/env python3
"""
检查 Isaac Sim 导出的 Go2-X5 arm 状态与 cuRobo FK 是否对齐。

输入：
    /tmp/go2_x5_isaac_state.json
    source/robot/go2_x5/curobo/go2_x5_arm.yml

输出：
    Isaac T_base_tcp
    cuRobo FK T_base_tcp
    position error
    orientation error

用途：
    确认 Isaac Sim 中的 arm_joint1~6、arm_base_link、grasp_tcp_link
    和 cuRobo robot model 的定义一致。

为什么这一步重要：
    后续 cuRobo 轨迹规划的输入是 q_arm 和 target_tcp_pose。
    如果 Isaac FK 与 cuRobo FK 不对齐，轨迹会在 Isaac 中表现为目标偏移、
    姿态错误，甚至看起来像 planner 或 controller 出错。

运行：
    先在 Isaac Sim Script Editor 中运行：
        scripts/isaac/01_export_go2_x5_state.py

    再在普通终端运行：
        cd /home/light/workspace/arm_vla

        PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
        /data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
          scripts/dev_tools/curobo/check_isaac_curobo_fk.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch


WORKSPACE = Path("/home/light/workspace/arm_vla")
CUROBO_SOURCE_ROOT = Path("/home/light/workspace/curobo")

DEFAULT_STATE_JSON = Path("/tmp/go2_x5_isaac_state.json")
DEFAULT_ROBOT_YAML = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.yml"

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

DEFAULT_POSITION_TOLERANCE_M = 0.02
DEFAULT_ORIENTATION_TOLERANCE_DEG = 5.0


if CUROBO_SOURCE_ROOT.exists():
    sys.path.insert(0, str(CUROBO_SOURCE_ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState

from scripts.math.SE3 import normalize_quat_wxyz, quat_angle_error_deg


def print_header(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_isaac_state(path: Path) -> dict:
    """读取 Isaac Sim 导出的 JSON。"""
    if not path.exists():
        raise FileNotFoundError(
            f"Isaac state JSON 不存在: {path}\n"
            "请先在 Isaac Sim Script Editor 中运行 scripts/isaac/01_export_go2_x5_state.py"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def check_isaac_json_schema(data: dict) -> None:
    """检查 JSON 是否包含 FK 对齐所需字段。"""
    print_header("1. 检查 Isaac state JSON")

    required_top_keys = ["planner_convention", "isaac_state", "poses"]
    missing = [key for key in required_top_keys if key not in data]
    if missing:
        raise RuntimeError(f"Isaac state JSON 缺少字段: {missing}")

    planner_convention = data["planner_convention"]
    active_joint_names = list(planner_convention["active_joint_names"])
    base_link = planner_convention["base_link"]
    tool_frame = planner_convention["tool_frame"]

    print("json base_link:", base_link)
    print("json tool_frame:", tool_frame)
    print("json active_joint_names:", active_joint_names)
    print("json created_at:", data.get("created_at"))
    print("json paths:", data.get("paths", {}))

    if base_link != EXPECTED_BASE_LINK:
        raise RuntimeError(f"JSON base_link 不一致: {base_link} != {EXPECTED_BASE_LINK}")
    if tool_frame != EXPECTED_TOOL_FRAME:
        raise RuntimeError(f"JSON tool_frame 不一致: {tool_frame} != {EXPECTED_TOOL_FRAME}")
    if active_joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"JSON active_joint_names 不一致: {active_joint_names}")

    q_arm = data["isaac_state"]["q_arm"]
    if len(q_arm) != len(EXPECTED_JOINT_NAMES):
        raise RuntimeError(f"q_arm 长度不正确: {len(q_arm)}")

    print("q_arm:", np.array2string(np.asarray(q_arm, dtype=float), precision=8))
    print("T_base_tcp position:", data["poses"]["base_tcp"]["position_xyz"])
    print("T_base_tcp quat_wxyz:", data["poses"]["base_tcp"]["quaternion_wxyz"])
    print("Isaac JSON 检查通过")


def create_planner(robot_yaml: Path) -> MotionPlanner:
    """创建 cuRobo MotionPlanner。"""
    print_header("2. 创建 cuRobo MotionPlanner")

    if not robot_yaml.exists():
        raise FileNotFoundError(f"cuRobo robot YAML 不存在: {robot_yaml}")

    print("robot_yaml:", robot_yaml)
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，cuRobo FK 无法运行。")

    print("cuda device:", torch.cuda.get_device_name(0))

    cfg = MotionPlannerCfg.create(
        robot=str(robot_yaml),
        scene_model=None,
        self_collision_check=True,
        use_cuda_graph=False,
        num_ik_seeds=4,
        num_trajopt_seeds=1,
    )
    planner = MotionPlanner(cfg)
    print("MotionPlanner 创建成功")
    return planner


def check_planner_metadata(planner: MotionPlanner) -> None:
    """检查 cuRobo 的 joint_names / tool_frames 是否和 Isaac JSON 约定一致。"""
    joint_names = list(planner.joint_names)
    tool_frames = list(planner.tool_frames)

    print("cuRobo joint_names:", joint_names)
    print("cuRobo tool_frames:", tool_frames)

    if joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"cuRobo joint_names 不一致: {joint_names}")

    if EXPECTED_TOOL_FRAME not in tool_frames:
        raise RuntimeError(f"cuRobo tool_frames 中缺少 {EXPECTED_TOOL_FRAME}: {tool_frames}")


def run_curobo_fk(planner: MotionPlanner, q_arm) -> tuple[np.ndarray, np.ndarray]:
    """用 q_arm 计算 cuRobo FK，返回 tool frame 在 base link 下的位置和姿态。"""
    q_tensor = torch.tensor(
        q_arm,
        device="cuda:0",
        dtype=torch.float32,
    ).unsqueeze(0)

    joint_state = JointState.from_position(
        position=q_tensor,
        joint_names=list(planner.joint_names),
    )

    kin_state = planner.compute_kinematics(joint_state)
    tool_pose = kin_state.tool_poses.get_link_pose(
        EXPECTED_TOOL_FRAME,
        make_contiguous=True,
    )

    position = tool_pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
    quaternion = tool_pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
    return position, normalize_quat_wxyz(quaternion)


def compare_fk(
    isaac_position,
    isaac_quaternion,
    curobo_position,
    curobo_quaternion,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
) -> bool:
    """打印并判断 FK 对齐误差。"""
    print_header("3. FK 对齐结果")

    isaac_position = np.asarray(isaac_position, dtype=float)
    isaac_quaternion = normalize_quat_wxyz(isaac_quaternion)
    curobo_position = np.asarray(curobo_position, dtype=float)
    curobo_quaternion = normalize_quat_wxyz(curobo_quaternion)

    position_error = float(np.linalg.norm(curobo_position - isaac_position))
    orientation_error_deg = quat_angle_error_deg(curobo_quaternion, isaac_quaternion)

    print("Isaac position:", isaac_position)
    print("cuRobo position:", curobo_position)
    print("position delta:", curobo_position - isaac_position)
    print("position error [m]:", position_error)
    print("position tolerance [m]:", position_tolerance_m)
    print()
    print("Isaac quat_wxyz:", isaac_quaternion)
    print("cuRobo quat_wxyz:", curobo_quaternion)
    print("orientation error [deg]:", orientation_error_deg)
    print("orientation tolerance [deg]:", orientation_tolerance_deg)

    passed = (
        position_error <= position_tolerance_m
        and orientation_error_deg <= orientation_tolerance_deg
    )

    if passed:
        print("\nFK 对齐通过")
    else:
        print("\nFK 对齐未通过")
        print("优先排查：base_link/tool_frame 是否一致、q_arm 顺序是否一致、URDF fixed joint 是否一致。")

    return passed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Check Isaac FK and cuRobo FK alignment for Go2-X5 arm.")
    parser.add_argument(
        "--state-json",
        type=Path,
        default=DEFAULT_STATE_JSON,
        help="Isaac Sim 导出的状态 JSON。",
    )
    parser.add_argument(
        "--robot-yaml",
        type=Path,
        default=DEFAULT_ROBOT_YAML,
        help="cuRobo Go2-X5 arm robot YAML。",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=DEFAULT_POSITION_TOLERANCE_M,
        help="位置误差通过阈值，单位 m。",
    )
    parser.add_argument(
        "--orientation-tolerance",
        type=float,
        default=DEFAULT_ORIENTATION_TOLERANCE_DEG,
        help="姿态误差通过阈值，单位 degree。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()

    print_header("Go2-X5 Isaac FK vs cuRobo FK Check")
    data = load_isaac_state(args.state_json)
    check_isaac_json_schema(data)

    q_arm = data["isaac_state"]["q_arm"]
    isaac_position = data["poses"]["base_tcp"]["position_xyz"]
    isaac_quaternion = data["poses"]["base_tcp"]["quaternion_wxyz"]

    planner = None
    try:
        planner = create_planner(args.robot_yaml)
        check_planner_metadata(planner)
        curobo_position, curobo_quaternion = run_curobo_fk(planner, q_arm)
        passed = compare_fk(
            isaac_position=isaac_position,
            isaac_quaternion=isaac_quaternion,
            curobo_position=curobo_position,
            curobo_quaternion=curobo_quaternion,
            position_tolerance_m=args.position_tolerance,
            orientation_tolerance_deg=args.orientation_tolerance,
        )
    finally:
        if planner is not None:
            planner.destroy()

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
