#!/usr/bin/env python3
"""
Go2-X5 arm 单个 TCP pose 轨迹规划 demo。

输入：
    /tmp/go2_x5_isaac_state.json
    source/robot/go2_x5/curobo/go2_x5_arm.yml

输出：
    /tmp/go2_x5_arm_plan_to_pose.json

功能：
    从 Isaac 导出的 q_arm 开始，规划到一个手动指定的 target TCP pose。
    这里只生成轨迹，不在 Isaac Sim 中执行。
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

WORKSPACE = Path("/home/light/workspace/arm_vla")
CUROBO_SOURCE_ROOT = Path("/home/light/workspace/curobo")

STATE_JSON = Path("/tmp/go2_x5_isaac_state.json")
ROBOT_YAML = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.yml"
ROBOT_URDF = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.urdf"
OUTPUT_JSON = Path("/tmp/go2_x5_arm_plan_to_pose.json")

EXPECTED_TOOL_FRAME = "arm_eef_link"
EXPECTED_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

# Isaac 物理仿真会产生 1e-4 rad 量级的数值抖动。
# 如果某个关节刚好在下限 0 附近，可能导出一个很小的负数。
# cuRobo 对 joint limit 很严格，所以进入 planner 前做一个小幅裁剪。
JOINT_LIMIT_MARGIN = 1.0e-5

# 第一版 demo 的目标是先拿到一条末端位姿收敛的轨迹。
# 如果 cuRobo 因 collision spheres / self collision matrix 判定 feasibility false，
# 但末端误差已经很小，脚本会保存调试轨迹，并明确标记 planner_success=False。
POSE_ACCEPTANCE_POSITION_M = 5.0e-3
POSE_ACCEPTANCE_ORIENTATION_RAD = 5.0e-2

if CUROBO_SOURCE_ROOT.exists():
    sys.path.insert(0, str(CUROBO_SOURCE_ROOT))

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState, Pose, GoalToolPose


# 加载isaac写入的json文件
def load_isaac_state(path:Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先在 Isaac Sim Script Editor 运行 "
            "scripts/isaac/1_dump_go2_x5_state.py"
        )
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    

# 从json文件获取start state 
def get_start_state_from_json(data:dict):
    joint_names = data["planner_convention"]["active_joint_names"]
    q_arm = np.asarray(data["isaac_state"]["q_arm"], dtype=np.float32)

    tcp_position = np.asarray(data["poses"]["base_tcp"]["position_xyz"], dtype=np.float32)
    tcp_quat = np.asarray(data["poses"]["base_tcp"]["quaternion_wxyz"], dtype=np.float32)

    if joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"joint_names 不一致: {joint_names}")

    return q_arm, tcp_position, tcp_quat


def load_joint_limits_from_urdf(urdf_path: Path) -> dict[str, tuple[float, float]]:
    """从 arm-only URDF 读取 active joints 的上下限。"""
    if not urdf_path.exists():
        raise FileNotFoundError(f"找不到 arm-only URDF: {urdf_path}")

    root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in EXPECTED_JOINT_NAMES:
            continue

        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"{name} 没有 <limit> 字段")

        limits[name] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )

    missing = [name for name in EXPECTED_JOINT_NAMES if name not in limits]
    if missing:
        raise RuntimeError(f"URDF 中缺少关节限位: {missing}")

    return limits


def clip_q_to_joint_limits(
    q_arm: np.ndarray,
    joint_limits: dict[str, tuple[float, float]],
) -> np.ndarray:
    """
    把 q_arm 裁剪到 URDF joint limit 内部。

    注意：
        这里只处理 Isaac 数值积分导致的极小越界。
        如果某个关节越界很大，应该回到 Isaac 中检查机器人状态，而不是强行裁剪。
    """
    q_clipped = np.asarray(q_arm, dtype=np.float32).copy()

    for index, joint_name in enumerate(EXPECTED_JOINT_NAMES):
        lower, upper = joint_limits[joint_name]
        safe_lower = lower + JOINT_LIMIT_MARGIN
        safe_upper = upper - JOINT_LIMIT_MARGIN
        before = float(q_clipped[index])
        q_clipped[index] = np.clip(q_clipped[index], safe_lower, safe_upper)
        after = float(q_clipped[index])

        if abs(after - before) > 0.0:
            print(
                "[joint-limit] "
                f"{joint_name}: {before:.9f} -> {after:.9f} "
                f"limit=[{lower:.6f}, {upper:.6f}]"
            )

    return q_clipped


# 创建规划器planner
def create_planner() -> MotionPlanner:
    print("torch.cuda.is_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用,cuRobo Planner无法运行")
    
    cfg = MotionPlannerCfg.create(
        robot = str(ROBOT_YAML),
        scene_model = None,
        self_collision_check = True,
        use_cuda_graph = False,
        num_ik_seeds = 16,
        num_trajopt_seeds = 4,
    )

    planner = MotionPlanner(cfg)
    print("planner.joint_names:", list(planner.joint_names))
    print("planner.tool_frames:", list(planner.tool_frames))

    planner.warmup(enable_graph=False, num_warmup_iterations=2)
    return planner


# 构造current JointState
def make_joint_state(q_arm: np.ndarray, planner: MotionPlanner) -> JointState:
    q_tensor = torch.tensor(
        q_arm,
        device="cuda:0",
        dtype=torch.float32,
    ).unsqueeze(0)

    return JointState.from_position(
        position=q_tensor,
        joint_names=list(planner.joint_names),
    )
    

# 构造目标TCP Pose
# target_tcp_position: 当前 TCP + [0.03, 0.00, 0.03]
# target_tcp_quat_wxyz: 与当前 TCP 姿态相同
def make_target_pose(
        current_position: np.ndarray,
        current_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # 单位：m。第一版目标要保守，否则 IK / trajopt 容易失败。
    # 不要设置成 1e-4 m 这种“几乎原地”的目标；
    # 对当前 cuRobo IK/TrajOpt 阈值来说，过近目标反而容易被判定不可行。
    offset = np.array([0.1, 0, 0.3], dtype=np.float32)

    target_position = current_position + offset
    target_quaternion = current_quaternion.copy()

    return target_position, target_quaternion


# 将目标pose转化为GoalToolPose
'''
Pose:
    单个 frame 的位姿，position 是 [B, 3]，quaternion 是 [B, 4]

GoalToolPose:
    planner 的目标输入，可以包含多个 tool frame 和多个 goal。
    对单 TCP pose 来说，只放 arm_eef_link 一个 frame。
'''
def make_goal_tool_pose(
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    planner: MotionPlanner,
) -> GoalToolPose:
    position_tensor = torch.tensor(
        target_position,
        device="cuda:0",
        dtype=torch.float32,
    ).reshape(1, 3)

    quat_tensor = torch.tensor(
        target_quaternion,
        device="cuda:0",
        dtype=torch.float32,
    ).reshape(1, 4)

    pose = Pose(
        position=position_tensor,
        quaternion=quat_tensor,
    )

    return GoalToolPose.from_poses(
        pose_dict={EXPECTED_TOOL_FRAME: pose},
        ordered_tool_frames=list(planner.tool_frames),
    )


def plan_to_pose(
        planner: MotionPlanner,
        current_state:JointState,
        goal_pose:GoalToolPose
):
    result = planner.plan_pose(
        goal_tool_poses=goal_pose,
        current_state=current_state,
        max_attempts=5,
        enable_graph_attempt=99,
    )

    if result is None:
        raise RuntimeError("cuRobo plan_pose 返回 None，IK 没有找到可用目标关节解。")

    success = bool(torch.count_nonzero(result.success).item() > 0)
    position_error = float(result.position_error.detach().min().item())
    rotation_error = float(result.rotation_error.detach().min().item())

    print("planner_success:", success)
    print("planner_position_error_m:", position_error)
    print("planner_rotation_error_rad:", rotation_error)
    print("planner_success_tensor:", result.success)

    if getattr(result, "seed_cost", None) is not None:
        print("planner_seed_cost:", result.seed_cost.detach().cpu().numpy())
    
    trajectory = result.get_interpolated_plan()
    q_traj = trajectory.position.detach().cpu().numpy()

    # 当前新版 cuRobo 常见 shape 是 [batch, seed, T, dof]。
    # 本 demo 只跑一个问题，所以统一整理成 [T, 6]。
    if q_traj.ndim < 2:
        raise RuntimeError(f"trajectory.position shape 异常: {q_traj.shape}")
    q_traj = q_traj.reshape(-1, q_traj.shape[-1])

    plan_info = {
        "planner_success": success,
        "planner_position_error_m": position_error,
        "planner_rotation_error_rad": rotation_error,
        "note": None,
    }

    if not success:
        pose_converged = (
            position_error <= POSE_ACCEPTANCE_POSITION_M
            and rotation_error <= POSE_ACCEPTANCE_ORIENTATION_RAD
        )
        if not pose_converged:
            raise RuntimeError(
                "cuRobo plan_pose 未成功，且末端误差未收敛。"
                f"position_error={position_error:.6g} m, "
                f"rotation_error={rotation_error:.6g} rad"
            )

        plan_info["note"] = (
            "末端 pose 已收敛，但 cuRobo success=False。"
            "这条轨迹只用于第一版调试/可视化，尚不能声明 collision-free。"
            "后续需要继续修 self-collision spheres / ignore matrix。"
        )
        print("[warning]", plan_info["note"])

    return q_traj, plan_info


# 四元数（wxyz顺序）转旋转矩阵
def quat_wxyz_to_rotmat(quat):
    w, x, y, z = normalize_quat(quat)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=float)


# 位置向量和四元数转齐次变换矩阵
def pose_to_matrix(position, quat_wxyz):
    T = np.eye(4, dtype=float)
    T[:3, :3] = quat_wxyz_to_rotmat(quat_wxyz)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


# 旋转矩阵转四元数（wxyz顺序）
def rotmat_to_quat_wxyz(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return normalize_quat([w, x, y, z])


# 齐次变换矩阵位置向量和四元数（wxyz顺序）
def matrix_to_pose(T):
    pos = T[:3, 3].copy()
    quat = rotmat_to_quat_wxyz(T[:3, :3])
    return pos, quat


# 归一化四元数
def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


# 将四元数误差转化为角度
def quat_error_deg(q_a, q_b) -> float:
    q_a = normalize_quat(q_a)
    q_b = normalize_quat(q_b)
    dot = abs(float(np.dot(q_a, q_b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


# Forward Kinematics
def run_fk(planner: MotionPlanner, q_arm: np.ndarray):
    joint_state = make_joint_state(q_arm, planner)
    kin_state = planner.compute_kinematics(joint_state)
    tool_pose = kin_state.tool_poses.get_link_pose(EXPECTED_TOOL_FRAME, make_contiguous=True)

    pos = tool_pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
    quat = tool_pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
    return pos, quat


# 基于FK计算每一个waypoint的TCP位姿
def compute_tcp_path(planner, q_traj, T_world_base):
    tcp_pos_base = []
    tcp_quat_base = []
    tcp_pos_world = []
    tcp_quat_world = []

    for q_arm in q_traj:
        pos_b, quat_b = run_fk(planner, q_arm)
        T_base_tcp = pose_to_matrix(pos_b, quat_b)
        T_world_tcp = T_world_base @ T_base_tcp
        pos_w, quat_w = matrix_to_pose(T_world_tcp)

        tcp_pos_base.append(pos_b.tolist())
        tcp_quat_base.append(quat_b.tolist())
        tcp_pos_world.append(pos_w.tolist())
        tcp_quat_world.append(quat_w.tolist())

    return {
        "tcp_position_base": tcp_pos_base,
        "tcp_quaternion_base": tcp_quat_base,
        "tcp_position_world": tcp_pos_world,
        "tcp_quaternion_world": tcp_quat_world,
    }


# 检查最终误差
def check_final_error(planner, q_final, target_position, target_quaternion):
    final_pos, final_quat = run_fk(planner, q_final)

    pos_error = float(np.linalg.norm(final_pos - target_position))
    ori_error = quat_error_deg(final_quat, target_quaternion)

    print("final_tcp_position:", final_pos)
    print("target_tcp_position:", target_position)
    print("final_position_error_m:", pos_error)
    print("final_orientation_error_deg:", ori_error)

    return pos_error, ori_error


# 保存最终trajectory json
def save_trajectory(
    q_traj: np.ndarray,
    target_position,
    target_quaternion,
    plan_info: dict,
    tcp_path: dict,
):
    payload = {
        "schema_version": 1,
        "robot_name": "go2_x5",
        "planner": "curobo.MotionPlanner.plan_pose",
        "plan_info": plan_info,
        "joint_names": EXPECTED_JOINT_NAMES,
        "tool_frame": EXPECTED_TOOL_FRAME,
        "target_tcp_pose_base": {
            "position_xyz": target_position.tolist(),
            "quaternion_wxyz": target_quaternion.tolist(),
        },
        "trajectory": {
            "q": q_traj.tolist(),
            "num_waypoints": int(q_traj.shape[0]),
            "num_joints": int(q_traj.shape[1]),
        },
    }

    # 把每个 waypoint 对应的 TCP pose 也写入 trajectory 字段
    payload["trajectory"].update(tcp_path)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("saved trajectory:", OUTPUT_JSON)


def main():
    print("========== Go2-X5 单 TCP pose 规划 ==========")

    # 读取Isaac Sim 导出的机器人状态，获取规划起点 q_arm 和当前 TCP pose
    data = load_isaac_state(STATE_JSON)
    q_start, current_pos, current_quat = get_start_state_from_json(data)

    print("q_start:", q_start)
    print("current_tcp_position:", current_pos)
    print("current_tcp_quat_wxyz:", current_quat)

    # 对 q_start 做 joint limit 裁剪
    joint_limits = load_joint_limits_from_urdf(ROBOT_URDF)
    q_start = clip_q_to_joint_limits(q_start, joint_limits)
    print("q_start_used_by_planner:", q_start)

    # 创建轨迹规划器
    planner = create_planner()

    # 构造规划输入：planner 的 current_state 和 goal_pose
    current_state = make_joint_state(q_start, planner)

    target_pos, target_quat = make_target_pose(current_pos, current_quat)
    print("target_tcp_position:", target_pos)
    print("target_tcp_quat_wxyz:", target_quat)

    goal_pose = make_goal_tool_pose(target_pos, target_quat, planner)

    # 生成trajectory
    q_traj, plan_info = plan_to_pose(planner, current_state, goal_pose)
    print("trajectory shape:", q_traj.shape)
    print("q_start_from_traj:", q_traj[0])
    print("q_final_from_traj:", q_traj[-1])

    # 基于FK计算trajectory中每个waypoint的TCP pose， 用于Isaac Sim中可视化和后续误差分析
    T_world_base = np.asarray(data["poses"]["world_base"]["matrix_4x4"], dtype=float)
    tcp_path = compute_tcp_path(planner, q_traj, T_world_base)
    print("tcp_path waypoints:", len(tcp_path["tcp_position_world"]))

    # 检查trajectory末端的TCP pose误差，验证是否达到预期目标。
    pos_error, ori_error = check_final_error(
        planner,
        q_traj[-1],
        target_pos,
        target_quat,
    )

    # 保存轨迹
    save_trajectory(q_traj, target_pos, target_quat, plan_info, tcp_path)

    print("========== planning complete ==========")
    print("position error [m]:", pos_error)
    print("orientation error [deg]:", ori_error)


if __name__ == "__main__":
    main()
