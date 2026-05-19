#!/usr/bin/env python3
"""
Go2-X5 抓取任务分段轨迹规划。

用途：
    读取 Isaac 导出的当前机械臂状态和目标物体抓取 JSON，
    用 cuRobo 依次规划：

        q_current -> pregrasp
        pregrasp  -> grasp
        close_gripper
        grasp     -> lift

    本脚本只生成分段轨迹，不在 Isaac Sim 中执行。

输入：
    /tmp/go2_x5_isaac_state.json
    /tmp/go2_x5_target_tcp_pose.json
    source/robot/go2_x5/curobo/go2_x5_arm.yml
    source/robot/go2_x5/curobo/go2_x5_arm.urdf

输出：
    /tmp/go2_x5_grasp_plan.json

和 4_demo_plan_to_pose.py 的关系：
    4_demo_plan_to_pose.py 是单个 TCP pose 的 smoke test。
    本脚本是抓取流程使用的分段 planner。
    后续正常抓取走本脚本；4_demo 保留用于单独调试某一个目标 pose。
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path("/home/light/workspace/arm_vla")
SCRIPTS_DIR = WORKSPACE / "scripts"
CUROBO_SOURCE_ROOT = Path("/home/light/workspace/curobo")

STATE_JSON = Path("/tmp/go2_x5_isaac_state.json")
TARGET_JSON = Path("/tmp/go2_x5_target_tcp_pose.json")
OUTPUT_JSON = Path("/tmp/go2_x5_grasp_plan.json")

ROBOT_YAML = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.yml"
ROBOT_URDF = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.urdf"

EXPECTED_TOOL_FRAME = "grasp_tcp_link"
EXPECTED_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

JOINT_LIMIT_MARGIN = 1.0e-5
TRAJECTORY_DT = 1.0 / 100.0

# 如果 cuRobo success=False，但末端 pose 误差已经很小，可以保存调试结果。
# 当前正式抓取先要求 success=True，避免把不可行轨迹送去执行。
REQUIRE_PLANNER_SUCCESS = True
POSE_ACCEPTANCE_POSITION_M = 5.0e-3
POSE_ACCEPTANCE_ORIENTATION_RAD = 5.0e-2

SEGMENT_TIMING = {
    # 从当前姿态到预抓取点，可以稍快。
    "move_to_pregrasp": {
        "min_duration": 2.0,
        "max_joint_speed": 0.75,
    },
    # approach 是靠近物体的关键阶段，速度慢一点。
    "approach_to_grasp": {
        "min_duration": 2.0,
        "max_joint_speed": 0.25,
    },
    # lift 抬起阶段也慢一点，后续夹住物体时更稳。
    "lift_object": {
        "min_duration": 1.5,
        "max_joint_speed": 0.45,
    },
}


if CUROBO_SOURCE_ROOT.exists():
    sys.path.insert(0, str(CUROBO_SOURCE_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

from SE3 import matrix_to_pose, pose_to_matrix, quat_error_deg


def print_header(title: str) -> None:
    """打印阶段标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_json(path: Path, hint: str) -> dict:
    """读取 JSON，并在文件缺失时给出下一步提示。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}。请先运行：{hint}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_isaac_state() -> dict:
    """读取 Isaac Sim 导出的机器人状态。"""
    return load_json(
        STATE_JSON,
        "Isaac Script Editor: scripts/isaac/1_dump_go2_x5_state.py",
    )


def load_grasp_target() -> dict:
    """读取由 2_generate_sim_grasp_target.py 生成的抓取目标。"""
    data = load_json(
        TARGET_JSON,
        "Isaac Script Editor: scripts/isaac/2_generate_sim_grasp_target.py",
    )

    if "poses" not in data:
        raise RuntimeError(
            f"{TARGET_JSON} 还是旧格式，缺少 poses 字段。"
            "请重新运行 scripts/isaac/2_generate_sim_grasp_target.py。"
        )

    required = ["pregrasp", "grasp", "lift"]
    missing = [name for name in required if name not in data["poses"]]
    if missing:
        raise RuntimeError(f"{TARGET_JSON} 缺少抓取分段目标: {missing}")

    return data


def get_start_q_from_isaac_state(data: dict) -> np.ndarray:
    """读取 q_arm，并检查 joint order 是否和 cuRobo 一致。"""
    joint_names = list(data["planner_convention"]["active_joint_names"])
    if joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"Isaac active_joint_names 不一致: {joint_names}")

    return np.asarray(data["isaac_state"]["q_arm"], dtype=np.float32)


def load_joint_limits_from_urdf(urdf_path: Path) -> dict[str, tuple[float, float]]:
    """从 arm-only URDF 读取 arm_joint1~6 的 joint limit。"""
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
            raise RuntimeError(f"{name} 缺少 <limit> 字段。")

        limits[name] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )

    missing = [name for name in EXPECTED_JOINT_NAMES if name not in limits]
    if missing:
        raise RuntimeError(f"URDF 中缺少关节限位: {missing}")

    return limits


def clip_q_to_joint_limits(q_arm: np.ndarray, joint_limits: dict[str, tuple[float, float]]) -> np.ndarray:
    """把 Isaac 数值积分导致的极小越界裁剪回 joint limit 内部。"""
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


def get_named_target_pose(target_data: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    从 target JSON 中读取某个分段目标 pose。

    当前只支持 arm_base_link frame，因为 cuRobo arm-only model 的 base_link
    就是 arm_base_link。
    """
    entry = target_data["poses"][name]
    frame = entry.get("frame")
    if frame != "arm_base_link":
        raise RuntimeError(f"{name} frame 必须是 arm_base_link，当前是 {frame}")

    position = np.asarray(entry["position_xyz"], dtype=np.float32)
    quaternion = np.asarray(entry["quaternion_wxyz"], dtype=np.float32)
    return position, quaternion


def create_planner() -> MotionPlanner:
    """创建 cuRobo MotionPlanner。"""
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，cuRobo MotionPlanner 无法运行。")

    cfg = MotionPlannerCfg.create(
        robot=str(ROBOT_YAML),
        scene_model=None,
        self_collision_check=True,
        use_cuda_graph=False,
        num_ik_seeds=16,
        num_trajopt_seeds=4,
    )

    planner = MotionPlanner(cfg)
    print("planner.joint_names:", list(planner.joint_names))
    print("planner.tool_frames:", list(planner.tool_frames))

    if list(planner.joint_names) != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"cuRobo joint_names 不一致: {list(planner.joint_names)}")
    if EXPECTED_TOOL_FRAME not in list(planner.tool_frames):
        raise RuntimeError(f"cuRobo tool_frames 缺少 {EXPECTED_TOOL_FRAME}")

    planner.warmup(enable_graph=False, num_warmup_iterations=2)
    return planner


def make_joint_state(q_arm: np.ndarray, planner: MotionPlanner) -> JointState:
    """把 numpy q_arm 转成 cuRobo JointState。"""
    q_tensor = torch.tensor(
        np.asarray(q_arm, dtype=np.float32),
        device="cuda:0",
        dtype=torch.float32,
    ).unsqueeze(0)

    return JointState.from_position(
        position=q_tensor,
        joint_names=list(planner.joint_names),
    )


def make_goal_tool_pose(
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    planner: MotionPlanner,
) -> GoalToolPose:
    """把目标 TCP pose 转成 cuRobo plan_pose 需要的 GoalToolPose。"""
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


def run_fk(planner: MotionPlanner, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用 cuRobo FK 计算 grasp_tcp_link 在 arm_base_link 下的 pose。"""
    joint_state = make_joint_state(q_arm, planner)
    kin_state = planner.compute_kinematics(joint_state)
    tool_pose = kin_state.tool_poses.get_link_pose(EXPECTED_TOOL_FRAME, make_contiguous=True)

    position = tool_pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
    quaternion = tool_pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
    return position, quaternion


def smoothstep5(u: np.ndarray) -> np.ndarray:
    """五次 S 曲线，用于对路径做平滑时间参数化。"""
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def compute_path_coordinate(q_path: np.ndarray) -> np.ndarray:
    """计算 joint-space 路径累计长度，并归一化到 [0, 1]。"""
    q_path = np.asarray(q_path, dtype=float)
    step_dist = np.linalg.norm(np.diff(q_path, axis=0), axis=1)
    path_s = np.concatenate([[0.0], np.cumsum(step_dist)])

    if path_s[-1] < 1.0e-12:
        return np.linspace(0.0, 1.0, len(q_path))

    return path_s / path_s[-1]


def estimate_duration(q_path: np.ndarray, segment_name: str) -> float:
    """根据当前分段配置估计轨迹执行时长。"""
    cfg = SEGMENT_TIMING[segment_name]
    q_delta = float(np.max(np.abs(np.asarray(q_path[-1]) - np.asarray(q_path[0]))))
    duration = q_delta / float(cfg["max_joint_speed"])
    return max(float(cfg["min_duration"]), duration)


def retime_joint_path_scurve(
    q_path: np.ndarray,
    segment_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对一个分段 joint path 做 S 曲线时间参数化。"""
    q_path = np.asarray(q_path, dtype=float)
    path_s = compute_path_coordinate(q_path)
    duration = estimate_duration(q_path, segment_name)

    num_steps = int(np.ceil(duration / TRAJECTORY_DT)) + 1
    time_from_start = np.linspace(0.0, duration, num_steps)

    u = time_from_start / duration
    s_query = smoothstep5(u)

    q = np.zeros((num_steps, q_path.shape[1]), dtype=float)
    for joint_index in range(q_path.shape[1]):
        q[:, joint_index] = np.interp(s_query, path_s, q_path[:, joint_index])

    qd = np.gradient(q, time_from_start, axis=0)
    qdd = np.gradient(qd, time_from_start, axis=0)

    return time_from_start, q, qd, qdd


def compute_tcp_path(planner: MotionPlanner, q_traj: np.ndarray, T_world_base: np.ndarray) -> dict:
    """计算一个分段中每个 waypoint 的 TCP pose。"""
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


def plan_pose_path(
    planner: MotionPlanner,
    q_start: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    segment_name: str,
) -> tuple[np.ndarray, dict]:
    """调用 cuRobo plan_pose，返回原始 joint path 和 planner 诊断信息。"""
    print_header(f"Plan Segment: {segment_name}")
    print("q_start:", np.array2string(np.asarray(q_start), precision=6))
    print("target_position:", target_position)
    print("target_quaternion:", target_quaternion)

    current_state = make_joint_state(q_start, planner)
    goal_pose = make_goal_tool_pose(target_position, target_quaternion, planner)

    result = planner.plan_pose(
        goal_tool_poses=goal_pose,
        current_state=current_state,
        max_attempts=5,
        enable_graph_attempt=99,
    )

    if result is None:
        raise RuntimeError(f"{segment_name}: cuRobo plan_pose 返回 None。")

    success = bool(torch.count_nonzero(result.success).item() > 0)
    position_error = float(result.position_error.detach().min().item())
    rotation_error = float(result.rotation_error.detach().min().item())

    print("planner_success:", success)
    print("planner_position_error_m:", position_error)
    print("planner_rotation_error_rad:", rotation_error)
    print("planner_success_tensor:", result.success)

    if getattr(result, "seed_cost", None) is not None:
        print("planner_seed_cost:", result.seed_cost.detach().cpu().numpy())

    pose_converged = (
        position_error <= POSE_ACCEPTANCE_POSITION_M
        and rotation_error <= POSE_ACCEPTANCE_ORIENTATION_RAD
    )

    if REQUIRE_PLANNER_SUCCESS and not success:
        raise RuntimeError(
            f"{segment_name}: cuRobo planner_success=False。"
            f"position_error={position_error:.6g} m, "
            f"rotation_error={rotation_error:.6g} rad。"
        )

    if not success and not pose_converged:
        raise RuntimeError(
            f"{segment_name}: 规划未成功，且末端误差未收敛。"
            f"position_error={position_error:.6g} m, "
            f"rotation_error={rotation_error:.6g} rad。"
        )

    trajectory = result.get_interpolated_plan()
    q_path = trajectory.position.detach().cpu().numpy()
    if q_path.ndim < 2:
        raise RuntimeError(f"{segment_name}: trajectory.position shape 异常: {q_path.shape}")
    q_path = q_path.reshape(-1, q_path.shape[-1])

    plan_info = {
        "planner_success": success,
        "planner_position_error_m": position_error,
        "planner_rotation_error_rad": rotation_error,
        "pose_converged": pose_converged,
        "raw_num_waypoints": int(q_path.shape[0]),
    }

    return q_path, plan_info


def build_motion_segment(
    planner: MotionPlanner,
    q_start: np.ndarray,
    target_name: str,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    segment_name: str,
    T_world_base: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """规划并封装一个 motion segment。"""
    q_path_raw, plan_info = plan_pose_path(
        planner=planner,
        q_start=q_start,
        target_position=target_position,
        target_quaternion=target_quaternion,
        segment_name=segment_name,
    )

    time_from_start, q_traj, qd_traj, qdd_traj = retime_joint_path_scurve(
        q_path_raw,
        segment_name=segment_name,
    )

    tcp_path = compute_tcp_path(planner, q_traj, T_world_base)
    final_pos, final_quat = run_fk(planner, q_traj[-1])

    final_position_error = float(np.linalg.norm(final_pos - target_position))
    final_orientation_error_deg = quat_error_deg(final_quat, target_quaternion)

    print("retimed trajectory shape:", q_traj.shape)
    print("duration_s:", float(time_from_start[-1]))
    print("final_position_error_m:", final_position_error)
    print("final_orientation_error_deg:", final_orientation_error_deg)

    segment = {
        "name": segment_name,
        "type": "motion",
        "target_name": target_name,
        "target_pose_base": {
            "position_xyz": target_position.tolist(),
            "quaternion_wxyz": target_quaternion.tolist(),
        },
        "plan_info": plan_info,
        "timing": {
            "dt": float(TRAJECTORY_DT),
            "duration_s": float(time_from_start[-1]),
            "num_waypoints": int(q_traj.shape[0]),
        },
        "final_error": {
            "position_m": final_position_error,
            "orientation_deg": final_orientation_error_deg,
        },
        "trajectory": {
            "time_from_start": time_from_start.tolist(),
            "q": q_traj.tolist(),
            "qd": qd_traj.tolist(),
            "qdd": qdd_traj.tolist(),
            **tcp_path,
        },
    }

    return segment, q_traj[-1].astype(np.float32)


def make_gripper_segment(name: str, q_target: float, gripper_joint_names: list[str]) -> dict:
    """创建一个夹爪动作 segment。轨迹执行脚本会真正发送该动作。"""
    return {
        "name": name,
        "type": "gripper",
        "joint_names": gripper_joint_names,
        "target_position": [float(q_target) for _ in gripper_joint_names],
    }


def main() -> None:
    print("========== Go2-X5 Grasp Segments Planning ==========")

    isaac_state = load_isaac_state()    # 读取isaac中机器人状态
    target_data = load_grasp_target()   # 读取目标夹爪位姿

    print("[input] state_json:", STATE_JSON)
    print("[input] target_json:", TARGET_JSON)
    print("[target] object:", target_data.get("source", {}).get("object_prim_path"))
    print("[target] sequence:", target_data.get("sequence"))

    q_start = get_start_q_from_isaac_state(isaac_state) # 从isaac状态中读取当前机械臂关节位置，并检查顺序是否正确
    joint_limits = load_joint_limits_from_urdf(ROBOT_URDF)  # 从 URDF 读取关节限位
    q_current = clip_q_to_joint_limits(q_start, joint_limits)   # 把 Isaac 的 q_current 裁剪回 joint limit
    T_world_base = np.asarray(isaac_state["poses"]["world_base"]["matrix_4x4"], dtype=float) # 获取 world_base 的变换矩阵 

    pregrasp_pos, pregrasp_quat = get_named_target_pose(target_data, "pregrasp")
    grasp_pos, grasp_quat = get_named_target_pose(target_data, "grasp")
    lift_pos, lift_quat = get_named_target_pose(target_data, "lift")

    gripper_info = target_data.get("gripper", {})
    gripper_joint_names = list(gripper_info.get("joint_names", ["arm_joint7", "arm_joint8"]))
    gripper_open = float(gripper_info.get("open_m", 0.04))
    gripper_close = float(gripper_info.get("close_m", 0.0))

    planner = None
    segments = []

    try:
        planner = create_planner()

        segments.append(make_gripper_segment("open_gripper", gripper_open, gripper_joint_names))

        segment, q_current = build_motion_segment(
            planner=planner,
            q_start=q_current,
            target_name="pregrasp",
            target_position=pregrasp_pos,
            target_quaternion=pregrasp_quat,
            segment_name="move_to_pregrasp",
            T_world_base=T_world_base,
        )
        segments.append(segment)

        segment, q_current = build_motion_segment(
            planner=planner,
            q_start=q_current,
            target_name="grasp",
            target_position=grasp_pos,
            target_quaternion=grasp_quat,
            segment_name="approach_to_grasp",
            T_world_base=T_world_base,
        )
        segments.append(segment)

        segments.append(make_gripper_segment("close_gripper", gripper_close, gripper_joint_names))

        segment, q_current = build_motion_segment(
            planner=planner,
            q_start=q_current,
            target_name="lift",
            target_position=lift_pos,
            target_quaternion=lift_quat,
            segment_name="lift_object",
            T_world_base=T_world_base,
        )
        segments.append(segment)

    finally:
        if planner is not None:
            planner.destroy()

    motion_segments = [segment for segment in segments if segment["type"] == "motion"]
    total_motion_duration = sum(segment["timing"]["duration_s"] for segment in motion_segments)
    all_success = all(segment["plan_info"]["planner_success"] for segment in motion_segments)

    payload = {
        "schema_version": 1,
        "robot_name": "go2_x5",
        "planner": "curobo.MotionPlanner.plan_pose segmented grasp",
        "source_state_json": str(STATE_JSON),
        "source_target_json": str(TARGET_JSON),
        "joint_names": EXPECTED_JOINT_NAMES,
        "tool_frame": EXPECTED_TOOL_FRAME,
        "object_prim_path": target_data.get("source", {}).get("object_prim_path"),
        "segments": segments,
        "summary": {
            "num_segments": len(segments),
            "num_motion_segments": len(motion_segments),
            "all_motion_segments_success": all_success,
            "total_motion_duration_s": float(total_motion_duration),
            "final_q_arm": q_current.tolist(),
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_header("Planning Summary")
    print("output:", OUTPUT_JSON)
    print("all_motion_segments_success:", all_success)
    print("num_segments:", len(segments))
    print("total_motion_duration_s:", total_motion_duration)
    print("final_q_arm:", q_current)
    print("========== grasp segment planning complete ==========")


if __name__ == "__main__":
    main()
