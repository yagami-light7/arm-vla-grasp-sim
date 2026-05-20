#!/usr/bin/env python3
"""
Go2-X5 抓取任务轨迹规划。

用途：
    读取 Isaac 导出的当前机械臂状态和目标物体抓取 JSON，
    用 cuRobo 先分段求解路径，再把 pregrasp 作为途经点合并成连续轨迹：

        open_gripper
        q_current -> pregrasp -> grasp   # 输出为一条连续 motion，避免 pregrasp 处切段抖动
        close_gripper
        grasp     -> lift

    本脚本只生成抓取计划，不在 Isaac Sim 中执行。

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
import time
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
TRAJECTORY_DT = 1.0 / 50

# 如果 cuRobo success=False，但末端 pose 误差已经很小，可以保存调试结果。
# 当前正式抓取先要求 success=True，避免把不可行轨迹送去执行。
REQUIRE_PLANNER_SUCCESS = True
POSE_ACCEPTANCE_POSITION_M = 5.0e-3
POSE_ACCEPTANCE_ORIENTATION_RAD = 5.0e-2

# 常驻 planner server 已经把初始化成本摊掉了，因此这里优先保证稳健性。
NUM_IK_SEEDS = 32
NUM_TRAJOPT_SEEDS = 8
PLAN_POSE_MAX_ATTEMPTS = 6
WORLD_COLLISION_ENABLED = True
WORLD_COLLISION_CACHE = {
    "primitive": 64,
    "mesh": 1,
}
WORLD_COLLISION_ACTIVATION_DISTANCE_M = 0.02

# pregrasp 是避开物体的安全点，但目标物体被放高、或者固定底座后
# arm_base_link 变低时，默认 10 cm 上方的 pregrasp 可能比 grasp 本身更难到达。
# 这里保留原始 pregrasp，然后逐步降低到 grasp 附近做自动回退。
PREGRASP_FALLBACK_OFFSETS_M = [0.05, 0.03, 0.015, 0.0]

# top-down grasp 只约束 TCP 下降轴时，绕 TCP 局部 +X 轴旋转是等价候选。
# 固定底座或物体方位变化后，某个 roll 角可能让腕部更容易到达。
TCP_ROLL_FALLBACK_DEG = [0.0, 90.0, -90.0, 180.0]

SEGMENT_TIMING = {
    # 仅用于规划日志中的子路径名称；最终不会作为独立 motion 输出。
    "move_to_pregrasp": {
        "min_duration": 1.0,
        "max_joint_speed": 1.0,
    },
    # 最终输出的连续 motion：q_current -> pregrasp -> grasp。
    # 名称继续使用 approach_to_grasp，是为了兼容执行脚本中已有的严格到位等待逻辑。
    "approach_to_grasp": {
        "min_duration": 1.0,
        "max_joint_speed": 1.0,
    },
    # lift 抬起阶段夹着物体，保持相对稳一点。
    "lift_object": {
        "min_duration": 1.0,
        "max_joint_speed": 0.80,
    },
    # 侧向抓取后不做竖直 lift，而是沿 approach_to_grasp 原路退回 chosen pregrasp。
    "retreat_object": {
        "min_duration": 1.0,
        "max_joint_speed": 0.80,
    },
}


if CUROBO_SOURCE_ROOT.exists():
    sys.path.insert(0, str(CUROBO_SOURCE_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose

from SE3 import matrix_to_pose, pose_to_matrix, quat_error_deg


SCRIPT_START_TIME = time.perf_counter()


class Profiler:
    """轻量耗时统计，用于定位 cuRobo 规划链路瓶颈。"""

    def __init__(self):
        self.records = []

    def add(self, label: str, duration_s: float) -> None:
        self.records.append((label, float(duration_s)))
        print(f"[profile] {label}: {duration_s:.3f}s")

    def summary(self) -> dict:
        return {
            label: duration_s
            for label, duration_s in self.records
        }

    def print_summary(self) -> None:
        print_header("Profiling Summary")
        for label, duration_s in self.records:
            print(f"[profile] {label:36s} {duration_s:8.3f}s")
        print(f"[profile] {'total_process_time':36s} {time.perf_counter() - SCRIPT_START_TIME:8.3f}s")


PROFILER = Profiler()


class profile_block:
    """with profile_block('name'): 统计代码块耗时。"""

    def __init__(self, label: str):
        self.label = label
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        PROFILER.add(self.label, time.perf_counter() - self.start)
        return False


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


def make_world_collision_scene(isaac_state: dict) -> Scene | None:
    """把 Isaac 导出的环境 collision cuboids 转成 cuRobo Scene。"""
    if not WORLD_COLLISION_ENABLED:
        return None

    world_collision = isaac_state.get("world_collision", {})
    cuboid_entries = world_collision.get("cuboids_base", [])
    cuboids = []

    for entry in cuboid_entries:
        pose_base = entry.get("pose_base", {})
        position = pose_base.get("position_xyz")
        quaternion = pose_base.get("quaternion_wxyz")
        dims = entry.get("dims_xyz")
        name = entry.get("name")

        if not name or position is None or quaternion is None or dims is None:
            continue

        dims = np.asarray(dims, dtype=float)
        if dims.shape != (3,) or np.any(dims <= 0.0):
            continue

        cuboids.append(
            Cuboid(
                name=str(name),
                pose=[
                    *np.asarray(position, dtype=float).tolist(),
                    *np.asarray(quaternion, dtype=float).tolist(),
                ],
                dims=dims.tolist(),
            )
        )

    if not cuboids:
        print("[world collision] no exported cuboids; planning without environment obstacles.")
        return None

    print(f"[world collision] loaded cuboids: {len(cuboids)}")
    for cuboid in cuboids[:8]:
        print(f"  - {cuboid.name}: pose={cuboid.pose}, dims={cuboid.dims}")
    if len(cuboids) > 8:
        print(f"  ... {len(cuboids) - 8} more")

    return Scene(cuboid=cuboids)


def update_planner_world(planner: MotionPlanner, world_scene: Scene | None) -> None:
    """把当前 Isaac 场景障碍物更新到 cuRobo planner。"""
    with profile_block("planner.update_world_collision"):
        try:
            planner.clear_scene_cache()
        except Exception:
            pass
        if world_scene is None:
            return
        planner.update_world(world_scene)


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

    with profile_block("MotionPlannerCfg.create"):
        cfg = MotionPlannerCfg.create(
            robot=str(ROBOT_YAML),
            scene_model=None,
            collision_cache=WORLD_COLLISION_CACHE if WORLD_COLLISION_ENABLED else None,
            self_collision_check=True,
            use_cuda_graph=False,
            num_ik_seeds=NUM_IK_SEEDS,
            num_trajopt_seeds=NUM_TRAJOPT_SEEDS,
            optimizer_collision_activation_distance=WORLD_COLLISION_ACTIVATION_DISTANCE_M,
        )

    with profile_block("MotionPlanner.__init__"):
        planner = MotionPlanner(cfg)
    print("planner.joint_names:", list(planner.joint_names))
    print("planner.tool_frames:", list(planner.tool_frames))

    if list(planner.joint_names) != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"cuRobo joint_names 不一致: {list(planner.joint_names)}")
    if EXPECTED_TOOL_FRAME not in list(planner.tool_frames):
        raise RuntimeError(f"cuRobo tool_frames 缺少 {EXPECTED_TOOL_FRAME}")

    # with profile_block("MotionPlanner.warmup"):
    #     planner.warmup(enable_graph=False, num_warmup_iterations=2)
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


def normalize_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """归一化 wxyz 四元数。"""
    q = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-9:
        raise RuntimeError(f"四元数范数异常: {quaternion}")
    return (q / norm).astype(np.float32)


def quat_multiply_wxyz(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """wxyz 四元数乘法，结果表示先 q_right 后 q_left 的组合。"""
    w1, x1, y1, z1 = normalize_quat_wxyz(q_left)
    w2, x2, y2, z2 = normalize_quat_wxyz(q_right)
    return normalize_quat_wxyz(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )
    )


def quat_roll_about_local_x(quaternion: np.ndarray, roll_deg: float) -> np.ndarray:
    """在 TCP 局部 +X 轴上追加一个 roll 角。"""
    half_angle = np.deg2rad(float(roll_deg)) * 0.5
    q_roll_local_x = np.array(
        [np.cos(half_angle), np.sin(half_angle), 0.0, 0.0],
        dtype=np.float32,
    )
    return quat_multiply_wxyz(quaternion, q_roll_local_x)


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


def remove_near_duplicate_waypoints(q_path: np.ndarray, tol: float = 1.0e-9) -> np.ndarray:
    """删除相邻重复 waypoint，避免路径累计长度中出现重复坐标。"""
    q_path = np.asarray(q_path, dtype=float)
    if q_path.shape[0] <= 2:
        return q_path

    step_dist = np.linalg.norm(np.diff(q_path, axis=0), axis=1)
    keep = np.concatenate([[True], step_dist > tol])
    keep[-1] = True
    return q_path[keep]


def compute_path_coordinate(q_path: np.ndarray) -> np.ndarray:
    """计算 joint-space 路径累计长度，并归一化到 [0, 1]。"""
    q_path = np.asarray(q_path, dtype=float)
    step_dist = np.linalg.norm(np.diff(q_path, axis=0), axis=1)
    path_s = np.concatenate([[0.0], np.cumsum(step_dist)])

    if path_s[-1] < 1.0e-12:
        return np.linspace(0.0, 1.0, len(q_path))

    return path_s / path_s[-1]


def estimate_duration(q_path: np.ndarray, segment_name: str) -> float:
    """根据当前分段配置估计轨迹执行时长。

    这里使用每个关节沿整条路径的累计运动量，而不是只看首末点差值。
    对合并后的 current -> pregrasp -> grasp 路径更稳，避免路径绕了一段但首末点差值较小导致 retime 过快。
    """
    cfg = SEGMENT_TIMING[segment_name]
    q_path = np.asarray(q_path, dtype=float)

    if q_path.shape[0] < 2:
        return float(cfg["min_duration"])

    joint_travel = np.sum(np.abs(np.diff(q_path, axis=0)), axis=0)
    max_joint_travel = float(np.max(joint_travel))
    duration = max_joint_travel / float(cfg["max_joint_speed"])
    return max(float(cfg["min_duration"]), duration)


def retime_joint_path_scurve(
    q_path: np.ndarray,
    segment_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对一个 joint path 做 S 曲线时间参数化。"""
    q_path = remove_near_duplicate_waypoints(np.asarray(q_path, dtype=float))
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

    with profile_block(f"plan_pose:{segment_name}"):
        result = planner.plan_pose(
            goal_tool_poses=goal_pose,
            current_state=current_state,
            max_attempts=PLAN_POSE_MAX_ATTEMPTS,
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


def summarize_plan_info(plan_info: dict) -> dict:
    """复制 plan_info 中适合写入 JSON 的标量诊断字段，避免循环引用。"""
    keys = [
        "planner_success",
        "planner_position_error_m",
        "planner_rotation_error_rad",
        "pose_converged",
        "raw_num_waypoints",
    ]
    return {
        key: plan_info[key]
        for key in keys
        if key in plan_info
    }


def make_pregrasp_position_candidates(
    pregrasp_position: np.ndarray,
    grasp_position: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """生成 pregrasp 自动回退候选。

    输入的 pregrasp 通常等于 grasp 沿 world/base +Z 上方 10 cm。
    如果这个点不可达，依次尝试 5 cm、3 cm、1.5 cm、0 cm。
    这些候选只改变 pregrasp 的位置，不改变 TCP 姿态。
    """
    pregrasp_position = np.asarray(pregrasp_position, dtype=np.float32)
    grasp_position = np.asarray(grasp_position, dtype=np.float32)
    delta = pregrasp_position - grasp_position
    original_offset = float(np.linalg.norm(delta))

    if original_offset < 1.0e-6:
        return [("original", pregrasp_position.copy())]

    direction = delta / original_offset
    offsets = [original_offset]
    offsets.extend(offset for offset in PREGRASP_FALLBACK_OFFSETS_M if offset < original_offset)

    candidates: list[tuple[str, np.ndarray]] = []
    for index, offset in enumerate(offsets):
        position = grasp_position + direction * float(offset)
        if any(np.allclose(position, existing, atol=1.0e-5) for _, existing in candidates):
            continue

        if index == 0:
            label = "original"
        else:
            label = f"offset_{offset:.3f}m"
        candidates.append((label, position.astype(np.float32)))

    return candidates


def make_tcp_roll_orientation_candidates(
    pregrasp_quaternion: np.ndarray,
    grasp_quaternion: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """生成绕 TCP 局部 +X 轴的等价 grasp 姿态候选。"""
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []

    for roll_deg in TCP_ROLL_FALLBACK_DEG:
        pre_q = quat_roll_about_local_x(pregrasp_quaternion, roll_deg)
        grasp_q = quat_roll_about_local_x(grasp_quaternion, roll_deg)

        if any(
            np.allclose(pre_q, old_pre_q, atol=1.0e-5)
            and np.allclose(grasp_q, old_grasp_q, atol=1.0e-5)
            for _, old_pre_q, old_grasp_q in candidates
        ):
            continue

        label = f"tcp_roll_{roll_deg:+.0f}deg"
        candidates.append((label, pre_q, grasp_q))

    return candidates


def build_motion_segment_from_raw_path(
    planner: MotionPlanner,
    q_path_raw: np.ndarray,
    plan_info: dict,
    target_name: str,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    segment_name: str,
    T_world_base: np.ndarray,
    extra_fields: dict | None = None,
) -> tuple[dict, np.ndarray]:
    """把已经得到的 raw joint path 重新时间参数化并封装成 motion segment。"""
    with profile_block(f"retime:{segment_name}"):
        time_from_start, q_traj, qd_traj, qdd_traj = retime_joint_path_scurve(
            q_path_raw,
            segment_name=segment_name,
        )

    with profile_block(f"compute_tcp_path:{segment_name}"):
        tcp_path = compute_tcp_path(planner, q_traj, T_world_base)

    with profile_block(f"final_fk:{segment_name}"):
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

    if extra_fields:
        segment.update(extra_fields)

    return segment, q_traj[-1].astype(np.float32)


def build_motion_segment(
    planner: MotionPlanner,
    q_start: np.ndarray,
    target_name: str,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    segment_name: str,
    T_world_base: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """规划并封装一个普通 motion segment。"""
    q_path_raw, plan_info = plan_pose_path(
        planner=planner,
        q_start=q_start,
        target_position=target_position,
        target_quaternion=target_quaternion,
        segment_name=segment_name,
    )

    return build_motion_segment_from_raw_path(
        planner=planner,
        q_path_raw=q_path_raw,
        plan_info=plan_info,
        target_name=target_name,
        target_position=target_position,
        target_quaternion=target_quaternion,
        segment_name=segment_name,
        T_world_base=T_world_base,
    )


def build_pregrasp_to_grasp_segment(
    planner: MotionPlanner,
    q_start: np.ndarray,
    pregrasp_position: np.ndarray,
    pregrasp_quaternion: np.ndarray,
    grasp_position: np.ndarray,
    grasp_quaternion: np.ndarray,
    T_world_base: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """规划 current -> pregrasp -> grasp，并输出为一条连续 motion。

    关键点：pregrasp 只作为途经点，不再作为 Isaac 执行时的独立 motion segment。
    这样执行脚本不会在 pregrasp 处调用 settle_arm_to_start，从根源上减少段间抖动。
    """
    attempts = []
    q_path_to_pregrasp = None
    q_path_to_grasp = None
    pregrasp_info = None
    grasp_info = None
    chosen_pregrasp_position = None
    chosen_pregrasp_label = None
    chosen_orientation_label = None
    chosen_pregrasp_quaternion = None
    chosen_grasp_quaternion = None

    orientation_candidates = make_tcp_roll_orientation_candidates(
        pregrasp_quaternion,
        grasp_quaternion,
    )
    position_candidates = make_pregrasp_position_candidates(
        pregrasp_position,
        grasp_position,
    )

    for orientation_label, candidate_pregrasp_quat, candidate_grasp_quat in orientation_candidates:
        for position_index, (position_label, candidate_position) in enumerate(position_candidates):
            pregrasp_segment_name = "move_to_pregrasp"
            if orientation_label != "tcp_roll_+0deg" or position_index > 0:
                pregrasp_segment_name = (
                    f"move_to_pregrasp_retry_{orientation_label}_{position_label}"
                )

            attempt_record = {
                "orientation_label": orientation_label,
                "position_label": position_label,
                "pregrasp_position_xyz": candidate_position.tolist(),
                "pregrasp_quaternion_wxyz": candidate_pregrasp_quat.tolist(),
                "grasp_quaternion_wxyz": candidate_grasp_quat.tolist(),
                "success": False,
            }

            try:
                candidate_path_to_pregrasp, candidate_pregrasp_info = plan_pose_path(
                    planner=planner,
                    q_start=q_start,
                    target_position=candidate_position,
                    target_quaternion=candidate_pregrasp_quat,
                    segment_name=pregrasp_segment_name,
                )
            except RuntimeError as exc:
                print(
                    "[pregrasp fallback] failed "
                    f"orientation={orientation_label}, position={position_label}, "
                    f"position_xyz={candidate_position}, reason={exc}"
                )
                attempt_record["stage"] = "move_to_pregrasp"
                attempt_record["error"] = str(exc)
                attempts.append(attempt_record)
                continue

            q_pregrasp = candidate_path_to_pregrasp[-1].astype(np.float32)

            grasp_segment_name = "approach_to_grasp"
            if orientation_label != "tcp_roll_+0deg" or position_index > 0:
                grasp_segment_name = (
                    f"approach_to_grasp_retry_{orientation_label}_{position_label}"
                )

            try:
                candidate_path_to_grasp, candidate_grasp_info = plan_pose_path(
                    planner=planner,
                    q_start=q_pregrasp,
                    target_position=grasp_position,
                    target_quaternion=candidate_grasp_quat,
                    segment_name=grasp_segment_name,
                )
            except RuntimeError as exc:
                print(
                    "[grasp fallback] failed "
                    f"orientation={orientation_label}, position={position_label}, "
                    f"reason={exc}"
                )
                attempt_record["stage"] = "approach_to_grasp"
                attempt_record["error"] = str(exc)
                attempt_record["pregrasp_plan_info"] = summarize_plan_info(
                    candidate_pregrasp_info
                )
                attempts.append(attempt_record)
                continue

            attempt_record["success"] = True
            attempt_record["stage"] = "complete"
            attempt_record["pregrasp_plan_info"] = summarize_plan_info(
                candidate_pregrasp_info
            )
            attempt_record["grasp_plan_info"] = summarize_plan_info(
                candidate_grasp_info
            )
            attempts.append(attempt_record)

            q_path_to_pregrasp = candidate_path_to_pregrasp
            q_path_to_grasp = candidate_path_to_grasp
            pregrasp_info = candidate_pregrasp_info
            grasp_info = candidate_grasp_info
            chosen_pregrasp_position = candidate_position
            chosen_pregrasp_label = position_label
            chosen_orientation_label = orientation_label
            chosen_pregrasp_quaternion = candidate_pregrasp_quat
            chosen_grasp_quaternion = candidate_grasp_quat
            break

        if q_path_to_pregrasp is not None and q_path_to_grasp is not None:
            break

    if q_path_to_pregrasp is None or q_path_to_grasp is None:
        brief = [
            (
                f"{item['orientation_label']}/{item['position_label']}:"
                f"{item.get('stage')}:{item.get('error')}"
            )
            for item in attempts[-8:]
        ]
        raise RuntimeError(
            "pregrasp/grasp 所有姿态和位置候选都失败。"
            "当前目标不是距离问题，而是抓取姿态、桌面约束或自碰撞约束下不可达。"
            "最近失败候选: "
            + " | ".join(brief)
        )

    if chosen_orientation_label != "tcp_roll_+0deg" or chosen_pregrasp_label != "original":
        print(
            "[pregrasp/grasp fallback] using "
            f"orientation={chosen_orientation_label}, "
            f"pregrasp={chosen_pregrasp_label}, "
            f"position={chosen_pregrasp_position}"
        )

    pregrasp_info["fallback"] = {
        "used": (
            chosen_orientation_label != "tcp_roll_+0deg"
            or chosen_pregrasp_label != "original"
        ),
        "chosen_orientation_label": chosen_orientation_label,
        "chosen_pregrasp_label": chosen_pregrasp_label,
        "original_pregrasp_position_xyz": np.asarray(pregrasp_position, dtype=float).tolist(),
        "chosen_pregrasp_position_xyz": np.asarray(chosen_pregrasp_position, dtype=float).tolist(),
        "original_grasp_quaternion_wxyz": np.asarray(grasp_quaternion, dtype=float).tolist(),
        "chosen_grasp_quaternion_wxyz": np.asarray(chosen_grasp_quaternion, dtype=float).tolist(),
        "attempts": attempts,
    }

    if q_path_to_grasp.shape[0] > 1:
        q_path_merged = np.vstack([q_path_to_pregrasp, q_path_to_grasp[1:]])
    else:
        q_path_merged = q_path_to_pregrasp.copy()

    return_home_position, return_home_quaternion = run_fk(planner, q_path_merged[0])

    plan_info = {
        "planner_success": bool(
            pregrasp_info["planner_success"] and grasp_info["planner_success"]
        ),
        "planner_position_error_m": grasp_info["planner_position_error_m"],
        "planner_rotation_error_rad": grasp_info["planner_rotation_error_rad"],
        "pose_converged": bool(
            pregrasp_info["pose_converged"] and grasp_info["pose_converged"]
        ),
        "raw_num_waypoints": int(q_path_merged.shape[0]),
        "merged_from": ["move_to_pregrasp", "approach_to_grasp"],
        "sub_plans": {
            "move_to_pregrasp": pregrasp_info,
            "approach_to_grasp": grasp_info,
        },
    }

    extra_fields = {
        "merged_motion_segments": ["move_to_pregrasp", "approach_to_grasp"],
        "reverse_approach_lift": {
            "description": "close_gripper 后沿 approach_to_grasp 的反向路径退回 chosen pregrasp，避免从 grasp 重新规划 lift 失败。",
            "q_path_raw": q_path_to_grasp[::-1].copy().tolist(),
            "target_pose_base": {
                "position_xyz": chosen_pregrasp_position.tolist(),
                "quaternion_wxyz": chosen_pregrasp_quaternion.tolist(),
            },
            "source_motion": "reverse of selected approach_to_grasp raw path",
            "pregrasp_fallback_label": chosen_pregrasp_label,
            "orientation_fallback_label": chosen_orientation_label,
        },
        "reverse_full_approach_return": {
            "description": "close_gripper 后沿合并后的 approach_to_grasp 完整反向路径退回本轮任务 home pose。",
            "q_path_raw": q_path_merged[::-1].copy().tolist(),
            "target_pose_base": {
                "position_xyz": return_home_position.tolist(),
                "quaternion_wxyz": return_home_quaternion.tolist(),
            },
            "source_motion": "reverse of merged current-to-pregrasp-to-grasp raw path",
            "pregrasp_fallback_label": chosen_pregrasp_label,
            "orientation_fallback_label": chosen_orientation_label,
        },
        "via_targets": [
            {
                "name": "pregrasp",
                "pose_base": {
                    "position_xyz": chosen_pregrasp_position.tolist(),
                    "quaternion_wxyz": chosen_pregrasp_quaternion.tolist(),
                },
                "original_position_xyz": np.asarray(pregrasp_position, dtype=float).tolist(),
                "fallback_label": chosen_pregrasp_label,
                "orientation_fallback_label": chosen_orientation_label,
            }
        ],
    }

    # 这里故意保留 name="approach_to_grasp"：
    # 1. 兼容执行脚本中 STRICT_POST_MOTION_WAIT_SEGMENTS={"approach_to_grasp"}
    # 2. close_gripper 前仍会等待机械臂真正到达 grasp 位姿
    return build_motion_segment_from_raw_path(
        planner=planner,
        q_path_raw=q_path_merged,
        plan_info=plan_info,
        target_name="grasp",
        target_position=grasp_position,
        target_quaternion=chosen_grasp_quaternion,
        segment_name="approach_to_grasp",
        T_world_base=T_world_base,
        extra_fields=extra_fields,
    )


def build_lift_segment_from_reverse_approach(
    planner: MotionPlanner,
    approach_segment: dict,
    T_world_base: np.ndarray,
    segment_name: str = "lift_object",
    target_name: str = "lift",
    reverse_info_key: str = "reverse_approach_lift",
) -> tuple[dict, np.ndarray]:
    """用 approach_to_grasp 的反向路径生成 lift/retreat 段。

    抓取后最稳定的第一版 lift 不是重新求解一个新目标，而是沿着刚刚接近物体
    的路径反向退出。这条路径已经被 cuRobo 验证过，且对 top-down 抓取等价于
    从 grasp 回到 pregrasp。
    """
    reverse_info = approach_segment.get(reverse_info_key)
    if not reverse_info:
        raise RuntimeError(f"approach segment 缺少 {reverse_info_key} 字段。")

    q_path_raw = np.asarray(reverse_info["q_path_raw"], dtype=np.float32)
    target_pose = reverse_info["target_pose_base"]
    target_position = np.asarray(target_pose["position_xyz"], dtype=np.float32)
    target_quaternion = np.asarray(target_pose["quaternion_wxyz"], dtype=np.float32)

    plan_info = {
        "planner_success": True,
        "planner_position_error_m": 0.0,
        "planner_rotation_error_rad": 0.0,
        "pose_converged": True,
        "raw_num_waypoints": int(q_path_raw.shape[0]),
        "generated_from": reverse_info_key,
        "note": "No additional cuRobo plan_pose call; this reuses the already feasible approach path in reverse.",
        reverse_info_key: {
            "pregrasp_fallback_label": reverse_info.get("pregrasp_fallback_label"),
            "orientation_fallback_label": reverse_info.get("orientation_fallback_label"),
        },
    }

    return build_motion_segment_from_raw_path(
        planner=planner,
        q_path_raw=q_path_raw,
        plan_info=plan_info,
        target_name=target_name,
        target_position=target_position,
        target_quaternion=target_quaternion,
        segment_name=segment_name,
        T_world_base=T_world_base,
        extra_fields={
            "retreat_strategy": "reverse_approach_path",
            "reverse_info_key": reverse_info_key,
            "source_motion": reverse_info.get("source_motion"),
        },
    )


def make_gripper_segment(name: str, q_target: float, gripper_joint_names: list[str]) -> dict:
    """创建一个夹爪动作 segment。轨迹执行脚本会真正发送该动作。"""
    return {
        "name": name,
        "type": "gripper",
        "joint_names": gripper_joint_names,
        "target_position": [float(q_target) for _ in gripper_joint_names],
    }


def plan_grasp_segments(
    planner: MotionPlanner | None = None,
    destroy_planner: bool = True,
) -> dict:
    """执行一次抓取分段规划。

    planner 为 None 时，本函数按原 standalone 脚本行为创建并销毁 MotionPlanner。
    常驻 planner server 会传入已经创建好的 planner，并设置 destroy_planner=False，
    从而避免每次任务重复初始化 cuRobo。
    """
    with profile_block("load_inputs"):
        isaac_state = load_isaac_state()    # 读取isaac中机器人状态
        target_data = load_grasp_target()   # 读取目标夹爪位姿

    print("[input] state_json:", STATE_JSON)
    print("[input] target_json:", TARGET_JSON)
    print("[target] object:", target_data.get("source", {}).get("object_prim_path"))
    print("[target] sequence:", target_data.get("sequence"))
    print("[target] grasp_mode:", target_data.get("source", {}).get("grasp_mode", "top_down"))

    with profile_block("prepare_state_and_targets"):
        q_start = get_start_q_from_isaac_state(isaac_state) # 从isaac状态中读取当前机械臂关节位置，并检查顺序是否正确
        joint_limits = load_joint_limits_from_urdf(ROBOT_URDF)  # 从 URDF 读取关节限位
        q_current = clip_q_to_joint_limits(q_start, joint_limits)   # 把 Isaac 的 q_current 裁剪回 joint limit
        T_world_base = np.asarray(isaac_state["poses"]["world_base"]["matrix_4x4"], dtype=float) # 获取 world_base 的变换矩阵 
        world_scene = make_world_collision_scene(isaac_state)

        pregrasp_pos, pregrasp_quat = get_named_target_pose(target_data, "pregrasp")
        grasp_pos, grasp_quat = get_named_target_pose(target_data, "grasp")
        lift_pos, lift_quat = get_named_target_pose(target_data, "lift")
        grasp_mode = target_data.get("source", {}).get("grasp_mode", "top_down")

        gripper_info = target_data.get("gripper", {})
        gripper_joint_names = list(gripper_info.get("joint_names", ["arm_joint7", "arm_joint8"]))
        gripper_open = float(gripper_info.get("open_m", 0.04))
        gripper_close = float(gripper_info.get("close_m", 0.0))

    owns_planner = planner is None
    segments = []

    try:
        if planner is None:
            with profile_block("create_planner_total"):
                planner = create_planner()

        update_planner_world(planner, world_scene)

        segments.append(make_gripper_segment("open_gripper", gripper_open, gripper_joint_names))

        # 关键修改：
        # 仍然让 cuRobo 分别求 current -> pregrasp 和 pregrasp -> grasp，
        # 但输出给 Isaac 的时候合并成一条连续 motion，避免 pregrasp 处切段抖动。
        segment, q_current = build_pregrasp_to_grasp_segment(
            planner=planner,
            q_start=q_current,
            pregrasp_position=pregrasp_pos,
            pregrasp_quaternion=pregrasp_quat,
            grasp_position=grasp_pos,
            grasp_quaternion=grasp_quat,
            T_world_base=T_world_base,
        )
        segments.append(segment)
        approach_segment = segment

        segments.append(make_gripper_segment("close_gripper", gripper_close, gripper_joint_names))

        if grasp_mode == "side":
            print("[side grasp] skip vertical lift; retreat by reversing approach path.")
            segment, q_current = build_lift_segment_from_reverse_approach(
                planner=planner,
                approach_segment=approach_segment,
                T_world_base=T_world_base,
                segment_name="retreat_object",
                target_name="home",
                reverse_info_key="reverse_full_approach_return",
            )
            segment["retreat_strategy"] = "reverse_full_approach_path_for_side_grasp"
        else:
            segment, q_current = build_lift_segment_from_reverse_approach(
                planner=planner,
                approach_segment=approach_segment,
                T_world_base=T_world_base,
            )
        segments.append(segment)

    finally:
        if planner is not None and owns_planner and destroy_planner:
            with profile_block("planner.destroy"):
                planner.destroy()

    motion_segments = [segment for segment in segments if segment["type"] == "motion"]
    total_motion_duration = sum(segment["timing"]["duration_s"] for segment in motion_segments)
    all_success = all(segment["plan_info"]["planner_success"] for segment in motion_segments)

    payload = {
        "schema_version": 1,
        "robot_name": "go2_x5",
        "planner": "curobo.MotionPlanner.plan_pose merged pregrasp-grasp",
        "source_state_json": str(STATE_JSON),
        "source_target_json": str(TARGET_JSON),
        "joint_names": EXPECTED_JOINT_NAMES,
        "tool_frame": EXPECTED_TOOL_FRAME,
        "object_prim_path": target_data.get("source", {}).get("object_prim_path"),
        "grasp_mode": grasp_mode,
        "segments": segments,
        "summary": {
            "num_segments": len(segments),
            "num_motion_segments": len(motion_segments),
            "all_motion_segments_success": all_success,
            "total_motion_duration_s": float(total_motion_duration),
            "final_q_arm": q_current.tolist(),
            "grasp_mode": grasp_mode,
        },
    }

    with profile_block("write_output_json"):
        OUTPUT_JSON.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return payload


def main() -> None:
    print("========== Go2-X5 Grasp Segments Planning ==========")
    PROFILER.add("process_start_to_main", time.perf_counter() - SCRIPT_START_TIME)

    payload = plan_grasp_segments()
    summary = payload["summary"]

    print_header("Planning Summary")
    print("output:", OUTPUT_JSON)
    print("all_motion_segments_success:", summary["all_motion_segments_success"])
    print("num_segments:", summary["num_segments"])
    print("total_motion_duration_s:", summary["total_motion_duration_s"])
    print("final_q_arm:", np.asarray(summary["final_q_arm"], dtype=np.float32))
    PROFILER.print_summary()
    print("========== grasp segment planning complete ==========")


if __name__ == "__main__":
    main()
