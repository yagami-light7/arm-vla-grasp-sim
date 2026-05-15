"""
Isaac Sim Script Editor demo：cuRobo TCP 轨迹规划与可视化。

运行方式：
1. 打开 Isaac Sim 5.1.0 GUI。
2. 打开已经包含 /World/mec_arm 的 stage。
3. 确认机器人 articulation root 是 /World/mec_arm/root_joint。
4. 在 Script Editor 中运行本脚本。

本脚本只做：
1. 从 Isaac Sim 读取当前关节角 q_current 和当前 TCP 世界位姿。
2. 将目标 TCP 位姿、cuboid 障碍物从 Isaac world frame 转到 robot base_link frame。
3. 调用独立 Python 子进程中的新版 cuRobo MotionPlanner 规划无碰撞关节轨迹。
4. 在子进程中对 cuRobo 输出轨迹做 S 曲线时间重采样和 FK。
5. 把子进程返回的 TCP path 在 Isaac viewport 中画出来。

本脚本不做：
1. 不发送 ArticulationAction。
2. 不驱动机械臂跟踪轨迹。
3. 不接 AnyGrasp / VLA。

四元数约定：
- 脚本顶部、cuRobo、打印输出都使用 wxyz。
- 坐标单位是米和弧度。
"""

import asyncio
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import omni
from pxr import Gf, Sdf, Usd, UsdGeom

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation


# 不在 Isaac Sim 进程里 import cuRobo。
# 原因：Isaac Sim 5.1 GUI 进程会预加载 omni.warp 1.8.2，
# 但当前 cuRobo 需要 warp_lang 1.13.0；两者在同一进程里冲突。
# 本脚本只负责读取 Isaac 状态和可视化，cuRobo 规划放到独立 Python 子进程。
PYTHON_EXECUTABLE = "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python"
EXTERNAL_PLANNER_SCRIPT = Path(
    "/home/light/workspace/arm_vla/mec_arm_sim/scripts/curobo/run_curobo_plan_external.py"
)
EXTERNAL_PLANNER_TIMEOUT_S = 240
BRIDGE_DIR = Path("/tmp/curobo_isaac_bridge")


# ==============================================================================
# 你当前机器人在 Isaac Sim 中的已确认路径
# ==============================================================================

ROBOT_ASSET_ROOT_PATH = "/World/mec_arm"
ARTICULATION_ROOT_PATH = "/World/mec_arm/root_joint"
ROBOT_BASE_FRAME_PATH = "/World/mec_arm/base_link"
TCP_FRAME_PATH = "/World/mec_arm/Empty_Link6/TCP_link"

EXPECTED_JOINT_ORDER = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
TOOL_FRAME_NAME = "TCP_link"

ROBOT_CONFIG_PATH = Path("/home/light/workspace/arm_vla/mec_arm_sim/configs/curobo/mec_arm.yml")


# ==============================================================================
# 第一版手动目标和障碍物配置
# ==============================================================================

# 如果你已经知道目标 TCP 世界位姿，把 None 改成：
# TARGET_TCP_POSE_WORLD = {
#     "position": [x, y, z],
#     "quaternion_wxyz": [qw, qx, qy, qz],
# }
#
# 保持 None 时，脚本会用“当前 TCP 位姿 + TARGET_TCP_WORLD_OFFSET_IF_NONE”生成一个小目标。
TARGET_TCP_POSE_WORLD = None
TARGET_TCP_WORLD_OFFSET_IF_NONE = np.array([0.08, -0.06, 0.04], dtype=float)

# 第一版只使用 cuboid 障碍物，不解析完整 3DGS collision mesh。
# position 为 None 时，会用“当前 TCP 世界位置 + offset_from_current_tcp_world”生成障碍物位置。
OBSTACLE_CUBOIDS_WORLD = [
    {
        "name": "debug_block_near_tcp",
        "dims": [0.08, 0.08, 0.16],
        "position": None,
        "offset_from_current_tcp_world": [0.04, 0.10, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
]


# ==============================================================================
# 规划、重采样、验证参数
# ==============================================================================

NUM_IK_SEEDS = 32
NUM_TRAJOPT_SEEDS = 4
MAX_PLANNING_ATTEMPTS = 5
USE_CUDA_GRAPH = False
RUN_PLANNER_WARMUP = False
SELF_COLLISION_CHECK = False
ALLOW_IK_FALLBACK = True

S_CURVE_DT = 0.02
S_CURVE_MIN_DURATION = 1.5
MAX_JOINT_VEL_RAD_S = np.array([1.2, 1.2, 1.2, 1.5, 1.5, 1.8], dtype=float)
MAX_JOINT_ACC_RAD_S2 = np.array([3.0, 3.0, 3.0, 4.0, 4.0, 5.0], dtype=float)

JOINT_LIMITS = {
    "Joint1": (-3.10, 3.10),
    "Joint2": (-2.60, 0.00),
    "Joint3": (0.00, 4.00),
    "Joint4": (-3.10, 3.10),
    "Joint5": (0.00, 3.10),
    "Joint6": (-1.57, 1.57),
}
JOINT_LIMIT_TOL = 1e-3

FINAL_POSITION_TOL_M = 0.02
FINAL_ORIENTATION_TOL_DEG = 5.0
SINGULARITY_CONDITION_WARN = 200.0
SINGULARITY_MAX_SAMPLES = 40
NUMERIC_JACOBIAN_EPS = 1e-4

START_FK_WARN_POS_M = 0.05
START_FK_WARN_ORI_DEG = 10.0

DEBUG_ROOT_PATH = "/World/debug_curobo_trajectory"
WAYPOINT_MARKER_RADIUS_M = 0.012
WAYPOINT_MARKER_STRIDE = 4
DRAW_OBSTACLE_MARKERS = True


# ==============================================================================
# 基础数学工具：SE(3)、四元数、坐标系转换
# ==============================================================================


def normalize_quat_wxyz(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError(f"四元数范数太小，无法归一化：{quat_wxyz}")
    return quat / norm


def quat_wxyz_to_rotmat(quat_wxyz):
    qw, qx, qy, qz = normalize_quat_wxyz(quat_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def rotmat_to_quat_wxyz(rotmat):
    matrix = np.asarray(rotmat, dtype=float)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diag = np.diag(matrix)
        index = int(np.argmax(diag))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s

    return normalize_quat_wxyz([qw, qx, qy, qz])


def pose_to_matrix(position_xyz, quaternion_wxyz):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quat_wxyz_to_rotmat(quaternion_wxyz)
    transform[:3, 3] = np.asarray(position_xyz, dtype=float)
    return transform


def matrix_to_pose_wxyz(transform):
    transform = np.asarray(transform, dtype=float)
    position = transform[:3, 3].copy()
    quaternion = rotmat_to_quat_wxyz(transform[:3, :3])
    return position, quaternion


def usd_pose_to_matrix(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim path 不存在：{prim_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    usd_matrix = cache.GetLocalToWorldTransform(prim)
    translation = usd_matrix.ExtractTranslation()
    rotation = usd_matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()

    position = np.array([translation[0], translation[1], translation[2]], dtype=float)
    quaternion = normalize_quat_wxyz(
        [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]]
    )
    return pose_to_matrix(position, quaternion)


def world_pose_to_base_pose(world_position, world_quaternion, world_to_base_matrix):
    world_pose = pose_to_matrix(world_position, world_quaternion)
    base_pose = world_to_base_matrix @ world_pose
    return matrix_to_pose_wxyz(base_pose)


def base_pose_to_world_pose(base_position, base_quaternion, base_to_world_matrix):
    base_pose = pose_to_matrix(base_position, base_quaternion)
    world_pose = base_to_world_matrix @ base_pose
    return matrix_to_pose_wxyz(world_pose)


def transform_points(transform, points_xyz):
    points = np.asarray(points_xyz, dtype=float)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
    transformed = (np.asarray(transform, dtype=float) @ homogeneous.T).T
    return transformed[:, :3]


def quat_conjugate_wxyz(quat):
    quat = normalize_quat_wxyz(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = normalize_quat_wxyz(q1)
    w2, x2, y2, z2 = normalize_quat_wxyz(q2)
    return normalize_quat_wxyz(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_angle_error_deg(q_a, q_b):
    q_a = normalize_quat_wxyz(q_a)
    q_b = normalize_quat_wxyz(q_b)
    dot = float(abs(np.dot(q_a, q_b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def quat_to_rotvec_wxyz(quat):
    quat = normalize_quat_wxyz(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:4]
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-10:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, quat[0])
    return vector / vector_norm * angle


# ==============================================================================
# Isaac Sim 读取工具
# ==============================================================================


def safe_numpy(value):
    array = np.asarray(value)
    if array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return array.astype(float, copy=False)


def get_dof_names(robot):
    try:
        return list(robot.dof_names)
    except Exception:
        pass

    view = getattr(robot, "_articulation_view", None)
    if view is not None:
        try:
            return list(view.dof_names)
        except Exception:
            pass

    return []


def resolve_target_tcp_pose_world(current_tcp_position, current_tcp_quaternion):
    if TARGET_TCP_POSE_WORLD is None:
        position = current_tcp_position + TARGET_TCP_WORLD_OFFSET_IF_NONE
        quaternion = current_tcp_quaternion
        return position, quaternion

    position = TARGET_TCP_POSE_WORLD.get("position")
    quaternion = TARGET_TCP_POSE_WORLD.get("quaternion_wxyz")

    if position is None:
        position = current_tcp_position + TARGET_TCP_WORLD_OFFSET_IF_NONE
    if quaternion is None:
        quaternion = current_tcp_quaternion

    return np.asarray(position, dtype=float), normalize_quat_wxyz(quaternion)


def resolve_obstacle_cuboids_world(current_tcp_position):
    cuboids = []
    for item in OBSTACLE_CUBOIDS_WORLD:
        name = item["name"]
        dims = np.asarray(item["dims"], dtype=float)
        position = item.get("position")
        quaternion = item.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])

        if position is None:
            offset = np.asarray(item.get("offset_from_current_tcp_world", [0.0, 0.0, 0.0]), dtype=float)
            position = current_tcp_position + offset

        cuboids.append(
            {
                "name": name,
                "dims": dims,
                "position": np.asarray(position, dtype=float),
                "quaternion_wxyz": normalize_quat_wxyz(quaternion),
            }
        )
    return cuboids


# ==============================================================================
# cuRobo 输入/输出工具
# ==============================================================================


def make_goal_tool_pose(position_base, quaternion_base, planner):
    position_tensor = torch.tensor(
        [[[[position_base.tolist()]]]], device=DEVICE, dtype=DTYPE
    )
    quaternion_tensor = torch.tensor(
        [[[[normalize_quat_wxyz(quaternion_base).tolist()]]]], device=DEVICE, dtype=DTYPE
    )
    return GoalToolPose(
        tool_frames=list(planner.tool_frames),
        position=position_tensor,
        quaternion=quaternion_tensor,
    )


def make_joint_state(q_current, joint_names):
    q_tensor = torch.as_tensor(q_current, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    return JointState.from_position(q_tensor, joint_names=list(joint_names))


def build_scene_dict_from_cuboids(cuboids_world, world_to_base_matrix):
    scene_cuboids = {}
    cuboids_base = []

    for cuboid in cuboids_world:
        position_base, quaternion_base = world_pose_to_base_pose(
            cuboid["position"],
            cuboid["quaternion_wxyz"],
            world_to_base_matrix,
        )
        pose_base = position_base.tolist() + normalize_quat_wxyz(quaternion_base).tolist()
        scene_cuboids[cuboid["name"]] = {
            "dims": cuboid["dims"].astype(float).tolist(),
            "pose": pose_base,
        }
        cuboids_base.append(
            {
                "name": cuboid["name"],
                "dims": cuboid["dims"],
                "position": position_base,
                "quaternion_wxyz": quaternion_base,
                "pose": pose_base,
            }
        )

    return {"cuboid": scene_cuboids, "mesh": {}}, cuboids_base


def json_sanitize(value):
    if isinstance(value, np.ndarray):
        return [json_sanitize(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def run_external_curobo_planner(request):
    if not EXTERNAL_PLANNER_SCRIPT.exists():
        raise FileNotFoundError(f"外部 cuRobo planner 脚本不存在：{EXTERNAL_PLANNER_SCRIPT}")

    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{os.getpid()}_{int(time.time() * 1000)}"
    request_path = BRIDGE_DIR / f"request_{stamp}.json"
    response_path = BRIDGE_DIR / f"response_{stamp}.json"

    request_path.write_text(
        json.dumps(json_sanitize(request), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cmd = [
        PYTHON_EXECUTABLE,
        str(EXTERNAL_PLANNER_SCRIPT),
        str(request_path),
        str(response_path),
    ]

    print("[cuRobo] 使用独立 Python 子进程规划，避免 Isaac Sim 内部 omni.warp 冲突。")
    print("[cuRobo] subprocess command:", " ".join(cmd))
    print("[cuRobo] request json:", request_path)
    print("[cuRobo] response json:", response_path)

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=EXTERNAL_PLANNER_TIMEOUT_S,
        check=False,
    )

    if completed.stdout:
        print("[cuRobo subprocess stdout]")
        print(completed.stdout.rstrip())
    if completed.stderr:
        print("[cuRobo subprocess stderr]")
        print(completed.stderr.rstrip())

    if completed.returncode != 0:
        raise RuntimeError(f"外部 cuRobo planner 进程失败，returncode={completed.returncode}")
    if not response_path.exists():
        raise RuntimeError(f"外部 cuRobo planner 没有写出 response json：{response_path}")

    response = json.loads(response_path.read_text(encoding="utf-8"))
    if not response.get("success", False):
        raise RuntimeError(f"cuRobo 规划失败：{response.get('error')}")

    return response


def extract_joint_trajectory(result):
    if result is None:
        raise RuntimeError("cuRobo 没有返回规划结果。")

    interpolated = result.get_interpolated_plan()
    if interpolated is None or interpolated.position is None:
        raise RuntimeError("cuRobo 结果里没有 interpolated trajectory。")

    q = interpolated.position.detach()
    if q.ndim == 3:
        q = q[0]
    elif q.ndim != 2:
        raise RuntimeError(f"未知轨迹 shape：{tuple(q.shape)}")

    return q.cpu().numpy().astype(float)


def fk_tcp_poses_base(planner, q_trajectory):
    q_trajectory = np.asarray(q_trajectory, dtype=float)
    q_tensor = torch.as_tensor(q_trajectory, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    joint_state = JointState.from_position(q_tensor, joint_names=list(planner.joint_names))
    kin_state = planner.compute_kinematics(joint_state)

    positions = kin_state.tool_poses.position.detach().cpu().numpy()
    quaternions = kin_state.tool_poses.quaternion.detach().cpu().numpy()

    # 输入 q 是 [1, T, dof]，输出 pose 是 [1, T, num_tool_frames, 3/4]。
    positions = positions[0, :, 0, :]
    quaternions = quaternions[0, :, 0, :]
    quaternions = np.asarray([normalize_quat_wxyz(q) for q in quaternions], dtype=float)
    return positions, quaternions


def fk_single_tcp_pose_base(planner, q):
    positions, quaternions = fk_tcp_poses_base(planner, np.asarray(q, dtype=float)[None, :])
    return positions[0], quaternions[0]


# ==============================================================================
# 核心算法 1：S 曲线时间重采样
# ==============================================================================


def smootherstep(u):
    """五次 smootherstep：位置、速度、加速度在起点和终点都连续。"""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def s_curve_resample_joint_trajectory(
    q_raw,
    dt=S_CURVE_DT,
    min_duration=S_CURVE_MIN_DURATION,
    max_joint_vel=MAX_JOINT_VEL_RAD_S,
    max_joint_acc=MAX_JOINT_ACC_RAD_S2,
):
    """
    对 cuRobo 输出的 joint path 做 S 曲线时间重采样。

    注意：
    - cuRobo 负责路径搜索、碰撞约束和优化。
    - 这里不改变路径几何，只改变沿路径前进的时间参数。
    - 第一版使用 joint-space chord length + smootherstep，目标是平滑可视化轨迹。
    """
    q_raw = np.asarray(q_raw, dtype=float)
    if q_raw.ndim != 2:
        raise ValueError(f"q_raw 必须是 [T, dof]，当前 shape={q_raw.shape}")
    if q_raw.shape[0] < 2:
        raise ValueError("轨迹至少需要两个 waypoint。")

    segment_lengths = np.linalg.norm(np.diff(q_raw, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    path_length = float(cumulative[-1])
    if path_length < 1e-9:
        raise ValueError("cuRobo 返回的轨迹几乎没有运动，无法做 S 曲线重采样。")

    joint_delta = np.max(np.abs(np.diff(q_raw, axis=0)), axis=0)
    total_joint_delta = np.abs(q_raw[-1] - q_raw[0])

    # smootherstep 的最大归一化速度约 1.875，最大归一化加速度约 5.8。
    duration_from_vel = float(np.max(1.875 * total_joint_delta / np.asarray(max_joint_vel)))
    duration_from_acc = float(np.max(np.sqrt(5.8 * np.maximum(total_joint_delta, joint_delta) / np.asarray(max_joint_acc))))
    duration = max(float(min_duration), duration_from_vel, duration_from_acc)

    sample_count = max(2, int(math.ceil(duration / dt)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    u = times / duration
    target_s = path_length * smootherstep(u)

    q_resampled = np.empty((sample_count, q_raw.shape[1]), dtype=float)
    for joint_index in range(q_raw.shape[1]):
        q_resampled[:, joint_index] = np.interp(target_s, cumulative, q_raw[:, joint_index])

    qd = np.gradient(q_resampled, times, axis=0, edge_order=1)
    qdd = np.gradient(qd, times, axis=0, edge_order=1)

    return {
        "q": q_resampled,
        "t": times,
        "qd": qd,
        "qdd": qdd,
        "duration": duration,
    }


# ==============================================================================
# 核心算法 2：关节限位和奇异性检查
# ==============================================================================


def check_joint_limits(q_trajectory, joint_names):
    q_trajectory = np.asarray(q_trajectory, dtype=float)
    violations = []
    for joint_index, joint_name in enumerate(joint_names):
        lower, upper = JOINT_LIMITS[joint_name]
        values = q_trajectory[:, joint_index]
        bad_indices = np.where((values < lower - JOINT_LIMIT_TOL) | (values > upper + JOINT_LIMIT_TOL))[0]
        if bad_indices.size:
            violations.append(
                {
                    "joint": joint_name,
                    "lower": lower,
                    "upper": upper,
                    "first_bad_index": int(bad_indices[0]),
                    "first_bad_value": float(values[bad_indices[0]]),
                }
            )
    return violations


def numerical_tcp_jacobian(planner, q):
    q = np.asarray(q, dtype=float)
    pos_0, quat_0 = fk_single_tcp_pose_base(planner, q)
    jacobian = np.zeros((6, q.shape[0]), dtype=float)

    for joint_index in range(q.shape[0]):
        q_eps = q.copy()
        q_eps[joint_index] += NUMERIC_JACOBIAN_EPS
        pos_eps, quat_eps = fk_single_tcp_pose_base(planner, q_eps)

        jacobian[:3, joint_index] = (pos_eps - pos_0) / NUMERIC_JACOBIAN_EPS

        delta_quat = quat_multiply_wxyz(quat_eps, quat_conjugate_wxyz(quat_0))
        jacobian[3:, joint_index] = quat_to_rotvec_wxyz(delta_quat) / NUMERIC_JACOBIAN_EPS

    return jacobian


def check_singularity_by_numeric_jacobian(planner, q_trajectory):
    q_trajectory = np.asarray(q_trajectory, dtype=float)
    sample_count = min(SINGULARITY_MAX_SAMPLES, q_trajectory.shape[0])
    sample_indices = np.unique(
        np.linspace(0, q_trajectory.shape[0] - 1, sample_count).round().astype(int)
    )

    records = []
    for index in sample_indices:
        jacobian = numerical_tcp_jacobian(planner, q_trajectory[index])
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        smallest = float(np.min(singular_values))
        largest = float(np.max(singular_values))
        condition = float("inf") if smallest < 1e-9 else largest / smallest
        records.append(
            {
                "index": int(index),
                "condition": condition,
                "sigma_min": smallest,
                "sigma_max": largest,
            }
        )

    max_record = max(records, key=lambda item: item["condition"])
    risky = [item for item in records if item["condition"] > SINGULARITY_CONDITION_WARN]
    return {
        "records": records,
        "max_condition": max_record["condition"],
        "max_record": max_record,
        "risky": risky,
    }


# ==============================================================================
# Isaac Sim 可视化
# ==============================================================================


def delete_debug_root(stage):
    root_path = Sdf.Path(DEBUG_ROOT_PATH)
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)


def create_colored_sphere(stage, path, position, color, radius):
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    return sphere.GetPrim()


def create_curve(stage, path, points_world, color, width):
    curve = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
    curve.CreateTypeAttr("linear")
    curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr([len(points_world)])
    curve.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points_world])
    curve.CreateWidthsAttr([float(width)])
    curve.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return curve.GetPrim()


def create_cuboid_marker(stage, path, position, quaternion_wxyz, dims, color):
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    cube.CreateDisplayOpacityAttr([0.35])

    qw, qx, qy, qz = normalize_quat_wxyz(quaternion_wxyz)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xform.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    xform.AddScaleOp().Set(Gf.Vec3f(float(dims[0]), float(dims[1]), float(dims[2])))
    return cube.GetPrim()


def draw_tcp_trajectory_markers(
    stage,
    tcp_positions_world,
    target_position_world,
    cuboids_world,
    singularity_risky_indices,
):
    delete_debug_root(stage)
    UsdGeom.Xform.Define(stage, Sdf.Path(DEBUG_ROOT_PATH))

    create_curve(
        stage,
        f"{DEBUG_ROOT_PATH}/tcp_path_curve",
        tcp_positions_world,
        color=(0.0, 0.85, 1.0),
        width=0.008,
    )

    risky_set = set(int(item["index"]) for item in singularity_risky_indices)
    for index, position in enumerate(tcp_positions_world):
        if index == 0:
            color = (0.0, 1.0, 0.2)
            radius = WAYPOINT_MARKER_RADIUS_M * 1.5
        elif index == len(tcp_positions_world) - 1:
            color = (1.0, 0.0, 0.0)
            radius = WAYPOINT_MARKER_RADIUS_M * 1.5
        elif index in risky_set:
            color = (1.0, 0.55, 0.0)
            radius = WAYPOINT_MARKER_RADIUS_M * 1.2
        elif index % WAYPOINT_MARKER_STRIDE == 0:
            color = (0.0, 0.65, 1.0)
            radius = WAYPOINT_MARKER_RADIUS_M
        else:
            continue

        create_colored_sphere(
            stage,
            f"{DEBUG_ROOT_PATH}/waypoints/wp_{index:04d}",
            position,
            color,
            radius,
        )

    create_colored_sphere(
        stage,
        f"{DEBUG_ROOT_PATH}/target_tcp_red",
        target_position_world,
        color=(1.0, 0.0, 0.0),
        radius=WAYPOINT_MARKER_RADIUS_M * 2.0,
    )

    if DRAW_OBSTACLE_MARKERS:
        for cuboid in cuboids_world:
            create_cuboid_marker(
                stage,
                f"{DEBUG_ROOT_PATH}/obstacles/{cuboid['name']}",
                cuboid["position"],
                cuboid["quaternion_wxyz"],
                cuboid["dims"],
                color=(1.0, 0.6, 0.05),
            )


# ==============================================================================
# 主流程
# ==============================================================================


async def demo_plan_tcp_path():
    print("\n========== cuRobo TCP 轨迹规划与 Isaac Sim 可视化 ==========")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD stage。")

    if not ROBOT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"cuRobo robot config 不存在：{ROBOT_CONFIG_PATH}")

    print("[环境] python executable:", sys.executable)
    print("[环境] external planner python:", PYTHON_EXECUTABLE)

    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    # 不 reset stage，避免改变你当前调好的机器人状态。
    await world.play_async()
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(prim_path=ARTICULATION_ROOT_PATH, name="curobo_demo_robot")
    robot.initialize()
    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 无效：{ARTICULATION_ROOT_PATH}")

    dof_names = get_dof_names(robot)
    print("[Isaac] DOF order:", dof_names)
    if dof_names != EXPECTED_JOINT_ORDER:
        raise RuntimeError(f"Isaac DOF 顺序不符合预期：{dof_names} != {EXPECTED_JOINT_ORDER}")

    q_current = safe_numpy(robot.get_joint_positions())
    print("[Isaac] q_current:", np.array2string(q_current, precision=5, suppress_small=False))

    world_from_base = usd_pose_to_matrix(stage, ROBOT_BASE_FRAME_PATH)
    base_from_world = np.linalg.inv(world_from_base)
    world_from_tcp = usd_pose_to_matrix(stage, TCP_FRAME_PATH)

    current_tcp_world_pos, current_tcp_world_quat = matrix_to_pose_wxyz(world_from_tcp)
    current_tcp_base_pos, current_tcp_base_quat = matrix_to_pose_wxyz(base_from_world @ world_from_tcp)

    print("[Isaac] 当前 TCP world position:", np.array2string(current_tcp_world_pos, precision=6))
    print("[Isaac] 当前 TCP world quat_wxyz:", np.array2string(current_tcp_world_quat, precision=6))
    print("[坐标] 当前 TCP base position:", np.array2string(current_tcp_base_pos, precision=6))
    print("[坐标] 当前 TCP base quat_wxyz:", np.array2string(current_tcp_base_quat, precision=6))

    target_world_pos, target_world_quat = resolve_target_tcp_pose_world(
        current_tcp_world_pos,
        current_tcp_world_quat,
    )
    target_base_pos, target_base_quat = world_pose_to_base_pose(
        target_world_pos,
        target_world_quat,
        base_from_world,
    )

    print("[目标] target TCP world position:", np.array2string(target_world_pos, precision=6))
    print("[目标] target TCP world quat_wxyz:", np.array2string(target_world_quat, precision=6))
    print("[目标] target TCP base position:", np.array2string(target_base_pos, precision=6))
    print("[目标] target TCP base quat_wxyz:", np.array2string(target_base_quat, precision=6))

    cuboids_world = resolve_obstacle_cuboids_world(current_tcp_world_pos)
    scene_dict, cuboids_base = build_scene_dict_from_cuboids(cuboids_world, base_from_world)

    print("[障碍物] cuboids in base frame:")
    for cuboid in cuboids_base:
        print(
            f"  - {cuboid['name']}: dims={cuboid['dims'].tolist()} "
            f"pose_base={np.array2string(np.asarray(cuboid['pose']), precision=5)}"
        )

    planner_request = {
        "robot_config_path": str(ROBOT_CONFIG_PATH),
        "joint_names": EXPECTED_JOINT_ORDER,
        "tool_frame": TOOL_FRAME_NAME,
        "q_current": q_current,
        "current_tcp_base": {
            "position": current_tcp_base_pos,
            "quaternion_wxyz": current_tcp_base_quat,
        },
        "target_tcp_base": {
            "position": target_base_pos,
            "quaternion_wxyz": target_base_quat,
        },
        "scene_model": scene_dict,
        "params": {
            "num_ik_seeds": NUM_IK_SEEDS,
            "num_trajopt_seeds": NUM_TRAJOPT_SEEDS,
            "max_planning_attempts": MAX_PLANNING_ATTEMPTS,
            "use_cuda_graph": USE_CUDA_GRAPH,
            "run_planner_warmup": RUN_PLANNER_WARMUP,
            "self_collision_check": SELF_COLLISION_CHECK,
            "allow_ik_fallback": ALLOW_IK_FALLBACK,
            "s_curve_dt": S_CURVE_DT,
            "s_curve_min_duration": S_CURVE_MIN_DURATION,
            "max_joint_vel_rad_s": MAX_JOINT_VEL_RAD_S,
            "max_joint_acc_rad_s2": MAX_JOINT_ACC_RAD_S2,
            "joint_limits": JOINT_LIMITS,
            "joint_limit_tol": JOINT_LIMIT_TOL,
            "singularity_condition_warn": SINGULARITY_CONDITION_WARN,
            "singularity_max_samples": SINGULARITY_MAX_SAMPLES,
            "numeric_jacobian_eps": NUMERIC_JACOBIAN_EPS,
        },
    }

    response = run_external_curobo_planner(planner_request)

    print("[cuRobo] planning success:", response.get("planner_success", response.get("success")))
    print("[cuRobo] planner.joint_names:", response["joint_names"])
    print("[cuRobo] planner.tool_frames:", response["tool_frames"])
    if response.get("fallback_used"):
        print(f"[cuRobo][fallback] 使用了 fallback：{response['fallback_used']}")
        print("[cuRobo][fallback] 原因：", response.get("fallback_info", {}).get("reason"))
        print(
            "[cuRobo][fallback] 注意：当前轨迹是 IK goal + joint-space 插值，"
            "不是 cuRobo trajopt 的 collision-free 结果。"
        )

    start_pos_error = float(response["start_position_error_m"])
    start_ori_error_deg = float(response["start_orientation_error_deg"])
    print(f"[坐标检查] cuRobo FK(q_current) 与 Isaac TCP 的 position error: {start_pos_error:.6f} m")
    print(f"[坐标检查] cuRobo FK(q_current) 与 Isaac TCP 的 orientation error: {start_ori_error_deg:.3f} deg")
    if start_pos_error > START_FK_WARN_POS_M or start_ori_error_deg > START_FK_WARN_ORI_DEG:
        print(
            "[坐标检查][警告] cuRobo URDF/base/TCP 与 Isaac stage 的 TCP 不完全一致。"
            "如果规划结果看起来偏移，优先检查 cuRobo URDF 的 TCP_link 固定关节和 base_link frame。"
        )

    q_raw = np.asarray(response["q_raw"], dtype=float)
    q_s = np.asarray(response["q_s"], dtype=float)
    print("[轨迹] raw trajectory shape:", q_raw.shape)
    print("[轨迹] S-curve trajectory shape:", q_s.shape)
    print(f"[轨迹] S-curve duration: {float(response['duration_s']):.3f} s, dt ~= {S_CURVE_DT:.3f} s")
    print(
        "[轨迹] max |joint velocity|:",
        np.array2string(np.asarray(response["max_abs_joint_velocity"], dtype=float), precision=4),
    )
    print(
        "[轨迹] max |joint acceleration|:",
        np.array2string(np.asarray(response["max_abs_joint_acceleration"], dtype=float), precision=4),
    )

    joint_limit_violations = response.get("joint_limit_violations", [])
    if joint_limit_violations:
        print("[关节限位][错误] 发现限位越界：")
        for item in joint_limit_violations:
            print("  ", item)
        raise RuntimeError("S 曲线轨迹存在关节限位越界。")
    print("[关节限位] no waypoint violates joint limits")

    tcp_positions_base = np.asarray(response["tcp_positions_base"], dtype=float)
    tcp_positions_world = transform_points(world_from_base, tcp_positions_base)

    final_pos_error = float(response["final_position_error_m"])
    final_ori_error_deg = float(response["final_orientation_error_deg"])
    print(f"[误差] final TCP position error: {final_pos_error:.6f} m")
    print(f"[误差] final TCP orientation error: {final_ori_error_deg:.3f} deg")

    singularity = response["singularity"]
    max_condition = singularity["max_condition"]
    if isinstance(max_condition, str):
        max_condition_text = max_condition
    else:
        max_condition_text = f"{float(max_condition):.3f}"
    print(f"[奇异性] max condition number: {max_condition_text}")

    risky = singularity.get("risky", [])
    if risky:
        print(
            f"[奇异性][警告] {len(risky)} 个采样点 condition number "
            f"> {SINGULARITY_CONDITION_WARN:.1f}"
        )
        for item in risky[:8]:
            print(
                f"  index={item['index']} cond={item['condition']} "
                f"sigma_min={item['sigma_min']}"
            )
    else:
        print("[奇异性] check passed")

    draw_tcp_trajectory_markers(
        stage=stage,
        tcp_positions_world=tcp_positions_world,
        target_position_world=target_world_pos,
        cuboids_world=cuboids_world,
        singularity_risky_indices=risky,
    )

    print(f"[可视化] trajectory marker path: {DEBUG_ROOT_PATH}")

    if final_pos_error > FINAL_POSITION_TOL_M or final_ori_error_deg > FINAL_ORIENTATION_TOL_DEG:
        print(
            "[结果][警告] 规划成功，但终点误差超过第一版阈值。"
            "后续需要检查目标坐标系、TCP frame 和 cuRobo 规划容差。"
        )
    else:
        print("[结果] 第一版 cuRobo TCP 轨迹规划 demo 通过。")

    print("========== demo 完成 ==========\n")


try:
    asyncio.ensure_future(demo_plan_tcp_path())
except Exception:
    traceback.print_exc()
