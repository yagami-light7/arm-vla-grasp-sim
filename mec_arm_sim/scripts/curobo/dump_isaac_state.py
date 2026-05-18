"""
从 Isaac Sim 当前 stage 导出机器人状态。

推荐运行位置：
    Isaac Sim 5.1.0 Script Editor

本脚本现在通过 robot profile 工作，默认 profile 是：
    mec_arm_sim/configs/robots/go2_x5.yaml

导出的 JSON 会同时包含：
    - Isaac 完整 DOF order
    - 完整 q / dq
    - arm_joint1 ~ arm_joint6 的 q_arm / dq_arm
    - T_world_base
    - T_world_tcp
    - T_base_tcp

后续普通 Python 脚本会读取这个 JSON，用 cuRobo FK 做对齐检查。
"""

from __future__ import annotations

import asyncio
import json
import math
import traceback
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
import yaml
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation


# 在 Script Editor 中调试时，最常改的是这里。
ROBOT_PROFILE_PATH = Path("/home/light/workspace/arm_vla/mec_arm_sim/configs/robots/go2_x5.yaml")


def normalize_quat_wxyz(quat) -> np.ndarray:
    """归一化 wxyz 四元数。"""
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        raise ValueError(f"四元数范数太小，无法归一化: {quat}")
    return quat / norm


def quat_wxyz_to_rotmat(quat) -> np.ndarray:
    """wxyz 四元数转 3x3 旋转矩阵。"""
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotmat_to_quat_wxyz(rotation) -> np.ndarray:
    """3x3 旋转矩阵转 wxyz 四元数。"""
    rotation = np.asarray(rotation, dtype=float)
    trace = np.trace(rotation)

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diag = np.diag(rotation)
        idx = int(np.argmax(diag))
        if idx == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif idx == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale

    return normalize_quat_wxyz([w, x, y, z])


def rpy_to_rotmat(rpy_xyz) -> np.ndarray:
    """URDF/Isaac 常用固定轴 rpy 转旋转矩阵。"""
    roll, pitch, yaw = [float(v) for v in rpy_xyz]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rot_z @ rot_y @ rot_x


def pose_to_matrix(position_xyz, quaternion_wxyz) -> np.ndarray:
    """position + wxyz quaternion 转 4x4 齐次矩阵。"""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quat_wxyz_to_rotmat(quaternion_wxyz)
    transform[:3, 3] = np.asarray(position_xyz, dtype=float)
    return transform


def xyz_rpy_to_matrix(xyz, rpy) -> np.ndarray:
    """xyz + rpy 转 4x4 齐次矩阵，用于 TCP fallback offset。"""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_rotmat(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def matrix_to_pose_wxyz(transform) -> tuple[np.ndarray, np.ndarray]:
    """4x4 齐次矩阵转 position + wxyz quaternion。"""
    transform = np.asarray(transform, dtype=float)
    return transform[:3, 3].copy(), rotmat_to_quat_wxyz(transform[:3, :3])


def matrix_to_list(transform) -> list[list[float]]:
    """4x4 numpy matrix 转 JSON 可写入的 nested list。"""
    return np.asarray(transform, dtype=float).tolist()


def pose_dict_from_matrix(transform) -> dict:
    """把 4x4 位姿矩阵转成 JSON 友好的 position/quaternion 字典。"""
    position, quaternion = matrix_to_pose_wxyz(transform)
    return {
        "position_xyz": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
    }


def safe_numpy(value) -> np.ndarray:
    """把 Isaac 返回的数据安全转成一维 numpy array。"""
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(float, copy=False)


def load_profile(path: Path) -> dict:
    """读取 robot profile。"""
    if not path.exists():
        raise FileNotFoundError(f"robot profile 不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stage():
    """获取当前 Isaac Sim GUI 中已经打开的 USD stage。"""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD stage。请先在 Isaac Sim GUI 中打开场景。")
    return stage


def prim_exists(stage, prim_path: str) -> bool:
    """检查 prim path 是否存在。"""
    return bool(prim_path) and stage.GetPrimAtPath(prim_path).IsValid()


def find_first_prim_by_name(stage, root_path: str, prim_name: str) -> str | None:
    """在 robot root 下按最后一级名称查找 prim。"""
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == prim_name:
            return str(prim.GetPath())
    return None


def resolve_frame_path(stage, configured_path: str, robot_root_path: str, frame_name: str) -> str:
    """
    解析 frame prim path。

    先尝试 profile 写死的 path；如果不存在，再在 robot_root 下按 frame 名称搜索。
    """
    if prim_exists(stage, configured_path):
        return configured_path

    found = find_first_prim_by_name(stage, robot_root_path, frame_name)
    if found is not None:
        print(f"[路径] {configured_path} 不存在，按名称找到 {frame_name}: {found}")
        return found

    raise RuntimeError(f"找不到 frame prim: configured={configured_path}, name={frame_name}")


def scan_articulation_roots(stage, robot_root_path: str) -> list[str]:
    """扫描带 UsdPhysics.ArticulationRootAPI 的 prim。"""
    robot_root = stage.GetPrimAtPath(robot_root_path)
    if not robot_root.IsValid():
        return []

    roots: list[str] = []
    for prim in Usd.PrimRange(robot_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
    return roots


def resolve_articulation_root_path(stage, configured_path: str, robot_root_path: str) -> str:
    """解析 articulation root path，避免硬编码路径不匹配时直接失败。"""
    if prim_exists(stage, configured_path):
        return configured_path

    roots = scan_articulation_roots(stage, robot_root_path)
    if len(roots) == 1:
        print(f"[路径] {configured_path} 不存在，扫描到 articulation root: {roots[0]}")
        return roots[0]
    if len(roots) > 1:
        raise RuntimeError(f"扫描到多个 articulation root，请在 profile 中明确指定: {roots}")

    raise RuntimeError(f"找不到 articulation root: {configured_path}")


def usd_pose_to_matrix(stage, prim_path: str) -> np.ndarray:
    """读取某个 prim 的 world pose，并转换成 4x4 矩阵。"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim path 不存在: {prim_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    usd_matrix = cache.GetLocalToWorldTransform(prim)
    translation = usd_matrix.ExtractTranslation()
    rotation = usd_matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()

    position = np.array([translation[0], translation[1], translation[2]], dtype=float)
    quaternion_wxyz = normalize_quat_wxyz([rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]])
    return pose_to_matrix(position, quaternion_wxyz)


def get_dof_names(robot) -> list[str]:
    """读取 Isaac Sim 内部 DOF 顺序。"""
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


def build_joint_indices(dof_names: list[str], joint_names: list[str]) -> list[int]:
    """把 planner joint_names 映射到 Isaac full DOF order 的索引。"""
    missing = [name for name in joint_names if name not in dof_names]
    if missing:
        raise RuntimeError(f"Isaac DOF order 中找不到这些关节: {missing}\n完整 DOF order: {dof_names}")
    return [dof_names.index(name) for name in joint_names]


async def create_robot_handle(articulation_root_path: str):
    """初始化 World，并创建 SingleArticulation 句柄。"""
    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    await world.play_async()
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(
        prim_path=articulation_root_path,
        name="dump_robot_state",
    )
    robot.initialize()

    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 无效: {articulation_root_path}")

    return robot


async def dump_isaac_state():
    """Script Editor 主流程。"""
    print("========== Dump Isaac Robot State ==========")

    profile = load_profile(ROBOT_PROFILE_PATH)
    stage = get_stage()

    isaac_cfg = profile["isaac"]
    planner_cfg = profile["planner"]
    output_path = Path(profile["outputs"]["isaac_state_json"])

    robot_root_path = str(isaac_cfg["robot_root_path"])
    articulation_root_path = resolve_articulation_root_path(
        stage,
        str(isaac_cfg["articulation_root_path"]),
        robot_root_path,
    )
    base_frame_path = resolve_frame_path(
        stage,
        str(isaac_cfg["base_frame_path"]),
        robot_root_path,
        str(planner_cfg["base_link"]),
    )

    tcp_mode = "direct_prim"
    if prim_exists(stage, str(isaac_cfg["tcp_frame_path"])):
        tcp_frame_path = str(isaac_cfg["tcp_frame_path"])
        T_world_tcp = usd_pose_to_matrix(stage, tcp_frame_path)
    else:
        tcp_mode = "fallback_offset"
        fallback_path = resolve_frame_path(
            stage,
            str(isaac_cfg["fallback_tcp_frame_path"]),
            robot_root_path,
            "arm_link6",
        )
        tcp_frame_path = fallback_path
        T_world_fallback = usd_pose_to_matrix(stage, fallback_path)
        T_fallback_tcp = xyz_rpy_to_matrix(
            isaac_cfg["fallback_tcp_offset_xyz"],
            isaac_cfg["fallback_tcp_offset_rpy"],
        )
        T_world_tcp = T_world_fallback @ T_fallback_tcp

    robot = await create_robot_handle(articulation_root_path)
    dof_names = get_dof_names(robot)
    if not dof_names:
        raise RuntimeError("无法读取 Isaac DOF order。")

    active_joints = list(planner_cfg["active_joints"])
    gripper_joints = list(planner_cfg.get("gripper_joints", []))
    active_indices = build_joint_indices(dof_names, active_joints)
    gripper_indices = [dof_names.index(name) for name in gripper_joints if name in dof_names]

    q_full = safe_numpy(robot.get_joint_positions())
    dq_full = safe_numpy(robot.get_joint_velocities())
    q_arm = q_full[active_indices]
    dq_arm = dq_full[active_indices]

    T_world_base = usd_pose_to_matrix(stage, base_frame_path)
    T_base_tcp = np.linalg.inv(T_world_base) @ T_world_tcp

    base_position, base_quat = matrix_to_pose_wxyz(T_world_base)
    tcp_world_position, tcp_world_quat = matrix_to_pose_wxyz(T_world_tcp)
    tcp_base_position, tcp_base_quat = matrix_to_pose_wxyz(T_base_tcp)

    print("[Isaac] robot root:", robot_root_path)
    print("[Isaac] articulation root:", articulation_root_path)
    print("[Isaac] base frame:", base_frame_path)
    print("[Isaac] tcp frame source:", tcp_frame_path, f"mode={tcp_mode}")
    print("[Isaac] full DOF order:", dof_names)
    print("[Isaac] active arm joints:", active_joints)
    print("[Isaac] active arm indices:", active_indices)
    print("[Isaac] q_arm:", np.array2string(q_arm, precision=6))
    print("[Isaac] dq_arm:", np.array2string(dq_arm, precision=6))
    print("[Isaac] TCP world position:", np.array2string(tcp_world_position, precision=6))
    print("[Isaac] TCP world quat_wxyz:", np.array2string(tcp_world_quat, precision=6))
    print("[Isaac] TCP base position:", np.array2string(tcp_base_position, precision=6))
    print("[Isaac] TCP base quat_wxyz:", np.array2string(tcp_base_quat, precision=6))

    output = {
        "robot_profile": str(ROBOT_PROFILE_PATH),
        "robot_name": profile["name"],
        "resolved_paths": {
            "robot_root_path": robot_root_path,
            "articulation_root_path": articulation_root_path,
            "base_frame_path": base_frame_path,
            "tcp_frame_path": tcp_frame_path,
            "tcp_mode": tcp_mode,
        },
        "planner": {
            "base_link": planner_cfg["base_link"],
            "tool_frame": planner_cfg["tool_frame"],
            "active_joints": active_joints,
            "gripper_joints": gripper_joints,
            "active_joint_indices": active_indices,
            "gripper_joint_indices": gripper_indices,
        },
        "isaac": {
            "dof_names": dof_names,
            "q_full": q_full.tolist(),
            "dq_full": dq_full.tolist(),
            "q_arm": q_arm.tolist(),
            "dq_arm": dq_arm.tolist(),
            "base_world": {
                "position_xyz": base_position.tolist(),
                "quaternion_wxyz": base_quat.tolist(),
            },
            "tcp_world": {
                "position_xyz": tcp_world_position.tolist(),
                "quaternion_wxyz": tcp_world_quat.tolist(),
            },
            "tcp_base": {
                "position_xyz": tcp_base_position.tolist(),
                "quaternion_wxyz": tcp_base_quat.tolist(),
            },
            "T_world_base": matrix_to_list(T_world_base),
            "T_world_tcp": matrix_to_list(T_world_tcp),
            "T_base_tcp": matrix_to_list(T_base_tcp),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[输出] 已保存 Isaac 状态 JSON:", output_path)
    print("========== Dump complete ==========")


asyncio.ensure_future(dump_isaac_state())
