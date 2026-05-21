"""
检查 Go2-X5 夹爪真实抓取 TCP 位置。

用途：
    当前 grasp_tcp_link 可能不在真实夹爪接触中心，导致 cuRobo 规划到位后
    夹爪几何仍然和物体有偏移或碰撞。

    本脚本读取 arm_link6、arm_link7、arm_link8、grasp_tcp_link 的 world pose / bbox，
    在 arm_link6 坐标系下估计几个候选 TCP offset，并在 Isaac viewport 中画出来。

运行位置：
    Isaac Sim 5.1.0 GUI -> Window -> Script Editor

输出：
    /tmp/go2_x5_grasp_tcp_candidate.json

颜色：
    蓝色: arm_link6 origin
    红色: 当前 grasp_tcp_link
    黄色: 两个 finger bbox center 的中点
    绿色: 推荐候选 grasp_tcp_link
    白色/灰色: 左右 finger bbox center
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom


ROBOT_ROOT_PATH = "/World/go2_x5"

WRIST_LINK_NAME = "arm_link6"
LEFT_FINGER_LINK_NAME = "arm_link7"
RIGHT_FINGER_LINK_NAME = "arm_link8"
CURRENT_EEF_LINK_NAME = "grasp_tcp_link"
CURRENT_EEF_FALLBACK_OFFSET_XYZ = np.array(
    [0.1425699970126152, 0.0, 0.0],
    dtype=float,
)

NEW_TCP_LINK_NAME = "grasp_tcp_link"
NEW_TCP_FIXED_JOINT_NAME = "grasp_tcp_fixed_joint"

DEBUG_ROOT_PATH = "/World/debug_gripper_tcp_inspection"
OUTPUT_JSON = Path("/tmp/go2_x5_grasp_tcp_candidate.json")

# 推荐 TCP 取 finger 前端往回一点，避免把 TCP 放在 mesh 最尖端。
# 如果绿色 marker 明显太靠前/太靠后，先改这个值再重新运行脚本。
FRONT_BACKOFF_M = 0.015


def get_stage():
    """获取当前 Isaac stage。"""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 Isaac stage。")
    return stage


def find_first_prim_by_name(stage, root_path: str, prim_name: str) -> str:
    """在 robot root 下按 prim name 查找第一个匹配的 prim path。"""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"robot root 不存在: {root_path}")

    for prim in Usd.PrimRange(root):
        if prim.GetName() == prim_name:
            return str(prim.GetPath())

    raise RuntimeError(f"在 {root_path} 下找不到 prim name={prim_name}")


def find_optional_prim_by_name(stage, root_path: str, prim_name: str) -> str | None:
    """按 prim name 查找 prim；找不到时返回 None。"""
    try:
        return find_first_prim_by_name(stage, root_path, prim_name)
    except RuntimeError:
        return None


def usd_world_pose_to_matrix(stage, prim_path: str) -> np.ndarray:
    """
    读取 prim world pose，转成标准 4x4 SE(3) 矩阵。

    标准约定：
        平移在最后一列，点使用 [x, y, z, 1]^T。
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim 不存在: {prim_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    usd_matrix = xform_cache.GetLocalToWorldTransform(prim)

    translation = usd_matrix.ExtractTranslation()
    rotation = usd_matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()

    position = np.array([translation[0], translation[1], translation[2]], dtype=float)
    quat_wxyz = np.array(
        [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
        dtype=float,
    )

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quat_wxyz_to_rotmat(quat_wxyz)
    transform[:3, 3] = position
    return transform


def normalize_quat_wxyz(quat) -> np.ndarray:
    """归一化 wxyz 四元数。"""
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        raise ValueError(f"四元数范数太小: {quat}")
    return quat / norm


def quat_wxyz_to_rotmat(quat) -> np.ndarray:
    """wxyz 四元数转旋转矩阵。"""
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_point(transform: np.ndarray, point_xyz) -> np.ndarray:
    """用 4x4 SE(3) 矩阵变换三维点。"""
    point_h = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=float)
    return (np.asarray(transform, dtype=float) @ point_h)[:3]


def compute_world_bbox(stage, prim_path: str) -> dict:
    """读取 prim 的 world aligned bbox。"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim 不存在: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = bound.ComputeAlignedBox()

    bbox_min = np.array(aligned_box.GetMin(), dtype=float)
    bbox_max = np.array(aligned_box.GetMax(), dtype=float)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_size = bbox_max - bbox_min

    return {
        "min": bbox_min,
        "max": bbox_max,
        "center": bbox_center,
        "size": bbox_size,
    }


def bbox_world_corners(bbox: dict) -> np.ndarray:
    """从 world aligned bbox 生成 8 个 corner。"""
    mn = bbox["min"]
    mx = bbox["max"]
    corners = []
    for x in [mn[0], mx[0]]:
        for y in [mn[1], mx[1]]:
            for z in [mn[2], mx[2]]:
                corners.append([x, y, z])
    return np.asarray(corners, dtype=float)


def bbox_in_link6_frame(T_link6_world: np.ndarray, bbox: dict) -> dict:
    """
    把 world aligned bbox 的 8 个角点转换到 arm_link6 frame。

    注意：
        这是对 finger 几何范围的近似估计，足够用于初步确定 TCP marker。
        最终仍以 Isaac viewport 中绿色 marker 的视觉位置为准。
    """
    corners_world = bbox_world_corners(bbox)
    corners_link6 = np.array(
        [transform_point(T_link6_world, corner) for corner in corners_world],
        dtype=float,
    )
    mn = corners_link6.min(axis=0)
    mx = corners_link6.max(axis=0)
    center = 0.5 * (mn + mx)
    return {
        "min": mn,
        "max": mx,
        "center": center,
        "size": mx - mn,
        "corners": corners_link6,
    }


def choose_front_axis(current_eef_link6: np.ndarray, combined_min: np.ndarray, combined_max: np.ndarray) -> tuple[int, float]:
    """
    判断 finger 从 arm_link6 往哪个 x 方向伸出。

    通常当前 grasp_tcp_link 在 x=0.08657 附近，finger 前端会在更大的 x。
    如果 mesh 导入方向相反，这里会自动选择离当前 eef 更远的一侧。
    """
    distance_to_min_x = abs(float(current_eef_link6[0] - combined_min[0]))
    distance_to_max_x = abs(float(combined_max[0] - current_eef_link6[0]))

    if distance_to_max_x >= distance_to_min_x:
        return 1, float(combined_max[0])
    return -1, float(combined_min[0])


def estimate_tcp_candidates(stage, paths: dict) -> dict:
    """估计当前 eef 和候选 grasp tcp 在 arm_link6 frame 下的位置。"""
    T_world_link6 = usd_world_pose_to_matrix(stage, paths["link6"])
    T_link6_world = np.linalg.inv(T_world_link6)

    if paths["eef"] is not None:
        T_world_eef = usd_world_pose_to_matrix(stage, paths["eef"])
        eef_world = T_world_eef[:3, 3]
        eef_link6 = transform_point(T_link6_world, eef_world)
        eef_mode = "direct_grasp_tcp_link_prim"
    else:
        eef_link6 = CURRENT_EEF_FALLBACK_OFFSET_XYZ.copy()
        eef_mode = "fallback_arm_link6_plus_urdf_fixed_offset"

    left_bbox_world = compute_world_bbox(stage, paths["left_finger"])
    right_bbox_world = compute_world_bbox(stage, paths["right_finger"])

    left_bbox_link6 = bbox_in_link6_frame(T_link6_world, left_bbox_world)
    right_bbox_link6 = bbox_in_link6_frame(T_link6_world, right_bbox_world)

    all_corners = np.vstack([left_bbox_link6["corners"], right_bbox_link6["corners"]])
    combined_min = all_corners.min(axis=0)
    combined_max = all_corners.max(axis=0)
    combined_center = 0.5 * (combined_min + combined_max)

    finger_center_midpoint = 0.5 * (left_bbox_link6["center"] + right_bbox_link6["center"])

    front_sign, front_x = choose_front_axis(eef_link6, combined_min, combined_max)
    recommended = np.array(
        [
            front_x - front_sign * FRONT_BACKOFF_M,
            0.0,
            combined_center[2],
        ],
        dtype=float,
    )

    current_eef_delta = recommended - eef_link6

    return {
        "T_world_link6": T_world_link6,
        "T_link6_world": T_link6_world,
        "current_eef_link6": eef_link6,
        "current_eef_mode": eef_mode,
        "left_bbox_link6": left_bbox_link6,
        "right_bbox_link6": right_bbox_link6,
        "combined_min_link6": combined_min,
        "combined_max_link6": combined_max,
        "combined_center_link6": combined_center,
        "finger_center_midpoint_link6": finger_center_midpoint,
        "recommended_grasp_tcp_link6": recommended,
        "recommended_delta_from_current_eef": current_eef_delta,
        "front_sign": front_sign,
        "front_x_raw": front_x,
    }


def create_marker(stage, path: str, position_world, color, radius=0.012):
    """创建一个彩色球 marker。"""
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*position_world))
    UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return sphere


def create_line(stage, path: str, points_world, color, width=0.004):
    """创建一条 debug line。"""
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points_world)])
    curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points_world])
    curve.CreateWidthsAttr([width])
    UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return curve


def draw_debug(stage, candidates: dict):
    """在 viewport 中画出当前 eef 和候选 TCP。"""
    if stage.GetPrimAtPath(DEBUG_ROOT_PATH).IsValid():
        stage.RemovePrim(DEBUG_ROOT_PATH)
    UsdGeom.Xform.Define(stage, DEBUG_ROOT_PATH)

    T_world_link6 = candidates["T_world_link6"]

    link6_origin_world = T_world_link6[:3, 3]
    eef_world = transform_point(T_world_link6, candidates["current_eef_link6"])
    finger_mid_world = transform_point(T_world_link6, candidates["finger_center_midpoint_link6"])
    recommended_world = transform_point(T_world_link6, candidates["recommended_grasp_tcp_link6"])
    left_center_world = transform_point(T_world_link6, candidates["left_bbox_link6"]["center"])
    right_center_world = transform_point(T_world_link6, candidates["right_bbox_link6"]["center"])

    create_marker(stage, DEBUG_ROOT_PATH + "/link6_origin_blue", link6_origin_world, (0.1, 0.3, 1.0), 0.012)
    create_marker(stage, DEBUG_ROOT_PATH + "/current_eef_red", eef_world, (1.0, 0.1, 0.1), 0.014)
    create_marker(stage, DEBUG_ROOT_PATH + "/finger_mid_yellow", finger_mid_world, (1.0, 0.85, 0.1), 0.014)
    create_marker(stage, DEBUG_ROOT_PATH + "/recommended_tcp_green", recommended_world, (0.1, 1.0, 0.2), 0.016)
    create_marker(stage, DEBUG_ROOT_PATH + "/left_finger_center_white", left_center_world, (1.0, 1.0, 1.0), 0.01)
    create_marker(stage, DEBUG_ROOT_PATH + "/right_finger_center_gray", right_center_world, (0.55, 0.55, 0.55), 0.01)

    create_line(
        stage,
        DEBUG_ROOT_PATH + "/eef_to_recommended_line",
        [eef_world, recommended_world],
        (0.1, 1.0, 0.2),
        width=0.006,
    )
    create_line(
        stage,
        DEBUG_ROOT_PATH + "/finger_center_line",
        [left_center_world, right_center_world],
        (1.0, 0.85, 0.1),
        width=0.004,
    )


def make_urdf_snippet(offset_xyz) -> str:
    """生成后续要写入 URDF 的 TCP link/fixed joint 片段。"""
    x, y, z = [float(v) for v in offset_xyz]
    return f"""  <link name="{NEW_TCP_LINK_NAME}">
    <inertial>
      <mass value="0.0"/>
      <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
      <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
  </link>
  <joint name="{NEW_TCP_FIXED_JOINT_NAME}" type="fixed">
    <origin xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="0 0 0"/>
    <parent link="{WRIST_LINK_NAME}"/>
    <child link="{NEW_TCP_LINK_NAME}"/>
  </joint>"""


def numpy_to_list_dict(data):
    """把嵌套 dict 中的 numpy array 转成 list，方便写 JSON。"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {key: numpy_to_list_dict(value) for key, value in data.items() if key != "corners"}
    if isinstance(data, (list, tuple)):
        return [numpy_to_list_dict(value) for value in data]
    return data


async def main():
    print("========== Inspect Go2-X5 Gripper TCP ==========")

    stage = get_stage()
    paths = {
        "link6": find_first_prim_by_name(stage, ROBOT_ROOT_PATH, WRIST_LINK_NAME),
        "left_finger": find_first_prim_by_name(stage, ROBOT_ROOT_PATH, LEFT_FINGER_LINK_NAME),
        "right_finger": find_first_prim_by_name(stage, ROBOT_ROOT_PATH, RIGHT_FINGER_LINK_NAME),
        "eef": find_optional_prim_by_name(stage, ROBOT_ROOT_PATH, CURRENT_EEF_LINK_NAME),
    }

    print("[paths]")
    for name, path in paths.items():
        print(f"  {name:12s}: {path if path is not None else '<fallback>'}")

    candidates = estimate_tcp_candidates(stage, paths)
    draw_debug(stage, candidates)

    recommended = candidates["recommended_grasp_tcp_link6"]
    delta = candidates["recommended_delta_from_current_eef"]

    urdf_snippet = make_urdf_snippet(recommended)

    payload = {
        "schema_version": 1,
        "script": "scripts/dev_tools/isaac/inspect_gripper_tcp.py",
        "robot_root_path": ROBOT_ROOT_PATH,
        "paths": paths,
        "frame": "arm_link6",
        "new_tcp_link_name": NEW_TCP_LINK_NAME,
        "new_tcp_fixed_joint_name": NEW_TCP_FIXED_JOINT_NAME,
        "front_backoff_m": FRONT_BACKOFF_M,
        "recommended_origin_xyz_in_arm_link6": recommended.tolist(),
        "recommended_delta_from_current_grasp_tcp_link": delta.tolist(),
        "urdf_snippet": urdf_snippet,
        "candidates": numpy_to_list_dict(candidates),
        "notes": [
            "绿色 marker 是推荐 grasp_tcp_link 位置。",
            "如果绿色 marker 不在两指接触中心，请调整 FRONT_BACKOFF_M 或根据 JSON 手动修正 recommended_origin_xyz_in_arm_link6。",
            "确认后，需要把 urdf_snippet 写入 full URDF，并重新生成 arm-only URDF 与 cuRobo YAML/XRDF。",
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[current eef xyz in arm_link6]:", candidates["current_eef_link6"])
    print("[current eef mode]:", candidates["current_eef_mode"])
    print("[finger midpoint xyz in arm_link6]:", candidates["finger_center_midpoint_link6"])
    print("[recommended grasp tcp xyz in arm_link6]:", recommended)
    print("[delta from current eef]:", delta)
    print("[output]", OUTPUT_JSON)
    print("[urdf snippet]")
    print(urdf_snippet)
    print("========== complete ==========")

    await omni.kit.app.get_app().next_update_async()


asyncio.ensure_future(main())
