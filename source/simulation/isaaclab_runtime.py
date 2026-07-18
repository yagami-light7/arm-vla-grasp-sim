"""由 pipeline 主循环驱动的 Isaac Lab 单环境 runtime。"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.interfaces import EpisodeSpec, RobotAction, SimulationState

from .receptacle_support import (
    inspect_task_receptacle_support_stage,
    inspect_task_receptacle_support_usd,
)
from .scene_runtime import resolve_scene_runtime_settings


FRONT_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/base/head_cam"
FRONT_CAMERA_MOUNT_POS_XYZ_M = (0.28, 0.0, 0.07)
FRONT_CAMERA_MOUNT_ROT_WXYZ = (0.5, -0.5, 0.5, -0.5)
WRIST_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera"
WRIST_CAMERA_CALIBRATION_FRAME = "arm_link6_T_camera_color_optical"
D436_CAMERA_RESOLUTION_WH = (640, 480)
WRIST_CAMERA_HAND_EYE_POS_XYZ_M = (
    0.0559054476,
    0.0026732239,
    0.0767149320,
)
# 标定板弯曲时仅凭 PnP 无法可靠恢复精确外参。根据实机 wrist 图像只看到双爪、
# 看不到夹爪根部的特征，在 camera_color_optical 坐标系沿 -Y 平移 20 mm，
# 将近端夹爪移到画面下方。禁止再沿 optical +Z 前移：该方向会缩短相机到 TCP
# 与抓持物体的深度，并使 30 mm near clipping 切入可乐 mesh。
WRIST_CAMERA_VISUAL_ALIGNMENT_OFFSET_CAMERA_XYZ_M = (0.0, -0.02, 0.0)
WRIST_CAMERA_MOUNT_POS_XYZ_M = (
    0.0666580792,
    0.0028071889,
    0.0935779972,
)
WRIST_CAMERA_MOUNT_ROT_WXYZ = (
    0.3377891849,
    -0.6214992221,
    0.6185057335,
    -0.3421810063,
)
D436_CAMERA_FX_PX = 383.44608095
D436_CAMERA_FY_PX = 383.52724198
D436_CAMERA_CX_PX = 324.33479864
D436_CAMERA_CY_PX = 238.90275478
D436_CAMERA_DISTORTION_COEFFICIENTS = (0.0,) * 12
D436_CAMERA_FALLBACK_FOCAL_LENGTH_MM = 18.0
D436_CAMERA_FALLBACK_FX_FY_PX = 383.486661465
D436_CAMERA_FALLBACK_CX_PX = 320.0
D436_CAMERA_FALLBACK_CY_PX = 240.0
# 普通 USD pinhole 仅作为 schema 不可用时的近似 fallback；精确渲染由 OpenCV schema 决定。
D436_CAMERA_FALLBACK_HORIZONTAL_APERTURE_MM = 30.040158257372415
D436_CAMERA_FALLBACK_VERTICAL_APERTURE_MM = 22.530118693029312
WRIST_CAMERA_NEAR_CLIPPING_M = 0.03
WRIST_CAMERA_TCP_OFFSET_LINK6_XYZ_M = (0.15757, 0.0, 0.0)


def _d436_camera_intrinsics_metadata() -> dict[str, Any]:
    """返回 front/wrist 共用的 D436 640x480 标定内参。"""

    width, height = D436_CAMERA_RESOLUTION_WH
    return {
        "resolution_wh": [width, height],
        "intrinsic_matrix": [
            [D436_CAMERA_FX_PX, 0.0, D436_CAMERA_CX_PX],
            [0.0, D436_CAMERA_FY_PX, D436_CAMERA_CY_PX],
            [0.0, 0.0, 1.0],
        ],
        "intrinsics": {
            "fx": D436_CAMERA_FX_PX,
            "fy": D436_CAMERA_FY_PX,
            "cx": D436_CAMERA_CX_PX,
            "cy": D436_CAMERA_CY_PX,
        },
        "distortion_model": "opencv_pinhole",
        "distortion_coefficients": list(D436_CAMERA_DISTORTION_COEFFICIENTS),
        "renderer_schema_requested": "OmniLensDistortionOpenCvPinholeAPI",
        "standard_usd_pinhole_fallback_intrinsics": {
            "fx": D436_CAMERA_FALLBACK_FX_FY_PX,
            "fy": D436_CAMERA_FALLBACK_FX_FY_PX,
            "cx": D436_CAMERA_FALLBACK_CX_PX,
            "cy": D436_CAMERA_FALLBACK_CY_PX,
        },
    }


def _front_camera_calibration_metadata() -> dict[str, Any]:
    """返回 front camera 的既有外参与新 D436 内参。"""

    return {
        "frame": "base_T_front_camera_color_optical",
        "parent_link": "base",
        "prim_path": FRONT_CAMERA_PRIM_PATH,
        "position_xyz_m": list(FRONT_CAMERA_MOUNT_POS_XYZ_M),
        "rotation_wxyz": list(FRONT_CAMERA_MOUNT_ROT_WXYZ),
        "convention": "ros",
        "extrinsics_source": "dwa_play_nav_cs",
        **_d436_camera_intrinsics_metadata(),
    }


def _wrist_camera_calibration_metadata() -> dict[str, Any]:
    """返回与 RGB 数据一同导出的 wrist camera 手眼标定参数。"""

    return {
        "frame": WRIST_CAMERA_CALIBRATION_FRAME,
        "parent_link": "arm_link6",
        "prim_path": WRIST_CAMERA_PRIM_PATH,
        "position_xyz_m": list(WRIST_CAMERA_MOUNT_POS_XYZ_M),
        "rotation_wxyz": list(WRIST_CAMERA_MOUNT_ROT_WXYZ),
        "convention": "ros",
        "extrinsics_source": "hand_eye_calibration_with_visual_alignment_v3",
        "raw_hand_eye_position_xyz_m": list(WRIST_CAMERA_HAND_EYE_POS_XYZ_M),
        "raw_hand_eye_rotation_wxyz": list(WRIST_CAMERA_MOUNT_ROT_WXYZ),
        "visual_alignment": {
            "method": "image_plane_vertical_reframe_with_grasp_clearance",
            "offset_frame": "camera_color_optical",
            "translation_xyz_m": list(
                WRIST_CAMERA_VISUAL_ALIGNMENT_OFFSET_CAMERA_XYZ_M
            ),
            "rotation_rpy_rad": [0.0, 0.0, 0.0],
            "raw_gripper_root_depth_m": 0.06710842,
            "corrected_gripper_root_depth_m": 0.06710842,
            "raw_tcp_depth_m": 0.12697417,
            "corrected_tcp_depth_m": 0.12697417,
            "near_clipping_range_m": WRIST_CAMERA_NEAR_CLIPPING_M,
            "gripper_root_expected_clipped": False,
            "predicted_initial_finger_top_v_px": 318.5,
            "alignment_goal": "move_gripper_base_below_image_keep_object_depth",
            "preserves_optical_depth": True,
            "metric_recalibration": False,
        },
        **_d436_camera_intrinsics_metadata(),
    }


def _normalized_quaternion_wxyz(raw: Any, *, field_name: str) -> tuple[float, ...]:
    """将 wxyz 四元数归一化，供纯 Python 相机安全检查使用。"""

    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是 4 维数值") from exc
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field_name} 必须是 4 维有限数值")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-12:
        raise ValueError(f"{field_name} 不能是零四元数")
    return tuple(value / norm for value in values)


def _rotate_vector_wxyz(quaternion: Any, vector: Any) -> tuple[float, float, float]:
    """使用 wxyz 四元数旋转三维向量。"""

    qw, qx, qy, qz = _normalized_quaternion_wxyz(
        quaternion,
        field_name="quaternion_wxyz",
    )
    try:
        vx, vy, vz = (float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise ValueError("vector 必须是 3 维数值") from exc
    if not all(math.isfinite(value) for value in (vx, vy, vz)):
        raise ValueError("vector 必须是 3 维有限数值")
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _compute_wrist_camera_object_clearance_sample(
    *,
    tcp_pose_world: Any,
    object_pose_world: Any,
    object_radius_m: float,
    object_half_length_m: float,
    near_clipping_m: float,
    minimum_surface_margin_m: float,
    camera_position_link6_xyz_m: Any = WRIST_CAMERA_MOUNT_POS_XYZ_M,
) -> dict[str, Any]:
    """计算圆柱目标与 wrist 相机近裁剪面的保守间距。"""

    try:
        tcp_pose = tuple(float(value) for value in tcp_pose_world)
        object_pose = tuple(float(value) for value in object_pose_world)
        camera_position = tuple(float(value) for value in camera_position_link6_xyz_m)
        radius = float(object_radius_m)
        half_length = float(object_half_length_m)
        near_clipping = float(near_clipping_m)
        minimum_margin = float(minimum_surface_margin_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("wrist camera clearance 输入必须是数值") from exc
    if len(tcp_pose) != 7 or len(object_pose) != 7 or len(camera_position) != 3:
        raise ValueError("tcp/object pose 必须为 7 维，camera position 必须为 3 维")
    if not all(
        math.isfinite(value)
        for value in (*tcp_pose, *object_pose, *camera_position)
    ):
        raise ValueError("wrist camera clearance pose 必须全部有限")
    if radius <= 0.0 or half_length <= 0.0 or near_clipping <= 0.0:
        raise ValueError("物体尺寸和 near clipping 必须为正数")
    if minimum_margin < 0.0 or not all(
        math.isfinite(value)
        for value in (radius, half_length, near_clipping, minimum_margin)
    ):
        raise ValueError("minimum surface margin 必须为非负有限数")

    tcp_quaternion = tcp_pose[3:7]
    object_quaternion = object_pose[3:7]
    tcp_offset_world = _rotate_vector_wxyz(
        tcp_quaternion,
        WRIST_CAMERA_TCP_OFFSET_LINK6_XYZ_M,
    )
    link6_position_world = tuple(
        tcp_pose[index] - tcp_offset_world[index] for index in range(3)
    )
    camera_offset_world = _rotate_vector_wxyz(tcp_quaternion, camera_position)
    camera_center_world = tuple(
        link6_position_world[index] + camera_offset_world[index]
        for index in range(3)
    )
    camera_axis_link6 = {
        "x": _rotate_vector_wxyz(WRIST_CAMERA_MOUNT_ROT_WXYZ, (1.0, 0.0, 0.0)),
        "y": _rotate_vector_wxyz(WRIST_CAMERA_MOUNT_ROT_WXYZ, (0.0, 1.0, 0.0)),
        "z": _rotate_vector_wxyz(WRIST_CAMERA_MOUNT_ROT_WXYZ, (0.0, 0.0, 1.0)),
    }
    camera_axis_world = {
        axis: _rotate_vector_wxyz(tcp_quaternion, vector)
        for axis, vector in camera_axis_link6.items()
    }
    relative_world = tuple(
        object_pose[index] - camera_center_world[index] for index in range(3)
    )

    def _dot(left: Any, right: Any) -> float:
        return float(sum(float(left[index]) * float(right[index]) for index in range(3)))

    center_camera = tuple(
        _dot(relative_world, camera_axis_world[axis]) for axis in ("x", "y", "z")
    )
    object_axis_world = _rotate_vector_wxyz(
        object_quaternion,
        (0.0, 0.0, 1.0),
    )
    axis_alignment = min(1.0, abs(_dot(camera_axis_world["z"], object_axis_world)))
    projected_half_depth = (
        half_length * axis_alignment
        + radius * math.sqrt(max(0.0, 1.0 - axis_alignment * axis_alignment))
    )
    center_depth = center_camera[2]
    surface_depth = center_depth - projected_half_depth
    far_surface_depth = center_depth + projected_half_depth
    surface_clearance = surface_depth - near_clipping

    width, height = D436_CAMERA_RESOLUTION_WH
    bounding_sphere_radius = math.hypot(half_length, radius)
    depth_for_fov = max(center_depth, near_clipping)
    horizontal_limit = (
        depth_for_fov
        * max(D436_CAMERA_CX_PX, width - D436_CAMERA_CX_PX)
        / D436_CAMERA_FX_PX
        + bounding_sphere_radius
    )
    vertical_limit = (
        depth_for_fov
        * max(D436_CAMERA_CY_PX, height - D436_CAMERA_CY_PX)
        / D436_CAMERA_FY_PX
        + bounding_sphere_radius
    )
    potentially_visible = bool(
        far_surface_depth > near_clipping
        and abs(center_camera[0]) <= horizontal_limit
        and abs(center_camera[1]) <= vertical_limit
    )
    verified = bool(not potentially_visible or surface_clearance >= minimum_margin)
    return {
        "shape": "cylinder_local_z",
        "camera_center_world_xyz_m": list(camera_center_world),
        "object_center_camera_xyz_m": list(center_camera),
        "object_axis_camera_z_abs_dot": axis_alignment,
        "projected_half_depth_m": projected_half_depth,
        "surface_depth_m": surface_depth,
        "far_surface_depth_m": far_surface_depth,
        "near_clipping_m": near_clipping,
        "minimum_surface_margin_m": minimum_margin,
        "surface_clearance_m": surface_clearance,
        "potentially_visible": potentially_visible,
        "near_plane_intersection": bool(
            potentially_visible
            and surface_depth < near_clipping < far_surface_depth
        ),
        "verified": verified,
    }


def _validate_d436_camera_calibration_resolution(
    camera_name: str,
    width: int,
    height: int,
) -> None:
    """拒绝把 640x480 标定参数静默用于其他渲染分辨率。"""

    expected_width, expected_height = D436_CAMERA_RESOLUTION_WH
    if (int(width), int(height)) != (expected_width, expected_height):
        raise ValueError(
            f"{camera_name} camera 的 D436 标定内参仅适用于 "
            f"{expected_width}x{expected_height}，当前请求为 {width}x{height}"
        )


def _apply_d436_camera_opencv_pinhole_schema(prim: Any) -> bool:
    """在 USD Camera 上写入 Isaac Sim 5.1 支持的完整 OpenCV 内参。"""

    from pxr import Gf

    try:
        schema_applied = prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
    except Exception:
        # IsaacLab 的精简 headless experience 可能未加载 lens-distortion schema。
        # 标准 USD pinhole fallback 已按平均焦距配置，不能因此阻塞 pipeline。
        return False
    if not schema_applied:
        return False
    attributes: tuple[tuple[str, Any], ...] = (
        ("omni:lensdistortion:model", "opencvPinhole"),
        (
            "omni:lensdistortion:opencvPinhole:imageSize",
            Gf.Vec2i(*D436_CAMERA_RESOLUTION_WH),
        ),
        ("omni:lensdistortion:opencvPinhole:fx", D436_CAMERA_FX_PX),
        ("omni:lensdistortion:opencvPinhole:fy", D436_CAMERA_FY_PX),
        ("omni:lensdistortion:opencvPinhole:cx", D436_CAMERA_CX_PX),
        ("omni:lensdistortion:opencvPinhole:cy", D436_CAMERA_CY_PX),
    )
    coefficient_names = (
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "k4",
        "k5",
        "k6",
        "s1",
        "s2",
        "s3",
        "s4",
    )
    attributes += tuple(
        (f"omni:lensdistortion:opencvPinhole:{name}", value)
        for name, value in zip(
            coefficient_names,
            D436_CAMERA_DISTORTION_COEFFICIENTS,
        )
    )
    for attribute_name, value in attributes:
        attribute = prim.GetAttribute(attribute_name)
        if not attribute.IsValid() or not attribute.Set(value):
            return False
    return True


def _enable_d436_lens_distortion_schema() -> dict[str, Any]:
    """显式启用 Isaac Sim camera schema；失败时由标准 USD pinhole 安全回退。"""

    extension_name = "omni.usd.schema.omni_lens_distortion"
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        enabled_before = bool(manager.is_extension_enabled(extension_name))
        if not enabled_before:
            manager.set_extension_enabled_immediate(extension_name, True)
        enabled_after = bool(manager.is_extension_enabled(extension_name))
    except Exception as exc:
        return {
            "requested": True,
            "extension": extension_name,
            "enabled": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "requested": True,
        "extension": extension_name,
        "enabled_before": enabled_before,
        "enabled": enabled_after,
    }


def _make_d436_camera_spawn_function() -> Any:
    """构造先写入标定 schema、再克隆到各环境的 Camera spawner。"""

    from isaaclab.sim.spawners.sensors.sensors import spawn_camera
    from isaaclab.sim.utils import clone

    @clone
    def _spawn_calibrated_d436_camera(
        prim_path: str,
        cfg: Any,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        # 调用未包装实现，确保标定 schema 在 clone 复制 source prim 前写入。
        prim = spawn_camera.__wrapped__(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        _apply_d436_camera_opencv_pinhole_schema(prim)
        return prim

    return _spawn_calibrated_d436_camera


def _overwrite_d436_intrinsic_matrices(matrices: Any) -> int:
    """把 IsaacLab 的居中近似 K 改为渲染器实际使用的标定 K。"""

    shape = tuple(int(value) for value in getattr(matrices, "shape", ()))
    if len(shape) < 2 or shape[-2:] != (3, 3):
        raise ValueError(f"camera intrinsic matrix shape 非法：{shape}")
    matrices[..., :, :] = 0.0
    matrices[..., 0, 0] = D436_CAMERA_FX_PX
    matrices[..., 0, 2] = D436_CAMERA_CX_PX
    matrices[..., 1, 1] = D436_CAMERA_FY_PX
    matrices[..., 1, 2] = D436_CAMERA_CY_PX
    matrices[..., 2, 2] = 1.0
    return math.prod(shape[:-2]) if len(shape) > 2 else 1


def _overwrite_d436_fallback_intrinsic_matrices(matrices: Any) -> int:
    """写入标准 USD pinhole 在当前分辨率下实际使用的中心主点近似 K。"""

    shape = tuple(int(value) for value in getattr(matrices, "shape", ()))
    if len(shape) < 2 or shape[-2:] != (3, 3):
        raise ValueError(f"camera intrinsic matrix shape 非法：{shape}")
    matrices[..., :, :] = 0.0
    matrices[..., 0, 0] = D436_CAMERA_FALLBACK_FX_FY_PX
    matrices[..., 0, 2] = D436_CAMERA_FALLBACK_CX_PX
    matrices[..., 1, 1] = D436_CAMERA_FALLBACK_FX_FY_PX
    matrices[..., 1, 2] = D436_CAMERA_FALLBACK_CY_PX
    matrices[..., 2, 2] = 1.0
    return math.prod(shape[:-2]) if len(shape) > 2 else 1


@dataclass(frozen=True)
class IsaacLabNavigationRuntimeConfig:
    """Isaac Lab 导航环境的固定配置。"""

    task_name: str = "RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0"
    agent_entry_point: str = "rsl_rl_cfg_entry_point"
    checkpoint: Path | None = None
    device: str = "cuda:0"
    standing_command_threshold: float = 0.0
    policy_action_warmup_steps: int = 0
    terrain_prim_path: str = "/World/scene_collision"
    collision_floor_proxy_profile: str | None = None
    visual_prim_path: str = "/World/gauss"
    enable_scene_visual: bool = False
    viewport_camera_prim_path: str = "/World/overview"
    hide_navigation_collision_visual: bool = True
    scene_light_mode: str = "camera"
    camera_light_intensity: float = 3500.0
    camera_light_radius: float = 2.0
    camera_light_name: str = "camera_light"
    # 数据相机严格对齐 DWA：Go2 头部前视、480x640 RGB、每个控制步更新。
    enable_front_camera: bool = False
    front_camera_height: int = 480
    front_camera_width: int = 640
    # 末端相机使用 arm_link6_T_camera_color_optical 手眼标定，只支持 640x480。
    enable_wrist_camera: bool = False
    wrist_camera_height: int = 480
    wrist_camera_width: int = 640
    # 绑定场景中已标定的 overview Camera，不生成或改写相机位姿。
    enable_overview_camera: bool = False
    overview_camera_prim_path: str = "/World/overview"
    overview_camera_height: int = 480
    overview_camera_width: int = 640
    patch_gripper_collision: bool = True
    gripper_collision_robot_root: str = "/World/go2_x5"
    gripper_collision_links: tuple[str, str] = ("arm_link7", "arm_link8")
    gripper_collision_approximation: str = "convexDecomposition"
    gripper_collision_contact_offset: float = 0.002
    gripper_collision_rest_offset: float = 0.0
    patch_apple_collision: bool = True
    apple_collision_root_path: str = "/World"
    apple_collision_keywords: tuple[str, ...] = ("apple", "Apple")
    apple_collision_approximation: str = "convexDecomposition"
    apple_collision_contact_offset: float = 0.001
    apple_collision_rest_offset: float = 0.0
    hide_object_collision_visual: bool = True
    object_collision_visual_root_path: str = "/World"
    object_collision_visual_hide_keywords: tuple[str, ...] = ("Apple_M_Apple",)
    object_collision_visual_keep_keywords: tuple[str, ...] = ("visual_video",)
    arm_joint_names: tuple[str, ...] = (
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
    )
    gripper_joint_names: tuple[str, str] = ("arm_joint7", "arm_joint8")
    gripper_open_position: float = 0.04
    gripper_close_position: float = 0.0
    place_release_clearance_min_m: float = 0.013
    place_pre_clearance_min_m: float = 0.06
    # 这是 cuRobo 规划用的虚拟障碍膨胀，不改变 Isaac 真实物理碰撞体。
    # 恢复稳定 baseline 的 2 cm；此前临时增大到 5 cm 会让规划器产生额外绕行。
    world_collision_padding_m: float = 0.02
    world_collision_vertical_padding_m: float = 0.02
    world_collision_min_size_m: float = 0.01
    world_collision_max_obstacles: int = 16
    world_collision_local_radius_m: float = 1.25
    # 稳定 baseline 会过滤大体积支撑物。把这类物体裁剪后重新加入会形成覆盖
    # 苹果 XY 的巨大 cuboid，迫使侧向 approach 先异常抬高再下降。
    world_collision_clip_large_support_obstacles: bool = False
    world_collision_large_obstacle_clip_half_extent_m: float = 0.45
    show_randomization_debug: bool = False


def _item(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _coerce_xyzyaw(value: Any) -> tuple[float, float, float, float] | None:
    """把 action metadata 中的 root lock 目标解析为 xyzyaw 四元组。"""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )
    except (TypeError, ValueError):
        return None


def _quat_to_yaw(quat_wxyz: Any) -> float:
    w, x, y, z = (_item(value) for value in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """按任务中的固定 RPY 计算姿态，用于只读诊断，不写回 PhysX。"""

    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _quat_angle_error_rad(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _quat_normalize_wxyz(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """归一化标量在前的四元数，拒绝无效零范数输入。"""

    norm = math.sqrt(sum(float(value) ** 2 for value in quat))
    if norm <= 1.0e-12:
        raise ValueError("四元数范数必须大于零。")
    return tuple(float(value) / norm for value in quat)  # type: ignore[return-value]


def _quat_multiply_wxyz(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """计算标量在前四元数的 Hamilton 乘积。"""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_conjugate_wxyz(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """返回单位四元数的共轭。"""

    return (quat[0], -quat[1], -quat[2], -quat[3])


def _quat_rotate_vector_wxyz(
    quat: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """用标量在前四元数旋转三维向量。"""

    normalized = _quat_normalize_wxyz(quat)
    vector_quat = (0.0, float(vector[0]), float(vector[1]), float(vector[2]))
    rotated = _quat_multiply_wxyz(
        _quat_multiply_wxyz(normalized, vector_quat),
        _quat_conjugate_wxyz(normalized),
    )
    return (rotated[1], rotated[2], rotated[3])


def _transform_authored_aabb_to_live_rigid_pose(
    *,
    authored_bbox_min: Any,
    authored_bbox_max: Any,
    authored_rigid_position: Any,
    authored_rigid_quaternion_wxyz: Any,
    live_rigid_position: Any,
    live_rigid_quaternion_wxyz: Any,
) -> dict[str, Any]:
    """把 authored world AABB 先还原到刚体局部系，再映射到 live PhysX pose。

    这样不会假设刚体原点恰好等于 Mesh bbox 中心。输入 AABB 是保守包围盒；
    物体旋转后输出仍是该包围盒刚性变换后的 world AABB。
    """

    import itertools
    import numpy as np

    def _vector(value: Any, *, field_name: str, length: int) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (length,) or not bool(np.all(np.isfinite(vector))):
            raise RuntimeError(f"{field_name} 必须是 {length} 个有限数值")
        return vector

    bbox_min = _vector(authored_bbox_min, field_name="authored_bbox_min", length=3)
    bbox_max = _vector(authored_bbox_max, field_name="authored_bbox_max", length=3)
    if bool(np.any(bbox_max <= bbox_min)):
        raise RuntimeError("authored object bbox 必须具有正尺寸")
    authored_position = _vector(
        authored_rigid_position,
        field_name="authored_rigid_position",
        length=3,
    )
    live_position = _vector(
        live_rigid_position,
        field_name="live_rigid_position",
        length=3,
    )
    authored_quaternion = tuple(
        _vector(
            authored_rigid_quaternion_wxyz,
            field_name="authored_rigid_quaternion_wxyz",
            length=4,
        ).tolist()
    )
    live_quaternion = tuple(
        _vector(
            live_rigid_quaternion_wxyz,
            field_name="live_rigid_quaternion_wxyz",
            length=4,
        ).tolist()
    )
    authored_quaternion = _quat_normalize_wxyz(authored_quaternion)  # type: ignore[arg-type]
    live_quaternion = _quat_normalize_wxyz(live_quaternion)  # type: ignore[arg-type]
    world_to_authored_rigid = _quat_conjugate_wxyz(authored_quaternion)

    authored_corners = [
        np.asarray(corner, dtype=float)
        for corner in itertools.product(
            (bbox_min[0], bbox_max[0]),
            (bbox_min[1], bbox_max[1]),
            (bbox_min[2], bbox_max[2]),
        )
    ]
    rigid_local_corners = [
        np.asarray(
            _quat_rotate_vector_wxyz(
                world_to_authored_rigid,
                tuple((corner - authored_position).tolist()),
            ),
            dtype=float,
        )
        for corner in authored_corners
    ]
    live_world_corners = np.stack(
        [
            live_position
            + np.asarray(
                _quat_rotate_vector_wxyz(
                    live_quaternion,
                    tuple(local_corner.tolist()),
                ),
                dtype=float,
            )
            for local_corner in rigid_local_corners
        ],
        axis=0,
    )
    live_bbox_min = np.min(live_world_corners, axis=0)
    live_bbox_max = np.max(live_world_corners, axis=0)
    live_bbox_center = 0.5 * (live_bbox_min + live_bbox_max)
    authored_bbox_center = 0.5 * (bbox_min + bbox_max)
    center_offset_rigid = np.asarray(
        _quat_rotate_vector_wxyz(
            world_to_authored_rigid,
            tuple((authored_bbox_center - authored_position).tolist()),
        ),
        dtype=float,
    )
    expected_live_center = live_position + np.asarray(
        _quat_rotate_vector_wxyz(
            live_quaternion,
            tuple(center_offset_rigid.tolist()),
        ),
        dtype=float,
    )
    if not bool(np.allclose(live_bbox_center, expected_live_center, atol=1.0e-9)):
        raise RuntimeError("live bbox center 与刚体局部中心偏移变换不一致")
    # authored world AABB 是当前资产已审计的保守 OBB。把它的三个 world 轴经
    # authored-rigid -> live-rigid 旋转，恢复 live PhysX 下的主轴方向。
    live_oriented_axes_world = []
    for authored_world_axis in np.eye(3, dtype=float):
        rigid_local_axis = _quat_rotate_vector_wxyz(
            world_to_authored_rigid,
            tuple(authored_world_axis.tolist()),
        )
        live_oriented_axes_world.append(
            list(
                _quat_rotate_vector_wxyz(
                    live_quaternion,
                    rigid_local_axis,
                )
            )
        )
    authored_bbox_size = bbox_max - bbox_min
    long_axis_index = int(np.argmax(authored_bbox_size))
    return {
        "min_xyz": live_bbox_min.tolist(),
        "max_xyz": live_bbox_max.tolist(),
        "center_xyz": live_bbox_center.tolist(),
        "size_xyz": (live_bbox_max - live_bbox_min).tolist(),
        "authored_bbox_center_xyz": authored_bbox_center.tolist(),
        "authored_rigid_position_xyz": authored_position.tolist(),
        "authored_rigid_quaternion_wxyz": list(authored_quaternion),
        "bbox_center_offset_rigid_xyz": center_offset_rigid.tolist(),
        "live_rigid_position_xyz": live_position.tolist(),
        "live_rigid_quaternion_wxyz": list(live_quaternion),
        "authored_bbox_size_xyz": authored_bbox_size.tolist(),
        "live_oriented_bbox_axes_world": live_oriented_axes_world,
        "long_axis_index": long_axis_index,
        "long_axis_world_xyz": live_oriented_axes_world[long_axis_index],
        "long_axis_length_m": float(authored_bbox_size[long_axis_index]),
        "transform_mode": "authored_world_aabb_via_live_rigid_pose",
    }


def _as_tuple(values: Any) -> tuple[float, ...]:
    if values is None:
        return ()
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(float(value) for value in values)


def _get_or_add_xform_op(xformable: Any, op_type: Any) -> Any:
    """复用引用层已有 xformOp，避免 AddXformOp 因同名属性已存在而失败。"""

    from pxr import UsdGeom

    attr_name_by_type = {
        UsdGeom.XformOp.TypeTranslate: "xformOp:translate",
        UsdGeom.XformOp.TypeOrient: "xformOp:orient",
        UsdGeom.XformOp.TypeScale: "xformOp:scale",
    }
    attr = xformable.GetPrim().GetAttribute(attr_name_by_type[op_type])
    if attr.IsValid():
        return UsdGeom.XformOp(attr)
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if op_type == UsdGeom.XformOp.TypeOrient:
        return xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    if op_type == UsdGeom.XformOp.TypeScale:
        return xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
    raise ValueError(f"unsupported xform op type: {op_type}")


def _set_translate_op(op: Any, xyz: tuple[float, float, float]) -> None:
    """按已有 op 精度写入 translate，避免把 double op 写成 float 值。"""

    from pxr import Gf, UsdGeom

    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Vec3d(*xyz))
    else:
        op.Set(Gf.Vec3f(*xyz))


def _set_orient_op(op: Any, quat_wxyz: tuple[float, float, float, float]) -> None:
    """按已有 op 精度写入 orient，保持引用资产 xform 属性可复用。"""

    from pxr import Gf, UsdGeom

    w, x, y, z = quat_wxyz
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _set_scale_op(op: Any, scale: tuple[float, float, float]) -> None:
    """按已有 op 精度写入资产 scale。"""

    from pxr import Gf, UsdGeom

    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Vec3d(*scale))
    else:
        op.Set(Gf.Vec3f(*scale))


def _xformable_world_pose(xformable: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """读取完整 xform stack 组合后的世界位姿，并剥离 scale。"""

    from pxr import Gf, Usd

    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    # 苹果根节点带 0.08 scale；直接 ExtractRotationQuat 会把 scale 混入
    # 四元数并产生非单位结果。Gf.Transform 会先完成 TRS 分解。
    rotation = Gf.Transform(matrix).GetRotation().GetQuat()
    imaginary = rotation.GetImaginary()
    return (
        tuple(float(translation[index]) for index in range(3)),
        (
            float(rotation.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _path_overlaps(path_a: str, path_b: str) -> bool:
    """判断两个 USD path 是否存在父子或相等关系。"""

    a = path_a.rstrip("/")
    b = path_b.rstrip("/")
    if not a or not b or a == "/" or b == "/":
        return False
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _path_is_under(path: str, parent_path: str) -> bool:
    """判断 path 是否等于或位于 parent_path 下。"""

    parent = parent_path.rstrip("/")
    return bool(parent) and (path == parent or path.startswith(parent + "/"))


def _prim_keyword_match_text(prim: Any) -> str:
    """汇总 prim path 与资产元数据，兼容 baseline 的干扰物识别方式。"""

    pieces = [str(prim.GetPath()), str(prim.GetName())]
    for metadata_name in ("references", "payload", "payloads", "assetInfo"):
        try:
            value = prim.GetMetadata(metadata_name)
        except Exception:
            value = None
        if value:
            pieces.append(str(value))
    return " ".join(pieces).lower()


def _dedupe_root_paths(paths: list[str]) -> list[str]:
    """只保留最浅层根路径，避免同一资产的子 prim 被重复处理。"""

    roots: list[str] = []
    for path in sorted(set(paths), key=lambda item: (item.count("/"), item)):
        if any(_path_is_under(path, root) for root in roots):
            continue
        roots.append(path)
    return roots


def _path_is_excluded_by_roots(path: str, excluded_roots: tuple[str, ...]) -> bool:
    """判断碰撞 prim 是否属于已隐藏的非任务物体。"""

    return any(_path_is_under(path, root) for root in excluded_roots)


def _distance_point_to_aabb_xy(point_xy: Any, bbox_min: Any, bbox_max: Any) -> float:
    """计算 XY 平面中点到 AABB 的距离；点在投影内部时为 0。"""

    import numpy as np

    point_xy = np.asarray(point_xy, dtype=float)
    min_xy = np.asarray(bbox_min[:2], dtype=float)
    max_xy = np.asarray(bbox_max[:2], dtype=float)
    delta = np.maximum(np.maximum(min_xy - point_xy, point_xy - max_xy), 0.0)
    return float(np.linalg.norm(delta))


def _derive_mesh_truth_place_pose(
    *,
    raw_place: dict[str, Any],
    receptacle_support_report: dict[str, Any] | None,
    pick_object_bbox: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """由运行时支撑 Mesh 与抓取前物体 bbox 推导最终物体中心。

    ``place_pose_world`` 仍作为 Phase-0 人工标注的一致性基准，但启用该模式后
    不再直接作为运行时 XYZ 来源。任何支撑几何、物体尺寸或标注点漂移都会显式
    失败，避免 CuRobo 静默使用过期目标。
    """

    mesh_config = raw_place.get("mesh_truth_target")
    if mesh_config is None:
        return None
    if not isinstance(mesh_config, dict):
        raise RuntimeError("task.place.mesh_truth_target 必须是对象")
    if not bool(mesh_config.get("enabled", False)):
        return None

    expected_sources = {
        "target_xy_source": "runtime_placement_region_center",
        "support_surface_source": "runtime_target_support_bbox_top",
        "object_support_extent_source": (
            "pick_live_object_bbox_center_to_min_z"
        ),
    }
    for field_name, expected_value in expected_sources.items():
        if mesh_config.get(field_name) != expected_value:
            raise RuntimeError(
                "task.place.mesh_truth_target source contract mismatch: "
                f"{field_name}={mesh_config.get(field_name)!r}, "
                f"expected={expected_value!r}"
            )
    if mesh_config.get("visual_localization_required") is not False:
        raise RuntimeError(
            "Mesh-truth place target 必须显式配置 visual_localization_required=false"
        )

    if not isinstance(receptacle_support_report, dict):
        raise RuntimeError("Mesh-truth place target 缺少运行时 receptacle support 报告")
    if receptacle_support_report.get("configured") is not True:
        raise RuntimeError("Mesh-truth place target 的 receptacle support 未配置")
    if receptacle_support_report.get("geometry_verified") is not True:
        raise RuntimeError("Mesh-truth place target 的运行时 support geometry 未验证")
    region_report = receptacle_support_report.get("placement_region_report")
    if not isinstance(region_report, dict) or region_report.get("verified") is not True:
        raise RuntimeError("Mesh-truth place target 的 placement region 未验证")

    placement_region = receptacle_support_report.get("placement_region")
    if not isinstance(placement_region, dict):
        raise RuntimeError("运行时 support 报告缺少 placement_region")
    if placement_region.get("frame") != "world":
        raise RuntimeError("Mesh-truth placement_region.frame 必须是 world")

    def _finite_float(value: Any, *, field_name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{field_name} 必须是有限数值") from exc
        if not math.isfinite(parsed):
            raise RuntimeError(f"{field_name} 必须是有限数值")
        return parsed

    def _finite_xyz(value: Any, *, field_name: str) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise RuntimeError(f"{field_name} 必须包含三个有限数值")
        return tuple(
            _finite_float(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )

    x_min = _finite_float(placement_region.get("x_min"), field_name="placement_region.x_min")
    x_max = _finite_float(placement_region.get("x_max"), field_name="placement_region.x_max")
    y_min = _finite_float(placement_region.get("y_min"), field_name="placement_region.y_min")
    y_max = _finite_float(placement_region.get("y_max"), field_name="placement_region.y_max")
    region_surface_z = _finite_float(
        placement_region.get("z_surface"),
        field_name="placement_region.z_surface",
    )
    if not x_min < x_max or not y_min < y_max:
        raise RuntimeError("Mesh-truth placement region 边界顺序无效")

    support_bbox_min = _finite_xyz(
        receptacle_support_report.get("world_bbox_min_xyz"),
        field_name="support_report.world_bbox_min_xyz",
    )
    support_bbox_max = _finite_xyz(
        receptacle_support_report.get("world_bbox_max_xyz"),
        field_name="support_report.world_bbox_max_xyz",
    )
    support_surface_z = _finite_float(
        receptacle_support_report.get("support_surface_z"),
        field_name="support_report.support_surface_z",
    )
    geometry_tolerance_m = _finite_float(
        mesh_config.get("configured_pose_consistency_tolerance_m", 1.0e-6),
        field_name=(
            "task.place.mesh_truth_target."
            "configured_pose_consistency_tolerance_m"
        ),
    )
    if geometry_tolerance_m < 0.0:
        raise RuntimeError("Mesh-truth geometry tolerance 不能为负数")
    if abs(support_surface_z - support_bbox_max[2]) > geometry_tolerance_m:
        raise RuntimeError("运行时 support_surface_z 与 support bbox 顶面不一致")
    if abs(region_surface_z - support_surface_z) > geometry_tolerance_m:
        raise RuntimeError("运行时 placement region 与 support bbox 顶面不一致")

    if not isinstance(pick_object_bbox, dict):
        raise RuntimeError("Mesh-truth place target 缺少抓取前实时物体 bbox")
    if pick_object_bbox.get("center_source") != "live_physx_object_pose":
        raise RuntimeError(
            "Mesh-truth place target 要求抓取前 bbox center 来自 live PhysX pose"
        )
    pick_bbox_min = _finite_xyz(
        pick_object_bbox.get("min_xyz"),
        field_name="pick_object_bbox.min_xyz",
    )
    pick_bbox_max = _finite_xyz(
        pick_object_bbox.get("max_xyz"),
        field_name="pick_object_bbox.max_xyz",
    )
    pick_bbox_center = _finite_xyz(
        pick_object_bbox.get("center_xyz"),
        field_name="pick_object_bbox.center_xyz",
    )
    object_bbox_center_to_min_z_m = pick_bbox_center[2] - pick_bbox_min[2]
    if object_bbox_center_to_min_z_m <= 0.0:
        raise RuntimeError("抓取前物体 bbox 的 center-to-min-z 必须为正数")
    expected_extent_field = "expected_object_bbox_center_to_min_z_m"
    if expected_extent_field in mesh_config:
        expected_extent_raw = mesh_config.get(expected_extent_field)
    else:
        # 兼容早期任务配置；新任务应使用准确表达 bbox 几何的字段名。
        expected_extent_field = "expected_object_center_to_support_m"
        expected_extent_raw = mesh_config.get(expected_extent_field)
    expected_extent_m = _finite_float(
        expected_extent_raw,
        field_name=f"task.place.mesh_truth_target.{expected_extent_field}",
    )
    extent_tolerance_m = _finite_float(
        mesh_config.get("object_extent_tolerance_m"),
        field_name="task.place.mesh_truth_target.object_extent_tolerance_m",
    )
    if extent_tolerance_m < 0.0:
        raise RuntimeError("Mesh-truth object extent tolerance 不能为负数")
    extent_error_m = abs(object_bbox_center_to_min_z_m - expected_extent_m)
    if extent_error_m > extent_tolerance_m:
        raise RuntimeError(
            "抓取前物体 Mesh extent 与任务标定不一致: "
            f"error_m={extent_error_m}, tolerance_m={extent_tolerance_m}"
        )

    configured_pose = raw_place.get("place_pose_world")
    if not isinstance(configured_pose, dict):
        raise RuntimeError("Mesh-truth place target 要求 place_pose_world 作为审计基准")
    configured_xyz = tuple(
        _finite_float(configured_pose.get(axis), field_name=f"place_pose_world.{axis}")
        for axis in ("x", "y", "z")
    )
    calibrated_xyz = (
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
        support_surface_z + expected_extent_m,
    )
    configured_pose_errors_m = {
        axis: abs(calibrated_xyz[index] - configured_xyz[index])
        for index, axis in enumerate(("x", "y", "z"))
    }
    configured_pose_max_abs_error_m = max(configured_pose_errors_m.values())
    if configured_pose_max_abs_error_m > geometry_tolerance_m:
        raise RuntimeError(
            "Mesh-derived place pose 与 configured place pose drifted: "
            f"max_abs_error_m={configured_pose_max_abs_error_m}, "
            f"tolerance_m={geometry_tolerance_m}"
        )

    # 运行目标使用本 episode 抓取前的 live bbox；静态标注只与离线标定半高比较。
    # 因此 PhysX settle 引起、且仍在 object_extent_tolerance_m 内的微小变化不会被
    # 更严格的静态标注漂移门禁误拒绝。
    derived_xyz = (
        calibrated_xyz[0],
        calibrated_xyz[1],
        support_surface_z + object_bbox_center_to_min_z_m,
    )
    runtime_to_configured_errors_m = {
        axis: abs(derived_xyz[index] - configured_xyz[index])
        for index, axis in enumerate(("x", "y", "z"))
    }

    payload = dict(configured_pose)
    payload.update(
        {
            "x": derived_xyz[0],
            "y": derived_xyz[1],
            "z": derived_xyz[2],
        }
    )
    report = {
        "configured": True,
        "enabled": True,
        "verified": True,
        "visual_localization_required": False,
        "visual_localization_used": False,
        **expected_sources,
        "target_receptacle_prim_path": receptacle_support_report.get(
            "target_receptacle_prim_path"
        ),
        "target_support_prim_path": receptacle_support_report.get(
            "target_support_prim_path"
        ),
        "support_report_source": receptacle_support_report.get("source"),
        "support_geometry_verified": True,
        "support_bbox_world": {
            "min_xyz": list(support_bbox_min),
            "max_xyz": list(support_bbox_max),
        },
        "placement_region_world": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_surface": region_surface_z,
        },
        "pick_object_bbox_world": {
            "min_xyz": list(pick_bbox_min),
            "max_xyz": list(pick_bbox_max),
            "center_xyz": list(pick_bbox_center),
            "center_source": pick_object_bbox.get("center_source"),
        },
        "object_bbox_center_to_min_z_m": object_bbox_center_to_min_z_m,
        "expected_object_bbox_center_to_min_z_m": expected_extent_m,
        "expected_object_extent_config_field": expected_extent_field,
        "object_extent_error_m": extent_error_m,
        "object_extent_tolerance_m": extent_tolerance_m,
        "object_extent_consistency_verified": True,
        "derived_place_pose_world": {
            "x": derived_xyz[0],
            "y": derived_xyz[1],
            "z": derived_xyz[2],
            "roll": float(payload.get("roll", 0.0)),
            "pitch": float(payload.get("pitch", 0.0)),
            "yaw": float(payload.get("yaw", 0.0)),
        },
        "configured_place_pose_world": {
            "x": configured_xyz[0],
            "y": configured_xyz[1],
            "z": configured_xyz[2],
        },
        "calibrated_place_pose_world": {
            "x": calibrated_xyz[0],
            "y": calibrated_xyz[1],
            "z": calibrated_xyz[2],
        },
        "configured_pose_errors_m": configured_pose_errors_m,
        "configured_pose_max_abs_error_m": configured_pose_max_abs_error_m,
        "configured_pose_consistency_tolerance_m": geometry_tolerance_m,
        "configured_pose_consistency_verified": True,
        "runtime_to_configured_pose_errors_m": runtime_to_configured_errors_m,
        "runtime_to_configured_pose_max_abs_error_m": max(
            runtime_to_configured_errors_m.values()
        ),
        "xyz_source": "runtime_mesh_truth",
        "orientation_source": "task_place_pose_world",
    }
    return payload, report


def _resolve_mesh_truth_manipulation_contract(
    raw_task: dict[str, Any],
) -> dict[str, Any]:
    """解析运行时 Mesh-truth 操作目标合同；旧任务默认不要求。"""

    raw_config = raw_task.get("mesh_truth_manipulation_targets")
    if raw_config is None:
        return {"configured": False, "required": False}
    if not isinstance(raw_config, dict):
        raise RuntimeError("task.mesh_truth_manipulation_targets 必须是对象")
    required = bool(raw_config.get("required", False))
    report = {
        **raw_config,
        "configured": True,
        "required": required,
    }
    if not required:
        return report
    expected = {
        "visual_localization_required": False,
        "pick_tcp_source": "runtime_live_object_bbox",
        "place_tcp_source": (
            "runtime_receptacle_bbox_plus_pick_object_bbox_plus_current_tcp_offset"
        ),
    }
    mismatches = {
        key: {"actual": raw_config.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if raw_config.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "task.mesh_truth_manipulation_targets contract mismatch: "
            f"{mismatches}"
        )
    return report


def _resolve_pick_grasp_mode(raw_task: dict[str, Any]) -> dict[str, str]:
    """解析 task.pick.grasp_mode，并保留 auto 的旧 side 兼容语义。"""

    raw_pick = raw_task.get("pick")
    raw_pick = raw_pick if isinstance(raw_pick, dict) else {}
    requested = str(raw_pick.get("grasp_mode") or "auto").strip().lower()
    requested = requested.replace("-", "_")
    if requested not in {"auto", "side", "top_down"}:
        raise RuntimeError(
            "task.pick.grasp_mode 必须是 auto、side 或 top_down，"
            f"当前为 {requested!r}"
        )
    resolved = "side" if requested == "auto" else requested
    return {"requested": requested, "resolved": resolved}


def _point_inside_aabb(point: Any, bbox_min: Any, bbox_max: Any, *, margin: float = 0.0) -> bool:
    """判断 point 是否在 AABB 内部。"""

    import numpy as np

    point = np.asarray(point, dtype=float)
    return bool(
        np.all(point >= np.asarray(bbox_min, dtype=float) - margin)
        and np.all(point <= np.asarray(bbox_max, dtype=float) + margin)
    )


def _sanitize_obstacle_name(prim_path: str, index: int) -> str:
    """把 USD path 转成 cuRobo obstacle name。"""

    safe = prim_path.strip("/").replace("/", "_").replace(":", "_")
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in safe)
    return f"obs_{index:03d}_{safe[-80:]}"


def _collision_vector(
    value: Any,
    *,
    field_name: str,
    length: int,
) -> Any:
    """严格解析任务中的碰撞向量，禁止 NaN 或隐式补维。"""

    import numpy as np

    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not bool(np.all(np.isfinite(vector))):
        raise RuntimeError(
            f"{field_name} 必须是 {length} 个有限数值，实际为 {value!r}"
        )
    return vector


def _task_world_collision_cuboids(
    *,
    raw_task: dict[str, Any],
    phase: str,
    T_world_base: Any,
    reference_point: Any,
    padding_xy_m: float,
    padding_z_m: float,
) -> list[dict[str, Any]]:
    """把任务显式声明的局部 world cuboid 转成 cuRobo base frame。

    该入口用于没有独立桌子 prim 的合并碰撞场景。只有任务显式配置时才生效，
    因而不会改变旧场景的默认 USD CollisionAPI 导出行为。
    """

    import numpy as np

    from source.manipulation.current_state_curobo import (
        pose_dict_from_matrix,
        pose_to_matrix,
    )

    if phase not in {"pick", "place"}:
        raise RuntimeError(f"不支持的任务碰撞阶段: {phase}")
    raw_phase = raw_task.get(phase) or {}
    if not isinstance(raw_phase, dict):
        raise RuntimeError(f"task.{phase} 必须是对象")
    raw_config = raw_phase.get("curobo_world_collision")
    if raw_config is None:
        return []
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"task.{phase}.curobo_world_collision 必须是对象")

    required = bool(raw_config.get("required", False))
    enabled = bool(raw_config.get("enabled", True))
    if not enabled:
        if required:
            raise RuntimeError(
                f"task.{phase}.curobo_world_collision 同时配置 required=true 和 enabled=false"
            )
        return []
    raw_cuboids = raw_config.get("cuboids_world", [])
    if not isinstance(raw_cuboids, list):
        raise RuntimeError(
            f"task.{phase}.curobo_world_collision.cuboids_world 必须是数组"
        )
    if required and not raw_cuboids:
        raise RuntimeError(f"task.{phase} 要求 CuRobo world collision，但未提供 cuboid")

    T_world_base = np.asarray(T_world_base, dtype=float)
    if T_world_base.shape != (4, 4) or not bool(np.all(np.isfinite(T_world_base))):
        raise RuntimeError("T_world_base 必须是有限的 4x4 矩阵")
    T_base_world = np.linalg.inv(T_world_base)
    reference_point = _collision_vector(
        reference_point,
        field_name="reference_point",
        length=3,
    )
    output: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for index, raw_cuboid in enumerate(raw_cuboids):
        config_path = (
            f"task.{phase}.curobo_world_collision.cuboids_world[{index}]"
        )
        if not isinstance(raw_cuboid, dict):
            raise RuntimeError(f"{config_path} 必须是对象")
        name = str(raw_cuboid.get("name") or "").strip()
        if not name:
            raise RuntimeError(f"{config_path}.name 不能为空")
        if name in seen_names:
            raise RuntimeError(f"{config_path}.name 重复: {name}")
        seen_names.add(name)
        if str(raw_cuboid.get("frame", "world")) != "world":
            raise RuntimeError(f"{config_path}.frame 当前只支持 world")

        center = _collision_vector(
            raw_cuboid.get("center_xyz"),
            field_name=f"{config_path}.center_xyz",
            length=3,
        )
        dims = _collision_vector(
            raw_cuboid.get("dims_xyz"),
            field_name=f"{config_path}.dims_xyz",
            length=3,
        )
        if bool(np.any(dims <= 0.0)):
            raise RuntimeError(f"{config_path}.dims_xyz 必须全部大于 0")
        quaternion = _collision_vector(
            raw_cuboid.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
            field_name=f"{config_path}.quaternion_wxyz",
            length=4,
        )
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 1.0e-8:
            raise RuntimeError(f"{config_path}.quaternion_wxyz 不能是零四元数")
        quaternion = quaternion / quaternion_norm
        padding_mode = str(raw_cuboid.get("padding_mode", "symmetric"))
        if padding_mode not in {"none", "symmetric", "preserve_top"}:
            raise RuntimeError(
                f"{config_path}.padding_mode 必须是 none/symmetric/preserve_top"
            )

        T_world_raw = pose_to_matrix(center, quaternion)
        T_world_obstacle = T_world_raw.copy()
        padded_dims = dims.copy()
        if padding_mode == "symmetric":
            padded_dims += np.asarray(
                [2.0 * padding_xy_m, 2.0 * padding_xy_m, 2.0 * padding_z_m],
                dtype=float,
            )
        elif padding_mode == "preserve_top":
            # 支撑面膨胀只向局部 -Z 延伸，不能把人工标定的桌面虚拟抬高。
            padded_dims += np.asarray(
                [2.0 * padding_xy_m, 2.0 * padding_xy_m, padding_z_m],
                dtype=float,
            )
            T_world_obstacle[:3, 3] -= (
                T_world_raw[:3, 2] * float(padding_z_m) * 0.5
            )

        T_base_obstacle = T_base_world @ T_world_obstacle
        source_prim_path = str(raw_cuboid.get("source_prim_path") or "")
        semantic_role = str(raw_cuboid.get("semantic_role") or "task_obstacle")
        output.append(
            {
                "prim_path": source_prim_path or f"task://{phase}/{name}",
                "type": "task_configured_world_cuboid",
                "task_configured": True,
                "task_collision_required": required,
                "task_collision_id": f"{phase}:{name}",
                "task_collision_name": name,
                "task_config_path": config_path,
                "phase": phase,
                "semantic_role": semantic_role,
                "distance_to_reference_xy_m": float(
                    np.linalg.norm(T_world_obstacle[:2, 3] - reference_point[:2])
                ),
                "dims_xyz": padded_dims.tolist(),
                "raw_dims_xyz": dims.tolist(),
                "pose_world": pose_dict_from_matrix(T_world_obstacle),
                "raw_pose_world": pose_dict_from_matrix(T_world_raw),
                "pose_base": pose_dict_from_matrix(T_base_obstacle),
                "padding_m": float(padding_xy_m),
                "padding_xy_m": float(padding_xy_m),
                "padding_z_m": float(padding_z_m),
                "padding_mode": padding_mode,
                "source_prim_path": source_prim_path or None,
                "source": dict(raw_cuboid.get("source") or {}),
                "clipped_from_large_obstacle": False,
            }
        )

    return output


def _collision_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, str]:
    """优先保留任务物体附近的碰撞体，避免 stage 遍历顺序挤掉桌面。"""

    prim_path = str(candidate["prim_path"])
    semantic_role = str(candidate.get("semantic_role") or "").lower()
    if candidate.get("task_collision_required"):
        collision_priority = 0
    elif candidate.get("task_configured"):
        collision_priority = 1
    elif semantic_role == "table_support" or any(
        keyword in prim_path.lower()
        for keyword in ("table", "tabletop", "desk", "counter")
    ):
        collision_priority = 2
    else:
        collision_priority = 3
    return (
        collision_priority,
        float(candidate["distance_to_reference_xy_m"]),
        prim_path,
    )


def _retarget_height_scanners(scene_cfg: Any, terrain_mesh_prim_path: str) -> tuple[str, ...]:
    """让地形高度扫描器跟随 runtime 实际导入的碰撞 Mesh。"""

    updated: list[str] = []
    for sensor_name in ("height_scanner", "height_scanner_base"):
        sensor_cfg = getattr(scene_cfg, sensor_name, None)
        if sensor_cfg is None:
            continue
        sensor_cfg.mesh_prim_paths = [terrain_mesh_prim_path]
        updated.append(sensor_name)
    return tuple(updated)


def _resolve_rigid_body_prim_path(stage: Any, object_root_path: str) -> str:
    """解析物体子树中的真实动态刚体，避免把外层定位 Xform 变成父刚体。"""

    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(object_root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"task object prim is unavailable: {object_root_path}")
    candidates: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        enabled_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if enabled_attr and enabled_attr.IsValid() and enabled_attr.Get() is False:
            continue
        candidates.append(str(prim.GetPath()))
    if not candidates:
        raise RuntimeError(f"no enabled RigidBodyAPI exists under {object_root_path}")
    return object_root_path if object_root_path in candidates else candidates[0]


def _collision_cuboid_diagnostics(
    cuboids: list[dict[str, Any]],
) -> dict[str, Any]:
    """记录桌面碰撞体是否进入 cuRobo 局部 world，避免只看规划结果猜测。"""

    table_keywords = ("table", "tabletop", "desk", "counter")
    table_paths = [
        str(cuboid.get("prim_path") or "")
        for cuboid in cuboids
        if str(cuboid.get("semantic_role") or "").lower() == "table_support"
        or any(
            keyword in str(cuboid.get("prim_path") or "").lower()
            for keyword in table_keywords
        )
    ]
    return {
        "collision_cuboids_table_present": bool(table_paths),
        "table_collision_prim_paths": table_paths,
        "task_configured_collision_ids": [
            str(cuboid.get("task_collision_id"))
            for cuboid in cuboids
            if cuboid.get("task_configured")
        ],
        "nearest_collision_prim_paths": [
            str(cuboid.get("prim_path") or "") for cuboid in cuboids[:8]
        ],
    }


class IsaacLabNavigationRuntime:
    """封装 Isaac Lab env，并把所有物理推进集中到 ``step``。"""

    def __init__(
        self,
        *,
        simulation_app: Any,
        project_root: str | Path,
        config: IsaacLabNavigationRuntimeConfig | None = None,
    ):
        self._simulation_app = simulation_app
        self._project_root = Path(project_root).expanduser().resolve()
        self._config = config or IsaacLabNavigationRuntimeConfig()
        self._env = None
        self._runtime = None
        self._adapter = None
        self._object = None
        self._settled_object_pose: tuple[float, ...] | None = None
        self._episode_spec: EpisodeSpec | None = None
        self._step_calls = 0
        self._closed = False
        self._action_prepared = False
        self._environment_terminated = False
        self._last_action = RobotAction.idle()
        self._pending_arm_tracking_target: dict[str, Any] | None = None
        self._manipulation_base_lock_active = False
        self._manipulation_support_joint_lock_active = False
        self._navigation_joint_pose_lock_active = False
        self._navigation_object_follow_active = False
        self._navigation_object_relative_pose: tuple[float, ...] | None = None
        self._navigation_object_follow_root_target: tuple[float, ...] | None = None
        self._navigation_object_follow_target_pose: tuple[float, ...] | None = None
        self._hidden_distractor_root_paths: tuple[str, ...] = ()
        self._viewport_config_attempts = 0
        self._metadata: dict[str, Any] = {
            "simulation_ready": False,
            "execution_provenance_verified": True,
            "used_base_teleport": False,
            "used_direct_joint_state": False,
            "used_object_teleport": False,
            "used_kinematic_object_follow": False,
            "used_visual_replay": False,
            "used_manipulation_base_lock": False,
            "used_manipulation_support_joint_lock": False,
            "used_navigation_base_lock": False,
            "used_navigation_support_joint_lock": False,
            "used_navigation_joint_pose_lock": False,
            "navigation_object_follow_active": False,
            "navigation_object_follow_apply_count": 0,
            "last_navigation_object_follow_report": None,
            "manipulation_base_lock_active": False,
            "manipulation_base_lock_apply_count": 0,
            "last_manipulation_base_lock_report": None,
            "last_navigation_base_lock_report": None,
            "manipulation_support_joint_lock_active": False,
            "manipulation_support_joint_lock_apply_count": 0,
            "last_manipulation_support_joint_lock_report": None,
            "last_navigation_support_joint_lock_report": None,
            "navigation_joint_pose_lock_active": False,
            "navigation_joint_pose_lock_apply_count": 0,
            "last_navigation_joint_pose_lock_report": None,
            "arm_joint_position_target_apply_count": 0,
            "last_arm_joint_position_target_report": None,
            "gripper_joint_position_target_apply_count": 0,
            "last_gripper_joint_position_target_report": None,
            "last_arm_action_report": None,
            "last_joint_action_report": None,
            "last_arm_tracking_report": None,
            "arm_tracking_peak_report": None,
            "arm_tracking_report": {
                "sample_count": 0,
                "max_abs_error": 0.0,
                "peak_report": None,
            },
            "arm_tracking_sample_count": 0,
            "arm_tracking_max_abs_error": 0.0,
            "last_gripper_action_report": None,
            "joint_action_apply_count": 0,
            "arm_joint_action_apply_count": 0,
            "gripper_joint_action_apply_count": 0,
            "gripper_close_apply_count": 0,
            "gripper_open_apply_count": 0,
            "world_count": 1,
            "opened_stage_count": 1,
        }

    def build(self, episode_spec: EpisodeSpec) -> None:
        if self._closed:
            raise RuntimeError("simulation runtime is closed")
        if self._env is not None:
            raise RuntimeError("Isaac Lab environment has already been built")
        self._episode_spec = episode_spec
        self._build_environment(episode_spec)
        if self._config.show_randomization_debug:
            import omni.usd

            from source.diagnostics import create_randomization_debug

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("randomization debug requires an open USD stage")
            self._metadata["randomization_debug"] = create_randomization_debug(
                stage,
                episode_spec.raw_task,
            )
        self._metadata.update(
            {
                "simulation_ready": True,
                "simulation_backend": "isaaclab_manager_based_rl",
                "scene_usd": str(self._resolve_path(episode_spec.scene_usd)),
                "nav_map": str(self._resolve_path(episode_spec.nav_map)),
                "articulation_prim_path": self._robot_prim_path(),
                "joint_names": list(getattr(self._adapter.robot, "joint_names", [])),
                "control_dt": float(self._runtime.step_dt),
                "physics_dt": float(self._runtime.physics_dt),
                "decimation": int(self._runtime.cfg.decimation),
                "checkpoint": str(self._resolve_checkpoint()),
            }
        )

    def reset(self, episode_spec: EpisodeSpec, *, seed: int) -> None:
        self._require_ready()
        self._episode_spec = episode_spec
        self._settled_object_pose = None
        reset_policy_warmup = getattr(self._adapter, "reset_policy_warmup", None)
        if callable(reset_policy_warmup):
            reset_policy_warmup()
        observations, _extras = self._runtime.reset(seed=seed)
        self._adapter.update_observations(self._to_tensor_dict(observations))
        self._environment_terminated = False
        self._action_prepared = False
        self._last_action = RobotAction.idle(source="episode_reset")
        self._adapter.set_arm_joint_target(None)
        self._adapter.set_direct_arm_action_override(False)
        self._adapter.set_gripper_joint_target(None)
        self._adapter.set_base_pose_lock(False)
        if hasattr(self._adapter, "set_support_joint_lock"):
            self._adapter.set_support_joint_lock(False)
        if hasattr(self._adapter, "set_navigation_joint_pose_lock"):
            self._adapter.set_navigation_joint_pose_lock(False)
        self._pending_arm_tracking_target = None
        self._manipulation_base_lock_active = False
        self._manipulation_support_joint_lock_active = False
        self._navigation_joint_pose_lock_active = False
        self._navigation_object_follow_active = False
        self._navigation_object_relative_pose = None
        self._navigation_object_follow_root_target = None
        self._navigation_object_follow_target_pose = None
        self._metadata.pop("wrist_camera_object_clearance_report", None)
        self._metadata.update(
            {
                "seed": int(seed),
                "episode_reset_complete": True,
                "used_episode_reset_pose": True,
                "last_arm_action_report": None,
                "last_joint_action_report": None,
                "last_arm_tracking_report": None,
                "arm_tracking_peak_report": None,
                "arm_tracking_report": {
                    "sample_count": 0,
                    "max_abs_error": 0.0,
                    "peak_report": None,
                },
                "arm_tracking_sample_count": 0,
                "arm_tracking_max_abs_error": 0.0,
                "last_gripper_action_report": None,
                "joint_action_apply_count": 0,
                "arm_joint_action_apply_count": 0,
                "gripper_joint_action_apply_count": 0,
                "gripper_close_apply_count": 0,
                "gripper_open_apply_count": 0,
                "used_direct_joint_state": False,
                "used_manipulation_base_lock": False,
                "used_manipulation_support_joint_lock": False,
                "used_navigation_base_lock": False,
                "used_navigation_support_joint_lock": False,
                "used_navigation_joint_pose_lock": False,
                "used_object_teleport": False,
                "used_kinematic_object_follow": False,
                "navigation_object_follow_active": False,
                "navigation_object_follow_apply_count": 0,
                "last_navigation_object_follow_report": None,
                "manipulation_base_lock_active": False,
                "manipulation_base_lock_apply_count": 0,
                "last_manipulation_base_lock_report": None,
                "last_navigation_base_lock_report": None,
                "manipulation_support_joint_lock_active": False,
                "manipulation_support_joint_lock_apply_count": 0,
                "last_manipulation_support_joint_lock_report": None,
                "last_navigation_support_joint_lock_report": None,
                "navigation_joint_pose_lock_active": False,
                "navigation_joint_pose_lock_apply_count": 0,
                "last_navigation_joint_pose_lock_report": None,
                # reset 事件写入初始位姿，不属于导航执行中的 teleport。
                "reset_pose_source": "isaaclab_reset_event",
            }
        )
        self._metadata["object_reset_for_navigation_report"] = (
            self._reset_object_pose_and_motion(
                episode_spec,
                sleep_until_contact=True,
                reason="episode_reset_before_navigation",
            )
        )
        self._metadata["object_pose_debug_after_reset"] = self._object_initial_pose_diagnostic(
            episode_spec,
            label="after_runtime_reset",
        )
        self.refresh_viewport(reason="reset_episode")

    def read(self) -> SimulationState:
        self._retry_viewport_after_stage_updates()
        if self._adapter is None:
            return SimulationState(
                step_index=self._step_calls,
                timestamp=float(self._step_calls) * 0.02,
                robot_root_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                robot_root_velocity=(0.0,) * 6,
                metadata=dict(self._metadata),
            )

        robot = self._adapter.robot
        self._consume_pending_arm_tracking_target(robot)
        root_position = robot.data.root_pos_w[0]
        root_quaternion = robot.data.root_quat_w[0]
        root_linear = robot.data.root_lin_vel_w[0]
        root_angular = robot.data.root_ang_vel_w[0]
        object_pose, object_velocity = self._read_object_state()
        tcp_pose = self._read_tcp_pose()
        self._update_wrist_camera_object_clearance(
            tcp_pose_world=tcp_pose,
            object_pose_world=object_pose,
        )
        metadata = {
            **self._metadata,
            "environment_terminated": self._environment_terminated,
            "joint_names": tuple(str(name) for name in getattr(robot, "joint_names", ())),
            "base_pose_xyyaw": (
                _item(root_position[0]),
                _item(root_position[1]),
                _quat_to_yaw(root_quaternion),
            ),
            "body_velocity": self._adapter.get_base_velocity_full(),
            "body_linear_velocity": _as_tuple(robot.data.root_lin_vel_b[0]),
            "last_action_source": self._last_action.source,
        }
        metadata.update(self._adapter.diagnostics())
        return SimulationState(
            step_index=self._step_calls,
            timestamp=float(self._step_calls) * float(self._runtime.step_dt),
            robot_root_pose=(
                *_as_tuple(root_position),
                *_as_tuple(root_quaternion),
            ),
            robot_root_velocity=(
                *_as_tuple(root_linear),
                *_as_tuple(root_angular),
            ),
            joint_positions=_as_tuple(robot.data.joint_pos[0]),
            joint_velocities=_as_tuple(robot.data.joint_vel[0]),
            tcp_pose=tcp_pose,
            object_pose=object_pose,
            object_velocity=object_velocity,
            camera_images=self._read_camera_images(),
            metadata=metadata,
        )

    def apply(self, action: RobotAction) -> None:
        self._require_ready()
        self._configure_manipulation_base_lock(action)
        arm_report = self._stage_arm_target(action)
        gripper_report = self._stage_gripper_target(action)
        if self._environment_terminated:
            self._adapter.apply_base_command(0.0, 0.0, 0.0)
        else:
            self._adapter.apply_base_command(*action.base_velocity)
        policy_action = self._adapter.compute_policy_action(refresh_observations=True)
        self._runtime.action_manager.process_action(policy_action.to(self._runtime.device))
        self._last_action = action
        self._metadata["last_arm_action_report"] = arm_report
        self._metadata["last_gripper_action_report"] = gripper_report
        self._record_joint_action_apply(action, arm_report, gripper_report)
        self._action_prepared = True

    def _configure_manipulation_base_lock(self, action: RobotAction) -> None:
        """按状态机请求启停 root/support lock，并记录非纯物理 provenance。"""

        manipulation_requested = bool(action.metadata.get("manipulation_base_lock", False))
        navigation_requested = bool(action.metadata.get("navigation_base_pose_lock", False))
        requested = manipulation_requested or navigation_requested
        phase = (
            action.metadata.get("navigation_base_pose_lock_phase")
            if navigation_requested
            else action.metadata.get("manipulation_base_lock_phase")
        )
        pose_xyzyaw = _coerce_xyzyaw(
            action.metadata.get("navigation_base_pose_lock_xyzyaw")
        )
        if navigation_requested and pose_xyzyaw is None:
            raise RuntimeError(
                "navigation base pose lock requires navigation_base_pose_lock_xyzyaw"
            )
        should_update_pose = navigation_requested and pose_xyzyaw is not None
        base_lock_was_active = self._manipulation_base_lock_active
        if requested and (
            not base_lock_was_active or should_update_pose
        ):
            report = self._adapter.set_base_pose_lock(True, pose_xyzyaw=pose_xyzyaw)
            if report.get("enabled") is not True:
                raise RuntimeError(f"failed to enable manipulation base lock: {report}")
            self._manipulation_base_lock_active = True
            base_report = {
                **report,
                "transition": (
                    "updated"
                    if base_lock_was_active and should_update_pose
                    else "enabled"
                ),
                "phase": phase,
                "source": "navigation" if navigation_requested else "manipulation",
            }
            metadata_update = {
                "used_base_teleport": True,
                "manipulation_base_lock_active": True,
                "last_manipulation_base_lock_report": base_report,
            }
            if manipulation_requested:
                metadata_update["used_manipulation_base_lock"] = True
            if navigation_requested:
                metadata_update["used_navigation_base_lock"] = True
                metadata_update["last_navigation_base_lock_report"] = base_report
            self._metadata.update(metadata_update)
        if not requested and self._manipulation_base_lock_active:
            report = self._adapter.set_base_pose_lock(False)
            self._manipulation_base_lock_active = False
            self._metadata.update(
                {
                    "manipulation_base_lock_active": False,
                    "last_manipulation_base_lock_report": {
                        **report,
                        "transition": "disabled",
                        "phase": phase,
                    },
                    "last_navigation_base_lock_report": {
                        **report,
                        "transition": "disabled",
                        "phase": phase,
                    },
                }
            )
        manipulation_support_requested = bool(action.metadata.get("manipulation_support_joint_lock", False))
        navigation_support_requested = bool(action.metadata.get("navigation_support_joint_lock", False))
        support_requested = manipulation_support_requested or navigation_support_requested
        support_phase = (
            action.metadata.get("navigation_support_joint_lock_phase")
            if navigation_support_requested
            else action.metadata.get("manipulation_support_joint_lock_phase")
        )
        navigation_dog_joint_positions = action.metadata.get(
            "navigation_dog_joint_positions"
        )
        navigation_dog_joint_names = action.metadata.get("navigation_dog_joint_names")
        if support_requested and not self._manipulation_support_joint_lock_active:
            if not hasattr(self._adapter, "set_support_joint_lock"):
                report = {
                    "enabled": False,
                    "reason": "adapter_missing_set_support_joint_lock",
                }
            else:
                if navigation_support_requested:
                    report = self._adapter.set_support_joint_lock(
                        True,
                        dog_joint_target=navigation_dog_joint_positions,
                        dog_joint_names=navigation_dog_joint_names,
                    )
                else:
                    report = self._adapter.set_support_joint_lock(True)
            self._manipulation_support_joint_lock_active = bool(report.get("enabled"))
            support_report = {
                **report,
                "transition": "enabled",
                "phase": support_phase,
                "source": (
                    "navigation"
                    if navigation_support_requested
                    else "manipulation"
                ),
            }
            metadata_update = {
                "used_direct_joint_state": bool(report.get("uses_direct_joint_state", False)),
                "manipulation_support_joint_lock_active": bool(report.get("enabled")),
                "last_manipulation_support_joint_lock_report": support_report,
            }
            if manipulation_support_requested:
                metadata_update["used_manipulation_support_joint_lock"] = bool(
                    report.get("enabled")
                )
            if navigation_support_requested:
                metadata_update["used_navigation_support_joint_lock"] = bool(
                    report.get("enabled")
                )
                metadata_update["last_navigation_support_joint_lock_report"] = (
                    support_report
                )
            self._metadata.update(metadata_update)
        if not support_requested and self._manipulation_support_joint_lock_active:
            report = self._adapter.set_support_joint_lock(False)
            self._manipulation_support_joint_lock_active = False
            self._metadata.update(
                {
                    "manipulation_support_joint_lock_active": False,
                    "last_manipulation_support_joint_lock_report": {
                        **report,
                        "transition": "disabled",
                        "phase": support_phase,
                    },
                    "last_navigation_support_joint_lock_report": {
                        **report,
                        "transition": "disabled",
                        "phase": support_phase,
                    },
                }
            )
        self._configure_navigation_joint_pose_lock(action)
        self._configure_navigation_object_follow(
            action,
            root_target_xyzyaw=pose_xyzyaw,
            navigation_base_lock_requested=navigation_requested,
        )

    def _configure_navigation_joint_pose_lock(self, action: RobotAction) -> None:
        """楼梯漂移期间强制保持四足和机械臂姿态，避免 root 锁定拉出奇异构型。"""

        requested = bool(action.metadata.get("navigation_full_body_joint_lock", False))
        phase = action.metadata.get("navigation_full_body_joint_lock_phase")
        if requested and not self._navigation_joint_pose_lock_active:
            if not hasattr(self._adapter, "set_navigation_joint_pose_lock"):
                report = {
                    "enabled": False,
                    "reason": "adapter_missing_set_navigation_joint_pose_lock",
                }
            else:
                report = self._adapter.set_navigation_joint_pose_lock(
                    True,
                    arm_joint_target=action.arm_joint_positions,
                    dog_joint_target=action.metadata.get(
                        "navigation_dog_joint_positions"
                    ),
                    dog_joint_names=action.metadata.get("navigation_dog_joint_names"),
                )
            self._navigation_joint_pose_lock_active = bool(report.get("enabled"))
            lock_report = {
                **report,
                "transition": "enabled",
                "phase": phase,
                "source": "navigation",
            }
            self._metadata.update(
                {
                    "used_navigation_joint_pose_lock": bool(report.get("enabled")),
                    "used_direct_joint_state": (
                        bool(self._metadata.get("used_direct_joint_state", False))
                        or bool(report.get("uses_direct_joint_state", False))
                    ),
                    "navigation_joint_pose_lock_active": bool(report.get("enabled")),
                    "last_navigation_joint_pose_lock_report": lock_report,
                }
            )
        if not requested and self._navigation_joint_pose_lock_active:
            report = self._adapter.set_navigation_joint_pose_lock(False)
            self._navigation_joint_pose_lock_active = False
            reset_policy_warmup = getattr(self._adapter, "reset_policy_warmup", None)
            policy_warmup_reset = False
            if callable(reset_policy_warmup):
                # 楼梯漂移期间写过 root 和关节状态；解除 direct joint lock 后，
                # 让 locomotion policy 重新渐入，避免第一帧目标阶跃。
                reset_policy_warmup()
                policy_warmup_reset = True
            self._metadata.update(
                {
                    "navigation_joint_pose_lock_active": False,
                    "last_navigation_joint_pose_lock_report": {
                        **report,
                        "transition": "disabled",
                        "phase": phase,
                        "source": "navigation",
                        "policy_warmup_reset": policy_warmup_reset,
                    },
                }
            )

    def _configure_navigation_object_follow(
        self,
        action: RobotAction,
        *,
        root_target_xyzyaw: tuple[float, float, float, float] | None,
        navigation_base_lock_requested: bool,
    ) -> None:
        """仅在 PCT 楼梯漂移期间同步携物，避免 root 写姿态时苹果留在原地。"""

        requested = bool(action.metadata.get("navigation_carry_object_follow", False))
        if requested and (
            not navigation_base_lock_requested or root_target_xyzyaw is None
        ):
            raise RuntimeError(
                "navigation carry object follow requires an active navigation base pose lock"
            )
        if requested:
            self._navigation_object_follow_root_target = tuple(
                float(value) for value in root_target_xyzyaw
            )
            if not self._navigation_object_follow_active:
                self._capture_navigation_object_follow()
            self._update_navigation_object_follow_target()
            return
        if self._navigation_object_follow_active:
            self._release_navigation_object_follow()

    def _capture_navigation_object_follow(self) -> None:
        """保存物体相对 TCP 的刚体变换，并让 PhysX 暂停自由落体。"""

        if self._object is None or self._adapter is None:
            raise RuntimeError("navigation carry object follow requires robot and object handles")
        tcp_pose = self._read_tcp_pose()
        if tcp_pose is None:
            raise RuntimeError("navigation carry object follow requires live TCP pose")
        tcp_position = tuple(float(value) for value in tcp_pose[:3])
        tcp_quaternion = _quat_normalize_wxyz(
            tuple(float(value) for value in tcp_pose[3:7])  # type: ignore[arg-type]
        )
        object_position_raw, object_quaternion_raw = self._object.get_world_pose()
        object_position = tuple(float(value) for value in _as_tuple(object_position_raw))
        object_quaternion = _quat_normalize_wxyz(
            tuple(float(value) for value in _as_tuple(object_quaternion_raw))  # type: ignore[arg-type]
        )
        inverse_tcp = _quat_conjugate_wxyz(tcp_quaternion)
        relative_position = _quat_rotate_vector_wxyz(
            inverse_tcp,
            (
                object_position[0] - tcp_position[0],
                object_position[1] - tcp_position[1],
                object_position[2] - tcp_position[2],
            ),
        )
        relative_quaternion = _quat_normalize_wxyz(
            _quat_multiply_wxyz(inverse_tcp, object_quaternion)
        )
        self._navigation_object_relative_pose = (
            *relative_position,
            *relative_quaternion,
        )
        self._navigation_object_follow_active = True
        sleep_report = self._set_object_sleeping(enabled=True)
        self._metadata.update(
            {
                "used_object_teleport": True,
                "used_kinematic_object_follow": True,
                "navigation_object_follow_active": True,
                "last_navigation_object_follow_report": {
                    "transition": "enabled",
                    "relative_pose_tcp": list(self._navigation_object_relative_pose),
                    "sleep_report": sleep_report,
                },
            }
        )

    def _update_navigation_object_follow_target(self) -> None:
        """用实时 TCP 姿态和下一 root 目标计算本控制步的物体世界位姿。"""

        relative_pose = self._navigation_object_relative_pose
        root_target = self._navigation_object_follow_root_target
        if relative_pose is None or root_target is None or self._adapter is None:
            raise RuntimeError("navigation carry object follow state is incomplete")
        tcp_pose = self._read_tcp_pose()
        if tcp_pose is None:
            raise RuntimeError("navigation carry object follow requires live TCP pose")
        robot = self._adapter.robot
        root_position = tuple(float(value) for value in _as_tuple(robot.data.root_pos_w[0]))
        root_quaternion = _quat_normalize_wxyz(
            tuple(float(value) for value in _as_tuple(robot.data.root_quat_w[0]))  # type: ignore[arg-type]
        )
        tcp_position = tuple(float(value) for value in tcp_pose[:3])
        tcp_quaternion = _quat_normalize_wxyz(
            tuple(float(value) for value in tcp_pose[3:7])  # type: ignore[arg-type]
        )
        inverse_root = _quat_conjugate_wxyz(root_quaternion)
        tcp_position_root = _quat_rotate_vector_wxyz(
            inverse_root,
            (
                tcp_position[0] - root_position[0],
                tcp_position[1] - root_position[1],
                tcp_position[2] - root_position[2],
            ),
        )
        tcp_quaternion_root = _quat_normalize_wxyz(
            _quat_multiply_wxyz(inverse_root, tcp_quaternion)
        )
        target_root_position = (root_target[0], root_target[1], root_target[2])
        target_root_quaternion = _quat_wxyz_from_rpy(0.0, 0.0, root_target[3])
        rotated_tcp_position = _quat_rotate_vector_wxyz(
            target_root_quaternion,
            tcp_position_root,
        )
        target_tcp_position = (
            target_root_position[0] + rotated_tcp_position[0],
            target_root_position[1] + rotated_tcp_position[1],
            target_root_position[2] + rotated_tcp_position[2],
        )
        target_tcp_quaternion = _quat_normalize_wxyz(
            _quat_multiply_wxyz(
                target_root_quaternion,
                tcp_quaternion_root,
            )
        )
        rotated_object_position = _quat_rotate_vector_wxyz(
            target_tcp_quaternion,
            (relative_pose[0], relative_pose[1], relative_pose[2]),
        )
        target_object_position = (
            target_tcp_position[0] + rotated_object_position[0],
            target_tcp_position[1] + rotated_object_position[1],
            target_tcp_position[2] + rotated_object_position[2],
        )
        target_object_quaternion = _quat_normalize_wxyz(
            _quat_multiply_wxyz(
                target_tcp_quaternion,
                tuple(relative_pose[3:7]),  # type: ignore[arg-type]
            )
        )
        self._navigation_object_follow_target_pose = (
            *target_object_position,
            *target_object_quaternion,
        )

    def _release_navigation_object_follow(self) -> None:
        """在楼梯漂移结束后恢复动态物理，并保留夹爪闭合产生的真实约束。"""

        self._apply_active_navigation_object_follow(timing="before_release")
        rigid_view = getattr(self._object, "_rigid_prim_view", None)
        if rigid_view is None or not hasattr(rigid_view, "set_velocities"):
            raise RuntimeError("navigation carry object follow requires rigid velocity API")
        import torch

        rigid_view.set_velocities(
            torch.zeros(
                (1, 6),
                dtype=torch.float32,
                device=getattr(self._runtime, "device", "cpu"),
            )
        )
        wake_report = self._set_object_sleeping(enabled=False)
        apply_count = int(self._metadata.get("navigation_object_follow_apply_count", 0))
        self._navigation_object_follow_active = False
        self._navigation_object_relative_pose = None
        self._navigation_object_follow_root_target = None
        self._navigation_object_follow_target_pose = None
        self._metadata.update(
            {
                "navigation_object_follow_active": False,
                "last_navigation_object_follow_report": {
                    "transition": "disabled",
                    "apply_count": apply_count,
                    "wake_report": wake_report,
                },
            }
        )

    def _apply_active_navigation_object_follow(self, *, timing: str) -> None:
        """把苹果写到当前 root 目标对应的夹持相对位姿。"""

        if not self._navigation_object_follow_active:
            return
        target_pose = self._navigation_object_follow_target_pose
        if target_pose is None or self._object is None:
            raise RuntimeError("navigation carry object follow state is incomplete")
        object_position = tuple(float(value) for value in target_pose[:3])
        object_quaternion = tuple(float(value) for value in target_pose[3:7])
        rigid_view = getattr(self._object, "_rigid_prim_view", None)
        if (
            rigid_view is None
            or not hasattr(rigid_view, "set_world_poses")
            or not hasattr(rigid_view, "set_velocities")
        ):
            raise RuntimeError("navigation carry object follow requires rigid pose APIs")
        import torch

        device = getattr(self._runtime, "device", "cpu")
        rigid_view.set_world_poses(
            positions=torch.tensor(
                [object_position],
                dtype=torch.float32,
                device=device,
            ),
            orientations=torch.tensor(
                [object_quaternion],
                dtype=torch.float32,
                device=device,
            ),
        )
        rigid_view.set_velocities(
            torch.zeros((1, 6), dtype=torch.float32, device=device)
        )
        apply_count = int(
            self._metadata.get("navigation_object_follow_apply_count", 0)
        ) + 1
        self._metadata.update(
            {
                "navigation_object_follow_active": True,
                "navigation_object_follow_apply_count": apply_count,
                "last_navigation_object_follow_report": {
                    "transition": "applied",
                    "timing": timing,
                    "apply_count": apply_count,
                    "object_pose": [*object_position, *object_quaternion],
                },
            }
        )

    def _apply_active_manipulation_base_lock(self, *, timing: str) -> None:
        if self._manipulation_base_lock_active:
            report = self._adapter.apply_base_pose_lock()
            if report.get("applied") is not True:
                raise RuntimeError(f"manipulation base lock was not applied: {report}")
            apply_count = int(self._metadata.get("manipulation_base_lock_apply_count", 0)) + 1
            self._metadata.update(
                {
                    "manipulation_base_lock_active": True,
                    "manipulation_base_lock_apply_count": apply_count,
                    "last_manipulation_base_lock_report": {
                        **report,
                        "timing": timing,
                        "apply_count": apply_count,
                    },
                }
            )
        self._apply_active_navigation_object_follow(timing=timing)
        if self._manipulation_support_joint_lock_active:
            report = self._adapter.apply_support_joint_lock()
            if report.get("applied") is not True:
                raise RuntimeError(f"manipulation support joint lock was not applied: {report}")
            apply_count = int(
                self._metadata.get("manipulation_support_joint_lock_apply_count", 0)
            ) + 1
            self._metadata.update(
                {
                    "manipulation_support_joint_lock_active": True,
                    "manipulation_support_joint_lock_apply_count": apply_count,
                    "last_manipulation_support_joint_lock_report": {
                        **report,
                        "timing": timing,
                        "apply_count": apply_count,
                    },
                }
            )
        if self._navigation_joint_pose_lock_active:
            if not hasattr(self._adapter, "apply_navigation_joint_pose_lock"):
                raise RuntimeError("navigation joint pose lock adapter method is unavailable")
            report = self._adapter.apply_navigation_joint_pose_lock()
            if report.get("applied") is not True:
                raise RuntimeError(f"navigation joint pose lock was not applied: {report}")
            apply_count = int(
                self._metadata.get("navigation_joint_pose_lock_apply_count", 0)
            ) + 1
            self._metadata.update(
                {
                    "used_navigation_joint_pose_lock": True,
                    "used_direct_joint_state": (
                        bool(self._metadata.get("used_direct_joint_state", False))
                        or bool(report.get("uses_direct_joint_state", False))
                    ),
                    "navigation_joint_pose_lock_active": True,
                    "navigation_joint_pose_lock_apply_count": apply_count,
                    "last_navigation_joint_pose_lock_report": {
                        **report,
                        "timing": timing,
                        "apply_count": apply_count,
                    },
                }
            )

    def _apply_staged_joint_position_targets(self, *, timing: str) -> None:
        """在 action manager 之后刷新 arm/gripper PD target，不直接改 joint state。"""

        if hasattr(self._adapter, "apply_arm_joint_target"):
            arm_report = self._adapter.apply_arm_joint_target()
        else:
            arm_report = {"applied": False, "reason": "adapter_missing_apply_arm_joint_target"}
        if arm_report.get("applied") is True:
            apply_count = int(
                self._metadata.get("arm_joint_position_target_apply_count", 0)
            ) + 1
            self._metadata.update(
                {
                    "arm_joint_position_target_apply_count": apply_count,
                    "last_arm_joint_position_target_report": {
                        **arm_report,
                        "timing": timing,
                        "apply_count": apply_count,
                        "world_step_owned_by_pipeline": True,
                    },
                }
            )

        if hasattr(self._adapter, "apply_gripper_joint_target"):
            gripper_report = self._adapter.apply_gripper_joint_target()
        else:
            gripper_report = {
                "applied": False,
                "reason": "adapter_missing_apply_gripper_joint_target",
            }
        if gripper_report.get("applied") is True:
            apply_count = int(
                self._metadata.get("gripper_joint_position_target_apply_count", 0)
            ) + 1
            self._metadata.update(
                {
                    "gripper_joint_position_target_apply_count": apply_count,
                    "last_gripper_joint_position_target_report": {
                        **gripper_report,
                        "timing": timing,
                        "apply_count": apply_count,
                        "world_step_owned_by_pipeline": True,
                    },
                }
            )

    def _record_joint_action_apply(
        self,
        action: RobotAction,
        arm_report: dict[str, Any],
        gripper_report: dict[str, Any],
    ) -> None:
        """记录 target staging 事实，供联调验证器区分控制成功与物体成功。"""

        arm_targeted = arm_report.get("target_staged") is True
        gripper_targeted = gripper_report.get("target_staged") is True
        if not arm_targeted and not gripper_targeted:
            return

        self._metadata["joint_action_apply_count"] = (
            int(self._metadata.get("joint_action_apply_count", 0)) + 1
        )
        if arm_targeted:
            self._metadata["arm_joint_action_apply_count"] = (
                int(self._metadata.get("arm_joint_action_apply_count", 0)) + 1
            )
        if gripper_targeted:
            self._metadata["gripper_joint_action_apply_count"] = (
                int(self._metadata.get("gripper_joint_action_apply_count", 0)) + 1
            )
        if gripper_targeted and action.gripper_command == "close":
            self._metadata["gripper_close_apply_count"] = (
                int(self._metadata.get("gripper_close_apply_count", 0)) + 1
            )
        if gripper_targeted and action.gripper_command == "open":
            self._metadata["gripper_open_apply_count"] = (
                int(self._metadata.get("gripper_open_apply_count", 0)) + 1
            )
        self._metadata["last_joint_action_report"] = {
            "applied": True,
            "source": action.source,
            "arm_targeted": arm_targeted,
            "gripper_targeted": gripper_targeted,
            "uses_direct_joint_state": False,
            "world_step_owned_by_pipeline": True,
        }

    def step(self, *, render: bool) -> None:
        self._require_ready()
        if not self._action_prepared:
            raise RuntimeError("apply must be called before step")

        is_rendering = bool(render or self._runtime.sim.has_rtx_sensors())
        self._runtime.recorder_manager.record_pre_step()
        for _ in range(self._runtime.cfg.decimation):
            self._runtime._sim_step_counter += 1
            self._apply_active_manipulation_base_lock(timing="before_physics_step")
            self._runtime.action_manager.apply_action()
            # action_manager 会重写 joint_pos action term 覆盖目标；这里在写入 sim 前
            # 重新刷新支撑锁和机械臂/夹爪 position target，确保本物理子步使用 pipeline 目标。
            self._apply_active_manipulation_base_lock(timing="after_action_manager")
            self._apply_staged_joint_position_targets(timing="after_action_manager")
            self._runtime.scene.write_data_to_sim()
            # 这里只有 runtime 执行底层 physics step，调用权来自 pipeline 唯一主循环。
            self._runtime.sim.step(render=False)
            self._apply_active_manipulation_base_lock(timing="after_physics_step")
            self._runtime.recorder_manager.record_post_physics_decimation_step()
            if (
                is_rendering
                and self._runtime._sim_step_counter % self._runtime.cfg.sim.render_interval == 0
            ):
                self._runtime.sim.render()
            self._runtime.scene.update(dt=self._runtime.physics_dt)

        self._finish_control_step()
        self._step_calls += 1
        self._action_prepared = False

    def close(self) -> None:
        if self._closed:
            return
        if self._env is not None:
            self._env.close()
        self._closed = True

    def _build_environment(self, episode_spec: EpisodeSpec) -> None:
        import gymnasium as gym
        import isaaclab.sim as sim_utils
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab.utils.assets import retrieve_file_path
        from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils.hydra import hydra_task_config
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        robot_lab_source = self._project_root / "source/robot_lab"
        if str(robot_lab_source) not in sys.path:
            sys.path.insert(0, str(robot_lab_source))
        import robot_lab.tasks  # noqa: F401

        from source.navigation.adapters.isaaclab_go2_adapter import Go2LocomotionAdapter

        build_result: dict[str, Any] = {}

        @hydra_task_config(self._config.task_name, self._config.agent_entry_point)
        def _create(
            env_cfg: ManagerBasedRLEnvCfg,
            agent_cfg: RslRlBaseRunnerCfg,
        ) -> None:
            self._metadata["visual_scene_report"] = self._load_visual_scene(episode_spec)
            self._metadata["d436_lens_distortion_schema_report"] = (
                _enable_d436_lens_distortion_schema()
            )
            self._configure_env(env_cfg, episode_spec, sim_utils)
            env_cfg.seed = int(agent_cfg.seed)
            env = gym.make(
                self._config.task_name,
                cfg=env_cfg,
                render_mode=(
                    "rgb_array"
                    if (
                        self._config.enable_front_camera
                        or self._config.enable_wrist_camera
                        or self._config.enable_overview_camera
                    )
                    else None
                ),
            )
            self._metadata["camera_runtime_intrinsics_report"] = (
                self._apply_d436_runtime_intrinsics(env.unwrapped)
            )
            if self._config.patch_gripper_collision or self._config.patch_apple_collision:
                from source.simulation.collision_patch import (
                    gripper_collision_patch_report,
                    keyword_collision_patch_report,
                )

                robot_spawn_cfg = env.unwrapped.scene["robot"].cfg.spawn
                if self._config.patch_gripper_collision:
                    patch_report = gripper_collision_patch_report(robot_spawn_cfg)
                    self._metadata["gripper_collision_patch_report"] = (
                        patch_report
                        if patch_report is not None
                        else {
                            "applied": False,
                            "reason": "spawn_patch_report_missing",
                            "patch_count": 0,
                        }
                    )
                if self._config.patch_apple_collision:
                    apple_patch_report = keyword_collision_patch_report(robot_spawn_cfg)
                    self._metadata["apple_collision_patch_report"] = (
                        apple_patch_report
                        if apple_patch_report is not None
                        else {
                            "applied": False,
                            "reason": "spawn_patch_report_missing",
                            "patch_count": 0,
                        }
                    )
            # collision patch 可能取消实例化并重新组合苹果 prim；相机采集前重新显示任务物体。
            import omni.usd

            stage_after_collision_patch = omni.usd.get_context().get_stage()
            self._metadata["object_visibility_after_spawn_report"] = (
                self._show_only_task_object(stage_after_collision_patch, episode_spec)
                if stage_after_collision_patch is not None
                else {"applied": False, "reason": "usd_stage_unavailable"}
            )
            self._metadata["object_collision_visual_hide_after_spawn_report"] = (
                self._hide_object_collision_visual(stage_after_collision_patch)
                if stage_after_collision_patch is not None
                else {"applied": False, "reason": "usd_stage_unavailable"}
            )
            self._metadata["scene_lighting_after_spawn_report"] = (
                self._configure_scene_lighting(
                    stage_after_collision_patch,
                    reason="after_spawn",
                )
                if stage_after_collision_patch is not None
                else {"applied": False, "reason": "usd_stage_unavailable"}
            )
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            # wrapper 构造时会触发 env.reset；GUI viewport 只在 reset 完成后切相机。
            self.refresh_viewport(reason="environment_reset")
            stage_after_reset = omni.usd.get_context().get_stage()
            self._metadata["scene_lighting_after_reset_report"] = (
                self._configure_scene_lighting(
                    stage_after_reset,
                    reason="after_environment_reset",
                )
                if stage_after_reset is not None
                else {"applied": False, "reason": "usd_stage_unavailable"}
            )
            checkpoint = retrieve_file_path(str(self._resolve_checkpoint()))
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(
                    wrapped,
                    agent_cfg.to_dict(),
                    log_dir=None,
                    device=agent_cfg.device,
                )
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(
                    wrapped,
                    agent_cfg.to_dict(),
                    log_dir=None,
                    device=agent_cfg.device,
                )
            else:
                raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
            adapter = Go2LocomotionAdapter(
                wrapped,
                policy,
                wrapped.get_observations(),
                standing_command_threshold=self._config.standing_command_threshold,
                policy_action_warmup_steps=self._config.policy_action_warmup_steps,
            )
            build_result.update({"env": wrapped, "runtime": wrapped.unwrapped, "adapter": adapter})

        original_argv = sys.argv
        try:
            # Hydra 只负责加载注册配置，pipeline CLI 参数不能泄漏为 Hydra override。
            sys.argv = [original_argv[0]]
            _create()
        finally:
            sys.argv = original_argv
        if not build_result:
            raise RuntimeError("Isaac Lab environment builder returned no result")
        self._env = build_result["env"]
        self._runtime = build_result["runtime"]
        self._adapter = build_result["adapter"]
        self._initialize_object_reader(episode_spec)
        self._metadata["object_pose_debug_after_physics_reader"] = (
            self._object_initial_pose_diagnostic(
                episode_spec,
                label="after_object_reader_initialize",
            )
        )

    def _apply_d436_runtime_intrinsics(self, runtime: Any) -> dict[str, Any]:
        """让 IsaacLab 对外暴露的 K 与 OpenCV schema 的实际渲染内参一致。"""

        camera_sensors = []
        if self._config.enable_front_camera:
            camera_sensors.append(("front", "head_camera"))
        if self._config.enable_wrist_camera:
            camera_sensors.append(("wrist", "arm_camera"))
        report: dict[str, Any] = {
            "applied": True,
            "intrinsics": _d436_camera_intrinsics_metadata(),
            "cameras": {},
        }
        for camera_name, sensor_name in camera_sensors:
            try:
                sensor = runtime.scene[sensor_name]
                matrices = sensor._data.intrinsic_matrices
                sensor_prims = sensor._sensor_prims
                camera_prim = sensor_prims[0].GetPrim()
                model_attribute = camera_prim.GetAttribute("omni:lensdistortion:model")
                renderer_schema_applied = bool(
                    model_attribute.IsValid()
                    and model_attribute.Get() == "opencvPinhole"
                )
                if renderer_schema_applied:
                    matrix_count = _overwrite_d436_intrinsic_matrices(matrices)
                    effective_intrinsics = {
                        "fx": D436_CAMERA_FX_PX,
                        "fy": D436_CAMERA_FY_PX,
                        "cx": D436_CAMERA_CX_PX,
                        "cy": D436_CAMERA_CY_PX,
                    }
                else:
                    matrix_count = _overwrite_d436_fallback_intrinsic_matrices(
                        matrices
                    )
                    effective_intrinsics = {
                        "fx": D436_CAMERA_FALLBACK_FX_FY_PX,
                        "fy": D436_CAMERA_FALLBACK_FX_FY_PX,
                        "cx": D436_CAMERA_FALLBACK_CX_PX,
                        "cy": D436_CAMERA_FALLBACK_CY_PX,
                    }
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                report["applied"] = False
                report["cameras"][camera_name] = {
                    "applied": False,
                    "sensor_name": sensor_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            report["cameras"][camera_name] = {
                "applied": True,
                "sensor_name": sensor_name,
                "matrix_count": matrix_count,
                "renderer_schema_applied": renderer_schema_applied,
                "effective_intrinsics": effective_intrinsics,
            }
        return report

    def _robot_prim_path(self) -> str:
        """兼容不同 Isaac Lab 版本的 Articulation 路径字段。"""

        robot = self._adapter.robot
        prim_path = getattr(robot, "prim_path", None)
        if prim_path:
            return str(prim_path)
        cfg = getattr(robot, "cfg", None)
        cfg_prim_path = getattr(cfg, "prim_path", "")
        return str(cfg_prim_path)

    def _configure_env(self, env_cfg: Any, episode_spec: EpisodeSpec, sim_utils: Any) -> None:
        from isaaclab.sensors import CameraCfg
        from isaaclab.terrains import TerrainImporterCfg
        from source.navigation.adapters.terrain_utils import write_collision_terrain_wrapper

        scene_usd = self._resolve_path(episode_spec.scene_usd)
        scene_runtime = resolve_scene_runtime_settings(
            episode_spec.raw_task,
            default_collision_prim_path=self._config.terrain_prim_path,
            default_visual_prim_path=self._config.visual_prim_path,
            default_collision_floor_proxy_profile=(
                self._config.collision_floor_proxy_profile
            ),
        )
        collision_prim_path = str(scene_runtime["collision_prim_path"])
        floor_proxy_profile = scene_runtime["collision_floor_proxy_profile"]
        self._metadata["scene_runtime_settings"] = scene_runtime
        from source.simulation.task_scene_pose import resolve_task_receptacle_pose

        receptacle_pose_settings = resolve_task_receptacle_pose(
            episode_spec.raw_task
        )
        if receptacle_pose_settings["configured"]:
            # 源 USDA 只保存模板位姿；episode 覆盖必须在组合 stage 中验证，
            # 否则会把预期的随机位移误报为资产漂移。
            self._metadata["task_receptacle_support_source_report"] = {
                "source": "isaaclab_source_scene_usd",
                "geometry_verified": None,
                "skipped": True,
                "reason": "episode_receptacle_pose_requires_composed_stage",
                "receptacle_pose": receptacle_pose_settings,
            }
        else:
            self._metadata["task_receptacle_support_source_report"] = (
                inspect_task_receptacle_support_usd(
                    scene_usd,
                    episode_spec.raw_task,
                    source="isaaclab_source_scene_usd",
                )
            )
        stage_report = self._validate_scene_collision(
            scene_usd,
            collision_prim_path,
        )
        self._metadata["stage_report"] = stage_report
        terrain_usd = write_collision_terrain_wrapper(
            scene_usd,
            collision_prim_path,
            floor_proxy_profile=floor_proxy_profile,
            source_prim_is_mesh=bool(stage_report["collision_root_is_mesh"]),
        )
        wrapper_stage_report = self._validate_scene_collision(
            terrain_usd,
            "/scene_collision",
        )
        self._metadata["collision_terrain_wrapper_report"] = wrapper_stage_report
        self._metadata["collision_floor_proxy_report"] = {
            "profile": floor_proxy_profile,
            "source_collision_prim_path": collision_prim_path,
            "terrain_wrapper": str(terrain_usd),
        }
        env_cfg.scene.num_envs = 1
        env_cfg.scene.env_spacing = 0.0
        env_cfg.sim.device = self._config.device
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/nav_collision",
            terrain_type="usd",
            usd_path=str(terrain_usd),
            debug_vis=False,
        )
        # TerrainImporter 会把 USD default prim 生成在 ``<prim_path>/terrain``；
        # RayCaster 只接受 Mesh prim，不能绑定外层 ``/World/nav_collision`` 容器。
        terrain_mesh_prim_path = f"{env_cfg.scene.terrain.prim_path}/terrain"
        updated_height_scanners = _retarget_height_scanners(
            env_cfg.scene,
            terrain_mesh_prim_path,
        )
        self._metadata["height_scanner_terrain_report"] = {
            "terrain_prim_path": env_cfg.scene.terrain.prim_path,
            "terrain_mesh_prim_path": terrain_mesh_prim_path,
            "updated_sensors": updated_height_scanners,
        }
        default_root_pos = tuple(float(value) for value in env_cfg.scene.robot.init_state.pos)
        start_z = default_root_pos[2] if episode_spec.start.z is None else float(episode_spec.start.z)
        start_offset = (
            float(episode_spec.start.x) - default_root_pos[0],
            float(episode_spec.start.y) - default_root_pos[1],
            start_z - default_root_pos[2],
        )
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (start_offset[0], start_offset[0]),
                "y": (start_offset[1], start_offset[1]),
                "z": (start_offset[2], start_offset[2]),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (episode_spec.start.yaw, episode_spec.start.yaw),
            },
            "velocity_range": {
                key: (0.0, 0.0)
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            },
        }
        self._metadata["episode_reset_pose_request"] = {
            "target_world_xyz_yaw": (
                float(episode_spec.start.x),
                float(episode_spec.start.y),
                start_z,
                float(episode_spec.start.yaw),
            ),
            "default_root_pos": default_root_pos,
            "pose_range_offset_xyz": start_offset,
            "event_semantics": "default_root_state_plus_offset",
        }
        env_cfg.observations.policy.enable_corruption = False
        for event_name in (
            "randomize_rigid_body_material",
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_others",
            "randomize_com_positions",
            "randomize_apply_external_force_torque",
            "push_robot",
            "randomize_push_robot",
            "randomize_actuator_gains",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        for curriculum_name in (
            "terrain_levels",
            "command_levels_lin_vel",
            "command_levels_ang_vel",
        ):
            if hasattr(env_cfg.curriculum, curriculum_name):
                setattr(env_cfg.curriculum, curriculum_name, None)
        env_cfg.terminations.time_out = None
        env_cfg.terminations.illegal_contact = None
        env_cfg.terminations.terrain_out_of_bounds = None
        if self._config.patch_gripper_collision or self._config.patch_apple_collision:
            from source.simulation.collision_patch import (
                install_gripper_collision_patch_on_spawn,
            )

            install_gripper_collision_patch_on_spawn(
                env_cfg.scene.robot.spawn,
                enable_gripper_patch=self._config.patch_gripper_collision,
                enable_keyword_patch=self._config.patch_apple_collision,
                robot_root=self._config.gripper_collision_robot_root,
                gripper_links=self._config.gripper_collision_links,
                approximation=self._config.gripper_collision_approximation,
                contact_offset=self._config.gripper_collision_contact_offset,
                rest_offset=self._config.gripper_collision_rest_offset,
                keyword_root_path=self._config.apple_collision_root_path,
                keywords=self._config.apple_collision_keywords,
                keyword_approximation=self._config.apple_collision_approximation,
                keyword_contact_offset=self._config.apple_collision_contact_offset,
                keyword_rest_offset=self._config.apple_collision_rest_offset,
            )
        if self._config.enable_front_camera:
            _validate_d436_camera_calibration_resolution(
                "front",
                self._config.front_camera_width,
                self._config.front_camera_height,
            )
            # 保留 DWA/play_nav_cs.py 的安装外参；内参改用 D436 640x480 标定值。
            env_cfg.scene.head_camera = CameraCfg(
                prim_path=FRONT_CAMERA_PRIM_PATH,
                update_period=0.0,
                height=self._config.front_camera_height,
                width=self._config.front_camera_width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    func=_make_d436_camera_spawn_function(),
                    focal_length=D436_CAMERA_FALLBACK_FOCAL_LENGTH_MM,
                    focus_distance=400.0,
                    horizontal_aperture=D436_CAMERA_FALLBACK_HORIZONTAL_APERTURE_MM,
                    vertical_aperture=D436_CAMERA_FALLBACK_VERTICAL_APERTURE_MM,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=FRONT_CAMERA_MOUNT_POS_XYZ_M,
                    rot=FRONT_CAMERA_MOUNT_ROT_WXYZ,
                    convention="ros",
                ),
            )
            self._metadata["front_camera_report"] = {
                "enabled": True,
                "name": "head_camera",
                "resolution_hw": [
                    self._config.front_camera_height,
                    self._config.front_camera_width,
                ],
                "data_types": ["rgb"],
                "source": "dwa_play_nav_cs",
                "calibration": _front_camera_calibration_metadata(),
            }
        if self._config.enable_wrist_camera:
            _validate_d436_camera_calibration_resolution(
                "wrist",
                self._config.wrist_camera_width,
                self._config.wrist_camera_height,
            )
            # 使用 arm_link6_T_camera_color_optical 手眼标定结果。
            env_cfg.scene.arm_camera = CameraCfg(
                prim_path=WRIST_CAMERA_PRIM_PATH,
                update_period=0.0,
                height=self._config.wrist_camera_height,
                width=self._config.wrist_camera_width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    func=_make_d436_camera_spawn_function(),
                    focal_length=D436_CAMERA_FALLBACK_FOCAL_LENGTH_MM,
                    focus_distance=400.0,
                    horizontal_aperture=D436_CAMERA_FALLBACK_HORIZONTAL_APERTURE_MM,
                    vertical_aperture=D436_CAMERA_FALLBACK_VERTICAL_APERTURE_MM,
                    clipping_range=(WRIST_CAMERA_NEAR_CLIPPING_M, 5.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=WRIST_CAMERA_MOUNT_POS_XYZ_M,
                    rot=WRIST_CAMERA_MOUNT_ROT_WXYZ,
                    convention="ros",
                ),
            )
            self._metadata["wrist_camera_report"] = {
                "enabled": True,
                "name": "arm_camera",
                "prim_path": "{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera",
                "resolution_hw": [
                    self._config.wrist_camera_height,
                    self._config.wrist_camera_width,
                ],
                "data_types": ["rgb"],
                "source": "hand_eye_calibration_with_visual_alignment_v3",
                "calibration": _wrist_camera_calibration_metadata(),
            }
        if self._config.enable_overview_camera:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            overview_prim = (
                None
                if stage is None
                else stage.GetPrimAtPath(self._config.overview_camera_prim_path)
            )
            if (
                overview_prim is not None
                and overview_prim.IsValid()
                and overview_prim.IsA(UsdGeom.Camera)
            ):
                env_cfg.scene.overview_camera = CameraCfg(
                    prim_path=self._config.overview_camera_prim_path,
                    update_period=0.0,
                    height=self._config.overview_camera_height,
                    width=self._config.overview_camera_width,
                    data_types=["rgb"],
                    spawn=None,
                )
                self._metadata["overview_camera_report"] = {
                    "enabled": True,
                    "bound_existing_stage_camera": True,
                    "name": "overview_camera",
                    "prim_path": self._config.overview_camera_prim_path,
                    "resolution_hw": [
                        self._config.overview_camera_height,
                        self._config.overview_camera_width,
                    ],
                    "data_types": ["rgb"],
                    "source": "authored_stage_camera",
                }
            else:
                self._metadata["overview_camera_report"] = {
                    "enabled": False,
                    "bound_existing_stage_camera": False,
                    "prim_path": self._config.overview_camera_prim_path,
                    "reason": "overview_camera_prim_unavailable",
                }

    def _load_visual_scene(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        import omni.usd
        from source.navigation.adapters.terrain_utils import write_visual_sublayer_wrapper

        scene_runtime = resolve_scene_runtime_settings(
            episode_spec.raw_task,
            default_collision_prim_path=self._config.terrain_prim_path,
            default_visual_prim_path=self._config.visual_prim_path,
            default_collision_floor_proxy_profile=(
                self._config.collision_floor_proxy_profile
            ),
        )
        collision_prim_path = str(scene_runtime["collision_prim_path"])
        visual_prim_path = str(scene_runtime["visual_prim_path"])
        wrapper = write_visual_sublayer_wrapper(
            self._resolve_path(episode_spec.scene_usd),
            visual_prim_path,
            excluded_prim_paths=(
                collision_prim_path,
                "/World/go2_x5",
                "/World/mec_arm_6dof",
            ),
            include_visual_prim=self._config.enable_scene_visual,
        )
        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            # AppLauncher 通常会创建 stage；这里兜底创建当前 stage，后续 env 仍在同一 stage 上构建。
            context.new_stage()
            stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage is unavailable before environment creation")
        root_layer = stage.GetRootLayer()
        if str(wrapper) not in root_layer.subLayerPaths:
            root_layer.subLayerPaths.append(str(wrapper))
        from source.simulation.task_scene_pose import apply_task_receptacle_pose

        receptacle_pose_report = apply_task_receptacle_pose(
            stage,
            episode_spec.raw_task,
        )
        self._metadata["task_receptacle_pose_report"] = receptacle_pose_report
        receptacle_support_report = inspect_task_receptacle_support_stage(
            stage,
            episode_spec.raw_task,
            source="isaaclab_visual_sublayer_stage",
        )
        self._metadata["task_receptacle_support_runtime_stage_report"] = (
            receptacle_support_report
        )
        self._metadata["object_pose_setup_report"] = self._apply_object_pose(episode_spec)
        object_visibility = self._show_only_task_object(stage, episode_spec)
        self._metadata["object_visibility_report"] = object_visibility
        object_collision_visual_hide = self._hide_object_collision_visual(stage)
        self._metadata["object_collision_visual_hide_report"] = (
            object_collision_visual_hide
        )
        scene_lighting = self._configure_scene_lighting(stage, reason="visual_scene_loaded")
        self._metadata["scene_lighting_report"] = scene_lighting
        return {
            "loaded": True,
            "load_mode": "sublayer",
            "wrapper_path": str(wrapper),
            "scene_usd": str(self._resolve_path(episode_spec.scene_usd)),
            "visual_prim_path": visual_prim_path,
            "collision_prim_path": collision_prim_path,
            "scene_runtime_settings": scene_runtime,
            "task_receptacle_support_report": receptacle_support_report,
            "task_receptacle_pose_report": receptacle_pose_report,
            "scene_visual_enabled": self._config.enable_scene_visual,
            "excluded_prim_paths": (
                collision_prim_path,
                "/World/go2_x5",
                "/World/mec_arm_6dof",
            ),
            "object_visibility": object_visibility,
            "object_collision_visual_hide": object_collision_visual_hide,
            "scene_lighting": scene_lighting,
        }

    def _configure_scene_lighting(self, stage: Any, *, reason: str) -> dict[str, Any]:
        """根据 runtime 配置切换 stage light / camera light。"""

        from source.simulation.lighting import configure_scene_lighting

        report = configure_scene_lighting(
            stage=stage,
            mode=self._config.scene_light_mode,
            camera_light_name=self._config.camera_light_name,
            camera_light_intensity=self._config.camera_light_intensity,
            camera_light_radius=self._config.camera_light_radius,
        )
        report["reason"] = reason
        return report

    def _hide_object_collision_visual(self, stage: Any) -> dict[str, Any]:
        """隐藏 Apple_M_Apple 碰撞视觉层，避免相机采集到占位碰撞网格。"""

        if not self._config.hide_object_collision_visual:
            return {"applied": False, "reason": "disabled_by_config"}
        from source.simulation.visibility_patch import hide_visual_prims_by_keywords

        return hide_visual_prims_by_keywords(
            root_path=self._config.object_collision_visual_root_path,
            hide_keywords=self._config.object_collision_visual_hide_keywords,
            keep_keywords=self._config.object_collision_visual_keep_keywords,
            stage=stage,
        )

    def _show_only_task_object(self, stage: Any, episode_spec: EpisodeSpec) -> dict[str, Any]:
        """复用 baseline 语义：隐藏 apple/orange/bottle 中的非任务物体。"""

        from pxr import Usd, UsdGeom

        object_prim_path = episode_spec.object_prim_path
        if not object_prim_path:
            self._hidden_distractor_root_paths = ()
            return {"applied": False, "reason": "object_prim_path_missing"}

        object_prefix = object_prim_path.rstrip("/") + "/"
        keywords = ("apple", "orange", "bottle")
        candidate_roots: list[str] = []
        hidden_paths: list[str] = []
        shown_paths: list[str] = []

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim_path == object_prim_path or prim_path.startswith(object_prefix):
                continue
            if any(keyword in _prim_keyword_match_text(prim) for keyword in keywords):
                candidate_roots.append(prim_path)

        hidden_root_paths = _dedupe_root_paths(candidate_roots)
        for root_path in hidden_root_paths:
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue
            for child in Usd.PrimRange(root_prim):
                child_path = str(child.GetPath())
                if child_path == object_prim_path or child_path.startswith(object_prefix):
                    continue
                if child.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(child).MakeInvisible()
                    hidden_paths.append(child_path)

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim_path == object_prim_path or prim_path.startswith(object_prefix):
                if prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeVisible()
                    shown_paths.append(prim_path)

        # cuRobo 必须使用同一组隐藏根，否则视觉上消失的占位物仍会阻塞 place 目标。
        self._hidden_distractor_root_paths = tuple(hidden_root_paths)
        return {
            "applied": True,
            "kept_object_prim_path": object_prim_path,
            "shown_paths": shown_paths,
            "hidden_root_paths": hidden_root_paths,
            "hidden_paths": hidden_paths,
            "planner_collision_exclusion_enabled": True,
        }

    def refresh_viewport(self, *, reason: str = "manual") -> dict[str, Any]:
        """重试配置 GUI viewpoint；只影响显示，不推进物理。"""

        return self._configure_viewport(reason=reason)

    def _retry_viewport_after_stage_updates(self) -> None:
        """IsaacLab 创建窗口和 sublayer 解析可能滞后，前几帧允许轻量重试。"""

        if self._config.viewport_camera_prim_path in {"", "none", "None"}:
            return
        report = self._metadata.get("viewport_report")
        if isinstance(report, dict) and report.get("camera_applied") is True:
            return
        if self._viewport_config_attempts >= 12:
            return
        if self._step_calls not in {0, 1, 2, 5, 10, 20, 40, 80}:
            return
        self.refresh_viewport(reason=f"retry_step_{self._step_calls}")

    def _configure_viewport(self, *, reason: str = "initial") -> dict[str, Any]:
        """复用 baseline 的 GUI 显示语义，不参与仿真控制。"""

        from source.simulation.viewport import configure_navigation_viewport

        self._viewport_config_attempts += 1
        report = configure_navigation_viewport(
            camera_prim_path=self._config.viewport_camera_prim_path,
            hide_collision_visual=self._config.hide_navigation_collision_visual,
        )
        report["configure_reason"] = reason
        report["configure_attempt"] = self._viewport_config_attempts
        self._metadata["viewport_report"] = report
        selected_camera = report.get("selected_camera_prim_path")
        if selected_camera:
            self._metadata["camera_prim_path"] = selected_camera
        return report

    def _validate_scene_collision(self, scene_usd: Path, prim_path: str) -> dict[str, Any]:
        """提前确认外部 collision payload 已挂载，避免空场景里假跑导航。"""

        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(scene_usd))
        if stage is None:
            raise RuntimeError(f"failed to open scene USD: {scene_usd}")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"scene collision prim does not exist: {prim_path} in {scene_usd}")
        mesh_count = 0
        mesh_prim_paths: list[str] = []
        collision_api_count = 0
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                mesh_count += 1
                mesh_prim_paths.append(str(child.GetPath()))
            if child.HasAPI(UsdPhysics.CollisionAPI):
                collision_api_count += 1
        if mesh_count == 0:
            raise RuntimeError(
                f"scene collision prim {prim_path} has no mesh geometry; "
                "check external payload mounts such as /mnt/sage_data."
            )
        return {
            "scene_usd": str(scene_usd),
            "collision_prim_path": prim_path,
            "collision_root_type": prim.GetTypeName(),
            "collision_root_is_mesh": bool(prim.IsA(UsdGeom.Mesh)),
            "mesh_prim_paths": mesh_prim_paths,
            "mesh_count": mesh_count,
            "collision_api_count": collision_api_count,
        }

    def _apply_object_pose(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        """按 baseline 写入苹果根节点局部姿态，并保留资产 unitsResolve。"""

        if not episode_spec.object_prim_path or episode_spec.object_initial_pose is None:
            return {"applied": False, "reason": "object_initial_pose_missing"}
        import omni.usd
        from pxr import UsdGeom

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(episode_spec.object_prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            raise RuntimeError(f"task object prim is unavailable: {episode_spec.object_prim_path}")
        x, y, z, roll, pitch, yaw = episode_spec.object_initial_pose
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        quat = (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
        xformable = UsdGeom.Xformable(prim)
        op_order_before = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
        # 任务中的固定 RPY 对应 Inspector 的 Orient 字段。苹果资产还包含
        # rotateX:unitsResolve=90°，该 op 是模型坐标系转换，不能删掉或把任务
        # 四元数直接当成最终刚体世界姿态。
        translate_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
        orient_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
        _set_translate_op(translate_op, (float(x), float(y), float(z)))
        _set_orient_op(orient_op, tuple(float(value) for value in quat))
        op_order_after = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
        authored_world_position, authored_world_quaternion = _xformable_world_pose(xformable)
        return {
            "applied": True,
            "object_prim_path": episode_spec.object_prim_path,
            "pose_world": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            },
            "quaternion_wxyz": [float(value) for value in quat],
            "task_quaternion_semantics": "root_orient_before_units_resolve",
            "reset_xform_stack": False,
            "xform_op_order_before": op_order_before,
            "xform_op_order_after": op_order_after,
            "units_resolve_preserved": any("unitsResolve" in name for name in op_order_after),
            "authored_world_position_xyz": list(authored_world_position),
            "authored_world_quaternion_wxyz": list(authored_world_quaternion),
        }

    def _object_initial_pose_diagnostic(
        self,
        episode_spec: EpisodeSpec,
        *,
        label: str,
    ) -> dict[str, Any]:
        """只读检查 PhysX 中物体首帧姿态是否仍等于 task 固定姿态。"""

        if episode_spec.object_initial_pose is None:
            return {"available": False, "label": label, "reason": "object_initial_pose_missing"}
        if self._object is None:
            return {"available": False, "label": label, "reason": "object_reader_unavailable"}
        if self._settled_object_pose is not None:
            expected_position = tuple(float(value) for value in self._settled_object_pose[:3])
            expected_quat = tuple(float(value) for value in self._settled_object_pose[3:7])
            baseline_source = "settled_physx_pose"
        else:
            x, y, z, roll, pitch, yaw = episode_spec.object_initial_pose
            expected_position = (float(x), float(y), float(z))
            pose_report = self._metadata.get("object_pose_setup_report") or {}
            expected_quat = tuple(
                float(value)
                for value in (
                    pose_report.get("authored_world_quaternion_wxyz")
                    or _quat_wxyz_from_rpy(float(roll), float(pitch), float(yaw))
                )
            )
            baseline_source = "task_pose"
        try:
            actual_position_raw, actual_quat_raw = self._object.get_world_pose()
            actual_position = _as_tuple(actual_position_raw)
            actual_quat = _as_tuple(actual_quat_raw)
        except Exception as exc:  # pragma: no cover - 真实 Isaac 后端异常。
            return {
                "available": False,
                "label": label,
                "reason": "live_object_pose_read_failed",
                "error": str(exc),
            }
        position_error = math.sqrt(
            sum((float(actual) - expected) ** 2 for actual, expected in zip(actual_position, expected_position))
        )
        orientation_error = _quat_angle_error_rad(actual_quat, expected_quat)
        return {
            "available": True,
            "label": label,
            "object_prim_path": episode_spec.object_prim_path,
            "expected_position_xyz": expected_position,
            "expected_quaternion_wxyz": expected_quat,
            "actual_pose": (*actual_position, *actual_quat),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "within_tolerance": position_error <= 0.02 and orientation_error <= 0.10,
            "read_only": True,
            "baseline_source": baseline_source,
        }

    def _initialize_object_reader(self, episode_spec: EpisodeSpec) -> None:
        if not episode_spec.object_prim_path:
            return
        import omni.usd
        from isaacsim.core.prims import SingleRigidPrim

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage is unavailable while initializing object reader")
        rigid_body_prim_path = _resolve_rigid_body_prim_path(
            stage,
            episode_spec.object_prim_path,
        )
        self._object = SingleRigidPrim(
            prim_path=rigid_body_prim_path,
            name="full_physics_navigation_object",
            reset_xform_properties=False,
        )
        self._object.initialize()
        self._metadata["object_reader_report"] = {
            "object_root_prim_path": episode_spec.object_prim_path,
            "rigid_body_prim_path": rigid_body_prim_path,
        }

    def prepare_object_for_pick(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        """对齐 baseline：规划前恢复 task 姿态并清零速度。

        苹果保持为动态刚体，只在接触前进入 PhysX sleep；机械臂接触会由
        PhysX 自动唤醒物体，抓取、carry 和 place 阶段不做 TCP 跟随。
        """

        report = self._reset_object_pose_and_motion(
            episode_spec,
            sleep_until_contact=True,
            reason="before_current_state_pick_planning",
        )
        self._metadata["object_prepare_for_pick_report"] = report
        return report

    def begin_object_settle(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        """唤醒任务物体，让其在导航前自然沉降到稳定接触姿态。"""

        del episode_spec
        if self._object is None:
            return {"applied": False, "reason": "object_reader_unavailable"}
        position, orientation = self._object.get_world_pose()
        report = {
            "applied": True,
            "initial_pose": [
                *list(_as_tuple(position)),
                *list(_as_tuple(orientation)),
            ],
            "wake_report": self._set_object_sleeping(enabled=False),
            "baseline_source": "physx_free_settle",
        }
        self._metadata["object_settle_begin_report"] = report
        return report

    def finalize_object_settle(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        """记录稳定 PhysX 位姿并冻结物体，供后续导航和 pick 共用。"""

        if self._object is None:
            return {"applied": False, "reason": "object_reader_unavailable"}
        import torch

        position, orientation = self._object.get_world_pose()
        settled_pose = (
            *tuple(float(value) for value in _as_tuple(position)),
            *tuple(float(value) for value in _as_tuple(orientation)),
        )
        rigid_view = getattr(self._object, "_rigid_prim_view", None)
        if rigid_view is None or not hasattr(rigid_view, "set_velocities"):
            raise RuntimeError("SingleRigidPrim 缺少 GPU 合并速度写入接口。")
        device = getattr(self._runtime, "device", "cpu")
        rigid_view.set_velocities(
            torch.zeros((1, 6), dtype=torch.float32, device=device)
        )
        sleep_report = self._set_object_sleeping(enabled=True)
        self._settled_object_pose = settled_pose

        requested_position = tuple(
            float(value) for value in episode_spec.object_initial_pose[:3]
        )
        pose_report = self._metadata.get("object_pose_setup_report") or {}
        requested_quaternion = tuple(
            float(value)
            for value in (
                pose_report.get("authored_world_quaternion_wxyz")
                or _quat_wxyz_from_rpy(*episode_spec.object_initial_pose[3:])
            )
        )
        requested_position_error = math.sqrt(
            sum(
                (float(actual) - expected) ** 2
                for actual, expected in zip(settled_pose[:3], requested_position)
            )
        )
        requested_orientation_error = _quat_angle_error_rad(
            settled_pose[3:7],
            requested_quaternion,
        )
        report = {
            "applied": True,
            "settled_pose": settled_pose,
            "requested_position_xyz": requested_position,
            "requested_quaternion_wxyz": requested_quaternion,
            "requested_position_error_m": requested_position_error,
            "requested_orientation_error_rad": requested_orientation_error,
            "sleep_report": sleep_report,
            "baseline_source": "settled_physx_pose",
        }
        self._metadata["object_settle_final_report"] = report
        self._metadata["object_pose_debug_after_reset"] = (
            self._object_initial_pose_diagnostic(
                episode_spec,
                label="after_object_physics_settle",
            )
        )
        return report

    def _reset_object_pose_and_motion(
        self,
        episode_spec: EpisodeSpec,
        *,
        sleep_until_contact: bool,
        reason: str,
    ) -> dict[str, Any]:
        if self._object is None or episode_spec.object_initial_pose is None:
            return {
                "applied": False,
                "reason": "object_reader_or_initial_pose_missing",
            }

        import torch

        x, y, z, roll, pitch, yaw = episode_spec.object_initial_pose
        if self._settled_object_pose is None:
            pose_report = self._apply_object_pose(episode_spec)
            target_position = (float(x), float(y), float(z))
            world_quaternion = tuple(
                float(value)
                for value in pose_report.get(
                    "authored_world_quaternion_wxyz",
                    _quat_wxyz_from_rpy(roll, pitch, yaw),
                )
            )
        else:
            target_position = tuple(float(value) for value in self._settled_object_pose[:3])
            world_quaternion = tuple(
                float(value) for value in self._settled_object_pose[3:7]
            )
            pose_report = {
                "applied": False,
                "reason": "reuse_settled_physx_pose",
                "settled_pose": self._settled_object_pose,
            }
        device = getattr(self._runtime, "device", "cpu")
        # 不能调用 SingleRigidPrim.set_world_pose：该 API 会把传入的世界四元数
        # 直接写回根 Orient，破坏任务 RPY 与 unitsResolve 的局部组合语义。
        # 物体位姿已在 stage/PhysX 初始化前写入；这里仅清速度并让其休眠。
        # SingleRigidPrim 没有公开 set_velocities，但内部 RigidPrim view 提供
        # GPU tensor pipeline 所需的合并速度 API。
        rigid_view = getattr(self._object, "_rigid_prim_view", None)
        if rigid_view is None or not hasattr(rigid_view, "set_velocities"):
            raise RuntimeError("SingleRigidPrim 缺少 GPU 合并速度写入接口。")
        rigid_view.set_velocities(
            torch.zeros((1, 6), dtype=torch.float32, device=device)
        )
        sleep_report = self._set_object_sleeping(enabled=sleep_until_contact)
        actual_position, actual_orientation = self._object.get_world_pose()
        report = {
            "applied": True,
            "reason": reason,
            "object_prim_path": episode_spec.object_prim_path,
            "target_position_xyz": list(target_position),
            "target_root_orient_quaternion_wxyz": list(
                _quat_wxyz_from_rpy(roll, pitch, yaw)
            ),
            "target_world_quaternion_wxyz": list(world_quaternion),
            "actual_position_xyz": list(_as_tuple(actual_position)),
            "actual_quaternion_wxyz": list(_as_tuple(actual_orientation)),
            "object_pose_apply_report": pose_report,
            "live_pose_write_skipped": True,
            "live_pose_write_skip_reason": "preserve_root_orient_and_units_resolve",
            "linear_velocity_zeroed": True,
            "angular_velocity_zeroed": True,
            "velocity_write_api": "set_velocities",
            "sleep_until_contact": bool(sleep_until_contact),
            "wake_policy": "physx_contact" if sleep_until_contact else "explicit",
            "sleep_report": sleep_report,
            "object_pose_reset_is_episode_setup": True,
            "object_pose_clamped_to_tcp": False,
        }
        return report

    def _set_object_sleeping(self, *, enabled: bool) -> dict[str, Any]:
        """让任务物体在导航期间保持初始姿态，接触时由 PhysX 自动唤醒。"""

        import omni.physx
        import omni.usd
        from pxr import PhysicsSchemaTools, Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None or self._episode_spec is None or not self._episode_spec.object_prim_path:
            return {"applied": False, "reason": "stage_or_object_path_missing"}
        root = stage.GetPrimAtPath(self._episode_spec.object_prim_path)
        if not root.IsValid():
            return {"applied": False, "reason": "object_prim_missing"}

        interface = omni.physx.get_physx_simulation_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        body_paths: list[str] = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            body_path = str(prim.GetPath())
            encoded_path = PhysicsSchemaTools.sdfPathToInt(prim.GetPath())
            if enabled:
                interface.put_to_sleep(stage_id, encoded_path)
            else:
                interface.wake_up(stage_id, encoded_path)
            body_paths.append(body_path)
        return {
            "applied": bool(body_paths),
            "sleeping": bool(enabled),
            "rigid_body_paths": body_paths,
        }

    def pause(self) -> dict[str, Any]:
        """暂停物理但保留 env，使 keep-window-open 不会释放控制后瘫倒。"""

        self._require_ready()
        report: dict[str, Any] = {"paused": False}
        try:
            self._runtime.sim.pause()
            report.update({"paused": True, "backend": "isaaclab_simulation_context"})
        except Exception as exc:
            report["simulation_context_error"] = str(exc)
            try:
                import omni.timeline

                omni.timeline.get_timeline_interface().pause()
                report.update({"paused": True, "backend": "omni_timeline"})
            except Exception as timeline_exc:
                report["timeline_error"] = str(timeline_exc)
        self._metadata["terminal_hold_report"] = report
        return report

    def export_current_curobo_pick_inputs(
        self,
        *,
        output_dir: str | Path,
        episode_spec: EpisodeSpec,
        state: SimulationState | None = None,
    ) -> dict[str, Any]:
        """导出当前 handoff 状态，供 cuRobo pick 在线重规划使用。"""

        self._require_ready()
        object_prim_path = episode_spec.object_prim_path
        if not object_prim_path:
            raise RuntimeError("当前 task 未配置 object_prim_path，无法生成 pick target。")

        import omni.usd
        import numpy as np

        from source.manipulation.current_state_curobo import (
            build_curobo_state_payload,
            build_grasp_target_payload,
            pose_dict_from_matrix,
            pose_to_matrix,
            write_json,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("当前没有 USD stage，无法导出 cuRobo 输入。")

        T_world_base, base_source = self._read_body_matrix("arm_base_link")
        T_world_tcp, tcp_source, tcp_mode = self._read_tcp_export_matrix()
        q_arm, dq_arm, arm_joint_ids = self._read_named_joint_state(self._config.arm_joint_names)
        q_gripper, dq_gripper, gripper_joint_ids = self._read_named_joint_state(
            self._config.gripper_joint_names
        )
        bbox = self._compute_object_bbox(stage, object_prim_path)
        mesh_truth_contract = _resolve_mesh_truth_manipulation_contract(
            episode_spec.raw_task or {}
        )
        grasp_mode = _resolve_pick_grasp_mode(episode_spec.raw_task or {})
        if (
            mesh_truth_contract["required"]
            and bbox.get("center_source") != "live_physx_object_pose"
        ):
            raise RuntimeError(
                "required Mesh-truth pick target 缺少 live PhysX object pose"
            )
        collision_cuboids = self._export_current_world_collision_cuboids(
            stage=stage,
            episode_spec=episode_spec,
            phase="pick",
            robot_root_path=self._robot_prim_path(),
            object_prim_path=object_prim_path,
            T_world_base=T_world_base,
            object_bbox_center=np.asarray(bbox["center_xyz"], dtype=float),
        )
        world_collision_metadata = self._world_collision_export_metadata(collision_cuboids)

        robot = self._adapter.robot
        joint_names = tuple(str(name) for name in getattr(robot, "joint_names", ()))
        q_full = _as_tuple(robot.data.joint_pos[0])
        dq_full = _as_tuple(robot.data.joint_vel[0])
        state_payload = build_curobo_state_payload(
            q_arm=q_arm,
            dq_arm=dq_arm,
            q_full=q_full,
            dq_full=dq_full,
            dof_names=joint_names,
            q_gripper=q_gripper,
            dq_gripper=dq_gripper,
            T_world_base=T_world_base,
            T_world_tcp=T_world_tcp,
            robot_root_path=self._robot_prim_path(),
            articulation_root_path=self._robot_prim_path(),
            base_frame_path=str(base_source),
            tcp_frame_path=str(tcp_source),
            tcp_mode=tcp_mode,
            world_collision_cuboids=collision_cuboids,
            world_collision_metadata=world_collision_metadata,
            source="IsaacLabNavigationRuntime.current_state_pick_replan",
        )
        target_payload = build_grasp_target_payload(
            grasp_mode=grasp_mode["resolved"],
            object_prim_path=object_prim_path,
            T_world_base=T_world_base,
            bbox_min=bbox["min_xyz"],
            bbox_max=bbox["max_xyz"],
            bbox_center=bbox["center_xyz"],
            bbox_size=bbox["size_xyz"],
            object_long_axis_world=(
                (bbox.get("live_bbox_transform") or {}).get(
                    "long_axis_world_xyz"
                )
            ),
        )
        pick_target_source = target_payload.get("source")
        pick_target_source = (
            pick_target_source if isinstance(pick_target_source, dict) else {}
        )
        expected_target_source_type = f"sim_object_bbox_{grasp_mode['resolved']}"
        mesh_truth_pick_target_report = {
            "configured": bool(mesh_truth_contract["configured"]),
            "required": bool(mesh_truth_contract["required"]),
            "verified": bool(
                pick_target_source.get("type") == expected_target_source_type
                and pick_target_source.get("grasp_mode") == grasp_mode["resolved"]
                and (
                    not mesh_truth_contract["required"]
                    or bbox.get("center_source") == "live_physx_object_pose"
                )
            ),
            "visual_localization_required": False,
            "visual_localization_used": False,
            "pick_tcp_source": "runtime_live_object_bbox",
            "requested_grasp_mode": grasp_mode["requested"],
            "resolved_grasp_mode": grasp_mode["resolved"],
            "expected_target_source_type": expected_target_source_type,
            "target_source_type": pick_target_source.get("type"),
            "object_prim_path": object_prim_path,
            "bbox_center_source": bbox.get("center_source"),
            "bbox_world": bbox,
        }
        if (
            mesh_truth_contract["required"]
            and mesh_truth_pick_target_report["verified"] is not True
        ):
            raise RuntimeError("required Mesh-truth pick target 未验证")

        state_json = write_json(output_path / "pick_state.json", state_payload)
        target_json = write_json(output_path / "pick_target.json", target_payload)
        report = {
            "state_json": state_json,
            "target_json": target_json,
            "object_prim_path": object_prim_path,
            "step_index": None if state is None else int(state.step_index),
            "arm_joint_ids": [int(index) for index in arm_joint_ids],
            "gripper_joint_ids": [int(index) for index in gripper_joint_ids],
            "base_source": str(base_source),
            "tcp_source": str(tcp_source),
            "tcp_mode": tcp_mode,
            "world_base": pose_dict_from_matrix(T_world_base),
            "bbox_world": bbox,
            "collision_cuboid_count": len(collision_cuboids),
            "world_collision_export": world_collision_metadata,
            **_collision_cuboid_diagnostics(collision_cuboids),
            "hidden_distractor_root_paths": list(self._hidden_distractor_root_paths),
            "pick_target": {
                "source": target_payload.get("source", {}),
                "diagnostics": target_payload.get("diagnostics", {}),
            },
            "mesh_truth_pick_target_report": mesh_truth_pick_target_report,
            "target_grasp_position_base": (
                target_payload.get("poses", {})
                .get("grasp", {})
                .get("position_xyz")
            ),
            "target_workspace_base": target_payload.get("diagnostics", {}).get(
                "target_workspace_base",
                {},
            ),
            "world_step_owned_by_pipeline": True,
        }
        self._metadata["last_current_state_curobo_pick_export"] = {
            **report,
            "state_json": str(state_json),
            "target_json": str(target_json),
        }
        self._metadata["last_mesh_truth_pick_target_report"] = (
            mesh_truth_pick_target_report
        )
        return report

    def read_object_bbox_world(self) -> dict[str, Any]:
        """只读返回当前任务物体 bbox，供 target-vs-execution drift 检查。"""

        self._require_ready()
        if self._episode_spec is None or not self._episode_spec.object_prim_path:
            raise RuntimeError("当前 episode 未配置 object_prim_path")
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("当前没有 USD stage，无法读取 object bbox")
        return self._compute_object_bbox(stage, self._episode_spec.object_prim_path)

    def export_current_curobo_place_inputs(
        self,
        *,
        output_dir: str | Path,
        episode_spec: EpisodeSpec,
        state: SimulationState | None = None,
        pick_grasp_quaternion_base: Any | None = None,
    ) -> dict[str, Any]:
        """导出当前 carry handoff 状态，供 cuRobo place 在线重规划使用。"""

        self._require_ready()
        object_prim_path = episode_spec.object_prim_path
        if not object_prim_path:
            raise RuntimeError("当前 task 未配置 object_prim_path，无法生成 place target。")

        import omni.usd
        import numpy as np

        from source.manipulation.current_state_curobo import (
            build_arm_place_target_payload,
            build_curobo_state_payload,
            pose_dict_from_matrix,
            write_json,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("当前没有 USD stage，无法导出 cuRobo place 输入。")

        mesh_truth_contract = _resolve_mesh_truth_manipulation_contract(
            episode_spec.raw_task or {}
        )
        receptacle_support_report = inspect_task_receptacle_support_stage(
            stage,
            episode_spec.raw_task or {},
            source="curobo_place_target_runtime_stage",
        )
        pick_export = self._metadata.get("last_current_state_curobo_pick_export")
        pick_export = pick_export if isinstance(pick_export, dict) else {}
        pick_object_bbox = pick_export.get("bbox_world")
        pick_target_report = pick_export.get("mesh_truth_pick_target_report")
        if mesh_truth_contract["required"] and (
            not isinstance(pick_target_report, dict)
            or pick_target_report.get("verified") is not True
        ):
            raise RuntimeError(
                "required Mesh-truth place target 缺少已验证的 pick target 报告"
            )
        place_pose_world = self._place_pose_world_from_episode(
            episode_spec,
            receptacle_support_report=receptacle_support_report,
            pick_object_bbox=(
                pick_object_bbox if isinstance(pick_object_bbox, dict) else None
            ),
        )

        T_world_base, base_source = self._read_body_matrix("arm_base_link")
        T_world_tcp, tcp_source, tcp_mode = self._read_tcp_export_matrix()
        q_arm, dq_arm, arm_joint_ids = self._read_named_joint_state(self._config.arm_joint_names)
        q_gripper, dq_gripper, gripper_joint_ids = self._read_named_joint_state(
            self._config.gripper_joint_names
        )
        bbox = self._compute_object_bbox(stage, object_prim_path)
        if (
            mesh_truth_contract["required"]
            and bbox.get("center_source") != "live_physx_object_pose"
        ):
            raise RuntimeError(
                "required Mesh-truth place target 缺少当前 live PhysX object pose"
            )
        mesh_truth_place_target_report = self._metadata.get(
            "last_mesh_truth_place_target_report"
        )
        if isinstance(mesh_truth_place_target_report, dict):
            mesh_truth_place_target_report = {
                **mesh_truth_place_target_report,
                "required": bool(mesh_truth_contract["required"]),
                "place_tcp_source": (
                    "runtime_receptacle_bbox_plus_pick_object_bbox_plus_current_tcp_offset"
                ),
                "current_object_bbox_center_source": bbox.get("center_source"),
                "current_object_center_live_verified": (
                    bbox.get("center_source") == "live_physx_object_pose"
                ),
                "current_tcp_offset_source": (
                    "runtime_current_tcp_and_live_object_center"
                ),
            }
            mesh_truth_place_target_report["verified"] = bool(
                mesh_truth_place_target_report.get("verified") is True
                and (
                    not mesh_truth_contract["required"]
                    or mesh_truth_place_target_report[
                        "current_object_center_live_verified"
                    ]
                )
            )
            self._metadata["last_mesh_truth_place_target_report"] = (
                mesh_truth_place_target_report
            )
        else:
            mesh_truth_place_target_report = {
                "configured": False,
                "required": bool(mesh_truth_contract["required"]),
                "verified": not bool(mesh_truth_contract["required"]),
            }
        if (
            mesh_truth_contract["required"]
            and mesh_truth_place_target_report.get("verified") is not True
        ):
            raise RuntimeError("required Mesh-truth place target 未验证")
        collision_cuboids = self._export_current_world_collision_cuboids(
            stage=stage,
            episode_spec=episode_spec,
            phase="place",
            robot_root_path=self._robot_prim_path(),
            object_prim_path=object_prim_path,
            T_world_base=T_world_base,
            object_bbox_center=np.asarray(bbox["center_xyz"], dtype=float),
        )
        world_collision_metadata = self._world_collision_export_metadata(collision_cuboids)

        robot = self._adapter.robot
        joint_names = tuple(str(name) for name in getattr(robot, "joint_names", ()))
        q_full = _as_tuple(robot.data.joint_pos[0])
        dq_full = _as_tuple(robot.data.joint_vel[0])
        state_payload = build_curobo_state_payload(
            q_arm=q_arm,
            dq_arm=dq_arm,
            q_full=q_full,
            dq_full=dq_full,
            dof_names=joint_names,
            q_gripper=q_gripper,
            dq_gripper=dq_gripper,
            T_world_base=T_world_base,
            T_world_tcp=T_world_tcp,
            robot_root_path=self._robot_prim_path(),
            articulation_root_path=self._robot_prim_path(),
            base_frame_path=str(base_source),
            tcp_frame_path=str(tcp_source),
            tcp_mode=tcp_mode,
            world_collision_cuboids=collision_cuboids,
            world_collision_metadata=world_collision_metadata,
            source="IsaacLabNavigationRuntime.current_state_place_replan",
        )
        target_payload = build_arm_place_target_payload(
            object_prim_path=object_prim_path,
            T_world_base=T_world_base,
            T_world_tcp=T_world_tcp,
            bbox_min=bbox["min_xyz"],
            bbox_max=bbox["max_xyz"],
            bbox_center=bbox["center_xyz"],
            bbox_size=bbox["size_xyz"],
            place_pose_world=place_pose_world,
            pick_grasp_quaternion_base=pick_grasp_quaternion_base,
        )

        state_json = write_json(output_path / "place_state.json", state_payload)
        target_json = write_json(output_path / "place_target.json", target_payload)
        report = {
            "state_json": state_json,
            "target_json": target_json,
            "object_prim_path": object_prim_path,
            "step_index": None if state is None else int(state.step_index),
            "arm_joint_ids": [int(index) for index in arm_joint_ids],
            "gripper_joint_ids": [int(index) for index in gripper_joint_ids],
            "base_source": str(base_source),
            "tcp_source": str(tcp_source),
            "tcp_mode": tcp_mode,
            "world_base": pose_dict_from_matrix(T_world_base),
            "world_tcp": pose_dict_from_matrix(T_world_tcp),
            "bbox_world": bbox,
            "collision_cuboid_count": len(collision_cuboids),
            "world_collision_export": world_collision_metadata,
            **_collision_cuboid_diagnostics(collision_cuboids),
            "hidden_distractor_root_paths": list(self._hidden_distractor_root_paths),
            "receptacle_support_report": receptacle_support_report,
            "mesh_truth_place_target_report": mesh_truth_place_target_report,
            "desired_final_object_center_world": (
                target_payload.get("source", {}).get("desired_final_object_center_world")
            ),
            "release_object_center_world": (
                target_payload.get("source", {}).get("release_object_center_world")
            ),
            "target_workspace_base": target_payload.get("diagnostics", {}).get(
                "target_workspace_base",
                {},
            ),
            "world_step_owned_by_pipeline": True,
        }
        self._metadata["last_current_state_curobo_place_export"] = {
            **report,
            "state_json": str(state_json),
            "target_json": str(target_json),
        }
        return report

    def _place_pose_world_from_episode(
        self,
        episode_spec: EpisodeSpec,
        *,
        receptacle_support_report: dict[str, Any] | None = None,
        pick_object_bbox: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """解析 place pose；可选由运行时 Mesh 真值覆盖静态 XYZ。"""

        raw_place = dict((episode_spec.raw_task or {}).get("place") or {})
        mesh_truth_result = _derive_mesh_truth_place_pose(
            raw_place=raw_place,
            receptacle_support_report=receptacle_support_report,
            pick_object_bbox=pick_object_bbox,
        )
        if mesh_truth_result is not None:
            payload, mesh_truth_report = mesh_truth_result
            if hasattr(self, "_metadata"):
                self._metadata["last_mesh_truth_place_target_report"] = (
                    mesh_truth_report
                )
        else:
            raw_pose = raw_place.get("place_pose_world")
            if isinstance(raw_pose, dict):
                payload = dict(raw_pose)
            elif episode_spec.place_target_pose is not None:
                x, y, z, roll, pitch, yaw = episode_spec.place_target_pose
                payload = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                }
            else:
                raise RuntimeError(
                    "当前 task 缺少 place.place_pose_world，无法生成 arm-place target。"
                )
        # baseline arm-place 只读取 clearance 字段。release_height/retreat_height
        # 属于旧 put 流程，映射后会把苹果抬高到 4~5 cm 再松爪，造成明显自由落体。
        for target_key in (
            "release_clearance",
            "pre_place_clearance",
            "retreat_clearance",
        ):
            if target_key not in payload and target_key in raw_place:
                payload[target_key] = raw_place[target_key]
        payload["release_clearance"] = max(
            float(payload.get("release_clearance", 0.0)),
            self._config.place_release_clearance_min_m,
        )
        payload["pre_place_clearance"] = max(
            float(payload.get("pre_place_clearance", 0.0)),
            payload["release_clearance"],
            self._config.place_pre_clearance_min_m,
        )
        return payload

    def _read_named_joint_state(
        self,
        joint_names: tuple[str, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...], list[int]]:
        """按给定关节顺序读取当前 q/dq，不改变 articulation 状态。"""

        robot = self._adapter.robot
        joint_ids, _names = robot.find_joints(list(joint_names), preserve_order=True)
        if len(joint_ids) != len(joint_names):
            raise RuntimeError(
                f"IsaacLab articulation 缺少关节 {joint_names}，实际 joint_ids={joint_ids}"
            )
        q = tuple(_item(robot.data.joint_pos[0, index]) for index in joint_ids)
        dq = tuple(_item(robot.data.joint_vel[0, index]) for index in joint_ids)
        return q, dq, [int(index) for index in joint_ids]

    def _read_body_matrix(self, body_name: str) -> tuple[Any, str]:
        """读取 link/frame 的 world SE(3)；fixed frame 缺失时按 baseline 回退。"""

        import numpy as np

        if body_name == "arm_base_link":
            # IsaacLab 会把 fixed arm_base_link 合并掉，USD prim 又不会随实时
            # PhysX root pose 更新；因此必须用实时 base body 加 URDF fixed joint。
            T_world_base_body, base_source = self._read_body_matrix_from_tensor("base")
            T_base_body_arm_base = np.eye(4, dtype=float)
            T_base_body_arm_base[:3, 3] = (0.12, 0.0, 0.05)
            return (
                T_world_base_body @ T_base_body_arm_base,
                f"{base_source}+fixed_arm_base_joint(0.12,0,0.05)",
            )

        try:
            return self._read_body_matrix_from_tensor(body_name)
        except RuntimeError:
            pass

        try:
            return self._read_stage_prim_matrix_by_name(body_name)
        except (RuntimeError, ModuleNotFoundError):
            raise

    def _read_body_matrix_from_tensor(self, body_name: str) -> tuple[Any, str]:
        """从 IsaacLab 实时 body tensor 读取 link 的 world SE(3)。"""

        from source.manipulation.current_state_curobo import pose_to_matrix

        robot = self._adapter.robot
        try:
            body_ids, _names = robot.find_bodies([body_name], preserve_order=True)
        except ValueError as exc:
            raise RuntimeError(f"IsaacLab robot body 中找不到 {body_name}") from exc
        if not body_ids:
            raise RuntimeError(f"IsaacLab robot body 中找不到 {body_name}")
        body_id = int(body_ids[0])
        matrix = pose_to_matrix(
            _as_tuple(robot.data.body_pos_w[0, body_id]),
            _as_tuple(robot.data.body_quat_w[0, body_id]),
        )
        return matrix, f"isaaclab_body:{body_name}"

    def _read_stage_prim_matrix_by_name(self, prim_name: str) -> tuple[Any, str]:
        """按 baseline 的方式从 USD stage 读取固定 frame prim。"""

        import omni.usd
        import numpy as np
        from pxr import Usd, UsdGeom

        from source.manipulation.current_state_curobo import pose_to_matrix

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("当前没有 USD stage")

        candidates = []
        for prim in stage.TraverseAll():
            if prim.IsValid() and prim.GetName() == prim_name:
                candidates.append(str(prim.GetPath()))
        if not candidates:
            raise RuntimeError(f"USD stage 中找不到 prim name: {prim_name}")

        preferred = [
            path
            for path in candidates
            if "/Robot" in path or "/go2_x5" in path or "/mec_arm_6dof" in path
        ]
        prim_path = sorted(preferred or candidates)[0]
        prim = stage.GetPrimAtPath(prim_path)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        usd_matrix = xform_cache.GetLocalToWorldTransform(prim)
        translation = usd_matrix.ExtractTranslation()
        rotation = usd_matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        matrix = pose_to_matrix(
            np.array([translation[0], translation[1], translation[2]], dtype=float),
            np.array([rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=float),
        )
        return matrix, f"usd_prim:{prim_path}"

    def _read_tcp_export_matrix(self) -> tuple[Any, str, str]:
        """优先读取 grasp_tcp_link；缺失时按 baseline 使用 arm_link6 固定偏移。"""

        import numpy as np

        try:
            return (
                *self._read_body_matrix_from_tensor("grasp_tcp_link"),
                "direct_tool_frame_body",
            )
        except RuntimeError:
            pass
        T_world_link6, source = self._read_body_matrix_from_tensor("arm_link6")
        # 该偏移和 scripts/isaac/01_export_go2_x5_state.py 中的 fallback 保持一致。
        T_link6_tcp = np.eye(4, dtype=float)
        T_link6_tcp[0, 3] = 0.15757
        return T_world_link6 @ T_link6_tcp, source, "fallback_parent_link_plus_fixed_offset"

    def _compute_object_bbox(self, stage: Any, object_prim_path: str) -> dict[str, Any]:
        """读取物体尺寸，并用实时 PhysX pose 修正动态物体中心。"""

        import numpy as np
        from pxr import Usd, UsdGeom

        prim = stage.GetPrimAtPath(object_prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"object prim 不存在: {object_prim_path}")
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        bound = bbox_cache.ComputeWorldBound(prim)
        aligned_box = bound.ComputeAlignedBox()
        bbox_min = np.asarray(aligned_box.GetMin(), dtype=float)
        bbox_max = np.asarray(aligned_box.GetMax(), dtype=float)
        if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)):
            raise RuntimeError(f"object bbox 非法: min={bbox_min}, max={bbox_max}")
        size = bbox_max - bbox_min
        center = 0.5 * (bbox_min + bbox_max)
        center_source = "usd_bbox"
        live_transform_report: dict[str, Any] | None = None
        if self._object is not None:
            # Fabric/PhysX 运动后，UsdGeom.BBoxCache 可能仍返回 authored 位姿。
            # 先把 authored world AABB 还原到刚体局部系，再用 live pose 映射，
            # 避免默认刚体原点恰好等于 Mesh bbox 中心。
            reader_report = self._metadata.get("object_reader_report")
            reader_report = reader_report if isinstance(reader_report, dict) else {}
            rigid_body_prim_path = str(
                reader_report.get("rigid_body_prim_path")
                or _resolve_rigid_body_prim_path(stage, object_prim_path)
            )
            rigid_body_prim = stage.GetPrimAtPath(rigid_body_prim_path)
            if not rigid_body_prim.IsValid() or not rigid_body_prim.IsA(UsdGeom.Xformable):
                raise RuntimeError(
                    f"object rigid body prim 不是有效 Xformable: {rigid_body_prim_path}"
                )
            authored_rigid_position, authored_rigid_quaternion = (
                _xformable_world_pose(UsdGeom.Xformable(rigid_body_prim))
            )
            live_position, live_orientation = self._object.get_world_pose()
            live_transform_report = _transform_authored_aabb_to_live_rigid_pose(
                authored_bbox_min=bbox_min,
                authored_bbox_max=bbox_max,
                authored_rigid_position=authored_rigid_position,
                authored_rigid_quaternion_wxyz=authored_rigid_quaternion,
                live_rigid_position=_as_tuple(live_position),
                live_rigid_quaternion_wxyz=_as_tuple(live_orientation),
            )
            bbox_min = np.asarray(live_transform_report["min_xyz"], dtype=float)
            bbox_max = np.asarray(live_transform_report["max_xyz"], dtype=float)
            center = np.asarray(live_transform_report["center_xyz"], dtype=float)
            size = np.asarray(live_transform_report["size_xyz"], dtype=float)
            center_source = "live_physx_object_pose"
        report = {
            "min_xyz": bbox_min.tolist(),
            "max_xyz": bbox_max.tolist(),
            "center_xyz": center.tolist(),
            "size_xyz": size.tolist(),
            "center_source": center_source,
            "read_only": True,
        }
        if live_transform_report is not None:
            report["live_bbox_transform"] = live_transform_report
        return report

    def _export_current_world_collision_cuboids(
        self,
        *,
        stage: Any,
        episode_spec: EpisodeSpec,
        phase: str,
        robot_root_path: str,
        object_prim_path: str,
        T_world_base: Any,
        object_bbox_center: Any,
    ) -> list[dict[str, Any]]:
        """导出局部环境 cuboid，避免 cuRobo 重规划穿过桌面。"""

        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics

        from source.manipulation.current_state_curobo import pose_dict_from_matrix

        padding_xy_m = float(self._config.world_collision_padding_m)
        padding_z_m = float(self._config.world_collision_vertical_padding_m)
        min_size_m = float(self._config.world_collision_min_size_m)
        max_obstacles = int(self._config.world_collision_max_obstacles)
        local_radius_m = float(self._config.world_collision_local_radius_m)
        clip_large_support = bool(self._config.world_collision_clip_large_support_obstacles)
        clip_half_extent_m = float(self._config.world_collision_large_obstacle_clip_half_extent_m)
        max_extent_m = 2.0
        max_height_m = 1.6
        max_volume_m3 = 2.5
        excluded_keywords = (
            "floorplan",
            "wall",
            "door",
            "window",
            "ceiling",
            "wardrobe",
        )
        excluded_prefixes = ("/World/debug_", "/World/Looks", "/World/Render")

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        T_world_base = np.asarray(T_world_base, dtype=float)
        T_base_world = np.linalg.inv(T_world_base)
        reference_point = np.asarray(object_bbox_center, dtype=float)
        base_position = T_world_base[:3, 3].copy()
        candidates = _task_world_collision_cuboids(
            raw_task=episode_spec.raw_task,
            phase=phase,
            T_world_base=T_world_base,
            reference_point=reference_point,
            padding_xy_m=padding_xy_m,
            padding_z_m=padding_z_m,
        )

        for prim in stage.TraverseAll():
            if not prim.IsValid() or not prim.IsActive():
                continue
            try:
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
            except Exception:
                continue
            prim_path = str(prim.GetPath())
            # IsaacLab 的 robot.prim_path 在部分版本中是 env regex，不能只靠 path overlap。
            if "/Robot" in prim_path or "/go2_x5" in prim_path:
                continue
            if _path_overlaps(prim_path, robot_root_path) or _path_overlaps(
                prim_path,
                object_prim_path,
            ):
                continue
            if _path_is_excluded_by_roots(
                prim_path,
                self._hidden_distractor_root_paths,
            ):
                continue
            prim_path_lower = prim_path.lower()
            if any(prim_path.startswith(prefix) for prefix in excluded_prefixes):
                continue
            if any(keyword in prim_path_lower for keyword in excluded_keywords):
                continue

            try:
                bound = bbox_cache.ComputeWorldBound(prim)
                aligned_box = bound.ComputeAlignedBox()
                bbox_min = np.asarray(aligned_box.GetMin(), dtype=float)
                bbox_max = np.asarray(aligned_box.GetMax(), dtype=float)
            except Exception:
                continue
            if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)):
                continue
            distance_to_reference = _distance_point_to_aabb_xy(
                reference_point[:2],
                bbox_min,
                bbox_max,
            )
            if distance_to_reference > local_radius_m:
                continue

            size = bbox_max - bbox_min
            if float(np.max(size)) < min_size_m:
                continue
            clipped_from_large_obstacle = False
            original_bbox_min = bbox_min.copy()
            original_bbox_max = bbox_max.copy()
            original_size = size.copy()
            if (
                float(np.max(size)) > max_extent_m
                or float(size[2]) > max_height_m
                or float(np.prod(size)) > max_volume_m3
            ):
                support_top_z = float(bbox_max[2])
                support_like_height = (
                    float(reference_point[2]) - 0.15
                    <= support_top_z
                    <= float(reference_point[2]) + 0.04
                )
                if not (clip_large_support and support_like_height):
                    continue
                clip_min_xy = reference_point[:2] - clip_half_extent_m
                clip_max_xy = reference_point[:2] + clip_half_extent_m
                clipped_min_xy = np.maximum(bbox_min[:2], clip_min_xy)
                clipped_max_xy = np.minimum(bbox_max[:2], clip_max_xy)
                if np.any(clipped_max_xy <= clipped_min_xy):
                    continue
                bbox_min = bbox_min.copy()
                bbox_max = bbox_max.copy()
                bbox_min[:2] = clipped_min_xy
                bbox_max[:2] = clipped_max_xy
                size = bbox_max - bbox_min
                if float(np.max(size)) < min_size_m:
                    continue
                clipped_from_large_obstacle = True
            if _point_inside_aabb(base_position, bbox_min, bbox_max, margin=padding_xy_m):
                continue

            center_world = 0.5 * (bbox_min + bbox_max)
            padding_xyz = np.asarray(
                [padding_xy_m, padding_xy_m, padding_z_m],
                dtype=float,
            )
            padded_size = np.maximum(size + 2.0 * padding_xyz, min_size_m)
            T_world_obstacle = np.eye(4, dtype=float)
            T_world_obstacle[:3, 3] = center_world
            T_base_obstacle = T_base_world @ T_world_obstacle
            candidates.append(
                {
                    "prim_path": prim_path,
                    "type": "cuboid_from_world_aabb",
                    "distance_to_reference_xy_m": distance_to_reference,
                    "dims_xyz": padded_size.tolist(),
                    "raw_bbox_world": {
                        "min_xyz": bbox_min.tolist(),
                        "max_xyz": bbox_max.tolist(),
                        "center_xyz": center_world.tolist(),
                        "size_xyz": size.tolist(),
                    },
                    "source_raw_bbox_world": {
                        "min_xyz": original_bbox_min.tolist(),
                        "max_xyz": original_bbox_max.tolist(),
                        "size_xyz": original_size.tolist(),
                    },
                    "pose_world": pose_dict_from_matrix(T_world_obstacle),
                    "pose_base": pose_dict_from_matrix(T_base_obstacle),
                    "padding_m": padding_xy_m,
                    "padding_xy_m": padding_xy_m,
                    "padding_z_m": padding_z_m,
                    "clipped_from_large_obstacle": clipped_from_large_obstacle,
                }
            )

        # stage 中 navigation terrain 通常先于桌面出现。必须先完成局部候选收集，
        # 再按苹果距离截断，否则前 16 个 terrain prim 会让 cuRobo 看不到桌面。
        candidates.sort(key=_collision_candidate_sort_key)
        obstacles = candidates[:max_obstacles]
        required_ids = {
            str(candidate["task_collision_id"])
            for candidate in candidates
            if candidate.get("task_collision_required")
        }
        selected_required_ids = {
            str(candidate["task_collision_id"])
            for candidate in obstacles
            if candidate.get("task_collision_required")
        }
        if required_ids != selected_required_ids:
            missing = sorted(required_ids - selected_required_ids)
            raise RuntimeError(
                "任务要求的 CuRobo world collision 被 max_obstacles 截断: "
                f"{missing}"
            )
        for index, obstacle in enumerate(obstacles):
            obstacle["name"] = _sanitize_obstacle_name(
                str(obstacle.get("task_collision_name") or obstacle["prim_path"]),
                index,
            )
        return obstacles

    def _world_collision_export_metadata(
        self,
        cuboids: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """记录 cuRobo 虚拟障碍导出参数，便于排查桌沿 clearance。"""

        padding_xy_m = float(self._config.world_collision_padding_m)
        padding_z_m = float(self._config.world_collision_vertical_padding_m)
        task_configured_count = sum(
            1 for cuboid in cuboids if cuboid.get("task_configured")
        )
        return {
            "representation": (
                "IsaacLab CollisionAPI world AABB plus task-configured world cuboids "
                "in arm_base_link frame"
                if task_configured_count
                else "IsaacLab current CollisionAPI world AABB exported as cuRobo "
                "cuboids in arm_base_link frame"
            ),
            "padding_m": padding_xy_m,
            "padding_xy_m": padding_xy_m,
            "padding_z_m": padding_z_m,
            "clearance_margin_m": padding_xy_m,
            "min_size_m": float(self._config.world_collision_min_size_m),
            "max_obstacles": int(self._config.world_collision_max_obstacles),
            "local_radius_m": float(self._config.world_collision_local_radius_m),
            "clip_large_support_obstacles": bool(
                self._config.world_collision_clip_large_support_obstacles
            ),
            "large_obstacle_clip_half_extent_m": float(
                self._config.world_collision_large_obstacle_clip_half_extent_m
            ),
            "obstacle_count": len(cuboids),
            "task_configured_obstacle_count": task_configured_count,
            "required_task_collision_count": sum(
                1 for cuboid in cuboids if cuboid.get("task_collision_required")
            ),
            "task_collision_ids": [
                str(cuboid.get("task_collision_id"))
                for cuboid in cuboids
                if cuboid.get("task_configured")
            ],
            "preserve_top_padding_count": sum(
                1
                for cuboid in cuboids
                if cuboid.get("padding_mode") == "preserve_top"
            ),
            "clipped_large_obstacle_count": sum(
                1 for cuboid in cuboids if cuboid.get("clipped_from_large_obstacle")
            ),
            "nearest_obstacle_distance_xy_m": (
                None
                if not cuboids
                else float(cuboids[0].get("distance_to_reference_xy_m", 0.0))
            ),
            "note": (
                "任务支撑代理可使用 preserve_top，使 padding_z 仅向局部 -Z 膨胀；"
                "USD AABB 仍沿用旧场景的对称膨胀语义。"
            ),
        }

    def _finish_control_step(self) -> None:
        import torch

        self._runtime.episode_length_buf += 1
        self._runtime.common_step_counter += 1
        self._runtime.reset_buf = self._runtime.termination_manager.compute()
        self._runtime.reset_terminated = self._runtime.termination_manager.terminated
        self._runtime.reset_time_outs = self._runtime.termination_manager.time_outs
        self._environment_terminated = bool(torch.any(self._runtime.reset_buf).item())
        self._runtime.reward_buf = self._runtime.reward_manager.compute(
            dt=self._runtime.step_dt
        )
        self._runtime.command_manager.compute(dt=self._runtime.step_dt)
        if "interval" in self._runtime.event_manager.available_modes:
            self._runtime.event_manager.apply(mode="interval", dt=self._runtime.step_dt)
        self._runtime.obs_buf = self._runtime.observation_manager.compute(
            update_history=True
        )
        self._runtime.recorder_manager.record_post_step()
        self._adapter.update_observations(self._to_tensor_dict(self._runtime.obs_buf))

    def _to_tensor_dict(self, observations: Any) -> Any:
        from tensordict import TensorDict

        if isinstance(observations, TensorDict):
            return observations
        return TensorDict(observations, batch_size=[self._runtime.num_envs])

    def _stage_gripper_target(self, action: RobotAction) -> dict[str, Any]:
        """把 pipeline 夹爪命令转成 Isaac Lab gripper position target。"""

        if action.gripper_command is None:
            return {
                "target_staged": False,
                "reason": "no_gripper_command",
            }

        metadata = action.metadata or {}
        if "gripper_joint_positions" in metadata:
            joint_names = tuple(
                str(name)
                for name in metadata.get("gripper_joint_names", self._config.gripper_joint_names)
            )
            if joint_names != tuple(self._config.gripper_joint_names):
                raise RuntimeError(
                    "IsaacLabNavigationRuntime only supports fixed gripper joint order: "
                    f"{self._config.gripper_joint_names}, got {joint_names}"
                )
            target = tuple(float(value) for value in metadata["gripper_joint_positions"])
            source = "metadata"
        elif action.gripper_command == "open":
            joint_names = tuple(self._config.gripper_joint_names)
            target = tuple(self._config.gripper_open_position for _ in joint_names)
            source = "command_open"
        elif action.gripper_command == "close":
            joint_names = tuple(self._config.gripper_joint_names)
            target = tuple(self._config.gripper_close_position for _ in joint_names)
            source = "command_close"
        elif action.gripper_command == "hold":
            # 没有显式 target 时不清空 adapter 内已有目标，避免导航阶段丢掉 close target。
            return {
                "target_staged": False,
                "gripper_command": action.gripper_command,
                "reason": "hold_without_explicit_target",
            }
        else:
            raise RuntimeError(f"unsupported gripper command: {action.gripper_command}")

        if len(target) != len(self._config.gripper_joint_names):
            raise RuntimeError(
                "gripper_joint_positions length does not match gripper_joint_names: "
                f"{len(target)} != {len(self._config.gripper_joint_names)}"
            )
        self._adapter.set_gripper_joint_target(target)
        return {
            "target_staged": True,
            "gripper_command": action.gripper_command,
            "target_source": source,
            "gripper_joint_names": joint_names,
            "gripper_joint_positions": target,
            # 这里只有 target staging；实际 physics step 仍由 pipeline 主循环触发。
            "world_step_owned_by_pipeline": True,
        }

    def _stage_arm_target(self, action: RobotAction) -> dict[str, Any]:
        """把 arm position target 写入当前 task 支持的机械臂控制通道。"""

        if action.arm_joint_positions is None:
            return {
                "target_staged": False,
                "reason": "no_arm_joint_target",
            }

        metadata = action.metadata or {}
        joint_names = tuple(
            str(name)
            for name in metadata.get("arm_joint_names", self._config.arm_joint_names)
        )
        expected_names = tuple(self._config.arm_joint_names)
        if joint_names != expected_names:
            raise RuntimeError(
                "IsaacLabNavigationRuntime only supports fixed arm joint order: "
                f"{expected_names}, got {joint_names}"
            )
        target = tuple(float(value) for value in action.arm_joint_positions)
        if len(target) != len(expected_names):
            raise RuntimeError(
                "arm_joint_positions length does not match arm_joint_names: "
                f"{len(target)} != {len(expected_names)}"
            )

        override_report = self._adapter.set_direct_arm_action_override(True)
        action_indices = tuple(int(index) for index in override_report.get("arm_action_indices") or ())
        direct_override_available = (
            bool(override_report.get("action_term_available"))
            and len(action_indices) == len(expected_names)
        )
        if direct_override_available:
            arm_control_mode = "policy_action_override"
            direct_override_enabled = True
        else:
            if not hasattr(self._adapter, "apply_arm_joint_target"):
                raise RuntimeError(
                    "Isaac Lab arm target control is unavailable: policy action does not expose "
                    f"all arm joints and adapter has no independent arm target path: {override_report}"
                )
            disable_report = self._adapter.set_direct_arm_action_override(False)
            self._metadata["last_direct_arm_action_override_disable_report"] = disable_report
            arm_control_mode = "independent_position_target"
            direct_override_enabled = False
        arm_velocity_hold = bool(metadata.get("segment_type") == "post_motion_hold")
        self._adapter.set_arm_joint_target(
            target,
            hold_velocity=arm_velocity_hold,
        )
        self._pending_arm_tracking_target = {
            "source": action.source,
            "pipeline_state": metadata.get("carry_arm_home_phase"),
            "joint_names": joint_names,
            "target_positions": target,
            "applied_step_index": getattr(self, "_step_calls", 0),
        }
        return {
            "target_staged": True,
            "arm_joint_names": joint_names,
            "arm_joint_positions": target,
            "arm_control_mode": arm_control_mode,
            "direct_arm_action_override": direct_override_enabled,
            "arm_action_indices": action_indices,
            "arm_velocity_hold": arm_velocity_hold,
            # 这里只替换 policy action 槽，禁止通过 direct joint state 制造执行结果。
            "uses_direct_joint_state": False,
            "world_step_owned_by_pipeline": True,
        }

    def _consume_pending_arm_tracking_target(self, robot: Any) -> None:
        pending = self._pending_arm_tracking_target
        if pending is None:
            return
        joint_ids = tuple(int(index) for index in self._adapter.arm_joint_ids)
        target_positions = tuple(float(value) for value in pending["target_positions"])
        joint_names = tuple(str(name) for name in pending["joint_names"])
        if len(joint_ids) != len(target_positions):
            self._pending_arm_tracking_target = None
            self._metadata["last_arm_tracking_report"] = {
                "available": False,
                "reason": "arm_joint_id_count_mismatch",
                "joint_ids": joint_ids,
                "target_positions": target_positions,
            }
            return

        actual_positions = _as_tuple(robot.data.joint_pos[0, list(joint_ids)])
        position_errors = tuple(
            actual - target
            for actual, target in zip(actual_positions, target_positions)
        )
        abs_errors = tuple(abs(value) for value in position_errors)
        peak_index = max(range(len(abs_errors)), key=abs_errors.__getitem__) if abs_errors else 0
        max_abs_error = abs_errors[peak_index] if abs_errors else 0.0
        report = {
            "available": True,
            "source": pending.get("source"),
            "pipeline_state": pending.get("pipeline_state"),
            "joint_names": joint_names,
            "joint_ids": joint_ids,
            "target_positions": target_positions,
            "actual_positions": actual_positions,
            "position_errors": position_errors,
            "max_abs_error": max_abs_error,
            "mean_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
            "peak_joint": {
                "joint_name": joint_names[peak_index] if peak_index < len(joint_names) else None,
                "joint_order_index": peak_index,
                "target_position": (
                    target_positions[peak_index] if peak_index < len(target_positions) else None
                ),
                "actual_position": (
                    actual_positions[peak_index] if peak_index < len(actual_positions) else None
                ),
                "position_error": (
                    position_errors[peak_index] if peak_index < len(position_errors) else None
                ),
                "abs_error": max_abs_error,
            },
            "applied_step_index": pending.get("applied_step_index"),
            "sample_step_index": self._step_calls,
        }
        sample_count = int(self._metadata.get("arm_tracking_sample_count", 0)) + 1
        previous_peak = self._metadata.get("arm_tracking_peak_report")
        if (
            not isinstance(previous_peak, dict)
            or max_abs_error > float(previous_peak.get("max_abs_error", -1.0))
        ):
            self._metadata["arm_tracking_peak_report"] = report
            previous_peak = report
        aggregate_max = max(
            float(self._metadata.get("arm_tracking_max_abs_error", 0.0)),
            max_abs_error,
        )
        self._metadata.update(
            {
                "last_arm_tracking_report": report,
                "arm_tracking_sample_count": sample_count,
                "arm_tracking_max_abs_error": aggregate_max,
                "arm_tracking_report": {
                    "sample_count": sample_count,
                    "max_abs_error": aggregate_max,
                    "peak_report": previous_peak,
                    "latest_report": report,
                },
            }
        )
        self._pending_arm_tracking_target = None

    def _read_object_state(
        self,
    ) -> tuple[
        tuple[float, float, float, float, float, float, float] | None,
        tuple[float, float, float, float, float, float] | None,
    ]:
        if self._object is None:
            return None, None
        position, orientation = self._object.get_world_pose()
        return (
            (*_as_tuple(position), *_as_tuple(orientation)),
            (
                *_as_tuple(self._object.get_linear_velocity()),
                *_as_tuple(self._object.get_angular_velocity()),
            ),
        )

    def _read_tcp_pose(
        self,
    ) -> tuple[float, float, float, float, float, float, float] | None:
        from source.manipulation.current_state_curobo import matrix_to_pose

        try:
            matrix, _source, _mode = self._read_tcp_export_matrix()
        except RuntimeError:
            return None
        position, quaternion = matrix_to_pose(matrix)
        return (*_as_tuple(position), *_as_tuple(quaternion))

    def _wrist_camera_object_clearance_config(self) -> dict[str, Any] | None:
        """解析任务级 wrist 近裁剪安全门禁。"""

        if self._episode_spec is None:
            return None
        raw_task = self._episode_spec.raw_task
        recording = raw_task.get("recording") if isinstance(raw_task, dict) else None
        recording = recording if isinstance(recording, dict) else {}
        raw = recording.get("wrist_camera_object_clearance")
        if not isinstance(raw, dict) or not raw.get("enabled", False):
            return None
        shape = str(raw.get("shape") or "").strip().lower()
        if shape != "cylinder_local_z":
            raise RuntimeError(
                "recording.wrist_camera_object_clearance.shape "
                "当前只支持 cylinder_local_z"
            )
        try:
            config = {
                "enabled": True,
                "required_for_training": bool(
                    raw.get("required_for_training", False)
                ),
                "shape": shape,
                "object_radius_m": float(raw["object_radius_m"]),
                "object_half_length_m": float(raw["object_half_length_m"]),
                "near_clipping_m": float(
                    raw.get("near_clipping_m", WRIST_CAMERA_NEAR_CLIPPING_M)
                ),
                "minimum_surface_margin_m": float(
                    raw.get("minimum_surface_margin_m", 0.01)
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "recording.wrist_camera_object_clearance 尺寸配置无效"
            ) from exc
        return config

    def _update_wrist_camera_object_clearance(
        self,
        *,
        tcp_pose_world: Any,
        object_pose_world: Any,
    ) -> None:
        """聚合整条 episode 中 wrist 近裁剪面与目标物体的最小间距。"""

        config = self._wrist_camera_object_clearance_config()
        if config is None:
            return
        previous = self._metadata.get("wrist_camera_object_clearance_report")
        previous = previous if isinstance(previous, dict) else {}
        sample_count = int(previous.get("sample_count", 0)) + 1
        unavailable_count = int(previous.get("unavailable_sample_count", 0))
        considered_count = int(previous.get("considered_sample_count", 0))
        violation_count = int(previous.get("violation_count", 0))
        worst_sample = previous.get("worst_sample")
        latest_sample = None
        if tcp_pose_world is None or object_pose_world is None:
            unavailable_count += 1
        else:
            latest_sample = _compute_wrist_camera_object_clearance_sample(
                tcp_pose_world=tcp_pose_world,
                object_pose_world=object_pose_world,
                object_radius_m=config["object_radius_m"],
                object_half_length_m=config["object_half_length_m"],
                near_clipping_m=config["near_clipping_m"],
                minimum_surface_margin_m=config["minimum_surface_margin_m"],
            )
            latest_sample.update(
                {
                    "step_index": self._step_calls,
                    "timestamp": (
                        float(self._step_calls) * float(self._runtime.step_dt)
                        if self._runtime is not None
                        else None
                    ),
                    "action_source": self._last_action.source,
                }
            )
            if latest_sample["potentially_visible"]:
                considered_count += 1
                if not latest_sample["verified"]:
                    violation_count += 1
                if (
                    not isinstance(worst_sample, dict)
                    or float(latest_sample["surface_clearance_m"])
                    < float(worst_sample.get("surface_clearance_m", math.inf))
                ):
                    worst_sample = dict(latest_sample)
        self._metadata["wrist_camera_object_clearance_report"] = {
            **config,
            "camera_extrinsics_source": (
                "hand_eye_calibration_with_visual_alignment_v3"
            ),
            "sample_count": sample_count,
            "unavailable_sample_count": unavailable_count,
            "considered_sample_count": considered_count,
            "violation_count": violation_count,
            "verified": bool(considered_count > 0 and violation_count == 0),
            "worst_sample": worst_sample,
            "latest_sample": latest_sample,
        }

    def _read_camera_images(self) -> dict[str, Any]:
        """只暴露当前渲染完成的 tensor；JPEG 编码由 5 Hz recorder 负责。"""

        if self._runtime is None:
            return {}
        images: dict[str, Any] = {}
        sensor_names = []
        if self._config.enable_front_camera:
            sensor_names.append(("front", "head_camera"))
        if self._config.enable_wrist_camera:
            sensor_names.append(("wrist", "arm_camera"))
        if self._config.enable_overview_camera:
            sensor_names.append(("overview", "overview_camera"))
        for camera_key, sensor_name in sensor_names:
            try:
                sensor = self._runtime.scene[sensor_name]
                rgb = sensor.data.output["rgb"]
            except (KeyError, TypeError, AttributeError):
                continue
            if rgb is None or getattr(rgb, "shape", (0,))[0] < 1:
                continue
            images[camera_key] = rgb[0, :, :, :3]
        metadata = getattr(self, "_metadata", None)
        if isinstance(metadata, dict):
            requested = [camera_key for camera_key, _sensor_name in sensor_names]
            metadata["camera_capture_report"] = {
                "requested_camera_keys": requested,
                "available_camera_keys": sorted(images),
                "missing_camera_keys": sorted(set(requested) - set(images)),
                "image_shapes": {
                    key: [int(value) for value in getattr(image, "shape", ())]
                    for key, image in images.items()
                },
            }
        return images

    def _resolve_checkpoint(self) -> Path:
        candidates = []
        if self._config.checkpoint is not None:
            candidates.append(self._config.checkpoint)
        candidates.extend(
            [
                self._project_root / "checkpoints/go2_x5/flat/model_8500.pt",
                Path("/home/light/workspace/arm_vla/checkpoints/go2_x5/flat/model_8500.pt"),
                Path("/home/light/workspace/DWA/flat/model_8500.pt"),
            ]
        )
        for candidate in candidates:
            path = Path(candidate).expanduser().resolve()
            if path.is_file():
                return path
        raise FileNotFoundError(f"Go2-X5 checkpoint does not exist: {candidates}")

    def _resolve_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        return path.resolve() if path.is_absolute() else (self._project_root / path).resolve()

    def _require_ready(self) -> None:
        if self._env is None or self._runtime is None or self._adapter is None:
            raise RuntimeError("Isaac Lab environment must be built first")
