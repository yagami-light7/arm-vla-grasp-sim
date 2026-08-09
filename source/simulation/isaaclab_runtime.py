"""由 pipeline 主循环驱动的 Isaac Lab 单环境 runtime。"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.interfaces import EpisodeSpec, RobotAction, SimulationState
from source.navigation.cmd_vel_to_policy import (
    CmdVelToPolicyAdapter,
    CmdVelToPolicyConfig,
    PolicyCommandWriteReport,
)
from source.navigation.isaac_depth_point_cloud import (
    DepthPointCloudConfig,
    camera_sensor_to_world_points,
)
from source.navigation.isaac_ros2_ogn_bridge import (
    IsaacRos2OgnBridge,
    IsaacRos2OgnBridgeConfig,
    OgnBsplineDiagnosticsSample,
    OgnGridMapObservationDiagnosticsSample,
    OgnPCTGoalSample,
    OgnStairExecutionFreezePublicationReport,
    enable_ros2_bridge_extension,
)

from .object_initialization import resolve_object_initialization_policy
from .dynamic_obstacles import (
    DynamicObstaclePlan,
    DynamicObstacleState,
    resolve_dynamic_obstacle_plan,
)
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
DYNAMIC_OBSTACLE_POINT_ASSOCIATION_TOLERANCE_M = 0.05
DYNAMIC_OBSTACLE_MOTION_SEPARATION_EPSILON_M = 1.0e-9
DYNAMIC_OBSTACLE_RELEVANCE_DISTANCE_M = 2.0
GO2_X5_DOUBLE_CYLINDER_RADIUS_M = 0.27
GO2_X5_DOUBLE_CYLINDER_OFFSET_M = 0.16
GO2_X5_ANY_YAW_CLEARANCE_RADIUS_M = (
    GO2_X5_DOUBLE_CYLINDER_RADIUS_M
    + GO2_X5_DOUBLE_CYLINDER_OFFSET_M
)
DYNAMIC_OBSTACLE_DETOUR_DEVIATION_MIN_M = 0.02
DYNAMIC_OBSTACLE_RECOVERY_MAX_DEVIATION_M = 0.02
DYNAMIC_OBSTACLE_RECOVERY_IMPROVEMENT_MIN_M = 0.01
ACTIVE_SENSING_PENDING_CONTROLLER_STATUS_LIMIT = 64
ACTIVE_SENSING_PENDING_POLICY_WRITE_LIMIT = 64
ACTIVE_SENSING_ATTEMPT_LIMIT = 32
# 普通 USD pinhole 仅作为 schema 不可用时的近似 fallback；精确渲染由 OpenCV schema 决定。
D436_CAMERA_FALLBACK_HORIZONTAL_APERTURE_MM = 30.040158257372415
D436_CAMERA_FALLBACK_VERTICAL_APERTURE_MM = 22.530118693029312
WRIST_CAMERA_NEAR_CLIPPING_M = 0.03
WRIST_CAMERA_TCP_OFFSET_LINK6_XYZ_M = (0.15757, 0.0, 0.0)

_STAIR_EXECUTION_RESUMED_PHASES = frozenset(
    {"released_stable", "resume"}
)


def _stair_execution_frozen_from_action(
    action: RobotAction,
    *,
    emergency_stop_latched: bool = False,
) -> tuple[bool, str | None, str]:
    """从单拍 action provenance 推导 SCAN planner 是否必须暂停。

    ``released_stable``/``resume`` 是唯一显式解冻阶段。其余带楼梯冻结
    provenance 的阶段（含 release 稳定窗口与 emergency hold）全部返回
    ``true``；字段缺失或新阶段未知时保持冻结，避免新楼梯动作静默放行。
    普通 action 没有楼梯 provenance，发布初始/持续 ``false``。
    """

    metadata = action.metadata
    raw_phase = metadata.get("navigation_scan_stair_freeze_phase")
    phase = (
        raw_phase.strip()
        if isinstance(raw_phase, str) and raw_phase.strip()
        else None
    )
    if metadata.get("navigation_stair_emergency_hold") is True:
        return True, phase, "stair_emergency_hold"
    if emergency_stop_latched:
        return True, phase, "navigation_emergency_stop_latched"
    if phase in _STAIR_EXECUTION_RESUMED_PHASES:
        return False, phase, "stair_execution_resumed"

    explicit_stair_freeze = (
        metadata.get("navigation_scan_stair_freeze") is True
    )
    raw_inhibit_reason = metadata.get("navigation_cmd_vel_inhibit_reason")
    stair_inhibit = (
        isinstance(raw_inhibit_reason, str)
        and raw_inhibit_reason.strip().startswith("scan_stair_")
    )
    if explicit_stair_freeze or phase is not None or stair_inhibit:
        return True, phase, (
            f"stair_phase:{phase}"
            if phase is not None
            else "stair_metadata_fail_closed"
        )
    return False, None, "ordinary_action"


def _object_pose_reset_gate(
    *,
    object_pose_required: bool,
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """判定 reset 后是否必须通过物体位姿一致性门禁。

    纯导航任务可以没有目标物体和初始物体位姿；此时诊断缺失是任务合同，
    不能被解释为物体在 Fabric/render 同步中发生漂移。
    """

    if not object_pose_required:
        return {
            "required": False,
            "verified": False,
            "skipped": True,
            "reason": "navigation_task_has_no_object_initial_pose",
        }
    verified = (
        diagnostic.get("available") is True
        and diagnostic.get("within_tolerance") is True
    )
    return {
        "required": True,
        "verified": verified,
        "skipped": False,
        "reason": None if verified else "object_pose_reset_diagnostic_failed",
    }


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
    auto_manage_viewport_camera: bool = True
    hide_navigation_collision_visual: bool = True
    scene_light_mode: str = "camera"
    camera_light_intensity: float = 3500.0
    camera_light_radius: float = 2.0
    camera_light_name: str = "camera_light"
    # 场景可覆盖 IsaacLab RenderCfg；NuRec 在 Isaac Sim 5.1 中不能使用默认 DLSS。
    render_antialiasing_mode: str | None = None
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
    # Headless dataset collection can render on the dataset sampling grid.
    # GUI and composite video keep this at one control step.
    camera_render_interval_control_steps: int = 1
    # 两项同时为 None 时保持旧 pipeline 完全不变；启用时必须同时提供，禁止
    # 只发布 Odometry 而静默缺失 SCAN 所需的在线点云。
    ros2_ogn_bridge_config: IsaacRos2OgnBridgeConfig | None = None
    depth_point_cloud_config: DepthPointCloudConfig | None = None
    # 非 None 时由 /cmd_vel 独占写入 locomotion policy；pipeline 传入的
    # base_velocity 会被明确旁路，机械臂与夹爪 action 仍沿原主循环执行。
    cmd_vel_to_policy_config: CmdVelToPolicyConfig | None = None
    # Multi-episode stage reuse requires randomized support colliders to expose a
    # live PhysX pose.  They remain immovable kinematic bodies during an episode.
    enable_relocatable_episode_supports: bool = False
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
    show_velocity_command_debug: bool = False


def _item(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _validated_effective_goal_provenance(
    raw_provenance: Any,
    *,
    position_base_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    """校验 PLY 地面、live 高度和最终 base 目标属于同一审计合同。"""

    if not isinstance(raw_provenance, dict):
        raise ValueError("生产 PCT goal 缺少 effective_goal_provenance。")
    provenance = copy.deepcopy(raw_provenance)
    if (
        provenance.get("schema") != "pct_effective_goal_height_v1"
        or provenance.get("height_semantics")
        != "collision_ground_plus_configured_body_height"
        or provenance.get("formula")
        != "effective_base_z=collision_ground_z+configured_body_height_m"
        or provenance.get("raw_task_z_used_as_height_evidence") is not False
    ):
        raise ValueError("effective_goal_provenance 高度合同非法。")
    projection = provenance.get("projection")
    calibration = provenance.get("calibration")
    if not isinstance(projection, dict) or not isinstance(calibration, dict):
        raise ValueError("effective_goal_provenance 缺少投影或 live 校准证据。")
    projected_base = projection.get("projected_base_sim_xyz")
    if not isinstance(projected_base, (list, tuple)) or len(projected_base) != 3:
        raise ValueError("effective_goal_provenance 缺少 projected base xyz。")
    try:
        normalized_projected_base = tuple(float(value) for value in projected_base)
        configured_height = float(provenance["configured_body_height_m"])
        projected_height = float(projection["configured_body_height_hint_m"])
        calibrated_hint = float(calibration["configured_body_height_hint_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("effective_goal_provenance 数值字段非法。") from exc
    if not all(
        math.isfinite(value)
        for value in (
            *normalized_projected_base,
            configured_height,
            projected_height,
            calibrated_hint,
        )
    ):
        raise ValueError("effective_goal_provenance 不能包含 NaN 或 Inf。")
    if any(
        abs(left - right) > 1.0e-9
        for left, right in zip(
            normalized_projected_base,
            position_base_xyz,
            strict=True,
        )
    ):
        raise ValueError("PCT goal xyz 与 effective height provenance 不一致。")
    if (
        abs(configured_height - projected_height) > 1.0e-12
        or abs(configured_height - calibrated_hint) > 1.0e-12
    ):
        raise ValueError("effective_goal_provenance 的 body height 不一致。")
    projection_sha = projection.get("collision_ply_sha256")
    calibration_sha = calibration.get("collision_ply_sha256")
    if (
        not isinstance(projection_sha, str)
        or len(projection_sha) != 64
        or projection_sha != calibration_sha
    ):
        raise ValueError("effective_goal_provenance 的 collision PLY 哈希不一致。")
    return provenance


def _coerce_xyzyaw(value: Any) -> tuple[float, float, float, float] | None:
    """把 action metadata 中的 root lock 目标解析为 xyzyaw 四元组。"""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        parsed = (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(item) for item in parsed) else None


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


def _episode_reset_pose_configuration(
    episode_spec: EpisodeSpec,
    *,
    default_root_pos: tuple[float, float, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact IsaacLab reset event parameters for one episode."""

    start_z = (
        default_root_pos[2]
        if episode_spec.start.z is None
        else float(episode_spec.start.z)
    )
    start_offset = (
        float(episode_spec.start.x) - default_root_pos[0],
        float(episode_spec.start.y) - default_root_pos[1],
        start_z - default_root_pos[2],
    )
    params = {
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
    report = {
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
    return params, report


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
        self._dynamic_obstacle_plan = DynamicObstaclePlan()
        self._episode_support_bodies: dict[str, dict[str, Any]] = {}
        self._settled_object_pose: tuple[float, ...] | None = None
        self._episode_spec: EpisodeSpec | None = None
        self._default_robot_root_pos: tuple[float, float, float] | None = None
        self._stage_reuse_fingerprint: dict[str, Any] | None = None
        self._stage_build_count = 0
        self._stage_reuse_count = 0
        self._step_calls = 0
        # ROS direct 模式使用跨 episode 连续的真实 physics-step 计数。现有
        # _step_calls 与 ManagerBasedEnv 计数会在 reset 时清零，不能用作 /clock。
        self._ros2_physics_step_count = 0
        # 动态障碍按 episode-local physics time 运动，而 /clock 在 stage reuse
        # 时连续递增。两者只能通过本 episode 的 ROS 时间偏移相互转换，禁止
        # 直接把跨 episode Header stamp 传给 DynamicObstaclePlan.state_at()。
        self._navigation_episode_ros_time_offset_s = 0.0
        self._ros2_ogn_bridge: IsaacRos2OgnBridge | None = None
        self._cmd_vel_to_policy: CmdVelToPolicyAdapter | None = None
        self._cmd_vel_owner_id = "scan_cmd_vel"
        self._navigation_emergency_stop_reason: str | None = None
        self._scan_policy_write_sequence = 0
        self._last_scan_cmd_vel_source_sequence: int | None = None
        self._last_scan_cmd_vel_source_receipt_timestamp: float | None = None
        self._scan_cmd_vel_sample_received_this_tick = False
        self._scan_cmd_vel_sample_drained_this_tick = False
        self._last_scan_cmd_vel_drain_sequence: int | None = None
        self._last_scan_cmd_vel_drain_receipt_timestamp: float | None = None
        self._last_pct_goal_request_identity: tuple[object, ...] | None = None
        self._last_pct_goal_sample: OgnPCTGoalSample | None = None
        self._ros2_odometry_publish_count = 0
        self._ros2_point_cloud_publish_count = 0
        self._closed = False
        self._performance_profiler: Any | None = None
        self._cached_camera_step: int | None = None
        self._cached_camera_images: dict[str, Any] = {}
        self._camera_render_generation = 0
        self._last_camera_render_step: int | None = None
        self._last_camera_render_reason: str | None = None
        self._action_prepared = False
        self._environment_terminated = False
        self._last_action = RobotAction.idle()
        self._pending_stair_probe_low_level_telemetry: dict[str, Any] | None = (
            None
        )
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
            "used_object_initialization_pose_stabilization": False,
            "object_initialization_pose_stabilization_apply_count": 0,
            "last_object_initialization_pose_stabilization_report": None,
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
            "stage_build_count": 0,
            "stage_reuse_count": 0,
            "scan_reference_path_last_report": None,
            "scan_controller_status_last_report": None,
            "scan_controller_status_lifecycle_report": (
                self._new_scan_controller_status_lifecycle_report()
            ),
            "grid_map_observation_diagnostics_last_report": None,
            "grid_map_observation_lifecycle_report": (
                self._new_grid_map_observation_lifecycle_report()
            ),
            "bspline_diagnostics_last_report": None,
            "bspline_diagnostics_lifecycle_report": (
                self._new_bspline_diagnostics_lifecycle_report()
            ),
            "active_sensing_lifecycle_report": (
                self._new_active_sensing_lifecycle_report()
            ),
            "dynamic_navigation_evidence_report": (
                self._new_dynamic_navigation_evidence_report(
                    DynamicObstaclePlan()
                )
            ),
        }

    def set_performance_profiler(self, profiler: Any | None) -> None:
        """Attach the episode profiler without coupling runtime interfaces to diagnostics."""

        self._performance_profiler = profiler

    @property
    def is_built(self) -> bool:
        """Whether the IsaacLab environment and stage are ready for reset."""

        return self._env is not None and self._runtime is not None

    def _hard_reset_stage_reuse_physics(
        self,
        episode_spec: EpisodeSpec,
    ) -> dict[str, Any]:
        """Recreate PhysX views while preserving the already-open USD stage.

        ``ManagerBasedEnv.reset`` only rewrites tensor state for one environment.
        It does not clear the articulation/contact solver warm-start accumulated by
        the previous episode.  In a reused stage that stale state can fold the Go2
        legs on the first physics step even though the visible reset tensors are
        correct.  A hard ``SimulationContext.reset`` stops and restarts the
        timeline, which recreates the PhysX simulation view without reopening or
        rebuilding the USD stage.

        Timeline STOP invalidates both IsaacLab asset handles and the standalone
        ``SingleRigidPrim`` readers used here.  IsaacLab assets reinitialize from
        their PLAY callbacks; the standalone object/support readers and calibrated
        camera matrices must be rebound explicitly before the episode reset.
        """

        started_at = time.perf_counter()
        control_step_before = int(self._step_calls)
        manager_sim_step_before = int(self._runtime._sim_step_counter)

        self._runtime.sim.reset(soft=False)
        # Match ManagerBasedEnv's initial construction sequence so freshly
        # reinitialized assets and sensors have populated data buffers.
        self._runtime.scene.update(dt=float(self._runtime.physics_dt))

        self._initialize_object_reader(episode_spec)
        self._initialize_episode_support_readers(episode_spec)
        camera_intrinsics_report = self._apply_d436_runtime_intrinsics(
            self._runtime
        )
        self._metadata["camera_runtime_intrinsics_report"] = (
            camera_intrinsics_report
        )
        self._runtime.sim.forward()
        ros2_bridge = getattr(self, "_ros2_ogn_bridge", None)
        if ros2_bridge is not None:
            # 同一 USD stage 的 STOP/PLAY 不重建 OGN graph；仅重绑可能失效的
            # AttributeValueHelper，并保留跨 episode 的最后发布时间。
            ros2_bridge.refresh_after_timeline_reset()

        return {
            "applied": True,
            "mode": "simulation_context_hard_reset",
            "soft": False,
            "usd_stage_reopened": False,
            "usd_stage_rebuilt": False,
            "physx_views_recreated": True,
            "standalone_tensor_readers_reinitialized": True,
            "camera_intrinsics_reapplied": camera_intrinsics_report,
            "ros2_ogn_graph_reused": ros2_bridge is not None,
            "control_step_before_after": [
                control_step_before,
                int(self._step_calls),
            ],
            "manager_sim_step_before_after": [
                manager_sim_step_before,
                int(self._runtime._sim_step_counter),
            ],
            "physics_time_advanced": False,
            "wall_seconds": time.perf_counter() - started_at,
        }

    def prepare_episode(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        """Build once, then reconfigure episode-level poses on the live stage."""

        if self._closed:
            raise RuntimeError("simulation runtime is closed")
        if not self.is_built:
            self.build(episode_spec)
            report = {
                "stage_reused": False,
                "stage_build_count": self._stage_build_count,
                "stage_reuse_count": self._stage_reuse_count,
                "episode_id": int(episode_spec.episode_id),
                "reason": "initial_stage_build",
            }
            self._metadata["stage_reuse_report"] = report
            return report

        started_at = time.perf_counter()
        fingerprint = self._episode_stage_fingerprint(episode_spec)
        if fingerprint != self._stage_reuse_fingerprint:
            raise RuntimeError(
                "episode is incompatible with the existing Isaac stage: "
                f"built={self._stage_reuse_fingerprint} requested={fingerprint}"
            )
        if self._default_robot_root_pos is None:
            raise RuntimeError("default robot root pose is unavailable for stage reuse")

        import omni.usd

        from source.simulation.task_scene_pose import apply_task_receptacle_pose

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage is unavailable during episode reuse")

        previous_episode_id = (
            None if self._episode_spec is None else int(self._episode_spec.episode_id)
        )
        self._episode_spec = episode_spec
        self._dynamic_obstacle_plan = resolve_dynamic_obstacle_plan(
            episode_spec.raw_task
        )
        self._metadata["dynamic_obstacle_configuration_report"] = (
            self._dynamic_obstacle_configuration_metadata(
                self._dynamic_obstacle_plan
            )
        )
        receptacle_pose_report = apply_task_receptacle_pose(
            stage,
            episode_spec.raw_task,
        )
        self._metadata["task_receptacle_pose_report"] = receptacle_pose_report
        # Author every episode-local USD pose before restarting PhysX so the new
        # simulation view is born from the requested scene configuration rather
        # than from the previous episode's terminal contact state.
        self._metadata["object_pose_setup_report"] = self._apply_object_pose(
            episode_spec
        )
        physics_context_reset_report = self._hard_reset_stage_reuse_physics(
            episode_spec
        )
        self._metadata["stage_reuse_physics_context_reset_report"] = (
            physics_context_reset_report
        )
        support_pose_write_report = self._write_episode_support_physics_poses(
            episode_spec,
            reason="stage_reuse_prepare_episode",
        )
        # Push the kinematic support tensor transforms into Fabric.  Unlike a
        # rewritten USD-static collider, this does not require a Kit update or a
        # physics-scene rebuild.
        self._runtime.sim.forward()
        support_pose_diagnostic = self._episode_support_pose_diagnostic(
            label="after_stage_reuse_prepare_forward",
        )
        if support_pose_diagnostic.get("verified") is not True:
            raise RuntimeError(
                "episode support pose changed during stage-reuse forward sync: "
                f"{support_pose_diagnostic}"
            )
        self._metadata["task_receptacle_support_runtime_stage_report"] = (
            inspect_task_receptacle_support_stage(
                stage,
                episode_spec.raw_task,
                source="isaaclab_reused_runtime_stage",
            )
        )
        self._metadata["object_visibility_report"] = self._show_only_task_object(
            stage,
            episode_spec,
        )
        self._metadata["object_collision_visual_hide_report"] = (
            self._hide_object_collision_visual(stage)
        )

        reset_params, reset_report = _episode_reset_pose_configuration(
            episode_spec,
            default_root_pos=self._default_robot_root_pos,
        )
        reset_term_cfg = copy.deepcopy(
            self._runtime.event_manager.get_term_cfg("randomize_reset_base")
        )
        reset_term_cfg.params = reset_params
        self._runtime.event_manager.set_term_cfg(
            "randomize_reset_base",
            reset_term_cfg,
        )
        self._metadata["episode_reset_pose_request"] = reset_report

        static_sync_report = {
            "applied": True,
            "reason": "kinematic_episode_support_tensor_sync",
            "support_pose_write_report": support_pose_write_report,
            "support_pose_diagnostic": support_pose_diagnostic,
            "render_required": False,
            "physics_time_advanced": False,
        }
        self._step_calls = 0
        self._runtime._sim_step_counter = 0
        self._cached_camera_step = None
        self._cached_camera_images = {}
        self._clear_previous_episode_metadata()
        self._stage_reuse_count += 1
        report = {
            "stage_reused": True,
            "stage_build_count": self._stage_build_count,
            "stage_reuse_count": self._stage_reuse_count,
            "previous_episode_id": previous_episode_id,
            "episode_id": int(episode_spec.episode_id),
            "scene_pose_reapplied": bool(
                receptacle_pose_report.get("any_scene_pose_configured")
            ),
            "robot_reset_event_updated": True,
            "physics_static_scene_sync": static_sync_report,
            "physics_context_reset": physics_context_reset_report,
            "physics_time_advanced": False,
            "prepare_wall_seconds": time.perf_counter() - started_at,
        }
        self._metadata.update(
            {
                "stage_build_count": self._stage_build_count,
                "stage_reuse_count": self._stage_reuse_count,
                "stage_reuse_report": report,
            }
        )
        return report

    def _episode_stage_fingerprint(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        scene_runtime = resolve_scene_runtime_settings(
            episode_spec.raw_task,
            default_collision_prim_path=self._config.terrain_prim_path,
            default_visual_prim_path=self._config.visual_prim_path,
            default_collision_floor_proxy_profile=(
                self._config.collision_floor_proxy_profile
            ),
        )
        support_settings = self._episode_support_pose_settings(episode_spec)
        dynamic_obstacle_plan = resolve_dynamic_obstacle_plan(
            episode_spec.raw_task
        )
        return {
            "scene_usd": str(self._resolve_path(episode_spec.scene_usd)),
            "nav_map": str(self._resolve_path(episode_spec.nav_map)),
            "object_prim_path": episode_spec.object_prim_path,
            "collision_prim_path": str(scene_runtime["collision_prim_path"]),
            "visual_prim_path": str(scene_runtime["visual_prim_path"]),
            "relocatable_episode_supports": bool(
                self._config.enable_relocatable_episode_supports
            ),
            "episode_support_prim_paths": {
                role: (
                    str(settings["prim_path"])
                    if settings.get("configured") is True
                    else None
                )
                for role, settings in support_settings.items()
            },
            "dynamic_obstacle_topology": (
                dynamic_obstacle_plan.topology_fingerprint()
            ),
        }

    def _clear_previous_episode_metadata(self) -> None:
        """Drop dynamic reports that must never leak into the next summary."""

        for key in (
            "last_current_state_curobo_pick_export",
            "last_current_state_curobo_place_export",
            "last_mesh_truth_pick_target_report",
            "last_mesh_truth_place_target_report",
            "object_settle_final_report",
            "object_settle_begin_report",
            "object_pose_debug_after_reset",
            "object_pose_debug_after_reset_render",
            "episode_support_reset_pose_write_report",
            "episode_support_pose_after_reset_forward",
            "episode_support_pose_after_reset_render",
            "camera_capture_report",
            "wrist_camera_object_clearance_report",
            "stair_probe_low_level_telemetry",
            "dynamic_obstacle_runtime_report",
            "dynamic_obstacle_pose_write_count",
            "dynamic_obstacle_lifecycle_report",
            "dynamic_obstacle_raw_cloud_last_report",
            "dynamic_obstacle_raw_cloud_lifecycle_report",
            "scan_controller_status_lifecycle_report",
            "grid_map_observation_diagnostics_last_report",
            "grid_map_observation_lifecycle_report",
            "bspline_diagnostics_last_report",
            "bspline_diagnostics_lifecycle_report",
            "active_sensing_lifecycle_report",
            "dynamic_navigation_evidence_report",
        ):
            self._metadata.pop(key, None)

    def build(self, episode_spec: EpisodeSpec) -> None:
        if self._closed:
            raise RuntimeError("simulation runtime is closed")
        if self._env is not None:
            raise RuntimeError("Isaac Lab environment has already been built")
        if self._config.camera_render_interval_control_steps < 1:
            raise ValueError("camera_render_interval_control_steps must be positive")
        self._validate_navigation_ros2_config()
        self._episode_spec = episode_spec
        self._build_environment_with_navigation_ros2(episode_spec)
        self._stage_build_count += 1
        self._stage_reuse_fingerprint = self._episode_stage_fingerprint(episode_spec)
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
                "camera_render_interval_control_steps": int(
                    self._effective_camera_render_interval_control_steps()
                ),
                "camera_capture_interval_control_steps": int(
                    self._config.camera_render_interval_control_steps
                ),
                "camera_render_hz": 1.0
                / (
                    float(self._runtime.step_dt)
                    * float(
                        self._effective_camera_render_interval_control_steps()
                    )
                ),
                "checkpoint": str(self._resolve_checkpoint()),
                "stage_build_count": self._stage_build_count,
                "stage_reuse_count": self._stage_reuse_count,
            }
        )

    def _validate_navigation_ros2_config(self) -> None:
        """校验 Isaac runtime 的 ROS 2 与深度点云配置必须成对且同坐标。"""

        bridge_config = self._config.ros2_ogn_bridge_config
        cloud_config = self._config.depth_point_cloud_config
        command_config = self._config.cmd_vel_to_policy_config
        if (bridge_config is None) != (cloud_config is None):
            raise ValueError(
                "ros2_ogn_bridge_config 与 depth_point_cloud_config 必须同时配置。"
            )
        if bridge_config is None or cloud_config is None:
            if command_config is not None:
                raise ValueError(
                    "cmd_vel_to_policy_config 需要同时启用 ROS 2 OGN bridge。"
                )
            return
        if bridge_config.enable_command_subscription != (
            command_config is not None
        ):
            raise ValueError(
                "OGN cmd_vel 订阅与 cmd_vel_to_policy_config 必须同时启用或关闭。"
            )
        if bridge_config.enable_goal_reached_subscription != (
            command_config is not None
        ):
            raise ValueError(
                "OGN goal_reached 订阅与 cmd_vel_to_policy_config "
                "必须同时启用或关闭。"
            )
        if bridge_config.enable_controller_status_subscription != (
            command_config is not None
        ):
            raise ValueError(
                "OGN controller_status 订阅与 cmd_vel_to_policy_config "
                "必须同时启用或关闭。"
            )
        if command_config is not None and (
            not bridge_config.enable_grid_map_diagnostics_subscription
            or not bridge_config.enable_bspline_diagnostics_subscription
        ):
            raise ValueError(
                "IsaacLab ROS 2 导航主链必须同时订阅 GridMap 与 B-spline "
                "typed diagnostics。"
            )
        if not bridge_config.enable_reference_path_subscription:
            raise ValueError(
                "IsaacLab ROS 2 导航 bridge 启用时必须订阅参考 Path。"
            )
        if (
            command_config is not None
            and not bridge_config.enable_pct_goal_publisher
        ):
            raise ValueError(
                "IsaacLab ROS 2 导航 bridge 启用时必须提供 /pct/goal publisher。"
            )
        if (
            command_config is not None
            and not bridge_config.enable_stair_execution_frozen_publisher
        ):
            raise ValueError(
                "IsaacLab ROS 2 导航主链必须提供楼梯执行冻结 publisher。"
            )
        if self._config.hide_navigation_collision_visual:
            raise ValueError(
                "RTX 导航深度必须显示 navigation collision mesh；"
                "请使用 navigation_visual_mode=collision。"
            )
        if bridge_config.odometry_source != "direct":
            raise ValueError("IsaacLab runtime 接入只支持 direct Odometry。")
        if (
            bridge_config.odom_frame_id != "world"
            or bridge_config.point_cloud_frame_id != "world"
        ):
            raise ValueError(
                "root tensor 与深度点云均为 world 坐标，ROS 2 frame 必须配置为 world。"
            )
        if cloud_config.sensor_name != "head_camera":
            raise ValueError("第一阶段在线导航点云只允许使用前视 head_camera。")
        if cloud_config.depth_key != "distance_to_image_plane":
            raise ValueError(
                "前视导航点云必须使用 distance_to_image_plane 深度。"
            )
        if cloud_config.environment_index != 0:
            raise ValueError("当前导航 runtime 是单环境，只允许 environment_index=0。")

    def _prepare_navigation_ros2_extension(self) -> dict[str, object] | None:
        """在场景与 RTX 相机创建前启用 ROS 2 bridge extension。"""

        if self._config.ros2_ogn_bridge_config is None:
            return None
        return enable_ros2_bridge_extension()

    def _build_environment_with_navigation_ros2(
        self,
        episode_spec: EpisodeSpec,
    ) -> None:
        """按 extension、环境、OGN 图的固定顺序建立导航运行时。"""

        # bridge 会按需加载 isaacsim.sensors.rtx。若等相机 render product 与
        # timeline 已建立后再加载，复杂场景可能在 extension startup 中访问到
        # 尚未稳定的 render prim，并以 ``Used null prim`` 失败。
        extension_report = self._prepare_navigation_ros2_extension()
        self._build_environment(episode_spec)
        self._initialize_navigation_ros2_bridge(
            extension_report=extension_report,
        )

    def _initialize_navigation_ros2_bridge(
        self,
        *,
        extension_report: dict[str, object] | None,
    ) -> None:
        """在 Isaac 环境与 stage 就绪后创建 ROS 2 OGN 发布图。"""

        bridge_config = self._config.ros2_ogn_bridge_config
        if bridge_config is None:
            self._metadata["navigation_ros2_bridge_report"] = {
                "enabled": False,
                "reason": "configuration_disabled",
            }
            return
        if extension_report is None:
            raise RuntimeError(
                "ROS 2 bridge extension 必须在 Isaac 环境创建前启用。"
            )
        bridge = IsaacRos2OgnBridge(bridge_config)
        bridge.setup()
        self._ros2_ogn_bridge = bridge
        command_config = self._config.cmd_vel_to_policy_config
        if command_config is not None:
            command_gate = CmdVelToPolicyAdapter(
                self._adapter,
                command_config,
                ownership_resource=("go2_x5_policy_command", 0),
            )
            command_gate.claim(
                self._cmd_vel_owner_id,
                self._navigation_ros2_timestamp(),
            )
            self._cmd_vel_to_policy = command_gate
        cloud_config = self._config.depth_point_cloud_config
        self._metadata["navigation_ros2_bridge_report"] = {
            "enabled": True,
            "extension": extension_report,
            "graph_path": bridge_config.graph_path,
            "odometry_topic": bridge_config.odometry_topic,
            "point_cloud_topic": bridge_config.point_cloud_topic,
            "clock_topic": bridge_config.clock_topic,
            "command_topic": (
                bridge_config.command_topic
                if bridge_config.enable_command_subscription
                else None
            ),
            "goal_reached_topic": (
                bridge_config.goal_reached_topic
                if bridge_config.enable_goal_reached_subscription
                else None
            ),
            "controller_status_topic": (
                bridge_config.controller_status_topic
                if bridge_config.enable_controller_status_subscription
                else None
            ),
            "grid_map_diagnostics_topic": (
                bridge_config.grid_map_diagnostics_topic
                if bridge_config.enable_grid_map_diagnostics_subscription
                else None
            ),
            "bspline_diagnostics_topic": (
                bridge_config.bspline_diagnostics_topic
                if bridge_config.enable_bspline_diagnostics_subscription
                else None
            ),
            "planning_diagnostics_qos_profile": (
                bridge_config.planning_diagnostics_qos_profile
                if (
                    bridge_config.enable_grid_map_diagnostics_subscription
                    or bridge_config.enable_bspline_diagnostics_subscription
                )
                else None
            ),
            "navigation_status_topic": (
                bridge_config.navigation_status_topic
                if bridge_config.enable_command_subscription
                else None
            ),
            "navigation_status_qos_profile": (
                bridge_config.navigation_status_qos_profile
                if bridge_config.enable_command_subscription
                else None
            ),
            "stair_execution_frozen_topic": (
                bridge_config.stair_execution_frozen_topic
                if bridge_config.enable_stair_execution_frozen_publisher
                else None
            ),
            "reference_path_topic": bridge_config.reference_path_topic,
            "pct_goal_topic": bridge_config.pct_goal_topic,
            "command_authority_enabled": command_config is not None,
            "navigation_status_gate_required": (
                None
                if command_config is None
                else bool(command_config.require_navigation_status)
            ),
            "navigation_status_timeout_s": (
                None
                if command_config is None
                else float(command_config.navigation_status_timeout_s)
            ),
            "goal_lifecycle_enabled": (
                bridge_config.enable_goal_reached_subscription
            ),
            "controller_status_subscription_enabled": (
                bridge_config.enable_controller_status_subscription
            ),
            "grid_map_diagnostics_subscription_enabled": (
                bridge_config.enable_grid_map_diagnostics_subscription
            ),
            "bspline_diagnostics_subscription_enabled": (
                bridge_config.enable_bspline_diagnostics_subscription
            ),
            "stair_execution_frozen_publisher_enabled": (
                bridge_config.enable_stair_execution_frozen_publisher
            ),
            "reference_path_subscription_enabled": (
                bridge_config.enable_reference_path_subscription
            ),
            "pct_goal_publisher_enabled": (
                bridge_config.enable_pct_goal_publisher
            ),
            "odom_frame_id": bridge_config.odom_frame_id,
            "base_frame_id": bridge_config.base_frame_id,
            "point_cloud_frame_id": bridge_config.point_cloud_frame_id,
            "depth_sensor_name": (
                None if cloud_config is None else cloud_config.sensor_name
            ),
            "point_cloud_publish_interval_control_steps": (
                None
                if cloud_config is None
                else cloud_config.publish_interval_control_steps
            ),
            "point_cloud_pixel_stride": (
                None if cloud_config is None else cloud_config.pixel_stride
            ),
            "point_cloud_max_points": (
                None if cloud_config is None else cloud_config.max_points
            ),
            "point_cloud_minimum_valid_points": (
                None
                if cloud_config is None
                else cloud_config.minimum_valid_points
            ),
            "continuous_time_source": "successful_physics_steps_x_physics_dt",
        }

    def reset(self, episode_spec: EpisodeSpec, *, seed: int) -> None:
        self._require_ready()
        self._episode_spec = episode_spec
        requested_dynamic_obstacle_plan = resolve_dynamic_obstacle_plan(
            episode_spec.raw_task
        )
        configured_dynamic_obstacle_plan = getattr(
            self,
            "_dynamic_obstacle_plan",
            DynamicObstaclePlan(),
        )
        if (
            requested_dynamic_obstacle_plan.topology_fingerprint()
            != configured_dynamic_obstacle_plan.topology_fingerprint()
        ):
            raise RuntimeError(
                "episode dynamic obstacle topology differs from the built Isaac stage"
            )
        self._dynamic_obstacle_plan = requested_dynamic_obstacle_plan
        self._metadata["dynamic_obstacle_configuration_report"] = (
            self._dynamic_obstacle_configuration_metadata(
                self._dynamic_obstacle_plan
            )
        )
        # Episode-local time and the internal render grid both restart at zero.
        # This preserves camera_capture_step == state.step_index after stage reuse.
        self._step_calls = 0
        self._pending_stair_probe_low_level_telemetry = None
        self._scan_policy_write_sequence = 0
        self._last_scan_cmd_vel_source_sequence = None
        self._last_scan_cmd_vel_source_receipt_timestamp = None
        self._scan_cmd_vel_sample_received_this_tick = False
        self._scan_cmd_vel_sample_drained_this_tick = False
        self._last_scan_cmd_vel_drain_sequence = None
        self._last_scan_cmd_vel_drain_receipt_timestamp = None
        self._last_pct_goal_request_identity = None
        self._last_pct_goal_sample = None
        self._runtime._sim_step_counter = 0
        self._navigation_episode_ros_time_offset_s = (
            self._navigation_ros2_timestamp()
        )
        self._settled_object_pose = None
        self._cached_camera_step = None
        self._cached_camera_images = {}
        self._last_camera_render_step = None
        self._last_camera_render_reason = None
        reset_policy_warmup = getattr(self._adapter, "reset_policy_warmup", None)
        if callable(reset_policy_warmup):
            reset_policy_warmup()
        # Clear every episode-local adapter override before ManagerBasedEnv writes
        # reset state back to PhysX.  This prevents a previous terminal carry/place
        # lock from being observed during the new episode's reset transaction.
        command_gate = getattr(self, "_cmd_vel_to_policy", None)
        if command_gate is None:
            self._adapter.apply_base_command(0.0, 0.0, 0.0)
        else:
            command_gate.reset(
                owner_id=self._cmd_vel_owner_id,
                now=self._navigation_ros2_timestamp(),
            )
        # reset() 已先由唯一命令门精确写零并清除旧输入；此后才允许新 episode
        # 解除失败锁存，且仍须重新收到新鲜 cmd_vel、Odometry 与点云。
        self._navigation_emergency_stop_reason = None
        self._adapter.set_arm_joint_target(None)
        self._adapter.set_direct_arm_action_override(False)
        self._adapter.set_gripper_joint_target(None)
        self._adapter.set_base_pose_lock(False)
        if hasattr(self._adapter, "set_support_joint_lock"):
            self._adapter.set_support_joint_lock(False)
        if hasattr(self._adapter, "set_navigation_joint_pose_lock"):
            self._adapter.set_navigation_joint_pose_lock(False)
        observations, _extras = self._runtime.reset(seed=seed)
        self._adapter.update_observations(self._to_tensor_dict(observations))
        self._environment_terminated = False
        self._action_prepared = False
        self._last_action = RobotAction.idle(source="episode_reset")
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
                "used_object_initialization_pose_stabilization": False,
                "object_initialization_pose_stabilization_apply_count": 0,
                "last_object_initialization_pose_stabilization_report": None,
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
                "scan_cmd_vel_last_write_report": None,
                "navigation_policy_gate_lifecycle_report": (
                    self._new_navigation_policy_gate_lifecycle_report()
                ),
                "scan_goal_reached_last_sample": None,
                "scan_reference_path_last_report": None,
                "scan_controller_status_last_report": None,
                "scan_controller_status_lifecycle_report": (
                    self._new_scan_controller_status_lifecycle_report()
                ),
                "grid_map_observation_diagnostics_last_report": None,
                "grid_map_observation_lifecycle_report": (
                    self._new_grid_map_observation_lifecycle_report(
                        ros_time_offset_s=(
                            self._navigation_episode_ros_time_offset_s
                        )
                    )
                ),
                "bspline_diagnostics_last_report": None,
                "bspline_diagnostics_lifecycle_report": (
                    self._new_bspline_diagnostics_lifecycle_report(
                        ros_time_offset_s=(
                            self._navigation_episode_ros_time_offset_s
                        )
                    )
                ),
                "active_sensing_lifecycle_report": (
                    self._new_active_sensing_lifecycle_report()
                ),
                "dynamic_navigation_evidence_report": (
                    self._new_dynamic_navigation_evidence_report(
                        self._dynamic_obstacle_plan,
                        ros_time_offset_s=(
                            self._navigation_episode_ros_time_offset_s
                        ),
                    )
                ),
                "scan_pct_goal_last_report": None,
                "navigation_stair_execution_frozen_last_publish_report": None,
                "navigation_joint_pose_lock_active": False,
                "navigation_joint_pose_lock_apply_count": 0,
                "last_navigation_joint_pose_lock_report": None,
                "dynamic_obstacle_pose_write_count": 0,
                "dynamic_obstacle_raw_cloud_last_report": None,
                # reset 事件写入初始位姿，不属于导航执行中的 teleport。
                "reset_pose_source": "isaaclab_reset_event",
                "object_initialization_policy": resolve_object_initialization_policy(
                    episode_spec.raw_task
                ),
            }
        )
        self._metadata["dynamic_obstacle_lifecycle_report"] = (
            self._new_dynamic_obstacle_lifecycle_report(
                self._dynamic_obstacle_plan,
                ros_time_offset_s=(
                    self._navigation_episode_ros_time_offset_s
                ),
            )
        )
        self._metadata["dynamic_obstacle_raw_cloud_lifecycle_report"] = (
            self._new_dynamic_obstacle_raw_cloud_lifecycle_report(
                self._dynamic_obstacle_plan
            )
        )
        self._metadata["episode_support_reset_pose_write_report"] = (
            self._write_episode_support_physics_poses(
                episode_spec,
                reason="episode_reset_after_manager_reset",
            )
        )
        self._metadata["object_reset_for_navigation_report"] = (
            self._reset_object_pose_and_motion(
                episode_spec,
                sleep_until_contact=True,
                reason="episode_reset_before_navigation",
            )
        )
        self._write_dynamic_obstacle_poses(
            elapsed_time_s=0.0,
            physics_step_index=0,
            reason="episode_reset_before_fabric_sync",
        )
        self._metadata["object_pose_debug_after_reset"] = self._object_initial_pose_diagnostic(
            episode_spec,
            label="after_runtime_reset",
        )
        # SingleRigidPrim writes the reset pose through a PhysX tensor view.  Push
        # that fresh state into Fabric before RTX reads it; otherwise a render can
        # re-expose the previous episode's cached transform even though PhysX was
        # already reset correctly.
        self._runtime.sim.forward()
        support_before_render = self._episode_support_pose_diagnostic(
            label="after_episode_reset_forward",
        )
        self._metadata["episode_support_pose_after_reset_forward"] = (
            support_before_render
        )
        if support_before_render.get("verified") is not True:
            raise RuntimeError(
                "episode support pose changed during reset forward sync: "
                f"{support_before_render}"
            )
        # ManagerBasedEnv reset deliberately does not rerender by default.  Without
        # this explicit no-physics render, the first image after reset belongs to
        # the previous robot/object state even if its logical timestamp says zero.
        self._metadata["episode_reset_camera_render_sync_report"] = (
            self._render_without_physics(
                valid_state_step=0,
                reason="episode_reset_state_sync",
            )
        )
        post_render_pose_report = self._object_initial_pose_diagnostic(
            episode_spec,
            label="after_runtime_reset_fabric_render_sync",
        )
        self._metadata["object_pose_debug_after_reset_render"] = (
            post_render_pose_report
        )
        object_pose_reset_gate = _object_pose_reset_gate(
            object_pose_required=(
                episode_spec.object_initial_pose is not None
            ),
            diagnostic=post_render_pose_report,
        )
        self._metadata["object_pose_reset_gate"] = object_pose_reset_gate
        support_after_render = self._episode_support_pose_diagnostic(
            label="after_episode_reset_render",
        )
        self._metadata["episode_support_pose_after_reset_render"] = (
            support_after_render
        )
        if support_after_render.get("verified") is not True:
            raise RuntimeError(
                "episode support pose changed during reset render sync: "
                f"{support_after_render}"
            )
        if (
            object_pose_reset_gate["required"]
            and not object_pose_reset_gate["verified"]
        ):
            raise RuntimeError(
                "object pose changed during reset Fabric/render synchronization: "
                f"{post_render_pose_report}"
            )
        self.refresh_viewport(reason="reset_episode")
        # transient-local writer 可能仍保存上一 episode 的 true；reset 全部
        # 成功后必须在同一连续时钟显式发布初始 false，禁止新任务继承冻结。
        self._publish_stair_execution_frozen_for_action(
            RobotAction.idle(source="episode_reset")
        )

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
        camera_images = self._read_camera_images()
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
            camera_images=camera_images,
            metadata=metadata,
        )

    def apply(self, action: RobotAction) -> None:
        self._require_ready()
        stair_probe_telemetry_requested = bool(
            action.metadata.get("stair_fixed_command_probe") is True
        )
        if not stair_probe_telemetry_requested:
            # Probe 报告只属于产生它的 action；下一条普通 action 到来时立即
            # 清除 runtime 缓存，不能让 verify/failed 状态继承最后一帧证据。
            self._pending_stair_probe_low_level_telemetry = None
            self._metadata.pop("stair_probe_low_level_telemetry", None)
        if action.metadata.get("navigation_emergency_stop") is True:
            raw_stop_reason = action.metadata.get(
                "navigation_emergency_stop_reason",
            )
            self._navigation_emergency_stop_reason = (
                raw_stop_reason.strip()
                if isinstance(raw_stop_reason, str) and raw_stop_reason.strip()
                else "navigation_failed"
            )
        stair_execution_frozen, _stair_phase, _stair_reason = (
            _stair_execution_frozen_from_action(
                action,
                emergency_stop_latched=(
                    getattr(
                        self,
                        "_navigation_emergency_stop_reason",
                        None,
                    )
                    is not None
                ),
            )
        )
        if action.metadata.get("skip_physics_step") is True:
            # ``skip_physics_step`` 是严格的无物理事务：既不能推进 PhysX，也不能
            # 提前推进 RL policy warmup、ActionManager history 或 actuator target。
            # 否则 reset 审计帧会吞掉第一条 warmup action，真正的首个物理步从
            # 第二条 action 开始，在不规则碰撞地面上会产生明显冲击甚至把机器狗
            # 直接打翻。状态机动作仍保留在 recorder/last_action 中用于审计。
            self._last_action = action
            self._action_prepared = False
            self._metadata["last_no_physics_action_report"] = {
                "skipped": True,
                "source": action.source,
                "skip_reason": action.metadata.get("skip_reason"),
                "policy_action_processed": False,
                "action_history_advanced": False,
                "physics_step_required": False,
            }
            # 无物理事务不会进入下方 action staging，但它仍是一次成功的
            # runtime action，必须刷新 transient-local planner 状态。
            self._publish_stair_execution_frozen_for_action(action)
            return

        if stair_execution_frozen:
            # true 必须在命令 drain、目标发布、对象/锁/关节 staging 与 policy
            # 推理之前送达。后续任一异常都让 durable true 留在图中，SCAN
            # 不会继承上一拍 false 继续优化；false 则只在整拍成功后发布。
            self._publish_stair_execution_frozen_for_action(action)
        command_gate = getattr(self, "_cmd_vel_to_policy", None)
        temporary_inhibit_requested = bool(
            action.metadata.get("navigation_cmd_vel_inhibit") is True
            or action.metadata.get("navigation_base_pose_lock") is True
        )
        raw_inhibit_reason = action.metadata.get(
            "navigation_cmd_vel_inhibit_reason"
        )
        temporary_inhibit_reason = (
            raw_inhibit_reason.strip()
            if (
                temporary_inhibit_requested
                and isinstance(raw_inhibit_reason, str)
                and raw_inhibit_reason.strip()
            )
            else "navigation_base_pose_lock"
            if temporary_inhibit_requested
            else None
        )
        command_report: PolicyCommandWriteReport | None = None
        if command_gate is not None and (
            self._environment_terminated
            or self._navigation_emergency_stop_reason is not None
            or temporary_inhibit_reason is not None
        ):
            # 冻结、临时抑制和急停必须先由当前唯一 owner 清零。后续对象
            # 初始化、root/关节锁或机械臂 staging 任一抛错时，上一拍非零
            # policy command 都不能继续留在 ManagerBasedEnv 的命令缓存中。
            command_report = self._apply_scan_cmd_vel_to_policy(
                environment_terminated=self._environment_terminated,
                emergency_stop_reason=(
                    self._navigation_emergency_stop_reason
                ),
                temporary_inhibit_reason=temporary_inhibit_reason,
            )
            self._record_scan_policy_write_report(
                action=action,
                command_report=command_report,
                temporary_inhibit_reason=temporary_inhibit_reason,
            )

        # 生产 PCT 目标只能在唯一 policy owner 已写零后发布。上一物理步已经
        # 先发布同一连续仿真时间域的 /clock 与 Odometry，PCT 收到目标时可
        # 立即用新鲜 base 位姿规划，不会在发目标后再补传状态。
        self._publish_pct_goal_request(action)

        self._apply_object_initialization_pose_stabilization(action)
        self._configure_manipulation_base_lock(action)
        arm_report = self._stage_arm_target(action)
        gripper_report = self._stage_gripper_target(action)
        if command_gate is not None:
            if command_report is None:
                command_report = self._apply_scan_cmd_vel_to_policy(
                    environment_terminated=self._environment_terminated,
                    emergency_stop_reason=(
                        self._navigation_emergency_stop_reason
                    ),
                    temporary_inhibit_reason=temporary_inhibit_reason,
                )
                self._record_scan_policy_write_report(
                    action=action,
                    command_report=command_report,
                    temporary_inhibit_reason=temporary_inhibit_reason,
                )
            self._poll_scan_goal_reached()
        elif self._environment_terminated:
            self._adapter.apply_base_command(0.0, 0.0, 0.0)
        else:
            self._adapter.apply_base_command(*action.base_velocity)
        if stair_probe_telemetry_requested:
            policy_action = self._adapter.compute_policy_action(
                refresh_observations=True,
                capture_stair_probe_telemetry=True,
            )
        else:
            policy_action = self._adapter.compute_policy_action(
                refresh_observations=True
            )
        self._update_velocity_command_visualization(action)
        self._runtime.action_manager.process_action(policy_action.to(self._runtime.device))
        if stair_probe_telemetry_requested:
            self._begin_stair_probe_low_level_telemetry(action)
        self._last_action = action
        self._metadata["last_arm_action_report"] = arm_report
        self._metadata["last_gripper_action_report"] = gripper_report
        self._record_joint_action_apply(action, arm_report, gripper_report)
        if not stair_execution_frozen:
            # 解冻只能发生在 action staging、policy 写入与所有锁配置均成功
            # 之后；publisher 任一错误直接中止本拍。
            self._publish_stair_execution_frozen_for_action(action)
        self._action_prepared = True

    def _publish_stair_execution_frozen_for_action(
        self,
        action: RobotAction,
    ) -> OgnStairExecutionFreezePublicationReport | None:
        """为当前精确 Path 发布类型化楼梯冻结快照并写入 metadata。"""

        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if (
            bridge is None
            or not bridge.config.enable_stair_execution_frozen_publisher
        ):
            return None
        if bridge.active_reference_path_stamp_ns <= 0:
            # reset 与 Path tombstone 期间没有可绑定的代际；类型化协议禁止
            # 发布无身份 false。SCAN 在收到下一代 Path 后保持 fail-closed，
            # 直到本 runtime 轮询到同一 Path 并发布第一条精确快照。
            return None
        frozen, phase, decision_reason = (
            _stair_execution_frozen_from_action(
                action,
                emergency_stop_latched=(
                    getattr(
                        self,
                        "_navigation_emergency_stop_reason",
                        None,
                    )
                    is not None
                ),
            )
        )
        report = bridge.publish_stair_execution_frozen(
            frozen,
            timestamp=self._navigation_ros2_timestamp(),
        )
        fixed_report = {
            "schema": "isaac_stair_execution_frozen_v1",
            "message_type": "scan_planner_msgs/msg/StairExecutionFreeze",
            "source": "isaac_action_metadata",
            "topic": report.source_topic,
            "publish_timestamp": float(report.publish_timestamp),
            "header": {
                "frame_id": bridge.config.odom_frame_id,
                "stamp": {
                    "sec": int(report.header_stamp_sec),
                    "nanosec": int(report.header_stamp_nanosec),
                },
            },
            "reference_path_stamp": {
                "sec": int(report.reference_path_stamp_sec),
                "nanosec": int(report.reference_path_stamp_nanosec),
            },
            "reference_path_stamp_ns": int(report.reference_path_stamp_ns),
            "writer_id": report.writer_id,
            "writer_epoch": report.writer_epoch,
            "sequence": int(report.sequence),
            "value": bool(report.value),
            "frozen": bool(report.frozen),
            "action_source": action.source,
            "action_phase": phase,
            "decision_reason": decision_reason,
        }
        self._metadata[
            "navigation_stair_execution_frozen_last_publish_report"
        ] = fixed_report
        return report

    @staticmethod
    def _new_navigation_policy_gate_lifecycle_report() -> dict[str, Any]:
        """建立 supervisor 状态、policy 许可和全局重规划的单 episode 证据。"""

        return {
            "schema": "navigation_policy_gate_lifecycle_v1",
            "policy_write_count": 0,
            "motion_allowed_write_count": 0,
            "identity_verified_tracking_write_count": 0,
            # policy 可在同一个 ControllerStatus 快照下连续写多拍；ring
            # 只为每个 typed snapshot 钉住一条代表性证据，不能拿总写入数
            # 推导 ring 长度。
            "identity_verified_tracking_snapshot_count": 0,
            "observed_status_sequence_count": 0,
            "identity_valid_observed_status_count": 0,
            "forced_zero_write_count": 0,
            "first_identity_verified_tracking_write": None,
            "last_identity_verified_tracking_write": None,
            "identity_verified_tracking_write_reports": [],
            "dropped_identity_verified_tracking_write_report_count": 0,
            "first_identity_valid_observed_status": None,
            "last_identity_valid_observed_status": None,
            "last_observed_status_sequence": None,
            "last_observed_state": None,
            "observed_state_transition_count": 0,
            "observed_state_counts": {},
            "observed_reason_counts": {},
            "maximum_consecutive_scan_failures": 0,
            "global_replan_requested_status_count": 0,
            "global_replan_in_flight_status_count": 0,
            "distinct_global_replan_request_ids": [],
            "distinct_pct_plan_ids": [],
            "first_global_replan_status": None,
            "last_global_replan_status": None,
            "global_replan_pending_recovery": False,
            "tracking_after_global_replan_observed": False,
            "global_replan_recovery_count": 0,
            "emergency_stop_observed_status_count": 0,
            "goal_reached_observed_status_count": 0,
            "last_observed_status": None,
            "last_write_sequence": None,
            "last_stop_reasons": [],
            "stop_reason_counts": {},
        }

    @staticmethod
    def _controller_status_snapshot_key(
        snapshot: object,
    ) -> tuple[int, int, int, int, int, int] | None:
        """提取不可变 typed controller snapshot 身份；非法证据失败关闭。"""

        if not isinstance(snapshot, dict) or snapshot.get("source") != (
            "ros2_scan_planner_msgs_controller_status"
        ):
            return None
        identity = snapshot.get("identity")
        if not isinstance(identity, dict):
            return None
        values = (
            identity.get("reference_path_stamp_ns"),
            identity.get("bspline_header_stamp_ns"),
            identity.get("start_time_ns"),
            identity.get("traj_id"),
            snapshot.get("status_sequence"),
            snapshot.get("state"),
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in values
        ):
            return None
        (
            reference_path_stamp_ns,
            bspline_header_stamp_ns,
            start_time_ns,
            traj_id,
            status_sequence,
            state,
        ) = values
        if (
            reference_path_stamp_ns <= 0
            or bspline_header_stamp_ns <= 0
            or start_time_ns <= 0
            or traj_id < 0
            or status_sequence <= 0
            or state not in {*range(13), 255}
        ):
            return None
        return (
            reference_path_stamp_ns,
            bspline_header_stamp_ns,
            start_time_ns,
            traj_id,
            status_sequence,
            state,
        )

    @classmethod
    def _pin_identity_verified_tracking_write_report(
        cls,
        lifecycle: dict[str, Any],
        evidence: dict[str, Any],
    ) -> bool:
        """为每个 typed snapshot 钉住首条 policy 写入，按 FIFO 有界保留。"""

        tracking_writes = lifecycle.get(
            "identity_verified_tracking_write_reports"
        )
        if not isinstance(tracking_writes, list):
            raise RuntimeError("policy gate identity tracking 写入 ring 非数组。")
        snapshot_key = cls._controller_status_snapshot_key(
            evidence.get("scan_controller_status_snapshot")
        )
        if snapshot_key is None:
            return False
        for retained_index, retained in enumerate(tracking_writes):
            if not isinstance(retained, dict):
                raise RuntimeError(
                    "policy gate identity tracking 写入证据不是对象。"
                )
            retained_key = cls._controller_status_snapshot_key(
                retained.get("scan_controller_status_snapshot")
            )
            if retained_key is None:
                raise RuntimeError(
                    "policy gate identity tracking 写入缺少合法 typed snapshot。"
                )
            if retained_key == snapshot_key:
                snapshot = evidence.get("scan_controller_status_snapshot")
                if (
                    isinstance(snapshot, dict)
                    and snapshot.get("active_sensing_yaw_only") is True
                ):
                    retained_command = retained.get("written_command")
                    incoming_command = evidence.get("written_command")
                    retained_abs_wz = (
                        abs(float(retained_command[2]))
                        if isinstance(retained_command, list)
                        and len(retained_command) == 3
                        else -1.0
                    )
                    incoming_abs_wz = (
                        abs(float(incoming_command[2]))
                        if isinstance(incoming_command, list)
                        and len(incoming_command) == 3
                        else -1.0
                    )
                    if (
                        retained_abs_wz <= 1.0e-12
                        and incoming_abs_wz > 1.0e-12
                    ):
                        # 主动观测必须保留真实非零旋转，而不是同 snapshot
                        # 下较早的零命令；首条非零写入一旦钉住就不再覆盖。
                        tracking_writes[retained_index] = copy.deepcopy(evidence)
                return False

        tracking_writes.append(copy.deepcopy(evidence))
        if len(tracking_writes) > 128:
            # 不同 snapshot 超界时只淘汰最旧的钉住证据；同 snapshot 的
            # 50 Hz 写入不会占用新槽位，也不会挤掉早期绕障证据。
            tracking_writes.pop(0)
            lifecycle[
                "dropped_identity_verified_tracking_write_report_count"
            ] = int(
                lifecycle.get(
                    "dropped_identity_verified_tracking_write_report_count",
                    0,
                )
            ) + 1
        return True

    def _record_scan_policy_write_report(
        self,
        *,
        action: RobotAction,
        command_report: PolicyCommandWriteReport,
        temporary_inhibit_reason: str | None,
    ) -> None:
        """记录唯一 owner 在当前 action 中实际执行的一次 policy 写入。"""

        self._scan_policy_write_sequence += 1
        command_gate = getattr(self, "_cmd_vel_to_policy", None)
        diagnostics_callback = getattr(
            command_gate,
            "navigation_gate_diagnostics",
            None,
        )
        navigation_gate_diagnostics = (
            diagnostics_callback()
            if callable(diagnostics_callback)
            else None
        )
        bridge = getattr(self, "_ros2_ogn_bridge", None)
        observed_callback = getattr(
            bridge,
            "navigation_status_observed_diagnostics",
            None,
        )
        navigation_status_observed_report = (
            observed_callback()
            if callable(observed_callback)
            else None
        )
        write_report = {
            "write_sequence": int(self._scan_policy_write_sequence),
            "timestamp": float(command_report.timestamp),
            "owner_id": command_report.owner_id,
            "requested_command": (
                None
                if command_report.requested_command is None
                else list(command_report.requested_command.as_tuple())
            ),
            "limited_target": list(
                command_report.limited_target.as_tuple()
            ),
            "written_command": list(
                command_report.written_command.as_tuple()
            ),
            "motion_allowed": bool(command_report.motion_allowed),
            "stop_reasons": list(command_report.stop_reasons),
            "clipped_axes": list(command_report.clipped_axes),
            "rate_limited_axes": list(command_report.rate_limited_axes),
            "navigation_emergency_stop_latched": (
                self._navigation_emergency_stop_reason is not None
            ),
            "navigation_emergency_stop_reason": (
                self._navigation_emergency_stop_reason
            ),
            "navigation_cmd_vel_inhibited": (
                temporary_inhibit_reason is not None
            ),
            "navigation_cmd_vel_inhibit_reason": temporary_inhibit_reason,
            "cmd_vel_source_sequence": getattr(
                self,
                "_last_scan_cmd_vel_source_sequence",
                None,
            ),
            "cmd_vel_source_receipt_timestamp": getattr(
                self,
                "_last_scan_cmd_vel_source_receipt_timestamp",
                None,
            ),
            "cmd_vel_sample_received_this_tick": bool(
                getattr(
                    self,
                    "_scan_cmd_vel_sample_received_this_tick",
                    False,
                )
            ),
            "cmd_vel_sample_drained_this_tick": bool(
                getattr(
                    self,
                    "_scan_cmd_vel_sample_drained_this_tick",
                    False,
                )
            ),
            "last_cmd_vel_drain_sequence": getattr(
                self,
                "_last_scan_cmd_vel_drain_sequence",
                None,
            ),
            "last_cmd_vel_drain_receipt_timestamp": getattr(
                self,
                "_last_scan_cmd_vel_drain_receipt_timestamp",
                None,
            ),
            "navigation_status_observed_report": (
                navigation_status_observed_report
            ),
            "policy_navigation_gate_consumed_report": (
                navigation_gate_diagnostics
            ),
            "pipeline_base_velocity_ignored": list(action.base_velocity),
        }
        self._metadata["scan_cmd_vel_last_write_report"] = write_report
        self._update_active_sensing_policy_write_evidence(write_report)

        lifecycle = self._metadata.get(
            "navigation_policy_gate_lifecycle_report"
        )
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("schema")
            != "navigation_policy_gate_lifecycle_v1"
        ):
            lifecycle = self._new_navigation_policy_gate_lifecycle_report()
            self._metadata[
                "navigation_policy_gate_lifecycle_report"
            ] = lifecycle
        lifecycle["policy_write_count"] = int(
            lifecycle.get("policy_write_count", 0)
        ) + 1
        if command_report.motion_allowed:
            lifecycle["motion_allowed_write_count"] = int(
                lifecycle.get("motion_allowed_write_count", 0)
            ) + 1
        if (
            not command_report.motion_allowed
            and command_report.written_command.as_tuple()
            == (0.0, 0.0, 0.0)
        ):
            lifecycle["forced_zero_write_count"] = int(
                lifecycle.get("forced_zero_write_count", 0)
            ) + 1
        permit_report = (
            navigation_gate_diagnostics.get("permit")
            if isinstance(navigation_gate_diagnostics, dict)
            else None
        )
        identity_verified_tracking = bool(
            command_report.motion_allowed
            and isinstance(permit_report, dict)
            and permit_report.get("state") == 3
            and permit_report.get("allow_tracking_command") is True
            and permit_report.get("force_zero_velocity") is False
            and permit_report.get("identity_valid") is True
            and navigation_gate_diagnostics.get(
                "command_identity_matches_permit"
            )
            is True
        )
        if identity_verified_tracking:
            lifecycle["identity_verified_tracking_write_count"] = int(
                lifecycle.get(
                    "identity_verified_tracking_write_count",
                    0,
                )
            ) + 1
            evidence = {
                "write_sequence": int(self._scan_policy_write_sequence),
                "timestamp": float(command_report.timestamp),
                "written_command": list(
                    command_report.written_command.as_tuple()
                ),
                "navigation_gate_diagnostics": (
                    navigation_gate_diagnostics
                ),
                # Twist 本身没有 Header/identity；至少把本次 policy 写入时
                # runtime 已接收的 controller typed 状态做不可变快照。最终
                # recovery 只能引用同一 TRACKING identity 的该快照。
                "scan_controller_status_snapshot": copy.deepcopy(
                    self._metadata.get("scan_controller_status_last_report")
                ),
            }
            if lifecycle.get(
                "first_identity_verified_tracking_write"
            ) is None:
                lifecycle[
                    "first_identity_verified_tracking_write"
                ] = evidence
            lifecycle["last_identity_verified_tracking_write"] = evidence
            pinned_new_snapshot = self._pin_identity_verified_tracking_write_report(
                lifecycle,
                evidence,
            )
            if pinned_new_snapshot:
                lifecycle["identity_verified_tracking_snapshot_count"] = int(
                    lifecycle.get(
                        "identity_verified_tracking_snapshot_count",
                        0,
                    )
                ) + 1
        observed_status = (
            navigation_status_observed_report.get("status")
            if isinstance(navigation_status_observed_report, dict)
            else None
        )
        if isinstance(observed_status, dict):
            observed_sequence = observed_status.get("status_sequence")
            if (
                isinstance(observed_sequence, int)
                and not isinstance(observed_sequence, bool)
                and observed_sequence
                != lifecycle.get("last_observed_status_sequence")
            ):
                lifecycle["observed_status_sequence_count"] = int(
                    lifecycle.get("observed_status_sequence_count", 0)
                ) + 1
                lifecycle["last_observed_status_sequence"] = int(
                    observed_sequence
                )
                observed_state = observed_status.get("state")
                if (
                    not isinstance(observed_state, int)
                    or isinstance(observed_state, bool)
                ):
                    raise RuntimeError("NavigationStatus.state 证据不是整数。")
                state_names = {
                    0: "idle",
                    1: "global_planning",
                    2: "local_planning",
                    3: "tracking",
                    4: "global_replan",
                    5: "emergency_stop",
                    6: "goal_reached",
                    255: "unknown",
                }
                state_name = state_names.get(
                    observed_state,
                    f"invalid_{observed_state}",
                )
                previous_state = lifecycle.get("last_observed_state")
                if (
                    previous_state is not None
                    and int(previous_state) != observed_state
                ):
                    lifecycle["observed_state_transition_count"] = int(
                        lifecycle.get("observed_state_transition_count", 0)
                    ) + 1
                lifecycle["last_observed_state"] = observed_state
                state_counts = lifecycle.get("observed_state_counts")
                reason_counts = lifecycle.get("observed_reason_counts")
                if not isinstance(state_counts, dict) or not isinstance(
                    reason_counts, dict
                ):
                    raise RuntimeError("NavigationStatus 生命周期计数字段非法。")
                state_counts[state_name] = int(state_counts.get(state_name, 0)) + 1
                observed_reason = str(observed_status.get("reason") or "<empty>")
                reason_counts[observed_reason] = int(
                    reason_counts.get(observed_reason, 0)
                ) + 1
                lifecycle["maximum_consecutive_scan_failures"] = max(
                    int(lifecycle.get("maximum_consecutive_scan_failures", 0)),
                    int(observed_status.get("consecutive_scan_failures", 0)),
                )
                for field_name, report_name in (
                    (
                        "global_replan_request_id",
                        "distinct_global_replan_request_ids",
                    ),
                    ("pct_plan_id", "distinct_pct_plan_ids"),
                ):
                    raw_identifier = observed_status.get(field_name)
                    if (
                        isinstance(raw_identifier, int)
                        and not isinstance(raw_identifier, bool)
                        and raw_identifier > 0
                    ):
                        identifiers = lifecycle.get(report_name)
                        if not isinstance(identifiers, list):
                            raise RuntimeError(
                                f"NavigationStatus {report_name} 不是数组。"
                            )
                        if raw_identifier not in identifiers:
                            if len(identifiers) >= 64:
                                raise RuntimeError(
                                    f"NavigationStatus {report_name} 超过 64 项。"
                                )
                            identifiers.append(int(raw_identifier))
                replan_active = bool(
                    observed_state == 4
                    or observed_status.get("global_replan_requested") is True
                    or observed_status.get("global_replan_in_flight") is True
                )
                if observed_status.get("global_replan_requested") is True:
                    lifecycle["global_replan_requested_status_count"] = int(
                        lifecycle.get(
                            "global_replan_requested_status_count",
                            0,
                        )
                    ) + 1
                if observed_status.get("global_replan_in_flight") is True:
                    lifecycle["global_replan_in_flight_status_count"] = int(
                        lifecycle.get(
                            "global_replan_in_flight_status_count",
                            0,
                        )
                    ) + 1
                observed_evidence = {
                    "write_sequence": int(self._scan_policy_write_sequence),
                    "timestamp": float(command_report.timestamp),
                    "navigation_status_observed_report": (
                        navigation_status_observed_report
                    ),
                }
                if replan_active:
                    if lifecycle.get("first_global_replan_status") is None:
                        lifecycle["first_global_replan_status"] = observed_evidence
                    lifecycle["last_global_replan_status"] = observed_evidence
                    lifecycle["global_replan_pending_recovery"] = True
                if observed_state == 5:
                    lifecycle["emergency_stop_observed_status_count"] = int(
                        lifecycle.get(
                            "emergency_stop_observed_status_count",
                            0,
                        )
                    ) + 1
                if observed_state == 6:
                    lifecycle["goal_reached_observed_status_count"] = int(
                        lifecycle.get(
                            "goal_reached_observed_status_count",
                            0,
                        )
                    ) + 1
                if (
                    observed_state == 3
                    and observed_status.get("identity_valid") is True
                    and lifecycle.get("global_replan_pending_recovery") is True
                ):
                    lifecycle["tracking_after_global_replan_observed"] = True
                    lifecycle["global_replan_recovery_count"] = int(
                        lifecycle.get("global_replan_recovery_count", 0)
                    ) + 1
                    lifecycle["global_replan_pending_recovery"] = False
                lifecycle["last_observed_status"] = observed_evidence
                if observed_status.get("identity_valid") is True:
                    lifecycle["identity_valid_observed_status_count"] = int(
                        lifecycle.get(
                            "identity_valid_observed_status_count",
                            0,
                        )
                    ) + 1
                    evidence = {
                        "write_sequence": int(
                            self._scan_policy_write_sequence
                        ),
                        "timestamp": float(command_report.timestamp),
                        "navigation_status_observed_report": (
                            navigation_status_observed_report
                        ),
                    }
                    if lifecycle.get(
                        "first_identity_valid_observed_status"
                    ) is None:
                        lifecycle[
                            "first_identity_valid_observed_status"
                        ] = evidence
                    lifecycle[
                        "last_identity_valid_observed_status"
                    ] = evidence
        lifecycle["last_write_sequence"] = int(
            self._scan_policy_write_sequence
        )
        lifecycle["last_stop_reasons"] = list(
            command_report.stop_reasons
        )
        stop_reason_counts = lifecycle.get("stop_reason_counts")
        if not isinstance(stop_reason_counts, dict):
            stop_reason_counts = {}
            lifecycle["stop_reason_counts"] = stop_reason_counts
        for reason in command_report.stop_reasons:
            stop_reason_counts[reason] = int(
                stop_reason_counts.get(reason, 0)
            ) + 1
        if getattr(
            self,
            "_dynamic_obstacle_plan",
            DynamicObstaclePlan(),
        ).enabled:
            # recovery 的最后一块证据来自 policy 写入。若后续没有新的
            # diagnostics/status 消息，也必须在本次写入后立刻重建聚合结果。
            self._refresh_dynamic_navigation_evidence_report()

    def _publish_pct_goal_request(self, action: RobotAction) -> None:
        """发布本代 goal，并在 PCT 回程证据出现前允许同 stamp 传输重试。"""

        raw_request = action.metadata.get("navigation_pct_goal_request")
        if raw_request is None:
            return
        if not isinstance(raw_request, dict):
            raise ValueError("navigation_pct_goal_request 必须是对象。")
        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if bridge is None or not bridge.config.enable_pct_goal_publisher:
            raise RuntimeError("收到 PCT goal 请求，但 OGN goal publisher 未启用。")
        generation = raw_request.get("generation")
        position = raw_request.get("position_base_xyz")
        yaw = raw_request.get("yaw")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or raw_request.get("frame_id") != "world"
            or raw_request.get("height_semantics") != "base"
            or not isinstance(position, (list, tuple))
            or len(position) != 3
        ):
            raise ValueError("navigation_pct_goal_request 合同非法。")
        try:
            normalized_position = tuple(float(value) for value in position)
            normalized_yaw = float(yaw)
        except (TypeError, ValueError) as exc:
            raise ValueError("PCT goal 必须包含有限 base xyz/yaw。") from exc
        if not all(
            math.isfinite(value)
            for value in (*normalized_position, normalized_yaw)
        ):
            raise ValueError("PCT goal 必须包含有限 base xyz/yaw。")
        provenance_required = raw_request.get(
            "effective_goal_provenance_required",
            False,
        )
        if not isinstance(provenance_required, bool):
            raise ValueError("effective_goal_provenance_required 必须是布尔值。")
        transport_retry = raw_request.get("transport_retry", False)
        if not isinstance(transport_retry, bool):
            raise ValueError("transport_retry 必须是布尔值。")
        effective_goal_provenance: dict[str, Any] | None = None
        if provenance_required:
            effective_goal_provenance = _validated_effective_goal_provenance(
                raw_request.get("effective_goal_provenance"),
                position_base_xyz=normalized_position,
            )
        provenance_identity = (
            None
            if effective_goal_provenance is None
            else json.dumps(
                effective_goal_provenance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        identity = (
            generation,
            *normalized_position,
            normalized_yaw,
            "world",
            "base",
            provenance_required,
            provenance_identity,
        )
        previous = self._last_pct_goal_request_identity
        if previous == identity:
            if not transport_retry:
                return
            expected_sample = self._last_pct_goal_sample
            if expected_sample is None:
                raise RuntimeError("PCT goal transport retry 缺少首发 sample。")
            sample = bridge.republish_last_pct_goal()
            if sample != expected_sample:
                raise RuntimeError("PCT goal transport retry 改变了 stamped sample。")
            previous_report = self._metadata.get("scan_pct_goal_last_report")
            if not isinstance(previous_report, dict):
                raise RuntimeError("PCT goal transport retry 缺少首发报告。")
            report = dict(previous_report)
            report.update(
                {
                    "transport_attempt_count": int(
                        bridge.pct_goal_transport_attempt_count
                    ),
                    "last_transport_attempt_control_step": int(
                        self._step_calls
                    ),
                    "dds_acknowledged": False,
                }
            )
            self._metadata["scan_pct_goal_last_report"] = report
            return
        if transport_retry:
            raise RuntimeError("PCT goal 首发前不能请求 transport retry。")
        if previous is not None and generation <= int(previous[0]):
            raise RuntimeError(
                "同一或回退的 PCT goal generation 不能改变目标 payload。"
            )
        timestamp = self._navigation_ros2_timestamp()
        if timestamp <= 0.0:
            # 零时刻没有合法 ROS stamp；不锁存 identity，让下一控制 tick 重试。
            return
        stamp_ns = int(round(timestamp * 1_000_000_000.0))
        sample = bridge.publish_pct_goal(
            normalized_position,
            normalized_yaw,
            stamp_ns=stamp_ns,
            frame_id="world",
        )
        self._last_pct_goal_request_identity = identity
        self._last_pct_goal_sample = sample
        report = {
            "published": True,
            "source": "isaac_ros2_ogn_pose_stamped",
            "topic": sample.source_topic,
            "frame_id": sample.frame_id,
            "stamp": sample.stamp,
            "sequence": int(sample.sequence),
            "generation": generation,
            "position_base_xyz": list(sample.position_base_xyz),
            "yaw": float(sample.yaw),
            "height_semantics": "base",
            "published_at_control_step": int(self._step_calls),
            "transport_attempt_count": int(
                bridge.pct_goal_transport_attempt_count
            ),
            "first_transport_attempt_control_step": int(self._step_calls),
            "last_transport_attempt_control_step": int(self._step_calls),
            # generic OGN evaluate_sync 只能证明本地触发成功；PCT 的新 Path
            # tombstone/非空 Path 才是跨 DDS 的接收证据。
            "dds_acknowledged": False,
        }
        if provenance_required:
            report["effective_goal_provenance_required"] = True
            report["effective_goal_provenance"] = effective_goal_provenance
        self._metadata["scan_pct_goal_last_report"] = report

    def _begin_stair_probe_low_level_telemetry(
        self,
        action: RobotAction,
    ) -> None:
        """绑定同一次固定命令 policy 推理与即将执行的 control step。"""

        getter = getattr(
            self._adapter,
            "get_stair_probe_policy_pre_step",
            None,
        )
        if callable(getter):
            try:
                pre_step = getter()
            except Exception as exc:
                pre_step = {
                    "available": False,
                    "unavailable_reason": (
                        "adapter_pre_step_telemetry_failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
        else:
            pre_step = {
                "available": False,
                "unavailable_reason": (
                    "adapter_pre_step_telemetry_interface_unavailable"
                ),
            }

        step_dt = float(getattr(self._runtime, "step_dt", 0.0))
        completed_control_step = int(self._step_calls) + 1
        report = {
            "schema": "stair_fixed_command_probe_low_level_v1",
            "probe_only": True,
            "complete": False,
            "alignment": {
                "control_step_index": completed_control_step,
                "pre_step_state_step_index": int(self._step_calls),
                "pre_step_timestamp_s": float(self._step_calls) * step_dt,
                "post_step_state_step_index": None,
                "post_step_timestamp_s": None,
                "physics_substep_count": int(
                    getattr(getattr(self._runtime, "cfg", None), "decimation", 0)
                ),
                "policy_input_and_action": (
                    "pre-step：process_action 已接收该 policy action，物理尚未推进"
                ),
                "contacts_and_pose": (
                    "post-step：完整 decimation 后最后 scene.update 的状态"
                ),
            },
            "pipeline_action": {
                "source": action.source,
                "base_velocity": [
                    float(value) for value in action.base_velocity
                ],
                "stair_probe_phase": action.metadata.get(
                    "stair_probe_phase"
                ),
            },
            "pre_step": pre_step,
            "post_step": {
                "available": False,
                "unavailable_reason": "physics_step_pending",
            },
        }
        self._pending_stair_probe_low_level_telemetry = report
        self._metadata["stair_probe_low_level_telemetry"] = report

    def _complete_stair_probe_low_level_telemetry(
        self,
        *,
        completed_control_step: int,
    ) -> None:
        """在物理步完成后补齐与 pre-step 一一对应的接触和位姿。"""

        report = getattr(
            self,
            "_pending_stair_probe_low_level_telemetry",
            None,
        )
        if report is None:
            return
        expected_step = int(
            report.get("alignment", {}).get("control_step_index", -1)
        )
        if expected_step != int(completed_control_step):
            report["complete"] = False
            report["post_step"] = {
                "available": False,
                "unavailable_reason": "control_step_alignment_mismatch",
                "expected_control_step": expected_step,
                "completed_control_step": int(completed_control_step),
            }
        else:
            getter = getattr(
                self._adapter,
                "capture_stair_probe_post_step",
                None,
            )
            if callable(getter):
                try:
                    post_step = getter()
                except Exception as exc:
                    post_step = {
                        "available": False,
                        "unavailable_reason": (
                            "adapter_post_step_telemetry_failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
            else:
                post_step = {
                    "available": False,
                    "unavailable_reason": (
                        "adapter_post_step_telemetry_interface_unavailable"
                    ),
                }
            step_dt = float(getattr(self._runtime, "step_dt", 0.0))
            report["alignment"].update(
                {
                    "post_step_state_step_index": int(
                        completed_control_step
                    ),
                    "post_step_timestamp_s": (
                        float(completed_control_step) * step_dt
                    ),
                }
            )
            report["post_step"] = post_step
            report["complete"] = True
        pre_step = report.get("pre_step", {})
        post_step = report.get("post_step", {})
        required_data_available = bool(
            isinstance(pre_step, dict)
            and pre_step.get("available") is True
            and isinstance(post_step, dict)
            and post_step.get("available") is True
        )
        report["available"] = required_data_available
        report["unavailable_reason"] = (
            None
            if required_data_available
            else "required_probe_component_unavailable"
        )
        self._metadata["stair_probe_low_level_telemetry"] = report
        self._pending_stair_probe_low_level_telemetry = None

    def _navigation_ros2_timestamp(self) -> float:
        """返回 OGN、命令安全门和 ROS 超时共用的连续仿真时间。"""

        runtime = getattr(self, "_runtime", None)
        physics_dt = float(getattr(runtime, "physics_dt", 0.0))
        return max(
            0.0,
            float(getattr(self, "_ros2_physics_step_count", 0))
            * physics_dt,
        )

    def _apply_scan_cmd_vel_to_policy(
        self,
        *,
        environment_terminated: bool,
        emergency_stop_reason: str | None = None,
        temporary_inhibit_reason: str | None = None,
    ) -> PolicyCommandWriteReport:
        """轮询新 ``/cmd_vel`` 并由唯一安全门写入 policy command buffer。"""

        bridge = getattr(self, "_ros2_ogn_bridge", None)
        command_gate = getattr(self, "_cmd_vel_to_policy", None)
        if bridge is None or command_gate is None:
            raise RuntimeError("SCAN cmd_vel authority 尚未完成初始化。")
        timestamp = self._navigation_ros2_timestamp()
        owner_id = self._cmd_vel_owner_id
        self._scan_cmd_vel_sample_received_this_tick = False
        self._scan_cmd_vel_sample_drained_this_tick = False
        command_gate.renew_control_lease(owner_id, timestamp)
        if environment_terminated:
            return command_gate.emergency_stop(
                owner_id=owner_id,
                now=timestamp,
                reason="environment_terminated",
            )
        if emergency_stop_reason is not None:
            reason = str(emergency_stop_reason).strip()
            if not reason:
                reason = "navigation_failed"
            return command_gate.emergency_stop(
                owner_id=owner_id,
                now=timestamp,
                reason=reason,
            )
        if temporary_inhibit_reason is not None:
            reason = str(temporary_inhibit_reason).strip()
            if not reason:
                reason = "navigation_base_pose_lock"
            report = command_gate.inhibit(
                owner_id=owner_id,
                now=timestamp,
                reason=reason,
            )
            # 锁定期间仍轮询并丢弃 OGN counter 的新 Twist；否则解锁首帧会把
            # 冻结期间积压的最后一条非零命令误当成新鲜输入。必须先由唯一
            # owner 写零再轮询，bridge 本身异常也不能让旧 policy 命令残留。
            if timestamp > 0.0:
                sample = bridge.poll_twist(receipt_timestamp=timestamp)
                if sample is not None:
                    self._last_scan_cmd_vel_source_sequence = int(
                        sample.sequence
                    )
                    self._last_scan_cmd_vel_source_receipt_timestamp = float(
                        sample.receipt_timestamp
                    )
                    self._scan_cmd_vel_sample_drained_this_tick = True
                    self._last_scan_cmd_vel_drain_sequence = int(
                        sample.sequence
                    )
                    self._last_scan_cmd_vel_drain_receipt_timestamp = float(
                        sample.receipt_timestamp
                    )
            return report

        active_zero_gate = self._consume_active_sensing_policy_zero_gate()
        if active_zero_gate is not None:
            # ControllerStatus 可能在 KeepLast(1) Twist 中的首拍零速
            # 被后续 yaw 覆盖后才到 runtime。必须先由唯一
            # owner 真实写零，再 drain 当前最新 Twist；drain 异常时
            # 零速已生效且 gate 保持 armed，下拍继续失败关闭。
            report = command_gate.inhibit(
                owner_id=owner_id,
                now=timestamp,
                reason="active_sensing_identity_zero_gate",
            )
            if timestamp > 0.0:
                sample = bridge.poll_twist(receipt_timestamp=timestamp)
                if sample is not None:
                    self._last_scan_cmd_vel_source_sequence = int(
                        sample.sequence
                    )
                    self._last_scan_cmd_vel_source_receipt_timestamp = float(
                        sample.receipt_timestamp
                    )
                    self._scan_cmd_vel_sample_drained_this_tick = True
                    self._last_scan_cmd_vel_drain_sequence = int(
                        sample.sequence
                    )
                    self._last_scan_cmd_vel_drain_receipt_timestamp = float(
                        sample.receipt_timestamp
                    )
            self._confirm_active_sensing_policy_zero_gate(active_zero_gate)
            return report

        # Twist 没有 Header；在零时刻尚无有效 /clock，保持 claim() 写入的零速。
        if timestamp > 0.0:
            sample = bridge.poll_twist(receipt_timestamp=timestamp)
            if sample is not None:
                self._last_scan_cmd_vel_source_sequence = int(sample.sequence)
                self._last_scan_cmd_vel_source_receipt_timestamp = float(
                    sample.receipt_timestamp
                )
                self._scan_cmd_vel_sample_received_this_tick = True
                command_gate.receive(
                    sample.planar_command,
                    sample.receipt_timestamp,
                    owner_id,
                )
        return command_gate.tick(timestamp, owner_id)

    def _poll_scan_goal_reached(self) -> None:
        """轮询 SCAN 完成事件，并保存带接收序号的本轮生命周期证据。"""

        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if bridge is None or not bridge.config.enable_goal_reached_subscription:
            return
        timestamp = self._navigation_ros2_timestamp()
        if timestamp <= 0.0:
            return
        sample = bridge.poll_goal_reached(receipt_timestamp=timestamp)
        if sample is None:
            return
        self._metadata["scan_goal_reached_last_sample"] = {
            "value": bool(sample.value),
            "receipt_timestamp": float(sample.receipt_timestamp),
            "sequence": int(sample.sequence),
        }

    @staticmethod
    def _new_scan_controller_status_lifecycle_report() -> dict[str, Any]:
        """建立 SCAN controller 单 episode 轨迹与恢复生命周期证据。"""

        return {
            "schema": "scan_controller_status_lifecycle_v1",
            "sample_count": 0,
            "active_sensing_status_count": 0,
            "first_status_sequence": None,
            "last_status_sequence": None,
            "maximum_acceptance_sequence": 0,
            "event_counts": {},
            "state_counts": {},
            "reason_counts": {},
            "accepted_status_count": 0,
            "trajectory_valid_status_count": 0,
            "candidate_rejection_count": 0,
            "goal_latched_same_path_candidate_rejection_count": 0,
            "unexpected_candidate_rejection_count": 0,
            "first_candidate_rejection_status": None,
            "last_candidate_rejection_status": None,
            "emergency_stop_status_count": 0,
            "tracking_status_count": 0,
            "tracking_status_reports": [],
            "dropped_tracking_status_report_count": 0,
            "first_tracking_status": None,
            "last_tracking_status": None,
            "goal_reached_status_count": 0,
            "distinct_accepted_trajectory_count": 0,
            "trajectory_replacement_count": 0,
            "accepted_trajectory_identities": [],
            "accepted_status_reports": [],
            "dropped_accepted_status_report_count": 0,
            "first_accepted_status": None,
            "last_accepted_status": None,
            "first_emergency_stop_status": None,
            "last_emergency_stop_status": None,
            "tracking_after_emergency_stop_observed": False,
            "emergency_stop_recovery_count": 0,
            "emergency_stop_pending_recovery": False,
            "last_status": None,
        }

    def _update_scan_controller_status_lifecycle_report(
        self,
        status_report: dict[str, Any],
    ) -> dict[str, Any]:
        """累计 controller 的严格递增状态，保留局部重规划与恢复证据。"""

        lifecycle = self._metadata.get(
            "scan_controller_status_lifecycle_report"
        )
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("schema")
            != "scan_controller_status_lifecycle_v1"
        ):
            lifecycle = self._new_scan_controller_status_lifecycle_report()

        status_sequence = int(status_report["status_sequence"])
        previous_sequence = lifecycle.get("last_status_sequence")
        if (
            previous_sequence is not None
            and status_sequence <= int(previous_sequence)
        ):
            raise RuntimeError(
                "SCAN controller status_sequence 在 runtime 生命周期中未严格递增。"
            )
        if lifecycle.get("first_status_sequence") is None:
            lifecycle["first_status_sequence"] = status_sequence
        lifecycle["last_status_sequence"] = status_sequence
        active_sensing = status_report.get("active_sensing_yaw_only") is True
        if active_sensing:
            lifecycle["active_sensing_status_count"] = int(
                lifecycle.get("active_sensing_status_count", 0)
            ) + 1
        else:
            lifecycle["sample_count"] = int(
                lifecycle.get("sample_count", 0)
            ) + 1
        lifecycle["maximum_acceptance_sequence"] = max(
            int(lifecycle.get("maximum_acceptance_sequence", 0)),
            int(status_report["acceptance_sequence"]),
        )

        event_names = {
            0: "initial",
            1: "accepted",
            2: "rejected",
            3: "invalidated",
            4: "state_changed",
            5: "duplicate",
        }
        state_names = {
            0: "waiting_for_trajectory",
            1: "waiting_for_reference_path",
            2: "waiting_for_odometry",
            3: "waiting_for_cloud",
            4: "trajectory_timeout",
            5: "odometry_timeout",
            6: "cloud_timeout",
            7: "invalid_clock",
            8: "emergency_stop",
            9: "aligning_yaw",
            10: "tracking",
            11: "trajectory_finished",
            12: "goal_reached",
            255: "unknown",
        }
        event = int(status_report["event"])
        state = int(status_report["state"])
        event_name = event_names.get(event, f"invalid_{event}")
        state_name = state_names.get(state, f"invalid_{state}")
        event_counts = lifecycle.get("event_counts")
        state_counts = lifecycle.get("state_counts")
        reason_counts = lifecycle.get("reason_counts")
        if not isinstance(event_counts, dict) or not isinstance(state_counts, dict):
            raise RuntimeError("SCAN controller lifecycle 计数字段非法。")
        if not isinstance(reason_counts, dict):
            raise RuntimeError("SCAN controller lifecycle reason_counts 非对象。")
        if not active_sensing:
            event_counts[event_name] = int(event_counts.get(event_name, 0)) + 1
            state_counts[state_name] = int(state_counts.get(state_name, 0)) + 1
        reason = str(status_report.get("reason") or "<empty>")
        if not active_sensing:
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

        if status_report.get("accepted") is True and not active_sensing:
            lifecycle["accepted_status_count"] = int(
                lifecycle.get("accepted_status_count", 0)
            ) + 1
            if lifecycle.get("first_accepted_status") is None:
                lifecycle["first_accepted_status"] = status_report
            lifecycle["last_accepted_status"] = status_report
            identity = status_report.get("identity")
            identities = lifecycle.get("accepted_trajectory_identities")
            if not isinstance(identity, dict) or not isinstance(identities, list):
                raise RuntimeError("SCAN controller accepted identity 证据非法。")
            if identity not in identities:
                if len(identities) >= 128:
                    raise RuntimeError("SCAN controller 单 episode 接受轨迹超过 128 代。")
                identities.append(identity)
            accepted_reports = lifecycle.get("accepted_status_reports")
            if not isinstance(accepted_reports, list):
                raise RuntimeError(
                    "SCAN controller accepted_status_reports 证据非法。"
                )
            accepted_reports.append(status_report)
            if len(accepted_reports) > 128:
                accepted_reports.pop(0)
                lifecycle["dropped_accepted_status_report_count"] = int(
                    lifecycle.get(
                        "dropped_accepted_status_report_count", 0
                    )
                ) + 1
            lifecycle["distinct_accepted_trajectory_count"] = len(identities)
            lifecycle["trajectory_replacement_count"] = max(
                0,
                len(identities) - 1,
            )
        if status_report.get("trajectory_valid") is True and not active_sensing:
            lifecycle["trajectory_valid_status_count"] = int(
                lifecycle.get("trajectory_valid_status_count", 0)
            ) + 1
        if event == 2 and not active_sensing:
            lifecycle["candidate_rejection_count"] = int(
                lifecycle.get("candidate_rejection_count", 0)
            ) + 1
            if lifecycle.get("first_candidate_rejection_status") is None:
                lifecycle["first_candidate_rejection_status"] = status_report
            lifecycle["last_candidate_rejection_status"] = status_report
            if self._is_goal_latched_same_path_candidate_rejection(
                status_report
            ):
                key = "goal_latched_same_path_candidate_rejection_count"
            else:
                key = "unexpected_candidate_rejection_count"
            lifecycle[key] = int(lifecycle.get(key, 0)) + 1
        if not active_sensing and (
            status_report.get("emergency_stop") is True or state == 8
        ):
            lifecycle["emergency_stop_status_count"] = int(
                lifecycle.get("emergency_stop_status_count", 0)
            ) + 1
            if lifecycle.get("first_emergency_stop_status") is None:
                lifecycle["first_emergency_stop_status"] = status_report
            lifecycle["last_emergency_stop_status"] = status_report
            lifecycle["emergency_stop_pending_recovery"] = True
        if state == 10 and not active_sensing:
            lifecycle["tracking_status_count"] = int(
                lifecycle.get("tracking_status_count", 0)
            ) + 1
            if lifecycle.get("first_tracking_status") is None:
                lifecycle["first_tracking_status"] = status_report
            lifecycle["last_tracking_status"] = status_report
            tracking_reports = lifecycle.get("tracking_status_reports")
            if not isinstance(tracking_reports, list):
                raise RuntimeError(
                    "SCAN controller tracking_status_reports 证据非法。"
                )
            tracking_reports.append(status_report)
            if len(tracking_reports) > 128:
                tracking_reports.pop(0)
                lifecycle["dropped_tracking_status_report_count"] = int(
                    lifecycle.get(
                        "dropped_tracking_status_report_count", 0
                    )
                ) + 1
            if lifecycle.get("emergency_stop_pending_recovery") is True:
                lifecycle["tracking_after_emergency_stop_observed"] = True
                lifecycle["emergency_stop_recovery_count"] = int(
                    lifecycle.get("emergency_stop_recovery_count", 0)
                ) + 1
                lifecycle["emergency_stop_pending_recovery"] = False
        if state == 12 and not active_sensing:
            lifecycle["goal_reached_status_count"] = int(
                lifecycle.get("goal_reached_status_count", 0)
            ) + 1
        lifecycle["last_status"] = status_report
        self._metadata[
            "scan_controller_status_lifecycle_report"
        ] = lifecycle
        self._update_active_sensing_from_controller_status(status_report)
        return lifecycle

    @staticmethod
    def _is_goal_latched_same_path_candidate_rejection(
        status_report: dict[str, Any],
    ) -> bool:
        """识别到达锁存后拒绝同一 Path 代迟到轨迹的安全事件。"""

        if not (
            int(status_report.get("event", -1)) == 2
            and int(status_report.get("state", -1)) == 12
            and status_report.get("accepted") is True
            and status_report.get("trajectory_valid") is True
            and status_report.get("is_final") is True
            and status_report.get("emergency_stop") is False
        ):
            return False
        identity = status_report.get("identity")
        candidate = status_report.get("candidate")
        if not isinstance(identity, dict) or not isinstance(candidate, dict):
            return False
        try:
            active_path_stamp_ns = int(identity["reference_path_stamp_ns"])
            candidate_path_stamp_ns = int(
                candidate["reference_path_stamp_ns"]
            )
            active_traj_id = int(identity["traj_id"])
            candidate_traj_id = int(candidate["traj_id"])
            active_bspline_stamp_ns = int(identity["bspline_header_stamp_ns"])
            candidate_bspline_stamp_ns = int(
                candidate["bspline_header_stamp_ns"]
            )
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            active_path_stamp_ns > 0
            and candidate_path_stamp_ns == active_path_stamp_ns
            and candidate_traj_id > active_traj_id > 0
            and candidate_bspline_stamp_ns > active_bspline_stamp_ns > 0
        )

    @staticmethod
    def _new_grid_map_observation_lifecycle_report(
        *,
        ros_time_offset_s: float = 0.0,
    ) -> dict[str, Any]:
        """建立过滤后点云、显式 miss 与动态障碍关联的单 episode 证据。"""

        return {
            "schema": "grid_map_observation_lifecycle_v1",
            "ros_time_offset_s": float(ros_time_offset_s),
            "sample_count": 0,
            "first_observation_sequence": None,
            "last_observation_sequence": None,
            "sequence_reset_count": 0,
            "canonical_empty_count": 0,
            "map_fusion_count": 0,
            "total_input_point_count": 0,
            "total_accepted_endpoint_count": 0,
            "total_hit_endpoint_count": 0,
            "total_explicit_free_endpoint_count": 0,
            "total_free_to_occupied_transition_count": 0,
            "total_explicit_free_miss_voxel_count": 0,
            "total_occupied_to_free_by_explicit_miss_count": 0,
            "total_occupied_removed_by_sliding_reset_count": 0,
            "dynamic_obstacle_hit_match_count": 0,
            "dynamic_obstacle_transition_hit_match_count": 0,
            "dynamic_obstacle_explicit_miss_clear_match_count": 0,
            "first_report": None,
            "last_report": None,
            "first_hit_report": None,
            "last_hit_report": None,
            "first_transition_hit_report": None,
            "last_transition_hit_report": None,
            "first_explicit_miss_clear_report": None,
            "last_explicit_miss_clear_report": None,
            "hit_reports": [],
            "transition_hit_reports": [],
            "diagnostic_reports": [],
            "dropped_diagnostic_report_count": 0,
        }

    @staticmethod
    def _new_bspline_diagnostics_lifecycle_report(
        *,
        ros_time_offset_s: float = 0.0,
    ) -> dict[str, Any]:
        """建立 B-spline identity、ordered corridor 与几何的单 episode 证据。"""

        return {
            "schema": "bspline_diagnostics_lifecycle_v1",
            "ros_time_offset_s": float(ros_time_offset_s),
            "sample_count": 0,
            "active_sensing_diagnostic_count": 0,
            "first_diagnostic_sequence": None,
            "last_diagnostic_sequence": None,
            "sequence_reset_count": 0,
            "distinct_trajectory_identity_count": 0,
            "trajectory_identities": [],
            "ordered_reference_checked_count": 0,
            "ordered_reference_safe_count": 0,
            "dynamic_obstacle_relevant_count": 0,
            "ordered_detour_candidate_count": 0,
            "first_report": None,
            "last_report": None,
            "diagnostic_reports": [],
            "dropped_diagnostic_report_count": 0,
        }

    @staticmethod
    def _new_active_sensing_lifecycle_report() -> dict[str, Any]:
        """建立主动感知从规划到恢复的单 episode typed 证据。"""

        return {
            "schema": "active_sensing_lifecycle_v1",
            "attempt_count": 0,
            "completed_attempt_count": 0,
            "failed_attempt_count": 0,
            "active_attempt_identity": None,
            "attempts": [],
            # planner/controller/GridMap 来自独立 topic，receipt 顺序
            # 不是源端因果顺序。pending 只在完整 identity 上回填，
            # 超界直接失败关闭，不淘汰未归属证据。
            "pending_active_controller_statuses": [],
            "pending_recovery_controller_statuses": [],
            "pending_active_policy_writes": [],
            # ControllerStatus 可能已经把首拍严格零 Twist 覆盖
            # 在 KeepLast(1) 中。runtime 对每个新 active identity
            # 强制一次真实 policy 零写入，再放行后续 yaw。
            "policy_zero_gate": None,
            "policy_zero_gate_armed_count": 0,
            "policy_zero_gate_consumed_count": 0,
            "policy_zero_gated_identities": [],
        }

    @staticmethod
    def _active_sensing_identity_key(
        identity: object,
    ) -> tuple[int, int, int, int] | None:
        """把完整 Path/B-spline identity 规范化为可比较键。"""

        if not isinstance(identity, dict):
            return None
        values = tuple(
            identity.get(field_name)
            for field_name in (
                "reference_path_stamp_ns",
                "bspline_header_stamp_ns",
                "start_time_ns",
                "traj_id",
            )
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in values
        ):
            return None
        return tuple(int(value) for value in values)  # type: ignore[return-value]

    @classmethod
    def _find_active_sensing_attempt(
        cls,
        lifecycle: dict[str, Any],
        identity: object,
    ) -> dict[str, Any] | None:
        """按完整 identity 查找主动感知尝试，不接受仅 traj_id 匹配。"""

        identity_key = cls._active_sensing_identity_key(identity)
        attempts = lifecycle.get("attempts")
        if identity_key is None or not isinstance(attempts, list):
            return None
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise RuntimeError("active sensing attempt 不是对象。")
            if (
                cls._active_sensing_identity_key(attempt.get("identity"))
                == identity_key
            ):
                return attempt
        return None

    @staticmethod
    def _new_active_sensing_attempt(
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """从 STARTED 快照建立一条主动感知尝试。"""

        return {
            "identity": copy.deepcopy(report["identity"]),
            "events": [],
            "event_reports": [],
            "started": None,
            "accepted": None,
            "yaw_stable": None,
            "completed": None,
            "failed": None,
            "policy_window_open": True,
            "planner_fusion": {
                "baseline": 0,
                "current": 0,
                "distinct": 0,
                "required": 3,
            },
            "post_settle_fused_observations": [],
            "controller_command_aggregate": {
                "sample_count": 0,
                "first_command": [0.0] * 6,
                "max_abs_vx": 0.0,
                "max_abs_vy": 0.0,
                "max_abs_wz": 0.0,
                "violation_count": 0,
                "first_status": None,
                "last_status": None,
            },
            "policy_command_aggregate": {
                "sample_count": 0,
                "first_command": None,
                "max_abs_vx": 0.0,
                "max_abs_vy": 0.0,
                "max_abs_wz": 0.0,
                "violation_count": 0,
                "first_write": None,
                # first/last 只能证明窗口边界；末拍可能因输入超时安全归零。
                # 首条非零旋转证明 settle 前的实际执行，最大角速度写入则
                # 约束聚合包络不能凭空声明。
                "first_rotation_write": None,
                "maximum_abs_wz_write": None,
                "last_write": None,
            },
            "pct_plan_ids": [],
            "recovery": {
                "identity": None,
                "reference_path_stamp_ns": None,
                "pct_plan_id": None,
                "stationary": None,
                "controller_state": None,
            },
        }

    def _active_sensing_lifecycle(self) -> dict[str, Any]:
        """返回当前 episode 主动感知聚合，缺失时创建空报告。"""

        lifecycle = self._metadata.get("active_sensing_lifecycle_report")
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("schema") != "active_sensing_lifecycle_v1"
        ):
            lifecycle = self._new_active_sensing_lifecycle_report()
            self._metadata["active_sensing_lifecycle_report"] = lifecycle
        return lifecycle

    @staticmethod
    def _append_active_sensing_pending(
        lifecycle: dict[str, Any],
        *,
        field_name: str,
        value: dict[str, Any],
        limit: int,
        label: str,
    ) -> None:
        """有界保留一条乱序证据；不允许静默丢失旧样本。"""

        pending = lifecycle.get(field_name)
        if not isinstance(pending, list):
            raise RuntimeError(f"{label} pending 字段不是数组。")
        if len(pending) >= limit:
            raise RuntimeError(f"{label} pending 超过有界容量 {limit}。")
        pending.append(copy.deepcopy(value))

    def _arm_active_sensing_policy_zero_gate(
        self,
        lifecycle: dict[str, Any],
        status_report: dict[str, Any],
    ) -> None:
        """首次观测新 active controller identity 时布署一次零写入。"""

        if (
            status_report.get("active_sensing_yaw_only") is not True
            or status_report.get("accepted") is not True
            or status_report.get("trajectory_valid") is not True
        ):
            return
        identity = status_report.get("identity")
        identity_key = self._active_sensing_identity_key(identity)
        if identity_key is None:
            raise RuntimeError("active controller 缺少完整 identity。")
        attempt = self._find_active_sensing_attempt(lifecycle, identity)
        if isinstance(attempt, dict) and (
            attempt.get("completed") is not None
            or attempt.get("failed") is not None
            or attempt.get("policy_window_open") is not True
        ):
            return
        gated_identities = lifecycle.get("policy_zero_gated_identities")
        if not isinstance(gated_identities, list):
            raise RuntimeError("active policy zero gate identity 证据不是数组。")
        if any(
            self._active_sensing_identity_key(retained) == identity_key
            for retained in gated_identities
        ):
            return
        current_gate = lifecycle.get("policy_zero_gate")
        if current_gate is not None:
            if not isinstance(current_gate, dict):
                raise RuntimeError("active policy zero gate 证据不是对象。")
            if (
                self._active_sensing_identity_key(
                    current_gate.get("identity")
                )
                == identity_key
            ):
                return
            raise RuntimeError("同时观测到两个未清零的 active identity。")
        lifecycle["policy_zero_gate"] = {
            "identity": copy.deepcopy(identity),
            "status_sequence": int(status_report["status_sequence"]),
            "receipt_timestamp": float(status_report["receipt_timestamp"]),
        }
        lifecycle["policy_zero_gate_armed_count"] = int(
            lifecycle.get("policy_zero_gate_armed_count", 0)
        ) + 1

    def _close_active_sensing_policy_window(
        self,
        lifecycle: dict[str, Any],
        attempt: dict[str, Any],
    ) -> None:
        """终态立即关闭 policy 归属窗口，不等 controller topic 追平。"""

        attempt["policy_window_open"] = False
        identity_key = self._active_sensing_identity_key(
            attempt.get("identity")
        )
        gate = lifecycle.get("policy_zero_gate")
        if (
            isinstance(gate, dict)
            and self._active_sensing_identity_key(gate.get("identity"))
            == identity_key
        ):
            lifecycle["policy_zero_gate"] = None

    def _consume_active_sensing_policy_zero_gate(
        self,
    ) -> dict[str, Any] | None:
        """取出当前零写入门；实际写零成功后由调用方确认。"""

        metadata = getattr(self, "_metadata", None)
        if not isinstance(metadata, dict):
            return None
        lifecycle = metadata.get("active_sensing_lifecycle_report")
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("schema") != "active_sensing_lifecycle_v1"
        ):
            return None
        gate = lifecycle.get("policy_zero_gate")
        if gate is None:
            return None
        if not isinstance(gate, dict) or self._active_sensing_identity_key(
            gate.get("identity")
        ) is None:
            raise RuntimeError("active policy zero gate 缺少完整 identity。")
        return copy.deepcopy(gate)

    def _confirm_active_sensing_policy_zero_gate(
        self,
        gate: dict[str, Any],
    ) -> None:
        """在唯一 policy owner 真实写零并完成 Twist drain 后收口门禁。"""

        lifecycle = self._active_sensing_lifecycle()
        current = lifecycle.get("policy_zero_gate")
        identity_key = self._active_sensing_identity_key(gate.get("identity"))
        if (
            not isinstance(current, dict)
            or self._active_sensing_identity_key(current.get("identity"))
            != identity_key
        ):
            raise RuntimeError("active policy zero gate 在写零期间发生变化。")
        gated_identities = lifecycle.get("policy_zero_gated_identities")
        if not isinstance(gated_identities, list):
            raise RuntimeError("active policy zero gate identity 证据不是数组。")
        if len(gated_identities) >= ACTIVE_SENSING_ATTEMPT_LIMIT:
            raise RuntimeError("active policy zero gate identity 超过单 episode 上限。")
        gated_identities.append(copy.deepcopy(gate["identity"]))
        lifecycle["policy_zero_gate"] = None
        lifecycle["policy_zero_gate_consumed_count"] = int(
            lifecycle.get("policy_zero_gate_consumed_count", 0)
        ) + 1

    @staticmethod
    def _new_dynamic_navigation_evidence_report(
        plan: DynamicObstaclePlan,
        *,
        ros_time_offset_s: float = 0.0,
    ) -> dict[str, Any]:
        """建立动态导航五项 typed 证据；任一叶子缺失都保持未验证。"""

        empty_leaf = {"verified": False, "reason": "evidence_not_observed"}
        return {
            "schema": "dynamic_navigation_evidence_v1",
            "enabled": bool(plan.enabled),
            "ros_time_offset_s": float(ros_time_offset_s),
            "verified": False,
            "obstacle_ids": [
                obstacle.obstacle_id for obstacle in plan.obstacles
            ],
            "post_filter_hit": dict(empty_leaf),
            "ordered_detour": dict(empty_leaf),
            "current_obstacle_clearance": dict(empty_leaf),
            "explicit_miss_ghost_clear": dict(empty_leaf),
            "trajectory_recovery": dict(empty_leaf),
        }

    @staticmethod
    def _bspline_diagnostics_identity(
        sample: OgnBsplineDiagnosticsSample,
    ) -> dict[str, Any]:
        """生成与 ControllerStatus 完全同构的 B-spline identity。"""

        return {
            "reference_path_stamp": {
                "sec": int(sample.reference_path_stamp_sec),
                "nanosec": int(sample.reference_path_stamp_nanosec),
            },
            "reference_path_stamp_ns": int(sample.reference_path_stamp_ns),
            "bspline_header_stamp": {
                "sec": int(sample.header_stamp_sec),
                "nanosec": int(sample.header_stamp_nanosec),
            },
            "bspline_header_stamp_ns": int(sample.header_stamp_ns),
            "start_time": {
                "sec": int(sample.start_time_sec),
                "nanosec": int(sample.start_time_nanosec),
            },
            "start_time_ns": int(sample.start_time_ns),
            "traj_id": int(sample.traj_id),
        }

    def _collect_active_sensing_fusion_evidence(
        self,
        attempt: dict[str, Any],
        *,
        receipt_cutoff: float | None = None,
        source_stamp_cutoff_ns: int | None = None,
    ) -> None:
        """收集稳定窗后的三条不同采集时刻 GridMap 真实融合证据。"""

        yaw_stable = attempt.get("yaw_stable")
        if not isinstance(yaw_stable, dict):
            return
        active = yaw_stable.get("active_sensing")
        if not isinstance(active, dict):
            raise RuntimeError("YAW_STABLE 快照缺少 active_sensing 对象。")
        settle_stamp_ns = active.get("settle_stamp_ns")
        if (
            not isinstance(settle_stamp_ns, int)
            or isinstance(settle_stamp_ns, bool)
            or settle_stamp_ns <= 0
        ):
            raise RuntimeError("YAW_STABLE 快照缺少非零 settle_stamp_ns。")

        grid_lifecycle = self._metadata.get(
            "grid_map_observation_lifecycle_report"
        )
        grid_reports = (
            grid_lifecycle.get("diagnostic_reports")
            if isinstance(grid_lifecycle, dict)
            else None
        )
        if not isinstance(grid_reports, list):
            return
        evidence = attempt.get("post_settle_fused_observations")
        if not isinstance(evidence, list):
            raise RuntimeError("主动感知 fused observation 证据不是数组。")
        retained_stamps = {
            int(item["header_stamp_ns"])
            for item in evidence
            if isinstance(item, dict)
            and isinstance(item.get("header_stamp_ns"), int)
        }
        for grid_report in grid_reports:
            if len(evidence) >= 3 or not isinstance(grid_report, dict):
                break
            header = grid_report.get("header")
            header_stamp_ns = (
                header.get("stamp_ns") if isinstance(header, dict) else None
            )
            receipt_timestamp = grid_report.get("receipt_timestamp")
            if (
                grid_report.get("map_fusion_performed") is not True
                or int(grid_report.get("accepted_endpoint_count", 0)) <= 0
                or not isinstance(header_stamp_ns, int)
                or isinstance(header_stamp_ns, bool)
                or header_stamp_ns <= settle_stamp_ns
                or (
                    source_stamp_cutoff_ns is not None
                    and header_stamp_ns > source_stamp_cutoff_ns + 1
                )
                or header_stamp_ns in retained_stamps
                or (
                    receipt_cutoff is not None
                    and isinstance(receipt_timestamp, (int, float))
                    and float(receipt_timestamp) > receipt_cutoff + 1.0e-9
                )
            ):
                continue
            evidence.append(
                {
                    "header_stamp_ns": int(header_stamp_ns),
                    "map_fusion_performed": True,
                    "accepted_endpoint_count": int(
                        grid_report["accepted_endpoint_count"]
                    ),
                    "observation_sequence": int(
                        grid_report["observation_sequence"]
                    ),
                    "header": copy.deepcopy(header),
                }
            )
            retained_stamps.add(int(header_stamp_ns))

    def _update_active_sensing_from_bspline_report(
        self,
        report: dict[str, Any],
    ) -> None:
        """按完整 identity 严格聚合主动感知事件，并发现后续运动恢复。"""

        lifecycle = self._active_sensing_lifecycle()
        active = report.get("active_sensing")
        if not isinstance(active, dict):
            raise RuntimeError("B-spline report 缺少 active_sensing 对象。")
        if active.get("enabled") is not True:
            if (
                report.get("stationary") is True
                or report.get("emergency_stop") is True
            ):
                return
            for attempt in lifecycle["attempts"]:
                if not isinstance(attempt, dict) or attempt.get("completed") is None:
                    continue
                recovery = attempt.get("recovery")
                identity = attempt.get("identity")
                if (
                    not isinstance(recovery, dict)
                    or recovery.get("identity") is not None
                    or not isinstance(identity, dict)
                    or report.get("identity") == identity
                    or report["identity"].get("reference_path_stamp_ns")
                    != identity.get("reference_path_stamp_ns")
                    or int(report["identity"]["bspline_header_stamp_ns"])
                    <= int(identity["bspline_header_stamp_ns"])
                    or int(report["identity"]["start_time_ns"])
                    <= int(identity["start_time_ns"])
                ):
                    continue
                recovery.update(
                    {
                        "identity": copy.deepcopy(report["identity"]),
                        "reference_path_stamp_ns": int(
                            report["identity"]["reference_path_stamp_ns"]
                        ),
                        # 只能由后续普通 controller identity 生效期间的
                        # NavigationStatus + policy 实写共同补齐，不能沿用
                        # active 窗口的 plan id 冒充恢复证据。
                        "pct_plan_id": None,
                        "stationary": False,
                    }
                )
                self._backfill_active_sensing_recovery_controller_status(
                    lifecycle,
                    attempt,
                )
            return

        event_names = {
            1: "STARTED",
            2: "ACCEPTED",
            3: "YAW_STABLE",
            4: "FUSION_PROGRESS",
            5: "COMPLETED",
            6: "FAILED",
        }
        event = int(active["event"])
        event_name = event_names.get(event)
        if event_name is None:
            raise RuntimeError("主动感知 report 携带未知事件。")
        identity = report.get("identity")
        attempt = self._find_active_sensing_attempt(lifecycle, identity)
        if event == 1:
            if attempt is not None:
                raise RuntimeError("同一主动感知 identity 重复 STARTED。")
            open_identity = lifecycle.get("active_attempt_identity")
            if open_identity is not None:
                raise RuntimeError(
                    "上一主动感知尝试未终止，拒绝新 STARTED。"
                )
            attempts = lifecycle.get("attempts")
            if (
                not isinstance(attempts, list)
                or len(attempts) >= ACTIVE_SENSING_ATTEMPT_LIMIT
            ):
                raise RuntimeError("单 episode 主动感知尝试超过 32 次。")
            attempt = self._new_active_sensing_attempt(report)
            attempts.append(attempt)
            lifecycle["attempt_count"] = len(attempts)
            lifecycle["active_attempt_identity"] = copy.deepcopy(identity)
            self._backfill_active_sensing_controller_statuses(
                lifecycle,
                attempt,
            )
        elif attempt is None:
            raise RuntimeError("主动感知事件缺少同 identity STARTED。")
        if attempt.get("completed") is not None or attempt.get("failed") is not None:
            raise RuntimeError(
                "主动感知终态之后仍收到同 identity 事件。"
            )

        events = attempt.get("events")
        event_reports = attempt.get("event_reports")
        if not isinstance(events, list) or not isinstance(event_reports, list):
            raise RuntimeError("主动感知事件证据不是数组。")
        previous_event = events[-1] if events else None
        allowed_previous = {
            "STARTED": {None},
            "ACCEPTED": {"STARTED"},
            "YAW_STABLE": {"ACCEPTED"},
            "FUSION_PROGRESS": {"YAW_STABLE", "FUSION_PROGRESS"},
            "COMPLETED": {"FUSION_PROGRESS"},
            "FAILED": {
                "STARTED",
                "ACCEPTED",
                "YAW_STABLE",
                "FUSION_PROGRESS",
            },
        }
        if previous_event not in allowed_previous[event_name]:
            raise RuntimeError(
                "主动感知事件顺序非法："
                f"{previous_event!r} -> {event_name}。"
            )
        events.append(event_name)
        event_reports.append(copy.deepcopy(report))
        snapshot_fields = {
            "STARTED": "started",
            "ACCEPTED": "accepted",
            "YAW_STABLE": "yaw_stable",
            "COMPLETED": "completed",
            "FAILED": "failed",
        }
        snapshot_field = snapshot_fields.get(event_name)
        if snapshot_field is not None:
            attempt[snapshot_field] = copy.deepcopy(report)
        if event_name == "ACCEPTED":
            self._backfill_active_sensing_policy_writes(
                lifecycle,
                attempt,
            )
        planner_fusion = attempt.get("planner_fusion")
        if not isinstance(planner_fusion, dict):
            raise RuntimeError("主动感知 planner_fusion 不是对象。")
        planner_fusion.update(
            {
                "baseline": int(active["fusion_baseline"]),
                "current": int(active["fusion_current"]),
                "distinct": int(active["fusion_distinct"]),
                "required": int(active["fusion_required"]),
            }
        )
        report_receipt_timestamp = float(report["receipt_timestamp"])
        self._collect_active_sensing_fusion_evidence(
            attempt,
            receipt_cutoff=report_receipt_timestamp,
            source_stamp_cutoff_ns=(
                int(round(report_receipt_timestamp * 1.0e9))
                if event_name == "COMPLETED"
                else None
            ),
        )
        if event_name == "COMPLETED":
            self._close_active_sensing_policy_window(lifecycle, attempt)
            lifecycle["completed_attempt_count"] = int(
                lifecycle.get("completed_attempt_count", 0)
            ) + 1
            lifecycle["active_attempt_identity"] = None
        elif event_name == "FAILED":
            self._close_active_sensing_policy_window(lifecycle, attempt)
            lifecycle["failed_attempt_count"] = int(
                lifecycle.get("failed_attempt_count", 0)
            ) + 1
            lifecycle["active_attempt_identity"] = None

    def _refresh_active_sensing_fusion_evidence(self) -> None:
        """在新 GridMap 诊断到达后更新未失败的主动感知窗口。"""

        lifecycle = self._active_sensing_lifecycle()
        for attempt in lifecycle["attempts"]:
            if not isinstance(attempt, dict):
                raise RuntimeError("active sensing attempt 不是对象。")
            # COMPLETED 诊断与三条 GridMap 诊断分属独立
            # topic；终态先到时仍允许回填已完成 attempt，但只接受
            # COMPLETED 接收时刻之前已采集的 source frame，不能让
            # 终态后的未来点云倒填成功窗口。
            # FAILED 尝试没有可认证的感知窗口，不予回填。
            if attempt.get("failed") is None:
                completed = attempt.get("completed")
                source_stamp_cutoff_ns: int | None = None
                if isinstance(completed, dict):
                    completed_receipt = completed.get("receipt_timestamp")
                    if (
                        not isinstance(completed_receipt, (int, float))
                        or isinstance(completed_receipt, bool)
                        or not math.isfinite(float(completed_receipt))
                    ):
                        raise RuntimeError(
                            "COMPLETED 主动感知快照缺少有限 receipt timestamp。"
                        )
                    source_stamp_cutoff_ns = int(
                        round(float(completed_receipt) * 1.0e9)
                    )
                self._collect_active_sensing_fusion_evidence(
                    attempt,
                    source_stamp_cutoff_ns=source_stamp_cutoff_ns,
                )

    @staticmethod
    def _merge_active_sensing_controller_status(
        attempt: dict[str, Any],
        status_report: dict[str, Any],
    ) -> None:
        """把一条同 identity controller 累计快照并入 attempt。"""

        aggregate = attempt.get("controller_command_aggregate")
        incoming = status_report.get("command_aggregate")
        if not isinstance(aggregate, dict) or not isinstance(incoming, dict):
            raise RuntimeError("active controller 命令聚合不是对象。")
        previous_count = int(aggregate["sample_count"])
        incoming_count = int(incoming["sample_count"])
        if incoming_count < previous_count:
            raise RuntimeError("active controller command_sample_count 发生回退。")
        for field_name in ("max_abs_vx", "max_abs_vy", "max_abs_wz"):
            if float(incoming[field_name]) + 1.0e-12 < float(aggregate[field_name]):
                raise RuntimeError(
                    f"active controller {field_name} 发生回退。"
                )
        if int(incoming["violation_count"]) < int(aggregate["violation_count"]):
            raise RuntimeError("active controller violation_count 发生回退。")
        if previous_count > 0 and incoming["first_command"] != aggregate[
            "first_command"
        ]:
            raise RuntimeError(
                "active controller first_command 在同 identity 内变化。"
            )
        aggregate.update(copy.deepcopy(incoming))
        if aggregate.get("first_status") is None:
            aggregate["first_status"] = copy.deepcopy(status_report)
        aggregate["last_status"] = copy.deepcopy(status_report)

    def _backfill_active_sensing_controller_statuses(
        self,
        lifecycle: dict[str, Any],
        attempt: dict[str, Any],
    ) -> None:
        """STARTED 到达后按原接收顺序回填同 identity controller。"""

        pending = lifecycle.get("pending_active_controller_statuses")
        if not isinstance(pending, list):
            raise RuntimeError("active controller pending 字段不是数组。")
        identity_key = self._active_sensing_identity_key(
            attempt.get("identity")
        )
        remaining: list[dict[str, Any]] = []
        for status_report in pending:
            if not isinstance(status_report, dict):
                raise RuntimeError("active controller pending 样本不是对象。")
            if (
                self._active_sensing_identity_key(
                    status_report.get("identity")
                )
                != identity_key
            ):
                remaining.append(status_report)
                continue
            self._merge_active_sensing_controller_status(
                attempt,
                status_report,
            )
        lifecycle["pending_active_controller_statuses"] = remaining

    def _retain_pending_recovery_controller_status(
        self,
        lifecycle: dict[str, Any],
        status_report: dict[str, Any],
    ) -> None:
        """在普通 B-spline 诊断落后时有界保留 recovery controller。"""

        attempts = lifecycle.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            return
        if not any(
            isinstance(attempt, dict)
            and attempt.get("started") is not None
            and attempt.get("failed") is None
            for attempt in attempts
        ):
            return
        identity_key = self._active_sensing_identity_key(
            status_report.get("identity")
        )
        if identity_key is None:
            raise RuntimeError("recovery controller 缺少完整 identity。")
        pending = lifecycle.get("pending_recovery_controller_statuses")
        if not isinstance(pending, list):
            raise RuntimeError("recovery controller pending 字段不是数组。")
        # ControllerStatus 是同 identity 的累计快照，只保留该
        # identity 最新状态即可完整回填，也避免 50 Hz 占满环。
        for index, retained in enumerate(pending):
            if not isinstance(retained, dict):
                raise RuntimeError("recovery controller pending 样本不是对象。")
            if (
                self._active_sensing_identity_key(retained.get("identity"))
                == identity_key
            ):
                pending[index] = copy.deepcopy(status_report)
                return
        self._append_active_sensing_pending(
            lifecycle,
            field_name="pending_recovery_controller_statuses",
            value=status_report,
            limit=ACTIVE_SENSING_PENDING_CONTROLLER_STATUS_LIMIT,
            label="recovery controller",
        )

    def _backfill_active_sensing_recovery_controller_status(
        self,
        lifecycle: dict[str, Any],
        attempt: dict[str, Any],
    ) -> None:
        """普通恢复诊断建档后回填已先到的 controller 状态。"""

        recovery = attempt.get("recovery")
        if not isinstance(recovery, dict):
            raise RuntimeError("active sensing recovery 不是对象。")
        identity_key = self._active_sensing_identity_key(
            recovery.get("identity")
        )
        if identity_key is None:
            return
        pending = lifecycle.get("pending_recovery_controller_statuses")
        if not isinstance(pending, list):
            raise RuntimeError("recovery controller pending 字段不是数组。")
        matching: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for status_report in pending:
            if not isinstance(status_report, dict):
                raise RuntimeError("recovery controller pending 样本不是对象。")
            if (
                self._active_sensing_identity_key(
                    status_report.get("identity")
                )
                == identity_key
            ):
                matching.append(status_report)
            else:
                remaining.append(status_report)
        if matching:
            latest = max(
                matching,
                key=lambda report: int(report["status_sequence"]),
            )
            recovery["controller_state"] = int(latest["state"])
        lifecycle["pending_recovery_controller_statuses"] = remaining

    def _update_active_sensing_from_controller_status(
        self,
        status_report: dict[str, Any],
    ) -> None:
        """保存 controller 实际 Twist 聚合，并补齐恢复轨迹 controller 状态。"""

        lifecycle = self._active_sensing_lifecycle()
        if status_report.get("active_sensing_yaw_only") is not True:
            identity_key = self._active_sensing_identity_key(
                status_report.get("identity")
            )
            matched = False
            for attempt in lifecycle["attempts"]:
                recovery = (
                    attempt.get("recovery")
                    if isinstance(attempt, dict)
                    else None
                )
                if (
                    isinstance(recovery, dict)
                    and self._active_sensing_identity_key(recovery.get("identity"))
                    == identity_key
                ):
                    recovery["controller_state"] = int(status_report["state"])
                    matched = True
            if not matched:
                self._retain_pending_recovery_controller_status(
                    lifecycle,
                    status_report,
                )
            return
        self._arm_active_sensing_policy_zero_gate(lifecycle, status_report)
        attempt = self._find_active_sensing_attempt(
            lifecycle,
            status_report.get("identity"),
        )
        if attempt is None:
            self._append_active_sensing_pending(
                lifecycle,
                field_name="pending_active_controller_statuses",
                value=status_report,
                limit=ACTIVE_SENSING_PENDING_CONTROLLER_STATUS_LIMIT,
                label="active controller",
            )
            return
        self._merge_active_sensing_controller_status(
            attempt,
            status_report,
        )

    @staticmethod
    def _active_sensing_policy_write_with_snapshot(
        write_report: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """冻结实写报告与当拍 controller identity，并校验三轴有限。"""

        command = write_report.get("written_command")
        if (
            not isinstance(command, list)
            or len(command) != 3
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in command
            )
        ):
            raise RuntimeError("policy active-window written_command 非有限三轴命令。")
        evidence_write = copy.deepcopy(write_report)
        consumed_gate = evidence_write.get(
            "policy_navigation_gate_consumed_report"
        )
        if isinstance(consumed_gate, dict):
            # active evidence 与全局 identity-verified write 使用同一规范键；
            # 保留 raw 键并复制别名，validator 可证明两者完全一致。
            evidence_write["navigation_gate_diagnostics"] = copy.deepcopy(
                consumed_gate
            )
        evidence_write["scan_controller_status_snapshot"] = copy.deepcopy(
            snapshot
        )
        return evidence_write

    @staticmethod
    def _merge_active_sensing_policy_write(
        attempt: dict[str, Any],
        evidence_write: dict[str, Any],
    ) -> None:
        """把一次唯一 owner 的真实 policy 写入并入 active 窗口。"""

        command = evidence_write["written_command"]
        vx, vy, wz = (float(value) for value in command)
        aggregate = attempt.get("policy_command_aggregate")
        if not isinstance(aggregate, dict):
            raise RuntimeError("active policy_command_aggregate 不是对象。")
        last_write = aggregate.get("last_write")
        if isinstance(last_write, dict) and int(
            evidence_write["write_sequence"]
        ) <= int(last_write["write_sequence"]):
            raise RuntimeError("active policy write_sequence 未严格递增。")
        aggregate["sample_count"] = int(aggregate["sample_count"]) + 1
        if aggregate.get("first_command") is None:
            aggregate["first_command"] = [vx, vy, wz]
            aggregate["first_write"] = copy.deepcopy(evidence_write)
        previous_max_abs_wz = float(aggregate["max_abs_wz"])
        aggregate["max_abs_vx"] = max(float(aggregate["max_abs_vx"]), abs(vx))
        aggregate["max_abs_vy"] = max(float(aggregate["max_abs_vy"]), abs(vy))
        aggregate["max_abs_wz"] = max(previous_max_abs_wz, abs(wz))
        if (
            evidence_write.get("motion_allowed") is True
            and abs(vx) <= 1.0e-12
            and abs(vy) <= 1.0e-12
            and abs(wz) > 1.0e-12
        ):
            if aggregate.get("first_rotation_write") is None:
                aggregate["first_rotation_write"] = copy.deepcopy(
                    evidence_write
                )
            if abs(wz) > previous_max_abs_wz + 1.0e-12:
                aggregate["maximum_abs_wz_write"] = copy.deepcopy(
                    evidence_write
                )
        if abs(vx) > 1.0e-12 or abs(vy) > 1.0e-12 or abs(wz) > 0.20 + 1.0e-12:
            aggregate["violation_count"] = int(aggregate["violation_count"]) + 1
        aggregate["last_write"] = evidence_write

        observed = evidence_write.get("navigation_status_observed_report")
        status = observed.get("status") if isinstance(observed, dict) else None
        if isinstance(status, dict):
            pct_plan_id = status.get("pct_plan_id")
            active_path_stamp_ns = status.get("active_path_stamp_ns")
            identity = attempt["identity"]
            if (
                isinstance(pct_plan_id, int)
                and not isinstance(pct_plan_id, bool)
                and pct_plan_id > 0
                and active_path_stamp_ns == identity["reference_path_stamp_ns"]
            ):
                pct_plan_ids = attempt.get("pct_plan_ids")
                if not isinstance(pct_plan_ids, list):
                    raise RuntimeError("active sensing pct_plan_ids 不是数组。")
                if pct_plan_id not in pct_plan_ids:
                    pct_plan_ids.append(int(pct_plan_id))

    def _retain_pending_active_sensing_policy_write(
        self,
        lifecycle: dict[str, Any],
        identity: dict[str, Any],
        evidence_write: dict[str, Any],
    ) -> None:
        """planner STARTED/ACCEPTED 落后时有界保留真实 policy 写入。"""

        pending = lifecycle.get("pending_active_policy_writes")
        if not isinstance(pending, list):
            raise RuntimeError("active policy pending 字段不是数组。")
        write_sequence = int(evidence_write["write_sequence"])
        if any(
            isinstance(retained, dict)
            and isinstance(retained.get("write"), dict)
            and int(retained["write"].get("write_sequence", -1))
            == write_sequence
            for retained in pending
        ):
            raise RuntimeError("active policy pending 包含重复 write_sequence。")
        self._append_active_sensing_pending(
            lifecycle,
            field_name="pending_active_policy_writes",
            value={
                "identity": copy.deepcopy(identity),
                "write": copy.deepcopy(evidence_write),
            },
            limit=ACTIVE_SENSING_PENDING_POLICY_WRITE_LIMIT,
            label="active policy write",
        )

    def _backfill_active_sensing_policy_writes(
        self,
        lifecycle: dict[str, Any],
        attempt: dict[str, Any],
    ) -> None:
        """planner ACCEPTED 到达后按真实 write_sequence 回填 policy。"""

        pending = lifecycle.get("pending_active_policy_writes")
        if not isinstance(pending, list):
            raise RuntimeError("active policy pending 字段不是数组。")
        identity_key = self._active_sensing_identity_key(
            attempt.get("identity")
        )
        matching: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for retained in pending:
            if not isinstance(retained, dict) or not isinstance(
                retained.get("write"), dict
            ):
                raise RuntimeError("active policy pending 样本非法。")
            if (
                self._active_sensing_identity_key(retained.get("identity"))
                == identity_key
            ):
                matching.append(retained["write"])
            else:
                remaining.append(retained)
        for evidence_write in sorted(
            matching,
            key=lambda item: int(item["write_sequence"]),
        ):
            self._merge_active_sensing_policy_write(
                attempt,
                evidence_write,
            )
        lifecycle["pending_active_policy_writes"] = remaining

    def _update_active_sensing_policy_write_evidence(
        self,
        write_report: dict[str, Any],
    ) -> None:
        """聚合 active controller 生效后每一次真正写入 policy 的命令。"""

        snapshot = self._metadata.get("scan_controller_status_last_report")
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("accepted") is not True
            or snapshot.get("trajectory_valid") is not True
        ):
            return
        lifecycle = self._active_sensing_lifecycle()
        observed = write_report.get("navigation_status_observed_report")
        status = observed.get("status") if isinstance(observed, dict) else None
        if snapshot.get("active_sensing_yaw_only") is not True:
            snapshot_key = self._active_sensing_identity_key(
                snapshot.get("identity")
            )
            for candidate in lifecycle["attempts"]:
                recovery = (
                    candidate.get("recovery")
                    if isinstance(candidate, dict)
                    else None
                )
                if (
                    not isinstance(recovery, dict)
                    or self._active_sensing_identity_key(
                        recovery.get("identity")
                    )
                    != snapshot_key
                ):
                    continue
                if isinstance(status, dict):
                    pct_plan_id = status.get("pct_plan_id")
                    if (
                        isinstance(pct_plan_id, int)
                        and not isinstance(pct_plan_id, bool)
                        and pct_plan_id > 0
                        and status.get("active_path_stamp_ns")
                        == recovery.get("reference_path_stamp_ns")
                    ):
                        recovery["pct_plan_id"] = int(pct_plan_id)
            return
        attempt = self._find_active_sensing_attempt(
            lifecycle,
            snapshot.get("identity"),
        )
        if isinstance(attempt, dict) and (
            attempt.get("completed") is not None
            or attempt.get("failed") is not None
            or attempt.get("policy_window_open") is not True
        ):
            # planner 终态比 controller 最后快照更权威。此后普通
            # recovery Twist 即使短暂伴随旧 active 快照，也不归入 active。
            return
        evidence_write = self._active_sensing_policy_write_with_snapshot(
            write_report,
            snapshot,
        )
        if attempt is None or attempt.get("accepted") is None:
            self._retain_pending_active_sensing_policy_write(
                lifecycle,
                snapshot["identity"],
                evidence_write,
            )
            return
        self._merge_active_sensing_policy_write(
            attempt,
            evidence_write,
        )

    def _dynamic_obstacle_episode_elapsed_at_ros_time(
        self,
        timestamp: float,
    ) -> float | None:
        """把连续 ROS 时钟转换为本 episode 时间；旧 episode 样本失败关闭。"""

        raw_timestamp = float(timestamp)
        offset = float(
            getattr(self, "_navigation_episode_ros_time_offset_s", 0.0)
        )
        if (
            not math.isfinite(raw_timestamp)
            or not math.isfinite(offset)
            or raw_timestamp < offset - 1.0e-9
        ):
            return None
        return max(0.0, raw_timestamp - offset)

    def _dynamic_obstacle_geometry_at(
        self,
        timestamp: float,
    ) -> tuple[tuple[Any, DynamicObstacleState], ...]:
        """返回 typed 诊断时间戳对应的动态障碍配置与确定性状态。"""

        plan = getattr(self, "_dynamic_obstacle_plan", DynamicObstaclePlan())
        if not plan.enabled:
            return ()
        elapsed_time_s = (
            self._dynamic_obstacle_episode_elapsed_at_ros_time(timestamp)
        )
        if elapsed_time_s is None:
            return ()
        states = {
            state.obstacle_id: state
            for state in plan.state_at(elapsed_time_s)
        }
        return tuple(
            (obstacle, states[obstacle.obstacle_id])
            for obstacle in plan.obstacles
        )

    @staticmethod
    def _dynamic_obstacle_point_geometry(
        point_world_xyz: tuple[float, float, float],
        obstacle: Any,
        state: DynamicObstacleState,
        *,
        tolerance_m: float = 0.0,
    ) -> tuple[bool, float]:
        """计算点是否落入旋转 cuboid，并返回到其 XY 边界的最短距离。"""

        delta_x = float(point_world_xyz[0]) - float(
            state.position_world_xyz[0]
        )
        delta_y = float(point_world_xyz[1]) - float(
            state.position_world_xyz[1]
        )
        delta_z = float(point_world_xyz[2]) - float(
            state.position_world_xyz[2]
        )
        cosine = math.cos(float(obstacle.yaw_rad))
        sine = math.sin(float(obstacle.yaw_rad))
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        half_x = 0.5 * float(obstacle.size_xyz_m[0])
        half_y = 0.5 * float(obstacle.size_xyz_m[1])
        half_z = 0.5 * float(obstacle.size_xyz_m[2])
        tolerance = float(tolerance_m)
        inside = bool(
            abs(local_x) <= half_x + tolerance
            and abs(local_y) <= half_y + tolerance
            and abs(delta_z) <= half_z + tolerance
        )
        clearance_xy = math.hypot(
            max(abs(local_x) - half_x, 0.0),
            max(abs(local_y) - half_y, 0.0),
        )
        return inside, clearance_xy

    @staticmethod
    def _dynamic_obstacle_motion_separation_report(
        obstacle_state_at_hit: object,
        obstacle_state_after_clear: object,
        *,
        map_resolution: object,
    ) -> dict[str, Any] | None:
        """核验障碍在 hit 与 clear 间确实移动了至少一个可测尺度。"""

        if not isinstance(obstacle_state_at_hit, dict) or not isinstance(
            obstacle_state_after_clear, dict
        ):
            return None
        hit_id = obstacle_state_at_hit.get("id")
        clear_id = obstacle_state_after_clear.get("id")
        if (
            not isinstance(hit_id, str)
            or not hit_id
            or clear_id != hit_id
        ):
            return None

        raw_hit_path_distance = obstacle_state_at_hit.get("path_distance_m")
        raw_clear_path_distance = obstacle_state_after_clear.get(
            "path_distance_m"
        )
        raw_hit_position = obstacle_state_at_hit.get("position_world_xyz")
        raw_clear_position = obstacle_state_after_clear.get(
            "position_world_xyz"
        )
        if (
            isinstance(map_resolution, bool)
            or not isinstance(map_resolution, (int, float))
            or isinstance(raw_hit_path_distance, bool)
            or not isinstance(raw_hit_path_distance, (int, float))
            or isinstance(raw_clear_path_distance, bool)
            or not isinstance(raw_clear_path_distance, (int, float))
            or not isinstance(raw_hit_position, (list, tuple))
            or len(raw_hit_position) != 3
            or not isinstance(raw_clear_position, (list, tuple))
            or len(raw_clear_position) != 3
        ):
            return None
        try:
            resolution = float(map_resolution)
            hit_path_distance = float(raw_hit_path_distance)
            clear_path_distance = float(raw_clear_path_distance)
            hit_position = tuple(float(value) for value in raw_hit_position)
            clear_position = tuple(
                float(value) for value in raw_clear_position
            )
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(resolution)
            or resolution <= 0.0
            or not math.isfinite(hit_path_distance)
            or not math.isfinite(clear_path_distance)
            or not all(math.isfinite(value) for value in hit_position)
            or not all(math.isfinite(value) for value in clear_position)
        ):
            return None

        path_distance_separation = abs(
            clear_path_distance - hit_path_distance
        )
        pose_position_separation = math.dist(
            hit_position,
            clear_position,
        )
        minimum_separation = max(
            resolution,
            DYNAMIC_OBSTACLE_POINT_ASSOCIATION_TOLERANCE_M,
        )
        separation_verified = bool(
            path_distance_separation
            + DYNAMIC_OBSTACLE_MOTION_SEPARATION_EPSILON_M
            >= minimum_separation
            and pose_position_separation
            + DYNAMIC_OBSTACLE_MOTION_SEPARATION_EPSILON_M
            >= minimum_separation
        )
        return {
            "obstacle_path_distance_separation_m": (
                path_distance_separation
            ),
            "obstacle_pose_position_separation_m": (
                pose_position_separation
            ),
            "minimum_obstacle_motion_separation_m": minimum_separation,
            "obstacle_motion_separation_verified": separation_verified,
        }

    @classmethod
    def _dynamic_obstacle_clear_match_motion_verified(
        cls,
        match: object,
    ) -> bool:
        """重算 clear match 的运动证据；字段缺失或被篡改时失败关闭。"""

        if not isinstance(match, dict):
            return False
        report = cls._dynamic_obstacle_motion_separation_report(
            match.get("obstacle_state_at_hit"),
            match.get("obstacle_state_after_clear"),
            map_resolution=match.get("map_resolution_m"),
        )
        if (
            report is None
            or report.get("obstacle_motion_separation_verified") is not True
            or match.get("obstacle_motion_separation_verified") is not True
        ):
            return False
        for field_name in (
            "obstacle_path_distance_separation_m",
            "obstacle_pose_position_separation_m",
            "minimum_obstacle_motion_separation_m",
        ):
            raw_value = match.get(field_name)
            expected_value = report[field_name]
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
                or not math.isclose(
                    float(raw_value),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                return False
        return True

    def _dynamic_obstacle_hit_matches(
        self,
        points: tuple[tuple[float, float, float], ...],
        voxel_indices: tuple[tuple[int, int, int], ...],
        *,
        timestamp: float,
        map_resolution: float,
    ) -> list[dict[str, Any]]:
        """把过滤后 hit 端点绑定到同一仿真时间的动态 cuboid。"""

        if len(points) != len(voxel_indices):
            raise RuntimeError("GridMap hit 点与 canonical voxel index 未对齐。")
        matches: list[dict[str, Any]] = []
        for point, voxel_index in zip(points, voxel_indices, strict=True):
            for obstacle, state in self._dynamic_obstacle_geometry_at(
                timestamp
            ):
                inside, clearance_xy = self._dynamic_obstacle_point_geometry(
                    point,
                    obstacle,
                    state,
                    tolerance_m=(
                        DYNAMIC_OBSTACLE_POINT_ASSOCIATION_TOLERANCE_M
                    ),
                )
                if not inside:
                    continue
                matches.append(
                    {
                        "obstacle_id": obstacle.obstacle_id,
                        "point_world_xyz": [float(value) for value in point],
                        "voxel_index_xyz": [int(value) for value in voxel_index],
                        "map_resolution_m": float(map_resolution),
                        "point_to_obstacle_xy_clearance_m": clearance_xy,
                        "association_tolerance_m": (
                            DYNAMIC_OBSTACLE_POINT_ASSOCIATION_TOLERANCE_M
                        ),
                        "obstacle_state": state.to_dict(),
                    }
                )
        return matches

    def _dynamic_obstacle_explicit_clear_matches(
        self,
        points: tuple[tuple[float, float, float], ...],
        voxel_indices: tuple[tuple[int, int, int], ...],
        transition_hit_sequences: tuple[int, ...],
        transition_hit_points: tuple[tuple[float, float, float], ...],
        transition_hit_header_stamps_ns: tuple[int, ...],
        *,
        timestamp: float,
        map_resolution: float,
    ) -> list[dict[str, Any]]:
        """把 explicit-miss clear 绑定到更早的 hit，排除滑窗清理冒充。"""

        geometries = {
            obstacle.obstacle_id: (obstacle, state)
            for obstacle, state in self._dynamic_obstacle_geometry_at(
                timestamp
            )
        }
        if not (
            len(points)
            == len(voxel_indices)
            == len(transition_hit_sequences)
            == len(transition_hit_points)
            == len(transition_hit_header_stamps_ns)
        ):
            raise RuntimeError("GridMap clear 点、voxel index 与 provenance 未对齐。")
        matches: list[dict[str, Any]] = []
        seen: set[tuple[tuple[int, int, int], str, int]] = set()
        for (
            clear_point,
            clear_voxel_index,
            hit_sequence,
            hit_point,
            hit_header_stamp_ns,
        ) in zip(
            points,
            voxel_indices,
            transition_hit_sequences,
            transition_hit_points,
            transition_hit_header_stamps_ns,
            strict=True,
        ):
            if hit_sequence <= 0 or hit_header_stamp_ns <= 0:
                continue
            hit_timestamp = float(hit_header_stamp_ns) * 1.0e-9
            hit_geometries = {
                obstacle.obstacle_id: (obstacle, state)
                for obstacle, state in self._dynamic_obstacle_geometry_at(
                    hit_timestamp
                )
            }
            for obstacle_id, (obstacle, state) in geometries.items():
                hit_geometry = hit_geometries.get(obstacle_id)
                if hit_geometry is None:
                    continue
                _hit_obstacle, hit_state = hit_geometry
                hit_inside, hit_clearance_xy = (
                    self._dynamic_obstacle_point_geometry(
                        hit_point,
                        obstacle,
                        hit_state,
                        tolerance_m=(
                            DYNAMIC_OBSTACLE_POINT_ASSOCIATION_TOLERANCE_M
                        ),
                    )
                )
                if not hit_inside:
                    continue
                motion_separation = (
                    self._dynamic_obstacle_motion_separation_report(
                        hit_state.to_dict(),
                        state.to_dict(),
                        map_resolution=map_resolution,
                    )
                )
                if (
                    motion_separation is None
                    or motion_separation.get(
                        "obstacle_motion_separation_verified"
                    )
                    is not True
                ):
                    continue
                voxel_separation_tolerance = (
                    0.5 * math.sqrt(3.0) * float(map_resolution) + 1.0e-9
                )
                still_inside, _ = self._dynamic_obstacle_point_geometry(
                    clear_point,
                    obstacle,
                    state,
                    tolerance_m=voxel_separation_tolerance,
                )
                if still_inside:
                    continue
                match_distance = math.dist(clear_point, hit_point)
                maximum_same_voxel_distance = (
                    0.5 * math.sqrt(3.0) * float(map_resolution) + 1.0e-9
                )
                if match_distance > maximum_same_voxel_distance:
                    raise RuntimeError(
                        "同一 canonical voxel 的 hit/clear 几何距离越界。"
                    )
                key = (
                    tuple(int(value) for value in clear_voxel_index),
                    obstacle_id,
                    int(hit_sequence),
                )
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    {
                        "obstacle_id": obstacle_id,
                        "point_world_xyz": [
                            float(value) for value in clear_point
                        ],
                        "matched_hit_observation_sequence": hit_sequence,
                        "matched_hit_header": {
                            "frame_id": "world",
                            "stamp": {
                                "sec": int(
                                    hit_header_stamp_ns // 1_000_000_000
                                ),
                                "nanosec": int(
                                    hit_header_stamp_ns % 1_000_000_000
                                ),
                            },
                            "stamp_ns": int(hit_header_stamp_ns),
                        },
                        "matched_hit_point_world_xyz": [
                            float(value) for value in hit_point
                        ],
                        "voxel_index_xyz": [
                            int(value) for value in clear_voxel_index
                        ],
                        "map_resolution_m": float(map_resolution),
                        "match_distance_m": match_distance,
                        "match_tolerance_m": maximum_same_voxel_distance,
                        "voxel_obstacle_separation_tolerance_m": (
                            voxel_separation_tolerance
                        ),
                        "matched_hit_provenance_verified": True,
                        "matched_hit_point_to_obstacle_xy_clearance_m": (
                            hit_clearance_xy
                        ),
                        "obstacle_state_at_hit": hit_state.to_dict(),
                        "obstacle_state_after_clear": state.to_dict(),
                        **motion_separation,
                        "sliding_reset_used": False,
                    }
                )
        if not matches:
            return []
        # 一条 clear report 只绑定一代 free→occupied 阈值穿越来源；否则
        # aggregate 无法用单一 observation identity 证明因果顺序。
        matched_sequence = min(
            int(match["matched_hit_observation_sequence"])
            for match in matches
        )
        return [
            match
            for match in matches
            if int(match["matched_hit_observation_sequence"])
            == matched_sequence
        ]

    def _update_grid_map_observation_lifecycle_report(
        self,
        sample: OgnGridMapObservationDiagnosticsSample,
    ) -> dict[str, Any]:
        """累计 GridMap typed 观测，并生成过滤后动态障碍 hit/clear 关联。"""

        lifecycle = self._metadata.get(
            "grid_map_observation_lifecycle_report"
        )
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("schema")
            != "grid_map_observation_lifecycle_v1"
        ):
            lifecycle = self._new_grid_map_observation_lifecycle_report(
                ros_time_offset_s=float(
                    getattr(
                        self,
                        "_navigation_episode_ros_time_offset_s",
                        0.0,
                    )
                )
            )
        expected_ros_time_offset_s = float(
            getattr(self, "_navigation_episode_ros_time_offset_s", 0.0)
        )
        if not math.isclose(
            float(lifecycle.get("ros_time_offset_s", math.nan)),
            expected_ros_time_offset_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("GridMap diagnostics 混入了其他 episode 时间域。")
        previous_sequence = lifecycle.get("last_observation_sequence")
        if previous_sequence is not None and sample.observation_sequence <= int(
            previous_sequence
        ):
            raise RuntimeError(
                "GridMap observation_sequence 在 runtime 生命周期中未严格递增。"
            )

        timestamp = float(sample.header_stamp_ns) * 1.0e-9
        ros_time_offset_s = float(
            getattr(self, "_navigation_episode_ros_time_offset_s", 0.0)
        )
        episode_elapsed_time_s = (
            self._dynamic_obstacle_episode_elapsed_at_ros_time(timestamp)
        )
        hit_matches = self._dynamic_obstacle_hit_matches(
            sample.hit_endpoint_samples,
            sample.hit_endpoint_sample_voxel_indices,
            timestamp=timestamp,
            map_resolution=sample.map_resolution,
        )
        transition_hit_matches = self._dynamic_obstacle_hit_matches(
            sample.free_to_occupied_transition_hit_samples,
            sample.free_to_occupied_transition_voxel_indices,
            timestamp=timestamp,
            map_resolution=sample.map_resolution,
        )
        clear_matches = self._dynamic_obstacle_explicit_clear_matches(
            sample.occupied_to_free_by_explicit_miss_samples,
            sample.occupied_to_free_sample_voxel_indices,
            sample.occupied_to_free_transition_hit_observation_sequences,
            sample.occupied_to_free_transition_hit_samples,
            sample.occupied_to_free_transition_hit_header_stamp_ns,
            timestamp=timestamp,
            map_resolution=sample.map_resolution,
        )
        report = {
            "source": "ros2_scan_grid_map_observation_diagnostics",
            "topic": sample.source_topic,
            "receipt_timestamp": float(sample.receipt_timestamp),
            "rx_sequence": int(sample.rx_sequence),
            "ros_time_offset_s": ros_time_offset_s,
            "episode_elapsed_time_s": episode_elapsed_time_s,
            "header": {
                "frame_id": sample.frame_id,
                "stamp": {
                    "sec": int(sample.header_stamp_sec),
                    "nanosec": int(sample.header_stamp_nanosec),
                },
                "stamp_ns": int(sample.header_stamp_ns),
            },
            "observation_sequence": int(sample.observation_sequence),
            "sensor_pose_stamp": {
                "sec": int(sample.sensor_pose_stamp_sec),
                "nanosec": int(sample.sensor_pose_stamp_nanosec),
                "stamp_ns": int(sample.sensor_pose_stamp_ns),
            },
            "sensor_origin_world_xyz": list(sample.sensor_origin),
            "canonical_empty": bool(sample.canonical_empty),
            "map_fusion_performed": bool(sample.map_fusion_performed),
            "map_resolution_m": float(sample.map_resolution),
            "input_point_count": int(sample.input_point_count),
            "accepted_endpoint_count": int(sample.accepted_endpoint_count),
            "hit_endpoint_count": int(sample.hit_endpoint_count),
            "explicit_free_endpoint_count": int(
                sample.explicit_free_endpoint_count
            ),
            "hit_endpoint_samples_truncated": bool(
                sample.hit_endpoint_samples_truncated
            ),
            "hit_endpoint_samples_world_xyz": [
                list(point) for point in sample.hit_endpoint_samples
            ],
            "hit_endpoint_sample_voxel_indices_xyz": [
                list(index)
                for index in sample.hit_endpoint_sample_voxel_indices
            ],
            "free_to_occupied_transition_count": int(
                sample.free_to_occupied_transition_count
            ),
            "free_to_occupied_transition_samples_truncated": bool(
                sample.free_to_occupied_transition_samples_truncated
            ),
            "free_to_occupied_transition_hit_samples_world_xyz": [
                list(point)
                for point in sample.free_to_occupied_transition_hit_samples
            ],
            "free_to_occupied_transition_voxel_indices_xyz": [
                list(index)
                for index in sample.free_to_occupied_transition_voxel_indices
            ],
            "explicit_free_miss_voxel_count": int(
                sample.explicit_free_miss_voxel_count
            ),
            "occupied_to_free_by_explicit_miss_count": int(
                sample.occupied_to_free_by_explicit_miss_count
            ),
            "occupied_to_free_samples_truncated": bool(
                sample.occupied_to_free_samples_truncated
            ),
            "occupied_to_free_by_explicit_miss_samples_world_xyz": [
                list(point)
                for point in (
                    sample.occupied_to_free_by_explicit_miss_samples
                )
            ],
            "occupied_to_free_sample_voxel_indices_xyz": [
                list(index)
                for index in sample.occupied_to_free_sample_voxel_indices
            ],
            "occupied_to_free_transition_hit_observation_sequences": [
                int(sequence)
                for sequence in (
                    sample.occupied_to_free_transition_hit_observation_sequences
                )
            ],
            "occupied_to_free_transition_hit_samples_world_xyz": [
                list(point)
                for point in sample.occupied_to_free_transition_hit_samples
            ],
            "occupied_to_free_transition_hit_header_stamp_ns": [
                int(stamp)
                for stamp in (
                    sample.occupied_to_free_transition_hit_header_stamp_ns
                )
            ],
            "occupied_removed_by_sliding_reset_count": int(
                sample.occupied_removed_by_sliding_reset_count
            ),
            "dynamic_obstacle_hit_matches": hit_matches,
            "dynamic_obstacle_transition_hit_matches": (
                transition_hit_matches
            ),
            "dynamic_obstacle_explicit_miss_clear_matches": clear_matches,
        }
        if lifecycle.get("first_observation_sequence") is None:
            lifecycle["first_observation_sequence"] = int(
                sample.observation_sequence
            )
            lifecycle["first_report"] = report
        lifecycle["last_observation_sequence"] = int(
            sample.observation_sequence
        )
        lifecycle["sample_count"] = int(lifecycle["sample_count"]) + 1
        lifecycle["canonical_empty_count"] = int(
            lifecycle["canonical_empty_count"]
        ) + int(sample.canonical_empty)
        lifecycle["map_fusion_count"] = int(
            lifecycle["map_fusion_count"]
        ) + int(sample.map_fusion_performed)
        for field_name in (
            "input_point_count",
            "accepted_endpoint_count",
            "hit_endpoint_count",
            "explicit_free_endpoint_count",
            "free_to_occupied_transition_count",
            "explicit_free_miss_voxel_count",
            "occupied_to_free_by_explicit_miss_count",
            "occupied_removed_by_sliding_reset_count",
        ):
            total_name = f"total_{field_name}"
            lifecycle[total_name] = int(lifecycle[total_name]) + int(
                report[field_name]
            )
        lifecycle["dynamic_obstacle_hit_match_count"] = int(
            lifecycle["dynamic_obstacle_hit_match_count"]
        ) + len(hit_matches)
        lifecycle["dynamic_obstacle_transition_hit_match_count"] = int(
            lifecycle["dynamic_obstacle_transition_hit_match_count"]
        ) + len(transition_hit_matches)
        lifecycle[
            "dynamic_obstacle_explicit_miss_clear_match_count"
        ] = int(
            lifecycle["dynamic_obstacle_explicit_miss_clear_match_count"]
        ) + len(clear_matches)
        lifecycle["last_report"] = report
        reports = lifecycle.get("diagnostic_reports")
        if not isinstance(reports, list):
            raise RuntimeError("GridMap lifecycle.diagnostic_reports 不是数组。")
        reports.append(report)
        if len(reports) > 128:
            reports.pop(0)
            lifecycle["dropped_diagnostic_report_count"] = int(
                lifecycle["dropped_diagnostic_report_count"]
            ) + 1
        if hit_matches:
            if lifecycle.get("first_hit_report") is None:
                lifecycle["first_hit_report"] = report
            lifecycle["last_hit_report"] = report
            hit_reports = lifecycle.get("hit_reports")
            if not isinstance(hit_reports, list):
                raise RuntimeError("GridMap lifecycle.hit_reports 不是数组。")
            hit_reports.append(report)
            if len(hit_reports) > 64:
                hit_reports.pop(0)
        if transition_hit_matches:
            if lifecycle.get("first_transition_hit_report") is None:
                lifecycle["first_transition_hit_report"] = report
            lifecycle["last_transition_hit_report"] = report
            transition_reports = lifecycle.get("transition_hit_reports")
            if not isinstance(transition_reports, list):
                raise RuntimeError(
                    "GridMap lifecycle.transition_hit_reports 不是数组。"
                )
            transition_reports.append(report)
            if len(transition_reports) > 64:
                transition_reports.pop(0)
        if clear_matches:
            if lifecycle.get("first_explicit_miss_clear_report") is None:
                lifecycle["first_explicit_miss_clear_report"] = report
            lifecycle["last_explicit_miss_clear_report"] = report
        self._metadata[
            "grid_map_observation_diagnostics_last_report"
        ] = report
        self._metadata[
            "grid_map_observation_lifecycle_report"
        ] = lifecycle
        self._refresh_active_sensing_fusion_evidence()
        return report

    def _update_bspline_diagnostics_lifecycle_report(
        self,
        sample: OgnBsplineDiagnosticsSample,
    ) -> dict[str, Any]:
        """累计 B-spline typed identity，并计算对当前动态 cuboid 的净空。"""

        lifecycle = self._metadata.get(
            "bspline_diagnostics_lifecycle_report"
        )
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("schema")
            != "bspline_diagnostics_lifecycle_v1"
        ):
            lifecycle = self._new_bspline_diagnostics_lifecycle_report(
                ros_time_offset_s=float(
                    getattr(
                        self,
                        "_navigation_episode_ros_time_offset_s",
                        0.0,
                    )
                )
            )
        expected_ros_time_offset_s = float(
            getattr(self, "_navigation_episode_ros_time_offset_s", 0.0)
        )
        if not math.isclose(
            float(lifecycle.get("ros_time_offset_s", math.nan)),
            expected_ros_time_offset_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("B-spline diagnostics 混入了其他 episode 时间域。")
        previous_sequence = lifecycle.get("last_diagnostic_sequence")
        if previous_sequence is not None and sample.diagnostic_sequence <= int(
            previous_sequence
        ):
            raise RuntimeError(
                "B-spline diagnostic_sequence 在 runtime 生命周期中未严格递增。"
            )

        identity = self._bspline_diagnostics_identity(sample)
        timestamp = float(sample.header_stamp_ns) * 1.0e-9
        ros_time_offset_s = float(
            getattr(self, "_navigation_episode_ros_time_offset_s", 0.0)
        )
        episode_elapsed_time_s = (
            self._dynamic_obstacle_episode_elapsed_at_ros_time(timestamp)
        )
        if not math.isclose(
            float(sample.double_cylinder_radius),
            GO2_X5_DOUBLE_CYLINDER_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ) or not math.isclose(
            float(sample.double_cylinder_offset),
            GO2_X5_DOUBLE_CYLINDER_OFFSET_M,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "B-spline diagnostics 的双圆柱包络与 Go2-X5 验收配置不一致。"
            )
        required_clearance_m = float(
            sample.double_cylinder_radius + sample.double_cylinder_offset
        )
        sample_interval_s = float(sample.trajectory_duration) / float(
            len(sample.trajectory_samples) - 1
        )
        sampling_clearance_margin_m = (
            float(sample.maximum_velocity_upper_bound)
            * sample_interval_s
            * 0.5
        )
        obstacle_clearances: list[dict[str, Any]] = []
        for obstacle, state in self._dynamic_obstacle_geometry_at(timestamp):
            trajectory_clearances = [
                self._dynamic_obstacle_point_geometry(
                    point,
                    obstacle,
                    state,
                )[1]
                for point in sample.trajectory_samples
            ]
            reference_clearances = [
                self._dynamic_obstacle_point_geometry(
                    point,
                    obstacle,
                    state,
                )[1]
                for point in sample.ordered_reference_samples
            ]
            minimum_sampled_clearance = min(trajectory_clearances)
            continuous_clearance_lower_bound = (
                minimum_sampled_clearance - sampling_clearance_margin_m
            )
            minimum_reference_clearance = (
                min(reference_clearances)
                if reference_clearances
                else None
            )
            reference_obstructed = bool(
                minimum_reference_clearance is not None
                and minimum_reference_clearance + 1.0e-9
                < required_clearance_m
            )
            clearance_verified = bool(
                continuous_clearance_lower_bound + 1.0e-9
                >= required_clearance_m
            )
            relevant = bool(
                min(
                    continuous_clearance_lower_bound,
                    (
                        minimum_reference_clearance
                        if minimum_reference_clearance is not None
                        else continuous_clearance_lower_bound
                    ),
                )
                <= DYNAMIC_OBSTACLE_RELEVANCE_DISTANCE_M
            )
            obstacle_clearances.append(
                {
                    "obstacle_id": obstacle.obstacle_id,
                    "obstacle_state": state.to_dict(),
                    "minimum_trajectory_center_to_obstacle_xy_m": (
                        minimum_sampled_clearance
                    ),
                    "trajectory_sample_interval_s": sample_interval_s,
                    "maximum_velocity_upper_bound_mps": float(
                        sample.maximum_velocity_upper_bound
                    ),
                    "sampling_clearance_margin_m": (
                        sampling_clearance_margin_m
                    ),
                    "continuous_clearance_lower_bound_m": (
                        continuous_clearance_lower_bound
                    ),
                    "continuous_clearance_verified": clearance_verified,
                    "minimum_ordered_reference_center_to_obstacle_xy_m": (
                        minimum_reference_clearance
                    ),
                    "required_clearance_m": (
                        required_clearance_m
                    ),
                    "clearance_verified": clearance_verified,
                    "reference_obstructed": reference_obstructed,
                    "reference_blocked_then_trajectory_clear": bool(
                        reference_obstructed and clearance_verified
                    ),
                    "relevant": relevant,
                    "relevance_distance_m": (
                        DYNAMIC_OBSTACLE_RELEVANCE_DISTANCE_M
                    ),
                }
            )
        relevant_clearances = [
            report for report in obstacle_clearances if report["relevant"]
        ]
        blocked_reference_clearances = [
            report
            for report in obstacle_clearances
            if report["reference_blocked_then_trajectory_clear"]
        ]
        ordered_detour_candidate = bool(
            sample.ordered_reference_checked
            and sample.ordered_reference_safe
            and not sample.stationary
            and not sample.emergency_stop
            and sample.maximum_trajectory_deviation + 1.0e-9
            >= DYNAMIC_OBSTACLE_DETOUR_DEVIATION_MIN_M
            and blocked_reference_clearances
        )
        report = {
            "source": "ros2_scan_bspline_diagnostics",
            "topic": sample.source_topic,
            "receipt_timestamp": float(sample.receipt_timestamp),
            "rx_sequence": int(sample.rx_sequence),
            "ros_time_offset_s": ros_time_offset_s,
            "episode_elapsed_time_s": episode_elapsed_time_s,
            "header": {
                "frame_id": sample.frame_id,
                "stamp": {
                    "sec": int(sample.header_stamp_sec),
                    "nanosec": int(sample.header_stamp_nanosec),
                },
                "stamp_ns": int(sample.header_stamp_ns),
            },
            "diagnostic_sequence": int(sample.diagnostic_sequence),
            "identity": identity,
            "is_final": bool(sample.is_final),
            "emergency_stop": bool(sample.emergency_stop),
            "stationary": bool(sample.stationary),
            "ordered_reference_checked": bool(
                sample.ordered_reference_checked
            ),
            "ordered_reference_safe": bool(sample.ordered_reference_safe),
            "maximum_trajectory_deviation_m": float(
                sample.maximum_trajectory_deviation
            ),
            "maximum_guide_anchor_deviation_m": float(
                sample.maximum_guide_anchor_deviation
            ),
            "maximum_guide_progress_lead_m": float(
                sample.maximum_guide_progress_lead
            ),
            "maximum_deviation_limit_m": float(
                sample.maximum_deviation_limit
            ),
            "maximum_progress_lead_limit_m": float(
                sample.maximum_progress_lead_limit
            ),
            "trajectory_duration_s": float(sample.trajectory_duration),
            "maximum_velocity_upper_bound_mps": float(
                sample.maximum_velocity_upper_bound
            ),
            "double_cylinder_radius_m": float(
                sample.double_cylinder_radius
            ),
            "double_cylinder_offset_m": float(
                sample.double_cylinder_offset
            ),
            "required_any_yaw_clearance_radius_m": required_clearance_m,
            "trajectory_sample_interval_s": sample_interval_s,
            "sampling_clearance_margin_m": sampling_clearance_margin_m,
            "detour_deviation_minimum_m": (
                DYNAMIC_OBSTACLE_DETOUR_DEVIATION_MIN_M
            ),
            "trajectory_sample_count_total": int(
                sample.trajectory_sample_count_total
            ),
            "trajectory_samples_truncated": bool(
                sample.trajectory_samples_truncated
            ),
            "trajectory_samples_world_xyz": [
                list(point) for point in sample.trajectory_samples
            ],
            "ordered_reference_sample_count_total": int(
                sample.ordered_reference_sample_count_total
            ),
            "ordered_reference_samples_truncated": bool(
                sample.ordered_reference_samples_truncated
            ),
            "ordered_reference_samples_world_xyz": [
                list(point) for point in sample.ordered_reference_samples
            ],
            "dynamic_obstacle_clearances": obstacle_clearances,
            "dynamic_obstacle_relevant": bool(relevant_clearances),
            "dynamic_obstacle_reference_obstructed": bool(
                blocked_reference_clearances
            ),
            "ordered_detour_candidate": ordered_detour_candidate,
            "active_sensing": {
                "enabled": bool(sample.active_sensing),
                "event": int(sample.active_sensing_event),
                "start_yaw": float(sample.active_sensing_start_yaw),
                "target_yaw": float(sample.active_sensing_target_yaw),
                "yaw_offset": float(sample.active_sensing_yaw_offset),
                "yaw_rate": float(sample.active_sensing_yaw_rate),
                "settle_stamp": {
                    "sec": int(sample.active_sensing_settle_stamp_sec),
                    "nanosec": int(
                        sample.active_sensing_settle_stamp_nanosec
                    ),
                },
                "settle_stamp_ns": int(
                    sample.active_sensing_settle_stamp_ns
                ),
                "settle_yaw_error": float(
                    sample.active_sensing_settle_yaw_error
                ),
                "settle_angular_speed": float(
                    sample.active_sensing_settle_angular_speed
                ),
                "stable_duration": float(
                    sample.active_sensing_stable_duration
                ),
                "fusion_baseline": int(
                    sample.active_sensing_fusion_baseline
                ),
                "fusion_current": int(
                    sample.active_sensing_fusion_current
                ),
                "fusion_distinct": int(
                    sample.active_sensing_fusion_distinct
                ),
                "fusion_required": int(
                    sample.active_sensing_fusion_required
                ),
                "completed": bool(sample.active_sensing_completed),
                "failed": bool(sample.active_sensing_failed),
                "reason": sample.active_sensing_reason,
            },
        }
        if lifecycle.get("first_diagnostic_sequence") is None:
            lifecycle["first_diagnostic_sequence"] = int(
                sample.diagnostic_sequence
            )
        lifecycle["last_diagnostic_sequence"] = int(
            sample.diagnostic_sequence
        )
        if sample.active_sensing:
            lifecycle["active_sensing_diagnostic_count"] = int(
                lifecycle.get("active_sensing_diagnostic_count", 0)
            ) + 1
        else:
            if lifecycle.get("first_report") is None:
                lifecycle["first_report"] = report
            lifecycle["sample_count"] = int(lifecycle["sample_count"]) + 1
            lifecycle["ordered_reference_checked_count"] = int(
                lifecycle["ordered_reference_checked_count"]
            ) + int(sample.ordered_reference_checked)
            lifecycle["ordered_reference_safe_count"] = int(
                lifecycle["ordered_reference_safe_count"]
            ) + int(sample.ordered_reference_safe)
            lifecycle["dynamic_obstacle_relevant_count"] = int(
                lifecycle["dynamic_obstacle_relevant_count"]
            ) + int(bool(relevant_clearances))
            lifecycle["ordered_detour_candidate_count"] = int(
                lifecycle["ordered_detour_candidate_count"]
            ) + int(ordered_detour_candidate)
            identities = lifecycle.get("trajectory_identities")
            if not isinstance(identities, list):
                raise RuntimeError(
                    "B-spline lifecycle.trajectory_identities 不是数组。"
                )
            if identity not in identities:
                if len(identities) >= 128:
                    raise RuntimeError(
                        "B-spline 单 episode typed identity 超过 128 代。"
                    )
                identities.append(identity)
            lifecycle["distinct_trajectory_identity_count"] = len(identities)
            lifecycle["last_report"] = report
            reports = lifecycle.get("diagnostic_reports")
            if not isinstance(reports, list):
                raise RuntimeError(
                    "B-spline lifecycle.diagnostic_reports 不是数组。"
                )
            reports.append(report)
            if len(reports) > 128:
                reports.pop(0)
                lifecycle["dropped_diagnostic_report_count"] = int(
                    lifecycle["dropped_diagnostic_report_count"]
                ) + 1
        self._metadata["bspline_diagnostics_last_report"] = report
        self._metadata["bspline_diagnostics_lifecycle_report"] = lifecycle
        self._update_active_sensing_from_bspline_report(report)
        return report

    def _refresh_dynamic_navigation_evidence_report(self) -> dict[str, Any]:
        """从 typed leaf 与 controller/policy identity 重建五项动态验收结果。"""

        plan = getattr(self, "_dynamic_obstacle_plan", DynamicObstaclePlan())
        aggregate = self._new_dynamic_navigation_evidence_report(
            plan,
            ros_time_offset_s=float(
                getattr(
                    self,
                    "_navigation_episode_ros_time_offset_s",
                    0.0,
                )
            ),
        )
        if not plan.enabled:
            self._metadata["dynamic_navigation_evidence_report"] = aggregate
            return aggregate
        grid_lifecycle = self._metadata.get(
            "grid_map_observation_lifecycle_report"
        )
        bspline_lifecycle = self._metadata.get(
            "bspline_diagnostics_lifecycle_report"
        )
        controller_lifecycle = self._metadata.get(
            "scan_controller_status_lifecycle_report"
        )
        policy_lifecycle = self._metadata.get(
            "navigation_policy_gate_lifecycle_report"
        )
        if not all(
            isinstance(report, dict)
            for report in (
                grid_lifecycle,
                bspline_lifecycle,
                controller_lifecycle,
                policy_lifecycle,
            )
        ):
            self._metadata["dynamic_navigation_evidence_report"] = aggregate
            return aggregate

        hit_report = grid_lifecycle.get("first_hit_report")
        clear_report = grid_lifecycle.get(
            "last_explicit_miss_clear_report"
        )
        if isinstance(hit_report, dict):
            hit_matches = hit_report.get("dynamic_obstacle_hit_matches")
            if isinstance(hit_matches, list) and hit_matches:
                aggregate["post_filter_hit"] = {
                    "verified": True,
                    "source": hit_report["source"],
                    "topic": hit_report["topic"],
                    "header": hit_report["header"],
                    "observation_sequence": hit_report[
                        "observation_sequence"
                    ],
                    "hit_endpoint_count": hit_report[
                        "hit_endpoint_count"
                    ],
                    "hit_endpoint_samples_world_xyz": hit_report[
                        "hit_endpoint_samples_world_xyz"
                    ],
                    "dynamic_obstacle_hit_matches": hit_matches,
                }
        if isinstance(clear_report, dict):
            clear_matches = clear_report.get(
                "dynamic_obstacle_explicit_miss_clear_matches"
            )
            hit_sequence = (
                int(clear_matches[0]["matched_hit_observation_sequence"])
                if isinstance(clear_matches, list)
                and clear_matches
                and isinstance(clear_matches[0], dict)
                else 0
            )
            clear_sequence = int(clear_report["observation_sequence"])
            sliding_count = int(
                clear_report["occupied_removed_by_sliding_reset_count"]
            )
            if (
                isinstance(clear_matches, list)
                and clear_matches
                and clear_sequence > hit_sequence
                and all(
                    isinstance(match, dict)
                    and match.get("matched_hit_provenance_verified") is True
                    and match.get("sliding_reset_used") is False
                    and self._dynamic_obstacle_clear_match_motion_verified(
                        match
                    )
                    for match in clear_matches
                )
                and int(
                    clear_report[
                        "occupied_to_free_by_explicit_miss_count"
                    ]
                )
                > 0
            ):
                aggregate["explicit_miss_ghost_clear"] = {
                    "verified": True,
                    "source": clear_report["source"],
                    "topic": clear_report["topic"],
                    "header": clear_report["header"],
                    "observation_sequence": clear_sequence,
                    "matched_hit_observation_sequence": hit_sequence,
                    "explicit_free_miss_voxel_count": clear_report[
                        "explicit_free_miss_voxel_count"
                    ],
                    "occupied_to_free_by_explicit_miss_count": clear_report[
                        "occupied_to_free_by_explicit_miss_count"
                    ],
                    "occupied_removed_by_sliding_reset_count": sliding_count,
                    "clear_matches": clear_matches,
                }

        accepted_status_reports = controller_lifecycle.get(
            "accepted_status_reports"
        )
        tracking_status_reports = controller_lifecycle.get(
            "tracking_status_reports"
        )
        identity_tracking_writes = policy_lifecycle.get(
            "identity_verified_tracking_write_reports"
        )
        diagnostics = bspline_lifecycle.get("diagnostic_reports")
        if not all(
            isinstance(reports, list)
            for reports in (
                accepted_status_reports,
                tracking_status_reports,
                identity_tracking_writes,
                diagnostics,
            )
        ):
            raise RuntimeError(
                "动态证据缺少 controller/policy/B-spline 有界状态数组。"
            )
        hit_header_ns = (
            int(hit_report["header"]["stamp_ns"])
            if isinstance(hit_report, dict)
            else None
        )
        clear_event_header_ns = (
            int(clear_report["header"]["stamp_ns"])
            if isinstance(clear_report, dict)
            else None
        )
        verified_clear_matches = (
            aggregate["explicit_miss_ghost_clear"].get("clear_matches")
            if aggregate["explicit_miss_ghost_clear"].get("verified") is True
            else None
        )
        if not isinstance(verified_clear_matches, list):
            verified_clear_matches = []
        detour_report = None
        detour_accepted_status = None
        detour_tracking_status = None
        detour_policy_tracking_write = None
        detour_causal_clear_match = None
        for candidate_report in diagnostics:
            if (
                not isinstance(candidate_report, dict)
                or candidate_report.get("ordered_detour_candidate") is not True
                or (
                    hit_header_ns is not None
                    and int(candidate_report["header"]["stamp_ns"])
                    < hit_header_ns
                )
                or (
                    clear_event_header_ns is not None
                    and int(candidate_report["header"]["stamp_ns"])
                    > clear_event_header_ns
                )
            ):
                continue
            candidate_header_ns = int(
                candidate_report["header"]["stamp_ns"]
            )
            candidate_receipt = float(
                candidate_report["receipt_timestamp"]
            )
            blocked_obstacle_ids = {
                str(clearance["obstacle_id"])
                for clearance in candidate_report.get(
                    "dynamic_obstacle_clearances", []
                )
                if isinstance(clearance, dict)
                and clearance.get(
                    "reference_blocked_then_trajectory_clear"
                )
                is True
            }
            causal_clear_match = next(
                (
                    match
                    for match in verified_clear_matches
                    if isinstance(match, dict)
                    and str(match.get("obstacle_id"))
                    in blocked_obstacle_ids
                    and int(match["matched_hit_header"]["stamp_ns"])
                    <= candidate_header_ns
                ),
                None,
            )
            if not isinstance(causal_clear_match, dict):
                continue
            matching_acceptance = next(
                (
                    status
                    for status in accepted_status_reports
                    if isinstance(status, dict)
                    and status.get("accepted") is True
                    and status.get("identity")
                    == candidate_report.get("identity")
                    and int(status["header"]["stamp_ns"])
                    >= candidate_header_ns
                    and float(status["receipt_timestamp"]) + 1.0e-9
                    >= candidate_receipt
                    and (
                        clear_event_header_ns is None
                        or int(status["header"]["stamp_ns"])
                        <= clear_event_header_ns
                    )
                    and (
                        not isinstance(clear_report, dict)
                        or float(status["receipt_timestamp"])
                        <= float(clear_report["receipt_timestamp"])
                        + 1.0e-9
                    )
                ),
                None,
            )
            if not isinstance(matching_acceptance, dict):
                continue
            matching_tracking = next(
                (
                    status
                    for status in tracking_status_reports
                    if isinstance(status, dict)
                    and int(status.get("state", -1)) == 10
                    and status.get("trajectory_valid") is True
                    and status.get("identity")
                    == candidate_report.get("identity")
                    and int(status["header"]["stamp_ns"])
                    >= int(matching_acceptance["header"]["stamp_ns"])
                    and float(status["receipt_timestamp"]) + 1.0e-9
                    >= float(matching_acceptance["receipt_timestamp"])
                    and (
                        clear_event_header_ns is None
                        or int(status["header"]["stamp_ns"])
                        <= clear_event_header_ns
                    )
                    and (
                        not isinstance(clear_report, dict)
                        or float(status["receipt_timestamp"])
                        <= float(clear_report["receipt_timestamp"])
                        + 1.0e-9
                    )
                ),
                None,
            )
            if not isinstance(matching_tracking, dict):
                continue
            matching_policy_write = next(
                (
                    write
                    for write in identity_tracking_writes
                    if isinstance(write, dict)
                    and write.get("scan_controller_status_snapshot")
                    == matching_tracking
                    and float(write.get("timestamp", -1.0)) + 1.0e-9
                    >= max(
                        candidate_receipt,
                        float(matching_tracking["receipt_timestamp"]),
                    )
                    and (
                        not isinstance(clear_report, dict)
                        or float(write.get("timestamp", math.inf))
                        + 1.0e-9
                        < float(clear_report["receipt_timestamp"])
                    )
                ),
                None,
            )
            if not isinstance(matching_policy_write, dict):
                continue
            detour_report = candidate_report
            detour_accepted_status = matching_acceptance
            detour_tracking_status = matching_tracking
            detour_policy_tracking_write = matching_policy_write
            detour_causal_clear_match = causal_clear_match
            break
        if isinstance(detour_report, dict):
            aggregate["ordered_detour"] = {
                "verified": True,
                "source": detour_report["source"],
                "topic": detour_report["topic"],
                "header": detour_report["header"],
                "diagnostic_sequence": detour_report[
                    "diagnostic_sequence"
                ],
                "identity": detour_report["identity"],
                "maximum_trajectory_deviation_m": detour_report[
                    "maximum_trajectory_deviation_m"
                ],
                "maximum_deviation_limit_m": detour_report[
                    "maximum_deviation_limit_m"
                ],
                "maximum_guide_progress_lead_m": detour_report[
                    "maximum_guide_progress_lead_m"
                ],
                "maximum_progress_lead_limit_m": detour_report[
                    "maximum_progress_lead_limit_m"
                ],
                "trajectory_samples_world_xyz": detour_report[
                    "trajectory_samples_world_xyz"
                ],
                "ordered_reference_samples_world_xyz": detour_report[
                    "ordered_reference_samples_world_xyz"
                ],
                "dynamic_obstacle_clearances": detour_report[
                    "dynamic_obstacle_clearances"
                ],
                "dynamic_obstacle_reference_obstructed": detour_report[
                    "dynamic_obstacle_reference_obstructed"
                ],
                "controller_identity_accepted": True,
                "controller_accepted_status": detour_accepted_status,
                "controller_tracking_status": detour_tracking_status,
                "policy_identity_valid_tracking": True,
                "policy_identity_verified_tracking_write": (
                    detour_policy_tracking_write
                ),
                "causal_map_transition_clear_match": (
                    detour_causal_clear_match
                ),
            }
            relevant_clearances = [
                clearance
                for clearance in detour_report[
                    "dynamic_obstacle_clearances"
                ]
                if clearance.get("relevant") is True
            ]
            clearance_verified = bool(
                relevant_clearances
                and all(
                    clearance.get("clearance_verified") is True
                    for clearance in relevant_clearances
                )
            )
            aggregate["current_obstacle_clearance"] = {
                "verified": clearance_verified,
                "source": detour_report["source"],
                "topic": detour_report["topic"],
                "header": detour_report["header"],
                "diagnostic_sequence": detour_report[
                    "diagnostic_sequence"
                ],
                "identity": detour_report["identity"],
                "required_clearance_m": GO2_X5_ANY_YAW_CLEARANCE_RADIUS_M,
                "obstacle_clearances": relevant_clearances,
                "reason": (
                    "all_relevant_obstacles_clear"
                    if clearance_verified
                    else "relevant_obstacle_clearance_below_required"
                ),
            }

        clear_header_ns = (
            int(clear_report["header"]["stamp_ns"])
            if isinstance(clear_report, dict)
            and aggregate["explicit_miss_ghost_clear"]["verified"] is True
            else None
        )
        recovery_report = None
        recovery_accepted_status = None
        controller_tracking_report = None
        policy_tracking_report = None
        if isinstance(detour_report, dict) and clear_header_ns is not None:
            for candidate_report in diagnostics:
                if (
                    not isinstance(candidate_report, dict)
                    or int(candidate_report["diagnostic_sequence"])
                    <= int(detour_report["diagnostic_sequence"])
                    or int(candidate_report["header"]["stamp_ns"])
                    < clear_header_ns
                    or candidate_report.get("identity")
                    == detour_report.get("identity")
                    or candidate_report.get("emergency_stop") is not False
                    or candidate_report.get("stationary") is not False
                    or candidate_report.get("ordered_reference_checked")
                    is not True
                    or candidate_report.get("ordered_reference_safe")
                    is not True
                    or candidate_report["identity"][
                        "reference_path_stamp_ns"
                    ]
                    != detour_report["identity"][
                        "reference_path_stamp_ns"
                    ]
                    or float(
                        candidate_report[
                            "maximum_trajectory_deviation_m"
                        ]
                    )
                    > DYNAMIC_OBSTACLE_RECOVERY_MAX_DEVIATION_M
                    + 1.0e-9
                    or float(
                        detour_report["maximum_trajectory_deviation_m"]
                    )
                    - float(
                        candidate_report[
                            "maximum_trajectory_deviation_m"
                        ]
                    )
                    < DYNAMIC_OBSTACLE_RECOVERY_IMPROVEMENT_MIN_M
                    - 1.0e-9
                ):
                    continue
                candidate_header_ns = int(
                    candidate_report["header"]["stamp_ns"]
                )
                candidate_receipt = float(
                    candidate_report["receipt_timestamp"]
                )
                candidate_acceptance = next(
                    (
                        status
                        for status in accepted_status_reports
                        if isinstance(status, dict)
                        and status.get("accepted") is True
                        and status.get("identity")
                        == candidate_report.get("identity")
                        and int(status["header"]["stamp_ns"])
                        >= candidate_header_ns
                        and float(status["receipt_timestamp"])
                        + 1.0e-9
                        >= candidate_receipt
                    ),
                    None,
                )
                if not isinstance(candidate_acceptance, dict):
                    continue
                candidate_tracking = next(
                    (
                        status
                        for status in tracking_status_reports
                        if isinstance(status, dict)
                        and int(status.get("state", -1)) == 10
                        and status.get("trajectory_valid") is True
                        and status.get("identity")
                        == candidate_report.get("identity")
                        and int(status["header"]["stamp_ns"])
                        >= int(candidate_acceptance["header"]["stamp_ns"])
                        and float(status["receipt_timestamp"])
                        + 1.0e-9
                        >= float(
                            candidate_acceptance["receipt_timestamp"]
                        )
                    ),
                    None,
                )
                if not isinstance(candidate_tracking, dict):
                    continue
                candidate_policy_write = next(
                    (
                        write
                        for write in identity_tracking_writes
                        if isinstance(write, dict)
                        and write.get(
                            "scan_controller_status_snapshot"
                        )
                        == candidate_tracking
                        and float(write.get("timestamp", -1.0))
                        + 1.0e-9
                        >= max(
                            candidate_receipt,
                            float(
                                candidate_tracking[
                                    "receipt_timestamp"
                                ]
                            ),
                        )
                    ),
                    None,
                )
                if not isinstance(candidate_policy_write, dict):
                    continue
                recovery_report = candidate_report
                recovery_accepted_status = candidate_acceptance
                controller_tracking_report = candidate_tracking
                policy_tracking_report = candidate_policy_write
                break
        if (
            isinstance(recovery_report, dict)
            and isinstance(recovery_accepted_status, dict)
            and isinstance(controller_tracking_report, dict)
            and isinstance(policy_tracking_report, dict)
        ):
            aggregate["trajectory_recovery"] = {
                "verified": True,
                "source": recovery_report["source"],
                "topic": recovery_report["topic"],
                "before_diagnostic_sequence": detour_report[
                    "diagnostic_sequence"
                ],
                "before_header": detour_report["header"],
                "before_detour_identity": detour_report["identity"],
                "after_recovery_identity": recovery_report["identity"],
                "after_diagnostic_sequence": recovery_report[
                    "diagnostic_sequence"
                ],
                "after_header": recovery_report["header"],
                "before_maximum_trajectory_deviation_m": detour_report[
                    "maximum_trajectory_deviation_m"
                ],
                "after_maximum_trajectory_deviation_m": recovery_report[
                    "maximum_trajectory_deviation_m"
                ],
                "recovery_maximum_deviation_m": (
                    DYNAMIC_OBSTACLE_RECOVERY_MAX_DEVIATION_M
                ),
                "recovery_minimum_improvement_m": (
                    DYNAMIC_OBSTACLE_RECOVERY_IMPROVEMENT_MIN_M
                ),
                "controller_tracking_status": controller_tracking_report,
                "controller_accepted_status": recovery_accepted_status,
                "controller_acceptance_sequence": controller_tracking_report[
                    "acceptance_sequence"
                ],
                "controller_status_sequence": controller_tracking_report[
                    "status_sequence"
                ],
                "policy_identity_valid_tracking": True,
                "policy_identity_verified_tracking_write": (
                    policy_tracking_report
                ),
                "same_reference_path_generation": True,
            }
        aggregate["verified"] = all(
            isinstance(aggregate[key], dict)
            and aggregate[key].get("verified") is True
            for key in (
                "post_filter_hit",
                "ordered_detour",
                "current_obstacle_clearance",
                "explicit_miss_ghost_clear",
                "trajectory_recovery",
            )
        )
        self._metadata["dynamic_navigation_evidence_report"] = aggregate
        return aggregate

    def _poll_grid_map_observation_diagnostics(
        self,
        *,
        receipt_timestamp: float,
    ) -> None:
        """轮询 GridMap typed 诊断并刷新动态障碍聚合证据。"""

        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if (
            bridge is None
            or not bridge.config.enable_grid_map_diagnostics_subscription
        ):
            return
        sample = bridge.poll_grid_map_observation_diagnostics(
            receipt_timestamp=receipt_timestamp,
        )
        if sample is None:
            return
        self._update_grid_map_observation_lifecycle_report(sample)
        self._refresh_dynamic_navigation_evidence_report()

    def _poll_bspline_diagnostics(self, *, receipt_timestamp: float) -> None:
        """轮询 B-spline typed 诊断并刷新动态障碍聚合证据。"""

        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if (
            bridge is None
            or not bridge.config.enable_bspline_diagnostics_subscription
        ):
            return
        sample = bridge.poll_bspline_diagnostics(
            receipt_timestamp=receipt_timestamp,
        )
        if sample is None:
            return
        self._update_bspline_diagnostics_lifecycle_report(sample)
        self._refresh_dynamic_navigation_evidence_report()

    def _poll_scan_controller_status(self, *, receipt_timestamp: float) -> None:
        """轮询 typed controller 状态，并保留精确 Path/B-spline identity。"""

        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if (
            bridge is None
            or not bridge.config.enable_controller_status_subscription
        ):
            return
        sample = bridge.poll_controller_status(
            receipt_timestamp=receipt_timestamp,
        )
        if sample is None:
            return
        identity = {
            "reference_path_stamp": sample.reference_path_stamp,
            "reference_path_stamp_ns": int(sample.reference_path_stamp_ns),
            "bspline_header_stamp": sample.bspline_header_stamp,
            "bspline_header_stamp_ns": int(sample.bspline_header_stamp_ns),
            "start_time": sample.start_time,
            "start_time_ns": int(sample.start_time_ns),
            "traj_id": int(sample.traj_id),
        }
        candidate = (
            {
                "reference_path_stamp": (
                    sample.candidate_reference_path_stamp
                ),
                "reference_path_stamp_ns": int(
                    sample.candidate_reference_path_stamp_ns
                ),
                "bspline_header_stamp": (
                    sample.candidate_bspline_header_stamp
                ),
                "bspline_header_stamp_ns": int(
                    sample.candidate_bspline_header_stamp_ns
                ),
                "start_time": sample.candidate_start_time,
                "start_time_ns": int(sample.candidate_start_time_ns),
                "traj_id": int(sample.candidate_traj_id),
            }
            if sample.candidate_present
            else None
        )
        status_report = {
            "source": "ros2_scan_planner_msgs_controller_status",
            "topic": sample.source_topic,
            "receipt_timestamp": float(sample.receipt_timestamp),
            "rx_sequence": int(sample.rx_sequence),
            "header": {
                "frame_id": sample.frame_id,
                "stamp": sample.header_stamp,
                "stamp_ns": int(sample.header_stamp_ns),
            },
            "status_sequence": int(sample.status_sequence),
            "acceptance_sequence": int(sample.acceptance_sequence),
            "event": int(sample.event),
            "state": int(sample.state),
            "reason": sample.reason,
            "accepted": bool(sample.accepted),
            "trajectory_valid": bool(sample.trajectory_valid),
            "is_final": bool(sample.is_final),
            "emergency_stop": bool(sample.emergency_stop),
            "active_sensing_yaw_only": bool(
                sample.active_sensing_yaw_only
            ),
            "command_aggregate": {
                "sample_count": int(sample.command_sample_count),
                "first_command": [
                    float(value) for value in sample.first_command
                ],
                "max_abs_vx": float(sample.max_abs_vx),
                "max_abs_vy": float(sample.max_abs_vy),
                "max_abs_wz": float(sample.max_abs_wz),
                "violation_count": int(sample.command_violation_count),
            },
            "identity": identity,
            "candidate": candidate,
        }
        self._metadata["scan_controller_status_last_report"] = status_report
        self._update_scan_controller_status_lifecycle_report(status_report)
        self._refresh_dynamic_navigation_evidence_report()

    def _poll_scan_reference_path(self) -> None:
        """轮询 live Path，并保存可由 ``SimulationState`` 消费的严格证据。"""

        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if (
            bridge is None
            or not bridge.config.enable_reference_path_subscription
        ):
            return
        sample = bridge.poll_reference_path()
        if sample is None:
            return
        self._metadata["scan_reference_path_last_report"] = {
            "points_ground_xyz": [
                [float(x), float(y), float(z)]
                for x, y, z in sample.points_ground_xyz
            ],
            "terminal_yaw": (
                None
                if sample.terminal_yaw is None
                else float(sample.terminal_yaw)
            ),
            "source": "ros2_nav_msgs_path",
            "topic": sample.source_topic,
            "frame_id": sample.frame_id,
            "stamp": sample.stamp,
            "sequence": int(sample.sequence),
            "points_sha256": sample.points_sha256,
            "cleared": len(sample.points_ground_xyz) == 0,
        }

    def _update_velocity_command_visualization(self, action: RobotAction) -> None:
        """按控制 tick 绘制实际进入 locomotion policy 的速度命令。"""

        if not self._config.show_velocity_command_debug:
            return
        from source.diagnostics.planned_trajectories import draw_velocity_command

        pose = self._adapter.get_base_pose_full()
        effective_command_getter = getattr(
            self._adapter,
            "get_effective_base_command",
            None,
        )
        effective_command = (
            effective_command_getter()
            if callable(effective_command_getter)
            else action.base_velocity
        )
        report = draw_velocity_command(
            robot_root_pose=(
                float(pose["x"]),
                float(pose["y"]),
                float(pose["z"]),
                *(float(value) for value in pose["quat_wxyz"]),
            ),
            base_velocity=effective_command,
            source=action.source,
        )
        self._metadata["velocity_command_visualization"] = {
            **report,
            "requested_base_velocity": [
                float(value) for value in action.base_velocity
            ],
            "effective_base_velocity": [
                float(value) for value in effective_command
            ],
        }

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
            if report.get("enabled") is not True:
                raise RuntimeError(
                    "failed to enable support joint lock: "
                    f"{support_report}"
                )
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
            if report.get("enabled") is not True:
                raise RuntimeError(
                    "failed to enable navigation joint pose lock: "
                    f"{lock_report}"
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
        profiler = self._performance_profiler
        control_started_at = time.perf_counter()
        started_at = time.perf_counter()
        self._runtime.recorder_manager.record_pre_step()
        if profiler is not None:
            profiler.record(
                "runtime.recorder_pre_step",
                time.perf_counter() - started_at,
            )
        for _ in range(self._runtime.cfg.decimation):
            prepare_started_at = time.perf_counter()
            self._runtime._sim_step_counter += 1
            self._apply_active_manipulation_base_lock(timing="before_physics_step")
            self._runtime.action_manager.apply_action()
            # action_manager 会重写 joint_pos action term 覆盖目标；这里在写入 sim 前
            # 重新刷新支撑锁和机械臂/夹爪 position target，确保本物理子步使用 pipeline 目标。
            self._apply_active_manipulation_base_lock(timing="after_action_manager")
            self._apply_staged_joint_position_targets(timing="after_action_manager")
            self._advance_dynamic_obstacles_for_physics_step()
            self._runtime.scene.write_data_to_sim()
            if profiler is not None:
                profiler.record(
                    "runtime.physics_prepare",
                    time.perf_counter() - prepare_started_at,
                )
            # 这里只有 runtime 执行底层 physics step，调用权来自 pipeline 唯一主循环。
            physics_started_at = time.perf_counter()
            self._runtime.sim.step(render=False)
            self._ros2_physics_step_count = (
                int(getattr(self, "_ros2_physics_step_count", 0)) + 1
            )
            if profiler is not None:
                profiler.record(
                    "runtime.physics_step",
                    time.perf_counter() - physics_started_at,
                )
            post_started_at = time.perf_counter()
            self._apply_active_manipulation_base_lock(timing="after_physics_step")
            self._runtime.recorder_manager.record_post_physics_decimation_step()
            if profiler is not None:
                profiler.record(
                    "runtime.physics_post_step",
                    time.perf_counter() - post_started_at,
                )
            if (
                is_rendering
                and self._runtime._sim_step_counter % self._runtime.cfg.sim.render_interval == 0
            ):
                render_started_at = time.perf_counter()
                self._runtime.sim.render()
                self._mark_camera_render(
                    valid_state_step=self._step_calls + 1,
                    reason="control_render_grid",
                )
                if profiler is not None:
                    profiler.record(
                        "runtime.rtx_render",
                        time.perf_counter() - render_started_at,
                    )
            scene_update_started_at = time.perf_counter()
            self._runtime.scene.update(dt=self._runtime.physics_dt)
            if profiler is not None:
                profiler.record(
                    "runtime.scene_update",
                    time.perf_counter() - scene_update_started_at,
                )

        finish_started_at = time.perf_counter()
        self._finish_control_step()
        if profiler is not None:
            profiler.record(
                "runtime.finish_control_step",
                time.perf_counter() - finish_started_at,
            )
            profiler.record(
                "runtime.control_step_total",
                time.perf_counter() - control_started_at,
            )
        self._step_calls += 1
        self._complete_stair_probe_low_level_telemetry(
            completed_control_step=self._step_calls,
        )
        self._action_prepared = False
        self._publish_navigation_ros2_observation(
            completed_control_step=self._step_calls,
        )

    def _publish_navigation_ros2_observation(
        self,
        *,
        completed_control_step: int,
    ) -> None:
        """在完整控制步结束后发布同源时间的 Odometry 与新鲜深度点云。"""

        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if bridge is None:
            return
        cloud_config = self._config.depth_point_cloud_config
        if cloud_config is None:
            raise RuntimeError("ROS 2 bridge 已启用但 depth_point_cloud_config 缺失。")
        timestamp = self._navigation_ros2_timestamp()
        if timestamp <= 0.0:
            raise RuntimeError("ROS 2 发布要求至少完成一个成功 physics step。")

        robot = self._adapter.robot
        bridge.update_odometry(
            _as_tuple(robot.data.root_pos_w[0]),
            _as_tuple(robot.data.root_quat_w[0]),
            _as_tuple(robot.data.root_lin_vel_b[0]),
            _as_tuple(robot.data.root_ang_vel_b[0]),
            timestamp,
        )
        self._ros2_odometry_publish_count += 1
        command_gate = getattr(self, "_cmd_vel_to_policy", None)
        if command_gate is not None:
            command_gate.mark_odometry(
                owner_id=self._cmd_vel_owner_id,
                received_at=timestamp,
            )

        cloud_due = (
            completed_control_step
            % cloud_config.publish_interval_control_steps
            == 0
        )
        cloud_published = False
        cloud_point_count = 0
        cloud_skip_reason: str | None = None
        dynamic_obstacle_raw_cloud_frame: dict[str, Any] | None = None
        if cloud_due:
            if self._last_camera_render_step != completed_control_step:
                # 不得把 reset 前或旧控制步的深度重新盖上当前时间戳。
                cloud_skip_reason = "stale_or_unrendered_depth_rejected"
            else:
                try:
                    sensor = self._runtime.scene[cloud_config.sensor_name]
                except (KeyError, TypeError) as exc:
                    raise RuntimeError(
                        f"Isaac scene 缺少导航深度相机 {cloud_config.sensor_name!r}。"
                    ) from exc
                points = camera_sensor_to_world_points(sensor, cloud_config)
                cloud_point_count = int(points.shape[0])
                if cloud_point_count < cloud_config.minimum_valid_points:
                    # 无效或覆盖不足的深度帧不能刷新点云新鲜度；保持上一帧
                    # 时间戳，让 controller 与 policy 安全门按超时主动停车。
                    cloud_skip_reason = "insufficient_valid_points"
                else:
                    dynamic_obstacle_plan = getattr(
                        self,
                        "_dynamic_obstacle_plan",
                        DynamicObstaclePlan(),
                    )
                    if dynamic_obstacle_plan.enabled:
                        dynamic_obstacle_raw_cloud_frame = (
                            self._update_dynamic_obstacle_raw_cloud_lifecycle_report(
                                points_world_xyz=points,
                                timestamp=timestamp,
                                completed_control_step=completed_control_step,
                            )
                        )
                    bridge.update_point_cloud(points, timestamp=timestamp)
                    self._ros2_point_cloud_publish_count += 1
                    cloud_published = True
                    if command_gate is not None:
                        command_gate.mark_point_cloud(
                            owner_id=self._cmd_vel_owner_id,
                            received_at=timestamp,
                        )
        else:
            cloud_skip_reason = "publish_interval_not_due"

        # Path 订阅与传感器发布共享同一个 OGN graph，每个成功控制步轮询一次；
        # 新报告会在紧随其后的 SimulationState.read() 中原样暴露。
        self._poll_scan_reference_path()
        self._poll_grid_map_observation_diagnostics(
            receipt_timestamp=timestamp
        )
        self._poll_bspline_diagnostics(receipt_timestamp=timestamp)
        # planner STARTED/ACCEPTED typed 事件必须先建档，随后同一控制拍收到的
        # controller aggregate 才能按完整 identity 归属，避免首条零命令误挂。
        self._poll_scan_controller_status(receipt_timestamp=timestamp)

        self._metadata["navigation_ros2_last_publish_report"] = {
            "completed_control_step": int(completed_control_step),
            "continuous_physics_step_count": int(self._ros2_physics_step_count),
            "timestamp": timestamp,
            "odometry_published": True,
            "odometry_publish_count": int(self._ros2_odometry_publish_count),
            "point_cloud_due": cloud_due,
            "point_cloud_published": cloud_published,
            "point_cloud_point_count": cloud_point_count,
            "point_cloud_publish_count": int(
                self._ros2_point_cloud_publish_count
            ),
            "point_cloud_skip_reason": cloud_skip_reason,
            "dynamic_obstacle_raw_cloud_frame": (
                dynamic_obstacle_raw_cloud_frame
            ),
            "depth_render_step": self._last_camera_render_step,
            "frame_id": bridge.config.odom_frame_id,
        }

    def close(self) -> None:
        if self._closed:
            return
        command_gate = getattr(self, "_cmd_vel_to_policy", None)
        if command_gate is not None:
            command_gate.release(
                owner_id=self._cmd_vel_owner_id,
                now=self._navigation_ros2_timestamp(),
            )
            self._cmd_vel_to_policy = None
        bridge = getattr(self, "_ros2_ogn_bridge", None)
        if bridge is not None:
            # 当前 env 生命周期与 stage 一致；释放 Python/Fabric 句柄，stage
            # 随 env/app 关闭。真正的跨 stage runtime 会显式重建 graph。
            bridge.invalidate_after_stage_reload()
            self._ros2_ogn_bridge = None
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
                        self._front_camera_sensor_enabled()
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
        self._initialize_episode_support_readers(episode_spec)
        self._metadata["object_pose_debug_after_physics_reader"] = (
            self._object_initial_pose_diagnostic(
                episode_spec,
                label="after_object_reader_initialize",
            )
        )

    def _apply_d436_runtime_intrinsics(self, runtime: Any) -> dict[str, Any]:
        """让 IsaacLab 对外暴露的 K 与 OpenCV schema 的实际渲染内参一致。"""

        camera_sensors = []
        if self._front_camera_sensor_enabled():
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

    @staticmethod
    def _dynamic_obstacle_configuration_metadata(
        plan: DynamicObstaclePlan,
    ) -> dict[str, Any]:
        """生成 build/reset/stage reuse 共用的动态障碍配置证据。"""

        report = plan.to_dict()
        report["registered_scene_assets"] = [
            {
                "id": obstacle.obstacle_id,
                "scene_asset_name": obstacle.scene_asset_name,
                "prim_path": obstacle.prim_path,
                "shape": "cuboid",
                "kinematic_enabled": True,
                "collision_enabled": True,
                "visible": True,
            }
            for obstacle in plan.obstacles
        ]
        return report

    @staticmethod
    def _new_dynamic_obstacle_lifecycle_report(
        plan: DynamicObstaclePlan,
        *,
        ros_time_offset_s: float = 0.0,
    ) -> dict[str, Any]:
        """建立单 episode 动态障碍运动证据，禁止跨 reset 复用旧样本。"""

        return {
            "schema": "dynamic_obstacle_lifecycle_v1",
            "enabled": bool(plan.enabled),
            "ros_time_offset_s": float(ros_time_offset_s),
            "obstacle_count": len(plan.obstacles),
            "pose_write_count": 0,
            "sample_frame_count": 0,
            "first_physics_step_index": None,
            "last_physics_step_index": None,
            "first_elapsed_time_s": None,
            "last_elapsed_time_s": None,
            "all_configured_obstacles_sampled": not plan.enabled,
            "all_configured_obstacles_moved": False,
            "maximum_path_distance_span_m": 0.0,
            "direction_transition_count": 0,
            "obstacles": {
                obstacle.obstacle_id: {
                    "scene_asset_name": obstacle.scene_asset_name,
                    "sample_count": 0,
                    "first_state": None,
                    "last_state": None,
                    "minimum_path_distance_m": None,
                    "maximum_path_distance_m": None,
                    "path_distance_span_m": 0.0,
                    "path_directions_seen": [],
                    "direction_transition_count": 0,
                    "waiting_for_start_seen": False,
                    "motion_started_seen": False,
                    "maximum_displacement_from_first_m": 0.0,
                }
                for obstacle in plan.obstacles
            },
        }

    @staticmethod
    def _new_dynamic_obstacle_raw_cloud_lifecycle_report(
        plan: DynamicObstaclePlan,
    ) -> dict[str, Any]:
        """建立 RTX 原始世界系点云对动态障碍的单 episode 命中证据。"""

        return {
            "schema": "dynamic_obstacle_raw_cloud_lifecycle_v1",
            "enabled": bool(plan.enabled),
            "source": "isaac_rtx_world_cloud_before_ros_filter",
            "proof_scope": "raw_cloud_visibility_only",
            "aabb_tolerance_m": 0.03,
            "sample_frame_count": 0,
            "frames_with_any_obstacle_points": 0,
            "frames_with_motion_started_obstacle_points": 0,
            "maximum_total_obstacle_point_count": 0,
            "first_detection": None,
            "last_detection": None,
            "all_configured_obstacles_observed": not plan.enabled,
            "obstacles": {
                obstacle.obstacle_id: {
                    "scene_asset_name": obstacle.scene_asset_name,
                    "sample_frame_count": 0,
                    "detected_frame_count": 0,
                    "maximum_point_count": 0,
                    "first_detection": None,
                    "last_detection": None,
                    "motion_started_detection_seen": False,
                    "path_directions_detected": [],
                    "minimum_detected_path_distance_m": None,
                    "maximum_detected_path_distance_m": None,
                    "detected_path_distance_span_m": 0.0,
                }
                for obstacle in plan.obstacles
            },
        }

    def _update_dynamic_obstacle_raw_cloud_lifecycle_report(
        self,
        *,
        points_world_xyz: Any,
        timestamp: float,
        completed_control_step: int,
    ) -> dict[str, Any]:
        """把原始 RTX 点与当前 cuboid 包络关联，但不冒充 ROS 过滤后点云。"""

        import numpy as np

        points = np.asarray(points_world_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("动态障碍点云证据要求 N×3 世界系数组。")
        if not bool(np.isfinite(points).all()):
            raise ValueError("动态障碍点云证据不能包含非有限坐标。")

        plan = getattr(self, "_dynamic_obstacle_plan", DynamicObstaclePlan())
        lifecycle = self._metadata.get(
            "dynamic_obstacle_raw_cloud_lifecycle_report"
        )
        expected_ids = {obstacle.obstacle_id for obstacle in plan.obstacles}
        lifecycle_ids = (
            set(lifecycle.get("obstacles", {}))
            if isinstance(lifecycle, dict)
            and isinstance(lifecycle.get("obstacles"), dict)
            else set()
        )
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("schema")
            != "dynamic_obstacle_raw_cloud_lifecycle_v1"
            or lifecycle_ids != expected_ids
        ):
            lifecycle = self._new_dynamic_obstacle_raw_cloud_lifecycle_report(
                plan
            )

        runtime = getattr(self, "_runtime", None)
        physics_dt = float(getattr(runtime, "physics_dt", 0.0))
        physics_step_index = int(getattr(runtime, "_sim_step_counter", 0))
        if not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise RuntimeError("动态障碍点云证据缺少有效 physics_dt。")
        elapsed_time_s = physics_step_index * physics_dt
        states = {
            state.obstacle_id: state
            for state in plan.state_at(elapsed_time_s)
        }
        tolerance = float(lifecycle["aabb_tolerance_m"])
        frame_obstacles: dict[str, Any] = {}
        total_count = 0
        motion_started_detected = False

        for obstacle in plan.obstacles:
            state = states[obstacle.obstacle_id]
            center = np.asarray(state.position_world_xyz, dtype=np.float64)
            delta = points - center
            cosine = math.cos(obstacle.yaw_rad)
            sine = math.sin(obstacle.yaw_rad)
            local_x = cosine * delta[:, 0] + sine * delta[:, 1]
            local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
            half_size = 0.5 * np.asarray(
                obstacle.size_xyz_m,
                dtype=np.float64,
            )
            inside = (
                (np.abs(local_x) <= half_size[0] + tolerance)
                & (np.abs(local_y) <= half_size[1] + tolerance)
                & (np.abs(delta[:, 2]) <= half_size[2] + tolerance)
            )
            point_count = int(np.count_nonzero(inside))
            total_count += point_count
            motion_started = bool(
                not state.waiting_for_start
                and state.path_distance_m > 1.0e-6
            )
            motion_started_detected = bool(
                motion_started_detected
                or (motion_started and point_count > 0)
            )
            frame_obstacles[obstacle.obstacle_id] = {
                "point_count": point_count,
                "state": state.to_dict(),
            }

            report = lifecycle["obstacles"].get(obstacle.obstacle_id)
            if not isinstance(report, dict):
                raise RuntimeError(
                    f"动态障碍原始点云生命周期缺少 {obstacle.obstacle_id!r}。"
                )
            report["sample_frame_count"] = int(
                report.get("sample_frame_count", 0)
            ) + 1
            if point_count <= 0:
                continue
            detection = {
                "timestamp": float(timestamp),
                "completed_control_step": int(completed_control_step),
                "physics_step_index": physics_step_index,
                "elapsed_time_s": elapsed_time_s,
                "point_count": point_count,
                "state": state.to_dict(),
            }
            report["detected_frame_count"] = int(
                report.get("detected_frame_count", 0)
            ) + 1
            report["maximum_point_count"] = max(
                int(report.get("maximum_point_count", 0)),
                point_count,
            )
            if report.get("first_detection") is None:
                report["first_detection"] = detection
            report["last_detection"] = detection
            report["motion_started_detection_seen"] = bool(
                report.get("motion_started_detection_seen", False)
                or motion_started
            )
            directions = set(report.get("path_directions_detected", []))
            directions.add(int(state.path_direction))
            report["path_directions_detected"] = sorted(directions)
            minimum_distance = report.get("minimum_detected_path_distance_m")
            maximum_distance = report.get("maximum_detected_path_distance_m")
            minimum_distance = (
                float(state.path_distance_m)
                if minimum_distance is None
                else min(float(minimum_distance), float(state.path_distance_m))
            )
            maximum_distance = (
                float(state.path_distance_m)
                if maximum_distance is None
                else max(float(maximum_distance), float(state.path_distance_m))
            )
            report["minimum_detected_path_distance_m"] = minimum_distance
            report["maximum_detected_path_distance_m"] = maximum_distance
            report["detected_path_distance_span_m"] = (
                maximum_distance - minimum_distance
            )

        frame_report = {
            "schema": "dynamic_obstacle_raw_cloud_frame_v1",
            "source": lifecycle["source"],
            "proof_scope": lifecycle["proof_scope"],
            "timestamp": float(timestamp),
            "completed_control_step": int(completed_control_step),
            "physics_step_index": physics_step_index,
            "elapsed_time_s": elapsed_time_s,
            "raw_point_count": int(points.shape[0]),
            "total_obstacle_point_count": total_count,
            "obstacles": frame_obstacles,
        }
        lifecycle["sample_frame_count"] = int(
            lifecycle.get("sample_frame_count", 0)
        ) + 1
        if total_count > 0:
            lifecycle["frames_with_any_obstacle_points"] = int(
                lifecycle.get("frames_with_any_obstacle_points", 0)
            ) + 1
            if lifecycle.get("first_detection") is None:
                lifecycle["first_detection"] = frame_report
            lifecycle["last_detection"] = frame_report
        if motion_started_detected:
            lifecycle["frames_with_motion_started_obstacle_points"] = int(
                lifecycle.get(
                    "frames_with_motion_started_obstacle_points",
                    0,
                )
            ) + 1
        lifecycle["maximum_total_obstacle_point_count"] = max(
            int(lifecycle.get("maximum_total_obstacle_point_count", 0)),
            total_count,
        )
        obstacle_reports = list(lifecycle["obstacles"].values())
        lifecycle["all_configured_obstacles_observed"] = bool(
            obstacle_reports
            and all(
                int(report.get("detected_frame_count", 0)) > 0
                for report in obstacle_reports
            )
        )
        self._metadata["dynamic_obstacle_raw_cloud_last_report"] = frame_report
        self._metadata[
            "dynamic_obstacle_raw_cloud_lifecycle_report"
        ] = lifecycle
        return frame_report

    def _update_dynamic_obstacle_lifecycle_report(
        self,
        *,
        states: tuple[DynamicObstacleState, ...],
        elapsed_time_s: float,
        physics_step_index: int,
        pose_write_count: int,
    ) -> dict[str, Any]:
        """累计真实 PhysX 写入后的运动范围，供最终 live 验收 fail-closed 使用。"""

        plan = getattr(self, "_dynamic_obstacle_plan", DynamicObstaclePlan())
        lifecycle = self._metadata.get("dynamic_obstacle_lifecycle_report")
        expected_ids = {obstacle.obstacle_id for obstacle in plan.obstacles}
        lifecycle_ids = (
            set(lifecycle.get("obstacles", {}))
            if isinstance(lifecycle, dict)
            and isinstance(lifecycle.get("obstacles"), dict)
            else set()
        )
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("schema") != "dynamic_obstacle_lifecycle_v1"
            or lifecycle_ids != expected_ids
        ):
            lifecycle = self._new_dynamic_obstacle_lifecycle_report(
                plan,
                ros_time_offset_s=float(
                    getattr(
                        self,
                        "_navigation_episode_ros_time_offset_s",
                        0.0,
                    )
                ),
            )

        previous_step = lifecycle.get("last_physics_step_index")
        previous_elapsed = lifecycle.get("last_elapsed_time_s")
        if previous_step is not None and int(physics_step_index) < int(previous_step):
            raise RuntimeError("动态障碍 physics step 回退但生命周期未 reset。")
        if previous_elapsed is not None and float(elapsed_time_s) < float(previous_elapsed):
            raise RuntimeError("动态障碍 episode 时间回退但生命周期未 reset。")

        if lifecycle.get("first_physics_step_index") is None:
            lifecycle["first_physics_step_index"] = int(physics_step_index)
            lifecycle["first_elapsed_time_s"] = float(elapsed_time_s)
        lifecycle["last_physics_step_index"] = int(physics_step_index)
        lifecycle["last_elapsed_time_s"] = float(elapsed_time_s)
        lifecycle["pose_write_count"] = int(pose_write_count)
        lifecycle["sample_frame_count"] = int(
            lifecycle.get("sample_frame_count", 0)
        ) + 1

        total_direction_transitions = 0
        maximum_span = 0.0
        for state in states:
            obstacle_report = lifecycle["obstacles"].get(state.obstacle_id)
            if not isinstance(obstacle_report, dict):
                raise RuntimeError(
                    f"动态障碍生命周期缺少配置 id {state.obstacle_id!r}。"
                )
            state_report = state.to_dict()
            if obstacle_report.get("first_state") is None:
                obstacle_report["first_state"] = state_report
            previous_state = obstacle_report.get("last_state")
            if isinstance(previous_state, dict):
                previous_direction = int(previous_state.get("path_direction", 0))
                if previous_direction != int(state.path_direction):
                    obstacle_report["direction_transition_count"] = int(
                        obstacle_report.get("direction_transition_count", 0)
                    ) + 1
            obstacle_report["last_state"] = state_report
            obstacle_report["sample_count"] = int(
                obstacle_report.get("sample_count", 0)
            ) + 1
            minimum_distance = obstacle_report.get("minimum_path_distance_m")
            maximum_distance = obstacle_report.get("maximum_path_distance_m")
            minimum_distance = (
                float(state.path_distance_m)
                if minimum_distance is None
                else min(float(minimum_distance), float(state.path_distance_m))
            )
            maximum_distance = (
                float(state.path_distance_m)
                if maximum_distance is None
                else max(float(maximum_distance), float(state.path_distance_m))
            )
            span = maximum_distance - minimum_distance
            obstacle_report["minimum_path_distance_m"] = minimum_distance
            obstacle_report["maximum_path_distance_m"] = maximum_distance
            obstacle_report["path_distance_span_m"] = span
            directions_seen = set(obstacle_report.get("path_directions_seen", []))
            directions_seen.add(int(state.path_direction))
            obstacle_report["path_directions_seen"] = sorted(directions_seen)
            obstacle_report["waiting_for_start_seen"] = bool(
                obstacle_report.get("waiting_for_start_seen", False)
                or state.waiting_for_start
            )
            obstacle_report["motion_started_seen"] = bool(
                obstacle_report.get("motion_started_seen", False)
                or (
                    not state.waiting_for_start
                    and float(state.path_distance_m) > 1.0e-6
                )
            )
            first_state = obstacle_report["first_state"]
            first_position = first_state.get("position_world_xyz")
            if not isinstance(first_position, list) or len(first_position) != 3:
                raise RuntimeError("动态障碍 first_state 位置证据非法。")
            obstacle_report["maximum_displacement_from_first_m"] = max(
                float(obstacle_report.get("maximum_displacement_from_first_m", 0.0)),
                math.dist(first_position, state.position_world_xyz),
            )
            total_direction_transitions += int(
                obstacle_report["direction_transition_count"]
            )
            maximum_span = max(maximum_span, span)

        obstacle_reports = list(lifecycle["obstacles"].values())
        lifecycle["all_configured_obstacles_sampled"] = bool(
            obstacle_reports
            and all(int(report.get("sample_count", 0)) > 0 for report in obstacle_reports)
        )
        lifecycle["all_configured_obstacles_moved"] = bool(
            obstacle_reports
            and all(
                bool(report.get("motion_started_seen"))
                and float(report.get("path_distance_span_m", 0.0)) > 1.0e-6
                for report in obstacle_reports
            )
        )
        lifecycle["maximum_path_distance_span_m"] = maximum_span
        lifecycle["direction_transition_count"] = total_direction_transitions
        self._metadata["dynamic_obstacle_lifecycle_report"] = lifecycle
        return lifecycle

    def _configure_dynamic_obstacles(
        self,
        env_cfg: Any,
        episode_spec: EpisodeSpec,
        sim_utils: Any,
    ) -> None:
        """把 task JSON 中的平地推车注册为 Isaac Lab kinematic rigid body。"""

        plan = resolve_dynamic_obstacle_plan(episode_spec.raw_task)
        self._dynamic_obstacle_plan = plan
        self._metadata["dynamic_obstacle_configuration_report"] = (
            self._dynamic_obstacle_configuration_metadata(plan)
        )
        self._metadata["dynamic_obstacle_lifecycle_report"] = (
            self._new_dynamic_obstacle_lifecycle_report(
                plan,
                ros_time_offset_s=float(
                    getattr(
                        self,
                        "_navigation_episode_ros_time_offset_s",
                        0.0,
                    )
                ),
            )
        )
        self._metadata["dynamic_obstacle_raw_cloud_lifecycle_report"] = (
            self._new_dynamic_obstacle_raw_cloud_lifecycle_report(plan)
        )
        self._metadata["dynamic_obstacle_raw_cloud_last_report"] = None
        if not plan.enabled:
            return

        from isaaclab.assets import RigidObjectCfg

        for obstacle in plan.obstacles:
            if hasattr(env_cfg.scene, obstacle.scene_asset_name):
                raise RuntimeError(
                    f"Isaac scene asset 名称冲突：{obstacle.scene_asset_name}"
                )
            rigid_object_cfg = RigidObjectCfg(
                prim_path=obstacle.prim_path,
                spawn=sim_utils.CuboidCfg(
                    size=obstacle.size_xyz_m,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                        max_depenetration_velocity=1.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=obstacle.mass_kg,
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8,
                        dynamic_friction=0.6,
                        restitution=0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=obstacle.color_rgb,
                        opacity=1.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=obstacle.waypoints_world_xyz[0],
                    rot=obstacle.orientation_world_wxyz,
                    lin_vel=(0.0, 0.0, 0.0),
                    ang_vel=(0.0, 0.0, 0.0),
                ),
            )
            setattr(env_cfg.scene, obstacle.scene_asset_name, rigid_object_cfg)

    def _write_dynamic_obstacle_poses(
        self,
        *,
        elapsed_time_s: float,
        physics_step_index: int,
        reason: str,
    ) -> dict[str, Any]:
        """把同一仿真时刻的全部动态障碍 WXYZ 目标写入 PhysX。"""

        plan = getattr(self, "_dynamic_obstacle_plan", DynamicObstaclePlan())
        if not plan.enabled:
            self._metadata["dynamic_obstacle_lifecycle_report"] = (
                self._new_dynamic_obstacle_lifecycle_report(
                    plan,
                    ros_time_offset_s=float(
                        getattr(
                            self,
                            "_navigation_episode_ros_time_offset_s",
                            0.0,
                        )
                    ),
                )
            )
            report = {
                "enabled": False,
                "reason": "configuration_disabled",
                "physics_step_index": int(physics_step_index),
                "elapsed_time_s": float(elapsed_time_s),
                "obstacles": [],
            }
            self._metadata["dynamic_obstacle_runtime_report"] = report
            return report

        states = plan.state_at(elapsed_time_s)
        state_reports: list[dict[str, Any]] = []
        for state in states:
            try:
                asset = self._runtime.scene[state.scene_asset_name]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Isaac scene 缺少动态障碍 asset {state.scene_asset_name!r}。"
                ) from exc
            try:
                root_pose = asset.data.root_pose_w.clone()
                shape = tuple(int(value) for value in root_pose.shape)
                if len(shape) != 2 or shape[0] < 1 or shape[1] < 7:
                    raise ValueError(f"root_pose_w shape 非法：{shape}")
                root_pose[:, :7] = root_pose.new_tensor(state.root_pose_wxyz())
                asset.write_root_pose_to_sim(root_pose)
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"动态障碍 {state.obstacle_id!r} 的 kinematic 位姿写入失败。"
                ) from exc
            state_reports.append(state.to_dict())

        write_count = int(
            self._metadata.get("dynamic_obstacle_pose_write_count", 0)
        ) + len(states)
        self._metadata["dynamic_obstacle_pose_write_count"] = write_count
        lifecycle_report = self._update_dynamic_obstacle_lifecycle_report(
            states=states,
            elapsed_time_s=elapsed_time_s,
            physics_step_index=physics_step_index,
            pose_write_count=write_count,
        )
        report = {
            "enabled": True,
            "reason": str(reason),
            "physics_step_index": int(physics_step_index),
            "elapsed_time_s": float(elapsed_time_s),
            "obstacle_count": len(states),
            "pose_write_count": write_count,
            "obstacles": state_reports,
            "root_lock_state_used": False,
            "time_source": "episode_physics_step_index_x_physics_dt",
            "lifecycle_schema": lifecycle_report["schema"],
        }
        self._metadata["dynamic_obstacle_runtime_report"] = report
        return report

    def _advance_dynamic_obstacles_for_physics_step(self) -> None:
        """在每个 physics 子步前写入该子步末端的确定性 kinematic 目标。"""

        plan = getattr(self, "_dynamic_obstacle_plan", DynamicObstaclePlan())
        if not plan.enabled:
            return
        physics_step_index = int(self._runtime._sim_step_counter)
        physics_dt = float(self._runtime.physics_dt)
        if physics_step_index < 1 or not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise RuntimeError("动态障碍要求正 physics step index 与有限正 physics_dt。")
        self._write_dynamic_obstacle_poses(
            elapsed_time_s=float(physics_step_index) * physics_dt,
            physics_step_index=physics_step_index,
            reason="before_scene_write_data_to_sim",
        )

    def _configure_env(self, env_cfg: Any, episode_spec: EpisodeSpec, sim_utils: Any) -> None:
        from isaaclab.sensors import CameraCfg
        from isaaclab.terrains import TerrainImporterCfg
        from source.navigation.adapters.terrain_utils import write_collision_terrain_wrapper

        if self._config.render_antialiasing_mode is not None:
            mode = str(self._config.render_antialiasing_mode)
            if mode not in {"Off", "FXAA", "DLSS", "TAA", "DLAA"}:
                raise ValueError(f"不支持的渲染抗锯齿模式：{mode}")
            env_cfg.sim.render.antialiasing_mode = mode
            self._metadata["render_antialiasing_report"] = {
                "configured": True,
                "mode": mode,
                "source": "scene_profile",
            }

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
        render_control_interval = (
            self._effective_camera_render_interval_control_steps()
        )
        env_cfg.sim.render_interval = int(env_cfg.decimation) * int(
            render_control_interval
        )
        self._metadata["camera_render_schedule"] = {
            "control_interval_steps": int(render_control_interval),
            "rgb_capture_interval_steps": int(
                self._config.camera_render_interval_control_steps
            ),
            "physics_interval_steps": int(env_cfg.sim.render_interval),
            "control_dt": float(env_cfg.sim.dt) * int(env_cfg.decimation),
            "render_hz": 1.0
            / (float(env_cfg.sim.dt) * float(env_cfg.sim.render_interval)),
            "physics_dt_unchanged": float(env_cfg.sim.dt),
            "decimation_unchanged": int(env_cfg.decimation),
        }
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
        self._configure_dynamic_obstacles(
            env_cfg,
            episode_spec,
            sim_utils,
        )
        default_root_pos = tuple(float(value) for value in env_cfg.scene.robot.init_state.pos)
        self._default_robot_root_pos = default_root_pos
        reset_params, reset_report = _episode_reset_pose_configuration(
            episode_spec,
            default_root_pos=default_root_pos,
        )
        env_cfg.events.randomize_reset_base.params = reset_params
        self._metadata["episode_reset_pose_request"] = reset_report
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
        if self._front_camera_sensor_enabled():
            _validate_d436_camera_calibration_resolution(
                "front",
                self._config.front_camera_width,
                self._config.front_camera_height,
            )
            front_data_types = []
            if self._config.enable_front_camera:
                front_data_types.append("rgb")
            if self._navigation_depth_camera_enabled():
                front_data_types.append("distance_to_image_plane")
            # 保留 DWA/play_nav_cs.py 的安装外参；内参改用 D436 640x480 标定值。
            env_cfg.scene.head_camera = CameraCfg(
                prim_path=FRONT_CAMERA_PRIM_PATH,
                update_period=0.0,
                height=self._config.front_camera_height,
                width=self._config.front_camera_width,
                data_types=front_data_types,
                depth_clipping_behavior="none",
                # 默认 False 会让随机器人运动的相机继续暴露初始化位姿，进而把
                # 后续深度云错误变换到旧位置。
                update_latest_camera_pose=self._navigation_depth_camera_enabled(),
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
                "data_types": list(front_data_types),
                "rgb_recording_enabled": self._config.enable_front_camera,
                "navigation_depth_enabled": self._navigation_depth_camera_enabled(),
                "update_latest_camera_pose": self._navigation_depth_camera_enabled(),
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
        if self._config.enable_relocatable_episode_supports:
            from source.simulation.task_scene_pose import (
                configure_task_supports_for_stage_reuse,
            )

            relocatable_support_report = configure_task_supports_for_stage_reuse(
                stage,
                episode_spec.raw_task,
            )
        else:
            relocatable_support_report = {
                "enabled": False,
                "configured_count": 0,
                "reason": "single_episode_or_stage_reuse_disabled",
            }
        self._metadata["relocatable_episode_support_report"] = (
            relocatable_support_report
        )
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
            "relocatable_episode_support_report": relocatable_support_report,
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

        from source.simulation.lighting import (
            configure_scene_lighting,
            resolve_scene_light_mode,
        )

        requested_mode = str(self._config.scene_light_mode).lower()
        resolved_mode = resolve_scene_light_mode(
            requested_mode,
            scene_visual_enabled=bool(self._config.enable_scene_visual),
        )

        report = configure_scene_lighting(
            stage=stage,
            mode=resolved_mode,
            camera_light_name=self._config.camera_light_name,
            camera_light_intensity=self._config.camera_light_intensity,
            camera_light_radius=self._config.camera_light_radius,
        )
        report["reason"] = reason
        report["requested_mode"] = requested_mode
        report["resolved_mode"] = resolved_mode
        report["scene_visual_enabled"] = bool(self._config.enable_scene_visual)
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
        """只保留任务物体，并停用非任务物体的渲染与物理。"""

        from pxr import Usd, UsdGeom

        object_prim_path = episode_spec.object_prim_path
        if not object_prim_path:
            self._hidden_distractor_root_paths = ()
            return {"applied": False, "reason": "object_prim_path_missing"}

        object_prim = stage.GetPrimAtPath(object_prim_path)
        if object_prim.IsValid() and not object_prim.IsActive():
            object_prim.SetActive(True)

        object_prefix = object_prim_path.rstrip("/") + "/"
        keywords = ("apple", "orange", "bottle")
        candidate_roots: list[str] = []
        hidden_paths: list[str] = []
        shown_paths: list[str] = []
        deactivated_roots: list[str] = []

        # 第二次调用发生在 collision patch 之后；TraverseAll 才能重新发现
        # 首次调用中已经停用的干扰物，并持续保留规划排除根路径。
        for prim in stage.TraverseAll():
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
            if root_prim.IsActive():
                root_prim.SetActive(False)
            if not root_prim.IsActive():
                deactivated_roots.append(root_path)

        for prim in stage.TraverseAll():
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
            "deactivated_root_paths": deactivated_roots,
            "distractor_physics_disabled": len(deactivated_roots) == len(hidden_root_paths),
            "planner_collision_exclusion_enabled": True,
        }

    def refresh_viewport(self, *, reason: str = "manual") -> dict[str, Any]:
        """重试配置 GUI viewpoint；只影响显示，不推进物理。"""

        return self._configure_viewport(reason=reason)

    def _navigation_depth_camera_enabled(self) -> bool:
        """是否为 ROS 2 导航请求了在线前视深度。"""

        return self._config.depth_point_cloud_config is not None

    def _front_camera_sensor_enabled(self) -> bool:
        """RGB 记录或导航深度任一启用时都必须创建 head_camera。"""

        return bool(
            self._config.enable_front_camera
            or self._navigation_depth_camera_enabled()
        )

    def _effective_camera_render_interval_control_steps(self) -> int:
        """返回同时覆盖 RGB 与点云采样网格的最小固定渲染节拍。"""

        interval = int(self._config.camera_render_interval_control_steps)
        cloud_config = self._config.depth_point_cloud_config
        if cloud_config is not None:
            # 只取较小间隔会漏掉另一网格：例如 RGB=10、点云=4 时，
            # 每 4 步渲染无法覆盖第 10 步。最大公约数可覆盖两者全部时刻。
            interval = math.gcd(
                interval,
                int(cloud_config.publish_interval_control_steps),
            )
        return interval

    def _camera_sensors_enabled(self) -> bool:
        return bool(
            self._front_camera_sensor_enabled()
            or self._config.enable_wrist_camera
            or self._config.enable_overview_camera
        )

    def _mark_camera_render(
        self,
        *,
        valid_state_step: int | None,
        reason: str,
    ) -> None:
        """Bind the latest RTX render to the exact state step it represents."""

        self._camera_render_generation += 1
        self._last_camera_render_step = (
            None if valid_state_step is None else int(valid_state_step)
        )
        self._last_camera_render_reason = str(reason)

    def _render_without_physics(
        self,
        *,
        valid_state_step: int | None,
        reason: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run one Kit/RTX update while physics stepping is disabled.

        IsaacLab's ``SimulationContext.render`` temporarily sets
        ``/app/player/playSimulations`` to false around ``app.update()``.  The
        before/after counters below make that no-task-physics contract auditable.
        """

        if self._runtime is None:
            return {"applied": False, "reason": "runtime_unavailable"}
        if not force and not self._camera_sensors_enabled():
            self._last_camera_render_step = None
            self._last_camera_render_reason = None
            return {"applied": False, "reason": "camera_sensors_disabled"}
        step_calls_before = int(self._step_calls)
        sim_step_before = int(self._runtime._sim_step_counter)
        started_at = time.perf_counter()
        self._runtime.sim.render()
        wall_seconds = time.perf_counter() - started_at
        step_calls_after = int(self._step_calls)
        sim_step_after = int(self._runtime._sim_step_counter)
        if step_calls_after != step_calls_before or sim_step_after != sim_step_before:
            raise RuntimeError(
                "no-physics render advanced simulation counters: "
                f"control={step_calls_before}->{step_calls_after}, "
                f"physics={sim_step_before}->{sim_step_after}"
            )
        self._mark_camera_render(
            valid_state_step=valid_state_step,
            reason=reason,
        )
        profiler = self._performance_profiler
        if profiler is not None:
            profiler.record("runtime.rtx_render_nonphysics_sync", wall_seconds)
        return {
            "applied": True,
            "reason": str(reason),
            "render_generation": int(self._camera_render_generation),
            "valid_state_step": valid_state_step,
            "control_step_before_after": [step_calls_before, step_calls_after],
            "physics_step_before_after": [sim_step_before, sim_step_after],
            "physics_time_advanced": False,
            "wall_seconds": wall_seconds,
        }

    def _retry_viewport_after_stage_updates(self) -> None:
        """IsaacLab 创建窗口和 sublayer 解析可能滞后，前几帧允许轻量重试。"""

        if not self._config.auto_manage_viewport_camera:
            return
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
            apply_camera=self._config.auto_manage_viewport_camera,
        )
        report["configure_reason"] = reason
        report["configure_attempt"] = self._viewport_config_attempts
        self._metadata["viewport_report"] = report
        selected_camera = report.get("selected_camera_prim_path")
        if selected_camera and report.get("camera_applied") is True:
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

    def _episode_support_pose_settings(
        self,
        episode_spec: EpisodeSpec,
    ) -> dict[str, dict[str, Any]]:
        from source.simulation.task_scene_pose import (
            resolve_task_pick_support_pose,
            resolve_task_receptacle_pose,
        )

        return {
            "pick": resolve_task_pick_support_pose(episode_spec.raw_task),
            "place": resolve_task_receptacle_pose(episode_spec.raw_task),
        }

    def _initialize_episode_support_readers(
        self,
        episode_spec: EpisodeSpec,
    ) -> None:
        """Bind tensor views for support roots made kinematic before PhysX start."""

        self._episode_support_bodies = {}
        if not self._config.enable_relocatable_episode_supports:
            self._metadata["episode_support_reader_report"] = {
                "enabled": False,
                "reader_count": 0,
                "reason": "relocatable_episode_supports_disabled",
            }
            return

        import omni.usd
        from isaacsim.core.prims import SingleRigidPrim
        from source.simulation.task_scene_pose import (
            inspect_episode_static_support_body_mode,
        )

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "Isaac stage is unavailable while initializing support readers"
            )
        reports: dict[str, Any] = {}
        for role, settings in self._episode_support_pose_settings(
            episode_spec
        ).items():
            if settings.get("configured") is not True:
                reports[role] = {
                    "configured": False,
                    "reason": settings.get("reason"),
                }
                continue
            prim_path = str(settings["prim_path"])
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.IsActive():
                raise RuntimeError(
                    f"episode support prim is unavailable: {prim_path}"
                )
            body_mode_report = inspect_episode_static_support_body_mode(prim)
            if body_mode_report["support_body_mode"] != "kinematic_episode_static":
                raise RuntimeError(
                    "relocatable episode support is not kinematic before PhysX "
                    f"initialization: role={role} report={body_mode_report}"
                )
            reader = SingleRigidPrim(
                prim_path=prim_path,
                name=f"full_physics_{role}_episode_support",
                reset_xform_properties=False,
            )
            reader.initialize()
            initial_position_raw, initial_quaternion_raw = reader.get_world_pose()
            initial_position = tuple(
                float(value) for value in _as_tuple(initial_position_raw)
            )
            initial_quaternion = tuple(
                float(value) for value in _as_tuple(initial_quaternion_raw)
            )
            self._episode_support_bodies[role] = {
                "reader": reader,
                "prim_path": prim_path,
                "target_position_xyz": initial_position,
                "target_quaternion_wxyz": initial_quaternion,
            }
            reports[role] = {
                "configured": True,
                "prim_path": prim_path,
                "initial_pose_xyz_wxyz": [
                    *initial_position,
                    *initial_quaternion,
                ],
                **body_mode_report,
            }
        self._metadata["episode_support_reader_report"] = {
            "enabled": True,
            "reader_count": len(self._episode_support_bodies),
            "supports": reports,
        }

    def _write_episode_support_physics_poses(
        self,
        episode_spec: EpisodeSpec,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Move kinematic supports through PhysX tensors without advancing time."""

        if not self._config.enable_relocatable_episode_supports:
            return {
                "applied": False,
                "verified": True,
                "reason": "relocatable_episode_supports_disabled",
            }
        if not self._episode_support_bodies:
            raise RuntimeError(
                "relocatable episode supports are enabled but no tensor readers exist"
            )

        import torch

        device = getattr(self._runtime, "device", "cpu")
        reports: dict[str, Any] = {}
        for role, settings in self._episode_support_pose_settings(
            episode_spec
        ).items():
            if settings.get("configured") is not True:
                continue
            body = self._episode_support_bodies.get(role)
            if body is None:
                raise RuntimeError(
                    f"missing live support tensor reader for role={role}"
                )
            reader = body["reader"]
            current_position_raw, current_quaternion_raw = reader.get_world_pose()
            current_position = tuple(
                float(value) for value in _as_tuple(current_position_raw)
            )
            current_quaternion = tuple(
                float(value) for value in _as_tuple(current_quaternion_raw)
            )
            pose = settings["pose_world"]
            target_position = (
                float(pose["x"]),
                float(pose["y"]),
                float(pose["z"]),
            )
            target_quaternion = (
                current_quaternion
                if settings.get("translation_only")
                else _quat_wxyz_from_rpy(
                    float(pose["roll"]),
                    float(pose["pitch"]),
                    float(pose["yaw"]),
                )
            )
            rigid_view = getattr(reader, "_rigid_prim_view", None)
            if rigid_view is None or not hasattr(rigid_view, "set_world_poses"):
                raise RuntimeError(
                    f"support reader lacks tensor pose API: role={role}"
                )
            rigid_view.set_world_poses(
                positions=torch.tensor(
                    [target_position],
                    dtype=torch.float32,
                    device=device,
                ),
                orientations=torch.tensor(
                    [target_quaternion],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            body["target_position_xyz"] = target_position
            body["target_quaternion_wxyz"] = target_quaternion
            actual_position_raw, actual_quaternion_raw = reader.get_world_pose()
            actual_position = tuple(
                float(value) for value in _as_tuple(actual_position_raw)
            )
            actual_quaternion = tuple(
                float(value) for value in _as_tuple(actual_quaternion_raw)
            )
            position_error = math.sqrt(
                sum(
                    (actual - expected) ** 2
                    for actual, expected in zip(actual_position, target_position)
                )
            )
            orientation_error = _quat_angle_error_rad(
                actual_quaternion,
                target_quaternion,
            )
            reports[role] = {
                "prim_path": body["prim_path"],
                "previous_pose_xyz_wxyz": [
                    *current_position,
                    *current_quaternion,
                ],
                "target_pose_xyz_wxyz": [
                    *target_position,
                    *target_quaternion,
                ],
                "actual_pose_xyz_wxyz": [
                    *actual_position,
                    *actual_quaternion,
                ],
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "verified": bool(
                    position_error <= 1.0e-4
                    and orientation_error <= 1.0e-4
                ),
            }
        verified = bool(reports) and all(
            report["verified"] is True for report in reports.values()
        )
        result = {
            "applied": True,
            "verified": verified,
            "reason": reason,
            "physics_time_advanced": False,
            "supports": reports,
        }
        if not verified:
            raise RuntimeError(f"episode support tensor pose write failed: {result}")
        return result

    def _episode_support_pose_diagnostic(self, *, label: str) -> dict[str, Any]:
        if not self._config.enable_relocatable_episode_supports:
            return {
                "available": False,
                "verified": True,
                "label": label,
                "reason": "relocatable_episode_supports_disabled",
            }
        reports: dict[str, Any] = {}
        for role, body in self._episode_support_bodies.items():
            position_raw, quaternion_raw = body["reader"].get_world_pose()
            position = tuple(float(value) for value in _as_tuple(position_raw))
            quaternion = tuple(float(value) for value in _as_tuple(quaternion_raw))
            target_position = tuple(body["target_position_xyz"])
            target_quaternion = tuple(body["target_quaternion_wxyz"])
            position_error = math.sqrt(
                sum(
                    (actual - expected) ** 2
                    for actual, expected in zip(position, target_position)
                )
            )
            orientation_error = _quat_angle_error_rad(
                quaternion,
                target_quaternion,
            )
            reports[role] = {
                "prim_path": body["prim_path"],
                "target_pose_xyz_wxyz": [
                    *target_position,
                    *target_quaternion,
                ],
                "actual_pose_xyz_wxyz": [*position, *quaternion],
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "verified": bool(
                    position_error <= 1.0e-4
                    and orientation_error <= 1.0e-4
                ),
            }
        verified = bool(reports) and all(
            report["verified"] is True for report in reports.values()
        )
        return {
            "available": bool(reports),
            "verified": verified,
            "label": label,
            "supports": reports,
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

    def _object_initialization_target_world_pose(
        self,
        episode_spec: EpisodeSpec,
        *,
        pose_report: dict[str, Any] | None = None,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ]:
        if episode_spec.object_initial_pose is None:
            raise RuntimeError("object initialization target pose is unavailable")
        x, y, z, roll, pitch, yaw = episode_spec.object_initial_pose
        report = pose_report or self._metadata.get("object_pose_setup_report") or {}
        authored_quaternion = report.get("authored_world_quaternion_wxyz")
        if isinstance(authored_quaternion, (list, tuple)) and len(
            authored_quaternion
        ) >= 4:
            world_quaternion = tuple(
                float(authored_quaternion[index]) for index in range(4)
            )
        else:
            world_quaternion = _quat_wxyz_from_rpy(roll, pitch, yaw)
        return (
            (float(x), float(y), float(z)),
            world_quaternion,
        )

    def _write_object_physics_state(
        self,
        *,
        position_xyz: tuple[float, float, float],
        quaternion_wxyz: tuple[float, float, float, float],
        velocity_xyz_rpy: tuple[float, float, float, float, float, float],
    ) -> dict[str, Any]:
        """Write the live PhysX state without rewriting authored USD xform ops."""

        if self._object is None:
            return {"applied": False, "reason": "object_reader_unavailable"}
        rigid_view = getattr(self._object, "_rigid_prim_view", None)
        if rigid_view is None or not hasattr(rigid_view, "set_world_poses"):
            raise RuntimeError("SingleRigidPrim 缺少 GPU world pose 写入接口。")
        if not hasattr(rigid_view, "set_velocities"):
            raise RuntimeError("SingleRigidPrim 缺少 GPU 合并速度写入接口。")
        import torch

        device = getattr(self._runtime, "device", "cpu")
        rigid_view.set_world_poses(
            positions=torch.tensor(
                [position_xyz],
                dtype=torch.float32,
                device=device,
            ),
            orientations=torch.tensor(
                [quaternion_wxyz],
                dtype=torch.float32,
                device=device,
            ),
        )
        rigid_view.set_velocities(
            torch.tensor(
                [velocity_xyz_rpy],
                dtype=torch.float32,
                device=device,
            )
        )
        return {
            "applied": True,
            "position_xyz": list(position_xyz),
            "quaternion_wxyz": list(quaternion_wxyz),
            "velocity_xyz_rpy": list(velocity_xyz_rpy),
            "pose_write_api": "RigidPrim.set_world_poses_physics_tensor",
            "usd_xform_ops_modified": False,
        }

    def _stabilize_object_initialization_pose(
        self,
        episode_spec: EpisodeSpec,
        *,
        timing: str,
        preserve_vertical_velocity: bool,
    ) -> dict[str, Any]:
        if self._object is None:
            return {"applied": False, "reason": "object_reader_unavailable"}
        requested_position, requested_quaternion = (
            self._object_initialization_target_world_pose(episode_spec)
        )
        current_position_raw, current_quaternion_raw = self._object.get_world_pose()
        current_position = tuple(float(value) for value in _as_tuple(current_position_raw))
        current_quaternion = tuple(
            float(value) for value in _as_tuple(current_quaternion_raw)
        )
        current_linear_velocity = tuple(
            float(value) for value in _as_tuple(self._object.get_linear_velocity())
        )
        target_position = (
            requested_position[0],
            requested_position[1],
            current_position[2],
        )
        target_velocity = (
            0.0,
            0.0,
            current_linear_velocity[2] if preserve_vertical_velocity else 0.0,
            0.0,
            0.0,
            0.0,
        )
        write_report = self._write_object_physics_state(
            position_xyz=target_position,
            quaternion_wxyz=requested_quaternion,
            velocity_xyz_rpy=target_velocity,
        )
        apply_count = int(
            self._metadata.get(
                "object_initialization_pose_stabilization_apply_count",
                0,
            )
        ) + 1
        report = {
            **write_report,
            "timing": timing,
            "apply_count": apply_count,
            "requested_position_xyz": list(requested_position),
            "pose_before": [*current_position, *current_quaternion],
            "preserved_current_z": True,
            "preserved_vertical_velocity": bool(preserve_vertical_velocity),
            "initialization_only": True,
        }
        self._metadata.update(
            {
                "used_object_initialization_pose_stabilization": True,
                "object_initialization_pose_stabilization_apply_count": apply_count,
                "last_object_initialization_pose_stabilization_report": report,
            }
        )
        return report

    def _apply_object_initialization_pose_stabilization(
        self,
        action: RobotAction,
    ) -> None:
        if action.metadata.get("object_settle_active") is not True:
            return
        if self._episode_spec is None:
            return
        policy = resolve_object_initialization_policy(self._episode_spec.raw_task)
        if not policy.get("enabled") or not policy.get(
            "stabilize_xy_and_orientation_during_settle"
        ):
            return
        stabilization_report = self._stabilize_object_initialization_pose(
            self._episode_spec,
            timing="before_object_settle_physics_step",
            preserve_vertical_velocity=True,
        )
        dynamic_steps = int(policy["dynamic_settle_steps_before_sleep"])
        if int(stabilization_report.get("apply_count", 0)) < dynamic_steps:
            return
        sleep_report = self._set_object_sleeping(enabled=True)
        stabilization_report.update(
            {
                "dynamic_settle_steps_before_sleep": dynamic_steps,
                "sleep_after_dynamic_settle": sleep_report,
                "supported_pose_frozen_until_contact": True,
            }
        )
        self._metadata[
            "last_object_initialization_pose_stabilization_report"
        ] = stabilization_report

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
        initialization_policy = resolve_object_initialization_policy(
            episode_spec.raw_task
        )
        stabilization_report: dict[str, Any] | None = None
        if initialization_policy.get("enabled") and initialization_policy.get(
            "stabilize_xy_and_orientation_during_settle"
        ):
            stabilization_report = self._stabilize_object_initialization_pose(
                episode_spec,
                timing="finalize_object_settle",
                preserve_vertical_velocity=False,
            )
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
            "object_initialization_policy": initialization_policy,
            "initialization_pose_stabilization_report": stabilization_report,
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
        initialization_policy = resolve_object_initialization_policy(
            episode_spec.raw_task
        )
        # ManagerBasedEnv.reset() only resets bodies owned by the IsaacLab scene.
        # Task objects referenced directly from the stage (for example the
        # multi-floor apple) keep the live PhysX pose reached during simulation
        # startup.  Re-authoring USD xform ops above is therefore insufficient:
        # Fabric can publish that stale live pose again on the first render.
        #
        # Restoring the episode pose is a reset contract for every task object,
        # independent of the optional supported-upright stabilization policy.
        # The physics-tensor write preserves the authored unitsResolve xform
        # stack and is still classified as episode setup, not navigation-time
        # object teleportation.
        live_pose_write_report = self._write_object_physics_state(
            position_xyz=target_position,
            quaternion_wxyz=world_quaternion,
            velocity_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
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
            "object_initialization_policy": initialization_policy,
            "live_pose_write_applied": bool(
                live_pose_write_report
                and live_pose_write_report.get("applied") is True
            ),
            "live_pose_write_report": live_pose_write_report,
            "live_pose_write_skipped": False,
            "live_pose_write_skip_reason": None,
            "live_pose_write_contract": "episode_reset_all_task_objects_v1",
            "policy_restore_pose_after_runtime_reset": bool(
                initialization_policy.get("enabled")
                and initialization_policy.get("restore_pose_after_runtime_reset")
            ),
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
        task_manipulation = (episode_spec.raw_task or {}).get(
            "manipulation_execution"
        )
        release_clearance_min_m = self._config.place_release_clearance_min_m
        if task_manipulation is not None:
            if not isinstance(task_manipulation, dict):
                raise RuntimeError("task.manipulation_execution 必须是对象")
            if "place_release_clearance_min_m" in task_manipulation:
                release_clearance_min_m = float(
                    task_manipulation["place_release_clearance_min_m"]
                )
                if (
                    not math.isfinite(release_clearance_min_m)
                    or release_clearance_min_m < 0.0
                ):
                    raise RuntimeError(
                        "task.manipulation_execution."
                        "place_release_clearance_min_m 必须是有限非负数"
                    )
        payload["release_clearance"] = max(
            float(payload.get("release_clearance", 0.0)),
            release_clearance_min_m,
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
        arm_velocity_hold = bool(
            metadata.get("segment_type") == "post_motion_hold"
            or metadata.get("arm_velocity_hold") is True
        )
        self._adapter.set_arm_joint_target(
            target,
            hold_velocity=arm_velocity_hold,
        )
        self._pending_arm_tracking_target = {
            "source": action.source,
            "pipeline_state": (
                metadata.get("carry_arm_home_phase")
                or metadata.get("navigation_arm_stow_phase")
            ),
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
        """Expose one camera set per render-grid step and cache duplicate reads."""

        started_at = time.perf_counter()
        if self._runtime is None:
            return {}
        interval = int(self._config.camera_render_interval_control_steps)
        step_calls = int(getattr(self, "_step_calls", 0))
        if step_calls % interval != 0:
            return {}
        if getattr(self, "_cached_camera_step", None) == step_calls:
            return dict(getattr(self, "_cached_camera_images", {}))
        render_step = self._last_camera_render_step
        if render_step != step_calls:
            # Never relabel a pre-reset or older render as the current state.  The
            # recorder may simply skip this control step; it must not receive a
            # visually stale packet with a fresh logical timestamp.
            self._metadata["camera_capture_report"] = {
                "requested_camera_keys": [
                    key
                    for key, enabled in (
                        ("front", self._config.enable_front_camera),
                        ("wrist", self._config.enable_wrist_camera),
                        ("overview", self._config.enable_overview_camera),
                    )
                    if enabled
                ],
                "available_camera_keys": [],
                "missing_camera_keys": [],
                "capture_step_index": step_calls,
                "render_step_index": render_step,
                "render_generation": int(self._camera_render_generation),
                "render_reason": self._last_camera_render_reason,
                "accepted": False,
                "reason": "stale_or_unrendered_state_rejected",
                "synchronization_source": "explicit_render_state_step_contract",
            }
            self._cached_camera_step = step_calls
            self._cached_camera_images = {}
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
                "capture_step_index": step_calls,
                "render_step_index": render_step,
                "render_generation": int(self._camera_render_generation),
                "render_reason": self._last_camera_render_reason,
                "capture_timestamp": (
                    float(step_calls) * float(getattr(self._runtime, "step_dt", 0.02))
                ),
                "render_interval_control_steps": interval,
                "accepted": bool(images),
                "synchronization_source": "explicit_render_state_step_contract",
            }
        self._cached_camera_step = step_calls
        self._cached_camera_images = dict(images)
        if getattr(self, "_performance_profiler", None) is not None:
            self._performance_profiler.record(
                "runtime.camera_tensor_read",
                time.perf_counter() - started_at,
                work_units=len(images),
            )
        return images

    def _resolve_checkpoint(self) -> Path:
        candidates = []
        if self._config.checkpoint is not None:
            candidates.append(self._config.checkpoint)
        candidates.extend(
            [
                self._project_root / "checkpoints/go2_x5/flat/model_8500.pt",
            ]
        )
        external_checkpoint = os.environ.get("GO2_X5_FLAT_CHECKPOINT")
        if external_checkpoint:
            external_path = Path(external_checkpoint).expanduser()
            candidates.insert(
                1,
                external_path
                if external_path.is_absolute()
                else self._project_root / external_path,
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
