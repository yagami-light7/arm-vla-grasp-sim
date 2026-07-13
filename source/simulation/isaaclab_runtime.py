"""由 pipeline 主循环驱动的 Isaac Lab 单环境 runtime。"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.interfaces import EpisodeSpec, RobotAction, SimulationState


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
    enable_scene_visual: bool = True
    viewport_camera_prim_path: str = "/World/Camera0"
    auto_manage_viewport_camera: bool = True
    hide_navigation_collision_visual: bool = True
    scene_light_mode: str = "camera"
    camera_light_intensity: float = 3500.0
    camera_light_radius: float = 2.0
    camera_light_name: str = "camera_light"
    # 数据相机严格对齐 DWA：Go2 头部前视、480x640 RGB、每个控制步更新。
    enable_front_camera: bool = False
    front_camera_height: int = 480
    front_camera_width: int = 640
    # 末端相机外参与 DWA ground-pick 环境保持一致；输出分辨率由 recorder 配置统一。
    enable_wrist_camera: bool = False
    wrist_camera_height: int = 480
    wrist_camera_width: int = 640
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


def _collision_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, str]:
    """优先保留任务物体附近的碰撞体，避免 stage 遍历顺序挤掉桌面。"""

    prim_path = str(candidate["prim_path"])
    table_priority = 0 if any(
        keyword in prim_path.lower()
        for keyword in ("table", "tabletop", "desk", "counter")
    ) else 1
    return (
        table_priority,
        float(candidate["distance_to_reference_xy_m"]),
        prim_path,
    )


def _retarget_height_scanners(scene_cfg: Any, terrain_prim_path: str) -> tuple[str, ...]:
    """让地形高度扫描器跟随 runtime 实际导入的碰撞地形。"""

    updated: list[str] = []
    for sensor_name in ("height_scanner", "height_scanner_base"):
        sensor_cfg = getattr(scene_cfg, sensor_name, None)
        if sensor_cfg is None:
            continue
        sensor_cfg.mesh_prim_paths = [terrain_prim_path]
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
        if any(
            keyword in str(cuboid.get("prim_path") or "").lower()
            for keyword in table_keywords
        )
    ]
    return {
        "collision_cuboids_table_present": bool(table_paths),
        "table_collision_prim_paths": table_paths,
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
            tcp_pose=self._read_tcp_pose(),
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
        self._update_velocity_command_visualization(action)
        self._runtime.action_manager.process_action(policy_action.to(self._runtime.device))
        self._last_action = action
        self._metadata["last_arm_action_report"] = arm_report
        self._metadata["last_gripper_action_report"] = gripper_report
        self._record_joint_action_apply(action, arm_report, gripper_report)
        self._action_prepared = True

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
                    )
                    else None
                ),
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
        stage_report = self._validate_scene_collision(scene_usd, self._config.terrain_prim_path)
        self._metadata["stage_report"] = stage_report
        terrain_usd = write_collision_terrain_wrapper(
            scene_usd,
            self._config.terrain_prim_path,
            floor_proxy_profile=self._config.collision_floor_proxy_profile,
        )
        self._metadata["collision_floor_proxy_report"] = {
            "profile": self._config.collision_floor_proxy_profile,
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
        updated_height_scanners = _retarget_height_scanners(
            env_cfg.scene,
            env_cfg.scene.terrain.prim_path,
        )
        self._metadata["height_scanner_terrain_report"] = {
            "terrain_prim_path": env_cfg.scene.terrain.prim_path,
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
            # 与 DWA/play_nav_cs.py 使用同一相机内参、外参和 ROS optical frame 约定。
            env_cfg.scene.head_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base/head_cam",
                update_period=0.0,
                height=self._config.front_camera_height,
                width=self._config.front_camera_width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.28, 0.0, 0.07),
                    rot=(0.5, -0.5, 0.5, -0.5),
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
            }
        if self._config.enable_wrist_camera:
            # 对齐 DWA ground-pick 的 arm_camera：挂在稳定存在的 arm_link6，
            # 相机原点于夹爪中心略微偏移，沿末端局部 +X 方向观察。
            env_cfg.scene.arm_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera",
                update_period=0.0,
                height=self._config.wrist_camera_height,
                width=self._config.wrist_camera_width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.03, 5.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.10),
                    rot=(0.353553, -0.612372, 0.612372, -0.353553),
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
                "source": "dwa_ground_pick_arm_camera",
            }

    def _load_visual_scene(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        import omni.usd
        from source.navigation.adapters.terrain_utils import write_visual_sublayer_wrapper

        wrapper = write_visual_sublayer_wrapper(
            self._resolve_path(episode_spec.scene_usd),
            self._config.visual_prim_path,
            excluded_prim_paths=(
                self._config.terrain_prim_path,
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
            "visual_prim_path": self._config.visual_prim_path,
            "scene_visual_enabled": self._config.enable_scene_visual,
            "excluded_prim_paths": (
                self._config.terrain_prim_path,
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
        collision_api_count = 0
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                mesh_count += 1
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
            build_side_grasp_target_payload,
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
        collision_cuboids = self._export_current_world_collision_cuboids(
            stage=stage,
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
        target_payload = build_side_grasp_target_payload(
            object_prim_path=object_prim_path,
            T_world_base=T_world_base,
            bbox_min=bbox["min_xyz"],
            bbox_max=bbox["max_xyz"],
            bbox_center=bbox["center_xyz"],
            bbox_size=bbox["size_xyz"],
        )

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

        place_pose_world = self._place_pose_world_from_episode(episode_spec)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("当前没有 USD stage，无法导出 cuRobo place 输入。")

        T_world_base, base_source = self._read_body_matrix("arm_base_link")
        T_world_tcp, tcp_source, tcp_mode = self._read_tcp_export_matrix()
        q_arm, dq_arm, arm_joint_ids = self._read_named_joint_state(self._config.arm_joint_names)
        q_gripper, dq_gripper, gripper_joint_ids = self._read_named_joint_state(
            self._config.gripper_joint_names
        )
        bbox = self._compute_object_bbox(stage, object_prim_path)
        collision_cuboids = self._export_current_world_collision_cuboids(
            stage=stage,
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

    def _place_pose_world_from_episode(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        """优先使用 task JSON 的 place_pose_world，并保留 baseline 支持的 clearance 字段。"""

        raw_place = dict((episode_spec.raw_task or {}).get("place") or {})
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
            raise RuntimeError("当前 task 缺少 place.place_pose_world，无法生成 arm-place target。")
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
        if self._object is not None:
            # 动态刚体运动后，UsdGeom.BBoxCache 可能仍返回 authored 初始位置。
            # 这里只读取 SingleRigidPrim 的实时 PhysX pose 修正中心，不修改物体状态。
            live_position, _live_orientation = self._object.get_world_pose()
            live_center = np.asarray(_as_tuple(live_position), dtype=float)
            if live_center.shape == (3,) and np.all(np.isfinite(live_center)):
                center = live_center
                bbox_min = center - 0.5 * size
                bbox_max = center + 0.5 * size
                center_source = "live_physx_object_pose"
        return {
            "min_xyz": bbox_min.tolist(),
            "max_xyz": bbox_max.tolist(),
            "center_xyz": center.tolist(),
            "size_xyz": size.tolist(),
            "center_source": center_source,
            "read_only": True,
        }

    def _export_current_world_collision_cuboids(
        self,
        *,
        stage: Any,
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
        candidates: list[dict[str, Any]] = []
        reference_point = np.asarray(object_bbox_center, dtype=float)
        base_position = T_world_base[:3, 3].copy()

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
        for index, obstacle in enumerate(obstacles):
            obstacle["name"] = _sanitize_obstacle_name(
                str(obstacle["prim_path"]),
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
        return {
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
            "clipped_large_obstacle_count": sum(
                1 for cuboid in cuboids if cuboid.get("clipped_from_large_obstacle")
            ),
            "nearest_obstacle_distance_xy_m": (
                None
                if not cuboids
                else float(cuboids[0].get("distance_to_reference_xy_m", 0.0))
            ),
            "note": (
                "padding_xy/clearance_margin 只水平膨胀传给 cuRobo 的规划障碍；"
                "padding_z 保持较小，避免把桌面虚拟抬高到抓取目标。"
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
