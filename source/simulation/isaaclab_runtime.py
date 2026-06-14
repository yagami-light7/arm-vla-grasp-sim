"""由 pipeline 主循环驱动的 Isaac Lab 单环境 runtime。"""

from __future__ import annotations

import math
import sys
import time
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
    terrain_prim_path: str = "/World/scene_collision"
    visual_prim_path: str = "/World/gauss"
    viewport_camera_prim_path: str = "/World/Camera_main"
    hide_navigation_collision_visual: bool = True
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
    show_randomization_debug: bool = False


def _item(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


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


def _collision_candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, str]:
    """优先保留任务物体附近的碰撞体，避免 stage 遍历顺序挤掉桌面。"""

    return (
        float(candidate["distance_to_reference_xy_m"]),
        str(candidate["prim_path"]),
    )


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
        self._episode_spec: EpisodeSpec | None = None
        self._step_calls = 0
        self._closed = False
        self._action_prepared = False
        self._environment_terminated = False
        self._last_action = RobotAction.idle()
        self._pending_arm_tracking_target: dict[str, Any] | None = None
        self._manipulation_base_lock_active = False
        self._manipulation_support_joint_lock_active = False
        self._hidden_distractor_root_paths: tuple[str, ...] = ()
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
            "manipulation_base_lock_active": False,
            "manipulation_base_lock_apply_count": 0,
            "last_manipulation_base_lock_report": None,
            "manipulation_support_joint_lock_active": False,
            "manipulation_support_joint_lock_apply_count": 0,
            "last_manipulation_support_joint_lock_report": None,
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
        self._pending_arm_tracking_target = None
        self._manipulation_base_lock_active = False
        self._manipulation_support_joint_lock_active = False
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
                "manipulation_base_lock_active": False,
                "manipulation_base_lock_apply_count": 0,
                "last_manipulation_base_lock_report": None,
                "manipulation_support_joint_lock_active": False,
                "manipulation_support_joint_lock_apply_count": 0,
                "last_manipulation_support_joint_lock_report": None,
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
        self._configure_viewport()

    def read(self) -> SimulationState:
        if self._adapter is None:
            return SimulationState(
                step_index=self._step_calls,
                timestamp=time.time(),
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
            "base_pose_xyyaw": (
                _item(root_position[0]),
                _item(root_position[1]),
                _quat_to_yaw(root_quaternion),
            ),
            "body_velocity": self._adapter.get_base_velocity_full(),
            "last_action_source": self._last_action.source,
        }
        metadata.update(self._adapter.diagnostics())
        return SimulationState(
            step_index=self._step_calls,
            timestamp=time.time(),
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

        requested = bool(action.metadata.get("manipulation_base_lock", False))
        phase = action.metadata.get("manipulation_base_lock_phase")
        if requested and not self._manipulation_base_lock_active:
            report = self._adapter.set_base_pose_lock(True)
            if report.get("enabled") is not True:
                raise RuntimeError(f"failed to enable manipulation base lock: {report}")
            self._manipulation_base_lock_active = True
            self._metadata.update(
                {
                    "used_base_teleport": True,
                    "used_manipulation_base_lock": True,
                    "manipulation_base_lock_active": True,
                    "last_manipulation_base_lock_report": {
                        **report,
                        "transition": "enabled",
                        "phase": phase,
                    },
                }
            )
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
                }
            )
        support_requested = bool(action.metadata.get("manipulation_support_joint_lock", False))
        support_phase = action.metadata.get("manipulation_support_joint_lock_phase")
        if support_requested and not self._manipulation_support_joint_lock_active:
            if not hasattr(self._adapter, "set_support_joint_lock"):
                report = {
                    "enabled": False,
                    "reason": "adapter_missing_set_support_joint_lock",
                }
            else:
                report = self._adapter.set_support_joint_lock(True)
            self._manipulation_support_joint_lock_active = bool(report.get("enabled"))
            self._metadata.update(
                {
                    "used_direct_joint_state": bool(report.get("uses_direct_joint_state", False)),
                    "used_manipulation_support_joint_lock": bool(report.get("enabled")),
                    "manipulation_support_joint_lock_active": bool(report.get("enabled")),
                    "last_manipulation_support_joint_lock_report": {
                        **report,
                        "transition": "enabled",
                        "phase": support_phase,
                    },
                }
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
            env = gym.make(self._config.task_name, cfg=env_cfg)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            # wrapper 构造时会触发 env.reset；GUI viewport 只在 reset 完成后切相机。
            self._configure_viewport()
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
            adapter = Go2LocomotionAdapter(wrapped, policy, wrapped.get_observations())
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
        from isaaclab.terrains import TerrainImporterCfg
        from source.navigation.adapters.terrain_utils import write_collision_terrain_wrapper

        scene_usd = self._resolve_path(episode_spec.scene_usd)
        stage_report = self._validate_scene_collision(scene_usd, self._config.terrain_prim_path)
        self._metadata["stage_report"] = stage_report
        terrain_usd = write_collision_terrain_wrapper(
            scene_usd,
            self._config.terrain_prim_path,
        )
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = self._config.device
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/nav_collision",
            terrain_type="usd",
            usd_path=str(terrain_usd),
            debug_vis=False,
        )
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (episode_spec.start.x, episode_spec.start.x),
                "y": (episode_spec.start.y, episode_spec.start.y),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (episode_spec.start.yaw, episode_spec.start.yaw),
            },
            "velocity_range": {
                key: (0.0, 0.0)
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            },
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
        return {
            "loaded": True,
            "load_mode": "sublayer",
            "wrapper_path": str(wrapper),
            "scene_usd": str(self._resolve_path(episode_spec.scene_usd)),
            "visual_prim_path": self._config.visual_prim_path,
            "excluded_prim_paths": (
                self._config.terrain_prim_path,
                "/World/go2_x5",
                "/World/mec_arm_6dof",
            ),
            "object_visibility": object_visibility,
        }

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

    def _configure_viewport(self) -> None:
        """复用 baseline 的 GUI 显示语义，不参与仿真控制。"""

        from source.simulation.viewport import configure_navigation_viewport

        report = configure_navigation_viewport(
            camera_prim_path=self._config.viewport_camera_prim_path,
            hide_collision_visual=self._config.hide_navigation_collision_visual,
        )
        self._metadata["viewport_report"] = report
        selected_camera = report.get("selected_camera_prim_path")
        if selected_camera:
            self._metadata["camera_prim_path"] = selected_camera

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
        }

    def _initialize_object_reader(self, episode_spec: EpisodeSpec) -> None:
        if not episode_spec.object_prim_path:
            return
        from isaacsim.core.prims import SingleRigidPrim

        self._object = SingleRigidPrim(
            prim_path=episode_spec.object_prim_path,
            name="full_physics_navigation_object",
            reset_xform_properties=False,
        )
        self._object.initialize()

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
        pose_report = self._apply_object_pose(episode_spec)
        world_quaternion = tuple(
            float(value)
            for value in pose_report.get(
                "authored_world_quaternion_wxyz",
                _quat_wxyz_from_rpy(roll, pitch, yaw),
            )
        )
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
            "target_position_xyz": [float(x), float(y), float(z)],
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
            "hidden_distractor_root_paths": list(self._hidden_distractor_root_paths),
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

        padding_m = 0.02
        min_size_m = 0.01
        max_obstacles = 16
        local_radius_m = 1.25
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
            size = bbox_max - bbox_min
            if float(np.max(size)) < min_size_m:
                continue
            if (
                float(np.max(size)) > max_extent_m
                or float(size[2]) > max_height_m
                or float(np.prod(size)) > max_volume_m3
            ):
                continue
            distance_to_reference = _distance_point_to_aabb_xy(
                reference_point[:2],
                bbox_min,
                bbox_max,
            )
            if distance_to_reference > local_radius_m:
                continue
            if _point_inside_aabb(base_position, bbox_min, bbox_max, margin=padding_m):
                continue

            center_world = 0.5 * (bbox_min + bbox_max)
            padded_size = np.maximum(size + 2.0 * padding_m, min_size_m)
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
                    "pose_world": pose_dict_from_matrix(T_world_obstacle),
                    "pose_base": pose_dict_from_matrix(T_base_obstacle),
                    "padding_m": padding_m,
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
        """把 arm position target 写入 policy action 槽，不直接改写关节状态。"""

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
        if (
            not override_report.get("action_term_available")
            or len(action_indices) != len(expected_names)
        ):
            raise RuntimeError(
                "Isaac Lab policy action does not expose all arm joints for direct target override: "
                f"{override_report}"
            )
        self._adapter.set_arm_joint_target(target)
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
            "direct_arm_action_override": True,
            "arm_action_indices": action_indices,
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
