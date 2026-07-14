"""单 Stage、单 World 的 Isaac Sim runtime。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from source.interfaces import EpisodeSpec, RobotAction, SimulationState
from source.simulation.action_applier import NamedJointActionApplier


@dataclass(frozen=True)
class IsaacSimulationConfig:
    """不暴露到命令行的 Isaac 配置。"""

    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    stage_load_updates: int = 30
    robot_name_hint: str = "go2_x5"
    collision_prim_path: str = "/World/scene_collision"
    table_prim_candidates: tuple[str, ...] = ("/World/table",)
    camera_prim_candidates: tuple[str, ...] = (
        "/World/Camera_main",
        "/World/camera_main",
    )
    tcp_prim_candidates: tuple[str, ...] = (
        "/World/go2_x5/arm_link6/grasp_tcp_link",
        "/World/go2_x5/grasp_tcp_link",
    )
    show_randomization_debug: bool = False


def _pose7_from_rpy(
    pose: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float, float]:
    x, y, z, roll, pitch, yaw = pose
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        x,
        y,
        z,
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _float_tuple(values: Any, *, length: int) -> tuple[float, ...]:
    if values is None:
        return (0.0,) * length
    flattened = values.detach().cpu().numpy() if hasattr(values, "detach") else values
    return tuple(float(value) for value in flattened)[:length]


def _get_or_add_xform_op(xformable: Any, op_type: Any) -> Any:
    """优先复用组合层中的 xform 属性，避免重复创建同名 op。"""

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


def _choose_larger_peak(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """保留最大误差样本，便于 summary 直接暴露最坏 tracking 点。"""

    if candidate is None:
        return current
    if current is None:
        return candidate
    if float(candidate.get("abs_error", 0.0)) > float(current.get("abs_error", 0.0)):
        return candidate
    return current


class IsaacSimulationRuntime:
    """统一持有 Isaac Stage、World、articulation 和状态读取器。"""

    def __init__(
        self,
        *,
        simulation_app: Any,
        project_root: str | Path,
        config: IsaacSimulationConfig | None = None,
    ):
        self._simulation_app = simulation_app
        self._project_root = Path(project_root).expanduser().resolve()
        self._config = config or IsaacSimulationConfig()
        self._world = None
        self._stage = None
        self._robot = None
        self._object = None
        self._joint_action_applier: NamedJointActionApplier | None = None
        self._episode_spec: EpisodeSpec | None = None
        self._object_root_path: str | None = None
        self._object_state_path: str | None = None
        self._tcp_prim_path: str | None = None
        self._camera_prim_path: str | None = None
        self._step_calls = 0
        self._reset_calls = 0
        self._closed = False
        self._last_action = RobotAction.idle()
        self._pending_arm_tracking_target: dict[str, Any] | None = None
        self._metadata: dict[str, Any] = {
            "simulation_ready": False,
            "execution_provenance_verified": False,
            "used_base_teleport": False,
            "used_direct_joint_state": False,
            "used_object_teleport": False,
            "used_kinematic_object_follow": False,
            "used_visual_replay": False,
            "last_joint_action_report": None,
            "last_arm_tracking_report": None,
            "arm_tracking_peak_report": None,
            "arm_tracking_report": {
                "sample_count": 0,
                "max_abs_error": None,
                "peak_report": None,
                "latest_max_abs_error": None,
                "latest_mean_abs_error": None,
                "segments": {},
                "joints": {},
            },
            "arm_tracking_sample_count": 0,
            "arm_tracking_max_abs_error": None,
            "joint_action_apply_count": 0,
            "arm_joint_action_apply_count": 0,
            "gripper_joint_action_apply_count": 0,
            "gripper_close_apply_count": 0,
            "gripper_open_apply_count": 0,
        }

    def build(self, episode_spec: EpisodeSpec) -> None:
        if self._closed:
            raise RuntimeError("simulation runtime is closed")
        if self._world is not None:
            raise RuntimeError("the Isaac stage has already been built")

        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim

        self._episode_spec = episode_spec
        scene_usd = self._resolve_project_path(episode_spec.scene_usd)
        if not scene_usd.exists():
            raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")

        usd_context = omni.usd.get_context()
        if usd_context.open_stage(str(scene_usd)) is False:
            raise RuntimeError(f"failed to open stage: {scene_usd}")
        for _ in range(max(1, self._config.stage_load_updates)):
            self._simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError(f"stage did not load: {scene_usd}")

        self._stage = stage
        self._object_root_path = episode_spec.object_prim_path
        activation_report = self._activate_required_prims(episode_spec)
        object_pose_report = self._apply_initial_object_pose(episode_spec)
        stage_report = self._inspect_stage(episode_spec)
        randomization_debug_report = None
        if self._config.show_randomization_debug:
            from source.diagnostics import create_randomization_debug

            randomization_debug_report = create_randomization_debug(
                stage,
                episode_spec.raw_task,
            )

        world = World.instance()
        if world is None:
            world = World(
                physics_dt=self._config.physics_dt,
                rendering_dt=self._config.rendering_dt,
                stage_units_in_meters=1.0,
            )
        world.play()
        world.initialize_physics()

        articulation_path = self._resolve_articulation_root()
        robot = SingleArticulation(
            prim_path=articulation_path,
            name="full_physics_go2_x5",
        )
        robot.initialize()
        if not robot.is_valid():
            raise RuntimeError(f"invalid articulation: {articulation_path}")

        object_state_path = self._resolve_dynamic_rigid_body(episode_spec.object_prim_path)
        object_prim = SingleRigidPrim(
            prim_path=object_state_path,
            name="full_physics_task_object",
            reset_xform_properties=False,
        )
        object_prim.initialize()
        # 对齐 baseline：PhysX/SingleRigidPrim 初始化后再写一次任务绝对位姿，
        # 避免初始化阶段把旧 rigid-body 状态同步回 USD。
        object_pose_after_physics_report = self._apply_initial_object_pose(episode_spec)
        object_prim.set_linear_velocity(np.zeros(3, dtype=np.float32))
        object_prim.set_angular_velocity(np.zeros(3, dtype=np.float32))

        self._world = world
        self._robot = robot
        self._object = object_prim
        self._joint_action_applier = NamedJointActionApplier(robot)
        self._object_state_path = object_state_path
        self._tcp_prim_path = self._resolve_tcp_prim_path()
        self._camera_prim_path = stage_report["camera_prim_path"]
        self._metadata.update(
            {
                "simulation_ready": True,
                "scene_usd": str(scene_usd),
                "world_count": 1,
                "opened_stage_count": 1,
                "articulation_prim_path": articulation_path,
                "object_root_prim_path": self._object_root_path,
                "object_state_prim_path": object_state_path,
                "tcp_prim_path": self._tcp_prim_path,
                "camera_prim_path": self._camera_prim_path,
                "stage_report": stage_report,
                "activation_report": activation_report,
                "object_pose_setup_report": object_pose_after_physics_report,
                "object_pose_setup_before_physics_report": object_pose_report,
                "object_pose_setup_after_physics_report": object_pose_after_physics_report,
                "randomization_debug": randomization_debug_report,
                "joint_names": list(robot.dof_names),
            }
        )

    def reset(self, episode_spec: EpisodeSpec, *, seed: int) -> None:
        self._require_ready()
        if self._reset_calls:
            raise RuntimeError("this phase supports one reset per Isaac runtime")

        import numpy as np

        self._episode_spec = episode_spec
        current_position, _ = self._robot.get_world_pose()
        root_z = float(current_position[2])
        half_yaw = episode_spec.start.yaw * 0.5
        self._robot.set_world_pose(
            position=np.asarray(
                [episode_spec.start.x, episode_spec.start.y, root_z],
                dtype=np.float32,
            ),
            orientation=np.asarray(
                [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
                dtype=np.float32,
            ),
        )
        self._robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self._robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        # build 后 pipeline 已推进过一次 world.step；reset 时按 baseline 再写一次绝对位姿。
        object_pose_reset_report = self._apply_initial_object_pose(episode_spec)
        self._object.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self._object.set_angular_velocity(np.zeros(3, dtype=np.float32))

        self._reset_calls += 1
        self._last_action = RobotAction.idle(source="episode_reset")
        self._metadata.update(
            {
                "seed": int(seed),
                "episode_reset_complete": True,
                "used_episode_reset_pose": True,
                "reset_robot_root_pose": (
                    episode_spec.start.x,
                    episode_spec.start.y,
                    root_z,
                    math.cos(half_yaw),
                    0.0,
                    0.0,
                    math.sin(half_yaw),
                ),
                "object_pose_setup_report": object_pose_reset_report,
                "object_pose_setup_after_reset_report": object_pose_reset_report,
                "object_pose_debug_after_reset": self._object_debug_snapshot(
                    episode_spec,
                    label="after_reset_pose_apply",
                ),
                # smoke 只验证场景与 reset，不声明执行来源已满足纯物理验收。
                "execution_provenance_verified": False,
            }
        )

    def read(self) -> SimulationState:
        if self._robot is None:
            return SimulationState(
                step_index=self._step_calls,
                timestamp=time.time(),
                robot_root_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                robot_root_velocity=(0.0,) * 6,
                metadata=dict(self._metadata),
            )

        robot_position, robot_orientation = self._robot.get_world_pose()
        robot_linear = self._robot.get_linear_velocity()
        robot_angular = self._robot.get_angular_velocity()
        joint_positions = self._robot.get_joint_positions()
        joint_velocities = self._robot.get_joint_velocities()
        joint_position_tuple = _float_tuple(joint_positions, length=len(self._robot.dof_names))
        joint_velocity_tuple = _float_tuple(joint_velocities, length=len(self._robot.dof_names))
        self._consume_pending_arm_tracking(joint_position_tuple)
        object_position, object_orientation = self._object.get_world_pose()
        object_linear = self._object.get_linear_velocity()
        object_angular = self._object.get_angular_velocity()

        return SimulationState(
            step_index=self._step_calls,
            timestamp=time.time(),
            robot_root_pose=(
                *_float_tuple(robot_position, length=3),
                *_float_tuple(robot_orientation, length=4),
            ),
            robot_root_velocity=(
                *_float_tuple(robot_linear, length=3),
                *_float_tuple(robot_angular, length=3),
            ),
            joint_positions=joint_position_tuple,
            joint_velocities=joint_velocity_tuple,
            tcp_pose=self._read_usd_world_pose(self._tcp_prim_path),
            object_pose=(
                *_float_tuple(object_position, length=3),
                *_float_tuple(object_orientation, length=4),
            ),
            object_velocity=(
                *_float_tuple(object_linear, length=3),
                *_float_tuple(object_angular, length=3),
            ),
            metadata={
                **self._metadata,
                "last_action_source": self._last_action.source,
                "object_pose_debug_latest": self._object_debug_snapshot(
                    self._episode_spec,
                    label="latest_read",
                )
                if hasattr(self, "_episode_spec") and self._episode_spec is not None
                else None,
            },
        )

    def _matrix_to_list(self, matrix: Any) -> list[list[float]] | None:
        if matrix is None:
            return None
        if isinstance(matrix, tuple):
            matrix = matrix[0]
        return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]

    def _rotation_matrix_to_rpy_deg(self, matrix: Any) -> dict[str, Any] | None:
        matrix_list = self._matrix_to_list(matrix)
        if matrix_list is None:
            return None
        rotation = [[float(matrix_list[row][col]) for col in range(3)] for row in range(3)]
        column_norms = []
        for col in range(3):
            norm = math.sqrt(sum(rotation[row][col] * rotation[row][col] for row in range(3)))
            column_norms.append(norm)
            if norm > 1.0e-9:
                for row in range(3):
                    rotation[row][col] /= norm
        pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
        if abs(math.cos(pitch)) > 1.0e-6:
            roll = math.atan2(rotation[2][1], rotation[2][2])
            yaw = math.atan2(rotation[1][0], rotation[0][0])
        else:
            roll = math.atan2(-rotation[1][2], rotation[1][1])
            yaw = 0.0
        return {
            "convention": "roll_x_pitch_y_yaw_z_degrees_from_matrix",
            "roll": math.degrees(roll),
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),
            "scale_removed_from_columns": column_norms,
        }

    def _object_debug_snapshot(
        self,
        episode_spec: EpisodeSpec | None,
        *,
        label: str,
    ) -> dict[str, Any] | None:
        if self._stage is None or episode_spec is None or not episode_spec.object_prim_path:
            return None
        from pxr import Usd, UsdGeom, UsdPhysics

        prim = self._stage.GetPrimAtPath(episode_spec.object_prim_path)
        report: dict[str, Any] = {
            "label": label,
            "object_prim_path": episode_spec.object_prim_path,
            "object_prim_valid": bool(prim.IsValid()),
            "object_state_prim_path": self._object_state_path,
        }
        if not prim.IsValid():
            return report
        rigid_paths = [
            str(child.GetPath())
            for child in Usd.PrimRange(prim)
            if child.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        report["rigid_body_prim_paths"] = rigid_paths
        target_paths = [episode_spec.object_prim_path]
        if self._object_state_path and self._object_state_path not in target_paths:
            target_paths.append(self._object_state_path)
        xform_targets: dict[str, Any] = {}
        for path in target_paths:
            target_prim = self._stage.GetPrimAtPath(path)
            if not target_prim.IsValid() or not target_prim.IsA(UsdGeom.Xformable):
                xform_targets[path] = {
                    "valid": bool(target_prim.IsValid()),
                    "xformable": False,
                }
                continue
            xformable = UsdGeom.Xformable(target_prim)
            local_transform = xformable.GetLocalTransformation()
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            xform_targets[path] = {
                "valid": True,
                "xformable": True,
                "xform_op_order": [op.GetOpName() for op in xformable.GetOrderedXformOps()],
                "local_euler_xyz_deg": self._rotation_matrix_to_rpy_deg(local_transform),
                "world_euler_xyz_deg": self._rotation_matrix_to_rpy_deg(world_transform),
                "local_transform": self._matrix_to_list(local_transform),
                "world_transform": self._matrix_to_list(world_transform),
            }
        report["xform_targets"] = xform_targets
        if self._object is not None:
            try:
                report["live_velocity"] = {
                    "linear": _float_tuple(self._object.get_linear_velocity(), length=3),
                    "angular": _float_tuple(self._object.get_angular_velocity(), length=3),
                }
            except Exception as exc:
                report["live_velocity_error"] = str(exc)
        return report

    def apply(self, action: RobotAction) -> None:
        nonzero_base = any(abs(value) > 1.0e-9 for value in action.base_velocity)
        if nonzero_base:
            raise RuntimeError(
                "IsaacSimulationRuntime currently accepts arm/gripper actions only; "
                "base commands require a navigation runtime"
            )
        if action.arm_joint_positions is not None or action.gripper_command is not None:
            if self._joint_action_applier is None:
                raise RuntimeError("joint action applier is not initialized")
            report = self._joint_action_applier.apply(action)
            self._metadata["last_joint_action_report"] = report
            self._record_joint_action_apply(action, report)
            self._record_pending_arm_tracking_target(action, report)
        self._last_action = action

    def step(self, *, render: bool) -> None:
        if self._world is None:
            return
        # 真实仿真的唯一推进点；调用权只属于 FullPhysicsPipeline 主循环。
        self._world.step(render=render)
        self._step_calls += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._world is not None and self._world.is_playing():
            self._world.pause()
        self._closed = True

    def _record_joint_action_apply(self, action: RobotAction, report: dict[str, Any]) -> None:
        if not report.get("applied"):
            return
        self._metadata["joint_action_apply_count"] = (
            int(self._metadata.get("joint_action_apply_count", 0)) + 1
        )
        if report.get("arm_targeted"):
            self._metadata["arm_joint_action_apply_count"] = (
                int(self._metadata.get("arm_joint_action_apply_count", 0)) + 1
            )
        if report.get("gripper_targeted"):
            self._metadata["gripper_joint_action_apply_count"] = (
                int(self._metadata.get("gripper_joint_action_apply_count", 0)) + 1
            )
        if action.gripper_command == "close" and report.get("gripper_targeted"):
            self._metadata["gripper_close_apply_count"] = (
                int(self._metadata.get("gripper_close_apply_count", 0)) + 1
            )
        if action.gripper_command == "open" and report.get("gripper_targeted"):
            self._metadata["gripper_open_apply_count"] = (
                int(self._metadata.get("gripper_open_apply_count", 0)) + 1
            )

    def _record_pending_arm_tracking_target(
        self,
        action: RobotAction,
        report: dict[str, Any],
    ) -> None:
        if not report.get("applied") or not report.get("arm_targeted"):
            return

        joint_names = tuple(str(name) for name in report.get("joint_names", ()))
        joint_indices = tuple(int(index) for index in report.get("joint_indices", ()))
        target_positions = tuple(float(value) for value in report.get("target_positions", ()))
        arm_joint_names = tuple(str(name) for name in report.get("arm_joint_names", ()))
        if not joint_names or not joint_indices or not target_positions or not arm_joint_names:
            self._metadata["last_arm_tracking_report"] = {
                "tracked": False,
                "reason": "missing_arm_joint_targets",
                "source": action.source,
            }
            return

        index_by_name = {
            name: joint_indices[index]
            for index, name in enumerate(joint_names)
            if index < len(joint_indices)
        }
        target_by_name = {
            name: target_positions[index]
            for index, name in enumerate(joint_names)
            if index < len(target_positions)
        }
        missing_names = [
            name for name in arm_joint_names if name not in index_by_name or name not in target_by_name
        ]
        if missing_names:
            self._metadata["last_arm_tracking_report"] = {
                "tracked": False,
                "reason": "missing_arm_joint_indices",
                "source": action.source,
                "missing_joint_names": missing_names,
            }
            return

        # 只记录目标，真实误差必须等 pipeline 推进一次 world.step 后再读取。
        self._pending_arm_tracking_target = {
            "source": action.source,
            "operation": action.metadata.get("operation"),
            "segment_index": action.metadata.get("segment_index"),
            "segment_name": action.metadata.get("segment_name"),
            "segment_type": action.metadata.get("segment_type"),
            "segment_tick": action.metadata.get("segment_tick"),
            "segment_ticks": action.metadata.get("segment_ticks"),
            "joint_names": arm_joint_names,
            "joint_indices": tuple(index_by_name[name] for name in arm_joint_names),
            "target_positions": tuple(target_by_name[name] for name in arm_joint_names),
            "applied_step_index": self._step_calls,
        }

    def _consume_pending_arm_tracking(self, joint_positions: tuple[float, ...]) -> None:
        pending = self._pending_arm_tracking_target
        if pending is None:
            return
        if self._step_calls <= int(pending.get("applied_step_index", -1)):
            return

        indices = tuple(int(index) for index in pending["joint_indices"])
        targets = tuple(float(value) for value in pending["target_positions"])
        if any(index < 0 or index >= len(joint_positions) for index in indices):
            self._metadata["last_arm_tracking_report"] = {
                "tracked": False,
                "reason": "joint_index_out_of_range",
                "source": pending.get("source"),
                "joint_indices": indices,
                "joint_position_count": len(joint_positions),
            }
            self._pending_arm_tracking_target = None
            return

        actuals = tuple(float(joint_positions[index]) for index in indices)
        errors = tuple(actual - target for actual, target in zip(actuals, targets))
        abs_errors = tuple(abs(value) for value in errors)
        max_abs_error = max(abs_errors) if abs_errors else 0.0
        mean_abs_error = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        l2_error = math.sqrt(sum(value * value for value in errors))
        report = {
            "tracked": True,
            "source": pending.get("source"),
            "operation": pending.get("operation"),
            "segment_index": pending.get("segment_index"),
            "segment_name": pending.get("segment_name"),
            "segment_type": pending.get("segment_type"),
            "segment_tick": pending.get("segment_tick"),
            "segment_ticks": pending.get("segment_ticks"),
            "applied_step_index": pending.get("applied_step_index"),
            "sample_step_index": self._step_calls,
            "joint_names": tuple(pending["joint_names"]),
            "joint_indices": indices,
            "target_positions": targets,
            "actual_positions": actuals,
            "position_errors": errors,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "l2_error": l2_error,
        }
        self._metadata["last_arm_tracking_report"] = report
        self._update_arm_tracking_aggregate(report)
        self._pending_arm_tracking_target = None

    def _update_arm_tracking_aggregate(self, report: dict[str, Any]) -> None:
        sample_count = int(self._metadata.get("arm_tracking_sample_count", 0)) + 1
        previous_max = self._metadata.get("arm_tracking_max_abs_error")
        max_abs_error = float(report["max_abs_error"])
        global_max = max_abs_error if previous_max is None else max(float(previous_max), max_abs_error)
        peak_report = self._make_arm_tracking_peak_report(report)
        previous_peak = self._metadata.get("arm_tracking_peak_report")
        global_peak = _choose_larger_peak(previous_peak, peak_report)
        segment_key = str(report.get("segment_name") or "unknown")
        aggregate = dict(self._metadata.get("arm_tracking_report") or {})
        segments = {
            str(name): dict(value)
            for name, value in dict(aggregate.get("segments") or {}).items()
        }
        joints = {
            str(name): dict(value)
            for name, value in dict(aggregate.get("joints") or {}).items()
        }
        segment_report = dict(segments.get(segment_key) or {})
        segment_count = int(segment_report.get("sample_count", 0)) + 1
        segment_previous_max = segment_report.get("max_abs_error")
        segment_max = (
            max_abs_error
            if segment_previous_max is None
            else max(float(segment_previous_max), max_abs_error)
        )
        segment_peak = _choose_larger_peak(segment_report.get("peak_report"), peak_report)
        segments[segment_key] = {
            "sample_count": segment_count,
            "operation": report.get("operation"),
            "segment_type": report.get("segment_type"),
            "max_abs_error": segment_max,
            "peak_report": segment_peak,
            "latest_max_abs_error": max_abs_error,
            "latest_mean_abs_error": float(report["mean_abs_error"]),
            "latest_step_index": report.get("sample_step_index"),
        }
        self._update_arm_tracking_joint_aggregates(report, joints)
        aggregate.update(
            {
                "sample_count": sample_count,
                "max_abs_error": global_max,
                "peak_report": global_peak,
                "latest_max_abs_error": max_abs_error,
                "latest_mean_abs_error": float(report["mean_abs_error"]),
                "latest_l2_error": float(report["l2_error"]),
                "latest_segment_name": report.get("segment_name"),
                "segments": segments,
                "joints": joints,
            }
        )
        self._metadata["arm_tracking_report"] = aggregate
        self._metadata["arm_tracking_peak_report"] = global_peak
        self._metadata["arm_tracking_sample_count"] = sample_count
        self._metadata["arm_tracking_max_abs_error"] = global_max

    def _make_arm_tracking_peak_report(self, report: dict[str, Any]) -> dict[str, Any] | None:
        errors = tuple(float(value) for value in report.get("position_errors", ()))
        if not errors:
            return None
        peak_order_index = max(range(len(errors)), key=lambda index: abs(errors[index]))
        joint_names = tuple(str(value) for value in report.get("joint_names", ()))
        joint_indices = tuple(int(value) for value in report.get("joint_indices", ()))
        targets = tuple(float(value) for value in report.get("target_positions", ()))
        actuals = tuple(float(value) for value in report.get("actual_positions", ()))
        error = errors[peak_order_index]
        return {
            "source": report.get("source"),
            "operation": report.get("operation"),
            "segment_index": report.get("segment_index"),
            "segment_name": report.get("segment_name"),
            "segment_type": report.get("segment_type"),
            "segment_tick": report.get("segment_tick"),
            "segment_ticks": report.get("segment_ticks"),
            "applied_step_index": report.get("applied_step_index"),
            "sample_step_index": report.get("sample_step_index"),
            "joint_order_index": peak_order_index,
            "joint_name": (
                joint_names[peak_order_index] if peak_order_index < len(joint_names) else None
            ),
            "joint_index": (
                joint_indices[peak_order_index] if peak_order_index < len(joint_indices) else None
            ),
            "target_position": (
                targets[peak_order_index] if peak_order_index < len(targets) else None
            ),
            "actual_position": (
                actuals[peak_order_index] if peak_order_index < len(actuals) else None
            ),
            "position_error": error,
            "abs_error": abs(error),
            "sample_max_abs_error": float(report["max_abs_error"]),
            "sample_mean_abs_error": float(report["mean_abs_error"]),
            "sample_l2_error": float(report["l2_error"]),
        }

    def _update_arm_tracking_joint_aggregates(
        self,
        report: dict[str, Any],
        joints: dict[str, dict[str, Any]],
    ) -> None:
        joint_names = tuple(str(value) for value in report.get("joint_names", ()))
        joint_indices = tuple(int(value) for value in report.get("joint_indices", ()))
        targets = tuple(float(value) for value in report.get("target_positions", ()))
        actuals = tuple(float(value) for value in report.get("actual_positions", ()))
        errors = tuple(float(value) for value in report.get("position_errors", ()))
        for order_index, error in enumerate(errors):
            joint_name = joint_names[order_index] if order_index < len(joint_names) else str(order_index)
            joint_report = dict(joints.get(joint_name) or {})
            joint_count = int(joint_report.get("sample_count", 0)) + 1
            abs_error = abs(error)
            previous_max = joint_report.get("max_abs_error")
            max_error = (
                abs_error if previous_max is None else max(float(previous_max), abs_error)
            )
            latest_peak = {
                "source": report.get("source"),
                "operation": report.get("operation"),
                "segment_index": report.get("segment_index"),
                "segment_name": report.get("segment_name"),
                "segment_type": report.get("segment_type"),
                "segment_tick": report.get("segment_tick"),
                "segment_ticks": report.get("segment_ticks"),
                "applied_step_index": report.get("applied_step_index"),
                "sample_step_index": report.get("sample_step_index"),
                "joint_order_index": order_index,
                "joint_name": joint_name,
                "joint_index": (
                    joint_indices[order_index] if order_index < len(joint_indices) else None
                ),
                "target_position": targets[order_index] if order_index < len(targets) else None,
                "actual_position": actuals[order_index] if order_index < len(actuals) else None,
                "position_error": error,
                "abs_error": abs_error,
            }
            joints[joint_name] = {
                "sample_count": joint_count,
                "joint_index": latest_peak["joint_index"],
                "max_abs_error": max_error,
                "peak_report": _choose_larger_peak(joint_report.get("peak_report"), latest_peak),
                "latest_abs_error": abs_error,
                "latest_position_error": error,
                "latest_target_position": latest_peak["target_position"],
                "latest_actual_position": latest_peak["actual_position"],
                "latest_segment_name": report.get("segment_name"),
                "latest_step_index": report.get("sample_step_index"),
            }

    def _resolve_project_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        return path.resolve() if path.is_absolute() else (self._project_root / path).resolve()

    def _require_ready(self) -> None:
        if self._world is None or self._robot is None or self._object is None:
            raise RuntimeError("stage must be built before reset")

    def _activate_required_prims(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        from pxr import UsdGeom

        required_paths = ["/World/go2_x5"]
        if episode_spec.object_prim_path:
            required_paths.append(episode_spec.object_prim_path)
        reports: list[dict[str, Any]] = []
        for required_path in required_paths:
            parts = [part for part in required_path.strip("/").split("/") if part]
            for index in range(1, len(parts) + 1):
                prim_path = "/" + "/".join(parts[:index])
                prim = self._stage.GetPrimAtPath(prim_path)
                report = {"prim_path": prim_path, "valid": bool(prim.IsValid())}
                if prim.IsValid():
                    report["was_active"] = bool(prim.IsActive())
                    if not prim.IsActive():
                        prim.SetActive(True)
                    report["is_active"] = bool(prim.IsActive())
                    if prim_path == required_path and prim.IsA(UsdGeom.Imageable):
                        UsdGeom.Imageable(prim).MakeVisible()
                        report["made_visible"] = True
                reports.append(report)
        missing = [item["prim_path"] for item in reports if not item["valid"]]
        if missing:
            raise RuntimeError(f"required stage prims are missing: {missing}")
        return {"required_paths": required_paths, "prim_reports": reports}

    def _apply_initial_object_pose(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        if episode_spec.object_initial_pose is None:
            return {"applied": False, "reason": "object_initial_pose_missing"}
        if not episode_spec.object_prim_path:
            raise RuntimeError("object_prim_path is required when object_initial_pose is set")

        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(episode_spec.object_prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            raise RuntimeError(f"object prim is not xformable: {episode_spec.object_prim_path}")

        xformable = UsdGeom.Xformable(prim)
        saved_scale = None
        op_order_before = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale and op.Get() is not None:
                saved_scale = tuple(float(value) for value in op.Get())
                break

        # 初始任务位姿必须是绝对位姿，不能叠加在资产已有旋转操作上。
        xformable.ClearXformOpOrder()
        for attr in list(prim.GetAttributes()):
            if attr.GetName().startswith("xformOp:") or attr.GetName() == "xformOpOrder":
                prim.RemoveProperty(attr.GetName())
        xformable = UsdGeom.Xformable(prim)
        translate_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
        orient_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
        pose7 = _pose7_from_rpy(episode_spec.object_initial_pose)
        if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            translate_op.Set(Gf.Vec3d(*pose7[:3]))
        else:
            translate_op.Set(Gf.Vec3f(*pose7[:3]))
        if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            orient_op.Set(Gf.Quatd(pose7[3], Gf.Vec3d(*pose7[4:])))
        else:
            orient_op.Set(Gf.Quatf(pose7[3], Gf.Vec3f(*pose7[4:])))
        ordered_ops = [translate_op, orient_op]
        if saved_scale is not None:
            scale_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeScale)
            if scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                scale_op.Set(Gf.Vec3d(*saved_scale))
            else:
                scale_op.Set(Gf.Vec3f(*saved_scale))
            ordered_ops.append(scale_op)
        xformable.SetXformOpOrder(ordered_ops)
        return {
            "applied": True,
            "object_prim_path": episode_spec.object_prim_path,
            "pose_wxyz": pose7,
            "xform_op_order_before": op_order_before,
            "xform_op_order_after": [op.GetOpName() for op in ordered_ops],
            "preserved_scale": saved_scale,
        }

    def _inspect_stage(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        from pxr import Usd, UsdGeom, UsdPhysics

        collision_root = self._stage.GetPrimAtPath(self._config.collision_prim_path)
        if not collision_root.IsValid():
            raise RuntimeError(
                f"collision prim does not exist: {self._config.collision_prim_path}"
            )
        collision_mesh_count = 0
        collision_api_count = 0
        for prim in Usd.PrimRange(collision_root):
            collision_mesh_count += int(prim.IsA(UsdGeom.Mesh))
            collision_api_count += int(prim.HasAPI(UsdPhysics.CollisionAPI))
        if collision_mesh_count == 0 or collision_api_count == 0:
            raise RuntimeError(
                "scene collision payload is empty; verify the /mnt/sage_data mount "
                f"for {self._config.collision_prim_path}"
            )

        camera_path = None
        for candidate in self._config.camera_prim_candidates:
            prim = self._stage.GetPrimAtPath(candidate)
            if prim.IsValid() and prim.IsA(UsdGeom.Camera):
                camera_path = candidate
                break
        if camera_path is None:
            for prim in self._stage.Traverse():
                if prim.IsA(UsdGeom.Camera):
                    camera_path = str(prim.GetPath())
                    break
        if camera_path is None:
            raise RuntimeError("no camera prim exists in the stage")

        table_path = next(
            (
                candidate
                for candidate in self._config.table_prim_candidates
                if self._stage.GetPrimAtPath(candidate).IsValid()
            ),
            self._config.collision_prim_path,
        )
        object_prim = self._stage.GetPrimAtPath(episode_spec.object_prim_path or "")
        if not object_prim.IsValid():
            raise RuntimeError(f"object prim does not exist: {episode_spec.object_prim_path}")
        return {
            "collision_prim_path": self._config.collision_prim_path,
            "collision_mesh_count": collision_mesh_count,
            "collision_api_count": collision_api_count,
            "table_support_prim_path": table_path,
            "camera_prim_path": camera_path,
            "object_prim_path": episode_spec.object_prim_path,
        }

    def _resolve_articulation_root(self) -> str:
        from pxr import UsdPhysics

        roots = [
            str(prim.GetPath())
            for prim in self._stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        matching = [
            path for path in roots if self._config.robot_name_hint.lower() in path.lower()
        ]
        if len(matching) == 1:
            return matching[0]
        if len(roots) == 1:
            return roots[0]
        raise RuntimeError(f"unable to choose Go2-X5 articulation root from: {roots}")

    def _resolve_dynamic_rigid_body(self, object_prim_path: str | None) -> str:
        from pxr import Usd, UsdPhysics

        if not object_prim_path:
            raise RuntimeError("task object prim path is missing")
        root = self._stage.GetPrimAtPath(object_prim_path)
        paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not paths:
            raise RuntimeError(f"no RigidBodyAPI exists under {object_prim_path}")
        return object_prim_path if object_prim_path in paths else paths[0]

    def _resolve_tcp_prim_path(self) -> str | None:
        from pxr import Usd

        for candidate in self._config.tcp_prim_candidates:
            if self._stage.GetPrimAtPath(candidate).IsValid():
                return candidate
        robot_root = self._stage.GetPrimAtPath("/World/go2_x5")
        if robot_root.IsValid():
            for prim in Usd.PrimRange(robot_root):
                if prim.GetName() == "grasp_tcp_link":
                    return str(prim.GetPath())
        return None

    def _read_usd_world_pose(
        self,
        prim_path: str | None,
    ) -> tuple[float, float, float, float, float, float, float] | None:
        if not prim_path:
            return None
        from pxr import UsdGeom

        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        return (
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
