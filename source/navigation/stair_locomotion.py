"""用于隔离验证低层楼梯 locomotion policy 的中心线跟踪组件。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from source.interfaces import NavGoal, NavPlan, RobotAction, SimulationState


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


class StairCenterlinePlanner:
    """返回场景标定的 PCT 楼梯中心线，不引入新的全局搜索变量。"""

    def __init__(
        self,
        path_3d: Sequence[Sequence[float]],
        *,
        visualization_path_3d: Sequence[Sequence[float]] | None = None,
    ):
        path = tuple(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in path_3d
        )
        if len(path) < 2:
            raise ValueError("楼梯中心线至少需要两个点。")
        self.path_3d = path
        self.visualization_path_3d = tuple(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in (
                path if visualization_path_3d is None else visualization_path_3d
            )
        )
        if len(self.visualization_path_3d) < 2:
            raise ValueError("楼梯可视化中心线至少需要两个点。")

    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        path = list(self.path_3d)
        path[0] = (
            float(state.robot_root_pose[0]),
            float(state.robot_root_pose[1]),
            float(state.robot_root_pose[2]),
        )
        path[-1] = (
            float(goal.x),
            float(goal.y),
            float(path[-1][2] if goal.z is None else goal.z),
        )
        path_3d = tuple(path)
        return NavPlan(
            goal=goal,
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct_stair_centerline",
                "path_3d": path_3d,
                "visualization_path_3d": self.visualization_path_3d,
                "controller": "stair_heading_tracker",
                "low_level_policy_isolation": True,
            },
        )


@dataclass(frozen=True)
class StairLocomotionExecutorConfig:
    """楼梯策略评测的固定控制参数。"""

    forward_velocity_mps: float = 0.25
    max_lateral_velocity_mps: float = 0.12
    max_angular_velocity_rps: float = 0.50
    heading_kp: float = 2.0
    cross_track_kp: float = 0.80
    full_speed_heading_error_rad: float = 0.10
    stop_forward_heading_error_rad: float = 0.30
    waypoint_tolerance_m: float = 0.22
    goal_tolerance_m: float = 0.25
    goal_z_tolerance_m: float = 0.45
    max_path_deviation_m: float = 0.45
    stall_timeout_s: float = 6.0
    stall_progress_m: float = 0.05

    def __post_init__(self) -> None:
        if self.forward_velocity_mps <= 0.0:
            raise ValueError("楼梯前进速度必须为正数。")
        if self.max_lateral_velocity_mps < 0.0:
            raise ValueError("楼梯横向纠偏速度不能为负数。")
        if self.max_angular_velocity_rps <= 0.0:
            raise ValueError("楼梯最大角速度必须为正数。")
        if self.stop_forward_heading_error_rad <= self.full_speed_heading_error_rad:
            raise ValueError("停止前进航向误差必须大于全速航向误差。")
        if self.goal_tolerance_m <= 0.0 or self.goal_z_tolerance_m < 0.0:
            raise ValueError("楼梯终点容差配置无效。")
        if self.max_path_deviation_m <= 0.0:
            raise ValueError("楼梯最大路径偏差必须为正数。")
        if self.stall_timeout_s <= 0.0 or self.stall_progress_m <= 0.0:
            raise ValueError("楼梯停滞检测参数必须为正数。")


class StairLocomotionExecutor:
    """沿楼梯分段切线控制机体航向，并用横向速度修正中心线偏差。"""

    def __init__(self, config: StairLocomotionExecutorConfig | None = None):
        self.config = config or StairLocomotionExecutorConfig()
        self._plan: NavPlan | None = None
        self._path: tuple[tuple[float, float, float], ...] = ()
        self._segment_lengths: tuple[float, ...] = ()
        self._cumulative_lengths: tuple[float, ...] = ()
        self._segment_index = 1
        self._done = False
        self._failed = False
        self._failure_reason = ""
        self._best_progress_m = 0.0
        self._last_progress_time_s: float | None = None
        self._status: dict[str, Any] = {
            "controller": "stair_heading_tracker",
            "ready": False,
        }

    def reset(self, plan: NavPlan) -> None:
        raw_path = plan.metadata.get("path_3d", plan.waypoints)
        path = tuple(
            (
                float(point[0]),
                float(point[1]),
                float(point[2]) if len(point) >= 3 else 0.0,
            )
            for point in raw_path
            if isinstance(point, (list, tuple)) and len(point) >= 2
        )
        if len(path) < 2:
            raise ValueError("楼梯 locomotion path 至少需要两个点。")
        lengths = tuple(
            math.hypot(end[0] - start[0], end[1] - start[1])
            for start, end in zip(path, path[1:])
        )
        if any(length <= 1.0e-6 for length in lengths):
            raise ValueError("楼梯 locomotion path 不能包含重复 XY 点。")
        cumulative = [0.0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)

        self._plan = plan
        self._path = path
        self._segment_lengths = lengths
        self._cumulative_lengths = tuple(cumulative)
        self._segment_index = 1
        self._done = False
        self._failed = False
        self._failure_reason = ""
        self._best_progress_m = 0.0
        self._last_progress_time_s = None
        self._status = {
            "controller": "stair_heading_tracker",
            "ready": True,
            "failed": False,
            "done": False,
            "path_point_count": len(path),
            "path_length_m": cumulative[-1],
            "segment_index": self._segment_index,
            "float_enabled": False,
            "base_pose_lock_requested": False,
        }

    def compute_action(self, state: SimulationState) -> RobotAction:
        tracking = self._update_tracking(state)
        if self._done or self._failed:
            return RobotAction(
                base_velocity=(0.0, 0.0, 0.0),
                source=(
                    "stair_locomotion_completed"
                    if self._done
                    else "stair_locomotion_failed"
                ),
                metadata=self._action_metadata(
                    tracking,
                    command=(0.0, 0.0, 0.0),
                ),
            )

        heading_error = float(tracking["heading_error_rad"])
        abs_heading_error = abs(heading_error)
        speed_scale = 1.0 - _clamp(
            (
                abs_heading_error
                - self.config.full_speed_heading_error_rad
            )
            / (
                self.config.stop_forward_heading_error_rad
                - self.config.full_speed_heading_error_rad
            ),
            0.0,
            1.0,
        )
        forward_world = self.config.forward_velocity_mps * speed_scale
        if int(tracking["segment_index"]) == len(self._path) - 1:
            forward_world *= _clamp(
                float(tracking["goal_distance_m"])
                / max(self.config.goal_tolerance_m * 2.0, 1.0e-6),
                0.35,
                1.0,
            )

        segment_heading = float(tracking["desired_heading_rad"])
        cross_track = float(tracking["signed_cross_track_error_m"])
        tangent_x = math.cos(segment_heading)
        tangent_y = math.sin(segment_heading)
        lateral_world = _clamp(
            -self.config.cross_track_kp * cross_track,
            -self.config.max_lateral_velocity_mps,
            self.config.max_lateral_velocity_mps,
        )
        normal_x = -tangent_y
        normal_y = tangent_x
        velocity_world_x = tangent_x * forward_world + normal_x * lateral_world
        velocity_world_y = tangent_y * forward_world + normal_y * lateral_world
        robot_yaw = float(tracking["robot_yaw_rad"])
        body_vx = (
            math.cos(robot_yaw) * velocity_world_x
            + math.sin(robot_yaw) * velocity_world_y
        )
        body_vy = (
            -math.sin(robot_yaw) * velocity_world_x
            + math.cos(robot_yaw) * velocity_world_y
        )
        angular_velocity = _clamp(
            self.config.heading_kp * heading_error,
            -self.config.max_angular_velocity_rps,
            self.config.max_angular_velocity_rps,
        )
        command = (float(body_vx), float(body_vy), float(angular_velocity))
        self._status["command"] = list(command)
        return RobotAction(
            base_velocity=command,
            source="stair_locomotion_heading_tracker",
            metadata=self._action_metadata(tracking, command=command),
        )

    def is_done(self, state: SimulationState) -> bool:
        self._update_tracking(state)
        return self._done

    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def _update_tracking(self, state: SimulationState) -> dict[str, float | int | bool | None]:
        if self._plan is None or not self._path:
            raise RuntimeError("楼梯 locomotion executor 尚未 reset。")
        pose_x = float(state.robot_root_pose[0])
        pose_y = float(state.robot_root_pose[1])
        pose_z = float(state.robot_root_pose[2])
        robot_yaw = _yaw_from_wxyz(state.robot_root_pose[3:7])

        while self._segment_index < len(self._path) - 1:
            segment = self._segment_projection(pose_x, pose_y, self._segment_index)
            if (
                float(segment["projection"]) < 0.92
                and float(segment["endpoint_distance_m"])
                > self.config.waypoint_tolerance_m
            ):
                break
            self._segment_index += 1

        segment = self._segment_projection(pose_x, pose_y, self._segment_index)
        desired_heading = float(segment["heading_rad"])
        heading_error = _wrap_angle(desired_heading - robot_yaw)
        progress = (
            self._cumulative_lengths[self._segment_index - 1]
            + _clamp(float(segment["projection"]), 0.0, 1.0)
            * self._segment_lengths[self._segment_index - 1]
        )
        goal = self._plan.goal
        goal_distance = math.hypot(pose_x - float(goal.x), pose_y - float(goal.y))
        goal_z_error = None if goal.z is None else abs(pose_z - float(goal.z))
        goal_reached = (
            goal_distance <= self.config.goal_tolerance_m
            and (
                goal_z_error is None
                or goal_z_error <= self.config.goal_z_tolerance_m
            )
        )
        self._done = bool(goal_reached)

        timestamp = float(state.timestamp)
        if progress >= self._best_progress_m + self.config.stall_progress_m:
            self._best_progress_m = progress
            self._last_progress_time_s = timestamp
        elif self._last_progress_time_s is None:
            self._last_progress_time_s = timestamp
        elif (
            not self._done
            and timestamp - self._last_progress_time_s
            >= self.config.stall_timeout_s
        ):
            self._failed = True
            self._failure_reason = "stair_locomotion_stalled"

        cross_track_error = abs(float(segment["signed_cross_track_error_m"]))
        if not self._done and cross_track_error > self.config.max_path_deviation_m:
            self._failed = True
            self._failure_reason = "stair_locomotion_path_deviation"

        tracking: dict[str, float | int | bool | None] = {
            "segment_index": self._segment_index,
            "segment_count": len(self._path) - 1,
            "robot_yaw_rad": robot_yaw,
            "desired_heading_rad": desired_heading,
            "heading_error_rad": heading_error,
            "signed_cross_track_error_m": float(
                segment["signed_cross_track_error_m"]
            ),
            "path_progress_m": progress,
            "best_path_progress_m": self._best_progress_m,
            "goal_distance_m": goal_distance,
            "goal_z_error_m": goal_z_error,
            "goal_reached": goal_reached,
        }
        self._status.update(tracking)
        self._status.update(
            {
                "failed": self._failed,
                "failure_reason": self._failure_reason,
                "done": self._done,
                "float_enabled": False,
                "base_pose_lock_requested": False,
            }
        )
        return tracking

    def _segment_projection(
        self,
        pose_x: float,
        pose_y: float,
        segment_index: int,
    ) -> dict[str, float]:
        start = self._path[segment_index - 1]
        end = self._path[segment_index]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = self._segment_lengths[segment_index - 1]
        tangent_x = dx / length
        tangent_y = dy / length
        relative_x = pose_x - start[0]
        relative_y = pose_y - start[1]
        projection = (relative_x * tangent_x + relative_y * tangent_y) / length
        signed_cross_track = tangent_x * relative_y - tangent_y * relative_x
        return {
            "projection": projection,
            "signed_cross_track_error_m": signed_cross_track,
            "endpoint_distance_m": math.hypot(pose_x - end[0], pose_y - end[1]),
            "heading_rad": math.atan2(dy, dx),
        }

    @staticmethod
    def _action_metadata(
        tracking: dict[str, float | int | bool | None],
        *,
        command: tuple[float, float, float],
    ) -> dict[str, Any]:
        return {
            "stair_locomotion_smoke": True,
            "navigation_controller": "stair_heading_tracker",
            "desired_stair_heading_rad": tracking["desired_heading_rad"],
            "stair_heading_error_rad": tracking["heading_error_rad"],
            "stair_cross_track_error_m": tracking[
                "signed_cross_track_error_m"
            ],
            "stair_path_progress_m": tracking["path_progress_m"],
            "stair_command_body_vx_mps": float(command[0]),
            "stair_command_body_vy_mps": float(command[1]),
            "stair_command_body_wz_rps": float(command[2]),
            "navigation_base_pose_lock": False,
        }
