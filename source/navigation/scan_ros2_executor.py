"""以 ROS 2 SCAN 完成事件驱动 pipeline 导航生命周期。

该执行器不生成速度命令。启用 ROS 2 bridge 时，``/cmd_vel`` 经唯一安全门
直接写入 locomotion policy；本类只消费仿真观测中的事件和实写报告，决定
何时允许 pipeline 从导航执行阶段进入物理到达验证阶段。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

from source.interfaces.navigation import NavGoal, NavPlan
from source.interfaces.simulation import RobotAction, SimulationState

from .scan_stair_freeze import (
    ScanReferencePath,
    ScanStairFreezeConfig,
    ScanStairFreezeController,
    hash_ground_path_points,
)


_GOAL_SAMPLE_KEYS = (
    "scan_goal_reached_last_sample",
    # 兼容早期原型使用的较长键名；正式 runtime 使用上面的短键名。
    "navigation_ros2_goal_reached_last_sample",
)
_POLICY_WRITE_REPORT_KEY = "scan_cmd_vel_last_write_report"
_REFERENCE_PATH_REPORT_KEY = "scan_reference_path_last_report"
_CONTROLLER_STATUS_REPORT_KEY = "scan_controller_status_last_report"
_PCT_GOAL_REPORT_KEY = "scan_pct_goal_last_report"
_NAVIGATION_STATE_GOAL_REACHED = 6
_CONTROLLER_STATE_GOAL_REACHED = 12
_SUPERVISOR_GOAL_REACHED_STOP_REASONS = frozenset(
    {
        "navigation_status_force_zero",
        "navigation_tracking_not_allowed",
    }
)
_ALLOWED_POST_GOAL_STOP_REASONS = frozenset(
    {
        "point_cloud_timeout",
    }
)
_TEMPORARY_NAVIGATION_STOP_REASONS = frozenset(
    {
        "body_height_preflight",
        "pct_goal_waiting_for_publish",
        "pct_goal_waiting_for_transport_ack",
        "pct_goal_waiting_for_path",
        "scan_stair_freeze",
        "scan_stair_freeze_release",
        "scan_stair_terminal_hold",
    }
)
_STAIR_SENSOR_STOP_REASONS = frozenset(
    {
        "missing_odometry",
        "odometry_from_future",
        "odometry_timeout",
        "missing_point_cloud",
        "point_cloud_from_future",
        "point_cloud_timeout",
    }
)
_MISSING_COMMAND_BOOTSTRAP_STOP_REASONS = frozenset(
    {
        "missing_cmd_vel",
        "missing_odometry",
        "missing_point_cloud",
        "missing_navigation_status",
    }
)


class ScanRos2LifecyclePlanner:
    """为外部 ROS 2 Path 建立 pipeline 生命周期占位计划。

    实际三维 Path 由 `/initial_path` 提供并由 SCAN 消费。本类不重新规划、
    不读取 A*/PCT 地图，也不把首尾直线伪装成在线参考路径。任务提供已校验
    手工 Path 时，计划会携带完整地面高度折线及哈希，供 Isaac 侧与 live
    ``/initial_path`` 代际核对；否则两个 waypoint 只承载本轮目标和来源。
    """

    def __init__(
        self,
        reference_path: ScanReferencePath | None = None,
        *,
        publish_pct_goal: bool = False,
    ) -> None:
        if not isinstance(publish_pct_goal, bool):
            raise TypeError("publish_pct_goal 必须是布尔值。")
        if reference_path is not None and publish_pct_goal:
            raise ValueError("手工 Path 与生产 PCT goal 发布不能同时启用。")
        self.reference_path = reference_path
        self.publish_pct_goal = publish_pct_goal

    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        """返回只承载目标与来源的生命周期计划。"""

        start = (
            float(state.robot_root_pose[0]),
            float(state.robot_root_pose[1]),
            float(state.robot_root_pose[2]),
        )
        end = (
            float(goal.x),
            float(goal.y),
            start[2] if goal.z is None else float(goal.z),
        )
        metadata: dict[str, Any] = {
            "planner": "external_ros2_path_lifecycle",
            "path_source": "/initial_path",
            "path_consumed_by": "scan_planner",
            "pipeline_waypoints_are_control_inputs": False,
        }
        if self.publish_pct_goal:
            if goal.z is None:
                raise ValueError("生产 PCT goal 必须显式提供 base 高度 z。")
            goal_values = (
                float(goal.x),
                float(goal.y),
                float(goal.z),
                float(goal.yaw),
            )
            if not all(math.isfinite(value) for value in goal_values):
                raise ValueError("生产 PCT goal 不能包含 NaN 或无穷值。")
            metadata.update(
                {
                    "path_source": "/pct/global_path",
                    "pct_goal_request": {
                        "frame_id": "world",
                        "position_base_xyz": goal_values[:3],
                        "yaw": goal_values[3],
                        "height_semantics": "base",
                    },
                }
            )
        waypoints: tuple[tuple[float, float, float], ...] = (start, end)
        if self.reference_path is not None:
            waypoints = self.reference_path.points_ground_xyz
            terminal_yaw = _terminal_ground_path_yaw(
                self.reference_path.points_ground_xyz,
            )
            metadata.update(
                {
                    "reference_path_3d_ground": (
                        self.reference_path.points_ground_xyz
                    ),
                    "reference_path_config": self.reference_path.source_path,
                    "reference_path_sha256": self.reference_path.sha256,
                    "reference_path_points_sha256": (
                        self.reference_path.points_sha256
                    ),
                    "reference_path_stair_segment_indices": (
                        self.reference_path.stair_segment_indices
                    ),
                    "reference_path_topic": self.reference_path.topic,
                    "reference_path_frame_id": self.reference_path.frame_id,
                    "reference_path_use_sim_time": (
                        self.reference_path.use_sim_time
                    ),
                    "reference_path_height_semantics": "ground",
                    "reference_path_point_count": len(
                        self.reference_path.points_ground_xyz
                    ),
                    "reference_path_terminal_yaw": terminal_yaw,
                }
            )
        return NavPlan(
            goal=goal,
            waypoints=waypoints,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class ScanRos2NavExecutorConfig:
    """定义 SCAN 完成事件与 policy 零速保持的验收门限。"""

    required_zero_write_ticks: int = 5
    zero_epsilon: float = 1.0e-6
    policy_owner_id: str = "scan_cmd_vel"
    progress_watchdog_timeout_s: float = 4.0
    progress_watchdog_min_displacement_m: float = 0.03
    progress_watchdog_min_forward_command_mps: float = 0.05
    require_live_reference_path: bool = False
    live_reference_path_timeout_s: float = 30.0
    pct_goal_publish_timeout_s: float = 3.0
    pct_goal_transport_ack_timeout_s: float = 3.0
    pct_goal_transport_retry_interval_s: float = 0.10
    controller_status_topic: str = "/planning/controller_status"
    controller_status_frame_id: str = "world"

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_zero_write_ticks, bool)
            or not isinstance(self.required_zero_write_ticks, int)
            or self.required_zero_write_ticks < 1
        ):
            raise ValueError("required_zero_write_ticks 必须是正整数。")
        if (
            isinstance(self.zero_epsilon, bool)
            or not isinstance(self.zero_epsilon, (int, float))
            or not math.isfinite(float(self.zero_epsilon))
            or float(self.zero_epsilon) < 0.0
        ):
            raise ValueError("zero_epsilon 必须是有限非负数。")
        if not isinstance(self.policy_owner_id, str) or not self.policy_owner_id:
            raise ValueError("policy_owner_id 必须是非空字符串。")
        if (
            not isinstance(self.controller_status_topic, str)
            or not self.controller_status_topic.startswith("/")
        ):
            raise ValueError("controller_status_topic 必须是绝对 ROS topic。")
        if self.controller_status_frame_id != "world":
            raise ValueError("controller_status_frame_id 当前必须严格为 world。")
        if not isinstance(self.require_live_reference_path, bool):
            raise TypeError("require_live_reference_path 必须是布尔值。")
        for name, value in (
            ("progress_watchdog_timeout_s", self.progress_watchdog_timeout_s),
            (
                "progress_watchdog_min_displacement_m",
                self.progress_watchdog_min_displacement_m,
            ),
            (
                "progress_watchdog_min_forward_command_mps",
                self.progress_watchdog_min_forward_command_mps,
            ),
            (
                "live_reference_path_timeout_s",
                self.live_reference_path_timeout_s,
            ),
            (
                "pct_goal_publish_timeout_s",
                self.pct_goal_publish_timeout_s,
            ),
            (
                "pct_goal_transport_ack_timeout_s",
                self.pct_goal_transport_ack_timeout_s,
            ),
            (
                "pct_goal_transport_retry_interval_s",
                self.pct_goal_transport_retry_interval_s,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} 必须是有限正数。")


class ScanRos2NavExecutor:
    """等待本轮 SCAN 完成事件和连续 policy 零速实写。

    每次 ``reset()`` 都建立新的导航代次。完成必须依次具备：

    1. 本轮收到一条新的 ``goal_reached=false``；
    2. 此后观察到无安全停车原因的非零 policy 实写；
    3. 此后收到一条新的 ``goal_reached=true``；
    4. true 之后连续若干控制 tick 由同一 owner 实际向 policy 写入精确零速。

    该顺序会拒绝 transient-local 遗留的旧 true，也不会把超时或急停造成的
    零速单独误报为正常到达。最终 true 已由 controller 使用新鲜 Odometry、
    终点距离和低速门限独立确认；因此 true 之后即使点云超时门继续强制零速，
    这些真实、连续的 policy 零速实写仍可作为保持证据。``is_done()`` 与
    ``compute_action()`` 可能在同一个 observation 上重复调用，所有输入均按
    消息或写入序号幂等消费。
    """

    def __init__(
        self,
        config: ScanRos2NavExecutorConfig | None = None,
        *,
        stair_freeze_config: ScanStairFreezeConfig | None = None,
        allow_carry_object_follow: bool = False,
    ) -> None:
        self.config = config or ScanRos2NavExecutorConfig()
        self._stair_freeze = ScanStairFreezeController(stair_freeze_config)
        self._allow_carry_object_follow = bool(allow_carry_object_follow)
        self._plan: NavPlan | None = None
        self._generation = 0
        self._phase = "idle"
        self._done = False
        self._success = False
        self._failed = False
        self._failure_reason = ""
        self._tick_index = 0
        self._last_observation_key: tuple[int, float] | None = None

        self._fresh_false_seen = False
        self._policy_activity_seen = False
        self._root_lock_progress_seen = False
        self._goal_true_seen = False
        self._goal_false_sequence: int | None = None
        self._goal_true_sequence: int | None = None
        self._goal_false_receipt_timestamp: float | None = None
        self._goal_true_receipt_timestamp: float | None = None
        self._activity_write_sequence: int | None = None
        self._activity_write_timestamp: float | None = None
        self._zero_write_streak = 0
        self._supervisor_goal_reached_zero_count = 0
        self._last_supervisor_goal_reached_zero_verified = False

        self._last_goal_identity: Hashable | None = None
        self._last_goal_sequence: int | None = None
        self._last_policy_write_identity: Hashable | None = None
        self._last_invalid_policy_write_identity: Hashable | None = None
        self._last_policy_write_sequence: int | None = None
        self._last_policy_write_step: int | None = None
        self._last_policy_write_timestamp: float | None = None
        self._last_requested_command: tuple[float, float, float] | None = None
        self._last_written_command: tuple[float, float, float] | None = None
        self._last_stop_reasons: tuple[str, ...] = ()
        self._last_navigation_cmd_vel_inhibited = False
        self._last_navigation_cmd_vel_inhibit_reason: str | None = None
        self._invalid_goal_sample_count = 0
        self._invalid_policy_write_count = 0
        self._premature_true_count = 0
        self._goal_true_waiting_for_supervisor_ack_count = 0
        self._post_goal_nonzero_write_count = 0
        self._goal_sequence_reset_count = 0
        self._policy_write_sequence_reset_count = 0
        self._progress_watchdog_active = False
        self._progress_watchdog_window_start_pose_xy: (
            tuple[float, float] | None
        ) = None
        self._progress_watchdog_window_start_timestamp: float | None = None
        self._progress_watchdog_window_sample_count = 0
        self._progress_watchdog_last_displacement_m: float | None = None
        self._progress_watchdog_last_goal_distance_m: float | None = None
        self._progress_watchdog_elapsed_without_progress_s = 0.0
        self._progress_watchdog_progress_event_count = 0
        self._progress_watchdog_reset_count = 0
        self._progress_watchdog_trigger_count = 0
        self._progress_watchdog_last_pause_reason = "not_started"
        self._progress_watchdog_failure_timestamp: float | None = None
        self._progress_watchdog_failure_pose_xy: tuple[float, float] | None = None
        self._invalid_progress_pose_count = 0
        self._expected_reference_path_points_sha256: str | None = None
        self._expected_reference_path_topic: str | None = None
        self._expected_reference_path_stair_segment_indices: (
            tuple[tuple[int, int], ...] | None
        ) = None
        self._live_reference_path_verified = False
        self._live_reference_path_sequence: int | None = None
        self._live_reference_path_stamp_ns: int | None = None
        self._live_reference_path_points_sha256: str | None = None
        self._live_reference_path_terminal_yaw: float | None = None
        self._live_reference_path_identity: Hashable | None = None
        self._live_reference_path_wait_started_timestamp: float | None = None
        self._live_reference_path_source: str | None = None
        self._live_reference_path_generation_count = 0
        self._live_reference_path_goal_bound = False
        self._live_reference_path_goal_xy_error_m: float | None = None
        self._live_reference_path_goal_z_error_m: float | None = None
        self._live_reference_path_goal_yaw_error_rad: float | None = None
        self._invalid_reference_path_report_count = 0
        self._pct_goal_required = False
        self._pct_goal_request: dict[str, object] | None = None
        self._pct_goal_acknowledged = False
        self._pct_goal_transport_acknowledged = False
        self._pct_goal_stamp_ns: int | None = None
        self._pct_goal_publish_sequence: int | None = None
        self._pct_goal_report_identity: Hashable | None = None
        self._pct_goal_wait_started_timestamp: float | None = None
        self._pct_goal_transport_wait_started_timestamp: float | None = None
        self._pct_goal_last_request_timestamp: float | None = None
        self._pct_goal_request_action_count = 0
        self._pct_goal_transport_retry_count = 0
        self._invalid_pct_goal_report_count = 0
        self._reference_path_tombstone_stamp_ns: int | None = None

    def reset(self, plan: NavPlan) -> None:
        """载入新导航目标，并清空上一段路径的全部完成证据。"""

        self._plan = plan
        self._generation += 1
        self._phase = "waiting_for_fresh_false"
        self._done = False
        self._success = False
        self._failed = False
        self._failure_reason = ""
        self._tick_index = 0
        self._last_observation_key = None

        self._fresh_false_seen = False
        self._policy_activity_seen = False
        self._root_lock_progress_seen = False
        self._goal_true_seen = False
        self._goal_false_sequence = None
        self._goal_true_sequence = None
        self._goal_false_receipt_timestamp = None
        self._goal_true_receipt_timestamp = None
        self._activity_write_sequence = None
        self._activity_write_timestamp = None
        self._zero_write_streak = 0
        self._supervisor_goal_reached_zero_count = 0
        self._last_supervisor_goal_reached_zero_verified = False

        self._last_goal_identity = None
        self._last_goal_sequence = None
        self._last_policy_write_identity = None
        self._last_invalid_policy_write_identity = None
        self._last_policy_write_sequence = None
        self._last_policy_write_step = None
        self._last_policy_write_timestamp = None
        self._last_requested_command = None
        self._last_written_command = None
        self._last_stop_reasons = ()
        self._last_navigation_cmd_vel_inhibited = False
        self._last_navigation_cmd_vel_inhibit_reason = None
        self._invalid_goal_sample_count = 0
        self._invalid_policy_write_count = 0
        self._premature_true_count = 0
        self._goal_true_waiting_for_supervisor_ack_count = 0
        self._post_goal_nonzero_write_count = 0
        self._goal_sequence_reset_count = 0
        self._policy_write_sequence_reset_count = 0
        self._progress_watchdog_active = False
        self._progress_watchdog_window_start_pose_xy = None
        self._progress_watchdog_window_start_timestamp = None
        self._progress_watchdog_window_sample_count = 0
        self._progress_watchdog_last_displacement_m = None
        self._progress_watchdog_last_goal_distance_m = None
        self._progress_watchdog_elapsed_without_progress_s = 0.0
        self._progress_watchdog_progress_event_count = 0
        self._progress_watchdog_reset_count = 0
        self._progress_watchdog_trigger_count = 0
        self._progress_watchdog_last_pause_reason = "not_started"
        self._progress_watchdog_failure_timestamp = None
        self._progress_watchdog_failure_pose_xy = None
        self._invalid_progress_pose_count = 0
        reference_path = plan.metadata.get("reference_path_3d_ground")
        expected_points_sha256 = plan.metadata.get(
            "reference_path_points_sha256"
        )
        self._expected_reference_path_points_sha256 = (
            str(expected_points_sha256)
            if isinstance(expected_points_sha256, str)
            and len(expected_points_sha256) == 64
            else None
        )
        expected_topic = plan.metadata.get("reference_path_topic")
        self._expected_reference_path_topic = (
            str(expected_topic)
            if isinstance(expected_topic, str) and expected_topic.startswith("/")
            else None
        )
        raw_stair_segments = plan.metadata.get(
            "reference_path_stair_segment_indices"
        )
        self._expected_reference_path_stair_segment_indices = (
            tuple(tuple(int(value) for value in segment) for segment in raw_stair_segments)
            if isinstance(raw_stair_segments, (list, tuple))
            else None
        )
        self._live_reference_path_verified = False
        self._live_reference_path_sequence = None
        self._live_reference_path_stamp_ns = None
        self._live_reference_path_points_sha256 = None
        self._live_reference_path_terminal_yaw = None
        self._live_reference_path_identity = None
        self._live_reference_path_wait_started_timestamp = None
        self._live_reference_path_source = None
        self._live_reference_path_generation_count = 0
        self._live_reference_path_goal_bound = False
        self._live_reference_path_goal_xy_error_m = None
        self._live_reference_path_goal_z_error_m = None
        self._live_reference_path_goal_yaw_error_rad = None
        self._invalid_reference_path_report_count = 0
        raw_pct_goal = plan.metadata.get("pct_goal_request")
        self._pct_goal_request = _normalize_pct_goal_request(raw_pct_goal)
        self._pct_goal_required = self._pct_goal_request is not None
        terminal_goal_base_xyzyaw = _nav_goal_base_xyzyaw(plan.goal)
        if self._pct_goal_required and not self.config.require_live_reference_path:
            raise ValueError(
                "生产 PCT goal 必须同时启用 live reference Path 验收。"
            )
        if self._pct_goal_required:
            if terminal_goal_base_xyzyaw is None:
                raise ValueError("生产 PCT goal 缺少 base 高度 terminal NavGoal。")
            request = self._pct_goal_request or {}
            request_position = request.get("position_base_xyz")
            request_yaw = request.get("yaw")
            if (
                request_position != terminal_goal_base_xyzyaw[:3]
                or request_yaw != terminal_goal_base_xyzyaw[3]
            ):
                raise ValueError("pct_goal_request 与 NavPlan.goal 不一致。")
        self._pct_goal_acknowledged = False
        self._pct_goal_transport_acknowledged = False
        self._pct_goal_stamp_ns = None
        self._pct_goal_publish_sequence = None
        self._pct_goal_report_identity = None
        self._pct_goal_wait_started_timestamp = None
        self._pct_goal_transport_wait_started_timestamp = None
        self._pct_goal_last_request_timestamp = None
        self._pct_goal_request_action_count = 0
        self._pct_goal_transport_retry_count = 0
        self._invalid_pct_goal_report_count = 0
        self._reference_path_tombstone_stamp_ns = None
        self._stair_freeze.reset(
            (
                None
                if self.config.require_live_reference_path
                else (
                    reference_path
                    if isinstance(reference_path, (list, tuple))
                    else None
                )
            ),
            path_source=(
                str(plan.metadata.get("reference_path_config"))
                if plan.metadata.get("reference_path_config") is not None
                else None
            ),
            path_sha256=(
                str(plan.metadata.get("reference_path_sha256"))
                if plan.metadata.get("reference_path_sha256") is not None
                else None
            ),
            path_points_sha256=(
                str(plan.metadata.get("reference_path_points_sha256"))
                if (
                    not self.config.require_live_reference_path
                    and plan.metadata.get("reference_path_points_sha256") is not None
                )
                else None
            ),
            path_terminal_yaw=(
                float(plan.metadata["reference_path_terminal_yaw"])
                if (
                    not self.config.require_live_reference_path
                    and isinstance(
                        plan.metadata.get("reference_path_terminal_yaw"),
                        (int, float),
                    )
                    and not isinstance(
                        plan.metadata.get("reference_path_terminal_yaw"),
                        bool,
                    )
                )
                else None
            ),
            terminal_goal_base_xyzyaw=terminal_goal_base_xyzyaw,
            stair_segment_indices=(
                plan.metadata.get("reference_path_stair_segment_indices")
                if (
                    not self.config.require_live_reference_path
                    and isinstance(
                    plan.metadata.get("reference_path_stair_segment_indices"),
                    (list, tuple),
                    )
                )
                else None
            ),
            carry_object_follow=(
                self._allow_carry_object_follow
                and plan.metadata.get("execution_phase") == "carry_nav_to_place"
            ),
        )

    def compute_action(self, state: SimulationState) -> RobotAction:
        """消费本 tick 的验收证据，但绝不与 ROS 2 ``/cmd_vel`` 争夺控制权。"""

        self._observe(state)
        if self._failed:
            return self._emergency_stop_action(
                state,
                source="scan_ros2_navigation_failed",
            )
        if self._pct_goal_required and not self._live_reference_path_verified:
            transport_acknowledged = self._pct_goal_transport_acknowledged
            retry_due = self._pct_goal_request_due(state)
            if self._failed:
                return self._emergency_stop_action(
                    state,
                    source="scan_ros2_navigation_failed",
                )
            metadata: dict[str, object] = {
                "navigation_cmd_vel_inhibit": True,
                "navigation_cmd_vel_inhibit_reason": (
                    "pct_goal_waiting_for_path"
                    if transport_acknowledged
                    else "pct_goal_waiting_for_transport_ack"
                    if self._pct_goal_acknowledged
                    else "pct_goal_waiting_for_publish"
                ),
            }
            if retry_due and not transport_acknowledged:
                transport_retry = self._pct_goal_acknowledged
                metadata["navigation_pct_goal_request"] = {
                    **(self._pct_goal_request or {}),
                    "generation": self._generation,
                    "transport_retry": transport_retry,
                }
                self._pct_goal_last_request_timestamp = float(state.timestamp)
                self._pct_goal_request_action_count += 1
                if transport_retry:
                    self._pct_goal_transport_retry_count += 1
            return RobotAction(
                base_velocity=(0.0, 0.0, 0.0),
                source=(
                    "scan_pct_goal_waiting_for_path"
                    if transport_acknowledged
                    else "scan_pct_goal_transport_retry"
                    if retry_due and self._pct_goal_acknowledged
                    else "scan_pct_goal_waiting_for_transport_ack"
                    if self._pct_goal_acknowledged
                    else "scan_pct_goal_publish"
                    if retry_due
                    else "scan_pct_goal_waiting_for_publish"
                ),
                metadata=metadata,
            )
        try:
            freeze_action = self._stair_freeze.compute_action(state)
        except (RuntimeError, ValueError) as exc:
            self._failed = True
            self._success = False
            stair_reason = self._stair_freeze.status().get("reason")
            self._failure_reason = (
                str(stair_reason)
                if isinstance(stair_reason, str)
                and stair_reason.startswith("stair_")
                else "scan_stair_freeze_failed"
            )
            self._phase = "failed"
            return self._emergency_stop_action(
                state,
                source="scan_stair_freeze_failed",
                extra_metadata={
                    "navigation_stair_freeze_error": str(exc),
                },
            )
        if freeze_action is not None:
            self._root_lock_progress_seen = bool(
                self._root_lock_progress_seen
                or self._stair_freeze.certified_progress_seen
            )
            return freeze_action
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            source="scan_ros2_navigation",
            metadata={},
        )

    def _pct_goal_request_due(self, state: SimulationState) -> bool:
        """按仿真时间节流同一目标的 transport 重触发请求。"""

        timestamp = float(state.timestamp)
        if not math.isfinite(timestamp):
            self._fail_reference_path("pct_goal_publish_timestamp_invalid")
            return False
        previous = self._pct_goal_last_request_timestamp
        if previous is None:
            return True
        if timestamp < previous:
            self._fail_reference_path("pct_goal_publish_timestamp_regressed")
            return False
        return (
            timestamp - previous
            >= self.config.pct_goal_transport_retry_interval_s
        )

    def _emergency_stop_action(
        self,
        state: SimulationState,
        *,
        source: str,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> RobotAction:
        """构造零速急停；楼梯锁已生效时必须保留最后 root 与关节锁。"""

        hold_action = self._stair_freeze.emergency_hold_action(
            state,
            reason=self._failure_reason,
        )
        metadata: dict[str, Any] = {}
        if hold_action is not None:
            metadata.update(hold_action.metadata)
            source = hold_action.source
        metadata.update(
            {
                "navigation_emergency_stop": True,
                "navigation_emergency_stop_reason": self._failure_reason,
                "navigation_global_replan_requested": True,
            }
        )
        if extra_metadata is not None:
            metadata.update(extra_metadata)
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            arm_joint_positions=(
                None if hold_action is None else hold_action.arm_joint_positions
            ),
            gripper_command=(
                None if hold_action is None else hold_action.gripper_command
            ),
            source=source,
            metadata=metadata,
        )

    def is_done(self, state: SimulationState) -> bool:
        """仅在完成事件和连续有效零速均通过后返回真。"""

        self._observe(state)
        return self._done

    def status(self) -> dict[str, Any]:
        """返回可写入 episode summary 的完整 SCAN 验收证据。"""

        goal = None
        goal_z = None
        execution_phase = None
        if self._plan is not None:
            goal = (
                float(self._plan.goal.x),
                float(self._plan.goal.y),
                float(self._plan.goal.yaw),
            )
            goal_z = (
                None
                if self._plan.goal.z is None
                else float(self._plan.goal.z)
            )
            execution_phase = self._plan.metadata.get("execution_phase")
        stair_freeze_status = self._stair_freeze.status()
        return {
            "backend": "scan_ros2_goal_event",
            "phase": self._phase,
            "generation": self._generation,
            "tick_index": self._tick_index,
            "done": self._done,
            "success": self._success,
            "failed": self._failed,
            "failure_reason": self._failure_reason,
            "goal": goal,
            "goal_z": goal_z,
            "execution_phase": execution_phase,
            "fresh_false_seen": self._fresh_false_seen,
            "policy_activity_seen": self._policy_activity_seen,
            "certified_root_lock_progress_seen": self._root_lock_progress_seen,
            "execution_activity_seen": (
                self._policy_activity_seen or self._root_lock_progress_seen
            ),
            "goal_rising_edge_seen": self._goal_true_seen,
            "goal_false_sequence": self._goal_false_sequence,
            "goal_true_sequence": self._goal_true_sequence,
            "goal_false_receipt_timestamp": (
                self._goal_false_receipt_timestamp
            ),
            "goal_true_receipt_timestamp": self._goal_true_receipt_timestamp,
            "activity_write_sequence": self._activity_write_sequence,
            "activity_write_timestamp": self._activity_write_timestamp,
            "zero_write_streak": self._zero_write_streak,
            "supervisor_goal_reached_zero_count": (
                self._supervisor_goal_reached_zero_count
            ),
            "last_supervisor_goal_reached_zero_verified": (
                self._last_supervisor_goal_reached_zero_verified
            ),
            "required_zero_write_ticks": (
                self.config.required_zero_write_ticks
            ),
            "last_policy_write_sequence": self._last_policy_write_sequence,
            "last_policy_write_timestamp": self._last_policy_write_timestamp,
            "last_requested_command": self._last_requested_command,
            "last_written_command": self._last_written_command,
            "last_stop_reasons": self._last_stop_reasons,
            "last_navigation_cmd_vel_inhibited": (
                self._last_navigation_cmd_vel_inhibited
            ),
            "last_navigation_cmd_vel_inhibit_reason": (
                self._last_navigation_cmd_vel_inhibit_reason
            ),
            "invalid_goal_sample_count": self._invalid_goal_sample_count,
            "invalid_policy_write_count": self._invalid_policy_write_count,
            "premature_true_count": self._premature_true_count,
            "goal_true_waiting_for_supervisor_ack_count": (
                self._goal_true_waiting_for_supervisor_ack_count
            ),
            "post_goal_nonzero_write_count": (
                self._post_goal_nonzero_write_count
            ),
            "goal_sequence_reset_count": self._goal_sequence_reset_count,
            "policy_write_sequence_reset_count": (
                self._policy_write_sequence_reset_count
            ),
            "progress_watchdog_active": self._progress_watchdog_active,
            "progress_watchdog_timeout_s": (
                self.config.progress_watchdog_timeout_s
            ),
            "progress_watchdog_min_displacement_m": (
                self.config.progress_watchdog_min_displacement_m
            ),
            "progress_watchdog_min_forward_command_mps": (
                self.config.progress_watchdog_min_forward_command_mps
            ),
            "progress_watchdog_source": "fixed_window_net_xy",
            "progress_watchdog_window_start_pose_xy": (
                self._progress_watchdog_window_start_pose_xy
            ),
            "progress_watchdog_window_start_timestamp": (
                self._progress_watchdog_window_start_timestamp
            ),
            "progress_watchdog_window_sample_count": (
                self._progress_watchdog_window_sample_count
            ),
            "progress_watchdog_last_displacement_m": (
                self._progress_watchdog_last_displacement_m
            ),
            "progress_watchdog_last_goal_distance_m": (
                self._progress_watchdog_last_goal_distance_m
            ),
            "progress_watchdog_elapsed_without_progress_s": (
                self._progress_watchdog_elapsed_without_progress_s
            ),
            "progress_watchdog_progress_event_count": (
                self._progress_watchdog_progress_event_count
            ),
            "progress_watchdog_reset_count": (
                self._progress_watchdog_reset_count
            ),
            "progress_watchdog_trigger_count": (
                self._progress_watchdog_trigger_count
            ),
            "progress_watchdog_last_pause_reason": (
                self._progress_watchdog_last_pause_reason
            ),
            "progress_watchdog_failure_timestamp": (
                self._progress_watchdog_failure_timestamp
            ),
            "progress_watchdog_failure_pose_xy": (
                self._progress_watchdog_failure_pose_xy
            ),
            "invalid_progress_pose_count": self._invalid_progress_pose_count,
            "scan_controller_goal_reached_verified": (
                self._goal_true_seen and self._done
            ),
            "policy_zero_hold_verified": (
                self._zero_write_streak
                >= self.config.required_zero_write_ticks
            ),
            "stair_freeze": stair_freeze_status,
            "stair_sensor_acquisition_pending": bool(
                stair_freeze_status.get("sensor_acquisition_pending") is True
            ),
            "stair_freeze_finish_ready": (
                self._stair_freeze.finish_ready
                and (
                    not self.config.require_live_reference_path
                    or self._live_reference_path_verified
                )
            ),
            "live_reference_path_required": (
                self.config.require_live_reference_path
            ),
            "live_reference_path_verified": self._live_reference_path_verified,
            "live_reference_path_timeout_s": (
                self.config.live_reference_path_timeout_s
            ),
            "live_reference_path_wait_started_timestamp": (
                self._live_reference_path_wait_started_timestamp
            ),
            "live_reference_path_sequence": self._live_reference_path_sequence,
            "live_reference_path_stamp_ns": self._live_reference_path_stamp_ns,
            "live_reference_path_points_sha256": (
                self._live_reference_path_points_sha256
            ),
            "live_reference_path_terminal_yaw": (
                self._live_reference_path_terminal_yaw
            ),
            "live_reference_path_source": self._live_reference_path_source,
            "live_reference_path_generation_count": (
                self._live_reference_path_generation_count
            ),
            "live_reference_path_goal_bound": (
                self._live_reference_path_goal_bound
            ),
            "live_reference_path_goal_xy_error_m": (
                self._live_reference_path_goal_xy_error_m
            ),
            "live_reference_path_goal_z_error_m": (
                self._live_reference_path_goal_z_error_m
            ),
            "live_reference_path_goal_yaw_error_rad": (
                self._live_reference_path_goal_yaw_error_rad
            ),
            "expected_reference_path_points_sha256": (
                self._expected_reference_path_points_sha256
            ),
            "expected_reference_path_topic": self._expected_reference_path_topic,
            "invalid_reference_path_report_count": (
                self._invalid_reference_path_report_count
            ),
            "pct_goal_required": self._pct_goal_required,
            "pct_goal_local_publish_triggered": self._pct_goal_acknowledged,
            "pct_goal_acknowledged": self._pct_goal_acknowledged,
            "pct_goal_transport_acknowledged": (
                self._pct_goal_transport_acknowledged
            ),
            "pct_goal_stamp_ns": self._pct_goal_stamp_ns,
            "pct_goal_publish_sequence": self._pct_goal_publish_sequence,
            "pct_goal_wait_started_timestamp": (
                self._pct_goal_wait_started_timestamp
            ),
            "pct_goal_publish_timeout_s": (
                self.config.pct_goal_publish_timeout_s
            ),
            "pct_goal_transport_wait_started_timestamp": (
                self._pct_goal_transport_wait_started_timestamp
            ),
            "pct_goal_transport_ack_timeout_s": (
                self.config.pct_goal_transport_ack_timeout_s
            ),
            "pct_goal_transport_retry_interval_s": (
                self.config.pct_goal_transport_retry_interval_s
            ),
            "pct_goal_request_action_count": (
                self._pct_goal_request_action_count
            ),
            "pct_goal_transport_retry_count": (
                self._pct_goal_transport_retry_count
            ),
            "invalid_pct_goal_report_count": (
                self._invalid_pct_goal_report_count
            ),
            "reference_path_tombstone_stamp_ns": (
                self._reference_path_tombstone_stamp_ns
            ),
            "acceptance_mode": (
                "fresh_false_execution_activity_true_stair_fresh_bspline_"
                "twist_or_terminal_hold_policy_zero_hold"
            ),
        }

    def _observe(self, state: SimulationState) -> None:
        if self._plan is None or self._done or self._failed:
            return
        observation_key = (int(state.step_index), float(state.timestamp))
        if observation_key != self._last_observation_key:
            self._last_observation_key = observation_key
            self._tick_index += 1
        self._consume_pct_goal_report(state)
        if self._failed:
            return
        self._check_pct_goal_publish_timeout(state)
        if self._failed or (
            self._pct_goal_required and not self._pct_goal_acknowledged
        ):
            return
        self._consume_live_reference_path(state)
        if self._failed:
            return
        self._check_pct_goal_transport_ack_timeout(state)
        if self._failed or (
            self._pct_goal_required
            and not self._pct_goal_transport_acknowledged
        ):
            return
        self._check_live_reference_path_timeout(state)
        if self._failed:
            return
        self._stair_freeze.observe_controller_status(
            state.metadata.get(_CONTROLLER_STATUS_REPORT_KEY),
            expected_topic=self.config.controller_status_topic,
            expected_frame_id=self.config.controller_status_frame_id,
        )
        self._stair_freeze.observe_policy_write(
            state.metadata.get(_POLICY_WRITE_REPORT_KEY),
            owner_id=self.config.policy_owner_id,
        )
        self._root_lock_progress_seen = bool(
            self._root_lock_progress_seen
            or self._stair_freeze.certified_progress_seen
        )

        # 同一个 observation 中先消费逻辑状态，再消费 policy 实写。这样
        # false 与非零实写可以建立本轮活动，而 true 与残留非零同 tick 时不会
        # 把该非零倒算到 true 之前。
        goal_true_before = self._goal_true_seen
        self._consume_goal_sample(state.metadata)
        goal_became_true = self._goal_true_seen and not goal_true_before
        self._consume_policy_write(
            state,
            count_post_goal_zero=not goal_became_true,
        )
        if (
            self._goal_true_seen
            and self._zero_write_streak
            >= self.config.required_zero_write_ticks
            and self._stair_freeze.finish_ready
            and (
                not self.config.require_live_reference_path
                or self._live_reference_path_verified
            )
        ):
            self._done = True
            self._success = True
            self._phase = "completed"

    def _consume_pct_goal_report(self, state: SimulationState) -> None:
        """绑定 runtime 实际发布的本代 ``/pct/goal`` 精确时间戳。"""

        if not self._pct_goal_required:
            return
        raw_report = state.metadata.get(_PCT_GOAL_REPORT_KEY)
        if raw_report is None:
            return
        if not isinstance(raw_report, Mapping):
            self._invalid_pct_goal_report_count += 1
            self._fail_reference_path("pct_goal_publish_report_invalid")
            return
        generation = raw_report.get("generation")
        sequence = raw_report.get("sequence")
        position = raw_report.get("position_base_xyz")
        yaw = raw_report.get("yaw")
        stamp = raw_report.get("stamp")
        if (
            raw_report.get("published") is not True
            or raw_report.get("source") != "isaac_ros2_ogn_pose_stamped"
            or raw_report.get("frame_id") != "world"
            or raw_report.get("height_semantics") != "base"
            or not isinstance(raw_report.get("topic"), str)
            or not str(raw_report.get("topic")).startswith("/")
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or not isinstance(position, (list, tuple))
            or len(position) != 3
            or not isinstance(stamp, Mapping)
        ):
            self._invalid_pct_goal_report_count += 1
            self._fail_reference_path("pct_goal_publish_report_invalid")
            return
        if generation < self._generation:
            return
        if generation > self._generation:
            self._fail_reference_path("pct_goal_publish_generation_ahead")
            return
        try:
            normalized_position = tuple(float(value) for value in position)
            normalized_yaw = float(yaw)
            stamp_ns = _stamp_mapping_to_ns(stamp, field_name="pct_goal.stamp")
        except (TypeError, ValueError) as exc:
            self._invalid_pct_goal_report_count += 1
            self._fail_reference_path(
                "pct_goal_publish_report_invalid",
                detail=str(exc),
            )
            return
        if not all(
            math.isfinite(value)
            for value in (*normalized_position, normalized_yaw)
        ):
            self._invalid_pct_goal_report_count += 1
            self._fail_reference_path("pct_goal_publish_report_invalid")
            return
        expected = self._pct_goal_request or {}
        if (
            normalized_position != expected.get("position_base_xyz")
            or normalized_yaw != expected.get("yaw")
            or raw_report.get("frame_id") != expected.get("frame_id")
            or raw_report.get("height_semantics")
            != expected.get("height_semantics")
            or raw_report.get("effective_goal_provenance_required", False)
            != expected.get("effective_goal_provenance_required", False)
            or (
                expected.get("effective_goal_provenance_required") is True
                and raw_report.get("effective_goal_provenance")
                != expected.get("effective_goal_provenance")
            )
        ):
            self._fail_reference_path("pct_goal_publish_report_mismatch")
            return
        identity = (generation, sequence, stamp_ns, *normalized_position, normalized_yaw)
        if identity == self._pct_goal_report_identity:
            return
        if self._pct_goal_acknowledged:
            self._fail_reference_path("pct_goal_publish_not_exactly_once")
            return
        self._pct_goal_acknowledged = True
        self._pct_goal_stamp_ns = stamp_ns
        self._pct_goal_publish_sequence = sequence
        self._pct_goal_report_identity = identity
        self._pct_goal_transport_wait_started_timestamp = float(state.timestamp)
        self._phase = "waiting_for_pct_goal_transport_ack"

    def _check_pct_goal_publish_timeout(self, state: SimulationState) -> None:
        """要求生产目标在有限仿真时间内得到 OGN 发布确认。"""

        if not self._pct_goal_required or self._pct_goal_acknowledged:
            return
        timestamp = float(state.timestamp)
        if not math.isfinite(timestamp):
            self._fail_reference_path("pct_goal_publish_timestamp_invalid")
            return
        if self._pct_goal_wait_started_timestamp is None:
            self._pct_goal_wait_started_timestamp = timestamp
            self._phase = "waiting_for_pct_goal_publish"
            return
        if (
            timestamp - self._pct_goal_wait_started_timestamp
            > self.config.pct_goal_publish_timeout_s
        ):
            self._fail_reference_path("pct_goal_publish_timeout")

    def _check_pct_goal_transport_ack_timeout(
        self,
        state: SimulationState,
    ) -> None:
        """要求 PCT 通过新 Path 代际确认已经收到同一 stamped goal。"""

        if (
            not self._pct_goal_required
            or not self._pct_goal_acknowledged
            or self._pct_goal_transport_acknowledged
        ):
            return
        timestamp = float(state.timestamp)
        if not math.isfinite(timestamp):
            self._fail_reference_path("pct_goal_transport_timestamp_invalid")
            return
        if self._pct_goal_transport_wait_started_timestamp is None:
            self._pct_goal_transport_wait_started_timestamp = timestamp
            self._phase = "waiting_for_pct_goal_transport_ack"
            return
        if (
            timestamp - self._pct_goal_transport_wait_started_timestamp
            > self.config.pct_goal_transport_ack_timeout_s
        ):
            self._fail_reference_path("pct_goal_transport_ack_timeout")

    def _consume_live_reference_path(self, state: SimulationState) -> None:
        """把 Isaac 侧实际收到的 Path 代际绑定到冻结状态机。"""

        raw_report = state.metadata.get(_REFERENCE_PATH_REPORT_KEY)
        if raw_report is None:
            return
        if not isinstance(raw_report, Mapping):
            self._invalid_reference_path_report_count += 1
            self._fail_reference_path("scan_reference_path_report_invalid")
            return
        sequence = raw_report.get("sequence")
        points = raw_report.get("points_ground_xyz")
        points_sha256 = raw_report.get("points_sha256")
        source = raw_report.get("source")
        topic = raw_report.get("topic")
        frame_id = raw_report.get("frame_id")
        stamp = raw_report.get("stamp")
        raw_terminal_yaw = raw_report.get("terminal_yaw")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or not isinstance(points, (list, tuple))
            or not isinstance(points_sha256, str)
            or len(points_sha256) != 64
            or source != "ros2_nav_msgs_path"
            or not isinstance(topic, str)
            or not topic.startswith("/")
            or frame_id != "world"
            or not isinstance(stamp, Mapping)
            or (
                raw_terminal_yaw is not None
                and (
                    isinstance(raw_terminal_yaw, bool)
                    or not isinstance(raw_terminal_yaw, (int, float))
                )
            )
        ):
            self._invalid_reference_path_report_count += 1
            self._fail_reference_path("scan_reference_path_report_invalid")
            return
        if (
            self._expected_reference_path_topic is not None
            and topic != self._expected_reference_path_topic
        ):
            self._fail_reference_path("scan_reference_path_topic_mismatch")
            return
        try:
            stamp_ns = _stamp_mapping_to_ns(
                stamp,
                field_name="reference_path.stamp",
            )
        except ValueError as exc:
            self._invalid_reference_path_report_count += 1
            self._fail_reference_path(
                "scan_reference_path_report_invalid",
                detail=str(exc),
            )
            return
        cleared = len(points) == 0
        if raw_report.get("cleared") is not cleared:
            self._invalid_reference_path_report_count += 1
            self._fail_reference_path("scan_reference_path_report_invalid")
            return
        terminal_yaw = (
            None
            if raw_terminal_yaw is None
            else _finite_number(raw_terminal_yaw)
        )
        if (
            (cleared and terminal_yaw is not None)
            or (not cleared and terminal_yaw is None)
        ):
            self._invalid_reference_path_report_count += 1
            self._fail_reference_path("scan_reference_path_report_invalid")
            return
        identity = (sequence, stamp_ns, points_sha256, terminal_yaw)
        if identity == self._live_reference_path_identity:
            return
        if (
            self._live_reference_path_sequence is not None
            and sequence < self._live_reference_path_sequence
        ):
            self._fail_reference_path("scan_reference_path_sequence_regressed")
            return
        try:
            actual_points_sha256 = hash_ground_path_points(points)
        except (TypeError, ValueError) as exc:
            self._invalid_reference_path_report_count += 1
            self._fail_reference_path(
                "scan_reference_path_report_invalid",
                detail=str(exc),
            )
            return
        if actual_points_sha256 != points_sha256:
            self._fail_reference_path("scan_reference_path_hash_invalid")
            return
        if self._pct_goal_required:
            goal_stamp_ns = self._pct_goal_stamp_ns
            if goal_stamp_ns is None:
                return
            if stamp_ns < goal_stamp_ns or (
                not cleared and stamp_ns == goal_stamp_ns
            ):
                # reset 后 metadata 可能仍携带上一导航阶段的 transient Path；
                # 旧代只能被忽略，绝不能绑定到当前 task goal。
                return
            if (
                not cleared
                and self._reference_path_tombstone_stamp_ns is not None
                and stamp_ns <= self._reference_path_tombstone_stamp_ns
            ):
                self._fail_reference_path(
                    "scan_reference_path_not_newer_than_tombstone"
                )
                return
        if (
            self._live_reference_path_stamp_ns is not None
            and stamp_ns < self._live_reference_path_stamp_ns
        ):
            return
        if (
            self._live_reference_path_stamp_ns is not None
            and stamp_ns == self._live_reference_path_stamp_ns
            and (
                self._live_reference_path_points_sha256 != points_sha256
                or self._live_reference_path_terminal_yaw != terminal_yaw
            )
        ):
            self._fail_reference_path("scan_reference_path_stamp_conflict")
            return
        freeze_phase = str(self._stair_freeze.status().get("phase"))
        freeze_active = freeze_phase in {
            "active",
            "full_lock_settle",
            "root_release_settle",
            "release_action_pending",
            "post_release_stabilizing",
            "resume_wait_fresh_cmd",
            "terminal_hold",
        }
        if cleared:
            if self._expected_reference_path_points_sha256 is not None:
                self._fail_reference_path("scan_reference_path_cleared")
                return
            if freeze_active:
                self._fail_reference_path(
                    "scan_reference_path_cleared_during_stair_freeze"
                )
                return
            if self._pct_goal_required:
                self._pct_goal_transport_acknowledged = True
                self._live_reference_path_wait_started_timestamp = float(
                    state.timestamp
                )
            self._reference_path_tombstone_stamp_ns = stamp_ns
            self._live_reference_path_verified = False
            self._live_reference_path_sequence = sequence
            self._live_reference_path_stamp_ns = stamp_ns
            self._live_reference_path_points_sha256 = points_sha256
            self._live_reference_path_terminal_yaw = None
            self._live_reference_path_identity = identity
            self._live_reference_path_source = str(source)
            self._live_reference_path_goal_bound = False
            self._live_reference_path_goal_xy_error_m = None
            self._live_reference_path_goal_z_error_m = None
            self._live_reference_path_goal_yaw_error_rad = None
            self._stair_freeze.reset(None)
            self._reset_acceptance_chain()
            self._phase = "waiting_for_live_reference_path"
            return
        if len(points) < 2:
            self._fail_reference_path("scan_reference_path_report_invalid")
            return
        expected_sha256 = self._expected_reference_path_points_sha256
        if expected_sha256 is not None and points_sha256 != expected_sha256:
            self._fail_reference_path("scan_reference_path_mismatch")
            return
        assert terminal_yaw is not None
        if not self._bind_live_reference_path_goal(
            points,
            terminal_yaw=terminal_yaw,
        ):
            return
        if self._pct_goal_required:
            # 非空当前代 Path 本身同样是 PCT 已收到 goal 的强证据；即使
            # depth=1 让中间 tombstone 被覆盖，也不能继续无意义重发。
            self._pct_goal_transport_acknowledged = True
        previous_sha256 = self._live_reference_path_points_sha256
        if (
            previous_sha256 is not None
            and previous_sha256 != points_sha256
            and freeze_active
        ):
            self._fail_reference_path(
                "scan_reference_path_changed_during_stair_freeze"
            )
            return
        stair_segments = (
            self._expected_reference_path_stair_segment_indices
            if expected_sha256 == points_sha256
            else None
        )
        if previous_sha256 != points_sha256 or not freeze_active:
            try:
                self._stair_freeze.reset(
                    points,
                    path_source=(
                        f"ros2:{topic}@{stamp.get('sec')}:{stamp.get('nanosec')}"
                    ),
                    path_points_sha256=points_sha256,
                    path_stamp_ns=stamp_ns,
                    path_terminal_yaw=terminal_yaw,
                    terminal_goal_base_xyzyaw=(
                        None
                        if self._plan is None
                        else _nav_goal_base_xyzyaw(self._plan.goal)
                    ),
                    stair_segment_indices=stair_segments,
                    carry_object_follow=(
                        self._allow_carry_object_follow
                        and self._plan is not None
                        and self._plan.metadata.get("execution_phase")
                        == "carry_nav_to_place"
                    ),
                )
            except (TypeError, ValueError) as exc:
                self._fail_reference_path(
                    "scan_reference_path_report_invalid",
                    detail=str(exc),
                )
                return
        self._live_reference_path_verified = True
        self._live_reference_path_sequence = sequence
        self._live_reference_path_stamp_ns = stamp_ns
        self._live_reference_path_points_sha256 = points_sha256
        self._live_reference_path_terminal_yaw = terminal_yaw
        self._live_reference_path_identity = identity
        self._live_reference_path_source = str(source)
        self._live_reference_path_generation_count += 1
        self._reset_acceptance_chain()

    def _bind_live_reference_path_goal(
        self,
        points: Sequence[Sequence[float]],
        *,
        terminal_yaw: float,
    ) -> bool:
        """证明 live Path 的地面末端与本代 base NavGoal 完全同源。"""

        if self._plan is None:
            self._fail_reference_path("scan_reference_path_goal_missing")
            return False
        goal = _nav_goal_base_xyzyaw(self._plan.goal)
        if goal is None:
            self._fail_reference_path("scan_reference_path_goal_z_missing")
            return False
        endpoint = points[-1]
        endpoint_x = _finite_number(endpoint[0])
        endpoint_y = _finite_number(endpoint[1])
        endpoint_ground_z = _finite_number(endpoint[2])
        if (
            endpoint_x is None
            or endpoint_y is None
            or endpoint_ground_z is None
        ):
            self._fail_reference_path("scan_reference_path_report_invalid")
            return False
        config = self._stair_freeze.config
        xy_error = math.hypot(endpoint_x - goal[0], endpoint_y - goal[1])
        z_error = abs(
            endpoint_ground_z + float(config.body_height_m) - goal[2]
        )
        yaw_error = abs(_normalize_angle(terminal_yaw - goal[3]))
        self._live_reference_path_goal_xy_error_m = xy_error
        self._live_reference_path_goal_z_error_m = z_error
        self._live_reference_path_goal_yaw_error_rad = yaw_error
        if xy_error > config.terminal_goal_xy_tolerance_m:
            self._fail_reference_path("scan_reference_path_goal_xy_mismatch")
            return False
        if z_error > config.terminal_goal_z_tolerance_m:
            self._fail_reference_path("scan_reference_path_goal_z_mismatch")
            return False
        if yaw_error > config.terminal_goal_yaw_tolerance_rad:
            self._fail_reference_path("scan_reference_path_goal_yaw_mismatch")
            return False
        self._live_reference_path_goal_bound = True
        return True

    def _check_live_reference_path_timeout(
        self,
        state: SimulationState,
    ) -> None:
        """要求 ROS bridge 在有限仿真时间内提供实际 Path 代际。"""

        if (
            not self.config.require_live_reference_path
            or self._live_reference_path_verified
        ):
            return
        timestamp = float(state.timestamp)
        if not math.isfinite(timestamp):
            self._fail_reference_path("scan_reference_path_timestamp_invalid")
            return
        if self._live_reference_path_wait_started_timestamp is None:
            self._live_reference_path_wait_started_timestamp = timestamp
            self._phase = "waiting_for_live_reference_path"
            return
        if (
            timestamp - self._live_reference_path_wait_started_timestamp
            > self.config.live_reference_path_timeout_s
        ):
            self._fail_reference_path("scan_reference_path_timeout")

    def _fail_reference_path(
        self,
        reason: str,
        *,
        detail: str | None = None,
    ) -> None:
        """锁存 Path 代际失败；后续动作只允许零速与全局重规划。"""

        self._failed = True
        self._success = False
        self._failure_reason = reason if detail is None else f"{reason}:{detail}"
        self._phase = "failed"

    def _consume_goal_sample(self, metadata: dict[str, Any]) -> None:
        sample = next(
            (
                metadata.get(key)
                for key in _GOAL_SAMPLE_KEYS
                if metadata.get(key) is not None
            ),
            None,
        )
        parsed = self._parse_goal_sample(sample)
        if parsed is None:
            if sample is not None:
                self._invalid_goal_sample_count += 1
            return
        value, receipt_timestamp, sequence, identity = parsed
        if identity == self._last_goal_identity:
            return
        if (
            sequence is not None
            and self._last_goal_sequence is not None
            and sequence < self._last_goal_sequence
        ):
            self._goal_sequence_reset_count += 1
            self._reset_acceptance_chain()
        self._last_goal_identity = identity
        if sequence is not None:
            self._last_goal_sequence = sequence

        if not value:
            if self._goal_true_seen:
                # true 后重新变 false 表示轨迹被撤销或开始了新一轮规划；
                # 必须重新证明本轮确实执行过非零 policy 活动。
                self._policy_activity_seen = False
                self._root_lock_progress_seen = False
                self._activity_write_sequence = None
                self._activity_write_timestamp = None
                self._reset_progress_watchdog()
            self._fresh_false_seen = True
            self._goal_true_seen = False
            self._goal_false_sequence = sequence
            self._goal_false_receipt_timestamp = receipt_timestamp
            self._goal_true_sequence = None
            self._goal_true_receipt_timestamp = None
            self._zero_write_streak = 0
            self._supervisor_goal_reached_zero_count = 0
            self._last_supervisor_goal_reached_zero_verified = False
            self._phase = (
                "tracking_waiting_for_goal"
                if self._policy_activity_seen
                else "waiting_for_policy_activity"
            )
            return

        if not (
            self._fresh_false_seen
            and (self._policy_activity_seen or self._root_lock_progress_seen)
        ):
            self._premature_true_count += 1
            return
        stair_status = self._stair_freeze.status()
        if (
            stair_status.get("applicable") is True
            and not self._stair_freeze.finish_ready
        ):
            # controller Bool 是持续状态，不带 Path/B-spline identity。冻结
            # 或解锁交接尚未完成时只能丢弃该 true；不能暂存后在下一阶段复用。
            # terminal hold 的 typed 轨迹已确认、仅等待 supervisor ACK 时，
            # 跨 topic 早到一拍属于正常等待，不计作协议错误；两种情况都必须
            # 在就绪后收到一个更晚的新序号 true。
            if self._stair_freeze.terminal_supervisor_transition_pending:
                self._goal_true_waiting_for_supervisor_ack_count += 1
                self._phase = "goal_reached_waiting_for_supervisor_ack"
                return
            self._premature_true_count += 1
            return
        if not self._goal_true_seen:
            self._goal_true_seen = True
            self._goal_true_sequence = sequence
            self._goal_true_receipt_timestamp = receipt_timestamp
            self._zero_write_streak = 0
            self._reset_progress_watchdog()
            self._phase = "goal_reached_waiting_for_zero_hold"

    def _consume_policy_write(
        self,
        state: SimulationState,
        *,
        count_post_goal_zero: bool,
    ) -> None:
        raw_report = state.metadata.get(_POLICY_WRITE_REPORT_KEY)
        parsed = self._parse_policy_write(raw_report, state)
        if parsed is None:
            invalid_identity = self._policy_write_observation_identity(
                raw_report,
                state,
            )
            if (
                raw_report is not None
                and invalid_identity
                != self._last_invalid_policy_write_identity
            ):
                self._invalid_policy_write_count += 1
                self._last_invalid_policy_write_identity = invalid_identity
            return
        (
            requested,
            written,
            stop_reasons,
            motion_allowed,
            owner_id,
            timestamp,
            sequence,
            navigation_cmd_vel_inhibited,
            navigation_cmd_vel_inhibit_reason,
            identity,
        ) = parsed
        if identity == self._last_policy_write_identity:
            return

        self._last_invalid_policy_write_identity = None

        sequence_regressed = (
            sequence is not None
            and self._last_policy_write_sequence is not None
            and sequence < self._last_policy_write_sequence
        )
        write_contiguous = self._policy_write_is_contiguous(
            state=state,
            sequence=sequence,
            timestamp=timestamp,
        )
        if sequence_regressed:
            self._policy_write_sequence_reset_count += 1
            self._reset_acceptance_chain()

        self._last_policy_write_identity = identity
        self._last_policy_write_sequence = sequence
        self._last_policy_write_step = int(state.step_index)
        self._last_policy_write_timestamp = timestamp
        self._last_requested_command = requested
        self._last_written_command = written
        self._last_stop_reasons = stop_reasons
        self._last_navigation_cmd_vel_inhibited = (
            navigation_cmd_vel_inhibited
        )
        self._last_navigation_cmd_vel_inhibit_reason = (
            navigation_cmd_vel_inhibit_reason
        )

        valid_write = (
            owner_id == self.config.policy_owner_id
            and motion_allowed is True
            and not stop_reasons
        )
        requested_zero = _command_is_zero(
            requested,
            epsilon=self.config.zero_epsilon,
        )
        written_zero = _command_is_zero(
            written,
            epsilon=self.config.zero_epsilon,
        )
        requested_nonzero = not requested_zero
        written_nonzero = not written_zero

        if not self._goal_true_seen:
            self._update_progress_watchdog(
                state=state,
                timestamp=timestamp,
                requested=requested,
                written=written,
                valid_write=valid_write,
                write_contiguous=write_contiguous,
            )
            if self._failed:
                return
            if (
                self._fresh_false_seen
                and valid_write
                and requested_nonzero
                and written_nonzero
            ):
                self._policy_activity_seen = True
                self._activity_write_sequence = sequence
                self._activity_write_timestamp = timestamp
                self._phase = "tracking_waiting_for_goal"
            return

        # Bool 与 Twist 属于不同 ROS topic，DDS 不保证跨 topic 顺序。同一个
        # observation 中伴随 true 的零速可能先于 true 发布，不能算作“到达后”
        # 的保持证据；只从下一条独立 policy 实写开始计数。
        if not count_post_goal_zero:
            return

        # true 之后验收的是 policy buffer 是否被唯一 owner 连续写成精确零速。
        # 点云超时会令 motion_allowed=false，但 controller 仍可在冻结
        # 执行时间后由新鲜 Odometry 物理确认到达，因此只对该原因
        # 开放例外。环境终止、预测碰撞、时钟回退或控制租约失效等
        # 安全停车不能成为成功保持证据。
        stair_status = self._stair_freeze.status()
        supervisor_goal_reached_zero = (
            self._supervisor_goal_reached_zero_verified(
                raw_report,
                state,
                requested=requested,
                written=written,
                stop_reasons=stop_reasons,
                motion_allowed=motion_allowed,
                owner_id=owner_id,
                write_timestamp=timestamp,
                navigation_cmd_vel_inhibited=(
                    navigation_cmd_vel_inhibited
                ),
                navigation_cmd_vel_inhibit_reason=(
                    navigation_cmd_vel_inhibit_reason
                ),
            )
        )
        self._last_supervisor_goal_reached_zero_verified = (
            supervisor_goal_reached_zero
        )
        terminal_hold_zero = (
            stair_status.get("phase") == "terminal_hold"
            and stair_status.get("terminal_component") is True
            and stair_status.get("terminal_goal_bound") is True
            and motion_allowed is False
            and stop_reasons == ("scan_stair_terminal_hold",)
            and navigation_cmd_vel_inhibited is True
            and navigation_cmd_vel_inhibit_reason
            == "scan_stair_terminal_hold"
        )
        stop_context_allowed = (
            (motion_allowed is True and not stop_reasons)
            or supervisor_goal_reached_zero
            or terminal_hold_zero
            or (
                motion_allowed is False
                and bool(stop_reasons)
                and set(stop_reasons).issubset(
                    _ALLOWED_POST_GOAL_STOP_REASONS
                )
            )
        )
        timestamp_after_goal = (
            self._goal_true_receipt_timestamp is not None
            and timestamp > self._goal_true_receipt_timestamp
        )
        valid_goal_zero = (
            owner_id == self.config.policy_owner_id
            and requested_zero
            and written_zero
            and stop_context_allowed
            and timestamp_after_goal
        )
        if not valid_goal_zero:
            if written_nonzero:
                self._post_goal_nonzero_write_count += 1
            self._zero_write_streak = 0
            return
        if supervisor_goal_reached_zero:
            self._supervisor_goal_reached_zero_count += 1
        if self._zero_write_streak == 0 or not write_contiguous:
            self._zero_write_streak = 1
        else:
            self._zero_write_streak += 1

    def _supervisor_goal_reached_zero_verified(
        self,
        raw_report: Any,
        state: SimulationState,
        *,
        requested: tuple[float, float, float],
        written: tuple[float, float, float],
        stop_reasons: tuple[str, ...],
        motion_allowed: bool,
        owner_id: str,
        write_timestamp: float,
        navigation_cmd_vel_inhibited: bool,
        navigation_cmd_vel_inhibit_reason: str | None,
    ) -> bool:
        """验证 supervisor 完成锁存产生的同代 policy 零速。"""

        if (
            not isinstance(raw_report, Mapping)
            or owner_id != self.config.policy_owner_id
            or motion_allowed is not False
            or navigation_cmd_vel_inhibited is not False
            or navigation_cmd_vel_inhibit_reason is not None
            or len(stop_reasons)
            != len(_SUPERVISOR_GOAL_REACHED_STOP_REASONS)
            or set(stop_reasons)
            != _SUPERVISOR_GOAL_REACHED_STOP_REASONS
            or not _command_is_zero(requested, epsilon=0.0)
            or not _command_is_zero(written, epsilon=0.0)
            or raw_report.get("navigation_emergency_stop_latched") is not False
            or raw_report.get("navigation_emergency_stop_reason") is not None
            or self._goal_true_receipt_timestamp is None
        ):
            return False

        diagnostics = raw_report.get("navigation_status_observed_report")
        gate = raw_report.get("policy_navigation_gate_consumed_report")
        controller = state.metadata.get(_CONTROLLER_STATUS_REPORT_KEY)
        if (
            not isinstance(diagnostics, Mapping)
            or diagnostics.get("schema")
            != "navigation_status_observed_diagnostics_v1"
            or diagnostics.get("topic") != "/navigation/status"
            or diagnostics.get("status_error") is not None
            or diagnostics.get("local_reference_path_identity_fault")
            is not None
            or not isinstance(gate, Mapping)
            or gate.get("schema")
            != "navigation_policy_gate_diagnostics_v1"
            or gate.get("required") is not True
            or gate.get("status_fault") is not None
            or gate.get("permit_received") is not True
            or not isinstance(controller, Mapping)
        ):
            return False

        status = diagnostics.get("status")
        permit = gate.get("permit")
        if not isinstance(status, Mapping) or not isinstance(permit, Mapping):
            return False

        receipt_timestamp = _finite_number(status.get("receipt_timestamp"))
        gate_timeout_s = _finite_number(gate.get("timeout_s"))
        goal_id = _positive_int(status.get("goal_id"))
        active_path_stamp_ns = _positive_int(
            status.get("active_path_stamp_ns")
        )
        local_goal_id = _positive_int(
            diagnostics.get("local_pct_goal_stamp_ns")
        )
        local_path_stamp_ns = _positive_int(
            diagnostics.get("local_active_path_stamp_ns")
        )
        header_stamp_ns = _positive_int(status.get("header_stamp_ns"))
        status_sequence = _positive_int(status.get("status_sequence"))
        state_revision = _positive_int(status.get("state_revision"))
        if (
            receipt_timestamp is None
            or gate_timeout_s is None
            or gate_timeout_s <= 0.0
            or receipt_timestamp > write_timestamp
            or write_timestamp - receipt_timestamp > gate_timeout_s
            or goal_id is None
            or active_path_stamp_ns is None
            or local_goal_id != goal_id
            or local_path_stamp_ns != active_path_stamp_ns
            or header_stamp_ns is None
            or status_sequence is None
            or state_revision is None
            or _positive_int(status.get("rx_sequence")) is None
            or _positive_int(status.get("pct_plan_id")) is None
            or status.get("state") != _NAVIGATION_STATE_GOAL_REACHED
            or status.get("reason") != "goal_reached"
            or status.get("allow_tracking_command") is not False
            or status.get("force_zero_velocity") is not True
            or status.get("stop_confirmed") is not True
            or status.get("identity_valid") is not True
            or status.get("global_replan_requested") is not False
            or status.get("global_replan_in_flight") is not False
            or _optional_nonnegative_int(
                status.get("global_replan_request_id")
            )
            is None
            or status.get("consecutive_scan_failures") != 0
        ):
            return False
        stale_inputs = status.get("stale_inputs")
        if not isinstance(stale_inputs, list) or not all(
            isinstance(value, str) for value in stale_inputs
        ):
            return False
        if (
            self._pct_goal_stamp_ns is not None
            and goal_id != self._pct_goal_stamp_ns
        ):
            return False
        if (
            self._live_reference_path_stamp_ns is not None
            and active_path_stamp_ns != self._live_reference_path_stamp_ns
        ):
            return False

        if (
            _finite_number(permit.get("received_at"))
            != receipt_timestamp
            or permit.get("header_stamp_ns") != header_stamp_ns
            or permit.get("status_sequence") != status_sequence
            or permit.get("state_revision") != state_revision
            or permit.get("goal_id") != goal_id
            or permit.get("active_path_stamp_ns") != active_path_stamp_ns
            or permit.get("state") != _NAVIGATION_STATE_GOAL_REACHED
            or permit.get("allow_tracking_command") is not False
            or permit.get("force_zero_velocity") is not True
            or permit.get("identity_valid") is not True
            or permit.get("reason") != "goal_reached"
        ):
            return False

        controller_receipt_timestamp = _finite_number(
            controller.get("receipt_timestamp")
        )
        controller_header = controller.get("header")
        controller_identity = controller.get("identity")
        if (
            controller.get("source")
            != "ros2_scan_planner_msgs_controller_status"
            or controller.get("topic") != self.config.controller_status_topic
            or controller_receipt_timestamp is None
            or controller_receipt_timestamp
            < self._goal_true_receipt_timestamp
            or controller_receipt_timestamp > write_timestamp
            or not isinstance(controller_header, Mapping)
            or controller_header.get("frame_id")
            != self.config.controller_status_frame_id
            or _positive_int(controller_header.get("stamp_ns")) is None
            or _positive_int(controller.get("status_sequence")) is None
            or _positive_int(controller.get("acceptance_sequence")) is None
            or controller.get("state") != _CONTROLLER_STATE_GOAL_REACHED
            or controller.get("accepted") is not True
            or controller.get("trajectory_valid") is not True
            or controller.get("is_final") is not True
            or controller.get("emergency_stop") is not False
            or not isinstance(controller_identity, Mapping)
            or controller_identity.get("reference_path_stamp_ns")
            != active_path_stamp_ns
            or _positive_int(
                controller_identity.get("bspline_header_stamp_ns")
            )
            is None
            or _positive_int(controller_identity.get("start_time_ns"))
            is None
            or _positive_int(controller_identity.get("traj_id")) is None
        ):
            return False
        return True

    def _policy_write_is_contiguous(
        self,
        *,
        state: SimulationState,
        sequence: int | None,
        timestamp: float,
    ) -> bool:
        step_contiguous = (
            self._last_policy_write_step is not None
            and int(state.step_index) == self._last_policy_write_step + 1
        )
        timestamp_increasing = (
            self._last_policy_write_timestamp is not None
            and timestamp > self._last_policy_write_timestamp
        )
        if sequence is not None:
            sequence_contiguous = (
                self._last_policy_write_sequence is not None
                and sequence == self._last_policy_write_sequence + 1
            )
        else:
            sequence_contiguous = self._last_policy_write_sequence is None
        return step_contiguous and timestamp_increasing and sequence_contiguous

    def _reset_acceptance_chain(self) -> None:
        """输入代次回退时丢弃全部完成证据，等待新的 false 起点。"""

        self._fresh_false_seen = False
        self._policy_activity_seen = False
        self._root_lock_progress_seen = False
        self._goal_true_seen = False
        self._goal_false_sequence = None
        self._goal_true_sequence = None
        self._goal_false_receipt_timestamp = None
        self._goal_true_receipt_timestamp = None
        self._activity_write_sequence = None
        self._activity_write_timestamp = None
        self._zero_write_streak = 0
        self._supervisor_goal_reached_zero_count = 0
        self._last_supervisor_goal_reached_zero_verified = False
        self._phase = "waiting_for_fresh_false"
        self._reset_progress_watchdog()

    def _update_progress_watchdog(
        self,
        *,
        state: SimulationState,
        timestamp: float,
        requested: tuple[float, float, float],
        written: tuple[float, float, float],
        valid_write: bool,
        write_contiguous: bool,
    ) -> None:
        """检测 policy 持续前推但机器人净位移不足的低层卡死。"""

        active_forward_write = (
            self._fresh_false_seen
            and valid_write
            and requested[0]
            >= self.config.progress_watchdog_min_forward_command_mps
            and written[0]
            >= self.config.progress_watchdog_min_forward_command_mps
        )
        if not active_forward_write:
            self._reset_progress_watchdog(reason="ineligible_policy_write")
            return

        pose_xy = self._planar_pose_xy(state)
        if pose_xy is None:
            self._invalid_progress_pose_count += 1
            self._reset_progress_watchdog(reason="invalid_robot_pose")
            return

        # 写入序号、控制步或写入时间不连续时不能跨缺口累计卡死时间。
        if self._progress_watchdog_active and not write_contiguous:
            self._reset_progress_watchdog(reason="policy_write_not_contiguous")

        self._progress_watchdog_last_goal_distance_m = (
            self._planar_goal_distance(pose_xy)
        )
        if not self._progress_watchdog_active:
            self._progress_watchdog_active = True
            self._progress_watchdog_window_start_pose_xy = pose_xy
            self._progress_watchdog_window_start_timestamp = timestamp
            self._progress_watchdog_window_sample_count = 1
            self._progress_watchdog_last_displacement_m = 0.0
            self._progress_watchdog_elapsed_without_progress_s = 0.0
            self._progress_watchdog_last_pause_reason = ""
            return

        start_pose = self._progress_watchdog_window_start_pose_xy
        start_timestamp = self._progress_watchdog_window_start_timestamp
        if start_pose is None or start_timestamp is None:
            self._reset_progress_watchdog(reason="invalid_watchdog_state")
            return

        self._progress_watchdog_window_sample_count += 1
        displacement = math.hypot(
            pose_xy[0] - start_pose[0],
            pose_xy[1] - start_pose[1],
        )
        self._progress_watchdog_last_displacement_m = displacement
        elapsed = max(0.0, timestamp - start_timestamp)
        self._progress_watchdog_elapsed_without_progress_s = elapsed
        if elapsed < self.config.progress_watchdog_timeout_s:
            return

        # 固定窗口只比较首尾净位移。即使中途碰撞抖动曾越过阈值，最终回到
        # 锚点也不能续命；正常局部绕障只要确有机体位移就不会被误报。
        if displacement >= self.config.progress_watchdog_min_displacement_m:
            self._progress_watchdog_window_start_pose_xy = pose_xy
            self._progress_watchdog_window_start_timestamp = timestamp
            self._progress_watchdog_window_sample_count = 1
            self._progress_watchdog_last_displacement_m = 0.0
            self._progress_watchdog_elapsed_without_progress_s = 0.0
            self._progress_watchdog_progress_event_count += 1
            return

        self._failed = True
        self._failure_reason = "locomotion_stall"
        self._phase = "failed"
        self._progress_watchdog_trigger_count += 1
        self._progress_watchdog_failure_timestamp = timestamp
        self._progress_watchdog_failure_pose_xy = pose_xy

    @staticmethod
    def _planar_pose_xy(
        state: SimulationState,
    ) -> tuple[float, float] | None:
        """返回有限的当前 base 世界坐标。"""

        if len(state.robot_root_pose) < 2:
            return None
        pose_x = _finite_number(state.robot_root_pose[0])
        pose_y = _finite_number(state.robot_root_pose[1])
        if pose_x is None or pose_y is None:
            return None
        return pose_x, pose_y

    def _planar_goal_distance(
        self,
        pose_xy: tuple[float, float],
    ) -> float | None:
        """仅为诊断返回当前 base 到 pipeline 目标的水平距离。"""

        if self._plan is None:
            return None
        return math.hypot(
            pose_xy[0] - float(self._plan.goal.x),
            pose_xy[1] - float(self._plan.goal.y),
        )

    def _reset_progress_watchdog(self, *, reason: str = "lifecycle_reset") -> None:
        """清空当前连续前进窗口，不丢弃累计诊断计数。"""

        if self._progress_watchdog_active:
            self._progress_watchdog_reset_count += 1
        self._progress_watchdog_active = False
        self._progress_watchdog_window_start_pose_xy = None
        self._progress_watchdog_window_start_timestamp = None
        self._progress_watchdog_window_sample_count = 0
        self._progress_watchdog_last_displacement_m = None
        self._progress_watchdog_last_goal_distance_m = None
        self._progress_watchdog_elapsed_without_progress_s = 0.0
        self._progress_watchdog_last_pause_reason = reason

    @staticmethod
    def _parse_goal_sample(
        raw_sample: Any,
    ) -> tuple[bool, float, int | None, Hashable] | None:
        if not isinstance(raw_sample, dict):
            return None
        raw_value = raw_sample.get(
            "value",
            raw_sample.get("goal_reached"),
        )
        if not isinstance(raw_value, bool):
            return None
        receipt_timestamp = _finite_number(
            raw_sample.get("receipt_timestamp"),
        )
        if receipt_timestamp is None or receipt_timestamp < 0.0:
            return None
        sequence = _optional_nonnegative_int(raw_sample.get("sequence"))
        if raw_sample.get("sequence") is not None and sequence is None:
            return None
        identity: Hashable = (
            ("sequence", sequence)
            if sequence is not None
            else ("receipt_timestamp", receipt_timestamp)
        )
        return raw_value, receipt_timestamp, sequence, identity

    @staticmethod
    def _policy_write_observation_identity(
        raw_report: Any,
        state: SimulationState,
    ) -> Hashable:
        """为不可解析实写建立去重键，避免同一 observation 重复计错。"""

        if isinstance(raw_report, Mapping):
            sequence = _optional_nonnegative_int(
                raw_report.get("write_sequence")
            )
            if sequence is not None:
                return ("invalid_write_sequence", sequence)
            timestamp = _finite_number(raw_report.get("timestamp"))
            if timestamp is not None and timestamp >= 0.0:
                return ("invalid_write_timestamp", timestamp)
        return (
            "invalid_observation",
            int(state.step_index),
            float(state.timestamp),
        )

    @staticmethod
    def _parse_policy_write(
        raw_report: Any,
        state: SimulationState,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[str, ...],
        bool,
        str,
        float,
        int | None,
        bool,
        str | None,
        Hashable,
    ] | None:
        if not isinstance(raw_report, dict):
            return None
        raw_stop_reasons = raw_report.get("stop_reasons")
        if not isinstance(raw_stop_reasons, (list, tuple)) or not all(
            isinstance(reason, str) for reason in raw_stop_reasons
        ):
            return None
        written = _finite_command(raw_report.get("written_command"))
        requested = _finite_command(raw_report.get("requested_command"))
        motion_allowed = raw_report.get("motion_allowed")
        owner_id = raw_report.get("owner_id")
        if not isinstance(motion_allowed, bool) or not isinstance(owner_id, str):
            return None
        navigation_cmd_vel_inhibited = raw_report.get(
            "navigation_cmd_vel_inhibited"
        )
        if not isinstance(navigation_cmd_vel_inhibited, bool):
            return None
        raw_inhibit_reason = raw_report.get(
            "navigation_cmd_vel_inhibit_reason"
        )
        if raw_inhibit_reason is None:
            navigation_cmd_vel_inhibit_reason = None
        elif isinstance(raw_inhibit_reason, str) and raw_inhibit_reason.strip():
            navigation_cmd_vel_inhibit_reason = raw_inhibit_reason.strip()
        else:
            return None
        stop_reason_set = set(raw_stop_reasons)
        temporary_navigation_stop = bool(
            len(raw_stop_reasons) == len(stop_reason_set)
            and stop_reason_set
            and navigation_cmd_vel_inhibited is True
            and navigation_cmd_vel_inhibit_reason
            in _TEMPORARY_NAVIGATION_STOP_REASONS
            and navigation_cmd_vel_inhibit_reason in stop_reason_set
            and stop_reason_set.issubset(
                {navigation_cmd_vel_inhibit_reason}
                | _STAIR_SENSOR_STOP_REASONS
            )
            and motion_allowed is False
            and written is not None
            and _command_is_zero(written, epsilon=0.0)
        )
        missing_command_bootstrap_stop = bool(
            len(raw_stop_reasons) == len(stop_reason_set)
            and "missing_cmd_vel" in stop_reason_set
            and stop_reason_set.issubset(
                _MISSING_COMMAND_BOOTSTRAP_STOP_REASONS
            )
            and motion_allowed is False
            and navigation_cmd_vel_inhibited is False
            and navigation_cmd_vel_inhibit_reason is None
            and written is not None
            and _command_is_zero(written, epsilon=0.0)
        )
        if requested is None and (
            temporary_navigation_stop or missing_command_bootstrap_stop
        ):
            # host 在 Path 等待、预检或楼梯冻结阶段没有上游 Twist；这类
            # 带精确抑制身份，或 policy 启动期明确缺少命令且实际写零的
            # fail-closed 报告，其有效请求语义就是零速度。
            requested = (0.0, 0.0, 0.0)
        if requested is None or written is None:
            return None
        timestamp = _finite_number(raw_report.get("timestamp"))
        if timestamp is None or timestamp < 0.0:
            return None
        sequence = _optional_nonnegative_int(raw_report.get("write_sequence"))
        if raw_report.get("write_sequence") is not None and sequence is None:
            return None
        identity: Hashable = (
            ("write_sequence", sequence)
            if sequence is not None
            else ("timestamp", timestamp)
        )
        return (
            requested,
            written,
            tuple(raw_stop_reasons),
            motion_allowed,
            owner_id,
            timestamp,
            sequence,
            navigation_cmd_vel_inhibited,
            navigation_cmd_vel_inhibit_reason,
            identity,
        )


def _normalize_pct_goal_request(raw_request: Any) -> dict[str, object] | None:
    """校验 pipeline 交给 runtime 的 base 高度 PCT 目标。"""

    if raw_request is None:
        return None
    if not isinstance(raw_request, Mapping):
        raise ValueError("pct_goal_request 必须是对象。")
    position = raw_request.get("position_base_xyz")
    yaw = raw_request.get("yaw")
    if (
        raw_request.get("frame_id") != "world"
        or raw_request.get("height_semantics") != "base"
        or not isinstance(position, (list, tuple))
        or len(position) != 3
    ):
        raise ValueError("pct_goal_request 必须使用 world frame 和 base 高度 xyz。")
    try:
        normalized_position = tuple(float(value) for value in position)
        normalized_yaw = float(yaw)
    except (TypeError, ValueError) as exc:
        raise ValueError("pct_goal_request 必须包含有限数值。") from exc
    if not all(
        math.isfinite(value)
        for value in (*normalized_position, normalized_yaw)
    ):
        raise ValueError("pct_goal_request 必须包含有限数值。")
    provenance_required = raw_request.get(
        "effective_goal_provenance_required",
        False,
    )
    if not isinstance(provenance_required, bool):
        raise ValueError("effective_goal_provenance_required 必须是布尔值。")
    raw_provenance = raw_request.get("effective_goal_provenance")
    if provenance_required and not isinstance(raw_provenance, Mapping):
        raise ValueError("生产 PCT goal 缺少有效高度 provenance。")
    normalized: dict[str, object] = {
        "frame_id": "world",
        "position_base_xyz": normalized_position,
        "yaw": normalized_yaw,
        "height_semantics": "base",
    }
    if provenance_required:
        normalized["effective_goal_provenance_required"] = True
        normalized["effective_goal_provenance"] = dict(raw_provenance or {})
    return normalized


def _nav_goal_base_xyzyaw(
    goal: NavGoal,
) -> tuple[float, float, float, float] | None:
    """把有明确 base 高度的 NavGoal 转为冻结目标合同。"""

    if goal.z is None:
        return None
    values = (float(goal.x), float(goal.y), float(goal.z), float(goal.yaw))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("NavGoal base xyzyaw 必须是有限数值。")
    return values


def _terminal_ground_path_yaw(
    points: Sequence[Sequence[float]],
) -> float:
    """返回手工 Path publisher 会写入末 Pose 的最后有效切向。"""

    if len(points) < 2:
        raise ValueError("参考 Path 至少需要两个点。")
    terminal_x = float(points[-1][0])
    terminal_y = float(points[-1][1])
    for previous in reversed(points[:-1]):
        dx = terminal_x - float(previous[0])
        dy = terminal_y - float(previous[1])
        if math.hypot(dx, dy) > 1.0e-9:
            return math.atan2(dy, dx)
    raise ValueError("参考 Path 末端没有可用的平面方向。")


def _normalize_angle(value: float) -> float:
    """把角度归一化到 ``[-pi, pi)``。"""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _stamp_mapping_to_ns(stamp: Mapping[str, Any], *, field_name: str) -> int:
    """把 ROS stamp 对象严格转换为正整数纳秒。"""

    sec = stamp.get("sec")
    nanosec = stamp.get("nanosec")
    if (
        isinstance(sec, bool)
        or not isinstance(sec, int)
        or sec < 0
        or isinstance(nanosec, bool)
        or not isinstance(nanosec, int)
        or not 0 <= nanosec < 1_000_000_000
    ):
        raise ValueError(f"{field_name} 范围非法。")
    stamp_ns = sec * 1_000_000_000 + nanosec
    if stamp_ns <= 0:
        raise ValueError(f"{field_name} 必须非零。")
    return stamp_ns


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_int(value: Any) -> int | None:
    """严格解析正整数，拒绝布尔值和零。"""

    parsed = _optional_nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _finite_command(value: Any) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return None
    parsed = tuple(_finite_number(item) for item in value)
    if any(item is None for item in parsed):
        return None
    return parsed  # type: ignore[return-value]


def _command_is_zero(
    command: tuple[float, float, float],
    *,
    epsilon: float,
) -> bool:
    return all(abs(value) <= epsilon for value in command)
