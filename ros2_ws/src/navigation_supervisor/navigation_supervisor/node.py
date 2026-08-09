"""PCT、SCAN 与闭环控制器之间的类型化 ROS 2 安全协调节点。

节点不发布 ``/cmd_vel``。它只汇总三类 planner/controller 状态、发布
``/navigation/status``，并在控制器已经证明停车后幂等调用 PCT REPLAN
服务。速度单写入者仍是现有 SCAN controller → policy adapter 链。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Hashable

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scan_planner_msgs.msg import (
    Bspline,
    ControllerStatus,
    NavigationStatus,
    PCTPlanningStatus,
    ScanPlanningStatus,
)
from scan_planner_msgs.srv import PCTPlanningCommand
from sensor_msgs.msg import PointCloud2, PointField

from navigation_core import (
    NavigationState,
    NavigationSupervisor,
    NavigationSupervisorConfig,
    SupervisorDecision,
)
from navigation_supervisor.contracts import (
    GoalIdentity,
    MonotonicSequence,
    PathIdentity,
    ReplanTransaction,
    SequenceDisposition,
    TrajectoryIdentity,
    bspline_valid_until_ns,
    finite_tuple,
    nanoseconds_to_seconds,
    stamp_to_nanoseconds,
)


NANOSECONDS_PER_SECOND = 1_000_000_000
SCAN_STAIR_EXECUTION_INHIBITED_REASON = "scan_stair_execution_inhibited"
SCAN_STAIR_RESUME_WAITING_REASON = "scan_stair_resume_waiting"
KNOWN_SCAN_STAIR_FREEZE_FAULTS = frozenset(
    {
        "scan_stair_freeze_frame_mismatch_fault",
        "scan_stair_freeze_protocol_fault",
        "scan_stair_freeze_snapshot_timeout_fault",
        "scan_stair_stop_publish_fault",
    }
)


@dataclass(frozen=True)
class _TrajectoryRecord:
    """已在真实 B-spline topic 上观察到的完整轨迹。"""

    identity: TrajectoryIdentity
    valid_until_ns: int
    signature: Hashable


@dataclass(frozen=True)
class _PendingBsplineRecord:
    """等待同代 PCT Path 对账的 B-spline 完整快照。"""

    message: Bspline
    received_at_ns: int
    identity: TrajectoryIdentity
    signature: Hashable


@dataclass(frozen=True)
class _PendingScanStatusRecord:
    """等待同代 PCT Path 对账的 SCAN typed status 快照。"""

    message: ScanPlanningStatus
    received_at_ns: int
    reference_path_stamp_ns: int
    status_sequence: int
    signature: Hashable


@dataclass(frozen=True)
class _ControllerSnapshot:
    """通过单调序列检查的 controller 状态快照。"""

    identity: TrajectoryIdentity | None
    event: int
    state: int
    accepted: bool
    trajectory_valid: bool
    status_sequence: int
    acceptance_sequence: int


def _time_from_nanoseconds(value_ns: int) -> Time:
    """把非负纳秒转成规范 ROS Time。"""

    value = max(int(value_ns), 0)
    return Time(
        sec=value // NANOSECONDS_PER_SECOND,
        nanosec=value % NANOSECONDS_PER_SECOND,
    )


def _normalize_frame(value: object) -> str:
    """统一移除 frame 前导斜线，拒绝空 frame。"""

    frame = str(value).strip().lstrip("/")
    if not frame:
        raise ValueError("frame_id 不能为空")
    return frame


def _is_canonical_empty_point_cloud(message: PointCloud2) -> bool:
    """识别 bridge/controller 共同约定的 canonical xyz32 空点云。"""

    expected_fields = (
        ("x", 0, PointField.FLOAT32, 1),
        ("y", 4, PointField.FLOAT32, 1),
        ("z", 8, PointField.FLOAT32, 1),
    )
    actual_fields = tuple(
        (
            str(field.name),
            int(field.offset),
            int(field.datatype),
            int(field.count),
        )
        for field in message.fields
    )
    return (
        int(message.height) == 1
        and int(message.width) == 0
        and not bool(message.is_bigendian)
        and int(message.point_step) == 12
        and int(message.row_step) == 0
        and len(message.data) == 0
        and bool(message.is_dense)
        and actual_fields == expected_fields
    )


def _pose_values(message: PoseStamped) -> tuple[float, ...]:
    """验证并返回完整位置和单位化前四元数 payload。"""

    pose = message.pose
    values = finite_tuple(
        (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
        field_name="PoseStamped.pose",
    )
    norm = math.sqrt(sum(value * value for value in values[3:]))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError("PoseStamped 四元数不能为零")
    return values


def _goal_identity(message: PoseStamped, goal_id: int) -> GoalIdentity:
    """从 PCT 保留的完整目标快照建立不可变身份。"""

    return GoalIdentity(
        goal_id=int(goal_id),
        stamp_ns=stamp_to_nanoseconds(message.header.stamp),
        frame_id=_normalize_frame(message.header.frame_id),
        pose=_pose_values(message),
    )


def _trajectory_identity_from_bspline(message: Bspline) -> TrajectoryIdentity:
    """读取 B-spline 的跨 topic 对账身份。"""

    return TrajectoryIdentity(
        reference_path_stamp_ns=stamp_to_nanoseconds(
            message.reference_path_stamp
        ),
        bspline_header_stamp_ns=stamp_to_nanoseconds(message.header.stamp),
        start_time_ns=stamp_to_nanoseconds(message.start_time),
        trajectory_id=int(message.traj_id),
        is_final=bool(message.is_final),
        emergency_stop=bool(message.emergency_stop),
    )


def _trajectory_identity_from_scan(
    message: ScanPlanningStatus,
) -> TrajectoryIdentity:
    """读取 SCAN typed status 携带的轨迹身份。"""

    if not message.trajectory_present:
        raise ValueError("SCAN 状态没有携带轨迹身份")
    return TrajectoryIdentity(
        reference_path_stamp_ns=stamp_to_nanoseconds(
            message.reference_path_stamp
        ),
        bspline_header_stamp_ns=stamp_to_nanoseconds(
            message.bspline_header_stamp
        ),
        start_time_ns=stamp_to_nanoseconds(message.trajectory_start_time),
        trajectory_id=int(message.trajectory_id),
        is_final=bool(message.trajectory_is_final),
        emergency_stop=bool(message.trajectory_emergency_stop),
    )


def _trajectory_identity_from_controller(
    message: ControllerStatus,
) -> TrajectoryIdentity | None:
    """读取 controller 当前或最近失效的已接受轨迹身份。"""

    if not message.accepted:
        return None
    return TrajectoryIdentity(
        reference_path_stamp_ns=stamp_to_nanoseconds(
            message.reference_path_stamp
        ),
        bspline_header_stamp_ns=stamp_to_nanoseconds(
            message.bspline_header_stamp
        ),
        start_time_ns=stamp_to_nanoseconds(message.start_time),
        trajectory_id=int(message.traj_id),
        is_final=bool(message.is_final),
        emergency_stop=bool(message.emergency_stop),
    )


class NavigationSupervisorNode(Node):
    """把依赖无关状态机接入 PCT、SCAN 与 controller ROS 2 图。"""

    def __init__(self) -> None:
        super().__init__("navigation_supervisor")
        self._declare_parameters()
        if not bool(self.get_parameter("use_sim_time").value):
            raise RuntimeError(
                "navigation_supervisor 必须使用 use_sim_time=true"
            )
        self._world_frame = _normalize_frame(
            self.get_parameter("frames.world").value
        )
        self._base_frame = _normalize_frame(
            self.get_parameter("frames.base").value
        )
        self._future_tolerance_ns = self._positive_duration_ns(
            "timeouts.future_tolerance_sec",
            allow_zero=True,
        )
        self._odometry_timeout_ns = self._positive_duration_ns(
            "timeouts.odometry_sec"
        )
        self._point_cloud_timeout_ns = self._positive_duration_ns(
            "timeouts.point_cloud_sec"
        )
        self._bspline_timeout_ns = self._positive_duration_ns(
            "timeouts.bspline_sec"
        )
        self._pending_path_evidence_timeout_ns = (
            self._positive_duration_ns(
                "timeouts.pending_path_evidence_sec"
            )
        )
        self._replan_retry_period_ns = self._positive_duration_ns(
            "timeouts.replan_retry_period_sec"
        )
        self._replan_response_ns = self._positive_duration_ns(
            "timeouts.replan_response_sec"
        )
        self._replan_service_wait_ns = self._positive_duration_ns(
            "timeouts.replan_service_wait_sec"
        )
        self._global_planning_timeout_ns = self._positive_duration_ns(
            "timeouts.global_planning_sec"
        )
        self._trajectory_expiry_grace_ns = (
            self._positive_duration_ns(
                "timeouts.trajectory_expiry_grace_sec"
            )
        )
        self._max_yaw_alignment_freeze_ns = (
            self._positive_duration_ns(
                "timeouts.max_yaw_alignment_freeze_sec"
            )
        )
        self._status_heartbeat_ns = self._positive_duration_ns(
            "status.heartbeat_sec"
        )
        self._replan_max_attempts = self._positive_int(
            "limits.replan_max_attempts"
        )
        self._max_global_replan_cycles = self._positive_int(
            "limits.max_global_replan_cycles"
        )
        self._max_path_points = self._positive_int("limits.max_path_points")
        self._max_pending_path_evidence = self._positive_int(
            "limits.max_pending_path_evidence"
        )

        self._core_config = NavigationSupervisorConfig(
            odometry_timeout_s=self._positive_float(
                "timeouts.odometry_sec"
            ),
            point_cloud_timeout_s=self._positive_float(
                "timeouts.point_cloud_sec"
            ),
            bspline_timeout_s=self._positive_float(
                "timeouts.bspline_sec"
            ),
            max_consecutive_scan_failures=self._positive_int(
                "limits.max_consecutive_scan_failures"
            ),
        )
        self._core = NavigationSupervisor(self._core_config)
        self._last_decision: SupervisorDecision | None = None
        self._last_clock_ns = 0
        self._last_odometry_stamp_ns = 0
        self._last_point_cloud_stamp_ns = 0
        self._epoch = 0
        self._adapter_fault_reason = ""
        self._protocol_poisoned = False
        self._fatal_epoch_reset = False

        self._goal: GoalIdentity | None = None
        self._goal_message: PoseStamped | None = None
        self._pct_plan_id = 0
        self._pct_active_path_stamp_ns = 0
        self._active_path: PathIdentity | None = None
        self._path_records: dict[int, PathIdentity] = {}
        self._path_signatures: dict[int, Hashable] = {}
        self._pct_successes: dict[
            int,
            tuple[int, int, int, int, int],
        ] = {}
        self._pct_failures: dict[int, tuple[int, int, int, int, str]] = {}
        self._pct_replan_ack_candidates: dict[
            int,
            tuple[int, int, int],
        ] = {}
        self._latest_tombstone_stamp_ns = 0
        self._last_pct_status_stamp_ns = 0
        self._pct_status_signatures: set[Hashable] = set()

        self._trajectories: dict[TrajectoryIdentity, _TrajectoryRecord] = {}
        self._scan_trajectory_identities: set[TrajectoryIdentity] = set()
        self._pending_bsplines: dict[
            int,
            dict[TrajectoryIdentity, _PendingBsplineRecord],
        ] = {}
        self._pending_scan_statuses: dict[
            int,
            dict[int, _PendingScanStatusRecord],
        ] = {}
        self._reported_tracking_identity: TrajectoryIdentity | None = None
        self._pending_emergency_identity: TrajectoryIdentity | None = None
        self._controller_snapshot: _ControllerSnapshot | None = None
        self._scan_sequence = MonotonicSequence()
        self._controller_sequence = MonotonicSequence()
        self._last_controller_acceptance_sequence = 0
        self._scan_failure_count = 0
        self._stop_confirmed = False
        self._stop_confirmation_after_sequence = -1
        self._recovery_block_reason = ""
        self._recovery_after_controller_sequence = -1
        self._recovery_blocked_identity: TrajectoryIdentity | None = None

        self._next_wire_request_id = 0
        self._replan: ReplanTransaction | None = None
        self._replan_request: PCTPlanningCommand.Request | None = None
        self._replan_future = None
        self._replan_attempt_token = 0
        self._global_planning_deadline_ns = 0
        self._global_planning_phase = ""
        self._global_replan_cycle_count = 0
        self._global_planning_terminal_reason = ""

        self._navigation_status_sequence = 0
        self._last_navigation_status_signature: Hashable | None = None
        self._last_navigation_status_publish_ns = 0

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self._positive_int("qos.sensor_depth"),
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self._positive_int("qos.path_depth"),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        cached_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self._positive_int("qos.status_depth"),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        trajectory_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self._positive_int("qos.trajectory_depth"),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._status_publisher = self.create_publisher(
            NavigationStatus,
            str(self.get_parameter("topics.navigation_status").value),
            cached_qos,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("topics.odometry").value),
            self._odometry_callback,
            sensor_qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("topics.point_cloud").value),
            self._point_cloud_callback,
            sensor_qos,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("topics.global_path").value),
            self._path_callback,
            path_qos,
        )
        self.create_subscription(
            PCTPlanningStatus,
            str(self.get_parameter("topics.pct_status").value),
            self._pct_status_callback,
            cached_qos,
        )
        self.create_subscription(
            Bspline,
            str(self.get_parameter("topics.bspline").value),
            self._bspline_callback,
            trajectory_qos,
        )
        self.create_subscription(
            ScanPlanningStatus,
            str(self.get_parameter("topics.scan_status").value),
            self._scan_status_callback,
            cached_qos,
        )
        self.create_subscription(
            ControllerStatus,
            str(self.get_parameter("topics.controller_status").value),
            self._controller_status_callback,
            cached_qos,
        )
        self._pct_client = self.create_client(
            PCTPlanningCommand,
            str(self.get_parameter("topics.pct_command_service").value),
        )
        self._timer = self.create_timer(
            self._positive_float("timeouts.timer_period_sec"),
            self._timer_callback,
            clock=self.get_clock(),
        )
        self.get_logger().info(
            "typed navigation supervisor 已启动；本节点不发布 /cmd_vel"
        )

    def _declare_parameters(self) -> None:
        """声明所有 frame、topic、QoS、超时和有界重试参数。"""

        self.declare_parameter("frames.world", "world")
        self.declare_parameter("frames.base", "base_link")
        self.declare_parameter("topics.odometry", "/body_pose")
        self.declare_parameter("topics.point_cloud", "/cloud_registered")
        self.declare_parameter("topics.global_path", "/pct/global_path")
        self.declare_parameter("topics.pct_status", "/pct/planning_status")
        self.declare_parameter("topics.bspline", "/planning/bspline")
        self.declare_parameter("topics.scan_status", "/planning/scan_status")
        self.declare_parameter(
            "topics.controller_status", "/planning/controller_status"
        )
        self.declare_parameter(
            "topics.navigation_status",
            "/navigation/status",
        )
        self.declare_parameter(
            "topics.pct_command_service", "/pct/planning_command"
        )
        self.declare_parameter("qos.sensor_depth", 5)
        self.declare_parameter("qos.path_depth", 1)
        self.declare_parameter("qos.trajectory_depth", 1)
        self.declare_parameter("qos.status_depth", 1)
        self.declare_parameter("timeouts.odometry_sec", 0.30)
        self.declare_parameter("timeouts.point_cloud_sec", 0.50)
        self.declare_parameter("timeouts.bspline_sec", 1.50)
        self.declare_parameter("timeouts.pending_path_evidence_sec", 2.00)
        self.declare_parameter("timeouts.future_tolerance_sec", 0.10)
        self.declare_parameter("timeouts.timer_period_sec", 0.05)
        self.declare_parameter("timeouts.replan_retry_period_sec", 0.50)
        self.declare_parameter("timeouts.replan_response_sec", 1.00)
        self.declare_parameter("timeouts.replan_service_wait_sec", 3.00)
        self.declare_parameter("timeouts.global_planning_sec", 15.00)
        self.declare_parameter(
            "timeouts.trajectory_expiry_grace_sec",
            3.00,
        )
        self.declare_parameter(
            "timeouts.max_yaw_alignment_freeze_sec",
            6.00,
        )
        self.declare_parameter("status.heartbeat_sec", 0.10)
        self.declare_parameter("limits.max_consecutive_scan_failures", 5)
        self.declare_parameter("limits.max_path_points", 4096)
        self.declare_parameter("limits.max_pending_path_evidence", 64)
        self.declare_parameter("limits.replan_max_attempts", 3)
        self.declare_parameter("limits.max_global_replan_cycles", 3)

    def _positive_float(self, name: str) -> float:
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            raise ValueError(f"{name} 必须是有限正数")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} 必须是有限正数")
        return result

    def _positive_duration_ns(
        self,
        name: str,
        *,
        allow_zero: bool = False,
    ) -> int:
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            raise ValueError(f"{name} 必须是有限非负秒数")
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"{name} 必须是有限非负秒数")
        result = int(round(seconds * NANOSECONDS_PER_SECOND))
        if result == 0 and not allow_zero:
            raise ValueError(f"{name} 必须大于零")
        return result

    def _positive_int(self, name: str) -> int:
        value = self.get_parameter(name).value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是正整数")
        return int(value)

    def _clock_now_ns(self) -> int | None:
        """读取 ROS 时钟并在回退时清空所有旧 epoch 运动意图。"""

        now_ns = int(self.get_clock().now().nanoseconds)
        if now_ns <= 0:
            return None
        if self._last_clock_ns > 0 and now_ns < self._last_clock_ns:
            self._reset_for_clock_rollback(now_ns)
        self._last_clock_ns = now_ns
        return now_ns

    def _reset_for_clock_rollback(self, now_ns: int) -> None:
        """时钟回拨时重建核心并 fail closed，禁止复活 transient 旧状态。"""

        self._epoch += 1
        self._core = NavigationSupervisor(self._core_config)
        self._last_decision = None
        self._last_odometry_stamp_ns = 0
        self._last_point_cloud_stamp_ns = 0
        self._goal = None
        self._goal_message = None
        self._pct_plan_id = 0
        self._pct_active_path_stamp_ns = 0
        self._active_path = None
        self._path_records.clear()
        self._path_signatures.clear()
        self._pct_successes.clear()
        self._pct_failures.clear()
        self._pct_replan_ack_candidates.clear()
        self._latest_tombstone_stamp_ns = 0
        self._last_pct_status_stamp_ns = 0
        self._pct_status_signatures.clear()
        self._trajectories.clear()
        self._scan_trajectory_identities.clear()
        self._clear_pending_path_evidence()
        self._reported_tracking_identity = None
        self._pending_emergency_identity = None
        self._controller_snapshot = None
        self._scan_sequence.reset()
        self._controller_sequence.reset()
        self._last_controller_acceptance_sequence = 0
        self._scan_failure_count = 0
        self._stop_confirmed = False
        self._stop_confirmation_after_sequence = -1
        self._recovery_block_reason = "clock_epoch_reset"
        self._recovery_after_controller_sequence = -1
        self._recovery_blocked_identity = None
        if self._replan is not None and not self._replan.terminal:
            self._replan.mark_terminal_error("clock_epoch_reset")
        self._replan = None
        self._replan_request = None
        self._replan_future = None
        self._replan_attempt_token += 1
        self._global_planning_deadline_ns = 0
        self._global_planning_phase = ""
        self._global_replan_cycle_count = 0
        self._global_planning_terminal_reason = ""
        self._adapter_fault_reason = "clock_epoch_reset"
        self._protocol_poisoned = True
        self._fatal_epoch_reset = True
        self._last_navigation_status_publish_ns = 0
        self.get_logger().error(
            f"ROS 时钟回拨到 {now_ns}ns，已清空旧导航 epoch"
        )

    def _message_stamp_ns(self, message: object, now_ns: int) -> int:
        """验证关键 header 时间非零且没有超前当前仿真时钟。"""

        if self._fatal_epoch_reset:
            raise ValueError("时钟 epoch 已回拨，必须重启完整导航 ROS 图")
        stamp_ns = stamp_to_nanoseconds(getattr(message, "header").stamp)
        if stamp_ns > now_ns + self._future_tolerance_ns:
            raise ValueError("消息时间戳超前于当前 ROS 时钟")
        return stamp_ns

    def _odometry_callback(self, message: Odometry) -> None:
        now_ns = self._clock_now_ns()
        if now_ns is None:
            return
        try:
            stamp_ns = self._message_stamp_ns(message, now_ns)
            if now_ns - stamp_ns > self._odometry_timeout_ns:
                raise ValueError("Odometry header 已超过安全超时")
            if stamp_ns <= self._last_odometry_stamp_ns:
                raise ValueError("Odometry header 未严格推进")
            if _normalize_frame(message.header.frame_id) != self._world_frame:
                raise ValueError("Odometry world frame 不匹配")
            if _normalize_frame(message.child_frame_id) != self._base_frame:
                raise ValueError("Odometry base frame 不匹配")
            values = finite_tuple(
                (
                    message.pose.pose.position.x,
                    message.pose.pose.position.y,
                    message.pose.pose.position.z,
                    message.pose.pose.orientation.x,
                    message.pose.pose.orientation.y,
                    message.pose.pose.orientation.z,
                    message.pose.pose.orientation.w,
                    message.twist.twist.linear.x,
                    message.twist.twist.linear.y,
                    message.twist.twist.linear.z,
                    message.twist.twist.angular.x,
                    message.twist.twist.angular.y,
                    message.twist.twist.angular.z,
                ),
                field_name="Odometry",
            )
            norm = math.sqrt(sum(value * value for value in values[3:7]))
            if abs(norm - 1.0) > 0.05:
                raise ValueError("Odometry 四元数不是单位四元数")
        except ValueError as exc:
            self._protocol_warning(f"拒绝 Odometry：{exc}")
            return
        self._last_odometry_stamp_ns = stamp_ns
        # /clock 与传感器 topic 是独立 DDS 回调，adapter 允许 header
        # 在 future_tolerance 内先于本回调读到的 /clock 一小拍。原始
        # stamp 仍用于严格单调与审计；core 的 freshness 时间只能
        # 使用不晚于当前 ROS 时钟的保守值，避免凭未来时间延长超时窗口。
        observed_at_ns = min(stamp_ns, now_ns)
        decision = self._core.observe_odometry(
            nanoseconds_to_seconds(now_ns),
            observed_at_s=nanoseconds_to_seconds(observed_at_ns),
        )
        self._apply_decision(decision, now_ns)
        self._try_clear_recoverable_emergency(now_ns)

    def _point_cloud_callback(self, message: PointCloud2) -> None:
        now_ns = self._clock_now_ns()
        if now_ns is None:
            return
        try:
            stamp_ns = self._message_stamp_ns(message, now_ns)
            if now_ns - stamp_ns > self._point_cloud_timeout_ns:
                raise ValueError("PointCloud2 header 已超过安全超时")
            if stamp_ns <= self._last_point_cloud_stamp_ns:
                raise ValueError("PointCloud2 header 未严格推进")
            if _normalize_frame(message.header.frame_id) != self._world_frame:
                raise ValueError("PointCloud2 world frame 不匹配")
            point_count = int(message.width) * int(message.height)
            if point_count <= 0:
                # bridge 在有效原始点全部被地面/自点过滤后会发布
                # canonical empty。它代表本帧没有保留障碍点，仍是
                # 新鲜传感器证据；其他空布局仍必须拒绝。
                if not _is_canonical_empty_point_cloud(message):
                    raise ValueError(
                        "空 PointCloud2 必须使用 canonical xyz32 非组织布局"
                    )
            else:
                field_names = {str(field.name) for field in message.fields}
                if int(message.point_step) <= 0:
                    raise ValueError("PointCloud2 point_step 必须为正数")
                if not {"x", "y", "z"}.issubset(field_names):
                    raise ValueError("PointCloud2 缺少 x/y/z 字段")
                minimum_bytes = (
                    max(int(message.row_step), 0) * int(message.height)
                )
                if len(message.data) < minimum_bytes:
                    raise ValueError("PointCloud2 data 小于 row_step 合同")
        except ValueError as exc:
            self._protocol_warning(f"拒绝 PointCloud2：{exc}")
            return
        self._last_point_cloud_stamp_ns = stamp_ns
        # 与 Odometry 使用同一条时间合同：保留原始 header 做重放
        # 检查，但不让 tolerance 内的 DDS 乱序为 core 预支新鲜度。
        observed_at_ns = min(stamp_ns, now_ns)
        decision = self._core.observe_point_cloud(
            nanoseconds_to_seconds(now_ns),
            observed_at_s=nanoseconds_to_seconds(observed_at_ns),
        )
        self._apply_decision(decision, now_ns)
        self._try_clear_recoverable_emergency(now_ns)

    def _path_callback(self, message: Path) -> None:
        now_ns = self._clock_now_ns()
        if now_ns is None:
            return
        self._expire_pending_path_evidence(now_ns)
        if self._expire_global_planning_deadline(now_ns):
            return
        try:
            stamp_ns = self._message_stamp_ns(message, now_ns)
            if _normalize_frame(message.header.frame_id) != self._world_frame:
                raise ValueError("Path world frame 不匹配")
            signature = self._path_signature(message, stamp_ns)
        except ValueError as exc:
            self._fail_closed(now_ns, f"invalid_global_path:{exc}")
            return

        previous = self._path_signatures.get(stamp_ns)
        if previous is not None:
            if previous != signature:
                self._fail_closed(now_ns, "conflicting_same_stamp_global_path")
            return
        if self._path_signatures and stamp_ns < max(self._path_signatures):
            self._protocol_warning("忽略早于已观察代际的全局 Path")
            return
        self._path_signatures[stamp_ns] = signature
        if not message.poses:
            removed_pending = (
                self._discard_pending_path_evidence_at_or_before(stamp_ns)
            )
            if removed_pending:
                self._fail_closed(
                    now_ns,
                    "pending_path_evidence_invalidated_by_tombstone",
                )
            invalidated_active = bool(
                self._active_path is not None
                and stamp_ns > self._active_path.stamp_ns
            )
            self._latest_tombstone_stamp_ns = max(
                self._latest_tombstone_stamp_ns,
                stamp_ns,
            )
            if self._global_planning_terminal_reason:
                self._protocol_warning(
                    "终止停车已锁存，仅缓存空 Path 等待新 goal"
                )
                self._prune_caches()
                self._publish_status(now_ns)
                return
            if invalidated_active:
                self._active_path = None
                self._reported_tracking_identity = None
                if self._core.state in {
                    NavigationState.LOCAL_PLANNING,
                    NavigationState.TRACKING,
                }:
                    self._arm_stop_confirmation()
                    self._latch_recovery("global_path_tombstone")
                    self._last_decision = self._core.report_emergency_stop(
                        nanoseconds_to_seconds(now_ns),
                        reason="global_path_tombstone",
                        request_global_replan=False,
                    )
            self._evaluate_stop_confirmation(now_ns)
            self._try_ack_replan_status(stamp_ns, now_ns)
            self._try_process_pct_failure(stamp_ns, now_ns)
            self._prune_caches()
            self._publish_status(now_ns)
            return

        try:
            identity = self._path_identity(message, stamp_ns)
        except ValueError as exc:
            self._fail_closed(now_ns, f"invalid_nonempty_global_path:{exc}")
            return
        self._path_records[stamp_ns] = identity
        if self._global_planning_terminal_reason:
            self._protocol_warning(
                "终止停车已锁存，仅缓存非空 Path 等待新 goal"
            )
            self._prune_caches()
            self._publish_status(now_ns)
            return
        self._try_ack_replan_status(stamp_ns, now_ns)
        self._try_accept_global_path(stamp_ns, now_ns)
        self._prune_caches()

    def _path_signature(self, message: Path, stamp_ns: int) -> Hashable:
        if len(message.poses) > self._max_path_points:
            raise ValueError("Path 点数超过安全上限")
        payload: list[Hashable] = []
        for index, pose in enumerate(message.poses):
            pose_frame = (
                _normalize_frame(pose.header.frame_id)
                if str(pose.header.frame_id).strip()
                else self._world_frame
            )
            if pose_frame != self._world_frame:
                raise ValueError(f"Path 第 {index} 个 pose frame 不匹配")
            pose_stamp_ns = stamp_to_nanoseconds(pose.header.stamp)
            if pose_stamp_ns != stamp_ns:
                raise ValueError(f"Path 第 {index} 个 pose stamp 不同代")
            payload.append((pose_stamp_ns, pose_frame, _pose_values(pose)))
        return (stamp_ns, self._world_frame, tuple(payload))

    def _path_identity(self, message: Path, stamp_ns: int) -> PathIdentity:
        if len(message.poses) < 2:
            raise ValueError("非空 Path 至少需要两个 pose")
        payload = tuple(self._path_signature(message, stamp_ns)[2])
        return PathIdentity(
            stamp_ns=stamp_ns,
            frame_id=self._world_frame,
            point_count=len(message.poses),
            payload=payload,
        )

    def _pct_status_callback(self, message: PCTPlanningStatus) -> None:
        now_ns = self._clock_now_ns()
        if now_ns is None:
            return
        if self._expire_global_planning_deadline(now_ns):
            return
        try:
            stamp_ns = self._message_stamp_ns(message, now_ns)
            if _normalize_frame(message.header.frame_id) != self._world_frame:
                raise ValueError("PCT status world frame 不匹配")
            signature = self._pct_status_signature(message)
            if stamp_ns < self._last_pct_status_stamp_ns:
                raise ValueError("PCT status 时间倒退")
        except ValueError as exc:
            self._protocol_warning(f"拒绝 PCT status：{exc}")
            return
        if signature in self._pct_status_signatures:
            return
        self._pct_status_signatures.add(signature)
        self._last_pct_status_stamp_ns = max(
            self._last_pct_status_stamp_ns,
            stamp_ns,
        )
        self._next_wire_request_id = max(
            self._next_wire_request_id,
            int(message.request_id),
        )
        incoming_plan_id = int(message.plan_id)
        if incoming_plan_id < self._pct_plan_id:
            self._protocol_warning("忽略旧 plan_id 的 PCT status")
            return

        if message.has_active_goal:
            try:
                goal = _goal_identity(
                    message.active_goal,
                    int(message.goal_id),
                )
            except ValueError as exc:
                self._fail_closed(now_ns, f"invalid_pct_active_goal:{exc}")
                return
            if goal.frame_id != self._world_frame:
                self._fail_closed(now_ns, "pct_active_goal_frame_mismatch")
                return
            if (
                self._global_planning_terminal_reason
                and self._goal is not None
                and goal.goal_id == self._goal.goal_id
            ):
                self._protocol_warning(
                    "忽略终止停车后属于同一 goal 的 PCT status"
                )
                return
            if self._goal is None or goal.goal_id != self._goal.goal_id:
                self._start_goal(goal, message.active_goal, now_ns)
            elif goal != self._goal:
                self._fail_closed(
                    now_ns,
                    "conflicting_pct_active_goal_snapshot",
                )
                return
            self._pct_active_path_stamp_ns = self._optional_stamp_ns(
                message.active_path_stamp
            )
        elif (
            int(message.state) == PCTPlanningStatus.IDLE
            or int(message.command) == PCTPlanningStatus.COMMAND_CANCEL
        ):
            self._cancel_goal(now_ns, "pct_goal_cancelled")
            return
        elif int(message.state) == PCTPlanningStatus.ERROR:
            # PCT 拒绝非法新 goal 时会撤销旧意图，并发布无活动目标 ERROR。
            self._cancel_goal(now_ns, "pct_invalid_goal_rejected")
            return

        if self._goal is None or int(message.goal_id) != self._goal.goal_id:
            self._protocol_warning("忽略不属于当前活动 goal 的 PCT status")
            return

        path_stamp_ns = self._optional_stamp_ns(message.path_stamp)
        active_path_stamp_ns = self._optional_stamp_ns(
            message.active_path_stamp
        )
        state = int(message.state)
        request_id = int(message.request_id)
        command = int(message.command)
        goal_stamp_ns = self._optional_stamp_ns(message.goal_stamp)
        if goal_stamp_ns != self._goal.stamp_ns:
            self._fail_closed(now_ns, "pct_status_goal_stamp_mismatch")
            return
        if path_stamp_ns <= 0 or active_path_stamp_ns != path_stamp_ns:
            self._fail_closed(now_ns, "pct_status_active_path_mismatch")
            return
        if incoming_plan_id <= 0:
            self._fail_closed(now_ns, "pct_status_plan_id_must_be_positive")
            return
        if self._pct_plan_id > 0 and incoming_plan_id > self._pct_plan_id:
            if command == PCTPlanningStatus.COMMAND_REPLAN:
                transaction = self._replan
                if transaction is None or request_id != transaction.request_id:
                    self._fail_closed(now_ns, "unexpected_pct_replan_plan_id")
                    return
            elif command != PCTPlanningStatus.COMMAND_PLAN:
                self._fail_closed(now_ns, "unexpected_pct_plan_generation")
                return
        self._pct_plan_id = incoming_plan_id
        self._pct_active_path_stamp_ns = path_stamp_ns

        if command == PCTPlanningStatus.COMMAND_REPLAN:
            transaction = self._replan
            if transaction is None or request_id != transaction.request_id:
                self._protocol_warning("忽略不属于当前事务的 PCT REPLAN status")
                return
            if path_stamp_ns <= transaction.expected_path_stamp_ns:
                self._fail_closed(now_ns, "pct_replan_path_not_strictly_newer")
                return
            self._pct_replan_ack_candidates[path_stamp_ns] = (
                request_id,
                incoming_plan_id,
                state,
            )
            self._try_ack_replan_status(path_stamp_ns, now_ns)
        elif command == PCTPlanningStatus.COMMAND_PLAN:
            if request_id < 0:
                self._fail_closed(now_ns, "pct_plan_request_id_invalid")
                return
        else:
            self._fail_closed(now_ns, "pct_status_command_invalid")
            return

        if state == PCTPlanningStatus.SUCCEEDED:
            if path_stamp_ns <= 0 or int(message.path_point_count) < 2:
                self._fail_closed(now_ns, "pct_success_without_nonempty_path")
                return
            self._pct_successes[path_stamp_ns] = (
                int(message.goal_id),
                incoming_plan_id,
                int(message.path_point_count),
                request_id,
                command,
            )
            self._try_accept_global_path(path_stamp_ns, now_ns)
        elif state in {PCTPlanningStatus.NO_PATH, PCTPlanningStatus.ERROR}:
            if int(message.path_point_count) != 0:
                self._fail_closed(now_ns, "pct_failure_has_nonzero_path_count")
                return
            self._pct_failures[path_stamp_ns] = (
                int(message.goal_id),
                incoming_plan_id,
                request_id,
                command,
                (
                    "pct_no_path"
                    if state == PCTPlanningStatus.NO_PATH
                    else "pct_planning_error"
                ),
            )
            self._try_process_pct_failure(path_stamp_ns, now_ns)
        else:
            if int(message.path_point_count) != 0:
                self._fail_closed(
                    now_ns,
                    "pct_planning_status_has_path_points",
                )
                return
            self._publish_status(now_ns)

    def _pct_status_signature(self, message: PCTPlanningStatus) -> Hashable:
        goal_signature: Hashable = None
        if message.has_active_goal:
            goal_signature = (
                int(message.goal_id),
                stamp_to_nanoseconds(message.active_goal.header.stamp),
                _normalize_frame(message.active_goal.header.frame_id),
                _pose_values(message.active_goal),
            )
        return (
            stamp_to_nanoseconds(message.header.stamp),
            int(message.plan_id),
            int(message.goal_id),
            int(message.request_id),
            int(message.command),
            int(message.state),
            self._optional_stamp_ns(message.goal_stamp),
            self._optional_stamp_ns(message.path_stamp),
            bool(message.has_active_goal),
            goal_signature,
            self._optional_stamp_ns(message.active_path_stamp),
            int(message.path_point_count),
            str(message.message),
        )

    def _try_ack_replan_status(self, path_stamp_ns: int, now_ns: int) -> None:
        """只用与实际 Path topic 对账后的 typed status 作为旁路 ACK。"""

        candidate = self._pct_replan_ack_candidates.get(path_stamp_ns)
        transaction = self._replan
        if candidate is None or transaction is None:
            return
        request_id, _plan_id, state = candidate
        if request_id != transaction.request_id:
            return
        signature = self._path_signatures.get(path_stamp_ns)
        if signature is None:
            return
        path_is_empty = not bool(signature[2])
        if state == PCTPlanningStatus.SUCCEEDED:
            if path_is_empty or path_stamp_ns not in self._path_records:
                return
        elif state in {
            PCTPlanningStatus.WAITING_FOR_ODOMETRY,
            PCTPlanningStatus.PLANNING,
            PCTPlanningStatus.NO_PATH,
            PCTPlanningStatus.ERROR,
        }:
            if not path_is_empty:
                self._fail_closed(
                    now_ns,
                    "pct_replan_status_path_kind_mismatch",
                )
                return
        else:
            return
        self._acknowledge_replan(now_ns)

    def _try_process_pct_failure(
        self,
        path_stamp_ns: int,
        now_ns: int,
    ) -> None:
        """等待真实空 Path 后，才把当前 PCT 失败推进到 core。"""

        failure = self._pct_failures.get(path_stamp_ns)
        signature = self._path_signatures.get(path_stamp_ns)
        if failure is None or signature is None or bool(signature[2]):
            return
        goal_id, plan_id, request_id, command, reason = failure
        if (
            self._goal is None
            or goal_id != self._goal.goal_id
            or plan_id != self._pct_plan_id
        ):
            return
        if command == PCTPlanningStatus.COMMAND_REPLAN:
            self._try_ack_replan_status(path_stamp_ns, now_ns)
            transaction = self._replan
            if (
                transaction is None
                or request_id != transaction.request_id
                or not transaction.acknowledged
            ):
                return
        if self._core.state is not NavigationState.GLOBAL_PLANNING:
            return
        decision = self._core.report_global_planning_failed(
            nanoseconds_to_seconds(now_ns),
            reason=reason,
        )
        self._clear_global_planning_deadline()
        if not self._stop_confirmed:
            self._arm_stop_confirmation()
        del self._pct_failures[path_stamp_ns]
        self._last_decision = decision
        if (
            command == PCTPlanningStatus.COMMAND_REPLAN
            and self._global_replan_cycle_count
            >= self._max_global_replan_cycles
        ):
            self._terminate_global_planning(
                now_ns,
                "global_replan_cycles_exhausted",
            )
            return
        self._apply_decision(decision, now_ns)

    def _start_goal(
        self,
        goal: GoalIdentity,
        message: PoseStamped,
        now_ns: int,
    ) -> None:
        replacing_goal = bool(
            self._goal is not None and self._goal.goal_id != goal.goal_id
        )
        if replacing_goal:
            self._clear_pending_path_evidence()
        self._goal = goal
        self._goal_message = deepcopy(message)
        self._pct_plan_id = 0
        self._pct_active_path_stamp_ns = 0
        self._active_path = None
        self._pct_successes.clear()
        self._pct_failures.clear()
        self._pct_replan_ack_candidates.clear()
        self._trajectories.clear()
        self._scan_trajectory_identities.clear()
        self._reported_tracking_identity = None
        self._pending_emergency_identity = None
        self._recovery_block_reason = ""
        self._recovery_after_controller_sequence = -1
        self._recovery_blocked_identity = None
        self._scan_failure_count = 0
        self._arm_stop_confirmation()
        self._replan = None
        self._replan_request = None
        self._replan_future = None
        self._replan_attempt_token += 1
        self._global_replan_cycle_count = 0
        self._global_planning_terminal_reason = ""
        self._begin_global_planning_deadline(now_ns, phase="initial")
        self._adapter_fault_reason = ""
        self._protocol_poisoned = False
        self._apply_decision(
            self._core.start_goal(nanoseconds_to_seconds(now_ns)),
            now_ns,
        )

    def _cancel_goal(self, now_ns: int, reason: str) -> None:
        if self._core.state is not NavigationState.IDLE:
            self._last_decision = self._core.cancel(
                nanoseconds_to_seconds(now_ns)
            )
        self._goal = None
        self._goal_message = None
        self._pct_active_path_stamp_ns = 0
        self._active_path = None
        self._clear_pending_path_evidence()
        self._reported_tracking_identity = None
        self._pending_emergency_identity = None
        self._recovery_block_reason = ""
        self._recovery_after_controller_sequence = -1
        self._recovery_blocked_identity = None
        self._replan = None
        self._replan_request = None
        self._replan_future = None
        self._clear_global_planning_deadline()
        self._global_replan_cycle_count = 0
        self._global_planning_terminal_reason = ""
        self._arm_stop_confirmation()
        self._adapter_fault_reason = reason
        self._publish_status(now_ns)

    def _try_accept_global_path(self, stamp_ns: int, now_ns: int) -> None:
        path = self._path_records.get(stamp_ns)
        success = self._pct_successes.get(stamp_ns)
        if path is None or success is None or self._goal is None:
            self._publish_status(now_ns)
            return
        goal_id, plan_id, point_count, request_id, command = success
        if (
            goal_id != self._goal.goal_id
            or plan_id != self._pct_plan_id
            or point_count != path.point_count
        ):
            self._fail_closed(now_ns, "pct_status_path_identity_mismatch")
            return
        if command == PCTPlanningStatus.COMMAND_REPLAN:
            transaction = self._replan
            if transaction is None or request_id != transaction.request_id:
                self._fail_closed(now_ns, "pct_success_request_mismatch")
                return
            self._try_ack_replan_status(stamp_ns, now_ns)
            if not transaction.acknowledged:
                return
        elif command != PCTPlanningStatus.COMMAND_PLAN:
            self._fail_closed(now_ns, "pct_success_command_mismatch")
            return
        if self._active_path == path:
            return
        if (
            self._active_path is not None
            and stamp_ns <= self._active_path.stamp_ns
        ):
            self._fail_closed(
                now_ns,
                "global_path_generation_not_strictly_newer",
            )
            return
        if self._core.state not in {
            NavigationState.GLOBAL_PLANNING,
            NavigationState.GLOBAL_REPLAN,
            NavigationState.EMERGENCY_STOP,
        }:
            self._fail_closed(
                now_ns,
                "unexpected_global_path_in_current_state",
            )
            return
        if self._has_pending_path_evidence_after(stamp_ns):
            self._discard_pending_path_evidence_at_or_before(stamp_ns)
            self._reject_path_while_forcing_zero(
                now_ns,
                "global_path_older_than_pending_path_evidence",
            )
            return
        if self._has_pending_path_evidence_before(stamp_ns):
            self._clear_pending_path_evidence()
            self._fail_closed(
                now_ns,
                "pending_path_evidence_old_generation",
            )
            return
        if (
            self._core.state is NavigationState.EMERGENCY_STOP
            and not self._decision().global_replan_requested
            and not self._decision().global_replan_in_flight
        ):
            self._fail_closed(
                now_ns,
                "protocol_fault_requires_new_goal_before_path",
            )
            return
        self._active_path = path
        self._pct_plan_id = plan_id
        self._pct_active_path_stamp_ns = stamp_ns
        self._reported_tracking_identity = None
        self._pending_emergency_identity = None
        self._scan_failure_count = 0
        self._recovery_block_reason = ""
        self._recovery_after_controller_sequence = -1
        self._recovery_blocked_identity = None
        self._replan = None
        self._replan_request = None
        self._replan_future = None
        self._clear_global_planning_deadline()
        self._global_replan_cycle_count = 0
        self._global_planning_terminal_reason = ""
        self._adapter_fault_reason = ""
        self._protocol_poisoned = False
        self._apply_decision(
            self._core.report_global_path_available(
                nanoseconds_to_seconds(now_ns)
            ),
            now_ns,
        )

        self._replay_pending_path_evidence(stamp_ns, now_ns)

    def _pending_path_evidence_count(self) -> int:
        """返回尚未与 PCT Path 对账的证据总数。"""

        return sum(map(len, self._pending_bsplines.values())) + sum(
            map(len, self._pending_scan_statuses.values())
        )

    def _reject_path_while_forcing_zero(
        self,
        now_ns: int,
        reason: str,
    ) -> None:
        """拒绝落后 Path，不改变当前已经强制零速的状态。"""

        self._adapter_fault_reason = str(reason)
        self._protocol_poisoned = True
        self._arm_stop_confirmation()
        self.get_logger().error(
            f"navigation supervisor 拒绝旧 Path 并保持零速：{reason}"
        )
        self._publish_status(now_ns)

    def _clear_pending_path_evidence(self) -> None:
        """清空所有尚未完成 Path 代际对账的运动证据。"""

        self._pending_bsplines.clear()
        self._pending_scan_statuses.clear()

    def _has_pending_path_evidence_before(self, stamp_ns: int) -> bool:
        """检查是否存在会污染新活动 Path 的旧代证据。"""

        return any(
            generation < stamp_ns
            for generation in (
                *self._pending_bsplines,
                *self._pending_scan_statuses,
            )
        )

    def _has_pending_path_evidence_after(self, stamp_ns: int) -> bool:
        """检查是否已观察到比候选活动 Path 更新的运动证据。"""

        return any(
            generation > stamp_ns
            for generation in (
                *self._pending_bsplines,
                *self._pending_scan_statuses,
            )
        )

    def _discard_pending_path_evidence_at_or_before(
        self,
        stamp_ns: int,
    ) -> int:
        """丢弃被 Path tombstone 覆盖的缓存并返回数量。"""

        removed = 0
        for mapping in (
            self._pending_bsplines,
            self._pending_scan_statuses,
        ):
            for generation in tuple(mapping):
                if generation <= stamp_ns:
                    removed += len(mapping.pop(generation))
        return removed

    def _expire_pending_path_evidence(self, now_ns: int) -> bool:
        """缓存超过有界乱序窗口时清空全部证据并锁存停车。"""

        records = (
            *(
                record
                for generation in self._pending_bsplines.values()
                for record in generation.values()
            ),
            *(
                record
                for generation in self._pending_scan_statuses.values()
                for record in generation.values()
            ),
        )
        if not any(
            now_ns - record.received_at_ns
            > self._pending_path_evidence_timeout_ns
            for record in records
        ):
            return False
        self._clear_pending_path_evidence()
        self._fail_closed(now_ns, "pending_path_evidence_timeout")
        return True

    def _ensure_pending_path_evidence_capacity(self, now_ns: int) -> bool:
        """为一条新乱序证据预留槽位，溢出时 fail closed。"""

        if (
            self._pending_path_evidence_count()
            < self._max_pending_path_evidence
        ):
            return True
        self._clear_pending_path_evidence()
        self._fail_closed(now_ns, "pending_path_evidence_capacity_exceeded")
        return False

    def _cache_pending_bspline(
        self,
        message: Bspline,
        now_ns: int,
        identity: TrajectoryIdentity,
        signature: Hashable,
    ) -> bool:
        """按 reference Path stamp 有界缓存提前到达的 B-spline。"""

        generation = identity.reference_path_stamp_ns
        previous = self._pending_bsplines.get(generation, {}).get(identity)
        if previous is not None:
            if previous.signature != signature:
                self._clear_pending_path_evidence()
                self._fail_closed(
                    now_ns,
                    "conflicting_pending_bspline_identity",
                )
                return False
            return True
        if not self._ensure_pending_path_evidence_capacity(now_ns):
            return False
        self._pending_bsplines.setdefault(generation, {})[identity] = (
            _PendingBsplineRecord(
                message=deepcopy(message),
                received_at_ns=now_ns,
                identity=identity,
                signature=signature,
            )
        )
        return True

    def _cache_pending_scan_status(
        self,
        message: ScanPlanningStatus,
        now_ns: int,
        reference_stamp_ns: int,
        signature: Hashable,
    ) -> bool:
        """按 reference Path stamp 和 status sequence 缓存 SCAN 状态。"""

        sequence = int(message.status_sequence)
        if sequence <= self._scan_sequence.latest:
            disposition = self._scan_sequence.observe(sequence, signature)
            if disposition is SequenceDisposition.CONFLICT:
                self._clear_pending_path_evidence()
                self._fail_closed(
                    now_ns,
                    "conflicting_scan_status_sequence",
                )
                return False
            return True
        for records in self._pending_scan_statuses.values():
            previous = records.get(sequence)
            if previous is None:
                continue
            if previous.signature != signature:
                self._clear_pending_path_evidence()
                self._fail_closed(
                    now_ns,
                    "conflicting_pending_scan_status_sequence",
                )
                return False
            return True
        if not self._ensure_pending_path_evidence_capacity(now_ns):
            return False
        self._pending_scan_statuses.setdefault(reference_stamp_ns, {})[
            sequence
        ] = _PendingScanStatusRecord(
            message=deepcopy(message),
            received_at_ns=now_ns,
            reference_path_stamp_ns=reference_stamp_ns,
            status_sequence=sequence,
            signature=signature,
        )
        return True

    def _replay_pending_path_evidence(
        self,
        stamp_ns: int,
        now_ns: int,
    ) -> None:
        """Path 激活后重放同代 B-spline，再按序重放 SCAN 状态。"""

        if self._expire_pending_path_evidence(now_ns):
            return
        bspline_records = tuple(
            self._pending_bsplines.pop(stamp_ns, {}).values()
        )
        scan_records = tuple(
            sorted(
                self._pending_scan_statuses.pop(stamp_ns, {}).values(),
                key=lambda record: record.status_sequence,
            )
        )
        for record in bspline_records:
            self._bspline_callback(deepcopy(record.message))
            if self._protocol_poisoned:
                return
        for record in scan_records:
            self._scan_status_callback(deepcopy(record.message))
            if self._protocol_poisoned:
                return

    def _bspline_callback(self, message: Bspline) -> None:
        now_ns = self._clock_now_ns()
        if now_ns is None:
            return
        if self._expire_pending_path_evidence(now_ns):
            return
        try:
            self._message_stamp_ns(message, now_ns)
            if _normalize_frame(message.header.frame_id) != self._world_frame:
                raise ValueError("B-spline world frame 不匹配")
            identity = _trajectory_identity_from_bspline(message)
            if (
                identity.reference_path_stamp_ns
                > now_ns + self._future_tolerance_ns
                or identity.start_time_ns
                > now_ns + self._future_tolerance_ns
            ):
                raise ValueError("B-spline 身份时间超前于当前 ROS 时钟")
            points = tuple(
                finite_tuple((point.x, point.y, point.z), field_name="pos_pts")
                for point in message.pos_pts
            )
            valid_until_ns = bspline_valid_until_ns(
                order=int(message.order),
                control_point_count=len(points),
                knots=message.knots,
                start_time_ns=identity.start_time_ns,
            )
            if len(message.knots) != len(points) + int(message.order) + 1:
                raise ValueError("B-spline knots 数量不满足 cp+order+1")
            yaw_points = finite_tuple(message.yaw_pts, field_name="yaw_pts")
            yaw_dt = float(message.yaw_dt)
            if not math.isfinite(yaw_dt):
                raise ValueError("B-spline yaw_dt 必须有限")
            if yaw_points and yaw_dt <= 0.0:
                raise ValueError("B-spline yaw_dt 必须为有限正数")
            signature = (
                identity,
                int(message.order),
                tuple(float(value) for value in message.knots),
                points,
                yaw_points,
                yaw_dt,
            )
        except (TypeError, ValueError) as exc:
            self._fail_closed(now_ns, f"invalid_bspline:{exc}")
            return
        reference_stamp_ns = identity.reference_path_stamp_ns
        if self._is_tombstone_retired_path_generation(reference_stamp_ns):
            previous = self._trajectories.get(identity)
            if previous is not None and previous.signature != signature:
                self._fail_closed(
                    now_ns,
                    "conflicting_retired_bspline_identity",
                )
                return
            # 权威空 Path 已经淘汰该代；晚到轨迹不能恢复运动，也不应把
            # 正常跨 topic 交接误判成永久协议故障。
            self._protocol_warning("忽略已被 tombstone 淘汰代际的 B-spline")
            return
        if self._active_path is None:
            if reference_stamp_ns <= self._latest_tombstone_stamp_ns:
                self._fail_closed(
                    now_ns,
                    "bspline_generation_not_newer_than_tombstone",
                )
                return
            self._cache_pending_bspline(
                message,
                now_ns,
                identity,
                signature,
            )
            return
        if reference_stamp_ns != self._active_path.stamp_ns:
            if reference_stamp_ns < self._active_path.stamp_ns:
                self._fail_closed(now_ns, "bspline_from_old_path_generation")
                return
            if self._cache_pending_bspline(
                message,
                now_ns,
                identity,
                signature,
            ):
                self._fail_closed(
                    now_ns,
                    "bspline_precedes_newer_active_path",
                )
            return
        previous = self._trajectories.get(identity)
        if previous is not None:
            if previous.signature != signature:
                self._fail_closed(now_ns, "conflicting_bspline_identity")
            return
        self._trajectories[identity] = _TrajectoryRecord(
            identity=identity,
            valid_until_ns=valid_until_ns,
            signature=signature,
        )
        if not identity.emergency_stop:
            self._stop_confirmed = False
        self._try_report_controller_tracking(now_ns)
        self._evaluate_stop_confirmation(now_ns)
        self._prune_caches()

    def _scan_status_callback(self, message: ScanPlanningStatus) -> None:
        now_ns = self._clock_now_ns()
        if now_ns is None:
            return
        if self._expire_pending_path_evidence(now_ns):
            return
        try:
            self._message_stamp_ns(message, now_ns)
            if _normalize_frame(message.header.frame_id) != self._world_frame:
                raise ValueError("SCAN status world frame 不匹配")
            signature = self._scan_status_signature(message)
        except ValueError as exc:
            self._fail_closed(now_ns, f"invalid_scan_status:{exc}")
            return

        event = int(message.event)
        path_bound = event not in {
            ScanPlanningStatus.EVENT_INITIAL,
            ScanPlanningStatus.EVENT_REFERENCE_CLEARED,
        }
        trajectory: TrajectoryIdentity | None = None
        try:
            self._validate_scan_status_policy(message)
            reference_stamp_ns = self._optional_stamp_ns(
                message.reference_path_stamp
            )
            if path_bound and reference_stamp_ns <= 0:
                raise ValueError("Path 绑定的 SCAN 事件缺少 reference stamp")
            if (
                path_bound
                and reference_stamp_ns
                > now_ns + self._future_tolerance_ns
            ):
                raise ValueError("SCAN reference stamp 超前于当前 ROS 时钟")
            if message.trajectory_present:
                if event not in {
                    ScanPlanningStatus.EVENT_TRAJECTORY_PUBLISHED,
                    ScanPlanningStatus.EVENT_EMERGENCY_STOP,
                    ScanPlanningStatus.EVENT_RECOVERED,
                    ScanPlanningStatus.EVENT_GOAL_HOLD,
                }:
                    raise ValueError("该 SCAN 事件不允许携带轨迹身份")
                trajectory = _trajectory_identity_from_scan(message)
                if trajectory.reference_path_stamp_ns != reference_stamp_ns:
                    raise ValueError("SCAN 轨迹与状态的 Path 代际不一致")
                if (
                    trajectory.bspline_header_stamp_ns
                    > now_ns + self._future_tolerance_ns
                    or trajectory.start_time_ns
                    > now_ns + self._future_tolerance_ns
                ):
                    raise ValueError("SCAN 轨迹身份时间超前于当前 ROS 时钟")
        except ValueError as exc:
            self._fail_closed(now_ns, f"scan_status_invariant:{exc}")
            return

        if path_bound:
            if self._is_tombstone_retired_path_generation(
                reference_stamp_ns
            ):
                try:
                    retired_disposition = self._scan_sequence.observe(
                        int(message.status_sequence),
                        signature,
                    )
                except ValueError as exc:
                    self._fail_closed(
                        now_ns,
                        f"invalid_scan_status:{exc}",
                    )
                    return
                if retired_disposition is SequenceDisposition.CONFLICT:
                    self._fail_closed(
                        now_ns,
                        "conflicting_retired_scan_status_sequence",
                    )
                    return
                # 仍消费 wire sequence 并保留同序冲突检查，但不让旧代状态
                # 写入轨迹、失败计数、恢复屏障或 supervisor core。
                self._protocol_warning("忽略已被 tombstone 淘汰代际的 SCAN 状态")
                return
            if self._active_path is None:
                if reference_stamp_ns <= self._latest_tombstone_stamp_ns:
                    self._fail_closed(
                        now_ns,
                        "scan_generation_not_newer_than_tombstone",
                    )
                    return
                self._cache_pending_scan_status(
                    message,
                    now_ns,
                    reference_stamp_ns,
                    signature,
                )
                return
            if reference_stamp_ns != self._active_path.stamp_ns:
                if reference_stamp_ns < self._active_path.stamp_ns:
                    self._fail_closed(
                        now_ns,
                        "scan_status_from_old_PCT_Path_generation",
                    )
                    return
                if self._cache_pending_scan_status(
                    message,
                    now_ns,
                    reference_stamp_ns,
                    signature,
                ):
                    self._fail_closed(
                        now_ns,
                        "scan_status_precedes_newer_active_path",
                    )
                return

        try:
            disposition = self._scan_sequence.observe(
                int(message.status_sequence),
                signature,
            )
        except ValueError as exc:
            self._fail_closed(now_ns, f"invalid_scan_status:{exc}")
            return
        if disposition in {
            SequenceDisposition.DUPLICATE,
            SequenceDisposition.STALE,
        }:
            return
        if disposition is SequenceDisposition.CONFLICT:
            self._fail_closed(now_ns, "conflicting_scan_status_sequence")
            return
        if trajectory is not None:
            self._scan_trajectory_identities.add(trajectory)
            if not trajectory.emergency_stop:
                self._stop_confirmed = False

        if event == ScanPlanningStatus.EVENT_PLANNING_FAILED:
            self._catch_up_scan_failures(
                int(message.consecutive_planning_failures),
                str(message.reason),
                now_ns,
            )
        elif event == ScanPlanningStatus.EVENT_PREDICTED_COLLISION:
            if self._core.state not in {
                NavigationState.IDLE,
                NavigationState.GOAL_REACHED,
            }:
                self._arm_stop_confirmation()
                if message.global_replan_recommended:
                    decision = self._core.report_predicted_collision(
                        nanoseconds_to_seconds(now_ns),
                        reason="scan_predicted_collision",
                    )
                else:
                    self._latch_recovery(
                        "scan_predicted_collision_local_stop"
                    )
                    decision = self._core.report_emergency_stop(
                        nanoseconds_to_seconds(now_ns),
                        reason="scan_predicted_collision_local_stop",
                        request_global_replan=False,
                    )
                self._apply_decision(decision, now_ns)
        elif event == ScanPlanningStatus.EVENT_EMERGENCY_STOP:
            if trajectory is not None:
                if not trajectory.emergency_stop:
                    self._fail_closed(
                        now_ns,
                        "scan_emergency_identity_not_emergency",
                    )
                    return
                self._pending_emergency_identity = trajectory
            if self._core.state not in {
                NavigationState.IDLE,
                NavigationState.GOAL_REACHED,
            } and not self._decision().global_replan_requested:
                self._arm_stop_confirmation()
                if not message.global_replan_recommended:
                    self._latch_recovery("scan_emergency_stop")
                self._apply_decision(
                    self._core.report_emergency_stop(
                        nanoseconds_to_seconds(now_ns),
                        reason="scan_emergency_stop",
                        request_global_replan=(
                            message.global_replan_recommended
                        ),
                    ),
                    now_ns,
                )
        elif event in {
            ScanPlanningStatus.EVENT_TRAJECTORY_PUBLISHED,
            ScanPlanningStatus.EVENT_RECOVERED,
        }:
            if trajectory is None or message.stop_required:
                self._fail_closed(
                    now_ns,
                    "scan_tracking_event_without_safe_trajectory",
                )
                return
            self._scan_failure_count = 0
            self._try_report_controller_tracking(now_ns)
        elif event == ScanPlanningStatus.EVENT_GOAL_HOLD:
            if trajectory is None or not trajectory.is_final:
                self._fail_closed(now_ns, "scan_goal_hold_missing_final")
                return
            self._scan_failure_count = 0
            self._try_report_controller_tracking(now_ns)
        elif event == ScanPlanningStatus.EVENT_REFERENCE_CLEARED:
            removed_pending = (
                self._pending_path_evidence_count()
                if reference_stamp_ns <= 0
                else self._discard_pending_path_evidence_at_or_before(
                    reference_stamp_ns
                )
            )
            if reference_stamp_ns <= 0:
                self._clear_pending_path_evidence()
            if removed_pending:
                self._fail_closed(
                    now_ns,
                    "pending_path_evidence_invalidated_by_scan_clear",
                )
                return
            if self._core.state in {
                NavigationState.LOCAL_PLANNING,
                NavigationState.TRACKING,
            }:
                self._arm_stop_confirmation()
                self._latch_recovery("scan_reference_cleared")
                self._apply_decision(
                    self._core.report_emergency_stop(
                        nanoseconds_to_seconds(now_ns),
                        reason="scan_reference_cleared",
                        request_global_replan=False,
                    ),
                    now_ns,
                )
        elif event in {
            ScanPlanningStatus.EVENT_STAIR_INHIBITED,
            ScanPlanningStatus.EVENT_STAIR_RESUME_WAITING,
        }:
            # 楼梯段由已认证 PCT Path 绑定的 root-lock 执行；supervisor 只
            # 保留 stop advisory，不制造无意义 PCT replan。
            if not message.stop_required or message.global_replan_recommended:
                self._fail_closed(now_ns, "stair_inhibit_policy_bits_invalid")
                return
            stair_reason = str(message.reason)
            if stair_reason in KNOWN_SCAN_STAIR_FREEZE_FAULTS:
                # 已知冻结协议故障是合法的类型化 fail-closed 报告，不是普通
                # 楼梯 ACK；保留精确 token 并锁存，后续 ACK 不得覆盖。
                self._fail_closed(now_ns, stair_reason)
                return
            decision = self._decision()
            if (
                not self._protocol_poisoned
                and not decision.global_replan_requested
                and not decision.global_replan_in_flight
                and self._core.state in {
                    NavigationState.LOCAL_PLANNING,
                    NavigationState.TRACKING,
                    NavigationState.EMERGENCY_STOP,
                }
            ):
                self._arm_stop_confirmation()
                self._latch_recovery(stair_reason)
                self._apply_decision(
                    self._core.report_emergency_stop(
                        nanoseconds_to_seconds(now_ns),
                        reason=stair_reason,
                        request_global_replan=False,
                    ),
                    now_ns,
                )
        self._evaluate_stop_confirmation(now_ns)
        self._publish_status(now_ns)

    def _validate_scan_status_policy(
        self,
        message: ScanPlanningStatus,
    ) -> None:
        """验证 SCAN typed event 的 state 和安全位没有自相矛盾。"""

        event = int(message.event)
        expected = {
            ScanPlanningStatus.EVENT_INITIAL: (
                ScanPlanningStatus.STATE_WAITING_FOR_REFERENCE,
                True,
            ),
            ScanPlanningStatus.EVENT_REFERENCE_ACCEPTED: (
                ScanPlanningStatus.STATE_PLANNING,
                True,
            ),
            ScanPlanningStatus.EVENT_REFERENCE_CLEARED: (
                ScanPlanningStatus.STATE_WAITING_FOR_REFERENCE,
                True,
            ),
            ScanPlanningStatus.EVENT_TRAJECTORY_PUBLISHED: (
                ScanPlanningStatus.STATE_TRACKING,
                False,
            ),
            ScanPlanningStatus.EVENT_PLANNING_FAILED: (
                ScanPlanningStatus.STATE_PLANNING,
                int(message.consecutive_planning_failures)
                >= self._core_config.max_consecutive_scan_failures,
            ),
            ScanPlanningStatus.EVENT_PREDICTED_COLLISION: (
                ScanPlanningStatus.STATE_EMERGENCY_STOP,
                True,
            ),
            ScanPlanningStatus.EVENT_EMERGENCY_STOP: (
                ScanPlanningStatus.STATE_EMERGENCY_STOP,
                True,
            ),
            ScanPlanningStatus.EVENT_RECOVERED: (
                ScanPlanningStatus.STATE_TRACKING,
                False,
            ),
            ScanPlanningStatus.EVENT_GOAL_HOLD: (
                ScanPlanningStatus.STATE_GOAL_HOLD,
                True,
            ),
            ScanPlanningStatus.EVENT_STAIR_INHIBITED: (
                ScanPlanningStatus.STATE_STAIR_INHIBITED,
                True,
            ),
            ScanPlanningStatus.EVENT_STAIR_RESUME_WAITING: (
                ScanPlanningStatus.STATE_STAIR_INHIBITED,
                True,
            ),
        }.get(event)
        if expected is None:
            raise ValueError("未知 SCAN event")
        expected_state, expected_stop = expected
        if int(message.state) != expected_state:
            raise ValueError("SCAN event/state 不匹配")
        if bool(message.stop_required) != expected_stop:
            raise ValueError("SCAN event/stop_required 不匹配")
        if (
            message.global_replan_recommended
            and not message.stop_required
        ):
            raise ValueError("请求全局重规划时必须同时要求停车")
        if event == ScanPlanningStatus.EVENT_PLANNING_FAILED:
            threshold_reached = (
                int(message.consecutive_planning_failures)
                >= self._core_config.max_consecutive_scan_failures
            )
            if bool(message.global_replan_recommended) != threshold_reached:
                raise ValueError("SCAN failure 阈值与 global replan 位不一致")
        if event in {
            ScanPlanningStatus.EVENT_INITIAL,
            ScanPlanningStatus.EVENT_REFERENCE_ACCEPTED,
            ScanPlanningStatus.EVENT_REFERENCE_CLEARED,
            ScanPlanningStatus.EVENT_TRAJECTORY_PUBLISHED,
            ScanPlanningStatus.EVENT_RECOVERED,
            ScanPlanningStatus.EVENT_GOAL_HOLD,
            ScanPlanningStatus.EVENT_STAIR_INHIBITED,
            ScanPlanningStatus.EVENT_STAIR_RESUME_WAITING,
        } and message.global_replan_recommended:
            raise ValueError("该 SCAN event 不允许请求全局重规划")
        if event == ScanPlanningStatus.EVENT_STAIR_INHIBITED:
            valid_stair_reasons = {
                SCAN_STAIR_EXECUTION_INHIBITED_REASON,
                *KNOWN_SCAN_STAIR_FREEZE_FAULTS,
            }
            if str(message.reason) not in valid_stair_reasons:
                raise ValueError(
                    "SCAN stair event/reason 合同不匹配："
                    "expected=execution ACK or known freeze fault, "
                    f"actual={message.reason}"
                )
        elif (
            event == ScanPlanningStatus.EVENT_STAIR_RESUME_WAITING
            and str(message.reason) != SCAN_STAIR_RESUME_WAITING_REASON
        ):
            raise ValueError(
                "SCAN stair event/reason 合同不匹配："
                f"expected={SCAN_STAIR_RESUME_WAITING_REASON}, "
                f"actual={message.reason}"
            )

    def _scan_status_signature(self, message: ScanPlanningStatus) -> Hashable:
        return (
            stamp_to_nanoseconds(message.header.stamp),
            int(message.event),
            int(message.state),
            self._optional_stamp_ns(message.reference_path_stamp),
            bool(message.trajectory_present),
            self._optional_stamp_ns(message.bspline_header_stamp),
            self._optional_stamp_ns(message.trajectory_start_time),
            int(message.trajectory_id),
            bool(message.trajectory_is_final),
            bool(message.trajectory_emergency_stop),
            int(message.consecutive_planning_failures),
            bool(message.stop_required),
            bool(message.global_replan_recommended),
            str(message.reason),
        )

    def _catch_up_scan_failures(
        self,
        reported_count: int,
        reason: str,
        now_ns: int,
    ) -> None:
        if reported_count < 1:
            self._fail_closed(now_ns, "scan_failure_count_must_be_positive")
            return
        bounded_count = min(
            reported_count,
            self._core_config.max_consecutive_scan_failures,
        )
        if bounded_count < self._scan_failure_count:
            self._protocol_warning("忽略倒退的 SCAN failure counter")
            return
        missing = bounded_count - self._scan_failure_count
        for _ in range(missing):
            if self._core.state not in {
                NavigationState.LOCAL_PLANNING,
                NavigationState.TRACKING,
            }:
                break
            decision = self._core.report_scan_failure(
                nanoseconds_to_seconds(now_ns),
                reason=str(reason).strip() or "scan_planning_failed",
            )
            self._last_decision = decision
        self._scan_failure_count = bounded_count
        if self._decision().global_replan_requested:
            self._arm_stop_confirmation()
        self._apply_decision(self._decision(), now_ns)

    def _controller_status_callback(self, message: ControllerStatus) -> None:
        now_ns = self._clock_now_ns()
        if now_ns is None:
            return
        try:
            self._message_stamp_ns(message, now_ns)
            if _normalize_frame(message.header.frame_id) != self._world_frame:
                raise ValueError("ControllerStatus world frame 不匹配")
            signature = self._controller_status_signature(message)
            disposition = self._controller_sequence.observe(
                int(message.status_sequence),
                signature,
            )
            identity = _trajectory_identity_from_controller(message)
        except ValueError as exc:
            self._fail_closed(now_ns, f"invalid_controller_status:{exc}")
            return
        if disposition in {
            SequenceDisposition.DUPLICATE,
            SequenceDisposition.STALE,
        }:
            return
        if disposition is SequenceDisposition.CONFLICT:
            self._fail_closed(now_ns, "conflicting_controller_status_sequence")
            return
        try:
            self._validate_controller_status(message, identity)
        except ValueError as exc:
            self._fail_closed(now_ns, f"controller_status_invariant:{exc}")
            return
        self._last_controller_acceptance_sequence = int(
            message.acceptance_sequence
        )
        self._controller_snapshot = _ControllerSnapshot(
            identity=identity,
            event=int(message.event),
            state=int(message.state),
            accepted=bool(message.accepted),
            trajectory_valid=bool(message.trajectory_valid),
            status_sequence=int(message.status_sequence),
            acceptance_sequence=int(message.acceptance_sequence),
        )
        if (
            int(message.event) == ControllerStatus.EVENT_ACCEPTED
            and identity is not None
            and not identity.emergency_stop
        ):
            self._stop_confirmed = False
        if (
            self._controller_snapshot.trajectory_valid
            and self._controller_snapshot.state
            in {
                ControllerStatus.STATE_TRACKING,
                ControllerStatus.STATE_ALIGNING_YAW,
            }
        ):
            self._stop_confirmed = False
        if (
            self._controller_finished_nonfinal_segment(
                self._controller_snapshot
            )
            and self._core.state
            in {NavigationState.LOCAL_PLANNING, NavigationState.TRACKING}
        ):
            self._reported_tracking_identity = None
            self._apply_decision(
                self._core.report_local_trajectory_finished(
                    nanoseconds_to_seconds(now_ns)
                ),
                now_ns,
            )
        elif (
            self._controller_requires_stop(self._controller_snapshot)
            and self._core.state
            in {NavigationState.LOCAL_PLANNING, NavigationState.TRACKING}
        ):
            self._arm_stop_confirmation()
            self._latch_recovery("controller_safe_stop")
            self._last_decision = self._core.report_emergency_stop(
                nanoseconds_to_seconds(now_ns),
                reason="controller_safe_stop",
                request_global_replan=False,
            )
        self._try_report_controller_tracking(now_ns)
        self._evaluate_stop_confirmation(now_ns)
        if (
            identity is not None
            and identity == self._reported_tracking_identity
            and identity.is_final
            and int(message.state) == ControllerStatus.STATE_GOAL_REACHED
            and bool(message.trajectory_valid)
            and self._core.state is NavigationState.TRACKING
        ):
            self._apply_decision(
                self._core.report_goal_reached(nanoseconds_to_seconds(now_ns)),
                now_ns,
            )
            return
        self._publish_status(now_ns)

    @staticmethod
    def _controller_requires_stop(snapshot: _ControllerSnapshot) -> bool:
        """返回 controller 是否报告了不能继续转发运动命令的状态。"""

        # controller 接收新 B-spline 时会先发布 EVENT_ACCEPTED，再在下一次
        # 控制 tick 发布 TRACKING。前一条消息仍携带接收前的 WAITING 状态，
        # 但 accepted+valid+normal identity 已证明新轨迹完成接管；这里只继续
        # 保持 LOCAL_PLANNING，等待后续 TRACKING 三元组，不能制造一次假的
        # EMERGENCY_STOP。若后续状态缺失，既有轨迹/输入超时仍会安全停车。
        if (
            snapshot.event == ControllerStatus.EVENT_ACCEPTED
            and snapshot.accepted
            and snapshot.trajectory_valid
            and snapshot.identity is not None
            and not snapshot.identity.emergency_stop
        ):
            return False

        return bool(
            snapshot.state
            in {
                ControllerStatus.STATE_WAITING_FOR_TRAJECTORY,
                ControllerStatus.STATE_WAITING_FOR_REFERENCE_PATH,
                ControllerStatus.STATE_WAITING_FOR_ODOMETRY,
                ControllerStatus.STATE_WAITING_FOR_CLOUD,
                ControllerStatus.STATE_TRAJECTORY_TIMEOUT,
                ControllerStatus.STATE_ODOMETRY_TIMEOUT,
                ControllerStatus.STATE_CLOUD_TIMEOUT,
                ControllerStatus.STATE_INVALID_CLOCK,
                ControllerStatus.STATE_EMERGENCY_STOP,
            }
            or (
                snapshot.event == ControllerStatus.EVENT_INVALIDATED
                and not snapshot.trajectory_valid
            )
        )

    @staticmethod
    def _controller_finished_nonfinal_segment(
        snapshot: _ControllerSnapshot,
    ) -> bool:
        """识别滚动规划中已安全结束、等待替换的非最终局部轨迹。"""

        return bool(
            snapshot.state == ControllerStatus.STATE_TRAJECTORY_FINISHED
            and snapshot.accepted
            and snapshot.trajectory_valid
            and snapshot.identity is not None
            and not snapshot.identity.is_final
            and not snapshot.identity.emergency_stop
        )

    def _validate_controller_status(
        self,
        message: ControllerStatus,
        identity: TrajectoryIdentity | None,
    ) -> None:
        """检查 acceptance 序列和 event/state/valid 的结构化不变量。"""

        acceptance = int(message.acceptance_sequence)
        first_observation = self._controller_sequence.latest == int(
            message.status_sequence
        ) and self._controller_snapshot is None
        if not first_observation:
            if acceptance < self._last_controller_acceptance_sequence:
                raise ValueError("acceptance_sequence 倒退")
        if bool(message.trajectory_valid) and not bool(message.accepted):
            raise ValueError("trajectory_valid 要求 accepted=true")
        if bool(message.accepted) != (identity is not None):
            raise ValueError("accepted 与轨迹身份存在性不一致")
        if identity is not None and identity.trajectory_id <= 0:
            raise ValueError("已接受 traj_id 必须为正数")
        event = int(message.event)
        if event not in {
            ControllerStatus.EVENT_INITIAL,
            ControllerStatus.EVENT_ACCEPTED,
            ControllerStatus.EVENT_REJECTED,
            ControllerStatus.EVENT_INVALIDATED,
            ControllerStatus.EVENT_STATE_CHANGED,
            ControllerStatus.EVENT_DUPLICATE,
        }:
            raise ValueError("未知 controller event")
        if int(message.state) not in {
            ControllerStatus.STATE_WAITING_FOR_TRAJECTORY,
            ControllerStatus.STATE_WAITING_FOR_REFERENCE_PATH,
            ControllerStatus.STATE_WAITING_FOR_ODOMETRY,
            ControllerStatus.STATE_WAITING_FOR_CLOUD,
            ControllerStatus.STATE_TRAJECTORY_TIMEOUT,
            ControllerStatus.STATE_ODOMETRY_TIMEOUT,
            ControllerStatus.STATE_CLOUD_TIMEOUT,
            ControllerStatus.STATE_INVALID_CLOCK,
            ControllerStatus.STATE_EMERGENCY_STOP,
            ControllerStatus.STATE_ALIGNING_YAW,
            ControllerStatus.STATE_TRACKING,
            ControllerStatus.STATE_TRAJECTORY_FINISHED,
            ControllerStatus.STATE_GOAL_REACHED,
        }:
            raise ValueError("未知 controller state")
        if bool(message.candidate_present) != (
            event == ControllerStatus.EVENT_REJECTED
        ):
            raise ValueError("candidate_present 只允许用于 REJECTED")
        if message.candidate_present:
            stamp_to_nanoseconds(message.candidate_reference_path_stamp)
            stamp_to_nanoseconds(message.candidate_bspline_header_stamp)
            stamp_to_nanoseconds(message.candidate_start_time)
            if int(message.candidate_traj_id) <= 0:
                raise ValueError("REJECTED candidate traj_id 必须为正数")
        elif not message.accepted:
            if any(
                self._optional_stamp_ns(stamp) != 0
                for stamp in (
                    message.reference_path_stamp,
                    message.bspline_header_stamp,
                    message.start_time,
                )
            ):
                raise ValueError("未接受状态不能携带已接受轨迹时间身份")
        if event == ControllerStatus.EVENT_ACCEPTED:
            if not message.accepted or not message.trajectory_valid:
                raise ValueError("ACCEPTED 事件必须保留有效轨迹")
            if (
                not first_observation
                and acceptance <= self._last_controller_acceptance_sequence
            ):
                raise ValueError("ACCEPTED 未严格推进 acceptance_sequence")
        elif event == ControllerStatus.EVENT_DUPLICATE:
            if not message.accepted or not message.trajectory_valid:
                raise ValueError("DUPLICATE 事件必须保留有效轨迹")
            if acceptance != self._last_controller_acceptance_sequence:
                raise ValueError("DUPLICATE 不能推进 acceptance_sequence")
        elif event == ControllerStatus.EVENT_INVALIDATED:
            if not message.accepted or message.trajectory_valid:
                raise ValueError("INVALIDATED 必须绑定已失效的已接受轨迹")
        if int(message.state) == ControllerStatus.STATE_GOAL_REACHED:
            if (
                identity is None
                or not message.trajectory_valid
                or not identity.is_final
                or identity.emergency_stop
            ):
                raise ValueError("GOAL_REACHED 必须绑定有效 normal final 轨迹")

    def _controller_status_signature(
        self,
        message: ControllerStatus,
    ) -> Hashable:
        return (
            stamp_to_nanoseconds(message.header.stamp),
            int(message.acceptance_sequence),
            int(message.event),
            self._optional_stamp_ns(message.reference_path_stamp),
            self._optional_stamp_ns(message.bspline_header_stamp),
            self._optional_stamp_ns(message.start_time),
            int(message.traj_id),
            bool(message.accepted),
            bool(message.trajectory_valid),
            bool(message.is_final),
            bool(message.emergency_stop),
            int(message.state),
            str(message.reason),
            bool(message.candidate_present),
            self._optional_stamp_ns(message.candidate_reference_path_stamp),
            self._optional_stamp_ns(message.candidate_bspline_header_stamp),
            self._optional_stamp_ns(message.candidate_start_time),
            int(message.candidate_traj_id),
        )

    def _try_report_controller_tracking(self, now_ns: int) -> None:
        snapshot = self._controller_snapshot
        if snapshot is None or snapshot.identity is None:
            return
        identity = snapshot.identity
        record = self._trajectories.get(identity)
        if record is None or identity not in self._scan_trajectory_identities:
            return
        if identity.emergency_stop:
            return
        if (
            not snapshot.accepted
            or not snapshot.trajectory_valid
            or snapshot.state
            not in {
                ControllerStatus.STATE_TRACKING,
                ControllerStatus.STATE_ALIGNING_YAW,
                ControllerStatus.STATE_GOAL_REACHED,
            }
        ):
            return
        if self._active_path is None or (
            identity.reference_path_stamp_ns != self._active_path.stamp_ns
        ):
            return
        soft_valid_until_ns = max(
            record.valid_until_ns + self._trajectory_expiry_grace_ns,
            identity.bspline_header_stamp_ns + self._bspline_timeout_ns,
        )
        valid_until_ns = soft_valid_until_ns
        if identity.is_final or (
            snapshot.state == ControllerStatus.STATE_ALIGNING_YAW
        ):
            # final hold 与已观察到的原地航向对齐可使用 controller 的固定
            # 冻结预算。上界来自不可变轨迹身份，状态心跳不能滚动累加。
            valid_until_ns += self._max_yaw_alignment_freeze_ns
        if valid_until_ns < now_ns:
            return
        if self._reported_tracking_identity == identity:
            try:
                decision = self._core.extend_scan_trajectory_validity(
                    nanoseconds_to_seconds(now_ns),
                    valid_until_s=nanoseconds_to_seconds(valid_until_ns),
                )
            except (RuntimeError, ValueError):
                return
            self._last_decision = decision
            self._try_clear_recoverable_emergency(now_ns)
            return
        if self._core.state not in {
            NavigationState.LOCAL_PLANNING,
            NavigationState.TRACKING,
            NavigationState.EMERGENCY_STOP,
        }:
            return
        try:
            decision = self._core.report_scan_success(
                nanoseconds_to_seconds(now_ns),
                valid_until_s=nanoseconds_to_seconds(valid_until_ns),
            )
        except RuntimeError:
            return
        self._reported_tracking_identity = identity
        self._scan_failure_count = 0
        self._apply_decision(decision, now_ns)
        self._try_clear_recoverable_emergency(now_ns)

    def _evaluate_stop_confirmation(self, now_ns: int) -> None:
        snapshot = self._controller_snapshot
        if snapshot is None:
            return
        if (
            snapshot.status_sequence
            <= self._stop_confirmation_after_sequence
            and not self._stop_confirmed
        ):
            return
        confirmed = False
        if (
            self._pending_emergency_identity is not None
            and snapshot is not None
        ):
            confirmed = bool(
                snapshot.identity == self._pending_emergency_identity
                and snapshot.accepted
                and snapshot.identity is not None
                and snapshot.identity.emergency_stop
                and snapshot.state == ControllerStatus.STATE_EMERGENCY_STOP
            )
        if not confirmed and snapshot is not None:
            identity = snapshot.identity
            confirmed = bool(
                snapshot.event == ControllerStatus.EVENT_INVALIDATED
                and not snapshot.trajectory_valid
                and identity is not None
                and self._latest_tombstone_stamp_ns
                > identity.reference_path_stamp_ns
            )
        if not confirmed and self._controller_is_stopped():
            confirmed = True
        if confirmed:
            self._stop_confirmed = True
            self._process_replan(now_ns)

    def _controller_is_stopped(self) -> bool:
        snapshot = self._controller_snapshot
        if snapshot is None:
            return False
        stopped_states = {
            ControllerStatus.STATE_WAITING_FOR_TRAJECTORY,
            ControllerStatus.STATE_WAITING_FOR_REFERENCE_PATH,
            ControllerStatus.STATE_WAITING_FOR_ODOMETRY,
            ControllerStatus.STATE_WAITING_FOR_CLOUD,
            ControllerStatus.STATE_TRAJECTORY_TIMEOUT,
            ControllerStatus.STATE_ODOMETRY_TIMEOUT,
            ControllerStatus.STATE_CLOUD_TIMEOUT,
            ControllerStatus.STATE_INVALID_CLOCK,
            ControllerStatus.STATE_EMERGENCY_STOP,
            ControllerStatus.STATE_TRAJECTORY_FINISHED,
            ControllerStatus.STATE_GOAL_REACHED,
        }
        if not snapshot.accepted:
            return snapshot.state in stopped_states
        return bool(
            not snapshot.trajectory_valid
            or snapshot.state in stopped_states
        )

    def _arm_stop_confirmation(self) -> None:
        """为当前 fault 绑定新的 controller 停车证据。"""

        snapshot = self._controller_snapshot
        if self._stop_confirmed and self._controller_is_stopped():
            return
        if (
            snapshot is not None
            and snapshot.acceptance_sequence == 0
            and self._controller_is_stopped()
        ):
            # acceptance_sequence=0 证明该 controller 进程从未接管轨迹；
            # 其余预 fault 停车快照都不能替代 fault 后的新证据。
            self._stop_confirmed = True
            return
        self._stop_confirmed = False
        self._stop_confirmation_after_sequence = (
            self._controller_sequence.latest
        )

    def _latch_recovery(self, reason: str) -> None:
        """锁存恢复屏障，要求 fault 后出现当前 Path 的新鲜执行 triple。"""

        self._recovery_block_reason = str(reason)
        self._recovery_after_controller_sequence = (
            self._controller_sequence.latest
        )
        self._recovery_blocked_identity = self._reported_tracking_identity
        self._reported_tracking_identity = None

    def _try_clear_recoverable_emergency(self, now_ns: int) -> None:
        decision = self._decision()
        if (
            self._protocol_poisoned
            or
            decision.state is not NavigationState.EMERGENCY_STOP
            or decision.global_replan_requested
            or decision.stale_inputs
        ):
            return
        if not self._recovery_block_reason:
            return
        snapshot = self._controller_snapshot
        identity = self._reported_tracking_identity
        if (
            self._active_path is None
            or snapshot is None
            or identity is None
            # 故障发生前已在执行的轨迹即使随后发布新的 TRACKING 或
            # ALIGNING_YAW 状态序号，也不能证明故障已经恢复。楼梯冻结
            # 期间旧 B-spline 仍可能因自身状态机切换而产生这类心跳；必须
            # 等 SCAN 在解除冻结后发布一条新身份轨迹，才能清除急停。
            or identity == self._recovery_blocked_identity
            or snapshot.status_sequence
            <= self._recovery_after_controller_sequence
            or snapshot.identity != identity
            or not snapshot.accepted
            or not snapshot.trajectory_valid
            or snapshot.state
            not in {
                ControllerStatus.STATE_TRACKING,
                ControllerStatus.STATE_ALIGNING_YAW,
            }
            or identity.reference_path_stamp_ns
            != self._active_path.stamp_ns
            or identity not in self._scan_trajectory_identities
            or identity not in self._trajectories
        ):
            return
        try:
            cleared = self._core.clear_emergency(
                nanoseconds_to_seconds(now_ns)
            )
        except RuntimeError:
            return
        self._recovery_block_reason = ""
        self._recovery_after_controller_sequence = -1
        self._recovery_blocked_identity = None
        self._apply_decision(cleared, now_ns)

    def _begin_global_planning_deadline(
        self,
        now_ns: int,
        *,
        phase: str,
    ) -> None:
        """为初始规划或已 ACK 的故障重规划建立结果截止时间。"""

        if phase not in {"initial", "replan"}:
            raise ValueError("全局规划阶段必须是 initial 或 replan")
        self._global_planning_phase = phase
        self._global_planning_deadline_ns = (
            now_ns + self._global_planning_timeout_ns
        )

    def _clear_global_planning_deadline(self) -> None:
        """清除已经由匹配 Path/result 完成的全局规划等待窗口。"""

        self._global_planning_deadline_ns = 0
        self._global_planning_phase = ""

    def _expire_global_planning_deadline(self, now_ns: int) -> bool:
        """截止时仍未对账到 PCT 结果与 Path 时锁存终止停车。"""

        if (
            self._global_planning_deadline_ns <= 0
            or now_ns < self._global_planning_deadline_ns
            or self._core.state is not NavigationState.GLOBAL_PLANNING
        ):
            return False
        reason = (
            "global_replan_result_timeout"
            if self._global_planning_phase == "replan"
            else "initial_global_planning_timeout"
        )
        self._terminate_global_planning(now_ns, reason)
        return True

    def _terminate_global_planning(self, now_ns: int, reason: str) -> None:
        """终止规划循环并禁止同一 goal 继续自动恢复或重试。"""

        if self._global_planning_terminal_reason:
            return
        terminal_reason = str(reason).strip() or "global_planning_terminal"
        self._clear_global_planning_deadline()
        self._global_planning_terminal_reason = terminal_reason
        self._adapter_fault_reason = terminal_reason
        self._protocol_poisoned = True
        self._latch_recovery("global_planning_terminal")
        transaction = self._replan
        if transaction is not None and not transaction.terminal_error:
            transaction.mark_terminal_error(terminal_reason)
        decision = self._decision()
        try:
            if (
                decision.global_replan_requested
                or decision.global_replan_in_flight
            ):
                self._last_decision = (
                    self._core.report_global_replan_transport_failed(
                        nanoseconds_to_seconds(now_ns),
                        reason=terminal_reason,
                    )
                )
            else:
                self._last_decision = self._core.report_emergency_stop(
                    nanoseconds_to_seconds(now_ns),
                    reason=terminal_reason,
                    request_global_replan=False,
                )
        except (RuntimeError, ValueError):
            self._last_decision = decision
        self._arm_stop_confirmation()
        self.get_logger().error(
            f"全局规划已终止并锁存停车：{terminal_reason}"
        )
        self._publish_status(now_ns)

    def _timer_callback(self) -> None:
        now_ns = self._clock_now_ns()
        if now_ns is None:
            return
        if self._expire_pending_path_evidence(now_ns):
            return
        if self._expire_global_planning_deadline(now_ns):
            return
        previous_state = self._core.state
        try:
            decision = self._core.tick(nanoseconds_to_seconds(now_ns))
        except ValueError as exc:
            self._fail_closed(now_ns, f"supervisor_clock_error:{exc}")
            return
        if (
            previous_state is NavigationState.TRACKING
            and decision.state is NavigationState.EMERGENCY_STOP
            and not decision.global_replan_requested
        ):
            self._arm_stop_confirmation()
            self._latch_recovery("supervisor_input_timeout")
        self._last_decision = decision
        self._try_clear_recoverable_emergency(now_ns)
        self._process_replan(now_ns)
        self._publish_status(now_ns)

    def _process_replan(self, now_ns: int) -> None:
        decision = self._decision()
        if self._global_planning_terminal_reason:
            return
        if not decision.global_replan_requested:
            return
        if (
            self._global_replan_cycle_count
            >= self._max_global_replan_cycles
        ):
            self._terminate_global_planning(
                now_ns,
                "global_replan_cycles_exhausted",
            )
            return
        if not self._stop_confirmed:
            return
        if (
            self._goal is None
            or self._goal_message is None
            or self._pct_active_path_stamp_ns <= 0
        ):
            self._adapter_fault_reason = "replan_missing_goal_or_path_snapshot"
            self._publish_status(now_ns)
            return
        self._ensure_replan_transaction(decision, now_ns)
        transaction = self._replan
        if transaction is None or transaction.terminal:
            return
        if transaction.response_expired(now_ns):
            old_future = self._replan_future
            if old_future is not None and hasattr(old_future, "cancel"):
                old_future.cancel()
            transaction.mark_retryable_failure()
            self._replan_future = None
            self._replan_attempt_token += 1
        if (
            transaction.attempts == 0
            and now_ns - transaction.request_stamp_ns
            >= self._replan_service_wait_ns
            and not self._pct_client.service_is_ready()
        ):
            self._terminate_replan(
                now_ns,
                "pct_replan_service_unavailable",
            )
            return
        if not transaction.can_send(now_ns):
            if transaction.exhausted and not transaction.in_flight:
                self._terminate_replan(
                    now_ns,
                    "pct_replan_attempts_exhausted",
                )
            return
        if not self._pct_client.service_is_ready():
            # 服务在已发送过请求后消失，也必须消耗有界 attempt，不能永久
            # 悬挂在同一 pending request 上。
            if transaction.attempts > 0:
                transaction.mark_sent(now_ns)
                transaction.mark_retryable_failure()
                if transaction.exhausted:
                    self._terminate_replan(
                        now_ns,
                        "pct_replan_service_lost",
                    )
            return
        assert self._replan_request is not None
        transaction.mark_sent(now_ns)
        self._replan_attempt_token += 1
        attempt_token = self._replan_attempt_token
        try:
            future = self._pct_client.call_async(self._replan_request)
        except Exception as exc:  # pragma: no cover - rclpy transport fault
            transaction.mark_retryable_failure()
            self._protocol_warning(f"PCT REPLAN call_async 失败：{exc}")
            return
        self._replan_future = future
        request_id = transaction.request_id
        epoch = transaction.epoch
        future.add_done_callback(
            lambda completed: self._replan_response_callback(
                completed,
                request_id=request_id,
                epoch=epoch,
                attempt_token=attempt_token,
            )
        )
        self._publish_status(now_ns)

    def _ensure_replan_transaction(
        self,
        decision: SupervisorDecision,
        now_ns: int,
    ) -> None:
        if (
            self._replan is not None
            and self._replan.core_request_id
            == decision.global_replan_request_id
        ):
            return
        self._next_wire_request_id += 1
        request_id = self._next_wire_request_id
        assert self._goal is not None
        assert self._goal_message is not None
        reason = str(decision.reason).strip() or "navigation_supervisor_replan"
        self._replan = ReplanTransaction(
            request_id=request_id,
            core_request_id=decision.global_replan_request_id,
            goal=self._goal,
            expected_path_stamp_ns=self._pct_active_path_stamp_ns,
            request_stamp_ns=now_ns,
            reason=reason,
            max_attempts=self._replan_max_attempts,
            retry_period_ns=self._replan_retry_period_ns,
            response_timeout_ns=self._replan_response_ns,
            epoch=self._epoch,
        )
        request = PCTPlanningCommand.Request()
        request.header.stamp = _time_from_nanoseconds(now_ns)
        request.header.frame_id = self._world_frame
        request.command = PCTPlanningCommand.Request.COMMAND_REPLAN
        request.goal_id = self._goal.goal_id
        request.request_id = request_id
        request.goal = deepcopy(self._goal_message)
        request.expected_path_stamp = _time_from_nanoseconds(
            self._pct_active_path_stamp_ns
        )
        request.reason = reason
        self._replan_request = request
        self._replan_future = None
        self._replan_attempt_token += 1

    def _replan_response_callback(
        self,
        future: object,
        *,
        request_id: int,
        epoch: int,
        attempt_token: int,
    ) -> None:
        now_ns = self._clock_now_ns()
        transaction = self._replan
        if (
            now_ns is None
            or transaction is None
            or transaction.request_id != request_id
            or transaction.epoch != epoch
            or transaction.terminal
        ):
            return
        current_attempt = self._replan_attempt_token == attempt_token
        try:
            response = future.result()
        except Exception as exc:
            if not current_attempt:
                return
            if transaction.in_flight:
                transaction.mark_retryable_failure()
            self._protocol_warning(f"PCT REPLAN 响应失败：{exc}")
            return
        if response is None:
            if not current_attempt:
                return
            if transaction.in_flight:
                transaction.mark_retryable_failure()
            self._protocol_warning("PCT REPLAN 返回空响应，等待有界重试")
            return
        accepted_disposition = int(response.disposition) in {
            PCTPlanningCommand.Response.DISPOSITION_ACCEPTED,
            PCTPlanningCommand.Response.DISPOSITION_DUPLICATE,
        }
        if not current_attempt and not accepted_disposition:
            return
        if int(response.goal_id) != transaction.goal.goal_id or (
            int(response.request_id) != transaction.request_id
        ):
            transaction.mark_terminal_error("pct_replan_response_id_mismatch")
        elif accepted_disposition:
            try:
                if not response.has_active_goal:
                    raise ValueError("ACK 缺少活动目标快照")
                response_goal = _goal_identity(
                    response.active_goal,
                    int(response.goal_id),
                )
                tombstone_ns = stamp_to_nanoseconds(response.tombstone_stamp)
                active_stamp_ns = stamp_to_nanoseconds(
                    response.active_path_stamp
                )
                if response_goal != transaction.goal:
                    raise ValueError("ACK 活动目标快照不一致")
                if tombstone_ns <= transaction.expected_path_stamp_ns:
                    raise ValueError("ACK tombstone 没有严格更新 Path 代际")
                if active_stamp_ns < tombstone_ns:
                    raise ValueError("ACK active_path_stamp 早于 tombstone")
            except ValueError as exc:
                transaction.mark_terminal_error(
                    f"invalid_pct_replan_ack:{exc}"
                )
            else:
                transaction.mark_acknowledged()
                self._pct_plan_id = int(response.plan_id)
                self._pct_active_path_stamp_ns = active_stamp_ns
                self._acknowledge_core_replan(now_ns)
                self._try_accept_global_path(active_stamp_ns, now_ns)
        else:
            transaction.mark_terminal_error(
                "pct_replan_rejected:"
                f"{int(response.disposition)}:{response.message}"
            )
        if transaction.terminal_error:
            self._terminate_replan(now_ns, transaction.terminal_error)
        self._publish_status(now_ns)

    def _terminate_replan(self, now_ns: int, reason: str) -> None:
        """锁存 transport terminal error，并同步终止 core phantom pending。"""

        transaction = self._replan
        if transaction is not None and not transaction.terminal_error:
            transaction.mark_terminal_error(reason)
        terminal_reason = (
            str(reason)
            if transaction is None
            else transaction.terminal_error
        )
        decision = self._decision()
        if (
            decision.global_replan_requested
            or decision.global_replan_in_flight
        ):
            try:
                self._last_decision = (
                    self._core.report_global_replan_transport_failed(
                        nanoseconds_to_seconds(now_ns),
                        reason=terminal_reason,
                    )
                )
            except RuntimeError:
                pass
        self._adapter_fault_reason = terminal_reason
        self._publish_status(now_ns)

    def _acknowledge_replan(self, now_ns: int) -> None:
        if self._replan is None:
            return
        if not self._replan.acknowledged:
            self._replan.mark_acknowledged()
        self._acknowledge_core_replan(now_ns)

    def _acknowledge_core_replan(self, now_ns: int) -> None:
        decision = self._decision()
        if decision.global_replan_requested and self._core.state in {
            NavigationState.GLOBAL_REPLAN,
            NavigationState.EMERGENCY_STOP,
        }:
            started = self._core.report_global_planning_started(
                nanoseconds_to_seconds(now_ns)
            )
            self._global_replan_cycle_count += 1
            self._begin_global_planning_deadline(now_ns, phase="replan")
            self._apply_decision(started, now_ns)

    def _fail_closed(self, now_ns: int, reason: str) -> None:
        self._adapter_fault_reason = str(reason)
        self._protocol_poisoned = True
        self._latch_recovery("protocol_fault")
        if self._core.state not in {
            NavigationState.IDLE,
            NavigationState.GOAL_REACHED,
        }:
            try:
                self._last_decision = self._core.report_emergency_stop(
                    nanoseconds_to_seconds(now_ns),
                    reason=str(reason),
                    request_global_replan=False,
                )
            except (RuntimeError, ValueError):
                pass
        self._arm_stop_confirmation()
        self.get_logger().error(f"navigation supervisor fail closed：{reason}")
        self._publish_status(now_ns)

    def _protocol_warning(self, text: str) -> None:
        self.get_logger().warning(str(text))

    def _apply_decision(
        self,
        decision: SupervisorDecision,
        now_ns: int,
    ) -> None:
        self._last_decision = decision
        self._process_replan(now_ns)
        self._publish_status(now_ns)

    def _decision(self) -> SupervisorDecision:
        if self._last_decision is not None:
            return self._last_decision
        # 只在尚无有效 ROS 时间的构造阶段使用；第一条状态不会在零时钟发布。
        return self._core.tick(0.0)

    def _publish_status(self, now_ns: int) -> None:
        decision = self._decision()
        transaction = self._replan
        reason = self._adapter_fault_reason or decision.reason
        signature = (
            self._epoch,
            decision.state_revision,
            0 if self._goal is None else self._goal.goal_id,
            decision.state,
            decision.allow_tracking_command,
            decision.force_zero_velocity,
            self._stop_confirmed,
            decision.global_replan_requested,
            decision.global_replan_in_flight,
            decision.global_replan_request_id,
            0 if transaction is None else transaction.request_id,
            0 if transaction is None else transaction.attempts,
            False if transaction is None else transaction.acknowledged,
            "" if transaction is None else transaction.terminal_error,
            self._pct_plan_id,
            self._pct_active_path_stamp_ns,
            decision.consecutive_scan_failures,
            decision.stale_inputs,
            reason,
        )
        heartbeat_due = bool(
            self._last_navigation_status_publish_ns <= 0
            or now_ns - self._last_navigation_status_publish_ns
            >= self._status_heartbeat_ns
        )
        if (
            signature == self._last_navigation_status_signature
            and not heartbeat_due
        ):
            return
        self._last_navigation_status_signature = signature
        self._last_navigation_status_publish_ns = now_ns
        self._navigation_status_sequence += 1
        message = NavigationStatus()
        message.header.stamp = _time_from_nanoseconds(now_ns)
        message.header.frame_id = self._world_frame
        message.status_sequence = self._navigation_status_sequence
        message.state_revision = decision.state_revision
        message.goal_id = 0 if self._goal is None else self._goal.goal_id
        message.state = self._navigation_state_code(decision.state)
        message.allow_tracking_command = decision.allow_tracking_command
        message.force_zero_velocity = decision.force_zero_velocity
        message.stop_confirmed = self._stop_confirmed
        message.global_replan_requested = decision.global_replan_requested
        message.global_replan_in_flight = decision.global_replan_in_flight
        message.global_replan_request_id = (
            decision.global_replan_request_id
        )
        message.pct_plan_id = self._pct_plan_id
        message.active_path_stamp = _time_from_nanoseconds(
            self._pct_active_path_stamp_ns
        )
        message.consecutive_scan_failures = (
            decision.consecutive_scan_failures
        )
        message.stale_inputs = list(decision.stale_inputs)
        message.reason = reason
        self._status_publisher.publish(message)

    @staticmethod
    def _navigation_state_code(state: NavigationState) -> int:
        return {
            NavigationState.IDLE: NavigationStatus.STATE_IDLE,
            NavigationState.GLOBAL_PLANNING: (
                NavigationStatus.STATE_GLOBAL_PLANNING
            ),
            NavigationState.LOCAL_PLANNING: (
                NavigationStatus.STATE_LOCAL_PLANNING
            ),
            NavigationState.TRACKING: NavigationStatus.STATE_TRACKING,
            NavigationState.GLOBAL_REPLAN: (
                NavigationStatus.STATE_GLOBAL_REPLAN
            ),
            NavigationState.EMERGENCY_STOP: (
                NavigationStatus.STATE_EMERGENCY_STOP
            ),
            NavigationState.GOAL_REACHED: NavigationStatus.STATE_GOAL_REACHED,
        }[state]

    @staticmethod
    def _optional_stamp_ns(stamp: Time) -> int:
        return stamp_to_nanoseconds(stamp, allow_zero=True)

    def _is_tombstone_retired_path_generation(self, stamp_ns: int) -> bool:
        """返回 Path 代际是否已被权威 tombstone 明确淘汰。"""

        if stamp_ns <= 0 or stamp_ns > self._latest_tombstone_stamp_ns:
            return False
        return bool(
            self._active_path is None
            or stamp_ns < self._active_path.stamp_ns
        )

    def _prune_caches(self) -> None:
        """限制诊断缓存，避免长 episode 随轨迹代际无界增长。"""

        for mapping in (
            self._path_records,
            self._path_signatures,
            self._pct_successes,
            self._pct_failures,
            self._pct_replan_ack_candidates,
        ):
            for key in sorted(mapping)[:-8]:
                del mapping[key]
        if len(self._trajectories) > 32:
            keep = set(list(self._trajectories)[-32:])
            self._trajectories = {
                identity: record
                for identity, record in self._trajectories.items()
                if identity in keep
            }
            self._scan_trajectory_identities.intersection_update(keep)


def main(args: list[str] | None = None) -> None:
    """运行 typed navigation supervisor，关闭时不遗留第二个 writer。"""

    rclpy.init(args=args)
    node: NavigationSupervisorNode | None = None
    try:
        node = NavigationSupervisorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # ros2 launch 关闭时可能在 spin() 返回后再次送达 SIGINT；清理阶段
        # 必须保持幂等，避免正常停止被记录成 exit -2 和 traceback。
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


__all__ = ["NavigationSupervisorNode", "main"]


if __name__ == "__main__":
    main()
