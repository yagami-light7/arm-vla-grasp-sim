"""navigation_supervisor ROS 2 节点的跨 topic 安全回归测试。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import permutations
import os
from typing import Callable

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path as PathMessage
import pytest
import rclpy
from scan_planner_msgs.msg import (
    Bspline,
    ControllerStatus,
    PCTPlanningStatus,
    ScanPlanningStatus,
)
from scan_planner_msgs.srv import PCTPlanningCommand
from sensor_msgs.msg import PointCloud2, PointField

from navigation_core import NavigationState
from navigation_supervisor.node import NavigationSupervisorNode


NANOSECONDS_PER_SECOND = 1_000_000_000
NOW_NS = 10 * NANOSECONDS_PER_SECOND
GOAL_STAMP_NS = 1 * NANOSECONDS_PER_SECOND
PATH_STAMP_NS = 4 * NANOSECONDS_PER_SECOND
BSPLINE_STAMP_NS = 9_600_000_000
START_STAMP_NS = 9_500_000_000


@dataclass
class _PublishedMessages:
    messages: list[object] = field(default_factory=list)

    def publish(self, message: object) -> None:
        self.messages.append(deepcopy(message))


@dataclass
class _NodeHarness:
    node: NavigationSupervisorNode
    clock_ns: list[int]
    published: _PublishedMessages


class _FakeFuture:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []
        self._response: object | None = None
        self._exception: Exception | None = None
        self.cancel_calls = 0

    def add_done_callback(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def cancel(self) -> bool:
        self.cancel_calls += 1
        return True

    def result(self) -> object:
        if self._exception is not None:
            raise self._exception
        if self._response is None:
            raise RuntimeError("future 尚未完成")
        return self._response

    def complete(self, response: object) -> None:
        self._response = response
        for callback in tuple(self._callbacks):
            callback(self)


class _NoneResultFuture:
    def result(self) -> None:
        return None


class _FakePCTClient:
    def __init__(self) -> None:
        self.ready = True
        self.requests: list[object] = []
        self.futures: list[_FakeFuture] = []

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request: object) -> _FakeFuture:
        future = _FakeFuture()
        self.requests.append(deepcopy(request))
        self.futures.append(future)
        return future


class _FakeClock:
    def __init__(self, now_ns: list[int]) -> None:
        self._now_ns = now_ns

    def now(self) -> object:
        return type("FakeNow", (), {"nanoseconds": self._now_ns[0]})()


@pytest.fixture(scope="module", autouse=True)
def _ros_context(tmp_path_factory: pytest.TempPathFactory):
    log_dir = tmp_path_factory.mktemp("ros_logs")
    os.environ["ROS_LOG_DIR"] = str(log_dir)
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init(
            args=["--ros-args", "-p", "use_sim_time:=true"],
        )
    yield
    if owns_context and rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _NodeHarness:
    node = NavigationSupervisorNode()
    clock_ns = [NOW_NS]
    published = _PublishedMessages()
    monkeypatch.setattr(node, "_clock_now_ns", lambda: clock_ns[0])
    monkeypatch.setattr(node, "_status_publisher", published)
    result = _NodeHarness(
        node=node,
        clock_ns=clock_ns,
        published=published,
    )
    yield result
    node.destroy_node()


def _time(value_ns: int) -> Time:
    return Time(
        sec=value_ns // NANOSECONDS_PER_SECOND,
        nanosec=value_ns % NANOSECONDS_PER_SECOND,
    )


def _goal(goal_stamp_ns: int = GOAL_STAMP_NS):
    from geometry_msgs.msg import PoseStamped

    message = PoseStamped()
    message.header.stamp = _time(goal_stamp_ns)
    message.header.frame_id = "world"
    message.pose.position.x = 6.0
    message.pose.position.y = -1.0
    message.pose.position.z = 0.3
    message.pose.orientation.w = 1.0
    return message


def _path(path_stamp_ns: int = PATH_STAMP_NS, *, empty: bool = False):
    from geometry_msgs.msg import PoseStamped

    message = PathMessage()
    message.header.stamp = _time(path_stamp_ns)
    message.header.frame_id = "world"
    if empty:
        return message
    for x_value in (0.0, 2.0):
        pose = PoseStamped()
        pose.header.stamp = _time(path_stamp_ns)
        pose.header.frame_id = "world"
        pose.pose.position.x = x_value
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


def _pct_status(
    *,
    path_stamp_ns: int = PATH_STAMP_NS,
    state: int = PCTPlanningStatus.SUCCEEDED,
    event_stamp_ns: int = 9_000_000_000,
    plan_id: int = 1,
    request_id: int = 1,
    command: int = PCTPlanningStatus.COMMAND_PLAN,
) -> PCTPlanningStatus:
    message = PCTPlanningStatus()
    message.header.stamp = _time(event_stamp_ns)
    message.header.frame_id = "world"
    message.plan_id = plan_id
    message.goal_id = 7
    message.request_id = request_id
    message.command = command
    message.state = state
    message.goal_stamp = _time(GOAL_STAMP_NS)
    message.path_stamp = _time(path_stamp_ns)
    message.has_active_goal = True
    message.active_goal = _goal()
    message.active_path_stamp = _time(path_stamp_ns)
    message.path_point_count = (
        2 if state == PCTPlanningStatus.SUCCEEDED else 0
    )
    message.message = "test"
    return message


def _odometry(stamp_ns: int) -> Odometry:
    message = Odometry()
    message.header.stamp = _time(stamp_ns)
    message.header.frame_id = "world"
    message.child_frame_id = "base_link"
    message.pose.pose.position.z = 0.3
    message.pose.pose.orientation.w = 1.0
    return message


def _point_cloud(stamp_ns: int) -> PointCloud2:
    message = PointCloud2()
    message.header.stamp = _time(stamp_ns)
    message.header.frame_id = "world"
    message.height = 1
    message.width = 1
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = 12
    message.data = bytes(12)
    message.is_dense = True
    return message


def _canonical_empty_point_cloud(stamp_ns: int) -> PointCloud2:
    """构造 bridge/controller 认证的 canonical xyz32 空点云。"""

    message = _point_cloud(stamp_ns)
    message.width = 0
    message.row_step = 0
    message.data = bytes()
    return message


def _bspline(
    *,
    path_stamp_ns: int = PATH_STAMP_NS,
    trajectory_id: int = 11,
    is_final: bool = False,
    emergency_stop: bool = False,
) -> Bspline:
    message = Bspline()
    message.header.stamp = _time(BSPLINE_STAMP_NS)
    message.header.frame_id = "world"
    message.order = 3
    message.traj_id = trajectory_id
    message.start_time = _time(START_STAMP_NS)
    message.reference_path_stamp = _time(path_stamp_ns)
    message.is_final = is_final
    message.emergency_stop = emergency_stop
    message.knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    message.pos_pts = [Point(x=float(index)) for index in range(6)]
    return message


def _scan_trajectory_status(
    *,
    sequence: int = 1,
    path_stamp_ns: int = PATH_STAMP_NS,
    trajectory_id: int = 11,
    is_final: bool = False,
) -> ScanPlanningStatus:
    message = ScanPlanningStatus()
    message.header.stamp = _time(9_700_000_000)
    message.header.frame_id = "world"
    message.status_sequence = sequence
    message.event = ScanPlanningStatus.EVENT_TRAJECTORY_PUBLISHED
    message.state = ScanPlanningStatus.STATE_TRACKING
    message.reference_path_stamp = _time(path_stamp_ns)
    message.trajectory_present = True
    message.bspline_header_stamp = _time(BSPLINE_STAMP_NS)
    message.trajectory_start_time = _time(START_STAMP_NS)
    message.trajectory_id = trajectory_id
    message.trajectory_is_final = is_final
    message.trajectory_emergency_stop = False
    message.stop_required = False
    message.global_replan_recommended = False
    message.reason = "trajectory_published"
    return message


def _scan_emergency_status(
    *,
    sequence: int = 2,
    path_stamp_ns: int = PATH_STAMP_NS,
    trajectory_id: int = 12,
) -> ScanPlanningStatus:
    """构造与急停 B-spline 身份完全一致的 SCAN 状态。"""

    message = _scan_trajectory_status(
        sequence=sequence,
        path_stamp_ns=path_stamp_ns,
        trajectory_id=trajectory_id,
    )
    message.event = ScanPlanningStatus.EVENT_EMERGENCY_STOP
    message.state = ScanPlanningStatus.STATE_EMERGENCY_STOP
    message.trajectory_emergency_stop = True
    message.stop_required = True
    message.reason = "scan_stair_emergency_stop"
    return message


def _collision_status(sequence: int = 2) -> ScanPlanningStatus:
    message = ScanPlanningStatus()
    message.header.stamp = _time(9_800_000_000)
    message.header.frame_id = "world"
    message.status_sequence = sequence
    message.event = ScanPlanningStatus.EVENT_PREDICTED_COLLISION
    message.state = ScanPlanningStatus.STATE_EMERGENCY_STOP
    message.reference_path_stamp = _time(PATH_STAMP_NS)
    message.trajectory_present = False
    message.stop_required = True
    message.global_replan_recommended = True
    message.reason = "predicted_collision"
    return message


def _stair_status(
    *,
    sequence: int = 1,
    event: int = ScanPlanningStatus.EVENT_STAIR_INHIBITED,
    reason: str | None = None,
    path_stamp_ns: int = PATH_STAMP_NS,
) -> ScanPlanningStatus:
    message = ScanPlanningStatus()
    message.header.stamp = _time(9_800_000_000)
    message.header.frame_id = "world"
    message.status_sequence = sequence
    message.event = event
    message.state = ScanPlanningStatus.STATE_STAIR_INHIBITED
    message.reference_path_stamp = _time(path_stamp_ns)
    message.trajectory_present = False
    message.stop_required = True
    message.global_replan_recommended = False
    if reason is None:
        reason = (
            "scan_stair_execution_inhibited"
            if event == ScanPlanningStatus.EVENT_STAIR_INHIBITED
            else "scan_stair_resume_waiting"
        )
    message.reason = reason
    return message


def _controller_status(
    *,
    sequence: int = 1,
    acceptance_sequence: int = 1,
    event: int = ControllerStatus.EVENT_ACCEPTED,
    state: int = ControllerStatus.STATE_TRACKING,
    path_stamp_ns: int = PATH_STAMP_NS,
    trajectory_id: int = 11,
    trajectory_valid: bool = True,
    is_final: bool = False,
    emergency_stop: bool = False,
) -> ControllerStatus:
    message = ControllerStatus()
    message.header.stamp = _time(9_900_000_000)
    message.header.frame_id = "world"
    message.status_sequence = sequence
    message.acceptance_sequence = acceptance_sequence
    message.event = event
    message.reference_path_stamp = _time(path_stamp_ns)
    message.bspline_header_stamp = _time(BSPLINE_STAMP_NS)
    message.start_time = _time(START_STAMP_NS)
    message.traj_id = trajectory_id
    message.accepted = True
    message.trajectory_valid = trajectory_valid
    message.is_final = is_final
    message.emergency_stop = emergency_stop
    message.state = state
    message.reason = "test"
    return message


def _bootstrap_path(
    node: NavigationSupervisorNode,
    *,
    status_first: bool = True,
) -> None:
    callbacks = (
        (node._pct_status_callback, _pct_status()),
        (node._path_callback, _path()),
    )
    if not status_first:
        callbacks = tuple(reversed(callbacks))
    for callback, message in callbacks:
        callback(message)
    assert node._core.state is NavigationState.LOCAL_PLANNING


def _bootstrap_tracking(node: NavigationSupervisorNode) -> None:
    _bootstrap_path(node)
    node._odometry_callback(_odometry(9_950_000_000))
    node._point_cloud_callback(_point_cloud(9_950_000_000))
    node._bspline_callback(_bspline())
    node._scan_status_callback(_scan_trajectory_status())
    node._controller_status_callback(_controller_status())
    assert node._core.state is NavigationState.TRACKING


def _refresh_tracking_sensors_until(
    harness: _NodeHarness,
    target_ns: int,
) -> None:
    """以小于最短 freshness 门限的步长推进仿真传感器。"""

    while harness.clock_ns[0] < target_ns:
        harness.clock_ns[0] = min(
            harness.clock_ns[0] + 200_000_000,
            target_ns,
        )
        stamp_ns = harness.clock_ns[0] - 10_000_000
        harness.node._odometry_callback(_odometry(stamp_ns))
        harness.node._point_cloud_callback(_point_cloud(stamp_ns))


def _stopped_controller_status(sequence: int = 2) -> ControllerStatus:
    return _controller_status(
        sequence=sequence,
        event=ControllerStatus.EVENT_INVALIDATED,
        state=ControllerStatus.STATE_EMERGENCY_STOP,
        trajectory_valid=False,
    )


def _finished_nonfinal_controller_status(
    sequence: int = 2,
) -> ControllerStatus:
    return _controller_status(
        sequence=sequence,
        event=ControllerStatus.EVENT_STATE_CHANGED,
        state=ControllerStatus.STATE_TRAJECTORY_FINISHED,
    )


def _accepted_replan_response(
    node: NavigationSupervisorNode,
    *,
    plan_id: int = 2,
    tombstone_ns: int = 5_000_000_000,
    active_path_stamp_ns: int | None = None,
):
    assert node._replan is not None
    response = PCTPlanningCommand.Response()
    response.header.stamp = _time(11_000_000_000)
    response.header.frame_id = "world"
    response.disposition = PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
    response.plan_id = plan_id
    response.goal_id = node._replan.goal.goal_id
    response.request_id = node._replan.request_id
    response.tombstone_stamp = _time(tombstone_ns)
    response.has_active_goal = True
    response.active_goal = deepcopy(node._goal_message)
    response.active_path_stamp = _time(
        tombstone_ns
        if active_path_stamp_ns is None
        else active_path_stamp_ns
    )
    response.message = "accepted"
    return response


def test_initialization_has_no_cmd_vel_publisher(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    publishers = dict(
        node.get_publisher_names_and_types_by_node(
            node.get_name(),
            node.get_namespace(),
        )
    )

    assert node.get_name() == "navigation_supervisor"
    assert node.get_parameter("use_sim_time").value is True
    assert node._global_planning_timeout_ns == 15 * NANOSECONDS_PER_SECOND
    assert node._pending_path_evidence_timeout_ns == (
        2 * NANOSECONDS_PER_SECOND
    )
    assert node._bspline_timeout_ns == 1_500_000_000
    assert node._trajectory_expiry_grace_ns == 3 * NANOSECONDS_PER_SECOND
    assert node._max_yaw_alignment_freeze_ns == (
        6 * NANOSECONDS_PER_SECOND
    )
    assert node._max_global_replan_cycles == 3
    assert node._max_pending_path_evidence == 64
    assert node._status_heartbeat_ns == 100_000_000
    assert publishers["/navigation/status"] == [
        "scan_planner_msgs/msg/NavigationStatus"
    ]
    assert "/cmd_vel" not in publishers


def test_navigation_status_heartbeat_refreshes_stamp_and_sequence(
    harness: _NodeHarness,
) -> None:
    node = harness.node

    node._timer_callback()
    first = harness.published.messages[-1]
    node._timer_callback()
    harness.clock_ns[0] += node._status_heartbeat_ns - 1
    node._timer_callback()

    assert len(harness.published.messages) == 1

    harness.clock_ns[0] += 1
    node._timer_callback()
    second = harness.published.messages[-1]

    assert len(harness.published.messages) == 2
    assert second.status_sequence == first.status_sequence + 1
    assert second.header.stamp == _time(harness.clock_ns[0])
    assert second.state_revision == first.state_revision


@pytest.mark.parametrize("source", ["odometry", "point_cloud"])
def test_sensor_rejects_stale_future_and_replayed_stamps(
    harness: _NodeHarness,
    source: str,
) -> None:
    node = harness.node
    if source == "odometry":
        callback = node._odometry_callback
        builder = _odometry
        last_stamp_name = "_last_odometry_stamp_ns"
        timeout_ns = node._odometry_timeout_ns
    else:
        callback = node._point_cloud_callback
        builder = _point_cloud
        last_stamp_name = "_last_point_cloud_stamp_ns"
        timeout_ns = node._point_cloud_timeout_ns
    accepted_stamp_ns = NOW_NS - 50_000_000

    callback(builder(accepted_stamp_ns))
    assert getattr(node, last_stamp_name) == accepted_stamp_ns

    callback(builder(NOW_NS - timeout_ns - 1))
    callback(builder(NOW_NS + node._future_tolerance_ns + 1))
    callback(builder(accepted_stamp_ns))

    assert getattr(node, last_stamp_name) == accepted_stamp_ns
    assert node._protocol_poisoned is False


@pytest.mark.parametrize("source", ["odometry", "point_cloud"])
def test_sensor_accepts_tolerance_future_stamp_without_advancing_core_clock(
    harness: _NodeHarness,
    source: str,
) -> None:
    """/clock 落后一拍时保留原始 stamp，但 core 不预支新鲜度。"""

    node = harness.node
    future_stamp_ns = NOW_NS + node._future_tolerance_ns
    if source == "odometry":
        callback = node._odometry_callback
        builder = _odometry
        last_stamp_name = "_last_odometry_stamp_ns"
        core_stamp_name = "_odometry_observed_at_s"
    else:
        callback = node._point_cloud_callback
        builder = _point_cloud
        last_stamp_name = "_last_point_cloud_stamp_ns"
        core_stamp_name = "_point_cloud_observed_at_s"

    callback(builder(future_stamp_ns))

    assert getattr(node, last_stamp_name) == future_stamp_ns
    assert getattr(node._core, core_stamp_name) == pytest.approx(
        NOW_NS / NANOSECONDS_PER_SECOND
    )
    assert node._protocol_poisoned is False


def test_canonical_empty_point_cloud_refreshes_freshness(
    harness: _NodeHarness,
) -> None:
    stamp_ns = NOW_NS - 50_000_000
    node = harness.node

    _bootstrap_path(node)
    node._odometry_callback(_odometry(stamp_ns))
    node._point_cloud_callback(_canonical_empty_point_cloud(stamp_ns))
    node._bspline_callback(_bspline())
    node._scan_status_callback(_scan_trajectory_status())
    node._controller_status_callback(_controller_status())

    assert node._last_point_cloud_stamp_ns == stamp_ns
    assert node._core.state is NavigationState.TRACKING
    assert node._decision().allow_tracking_command is True
    assert node._protocol_poisoned is False


@pytest.mark.parametrize(
    "mutation",
    [
        "height",
        "bigendian",
        "point_step",
        "row_step",
        "data",
        "dense",
        "field_count",
        "field_name",
        "field_offset",
        "field_datatype",
        "field_scalar_count",
    ],
)
def test_malformed_empty_point_cloud_does_not_refresh_freshness(
    harness: _NodeHarness,
    mutation: str,
) -> None:
    message = _canonical_empty_point_cloud(NOW_NS - 50_000_000)
    if mutation == "height":
        message.height = 2
    elif mutation == "bigendian":
        message.is_bigendian = True
    elif mutation == "point_step":
        message.point_step = 16
    elif mutation == "row_step":
        message.row_step = 12
    elif mutation == "data":
        message.data = bytes(12)
    elif mutation == "dense":
        message.is_dense = False
    elif mutation == "field_count":
        message.fields = message.fields[:2]
    elif mutation == "field_name":
        message.fields[0].name = "intensity"
    elif mutation == "field_offset":
        message.fields[1].offset = 5
    elif mutation == "field_datatype":
        message.fields[2].datatype = PointField.FLOAT64
    elif mutation == "field_scalar_count":
        message.fields[0].count = 2

    harness.node._point_cloud_callback(message)

    assert harness.node._last_point_cloud_stamp_ns == 0
    assert harness.node._protocol_poisoned is False


def test_tracking_sensor_timeout_still_enters_emergency_stop(
    harness: _NodeHarness,
) -> None:
    """代际采集完成后仍保留执行中的输入超时急停语义."""
    node = harness.node
    _bootstrap_tracking(node)
    harness.clock_ns[0] = 10_300_000_001

    node._timer_callback()

    decision = node._decision()
    assert decision.state is NavigationState.EMERGENCY_STOP
    assert decision.stale_inputs == ("odometry",)
    assert decision.reason == "input_timeout:odometry"
    assert decision.force_zero_velocity is True
    assert decision.global_replan_requested is False


def test_nonfinal_trajectory_finish_returns_to_local_planning(
    harness: _NodeHarness,
) -> None:
    """局部轨迹自然结束只等待下一段，不锁存 recovery 或急停。"""

    node = harness.node
    _bootstrap_tracking(node)

    node._controller_status_callback(_finished_nonfinal_controller_status())

    waiting = node._decision()
    assert waiting.state is NavigationState.LOCAL_PLANNING
    assert waiting.reason == "local_trajectory_finished"
    assert waiting.force_zero_velocity is True
    assert waiting.global_replan_requested is False
    assert node._recovery_block_reason == ""
    assert node._reported_tracking_identity is None

    node._bspline_callback(_bspline(trajectory_id=12))
    node._scan_status_callback(
        _scan_trajectory_status(sequence=2, trajectory_id=12)
    )
    node._controller_status_callback(
        _controller_status(
            sequence=3,
            acceptance_sequence=2,
            trajectory_id=12,
        )
    )

    resumed = node._decision()
    assert resumed.state is NavigationState.TRACKING
    assert resumed.allow_tracking_command is True
    assert resumed.global_replan_requested is False


def test_valid_acceptance_with_pre_tick_waiting_state_is_not_emergency(
    harness: _NodeHarness,
) -> None:
    """B-spline 接管与首个控制 tick 之间的状态快照不能制造假急停。"""

    node = harness.node
    _bootstrap_path(node)
    node._odometry_callback(_odometry(9_950_000_000))
    node._point_cloud_callback(_point_cloud(9_950_000_000))
    node._bspline_callback(_bspline())
    node._scan_status_callback(_scan_trajectory_status())

    node._controller_status_callback(
        _controller_status(
            event=ControllerStatus.EVENT_ACCEPTED,
            state=ControllerStatus.STATE_WAITING_FOR_TRAJECTORY,
        )
    )

    waiting = node._decision()
    assert waiting.state is NavigationState.LOCAL_PLANNING
    assert waiting.reason == "global_path_available"
    assert waiting.force_zero_velocity is True
    assert node._recovery_block_reason == ""

    node._controller_status_callback(
        _controller_status(
            sequence=2,
            event=ControllerStatus.EVENT_STATE_CHANGED,
            state=ControllerStatus.STATE_TRACKING,
        )
    )

    tracking = node._decision()
    assert tracking.state is NavigationState.TRACKING
    assert tracking.allow_tracking_command is True
    assert tracking.force_zero_velocity is False


def test_controller_timeout_still_enters_emergency_stop(
    harness: _NodeHarness,
) -> None:
    """自然结束的窄化不能放松真实 controller 超时急停。"""

    node = harness.node
    _bootstrap_tracking(node)
    timeout = _controller_status(
        sequence=2,
        event=ControllerStatus.EVENT_STATE_CHANGED,
        state=ControllerStatus.STATE_TRAJECTORY_TIMEOUT,
        trajectory_valid=False,
    )

    node._controller_status_callback(timeout)

    decision = node._decision()
    assert decision.state is NavigationState.EMERGENCY_STOP
    assert decision.reason == "controller_safe_stop"
    assert decision.force_zero_velocity is True


@pytest.mark.parametrize(
    ("event", "expected_reason"),
    [
        (
            ScanPlanningStatus.EVENT_STAIR_INHIBITED,
            "scan_stair_execution_inhibited",
        ),
        (
            ScanPlanningStatus.EVENT_STAIR_RESUME_WAITING,
            "scan_stair_resume_waiting",
        ),
    ],
)
def test_exact_stair_contract_generates_stop_advisory(
    harness: _NodeHarness,
    event: int,
    expected_reason: str,
) -> None:
    node = harness.node
    _bootstrap_path(node)

    node._scan_status_callback(_stair_status(event=event))

    decision = node._decision()
    assert decision.state is NavigationState.EMERGENCY_STOP
    assert decision.reason == expected_reason
    assert decision.global_replan_requested is False
    assert decision.global_replan_in_flight is False
    assert node._protocol_poisoned is False


def test_stair_inhibit_overrides_prior_controller_safe_stop_reason(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    _bootstrap_tracking(node)
    node._controller_status_callback(_stopped_controller_status())
    assert node._decision().reason == "controller_safe_stop"

    node._scan_status_callback(_stair_status(sequence=2))

    decision = node._decision()
    assert decision.state is NavigationState.EMERGENCY_STOP
    assert decision.reason == "scan_stair_execution_inhibited"
    assert decision.global_replan_requested is False
    assert node._protocol_poisoned is False


@pytest.mark.parametrize(
    "old_state",
    [
        ControllerStatus.STATE_TRACKING,
        ControllerStatus.STATE_ALIGNING_YAW,
    ],
)
def test_stair_inhibit_requires_new_trajectory_identity_before_recovery(
    harness: _NodeHarness,
    old_state: int,
) -> None:
    """旧 B-spline 的状态心跳不能提前解除楼梯冻结停车。"""

    node = harness.node
    _bootstrap_tracking(node)
    blocked_identity = node._reported_tracking_identity
    assert blocked_identity is not None

    node._scan_status_callback(_stair_status(sequence=2))
    assert node._decision().state is NavigationState.EMERGENCY_STOP
    assert node._recovery_blocked_identity == blocked_identity

    node._controller_status_callback(
        _controller_status(
            sequence=2,
            event=ControllerStatus.EVENT_STATE_CHANGED,
            state=old_state,
        )
    )

    blocked = node._decision()
    assert blocked.state is NavigationState.EMERGENCY_STOP
    assert blocked.reason == "scan_stair_execution_inhibited"
    assert blocked.allow_tracking_command is False
    assert blocked.force_zero_velocity is True

    node._bspline_callback(_bspline(trajectory_id=12))
    node._scan_status_callback(
        _scan_trajectory_status(sequence=3, trajectory_id=12)
    )
    node._controller_status_callback(
        _controller_status(
            sequence=3,
            acceptance_sequence=2,
            trajectory_id=12,
        )
    )

    recovered = node._decision()
    assert recovered.state is NavigationState.TRACKING
    assert recovered.reason == "emergency_cleared"
    assert recovered.allow_tracking_command is True
    assert recovered.force_zero_velocity is False


def test_stair_emergency_trajectory_confirms_stop_before_root_lock(
    harness: _NodeHarness,
) -> None:
    """楼梯冻结的急停轨迹必须形成 controller 可核验的停车证据。"""

    node = harness.node
    _bootstrap_tracking(node)

    node._bspline_callback(
        _bspline(trajectory_id=12, emergency_stop=True)
    )
    node._scan_status_callback(_scan_emergency_status())
    node._controller_status_callback(
        _controller_status(
            sequence=2,
            acceptance_sequence=2,
            event=ControllerStatus.EVENT_ACCEPTED,
            trajectory_id=12,
            emergency_stop=True,
        )
    )
    node._scan_status_callback(_stair_status(sequence=3))

    waiting = node._decision()
    assert waiting.state is NavigationState.EMERGENCY_STOP
    assert waiting.reason == "scan_stair_execution_inhibited"
    assert node._stop_confirmed is False

    node._controller_status_callback(
        _controller_status(
            sequence=3,
            acceptance_sequence=2,
            event=ControllerStatus.EVENT_STATE_CHANGED,
            state=ControllerStatus.STATE_EMERGENCY_STOP,
            trajectory_id=12,
            emergency_stop=True,
        )
    )

    confirmed = node._decision()
    assert confirmed.state is NavigationState.EMERGENCY_STOP
    assert confirmed.reason == "scan_stair_execution_inhibited"
    assert node._stop_confirmed is True


def test_stair_ack_does_not_override_active_global_replan(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    _bootstrap_tracking(node)
    node._last_decision = node._core.report_predicted_collision(
        NOW_NS / NANOSECONDS_PER_SECOND,
        reason="collision_requires_global_replan",
    )
    blocked_reason = node._decision().reason
    assert node._decision().global_replan_requested is True

    node._scan_status_callback(_stair_status(sequence=2))

    decision = node._decision()
    assert decision.global_replan_requested is True
    assert decision.reason == blocked_reason
    assert decision.reason != "scan_stair_execution_inhibited"
    assert node._protocol_poisoned is False


@pytest.mark.parametrize(
    "fault_reason",
    [
        "scan_stair_freeze_frame_mismatch_fault",
        "scan_stair_freeze_protocol_fault",
        "scan_stair_freeze_snapshot_timeout_fault",
        "scan_stair_stop_publish_fault",
    ],
)
def test_known_stair_fault_preserves_exact_fail_closed_reason(
    harness: _NodeHarness,
    fault_reason: str,
) -> None:
    node = harness.node
    _bootstrap_path(node)

    node._scan_status_callback(_stair_status(reason=fault_reason))

    decision = node._decision()
    assert decision.state is NavigationState.EMERGENCY_STOP
    assert decision.reason == fault_reason
    assert decision.force_zero_velocity is True
    assert decision.allow_tracking_command is False
    assert decision.reason != "scan_stair_execution_inhibited"
    assert node._protocol_poisoned is True


@pytest.mark.parametrize(
    "invalid_reason",
    ["scan_stair_resume_waiting", "unknown_stair_reason"],
)
def test_stair_mismatched_or_unknown_reason_poison_is_invariant(
    harness: _NodeHarness,
    invalid_reason: str,
) -> None:
    node = harness.node
    _bootstrap_path(node)

    node._scan_status_callback(_stair_status(reason=invalid_reason))

    decision = node._decision()
    assert decision.state is NavigationState.EMERGENCY_STOP
    assert decision.reason.startswith("scan_status_invariant:")
    assert "event/reason" in decision.reason
    assert node._protocol_poisoned is True


def test_protocol_poison_cannot_be_overwritten_by_later_legal_stair_ack(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    _bootstrap_path(node)
    node._scan_status_callback(
        _stair_status(reason="scan_stair_freeze_protocol_fault")
    )
    poisoned_reason = node._decision().reason
    assert poisoned_reason == "scan_stair_freeze_protocol_fault"
    assert node._protocol_poisoned is True

    node._scan_status_callback(_stair_status(sequence=2))

    assert node._protocol_poisoned is True
    assert node._decision().reason == poisoned_reason
    assert node._decision().reason != "scan_stair_execution_inhibited"


@pytest.mark.parametrize("status_first", [True, False])
def test_pct_path_and_success_status_are_order_independent(
    harness: _NodeHarness,
    status_first: bool,
) -> None:
    _bootstrap_path(harness.node, status_first=status_first)

    assert harness.node._active_path is not None
    assert harness.node._active_path.stamp_ns == PATH_STAMP_NS
    assert harness.node._pct_plan_id == 1
    assert harness.node._global_planning_deadline_ns == 0
    assert harness.node._global_planning_phase == ""


def test_path_acceptance_resets_freshness_but_preserves_sensor_watermarks(
    harness: _NodeHarness,
) -> None:
    """切换 Path 代次不能让旧 freshness 或重放消息通过安全门."""
    node = harness.node
    sensor_stamp_ns = 9_950_000_000
    node._pct_status_callback(_pct_status())
    node._odometry_callback(_odometry(sensor_stamp_ns))
    node._point_cloud_callback(_point_cloud(sensor_stamp_ns))

    node._path_callback(_path())

    assert node._core.state is NavigationState.LOCAL_PLANNING
    assert node._decision().stale_inputs == (
        "odometry",
        "point_cloud",
        "bspline",
    )
    assert node._last_odometry_stamp_ns == sensor_stamp_ns
    assert node._last_point_cloud_stamp_ns == sensor_stamp_ns

    node._odometry_callback(_odometry(sensor_stamp_ns))
    node._point_cloud_callback(_point_cloud(sensor_stamp_ns))
    assert node._decision().stale_inputs == (
        "odometry",
        "point_cloud",
        "bspline",
    )

    next_sensor_stamp_ns = sensor_stamp_ns + 10_000_000
    node._odometry_callback(_odometry(next_sensor_stamp_ns))
    assert node._decision().stale_inputs == ("point_cloud", "bspline")
    node._point_cloud_callback(_point_cloud(next_sensor_stamp_ns))
    assert node._decision().stale_inputs == ("bspline",)


@pytest.mark.parametrize("status_first", [True, False])
def test_pct_no_path_waits_for_matching_empty_path(
    harness: _NodeHarness,
    status_first: bool,
) -> None:
    node = harness.node
    failure_stamp_ns = 5_000_000_000
    status = _pct_status(
        path_stamp_ns=failure_stamp_ns,
        state=PCTPlanningStatus.NO_PATH,
    )
    empty_path = _path(failure_stamp_ns, empty=True)

    first, second = (
        (node._pct_status_callback, status),
        (node._path_callback, empty_path),
    )
    if not status_first:
        first, second = second, first
    first[0](first[1])
    if status_first:
        assert node._core.state is NavigationState.GLOBAL_PLANNING
        assert node._replan is None
    else:
        assert node._core.state is NavigationState.IDLE
    second[0](second[1])

    assert node._core.state is NavigationState.GLOBAL_REPLAN
    assert node._decision().global_replan_requested is True
    assert node._replan is None
    assert node._global_planning_deadline_ns == 0
    assert node._global_replan_cycle_count == 0


def test_initial_global_planning_deadline_latches_terminal_stop(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    node._pct_status_callback(
        _pct_status(state=PCTPlanningStatus.PLANNING),
    )

    assert node._core.state is NavigationState.GLOBAL_PLANNING
    assert node._global_planning_phase == "initial"
    assert node._global_planning_deadline_ns == (
        NOW_NS + node._global_planning_timeout_ns
    )

    harness.clock_ns[0] = node._global_planning_deadline_ns
    node._timer_callback()

    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert node._decision().global_replan_requested is False
    assert node._decision().global_replan_in_flight is False
    assert node._global_planning_terminal_reason == (
        "initial_global_planning_timeout"
    )
    assert node._adapter_fault_reason == "initial_global_planning_timeout"
    assert node._protocol_poisoned is True
    assert node._global_replan_cycle_count == 0


def test_replan_ack_deadline_rejects_late_path_and_stops(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = harness.node
    client = _FakePCTClient()
    monkeypatch.setattr(node, "_pct_client", client)
    _bootstrap_tracking(node)
    node._scan_status_callback(_collision_status())
    node._controller_status_callback(_stopped_controller_status())

    client.futures[0].complete(_accepted_replan_response(node))

    assert node._core.state is NavigationState.GLOBAL_PLANNING
    assert node._global_planning_phase == "replan"
    assert node._global_replan_cycle_count == 1
    deadline_ns = node._global_planning_deadline_ns

    harness.clock_ns[0] = deadline_ns
    node._path_callback(_path(5_000_000_000))

    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert node._global_planning_terminal_reason == (
        "global_replan_result_timeout"
    )
    assert 5_000_000_000 not in node._path_records
    assert node._decision().global_replan_requested is False
    assert node._decision().global_replan_in_flight is False
    assert len(client.requests) == 1


def test_successful_replan_clears_deadline_and_cycle_budget(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = harness.node
    client = _FakePCTClient()
    monkeypatch.setattr(node, "_pct_client", client)
    _bootstrap_tracking(node)
    node._scan_status_callback(_collision_status())
    node._controller_status_callback(_stopped_controller_status())
    transaction = node._replan
    assert transaction is not None
    client.futures[0].complete(_accepted_replan_response(node))
    success = _pct_status(
        path_stamp_ns=5_000_000_000,
        state=PCTPlanningStatus.SUCCEEDED,
        event_stamp_ns=NOW_NS,
        plan_id=2,
        request_id=transaction.request_id,
        command=PCTPlanningStatus.COMMAND_REPLAN,
    )
    node._odometry_callback(_odometry(9_960_000_000))
    node._point_cloud_callback(_point_cloud(9_960_000_000))

    node._pct_status_callback(success)
    node._path_callback(_path(5_000_000_000))

    assert node._core.state is NavigationState.LOCAL_PLANNING
    assert node._decision().stale_inputs == (
        "odometry",
        "point_cloud",
        "bspline",
    )
    node._odometry_callback(_odometry(9_970_000_000))
    node._point_cloud_callback(_point_cloud(9_970_000_000))
    assert node._decision().stale_inputs == ("bspline",)
    assert node._global_planning_deadline_ns == 0
    assert node._global_planning_phase == ""
    assert node._global_replan_cycle_count == 0
    assert node._global_planning_terminal_reason == ""


def test_successful_replan_quarantines_only_authenticated_retired_generation(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合法重规划交接允许旧尾包，但恢复必须使用新代完整执行证据。"""

    node = harness.node
    client = _FakePCTClient()
    monkeypatch.setattr(node, "_pct_client", client)
    _bootstrap_tracking(node)
    node._scan_status_callback(_collision_status())
    node._controller_status_callback(_stopped_controller_status())
    transaction = node._replan
    assert transaction is not None

    tombstone_ns = 4_500_000_000
    new_path_stamp_ns = 5_000_000_000
    client.futures[0].complete(
        _accepted_replan_response(
            node,
            tombstone_ns=tombstone_ns,
            active_path_stamp_ns=new_path_stamp_ns,
        )
    )
    node._path_callback(_path(tombstone_ns, empty=True))
    node._pct_status_callback(
        _pct_status(
            path_stamp_ns=new_path_stamp_ns,
            plan_id=2,
            request_id=transaction.request_id,
            command=PCTPlanningStatus.COMMAND_REPLAN,
        )
    )
    node._path_callback(_path(new_path_stamp_ns))

    assert node._latest_tombstone_stamp_ns == tombstone_ns
    assert node._active_path is not None
    assert node._active_path.stamp_ns == new_path_stamp_ns
    assert node._core.state is NavigationState.LOCAL_PLANNING

    node._controller_status_callback(_stopped_controller_status(sequence=3))
    assert node._protocol_poisoned is False
    assert node._core.state is NavigationState.EMERGENCY_STOP

    node._bspline_callback(
        _bspline(path_stamp_ns=PATH_STAMP_NS, trajectory_id=12)
    )
    node._scan_status_callback(
        _stair_status(
            sequence=3,
            path_stamp_ns=PATH_STAMP_NS,
            reason="scan_stair_freeze_protocol_fault",
        )
    )

    assert node._protocol_poisoned is False
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert node._scan_sequence.latest == 3
    assert all(
        identity.trajectory_id != 12
        for identity in node._trajectories
    )

    node._odometry_callback(_odometry(9_970_000_000))
    node._point_cloud_callback(_point_cloud(9_970_000_000))
    node._bspline_callback(
        _bspline(path_stamp_ns=new_path_stamp_ns, trajectory_id=22)
    )
    node._scan_status_callback(
        _scan_trajectory_status(
            sequence=4,
            path_stamp_ns=new_path_stamp_ns,
            trajectory_id=22,
        )
    )
    controller = _controller_status(
        sequence=4,
        path_stamp_ns=new_path_stamp_ns,
        trajectory_id=22,
    )
    controller.acceptance_sequence = 2
    node._controller_status_callback(controller)

    assert node._protocol_poisoned is False
    assert node._core.state is NavigationState.TRACKING
    assert node._reported_tracking_identity is not None
    assert (
        node._reported_tracking_identity.reference_path_stamp_ns
        == new_path_stamp_ns
    )


def test_retired_scan_generation_keeps_same_sequence_conflict_fail_closed(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离旧尾包不能放弃同一 wire sequence 的冲突检测。"""

    node = harness.node
    client = _FakePCTClient()
    monkeypatch.setattr(node, "_pct_client", client)
    _bootstrap_tracking(node)
    node._scan_status_callback(_collision_status())
    node._controller_status_callback(_stopped_controller_status())
    transaction = node._replan
    assert transaction is not None

    tombstone_ns = 4_500_000_000
    new_path_stamp_ns = 5_000_000_000
    client.futures[0].complete(
        _accepted_replan_response(
            node,
            tombstone_ns=tombstone_ns,
            active_path_stamp_ns=new_path_stamp_ns,
        )
    )
    node._path_callback(_path(tombstone_ns, empty=True))
    node._pct_status_callback(
        _pct_status(
            path_stamp_ns=new_path_stamp_ns,
            plan_id=2,
            request_id=transaction.request_id,
            command=PCTPlanningStatus.COMMAND_REPLAN,
        )
    )
    node._path_callback(_path(new_path_stamp_ns))

    node._scan_status_callback(
        _stair_status(
            sequence=3,
            path_stamp_ns=PATH_STAMP_NS,
            reason="scan_stair_freeze_protocol_fault",
        )
    )
    node._scan_status_callback(
        _stair_status(
            sequence=3,
            path_stamp_ns=PATH_STAMP_NS,
            reason="scan_stair_freeze_frame_mismatch_fault",
        )
    )

    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert node._adapter_fault_reason == (
        "conflicting_retired_scan_status_sequence"
    )


def test_global_replan_cycle_budget_stops_without_fourth_request(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = harness.node
    client = _FakePCTClient()
    monkeypatch.setattr(node, "_pct_client", client)
    _bootstrap_tracking(node)
    node._scan_status_callback(_collision_status())
    node._controller_status_callback(_stopped_controller_status())

    for cycle in range(1, node._max_global_replan_cycles + 1):
        transaction = node._replan
        assert transaction is not None
        result_stamp_ns = PATH_STAMP_NS + cycle * NANOSECONDS_PER_SECOND
        plan_id = cycle + 1
        client.futures[-1].complete(
            _accepted_replan_response(
                node,
                plan_id=plan_id,
                tombstone_ns=result_stamp_ns,
            )
        )
        assert node._global_replan_cycle_count == cycle
        assert node._core.state is NavigationState.GLOBAL_PLANNING
        failure = _pct_status(
            path_stamp_ns=result_stamp_ns,
            state=PCTPlanningStatus.NO_PATH,
            event_stamp_ns=NOW_NS,
            plan_id=plan_id,
            request_id=transaction.request_id,
            command=PCTPlanningStatus.COMMAND_REPLAN,
        )

        node._pct_status_callback(failure)
        node._path_callback(_path(result_stamp_ns, empty=True))

        if cycle < node._max_global_replan_cycles:
            assert len(client.requests) == cycle + 1
            assert node._core.state is NavigationState.GLOBAL_REPLAN

    assert len(client.requests) == node._max_global_replan_cycles
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert node._decision().global_replan_requested is False
    assert node._decision().global_replan_in_flight is False
    assert node._global_planning_terminal_reason == (
        "global_replan_cycles_exhausted"
    )
    assert node._protocol_poisoned is True


@pytest.mark.parametrize(
    "arrival_order",
    tuple(permutations(("bspline", "scan", "controller"))),
)
def test_tracking_requires_all_three_trajectory_observations_in_any_order(
    harness: _NodeHarness,
    arrival_order: tuple[str, str, str],
) -> None:
    node = harness.node
    _bootstrap_path(node)
    node._odometry_callback(_odometry(9_950_000_000))
    node._point_cloud_callback(_point_cloud(9_950_000_000))
    callbacks = {
        "bspline": (node._bspline_callback, _bspline()),
        "scan": (node._scan_status_callback, _scan_trajectory_status()),
        "controller": (
            node._controller_status_callback,
            _controller_status(),
        ),
    }

    for name in arrival_order:
        callback, message = callbacks[name]
        callback(message)

    assert node._core.state is NavigationState.TRACKING
    assert node._reported_tracking_identity is not None
    assert node._reported_tracking_identity.trajectory_id == 11


def test_nonfinal_yaw_alignment_uses_immutable_controller_hard_bound(
    harness: _NodeHarness,
) -> None:
    """航向状态抖动可提升但不能滚动延长同一轨迹的硬截止。"""

    node = harness.node
    _bootstrap_tracking(node)
    nominal_end_ns = START_STAMP_NS + 3 * NANOSECONDS_PER_SECOND
    soft_expiry_ns = nominal_end_ns + node._trajectory_expiry_grace_ns
    hard_expiry_ns = soft_expiry_ns + node._max_yaw_alignment_freeze_ns
    assert node._core._bspline_valid_until_s == pytest.approx(
        soft_expiry_ns / NANOSECONDS_PER_SECOND
    )

    node._controller_status_callback(
        _controller_status(
            sequence=2,
            event=ControllerStatus.EVENT_STATE_CHANGED,
            state=ControllerStatus.STATE_ALIGNING_YAW,
            is_final=False,
        )
    )
    assert node._core._bspline_valid_until_s == pytest.approx(
        hard_expiry_ns / NANOSECONDS_PER_SECOND
    )
    node._controller_status_callback(
        _controller_status(
            sequence=3,
            event=ControllerStatus.EVENT_STATE_CHANGED,
            state=ControllerStatus.STATE_TRACKING,
            is_final=False,
        )
    )
    node._controller_status_callback(
        _controller_status(
            sequence=4,
            event=ControllerStatus.EVENT_STATE_CHANGED,
            state=ControllerStatus.STATE_ALIGNING_YAW,
            is_final=False,
        )
    )
    assert node._core._bspline_valid_until_s == pytest.approx(
        hard_expiry_ns / NANOSECONDS_PER_SECOND
    )

    _refresh_tracking_sensors_until(harness, nominal_end_ns + 1)
    node._timer_callback()

    assert node._core.state is NavigationState.TRACKING
    assert node._decision().allow_tracking_command is True

    _refresh_tracking_sensors_until(harness, hard_expiry_ns)
    node._timer_callback()
    assert node._core.state is NavigationState.TRACKING

    harness.clock_ns[0] = hard_expiry_ns + 1
    node._timer_callback()

    decision = node._decision()
    assert decision.state is NavigationState.EMERGENCY_STOP
    assert decision.stale_inputs == ("bspline",)
    assert decision.reason == "input_timeout:bspline"
    assert decision.force_zero_velocity is True


def test_nonfinal_without_yaw_alignment_stops_at_soft_expiry(
    harness: _NodeHarness,
) -> None:
    """未进入航向对齐的普通轨迹不能预支额外六秒冻结预算。"""

    node = harness.node
    _bootstrap_tracking(node)
    nominal_end_ns = START_STAMP_NS + 3 * NANOSECONDS_PER_SECOND
    soft_expiry_ns = nominal_end_ns + node._trajectory_expiry_grace_ns
    _refresh_tracking_sensors_until(harness, soft_expiry_ns)
    node._timer_callback()
    assert node._core.state is NavigationState.TRACKING

    harness.clock_ns[0] = soft_expiry_ns + 1
    node._timer_callback()

    decision = node._decision()
    assert decision.state is NavigationState.EMERGENCY_STOP
    assert decision.stale_inputs == ("bspline",)
    assert decision.reason == "input_timeout:bspline"


@pytest.mark.parametrize(
    "arrival_order",
    tuple(permutations(("path", "bspline", "scan"))),
)
def test_path_bspline_and_scan_can_arrive_in_any_order(
    harness: _NodeHarness,
    arrival_order: tuple[str, str, str],
) -> None:
    node = harness.node
    node._pct_status_callback(_pct_status())
    node._odometry_callback(_odometry(9_950_000_000))
    node._point_cloud_callback(_point_cloud(9_950_000_000))
    node._controller_status_callback(_controller_status())
    callbacks = {
        "path": (node._path_callback, _path()),
        "bspline": (node._bspline_callback, _bspline()),
        "scan": (node._scan_status_callback, _scan_trajectory_status()),
    }

    for name in arrival_order:
        callback, message = callbacks[name]
        callback(message)

    assert node._core.state is NavigationState.LOCAL_PLANNING
    assert node._decision().stale_inputs == ("odometry", "point_cloud")
    node._odometry_callback(_odometry(9_960_000_000))
    node._point_cloud_callback(_point_cloud(9_960_000_000))

    assert node._core.state is NavigationState.TRACKING
    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is False


def test_trajectory_evidence_can_precede_pct_status_and_path(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    node._odometry_callback(_odometry(9_950_000_000))
    node._point_cloud_callback(_point_cloud(9_950_000_000))
    node._bspline_callback(_bspline())
    node._scan_status_callback(_scan_trajectory_status())
    node._controller_status_callback(_controller_status())

    assert node._pending_path_evidence_count() == 2
    assert node._core.state is NavigationState.IDLE

    node._pct_status_callback(_pct_status())
    node._path_callback(_path())

    assert node._core.state is NavigationState.LOCAL_PLANNING
    assert node._decision().stale_inputs == ("odometry", "point_cloud")
    node._odometry_callback(_odometry(9_960_000_000))
    node._point_cloud_callback(_point_cloud(9_960_000_000))

    assert node._core.state is NavigationState.TRACKING
    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is False


def test_pending_scan_statuses_replay_in_sequence_order(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    node._pct_status_callback(_pct_status())
    node._odometry_callback(_odometry(9_950_000_000))
    node._point_cloud_callback(_point_cloud(9_950_000_000))
    node._controller_status_callback(_controller_status())
    node._bspline_callback(_bspline())
    node._scan_status_callback(_scan_trajectory_status(sequence=2))
    node._scan_status_callback(_scan_trajectory_status(sequence=1))

    node._path_callback(_path())

    assert node._core.state is NavigationState.LOCAL_PLANNING
    assert node._decision().stale_inputs == ("odometry", "point_cloud")
    node._odometry_callback(_odometry(9_960_000_000))
    node._point_cloud_callback(_point_cloud(9_960_000_000))

    assert node._scan_sequence.latest == 2
    assert node._core.state is NavigationState.TRACKING
    assert node._pending_path_evidence_count() == 0


def test_conflicting_pending_bspline_clears_cache_and_stops(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    node._pct_status_callback(_pct_status())
    node._bspline_callback(_bspline())
    conflict = _bspline()
    conflict.pos_pts[0].y = 0.25

    node._bspline_callback(conflict)

    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert "conflicting_pending_bspline" in node._adapter_fault_reason


def test_conflicting_pending_scan_sequence_clears_cache_and_stops(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    node._pct_status_callback(_pct_status())
    node._scan_status_callback(_scan_trajectory_status())
    conflict = _scan_trajectory_status()
    conflict.reason = "conflicting_payload"

    node._scan_status_callback(conflict)

    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert "conflicting_pending_scan" in node._adapter_fault_reason


def test_pending_path_evidence_timeout_clears_cache_and_stops(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    node._pct_status_callback(_pct_status())
    node._bspline_callback(_bspline())
    assert node._pending_path_evidence_count() == 1

    harness.clock_ns[0] += node._pending_path_evidence_timeout_ns + 1
    node._timer_callback()

    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert node._adapter_fault_reason == "pending_path_evidence_timeout"


@pytest.mark.parametrize("trigger", ("bspline", "scan"))
def test_timeout_triggering_callback_cannot_repopulate_pending_cache(
    harness: _NodeHarness,
    trigger: str,
) -> None:
    node = harness.node
    node._pct_status_callback(_pct_status())
    if trigger == "bspline":
        node._scan_status_callback(_scan_trajectory_status())
        callback, message = node._bspline_callback, _bspline()
    else:
        node._bspline_callback(_bspline())
        callback, message = (
            node._scan_status_callback,
            _scan_trajectory_status(),
        )
    assert node._pending_path_evidence_count() == 1

    harness.clock_ns[0] += node._pending_path_evidence_timeout_ns + 1
    callback(message)

    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert node._adapter_fault_reason == "pending_path_evidence_timeout"


def test_pending_path_evidence_capacity_overflow_stops(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    node._max_pending_path_evidence = 1
    node._pct_status_callback(_pct_status())
    node._bspline_callback(_bspline())

    node._scan_status_callback(_scan_trajectory_status())

    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert "capacity_exceeded" in node._adapter_fault_reason


def test_path_tombstone_clears_pending_evidence_and_stops(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    node._pct_status_callback(_pct_status())
    node._bspline_callback(_bspline())
    node._scan_status_callback(_scan_trajectory_status())

    node._path_callback(_path(PATH_STAMP_NS + 1, empty=True))

    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert "tombstone" in node._adapter_fault_reason


def test_old_pending_generation_cannot_pollute_new_path(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    new_path_stamp_ns = PATH_STAMP_NS + 1
    node._bspline_callback(_bspline())
    node._scan_status_callback(_scan_trajectory_status())
    node._pct_status_callback(
        _pct_status(path_stamp_ns=new_path_stamp_ns),
    )

    node._path_callback(_path(new_path_stamp_ns))

    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert node._trajectories == {}
    assert node._scan_trajectory_identities == set()
    assert "old_generation" in node._adapter_fault_reason


def test_observed_future_evidence_blocks_older_path_until_new_path_arrives(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    future_path_stamp_ns = PATH_STAMP_NS + 1
    node._odometry_callback(_odometry(9_950_000_000))
    node._point_cloud_callback(_point_cloud(9_950_000_000))
    node._bspline_callback(_bspline(path_stamp_ns=future_path_stamp_ns))
    node._scan_status_callback(
        _scan_trajectory_status(path_stamp_ns=future_path_stamp_ns),
    )
    node._controller_status_callback(
        _controller_status(path_stamp_ns=future_path_stamp_ns),
    )
    node._pct_status_callback(_pct_status())

    node._path_callback(_path())

    assert node._active_path is None
    assert node._pending_path_evidence_count() == 2
    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.GLOBAL_PLANNING
    assert node._decision().force_zero_velocity is True
    assert "older_than_pending" in node._adapter_fault_reason

    node._pct_status_callback(
        _pct_status(path_stamp_ns=future_path_stamp_ns, plan_id=2),
    )
    node._path_callback(_path(future_path_stamp_ns))

    assert node._core.state is NavigationState.LOCAL_PLANNING
    assert node._decision().stale_inputs == ("odometry", "point_cloud")
    node._odometry_callback(_odometry(9_960_000_000))
    node._point_cloud_callback(_point_cloud(9_960_000_000))

    assert node._active_path is not None
    assert node._active_path.stamp_ns == future_path_stamp_ns
    assert node._pending_path_evidence_count() == 0
    assert node._protocol_poisoned is False
    assert node._core.state is NavigationState.TRACKING


def test_scan_event_from_old_path_generation_is_rejected(
    harness: _NodeHarness,
) -> None:
    node = harness.node
    _bootstrap_path(node)

    node._scan_status_callback(
        _scan_trajectory_status(path_stamp_ns=PATH_STAMP_NS - 1),
    )

    assert node._protocol_poisoned is True
    assert node._core.state is NavigationState.EMERGENCY_STOP
    assert not node._scan_trajectory_identities
    assert "Path" in node._adapter_fault_reason


def test_collision_waits_for_post_fault_controller_stop_evidence(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = harness.node
    client = _FakePCTClient()
    monkeypatch.setattr(node, "_pct_client", client)
    _bootstrap_tracking(node)

    node._scan_status_callback(_collision_status())

    assert node._decision().global_replan_requested is True
    assert node._stop_confirmed is False
    assert client.requests == []

    node._controller_status_callback(_stopped_controller_status())

    assert node._stop_confirmed is True
    assert len(client.requests) == 1
    assert node._replan is not None
    assert node._replan.in_flight is True


def test_replan_retry_reuses_payload_and_accepts_late_success(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = harness.node
    client = _FakePCTClient()
    monkeypatch.setattr(node, "_pct_client", client)
    _bootstrap_tracking(node)
    node._scan_status_callback(_collision_status())
    node._controller_status_callback(_stopped_controller_status())
    assert len(client.requests) == 1
    first_future = client.futures[0]

    harness.clock_ns[0] += node._replan_response_ns
    node._process_replan(harness.clock_ns[0])

    assert first_future.cancel_calls == 1
    assert len(client.requests) == 2
    assert client.requests[0] == client.requests[1]
    assert node._replan is not None
    assert node._replan.attempts == 2

    node._replan_response_callback(
        _NoneResultFuture(),
        request_id=node._replan.request_id,
        epoch=node._replan.epoch,
        attempt_token=node._replan_attempt_token - 1,
    )
    assert node._replan.in_flight is True

    first_future.complete(_accepted_replan_response(node))

    assert node._replan.acknowledged is True
    assert node._core.state is NavigationState.GLOBAL_PLANNING
    assert node._decision().global_replan_in_flight is True


def test_current_empty_replan_response_becomes_retryable_failure(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = harness.node
    client = _FakePCTClient()
    monkeypatch.setattr(node, "_pct_client", client)
    _bootstrap_tracking(node)
    node._scan_status_callback(_collision_status())
    node._controller_status_callback(_stopped_controller_status())
    transaction = node._replan
    assert transaction is not None
    assert transaction.in_flight is True

    node._replan_response_callback(
        _NoneResultFuture(),
        request_id=transaction.request_id,
        epoch=transaction.epoch,
        attempt_token=node._replan_attempt_token,
    )

    assert transaction.in_flight is False
    assert transaction.acknowledged is False
    assert transaction.terminal_error == ""
    assert transaction.attempts == 1


def test_clock_rollback_clears_epoch_and_rejects_new_messages(
    harness: _NodeHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = harness.node
    _bootstrap_tracking(node)
    rollback_clock_ns = [NOW_NS - 1]
    monkeypatch.setattr(
        NavigationSupervisorNode,
        "get_clock",
        lambda _self: _FakeClock(rollback_clock_ns),
    )
    node._last_clock_ns = NOW_NS

    observed_ns = NavigationSupervisorNode._clock_now_ns(node)

    assert observed_ns == rollback_clock_ns[0]
    assert node._epoch == 1
    assert node._core.state is NavigationState.IDLE
    assert node._goal is None
    assert node._active_path is None
    assert node._trajectories == {}
    assert node._protocol_poisoned is True
    assert node._fatal_epoch_reset is True

    node._odometry_callback(_odometry(rollback_clock_ns[0]))
    assert node._last_odometry_stamp_ns == 0
