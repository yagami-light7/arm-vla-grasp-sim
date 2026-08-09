"""PCT ROS 2 adapter 的异步代际、消息和 QoS 合同测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from scan_planner_msgs.msg import Bspline, PCTPlanningStatus, ScanPlanningStatus
from scan_planner_msgs.srv import PCTPlanningCommand

from pct_ros2_adapter.backend import (
    PCTBackendError,
    PCTBackendPlan,
    PCTNoPathError,
    create_global_planner_backend,
)
from pct_ros2_adapter.node import PCTROS2Adapter
from probe_live_multifloor_pct_scan_chain import (
    _execution_expected,
    _expected_execution_goal,
    _matching_scan_trajectory_pair,
)


class _CapturePublisher:
    """只记录发布消息的测试替身。"""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


@dataclass
class _FakeBackend:
    """返回与输入目标绑定的两点地面 Path。"""

    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def plan(
        self,
        *,
        start_base_xyz,
        goal_base_xyz,
        goal_yaw,
    ) -> PCTBackendPlan:
        self.calls.append(
            {
                "start": tuple(start_base_xyz),
                "goal": tuple(goal_base_xyz),
                "yaw": float(goal_yaw),
            }
        )
        if self.error is not None:
            raise self.error
        return _plan_for(start_base_xyz, goal_base_xyz)


class _FirstCallBlockingBackend(_FakeBackend):
    """只阻塞第一代规划，用于验证旧结果不会越代发布。"""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def plan(
        self,
        *,
        start_base_xyz,
        goal_base_xyz,
        goal_yaw,
    ) -> PCTBackendPlan:
        call_index = len(self.calls)
        self.calls.append(
            {
                "start": tuple(start_base_xyz),
                "goal": tuple(goal_base_xyz),
                "yaw": float(goal_yaw),
            }
        )
        if call_index == 0:
            self.started.set()
            if not self.release.wait(timeout=2.0):
                raise TimeoutError("测试未释放第一代 PCT 规划")
        return _plan_for(start_base_xyz, goal_base_xyz)


class _CancellableBlockingBackend(_FirstCallBlockingBackend):
    """记录 adapter 的显式取消请求。"""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_count = 0

    def cancel_current_plan(self) -> None:
        self.cancel_count += 1


class _InterruptibleFirstCallBackend(_FakeBackend):
    """第一代一直阻塞到显式取消，第二代正常返回。"""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.cancel_requested = threading.Event()
        self.cancel_count = 0

    def plan(
        self,
        *,
        start_base_xyz,
        goal_base_xyz,
        goal_yaw,
    ) -> PCTBackendPlan:
        call_index = len(self.calls)
        self.calls.append(
            {
                "start": tuple(start_base_xyz),
                "goal": tuple(goal_base_xyz),
                "yaw": float(goal_yaw),
            }
        )
        if call_index == 0:
            self.started.set()
            if not self.cancel_requested.wait(timeout=2.0):
                raise TimeoutError("测试没有取消第一代 native PCT 规划")
            raise PCTBackendError("第一代 native PCT 规划已取消")
        return _plan_for(start_base_xyz, goal_base_xyz)

    def cancel_current_plan(self) -> None:
        self.cancel_count += 1
        self.cancel_requested.set()


class _CountingCancellableBackend(_FakeBackend):
    """记录懒初始化完成后补发的取消，不阻塞正常新代规划。"""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_count = 0

    def cancel_current_plan(self) -> None:
        self.cancel_count += 1


class _PrepareBarrierBackend(_FakeBackend):
    """把第一代卡在 prepare/reset，验证取消握手不会被随后清除。"""

    def __init__(self) -> None:
        super().__init__()
        self.prepare_started = threading.Event()
        self.release_prepare = threading.Event()
        self.prepare_count = 0
        self.cancel_count = 0
        self.prepared = False

    def prepare_plan(self, cancel_event: threading.Event) -> None:
        prepare_index = self.prepare_count
        self.prepare_count += 1
        self.prepared = False
        if prepare_index == 0:
            self.prepare_started.set()
            if not self.release_prepare.wait(timeout=2.0):
                raise TimeoutError("测试没有释放 backend prepare barrier")
        if cancel_event.is_set():
            self.cancel_current_plan()
            raise PCTBackendError("prepare 期间到达取消")
        self.prepared = True

    def plan(
        self,
        *,
        start_base_xyz,
        goal_base_xyz,
        goal_yaw,
    ) -> PCTBackendPlan:
        if not self.prepared:
            raise AssertionError("node 必须先完成 prepare_plan 握手")
        self.prepared = False
        return super().plan(
            start_base_xyz=start_base_xyz,
            goal_base_xyz=goal_base_xyz,
            goal_yaw=goal_yaw,
        )

    def cancel_current_plan(self) -> None:
        self.cancel_count += 1


@pytest.fixture
def node_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """创建使用固定非零仿真时间的节点，并统一回收 worker。"""

    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init()
    nodes: list[PCTROS2Adapter] = []

    def factory(
        backend: _FakeBackend | None,
        *,
        backend_factory=create_global_planner_backend,
    ):
        node = PCTROS2Adapter(
            backend=backend,
            backend_factory=backend_factory,
        )
        node._clock_now_ns = lambda: 10_000_000_000
        path_capture = _CapturePublisher()
        scan_path_capture = _CapturePublisher()
        status_capture = _CapturePublisher()
        node._path_publisher = path_capture
        node._scan_path_publisher = scan_path_capture
        node._status_publisher = status_capture
        nodes.append(node)
        return node, path_capture, status_capture

    try:
        yield factory
    finally:
        for node in nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _odometry(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.30,
    stamp_sec: int = 10,
    stamp_nanosec: int = 0,
    frame_id: str = "world",
) -> Odometry:
    message = Odometry()
    message.header.stamp = Time(sec=stamp_sec, nanosec=stamp_nanosec)
    message.header.frame_id = frame_id
    message.child_frame_id = "base_link"
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.position.z = z
    message.pose.pose.orientation.w = 1.0
    return message


def _goal(
    *,
    x: float = 1.0,
    y: float = 0.0,
    z: float = 0.30,
    yaw: float = 0.0,
    stamp_sec: int = 10,
    stamp_nanosec: int = 0,
    frame_id: str = "world",
) -> PoseStamped:
    message = PoseStamped()
    message.header.stamp = Time(sec=stamp_sec, nanosec=stamp_nanosec)
    message.header.frame_id = frame_id
    message.pose.position.x = x
    message.pose.position.y = y
    message.pose.position.z = z
    message.pose.orientation.z = math.sin(0.5 * yaw)
    message.pose.orientation.w = math.cos(0.5 * yaw)
    return message


def _plan_for(start_base_xyz, goal_base_xyz) -> PCTBackendPlan:
    start = tuple(float(value) for value in start_base_xyz)
    goal = tuple(float(value) for value in goal_base_xyz)
    return PCTBackendPlan(
        points_xyz=(
            (start[0], start[1], start[2] - 0.30),
            (goal[0], goal[1], goal[2] - 0.30),
        ),
        metadata={"height_semantics": "ground_height"},
    )


def _command(
    command: int,
    *,
    goal_id: int = 7,
    request_id: int = 1,
    goal: PoseStamped | None = None,
    expected_path_stamp_ns: int = 0,
    reason: str = "test",
) -> PCTPlanningCommand.Request:
    """构造同一 world/仿真时间域的 typed PCT command。"""

    request = PCTPlanningCommand.Request()
    request.header.stamp = Time(sec=10)
    request.header.frame_id = "world"
    request.command = int(command)
    request.goal_id = int(goal_id)
    request.request_id = int(request_id)
    request.goal = _goal() if goal is None else goal
    if expected_path_stamp_ns > 0:
        request.expected_path_stamp = Time(
            sec=expected_path_stamp_ns // 1_000_000_000,
            nanosec=expected_path_stamp_ns % 1_000_000_000,
        )
    request.reason = reason
    return request


def _call_command(
    node: PCTROS2Adapter,
    request: PCTPlanningCommand.Request,
) -> PCTPlanningCommand.Response:
    return node._command_callback(request, PCTPlanningCommand.Response())


def _poll_until_idle(node: PCTROS2Adapter, *, timeout_sec: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        node._poll_planning_result()
        if node._future is None and node._pending_plan is None:
            return
        time.sleep(0.005)
    raise AssertionError("PCT worker 未在测试时限内完成")


def _stamp_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def test_node_forces_sim_time_and_declares_required_qos(
    node_factory,
) -> None:
    node, _paths, _statuses = node_factory(_FakeBackend())
    assert node.get_parameter("use_sim_time").value is True

    path_info = node.get_publishers_info_by_topic("/pct/global_path")
    scan_path_info = node.get_publishers_info_by_topic("/initial_path")
    status_info = node.get_publishers_info_by_topic("/pct/planning_status")
    odometry_info = node.get_subscriptions_info_by_topic("/body_pose")
    goal_info = node.get_subscriptions_info_by_topic("/pct/goal")
    assert len(path_info) == len(scan_path_info) == len(status_info) == 1
    assert len(odometry_info) == len(goal_info) == 1
    for endpoint in (*path_info, *scan_path_info, *status_info):
        assert endpoint.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert endpoint.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert (
        odometry_info[0].qos_profile.reliability
        == ReliabilityPolicy.BEST_EFFORT
    )
    assert odometry_info[0].qos_profile.durability == DurabilityPolicy.VOLATILE
    assert goal_info[0].qos_profile.reliability == ReliabilityPolicy.RELIABLE
    assert goal_info[0].qos_profile.durability == DurabilityPolicy.VOLATILE
    services = dict(node.get_service_names_and_types())
    assert services["/pct/planning_command"] == [
        "scan_planner_msgs/srv/PCTPlanningCommand"
    ]


def test_source_and_scan_path_outputs_preserve_identical_generations(
    node_factory,
) -> None:
    node, source_paths, _statuses = node_factory(_FakeBackend())
    scan_paths = node._scan_path_publisher

    node._odometry_callback(_odometry())
    node._goal_callback(_goal())
    _poll_until_idle(node)

    assert len(source_paths.messages) == len(scan_paths.messages) == 2
    for source, scan in zip(source_paths.messages, scan_paths.messages):
        assert source is scan
        assert source.header.stamp == scan.header.stamp
        assert source.header.frame_id == scan.header.frame_id == "world"
        assert source.poses == scan.poses
    assert source_paths.messages[0].poses == []
    assert len(source_paths.messages[1].poses) == 2


def test_backend_assets_preload_before_first_goal(node_factory) -> None:
    backend = _FakeBackend()
    factory_started = threading.Event()
    release_factory = threading.Event()

    def blocking_factory(_config: object) -> _FakeBackend:
        factory_started.set()
        if not release_factory.wait(timeout=2.0):
            raise TimeoutError("测试没有释放 PCT backend 预热")
        return backend

    node, paths, statuses = node_factory(
        None,
        backend_factory=blocking_factory,
    )
    assert factory_started.wait(timeout=1.0)
    assert backend.calls == []

    release_factory.set()
    node._odometry_callback(_odometry())
    node._goal_callback(_goal())
    _poll_until_idle(node)

    assert len(backend.calls) == 1
    assert len([message for message in paths.messages if message.poses]) == 1
    assert statuses.messages[-1].state == PCTPlanningStatus.SUCCEEDED


def test_node_reads_complete_coordinate_transform_parameters(
    node_factory,
) -> None:
    node, _paths, _statuses = node_factory(_FakeBackend())
    values = {
        "planner.pct_offset_x": 1.1,
        "planner.pct_offset_y": -2.2,
        "planner.pct_offset_z": 3.3,
        "planner.pct_scale_x": 0.7,
        "planner.pct_scale_y": -1.2,
        "planner.pct_scale_z": 1.4,
        "planner.pct_rotation_x_rad": 0.1,
        "planner.pct_rotation_y_rad": -0.2,
        "planner.pct_rotation_z_rad": 0.3,
    }
    results = node.set_parameters(
        [Parameter(name, value=value) for name, value in values.items()]
    )
    assert all(result.successful for result in results)

    config = node._read_backend_config()
    for parameter_name, expected in values.items():
        field_name = parameter_name.removeprefix("planner.")
        assert getattr(config, field_name) == pytest.approx(expected)


def test_node_reads_explicit_upstream_backend_parameters(node_factory) -> None:
    node, _paths, _statuses = node_factory(_FakeBackend())
    results = node.set_parameters(
        [
            Parameter("planner.backend_kind", value="upstream"),
            Parameter(
                "planner.upstream_source_root",
                value="external/PCT_planner",
            ),
            Parameter("planner.upstream_use_quintic", value=False),
            Parameter("planner.upstream_max_heading_rate", value=4.5),
            Parameter(
                "planner.upstream_astar_step_cost_weight",
                value=0.42,
            ),
            Parameter(
                "planner.upstream_body_clearance_enabled",
                value=True,
            ),
            Parameter(
                "planner.upstream_body_clearance_radius_m",
                value=0.75,
            ),
            Parameter(
                "planner.upstream_body_clearance_maximum_cost",
                value=18.0,
            ),
            Parameter(
                "planner.upstream_body_clearance_power",
                value=1.8,
            ),
            Parameter(
                "planner.upstream_stair_profile_path",
                value="configs/navigation/pct_multifloor_stair_profile.json",
            ),
            Parameter(
                "planner.upstream_stair_profile_match_tolerance_m",
                value=0.55,
            ),
            Parameter(
                "planner.upstream_tomogram_path",
                value="source/scene/multifloor/mutifloor.pickle",
            ),
        ]
    )
    assert all(result.successful for result in results)

    config = node._read_backend_config()

    assert config.backend_kind == "upstream"
    assert config.upstream_source_root == (
        Path(__file__).resolve().parents[4] / "external/PCT_planner"
    ).resolve()
    assert config.upstream_use_quintic is False
    assert config.upstream_max_heading_rate == pytest.approx(4.5)
    assert config.upstream_astar_step_cost_weight == pytest.approx(0.42)
    assert config.upstream_body_clearance_enabled is True
    assert config.upstream_body_clearance_radius_m == pytest.approx(0.75)
    assert config.upstream_body_clearance_maximum_cost == pytest.approx(18.0)
    assert config.upstream_body_clearance_power == pytest.approx(1.8)
    assert config.upstream_stair_profile_path == (
        Path(__file__).resolve().parents[4]
        / "configs/navigation/pct_multifloor_stair_profile.json"
    ).resolve()
    assert config.upstream_stair_profile_match_tolerance_m == pytest.approx(
        0.55
    )
    assert config.tomogram_path.name == "mutifloor.pickle"


@pytest.mark.parametrize(
    "parameter_name",
    (
        "planner.pct_scale_x",
        "planner.pct_scale_y",
        "planner.pct_scale_z",
    ),
)
def test_node_rejects_zero_coordinate_scale(
    node_factory,
    parameter_name: str,
) -> None:
    node, _paths, _statuses = node_factory(_FakeBackend())
    result = node.set_parameters([Parameter(parameter_name, value=0.0)])[0]
    assert result.successful

    with pytest.raises(ValueError, match="不能为零"):
        node._read_backend_config()


def test_fresh_odometry_and_goal_publish_ground_path_and_matching_status(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._goal_callback(_goal(yaw=0.4))
    _poll_until_idle(node)

    assert len(backend.calls) == 1
    assert len(paths.messages) == 2
    invalidation, result = paths.messages
    assert invalidation.poses == []
    assert result.header.frame_id == "world"
    assert len(result.poses) == 2
    assert [pose.pose.position.z for pose in result.poses] == [0.0, 0.0]
    assert all(pose.header.frame_id == "world" for pose in result.poses)
    assert all(
        _stamp_ns(pose.header.stamp) == _stamp_ns(result.header.stamp)
        for pose in result.poses
    )
    success = statuses.messages[-1]
    assert success.state == PCTPlanningStatus.SUCCEEDED
    assert success.plan_id == 1
    assert success.goal_id == 10_000_000_000
    assert success.request_id == 0
    assert success.command == PCTPlanningStatus.COMMAND_PLAN
    assert success.has_active_goal is True
    assert success.path_point_count == 2
    assert _stamp_ns(success.header.stamp) == _stamp_ns(result.header.stamp)
    assert _stamp_ns(success.path_stamp) == _stamp_ns(result.header.stamp)
    assert result.poses[-1].pose.orientation.z == pytest.approx(
        math.sin(0.2)
    )


def test_ground_path_height_is_not_incremented_by_body_height(
    node_factory,
) -> None:
    class NonzeroGroundBackend(_FakeBackend):
        def plan(self, *, start_base_xyz, goal_base_xyz, goal_yaw):
            super().plan(
                start_base_xyz=start_base_xyz,
                goal_base_xyz=goal_base_xyz,
                goal_yaw=goal_yaw,
            )
            return PCTBackendPlan(
                points_xyz=((0.0, 0.0, 0.17), (1.0, 0.0, 0.23)),
                metadata={"height_semantics": "ground_height"},
            )

    node, paths, statuses = node_factory(NonzeroGroundBackend())
    node._odometry_callback(_odometry(z=0.47))
    node._goal_callback(_goal(z=0.53))
    _poll_until_idle(node)

    assert statuses.messages[-1].state == PCTPlanningStatus.SUCCEEDED
    result = paths.messages[-1]
    assert [pose.pose.position.z for pose in result.poses] == pytest.approx(
        [0.17, 0.23]
    )


@pytest.mark.parametrize(
    "odometry",
    [
        None,
        _odometry(stamp_sec=9),
        _odometry(stamp_sec=11),
        _odometry(frame_id="sensor"),
    ],
)
def test_missing_stale_future_or_wrong_frame_odometry_never_calls_backend(
    node_factory,
    odometry: Odometry | None,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    if odometry is not None:
        node._odometry_callback(odometry)
    node._goal_callback(_goal())
    node._poll_planning_result()

    assert backend.calls == []
    assert len(paths.messages) == 1
    assert paths.messages[0].poses == []
    assert statuses.messages[-1].state == PCTPlanningStatus.WAITING_FOR_ODOMETRY


@pytest.mark.parametrize(
    ("error", "expected_state"),
    [
        (PCTNoPathError("严格 gateway 下无路径"), PCTPlanningStatus.NO_PATH),
        (RuntimeError("资产损坏"), PCTPlanningStatus.ERROR),
    ],
)
def test_no_path_and_error_publish_new_empty_generation(
    node_factory,
    error: Exception,
    expected_state: int,
) -> None:
    node, paths, statuses = node_factory(_FakeBackend(error=error))
    node._odometry_callback(_odometry())
    node._goal_callback(_goal())
    _poll_until_idle(node)

    assert len(paths.messages) == 2
    assert all(message.poses == [] for message in paths.messages)
    assert _stamp_ns(paths.messages[1].header.stamp) > _stamp_ns(
        paths.messages[0].header.stamp
    )
    assert statuses.messages[-1].state == expected_state
    assert _stamp_ns(statuses.messages[-1].path_stamp) == _stamp_ns(
        paths.messages[-1].header.stamp
    )


def test_duplicate_goal_is_idempotent_and_does_not_consume_plan_id(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    goal = _goal()
    node._goal_callback(goal)
    _poll_until_idle(node)
    published_count = len(paths.messages)
    status_count = len(statuses.messages)

    node._goal_callback(goal)
    node._poll_planning_result()

    assert len(backend.calls) == 1
    assert len(paths.messages) == published_count
    assert len(statuses.messages) == status_count
    assert node._plan_id == 1


def test_new_goal_supersedes_blocked_result_and_only_latest_path_is_published(
    node_factory,
) -> None:
    backend = _FirstCallBlockingBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._goal_callback(_goal(x=1.0))
    assert backend.started.wait(timeout=1.0)
    node._goal_callback(_goal(x=2.0, stamp_nanosec=1))
    backend.release.set()
    _poll_until_idle(node)

    assert len(backend.calls) == 2
    nonempty_paths = [message for message in paths.messages if message.poses]
    assert len(nonempty_paths) == 1
    assert nonempty_paths[0].poses[-1].pose.position.x == pytest.approx(2.0)
    successes = [
        message
        for message in statuses.messages
        if message.state == PCTPlanningStatus.SUCCEEDED
    ]
    assert len(successes) == 1
    assert successes[0].plan_id == 2


def test_cancel_during_lazy_backend_factory_skips_old_job_and_runs_new_goal(
    node_factory,
) -> None:
    backend = _CountingCancellableBackend()
    node, paths, statuses = node_factory(_FakeBackend())
    factory_started = threading.Event()
    release_factory = threading.Event()

    def blocking_factory(_config: object) -> _CountingCancellableBackend:
        factory_started.set()
        if not release_factory.wait(timeout=2.0):
            raise TimeoutError("测试没有释放 PCT backend factory")
        return backend

    # 只为覆盖首次 backend 尚未赋值的窗口；构造节点时的测试 backend 未运行。
    node._backend = None
    node._backend_config = object()
    node._backend_factory = blocking_factory
    node._odometry_callback(_odometry())
    node._goal_callback(_goal(x=1.0))
    assert factory_started.wait(timeout=1.0)

    node._goal_callback(_goal(x=2.0, stamp_nanosec=1))
    assert node._future_job is not None
    assert node._future_job.cancel_event.is_set()
    release_factory.set()
    _poll_until_idle(node)

    assert backend.cancel_count == 1
    assert len(backend.calls) == 1
    assert backend.calls[0]["goal"] == pytest.approx((2.0, 0.0, 0.30))
    nonempty_paths = [message for message in paths.messages if message.poses]
    assert len(nonempty_paths) == 1
    assert nonempty_paths[0].poses[-1].pose.position.x == pytest.approx(2.0)
    successes = [
        message
        for message in statuses.messages
        if message.state == PCTPlanningStatus.SUCCEEDED
    ]
    assert len(successes) == 1
    assert successes[0].plan_id == 2


def test_native_running_plan_is_interrupted_by_typed_replan_and_new_generation_wins(
    node_factory,
) -> None:
    backend = _InterruptibleFirstCallBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    initial = _call_command(
        node,
        _command(PCTPlanningCommand.Request.COMMAND_PLAN),
    )
    assert backend.started.wait(timeout=1.0)

    replan = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_REPLAN,
            request_id=2,
            expected_path_stamp_ns=_stamp_ns(initial.tombstone_stamp),
        ),
    )
    assert replan.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
    )
    assert replan.plan_id == 2
    _poll_until_idle(node)

    assert backend.cancel_count == 1
    assert len(backend.calls) == 2
    nonempty_paths = [message for message in paths.messages if message.poses]
    assert len(nonempty_paths) == 1
    successes = [
        message
        for message in statuses.messages
        if message.state == PCTPlanningStatus.SUCCEEDED
    ]
    assert len(successes) == 1
    assert successes[0].plan_id == 2
    assert successes[0].command == PCTPlanningStatus.COMMAND_REPLAN


def test_typed_replan_during_prepare_barrier_reissues_cancel_before_new_plan(
    node_factory,
) -> None:
    backend = _PrepareBarrierBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    initial = _call_command(
        node,
        _command(PCTPlanningCommand.Request.COMMAND_PLAN),
    )
    assert backend.prepare_started.wait(timeout=1.0)

    replan = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_REPLAN,
            request_id=2,
            expected_path_stamp_ns=_stamp_ns(initial.tombstone_stamp),
        ),
    )
    assert replan.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
    )
    backend.release_prepare.set()
    _poll_until_idle(node)

    # 第一次由 node 的即时抢占发出，第二次由 prepare 的 event 复查补发。
    assert backend.cancel_count == 2
    assert backend.prepare_count == 2
    assert len(backend.calls) == 1
    assert len([message for message in paths.messages if message.poses]) == 1
    successes = [
        message
        for message in statuses.messages
        if message.state == PCTPlanningStatus.SUCCEEDED
    ]
    assert len(successes) == 1
    assert successes[0].plan_id == 2
    assert successes[0].command == PCTPlanningStatus.COMMAND_REPLAN


def test_new_stamp_same_geometry_starts_a_new_planning_generation(
    node_factory,
) -> None:
    backend = _FirstCallBlockingBackend()
    node, paths, _statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._goal_callback(_goal(x=1.0, stamp_sec=10))
    assert backend.started.wait(timeout=1.0)

    node._goal_callback(
        _goal(x=1.0, stamp_sec=10, stamp_nanosec=100_000_000)
    )
    backend.release.set()
    _poll_until_idle(node)

    assert len(backend.calls) == 2
    assert node._plan_id == 2
    assert len([message for message in paths.messages if message.poses]) == 1


def test_old_goal_stamp_cannot_supersede_newer_generation(node_factory) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._goal_callback(_goal(x=2.0, stamp_sec=10, stamp_nanosec=100))
    _poll_until_idle(node)
    path_count = len(paths.messages)
    status_count = len(statuses.messages)

    node._goal_callback(_goal(x=1.0, stamp_sec=10, stamp_nanosec=99))

    assert len(backend.calls) == 1
    assert node._plan_id == 1
    assert len(paths.messages) == path_count
    assert len(statuses.messages) == status_count


def test_same_stamp_conflicting_goal_invalidates_generation(node_factory) -> None:
    backend = _FirstCallBlockingBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._goal_callback(_goal(x=1.0, stamp_sec=10, stamp_nanosec=100))
    assert backend.started.wait(timeout=1.0)

    node._goal_callback(_goal(x=2.0, stamp_sec=10, stamp_nanosec=100))
    backend.release.set()
    _poll_until_idle(node)

    assert node._plan_id == 2
    assert node._navigation_goal is None
    assert all(not message.poses for message in paths.messages)
    assert statuses.messages[-1].state == PCTPlanningStatus.ERROR
    assert "payload 冲突" in statuses.messages[-1].message


def test_invalid_new_goal_supersedes_active_job_and_clears_old_path(
    node_factory,
) -> None:
    backend = _FirstCallBlockingBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._goal_callback(_goal(x=1.0))
    assert backend.started.wait(timeout=1.0)

    node._goal_callback(_goal(x=2.0, frame_id="sensor"))
    backend.release.set()
    _poll_until_idle(node)

    assert node._plan_id == 2
    assert node._navigation_goal is None
    assert all(message.poses == [] for message in paths.messages)
    assert statuses.messages[-1].state == PCTPlanningStatus.ERROR
    assert statuses.messages[-1].plan_id == 2
    assert not any(
        message.state == PCTPlanningStatus.SUCCEEDED
        for message in statuses.messages
    )


def test_startup_generation_clears_stale_path_after_adapter_restart(
    node_factory,
) -> None:
    node, paths, statuses = node_factory(_FakeBackend())

    node._publish_startup_generation_if_ready()
    node._publish_startup_generation_if_ready()

    assert len(paths.messages) == 1
    assert paths.messages[0].poses == []
    assert statuses.messages[-1].state == PCTPlanningStatus.IDLE
    assert statuses.messages[-1].plan_id == 0
    assert _stamp_ns(statuses.messages[-1].path_stamp) == _stamp_ns(
        paths.messages[0].header.stamp
    )


def test_future_goal_is_rejected_without_poisoning_corrected_resend(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())

    node._goal_callback(_goal(x=1.0, stamp_sec=11))
    assert backend.calls == []
    assert len(paths.messages) == 1
    assert paths.messages[-1].poses == []
    assert statuses.messages[-1].state == PCTPlanningStatus.ERROR
    assert node._plan_id == 1

    node._goal_callback(_goal(x=1.0, stamp_sec=10))
    _poll_until_idle(node)

    assert len(backend.calls) == 1
    assert node._plan_id == 2
    assert statuses.messages[-1].state == PCTPlanningStatus.SUCCEEDED
    assert _stamp_ns(paths.messages[-1].header.stamp) < 10_100_000_000


def test_future_odometry_does_not_poison_corrected_sim_time_sample(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, _paths, statuses = node_factory(backend)

    node._odometry_callback(_odometry(stamp_sec=11))
    assert node._latest_odometry is None
    node._goal_callback(_goal())
    assert statuses.messages[-1].state == PCTPlanningStatus.WAITING_FOR_ODOMETRY

    node._odometry_callback(_odometry(stamp_sec=10))
    _poll_until_idle(node)

    assert len(backend.calls) == 1
    assert statuses.messages[-1].state == PCTPlanningStatus.SUCCEEDED


@pytest.mark.parametrize("metadata", ({}, {"height_semantics": "base_height"}))
def test_backend_must_declare_ground_height_semantics(
    node_factory,
    metadata: dict[str, str],
) -> None:
    class MetadataBackend(_FakeBackend):
        def plan(self, *, start_base_xyz, goal_base_xyz, goal_yaw):
            plan = super().plan(
                start_base_xyz=start_base_xyz,
                goal_base_xyz=goal_base_xyz,
                goal_yaw=goal_yaw,
            )
            return PCTBackendPlan(points_xyz=plan.points_xyz, metadata=metadata)

    backend = MetadataBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._goal_callback(_goal())
    _poll_until_idle(node)

    assert all(message.poses == [] for message in paths.messages)
    assert statuses.messages[-1].state == PCTPlanningStatus.ERROR
    assert "height_semantics=ground_height" in statuses.messages[-1].message


def test_too_short_success_path_is_converted_to_error_and_empty_path(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._goal_callback(_goal(x=0.01))
    _poll_until_idle(node)

    assert all(message.poses == [] for message in paths.messages)
    assert statuses.messages[-1].state == PCTPlanningStatus.ERROR
    assert "不能发布" in statuses.messages[-1].message


def test_typed_plan_ack_and_status_keep_active_goal_after_success(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())

    accepted = _call_command(
        node,
        _command(PCTPlanningCommand.Request.COMMAND_PLAN),
    )
    tombstone_ns = _stamp_ns(accepted.tombstone_stamp)
    assert accepted.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
    )
    assert accepted.plan_id == 1
    assert accepted.goal_id == 7
    assert accepted.request_id == 1
    assert accepted.has_active_goal is True
    assert tombstone_ns > 0

    _poll_until_idle(node)

    assert len(backend.calls) == 1
    assert len(paths.messages) == 2
    assert paths.messages[0].poses == []
    assert _stamp_ns(paths.messages[1].header.stamp) > tombstone_ns
    assert node._navigation_goal is not None
    assert node._navigation_goal.goal_id == 7
    success = statuses.messages[-1]
    assert success.state == PCTPlanningStatus.SUCCEEDED
    assert success.goal_id == 7
    assert success.request_id == 1
    assert success.command == PCTPlanningStatus.COMMAND_PLAN
    assert success.has_active_goal is True
    assert _stamp_ns(success.active_path_stamp) == _stamp_ns(
        paths.messages[-1].header.stamp
    )


def test_exact_command_retry_returns_duplicate_without_side_effects(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    request = _command(PCTPlanningCommand.Request.COMMAND_PLAN)
    first = _call_command(node, request)
    _poll_until_idle(node)
    counts = (len(backend.calls), len(paths.messages), len(statuses.messages))

    duplicate = _call_command(node, request)

    assert duplicate.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_DUPLICATE
    )
    assert duplicate.plan_id == first.plan_id
    assert duplicate.tombstone_stamp == first.tombstone_stamp
    assert (len(backend.calls), len(paths.messages), len(statuses.messages)) == counts


def test_same_request_id_conflict_and_old_request_are_side_effect_free(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    accepted = _call_command(
        node,
        _command(PCTPlanningCommand.Request.COMMAND_PLAN, request_id=2),
    )
    _poll_until_idle(node)
    counts = (len(backend.calls), len(paths.messages), len(statuses.messages))

    conflict = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_PLAN,
            request_id=2,
            goal=_goal(x=2.0),
        ),
    )
    stale = _call_command(
        node,
        _command(PCTPlanningCommand.Request.COMMAND_PLAN, request_id=1),
    )

    assert accepted.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
    )
    assert conflict.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_CONFLICT
    )
    assert stale.disposition == PCTPlanningCommand.Response.DISPOSITION_STALE
    assert (len(backend.calls), len(paths.messages), len(statuses.messages)) == counts


def test_replan_same_goal_uses_latest_odometry_and_new_path_generation(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    _call_command(node, _command(PCTPlanningCommand.Request.COMMAND_PLAN))
    _poll_until_idle(node)
    first_path_stamp_ns = _stamp_ns(paths.messages[-1].header.stamp)

    node._odometry_callback(_odometry(x=0.25))
    replan = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_REPLAN,
            request_id=2,
            expected_path_stamp_ns=first_path_stamp_ns,
        ),
    )
    replan_tombstone_ns = _stamp_ns(replan.tombstone_stamp)
    assert replan.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
    )
    assert replan.plan_id == 2
    assert replan_tombstone_ns > first_path_stamp_ns
    _poll_until_idle(node)

    assert len(backend.calls) == 2
    assert backend.calls[-1]["start"] == pytest.approx((0.25, 0.0, 0.30))
    assert backend.calls[-1]["goal"] == pytest.approx((1.0, 0.0, 0.30))
    assert _stamp_ns(paths.messages[-1].header.stamp) > replan_tombstone_ns
    success = statuses.messages[-1]
    assert success.request_id == 2
    assert success.command == PCTPlanningStatus.COMMAND_REPLAN
    assert success.has_active_goal is True


def test_replan_rejects_missing_goal_changed_goal_and_stale_path(
    node_factory,
) -> None:
    backend = _FakeBackend()
    node, paths, statuses = node_factory(backend)
    missing = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_REPLAN,
            expected_path_stamp_ns=1,
        ),
    )
    assert missing.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_REJECTED
    )

    node._odometry_callback(_odometry())
    initial = _call_command(
        node,
        _command(PCTPlanningCommand.Request.COMMAND_PLAN),
    )
    _poll_until_idle(node)
    current_path_stamp_ns = _stamp_ns(paths.messages[-1].header.stamp)
    counts = (len(backend.calls), len(paths.messages), len(statuses.messages))

    changed = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_REPLAN,
            request_id=2,
            goal=_goal(x=2.0),
            expected_path_stamp_ns=current_path_stamp_ns,
        ),
    )
    stale = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_REPLAN,
            request_id=2,
            expected_path_stamp_ns=_stamp_ns(initial.tombstone_stamp),
        ),
    )
    assert changed.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_CONFLICT
    )
    assert stale.disposition == PCTPlanningCommand.Response.DISPOSITION_STALE
    assert (len(backend.calls), len(paths.messages), len(statuses.messages)) == counts


def test_replan_rejects_same_yaw_with_different_wire_quaternion(
    node_factory,
) -> None:
    """几何等价的 q/-q 也不能替换活动目标的完整 wire 快照。"""

    backend = _FakeBackend()
    node, paths, _statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    node._command_callback(
        _command(PCTPlanningCommand.Request.COMMAND_PLAN),
        PCTPlanningCommand.Response(),
    )
    _poll_until_idle(node)
    current_path_stamp_ns = _stamp_ns(paths.messages[-1].header.stamp)
    equivalent = _goal()
    equivalent.pose.orientation.z *= -1.0
    equivalent.pose.orientation.w *= -1.0

    response = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_REPLAN,
            request_id=2,
            goal=equivalent,
            expected_path_stamp_ns=current_path_stamp_ns,
        ),
    )

    assert response.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_CONFLICT
    )
    assert len(backend.calls) == 1


def test_cancel_ack_invalidates_worker_and_duplicate_cancel_is_idempotent(
    node_factory,
) -> None:
    backend = _CancellableBlockingBackend()
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    plan = _call_command(
        node,
        _command(PCTPlanningCommand.Request.COMMAND_PLAN),
    )
    assert backend.started.wait(timeout=1.0)
    cancel_request = _command(
        PCTPlanningCommand.Request.COMMAND_CANCEL,
        request_id=2,
        expected_path_stamp_ns=_stamp_ns(plan.tombstone_stamp),
    )

    cancelled = _call_command(node, cancel_request)

    assert cancelled.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
    )
    assert cancelled.has_active_goal is False
    assert backend.cancel_count == 1
    assert node._navigation_goal is None
    assert node._pending_plan is None
    assert _stamp_ns(cancelled.tombstone_stamp) > _stamp_ns(plan.tombstone_stamp)
    duplicate = _call_command(node, cancel_request)
    assert duplicate.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_DUPLICATE
    )
    assert len(paths.messages) == 2
    assert backend.cancel_count == 1

    backend.release.set()
    _poll_until_idle(node)
    assert all(not path.poses for path in paths.messages)
    assert statuses.messages[-1].state == PCTPlanningStatus.IDLE
    assert statuses.messages[-1].command == PCTPlanningStatus.COMMAND_CANCEL
    assert statuses.messages[-1].has_active_goal is False


def test_failure_retains_goal_and_allows_typed_replan(
    node_factory,
) -> None:
    backend = _FakeBackend(error=PCTNoPathError("blocked"))
    node, paths, statuses = node_factory(backend)
    node._odometry_callback(_odometry())
    _call_command(node, _command(PCTPlanningCommand.Request.COMMAND_PLAN))
    _poll_until_idle(node)

    assert statuses.messages[-1].state == PCTPlanningStatus.NO_PATH
    assert statuses.messages[-1].has_active_goal is True
    assert node._navigation_goal is not None
    failure_stamp_ns = _stamp_ns(paths.messages[-1].header.stamp)

    backend.error = None
    accepted = _call_command(
        node,
        _command(
            PCTPlanningCommand.Request.COMMAND_REPLAN,
            request_id=2,
            expected_path_stamp_ns=failure_stamp_ns,
        ),
    )
    assert accepted.disposition == (
        PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
    )
    _poll_until_idle(node)
    assert statuses.messages[-1].state == PCTPlanningStatus.SUCCEEDED
    assert statuses.messages[-1].request_id == 2


def _trajectory_pair_fixture(
    trajectory_id: int,
    *,
    header_nanosec: int,
) -> tuple[Bspline, ScanPlanningStatus]:
    """
    @brief 构造一组完整 identity 一致的 B-spline 与 SCAN 状态
    @param trajectory_id 轨迹编号
    @param header_nanosec B-spline Header 的纳秒字段
    @return 可供在线探针配对测试使用的消息二元组
    """

    spline = Bspline()
    spline.reference_path_stamp = Time(sec=4, nanosec=40_000_000)
    spline.header.stamp = Time(sec=6, nanosec=header_nanosec)
    spline.start_time = Time(sec=6, nanosec=header_nanosec)
    spline.traj_id = trajectory_id

    status = ScanPlanningStatus()
    status.reference_path_stamp = spline.reference_path_stamp
    status.bspline_header_stamp = spline.header.stamp
    status.trajectory_start_time = spline.start_time
    status.trajectory_id = trajectory_id
    return spline, status


def test_live_probe_pairs_scan_messages_by_full_trajectory_identity() -> None:
    first_spline, first_status = _trajectory_pair_fixture(
        7,
        header_nanosec=100,
    )
    latest_spline, latest_status = _trajectory_pair_fixture(
        8,
        header_nanosec=200,
    )

    pair = _matching_scan_trajectory_pair(
        [first_spline, latest_spline],
        [latest_status, first_status],
    )

    assert pair == (latest_spline, latest_status)


def test_live_probe_waits_when_scan_identity_has_no_exact_pair() -> None:
    spline, _status = _trajectory_pair_fixture(
        9,
        header_nanosec=300,
    )
    _other_spline, other_status = _trajectory_pair_fixture(
        10,
        header_nanosec=400,
    )

    assert _matching_scan_trajectory_pair([spline], [other_status]) is None


def test_live_probe_distinguishes_planning_and_execution_modes() -> None:
    planning_args = SimpleNamespace(
        expect_flat_execution=False,
        expect_crossfloor_execution=False,
    )
    flat_args = SimpleNamespace(
        expect_flat_execution=True,
        expect_crossfloor_execution=False,
    )
    crossfloor_args = SimpleNamespace(
        expect_flat_execution=False,
        expect_crossfloor_execution=True,
    )

    assert not _execution_expected(planning_args)
    assert _expected_execution_goal(planning_args) is None
    assert _execution_expected(flat_args)
    assert _expected_execution_goal(flat_args) is not None
    assert _execution_expected(crossfloor_args)
    crossfloor_goal = _expected_execution_goal(crossfloor_args)
    assert crossfloor_goal is not None
    assert crossfloor_goal[2] > 3.0
