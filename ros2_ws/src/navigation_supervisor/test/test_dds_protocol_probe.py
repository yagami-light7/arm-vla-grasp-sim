"""navigation_supervisor 的 opt-in 真实多节点 DDS 协议探针。"""

from __future__ import annotations

from copy import deepcopy
import os
from threading import Lock, Thread
import time

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
import pytest
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import (
    MultiThreadedExecutor,
    SingleThreadedExecutor,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.serialization import serialize_message
from rosgraph_msgs.msg import Clock
from scan_planner_msgs.msg import (
    Bspline,
    ControllerStatus,
    NavigationStatus,
    PCTPlanningStatus,
    ScanPlanningStatus,
)
from scan_planner_msgs.srv import PCTPlanningCommand
from sensor_msgs.msg import PointCloud2, PointField

from navigation_core import NavigationState
from navigation_supervisor.node import NavigationSupervisorNode


NANOSECONDS_PER_SECOND = 1_000_000_000
INITIAL_TIME_NS = 10 * NANOSECONDS_PER_SECOND
GOAL_STAMP_NS = 1 * NANOSECONDS_PER_SECOND
PATH_STAMP_NS = 4 * NANOSECONDS_PER_SECOND
BSPLINE_STAMP_NS = 9_600_000_000
START_STAMP_NS = 9_500_000_000
RUN_DDS_PROBES = os.environ.get(
    "RUN_NAVIGATION_SUPERVISOR_DDS_PROBES",
) == "1"


pytestmark = pytest.mark.skipif(
    not RUN_DDS_PROBES,
    reason=(
        "真实 DDS probe 需显式设置 "
        "RUN_NAVIGATION_SUPERVISOR_DDS_PROBES=1"
    ),
)


def _time(value_ns: int) -> Time:
    return Time(
        sec=value_ns // NANOSECONDS_PER_SECOND,
        nanosec=value_ns % NANOSECONDS_PER_SECOND,
    )


def _goal() -> PoseStamped:
    message = PoseStamped()
    message.header.stamp = _time(GOAL_STAMP_NS)
    message.header.frame_id = "world"
    message.pose.position.x = 6.0
    message.pose.position.z = 0.3
    message.pose.orientation.w = 1.0
    return message


def _path(
    path_stamp_ns: int = PATH_STAMP_NS,
    *,
    empty: bool = False,
) -> Path:
    message = Path()
    message.header.stamp = _time(path_stamp_ns)
    message.header.frame_id = "world"
    if empty:
        return message
    for x_value in (0.0, 2.0):
        pose = PoseStamped()
        pose.header = deepcopy(message.header)
        pose.pose.position.x = x_value
        pose.pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


def _pct_success(
    *,
    path_stamp_ns: int = PATH_STAMP_NS,
    plan_id: int = 1,
    request_id: int = 1,
    command: int = PCTPlanningStatus.COMMAND_PLAN,
    event_stamp_ns: int = 9_000_000_000,
) -> PCTPlanningStatus:
    message = PCTPlanningStatus()
    message.header.stamp = _time(event_stamp_ns)
    message.header.frame_id = "world"
    message.plan_id = plan_id
    message.goal_id = 7
    message.request_id = request_id
    message.command = command
    message.state = PCTPlanningStatus.SUCCEEDED
    message.goal_stamp = _time(GOAL_STAMP_NS)
    message.path_stamp = _time(path_stamp_ns)
    message.has_active_goal = True
    message.active_goal = _goal()
    message.active_path_stamp = _time(path_stamp_ns)
    message.path_point_count = 2
    message.message = "dds_probe_success"
    return message


def _bspline() -> Bspline:
    message = Bspline()
    message.header.stamp = _time(BSPLINE_STAMP_NS)
    message.header.frame_id = "world"
    message.order = 3
    message.traj_id = 11
    message.start_time = _time(START_STAMP_NS)
    message.reference_path_stamp = _time(PATH_STAMP_NS)
    message.knots = [
        0.0,
        0.0,
        0.0,
        0.0,
        5.0,
        10.0,
        20.0,
        20.0,
        20.0,
        20.0,
    ]
    message.pos_pts = [Point(x=float(index)) for index in range(6)]
    return message


def _scan_trajectory(sequence: int = 1) -> ScanPlanningStatus:
    message = ScanPlanningStatus()
    message.header.stamp = _time(9_700_000_000)
    message.header.frame_id = "world"
    message.status_sequence = sequence
    message.event = ScanPlanningStatus.EVENT_TRAJECTORY_PUBLISHED
    message.state = ScanPlanningStatus.STATE_TRACKING
    message.reference_path_stamp = _time(PATH_STAMP_NS)
    message.trajectory_present = True
    message.bspline_header_stamp = _time(BSPLINE_STAMP_NS)
    message.trajectory_start_time = _time(START_STAMP_NS)
    message.trajectory_id = 11
    message.stop_required = False
    message.global_replan_recommended = False
    message.reason = "dds_probe_trajectory"
    return message


def _scan_failures(sequence: int = 2) -> ScanPlanningStatus:
    message = ScanPlanningStatus()
    message.header.stamp = _time(9_950_000_000)
    message.header.frame_id = "world"
    message.status_sequence = sequence
    message.event = ScanPlanningStatus.EVENT_PLANNING_FAILED
    message.state = ScanPlanningStatus.STATE_PLANNING
    message.reference_path_stamp = _time(PATH_STAMP_NS)
    message.consecutive_planning_failures = 5
    message.stop_required = True
    message.global_replan_recommended = True
    message.reason = "dds_probe_scan_failures"
    return message


def _controller(
    *,
    sequence: int = 1,
    valid: bool = True,
) -> ControllerStatus:
    message = ControllerStatus()
    message.header.stamp = _time(9_900_000_000 if valid else 9_950_000_000)
    message.header.frame_id = "world"
    message.status_sequence = sequence
    message.acceptance_sequence = 1
    message.event = (
        ControllerStatus.EVENT_ACCEPTED
        if valid
        else ControllerStatus.EVENT_INVALIDATED
    )
    message.reference_path_stamp = _time(PATH_STAMP_NS)
    message.bspline_header_stamp = _time(BSPLINE_STAMP_NS)
    message.start_time = _time(START_STAMP_NS)
    message.traj_id = 11
    message.accepted = True
    message.trajectory_valid = valid
    message.state = (
        ControllerStatus.STATE_TRACKING
        if valid
        else ControllerStatus.STATE_EMERGENCY_STOP
    )
    message.reason = "dds_probe_controller"
    return message


def _odometry(stamp_ns: int) -> Odometry:
    message = Odometry()
    message.header.stamp = _time(stamp_ns)
    message.header.frame_id = "world"
    message.child_frame_id = "base_link"
    message.pose.pose.position.z = 0.3
    message.pose.pose.orientation.w = 1.0
    return message


def _cloud(stamp_ns: int) -> PointCloud2:
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
    message.point_step = 12
    message.row_step = 12
    message.data = bytes(12)
    message.is_dense = True
    return message


def _cached_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _probe_domain(offset: int) -> int:
    try:
        base = int(os.environ["ROS_DOMAIN_ID"])
    except (KeyError, ValueError) as exc:
        raise AssertionError(
            "启用 DDS probe 时必须显式设置整数 ROS_DOMAIN_ID"
        ) from exc
    domain_id = base + offset
    if not 0 <= domain_id <= 232:
        raise AssertionError(
            "ROS_DOMAIN_ID 加 probe offset 后必须位于 0..232"
        )
    return domain_id


class _DelayedPCTService(Node):
    """首次响应延迟、第二次返回 DUPLICATE 的真实 ROS 2 service。"""

    def __init__(self, name: str) -> None:
        super().__init__(name, use_global_arguments=False)
        self.payloads: list[bytes] = []
        self.request_ids: list[int] = []
        self._lock = Lock()
        self.create_service(
            PCTPlanningCommand,
            "/pct/planning_command",
            self._callback,
            callback_group=ReentrantCallbackGroup(),
        )

    def _callback(self, request, response):
        payload = bytes(serialize_message(request))
        with self._lock:
            self.payloads.append(payload)
            self.request_ids.append(int(request.request_id))
            occurrence = len(self.payloads)
        if occurrence == 1:
            time.sleep(1.40)
        response.header = deepcopy(request.header)
        response.disposition = (
            PCTPlanningCommand.Response.DISPOSITION_ACCEPTED
            if occurrence == 1
            else PCTPlanningCommand.Response.DISPOSITION_DUPLICATE
        )
        response.plan_id = 2
        response.goal_id = request.goal_id
        response.request_id = request.request_id
        response.tombstone_stamp = _time(5_000_000_000)
        response.has_active_goal = True
        response.active_goal = deepcopy(request.goal)
        response.active_path_stamp = _time(5_000_000_000)
        response.message = "dds_probe_ack"
        return response


class _DDSGraph:
    """管理独立 domain、真实 DDS endpoints、仿真时钟和硬超时。"""

    def __init__(self, *, domain_id: int, multithreaded: bool) -> None:
        if rclpy.ok():
            rclpy.shutdown()
        rclpy.init(
            args=["--ros-args", "-p", "use_sim_time:=true"],
            domain_id=domain_id,
        )
        self.driver = Node(
            f"navigation_supervisor_dds_driver_{domain_id}",
            use_global_arguments=False,
        )
        self.executor = (
            MultiThreadedExecutor(num_threads=4)
            if multithreaded
            else SingleThreadedExecutor()
        )
        self.nodes: list[Node] = []
        self.thread: Thread | None = None
        self.thread_errors: list[BaseException] = []
        self.sim_time_ns = INITIAL_TIME_NS
        cached_qos = _cached_qos()
        sensor_qos = _sensor_qos()
        self.clock_publisher = self.driver.create_publisher(
            Clock,
            "/clock",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )
        self.path_publisher = self.driver.create_publisher(
            Path,
            "/pct/global_path",
            cached_qos,
        )
        self.pct_status_publisher = self.driver.create_publisher(
            PCTPlanningStatus,
            "/pct/planning_status",
            cached_qos,
        )
        self.bspline_publisher = self.driver.create_publisher(
            Bspline,
            "/planning/bspline",
            cached_qos,
        )
        self.scan_status_publisher = self.driver.create_publisher(
            ScanPlanningStatus,
            "/planning/scan_status",
            cached_qos,
        )
        self.controller_status_publisher = self.driver.create_publisher(
            ControllerStatus,
            "/planning/controller_status",
            cached_qos,
        )
        self.odometry_publisher = self.driver.create_publisher(
            Odometry,
            "/body_pose",
            sensor_qos,
        )
        self.cloud_publisher = self.driver.create_publisher(
            PointCloud2,
            "/cloud_registered",
            sensor_qos,
        )
        self.navigation_statuses: list[NavigationStatus] = []
        self.driver.create_subscription(
            NavigationStatus,
            "/navigation/status",
            self.navigation_statuses.append,
            cached_qos,
        )

    def add_nodes(self, *nodes: Node) -> None:
        for node in (*nodes, self.driver):
            self.executor.add_node(node)
            self.nodes.append(node)

    def start(self) -> None:
        def _spin() -> None:
            try:
                self.executor.spin()
            except BaseException as exc:  # pragma: no cover - DDS runtime
                self.thread_errors.append(exc)

        self.thread = Thread(target=_spin, daemon=True)
        self.thread.start()

    def publish_clock(self) -> None:
        self.clock_publisher.publish(Clock(clock=_time(self.sim_time_ns)))

    def pulse(self, *, sensors: bool) -> None:
        self.sim_time_ns += 10_000_000
        self.publish_clock()
        if sensors:
            self.odometry_publisher.publish(_odometry(self.sim_time_ns))
            self.cloud_publisher.publish(_cloud(self.sim_time_ns))
        time.sleep(0.01)
        if self.thread_errors:
            raise AssertionError(
                f"DDS executor 异常：{self.thread_errors!r}"
            )

    def wait_until(
        self,
        predicate,
        *,
        timeout_sec: float,
        sensors: bool,
    ) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.pulse(sensors=sensors)
            if predicate():
                return
        states = [int(message.state) for message in self.navigation_statuses]
        raise AssertionError(
            f"DDS probe 在 {timeout_sec:.1f}s 内未完成；states={states[-10:]}"
        )

    def close(self) -> None:
        self.executor.shutdown(timeout_sec=2.0)
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        for node in reversed(self.nodes):
            node.destroy_node()
        self.nodes.clear()
        if rclpy.ok():
            rclpy.shutdown()


def _publisher_matches(graph: _DDSGraph) -> bool:
    publishers = (
        graph.clock_publisher,
        graph.path_publisher,
        graph.pct_status_publisher,
        graph.bspline_publisher,
        graph.scan_status_publisher,
        graph.controller_status_publisher,
        graph.odometry_publisher,
        graph.cloud_publisher,
    )
    return all(
        publisher.get_subscription_count() > 0
        for publisher in publishers
    )


def _latest_state(graph: _DDSGraph, state: int) -> bool:
    return bool(
        graph.navigation_statuses
        and int(graph.navigation_statuses[-1].state) == state
    )


def test_late_join_receives_transient_local_protocol_snapshot() -> None:
    """晚加入 supervisor 从五个 durable writer 恢复完整 tracking 快照。"""

    graph = _DDSGraph(domain_id=_probe_domain(0), multithreaded=False)
    try:
        graph.path_publisher.publish(_path())
        graph.pct_status_publisher.publish(_pct_success())
        graph.bspline_publisher.publish(_bspline())
        graph.scan_status_publisher.publish(_scan_trajectory())
        graph.controller_status_publisher.publish(_controller())

        supervisor = NavigationSupervisorNode()
        # 本 probe 只隔离 durable protocol topics；固定正 ROS 时间避免把
        # volatile /clock 的启动竞态误算成 transient-local 缓存失败。
        supervisor._clock_now_ns = lambda: graph.sim_time_ns
        graph.add_nodes(supervisor)
        discovery_deadline = time.monotonic() + 3.0
        while (
            time.monotonic() < discovery_deadline
            and not _publisher_matches(graph)
        ):
            graph.publish_clock()
            time.sleep(0.01)
        assert _publisher_matches(graph), "late-join DDS endpoints 未完成匹配"

        graph.start()
        graph.wait_until(
            lambda: _latest_state(graph, NavigationStatus.STATE_TRACKING),
            timeout_sec=4.0,
            sensors=True,
        )

        assert supervisor._active_path is not None
        assert supervisor._active_path.stamp_ns == PATH_STAMP_NS
        assert len(supervisor._trajectories) == 1
        assert supervisor._scan_sequence.latest == 1
        assert supervisor._controller_sequence.latest == 1
        assert supervisor._core.state is NavigationState.TRACKING
    finally:
        graph.close()


def test_pending_trajectory_evidence_tracks_after_matching_path() -> None:
    """真实 DDS 乱序先缓存轨迹证据，匹配 Path 后恢复身份有效跟踪。"""

    graph = _DDSGraph(domain_id=_probe_domain(2), multithreaded=False)
    supervisor = NavigationSupervisorNode()
    try:
        graph.add_nodes(supervisor)
        graph.start()
        graph.wait_until(
            lambda: _publisher_matches(graph),
            timeout_sec=3.0,
            sensors=True,
        )

        graph.pct_status_publisher.publish(_pct_success())
        graph.wait_until(
            lambda: (
                supervisor._goal is not None
                and supervisor._core.state
                is NavigationState.GLOBAL_PLANNING
            ),
            timeout_sec=2.0,
            sensors=True,
        )
        graph.controller_status_publisher.publish(_controller())
        graph.bspline_publisher.publish(_bspline())
        graph.scan_status_publisher.publish(_scan_trajectory())
        graph.wait_until(
            lambda: (
                supervisor._pending_path_evidence_count() == 2
                and supervisor._controller_sequence.latest == 1
            ),
            timeout_sec=2.0,
            sensors=True,
        )

        stopped = graph.navigation_statuses[-1]
        assert supervisor._active_path is None
        assert supervisor._scan_sequence.latest == -1
        assert bool(stopped.force_zero_velocity) is True
        assert bool(stopped.allow_tracking_command) is False

        graph.path_publisher.publish(_path())
        graph.wait_until(
            lambda: (
                _latest_state(graph, NavigationStatus.STATE_TRACKING)
                and supervisor._reported_tracking_identity is not None
                and supervisor._pending_path_evidence_count() == 0
            ),
            timeout_sec=2.0,
            sensors=True,
        )

        tracking = graph.navigation_statuses[-1]
        identity = supervisor._reported_tracking_identity
        assert identity is not None
        assert identity.reference_path_stamp_ns == PATH_STAMP_NS
        assert identity.trajectory_id == 11
        assert identity in supervisor._trajectories
        assert identity in supervisor._scan_trajectory_identities
        assert bool(tracking.force_zero_velocity) is False
        assert bool(tracking.allow_tracking_command) is True
        assert supervisor._protocol_poisoned is False
    finally:
        graph.close()


def test_tombstone_prevents_cached_evidence_resurrection() -> None:
    """同代 tombstone 清缓存，随后同 stamp 非空 Path 不能复活运动。"""

    graph = _DDSGraph(domain_id=_probe_domain(3), multithreaded=False)
    supervisor = NavigationSupervisorNode()
    try:
        graph.add_nodes(supervisor)
        graph.start()
        graph.wait_until(
            lambda: _publisher_matches(graph),
            timeout_sec=3.0,
            sensors=True,
        )

        graph.pct_status_publisher.publish(_pct_success())
        graph.controller_status_publisher.publish(_controller())
        graph.bspline_publisher.publish(_bspline())
        graph.scan_status_publisher.publish(_scan_trajectory())
        graph.wait_until(
            lambda: supervisor._pending_path_evidence_count() == 2,
            timeout_sec=2.0,
            sensors=True,
        )

        graph.path_publisher.publish(_path(empty=True))
        graph.wait_until(
            lambda: (
                supervisor._pending_path_evidence_count() == 0
                and supervisor._core.state
                is NavigationState.EMERGENCY_STOP
            ),
            timeout_sec=2.0,
            sensors=True,
        )
        assert supervisor._active_path is None
        assert supervisor._trajectories == {}
        assert supervisor._scan_trajectory_identities == set()

        graph.bspline_publisher.publish(_bspline())
        graph.wait_until(
            lambda: supervisor._adapter_fault_reason
            == "bspline_generation_not_newer_than_tombstone",
            timeout_sec=2.0,
            sensors=True,
        )
        graph.scan_status_publisher.publish(_scan_trajectory())
        graph.wait_until(
            lambda: supervisor._adapter_fault_reason
            == "scan_generation_not_newer_than_tombstone",
            timeout_sec=2.0,
            sensors=True,
        )
        assert supervisor._pending_path_evidence_count() == 0

        graph.path_publisher.publish(_path())
        graph.wait_until(
            lambda: supervisor._adapter_fault_reason
            == "conflicting_same_stamp_global_path",
            timeout_sec=2.0,
            sensors=True,
        )

        stopped = graph.navigation_statuses[-1]
        assert supervisor._active_path is None
        assert supervisor._pending_path_evidence_count() == 0
        assert supervisor._protocol_poisoned is True
        assert supervisor._core.state is NavigationState.EMERGENCY_STOP
        assert bool(stopped.force_zero_velocity) is True
        assert bool(stopped.allow_tracking_command) is False
    finally:
        graph.close()


def test_replan_service_retry_is_one_fixed_logical_request() -> None:
    """延迟 service 使传输重试，但固定 payload 只形成一个规划周期。"""

    graph = _DDSGraph(domain_id=_probe_domain(1), multithreaded=True)
    service = _DelayedPCTService("navigation_supervisor_dds_pct_service")
    supervisor = NavigationSupervisorNode()
    try:
        graph.add_nodes(supervisor, service)
        graph.start()
        graph.wait_until(
            lambda: (
                _publisher_matches(graph)
                and supervisor._pct_client.service_is_ready()
            ),
            timeout_sec=3.0,
            sensors=True,
        )

        graph.path_publisher.publish(_path())
        graph.pct_status_publisher.publish(_pct_success())
        graph.wait_until(
            lambda: _latest_state(
                graph,
                NavigationStatus.STATE_LOCAL_PLANNING,
            ),
            timeout_sec=2.0,
            sensors=True,
        )
        graph.bspline_publisher.publish(_bspline())
        graph.scan_status_publisher.publish(_scan_trajectory())
        graph.controller_status_publisher.publish(_controller())
        graph.wait_until(
            lambda: _latest_state(graph, NavigationStatus.STATE_TRACKING),
            timeout_sec=2.0,
            sensors=True,
        )

        graph.scan_status_publisher.publish(_scan_failures())
        graph.wait_until(
            lambda: bool(
                graph.navigation_statuses
                and graph.navigation_statuses[-1].global_replan_requested
            ),
            timeout_sec=2.0,
            sensors=True,
        )
        graph.controller_status_publisher.publish(
            _controller(sequence=2, valid=False),
        )
        graph.wait_until(
            lambda: (
                len(service.payloads) >= 2
                and supervisor._global_replan_cycle_count == 1
                and supervisor._core.state is NavigationState.GLOBAL_PLANNING
            ),
            timeout_sec=4.0,
            sensors=True,
        )

        assert len(service.payloads) == 2
        assert service.payloads[0] == service.payloads[1]
        assert service.request_ids == [2, 2]
        assert supervisor._replan is not None
        assert supervisor._replan.attempts == 2
        request_id = supervisor._replan.request_id

        graph.path_publisher.publish(_path(5_000_000_000))
        graph.pct_status_publisher.publish(
            _pct_success(
                path_stamp_ns=5_000_000_000,
                plan_id=2,
                request_id=request_id,
                command=PCTPlanningStatus.COMMAND_REPLAN,
                event_stamp_ns=graph.sim_time_ns,
            )
        )
        graph.wait_until(
            lambda: (
                _latest_state(
                    graph,
                    NavigationStatus.STATE_LOCAL_PLANNING,
                )
                and supervisor._global_replan_cycle_count == 0
            ),
            timeout_sec=2.0,
            sensors=True,
        )

        assert len(service.payloads) == 2
        assert supervisor._active_path is not None
        assert supervisor._active_path.stamp_ns == 5_000_000_000
        assert supervisor._global_planning_terminal_reason == ""
    finally:
        graph.close()
