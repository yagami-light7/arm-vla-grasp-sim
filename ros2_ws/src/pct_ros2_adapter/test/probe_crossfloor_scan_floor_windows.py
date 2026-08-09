#!/usr/bin/env python3
"""用真实跨层 PCT 路线逐段审计 SCAN 的平地局部轨迹速度。"""

from __future__ import annotations

import json
import math
from pathlib import Path as FilePath
import struct
import sys
import time
import uuid

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from scan_planner_msgs.msg import (
    Bspline,
    BsplineDiagnostics,
    GridMapObservationDiagnostics,
    ScanPlanningStatus,
    StairExecutionFreeze,
)
from sensor_msgs.msg import PointCloud2, PointField

from multifloor_probe_contract import load_multifloor_probe_contract


START_BASE_XYZ = (
    -3.4748268127441406,
    6.524534225463867,
    0.16373473405838013,
)
PROJECT_ROOT = FilePath(__file__).resolve().parents[4]
PROBE_CONTRACT = load_multifloor_probe_contract(PROJECT_ROOT)
GOAL_BASE_XYZ = PROBE_CONTRACT.goal_base_xyz
GOAL_YAW = PROBE_CONTRACT.goal_yaw
BODY_HEIGHT_M = PROBE_CONTRACT.body_height_m
REFERENCE_CRUISE_SPEED_MPS = 0.60
STAIR_EXIT_SPEED_MPS = 0.30
FLOOR_SAMPLE_SPACING_M = 0.65
STAIR_ACTIVATION_BUFFER_M = 0.35
FINAL_WINDOW_BUFFER_M = 1.30
TERMINAL_WINDOW_SPECS = (
    # 终点本身仍比 1.20 m 局部前视远 0.05 m，但 planner 内门允许的
    # 安全捕获点已经进入前视范围；这是 live 中“到过目标却仍非 final”的回归门。
    (1.25, 0.60),
    (1.20, 0.60),
    (0.75, 0.40),
    (0.30, 0.20),
    (0.10, 0.05),
)
EXPECTED_PATH_POINT_COUNT = 146
MINIMUM_CRUISE_WINDOW_COUNT = 8
MINIMUM_FLOOR_WINDOW_VELOCITY_MPS = 0.45
MAXIMUM_FLOOR_WINDOW_DURATION_S = 3.20
MAXIMUM_TERMINAL_CAPTURE_XY_ERROR_M = 0.061
MAXIMUM_TERMINAL_TRAJECTORY_DURATION_S = 3.20
MAXIMUM_TERMINAL_TANGENT_YAW_ERROR_RAD = 0.10


def _time_message(seconds: float) -> Time:
    """把正浮点秒转换为 ROS Time。"""

    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def _time_ns(stamp: Time) -> int:
    """返回 ROS Time 的整数纳秒。"""

    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _same_stamp(lhs: Time, rhs: Time) -> bool:
    """比较两个 ROS Time 的完整整数身份。"""

    return _time_ns(lhs) == _time_ns(rhs)


def _evaluate_bspline(
    message: Bspline,
    sample_count: int = 240,
) -> tuple[tuple[float, float, float], ...]:
    """按消息 knot 域执行 De Boor 采样，用于复核最终捕获点。"""

    control_points = tuple(
        (point.x, point.y, point.z) for point in message.pos_pts
    )
    order = int(message.order)
    knots = tuple(float(value) for value in message.knots)
    if (
        len(control_points) < order + 1
        or len(knots) != len(control_points) + order + 1
    ):
        raise RuntimeError("B-spline 控制点或 knot 数量非法")
    domain_start = knots[order]
    domain_end = knots[len(control_points)]
    if domain_end <= domain_start:
        raise RuntimeError("B-spline 有效时间域为空")

    def evaluate(parameter: float) -> tuple[float, float, float]:
        bounded = min(max(parameter, domain_start), domain_end)
        span = order
        while span + 1 < len(knots) and knots[span + 1] < bounded:
            span += 1
        span = min(span, len(control_points) - 1)
        values = [
            list(control_points[span - order + index])
            for index in range(order + 1)
        ]
        for recursion in range(1, order + 1):
            for index in range(order, recursion - 1, -1):
                lower_index = index + span - order
                upper_index = index + 1 + span - recursion
                denominator = knots[upper_index] - knots[lower_index]
                if denominator <= 0.0:
                    raise RuntimeError("B-spline knot 区间退化")
                alpha = (bounded - knots[lower_index]) / denominator
                values[index] = [
                    (1.0 - alpha) * values[index - 1][axis]
                    + alpha * values[index][axis]
                    for axis in range(3)
                ]
        return tuple(values[order])

    return tuple(
        evaluate(
            domain_start
            + (domain_end - domain_start) * index / sample_count
        )
        for index in range(sample_count + 1)
    )


def _cumulative_lengths(
    points: tuple[tuple[float, float, float], ...],
) -> tuple[float, ...]:
    """计算三维折线的累计弧长。"""

    lengths = [0.0]
    for start, end in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(start, end))
    return tuple(lengths)


def _sample_at_progress(
    points: tuple[tuple[float, float, float], ...],
    lengths: tuple[float, ...],
    progress_m: float,
) -> tuple[tuple[float, float, float], int]:
    """按三维弧长插值，并返回插值点之后的首个原 Path 索引。"""

    bounded = min(max(float(progress_m), 0.0), lengths[-1])
    for index in range(1, len(points)):
        if lengths[index] + 1.0e-12 < bounded:
            continue
        segment_length = lengths[index] - lengths[index - 1]
        ratio = 0.0 if segment_length <= 1.0e-12 else (
            bounded - lengths[index - 1]
        ) / segment_length
        start = points[index - 1]
        end = points[index]
        point = tuple(
            start[axis] + ratio * (end[axis] - start[axis])
            for axis in range(3)
        )
        following_index = index
        if math.dist(point, end) <= 1.0e-9:
            following_index = index + 1
        return point, following_index
    return points[-1], len(points)


def _suffix_from_progress(
    points: tuple[tuple[float, float, float], ...],
    lengths: tuple[float, ...],
    progress_m: float,
) -> tuple[tuple[float, float, float], ...]:
    """从指定进度构造不重复首点、保持精确终点的 Path 后缀。"""

    first, following_index = _sample_at_progress(points, lengths, progress_m)
    suffix = (first, *points[following_index:])
    if len(suffix) < 2:
        raise ValueError("Path 后缀不足两个点")
    return suffix


def _tangent_yaw(
    points: tuple[tuple[float, float, float], ...],
    lengths: tuple[float, ...],
    progress_m: float,
) -> float:
    """使用前后 5 cm 弧长差分估计当前地面路径切向 yaw。"""

    before, _ = _sample_at_progress(points, lengths, progress_m - 0.05)
    after, _ = _sample_at_progress(points, lengths, progress_m + 0.05)
    delta_x = after[0] - before[0]
    delta_y = after[1] - before[1]
    if math.hypot(delta_x, delta_y) <= 1.0e-9:
        return GOAL_YAW
    return math.atan2(delta_y, delta_x)


def _nearest_anchor_index(
    points: tuple[tuple[float, float, float], ...],
    anchor: tuple[float, float, float],
) -> int:
    """按 XY 找到 Path 中的标定楼梯锚点，并要求误差为毫米级。"""

    index = min(
        range(len(points)),
        key=lambda candidate: math.dist(
            points[candidate][:2], anchor[:2]
        ),
    )
    error_xy = math.dist(points[index][:2], anchor[:2])
    if error_xy > 1.0e-3:
        raise RuntimeError(
            f"PCT Path 缺少楼梯锚点 {anchor[:2]}：误差 {error_xy:.6f} m"
        )
    return index


def _floor_sample_progresses(
    stair_entry_progress_m: float,
    stair_exit_progress_m: float,
    total_progress_m: float,
) -> tuple[tuple[str, float, float], ...]:
    """生成楼梯接管区之外的起点楼面和目标楼面审计位置。"""

    samples: list[tuple[str, float, float]] = []
    start_limit = max(
        0.0,
        stair_entry_progress_m - STAIR_ACTIVATION_BUFFER_M,
    )
    progress = 0.0
    while progress <= start_limit + 1.0e-9:
        if progress + 1.0e-9 >= start_limit:
            break
        speed = 0.0 if not samples else REFERENCE_CRUISE_SPEED_MPS
        samples.append(("start_floor", progress, speed))
        progress += FLOOR_SAMPLE_SPACING_M

    goal_limit = max(
        stair_exit_progress_m,
        total_progress_m - FINAL_WINDOW_BUFFER_M,
    )
    progress = stair_exit_progress_m
    first_goal_sample = True
    while progress <= goal_limit + 1.0e-9:
        speed = (
            STAIR_EXIT_SPEED_MPS
            if first_goal_sample
            else REFERENCE_CRUISE_SPEED_MPS
        )
        samples.append(("goal_floor", progress, speed))
        first_goal_sample = False
        progress += FLOOR_SAMPLE_SPACING_M
    if (
        samples
        and samples[-1][0] == "goal_floor"
        and goal_limit - samples[-1][1] >= 0.25
    ):
        samples.append(
            ("goal_floor", goal_limit, REFERENCE_CRUISE_SPEED_MPS)
        )
    return tuple(samples)


def _terminal_sample_progresses(
    total_progress_m: float,
) -> tuple[tuple[str, float, float], ...]:
    """生成最后一个局部窗口内由 SCAN 制动并捕获终点的审计位置。"""

    return tuple(
        (
            "terminal_window",
            max(0.0, total_progress_m - remaining_m),
            forward_speed_mps,
        )
        for remaining_m, forward_speed_mps in TERMINAL_WINDOW_SPECS
    )


def _synthetic_free_space_cloud(
    stamp: Time,
    center: tuple[float, float, float],
    occupied_points: tuple[tuple[float, float, float], ...] = (),
) -> PointCloud2:
    """用显式 free 环形射线和可选 hit 建立当前局部占据。"""

    free_points = [
        (
            center[0] + 4.0 * math.cos(2.0 * math.pi * index / 144.0),
            center[1] + 4.0 * math.sin(2.0 * math.pi * index / 144.0),
            center[2],
        )
        for index in range(144)
    ]
    tagged_points = [
        (*point, 0) for point in free_points
    ] + [
        (*point, 1) for point in occupied_points
    ]
    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = "world"
    cloud.height = 1
    cloud.width = len(tagged_points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name="ray_endpoint_type",
            offset=12,
            datatype=PointField.UINT8,
            count=1,
        ),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 13
    cloud.row_step = cloud.point_step * cloud.width
    cloud.data = b"".join(
        struct.pack("<fffB", *point) for point in tagged_points
    )
    cloud.is_dense = True
    return cloud


def _production_backend(project_root: FilePath):
    """创建与主运行链参数完全相同的固定 upstream backend。"""

    package_source = project_root / "ros2_ws/src/pct_ros2_adapter"
    if str(package_source) not in sys.path:
        sys.path.insert(0, str(package_source))
    from pct_ros2_adapter.backend import (  # noqa: PLC0415
        PCTBackendConfig,
        create_global_planner_backend,
    )

    stair_profile_path = (
        project_root / "configs/navigation/pct_multifloor_stair_profile.json"
    )
    config = PCTBackendConfig(
        project_root=project_root,
        backend_kind="upstream",
        upstream_source_root=project_root / "external/PCT_planner",
        tomogram_path=(
            project_root
            / "source/scene/multifloor/mutifloor_upstream.pickle"
        ),
        walkable_path=(
            project_root
            / "source/scene/multifloor/mutifloor_ply_walkable.npy"
        ),
        collision_ply_path=(
            project_root
            / "source/scene/multifloor/ply/3dgs_collision.ply"
        ),
        upstream_stair_profile_path=stair_profile_path,
        upstream_body_clearance_enabled=True,
        upstream_body_clearance_radius_m=0.80,
        upstream_body_clearance_maximum_cost=20.0,
        upstream_body_clearance_power=2.0,
        upstream_same_layer_shortcut_clearance_m=0.27,
        upstream_same_layer_shortcut_max_segment_m=10.0,
        slice_query_root_to_floor_m=BODY_HEIGHT_M,
        goal_base_to_ground_m=BODY_HEIGHT_M,
        path_sample_spacing_m=0.20,
    )
    return create_global_planner_backend(config)


def _production_path(
    project_root: FilePath,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    """直接调用固定 upstream core 生成 phase225 精确请求的新路线。"""

    backend = _production_backend(project_root)
    plan = backend.plan(
        start_base_xyz=START_BASE_XYZ,
        goal_base_xyz=GOAL_BASE_XYZ,
        goal_yaw=GOAL_YAW,
    )
    points = tuple(tuple(float(value) for value in point) for point in plan.points_xyz)
    if len(points) != EXPECTED_PATH_POINT_COUNT:
        raise RuntimeError(
            "真实生产请求的 PCT Path 点数发生未审计变化："
            f"expected={EXPECTED_PATH_POINT_COUNT}, actual={len(points)}"
        )
    stair_profile_path = (
        project_root / "configs/navigation/pct_multifloor_stair_profile.json"
    )
    profile_payload = json.loads(stair_profile_path.read_text(encoding="utf-8"))
    anchors = tuple(
        tuple(float(value) for value in point)
        for point in profile_payload["anchors_sim_ground_xyz"]
    )
    return points, anchors


class FloorWindowProbe:
    """向生产 SCAN 节点发布逐段 Path 后缀并收集同代轨迹诊断。"""

    def __init__(
        self,
        node: Node,
        *,
        occupied_cloud_points: tuple[tuple[float, float, float], ...] = (),
    ) -> None:
        self.node = node
        self.occupied_cloud_points = occupied_cloud_points
        sensor_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        path_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        diagnostic_qos = QoSProfile(
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        freeze_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.clock_publisher = node.create_publisher(Clock, "/clock", 1)
        self.odometry_publisher = node.create_publisher(
            Odometry, "/body_pose", sensor_qos
        )
        self.cloud_publisher = node.create_publisher(
            PointCloud2, "/cloud_registered", sensor_qos
        )
        self.path_publisher = node.create_publisher(
            Path, "/pct/global_path", path_qos
        )
        self.freeze_publisher = node.create_publisher(
            StairExecutionFreeze,
            "/planning/stair_execution_frozen",
            freeze_qos,
        )
        self.splines: list[Bspline] = []
        self.diagnostics: list[BsplineDiagnostics] = []
        self.map_diagnostics: list[GridMapObservationDiagnostics] = []
        self.statuses: list[ScanPlanningStatus] = []
        node.create_subscription(
            Bspline, "/planning/bspline", self.splines.append, 32
        )
        node.create_subscription(
            BsplineDiagnostics,
            "/planning/bspline_diagnostics",
            self.diagnostics.append,
            diagnostic_qos,
        )
        node.create_subscription(
            ScanPlanningStatus,
            "/planning/scan_status",
            self.statuses.append,
            diagnostic_qos,
        )
        node.create_subscription(
            GridMapObservationDiagnostics,
            "/planning/grid_map_observation_diagnostics",
            self.map_diagnostics.append,
            diagnostic_qos,
        )
        self.current_time = 10.0
        self.next_cloud_time = self.current_time
        self.freeze_sequence = 0
        self.freeze_writer_epoch = uuid.uuid4().hex

    def wait_for_graph(self) -> None:
        """等待 SCAN 与 controller 发现所有探针输入。"""

        started_at = time.monotonic()
        while time.monotonic() - started_at < 8.0:
            if (
                self.odometry_publisher.get_subscription_count() >= 2
                and self.cloud_publisher.get_subscription_count() >= 2
                and self.path_publisher.get_subscription_count() >= 2
                and self.freeze_publisher.get_subscription_count() >= 1
            ):
                return
            self._spin_once(timeout_sec=0.02)
            time.sleep(0.02)
        raise RuntimeError(
            "没有发现完整 SCAN/controller 图："
            f"odom={self.odometry_publisher.get_subscription_count()}, "
            f"cloud={self.cloud_publisher.get_subscription_count()}, "
            f"path={self.path_publisher.get_subscription_count()}, "
            f"freeze={self.freeze_publisher.get_subscription_count()}"
        )

    def _spin_once(self, *, timeout_sec: float = 0.005) -> None:
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def _odometry(
        self,
        stamp: Time,
        ground_point: tuple[float, float, float],
        yaw: float,
        forward_speed_mps: float,
    ) -> Odometry:
        """构造机体系前向速度与世界系路径切向一致的理想 Odometry。"""

        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.child_frame_id = "base_link"
        message.pose.pose.position.x = ground_point[0]
        message.pose.pose.position.y = ground_point[1]
        message.pose.pose.position.z = ground_point[2] + BODY_HEIGHT_M
        message.pose.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.pose.orientation.w = math.cos(0.5 * yaw)
        message.twist.twist.linear.x = forward_speed_mps
        return message

    def _freeze_snapshot(self, stamp: Time, path_stamp: Time) -> StairExecutionFreeze:
        """持续声明当前 Path 位于非冻结平地段。"""

        self.freeze_sequence += 1
        snapshot = StairExecutionFreeze()
        snapshot.header.stamp = stamp
        snapshot.header.frame_id = "world"
        snapshot.reference_path_stamp = path_stamp
        snapshot.writer_id = "pct_scan_floor_window_probe"
        snapshot.writer_epoch = self.freeze_writer_epoch
        snapshot.sequence = self.freeze_sequence
        snapshot.frozen = False
        return snapshot

    def tick(
        self,
        *,
        ground_point: tuple[float, float, float],
        yaw: float,
        forward_speed_mps: float,
        path_stamp: Time | None,
    ) -> None:
        """推进一拍连续仿真时钟和新鲜传感器输入。"""

        stamp = _time_message(self.current_time)
        self.clock_publisher.publish(Clock(clock=stamp))
        odometry = self._odometry(
            stamp, ground_point, yaw, forward_speed_mps
        )
        self.odometry_publisher.publish(odometry)
        if self.current_time + 1.0e-9 >= self.next_cloud_time:
            self.cloud_publisher.publish(
                _synthetic_free_space_cloud(
                    stamp,
                    (
                        ground_point[0],
                        ground_point[1],
                        ground_point[2] + BODY_HEIGHT_M,
                    ),
                    self.occupied_cloud_points,
                )
            )
            self.next_cloud_time += 0.10
        if path_stamp is not None:
            self.freeze_publisher.publish(
                self._freeze_snapshot(stamp, path_stamp)
            )
        self._spin_once()
        self.current_time += 0.02
        time.sleep(0.022)

    def run_for(
        self,
        duration_sec: float,
        *,
        ground_point: tuple[float, float, float],
        yaw: float,
        forward_speed_mps: float,
        path_stamp: Time | None,
    ) -> None:
        """按接近真实时间的 50 Hz 连续发布指定状态。"""

        end_time = self.current_time + duration_sec
        while self.current_time + 1.0e-9 < end_time:
            self.tick(
                ground_point=ground_point,
                yaw=yaw,
                forward_speed_mps=forward_speed_mps,
                path_stamp=path_stamp,
            )

    def publish_tombstone(
        self,
        ground_point: tuple[float, float, float],
        yaw: float,
        forward_speed_mps: float,
    ) -> None:
        """清除上一个审计 Path，再用新 Odometry 预热速度滤波器。"""

        tombstone = Path()
        tombstone.header.stamp = _time_message(self.current_time)
        tombstone.header.frame_id = "world"
        self.path_publisher.publish(tombstone)
        self.run_for(
            0.12,
            ground_point=ground_point,
            yaw=yaw,
            forward_speed_mps=forward_speed_mps,
            path_stamp=None,
        )
        self.run_for(
            0.38,
            ground_point=ground_point,
            yaw=yaw,
            forward_speed_mps=forward_speed_mps,
            path_stamp=None,
        )

    def publish_path(
        self,
        points: tuple[tuple[float, float, float], ...],
    ) -> tuple[Path, Time]:
        """发布带有效切向 yaw 和精确终端 yaw 的 Path 后缀。"""

        stamp = _time_message(self.current_time)
        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        for index, point in enumerate(points):
            yaw = GOAL_YAW
            if index + 1 < len(points):
                following = points[index + 1]
                delta_x = following[0] - point[0]
                delta_y = following[1] - point[1]
                if math.hypot(delta_x, delta_y) > 1.0e-9:
                    yaw = math.atan2(delta_y, delta_x)
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            message.poses.append(pose)
        self.path_publisher.publish(message)
        return message, stamp

    def wait_for_trajectory(
        self,
        *,
        path_stamp: Time,
        ground_point: tuple[float, float, float],
        yaw: float,
        forward_speed_mps: float,
    ) -> tuple[Bspline, BsplineDiagnostics]:
        """等待与当前 Path 完整同代的正常 B-spline 和诊断。"""

        started_at = time.monotonic()
        while time.monotonic() - started_at < 5.0:
            self.tick(
                ground_point=ground_point,
                yaw=yaw,
                forward_speed_mps=forward_speed_mps,
                path_stamp=path_stamp,
            )
            matching_splines = [
                spline
                for spline in self.splines
                if spline.pos_pts
                and not spline.emergency_stop
                and _same_stamp(spline.reference_path_stamp, path_stamp)
            ]
            for spline in reversed(matching_splines):
                matching_diagnostics = [
                    diagnostic
                    for diagnostic in self.diagnostics
                    if diagnostic.traj_id == spline.traj_id
                    and _same_stamp(
                        diagnostic.reference_path_stamp, path_stamp
                    )
                    and _same_stamp(diagnostic.header.stamp, spline.header.stamp)
                    and _same_stamp(diagnostic.start_time, spline.start_time)
                ]
                if matching_diagnostics:
                    return spline, matching_diagnostics[-1]
        recent_statuses = [
            (status.event, status.state, status.reason)
            for status in self.statuses
            if _same_stamp(status.reference_path_stamp, path_stamp)
        ][-8:]
        raise RuntimeError(
            "SCAN 平地窗口没有产生正常轨迹："
            f"path_stamp_ns={_time_ns(path_stamp)}, "
            f"statuses={recent_statuses}"
        )


def main() -> None:
    """逐段发布真实 Path 后缀并输出楼面与终点轨迹分布。"""

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("真实 upstream 扩展和 ROS 2 Humble 要求 Python 3.10")
    project_root = PROJECT_ROOT
    points, stair_anchors = _production_path(project_root)
    lengths = _cumulative_lengths(points)
    entry_index = _nearest_anchor_index(points, stair_anchors[0])
    exit_index = _nearest_anchor_index(points, stair_anchors[-1])
    entry_progress = lengths[entry_index]
    exit_progress = lengths[exit_index]
    terminal_tangent_yaw = _tangent_yaw(points, lengths, lengths[-1])
    terminal_tangent_yaw_error = abs(
        math.atan2(
            math.sin(GOAL_YAW - terminal_tangent_yaw),
            math.cos(GOAL_YAW - terminal_tangent_yaw),
        )
    )
    if (
        terminal_tangent_yaw_error
        > MAXIMUM_TERMINAL_TANGENT_YAW_ERROR_RAD + 1.0e-9
    ):
        raise RuntimeError(
            "PCT 末段切向与任务 terminal yaw 差异过大："
            f"error={terminal_tangent_yaw_error:.6f} rad"
        )
    floor_samples = _floor_sample_progresses(
        entry_progress, exit_progress, lengths[-1]
    )
    if len(floor_samples) < MINIMUM_CRUISE_WINDOW_COUNT:
        raise RuntimeError("真实跨层路线没有形成足量平地审计窗口")
    terminal_samples = _terminal_sample_progresses(lengths[-1])
    samples = (*floor_samples, *terminal_samples)

    rclpy.init()
    node = Node("pct_scan_crossfloor_floor_window_probe")
    probe = FloorWindowProbe(node)
    results: list[dict[str, object]] = []
    try:
        probe.wait_for_graph()
        for label, progress, forward_speed in samples:
            point, _ = _sample_at_progress(points, lengths, progress)
            yaw = _tangent_yaw(points, lengths, progress)
            suffix = _suffix_from_progress(points, lengths, progress)
            probe.publish_tombstone(point, yaw, forward_speed)
            _, path_stamp = probe.publish_path(suffix)
            # 先让 Path 回调绑定新代际，再持续发送同 stamp 非冻结快照。
            probe.run_for(
                0.08,
                ground_point=point,
                yaw=yaw,
                forward_speed_mps=forward_speed,
                path_stamp=None,
            )
            spline, diagnostic = probe.wait_for_trajectory(
                path_stamp=path_stamp,
                ground_point=point,
                yaw=yaw,
                forward_speed_mps=forward_speed,
            )
            if (
                not diagnostic.ordered_reference_checked
                or not diagnostic.ordered_reference_safe
                or diagnostic.active_sensing
                or diagnostic.stationary
            ):
                raise RuntimeError(
                    "平地窗口轨迹没有通过 ordered-reference 动态合同："
                    f"label={label}, progress={progress:.3f}"
                )
            trajectory_samples = _evaluate_bspline(spline)
            trajectory_endpoint = trajectory_samples[-1]
            results.append(
                {
                    "floor": label,
                    "progress_m": progress,
                    "remaining_path_m": lengths[-1] - progress,
                    "input_speed_mps": forward_speed,
                    "trajectory_id": int(spline.traj_id),
                    "is_final": bool(spline.is_final),
                    "duration_s": float(diagnostic.trajectory_duration),
                    "velocity_upper_bound_mps": float(
                        diagnostic.maximum_velocity_upper_bound
                    ),
                    "ordered_reference_points": int(
                        diagnostic.ordered_reference_sample_count_total
                    ),
                    "maximum_trajectory_deviation_m": float(
                        diagnostic.maximum_trajectory_deviation
                    ),
                    "trajectory_endpoint_goal_xy_error_m": math.dist(
                        trajectory_endpoint[:2], GOAL_BASE_XYZ[:2]
                    ),
                    "trajectory_endpoint_goal_z_error_m": abs(
                        trajectory_endpoint[2] - GOAL_BASE_XYZ[2]
                    ),
                }
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    non_final = [result for result in results if not result["is_final"]]
    if len(non_final) < MINIMUM_CRUISE_WINDOW_COUNT:
        raise RuntimeError("非最终 SCAN 平地轨迹数量不足")
    durations = [float(result["duration_s"]) for result in non_final]
    velocities = [
        float(result["velocity_upper_bound_mps"]) for result in non_final
    ]
    minimum_velocity = min(velocities)
    maximum_duration = max(durations)
    if minimum_velocity + 1.0e-9 < MINIMUM_FLOOR_WINDOW_VELOCITY_MPS:
        raise RuntimeError(
            "真实跨层路线仍含过慢 SCAN 平地窗口："
            f"minimum={minimum_velocity:.6f} m/s"
        )
    if maximum_duration > MAXIMUM_FLOOR_WINDOW_DURATION_S + 1.0e-9:
        raise RuntimeError(
            "真实跨层路线仍含过长 SCAN 平地窗口："
            f"maximum={maximum_duration:.6f} s"
        )
    terminal_results = [
        result
        for result in results
        if result["floor"] == "terminal_window"
    ]
    if len(terminal_results) != len(TERMINAL_WINDOW_SPECS):
        raise RuntimeError("终点 SCAN 轨迹审计窗口数量不完整")
    if any(not result["is_final"] for result in terminal_results):
        raise RuntimeError("终点局部窗口没有生成 moving final B-spline")
    maximum_terminal_capture_error = max(
        float(result["trajectory_endpoint_goal_xy_error_m"])
        for result in terminal_results
    )
    if (
        maximum_terminal_capture_error
        > MAXIMUM_TERMINAL_CAPTURE_XY_ERROR_M + 1.0e-9
    ):
        raise RuntimeError(
            "终点 B-spline 捕获点超出 planner 到达内门："
            f"maximum={maximum_terminal_capture_error:.6f} m"
        )
    terminal_durations = [
        float(result["duration_s"]) for result in terminal_results
    ]
    maximum_terminal_duration = max(terminal_durations)
    if (
        maximum_terminal_duration
        > MAXIMUM_TERMINAL_TRAJECTORY_DURATION_S + 1.0e-9
    ):
        raise RuntimeError(
            "终点 SCAN 轨迹时长超过自动性能门："
            f"maximum={maximum_terminal_duration:.6f} s"
        )
    report = {
        "result": "PASS",
        "evidence_kind": "synthetic_free_space_ideal_odometry_floor_window_sweep",
        "path_point_count": len(points),
        "path_length_m": lengths[-1],
        "stair_entry_progress_m": entry_progress,
        "stair_exit_progress_m": exit_progress,
        "window_count": len(results),
        "non_final_window_count": len(non_final),
        "minimum_velocity_upper_bound_mps": minimum_velocity,
        "maximum_trajectory_duration_s": maximum_duration,
        "mean_trajectory_duration_s": sum(durations) / len(durations),
        "minimum_velocity_gate_mps": MINIMUM_FLOOR_WINDOW_VELOCITY_MPS,
        "maximum_duration_gate_s": MAXIMUM_FLOOR_WINDOW_DURATION_S,
        "terminal_window_count": len(terminal_results),
        "maximum_terminal_trajectory_duration_s": (
            maximum_terminal_duration
        ),
        "maximum_terminal_trajectory_duration_gate_s": (
            MAXIMUM_TERMINAL_TRAJECTORY_DURATION_S
        ),
        "mean_terminal_trajectory_duration_s": (
            sum(terminal_durations) / len(terminal_durations)
        ),
        "maximum_terminal_capture_xy_error_m": (
            maximum_terminal_capture_error
        ),
        "maximum_terminal_capture_xy_error_gate_m": (
            MAXIMUM_TERMINAL_CAPTURE_XY_ERROR_M
        ),
        "terminal_tangent_yaw_rad": terminal_tangent_yaw,
        "terminal_goal_yaw_error_rad": terminal_tangent_yaw_error,
        "terminal_goal_yaw_error_gate_rad": (
            MAXIMUM_TERMINAL_TANGENT_YAW_ERROR_RAD
        ),
        "goal_base_xyz": list(GOAL_BASE_XYZ),
        "raw_task_goal_base_xyz": list(PROBE_CONTRACT.raw_goal_base_xyz),
        "goal_ground_surface_z_m": PROBE_CONTRACT.ground_surface_z_m,
        "goal_ground_face_index": PROBE_CONTRACT.ground_face_index,
        "body_height_m": BODY_HEIGHT_M,
        "terminal_windows": terminal_results,
        "windows": results,
    }
    print(
        "PCT_SCAN_FLOOR_WINDOWS_OK "
        + json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
