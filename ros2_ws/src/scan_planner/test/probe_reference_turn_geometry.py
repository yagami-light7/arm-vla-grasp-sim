#!/usr/bin/env python3
"""用空旷 90° Path 验证 SCAN 不会把局部 guide 优化成对角捷径。"""

from __future__ import annotations

import math
import struct
import time

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from scan_planner_msgs.msg import Bspline
from sensor_msgs.msg import PointCloud2, PointField


GROUND_POINTS = (
    (-0.54684120694198, 5.052347877290755, -0.15357718367978318),
    (-0.24684120694198, 5.052347877290755, -0.15067181352124973),
    (0.05315879305802, 5.052347877290755, -0.14421487341800926),
    (0.05315879305802, 5.352347877290755, -0.1386447881840442),
    (0.05315879305802, 5.652347877290755, -0.12837291433207745),
)
BODY_HEIGHT_M = 0.30


def _time_message(seconds: float) -> Time:
    """把正浮点秒转换为 ROS Time。"""

    whole = int(math.floor(seconds))
    nanosecond = int(round((seconds - whole) * 1.0e9))
    if nanosecond >= 1_000_000_000:
        whole += 1
        nanosecond -= 1_000_000_000
    return Time(sec=whole, nanosec=nanosecond)


def _free_space_cloud(stamp: Time, center: tuple[float, float, float]) -> PointCloud2:
    """发布远端环形回波，让 sliding map 覆盖转角并保持近场自由。"""

    points = [
        (
            center[0] + 4.0 * math.cos(2.0 * math.pi * index / 144.0),
            center[1] + 4.0 * math.sin(2.0 * math.pi * index / 144.0),
            center[2],
        )
        for index in range(144)
    ]
    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = "world"
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.point_step = 12
    cloud.row_step = cloud.point_step * cloud.width
    cloud.data = b"".join(struct.pack("<fff", *point) for point in points)
    cloud.is_dense = True
    return cloud


def _path_message(stamp: Time) -> Path:
    """构造带有效切向 yaw 和地面高度语义的 L 形 Path。"""

    message = Path()
    message.header.stamp = stamp
    message.header.frame_id = "world"
    for index, point in enumerate(GROUND_POINTS):
        if index + 1 < len(GROUND_POINTS):
            following = GROUND_POINTS[index + 1]
            yaw = math.atan2(following[1] - point[1], following[0] - point[0])
        else:
            yaw = math.pi / 2.0
        pose = PoseStamped()
        pose.header = message.header
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = point[2]
        pose.pose.orientation.z = math.sin(0.5 * yaw)
        pose.pose.orientation.w = math.cos(0.5 * yaw)
        message.poses.append(pose)
    return message


def _evaluate_bspline(message: Bspline, sample_count: int = 240) -> list[tuple[float, float, float]]:
    """按 SCAN UniformBspline 的 knot 域执行 De Boor 采样。"""

    control_points = [(point.x, point.y, point.z) for point in message.pos_pts]
    order = int(message.order)
    knots = [float(value) for value in message.knots]
    if len(control_points) < order + 1 or len(knots) != len(control_points) + order + 1:
        raise RuntimeError("B-spline 控制点或 knot 数量非法")
    domain_start = knots[order]
    domain_end = knots[len(control_points)]
    if not domain_end > domain_start:
        raise RuntimeError("B-spline 有效时间域为空")

    def evaluate(parameter: float) -> tuple[float, float, float]:
        bounded = min(max(parameter, domain_start), domain_end)
        span = order
        while span + 1 < len(knots) and knots[span + 1] < bounded:
            span += 1
        span = min(span, len(control_points) - 1)
        values = [list(control_points[span - order + index]) for index in range(order + 1)]
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

    return [
        evaluate(domain_start + (domain_end - domain_start) * index / sample_count)
        for index in range(sample_count + 1)
    ]


def _segment_distance_xy(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    """返回平面点到线段的最短距离。"""

    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    ratio = 0.0 if length_squared <= 1.0e-18 else max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y)
            / length_squared,
        ),
    )
    nearest_x = start[0] + ratio * delta_x
    nearest_y = start[1] + ratio * delta_y
    return math.hypot(point[0] - nearest_x, point[1] - nearest_y)


def main() -> None:
    """等待跨转角的第一条正常 B-spline，并执行几何安全门。"""

    rclpy.init()
    node = Node("scan_reference_turn_geometry_probe")
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
    clock_publisher = node.create_publisher(Clock, "/clock", 1)
    odometry_publisher = node.create_publisher(Odometry, "/body_pose", sensor_qos)
    cloud_publisher = node.create_publisher(PointCloud2, "/cloud_registered", sensor_qos)
    path_publisher = node.create_publisher(Path, "/initial_path", path_qos)
    splines: list[Bspline] = []
    node.create_subscription(Bspline, "/planning/bspline", splines.append, 10)

    # 从首段中点开始，使 0.60 m 前视窗口同时包含 90° 转角前后各 0.30 m。
    start_ground = GROUND_POINTS[1]
    base_position = (
        start_ground[0],
        start_ground[1],
        start_ground[2] + BODY_HEIGHT_M,
    )
    odometry = Odometry()
    odometry.header.frame_id = "world"
    odometry.child_frame_id = "base_link"
    odometry.pose.pose.position.x = base_position[0]
    odometry.pose.pose.position.y = base_position[1]
    odometry.pose.pose.position.z = base_position[2]
    odometry.pose.pose.orientation.w = 1.0

    current_time = 10.0
    started_at = time.monotonic()
    path_sent = False
    path_stamp: Time | None = None
    try:
        while time.monotonic() - started_at < 10.0:
            stamp = _time_message(current_time)
            clock_publisher.publish(Clock(clock=stamp))
            odometry.header.stamp = stamp
            odometry_publisher.publish(odometry)
            cloud_publisher.publish(_free_space_cloud(stamp, base_position))
            if (
                not path_sent
                and path_publisher.get_subscription_count() > 0
                and time.monotonic() - started_at >= 1.0
            ):
                path_publisher.publish(_path_message(stamp))
                path_stamp = stamp
                path_sent = True
            rclpy.spin_once(node, timeout_sec=0.005)

            normal = [
                spline
                for spline in splines
                if spline.pos_pts and not spline.emergency_stop
            ]
            if normal:
                spline = normal[-1]
                if path_stamp is None or (
                    spline.reference_path_stamp.sec != path_stamp.sec
                    or spline.reference_path_stamp.nanosec != path_stamp.nanosec
                ):
                    raise RuntimeError("B-spline 与 L 形 Path 代际不一致")
                samples = _evaluate_bspline(spline)
                corner = GROUND_POINTS[2]
                corner_base = (
                    corner[0], corner[1], corner[2] + BODY_HEIGHT_M
                )
                minimum_corner_distance_xy = min(
                    math.hypot(point[0] - corner_base[0], point[1] - corner_base[1])
                    for point in samples
                )
                corridor_deviation_xy = max(
                    min(
                        _segment_distance_xy(point, GROUND_POINTS[1], GROUND_POINTS[2]),
                        _segment_distance_xy(point, GROUND_POINTS[2], GROUND_POINTS[3]),
                    )
                    for point in samples
                )
                if minimum_corner_distance_xy > 0.10:
                    raise RuntimeError(
                        "SCAN 在空旷 90° guide 上切角："
                        f"距折角 {minimum_corner_distance_xy:.6f} m"
                    )
                if corridor_deviation_xy > 0.10:
                    raise RuntimeError(
                        "SCAN 在空旷 90° guide 上偏离折线走廊："
                        f"最大偏差 {corridor_deviation_xy:.6f} m"
                    )
                print(
                    "SCAN_TURN_GEOMETRY_OK "
                    f"spline_points={len(spline.pos_pts)} "
                    f"corner_distance_xy={minimum_corner_distance_xy:.6f} "
                    f"corridor_deviation_xy={corridor_deviation_xy:.6f}"
                )
                return
            current_time += 0.02
            time.sleep(0.015)
        raise RuntimeError(
            "SCAN 90° 转角探针超时："
            f"path_sent={path_sent}，splines={len(splines)}"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
