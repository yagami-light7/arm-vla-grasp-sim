#!/usr/bin/env python3
"""验收独立进程官方 PCT ROS 2 adapter 的真实 Path 生命周期。"""

from __future__ import annotations

import json
import math
from pathlib import Path as FilesystemPath
import time

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from rosgraph_msgs.msg import Clock
from scan_planner_msgs.msg import PCTPlanningStatus


START_BASE_XYZ = (-3.50493, 6.74910, 0.1251194919198741)
GOAL_BASE_XYZ = (0.50, 0.30, 3.309128817237528)
BODY_HEIGHT_M = 0.30
PROJECT_ROOT = FilesystemPath(__file__).resolve().parents[4]
STAIR_PROFILE_PATH = (
    PROJECT_ROOT / "configs/navigation/pct_multifloor_stair_profile.json"
)


def _time_message(seconds: float) -> Time:
    """把非负浮点秒转换为规范 ROS Time。"""

    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def _same_stamp(first: Time, second: Time) -> bool:
    return first.sec == second.sec and first.nanosec == second.nanosec


def _validate_success(
    path: Path,
    scan_path: Path,
    status: PCTPlanningStatus,
) -> None:
    """验证 typed 状态、PCT 源 Path 与 SCAN 输入 Path 属于同一代。"""

    if not path.poses:
        raise RuntimeError("SUCCEEDED 对应的 Path 为空")
    if status.path_point_count != len(path.poses):
        raise RuntimeError(
            "typed 状态点数与 Path 不一致："
            f"{status.path_point_count} != {len(path.poses)}"
        )
    if not _same_stamp(status.path_stamp, path.header.stamp):
        raise RuntimeError("typed 状态与 Path 不属于同一代")
    if serialize_message(path) != serialize_message(scan_path):
        raise RuntimeError("/initial_path 没有原样保留 PCT Path 的完整 payload")
    if path.header.frame_id != "world":
        raise RuntimeError(f"Path frame 错误：{path.header.frame_id!r}")
    if any(
        pose.header.frame_id != path.header.frame_id
        or not _same_stamp(pose.header.stamp, path.header.stamp)
        for pose in path.poses
    ):
        raise RuntimeError("Path 内部 pose 的 frame 或时间戳不统一")

    points = tuple(
        (
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            float(pose.pose.position.z),
        )
        for pose in path.poses
    )
    if not all(math.isfinite(value) for point in points for value in point):
        raise RuntimeError("ROS Path 含非有限值")
    start_xy_error = math.dist(points[0][:2], START_BASE_XYZ[:2])
    goal_xy_error = math.dist(points[-1][:2], GOAL_BASE_XYZ[:2])
    start_z_error = abs(points[0][2] + BODY_HEIGHT_M - START_BASE_XYZ[2])
    goal_z_error = abs(points[-1][2] + BODY_HEIGHT_M - GOAL_BASE_XYZ[2])
    vertical_span = max(point[2] for point in points) - min(
        point[2] for point in points
    )
    if start_xy_error > 1.0e-6 or goal_xy_error > 1.0e-6:
        raise RuntimeError(
            "ROS Path 端点 XY 不符："
            f"start={start_xy_error:.9f}, goal={goal_xy_error:.9f}"
        )
    if start_z_error > 0.08 or goal_z_error > 0.08:
        raise RuntimeError(
            "ROS Path ground_height 合同错误："
            f"start={start_z_error:.6f}, goal={goal_z_error:.6f}"
        )
    if vertical_span < 2.50:
        raise RuntimeError(f"ROS Path 未跨楼层：高度跨度 {vertical_span:.3f} m")
    profile = json.loads(STAIR_PROFILE_PATH.read_text(encoding="utf-8"))
    anchors = tuple(
        tuple(float(value) for value in point)
        for point in profile["anchors_sim_ground_xyz"]
    )
    maximum_anchor_xy_error_m = max(
        min(math.dist(anchor[:2], point[:2]) for point in points)
        for anchor in anchors
    )
    if maximum_anchor_xy_error_m > 1.0e-6:
        raise RuntimeError(
            "ROS Path 没有经过标定楼梯中心线："
            f"最大 XY 误差 {maximum_anchor_xy_error_m:.9f} m"
        )

    print(
        "UPSTREAM_PCT_ROS_LIFECYCLE_OK "
        f"plan_id={status.plan_id} "
        f"goal_id={status.goal_id} "
        f"points={len(points)} "
        f"scan_alias_points={len(scan_path.poses)} "
        f"vertical_span_m={vertical_span:.6f} "
        f"stair_anchor_xy_error_m={maximum_anchor_xy_error_m:.9f} "
        f"start_z_error_m={start_z_error:.6f} "
        f"goal_z_error_m={goal_z_error:.6f}"
    )


def main() -> None:
    """持续提供仿真时钟与里程计，等待官方 backend 发布成功代际。"""

    rclpy.init()
    node = Node("upstream_pct_ros_lifecycle_probe")
    sensor_qos = QoSProfile(
        depth=20,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    cached_qos = QoSProfile(
        depth=4,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    clock_publisher = node.create_publisher(Clock, "/clock", 1)
    odometry_publisher = node.create_publisher(
        Odometry,
        "/body_pose",
        sensor_qos,
    )
    goal_publisher = node.create_publisher(PoseStamped, "/pct/goal", 1)
    paths: list[Path] = []
    scan_paths: list[Path] = []
    statuses: list[PCTPlanningStatus] = []
    node.create_subscription(
        Path,
        "/pct/global_path",
        paths.append,
        cached_qos,
    )
    node.create_subscription(
        Path,
        "/initial_path",
        scan_paths.append,
        cached_qos,
    )
    node.create_subscription(
        PCTPlanningStatus,
        "/pct/planning_status",
        statuses.append,
        cached_qos,
    )

    odometry = Odometry()
    odometry.header.frame_id = "world"
    odometry.child_frame_id = "base_link"
    odometry.pose.pose.position.x = START_BASE_XYZ[0]
    odometry.pose.pose.position.y = START_BASE_XYZ[1]
    odometry.pose.pose.position.z = START_BASE_XYZ[2]
    odometry.pose.pose.orientation.w = 1.0

    current_time = 10.0
    started_at = time.monotonic()
    goal_sent = False
    try:
        while time.monotonic() - started_at < 45.0:
            stamp = _time_message(current_time)
            clock_publisher.publish(Clock(clock=stamp))
            odometry.header.stamp = stamp
            odometry_publisher.publish(odometry)
            if (
                not goal_sent
                and goal_publisher.get_subscription_count() > 0
                and time.monotonic() - started_at >= 1.0
            ):
                goal = PoseStamped()
                goal.header.stamp = stamp
                goal.header.frame_id = "world"
                goal.pose.position.x = GOAL_BASE_XYZ[0]
                goal.pose.position.y = GOAL_BASE_XYZ[1]
                goal.pose.position.z = GOAL_BASE_XYZ[2]
                goal.pose.orientation.w = 1.0
                goal_publisher.publish(goal)
                goal_sent = True

            rclpy.spin_once(node, timeout_sec=0.01)
            failures = [
                status
                for status in statuses
                if status.state
                in (PCTPlanningStatus.NO_PATH, PCTPlanningStatus.ERROR)
            ]
            if failures:
                failure = failures[-1]
                raise RuntimeError(
                    "官方 PCT ROS 2 规划失败："
                    f"state={failure.state}, message={failure.message}"
                )
            successes = [
                status
                for status in statuses
                if status.state == PCTPlanningStatus.SUCCEEDED
            ]
            if successes:
                success = successes[-1]
                matching_paths = [
                    path
                    for path in paths
                    if path.poses
                    and _same_stamp(path.header.stamp, success.path_stamp)
                ]
                matching_scan_paths = [
                    path
                    for path in scan_paths
                    if path.poses
                    and _same_stamp(path.header.stamp, success.path_stamp)
                ]
                if matching_paths and matching_scan_paths:
                    _validate_success(
                        matching_paths[-1],
                        matching_scan_paths[-1],
                        success,
                    )
                    return
            current_time += 0.02
            time.sleep(0.02)
        raise RuntimeError(
            "官方 PCT ROS 2 生命周期超时："
            f"goal_sent={goal_sent}, "
            f"paths={[len(path.poses) for path in paths]}, "
            f"scan_paths={[len(path.poses) for path in scan_paths]}, "
            f"statuses={[(s.state, s.message) for s in statuses]}"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
