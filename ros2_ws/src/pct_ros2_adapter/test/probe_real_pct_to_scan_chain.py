#!/usr/bin/env python3
"""用真实 multi-floor PCT 资产验收 PCT→SCAN 首段 ROS 2 链。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path as FilePath
import struct
import time
import uuid

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from scan_planner_msgs.msg import (
    Bspline,
    BsplineDiagnostics,
    GridMapObservationDiagnostics,
    PCTPlanningStatus,
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
MINIMUM_FIRST_SPLINE_VELOCITY_MPS = 0.50
MAXIMUM_FIRST_SPLINE_DURATION_S = 3.20


def _parse_args() -> argparse.Namespace:
    """解析是否只验收局部规划、不启动速度控制。"""

    parser = argparse.ArgumentParser(
        description="验收真实 PCT Path 到 SCAN 首段 B-spline"
    )
    parser.add_argument(
        "--planning-only",
        action="store_true",
        help=(
            "只要求正常 B-spline，并严格要求 /cmd_vel 没有发布者；"
            "默认模式仍验收到非零 controller 命令"
        ),
    )
    return parser.parse_args()


def _time_message(seconds: float) -> Time:
    """把正浮点秒转换为 ROS Time。"""

    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def _synthetic_free_space_cloud(message_stamp: Time) -> PointCloud2:
    """用显式 free 环形射线建立起点附近的已知自由空间。"""

    points = [
        (
            START_BASE_XYZ[0]
            + 4.0 * math.cos(2.0 * math.pi * index / 144.0),
            START_BASE_XYZ[1]
            + 4.0 * math.sin(2.0 * math.pi * index / 144.0),
            START_BASE_XYZ[2],
        )
        for index in range(144)
    ]
    cloud = PointCloud2()
    cloud.header.stamp = message_stamp
    cloud.header.frame_id = "world"
    cloud.height = 1
    cloud.width = len(points)
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
        struct.pack("<fffB", *point, 0) for point in points
    )
    cloud.is_dense = True
    return cloud


def _same_stamp(lhs: Time, rhs: Time) -> bool:
    return lhs.sec == rhs.sec and lhs.nanosec == rhs.nanosec


def main() -> None:
    """发布确定性输入，等待同代 PCT Path 与正常 B-spline。"""

    args = _parse_args()
    rclpy.init()
    node = Node("pct_scan_real_chain_probe")
    sensor_qos = QoSProfile(
        depth=100,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    cached_qos = QoSProfile(
        depth=2,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    diagnostic_qos = QoSProfile(
        depth=64,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    stair_freeze_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    clock_publisher = node.create_publisher(Clock, "/clock", 1)
    odometry_publisher = node.create_publisher(
        Odometry,
        "/body_pose",
        sensor_qos,
    )
    cloud_publisher = node.create_publisher(
        PointCloud2,
        "/cloud_registered",
        sensor_qos,
    )
    goal_publisher = node.create_publisher(PoseStamped, "/pct/goal", 1)
    stair_freeze_publisher = node.create_publisher(
        StairExecutionFreeze,
        "/planning/stair_execution_frozen",
        stair_freeze_qos,
    )
    paths: list[Path] = []
    scan_paths: list[Path] = []
    statuses: list[PCTPlanningStatus] = []
    splines: list[Bspline] = []
    commands: list[Twist] = []
    bspline_diagnostics: list[BsplineDiagnostics] = []
    map_diagnostics: list[GridMapObservationDiagnostics] = []
    node.create_subscription(Path, "/pct/global_path", paths.append, cached_qos)
    node.create_subscription(Path, "/initial_path", scan_paths.append, cached_qos)
    node.create_subscription(
        PCTPlanningStatus,
        "/pct/planning_status",
        statuses.append,
        cached_qos,
    )
    node.create_subscription(Bspline, "/planning/bspline", splines.append, 10)
    node.create_subscription(Twist, "/cmd_vel", commands.append, 10)
    node.create_subscription(
        BsplineDiagnostics,
        "/planning/bspline_diagnostics",
        bspline_diagnostics.append,
        diagnostic_qos,
    )
    node.create_subscription(
        GridMapObservationDiagnostics,
        "/planning/grid_map_observation_diagnostics",
        map_diagnostics.append,
        diagnostic_qos,
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
    next_cloud_time = current_time
    stair_freeze_sequence = 0
    stair_freeze_writer_epoch = uuid.uuid4().hex
    try:
        while time.monotonic() - started_at < 15.0:
            message_stamp = _time_message(current_time)
            clock_publisher.publish(Clock(clock=message_stamp))
            inputs_discovered = (
                odometry_publisher.get_subscription_count() > 0
                and cloud_publisher.get_subscription_count() > 0
            )
            if not inputs_discovered:
                rclpy.spin_once(node, timeout_sec=0.01)
                time.sleep(0.01)
                continue
            odometry.header.stamp = message_stamp
            odometry_publisher.publish(odometry)
            # GridMap 以 20 Hz wall timer 融合占据。点云若快于融合 timer，
            # 每帧回调都会立刻挂起下一次更新，reference Path 的严格 ready
            # 门可能在每个 Odometry 回调时都看到 occ_need_update。这里保持
            # Odometry 40 Hz、点云 10 Hz，给融合与 Path 激活留下确定窗口。
            if current_time + 1.0e-9 >= next_cloud_time:
                cloud_publisher.publish(
                    _synthetic_free_space_cloud(message_stamp)
                )
                next_cloud_time += 0.10
            if (
                not goal_sent
                and goal_publisher.get_subscription_count() > 0
                and any(
                    diagnostic.map_fusion_performed
                    and diagnostic.accepted_endpoint_count > 0
                    for diagnostic in map_diagnostics
                )
                and time.monotonic() - started_at >= 1.0
            ):
                goal = PoseStamped()
                goal.header.stamp = message_stamp
                goal.header.frame_id = "world"
                goal.pose.position.x = GOAL_BASE_XYZ[0]
                goal.pose.position.y = GOAL_BASE_XYZ[1]
                goal.pose.position.z = GOAL_BASE_XYZ[2]
                goal.pose.orientation.z = math.sin(PROBE_CONTRACT.goal_yaw / 2.0)
                goal.pose.orientation.w = math.cos(PROBE_CONTRACT.goal_yaw / 2.0)
                goal_publisher.publish(goal)
                goal_sent = True
            successful_paths = [message for message in paths if message.poses]
            if successful_paths:
                # 生产 SCAN 会在楼梯执行器快照缺失或过期时失败关闭。CPU
                # 探针没有 Isaac root-lock，但必须按同一 typed 合同持续声明
                # 当前 Path 处于非冻结段，不能通过关闭安全门制造假阳性。
                stair_freeze_sequence += 1
                freeze_snapshot = StairExecutionFreeze()
                freeze_snapshot.header.stamp = message_stamp
                freeze_snapshot.header.frame_id = "world"
                freeze_snapshot.reference_path_stamp = (
                    successful_paths[-1].header.stamp
                )
                freeze_snapshot.writer_id = "pct_scan_real_chain_probe"
                freeze_snapshot.writer_epoch = stair_freeze_writer_epoch
                freeze_snapshot.sequence = stair_freeze_sequence
                freeze_snapshot.frozen = False
                stair_freeze_publisher.publish(freeze_snapshot)
            rclpy.spin_once(node, timeout_sec=0.005)

            normal_splines = [
                message
                for message in splines
                if message.pos_pts and not message.emergency_stop
            ]
            nonzero_commands = [
                message
                for message in commands
                if max(
                    abs(message.linear.x),
                    abs(message.linear.y),
                    abs(message.angular.z),
                )
                > 1.0e-6
            ]
            execution_ready = bool(nonzero_commands) or args.planning_only
            if successful_paths and normal_splines and execution_ready:
                path = successful_paths[-1]
                spline = normal_splines[-1]
                matching_scan_paths = [
                    message
                    for message in scan_paths
                    if message.poses
                    and _same_stamp(message.header.stamp, path.header.stamp)
                ]
                if not matching_scan_paths:
                    current_time += 0.02
                    time.sleep(0.025)
                    continue
                scan_path = matching_scan_paths[-1]
                # ROS 消息的字段级相等会覆盖 header 与每个 PoseStamped。
                # CDR 字节包含无语义的对齐填充，不应作为 Topic 合同判据。
                if path != scan_path:
                    raise RuntimeError(
                        "/initial_path 没有原样保留 PCT Path 字段"
                    )
                matching_bspline_diagnostics = [
                    diagnostic
                    for diagnostic in bspline_diagnostics
                    if diagnostic.traj_id == spline.traj_id
                    and _same_stamp(diagnostic.header.stamp, spline.header.stamp)
                    and _same_stamp(diagnostic.start_time, spline.start_time)
                    and _same_stamp(
                        diagnostic.reference_path_stamp,
                        spline.reference_path_stamp,
                    )
                ]
                if not matching_bspline_diagnostics:
                    current_time += 0.02
                    time.sleep(0.025)
                    continue
                spline_diagnostic = matching_bspline_diagnostics[-1]
                if (
                    spline_diagnostic.maximum_velocity_upper_bound
                    < MINIMUM_FIRST_SPLINE_VELOCITY_MPS
                ):
                    raise RuntimeError(
                        "跨层首段 SCAN 轨迹没有恢复巡航速度："
                        f"{spline_diagnostic.maximum_velocity_upper_bound:.6f} m/s"
                    )
                if (
                    spline_diagnostic.trajectory_duration
                    > MAXIMUM_FIRST_SPLINE_DURATION_S
                ):
                    raise RuntimeError(
                        "跨层首段 SCAN 轨迹时间仍然过长："
                        f"{spline_diagnostic.trajectory_duration:.6f} s"
                    )
                successes = [
                    status
                    for status in statuses
                    if status.state == PCTPlanningStatus.SUCCEEDED
                ]
                if not successes:
                    raise RuntimeError("真实 PCT Path 缺少 typed 成功状态")
                if successes[-1].path_point_count != len(path.poses):
                    raise RuntimeError("真实 PCT Path 与 typed 状态点数不一致")
                vertical_span = max(
                    pose.pose.position.z for pose in path.poses
                ) - min(pose.pose.position.z for pose in path.poses)
                if vertical_span < 2.5:
                    raise RuntimeError("真实 PCT Path 没有形成跨楼层高度变化")
                if not _same_stamp(
                    spline.reference_path_stamp,
                    path.header.stamp,
                ):
                    raise RuntimeError("B-spline 与 PCT Path 代际时间戳不一致")
                fused_diagnostics = [
                    diagnostic
                    for diagnostic in map_diagnostics
                    if diagnostic.map_fusion_performed
                    and diagnostic.accepted_endpoint_count > 0
                ]
                if not fused_diagnostics:
                    raise RuntimeError("缺少已融合的 typed GridMap 观测证据")
                map_diagnostic = fused_diagnostics[-1]
                command_publishers = node.get_publishers_info_by_topic(
                    "/cmd_vel"
                )
                if args.planning_only and command_publishers:
                    publisher_names = sorted(
                        endpoint.node_name for endpoint in command_publishers
                    )
                    raise RuntimeError(
                        "只规划模式检测到 /cmd_vel 发布者："
                        f"{publisher_names}"
                    )
                result_label = (
                    "PCT_SCAN_PLANNING_ONLY_OK"
                    if args.planning_only
                    else "PCT_SCAN_CHAIN_OK"
                )
                print(
                    f"{result_label} "
                    f"path_points={len(path.poses)} "
                    f"scan_alias_points={len(scan_path.poses)} "
                    f"spline_points={len(spline.pos_pts)} "
                    f"spline_order={spline.order} "
                    f"spline_final={spline.is_final} "
                    "spline_duration_s="
                    f"{spline_diagnostic.trajectory_duration:.6f} "
                    "spline_velocity_upper_bound_mps="
                    f"{spline_diagnostic.maximum_velocity_upper_bound:.6f} "
                    f"cmd_nonzero={len(nonzero_commands)} "
                    f"cmd_publishers={len(command_publishers)} "
                    f"pct_plan_id={successes[-1].plan_id} "
                    "goal_base_xyz="
                    f"{','.join(f'{value:.6f}' for value in GOAL_BASE_XYZ)} "
                    f"body_height_m={PROBE_CONTRACT.body_height_m:.6f} "
                    f"map_observation_sequence={map_diagnostic.observation_sequence} "
                    f"map_accepted_endpoints={map_diagnostic.accepted_endpoint_count}"
                )
                return
            current_time += 0.02
            # 地图融合与 SCAN FSM 使用独立 timer；仿真时钟不能跑快于墙钟，
            # 否则新鲜度门会把仍在 DDS 队列中的输入判为过期。
            time.sleep(0.025)
        recent_map_diagnostics = [
            (
                diagnostic.observation_sequence,
                diagnostic.map_fusion_performed,
                diagnostic.accepted_endpoint_count,
            )
            for diagnostic in map_diagnostics[-5:]
        ]
        raise RuntimeError(
            "PCT→SCAN 探针超时："
            f"paths={[len(path.poses) for path in paths]}，"
            f"scan_paths={[len(path.poses) for path in scan_paths]}，"
            f"statuses={[(status.state, status.message) for status in statuses]}，"
            f"splines={[(len(s.pos_pts), s.emergency_stop) for s in splines]}，"
            f"commands={len(commands)}，"
            "map_diagnostics="
            f"{recent_map_diagnostics}"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
