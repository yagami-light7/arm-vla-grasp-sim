#!/usr/bin/env python3
"""只读验收真实 Isaac multifloor 的 PCT→SCAN 规划或闭环执行链。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path as FilePath
import time

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scan_planner_msgs.msg import (
    Bspline,
    ControllerStatus,
    GridMapObservationDiagnostics,
    PCTPlanningStatus,
    ScanPlanningStatus,
    StairExecutionFreeze,
)
from sensor_msgs.msg import PointCloud2

from multifloor_probe_contract import load_multifloor_probe_contract


PROJECT_ROOT = FilePath(__file__).resolve().parents[4]
PROBE_CONTRACT = load_multifloor_probe_contract(PROJECT_ROOT)
EXPECTED_START_BASE_XYZ = (
    -3.4748268127441406,
    6.524534225463867,
    0.1636741725990273,
)
BODY_HEIGHT_M = 0.338
EXPECTED_FLAT_GOAL_BASE_XYZ = (
    -3.1032249450683596,
    5.063706207275391,
    0.18431236715521436,
)
EXPECTED_CROSSFLOOR_GOAL_BASE_XYZ = (
    0.4,
    -0.02,
    3.339914814802456,
)


def _parse_args() -> argparse.Namespace:
    """
    @brief 解析真实在线链只读验收参数
    @return 命令行参数命名空间
    """

    parser = argparse.ArgumentParser(
        description=(
            "只订阅真实 Isaac multifloor 的 raw/canonical 观测、"
            "PCT Path 与 SCAN B-spline；绝不发布控制命令"
        )
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=45.0,
        help="等待完整链形成的墙钟超时，单位 s",
    )
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--expect-flat-execution",
        action="store_true",
        help=(
            "验收标定起点到同层短目标的真实闭环执行；默认仍验收跨层只规划"
        ),
    )
    execution_mode.add_argument(
        "--expect-crossfloor-execution",
        action="store_true",
        help="验收标定起点到默认二楼目标的真实闭环执行",
    )
    parser.add_argument(
        "--minimum-planar-displacement-m",
        type=float,
        default=0.25,
        help="执行模式要求的最小真实平面位移，单位 m",
    )
    parser.add_argument(
        "--minimum-vertical-displacement-m",
        type=float,
        default=2.5,
        help="跨层执行模式要求的最小真实爬升高度，单位 m",
    )
    parser.add_argument(
        "--allow-late-join",
        action="store_true",
        help=(
            "仅用于执行结束后的只读补验：允许探针晚加入，并以标定起点"
            "计算位移；正常验收不要启用"
        ),
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout_sec) or args.timeout_sec <= 0.0:
        parser.error("--timeout-sec 必须是有限正数")
    if (
        not math.isfinite(args.minimum_planar_displacement_m)
        or args.minimum_planar_displacement_m <= 0.0
    ):
        parser.error("--minimum-planar-displacement-m 必须是有限正数")
    if (
        not math.isfinite(args.minimum_vertical_displacement_m)
        or args.minimum_vertical_displacement_m <= 0.0
    ):
        parser.error("--minimum-vertical-displacement-m 必须是有限正数")
    if args.allow_late_join and not _execution_expected(args):
        parser.error("--allow-late-join 只能与执行验收模式同时使用")
    return args


def _execution_expected(args: argparse.Namespace) -> bool:
    """
    @brief 判断本次是否需要验收真实速度闭环执行
    @param args 已解析的命令行参数
    @return 同层或跨层执行模式时为真
    """

    return bool(
        args.expect_flat_execution
        or args.expect_crossfloor_execution
    )


def _expected_execution_goal(
    args: argparse.Namespace,
) -> tuple[float, float, float] | None:
    """
    @brief 返回当前执行验收模式对应的 base 目标
    @param args 已解析的命令行参数
    @return 执行目标 xyz；只规划模式返回 None
    """

    if args.expect_flat_execution:
        return EXPECTED_FLAT_GOAL_BASE_XYZ
    if args.expect_crossfloor_execution:
        return EXPECTED_CROSSFLOOR_GOAL_BASE_XYZ
    return None


def _stamp_ns(stamp: Time) -> int:
    """
    @brief 把 ROS Time 转成整数纳秒身份
    @param stamp ROS builtin_interfaces/Time
    @return 纳秒时间戳
    """

    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _same_stamp(lhs: Time, rhs: Time) -> bool:
    """
    @brief 判断两个 ROS Time 是否逐字段相同
    @param lhs 左侧时间
    @param rhs 右侧时间
    @return sec 与 nanosec 都相同时为真
    """

    return lhs.sec == rhs.sec and lhs.nanosec == rhs.nanosec


def _finite_odometry(message: Odometry) -> bool:
    """
    @brief 检查 Odometry 导航所需的位姿和速度是否全为有限数
    @param message 待检查的里程计消息
    @return 所有必要字段有限时为真
    """

    pose = message.pose.pose
    twist = message.twist.twist
    values = (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
        twist.linear.x,
        twist.linear.y,
        twist.linear.z,
        twist.angular.x,
        twist.angular.y,
        twist.angular.z,
    )
    return all(math.isfinite(value) for value in values)


def _field_names(message: PointCloud2) -> set[str]:
    """
    @brief 提取 PointCloud2 字段名称集合
    @param message 待检查的点云消息
    @return 字段名集合
    """

    return {field.name for field in message.fields}


def _bounded_append(messages: list, message, *, limit: int = 256) -> None:
    """
    @brief 保存最近的 ROS 消息并限制内存占用
    @param messages 对应 Topic 的消息列表
    @param message 新收到的 ROS 消息
    @param limit 最多保留的消息数量
    @return 无返回值
    """

    messages.append(message)
    if len(messages) > limit:
        del messages[:-limit]


def _matching_by_stamp(raw_messages: list, canonical_messages: list):
    """
    @brief 查找 raw 与 canonical Topic 最近的同时间戳消息对
    @param raw_messages 原始 Topic 消息序列
    @param canonical_messages 规范化 Topic 消息序列
    @return 找到时返回二元组，否则返回 None
    """

    canonical_by_stamp = {
        _stamp_ns(message.header.stamp): message
        for message in canonical_messages
    }
    for raw_message in reversed(raw_messages):
        canonical_message = canonical_by_stamp.get(
            _stamp_ns(raw_message.header.stamp)
        )
        if canonical_message is not None:
            return raw_message, canonical_message
    return None


def _matching_scan_trajectory_pair(
    splines: list[Bspline],
    statuses: list[ScanPlanningStatus],
) -> tuple[Bspline, ScanPlanningStatus] | None:
    """
    @brief 按完整轨迹身份配对 SCAN B-spline 与类型化状态
    @param splines 已按当前 Path 代际筛选的 B-spline 消息
    @param statuses 已按当前 Path 代际筛选的 SCAN 状态消息
    @return 最近一组完整身份一致的消息；尚未形成时返回 None
    """

    status_by_identity = {
        (
            _stamp_ns(status.reference_path_stamp),
            _stamp_ns(status.bspline_header_stamp),
            _stamp_ns(status.trajectory_start_time),
            int(status.trajectory_id),
        ): status
        for status in statuses
    }
    for spline in reversed(splines):
        identity = (
            _stamp_ns(spline.reference_path_stamp),
            _stamp_ns(spline.header.stamp),
            _stamp_ns(spline.start_time),
            int(spline.traj_id),
        )
        status = status_by_identity.get(identity)
        if status is not None:
            return spline, status
    return None


def _publisher_names(node: Node, topic: str) -> list[str]:
    """
    @brief 查询指定 Topic 的发布节点名称
    @param node 用于图查询的 rclpy 节点
    @param topic 绝对 Topic 名称
    @return 排序后的发布节点名称
    """

    return sorted(
        endpoint.node_name
        for endpoint in node.get_publishers_info_by_topic(topic)
    )


def _subscription_names(node: Node, topic: str) -> list[str]:
    """
    @brief 查询指定 Topic 的订阅节点名称
    @param node 用于图查询的 rclpy 节点
    @param topic 绝对 Topic 名称
    @return 排序后的订阅节点名称
    """

    return sorted(
        endpoint.node_name
        for endpoint in node.get_subscriptions_info_by_topic(topic)
    )


def main() -> None:
    """
    @brief 验收真实 Isaac→bridge→PCT→SCAN 的规划或闭环执行链
    @return 验收通过后打印单行证据，失败时抛出 RuntimeError
    """

    args = _parse_args()
    execution_expected = _execution_expected(args)
    expected_execution_goal = _expected_execution_goal(args)
    rclpy.init()
    node = Node("live_multifloor_pct_scan_probe")
    sensor_qos = QoSProfile(
        depth=128,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    cached_qos = QoSProfile(
        depth=8,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    diagnostic_qos = QoSProfile(
        depth=64,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    raw_odometry: list[Odometry] = []
    canonical_odometry: list[Odometry] = []
    raw_clouds: list[PointCloud2] = []
    canonical_clouds: list[PointCloud2] = []
    pct_paths: list[Path] = []
    scan_paths: list[Path] = []
    pct_statuses: list[PCTPlanningStatus] = []
    splines: list[Bspline] = []
    scan_statuses: list[ScanPlanningStatus] = []
    controller_statuses: list[ControllerStatus] = []
    freeze_snapshots: list[StairExecutionFreeze] = []
    map_diagnostics: list[GridMapObservationDiagnostics] = []

    def subscribe(message_type, topic, destination, qos) -> None:
        """
        @brief 创建一个保留最近消息的只读订阅
        @param message_type ROS 消息类型
        @param topic 绝对 Topic 名称
        @param destination 保存消息的列表
        @param qos 与生产发布端匹配的 QoS
        @return 无返回值
        """

        node.create_subscription(
            message_type,
            topic,
            lambda message: _bounded_append(destination, message),
            qos,
        )

    subscribe(Odometry, "/isaac/body_pose_raw", raw_odometry, sensor_qos)
    subscribe(Odometry, "/body_pose", canonical_odometry, sensor_qos)
    subscribe(
        PointCloud2,
        "/isaac/cloud_registered_raw",
        raw_clouds,
        sensor_qos,
    )
    subscribe(
        PointCloud2,
        "/cloud_registered",
        canonical_clouds,
        sensor_qos,
    )
    subscribe(Path, "/pct/global_path", pct_paths, cached_qos)
    subscribe(Path, "/initial_path", scan_paths, cached_qos)
    subscribe(
        PCTPlanningStatus,
        "/pct/planning_status",
        pct_statuses,
        cached_qos,
    )
    subscribe(Bspline, "/planning/bspline", splines, cached_qos)
    subscribe(
        ScanPlanningStatus,
        "/planning/scan_status",
        scan_statuses,
        diagnostic_qos,
    )
    if execution_expected:
        subscribe(
            ControllerStatus,
            "/planning/controller_status",
            controller_statuses,
            diagnostic_qos,
        )
    subscribe(
        StairExecutionFreeze,
        "/planning/stair_execution_frozen",
        freeze_snapshots,
        cached_qos,
    )
    subscribe(
        GridMapObservationDiagnostics,
        "/planning/grid_map_observation_diagnostics",
        map_diagnostics,
        diagnostic_qos,
    )

    started_at = time.monotonic()
    initial_raw_position: tuple[float, float, float] | None = (
        EXPECTED_START_BASE_XYZ if args.allow_late_join else None
    )
    start_observation_verified = not args.allow_late_join
    try:
        while time.monotonic() - started_at < args.timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.05)

            if raw_odometry and initial_raw_position is None:
                first_position = raw_odometry[0].pose.pose.position
                initial_raw_position = (
                    float(first_position.x),
                    float(first_position.y),
                    float(first_position.z),
                )
                initial_xy_error = math.hypot(
                    initial_raw_position[0] - EXPECTED_START_BASE_XYZ[0],
                    initial_raw_position[1] - EXPECTED_START_BASE_XYZ[1],
                )
                initial_z_error = abs(
                    initial_raw_position[2] - EXPECTED_START_BASE_XYZ[2]
                )
                if initial_xy_error > 0.35 or initial_z_error > 0.15:
                    raise RuntimeError(
                        "探针没有从标定起点观察到本轮 episode："
                        f"xy_error={initial_xy_error:.6f}m, "
                        f"z_error={initial_z_error:.6f}m；"
                        "请先启动探针，再启动 Isaac"
                    )

            odometry_pair = _matching_by_stamp(
                raw_odometry,
                canonical_odometry,
            )
            cloud_pair = _matching_by_stamp(raw_clouds, canonical_clouds)
            successful_paths = [path for path in pct_paths if path.poses]
            successful_statuses = [
                status
                for status in pct_statuses
                if status.state == PCTPlanningStatus.SUCCEEDED
            ]
            normal_splines = [
                spline
                for spline in splines
                if spline.pos_pts and not spline.emergency_stop
            ]
            fused_observations = [
                diagnostic
                for diagnostic in map_diagnostics
                if diagnostic.map_fusion_performed
                and diagnostic.accepted_endpoint_count > 0
            ]
            if not all(
                (
                    odometry_pair,
                    cloud_pair,
                    successful_paths,
                    scan_paths,
                    successful_statuses,
                    normal_splines,
                    scan_statuses,
                    freeze_snapshots,
                    fused_observations,
                    initial_raw_position,
                )
            ):
                continue
            if execution_expected and not controller_statuses:
                continue

            raw_odom, canonical_odom = odometry_pair
            raw_cloud, canonical_cloud = cloud_pair
            path = successful_paths[-1]
            matching_scan_paths = [
                candidate
                for candidate in scan_paths
                if candidate.poses
                and _same_stamp(candidate.header.stamp, path.header.stamp)
            ]
            matching_splines = [
                spline
                for spline in normal_splines
                if _same_stamp(
                    spline.reference_path_stamp,
                    path.header.stamp,
                )
            ]
            matching_statuses = [
                status
                for status in scan_statuses
                if status.trajectory_present
                and not status.stop_required
                and not status.global_replan_recommended
                and _same_stamp(
                    status.reference_path_stamp,
                    path.header.stamp,
                )
            ]
            matching_trajectory_pair = _matching_scan_trajectory_pair(
                matching_splines,
                matching_statuses,
            )
            matching_freezes = [
                snapshot
                for snapshot in freeze_snapshots
                if not snapshot.frozen
                and _same_stamp(
                    snapshot.reference_path_stamp,
                    path.header.stamp,
                )
            ]
            matching_controller_statuses = [
                status
                for status in controller_statuses
                if status.accepted
                and not status.emergency_stop
                and status.command_violation_count == 0
                and status.command_sample_count > 0
                and _same_stamp(
                    status.reference_path_stamp,
                    path.header.stamp,
                )
            ]
            moving_controller_statuses = [
                status
                for status in matching_controller_statuses
                if max(status.max_abs_vx, status.max_abs_vy) > 0.01
            ]
            goal_reached_statuses = [
                status
                for status in matching_controller_statuses
                if status.state == ControllerStatus.STATE_GOAL_REACHED
            ]
            if not all(
                (
                    matching_scan_paths,
                    matching_trajectory_pair,
                    matching_freezes,
                )
            ):
                continue
            if execution_expected and not (
                moving_controller_statuses
                and goal_reached_statuses
            ):
                continue

            scan_path = matching_scan_paths[-1]
            assert matching_trajectory_pair is not None
            spline, scan_status = matching_trajectory_pair
            pct_status = successful_statuses[-1]
            map_diagnostic = fused_observations[-1]
            controller_status = (
                goal_reached_statuses[-1]
                if goal_reached_statuses
                else None
            )
            command_evidence_status = (
                max(
                    moving_controller_statuses,
                    key=lambda status: max(
                        status.max_abs_vx,
                        status.max_abs_vy,
                    ),
                )
                if moving_controller_statuses
                else None
            )

            if raw_odom.header.frame_id != "world":
                raise RuntimeError("raw Odometry frame_id 不是 world")
            if raw_odom.child_frame_id != "base_link":
                raise RuntimeError("raw Odometry child_frame_id 不是 base_link")
            if canonical_odom.header.frame_id != "world":
                raise RuntimeError("canonical Odometry frame_id 不是 world")
            if canonical_odom.child_frame_id != "base_link":
                raise RuntimeError(
                    "canonical Odometry child_frame_id 不是 base_link"
                )
            if not _finite_odometry(raw_odom) or not _finite_odometry(
                canonical_odom
            ):
                raise RuntimeError("raw/canonical Odometry 包含 NaN 或 Inf")

            raw_position = raw_odom.pose.pose.position
            assert initial_raw_position is not None
            start_xy_error = math.hypot(
                initial_raw_position[0] - EXPECTED_START_BASE_XYZ[0],
                initial_raw_position[1] - EXPECTED_START_BASE_XYZ[1],
            )
            start_z_error = abs(
                initial_raw_position[2] - EXPECTED_START_BASE_XYZ[2]
            )
            if start_xy_error > 0.35 or start_z_error > 0.15:
                raise RuntimeError(
                    "Isaac 标准 Go2 没有稳定在标定起点："
                    f"xy_error={start_xy_error:.6f}m, "
                    f"z_error={start_z_error:.6f}m"
                )
            planar_displacement = math.hypot(
                raw_position.x - initial_raw_position[0],
                raw_position.y - initial_raw_position[1],
            )
            vertical_displacement = (
                raw_position.z - initial_raw_position[2]
            )

            if raw_cloud.header.frame_id != "world":
                raise RuntimeError("raw PointCloud2 frame_id 不是 world")
            if canonical_cloud.header.frame_id != "world":
                raise RuntimeError("canonical PointCloud2 frame_id 不是 world")
            if not {"x", "y", "z"}.issubset(_field_names(raw_cloud)):
                raise RuntimeError("raw PointCloud2 缺少 x/y/z 字段")
            if not {
                "x",
                "y",
                "z",
                "ray_endpoint_type",
            }.issubset(_field_names(canonical_cloud)):
                raise RuntimeError(
                    "canonical PointCloud2 缺少 ray_endpoint_type 合同字段"
                )
            raw_point_count = int(raw_cloud.width) * int(raw_cloud.height)
            canonical_point_count = (
                int(canonical_cloud.width) * int(canonical_cloud.height)
            )
            if raw_point_count <= 0 or canonical_point_count <= 0:
                raise RuntimeError("raw/canonical PointCloud2 为空")

            if path != scan_path:
                raise RuntimeError("/initial_path 没有原样保留 PCT Path 字段")
            if path.header.frame_id != "world":
                raise RuntimeError("PCT Path frame_id 不是 world")
            if pct_status.path_point_count != len(path.poses):
                raise RuntimeError("PCT typed 状态与 Path 点数不一致")
            vertical_span = max(
                pose.pose.position.z for pose in path.poses
            ) - min(pose.pose.position.z for pose in path.poses)
            if not args.expect_flat_execution and vertical_span < 2.5:
                raise RuntimeError("真实 PCT Path 没有形成跨楼层高度变化")
            if execution_expected:
                assert expected_execution_goal is not None
                goal_ground = path.poses[-1].pose.position
                goal_base_xyz = (
                    goal_ground.x,
                    goal_ground.y,
                    goal_ground.z + BODY_HEIGHT_M,
                )
                goal_error = math.dist(
                    goal_base_xyz,
                    expected_execution_goal,
                )
                if goal_error > 0.03:
                    raise RuntimeError(
                        "执行模式收到的不是标定目标："
                        f"goal_error={goal_error:.6f}m"
                    )
                final_goal_xy_error = math.hypot(
                    raw_position.x - expected_execution_goal[0],
                    raw_position.y - expected_execution_goal[1],
                )
                final_goal_z_error = abs(
                    raw_position.z - expected_execution_goal[2]
                )
                if final_goal_xy_error > 0.20:
                    raise RuntimeError(
                        "GOAL_REACHED 后机器人与目标的平面距离过大："
                        f"goal_xy_error={final_goal_xy_error:.6f}m"
                    )
                if final_goal_z_error > 0.20:
                    raise RuntimeError(
                        "GOAL_REACHED 后机器人与目标的高度距离过大："
                        f"goal_z_error={final_goal_z_error:.6f}m"
                    )
            else:
                final_goal_xy_error = 0.0
                final_goal_z_error = 0.0
            if args.expect_flat_execution:
                if vertical_span > 0.25:
                    raise RuntimeError("同层短目标 Path 出现异常跨层高度")

            if spline.order != 3 or len(spline.pos_pts) < 4:
                raise RuntimeError("SCAN 没有发布合法三次 B-spline")
            if scan_status.trajectory_emergency_stop:
                raise RuntimeError("SCAN typed 状态报告 emergency_stop")

            cmd_publishers = _publisher_names(node, "/cmd_vel")
            cmd_subscriptions = _subscription_names(node, "/cmd_vel")
            if not execution_expected and (
                cmd_publishers or cmd_subscriptions
            ):
                raise RuntimeError(
                    "只规划模式不应创建 /cmd_vel 端点："
                    f"publishers={cmd_publishers}, "
                    f"subscriptions={cmd_subscriptions}"
                )
            if execution_expected:
                if len(cmd_publishers) != 1 or len(cmd_subscriptions) != 1:
                    raise RuntimeError(
                        "执行模式必须恰好一发一收 /cmd_vel："
                        f"publishers={cmd_publishers}, "
                        f"subscriptions={cmd_subscriptions}"
                    )
                assert controller_status is not None
                assert command_evidence_status is not None
                if command_evidence_status.max_abs_vx > 0.800001:
                    raise RuntimeError("controller vx 超过 MoE-CTS policy 包络")
                if command_evidence_status.max_abs_vy > 0.500001:
                    raise RuntimeError("controller vy 超过 MoE-CTS policy 包络")
                if command_evidence_status.max_abs_wz > 0.800001:
                    raise RuntimeError("controller wz 超过 MoE-CTS policy 包络")
                maximum_planar_command = max(
                    command_evidence_status.max_abs_vx,
                    command_evidence_status.max_abs_vy,
                )
                if maximum_planar_command <= 0.01:
                    continue
                if planar_displacement < args.minimum_planar_displacement_m:
                    continue
                if (
                    args.expect_crossfloor_execution
                    and vertical_displacement
                    < args.minimum_vertical_displacement_m
                ):
                    continue

            required_publishers = {
                "/isaac/body_pose_raw": 1,
                "/isaac/cloud_registered_raw": 1,
                "/body_pose": 1,
                "/cloud_registered": 1,
                "/pct/global_path": 1,
                "/initial_path": 1,
                "/planning/bspline": 1,
            }
            duplicate_topics = {
                topic: names
                for topic, expected_count in required_publishers.items()
                if len(names := _publisher_names(node, topic))
                != expected_count
            }
            if duplicate_topics:
                raise RuntimeError(
                    "导航主链存在缺失或重复 publisher："
                    f"{duplicate_topics}"
                )

            elapsed = time.monotonic() - started_at
            if args.expect_flat_execution:
                assert controller_status is not None
                assert command_evidence_status is not None
                print(
                    "LIVE_MULTIFLOOR_PCT_SCAN_FLAT_EXECUTION_OK "
                    f"elapsed_s={elapsed:.3f} "
                    f"raw_points={raw_point_count} "
                    f"canonical_points={canonical_point_count} "
                    f"path_points={len(path.poses)} "
                    f"spline_points={len(spline.pos_pts)} "
                    f"traj_id={spline.traj_id} "
                    f"controller_state={controller_status.state} "
                    "command_samples="
                    f"{command_evidence_status.command_sample_count} "
                    f"max_abs_vx={command_evidence_status.max_abs_vx:.6f} "
                    f"max_abs_vy={command_evidence_status.max_abs_vy:.6f} "
                    f"max_abs_wz={command_evidence_status.max_abs_wz:.6f} "
                    f"planar_displacement_m={planar_displacement:.6f} "
                    f"start_xy_error_m={start_xy_error:.6f} "
                    f"start_z_error_m={start_z_error:.6f} "
                    "start_observation_verified="
                    f"{str(start_observation_verified).lower()} "
                    "cmd_publishers=1 cmd_subscriptions=1"
                )
                return
            if args.expect_crossfloor_execution:
                assert controller_status is not None
                assert command_evidence_status is not None
                print(
                    "LIVE_MULTIFLOOR_PCT_SCAN_CROSSFLOOR_EXECUTION_OK "
                    f"elapsed_s={elapsed:.3f} "
                    f"raw_points={raw_point_count} "
                    f"canonical_points={canonical_point_count} "
                    f"path_points={len(path.poses)} "
                    f"vertical_span_m={vertical_span:.6f} "
                    f"spline_points={len(spline.pos_pts)} "
                    f"traj_id={spline.traj_id} "
                    f"controller_state={controller_status.state} "
                    "command_samples="
                    f"{command_evidence_status.command_sample_count} "
                    f"max_abs_vx={command_evidence_status.max_abs_vx:.6f} "
                    f"max_abs_vy={command_evidence_status.max_abs_vy:.6f} "
                    f"max_abs_wz={command_evidence_status.max_abs_wz:.6f} "
                    f"planar_displacement_m={planar_displacement:.6f} "
                    f"vertical_displacement_m={vertical_displacement:.6f} "
                    f"goal_xy_error_m={final_goal_xy_error:.6f} "
                    f"goal_z_error_m={final_goal_z_error:.6f} "
                    f"start_xy_error_m={start_xy_error:.6f} "
                    f"start_z_error_m={start_z_error:.6f} "
                    "start_observation_verified="
                    f"{str(start_observation_verified).lower()} "
                    "cmd_publishers=1 cmd_subscriptions=1"
                )
                return
            print(
                "LIVE_MULTIFLOOR_PCT_SCAN_OK "
                f"elapsed_s={elapsed:.3f} "
                f"raw_points={raw_point_count} "
                f"canonical_points={canonical_point_count} "
                f"map_accepted_endpoints="
                f"{map_diagnostic.accepted_endpoint_count} "
                f"path_points={len(path.poses)} "
                f"vertical_span_m={vertical_span:.6f} "
                f"spline_points={len(spline.pos_pts)} "
                f"spline_order={spline.order} "
                f"traj_id={spline.traj_id} "
                f"start_xy_error_m={start_xy_error:.6f} "
                f"start_z_error_m={start_z_error:.6f} "
                "cmd_publishers=0 cmd_subscriptions=0"
            )
            return

        raise RuntimeError(
            "真实 multifloor PCT→SCAN 验收超时："
            f"raw_odom={len(raw_odometry)}, "
            f"body_pose={len(canonical_odometry)}, "
            f"raw_cloud={len(raw_clouds)}, "
            f"cloud={len(canonical_clouds)}, "
            f"pct_paths={[len(path.poses) for path in pct_paths[-3:]]}, "
            f"scan_paths={[len(path.poses) for path in scan_paths[-3:]]}, "
            f"pct_states={[status.state for status in pct_statuses[-5:]]}, "
            f"splines={[(len(s.pos_pts), s.emergency_stop) for s in splines[-5:]]}, "
            f"scan_states={[status.state for status in scan_statuses[-5:]]}, "
            "controller_states="
            f"{[status.state for status in controller_statuses[-5:]]}, "
            "map_observations="
            f"{[(d.map_fusion_performed, d.accepted_endpoint_count) for d in map_diagnostics[-5:]]}"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
