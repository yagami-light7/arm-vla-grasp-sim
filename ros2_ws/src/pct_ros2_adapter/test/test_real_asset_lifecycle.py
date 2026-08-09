"""真实多楼层资产上的 PCT ROS 2 topic 生命周期验收。"""

from __future__ import annotations

import math
from pathlib import Path
import time

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMessage
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from scan_planner_msgs.msg import PCTPlanningStatus

from pct_ros2_adapter.node import PCTROS2Adapter
from source.interfaces import SimulationState
from source.navigation.scan_stair_freeze import (
    ScanStairFreezeConfig,
    ScanStairFreezeController,
    extract_stair_components,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REAL_ASSETS = (
    PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle",
    PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy",
    PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_collision.ply",
)


def _time_message(seconds: float) -> Time:
    whole = int(math.floor(seconds))
    nanosecond = int(round((seconds - whole) * 1.0e9))
    if nanosecond >= 1_000_000_000:
        whole += 1
        nanosecond -= 1_000_000_000
    return Time(sec=whole, nanosec=nanosecond)


def _spin_until(
    executor: SingleThreadedExecutor,
    predicate,
    *,
    timeout_sec: float,
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("ROS 2 生命周期未在测试时限内完成")


@pytest.mark.skipif(
    not all(path.is_file() for path in REAL_ASSETS),
    reason="本机未准备 ignored 的 multi-floor PCT 真实资产",
)
def test_real_asset_goal_publishes_ground_path_and_typed_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """经真实 ROS topic 输入时发布 169 点跨层地面 Path。"""

    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("PCT_SCAN_PROJECT_ROOT", str(PROJECT_ROOT))
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init()
    adapter = PCTROS2Adapter()
    driver = Node("pct_real_asset_lifecycle_driver")
    executor = SingleThreadedExecutor()
    executor.add_node(adapter)
    executor.add_node(driver)

    sensor_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    cached_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    clock_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    clock_publisher = driver.create_publisher(Clock, "/clock", clock_qos)
    odometry_publisher = driver.create_publisher(
        Odometry,
        "/body_pose",
        sensor_qos,
    )
    goal_publisher = driver.create_publisher(
        PoseStamped,
        "/pct/goal",
        1,
    )
    paths: list[PathMessage] = []
    statuses: list[PCTPlanningStatus] = []
    driver.create_subscription(
        PathMessage,
        "/pct/global_path",
        paths.append,
        cached_qos,
    )
    driver.create_subscription(
        PCTPlanningStatus,
        "/pct/planning_status",
        statuses.append,
        cached_qos,
    )

    current_time = 10.0
    odometry = Odometry()
    odometry.header.frame_id = "world"
    odometry.child_frame_id = "base_link"
    odometry.pose.pose.position.x = -3.50493
    odometry.pose.pose.position.y = 6.7491
    odometry.pose.pose.position.z = 0.1251194919198741
    odometry.pose.pose.orientation.w = 1.0
    try:
        for _ in range(5):
            stamp = _time_message(current_time)
            clock_publisher.publish(Clock(clock=stamp))
            target_ns = int(round(current_time * 1.0e9))
            _spin_until(
                executor,
                lambda: adapter._clock_now_ns() >= target_ns,
                timeout_sec=0.5,
            )
            current_time += 0.02
        assert adapter._clock_now_ns() > 0
        for _ in range(3):
            stamp = _time_message(current_time)
            clock_publisher.publish(Clock(clock=stamp))
            target_ns = int(round(current_time * 1.0e9))
            _spin_until(
                executor,
                lambda: adapter._clock_now_ns() >= target_ns,
                timeout_sec=0.5,
            )
            odometry.header.stamp = stamp
            odometry_publisher.publish(odometry)
            executor.spin_once(timeout_sec=0.05)
            current_time += 0.02
        _spin_until(
            executor,
            lambda: adapter._latest_odometry is not None,
            timeout_sec=1.0,
        )

        goal = PoseStamped()
        goal.header.stamp = _time_message(current_time)
        goal.header.frame_id = "world"
        goal.pose.position.x = 0.4
        goal.pose.position.y = -0.1
        goal.pose.position.z = 3.303963228415887
        goal.pose.orientation.z = math.sin(-math.pi / 4.0)
        goal.pose.orientation.w = math.cos(-math.pi / 4.0)
        goal_publisher.publish(goal)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(
            status.state == PCTPlanningStatus.SUCCEEDED
            for status in statuses
        ):
            stamp = _time_message(current_time)
            clock_publisher.publish(Clock(clock=stamp))
            target_ns = int(round(current_time * 1.0e9))
            _spin_until(
                executor,
                lambda: adapter._clock_now_ns() >= target_ns,
                timeout_sec=0.5,
            )
            odometry.header.stamp = stamp
            odometry_publisher.publish(odometry)
            executor.spin_once(timeout_sec=0.02)
            current_time += 0.02

        successes = [
            status
            for status in statuses
            if status.state == PCTPlanningStatus.SUCCEEDED
        ]
        nonempty_paths = [path for path in paths if path.poses]
        assert len(successes) == 1
        assert len(nonempty_paths) == 1
        path = nonempty_paths[0]
        assert successes[0].plan_id == 1
        assert successes[0].path_point_count == 169
        assert len(path.poses) == 169
        assert path.header.frame_id == "world"
        assert path.poses[-1].pose.position.x == pytest.approx(0.4)
        assert path.poses[-1].pose.position.y == pytest.approx(-0.1)
        assert path.poses[-1].pose.position.z == pytest.approx(
            3.0039632284158873
        )
        points_ground_xyz = tuple(
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            )
            for pose in path.poses
        )
        stair_components = extract_stair_components(
            points_ground_xyz,
            ScanStairFreezeConfig(),
        )
        assert len(stair_components) == 1
        assert max(point[2] for point in stair_components[0]) - min(
            point[2] for point in stair_components[0]
        ) > 2.0

        # 生产 PCT Path 不携带手工楼梯索引；用同一条真实 ROS 输出证明几何
        # 识别结果会在楼梯组件入口直接切换到 pct_scene 同构的底盘冻结动作。
        stair_controller = ScanStairFreezeController(ScanStairFreezeConfig())
        stair_controller.reset(
            points_ground_xyz,
            path_source="ros2:/pct/global_path",
            path_stamp_ns=(
                int(path.header.stamp.sec) * 1_000_000_000
                + int(path.header.stamp.nanosec)
            ),
        )
        entry = stair_components[0][0]
        freeze_action = stair_controller.compute_action(
            SimulationState(
                step_index=0,
                timestamp=current_time,
                robot_root_pose=(
                    entry[0],
                    entry[1],
                    entry[2] + stair_controller.config.body_height_m,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                robot_root_velocity=(0.0,) * 6,
            )
        )
        assert freeze_action is not None
        assert freeze_action.source == "scan_stair_freeze_activated"
        assert freeze_action.base_velocity == (0.0, 0.0, 0.0)
        assert freeze_action.metadata["navigation_base_pose_lock"] is True
        assert freeze_action.metadata["navigation_support_joint_lock"] is True
        assert freeze_action.metadata["navigation_full_body_joint_lock"] is True
        assert freeze_action.metadata["navigation_cmd_vel_inhibit"] is True
        freeze_status = stair_controller.status()
        assert freeze_status["component_source"] == "geometry_heuristic"
        assert freeze_status["component_count"] == 1
        assert freeze_status["phase"] == "active"
        assert any(not message.poses for message in paths)
    finally:
        executor.shutdown(timeout_sec=1.0)
        adapter.destroy_node()
        driver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
