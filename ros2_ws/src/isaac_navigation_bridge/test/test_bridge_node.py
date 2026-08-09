"""Isaac 导航桥节点合同测试。"""

from __future__ import annotations

from pathlib import Path
import math
import time

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMessage
import numpy as np
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from isaac_navigation_bridge.bridge_node import IsaacNavigationBridge


class _CapturePublisher:
    """只记录发布消息的测试替身。"""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


@pytest.fixture
def bridge_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> IsaacNavigationBridge:
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init()
    node = IsaacNavigationBridge()
    try:
        yield node
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _odometry(stamp_sec: int = 10) -> Odometry:
    message = Odometry()
    message.header.stamp = Time(sec=stamp_sec)
    message.header.frame_id = "world"
    message.child_frame_id = "base_link"
    message.pose.pose.position.z = 0.30
    message.pose.pose.orientation.w = 1.0
    return message


def _cloud(
    stamp_sec: int = 10,
    frame_id: str = "world",
    points: tuple[tuple[float, float, float], ...] = tuple(
        (1.0, 0.0, 0.30) for _ in range(64)
    ),
):
    header = Header(stamp=Time(sec=stamp_sec), frame_id=frame_id)
    return point_cloud2.create_cloud_xyz32(
        header,
        points,
    )


def _path(
    stamp_sec: int = 10,
    frame_id: str = "world",
    *,
    pose_stamp_sec: int | None = None,
    pose_frame_id: str = "world",
    points: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.15),
    ),
    orientation_w: float = 1.0,
) -> PathMessage:
    message = PathMessage()
    message.header = Header(stamp=Time(sec=stamp_sec), frame_id=frame_id)
    pose_stamp = stamp_sec if pose_stamp_sec is None else pose_stamp_sec
    for x, y, z in points:
        pose = PoseStamped()
        pose.header = Header(
            stamp=Time(sec=pose_stamp),
            frame_id=pose_frame_id,
        )
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = orientation_w
        message.poses.append(pose)
    return message


def test_node_forces_sim_time_and_sensor_data_qos(
    bridge_node: IsaacNavigationBridge,
) -> None:
    assert bridge_node.get_parameter("use_sim_time").value is True
    assert bridge_node._filter_config.path_ground_clearance_m == pytest.approx(
        0.05
    )
    assert (
        bridge_node._filter_config.path_ground_stair_minimum_slope
        == pytest.approx(0.20)
    )
    assert (
        bridge_node._filter_config.path_ground_stair_clearance_m
        == pytest.approx(0.09)
    )
    assert (
        bridge_node._filter_config.path_ground_stair_band_down_m
        == pytest.approx(0.09)
    )

    publisher_info = bridge_node.get_publishers_info_by_topic("/body_pose")
    subscription_info = bridge_node.get_subscriptions_info_by_topic(
        "/isaac/body_pose_raw"
    )
    path_subscription_info = bridge_node.get_subscriptions_info_by_topic(
        "/initial_path"
    )
    assert len(publisher_info) == 1
    assert len(subscription_info) == 1
    for endpoint in (*publisher_info, *subscription_info):
        assert endpoint.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
        assert endpoint.qos_profile.durability == DurabilityPolicy.VOLATILE
    assert len(path_subscription_info) == 1
    assert (
        path_subscription_info[0].qos_profile.reliability
        == ReliabilityPolicy.RELIABLE
    )
    assert (
        path_subscription_info[0].qos_profile.durability
        == DurabilityPolicy.TRANSIENT_LOCAL
    )


def test_node_caches_only_valid_world_ground_path(
    bridge_node: IsaacNavigationBridge,
) -> None:
    bridge_node._path_callback(_path())
    cache = bridge_node._ground_path_cache
    assert cache is not None
    np.testing.assert_allclose(
        cache.path.points_world_xyz,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.15],
        ],
    )
    generation = cache.generation

    bridge_node._path_callback(_path(stamp_sec=0))
    bridge_node._path_callback(_path(frame_id="sensor"))
    assert bridge_node._ground_path_cache is cache
    assert bridge_node._ground_path_cache_generation == generation


def test_path_cache_mirrors_scan_pose_validation_and_spacing(
    bridge_node: IsaacNavigationBridge,
) -> None:
    bridge_node._path_callback(
        _path(
            stamp_sec=10,
            pose_frame_id="",
            orientation_w=2.0,
            points=(
                (0.0, 0.0, 0.0),
                (0.01, 0.0, 0.0),
                (1.0, 0.0, 0.15),
            ),
        )
    )
    cache = bridge_node._ground_path_cache
    assert cache is not None
    assert cache.path.point_count == 2

    invalid_pose_stamp = _path(stamp_sec=11, pose_stamp_sec=0)
    invalid_pose_frame = _path(stamp_sec=11, pose_frame_id="sensor")
    invalid_orientation = _path(stamp_sec=11)
    invalid_orientation.poses[0].pose.orientation.w = math.nan
    zero_orientation = _path(stamp_sec=11, orientation_w=0.0)
    too_close = _path(
        stamp_sec=11,
        points=((0.0, 0.0, 0.0), (0.01, 0.0, 0.0)),
    )
    for message in (
        invalid_pose_stamp,
        invalid_pose_frame,
        invalid_orientation,
        zero_orientation,
        too_close,
    ):
        bridge_node._path_callback(message)
        assert bridge_node._ground_path_cache is cache
        assert bridge_node._latest_ground_path_stamp_ns == 10_000_000_000


def test_empty_path_clears_cache_and_old_stamp_cannot_restore_it(
    bridge_node: IsaacNavigationBridge,
) -> None:
    bridge_node._path_callback(_path(stamp_sec=20))
    first_cache = bridge_node._ground_path_cache
    assert first_cache is not None

    empty = PathMessage()
    empty.header = Header(stamp=Time(sec=21), frame_id="world")
    bridge_node._path_callback(empty)
    assert bridge_node._ground_path_cache is None
    assert bridge_node._ground_path_cache_generation == 2
    assert bridge_node._latest_ground_path_stamp_ns == 21_000_000_000

    bridge_node._path_callback(_path(stamp_sec=20))
    assert bridge_node._ground_path_cache is None
    assert bridge_node._ground_path_cache_generation == 2

    bridge_node._path_callback(_path(stamp_sec=22))
    second_cache = bridge_node._ground_path_cache
    assert second_cache is not None
    assert second_cache.generation == 3
    bridge_node._path_callback(empty)
    assert bridge_node._ground_path_cache is second_cache


def test_new_path_drops_older_cloud_and_publishes_support_clearing_rays(
    bridge_node: IsaacNavigationBridge,
) -> None:
    cloud_capture = _CapturePublisher()
    bridge_node._cloud_publisher = cloud_capture
    bridge_node._body_pose_callback(_odometry(stamp_sec=10))
    bridge_node._path_callback(_path(stamp_sec=11))
    ramp_point = tuple((1.0, 0.0, 0.15) for _ in range(64))

    bridge_node._cloud_callback(_cloud(stamp_sec=10, points=ramp_point))
    assert cloud_capture.messages == []

    bridge_node._body_pose_callback(_odometry(stamp_sec=11))
    bridge_node._cloud_callback(_cloud(stamp_sec=11, points=ramp_point))
    current_points = point_cloud2.read_points(
        cloud_capture.messages[-1],
        field_names=["x", "y", "z"],
    )
    assert current_points.shape[0] == 64
    clearing_cloud = cloud_capture.messages[-1]
    assert clearing_cloud.height == 1
    assert clearing_cloud.width == 64
    assert clearing_cloud.is_bigendian is False
    assert clearing_cloud.point_step == 13
    assert clearing_cloud.row_step == 64 * 13
    assert clearing_cloud.is_dense is True
    assert [field.name for field in clearing_cloud.fields] == [
        "x",
        "y",
        "z",
        "ray_endpoint_type",
    ]
    assert all(
        clearing_cloud.data[index * clearing_cloud.point_step + 12] == 0
        for index in range(clearing_cloud.width)
    )
    cache = bridge_node._ground_path_cache
    assert cache is not None
    assert cache.generation == bridge_node._ground_path_cache_generation
    assert cache.progress_m == pytest.approx(0.0)


@pytest.mark.parametrize(
    "points",
    [
        (),
        tuple((1.0, 0.0, 0.30) for _ in range(63)),
        tuple((math.nan, 0.0, 0.30) for _ in range(64)),
    ],
)
def test_cloud_rejects_empty_sparse_or_nonfinite_raw_input(
    bridge_node: IsaacNavigationBridge,
    points: tuple[tuple[float, float, float], ...],
) -> None:
    cloud_capture = _CapturePublisher()
    bridge_node._cloud_publisher = cloud_capture
    bridge_node._body_pose_callback(_odometry(stamp_sec=10))
    bridge_node._path_callback(_path(stamp_sec=10))

    bridge_node._cloud_callback(_cloud(stamp_sec=10, points=points))

    assert cloud_capture.messages == []


def test_cloud_requires_fresh_time_aligned_odometry_and_matching_frame(
    bridge_node: IsaacNavigationBridge,
) -> None:
    odom_capture = _CapturePublisher()
    cloud_capture = _CapturePublisher()
    bridge_node._body_pose_publisher = odom_capture
    bridge_node._cloud_publisher = cloud_capture

    bridge_node._body_pose_callback(_odometry())
    bridge_node._path_callback(_path())
    bridge_node._cloud_callback(_cloud())
    assert len(odom_capture.messages) == 1
    assert len(cloud_capture.messages) == 1

    bridge_node._latest_odom_received_monotonic = (
        time.monotonic() - bridge_node._odom_timeout_sec - 0.1
    )
    bridge_node._cloud_callback(_cloud())
    assert len(cloud_capture.messages) == 1

    bridge_node._latest_odom_received_monotonic = time.monotonic()
    bridge_node._cloud_callback(_cloud(stamp_sec=11))
    assert len(cloud_capture.messages) == 1

    bridge_node._cloud_callback(_cloud(frame_id="sensor"))
    assert len(cloud_capture.messages) == 1
