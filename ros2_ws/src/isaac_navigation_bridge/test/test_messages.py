"""ROS 消息规范化测试。"""

import math

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
import numpy as np
import pytest
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from isaac_navigation_bridge.geometry import PointCloudFilterConfig
from isaac_navigation_bridge.messages import (
    base_pose_from_odometry,
    base_transform_from_odometry,
    convert_point_cloud,
    normalize_odometry,
    normalized_stamp,
    stamp_to_nanoseconds,
)


def _stamp(sec: int, nanosec: int = 0) -> Time:
    return Time(sec=sec, nanosec=nanosec)


def test_odometry_normalizes_stamp_frames_quaternion_and_nonfinite_velocity() -> None:
    message = Odometry()
    message.pose.pose.position.x = 1.0
    message.pose.pose.position.y = -2.0
    message.pose.pose.position.z = 0.45
    message.header.frame_id = "/world"
    message.child_frame_id = "/base_link"
    message.pose.pose.orientation.x = 0.0
    message.pose.pose.orientation.y = 0.0
    message.pose.pose.orientation.z = 0.0
    message.pose.pose.orientation.w = 2.0
    message.twist.twist.linear.x = math.nan
    message.twist.twist.angular.z = math.inf
    message.pose.covariance[0] = math.nan

    normalized = normalize_odometry(
        message,
        fallback_stamp=_stamp(12, 34),
        frame_id="/world",
        child_frame_id="/base_link",
    )

    assert normalized.header.stamp == _stamp(12, 34)
    assert normalized.header.frame_id == "world"
    assert normalized.child_frame_id == "base_link"
    assert normalized.pose.pose.orientation.w == pytest.approx(1.0)
    assert normalized.twist.twist.linear.x == 0.0
    assert normalized.twist.twist.angular.z == 0.0
    assert normalized.pose.covariance[0] == 0.0
    assert message.header.frame_id == "/world"


def test_odometry_rejects_nonfinite_position() -> None:
    message = Odometry()
    message.header.frame_id = "world"
    message.child_frame_id = "base_link"
    message.pose.pose.orientation.x = 0.0
    message.pose.pose.orientation.y = 0.0
    message.pose.pose.orientation.z = 0.0
    message.pose.pose.orientation.w = 0.0
    message.pose.pose.position.x = math.nan
    with pytest.raises(ValueError, match="位置"):
        normalize_odometry(
            message,
            fallback_stamp=_stamp(1),
            frame_id="world",
            child_frame_id="base_link",
        )


def test_odometry_rejects_invalid_quaternion_and_frame_mismatch() -> None:
    message = Odometry()
    message.header.frame_id = "world"
    message.child_frame_id = "base_link"
    message.pose.pose.orientation.x = 0.0
    message.pose.pose.orientation.y = 0.0
    message.pose.pose.orientation.z = 0.0
    message.pose.pose.orientation.w = 0.0

    with pytest.raises(ValueError, match="四元数"):
        normalize_odometry(
            message,
            fallback_stamp=_stamp(1),
            frame_id="world",
            child_frame_id="base_link",
        )

    message.pose.pose.orientation.w = 1.0
    message.header.frame_id = "sensor"
    with pytest.raises(ValueError, match="不匹配"):
        normalize_odometry(
            message,
            fallback_stamp=_stamp(1),
            frame_id="world",
            child_frame_id="base_link",
        )


def test_base_pose_extracts_yaw() -> None:
    message = Odometry()
    message.pose.pose.position.x = 1.0
    message.pose.pose.position.y = 2.0
    message.pose.pose.position.z = 3.0
    message.pose.pose.orientation.z = math.sin(math.pi / 4.0)
    message.pose.pose.orientation.w = math.cos(math.pi / 4.0)

    position, yaw = base_pose_from_odometry(message)

    assert position == (1.0, 2.0, 3.0)
    assert yaw == pytest.approx(math.pi / 2.0)


def test_base_transform_preserves_normalized_roll_pitch_yaw() -> None:
    message = Odometry()
    message.pose.pose.position.x = 1.0
    message.pose.pose.position.y = 2.0
    message.pose.pose.position.z = 3.0
    message.pose.pose.orientation.x = 0.2
    message.pose.pose.orientation.y = -0.3
    message.pose.pose.orientation.z = 0.4
    message.pose.pose.orientation.w = 0.5

    position, orientation = base_transform_from_odometry(message)

    assert position == (1.0, 2.0, 3.0)
    assert math.sqrt(sum(value * value for value in orientation)) == pytest.approx(
        1.0
    )
    assert orientation == pytest.approx(
        tuple(value / math.sqrt(0.54) for value in (0.2, -0.3, 0.4, 0.5))
    )


def test_cloud_conversion_preserves_measured_support_as_free_ray_endpoints() -> None:
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name="intensity",
            offset=12,
            datatype=PointField.FLOAT32,
            count=1,
        ),
    ]
    header = Header(stamp=Time(), frame_id="/world")
    message = point_cloud2.create_cloud(
        header,
        fields,
        [
            (1.0, 0.0, 0.10, 10.0),
            (0.28, 0.0, 0.10, 20.0),
            (2.0, 0.0, 0.10, 30.0),
            (0.8, 0.8, 0.00, 40.0),
            (math.nan, 0.0, 0.10, 50.0),
        ],
    )
    config = PointCloudFilterConfig(
        body_height_m=0.45,
        ground_clearance_m=0.03,
        double_cylinder_radius_m=0.36,
        double_cylinder_offset_m=0.28,
    )

    output = convert_point_cloud(
        message,
        fallback_stamp=_stamp(20, 5),
        frame_id="/world",
        base_position_world_xyz=(0.0, 0.0, 0.45),
        base_yaw_rad=0.0,
        filter_config=config,
    )

    assert output.header.stamp == _stamp(20, 5)
    assert output.header.frame_id == "world"
    assert [field.name for field in output.fields] == [
        "x",
        "y",
        "z",
        "ray_endpoint_type",
    ]
    assert output.point_step == 13
    assert output.is_dense is True
    structured_points = point_cloud2.read_points(
        output,
        field_names=["x", "y", "z"],
    )
    points = np.column_stack(
        (
            structured_points["x"],
            structured_points["y"],
            structured_points["z"],
        )
    )
    np.testing.assert_allclose(
        points,
        np.asarray(
            [
                [1.0, 0.0, 0.10],
                [2.0, 0.0, 0.10],
                [0.8, 0.8, 0.00],
            ],
            dtype=np.float32,
        ),
    )
    assert [
        output.data[index * output.point_step + 12]
        for index in range(output.width)
    ] == [1, 1, 0]


def test_cloud_conversion_emits_exact_canonical_empty_only_after_valid_filtering() -> None:
    header = Header(stamp=_stamp(30), frame_id="world")
    raw = point_cloud2.create_cloud_xyz32(
        header,
        tuple((0.0, 0.0, 0.30) for _ in range(64)),
    )
    output = convert_point_cloud(
        raw,
        fallback_stamp=_stamp(30),
        frame_id="world",
        base_position_world_xyz=(0.0, 0.0, 0.30),
        base_yaw_rad=0.0,
        filter_config=PointCloudFilterConfig(),
        minimum_valid_input_points=64,
    )

    assert output.height == 1
    assert output.width == 0
    assert output.is_bigendian is False
    assert output.point_step == 12
    assert output.row_step == 0
    assert len(output.data) == 0
    assert output.is_dense is True
    assert [field.name for field in output.fields] == ["x", "y", "z"]
    assert [field.offset for field in output.fields] == [0, 4, 8]
    assert all(field.datatype == PointField.FLOAT32 for field in output.fields)
    assert all(field.count == 1 for field in output.fields)


@pytest.mark.parametrize(
    "points",
    [
        (),
        tuple((1.0, 0.0, 0.30) for _ in range(63)),
        tuple((math.nan, 0.0, 0.30) for _ in range(64)),
    ],
)
def test_cloud_conversion_rejects_invalid_raw_input_before_filtering(
    points: tuple[tuple[float, float, float], ...],
) -> None:
    raw = point_cloud2.create_cloud_xyz32(
        Header(stamp=_stamp(30), frame_id="world"),
        points,
    )

    with pytest.raises(ValueError, match="有限 xyz 点数不足"):
        convert_point_cloud(
            raw,
            fallback_stamp=_stamp(30),
            frame_id="world",
            base_position_world_xyz=(0.0, 0.0, 0.30),
            base_yaw_rad=0.0,
            filter_config=PointCloudFilterConfig(),
            minimum_valid_input_points=64,
        )


def test_stamp_requires_input_or_clock_to_be_nonzero() -> None:
    with pytest.raises(ValueError, match="均为零"):
        normalized_stamp(Time(), Time())


def test_stamp_to_nanoseconds_requires_valid_nonzero_stamp() -> None:
    assert stamp_to_nanoseconds(_stamp(2, 3)) == 2_000_000_003
    with pytest.raises(ValueError, match="非零"):
        stamp_to_nanoseconds(Time())
