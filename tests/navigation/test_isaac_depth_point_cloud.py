from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from source.navigation.isaac_depth_point_cloud import (
    DepthPointCloudConfig,
    camera_sensor_to_world_points,
    depth_image_to_world_points,
)


def test_identity_camera_unprojects_ros_optical_coordinates() -> None:
    depth = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    intrinsics = np.eye(3, dtype=np.float32)

    points = depth_image_to_world_points(
        depth,
        intrinsics,
        np.zeros(3, dtype=np.float32),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        DepthPointCloudConfig(
            pixel_stride=1,
            min_depth_m=0.0,
            max_depth_m=10.0,
        ),
    )

    np.testing.assert_allclose(
        points,
        np.asarray(
            [
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 2.0],
                [0.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    assert points.dtype == np.float32
    assert points.flags.c_contiguous


def test_world_transform_uses_wxyz_rotation_and_translation() -> None:
    half_sqrt = np.sqrt(0.5)
    points = depth_image_to_world_points(
        np.asarray([[2.0]], dtype=np.float32),
        np.asarray(
            [
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray((10.0, 20.0, 30.0), dtype=np.float32),
        np.asarray((half_sqrt, 0.0, 0.0, half_sqrt), dtype=np.float32),
        DepthPointCloudConfig(
            pixel_stride=1,
            min_depth_m=0.0,
            max_depth_m=10.0,
        ),
    )

    # 相机点为 (2, 0, 2)，绕世界 Z 轴正转 90° 后为 (0, 2, 2)。
    np.testing.assert_allclose(points, [[10.0, 22.0, 32.0]], atol=1.0e-6)


def test_invalid_depth_range_stride_and_point_cap_are_deterministic() -> None:
    depth = np.asarray(
        [
            [np.nan, 1.0, 2.0, np.inf],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
        dtype=np.float32,
    )
    config = DepthPointCloudConfig(
        pixel_stride=1,
        min_depth_m=2.0,
        max_depth_m=8.0,
        max_points=3,
        minimum_valid_points=1,
    )

    first = depth_image_to_world_points(
        depth,
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.asarray((2.0, 0.0, 0.0, 0.0), dtype=np.float32),
        config,
    )
    second = depth_image_to_world_points(
        depth,
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        config,
    )

    assert first.shape == (3, 3)
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:, 2] >= 2.0)
    assert np.all(first[:, 2] <= 8.0)


def test_camera_sensor_reads_requested_environment_and_pose() -> None:
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            output={
                "distance_to_image_plane": np.asarray(
                    [
                        [[[1.0]]],
                        [[[2.0]]],
                    ],
                    dtype=np.float32,
                )
            },
            intrinsic_matrices=np.stack(
                (np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32))
            ),
            pos_w=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                ],
                dtype=np.float32,
            ),
            quat_w_ros=np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
    )

    points = camera_sensor_to_world_points(
        sensor,
        DepthPointCloudConfig(
            environment_index=1,
            pixel_stride=1,
            min_depth_m=0.0,
            max_depth_m=5.0,
        ),
    )

    np.testing.assert_allclose(points, [[1.0, 2.0, 5.0]])


def test_camera_sensor_stride_matches_original_pixels_with_off_center_intrinsics() -> None:
    depth = np.arange(1, 21, dtype=np.float32).reshape(1, 4, 5, 1)
    intrinsics = np.asarray(
        [
            [7.0, 0.0, 1.25],
            [0.0, 9.0, 0.75],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            output={"distance_to_image_plane": depth},
            intrinsic_matrices=intrinsics[None, ...],
            pos_w=np.zeros((1, 3), dtype=np.float32),
            quat_w_ros=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
    )
    config = DepthPointCloudConfig(
        pixel_stride=2,
        min_depth_m=0.0,
        max_depth_m=30.0,
    )

    optimized = camera_sensor_to_world_points(sensor, config)
    reference = depth_image_to_world_points(
        depth[0],
        intrinsics,
        np.zeros(3, dtype=np.float32),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        config,
    )

    np.testing.assert_allclose(optimized, reference, atol=1.0e-6)


def test_front_mount_ros_quaternion_rotates_optical_forward_to_robot_forward() -> None:
    points = depth_image_to_world_points(
        np.asarray([[2.0]], dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.asarray((0.28, 0.0, 0.07), dtype=np.float32),
        np.asarray((0.5, -0.5, 0.5, -0.5), dtype=np.float32),
        DepthPointCloudConfig(
            pixel_stride=1,
            min_depth_m=0.0,
            max_depth_m=3.0,
        ),
    )

    # 真实 head_camera 安装四元数把 optical +Z 对齐到机器人 +X。
    np.testing.assert_allclose(points, [[2.28, 0.0, 0.07]], atol=1.0e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sensor_name": ""},
        {"environment_index": -1},
        {"pixel_stride": 0},
        {"max_points": True},
        {"minimum_valid_points": 0},
        {"minimum_valid_points": 4, "max_points": 3},
        {"publish_interval_control_steps": 0},
        {"min_depth_m": -0.1},
        {"max_depth_m": 0.1, "min_depth_m": 0.1},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        DepthPointCloudConfig(**kwargs)


def test_conversion_rejects_bad_shapes_intrinsics_and_quaternion() -> None:
    config = DepthPointCloudConfig()

    with pytest.raises(ValueError, match="depth_image"):
        depth_image_to_world_points(
            np.zeros((1, 2, 2), dtype=np.float32),
            np.eye(3),
            np.zeros(3),
            (1.0, 0.0, 0.0, 0.0),
            config,
        )
    with pytest.raises(ValueError, match="fx"):
        depth_image_to_world_points(
            np.ones((1, 1), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.zeros(3),
            (1.0, 0.0, 0.0, 0.0),
            config,
        )
    with pytest.raises(ValueError, match="零四元数"):
        depth_image_to_world_points(
            np.ones((1, 1), dtype=np.float32),
            np.eye(3),
            np.zeros(3),
            (0.0, 0.0, 0.0, 0.0),
            config,
        )
