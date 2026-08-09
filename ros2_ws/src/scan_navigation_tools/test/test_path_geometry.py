"""手工路径纯几何与配置校验测试；本文件不导入 ROS。"""

import math

import pytest

from scan_navigation_tools.path_geometry import (
    prepare_path_points,
    validate_frame_id,
    validate_topic_name,
)


def test_flat_points_keep_ground_height_and_compute_yaw() -> None:
    points = prepare_path_points(
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.2, 2.0, 1.0, 0.4],
        min_point_distance_m=0.01,
    )

    assert [(point.x, point.y, point.z) for point in points] == [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.2),
        (2.0, 1.0, 0.4),
    ]
    assert points[0].yaw == pytest.approx(math.pi / 4.0)
    assert points[1].yaw == pytest.approx(0.0)
    assert points[2].yaw == pytest.approx(0.0)


def test_negative_y_stair_path_keeps_height_and_computes_yaw() -> None:
    """沿世界坐标-Y方向上楼时，yaw应保持为 -pi/2"""

    points = prepare_path_points(
        [
            1.5,
            -5.7,
            -0.1276,
            1.5,
            -6.1,
            -0.02,
            1.5,
            -6.5,
            0.18,
        ],
        min_point_distance_m=0.02,
    )
    # 地面高度必须原样保持，几何层不增加body_height
    assert [point.z for point in points] == pytest.approx([-0.1276, -0.02, 0.18])

    # x不变 y持续减小 因此水平前进方向是世界坐标-Y
    assert [point.yaw for point in points] == pytest.approx([-math.pi / 2.0] * 3)

def test_adjacent_near_duplicates_are_removed() -> None:
    points = prepare_path_points(
        [
            0.0,
            0.0,
            0.0,
            0.005,
            0.0,
            0.0,
            1.0,
            0.0,
            0.1,
        ],
        min_point_distance_m=0.01,
    )

    assert len(points) == 2
    assert (points[0].x, points[1].x) == (0.0, 1.0)


def test_vertical_prefix_uses_next_available_planar_direction() -> None:
    points = prepare_path_points(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0, 0.2],
        min_point_distance_m=0.01,
    )

    assert [point.yaw for point in points] == pytest.approx(
        [math.pi / 2.0, math.pi / 2.0, math.pi / 2.0]
    )


@pytest.mark.parametrize(
    "points_xyz",
    [
        [],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, math.nan, 0.0, 0.0],
        [0.0, 0.0, 0.0, math.inf, 0.0, 0.0],
        [0.0, 0.0, 0.0, "1.0", 0.0, 0.0],
    ],
)
def test_invalid_point_arrays_fail_loud(points_xyz: list[object]) -> None:
    with pytest.raises(ValueError):
        prepare_path_points(points_xyz, min_point_distance_m=0.01)


def test_deduplication_cannot_leave_a_single_point() -> None:
    with pytest.raises(ValueError, match="至少 2 个点"):
        prepare_path_points(
            [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
            min_point_distance_m=0.01,
        )


@pytest.mark.parametrize("distance", [-0.1, math.nan, math.inf, True])
def test_invalid_minimum_distance_fails_loud(distance: object) -> None:
    with pytest.raises(ValueError):
        prepare_path_points(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            min_point_distance_m=distance,
        )


def test_topic_and_frame_names_are_validated() -> None:
    assert validate_topic_name("/initial_path") == "/initial_path"
    assert validate_topic_name("planning/initial_path") == (
        "planning/initial_path"
    )
    assert validate_frame_id("world") == "world"
    assert validate_frame_id("map/world") == "map/world"

    for invalid_topic in ("", " /initial_path", "/initial path", "//path"):
        with pytest.raises(ValueError):
            validate_topic_name(invalid_topic)
    for invalid_frame in ("", "/world", "world frame", "world//base"):
        with pytest.raises(ValueError):
            validate_frame_id(invalid_frame)
