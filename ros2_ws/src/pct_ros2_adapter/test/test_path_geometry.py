import math

import numpy as np
import pytest

from pct_ros2_adapter.path_geometry import (
    normalize_frame_id,
    prepare_ground_path,
    quaternion_xyzw_to_yaw,
)


def test_frame_normalization_keeps_valid_hierarchy() -> None:
    assert normalize_frame_id(" /robot/map ", field_name="frame") == "robot/map"


@pytest.mark.parametrize("value", ("", "map//odom", "../map", "map/./odom"))
def test_frame_normalization_rejects_ambiguous_names(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_frame_id(value, field_name="frame")


def test_scaled_quaternion_is_normalized_before_yaw_conversion() -> None:
    yaw = 1.2
    quaternion = (
        0.0,
        0.0,
        4.0 * math.sin(0.5 * yaw),
        4.0 * math.cos(0.5 * yaw),
    )

    assert quaternion_xyzw_to_yaw(quaternion) == pytest.approx(yaw)


@pytest.mark.parametrize(
    "quaternion",
    ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, math.nan, 1.0), (0.0, 1.0)),
)
def test_invalid_quaternion_is_rejected(quaternion: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        quaternion_xyzw_to_yaw(quaternion)


def test_ground_path_keeps_height_and_generates_forward_yaw() -> None:
    path = prepare_ground_path(
        ((0.0, 0.0, -0.12), (1.0, 0.0, 0.20), (1.0, 1.0, 1.44)),
        terminal_yaw=-1.1,
        minimum_point_spacing_m=0.01,
    )

    assert tuple(point.z for point in path) == pytest.approx((-0.12, 0.20, 1.44))
    assert tuple(point.yaw for point in path) == pytest.approx(
        (0.0, math.pi / 2.0, -1.1)
    )


def test_near_duplicate_filter_never_drops_exact_terminal_point() -> None:
    path = prepare_ground_path(
        ((0.0, 0.0, 0.0), (0.98, 0.0, 0.0), (1.0, 0.0, 0.0)),
        terminal_yaw=0.7,
        minimum_point_spacing_m=0.05,
    )

    assert np.asarray(
        tuple((point.x, point.y, point.z) for point in path)
    ) == pytest.approx(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    )
    assert path[-1].yaw == pytest.approx(0.7)


def test_short_nonzero_path_is_rejected_before_scan_deduplicates_it() -> None:
    with pytest.raises(ValueError, match="不能发布"):
        prepare_ground_path(
            ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0)),
            terminal_yaw=0.0,
            minimum_point_spacing_m=0.05,
        )


def test_identical_path_is_rejected_after_deduplication() -> None:
    with pytest.raises(ValueError, match="\u53bb\u91cd"):
        prepare_ground_path(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            terminal_yaw=0.0,
            minimum_point_spacing_m=0.05,
        )
