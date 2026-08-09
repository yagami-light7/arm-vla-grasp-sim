"""点云几何过滤测试。"""

import math
import time

import numpy as np
import pytest

from isaac_navigation_bridge.geometry import (
    OrderedGroundPath,
    PointCloudFilterConfig,
    filter_points_xyz,
    path_ground_band_mask,
)


def _body_to_world(
    points_body: np.ndarray,
    *,
    base_xyz: tuple[float, float, float],
    yaw: float,
) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    result = np.empty_like(points_body, dtype=np.float64)
    result[:, 0] = (
        base_xyz[0] + cosine * points_body[:, 0] - sine * points_body[:, 1]
    )
    result[:, 1] = (
        base_xyz[1] + sine * points_body[:, 0] + cosine * points_body[:, 1]
    )
    result[:, 2] = base_xyz[2] + points_body[:, 2]
    return result


def _pitched_body_to_world(
    points_body: np.ndarray,
    *,
    base_xyz: tuple[float, float, float],
    pitch: float,
) -> np.ndarray:
    """把机体系测试点按绕 Y 轴姿态转换到世界系。"""

    cosine = math.cos(pitch)
    sine = math.sin(pitch)
    result = np.empty_like(points_body, dtype=np.float64)
    result[:, 0] = (
        base_xyz[0]
        + cosine * points_body[:, 0]
        + sine * points_body[:, 2]
    )
    result[:, 1] = base_xyz[1] + points_body[:, 1]
    result[:, 2] = (
        base_xyz[2]
        - sine * points_body[:, 0]
        + cosine * points_body[:, 2]
    )
    return result


def test_defaults_match_go2_x5_navigation_envelope() -> None:
    config = PointCloudFilterConfig()

    assert config.body_height_m == pytest.approx(0.30)
    assert config.ground_band_down_m == pytest.approx(0.03)
    assert config.filter_path_ground is True
    assert config.path_ground_corridor_radius_m == pytest.approx(0.70)
    assert config.path_ground_clearance_m == pytest.approx(0.05)
    assert config.path_ground_stair_minimum_slope == pytest.approx(0.20)
    assert config.path_ground_stair_clearance_m == pytest.approx(0.09)
    # 地面廊道需覆盖携臂双圆柱、SCAN 碰撞代价距离和最小噪声余量。
    assert (
        config.path_ground_corridor_radius_m
        >= config.double_cylinder_radius_m + 0.20 + 0.10
    )
    assert config.path_min_point_spacing_m == pytest.approx(0.05)
    assert config.path_ground_backward_arc_m == pytest.approx(1.0)
    assert config.path_ground_forward_arc_m == pytest.approx(3.0)
    assert config.double_cylinder_radius_m == pytest.approx(0.27)
    assert config.double_cylinder_offset_m == pytest.approx(0.16)
    assert config.self_z_min_m == pytest.approx(-0.40)
    assert config.self_z_max_m == pytest.approx(0.50)


def test_filter_removes_ground_self_range_crop_and_nonfinite_points() -> None:
    config = PointCloudFilterConfig(
        range_max_m=8.0,
        crop_min_xyz_m=(-5.0, -5.0, -0.60),
        crop_max_xyz_m=(5.0, 5.0, 2.0),
        body_height_m=0.45,
        ground_clearance_m=0.03,
        double_cylinder_radius_m=0.36,
        double_cylinder_offset_m=0.28,
        self_z_min_m=-0.48,
        self_z_max_m=0.75,
    )
    base = (10.0, 20.0, 1.0)
    yaw = math.pi / 2.0
    body_points = np.asarray(
        [
            [2.0, 0.0, 0.10],  # 保留：前方障碍
            [0.28, 0.0, 0.10],  # 删除：前圆柱自点
            [-0.28, 0.10, 0.20],  # 删除：后圆柱自点
            [0.80, 0.80, -0.44],  # 删除：地面 clearance 内
            [0.80, 0.80, -0.52],  # 保留：更低一级台阶
            [0.80, 0.80, -0.35],  # 保留：高于地面窄带
            [0.0, 0.0, 0.90],  # 保留：高于自点竖直包络
            [5.50, 0.0, 0.10],  # 删除：x 裁剪范围外
            [9.00, 0.0, 0.10],  # 删除：距离范围外
            [math.nan, 0.0, 0.0],  # 删除：非有限
        ],
        dtype=np.float64,
    )
    world_points = _body_to_world(body_points, base_xyz=base, yaw=yaw)

    filtered = filter_points_xyz(
        world_points,
        base_position_world_xyz=base,
        base_yaw_rad=yaw,
        config=config,
    )

    expected = world_points[[0, 4, 5, 6]].astype(np.float32)
    np.testing.assert_allclose(filtered, expected)
    assert filtered.dtype == np.float32


def test_double_cylinder_follows_base_yaw() -> None:
    config = PointCloudFilterConfig(
        filter_ground=False,
        crop_min_xyz_m=(-2.0, -2.0, -2.0),
        crop_max_xyz_m=(2.0, 2.0, 2.0),
        double_cylinder_radius_m=0.20,
        double_cylinder_offset_m=0.50,
        self_z_min_m=-0.5,
        self_z_max_m=0.5,
    )
    base = (3.0, 4.0, 1.0)
    yaw = math.pi / 2.0
    points = np.asarray(
        [
            [3.0, 4.50, 1.0],  # yaw 后的前圆柱中心
            [3.0, 3.50, 1.0],  # yaw 后的后圆柱中心
            [3.40, 4.00, 1.0],  # 圆柱外
        ]
    )

    filtered = filter_points_xyz(
        points,
        base_position_world_xyz=base,
        base_yaw_rad=yaw,
        config=config,
    )

    np.testing.assert_allclose(filtered, points[[2]].astype(np.float32))


def test_full_base_orientation_removes_pitched_support_plane() -> None:
    config = PointCloudFilterConfig(
        filter_self=False,
        crop_min_xyz_m=(-2.0, -2.0, -1.0),
        crop_max_xyz_m=(2.0, 2.0, 1.0),
        body_height_m=0.30,
        ground_clearance_m=0.03,
        ground_band_down_m=0.03,
    )
    base = (3.0, 4.0, 1.0)
    pitch = -math.atan(0.15)
    body_points = np.asarray(
        [
            [0.0, 0.50, -0.30],
            [0.6, 0.50, -0.30],
            [1.2, 0.50, -0.30],
            [0.6, 0.50, -0.15],
        ],
        dtype=np.float64,
    )
    world_points = _pitched_body_to_world(
        body_points,
        base_xyz=base,
        pitch=pitch,
    )
    orientation = (
        0.0,
        2.0 * math.sin(0.5 * pitch),
        0.0,
        2.0 * math.cos(0.5 * pitch),
    )

    full_orientation_filtered = filter_points_xyz(
        world_points,
        base_position_world_xyz=base,
        base_yaw_rad=None,
        base_orientation_world_xyzw=orientation,
        config=config,
    )
    yaw_only_filtered = filter_points_xyz(
        world_points,
        base_position_world_xyz=base,
        base_yaw_rad=0.0,
        config=config,
    )

    np.testing.assert_allclose(
        full_orientation_filtered,
        world_points[[3]].astype(np.float32),
    )
    assert any(
        np.allclose(point, world_points[2])
        for point in yaw_only_filtered
    )


def test_full_base_orientation_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="零四元数"):
        filter_points_xyz(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
            base_position_world_xyz=(0.0, 0.0, 0.0),
            base_yaw_rad=None,
            base_orientation_world_xyzw=(0.0, 0.0, 0.0, 0.0),
            config=PointCloudFilterConfig(),
        )


def test_path_ground_corridor_removes_upcoming_ramp_support() -> None:
    config = PointCloudFilterConfig(
        filter_self=False,
        crop_min_xyz_m=(-2.0, -2.0, -1.0),
        crop_max_xyz_m=(3.0, 2.0, 1.0),
        body_height_m=0.30,
        ground_clearance_m=0.03,
        ground_band_down_m=0.03,
        path_ground_corridor_radius_m=0.38,
        path_ground_clearance_m=0.05,
        path_ground_band_down_m=0.05,
    )
    base = (-0.5, 0.0, 0.30)
    ground_path = OrderedGroundPath(
        [
            [-0.5, 0.0, 0.00],
            [0.0, 0.0, 0.00],
            [0.6, 0.0, 0.09],
            [1.2, 0.0, 0.18],
            [1.7, 0.0, 0.18],
        ]
    )
    progress = ground_path.project_progress(
        (-0.5, 0.0, 0.0),
        previous_progress_m=None,
        backward_arc_m=1.0,
        forward_arc_m=3.0,
    )
    local_segments = ground_path.local_segments(
        progress,
        backward_arc_m=1.0,
        forward_arc_m=3.0,
    )
    world_points = np.asarray(
        [
            [0.0, 0.10, 0.00],
            [0.6, 0.10, 0.09],
            [1.2, 0.10, 0.18],
            [0.6, 0.10, 0.20],
            [0.6, 0.60, 0.09],
        ],
        dtype=np.float64,
    )

    filtered = filter_points_xyz(
        world_points,
        base_position_world_xyz=base,
        base_yaw_rad=0.0,
        local_ground_path_segments=local_segments,
        config=config,
    )

    np.testing.assert_allclose(
        filtered,
        world_points[[3, 4]].astype(np.float32),
    )


def test_stair_support_tail_removes_only_support_not_obstacles() -> None:
    """楼梯终点后的支撑 Path 不能顺带删除低障碍或头顶障碍。"""

    two_step_points = np.asarray(
        [
            [1.5, 5.7, -0.12757488791649585],
            [1.5060733333333334, 5.892276666666667, -0.12973559188322342],
            [1.5121466666666665, 6.084553333333334, -0.13441230906553495],
            [1.51822, 6.27683, -0.0067505603638905705],
            [1.541865882352941, 6.468079411764706, 0.06047143123673176],
            [1.5655117647058823, 6.659328823529412, 0.17794769798269805],
        ],
        dtype=np.float64,
    )
    support_points = np.vstack(
        (
            two_step_points,
            [1.5891576470588233, 6.850578235294118, 0.324499825739186],
            [1.6128035294117646, 7.041827647058824, 0.34594065578946614],
            [1.6364494117647057, 7.23307705882353, 0.4864545798344777],
        )
    )
    query_ground = two_step_points[-1]
    candidate_points = np.asarray(
        [
            [1.58514451, 6.81811974, 0.286251],
            [1.58514451, 6.81811974, 0.40],
            [1.58514451, 6.81811974, 0.927947697982698],
        ],
        dtype=np.float64,
    )

    def mask_for(path_points: np.ndarray) -> np.ndarray:
        path = OrderedGroundPath(path_points)
        progress = path.project_progress(
            query_ground,
            previous_progress_m=None,
            backward_arc_m=1.0,
            forward_arc_m=3.0,
        )
        return path_ground_band_mask(
            candidate_points,
            path.local_segments(
                progress,
                backward_arc_m=1.0,
                forward_arc_m=3.0,
            ),
            corridor_radius_m=0.70,
            clearance_m=0.05,
            band_down_m=0.05,
        )

    assert mask_for(two_step_points).tolist() == [False, False, False]
    assert mask_for(support_points).tolist() == [True, False, False]


def test_path_ground_stair_clearance_is_gated_by_segment_slope() -> None:
    """楼梯间隙只能用于陡坡段，不得放宽平地和缓坡障碍门。"""

    paths_and_points = (
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.40]],
            [[0.5, 0.1, 0.28], [0.5, 0.1, 0.30]],
            [True, False],
        ),
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.15]],
            [[0.5, 0.1, 0.155]],
            [False],
        ),
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.1, 0.08]],
            [False],
        ),
    )
    for path_points, query_points, expected in paths_and_points:
        path = OrderedGroundPath(path_points)
        local_segments = path.local_segments(
            0.0,
            backward_arc_m=0.0,
            forward_arc_m=2.0,
        )
        mask = path_ground_band_mask(
            np.asarray(query_points, dtype=np.float64),
            local_segments,
            corridor_radius_m=0.70,
            clearance_m=0.05,
            band_down_m=0.05,
            stair_minimum_slope=0.20,
            stair_clearance_m=0.09,
        )
        assert mask.tolist() == expected


def test_path_ground_stair_lower_band_is_gated_by_segment_slope() -> None:
    """
    @brief 验证台阶下踏面只在陡路径段使用楼梯下方窄带
    @return 无；平地障碍被吞掉或楼梯支撑面漏检时由 pytest 报告
    """

    cases = (
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.40]],
            [[0.5, 0.1, 0.135], [0.5, 0.1, 0.105]],
            [True, False],
        ),
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.15]],
            [[0.5, 0.1, 0.010]],
            [False],
        ),
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.1, -0.065]],
            [False],
        ),
    )
    for path_points, query_points, expected in cases:
        path = OrderedGroundPath(path_points)
        mask = path_ground_band_mask(
            np.asarray(query_points, dtype=np.float64),
            path.local_segments(
                0.0,
                backward_arc_m=0.0,
                forward_arc_m=2.0,
            ),
            corridor_radius_m=0.70,
            clearance_m=0.05,
            band_down_m=0.05,
            stair_minimum_slope=0.20,
            stair_clearance_m=0.09,
            stair_band_down_m=0.09,
        )
        assert mask.tolist() == expected


def test_ordered_path_uses_xyz_progress_to_isolate_overlapping_floors() -> None:
    path = OrderedGroundPath(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],  # 三维退化段必须安全跳过
            [0.0, 0.0, 3.0],
        ]
    )

    progress = path.project_progress(
        (1.0, 0.0, 3.0),
        previous_progress_m=None,
        backward_arc_m=1.0,
        forward_arc_m=3.0,
    )
    local_segments = path.local_segments(
        progress,
        backward_arc_m=1.0,
        forward_arc_m=2.0,
    )
    mask = path_ground_band_mask(
        np.asarray(
            [
                [1.0, 0.1, 0.0],
                [1.0, 0.1, 3.0],
                [1.0, 0.1, 3.11],
            ]
        ),
        local_segments,
        corridor_radius_m=0.38,
        clearance_m=0.05,
        band_down_m=0.05,
    )

    assert progress == pytest.approx(6.0)
    assert local_segments.segment_count == 1
    assert mask.tolist() == [False, True, False]


def test_ordered_path_progress_is_monotonic_on_revisited_geometry() -> None:
    path = OrderedGroundPath(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    on_revisited_segment = path.project_progress(
        (0.5, 0.0, 0.0),
        previous_progress_m=8.5,
        backward_arc_m=1.0,
        forward_arc_m=1.0,
    )
    after_backward_motion = path.project_progress(
        (0.1, 0.0, 0.0),
        previous_progress_m=on_revisited_segment,
        backward_arc_m=1.0,
        forward_arc_m=1.0,
    )

    assert on_revisited_segment == pytest.approx(8.5)
    assert after_backward_motion == pytest.approx(8.5)


def test_local_path_filter_cost_does_not_scale_with_full_waypoint_count() -> None:
    """30k 点性能门限同时约束绝对耗时和长路径扩张。"""

    random = np.random.default_rng(20260731)
    points = np.column_stack(
        (
            random.uniform(4.0, 8.0, 30_000),
            random.uniform(-0.5, 0.5, 30_000),
            random.uniform(-0.08, 0.20, 30_000),
        )
    )

    elapsed: list[float] = []
    segment_counts: list[int] = []
    for waypoint_count in (250, 1000):
        path = OrderedGroundPath(
            np.column_stack(
                (
                    np.arange(waypoint_count, dtype=np.float64) * 0.05,
                    np.zeros(waypoint_count),
                    np.zeros(waypoint_count),
                )
            )
        )
        progress = path.project_progress(
            (5.0, 0.0, 0.0),
            previous_progress_m=None,
            backward_arc_m=1.0,
            forward_arc_m=3.0,
        )
        local_segments = path.local_segments(
            progress,
            backward_arc_m=1.0,
            forward_arc_m=3.0,
        )
        path_ground_band_mask(
            points[:128],
            local_segments,
            corridor_radius_m=0.38,
            clearance_m=0.05,
            band_down_m=0.05,
        )
        started = time.perf_counter()
        mask = path_ground_band_mask(
            points,
            local_segments,
            corridor_radius_m=0.38,
            clearance_m=0.05,
            band_down_m=0.05,
        )
        elapsed.append(time.perf_counter() - started)
        segment_counts.append(local_segments.segment_count)
        assert mask.shape == (30_000,)

    assert max(segment_counts) <= 82
    assert abs(segment_counts[1] - segment_counts[0]) <= 1
    assert max(elapsed) < 1.25
    assert abs(elapsed[1] - elapsed[0]) < 0.35


@pytest.mark.parametrize(
    "kwargs",
    [
        {"range_min_m": 2.0, "range_max_m": 1.0},
        {"crop_min_xyz_m": (0.0, 0.0, 0.0), "crop_max_xyz_m": (0.0, 1.0, 1.0)},
        {"body_height_m": 0.0},
        {"path_ground_corridor_radius_m": -0.1},
        {"path_ground_stair_minimum_slope": 0.0},
        {"path_ground_stair_clearance_m": 0.04},
        {"path_min_point_spacing_m": 0.0},
        {"path_ground_backward_arc_m": -0.1},
        {"path_ground_forward_arc_m": 0.0},
        {"double_cylinder_radius_m": -0.1},
        {"self_z_min_m": 1.0, "self_z_max_m": 0.0},
    ],
)
def test_invalid_filter_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PointCloudFilterConfig(**kwargs)
