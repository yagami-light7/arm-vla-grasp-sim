"""验证标准 Go2 + MoE-CTS 确定性楼梯 Path 的几何合同。"""

from pathlib import Path

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PATH_CONFIG = PACKAGE_ROOT / "config" / "go2_moe_cts_stair_path.yaml"
WIDE_PATH_CONFIG = (
    PACKAGE_ROOT / "config" / "go2_moe_cts_stair_wide_path.yaml"
)


def _parameters(config_path: Path = PATH_CONFIG) -> dict[str, object]:
    """
    @brief 读取指定手工楼梯 Path 的 ROS 2 参数
    @param config_path 待读取的 YAML 文件
    @return manual_path_publisher 的参数字典
    """

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload["manual_path_publisher"]["ros__parameters"]


def _points_xyz(
    config_path: Path = PATH_CONFIG,
) -> list[tuple[float, float, float]]:
    """
    @brief 从 ROS 2 参数 YAML 读取展平的三维点列
    @param config_path 待读取的楼梯 Path YAML
    @return 按发布顺序排列的 world 地面高度点
    """

    parameters = _parameters(config_path)
    flat_points = parameters["points_xyz"]
    return [
        tuple(float(value) for value in flat_points[index : index + 3])
        for index in range(0, len(flat_points), 3)
    ]


def test_stair_path_matches_live_low_platform_and_top_landing() -> None:
    """
    @brief 验证路径首尾绑定当前 live 高度场且顶部留有驻留平台
    @return 无；几何漂移时由 pytest 报告
    """

    points = _points_xyz()

    assert len(points) == 33
    assert points[0] == pytest.approx((-0.05, 0.02, -2.47))
    assert points[-1] == pytest.approx((4.40, 0.02, 0.0))
    assert all(point[1] == pytest.approx(0.02) for point in points)
    assert [point[2] for point in points[:6]] == pytest.approx(
        [-2.47] * 6
    )
    assert [point[2] for point in points[-3:]] == pytest.approx([0.0] * 3)
    assert points[-1][0] - points[-3][0] >= 0.50


def test_each_measured_tread_has_a_horizontal_support_segment() -> None:
    """
    @brief 验证十二级中间踏面均以首尾点表示而不是连续斜坡
    @return 无；踏面数量、高度或实际 0.20 m 节距漂移时由 pytest 报告
    """

    points = _points_xyz()
    tread_points = points[6:30]

    assert len(tread_points) == 24
    for level in range(1, 13):
        begin = tread_points[2 * (level - 1)]
        end = tread_points[2 * (level - 1) + 1]
        expected_height = -2.47 + 0.19 * level
        expected_begin_x = 1.41 + 0.20 * (level - 1)

        assert begin[0] == pytest.approx(expected_begin_x)
        assert end[0] == pytest.approx(expected_begin_x + 0.18)
        assert begin[2] == pytest.approx(expected_height)
        assert end[2] == pytest.approx(expected_height)


def test_stair_path_explicitly_disables_root_lock_freeze_indices() -> None:
    """
    @brief 验证标准 Go2 物理爬楼支线没有配置 root-lock 冻结区间
    @return 无；出现冻结索引时由 pytest 报告
    """

    parameters = _parameters()

    assert parameters["scan_stair_freeze_stair_segment_indices"] == [-1, -1]


def test_wide_stair_path_matches_quantized_height_field_geometry() -> None:
    """
    @brief 验证 0.31 m 请求值对应真实 0.30 m 踏面和九次抬升
    @return 无；宽踏面 Path 与高度场离散几何不一致时由 pytest 报告
    """

    points = _points_xyz(WIDE_PATH_CONFIG)

    assert len(points) == 25
    assert points[0] == pytest.approx((-0.05, 0.02, -1.71))
    assert points[-1] == pytest.approx((4.40, 0.02, 0.0))
    assert all(point[1] == pytest.approx(0.02) for point in points)
    assert [point[2] for point in points[:6]] == pytest.approx(
        [-1.71] * 6
    )
    assert [point[2] for point in points[-3:]] == pytest.approx([0.0] * 3)

    tread_points = points[6:22]
    assert len(tread_points) == 16
    for level in range(1, 9):
        begin = tread_points[2 * (level - 1)]
        end = tread_points[2 * (level - 1) + 1]
        expected_height = -1.71 + 0.19 * level
        expected_begin_x = 1.31 + 0.30 * (level - 1)

        assert begin[0] == pytest.approx(expected_begin_x)
        assert end[0] == pytest.approx(expected_begin_x + 0.28)
        assert begin[2] == pytest.approx(expected_height)
        assert end[2] == pytest.approx(expected_height)

    assert points[22] == pytest.approx((3.71, 0.02, 0.0))
    assert points[-1][0] - points[22][0] >= 0.50
    assert _parameters(WIDE_PATH_CONFIG)[
        "scan_stair_freeze_stair_segment_indices"
    ] == [-1, -1]
