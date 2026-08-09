"""验证标准 Go2 + MoE-CTS 单跑楼梯 launch 与调参合同。"""

from pathlib import Path
from typing import Any

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TUNING_CONFIG = (
    PACKAGE_ROOT / "config" / "go2_moe_cts_stair_tuning.yaml"
)
FAST_TUNING_CONFIG = (
    PACKAGE_ROOT / "config" / "go2_moe_cts_stair_fast_tuning.yaml"
)
WIDE_FAST_TUNING_CONFIG = (
    PACKAGE_ROOT
    / "config"
    / "go2_moe_cts_stair_wide_fast_tuning.yaml"
)
LAUNCH_FILE = (
    PACKAGE_ROOT / "launch" / "go2_moe_cts_stair_navigation.launch.py"
)
FAST_LAUNCH_FILE = (
    PACKAGE_ROOT
    / "launch"
    / "go2_moe_cts_stair_fast_navigation.launch.py"
)
WIDE_FAST_LAUNCH_FILE = (
    PACKAGE_ROOT
    / "launch"
    / "go2_moe_cts_stair_wide_fast_navigation.launch.py"
)


def _parameters(
    node_name: str,
    config_path: Path = TUNING_CONFIG,
) -> dict[str, Any]:
    """
    @brief 读取指定 ROS 2 节点在楼梯调参 YAML 中的参数
    @param node_name YAML 顶层节点名称
    @param config_path 待读取的楼梯调参文件
    @return 对应 ros__parameters 字典
    """

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload[node_name]["ros__parameters"]


def test_stair_planner_and_controller_share_one_motion_envelope() -> None:
    """
    @brief 验证规划、优化与控制速度加速度上限保持一致
    @return 无；任一边界不一致时由 pytest 报告
    """

    planner = _parameters("scan_planner_node")
    controller = _parameters("scan_controller")

    assert planner["manager.max_vel"] == 0.45
    assert planner["optimization.max_vel"] == 0.45
    assert controller["limits.max_vx"] == 0.45
    assert planner["manager.max_acc"] == 0.80
    assert planner["optimization.max_acc"] == 0.80
    assert controller["limits.max_ax"] == 0.80
    assert 0.0 < planner["fsm.reference_cruise_speed"] < 0.45


def test_stair_point_cloud_uses_world_height_scanner_contract() -> None:
    """
    @brief 验证楼梯点云不重复应用相机外参且绑定 live 低平台高度
    @return 无；传感器合同漂移时由 pytest 报告
    """

    planner = _parameters("scan_planner_node")

    assert planner["grid_map.need_extrinsic"] is False
    assert planner["grid_map.ground_height"] == -2.47
    assert planner["fsm.planning_horizon"] <= 0.80


def test_fast_stair_profile_raises_only_safe_motion_envelope() -> None:
    """
    @brief 验证快速配置提高纵向速度且仍位于 MoE-CTS 安全门内
    @return 无；速度越界或非目标参数漂移时由 pytest 报告
    """

    baseline_planner = _parameters("scan_planner_node")
    baseline_controller = _parameters("scan_controller")
    planner = _parameters("scan_planner_node", FAST_TUNING_CONFIG)
    controller = _parameters("scan_controller", FAST_TUNING_CONFIG)

    assert planner["fsm.reference_cruise_speed"] == 0.60
    assert planner["manager.max_vel"] == 0.65
    assert planner["optimization.max_vel"] == 0.65
    assert controller["limits.max_vx"] == 0.65
    assert planner["manager.max_acc"] == 1.20
    assert planner["optimization.max_acc"] == 1.20
    assert controller["limits.max_ax"] == 1.20
    assert planner["manager.reference_profile_acceleration_scale"] == 0.75
    assert planner["manager.feasibility_tolerance"] == 0.0075
    assert (
        planner["manager.max_acc"]
        * planner["manager.reference_profile_acceleration_scale"]
        == pytest.approx(0.90)
    )
    assert planner["fsm.reference_velocity_filter_time_constant_sec"] == 0.12
    assert controller["finish.min_approach_speed"] == 0.28

    assert planner["fsm.reference_cruise_speed"] > (
        baseline_planner["fsm.reference_cruise_speed"]
    )
    assert controller["limits.max_vx"] > baseline_controller["limits.max_vx"]
    assert 0.0 < controller["limits.max_vx"] < 0.80
    assert 0.0 < controller["limits.max_ax"] < 2.0
    assert controller["limits.max_vy"] == baseline_controller["limits.max_vy"]
    assert controller["limits.max_yaw_rate"] == (
        baseline_controller["limits.max_yaw_rate"]
    )
    assert planner["grid_map.need_extrinsic"] is False
    assert planner["grid_map.ground_height"] == -2.47


def test_stair_launch_keeps_pct_and_supervisor_out_of_first_flight() -> None:
    """
    @brief 验证首轮只启动手工 Path、SCAN 与 controller
    @return 无；阶段边界被意外扩大时由 pytest 报告
    """

    source = LAUNCH_FILE.read_text(encoding="utf-8")

    assert '"start_scan": "true"' in source
    assert '"start_controller": "true"' in source
    assert '"start_manual_path": "true"' in source
    assert '"start_pct": "false"' in source
    assert '"start_supervisor": "false"' in source
    assert '"body_height_m": "0.342"' in source
    assert "go2_moe_cts_stair_path.yaml" in source
    assert "go2_moe_cts_stair_tuning.yaml" in source
    assert 'LaunchConfiguration("stair_path_config_file")' in source
    assert 'LaunchConfiguration("stair_tuning_config_file")' in source


def test_fast_stair_launch_only_replaces_tuning_file() -> None:
    """
    @brief 验证快速入口复用同一楼梯链且只选择快速覆盖层
    @return 无；入口绕开基线组合关系时由 pytest 报告
    """

    source = FAST_LAUNCH_FILE.read_text(encoding="utf-8")

    assert "go2_moe_cts_stair_navigation.launch.py" in source
    assert "go2_moe_cts_stair_fast_tuning.yaml" in source
    assert '"stair_tuning_config_file": fast_tuning_config' in source
    assert "scan_controller_node" not in source
    assert "manual_path_publisher" not in source


def test_wide_fast_profile_uses_user_high_speed_envelope() -> None:
    """
    @brief 验证宽踏面配置使用用户选定的高速包络且满足采样门
    @return 无；用户参数漂移或容差过宽时由 pytest 报告
    """

    wide_planner = _parameters(
        "scan_planner_node",
        WIDE_FAST_TUNING_CONFIG,
    )
    wide_controller = _parameters(
        "scan_controller",
        WIDE_FAST_TUNING_CONFIG,
    )

    assert wide_planner["grid_map.ground_height"] == -1.71
    assert wide_planner["fsm.reference_cruise_speed"] == 1.00
    assert wide_planner["manager.max_vel"] == 1.00
    assert wide_planner["manager.max_acc"] == 1.50
    assert wide_planner["optimization.max_vel"] == 1.00
    assert wide_planner["optimization.max_acc"] == 1.50
    assert wide_controller["limits.max_vx"] == 1.50
    assert wide_controller["limits.max_vy"] == 0.5
    assert wide_controller["limits.max_yaw_rate"] == 0.8
    assert wide_controller["limits.max_ax"] == 2.50
    assert wide_controller["limits.max_ay"] == 0.80
    assert wide_controller["limits.max_yaw_acc"] == 1.50
    assert wide_controller["controller.stair_heading_lock_enabled"] is True
    assert (
        wide_controller["controller.stair_heading_lock_half_window_arc_m"]
        == 0.45
    )
    assert (
        wide_controller["controller.stair_heading_lock_min_pitch_rad"]
        == 0.45
    )
    assert wide_controller["controller.stair_forward_speed_floor"] == 1.00

    tolerance = wide_planner["manager.feasibility_tolerance"]
    assert wide_planner["manager.max_vel"] * tolerance + 1.0e-4 <= 0.005
    assert wide_planner["manager.max_acc"] * tolerance + 1.0e-4 <= 0.01


def test_wide_fast_launch_selects_matching_path_and_tuning() -> None:
    """
    @brief 验证宽踏面入口同时替换 Path 与地面高度调参文件
    @return 无；物理地形、Path 与 SCAN 地面合同可能错位时由 pytest 报告
    """

    source = WIDE_FAST_LAUNCH_FILE.read_text(encoding="utf-8")

    assert "go2_moe_cts_stair_navigation.launch.py" in source
    assert "go2_moe_cts_stair_wide_path.yaml" in source
    assert "go2_moe_cts_stair_wide_fast_tuning.yaml" in source
    assert '"stair_path_config_file": wide_path_config' in source
    assert '"stair_tuning_config_file": wide_fast_tuning_config' in source
    assert "scan_controller_node" not in source
    assert "manual_path_publisher" not in source
