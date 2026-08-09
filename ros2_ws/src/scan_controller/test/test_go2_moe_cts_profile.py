"""验证标准 Go2 + MoE-CTS controller 覆盖层的安全边界。"""

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PACKAGE_ROOT / "config" / "controller.yaml"
PROFILE_CONFIG = PACKAGE_ROOT / "config" / "go2_moe_cts_flat.yaml"
LAUNCH_FILE = (
    PACKAGE_ROOT / "launch" / "go2_moe_cts_flat_controller.launch.py"
)


def _parameters(config_path: Path) -> dict[str, Any]:
    """
    @brief 读取 scan_controller ROS 2 参数字典
    @param config_path 待检查的 YAML 文件路径
    @return scan_controller.ros__parameters 中的参数字典
    """

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload["scan_controller"]["ros__parameters"]


def test_profile_is_a_narrow_standard_go2_overlay() -> None:
    """
    @brief 验证覆盖层只包含标准 Go2 验收所需的显式参数
    @return 无；断言失败时由 pytest 报告不安全字段
    """

    profile = _parameters(PROFILE_CONFIG)

    assert set(profile) == {
        "reference_path.body_height_m",
        "controller.publish_rate_hz",
        "limits.max_vx",
        "limits.max_vy",
        "limits.max_yaw_rate",
        "limits.max_ax",
        "limits.max_ay",
        "limits.max_yaw_acc",
        "finish.min_approach_speed",
    }


def test_profile_stays_inside_moe_cts_command_gate() -> None:
    """
    @brief 验证 controller 包络严格位于 MoE-CTS Isaac 安全门以内
    @return 无；任一速度或变化率越界时由 pytest 报告
    """

    profile = _parameters(PROFILE_CONFIG)

    assert profile["reference_path.body_height_m"] == 0.342
    assert profile["controller.publish_rate_hz"] == 50.0
    assert 0.0 < profile["limits.max_vx"] <= 0.80
    assert 0.0 < profile["limits.max_vy"] <= 0.50
    assert 0.0 < profile["limits.max_yaw_rate"] <= 0.80
    assert 0.0 < profile["limits.max_ax"] <= 2.0
    assert 0.0 < profile["limits.max_ay"] <= 1.5
    assert 0.0 < profile["limits.max_yaw_acc"] <= 2.5
    assert (
        0.0
        < profile["finish.min_approach_speed"]
        <= profile["limits.max_vx"]
    )


def test_profile_keeps_generic_controller_safety_contract() -> None:
    """
    @brief 验证覆盖层没有替换通用 controller 的关键安全参数
    @return 无；通用安全合同漂移时由 pytest 报告
    """

    base = _parameters(BASE_CONFIG)
    profile = _parameters(PROFILE_CONFIG)

    assert "timeouts.odom_sec" not in profile
    assert "timeouts.cloud_sec" not in profile
    assert "timeouts.bspline_sec" not in profile
    assert base["timeouts.odom_sec"] == 0.30
    assert base["timeouts.cloud_sec"] == 0.50
    assert base["timeouts.bspline_sec"] == 1.50
    assert base["qos.controller_status_depth"] == 64


def test_launch_composes_base_then_profile() -> None:
    """
    @brief 验证 launch 按通用配置、标准 Go2 覆盖层的顺序加载
    @return 无；配置顺序或唯一节点入口改变时由 pytest 报告
    """

    source = LAUNCH_FILE.read_text(encoding="utf-8")
    assert "parameters=[\n                    base_config_file,\n                    profile_config_file," in source
    assert source.count('package="scan_controller"') == 1
    assert source.count('executable="scan_controller_node"') == 1
