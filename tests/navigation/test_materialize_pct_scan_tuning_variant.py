from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.navigation import materialize_pct_scan_tuning_variant as variant


REAL_RECIPE = (
    variant.PROJECT_ROOT
    / "configs/navigation/pct_scan_yaw_rate_075_experiment.yaml"
)
TERMINAL_YAW_RECIPE = (
    variant.PROJECT_ROOT
    / "configs/navigation/pct_scan_yaw075_terminal_yaw012_experiment.yaml"
)


def _load_yaml(path: Path) -> dict[str, object]:
    """读取测试配置并保证顶层类型明确。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_recipe(
    path: Path,
    *,
    overrides: dict[str, object],
    expected_count: int = 1,
) -> None:
    """写入指向真实生产 YAML 的最小测试配方。"""

    path.write_text(
        yaml.safe_dump(
            {
                "schema": "pct_scan_tuning_variant_v1",
                "experiment_id": "unit_test_variant",
                "base_config": str(
                    variant.PROJECT_ROOT
                    / "ros2_ws/src/isaac_navigation_bridge/config/"
                    "pct_scan_tuning.yaml"
                ),
                "expected_override_count": expected_count,
                "overrides": overrides,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_real_yaw_recipe_changes_exactly_one_existing_leaf(tmp_path: Path) -> None:
    output = tmp_path / "yaw_rate_075.yaml"

    report = variant.materialize_variant(REAL_RECIPE, output)

    assert report["experiment_id"] == "go2_x5_mainline_yaw_rate_075"
    assert report["semantic_change_count"] == 1
    assert report["semantic_changes"] == [
        {
            "path": (
                "scan_controller.ros__parameters.limits.max_yaw_rate"
            ),
            "before": 0.60,
            "after": 0.75,
        }
    ]
    assert len(report["recipe_sha256"]) == 64
    assert len(report["base_sha256"]) == 64
    assert len(report["output_sha256"]) == 64

    recipe_payload = _load_yaml(REAL_RECIPE)
    base_path = variant._resolve_base_path(
        REAL_RECIPE,
        recipe_payload["base_config"],
    )
    expected = copy.deepcopy(_load_yaml(base_path))
    expected["scan_controller"]["ros__parameters"][
        "limits.max_yaw_rate"
    ] = 0.75
    resolved = _load_yaml(output)

    assert resolved == expected
    assert resolved["navigation_contract"]["ros__parameters"][
        "body_height_m"
    ] == pytest.approx(0.338)


def test_terminal_yaw_recipe_keeps_fast_turn_and_tightens_final_alignment(
    tmp_path: Path,
) -> None:
    output = tmp_path / "yaw075_terminal_yaw012.yaml"

    report = variant.materialize_variant(TERMINAL_YAW_RECIPE, output)

    assert report["experiment_id"] == "go2_x5_yaw075_terminal_yaw012"
    assert report["semantic_change_count"] == 2
    assert report["semantic_changes"] == [
        {
            "path": "scan_controller.ros__parameters.limits.max_yaw_rate",
            "before": 0.60,
            "after": 0.75,
        },
        {
            "path": (
                "scan_controller.ros__parameters."
                "finish.yaw_control_deadband"
            ),
            "before": 0.18,
            "after": 0.12,
        },
    ]

    resolved = _load_yaml(output)
    parameters = resolved["scan_controller"]["ros__parameters"]
    assert parameters["limits.max_yaw_rate"] == pytest.approx(0.75)
    assert parameters["finish.yaw_control_deadband"] == pytest.approx(0.12)
    assert parameters["finish.max_yaw_error"] == pytest.approx(0.20)
    assert parameters["finish.capture_stable_dwell_sec"] == pytest.approx(0.50)


def test_materializer_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "already_exists.yaml"
    output.write_text("user-owned\n", encoding="utf-8")

    with pytest.raises(variant.VariantError, match="原本不存在"):
        variant.materialize_variant(REAL_RECIPE, output)

    assert output.read_text(encoding="utf-8") == "user-owned\n"


def test_materializer_rejects_unknown_parameter_key(tmp_path: Path) -> None:
    recipe = tmp_path / "unknown_key.yaml"
    _write_recipe(
        recipe,
        overrides={
            "scan_controller": {
                "ros__parameters": {"limits.max_yaw_rtae": 0.75}
            }
        },
    )

    with pytest.raises(variant.VariantError, match="不存在的生产键"):
        variant.materialize_variant(recipe, tmp_path / "unused.yaml")


def test_materializer_enforces_declared_override_count(tmp_path: Path) -> None:
    recipe = tmp_path / "wrong_count.yaml"
    _write_recipe(
        recipe,
        overrides={
            "scan_controller": {
                "ros__parameters": {"limits.max_yaw_rate": 0.75}
            }
        },
        expected_count=2,
    )

    with pytest.raises(variant.VariantError, match="覆盖数"):
        variant.materialize_variant(recipe, tmp_path / "unused.yaml")
