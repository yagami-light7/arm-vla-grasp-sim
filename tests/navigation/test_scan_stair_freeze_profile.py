"""SCAN 楼梯冻结生产 profile 的 schema、数值与场景绑定测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from source.navigation.scan_stair_freeze import (
    SCAN_STAIR_FREEZE_DOG_JOINT_NAMES,
    SCAN_STAIR_FREEZE_DOG_STAND_JOINT_POSITIONS,
)
from source.navigation.scan_stair_freeze_profile import (
    ScanStairFreezeProfileError,
    bind_pipeline_navigation_settings,
    compute_scan_stair_freeze_contract_sha256,
    load_scan_stair_freeze_profile,
    load_scene_scan_stair_freeze_profile,
)
from source.scene.profiles import load_scene_profile
from source.pipeline.config import NavigationSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "configs/navigation/scan_stair_freeze_go2_x5_multifloor_v1.json"
)
PCT_TOPOLOGY_PROFILE_PATH = (
    PROJECT_ROOT / "configs/navigation/pct_multifloor_stair_profile.json"
)


def _production_payload() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _write_profile(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    refresh_digest: bool = True,
) -> Path:
    output = tmp_path / "freeze_profile.json"
    if refresh_digest:
        payload["contract_sha256"] = (
            compute_scan_stair_freeze_contract_sha256(payload)
        )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def test_multi_floor_binds_one_validated_go2_x5_freeze_profile() -> None:
    scene = load_scene_profile("multi_floor", PROJECT_ROOT)
    profile = load_scene_scan_stair_freeze_profile(scene, PROJECT_ROOT)
    topology = json.loads(PCT_TOPOLOGY_PROFILE_PATH.read_text(encoding="utf-8"))

    assert scene.scan_stair_freeze_profile == (
        "configs/navigation/scan_stair_freeze_go2_x5_multifloor_v1.json"
    )
    assert scene.scan_stair_freeze_profile in scene.required_assets
    assert profile.profile_id == "go2_x5_multifloor_scan_stair_freeze_v1"
    assert profile.scene == "multi_floor"
    assert profile.robot == "go2_x5"
    assert profile.controller == "scan_stair_freeze"
    assert profile.source_branch == "pct-scene"
    assert profile.baseline_behavior == "chassis_root_lock"
    assert profile.pct_topology_profile_reused is False
    assert profile.execution_contract.path_height_semantics == "ground"
    assert profile.execution_contract.root_target_height_semantics == (
        "ground_plus_body_height_once"
    )
    assert profile.execution_contract.cmd_vel_during_freeze == (0.0, 0.0, 0.0)
    assert profile.execution_contract.non_physical_root_lock_workaround is True
    assert profile.fixed_navigation_posture.arm_posture == "stow"
    assert profile.fixed_navigation_posture.arm_fixed_during_navigation is True
    assert profile.fixed_navigation_posture.dog_joint_names == tuple(
        SCAN_STAIR_FREEZE_DOG_JOINT_NAMES
    )
    assert profile.fixed_navigation_posture.dog_stand_joint_positions_rad == tuple(
        SCAN_STAIR_FREEZE_DOG_STAND_JOINT_POSITIONS
    )

    assert profile.config.speed_mps == pytest.approx(0.18)
    assert profile.config.approach_distance_m == pytest.approx(1.50)
    # phase225 实测中 SCAN 在距扩展楼梯组件 0.226 m 时已因机身碰撞停车；
    # 生产接管半径必须覆盖该位置，并保留一定的跟踪误差余量。
    assert profile.config.activation_radius_m == pytest.approx(0.35)
    assert profile.config.exit_distance_m == pytest.approx(0.40)
    assert profile.config.full_lock_settle_time_s == pytest.approx(1.20)
    assert profile.config.root_release_settle_time_s == pytest.approx(1.00)
    assert profile.config.post_release_stable_time_s == pytest.approx(0.50)
    assert profile.config.post_release_max_linear_speed_mps == pytest.approx(
        0.12
    )
    assert profile.config.post_release_max_angular_speed_rps == pytest.approx(
        0.35
    )
    assert profile.config.post_release_max_tilt_rad == pytest.approx(0.35)
    assert profile.config.require_supervisor_sensor_status is True
    assert profile.config.supervisor_sensor_status_timeout_s == pytest.approx(0.25)

    audit = profile.audit_report()
    assert audit["contract_sha256"] == (
        "d88694a5765a5a768fa0649bb7598e5be0b049f9a23277e0cbc61d005f5fb329"
    )
    assert len(audit["source_sha256"]) == 64
    assert audit["pct_topology_profile_reused"] is False
    assert "base_tomogram_contract" in topology
    assert "anchors_sim_ground_xyz" in topology
    assert "base_tomogram_contract" not in _production_payload()
    assert "anchors_sim_ground_xyz" not in _production_payload()


def test_pipeline_overrides_use_unique_navigation_body_height_contract() -> None:
    profile = load_scan_stair_freeze_profile(PROFILE_PATH)
    overrides = profile.pipeline_navigation_overrides()

    assert overrides["navigation_body_height_m"] == pytest.approx(0.338)
    assert "scan_stair_freeze_body_height_m" not in overrides
    assert "scan_stair_freeze_default_control_dt_s" not in overrides
    assert overrides["scan_stair_freeze_max_control_dt_s"] == pytest.approx(0.20)
    assert overrides["scan_stair_freeze_speed_mps"] == pytest.approx(0.18)
    assert overrides["scan_stair_freeze_require_supervisor_sensor_status"] is True
    assert profile.validate_runtime_bindings(
        navigation_body_height_m=0.338,
        control_dt_s=0.02,
    ) is profile.config
    with pytest.raises(ScanStairFreezeProfileError, match="唯一 navigation_body_height"):
        profile.validate_runtime_bindings(
            navigation_body_height_m=0.31,
            control_dt_s=0.02,
        )
    with pytest.raises(ScanStairFreezeProfileError, match="实际控制周期"):
        profile.validate_runtime_bindings(
            navigation_body_height_m=0.338,
            control_dt_s=0.01,
        )


def test_profile_binds_all_freeze_values_to_immutable_navigation_copy() -> None:
    profile = load_scan_stair_freeze_profile(PROFILE_PATH)
    original = NavigationSettings(
        navigation_body_height_m=0.338,
        scan_stair_freeze_speed_mps=0.22,
        scan_stair_freeze_require_supervisor_sensor_status=False,
        scan_stair_freeze_max_control_dt_s=0.33,
    )

    bound = bind_pipeline_navigation_settings(original, profile)

    assert bound is not original
    assert original.navigation_body_height_m == pytest.approx(0.338)
    assert original.scan_stair_freeze_speed_mps == pytest.approx(0.22)
    assert original.scan_stair_freeze_require_supervisor_sensor_status is False
    assert original.scan_stair_freeze_max_control_dt_s == pytest.approx(0.33)
    assert bound.navigation_body_height_m == pytest.approx(0.338)
    assert bound.scan_stair_freeze_speed_mps == pytest.approx(0.18)
    assert bound.scan_stair_freeze_approach_distance_m == pytest.approx(1.50)
    assert bound.scan_stair_freeze_exit_distance_m == pytest.approx(0.40)
    assert bound.scan_stair_freeze_full_lock_settle_time_s == pytest.approx(1.20)
    assert bound.scan_stair_freeze_root_release_settle_time_s == pytest.approx(1.00)
    assert bound.scan_stair_freeze_post_release_max_tilt_rad == pytest.approx(0.35)
    assert bound.scan_stair_freeze_require_supervisor_sensor_status is True
    assert bound.scan_stair_freeze_max_control_dt_s == pytest.approx(0.20)


def test_profile_binding_rejects_explicit_height_or_enable_conflict() -> None:
    profile = load_scan_stair_freeze_profile(PROFILE_PATH)

    with pytest.raises(
        ScanStairFreezeProfileError,
        match="唯一 navigation_body_height_m",
    ):
        bind_pipeline_navigation_settings(
            NavigationSettings(navigation_body_height_m=0.31),
            profile,
        )
    with pytest.raises(
        ScanStairFreezeProfileError,
        match="scan_stair_freeze_enabled",
    ):
        bind_pipeline_navigation_settings(
            NavigationSettings(scan_stair_freeze_enabled=False),
            profile,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload.__setitem__("unexpected", 1),
            "未知字段",
        ),
        (
            lambda payload: payload.__setitem__("schema_version", True),
            "schema_version",
        ),
        (
            lambda payload: payload["parameters"].__setitem__(
                "unknown_threshold", 0.1
            ),
            "未知字段",
        ),
        (
            lambda payload: payload["parameters"].pop("exit_distance_m"),
            "缺少字段",
        ),
        (
            lambda payload: payload["execution_contract"].__setitem__(
                "path_height_semantics", "base"
            ),
            "path_height_semantics",
        ),
        (
            lambda payload: payload["baseline_provenance"].__setitem__(
                "pct_topology_profile_reused", True
            ),
            "禁止复用 PCT 拓扑",
        ),
        (
            lambda payload: payload["parameters"].__setitem__(
                "speed_mps", 0.31
            ),
            "不能超过 0.30",
        ),
        (
            lambda payload: payload["parameters"].__setitem__(
                "post_release_stabilization_timeout_s", 0.49
            ),
            "稳定超时",
        ),
        (
            lambda payload: payload["parameters"].__setitem__(
                "require_supervisor_sensor_status", False
            ),
            "supervisor 传感器状态门",
        ),
    ],
)
def test_profile_rejects_schema_and_production_constraint_drift(
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    payload = copy.deepcopy(_production_payload())
    mutation(payload)
    source = _write_profile(tmp_path, payload)

    with pytest.raises(ScanStairFreezeProfileError, match=match):
        load_scan_stair_freeze_profile(source)


def test_profile_rejects_runtime_posture_drift(tmp_path: Path) -> None:
    payload = copy.deepcopy(_production_payload())
    payload["fixed_navigation_posture"]["dog_stand_joint_positions_rad"][0] = 0.2
    source = _write_profile(tmp_path, payload)

    with pytest.raises(ScanStairFreezeProfileError, match="runtime 常量"):
        load_scan_stair_freeze_profile(source)


def test_profile_rejects_content_changed_without_digest_update(tmp_path: Path) -> None:
    payload = copy.deepcopy(_production_payload())
    payload["description"] = "内容发生了未审计的变化。"
    source = _write_profile(tmp_path, payload, refresh_digest=False)

    with pytest.raises(ScanStairFreezeProfileError, match="contract_sha256"):
        load_scan_stair_freeze_profile(source)


def test_profile_rejects_duplicate_and_nonfinite_json_fields(tmp_path: Path) -> None:
    original = PROFILE_PATH.read_text(encoding="utf-8")
    duplicate = original.replace(
        '  "schema_version": 1,',
        '  "schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ScanStairFreezeProfileError, match="重复字段"):
        load_scan_stair_freeze_profile(duplicate_path)

    nonfinite = original.replace('"speed_mps": 0.18', '"speed_mps": NaN', 1)
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(ScanStairFreezeProfileError, match="非有限 JSON 数值"):
        load_scan_stair_freeze_profile(nonfinite_path)


def test_scene_binding_rejects_untracked_or_cross_scene_profile(tmp_path: Path) -> None:
    scene = load_scene_profile("multi_floor", PROJECT_ROOT)
    untracked = copy.copy(scene)
    object.__setattr__(untracked, "required_assets", ())
    with pytest.raises(ScanStairFreezeProfileError, match="required_assets"):
        load_scene_scan_stair_freeze_profile(untracked, PROJECT_ROOT)

    with pytest.raises(ScanStairFreezeProfileError, match="场景不匹配"):
        load_scan_stair_freeze_profile(PROFILE_PATH, expected_scene="liangzhu")
