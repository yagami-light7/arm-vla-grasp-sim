from __future__ import annotations

import math
from dataclasses import replace
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from source.interfaces import NavGoal, SimulationState
from source.navigation import (
    AStarNavPlanner,
    FixedCommandStairProbeExecutor,
    FixedCommandStairProbePlanner,
    PCTNavPlanner,
)
from source.navigation.scan_ros2_executor import (
    ScanRos2LifecyclePlanner,
    ScanRos2NavExecutor,
)
from source.navigation.navlib import OccupancyGridMap
from source.navigation.pct_adapter import PCTPlannerConfig
from source.pipeline import FullPhysicsConfig, NavigationSettings
from source.pipeline.config import PCT_MULTIFLOOR_LOCOMOTION_TASK
from source.pipeline.navigation_smoke import (
    _create_legacy_navigation_components_for_tests,
    _navigation_carry_smoke_start,
    _stair_locomotion_smoke_spec,
    create_navigation_carry_smoke_pipeline,
    create_navigation_components,
    create_navigation_smoke_pipeline,
    create_stair_locomotion_smoke_pipeline,
)
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
LOCAL_PCT_SERVER = PROJECT_ROOT / "scripts/navigation/pct_grid_server.py"
LOCAL_PCT_TOMOGRAM = PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle"
LOCAL_PCT_WALKABLE = PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy"
LOCAL_PCT_COLLISION = (
    PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_collision.ply"
)


def _write_flat_nav_map(tmp_path: Path) -> Path:
    map_dir = tmp_path / "flat_nav_map"
    map_dir.mkdir()
    np.save(map_dir / "occupancy.npy", np.zeros((20, 20), dtype=bool))
    map_json = map_dir / "map.json"
    map_json.write_text(
        json.dumps(
            {
                "image": "occupancy.npy",
                "resolution": 0.2,
                "origin": [-2.0, -2.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    return map_json


def _write_pct_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    tomogram_path = tmp_path / "tomogram.pickle"
    walkable_path = tmp_path / "walkable.npy"
    collision_ply_path = tmp_path / "collision.ply"
    traversability = np.full((4, 20, 20), 50.0, dtype=np.float32)
    zeros = np.zeros_like(traversability)
    tomogram = {
        "data": np.stack(
            [traversability, zeros, zeros, zeros, zeros],
            axis=0,
        ),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0], dtype=np.float64),
        "slice_h0": 0.0,
        "slice_dh": 0.2,
    }
    with tomogram_path.open("wb") as stream:
        pickle.dump(tomogram, stream)
    np.save(walkable_path, np.ones((4, 20, 20), dtype=bool))
    ply_header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    far_vertex = np.array(
        [(100.0, 100.0, 100.0)],
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]),
    )
    collision_ply_path.write_bytes(ply_header + far_vertex.tobytes())
    return tomogram_path, walkable_path, collision_ply_path


def _config(tmp_path: Path, navigation: NavigationSettings) -> FullPhysicsConfig:
    if navigation.global_planner == "pct":
        navigation = replace(
            navigation,
            pct_server_script=(
                navigation.pct_server_script or LOCAL_PCT_SERVER
            ),
            pct_tomogram_path=(
                navigation.pct_tomogram_path or LOCAL_PCT_TOMOGRAM
            ),
            pct_walkable_path=(
                navigation.pct_walkable_path or LOCAL_PCT_WALKABLE
            ),
            pct_collision_ply_path=(
                navigation.pct_collision_ply_path or LOCAL_PCT_COLLISION
            ),
        )
    return FullPhysicsConfig(
        task_json=TASK_PATH,
        output_dir=tmp_path,
        dry_run=True,
        navigation=navigation,
    )


def _spec_with_open_nav_map(tmp_path: Path):
    """为 A*/fallback 测试生成独立地图，避免依赖旧 839920 本地资产。"""

    map_dir = tmp_path / "nav_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    grid = OccupancyGridMap(
        np.zeros((160, 160), dtype=bool),
        0.1,
        (-6.0, -3.0, 0.0),
    )
    image_path = map_dir / "occupancy.pgm"
    meta_path = map_dir / "map.json"
    grid.save_pgm(image_path)
    grid.save_meta_file(meta_path, image_path=image_path.name)
    return replace(JsonTaskProvider().load(TASK_PATH), nav_map=str(meta_path))


def _state(x: float, y: float, z: float = 0.35) -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(x, y, z, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0,) * 6,
    )


def test_global_planner_astar_selects_astar(tmp_path: Path) -> None:
    spec = replace(
        JsonTaskProvider().load(TASK_PATH),
        nav_map=str(_write_flat_nav_map(tmp_path)),
    )
    planner, _executor, _verifier = _create_legacy_navigation_components_for_tests(
        config=_config(tmp_path, NavigationSettings(global_planner="astar")),
        episode_spec=spec,
    )

    assert isinstance(planner, AStarNavPlanner)


def test_ros2_bridge_selects_scan_executor_without_constructing_dwa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = replace(
        JsonTaskProvider().load(TASK_PATH),
        nav_map=str(_write_flat_nav_map(tmp_path)),
    )

    def _unexpected_dwa(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ROS 2 SCAN 链不应构造 DWA 配置或局部地图")

    monkeypatch.setattr(
        "source.pipeline.navigation_smoke._build_dwa_config",
        _unexpected_dwa,
    )
    monkeypatch.setattr(
        "source.pipeline.navigation_smoke._open_local_grid_map",
        _unexpected_dwa,
    )
    monkeypatch.setattr(
        "source.pipeline.navigation_smoke._create_navigation_planner",
        _unexpected_dwa,
    )

    planner, executor, _verifier = create_navigation_components(
        config=_config(tmp_path, NavigationSettings(global_planner="astar")),
        episode_spec=spec,
    )

    assert isinstance(planner, ScanRos2LifecyclePlanner)
    assert isinstance(executor, ScanRos2NavExecutor)


def test_ros2_bridge_external_path_does_not_require_map_or_pct_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """手工三维 Path 场景不应为 pipeline 生命周期伪造平面地图。"""

    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_smoke_scan_ramp.json"
    )

    def _unexpected_planner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("外部 ROS 2 Path 不应构造内部 A*/PCT planner")

    monkeypatch.setattr(
        "source.pipeline.navigation_smoke._create_navigation_planner",
        _unexpected_planner,
    )
    planner, executor, _verifier = create_navigation_components(
        config=FullPhysicsConfig(
            task_json=TASK_PATH,
            output_dir=tmp_path,
            navigation=NavigationSettings(
                global_planner="pct",
                pct_enabled=True,
                pct_fallback_to_astar=False,
            ),
        ),
        episode_spec=spec,
    )

    assert spec.nav_map == ""
    assert isinstance(planner, ScanRos2LifecyclePlanner)
    assert isinstance(executor, ScanRos2NavExecutor)


def test_ros2_bridge_loads_task_manual_path_for_stair_freeze(
    tmp_path: Path,
) -> None:
    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_smoke_scan_multifloor_stair_two_step.json"
    )
    planner, executor, _verifier = create_navigation_components(
        config=FullPhysicsConfig(
            task_json=TASK_PATH,
            output_dir=tmp_path,
            navigation=NavigationSettings(global_planner="pct"),
        ),
        episode_spec=spec,
    )

    plan = planner.plan(
        _state(spec.start.x, spec.start.y, float(spec.start.z or 0.0)),
        spec.pick_goal,
    )
    executor.reset(plan)

    assert len(plan.waypoints) == 6
    assert plan.metadata["reference_path_height_semantics"] == "ground"
    assert plan.waypoints[0][2] == pytest.approx(-0.12757488791649585)
    status = executor.status()
    assert status["live_reference_path_required"] is True
    assert status["live_reference_path_verified"] is False
    assert status["stair_freeze"]["component_count"] == 0
    assert plan.metadata["reference_path_stair_segment_indices"] == ((2, 5),)
    assert status["stair_freeze"]["phase"] == "not_applicable"
    assert status["stair_freeze"]["reason"] == "reference_path_unavailable"


def test_global_planner_pct_rejects_astar_fallback(tmp_path: Path) -> None:
    nav_map = _write_flat_nav_map(tmp_path)
    tomogram_path, walkable_path, collision_ply_path = _write_pct_assets(tmp_path)
    spec = replace(JsonTaskProvider().load(TASK_PATH), nav_map=str(nav_map))
    with pytest.raises(ValueError, match="禁止 PCT→A\\* fallback"):
        _create_legacy_navigation_components_for_tests(
            config=_config(
                tmp_path,
                NavigationSettings(
                    global_planner="pct",
                    pct_enabled=True,
                    pct_planner_root=tmp_path / "pct",
                    pct_fallback_to_astar=True,
                    pct_tomogram_path=tomogram_path,
                    pct_walkable_path=walkable_path,
                    pct_collision_ply_path=collision_ply_path,
                ),
            ),
            episode_spec=spec,
        )


def test_pct_rejects_missing_scene_asset_paths(tmp_path: Path) -> None:
    """新场景漏配地图时必须失败，不能静默回落到别墅 PCT 资产。"""

    spec = _spec_with_open_nav_map(tmp_path)
    config = FullPhysicsConfig(
        task_json=TASK_PATH,
        output_dir=tmp_path,
        dry_run=True,
        navigation=NavigationSettings(
            global_planner="pct",
            pct_enabled=True,
            pct_fallback_to_astar=False,
        ),
    )

    with pytest.raises(ValueError, match="pct_tomogram_path"):
        _create_legacy_navigation_components_for_tests(
            config=config,
            episode_spec=spec,
        )


def test_pct_without_fallback_allows_missing_flat_nav_map(tmp_path: Path) -> None:
    tomogram_path, walkable_path, collision_ply_path = _write_pct_assets(tmp_path)
    spec = replace(
        JsonTaskProvider().load(TASK_PATH),
        nav_map="source/scene/multifloor/nav_map/map.json",
    )

    planner, executor, _verifier = _create_legacy_navigation_components_for_tests(
        config=_config(
            tmp_path,
            NavigationSettings(
                global_planner="pct",
                pct_enabled=True,
                pct_fallback_to_astar=False,
                pct_tomogram_path=tomogram_path,
                pct_walkable_path=walkable_path,
                pct_collision_ply_path=collision_ply_path,
            ),
        ),
        episode_spec=spec,
    )

    assert isinstance(planner, PCTNavPlanner)
    assert planner.config.server_script == LOCAL_PCT_SERVER
    assert planner.config.tomogram_path == tomogram_path
    assert planner.config.walkable_path == walkable_path
    assert executor.local_map is not None


def test_pct_with_fallback_and_missing_flat_nav_map_is_rejected(tmp_path: Path) -> None:
    tomogram_path, walkable_path, collision_ply_path = _write_pct_assets(tmp_path)
    spec = replace(
        JsonTaskProvider().load(TASK_PATH),
        nav_map="source/scene/multifloor/nav_map/map.json",
    )

    with pytest.raises(ValueError, match="禁止 PCT→A\\* fallback"):
        _create_legacy_navigation_components_for_tests(
            config=_config(
                tmp_path,
                NavigationSettings(
                    global_planner="pct",
                    pct_enabled=True,
                    pct_fallback_to_astar=True,
                    pct_tomogram_path=tomogram_path,
                    pct_walkable_path=walkable_path,
                    pct_collision_ply_path=collision_ply_path,
                ),
            ),
            episode_spec=spec,
        )


def test_navigation_carry_smoke_uses_task_stable_start() -> None:
    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
    )

    start, source = _navigation_carry_smoke_start(spec)

    assert source == "carry.smoke_start"
    assert start == NavGoal(
        x=-3.48391,
        y=6.57414,
        z=0.36742,
        yaw=1.67247,
        floor_id="F1",
        slice_id=None,
    )


def test_navigation_carry_smoke_without_override_uses_pick_goal() -> None:
    spec = JsonTaskProvider().load(TASK_PATH)

    start, source = _navigation_carry_smoke_start(spec)

    assert source == "pick.base_goal"
    assert start is spec.pick_goal


def test_pct_navigation_carry_smoke_uses_multifloor_step_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
    )
    config = _config(
        tmp_path,
        NavigationSettings(global_planner="pct", pct_enabled=True),
    )
    config = replace(
        config,
        locomotion=replace(
            config.locomotion,
            policy_profile="pct_multifloor",
            locomotion_task=PCT_MULTIFLOOR_LOCOMOTION_TASK,
            locomotion_checkpoint=(
                PROJECT_ROOT / "checkpoints/go2_x5/pct_multifloor/model_26000.pt"
            ),
        ),
    )
    captured: dict[str, object] = {}

    def fake_create_navigation_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "source.pipeline.navigation_smoke._create_navigation_pipeline",
        fake_create_navigation_pipeline,
    )

    result = create_navigation_carry_smoke_pipeline(
        config=config,
        episode_spec=spec,
        episode_seed=0,
        episode_dir=tmp_path,
        simulation=object(),
    )

    assert result is not None
    carry_config = captured["config"]
    assert isinstance(carry_config, FullPhysicsConfig)
    assert carry_config.limits.navigation == 12000
    assert carry_config.limits.episode >= 15000


@pytest.mark.parametrize(
    ("extended_state_limits", "expected_navigation_limit"),
    [(True, 12000), (False, 5000)],
)
def test_pct_navigation_smoke_honors_task_step_budget(
    tmp_path: Path,
    monkeypatch,
    extended_state_limits: bool,
    expected_navigation_limit: int,
) -> None:
    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
    )
    spec = replace(
        spec,
        raw_task={
            **spec.raw_task,
            "navigation_execution": {
                **spec.raw_task["navigation_execution"],
                "extended_state_limits": extended_state_limits,
            },
        },
    )
    config = _config(
        tmp_path,
        NavigationSettings(global_planner="pct", pct_enabled=True),
    )
    config = replace(
        config,
        locomotion=replace(
            config.locomotion,
            policy_profile="pct_multifloor",
            locomotion_task=PCT_MULTIFLOOR_LOCOMOTION_TASK,
            locomotion_checkpoint=(
                PROJECT_ROOT / "checkpoints/go2_x5/pct_multifloor/model_26000.pt"
            ),
        ),
    )
    captured: dict[str, object] = {}

    def fake_create_navigation_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "source.pipeline.navigation_smoke._create_navigation_pipeline",
        fake_create_navigation_pipeline,
    )

    result = create_navigation_smoke_pipeline(
        config=config,
        episode_spec=spec,
        episode_seed=0,
        episode_dir=tmp_path,
        simulation=object(),
    )

    assert result is not None
    smoke_config = captured["config"]
    assert isinstance(smoke_config, FullPhysicsConfig)
    assert smoke_config.limits.navigation == expected_navigation_limit
    assert smoke_config.limits.episode == 15000
    assert smoke_config.navigation.final_position_tolerance == pytest.approx(0.08)
    assert smoke_config.navigation.place_position_tolerance == pytest.approx(0.08)
    assert smoke_config.navigation.final_yaw_tolerance == pytest.approx(0.20)
    assert smoke_config.navigation.stable_linear_velocity == pytest.approx(0.05)
    assert smoke_config.navigation.stable_angular_velocity == pytest.approx(0.10)
    assert smoke_config.navigation.require_yaw_alignment is True
    assert smoke_config.navigation.require_stable_base is True


def test_stair_locomotion_smoke_extends_goal_beyond_calibrated_exit(
    tmp_path: Path,
) -> None:
    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
    )
    config = _config(
        tmp_path,
        NavigationSettings(global_planner="pct", pct_enabled=True),
    )

    stair_spec = _stair_locomotion_smoke_spec(config, spec)

    assert stair_spec.start.x == 1.5
    assert stair_spec.start.y == 5.7
    assert stair_spec.start.z == 0.36742
    assert stair_spec.start.yaw == pytest.approx(
        math.atan2(6.27683 - 5.7, 1.51822 - 1.5)
    )
    exit_heading = math.atan2(7.05 - 7.79872, 2.70 - 2.69841)
    assert stair_spec.pick_goal.x == pytest.approx(2.70 + math.cos(exit_heading))
    assert stair_spec.pick_goal.y == pytest.approx(7.05 + math.sin(exit_heading))
    assert stair_spec.pick_goal.z == 3.62628
    assert stair_spec.place_goal is None
    assert stair_spec.object_prim_path is None
    assert stair_spec.raw_task["runtime_override"]["float_enabled"] is False
    assert stair_spec.raw_task["runtime_override"]["global_planner"] == "pct"
    assert stair_spec.raw_task["runtime_override"]["global_path"] == (
        "pct_online_path_3d"
    )
    assert stair_spec.raw_task["runtime_override"]["manual_centerline"] is False
    assert stair_spec.raw_task["runtime_override"]["stair_exit_xyz"] == [
        2.70,
        7.05,
        3.0,
    ]
    assert stair_spec.raw_task["runtime_override"]["exit_extension_m"] == 1.0
    assert "centerline" not in stair_spec.raw_task["runtime_override"]


def test_stair_locomotion_smoke_defaults_to_scan_goal_lifecycle(
    tmp_path: Path,
) -> None:
    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
    )
    config = FullPhysicsConfig(
        task_json=TASK_PATH,
        output_dir=tmp_path,
        navigation=NavigationSettings(
            global_planner="pct",
            pct_enabled=True,
            pct_server_script=LOCAL_PCT_SERVER,
            pct_tomogram_path=LOCAL_PCT_TOMOGRAM,
            pct_walkable_path=LOCAL_PCT_WALKABLE,
            pct_collision_ply_path=LOCAL_PCT_COLLISION,
            pct_stair_float_enabled=True,
        ),
        stair_locomotion_smoke=True,
    )

    pipeline = create_stair_locomotion_smoke_pipeline(
        config=config,
        episode_spec=spec,
        episode_seed=0,
        episode_dir=tmp_path,
        simulation=object(),
    )

    assert isinstance(pipeline.nav_planner, ScanRos2LifecyclePlanner)
    assert isinstance(pipeline.machine.nav_executor, ScanRos2NavExecutor)
    assert pipeline.nav_planner.publish_pct_goal is True
    assert pipeline.config.navigation.pct_stair_float_enabled is False
    assert pipeline.config.navigation.global_planner == "pct"
    assert pipeline.config.navigation.pct_enabled is False
    assert pipeline.config.stair_locomotion_smoke is True
    assert pipeline.episode_spec.raw_task["runtime_override"]["controller"] == (
        "scan_stair_freeze"
    )
    assert pipeline.episode_spec.raw_task["runtime_override"]["dwa_enabled"] is False


def test_stair_ros2_bridge_uses_external_path_lifecycle_without_pct_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """手工楼梯 Path 模式不得再启动 pipeline 内部 PCT 客户端。"""

    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
    )
    config = FullPhysicsConfig(
        task_json=TASK_PATH,
        output_dir=tmp_path,
        navigation=NavigationSettings(
            global_planner="pct",
            pct_enabled=True,
            pct_collision_ply_path=LOCAL_PCT_COLLISION,
            pct_stair_float_enabled=True,
        ),
        stair_locomotion_smoke=True,
    )

    def _unexpected_planner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ROS 2 手工楼梯 Path 不应构造 pipeline PCT 客户端")

    monkeypatch.setattr(
        "source.pipeline.navigation_smoke._create_navigation_planner",
        _unexpected_planner,
    )
    pipeline = create_stair_locomotion_smoke_pipeline(
        config=config,
        episode_spec=spec,
        episode_seed=0,
        episode_dir=tmp_path,
        simulation=object(),
    )

    assert isinstance(pipeline.nav_planner, ScanRos2LifecyclePlanner)
    assert isinstance(pipeline.machine.nav_executor, ScanRos2NavExecutor)
    assert pipeline.config.navigation.pct_stair_float_enabled is False


def test_manual_scan_stair_smoke_preserves_task_path_and_enables_freeze(
    tmp_path: Path,
) -> None:
    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_smoke_scan_multifloor_stair_two_step.json"
    )
    config = FullPhysicsConfig(
        task_json=TASK_PATH,
        output_dir=tmp_path,
        navigation=NavigationSettings(global_planner="pct", pct_enabled=True),
        stair_locomotion_smoke=True,
    )

    pipeline = create_stair_locomotion_smoke_pipeline(
        config=config,
        episode_spec=spec,
        episode_seed=0,
        episode_dir=tmp_path,
        simulation=object(),
    )
    plan = pipeline.nav_planner.plan(
        _state(spec.start.x, spec.start.y, float(spec.start.z or 0.0)),
        pipeline.episode_spec.pick_goal,
    )

    assert pipeline.episode_spec.start == spec.start
    assert pipeline.episode_spec.pick_goal == spec.pick_goal
    assert len(plan.waypoints) == 6
    assert pipeline.config.navigation.pct_enabled is False
    assert pipeline.config.navigation.scan_stair_freeze_enabled is True
    assert pipeline.machine.nav_executor.config.require_live_reference_path is True
    assert plan.metadata["reference_path_stair_segment_indices"] == ((2, 5),)
    assert len(plan.metadata["reference_path_points_sha256"]) == 64
    override = pipeline.episode_spec.raw_task["runtime_override"]
    assert override["controller"] == "scan_stair_freeze"
    assert override["global_planner"] == "external_ros2_path"
    assert override["dwa_enabled"] is False
    assert override["pure_physics"] is False


def test_stair_fixed_command_probe_preserves_task_pose_and_bypasses_planners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_smoke_scan_multifloor_stair_two_step.json"
    )
    config = FullPhysicsConfig(
        task_json=TASK_PATH,
        output_dir=tmp_path,
        navigation=NavigationSettings(
            global_planner="pct",
            pct_enabled=True,
            stair_fixed_command_probe=True,
            stair_probe_forward_velocity_mps=0.30,
            stair_probe_drive_duration_s=3.20,
        ),
        stair_locomotion_smoke=True,
    )

    def _unexpected_planner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("固定速度探针不应构造 PCT/A*/DWA/SCAN planner")

    monkeypatch.setattr(
        "source.pipeline.navigation_smoke._create_navigation_planner",
        _unexpected_planner,
    )
    pipeline = create_stair_locomotion_smoke_pipeline(
        config=config,
        episode_spec=spec,
        episode_seed=0,
        episode_dir=tmp_path,
        simulation=object(),
    )

    assert isinstance(pipeline.nav_planner, FixedCommandStairProbePlanner)
    assert isinstance(
        pipeline.machine.nav_executor,
        FixedCommandStairProbeExecutor,
    )
    assert pipeline.config.navigation.global_planner == "bypassed"
    assert pipeline.config.navigation.pct_enabled is False
    assert pipeline.config.navigation.pct_fallback_to_astar is False
    assert pipeline.config.navigation.pct_stair_float_enabled is False
    assert pipeline.episode_spec.start == spec.start
    assert pipeline.episode_spec.pick_goal == spec.pick_goal
    override = pipeline.episode_spec.raw_task["runtime_override"]
    assert override["global_planner"] == "bypassed"
    assert override["scan_enabled"] is False
    assert override["pct_enabled"] is False
    assert override["float_enabled"] is False
    assert override["requested_command_vx_vy_wz"] == [0.30, 0.0, 0.0]
    assert override["drive_duration_s"] == pytest.approx(3.20)
    summary = pipeline._build_summary(
        started_at=0.0,
        duration_steps=0,
        final_state=_state(
            spec.start.x,
            spec.start.y,
            float(spec.start.z or 0.0),
        ),
        last_action={},
    )
    assert summary["navigation_acceptance"]["global_planner"] == "bypassed"
    assert summary["task_config"]["runtime_override"]["pct_enabled"] is False

def test_stair_fixed_command_probe_requires_stair_smoke_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="stair_fixed_command_probe requires stair_locomotion_smoke=True",
    ):
        FullPhysicsConfig(
            task_json=TASK_PATH,
            output_dir=tmp_path,
            navigation=NavigationSettings(stair_fixed_command_probe=True),
        )


def test_bypassed_global_planner_is_private_to_fixed_stair_probe(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="global_planner must be one of"):
        FullPhysicsConfig(
            task_json=TASK_PATH,
            output_dir=tmp_path,
            navigation=NavigationSettings(global_planner="bypassed"),
            stair_locomotion_smoke=True,
        )


def test_pct_config_rejects_explicit_astar_fallback() -> None:
    with pytest.raises(ValueError, match="禁止 PCT→A\\* fallback"):
        PCTPlannerConfig(enabled=True, fallback_to_astar=True)
