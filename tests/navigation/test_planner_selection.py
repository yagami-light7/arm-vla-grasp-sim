from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from source.interfaces import NavGoal, SimulationState
from source.navigation import AStarNavPlanner, PCTNavPlanner
from source.navigation.navlib import OccupancyGridMap
from source.navigation.pct_adapter import PCTPlannerConfig
from source.pipeline import FullPhysicsConfig, NavigationSettings
from source.pipeline.config import PCT_MULTIFLOOR_LOCOMOTION_TASK
from source.pipeline.navigation_smoke import (
    _navigation_carry_smoke_start,
    create_navigation_carry_smoke_pipeline,
    create_navigation_components,
)
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
LOCAL_PCT_SERVER = PROJECT_ROOT / "scripts/navigation/pct_grid_server.py"
LOCAL_PCT_TOMOGRAM = PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle"
LOCAL_PCT_WALKABLE = PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy"


def _config(tmp_path: Path, navigation: NavigationSettings) -> FullPhysicsConfig:
    return FullPhysicsConfig(
        task_json=TASK_PATH,
        output_dir=tmp_path,
        dry_run=True,
        navigation=navigation,
    )


def _state(x: float, y: float, z: float = 0.35) -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(x, y, z, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0,) * 6,
    )


def test_global_planner_astar_selects_astar(tmp_path: Path) -> None:
    spec = JsonTaskProvider().load(TASK_PATH)
    planner, _executor, _verifier = create_navigation_components(
        config=_config(tmp_path, NavigationSettings(global_planner="astar")),
        episode_spec=spec,
    )

    assert isinstance(planner, AStarNavPlanner)


def test_global_planner_pct_selects_pct_with_astar_fallback(tmp_path: Path) -> None:
    spec = JsonTaskProvider().load(TASK_PATH)
    planner, _executor, _verifier = create_navigation_components(
        config=_config(
            tmp_path,
            NavigationSettings(
                global_planner="pct",
                pct_enabled=True,
                pct_planner_root=tmp_path / "pct",
                pct_fallback_to_astar=True,
            ),
        ),
        episode_spec=spec,
    )

    assert isinstance(planner, PCTNavPlanner)
    assert isinstance(planner.fallback_planner, AStarNavPlanner)


def test_pct_without_fallback_allows_missing_flat_nav_map(tmp_path: Path) -> None:
    spec = replace(
        JsonTaskProvider().load(TASK_PATH),
        nav_map="source/scene/multifloor/nav_map/map.json",
    )

    planner, executor, _verifier = create_navigation_components(
        config=_config(
            tmp_path,
            NavigationSettings(
                global_planner="pct",
                pct_enabled=True,
                pct_fallback_to_astar=False,
            ),
        ),
        episode_spec=spec,
    )

    assert isinstance(planner, PCTNavPlanner)
    assert planner.fallback_planner is None
    assert planner.config.server_script == LOCAL_PCT_SERVER
    assert planner.config.tomogram_path == LOCAL_PCT_TOMOGRAM
    assert planner.config.walkable_path == LOCAL_PCT_WALKABLE
    assert executor.local_map is not None


def test_pct_with_fallback_and_missing_flat_nav_map_disables_fallback(tmp_path: Path) -> None:
    spec = replace(
        JsonTaskProvider().load(TASK_PATH),
        nav_map="source/scene/multifloor/nav_map/map.json",
    )

    planner, executor, _verifier = create_navigation_components(
        config=_config(
            tmp_path,
            NavigationSettings(
                global_planner="pct",
                pct_enabled=True,
                pct_fallback_to_astar=True,
            ),
        ),
        episode_spec=spec,
    )

    assert isinstance(planner, PCTNavPlanner)
    assert planner.fallback_planner is None
    assert planner.config.fallback_to_astar is False
    assert executor.local_map is not None


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


def test_pct_failure_falls_back_to_astar() -> None:
    class FailingClient:
        def plan(self, *, start, end):
            del start, end
            raise RuntimeError("PCT unavailable")

    grid = OccupancyGridMap(
        np.zeros((20, 20), dtype=bool),
        0.1,
        (0.0, 0.0, 0.0),
    )
    planner = PCTNavPlanner(
        PCTPlannerConfig(enabled=True, fallback_to_astar=True),
        client=FailingClient(),
        fallback_planner=AStarNavPlanner(grid_map=grid),
    )

    plan = planner.plan(_state(0.15, 0.15), NavGoal(x=1.15, y=0.15, yaw=0.0))

    assert plan.metadata["planner"] == "astar_fallback_after_pct_failure"
    assert "PCT unavailable" in plan.metadata["pct_failure_reason"]
    assert plan.waypoints[0] == (0.15, 0.15)
    assert plan.waypoints[-1] == (1.15, 0.15)
