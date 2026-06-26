from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from source.interfaces import NavGoal, SimulationState
from source.navigation import AStarNavPlanner, PCTNavPlanner
from source.navigation.navlib import OccupancyGridMap
from source.navigation.pct_adapter import PCTPlannerConfig
from source.pipeline import FullPhysicsConfig, NavigationSettings
from source.pipeline.navigation_smoke import create_navigation_components
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


def test_pct_with_fallback_requires_existing_flat_nav_map(tmp_path: Path) -> None:
    spec = replace(
        JsonTaskProvider().load(TASK_PATH),
        nav_map="source/scene/multifloor/nav_map/map.json",
    )

    try:
        create_navigation_components(
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
    except ValueError as exc:
        assert "requires an existing nav_map" in str(exc)
    else:
        raise AssertionError("missing nav_map should fail when PCT fallback is enabled")


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
