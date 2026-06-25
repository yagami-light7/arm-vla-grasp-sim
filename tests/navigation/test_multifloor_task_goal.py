from __future__ import annotations

from pathlib import Path

from source.diagnostics import NavigationEpisodeVerifier
from source.interfaces import SimulationState
from source.pipeline import BaseGoalRandomizationSettings, RandomizationSettings
from source.tasks import JsonTaskProvider, prepare_episode_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"


def _state(x: float, y: float, z: float) -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(x, y, z, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0,) * 6,
    )


def test_prepare_episode_spec_preserves_multifloor_goal_fields() -> None:
    base = JsonTaskProvider().load(TASK_PATH)
    episode = prepare_episode_spec(
        base,
        episode_id=42,
        seed=7,
        settings=RandomizationSettings(
            enabled=False,
            base_goal=BaseGoalRandomizationSettings(enabled=False),
        ),
    )

    assert episode.start.z == 0.35
    assert episode.start.floor_id == "F1"
    assert episode.start.slice_id == 0
    assert episode.pick_goal.z == 0.35
    assert episode.pick_goal.floor_id == "F1"
    assert episode.pick_goal.slice_id == 0
    assert episode.place_goal is not None
    assert episode.place_goal.z == 1.8
    assert episode.place_goal.floor_id == "F2"
    assert episode.place_goal.slice_id == 1


def test_navigation_verifier_checks_z_only_when_goal_has_z() -> None:
    episode = JsonTaskProvider().load(TASK_PATH)
    verifier = NavigationEpisodeVerifier(
        position_tolerance=0.10,
        goal_z_tolerance=0.20,
    )

    bad_z = verifier.verify_place_reachable(
        _state(episode.place_goal.x, episode.place_goal.y, 0.35),
        episode,
    )
    good_z = verifier.verify_place_reachable(
        _state(episode.place_goal.x, episode.place_goal.y, episode.place_goal.z),
        episode,
    )

    assert not bad_z.success
    assert bad_z.metadata["z_check_enabled"] is True
    assert bad_z.metadata["z_reached"] is False
    assert good_z.success
    assert good_z.metadata["z_reached"] is True
