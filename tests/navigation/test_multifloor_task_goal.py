from __future__ import annotations

import math
from pathlib import Path

import pytest

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

    # start.z 表示机器人 root 高度，不是地面高度。
    assert episode.start.z == 0.36742
    assert episode.start.floor_id == "F1"
    assert episode.start.slice_id is None
    assert episode.pick_goal.z == 0.36742
    assert episode.pick_goal.floor_id == "F1"
    assert episode.pick_goal.slice_id is None
    assert episode.place_goal is not None
    assert episode.place_goal.z == 3.62628
    assert episode.place_goal.floor_id == "F2"
    assert episode.place_goal.slice_id is None


def test_multifloor_fixed_base_goals_are_not_object_centers() -> None:
    base = JsonTaskProvider().load(TASK_PATH)
    assert base.object_initial_pose is not None
    assert base.place_target_pose is not None
    assert base.place_goal is not None

    pick_distance = math.hypot(
        base.pick_goal.x - base.object_initial_pose[0],
        base.pick_goal.y - base.object_initial_pose[1],
    )
    place_distance = math.hypot(
        base.place_goal.x - base.place_target_pose[0],
        base.place_goal.y - base.place_target_pose[1],
    )

    # 固定 pick 站位必须同时满足桌边安全距离和 X5 机械臂工作空间。
    assert 0.45 <= pick_distance <= 0.60
    assert 0.20 <= place_distance <= 0.80
    assert base.object_prim_path == "/World/apple_01"
    assert base.object_initial_pose[:3] == pytest.approx(
        (-3.5059430599212646, 7.245284080505371, 0.5628772974014282)
    )
    assert base.object_initial_pose[3:] == pytest.approx(
        (0.033854853713603555, -0.019953342379394102, -0.010068430908084697)
    )


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
