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
    assert base.raw_task["place"][
        "place_release_peak_downward_speed_tolerance_mps"
    ] == pytest.approx(0.55)


def test_multifloor_lerobot_uses_fixed_six_subtask_contract() -> None:
    task = JsonTaskProvider().load(TASK_PATH).raw_task
    recording = task["recording"]
    segmentation = task["subtask_segmentation"]

    # 别墅与良渚必须共享同一训练数据契约，不能因场景切换退回未分段导出。
    assert recording["front_camera"] is True
    assert recording["wrist_camera"] is True
    assert segmentation == {
        "enabled": True,
        "schema": "nav_straight_turn_stop__arm_approach_contact_retreat_v1",
        "directory_export": True,
        "output_layout": "episodes_task_episode_subtask_front_wrist_v3",
        "min_segment_frames": 3,
        "hysteresis_frames": 2,
        "navigation": {
            "stop_command_linear_max_mps": 0.03,
            "stop_command_angular_max_rps": 0.08,
            "stop_measured_linear_max_mps": 0.08,
            "stop_measured_angular_max_rps": 0.2,
            "turn_command_angular_min_rps": 0.12,
            "turn_measured_angular_min_rps": 0.25,
            "turn_yaw_delta_min_rad": 0.03,
        },
        "contact_label_source": "heuristic_action_and_kinematics",
    }


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
