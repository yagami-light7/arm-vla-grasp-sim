from types import SimpleNamespace

from source.diagnostics.planned_trajectories import (
    manipulation_tcp_segments,
    navigation_path_points,
)


def test_navigation_path_points_prefers_pct_path_3d() -> None:
    plan = SimpleNamespace(
        metadata={"path_3d": ((1.0, 2.0, 0.5), (3.0, 4.0, 1.5))},
        waypoints=((8.0, 9.0),),
        goal=SimpleNamespace(z=7.0),
    )

    assert navigation_path_points(plan) == (
        (1.0, 2.0, 0.5),
        (3.0, 4.0, 1.5),
    )


def test_navigation_path_points_falls_back_to_xy_waypoints() -> None:
    plan = SimpleNamespace(
        metadata={},
        waypoints=((1.0, 2.0), (3.0, 4.0)),
        goal=SimpleNamespace(z=2.5),
    )

    assert navigation_path_points(plan) == (
        (1.0, 2.0, 2.5),
        (3.0, 4.0, 2.5),
    )


def test_manipulation_tcp_segments_uses_only_curobo_world_paths() -> None:
    plan = SimpleNamespace(
        metadata={
            "segments": (
                {"name": "open_gripper", "type": "gripper"},
                {
                    "name": "move_to_pregrasp",
                    "type": "motion",
                    "trajectory": {
                        "tcp_position_world": (
                            (0.1, 0.2, 0.3),
                            (0.4, 0.5, 0.6),
                        )
                    },
                },
                {
                    "name": "synthetic_return_home",
                    "type": "motion",
                    "trajectory": {"q": ((0.0,) * 6, (0.1,) * 6)},
                },
            )
        }
    )

    assert manipulation_tcp_segments(plan) == (
        {
            "index": 1,
            "name": "move_to_pregrasp",
            "points": ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
        },
    )
