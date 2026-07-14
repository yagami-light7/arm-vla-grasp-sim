from types import SimpleNamespace

import pytest

from source.diagnostics import planned_trajectories as trajectory_debug
from source.diagnostics.planned_trajectories import (
    manipulation_tcp_segments,
    navigation_path_points,
    velocity_command_guide_geometry,
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


def test_navigation_path_points_prefers_surface_visualization_path() -> None:
    plan = SimpleNamespace(
        metadata={
            "path_3d": ((1.0, 2.0, 0.5), (3.0, 4.0, 1.5)),
            "visualization_path_3d": (
                (1.0, 2.0, 0.05),
                (2.0, 3.0, 0.55),
                (3.0, 4.0, 1.05),
            ),
        },
        waypoints=(),
        goal=SimpleNamespace(z=0.0),
    )

    assert navigation_path_points(plan) == (
        (1.0, 2.0, 0.05),
        (2.0, 3.0, 0.55),
        (3.0, 4.0, 1.05),
    )


def test_velocity_command_guide_rotates_body_command_to_world() -> None:
    geometry = velocity_command_guide_geometry(
        (
            1.0,
            2.0,
            0.4,
            2.0**-0.5,
            0.0,
            0.0,
            2.0**-0.5,
        ),
        (0.25, 0.10, 0.50),
    )

    assert geometry["linear_velocity_world"] == pytest.approx((-0.10, 0.25))
    assert geometry["command_body"] == pytest.approx((0.25, 0.10, 0.50))
    assert geometry["linear_visible"] is True
    assert geometry["angular_visible"] is True
    linear_start, linear_end = geometry["linear_curves"][0]
    assert linear_start == pytest.approx((1.0, 2.0, 0.88))
    assert linear_end == pytest.approx((0.84, 2.40, 0.88))


def test_velocity_command_uses_official_debug_draw_lines(monkeypatch) -> None:
    class FakeDebugDraw:
        def __init__(self) -> None:
            self.clear_calls = 0
            self.draw_calls = []
            self.line_count = 0

        def clear_lines(self) -> None:
            self.clear_calls += 1
            self.line_count = 0

        def draw_lines(self, starts, ends, colors, widths) -> None:
            self.draw_calls.append((starts, ends, colors, widths))
            self.line_count = len(starts)

        def get_num_lines(self) -> int:
            return self.line_count

    debug_draw = FakeDebugDraw()
    monkeypatch.setattr(
        trajectory_debug,
        "_acquire_velocity_debug_draw_interface",
        lambda: debug_draw,
    )

    report = trajectory_debug.draw_velocity_command(
        robot_root_pose=(1.0, 2.0, 0.4, 1.0, 0.0, 0.0, 0.0),
        base_velocity=(0.25, 0.10, 0.50),
        source="stair_locomotion_heading_tracker",
    )

    assert report["available"] is True
    assert report["renderer"] == "isaacsim.util.debug_draw"
    assert report["fallback_used"] is False
    assert report["debug_draw_line_count"] > 3
    assert report["debug_draw_reported_line_count"] == report[
        "debug_draw_line_count"
    ]
    assert debug_draw.clear_calls == 1
    assert len(debug_draw.draw_calls) == 1
    starts, ends, colors, widths = debug_draw.draw_calls[0]
    assert len(starts) == len(ends) == len(colors) == len(widths)
    assert colors[0] == (0.20, 1.0, 0.25, 1.0)
    assert colors[-1] == (1.0, 0.35, 0.05, 1.0)
    assert widths[0] == 5.0
    assert widths[-1] == 4.0

    stopped_report = trajectory_debug.draw_velocity_command(
        robot_root_pose=(1.0, 2.0, 0.4, 1.0, 0.0, 0.0, 0.0),
        base_velocity=(0.0, 0.0, 0.0),
        source="stair_locomotion_completed",
    )

    assert stopped_report["debug_draw_line_count"] == 0
    assert stopped_report["debug_draw_reported_line_count"] == 0
    assert debug_draw.clear_calls == 2
    assert len(debug_draw.draw_calls) == 1


def test_velocity_command_falls_back_to_usd_when_debug_draw_fails(
    monkeypatch,
) -> None:
    def fail_debug_draw():
        raise RuntimeError("extension unavailable")

    monkeypatch.setattr(
        trajectory_debug,
        "_acquire_velocity_debug_draw_interface",
        fail_debug_draw,
    )
    monkeypatch.setattr(
        trajectory_debug,
        "_draw_velocity_command_with_usd",
        lambda **_kwargs: {
            "available": True,
            "type": "stair_velocity_command",
        },
    )

    report = trajectory_debug.draw_velocity_command(
        robot_root_pose=(1.0, 2.0, 0.4, 1.0, 0.0, 0.0, 0.0),
        base_velocity=(0.25, 0.0, 0.0),
        source="stair_locomotion_heading_tracker",
    )

    assert report["available"] is True
    assert report["renderer"] == "usd_basis_curves_fallback"
    assert report["fallback_used"] is True
    assert report["debug_draw_error"] == (
        "RuntimeError: extension unavailable"
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
