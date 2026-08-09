from __future__ import annotations

import numpy as np
import pytest

from scripts.navigation.pct_grid_server import (
    _State,
    _astar,
    _build_gateway_mask,
    _build_stair_surface_mask,
    _compress_path,
    _handle_request,
    _neighbors,
    _ordered_stair_midpoints,
    _parse_cross_floor_gateways,
    _same_floor_direct_path,
    _snap_to_walkable,
    _stair_progress_allowed,
    _stair_same_slice_progress_allowed,
    _vertical_transition_path_is_walkable,
    plan_request,
)


class _FakeState:
    def __init__(
        self,
        *,
        n_slice: int = 6,
        blocked: set[tuple[int, int, int]] | None = None,
    ) -> None:
        self.n_slice = n_slice
        self.resolution = 0.2
        self.slice_dh = 0.5
        self.blocked = blocked or set()

    def is_walkable(
        self,
        node: tuple[int, int, int],
        *,
        hard_obstacle_mask: np.ndarray | None = None,
    ) -> bool:
        del hard_obstacle_mask
        slice_index, x, y = node
        return (
            0 <= slice_index < self.n_slice
            and 0 <= x < 30
            and 0 <= y < 30
            and node not in self.blocked
        )


class _SnapState:
    def __init__(self) -> None:
        self.n_slice = 1
        self.dimx = 7
        self.dimy = 7
        self.free = {(0, 2, 3), (0, 4, 3)}

    def pct_xy_to_grid(self, xy: np.ndarray) -> tuple[int, int]:
        return int(round(float(xy[0]))), int(round(float(xy[1])))

    def grid_to_pct_xyz(self, node: tuple[int, int, int]) -> list[float]:
        return [float(node[1]), float(node[2]), float(node[0])]

    def is_walkable(
        self,
        node: tuple[int, int, int],
        *,
        hard_obstacle_mask: np.ndarray | None = None,
    ) -> bool:
        del hard_obstacle_mask
        return node in self.free


def _open_grid_state(
    walkable: np.ndarray,
    *,
    grid_timeout_sec: float = 10.0,
) -> _State:
    """构造不含硬障碍的小型 PCT 三维栅格。"""

    traversability = np.ones(walkable.shape, dtype=np.float32)
    return _State(
        tomogram={
            "data": np.stack([traversability] * 5, axis=0),
            "resolution": 0.2,
            "center": np.array([0.0, 0.0]),
            "slice_h0": 0.0,
            "slice_dh": 0.5,
        },
        walkable=walkable,
        robot_root_to_floor_m=0.0,
        grid_timeout_sec=grid_timeout_sec,
    )


def test_snap_to_walkable_breaks_equal_distance_tie_toward_route_goal() -> None:
    state = _SnapState()

    node, distance = _snap_to_walkable(
        state,  # type: ignore[arg-type]
        np.asarray([3.0, 3.0]),
        0,
        preferred_xy=np.asarray([10.0, 3.0]),
    )

    assert distance == 1
    assert node == (0, 4, 3)


def test_snap_to_walkable_never_falls_back_to_same_xy_on_wrong_slice() -> None:
    walkable = np.zeros((3, 5, 5), dtype=bool)
    state = _open_grid_state(walkable)
    state.traversability.fill(0.0)
    query_xy = np.asarray([0.0, 0.0])
    query_x, query_y = state.pct_xy_to_grid(query_xy)
    state.walkable[2, query_x, query_y] = True

    with pytest.raises(RuntimeError, match=r"slice 0 .*找不到可走格"):
        _snap_to_walkable(state, query_xy, 0)

    response = plan_request(
        state,
        start=(0.0, 0.0, 0.0),
        end=(0.0, 0.0, 1.0),
    )
    assert response["status"] == "error"
    assert "slice 0" in response["msg"]


def test_astar_cancel_check_interrupts_an_active_search() -> None:
    state = _open_grid_state(np.ones((1, 9, 9), dtype=bool))
    cancel_call_count = 0

    def cancel_after_first_expansion() -> bool:
        nonlocal cancel_call_count
        cancel_call_count += 1
        return cancel_call_count >= 2

    with pytest.raises(InterruptedError, match="规划已取消"):
        _astar(
            state,
            (0, 1, 1),
            (0, 7, 7),
            cancel_check=cancel_after_first_expansion,
        )

    assert cancel_call_count == 2


def test_astar_deadline_interrupts_an_active_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _open_grid_state(np.ones((1, 9, 9), dtype=bool))
    clock_samples = iter((0.0, 2.0))
    monkeypatch.setattr(
        "scripts.navigation.pct_grid_server.time.monotonic",
        lambda: next(clock_samples),
    )

    with pytest.raises(TimeoutError, match="超过截止时间"):
        _astar(
            state,
            (0, 1, 1),
            (0, 7, 7),
            deadline_monotonic=1.0,
        )


def test_plan_request_honors_preexisting_cancellation() -> None:
    state = _open_grid_state(np.ones((1, 5, 5), dtype=bool))

    response = plan_request(
        state,
        start=(0.0, 0.0, 0.0),
        end=(0.2, 0.0, 0.0),
        cancel_check=lambda: True,
    )

    assert response["status"] == "error"
    assert response["msg"] == "PCT grid 规划已取消。"
    assert "InterruptedError" in response["traceback"]


def test_same_floor_direct_path_accepts_clear_robot_width_corridor() -> None:
    state = _FakeState()

    path = _same_floor_direct_path(
        state,
        (2, 5, 5),
        (2, 7, 20),
        corridor_radius_cells=1,
    )

    assert path == [(2, 5, 5), (2, 7, 20)]


def test_compress_path_preserves_every_vertical_slice_transition() -> None:
    path = [
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.4, 0.2, 0.5],
        [0.6, 0.4, 1.0],
        [1.4, 0.4, 1.0],
    ]

    compressed = _compress_path(path)

    assert [0.4, 0.2, 0.5] in compressed
    assert [0.6, 0.4, 1.0] in compressed


def test_state_owned_search_and_compression_limits_are_honored() -> None:
    traversability = np.ones((1, 5, 5), dtype=np.float32)
    state = _State(
        tomogram={
            "data": np.stack([traversability] * 5, axis=0),
            "resolution": 0.2,
            "center": np.array([0.0, 0.0]),
            "slice_h0": 0.0,
            "slice_dh": 0.5,
        },
        walkable=np.ones((1, 5, 5), dtype=bool),
        grid_max_expansions=1,
        grid_compress_max_segment_m=0.30,
    )

    with pytest.raises(TimeoutError, match="最大扩展数: 1"):
        _astar(state, (0, 1, 1), (0, 3, 3))
    compressed = _compress_path(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        max_segment_length_m=state.grid_compress_max_segment_m,
    )
    assert compressed == [
        [0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.6, 0.0, 0.0],
    ]


def test_same_floor_direct_path_rejects_blocked_corridor_across_neighbor_slices() -> None:
    blocked = {
        (slice_index, 6, 12)
        for slice_index in (1, 2, 3)
    }
    state = _FakeState(blocked=blocked)

    path = _same_floor_direct_path(
        state,
        (2, 5, 5),
        (2, 7, 20),
        corridor_radius_cells=1,
    )

    assert path is None


def test_same_floor_direct_path_does_not_shortcut_multiple_slices() -> None:
    state = _FakeState()

    path = _same_floor_direct_path(
        state,
        (1, 5, 5),
        (4, 7, 20),
    )

    assert path is None


def test_state_hard_obstacle_overrides_walkable_assets() -> None:
    traversability = np.ones((3, 5, 5), dtype=np.float32)
    tomogram = {
        "data": np.stack(
            [
                traversability,
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
            ],
            axis=0,
        ),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    hard_obstacles = np.zeros((5, 5), dtype=bool)
    hard_obstacles[2, 2] = True
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((3, 5, 5), dtype=bool),
        hard_obstacle_mask=hard_obstacles,
        hard_obstacle_min_slices=6,
    )

    assert not state.is_walkable((1, 2, 2))
    assert state.is_walkable((1, 2, 3))


def test_state_slice_volume_only_blocks_matching_floor() -> None:
    traversability = np.ones((3, 5, 5), dtype=np.float32)
    tomogram = {
        "data": np.stack(
            [
                traversability,
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
            ],
            axis=0,
        ),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    obstacle_volume = np.zeros((3, 5, 5), dtype=bool)
    obstacle_volume[0, 2, 2] = True
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((3, 5, 5), dtype=bool),
        hard_obstacle_mask=obstacle_volume,
        hard_obstacle_min_slices=6,
    )

    assert not state.is_walkable((0, 2, 2))
    assert state.is_walkable((1, 2, 2))


def test_neighbors_reject_diagonal_corner_cutting() -> None:
    state = _FakeState(
        blocked={
            (2, 6, 5),
            (2, 5, 6),
        }
    )

    neighbors = {node for node, _cost in _neighbors(state, (2, 5, 5))}

    assert (2, 6, 6) not in neighbors


def test_neighbors_reject_pure_vertical_slice_transition() -> None:
    state = _FakeState()

    neighbors = {node for node, _cost in _neighbors(state, (2, 5, 5))}

    assert (1, 5, 5) not in neighbors
    assert (3, 5, 5) not in neighbors
    assert (3, 6, 5) in neighbors


def test_vertical_transition_accepts_support_switch_between_slices() -> None:
    state = _FakeState(
        blocked={
            (2, 8, 5),
            (2, 9, 5),
            (3, 6, 5),
        }
    )

    assert _vertical_transition_path_is_walkable(
        state,
        (2, 5, 5),
        (3, 9, 5),
        hard_obstacle_mask=None,
    )


def test_neighbors_limit_vertical_transition_to_gateway_mask() -> None:
    state = _FakeState()
    gateway_mask = np.zeros((30, 30), dtype=bool)
    gateway_mask[7, 5] = True
    gateway_mask[8, 5] = True

    outside_neighbors = {
        node
        for node, _cost in _neighbors(
            state,
            (2, 5, 5),
            vertical_gateway_mask=gateway_mask,
        )
    }
    gateway_neighbors = {
        node
        for node, _cost in _neighbors(
            state,
            (2, 7, 5),
            vertical_gateway_mask=gateway_mask,
        )
    }

    assert (3, 6, 5) not in outside_neighbors
    assert (3, 8, 5) in gateway_neighbors


def test_neighbors_limit_vertical_transition_to_3d_stair_surface_mask() -> None:
    state = _FakeState()
    surface_mask = np.zeros((6, 30, 30), dtype=bool)
    surface_mask[2, 7, 5] = True
    surface_mask[3, 8, 5] = True

    neighbors = {
        node
        for node, _cost in _neighbors(
            state,
            (2, 7, 5),
            vertical_gateway_mask=surface_mask,
        )
    }

    assert (3, 8, 5) in neighbors
    assert (3, 7, 6) not in neighbors


def test_neighbors_enforce_monotonic_vertical_direction() -> None:
    state = _FakeState()

    upward = {
        node
        for node, _cost in _neighbors(
            state,
            (2, 5, 5),
            vertical_direction=1,
        )
    }

    assert (3, 6, 5) in upward
    assert (1, 6, 5) not in upward
    assert (2, 6, 5) in upward


def test_parse_cross_floor_gateways_accepts_json_and_semicolon() -> None:
    assert _parse_cross_floor_gateways("[[1.0,2.0,3.0]]") == ((1.0, 2.0, 3.0),)
    assert _parse_cross_floor_gateways("1,2,3;4,5,6") == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    )


def test_gateway_mask_connects_stair_entry_to_exit_as_corridor() -> None:
    traversability = np.ones((3, 20, 20), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((3, 20, 20), dtype=bool),
    )

    mask = _build_gateway_mask(
        state,
        ((-0.8, 0.0, 0.0),),
        ((0.8, 0.0, 1.0),),
        radius_m=0.25,
    )

    assert mask is not None
    mid_x, mid_y = state.pct_xy_to_grid(np.array([0.0, 0.0]))
    outside_x, outside_y = state.pct_xy_to_grid(np.array([0.0, 0.8]))
    assert mask[mid_x, mid_y]
    assert not mask[outside_x, outside_y]


def test_gateway_mask_uses_stair_midpoint_polyline() -> None:
    traversability = np.ones((3, 30, 30), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((3, 30, 30), dtype=bool),
    )

    mask = _build_gateway_mask(
        state,
        ((-1.0, 0.0, 0.0),),
        ((1.0, 0.0, 2.0),),
        ((0.0, 1.0, 1.0),),
        radius_m=0.25,
    )

    assert mask is not None
    corner_x, corner_y = state.pct_xy_to_grid(np.array([0.0, 1.0]))
    shortcut_x, shortcut_y = state.pct_xy_to_grid(np.array([0.0, 0.0]))
    assert mask[corner_x, corner_y]
    assert not mask[shortcut_x, shortcut_y]


def test_stair_midpoints_are_ordered_by_ascent_height() -> None:
    unordered = (
        (2.89841, 7.79872, 2.61031),
        (2.94512, 9.14634, 1.64666),
        (1.51822, 6.27683, 0.29486),
        (1.9202, 9.52807, 1.71919),
    )

    ordered = _ordered_stair_midpoints(
        (1.5, 5.7, 0.6),
        (1.9, 8.0, 3.0),
        unordered,
    )

    assert ordered == (
        (1.51822, 6.27683, 0.29486),
        (1.9202, 9.52807, 1.71919),
        (2.94512, 9.14634, 1.64666),
        (2.89841, 7.79872, 2.61031),
    )


def test_stair_vertical_mask_is_narrower_than_gateway_corridor() -> None:
    traversability = np.ones((3, 20, 20), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((3, 20, 20), dtype=bool),
        cross_floor_gateways=((-0.8, 0.0, 0.0),),
        cross_floor_stair_exits=((0.8, 0.0, 1.0),),
        cross_floor_gateway_radius_m=0.6,
        stair_vertical_radius_m=0.25,
    )

    near_line = state.pct_xy_to_grid(np.array([0.0, 0.0]))
    rail_side = state.pct_xy_to_grid(np.array([0.0, 0.4]))

    assert state.cross_floor_gateway_mask is not None
    assert state.cross_floor_stair_vertical_mask is not None
    assert state.cross_floor_gateway_mask[rail_side]
    assert not state.cross_floor_stair_vertical_mask[1, rail_side[0], rail_side[1]]
    assert state.cross_floor_stair_vertical_mask[1, near_line[0], near_line[1]]


def test_stair_surface_mask_rejects_height_mismatched_handrail_side() -> None:
    traversability = np.ones((5, 40, 40), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((5, 40, 40), dtype=bool),
    )

    mask = _build_stair_surface_mask(
        state,
        ((-1.0, 0.0, 0.0),),
        ((1.0, 0.0, 2.0),),
        radius_m=0.25,
    )

    assert mask is not None
    matched_xy = state.pct_xy_to_grid(np.array([0.0, 0.0]))
    too_early_xy = state.pct_xy_to_grid(np.array([1.0, 0.0]))
    assert mask[2, matched_xy[0], matched_xy[1]]
    assert not mask[2, too_early_xy[0], too_early_xy[1]]


def test_stair_progress_rejects_slice_mismatched_ascent() -> None:
    traversability = np.ones((4, 20, 20), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((4, 20, 20), dtype=bool),
        cross_floor_gateways=((-0.6, 0.0, 0.0),),
        cross_floor_stair_exits=((0.6, 0.0, 1.5),),
        cross_floor_gateway_radius_m=0.6,
        stair_progress_tolerance=0.12,
    )
    state.cross_floor_stair_vertical_mask = None
    assert state.cross_floor_gateway_progress is not None

    current_xy = state.pct_xy_to_grid(np.array([-0.6, 0.0]))
    matched_xy = state.pct_xy_to_grid(np.array([-0.2, 0.0]))
    too_fast_xy = state.pct_xy_to_grid(np.array([0.6, 0.0]))
    current = (0, current_xy[0], current_xy[1])
    matched = (1, matched_xy[0], matched_xy[1])
    too_fast = (1, too_fast_xy[0], too_fast_xy[1])

    assert _stair_progress_allowed(
        state,
        current,
        matched,
        vertical_direction=1,
        stair_slice_range=(0, 3),
    )
    assert not _stair_progress_allowed(
        state,
        current,
        too_fast,
        vertical_direction=1,
        stair_slice_range=(0, 3),
    )


def test_stair_expected_progress_preserves_measured_waypoint_order() -> None:
    traversability = np.ones((7, 40, 40), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((7, 40, 40), dtype=bool),
        cross_floor_gateways=((1.5, 5.7, 0.6),),
        cross_floor_stair_midpoints=(
            (1.51822, 6.27683, 0.29486),
            (2.94512, 9.14634, 1.64666),
            (1.9202, 9.52807, 1.71919),
            (2.89841, 7.79872, 2.61031),
        ),
        cross_floor_stair_exits=((1.9, 8.0, 3.0),),
        cross_floor_gateway_radius_m=0.6,
    )

    expected = state.cross_floor_gateway_expected_progress_by_slice

    assert expected is not None
    assert 0.0 <= expected[state.z_to_slice(0.5)] < expected[state.z_to_slice(1.0)]
    assert expected[state.z_to_slice(1.5)] < expected[state.z_to_slice(2.0)]
    assert expected[state.z_to_slice(2.0)] < expected[state.z_to_slice(2.5)]


def test_stair_same_slice_progress_rejects_mid_stair_backtracking() -> None:
    traversability = np.ones((4, 20, 20), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((4, 20, 20), dtype=bool),
        cross_floor_gateways=((-0.6, 0.0, 0.0),),
        cross_floor_stair_exits=((0.6, 0.0, 1.5),),
        cross_floor_gateway_radius_m=0.6,
        stair_progress_tolerance=0.12,
    )
    state.cross_floor_stair_vertical_mask = None

    current_xy = state.pct_xy_to_grid(np.array([0.0, 0.0]))
    forward_xy = state.pct_xy_to_grid(np.array([0.2, 0.0]))
    backward_xy = state.pct_xy_to_grid(np.array([-0.4, 0.0]))
    current = (2, current_xy[0], current_xy[1])
    forward = (2, forward_xy[0], forward_xy[1])
    backward = (2, backward_xy[0], backward_xy[1])

    assert _stair_same_slice_progress_allowed(
        state,
        current,
        forward,
        vertical_direction=1,
        stair_slice_range=(0, 3),
    )
    assert not _stair_same_slice_progress_allowed(
        state,
        current,
        backward,
        vertical_direction=1,
        stair_slice_range=(0, 3),
    )


def test_request_uses_robot_root_to_floor_offset_for_slice_selection() -> None:
    traversability = np.ones((4, 9, 9), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((4, 9, 9), dtype=bool),
        robot_root_to_floor_m=0.45,
    )

    response = _handle_request(
        state,
        '{"start": [0.0, 0.0, 0.45], "end": [0.2, 0.0, 0.95]}',
    )

    assert response["status"] == "ok"
    assert response["slice_start"] == 0
    assert response["slice_end"] == 1
    assert response["snapped_start_slice"] == 0
    assert response["snapped_end_slice"] == 1
    assert response["snap_start_slice_delta"] == 0
    assert response["snap_end_slice_delta"] == 0
    assert response["snapped_start_xyz"][2] == 0.0
    assert response["snapped_end_xyz"][2] == 0.5
    assert response["planning_start_z"] == 0.0
    assert np.isclose(response["planning_end_z"], 0.5)


def test_typed_plan_request_matches_legacy_json_boundary() -> None:
    traversability = np.ones((2, 5, 5), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((2, 5, 5), dtype=bool),
        robot_root_to_floor_m=0.0,
    )

    typed = plan_request(
        state,
        start=(0.0, 0.0, 0.0),
        end=(0.2, 0.0, 0.0),
    )
    legacy = _handle_request(
        state,
        '{"start":[0.0,0.0,0.0],"end":[0.2,0.0,0.0]}',
    )

    assert typed == legacy
    assert typed["status"] == "ok"


def test_typed_plan_request_rejects_nonfinite_or_wrong_shape_input() -> None:
    traversability = np.ones((1, 3, 3), dtype=np.float32)
    tomogram = {
        "data": np.stack([traversability] * 5, axis=0),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((1, 3, 3), dtype=bool),
    )

    wrong_shape = plan_request(state, start=(0.0, 0.0), end=(0.0, 0.0, 0.0))
    nonfinite = plan_request(
        state,
        start=(0.0, 0.0, 0.0),
        end=(float("nan"), 0.0, 0.0),
    )

    assert wrong_shape["status"] == "error"
    assert "3 个坐标" in wrong_shape["msg"]
    assert nonfinite["status"] == "error"
    assert "NaN 或 Inf" in nonfinite["msg"]


def test_no_path_response_includes_hard_obstacle_diagnostics() -> None:
    traversability = np.ones((1, 5, 5), dtype=np.float32)
    tomogram = {
        "data": np.stack(
            [
                traversability,
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
            ],
            axis=0,
        ),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    hard_obstacles = np.zeros((5, 5), dtype=bool)
    hard_obstacles[2, :] = True
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((1, 5, 5), dtype=bool),
        hard_obstacle_mask=hard_obstacles,
        hard_obstacle_min_slices=7,
    )

    response = _handle_request(
        state,
        '{"start": [-0.2, 0.0, 0.0], "end": [0.2, 0.0, 0.0]}',
    )

    assert response["status"] == "no_path"
    assert response["hard_obstacle_cells"] == 5
    assert response["hard_obstacle_min_slices"] == 7


def test_cross_floor_request_uses_cross_floor_hard_obstacle_mask() -> None:
    traversability = np.ones((3, 5, 5), dtype=np.float32)
    tomogram = {
        "data": np.stack(
            [
                traversability,
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
            ],
            axis=0,
        ),
        "resolution": 0.2,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    same_floor_obstacles = np.zeros((5, 5), dtype=bool)
    same_floor_obstacles[2, :] = True
    cross_floor_obstacles = np.zeros((5, 5), dtype=bool)
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((3, 5, 5), dtype=bool),
        hard_obstacle_mask=same_floor_obstacles,
        hard_obstacle_min_slices=7,
        cross_floor_hard_obstacle_mask=cross_floor_obstacles,
        cross_floor_hard_obstacle_min_slices=9,
        robot_root_to_floor_m=0.0,
    )

    response = _handle_request(
        state,
        '{"start": [-0.2, 0.0, 0.0], "end": [0.2, 0.0, 1.0]}',
    )

    assert response["status"] == "ok"
    assert response["cross_floor"] is True
    assert response["hard_obstacle_cells"] == 0
    assert response["hard_obstacle_min_slices"] == 9
    assert response["default_hard_obstacle_min_slices"] == 7
    assert response["cross_floor_hard_obstacle_min_slices"] == 9


def test_cross_floor_request_reports_gateway_constraints() -> None:
    traversability = np.ones((3, 7, 7), dtype=np.float32)
    tomogram = {
        "data": np.stack(
            [
                traversability,
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
            ],
            axis=0,
        ),
        "resolution": 1.0,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 1.0,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((3, 7, 7), dtype=bool),
        cross_floor_gateways=((2.0, 0.0, 0.0),),
        cross_floor_gateway_radius_m=1.5,
        robot_root_to_floor_m=0.0,
        stair_min_horizontal_per_slice_m=1.0,
        stair_max_horizontal_per_slice_m=1.5,
    )

    response = _handle_request(
        state,
        '{"start": [-1.0, 0.0, 0.0], "end": [-1.0, 1.0, 2.0]}',
    )

    assert response["status"] == "ok"
    assert response["cross_floor_gateway_count"] == 1
    assert response["cross_floor_gateway_cells"] > 1
    assert response["cross_floor_gateway_mode"] == "strict_monotonic"
    assert response["stair_constraint_mode"] == "corridor_2d"
    assert [2.0, 0.0, 0.0] in response["traj"]
    z_values = [point[2] for point in response["traj"]]
    assert z_values == sorted(z_values)


def test_cross_floor_gateway_does_not_relax_when_strict_route_is_missing() -> None:
    traversability = np.ones((3, 7, 7), dtype=np.float32)
    tomogram = {
        "data": np.stack(
            [
                traversability,
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
                np.zeros_like(traversability),
            ],
            axis=0,
        ),
        "resolution": 1.0,
        "center": np.array([0.0, 0.0]),
        "slice_h0": 0.0,
        "slice_dh": 1.0,
    }
    state = _State(
        tomogram=tomogram,
        walkable=np.ones((3, 7, 7), dtype=bool),
        cross_floor_gateways=((2.0, 0.0, 0.0),),
        cross_floor_gateway_radius_m=0.1,
    )

    response = _handle_request(
        state,
        '{"start": [-1.0, 0.0, 0.0], "end": [-1.0, 1.0, 2.0]}',
    )

    assert response["status"] == "no_path"
    assert response["cross_floor_gateway_mode"] == "no_gateway_path"
