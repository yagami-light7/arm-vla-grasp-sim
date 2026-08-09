from __future__ import annotations

import math
import os
from pathlib import Path
import struct
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pct_ros2_adapter.backend import (
    DirectPCTBackend,
    PCTBackendConfig,
    PCTBackendError,
    PCTNoPathError,
    _enforce_snap_distance,
    _sample_polyline,
    xyz_triples,
)
from source.navigation import pct_adapter as legacy_pct


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _write_plane_ply(path: Path) -> None:
    vertices = np.asarray(
        [
            (-1.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 2.0, 0.0),
            (-1.0, 2.0, 0.0),
        ],
        dtype="<f4",
    )
    faces = ((0, 1, 2), (0, 2, 3))
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    payload = bytearray(header)
    payload.extend(vertices.tobytes())
    for face in faces:
        payload.extend(struct.pack("<Biii", 3, *face))
    path.write_bytes(payload)


def _backend_config(tmp_path: Path, **overrides: Any) -> PCTBackendConfig:
    tomogram = tmp_path / "map.pickle"
    walkable = tmp_path / "walkable.npy"
    collision = tmp_path / "collision.ply"
    tomogram.write_bytes(b"typed-grid-test")
    walkable.write_bytes(b"typed-grid-test")
    _write_plane_ply(collision)
    values: dict[str, Any] = {
        "project_root": PROJECT_ROOT,
        "tomogram_path": tomogram,
        "walkable_path": walkable,
        "collision_ply_path": collision,
        "coord_mode": "identity",
        "cross_floor_gateway_points": (),
        "cross_floor_stair_exit_points": (),
        "cross_floor_stair_midpoint_points": (),
        "ground_projection_max_z_error_m": 0.60,
        "path_sample_spacing_m": 0.20,
        "maximum_snap_distance_m": 0.25,
        "grid_max_expansions": 123,
        "grid_compress_max_segment_m": 0.37,
    }
    values.update(overrides)
    return PCTBackendConfig(**values)


def _fake_grid(
    response: dict[str, Any] | object,
    observations: dict[str, Any],
) -> ModuleType:
    module = ModuleType("typed_fake_pct_grid")

    def load_state_from_environment() -> object:
        observations["load_environment"] = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("PCT_")
        }
        return object()

    def plan_request(
        state: object,
        *,
        start: Any,
        end: Any,
        cancel_check=None,
    ) -> dict[str, Any] | object:
        del state
        observations["start"] = tuple(float(value) for value in start)
        observations["end"] = tuple(float(value) for value in end)
        observations["plan_grid_max_expansions"] = os.environ.get(
            "PCT_GRID_MAX_EXPANSIONS"
        )
        observations["cancel_check"] = cancel_check
        return response

    module.load_state_from_environment = load_state_from_environment
    module.plan_request = plan_request
    return module


def _forbid_legacy_json_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class ForbiddenClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("DirectPCTBackend 不得实例化旧 JSON client")

    monkeypatch.setattr(legacy_pct, "PCTPlannerClient", ForbiddenClient)


def test_direct_backend_uses_typed_call_and_publishes_ground_height(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations: dict[str, Any] = {}
    response = {
        "status": "ok",
        "traj": [[0.1, 0.1, 0.5], [0.8, 0.1, 0.5]],
        "cross_floor": False,
        "snap_start_dist": 0,
        "snap_end_dist": 0,
        "slice_start": 0,
        "slice_end": 0,
        "snapped_start_slice": 0,
        "snapped_end_slice": 0,
        "snap_start_slice_delta": 0,
        "snap_end_slice_delta": 0,
        "snap_start_distance_m": 0.01,
        "snap_end_distance_m": 0.10,
    }
    monkeypatch.setenv("PCT_GRID_MAX_EXPANSIONS", "ambient-value")
    _forbid_legacy_json_client(monkeypatch)
    backend = DirectPCTBackend(
        _backend_config(tmp_path),
        planner_module=legacy_pct,
        grid_module=_fake_grid(response, observations),
    )

    assert observations["load_environment"]["PCT_GRID_MAX_EXPANSIONS"] == "123"
    assert (
        observations["load_environment"]["PCT_GRID_COMPRESS_MAX_SEGMENT_M"]
        == "0.37"
    )
    assert observations["load_environment"]["PCT_CROSS_FLOOR_GATEWAYS_PCT"] == ""
    assert os.environ["PCT_GRID_MAX_EXPANSIONS"] == "ambient-value"

    plan = backend.plan(
        start_base_xyz=(0.1, 0.1, 0.30),
        goal_base_xyz=(0.9, 0.1, 0.30),
        goal_yaw=0.4,
    )

    assert observations["start"] == pytest.approx((0.1, 0.1, 0.30))
    assert observations["end"] == pytest.approx((0.9, 0.1, 0.30))
    assert observations["plan_grid_max_expansions"] == "ambient-value"
    assert callable(observations["cancel_check"])
    assert plan.points_xyz[-1] == pytest.approx((0.9, 0.1, 0.0))
    assert all(point[2] == pytest.approx(0.0) for point in plan.points_xyz)
    assert max(
        math.dist(start, end)
        for start, end in zip(plan.points_xyz, plan.points_xyz[1:])
    ) <= 0.20 + 1.0e-9
    assert plan.metadata["path_3d"] == plan.points_xyz
    assert plan.metadata["coarse_slice_path_3d"][-1] == pytest.approx(
        (0.8, 0.1, 0.5)
    )
    assert plan.metadata["height_semantics"] == "ground_height"
    assert plan.metadata["slice_query_root_to_floor_m"] == pytest.approx(0.30)
    assert plan.metadata["goal_base_to_ground_m"] == pytest.approx(0.30)
    assert plan.metadata["ground_projection_max_hint_error_m"] == pytest.approx(
        0.5
    )
    maximum_index = plan.metadata["ground_projection_max_hint_error_index"]
    assert plan.metadata["ground_projection_max_hint_input_pct_xyz"][2] == (
        pytest.approx(0.5)
    )
    assert plan.metadata["ground_projection_max_hint_output_pct_xyz"] == (
        pytest.approx(plan.points_xyz[maximum_index])
    )
    assert plan.metadata["requested_goal_appended"] is True
    assert plan.metadata["transport"] == "direct_in_process_ros2"


def test_backend_propagates_complete_coordinate_transform(
    tmp_path: Path,
) -> None:
    coordinate_transform = {
        "pct_offset_x": 1.1,
        "pct_offset_y": -2.2,
        "pct_offset_z": 3.3,
        "pct_scale_x": 0.7,
        "pct_scale_y": -1.2,
        "pct_scale_z": 1.4,
        "pct_rotation_x_rad": 0.1,
        "pct_rotation_y_rad": -0.2,
        "pct_rotation_z_rad": 0.3,
    }
    backend = DirectPCTBackend(
        _backend_config(tmp_path, **coordinate_transform),
        planner_module=legacy_pct,
        grid_module=_fake_grid({"status": "no_path"}, {}),
    )

    for field_name, expected in coordinate_transform.items():
        assert getattr(backend.config, field_name) == pytest.approx(expected)
        assert getattr(backend._legacy_config, field_name) == pytest.approx(
            expected
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("pct_scale_x", 0.0),
        ("pct_scale_y", 0.0),
        ("pct_scale_z", 0.0),
        ("pct_offset_z", math.inf),
        ("pct_rotation_x_rad", math.nan),
        ("pct_rotation_y_rad", math.inf),
        ("pct_rotation_z_rad", -math.inf),
    ],
)
def test_backend_rejects_singular_or_nonfinite_coordinate_transform(
    tmp_path: Path,
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="PCT 坐标"):
        DirectPCTBackend(
            _backend_config(tmp_path, **{field_name: value}),
            planner_module=legacy_pct,
            grid_module=_fake_grid({"status": "no_path"}, {}),
        )


def test_exact_goal_xy_replaces_coarse_slice_height_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "status": "ok",
        "traj": [[0.1, 0.1, 0.5], [0.9, 0.1, 0.5]],
        "cross_floor": False,
        "snap_start_dist": 0,
        "snap_end_dist": 0,
        "slice_start": 0,
        "slice_end": 0,
        "snapped_start_slice": 0,
        "snapped_end_slice": 0,
        "snap_start_slice_delta": 0,
        "snap_end_slice_delta": 0,
        "snap_start_distance_m": 0.0,
        "snap_end_distance_m": 0.0,
    }
    _forbid_legacy_json_client(monkeypatch)
    backend = DirectPCTBackend(
        _backend_config(tmp_path),
        planner_module=legacy_pct,
        grid_module=_fake_grid(response, {}),
    )

    plan = backend.plan(
        start_base_xyz=(0.1, 0.1, 0.30),
        goal_base_xyz=(0.9, 0.1, 0.30),
        goal_yaw=0.0,
    )

    assert plan.points_xyz[-1] == pytest.approx((0.9, 0.1, 0.0))
    assert plan.metadata["requested_goal_appended"] is False


def test_ground_projection_failure_is_typed_and_keeps_point_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "status": "ok",
        "traj": [[0.1, 0.1, 0.5], [0.9, 0.1, 0.5]],
        "cross_floor": False,
        "snap_start_dist": 0,
        "snap_end_dist": 0,
        "slice_start": 0,
        "slice_end": 0,
        "snapped_start_slice": 0,
        "snapped_end_slice": 0,
        "snap_start_slice_delta": 0,
        "snap_end_slice_delta": 0,
        "snap_start_distance_m": 0.0,
        "snap_end_distance_m": 0.0,
    }
    _forbid_legacy_json_client(monkeypatch)
    backend = DirectPCTBackend(
        _backend_config(tmp_path, ground_projection_max_z_error_m=0.20),
        planner_module=legacy_pct,
        grid_module=_fake_grid(response, {}),
    )

    with pytest.raises(PCTBackendError, match=r"points_pct_xyz\[0\]"):
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.30),
            goal_base_xyz=(0.9, 0.1, 0.30),
            goal_yaw=0.0,
        )


@pytest.mark.parametrize(
    ("response", "exception_type", "message"),
    [
        ({"status": "no_path", "msg": "blocked"}, PCTNoPathError, "blocked"),
        ({"status": "error", "msg": "broken"}, PCTBackendError, "broken"),
        ([], PCTBackendError, "\u975e\u5bf9\u8c61"),
    ],
)
def test_direct_backend_preserves_typed_grid_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: object,
    exception_type: type[Exception],
    message: str,
) -> None:
    _forbid_legacy_json_client(monkeypatch)
    backend = DirectPCTBackend(
        _backend_config(tmp_path),
        planner_module=legacy_pct,
        grid_module=_fake_grid(response, {}),
    )

    with pytest.raises(exception_type, match=message) as captured:
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.3),
            goal_base_xyz=(0.9, 0.1, 0.3),
            goal_yaw=0.0,
        )
    assert captured.value.__cause__ is not captured.value


def test_snap_gate_requires_both_metric_distances() -> None:
    with pytest.raises(PCTBackendError, match="snap_end_distance_m"):
        _enforce_snap_distance(
            {"snap_start_distance_m": 0.0},
            maximum_snap_distance_m=0.25,
        )
    with pytest.raises(PCTNoPathError, match="0.251"):
        _enforce_snap_distance(
            {
                "snap_start_distance_m": 0.0,
                "snap_end_distance_m": 0.251,
            },
            maximum_snap_distance_m=0.25,
        )


def test_endpoint_height_gates_reject_inconsistent_start_or_goal_z(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "status": "ok",
        "traj": [[0.1, 0.1, 0.0], [0.9, 0.1, 0.0]],
        "cross_floor": False,
        "snap_start_dist": 0,
        "snap_end_dist": 0,
        "slice_start": 0,
        "slice_end": 0,
        "snapped_start_slice": 0,
        "snapped_end_slice": 0,
        "snap_start_slice_delta": 0,
        "snap_end_slice_delta": 0,
        "snap_start_distance_m": 0.0,
        "snap_end_distance_m": 0.0,
    }
    backend = DirectPCTBackend(
        _backend_config(tmp_path),
        planner_module=legacy_pct,
        grid_module=_fake_grid(response, {}),
    )

    with pytest.raises(PCTBackendError, match="start z"):
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.50),
            goal_base_xyz=(0.9, 0.1, 0.30),
            goal_yaw=0.0,
        )
    with pytest.raises(PCTBackendError, match="goal z"):
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.30),
            goal_base_xyz=(0.9, 0.1, 0.50),
            goal_yaw=0.0,
        )


def test_exact_goal_is_rejected_when_requested_grid_cell_was_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "status": "ok",
        "traj": [[0.1, 0.1, 0.0], [0.8, 0.1, 0.0]],
        "cross_floor": False,
        "snap_start_dist": 0,
        "snap_end_dist": 1,
        "slice_start": 0,
        "slice_end": 0,
        "snapped_start_slice": 0,
        "snapped_end_slice": 0,
        "snap_start_slice_delta": 0,
        "snap_end_slice_delta": 0,
        "snap_start_distance_m": 0.0,
        "snap_end_distance_m": 0.10,
    }
    backend = DirectPCTBackend(
        _backend_config(tmp_path),
        planner_module=legacy_pct,
        grid_module=_fake_grid(response, {}),
    )

    with pytest.raises(PCTNoPathError, match="目标所在栅格不可走"):
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.30),
            goal_base_xyz=(0.9, 0.1, 0.30),
            goal_yaw=0.0,
        )


def test_backend_rejects_cross_floor_snap_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "status": "ok",
        "traj": [[0.1, 0.1, 0.0], [0.9, 0.1, 0.0]],
        "cross_floor": False,
        "snap_start_dist": 0,
        "snap_end_dist": 0,
        "slice_start": 0,
        "slice_end": 0,
        "snapped_start_slice": 0,
        "snapped_end_slice": 1,
        "snap_start_slice_delta": 0,
        "snap_end_slice_delta": 1,
        "snap_start_distance_m": 0.0,
        "snap_end_distance_m": 0.0,
    }
    backend = DirectPCTBackend(
        _backend_config(tmp_path),
        planner_module=legacy_pct,
        grid_module=_fake_grid(response, {}),
    )

    with pytest.raises(PCTNoPathError, match="snap 跨层"):
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.30),
            goal_base_xyz=(0.9, 0.1, 0.30),
            goal_yaw=0.0,
        )


def test_polyline_resampling_preserves_vertices_and_bounds_segments() -> None:
    sampled = _sample_polyline(
        ((0.0, 0.0, 0.0), (0.45, 0.0, 0.0), (0.45, 0.3, 0.0)),
        spacing_m=0.20,
    )

    assert sampled[0] == (0.0, 0.0, 0.0)
    assert (0.45, 0.0, 0.0) in sampled
    assert sampled[-1] == (0.45, 0.3, 0.0)
    assert max(
        math.dist(start, end)
        for start, end in zip(sampled, sampled[1:])
    ) <= 0.20 + 1.0e-12


def test_xyz_triples_rejects_partial_or_nonfinite_values() -> None:
    assert xyz_triples((1, 2, 3, 4, 5, 6), field_name="points") == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    )
    with pytest.raises(ValueError, match="\u4e09\u5143\u7ec4"):
        xyz_triples((1, 2), field_name="points")
    with pytest.raises(ValueError, match="NaN"):
        xyz_triples((1, 2, math.nan), field_name="points")
