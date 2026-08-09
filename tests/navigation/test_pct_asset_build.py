"""测试 PCT collision PLY 建图在缺少 Open3D 时仍可确定性采样。"""

from __future__ import annotations

from hashlib import sha256
import json
import struct
from pathlib import Path

import numpy as np

from scripts.navigation import build_pct_multifloor_assets as asset_builder


def _write_triangle_ply(path: Path) -> None:
    """写入两个等面积三角面的 binary little-endian PLY。"""

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 6\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 2\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    vertices = (
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
    )
    faces = struct.pack("<BiiiBiii", 3, 0, 1, 2, 3, 3, 4, 5)
    path.write_bytes(header + struct.pack("<18f", *vertices) + faces)


def test_numpy_ply_sampling_fallback_is_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """同一 seed 必须生成相同点集，保证 tomogram 构建可追溯。"""

    collision_ply = tmp_path / "collision.ply"
    _write_triangle_ply(collision_ply)
    monkeypatch.setattr(asset_builder, "o3d", None)

    first, first_backend = asset_builder._load_collision_points(
        collision_ply,
        sample_points=1000,
        random_seed=17,
    )
    second, second_backend = asset_builder._load_collision_points(
        collision_ply,
        sample_points=1000,
        random_seed=17,
    )

    assert first_backend == "numpy_binary_ply_area_sampling"
    assert second_backend == first_backend
    assert first.shape == (1000, 3)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert bool(
        np.all(
            np.isclose(first[:, 2], 0.0, atol=1.0e-6)
            | np.isclose(first[:, 2], 1.0, atol=1.0e-6)
        )
    )
    assert bool(np.any(np.isclose(first[:, 2], 0.0, atol=1.0e-6)))
    assert bool(np.any(np.isclose(first[:, 2], 1.0, atol=1.0e-6)))


def test_numpy_ply_vertex_mode_does_not_require_open3d(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """sample_points=0 应直接读取顶点，便于低成本资产体检。"""

    collision_ply = tmp_path / "collision.ply"
    _write_triangle_ply(collision_ply)
    monkeypatch.setattr(asset_builder, "o3d", None)

    points, backend = asset_builder._load_collision_points(
        collision_ply,
        sample_points=0,
        random_seed=0,
    )

    assert backend == "numpy_binary_ply_vertices"
    assert points.shape == (6, 3)
    assert points.dtype == np.float32


def test_upstream_stair_profile_injects_typed_gateway_from_ground_anchors(
    tmp_path: Path,
) -> None:
    """楼梯先验只修改局部 cell，并生成官方 wrapper 可识别的向上 gateway。"""

    collision_ply = tmp_path / "collision.ply"
    _write_triangle_ply(collision_ply)
    data = np.zeros((5, 2, 12, 12), dtype=np.float16)
    data[0] = 50.0
    data[3, 0] = 0.0
    data[3, 1] = 1.0
    data[4] = data[3] + 2.0
    tomogram = {
        "data": data,
        "resolution": 0.1,
        "center": np.asarray((0.5, 0.5), dtype=np.float64),
        "slice_h0": -0.5,
        "slice_dh": 0.5,
        "simplified_slice_indices": (0, 1),
        "pct_scan_asset_kind": "upstream_official_semantics_v1",
    }
    profile = {
        "schema_version": 1,
        "source_branch": "synthetic-test",
        "asset_kind": "synthetic_stair_profile",
        "collision_ply_sha256": sha256(collision_ply.read_bytes()).hexdigest(),
        "base_tomogram_contract": {
            "data_sha256": sha256(data.tobytes()).hexdigest(),
            "shape": list(data.shape),
            "resolution": 0.1,
            "center": [0.5, 0.5],
            "slice_h0": -0.5,
            "slice_dh": 0.5,
            "simplified_slice_indices": [0, 1],
        },
        "coordinate_transform": {
            "coord_mode": "identity",
            "pct_offset_x": 0.0,
            "pct_offset_y": 0.0,
            "pct_offset_z": 0.0,
            "pct_scale_x": 1.0,
            "pct_scale_y": 1.0,
            "pct_scale_z": 1.0,
            "pct_rotation_x_rad": 0.0,
            "pct_rotation_y_rad": 0.0,
            "pct_rotation_z_rad": 0.0,
        },
        "sampling_spacing_m": 0.1,
        "corridor_radius_cells": 0,
        "corridor_cost": 1.0,
        "corridor_clearance_m": 2.0,
        "projection_max_hint_error_m": 0.6,
        "logical_layer_band": {
            "first_layer": 0,
            "last_layer": 1,
            "ground_z_origin_m": 0.0,
            "height_step_m": 1.0,
        },
        "anchors_sim_ground_xyz": [
            [0.20, 0.20, 0.0],
            [0.30, 0.20, 1.0],
        ],
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    report = asset_builder._apply_upstream_stair_profile(
        tomogram,
        collision_ply=collision_ply,
        profile_path=profile_path,
    )

    gateway_up, _ = asset_builder._upstream_gateway_counts(tomogram)
    assert tomogram["pct_scan_asset_kind"] == "synthetic_stair_profile"
    assert report["injected_gateway_count"] == 1
    assert report["sampled_point_count"] > 2
    assert report["changed_cost_cell_count"] > 0
    assert gateway_up >= 1
