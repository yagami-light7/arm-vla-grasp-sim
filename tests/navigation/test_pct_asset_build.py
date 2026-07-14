"""测试 PCT collision PLY 建图在缺少 Open3D 时仍可确定性采样。"""

from __future__ import annotations

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
