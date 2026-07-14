#!/usr/bin/env python3

"""从 multifloor collision PLY 生成本地 PCT tomogram 和 walkable map。"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes

try:
    import open3d as o3d
except ModuleNotFoundError:  # 当前 Isaac 环境允许使用确定性 NumPy fallback。
    o3d = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from source.scene.placement_support import load_binary_triangle_ply  # noqa: E402

DEFAULT_COLLISION_PLY = PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_collision.ply"
DEFAULT_TOMOGRAM = PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle"
DEFAULT_WALKABLE = PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/pct_multifloor_asset_build_report.json"


def main() -> int:
    args = _parse_args()
    collision_ply = _project_path(args.collision_ply)
    output_tomogram = _project_path(args.output_tomogram)
    output_walkable = _project_path(args.output_walkable)
    report_output = _project_path(args.report_output)

    if not collision_ply.is_file():
        raise FileNotFoundError(f"collision PLY 不存在: {collision_ply}")
    if collision_ply.is_symlink():
        raise RuntimeError(f"collision PLY 不能是软链接: {collision_ply}")

    t0 = time.time()
    points, sampling_backend = _load_collision_points(
        collision_ply,
        sample_points=int(args.sample_points),
        random_seed=int(args.random_seed),
    )
    points = _filter_points(points, z_min=args.z_min, z_max=args.z_max)
    grid = _make_grid(points, resolution=float(args.resolution), slice_dh=float(args.slice_dh), padding=float(args.padding))
    has_surface = _voxelize(points, grid)
    walkable = _compute_walkable(
        has_surface,
        dilation_radius=int(args.dilation_radius),
        max_wall_slices=int(args.max_wall_slices),
    )
    tomogram = _build_tomogram(grid, walkable)

    output_tomogram.parent.mkdir(parents=True, exist_ok=True)
    output_walkable.parent.mkdir(parents=True, exist_ok=True)
    with output_tomogram.open("wb") as stream:
        pickle.dump(tomogram, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(output_walkable, walkable)

    report = {
        "collision_ply": str(collision_ply),
        "output_tomogram": str(output_tomogram),
        "output_walkable": str(output_walkable),
        "sample_points": int(args.sample_points),
        "sampling_backend": sampling_backend,
        "random_seed": int(args.random_seed),
        "point_count_after_filter": int(len(points)),
        "resolution": float(args.resolution),
        "slice_dh": float(args.slice_dh),
        "center": grid.center.tolist(),
        "slice_h0": float(grid.slice_h0),
        "shape": list(walkable.shape),
        "walkable_count": int(walkable.sum()),
        "build_seconds": round(time.time() - t0, 3),
        "tomogram_size_bytes": int(output_tomogram.stat().st_size),
        "walkable_size_bytes": int(output_walkable.stat().st_size),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


class _Grid:
    """保存 PCT grid 坐标参数。"""

    def __init__(
        self,
        *,
        resolution: float,
        slice_dh: float,
        center: np.ndarray,
        dimx: int,
        dimy: int,
        n_slice: int,
        slice_h0: float,
    ) -> None:
        self.resolution = float(resolution)
        self.slice_dh = float(slice_dh)
        self.center = np.asarray(center, dtype=np.float64)
        self.dimx = int(dimx)
        self.dimy = int(dimy)
        self.n_slice = int(n_slice)
        self.slice_h0 = float(slice_h0)
        self.offset = np.array([self.dimx // 2, self.dimy // 2], dtype=np.int32)


def _load_collision_points(
    path: Path,
    *,
    sample_points: int,
    random_seed: int,
) -> tuple[np.ndarray, str]:
    """优先使用 Open3D；缺失时对 binary triangle PLY 做确定性面积采样。"""

    if o3d is not None:
        mesh = o3d.io.read_triangle_mesh(os.fspath(path))
        if mesh.is_empty():
            raise RuntimeError(f"collision mesh 为空: {path}")
        if sample_points <= 0:
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            if len(vertices) == 0:
                raise RuntimeError("collision mesh 没有 vertex")
            return vertices, "open3d_vertices"
        sampled = mesh.sample_points_uniformly(number_of_points=sample_points)
        points = np.asarray(sampled.points, dtype=np.float32)
        if len(points) == 0:
            raise RuntimeError("collision mesh 采样后没有点")
        return points, "open3d_uniform_surface"

    vertices, face_indices = load_binary_triangle_ply(path)
    vertices_f32 = np.asarray(vertices, dtype=np.float32)
    if len(vertices_f32) == 0:
        raise RuntimeError("collision mesh 没有 vertex")
    if sample_points <= 0:
        return vertices_f32, "numpy_binary_ply_vertices"

    triangles = vertices_f32[face_indices]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(first_edges, second_edges), axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 1.0e-12)
    if not bool(np.any(valid)):
        raise RuntimeError("collision mesh 没有有效三角面")
    valid_triangles = triangles[valid]
    probabilities = areas[valid].astype(np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(int(random_seed))
    selected = rng.choice(
        len(valid_triangles),
        size=int(sample_points),
        replace=True,
        p=probabilities,
    )
    sampled_triangles = valid_triangles[selected]
    first_random = np.sqrt(rng.random(int(sample_points), dtype=np.float32))
    second_random = rng.random(int(sample_points), dtype=np.float32)
    weight_a = 1.0 - first_random
    weight_b = first_random * (1.0 - second_random)
    weight_c = first_random * second_random
    points = (
        weight_a[:, None] * sampled_triangles[:, 0]
        + weight_b[:, None] * sampled_triangles[:, 1]
        + weight_c[:, None] * sampled_triangles[:, 2]
    ).astype(np.float32)
    return points, "numpy_binary_ply_area_sampling"


def _filter_points(points: np.ndarray, *, z_min: float | None, z_max: float | None) -> np.ndarray:
    mask = np.isfinite(points).all(axis=1)
    if z_min is not None:
        mask &= points[:, 2] >= float(z_min)
    if z_max is not None:
        mask &= points[:, 2] <= float(z_max)
    filtered = points[mask]
    if len(filtered) == 0:
        raise RuntimeError("过滤后没有点，检查 z_min / z_max")
    return filtered


def _make_grid(points: np.ndarray, *, resolution: float, slice_dh: float, padding: float) -> _Grid:
    if resolution <= 0.0 or slice_dh <= 0.0:
        raise ValueError("resolution 和 slice_dh 必须为正数")
    p_min = points.min(axis=0).astype(np.float64)
    p_max = points.max(axis=0).astype(np.float64)
    p_min[:2] -= float(padding)
    p_max[:2] += float(padding)
    ground_h = math.floor((float(p_min[2]) - float(slice_dh)) / float(slice_dh)) * float(slice_dh)
    top_h = math.ceil((float(p_max[2]) + float(slice_dh)) / float(slice_dh)) * float(slice_dh)
    center = (p_min[:2] + p_max[:2]) / 2.0
    dimx = int(math.ceil((p_max[0] - p_min[0]) / resolution)) + 4
    dimy = int(math.ceil((p_max[1] - p_min[1]) / resolution)) + 4
    n_slice = int(math.ceil((top_h - ground_h) / slice_dh))
    return _Grid(
        resolution=resolution,
        slice_dh=slice_dh,
        center=center,
        dimx=dimx,
        dimy=dimy,
        n_slice=n_slice,
        slice_h0=ground_h + slice_dh,
    )


def _voxelize(points: np.ndarray, grid: _Grid) -> np.ndarray:
    xi = np.round((points[:, 0] - grid.center[0]) / grid.resolution).astype(np.int32) + grid.offset[0]
    yj = np.round((points[:, 1] - grid.center[1]) / grid.resolution).astype(np.int32) + grid.offset[1]
    si = np.round((points[:, 2] - grid.slice_h0) / grid.slice_dh).astype(np.int32)
    valid = (xi >= 0) & (xi < grid.dimx) & (yj >= 0) & (yj < grid.dimy) & (si >= 0) & (si < grid.n_slice)
    has_surface = np.zeros((grid.n_slice, grid.dimx, grid.dimy), dtype=bool)
    has_surface[si[valid], xi[valid], yj[valid]] = True
    return has_surface


def _compute_walkable(has_surface: np.ndarray, *, dilation_radius: int, max_wall_slices: int) -> np.ndarray:
    if dilation_radius < 0:
        raise ValueError("dilation_radius 不能为负数")
    if max_wall_slices <= 0:
        raise ValueError("max_wall_slices 必须为正数")
    slice_count = has_surface.sum(axis=0)
    floor_like = slice_count <= max_wall_slices
    floor_surface = has_surface & floor_like[np.newaxis, :, :]

    kernel_width = dilation_radius * 2 + 1
    struct2d = np.ones((kernel_width, kernel_width), dtype=bool)
    expanded = np.zeros_like(floor_surface)
    for si in range(floor_surface.shape[0]):
        expanded[si] = binary_dilation(floor_surface[si], struct2d)

    walkable = np.zeros_like(expanded)
    for si in range(expanded.shape[0]):
        obstacle_above = has_surface[si + 1] if si + 1 < expanded.shape[0] else np.zeros_like(expanded[si])
        walkable[si] = expanded[si] & ~obstacle_above
        walkable[si] = binary_fill_holes(walkable[si])
    return walkable


def _build_tomogram(grid: _Grid, walkable: np.ndarray) -> dict[str, Any]:
    layers_t = np.full(walkable.shape, 50.0, dtype=np.float32)
    layers_t[walkable] = 1.0
    trav_grad_x = np.zeros_like(layers_t)
    trav_grad_y = np.zeros_like(layers_t)
    layers_g = np.full(walkable.shape, np.nan, dtype=np.float32)
    layers_c = np.full(walkable.shape, np.nan, dtype=np.float32)
    for si in range(walkable.shape[0]):
        z = grid.slice_h0 + si * grid.slice_dh
        layers_g[si][walkable[si]] = z
        layers_c[si][walkable[si]] = z + grid.slice_dh
    data = np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)).astype(np.float16)
    return {
        "data": data,
        "resolution": float(grid.resolution),
        "center": grid.center.astype(np.float64),
        "slice_h0": float(grid.slice_h0),
        "slice_dh": float(grid.slice_dh),
    }


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成本仓库 PCT-compatible 多楼层导航资产。")
    parser.add_argument("--collision-ply", default=os.fspath(DEFAULT_COLLISION_PLY), help="collision mesh PLY。")
    parser.add_argument("--output-tomogram", default=os.fspath(DEFAULT_TOMOGRAM), help="输出 tomogram pickle。")
    parser.add_argument("--output-walkable", default=os.fspath(DEFAULT_WALKABLE), help="输出 walkable npy。")
    parser.add_argument("--report-output", default=os.fspath(DEFAULT_REPORT), help="输出构建报告 JSON。")
    parser.add_argument("--resolution", type=float, default=0.20, help="XY grid 分辨率，单位米。")
    parser.add_argument("--slice-dh", type=float, default=0.50, help="tomogram slice 高度，单位米。")
    parser.add_argument("--padding", type=float, default=2.0, help="XY 边界外扩，单位米。")
    parser.add_argument("--sample-points", type=int, default=1_500_000, help="从 collision mesh 采样的点数。")
    parser.add_argument("--random-seed", type=int, default=0, help="NumPy PLY fallback 的确定性采样 seed。")
    parser.add_argument("--dilation-radius", type=int, default=1, help="walkable XY 膨胀半径，单位 grid cell。")
    parser.add_argument("--max-wall-slices", type=int, default=15, help="同一 XY 跨越超过该 slice 数时视为墙面。")
    parser.add_argument("--z-min", type=float, default=None, help="可选最低 z 过滤。")
    parser.add_argument("--z-max", type=float, default=None, help="可选最高 z 过滤。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
