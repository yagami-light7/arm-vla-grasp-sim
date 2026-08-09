#!/usr/bin/env python3

"""从 multifloor collision PLY 生成本地 PCT tomogram 和 walkable map。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, convolve

try:
    import open3d as o3d
except ModuleNotFoundError:  # 当前 Isaac 环境允许使用确定性 NumPy fallback。
    o3d = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))
PCT_ROS2_SOURCE = PROJECT_ROOT / "ros2_ws/src/pct_ros2_adapter"
if os.fspath(PCT_ROS2_SOURCE) not in sys.path:
    sys.path.insert(0, os.fspath(PCT_ROS2_SOURCE))

from pct_ros2_adapter.ground_surface import TriangleGroundProjector  # noqa: E402
from source.navigation.pct_adapter import sim_to_pct_xyz  # noqa: E402
from source.scene.placement_support import load_binary_triangle_ply  # noqa: E402

DEFAULT_COLLISION_PLY = PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_collision.ply"
DEFAULT_TOMOGRAM = PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle"
DEFAULT_UPSTREAM_TOMOGRAM = (
    PROJECT_ROOT / "source/scene/multifloor/mutifloor_upstream.pickle"
)
DEFAULT_WALKABLE = PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/pct_multifloor_asset_build_report.json"
DEFAULT_STAIR_PROFILE = (
    PROJECT_ROOT / "configs/navigation/pct_multifloor_stair_profile.json"
)


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
        backend=str(args.sampling_backend),
    )
    points = _filter_points(points, z_min=args.z_min, z_max=args.z_max)
    grid = _make_grid(points, resolution=float(args.resolution), slice_dh=float(args.slice_dh), padding=float(args.padding))
    has_surface = _voxelize(points, grid)
    walkable = _compute_walkable(
        has_surface,
        dilation_radius=int(args.dilation_radius),
        max_wall_slices=int(args.max_wall_slices),
    )
    if args.tomogram_kind == "upstream":
        tomogram = _build_upstream_tomogram(
            grid,
            points,
            kernel_size=int(args.traversability_kernel_size),
            interval_min=float(args.interval_min),
            interval_free=float(args.interval_free),
            slope_max_rad=float(args.slope_max_rad),
            step_max_m=float(args.step_max_m),
            standable_ratio=float(args.standable_ratio),
            cost_barrier=float(args.cost_barrier),
            safe_margin_m=float(args.safe_margin_m),
            inflation_m=float(args.inflation_m),
        )
        stair_profile_path = _optional_project_path(
            args.stair_corridor_profile
        )
        stair_profile_report = (
            None
            if stair_profile_path is None
            else _apply_upstream_stair_profile(
                tomogram,
                collision_ply=collision_ply,
                profile_path=stair_profile_path,
            )
        )
    else:
        tomogram = _build_tomogram(grid, walkable)
        stair_profile_report = None

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
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "tomogram_kind": str(args.tomogram_kind),
        "traversability_parameters": {
            "kernel_size": int(args.traversability_kernel_size),
            "interval_min": float(args.interval_min),
            "interval_free": float(args.interval_free),
            "slope_max_rad": float(args.slope_max_rad),
            "step_max_m": float(args.step_max_m),
            "standable_ratio": float(args.standable_ratio),
            "cost_barrier": float(args.cost_barrier),
            "safe_margin_m": float(args.safe_margin_m),
            "inflation_m": float(args.inflation_m),
        },
        "stair_profile": stair_profile_report,
        "random_seed": int(args.random_seed),
        "point_count_after_filter": int(len(points)),
        "resolution": float(args.resolution),
        "slice_dh": float(args.slice_dh),
        "center": grid.center.tolist(),
        "slice_h0": float(grid.slice_h0),
        "shape": list(walkable.shape),
        "walkable_count": int(walkable.sum()),
        "tomogram_layer_count": int(np.asarray(tomogram["data"]).shape[1]),
        "tomogram_gateway_up_count": int(
            _upstream_gateway_counts(tomogram)[0]
        ),
        "tomogram_gateway_down_count": int(
            _upstream_gateway_counts(tomogram)[1]
        ),
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
    backend: str = "auto",
) -> tuple[np.ndarray, str]:
    """按显式后端采样；正式资产默认使用可复现的 NumPy 面积采样。"""

    backend_name = str(backend).strip().lower()
    if backend_name not in {"auto", "numpy", "open3d"}:
        raise ValueError("sampling_backend 只允许 auto、numpy 或 open3d")
    use_open3d = backend_name == "open3d" or (
        backend_name == "auto" and o3d is not None
    )
    if use_open3d:
        if o3d is None:
            raise RuntimeError("显式请求 Open3D 采样，但当前环境未安装 open3d")
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


def _build_upstream_tomogram(
    grid: _Grid,
    points: np.ndarray,
    *,
    kernel_size: int = 7,
    interval_min: float = 0.50,
    interval_free: float = 0.65,
    slope_max_rad: float = 0.40,
    step_max_m: float = 0.17,
    standable_ratio: float = 0.20,
    cost_barrier: float = 50.0,
    safe_margin_m: float = 0.40,
    inflation_m: float = 0.20,
) -> dict[str, Any]:
    """在 CPU 上复现官方 tomography 的五通道与 layer simplification。"""

    _validate_upstream_tomography_parameters(
        grid,
        kernel_size=kernel_size,
        interval_min=interval_min,
        interval_free=interval_free,
        slope_max_rad=slope_max_rad,
        step_max_m=step_max_m,
        standable_ratio=standable_ratio,
        cost_barrier=cost_barrier,
        safe_margin_m=safe_margin_m,
        inflation_m=inflation_m,
    )
    ground, ceiling = _official_ground_and_ceiling_layers(points, grid)
    ground_internal = np.nan_to_num(
        ground,
        nan=-1.0e6,
        posinf=-1.0e6,
        neginf=-1.0e6,
    ).astype(np.float32, copy=False)
    ceiling_internal = np.nan_to_num(
        ceiling,
        nan=1.0e6,
        posinf=1.0e6,
        neginf=1.0e6,
    ).astype(np.float32, copy=False)
    traversability = _official_traversability_cost(
        ground_internal,
        ceiling_internal,
        resolution=grid.resolution,
        kernel_size=kernel_size,
        interval_min=interval_min,
        interval_free=interval_free,
        slope_max_rad=slope_max_rad,
        step_max_m=step_max_m,
        standable_ratio=standable_ratio,
        cost_barrier=cost_barrier,
    )
    inflated = _official_inflation(
        traversability,
        resolution=grid.resolution,
        safe_margin_m=safe_margin_m,
        inflation_m=inflation_m,
    )
    simplified_indices = _official_simplified_layer_indices(
        ground_internal,
        inflated,
        cost_barrier=cost_barrier,
    )
    simplified_index_array = np.asarray(simplified_indices, dtype=np.intp)
    layers_t = inflated[simplified_index_array]
    layers_g = ground_internal[simplified_index_array]
    layers_c = ceiling_internal[simplified_index_array]
    trav_grad_x = np.zeros_like(layers_g)
    trav_grad_y = np.zeros_like(layers_g)
    trav_grad_x[:, 1:-1, :] = (
        layers_t[:, 2:, :] - layers_t[:, :-2, :]
    )
    trav_grad_y[:, :, 1:-1] = (
        layers_t[:, :, 2:] - layers_t[:, :, :-2]
    )
    layers_g = np.where(layers_g > -1.0e6, layers_g, np.nan)
    layers_c = np.where(layers_c < 1.0e6, layers_c, np.nan)
    data = np.stack(
        (layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)
    ).astype(np.float16)
    return {
        "data": data,
        "resolution": float(grid.resolution),
        "center": grid.center.astype(np.float64),
        # 官方 export 保留初始 slice 元数据；简化后的 layer 不再等距。
        "slice_h0": float(grid.slice_h0),
        "slice_dh": float(grid.slice_dh),
        "pct_scan_asset_kind": "upstream_official_semantics_v1",
        "simplified_slice_indices": tuple(int(i) for i in simplified_indices),
    }


def _apply_upstream_stair_profile(
    tomogram: dict[str, Any],
    *,
    collision_ply: Path,
    profile_path: Path,
) -> dict[str, Any]:
    """把 pct-scene 的楼梯中心线写成官方 A* 可识别的局部 gateway。"""

    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes.decode("utf-8"))
    if not isinstance(profile, dict) or int(profile.get("schema_version", 0)) != 1:
        raise ValueError("楼梯 profile 必须是 schema_version=1 的 JSON 对象")
    if sha256(collision_ply.read_bytes()).hexdigest() != str(
        profile.get("collision_ply_sha256", "")
    ):
        raise ValueError("楼梯 profile 与当前 collision PLY 哈希不一致")

    base_data = np.asarray(tomogram["data"])
    contract = profile.get("base_tomogram_contract")
    if not isinstance(contract, dict):
        raise ValueError("楼梯 profile 缺少 base_tomogram_contract")
    base_data_hash = sha256(base_data.tobytes()).hexdigest()
    if base_data_hash != str(contract.get("data_sha256", "")):
        raise ValueError(
            "楼梯 profile 只能应用到固定参数生成的官方 tomogram："
            f"data_sha256={base_data_hash}"
        )
    expected_shape = tuple(int(value) for value in contract.get("shape", ()))
    if base_data.shape != expected_shape:
        raise ValueError(
            f"楼梯 profile tomogram shape 不匹配：{base_data.shape} != "
            f"{expected_shape}"
        )
    _require_close(
        float(tomogram["resolution"]),
        float(contract["resolution"]),
        field_name="resolution",
    )
    _require_close(
        float(tomogram["slice_h0"]),
        float(contract["slice_h0"]),
        field_name="slice_h0",
    )
    _require_close(
        float(tomogram["slice_dh"]),
        float(contract["slice_dh"]),
        field_name="slice_dh",
    )
    expected_center = np.asarray(contract.get("center"), dtype=np.float64)
    actual_center = np.asarray(tomogram["center"], dtype=np.float64)
    if expected_center.shape != (2,) or not np.allclose(
        actual_center,
        expected_center,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("楼梯 profile tomogram center 不匹配")
    expected_indices = tuple(
        int(value) for value in contract.get("simplified_slice_indices", ())
    )
    actual_indices = tuple(
        int(value) for value in tomogram.get("simplified_slice_indices", ())
    )
    if actual_indices != expected_indices:
        raise ValueError("楼梯 profile simplified layer 索引不匹配")

    anchors_raw = profile.get("anchors_sim_ground_xyz")
    if not isinstance(anchors_raw, list) or len(anchors_raw) < 2:
        raise ValueError("楼梯 profile 至少需要两个 Sim ground anchor")
    anchors = tuple(
        _finite_profile_xyz(value, field_name=f"anchors[{index}]")
        for index, value in enumerate(anchors_raw)
    )
    spacing_m = _positive_profile_number(
        profile,
        "sampling_spacing_m",
    )
    sampled_sim = _sample_profile_polyline(anchors, spacing_m=spacing_m)

    transform_raw = profile.get("coordinate_transform")
    if not isinstance(transform_raw, dict):
        raise ValueError("楼梯 profile 缺少 coordinate_transform")
    coordinate_arguments: dict[str, float | str] = {
        "coord_mode": str(transform_raw.get("coord_mode", "")),
    }
    for name in (
        "pct_offset_x",
        "pct_offset_y",
        "pct_offset_z",
        "pct_scale_x",
        "pct_scale_y",
        "pct_scale_z",
        "pct_rotation_x_rad",
        "pct_rotation_y_rad",
        "pct_rotation_z_rad",
    ):
        coordinate_arguments[name] = float(transform_raw[name])

    maximum_projection_error_m = _positive_profile_number(
        profile,
        "projection_max_hint_error_m",
    )
    vertices, faces = load_binary_triangle_ply(collision_ply)
    projector = TriangleGroundProjector(
        vertices,
        faces,
        maximum_hint_error_m=maximum_projection_error_m,
    )
    projected_pct: list[tuple[float, float, float]] = []
    projection_errors: list[float] = []
    for point_sim in sampled_sim:
        point_pct = sim_to_pct_xyz(point_sim, **coordinate_arguments)
        projection = projector.project(
            x=point_pct[0],
            y=point_pct[1],
            z_hint=point_pct[2],
        )
        projected_pct.append((point_pct[0], point_pct[1], projection.z))
        projection_errors.append(projection.hint_error_m)

    band = profile.get("logical_layer_band")
    if not isinstance(band, dict):
        raise ValueError("楼梯 profile 缺少 logical_layer_band")
    first_layer = int(band["first_layer"])
    last_layer = int(band["last_layer"])
    ground_z_origin_m = float(band["ground_z_origin_m"])
    height_step_m = float(band["height_step_m"])
    if (
        first_layer < 0
        or last_layer < first_layer
        or last_layer >= base_data.shape[1]
        or not math.isfinite(ground_z_origin_m)
        or not math.isfinite(height_step_m)
        or height_step_m <= 0.0
    ):
        raise ValueError("楼梯 profile logical_layer_band 非法")

    radius_cells = int(profile["corridor_radius_cells"])
    if radius_cells < 0:
        raise ValueError("楼梯 profile corridor_radius_cells 不能为负数")
    corridor_cost = _positive_profile_number(profile, "corridor_cost")
    clearance_m = _positive_profile_number(
        profile,
        "corridor_clearance_m",
    )
    resolution = float(tomogram["resolution"])
    dim_x, dim_y = base_data.shape[2:]
    working = base_data.astype(np.float32, copy=True)
    stamped_cells: set[tuple[int, int, int]] = set()
    projected_records: list[tuple[int, int, int, float]] = []
    for point_index, point_pct in enumerate(projected_pct):
        layer = _stair_profile_layer(
            point_pct[2],
            first_layer=first_layer,
            last_layer=last_layer,
            ground_z_origin_m=ground_z_origin_m,
            height_step_m=height_step_m,
        )
        grid_x = int(
            np.rint((point_pct[0] - actual_center[0]) / resolution)
        ) + dim_x // 2
        grid_y = int(
            np.rint((point_pct[1] - actual_center[1]) / resolution)
        ) + dim_y // 2
        if (
            grid_x - radius_cells < 0
            or grid_x + radius_cells >= dim_x
            or grid_y - radius_cells < 0
            or grid_y + radius_cells >= dim_y
        ):
            raise ValueError(
                f"楼梯 profile 点 {point_index} 的 corridor 超出 tomogram"
            )
        x_slice = slice(grid_x - radius_cells, grid_x + radius_cells + 1)
        y_slice = slice(grid_y - radius_cells, grid_y + radius_cells + 1)
        working[0, layer, x_slice, y_slice] = corridor_cost
        working[3, layer, x_slice, y_slice] = point_pct[2]
        working[4, layer, x_slice, y_slice] = point_pct[2] + clearance_m
        for stamp_x in range(grid_x - radius_cells, grid_x + radius_cells + 1):
            for stamp_y in range(grid_y - radius_cells, grid_y + radius_cells + 1):
                stamped_cells.add((layer, stamp_x, stamp_y))
        projected_records.append((layer, grid_x, grid_y, point_pct[2]))

    gateway_cells: list[tuple[int, int, int]] = []
    for upper_layer in range(first_layer + 1, last_layer + 1):
        try:
            _, grid_x, grid_y, ground_z = next(
                record
                for record in projected_records
                if record[0] == upper_layer
            )
        except StopIteration as exc:
            raise ValueError(
                f"楼梯 profile 没有覆盖 logical layer {upper_layer}"
            ) from exc
        lower_layer = upper_layer - 1
        working[0, lower_layer, grid_x, grid_y] = 50.0
        working[0, upper_layer, grid_x, grid_y] = corridor_cost
        working[3, lower_layer : upper_layer + 1, grid_x, grid_y] = ground_z
        working[4, lower_layer : upper_layer + 1, grid_x, grid_y] = (
            ground_z + clearance_m
        )
        gateway_cells.append((lower_layer, grid_x, grid_y))

    working[1] = 0.0
    working[2] = 0.0
    working[1, :, 1:-1, :] = (
        working[0, :, 2:, :] - working[0, :, :-2, :]
    )
    working[2, :, :, 1:-1] = (
        working[0, :, :, 2:] - working[0, :, :, :-2]
    )
    tomogram["data"] = working.astype(np.float16)
    tomogram["pct_scan_asset_kind"] = str(profile["asset_kind"])
    tomogram["stair_profile_schema_version"] = 1
    tomogram["stair_profile_sha256"] = sha256(profile_bytes).hexdigest()
    tomogram["stair_profile_source_branch"] = str(profile["source_branch"])
    tomogram["stair_profile_point_count"] = len(projected_pct)

    gateway_up_count, gateway_down_count = _upstream_gateway_counts(tomogram)
    expected_gateway_counts = profile.get("expected_final_gateway_counts")
    if isinstance(expected_gateway_counts, dict) and (
        gateway_up_count != int(expected_gateway_counts["up"])
        or gateway_down_count != int(expected_gateway_counts["down"])
    ):
        raise ValueError(
            "楼梯 profile 最终 gateway 数量与固定资产合同不一致："
            f"up={gateway_up_count}, down={gateway_down_count}"
        )
    final_data = np.asarray(tomogram["data"])
    changed_cost_cells = int(
        np.count_nonzero(final_data[0] != base_data[0])
    )
    projected_array = np.asarray(projected_pct, dtype=np.float64)
    return {
        "path": str(profile_path),
        "sha256": tomogram["stair_profile_sha256"],
        "source_branch": tomogram["stair_profile_source_branch"],
        "asset_kind": tomogram["pct_scan_asset_kind"],
        "base_data_sha256": base_data_hash,
        "sampled_point_count": len(projected_pct),
        "stamped_cell_count": len(stamped_cells),
        "changed_cost_cell_count": changed_cost_cells,
        "injected_gateway_count": len(gateway_cells),
        "gateway_up_count": gateway_up_count,
        "gateway_down_count": gateway_down_count,
        "maximum_projection_hint_error_m": max(projection_errors),
        "pct_bbox_min": projected_array.min(axis=0).tolist(),
        "pct_bbox_max": projected_array.max(axis=0).tolist(),
    }


def _sample_profile_polyline(
    anchors: tuple[tuple[float, float, float], ...],
    *,
    spacing_m: float,
) -> tuple[tuple[float, float, float], ...]:
    """按固定三维间距加密楼梯 profile，并保留最后一个 anchor。"""

    output: list[tuple[float, float, float]] = []
    for first, second in zip(anchors, anchors[1:]):
        distance = math.dist(first, second)
        divisions = max(1, int(math.ceil(distance / spacing_m)))
        for index in range(divisions):
            ratio = index / divisions
            output.append(
                tuple(
                    float(first[axis] + (second[axis] - first[axis]) * ratio)
                    for axis in range(3)
                )
            )
    output.append(anchors[-1])
    return tuple(output)


def _stair_profile_layer(
    ground_z: float,
    *,
    first_layer: int,
    last_layer: int,
    ground_z_origin_m: float,
    height_step_m: float,
) -> int:
    """应用本场景楼梯 band；该函数不能用于普通 endpoint 选层。"""

    layer = (
        first_layer
        if ground_z <= ground_z_origin_m
        else first_layer
        + int(math.ceil((ground_z - ground_z_origin_m) / height_step_m))
    )
    if layer < first_layer or layer > last_layer:
        raise ValueError(
            f"楼梯 profile 投影高度 {ground_z:.3f} 超出 logical layer band"
        )
    return layer


def _finite_profile_xyz(
    value: object,
    *,
    field_name: str,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"楼梯 profile {field_name} 必须是 xyz 三元组")
    point = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"楼梯 profile {field_name} 含 NaN 或 Inf")
    return point


def _positive_profile_number(profile: dict[str, Any], field_name: str) -> float:
    value = float(profile[field_name])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"楼梯 profile {field_name} 必须是有限正数")
    return value


def _require_close(actual: float, expected: float, *, field_name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"楼梯 profile {field_name} 不匹配：{actual} != {expected}"
        )


def _official_ground_and_ceiling_layers(
    points: np.ndarray,
    grid: _Grid,
) -> tuple[np.ndarray, np.ndarray]:
    """复现官方 kernel：每个 slice 保存其下最高点与其上最低点。"""

    points_f32 = np.asarray(points, dtype=np.float32)
    if points_f32.ndim != 2 or points_f32.shape[1] != 3:
        raise ValueError("points 必须是 N×3")
    xi = np.round(
        (points_f32[:, 0] - grid.center[0]) / grid.resolution
    ).astype(np.int32) + grid.offset[0]
    yj = np.round(
        (points_f32[:, 1] - grid.center[1]) / grid.resolution
    ).astype(np.int32) + grid.offset[1]
    valid = (
        np.isfinite(points_f32).all(axis=1)
        & (xi >= 0)
        & (xi < grid.dimx)
        & (yj >= 0)
        & (yj < grid.dimy)
    )
    if not bool(np.any(valid)):
        raise RuntimeError("没有点落入官方 tomogram XY 网格")
    z = points_f32[valid, 2]
    flat_cell = xi[valid] * grid.dimy + yj[valid]
    first_ground_slice = np.ceil(
        (z.astype(np.float64) - grid.slice_h0) / grid.slice_dh
    ).astype(np.int32)
    cell_count = grid.dimx * grid.dimy

    ground_sources = np.full(
        (grid.n_slice, cell_count),
        -np.inf,
        dtype=np.float32,
    )
    contributes_ground = first_ground_slice < grid.n_slice
    np.maximum.at(
        ground_sources,
        (
            np.maximum(first_ground_slice[contributes_ground], 0),
            flat_cell[contributes_ground],
        ),
        z[contributes_ground],
    )
    ground = np.maximum.accumulate(ground_sources, axis=0)

    ceiling_sources = np.full(
        (grid.n_slice + 1, cell_count),
        np.inf,
        dtype=np.float32,
    )
    contributes_ceiling = first_ground_slice > 0
    np.minimum.at(
        ceiling_sources,
        (
            np.minimum(
                first_ground_slice[contributes_ceiling],
                grid.n_slice,
            ),
            flat_cell[contributes_ceiling],
        ),
        z[contributes_ceiling],
    )
    ceiling_suffix = np.minimum.accumulate(ceiling_sources[::-1], axis=0)[
        ::-1
    ]
    ceiling = ceiling_suffix[1:]
    ground[~np.isfinite(ground)] = np.nan
    ceiling[~np.isfinite(ceiling)] = np.nan
    return (
        ground.reshape(grid.n_slice, grid.dimx, grid.dimy),
        ceiling.reshape(grid.n_slice, grid.dimx, grid.dimy),
    )


def _official_traversability_cost(
    ground: np.ndarray,
    ceiling: np.ndarray,
    *,
    resolution: float,
    kernel_size: int,
    interval_min: float,
    interval_free: float,
    slope_max_rad: float,
    step_max_m: float,
    standable_ratio: float,
    cost_barrier: float,
) -> np.ndarray:
    """按官方 CuPy kernel 的分支次序计算未膨胀 traversability。"""

    grad_mag_sq = np.zeros_like(ground, dtype=np.float32)
    grad_mag_max = np.zeros_like(ground, dtype=np.float32)
    diff_x_sq = np.maximum(
        (ground[:, 1:-1, :] - ground[:, :-2, :]) ** 2,
        (ground[:, 1:-1, :] - ground[:, 2:, :]) ** 2,
    )
    diff_y_sq = np.maximum(
        (ground[:, :, 1:-1] - ground[:, :, :-2]) ** 2,
        (ground[:, :, 1:-1] - ground[:, :, 2:]) ** 2,
    )
    grad_mag_sq[:, 1:-1, 1:-1] = (
        diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
    )
    grad_mag_max[:, 1:-1, 1:-1] = np.maximum(
        diff_x_sq[:, :, 1:-1],
        diff_y_sq[:, 1:-1, :],
    )

    interval = ceiling - ground
    step_stand_sq = (
        1.2 * float(resolution) * math.tan(float(slope_max_rad))
    ) ** 2
    step_cross_sq = float(step_max_m) ** 2
    half_kernel = int(kernel_size / 2)
    standable_threshold = int(
        float(standable_ratio) * (2 * half_kernel + 1) ** 2
    ) - 1
    standable_count = convolve(
        (grad_mag_sq < step_stand_sq).astype(np.int16),
        np.ones((1, kernel_size, kernel_size), dtype=np.int16),
        mode="constant",
        cval=0,
    )

    cost = np.maximum(
        0.0,
        20.0 * (float(interval_free) - interval),
    ).astype(np.float32)
    low_clearance = interval < float(interval_min)
    standing = grad_mag_sq <= step_stand_sq
    standing_valid = standing & ~low_clearance
    cost[standing_valid] += (
        15.0 * grad_mag_sq[standing_valid] / step_stand_sq
    )
    crossing = (
        ~standing
        & ~low_clearance
        & (grad_mag_max <= step_cross_sq)
        & (standable_count >= standable_threshold)
    )
    cost[crossing] += 20.0 * grad_mag_max[crossing] / step_cross_sq
    blocked = low_clearance | (~standing & ~crossing)
    cost[blocked] = float(cost_barrier)
    return cost


def _official_inflation(
    traversability: np.ndarray,
    *,
    resolution: float,
    safe_margin_m: float,
    inflation_m: float,
) -> np.ndarray:
    """复现官方带距离权重的 max inflation kernel。"""

    half_kernel = int(
        (float(safe_margin_m) + float(inflation_m)) / float(resolution)
    )
    inflated = np.zeros_like(traversability, dtype=np.float32)
    denominator = float(safe_margin_m) + float(resolution)
    for dx in range(-half_kernel, half_kernel + 1):
        for dy in range(-half_kernel, half_kernel + 1):
            distance = math.hypot(dx * resolution, dy * resolution)
            score = float(
                np.clip(
                    1.0 - (distance - inflation_m) / denominator,
                    0.0,
                    1.0,
                )
            )
            source_x = slice(max(0, -dx), min(traversability.shape[1], traversability.shape[1] - dx))
            source_y = slice(max(0, -dy), min(traversability.shape[2], traversability.shape[2] - dy))
            target_x = slice(max(0, dx), min(traversability.shape[1], traversability.shape[1] + dx))
            target_y = slice(max(0, dy), min(traversability.shape[2], traversability.shape[2] + dy))
            np.maximum(
                inflated[:, target_x, target_y],
                traversability[:, source_x, source_y] * score,
                out=inflated[:, target_x, target_y],
            )
    return inflated


def _official_simplified_layer_indices(
    ground: np.ndarray,
    inflated: np.ndarray,
    *,
    cost_barrier: float,
) -> tuple[int, ...]:
    """逐句复现官方非均匀 layer simplification 索引选择。"""

    indices = [0]
    if ground.shape[0] > 1:
        lower_index, moving_index = 0, 1
        height_difference = ground[1:] - ground[:-1]
        while moving_index < ground.shape[0] - 2:
            unique = (
                (
                    (ground[moving_index] - ground[lower_index] > 0.0)
                    | (inflated[lower_index] > inflated[moving_index])
                )
                & (height_difference[moving_index] > 0.0)
                & (inflated[moving_index] < float(cost_barrier))
            )
            if bool(np.any(unique)):
                indices.append(moving_index)
                lower_index = moving_index
            moving_index += 1
        indices.append(moving_index)
    return tuple(indices)


def _upstream_gateway_counts(tomogram: dict[str, Any]) -> tuple[int, int]:
    """按官方 wrapper 规则统计相邻简化层的上下 gateway。"""

    data = np.asarray(tomogram["data"], dtype=np.float32)
    traversability = data[0]
    ground = np.nan_to_num(data[3], nan=-100.0)
    diff_t = traversability[1:] - traversability[:-1]
    diff_g = np.abs(ground[1:] - ground[:-1])
    gateway_up = (diff_t < -8.0) & (diff_g < 0.1)
    gateway_down = (diff_t > 8.0) & (diff_g < 0.1)
    return int(gateway_up.sum()), int(gateway_down.sum())


def _validate_upstream_tomography_parameters(
    grid: _Grid,
    **parameters: float | int,
) -> None:
    """在分配大数组前拒绝与官方 kernel 不兼容的参数。"""

    kernel_size = int(parameters["kernel_size"])
    if grid.n_slice < 3:
        raise ValueError("官方 layer simplification 至少需要 3 个初始 slice")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("traversability kernel_size 必须是正奇数")
    for name, value in parameters.items():
        if name == "kernel_size":
            continue
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} 必须是有限正数")
    if not 0.0 < float(parameters["standable_ratio"]) <= 1.0:
        raise ValueError("standable_ratio 必须在 (0, 1] 内")


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _optional_project_path(raw_path: str | Path | None) -> Path | None:
    """把 none/空值解释为禁用，其余路径按仓库根目录解析。"""

    if raw_path is None or str(raw_path).strip().lower() in {"", "none"}:
        return None
    path = _project_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"楼梯 corridor profile 不存在: {path}")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成本仓库 PCT 多楼层导航资产。")
    parser.add_argument("--collision-ply", default=os.fspath(DEFAULT_COLLISION_PLY), help="collision mesh PLY。")
    parser.add_argument("--output-tomogram", default=os.fspath(DEFAULT_TOMOGRAM), help="输出 tomogram pickle。")
    parser.add_argument("--output-walkable", default=os.fspath(DEFAULT_WALKABLE), help="输出 walkable npy。")
    parser.add_argument("--report-output", default=os.fspath(DEFAULT_REPORT), help="输出构建报告 JSON。")
    parser.add_argument("--resolution", type=float, default=0.20, help="XY grid 分辨率，单位米。")
    parser.add_argument("--slice-dh", type=float, default=0.50, help="tomogram slice 高度，单位米。")
    parser.add_argument("--padding", type=float, default=2.0, help="XY 边界外扩，单位米。")
    parser.add_argument("--sample-points", type=int, default=1_500_000, help="从 collision mesh 采样的点数。")
    parser.add_argument("--random-seed", type=int, default=0, help="NumPy PLY fallback 的确定性采样 seed。")
    parser.add_argument(
        "--sampling-backend",
        choices=("numpy", "open3d", "auto"),
        default="numpy",
        help="正式资产默认用确定性 NumPy 面积采样；auto 可保留旧行为。",
    )
    parser.add_argument("--dilation-radius", type=int, default=1, help="walkable XY 膨胀半径，单位 grid cell。")
    parser.add_argument("--max-wall-slices", type=int, default=15, help="同一 XY 跨越超过该 slice 数时视为墙面。")
    parser.add_argument("--z-min", type=float, default=None, help="可选最低 z 过滤。")
    parser.add_argument("--z-max", type=float, default=None, help="可选最高 z 过滤。")
    parser.add_argument(
        "--tomogram-kind",
        choices=("compatible", "upstream"),
        default="compatible",
        help="compatible 保留旧网格语义；upstream 复现官方五通道和 gateway。",
    )
    parser.add_argument(
        "--stair-corridor-profile",
        default=os.fspath(DEFAULT_STAIR_PROFILE),
        help="upstream 楼梯拓扑 profile JSON；传 none 可禁用。",
    )
    parser.add_argument("--traversability-kernel-size", type=int, default=7)
    parser.add_argument("--interval-min", type=float, default=0.50)
    parser.add_argument("--interval-free", type=float, default=0.65)
    parser.add_argument("--slope-max-rad", type=float, default=0.40)
    parser.add_argument("--step-max-m", type=float, default=0.65)
    parser.add_argument("--standable-ratio", type=float, default=0.05)
    parser.add_argument("--cost-barrier", type=float, default=50.0)
    parser.add_argument("--safe-margin-m", type=float, default=0.0001)
    parser.add_argument("--inflation-m", type=float, default=0.0001)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
