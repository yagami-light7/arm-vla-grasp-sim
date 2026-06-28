"""将 PCT walkable slice 转为当前 DWA 使用的 2D 局部地图。"""

from __future__ import annotations

import math
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from source.navigation.navlib import OccupancyGridMap

from .pct_adapter import pct_to_sim_xyz

PCT_TRAVERSABILITY_BARRIER = 49.0


def load_pct_slice_local_map(
    *,
    tomogram_path: str | Path,
    walkable_path: str | Path,
    z: float,
    coord_mode: str = "sim_to_pct_180deg",
    pct_offset_x: float = 0.0,
    pct_offset_y: float = 0.0,
    pct_scale_x: float = 1.0,
    pct_scale_y: float = 1.0,
    slice_neighbor_radius: int = 1,
    traversability_barrier: float = PCT_TRAVERSABILITY_BARRIER,
    free_dilation_radius_cells: int = 0,
    collision_ply_path: str | Path | None = None,
    vertical_obstacle_min_slices: int = 5,
    vertical_obstacle_dilation_radius_cells: int = 0,
    robot_root_to_floor_m: float = 0.0,
    body_obstacle_min_height_m: float | None = None,
    body_obstacle_max_height_m: float = 1.0,
) -> OccupancyGridMap:
    """读取 PCT 资产，并把目标 z 所在 slice 转成 Isaac Sim XY 占用图。"""

    return load_pct_route_local_map(
        tomogram_path=tomogram_path,
        walkable_path=walkable_path,
        z_values=(z,),
        coord_mode=coord_mode,
        pct_offset_x=pct_offset_x,
        pct_offset_y=pct_offset_y,
        pct_scale_x=pct_scale_x,
        pct_scale_y=pct_scale_y,
        slice_neighbor_radius=slice_neighbor_radius,
        traversability_barrier=traversability_barrier,
        free_dilation_radius_cells=free_dilation_radius_cells,
        collision_ply_path=collision_ply_path,
        vertical_obstacle_min_slices=vertical_obstacle_min_slices,
        vertical_obstacle_dilation_radius_cells=vertical_obstacle_dilation_radius_cells,
        robot_root_to_floor_m=robot_root_to_floor_m,
        body_obstacle_min_height_m=body_obstacle_min_height_m,
        body_obstacle_max_height_m=body_obstacle_max_height_m,
    )


def load_pct_route_local_map(
    *,
    tomogram_path: str | Path,
    walkable_path: str | Path,
    z_values: Sequence[float],
    coord_mode: str = "sim_to_pct_180deg",
    pct_offset_x: float = 0.0,
    pct_offset_y: float = 0.0,
    pct_scale_x: float = 1.0,
    pct_scale_y: float = 1.0,
    slice_neighbor_radius: int = 1,
    traversability_barrier: float = PCT_TRAVERSABILITY_BARRIER,
    free_dilation_radius_cells: int = 0,
    collision_ply_path: str | Path | None = None,
    vertical_obstacle_min_slices: int = 5,
    vertical_obstacle_dilation_radius_cells: int = 0,
    robot_root_to_floor_m: float = 0.0,
    body_obstacle_min_height_m: float | None = None,
    body_obstacle_max_height_m: float = 1.0,
) -> OccupancyGridMap:
    """合并路径涉及的高度切片，供二维局部控制器跟踪跨楼层 PCT 路径。"""

    route_z_values = tuple(float(value) for value in z_values)
    if not route_z_values:
        raise ValueError("PCT route local map 至少需要一个 z。")
    tomogram = _load_tomogram(Path(tomogram_path).expanduser())
    walkable = np.load(Path(walkable_path).expanduser()).astype(bool, copy=False)
    traversability = np.asarray(tomogram["data"][0], dtype=np.float32)
    resolution = float(tomogram["resolution"])
    center = np.asarray(tomogram["center"], dtype=np.float64)
    slice_h0 = float(tomogram["slice_h0"])
    slice_dh = float(tomogram["slice_dh"])
    if walkable.ndim != 3:
        raise ValueError("PCT walkable map 必须是 3 维数组。")
    if traversability.ndim != 3:
        raise ValueError("PCT tomogram traversability 必须是 3 维数组。")
    if slice_neighbor_radius < 0:
        raise ValueError("slice_neighbor_radius 不能为负数。")
    if (
        collision_ply_path is not None
        and body_obstacle_min_height_m is None
        and vertical_obstacle_min_slices <= 0
    ):
        raise ValueError("启用 collision PLY 障碍层时 vertical_obstacle_min_slices 必须为正数。")
    if vertical_obstacle_dilation_radius_cells < 0:
        raise ValueError("vertical_obstacle_dilation_radius_cells 不能为负数。")
    if not math.isclose(
        abs(float(pct_scale_x)),
        abs(float(pct_scale_y)),
        rel_tol=1.0e-6,
        abs_tol=1.0e-9,
    ):
        raise ValueError("当前 DWA 局部地图只支持 x/y 等比例 PCT 坐标缩放。")
    n_slice = min(walkable.shape[0], traversability.shape[0])
    walkable_slice = np.zeros(
        (
            min(int(traversability.shape[1]), int(walkable.shape[1])),
            min(int(traversability.shape[2]), int(walkable.shape[2])),
        ),
        dtype=bool,
    )
    route_slice_indices: set[int] = set()
    for route_z in route_z_values:
        slice_index = _z_to_slice(
            z=route_z - float(robot_root_to_floor_m),
            slice_h0=slice_h0,
            slice_dh=slice_dh,
            n_slice=n_slice,
        )
        route_slice_indices.add(slice_index)
        walkable_slice |= pct_combined_walkable_slice(
            tomogram=tomogram,
            walkable=walkable,
            slice_index=slice_index,
            slice_neighbor_radius=slice_neighbor_radius,
            traversability_barrier=traversability_barrier,
        )
    if collision_ply_path is not None:
        if body_obstacle_min_height_m is None:
            collision_obstacles = pct_vertical_obstacle_mask_from_ply(
                collision_ply_path=collision_ply_path,
                tomogram=tomogram,
                vertical_obstacle_min_slices=vertical_obstacle_min_slices,
            )
        else:
            body_volume = pct_robot_body_obstacle_volume_from_ply(
                collision_ply_path=collision_ply_path,
                tomogram=tomogram,
                min_height_m=float(body_obstacle_min_height_m),
                max_height_m=float(body_obstacle_max_height_m),
            )
            collision_obstacles = np.any(
                body_volume[sorted(route_slice_indices)],
                axis=0,
            )
        if vertical_obstacle_dilation_radius_cells > 0:
            collision_obstacles = _dilate_obstacle_cells(
                collision_obstacles,
                radius_cells=vertical_obstacle_dilation_radius_cells,
            )
        if collision_obstacles.shape != walkable_slice.shape:
            raise ValueError("PCT collision obstacle mask 与 walkable slice 尺寸不一致。")
        walkable_slice = walkable_slice & ~collision_obstacles
    return pct_walkable_slice_to_sim_grid(
        walkable_slice=walkable_slice,
        resolution=resolution,
        center=center,
        coord_mode=coord_mode,
        pct_offset_x=pct_offset_x,
        pct_offset_y=pct_offset_y,
        pct_scale_x=pct_scale_x,
        pct_scale_y=pct_scale_y,
        free_dilation_radius_cells=free_dilation_radius_cells,
    )


def pct_combined_walkable_slice(
    *,
    tomogram: dict[str, Any],
    walkable: np.ndarray,
    slice_index: int,
    slice_neighbor_radius: int = 1,
    traversability_barrier: float = PCT_TRAVERSABILITY_BARRIER,
) -> np.ndarray:
    """复用 PCT server 的可走判定，并允许合并相邻高度 slice。"""

    traversability = np.asarray(tomogram["data"][0], dtype=np.float32)
    if traversability.ndim != 3 or walkable.ndim != 3:
        raise ValueError("PCT tomogram 和 walkable 必须都是 3 维数组。")
    n_slice = min(int(traversability.shape[0]), int(walkable.shape[0]))
    dimx = min(int(traversability.shape[1]), int(walkable.shape[1]))
    dimy = min(int(traversability.shape[2]), int(walkable.shape[2]))
    if n_slice < 1 or dimx < 1 or dimy < 1:
        raise ValueError("PCT tomogram 或 walkable 为空。")
    if slice_neighbor_radius < 0:
        raise ValueError("slice_neighbor_radius 不能为负数。")
    center_slice = int(np.clip(int(slice_index), 0, n_slice - 1))
    start = max(0, center_slice - int(slice_neighbor_radius))
    end = min(n_slice, center_slice + int(slice_neighbor_radius) + 1)
    traversability_crop = traversability[start:end, :dimx, :dimy]
    walkable_crop = walkable[start:end, :dimx, :dimy].astype(bool, copy=False)
    traversability_ok = (
        (traversability_crop > 0.0)
        & (traversability_crop < float(traversability_barrier))
    )
    return np.any(traversability_ok | walkable_crop, axis=0)


def pct_vertical_obstacle_mask_from_ply(
    *,
    collision_ply_path: str | Path,
    tomogram: dict[str, Any],
    vertical_obstacle_min_slices: int = 5,
) -> np.ndarray:
    """从 collision PLY 中提取跨多层高度的墙体/柱体障碍。"""

    if vertical_obstacle_min_slices <= 0:
        raise ValueError("vertical_obstacle_min_slices 必须为正数。")
    points = _read_binary_little_endian_ply_xyz(Path(collision_ply_path).expanduser())
    traversability = np.asarray(tomogram["data"][0], dtype=np.float32)
    if traversability.ndim != 3:
        raise ValueError("PCT tomogram traversability 必须是 3 维数组。")
    n_slice, dimx, dimy = (
        int(traversability.shape[0]),
        int(traversability.shape[1]),
        int(traversability.shape[2]),
    )
    resolution = float(tomogram["resolution"])
    center = np.asarray(tomogram["center"], dtype=np.float64)
    slice_h0 = float(tomogram["slice_h0"])
    slice_dh = float(tomogram["slice_dh"])
    offset = np.array([dimx // 2, dimy // 2], dtype=np.int32)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    xi = np.round((points[:, 0] - center[0]) / resolution).astype(np.int32) + int(offset[0])
    yj = np.round((points[:, 1] - center[1]) / resolution).astype(np.int32) + int(offset[1])
    si = np.round((points[:, 2] - slice_h0) / slice_dh).astype(np.int32)
    valid = (
        (xi >= 0)
        & (xi < dimx)
        & (yj >= 0)
        & (yj < dimy)
        & (si >= 0)
        & (si < n_slice)
    )
    if not np.any(valid):
        return np.zeros((dimx, dimy), dtype=bool)
    flat_slice_xy = (si[valid].astype(np.int64) * dimx + xi[valid].astype(np.int64)) * dimy + yj[
        valid
    ].astype(np.int64)
    unique_slice_xy = np.unique(flat_slice_xy)
    xy_flat = unique_slice_xy % (dimx * dimy)
    slice_count = np.bincount(xy_flat, minlength=dimx * dimy).reshape(dimx, dimy)
    return slice_count >= int(vertical_obstacle_min_slices)


def pct_vertical_obstacle_volume_from_ply(
    *,
    collision_ply_path: str | Path,
    tomogram: dict[str, Any],
    vertical_obstacle_min_slices: int = 5,
) -> np.ndarray:
    """从 collision PLY 生成逐 slice 墙体障碍，避免不同楼层互相遮挡。"""

    if vertical_obstacle_min_slices <= 0:
        raise ValueError("vertical_obstacle_min_slices 必须为正数。")
    points = _read_binary_little_endian_ply_xyz(Path(collision_ply_path).expanduser())
    traversability = np.asarray(tomogram["data"][0], dtype=np.float32)
    if traversability.ndim != 3:
        raise ValueError("PCT tomogram traversability 必须是 3 维数组。")
    n_slice, dimx, dimy = (
        int(traversability.shape[0]),
        int(traversability.shape[1]),
        int(traversability.shape[2]),
    )
    resolution = float(tomogram["resolution"])
    center = np.asarray(tomogram["center"], dtype=np.float64)
    slice_h0 = float(tomogram["slice_h0"])
    slice_dh = float(tomogram["slice_dh"])
    offset = np.array([dimx // 2, dimy // 2], dtype=np.int32)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    xi = np.round((points[:, 0] - center[0]) / resolution).astype(np.int32) + int(offset[0])
    yj = np.round((points[:, 1] - center[1]) / resolution).astype(np.int32) + int(offset[1])
    si = np.round((points[:, 2] - slice_h0) / slice_dh).astype(np.int32)
    valid = (
        (xi >= 0)
        & (xi < dimx)
        & (yj >= 0)
        & (yj < dimy)
        & (si >= 0)
        & (si < n_slice)
    )
    if not np.any(valid):
        return np.zeros((n_slice, dimx, dimy), dtype=bool)

    flat_slice_xy = (
        (si[valid].astype(np.int64) * dimx + xi[valid].astype(np.int64)) * dimy
        + yj[valid].astype(np.int64)
    )
    unique_slice_xy = np.unique(flat_slice_xy)
    unique_slice = unique_slice_xy // (dimx * dimy)
    unique_xy = unique_slice_xy % (dimx * dimy)
    slice_count = np.bincount(unique_xy, minlength=dimx * dimy)
    vertical_xy = slice_count >= int(vertical_obstacle_min_slices)

    volume = np.zeros((n_slice, dimx * dimy), dtype=bool)
    selected = vertical_xy[unique_xy]
    volume[unique_slice[selected], unique_xy[selected]] = True
    return volume.reshape(n_slice, dimx, dimy)


def pct_robot_body_obstacle_volume_from_ply(
    *,
    collision_ply_path: str | Path,
    tomogram: dict[str, Any],
    min_height_m: float = 0.30,
    max_height_m: float = 1.0,
) -> np.ndarray:
    """按机器人身体净空生成逐 slice 障碍，覆盖薄墙、桌椅和其他家具。"""

    min_height = float(min_height_m)
    max_height = float(max_height_m)
    if min_height < 0.0:
        raise ValueError("机器人身体障碍最小高度不能为负数。")
    if max_height <= min_height:
        raise ValueError("机器人身体障碍最大高度必须大于最小高度。")
    vertices, faces = _read_binary_little_endian_ply_mesh(
        Path(collision_ply_path).expanduser()
    )
    samples = _sample_triangle_mesh(vertices, faces)
    traversability = np.asarray(tomogram["data"][0], dtype=np.float32)
    if traversability.ndim != 3:
        raise ValueError("PCT tomogram traversability 必须是 3 维数组。")
    n_slice, dimx, dimy = (
        int(traversability.shape[0]),
        int(traversability.shape[1]),
        int(traversability.shape[2]),
    )
    resolution = float(tomogram["resolution"])
    center = np.asarray(tomogram["center"], dtype=np.float64)
    slice_h0 = float(tomogram["slice_h0"])
    slice_dh = float(tomogram["slice_dh"])
    offset = np.array([dimx // 2, dimy // 2], dtype=np.int32)

    finite = np.isfinite(samples).all(axis=1)
    samples = samples[finite]
    xi = np.round((samples[:, 0] - center[0]) / resolution).astype(np.int32) + int(
        offset[0]
    )
    yj = np.round((samples[:, 1] - center[1]) / resolution).astype(np.int32) + int(
        offset[1]
    )
    valid_xy = (
        (xi >= 0)
        & (xi < dimx)
        & (yj >= 0)
        & (yj < dimy)
    )
    xi = xi[valid_xy]
    yj = yj[valid_xy]
    sample_z = samples[valid_xy, 2]

    volume = np.zeros((n_slice, dimx, dimy), dtype=bool)
    for slice_index in range(n_slice):
        floor_z = slice_h0 + slice_index * slice_dh
        in_body_band = (
            (sample_z >= floor_z + min_height)
            & (sample_z <= floor_z + max_height)
        )
        volume[slice_index, xi[in_body_band], yj[in_body_band]] = True
    return volume


def _sample_triangle_mesh(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """用顶点、边中点和面中心填充稀疏三角面，减少薄障碍漏栅格。"""

    if faces.size == 0:
        return vertices
    triangles = vertices[faces]
    return np.concatenate(
        (
            vertices,
            triangles.mean(axis=1),
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        ),
        axis=0,
    )


def load_pct_vertical_obstacle_grid(
    *,
    tomogram_path: str | Path,
    collision_ply_path: str | Path,
    vertical_obstacle_min_slices: int,
    coord_mode: str = "sim_to_pct_180deg",
    pct_offset_x: float = 0.0,
    pct_offset_y: float = 0.0,
    pct_scale_x: float = 1.0,
    pct_scale_y: float = 1.0,
    dilation_radius_cells: int = 0,
) -> OccupancyGridMap:
    """把 collision PLY 的跨层硬障碍转换为 Isaac Sim 二维占用图。"""

    if dilation_radius_cells < 0:
        raise ValueError("dilation_radius_cells 不能为负数。")
    tomogram = _load_tomogram(Path(tomogram_path).expanduser())
    obstacle_mask = pct_vertical_obstacle_mask_from_ply(
        collision_ply_path=collision_ply_path,
        tomogram=tomogram,
        vertical_obstacle_min_slices=vertical_obstacle_min_slices,
    )
    if dilation_radius_cells > 0:
        obstacle_mask = _dilate_obstacle_cells(
            obstacle_mask,
            radius_cells=int(dilation_radius_cells),
        )
    return pct_walkable_slice_to_sim_grid(
        walkable_slice=~obstacle_mask,
        resolution=float(tomogram["resolution"]),
        center=np.asarray(tomogram["center"], dtype=np.float64),
        coord_mode=coord_mode,
        pct_offset_x=pct_offset_x,
        pct_offset_y=pct_offset_y,
        pct_scale_x=pct_scale_x,
        pct_scale_y=pct_scale_y,
    )


def pct_walkable_slice_to_sim_grid(
    *,
    walkable_slice: np.ndarray,
    resolution: float,
    center: np.ndarray,
    coord_mode: str = "sim_to_pct_180deg",
    pct_offset_x: float = 0.0,
    pct_offset_y: float = 0.0,
    pct_scale_x: float = 1.0,
    pct_scale_y: float = 1.0,
    free_dilation_radius_cells: int = 0,
) -> OccupancyGridMap:
    """把单层 PCT 可走 mask 转为 DWA 的 sim 坐标 OccupancyGridMap。"""

    if walkable_slice.ndim != 2:
        raise ValueError("PCT walkable slice 必须是 2 维数组。")
    dimx, dimy = int(walkable_slice.shape[0]), int(walkable_slice.shape[1])
    if dimx < 1 or dimy < 1:
        raise ValueError("PCT walkable slice 不能为空。")
    if free_dilation_radius_cells < 0:
        raise ValueError("free_dilation_radius_cells 不能为负数。")
    if free_dilation_radius_cells > 0:
        walkable_slice = _dilate_free_cells(
            walkable_slice.astype(bool, copy=False),
            radius_cells=int(free_dilation_radius_cells),
        )
    sim_resolution = float(resolution) / abs(float(pct_scale_x))
    offset_x = dimx // 2
    offset_y = dimy // 2

    pct_x_centers = (
        (np.arange(dimx, dtype=np.float64) - offset_x)
        * float(resolution)
        + float(center[0])
    )
    pct_y_centers = (
        (np.arange(dimy, dtype=np.float64) - offset_y)
        * float(resolution)
        + float(center[1])
    )
    sim_x_centers = np.asarray(
        [
            pct_to_sim_xyz(
                (float(pct_x), float(pct_y_centers[0]), 0.0),
                coord_mode=coord_mode,
                pct_offset_x=pct_offset_x,
                pct_offset_y=pct_offset_y,
                pct_scale_x=pct_scale_x,
                pct_scale_y=pct_scale_y,
            )[0]
            for pct_x in pct_x_centers
        ],
        dtype=np.float64,
    )
    sim_y_centers = np.asarray(
        [
            pct_to_sim_xyz(
                (float(pct_x_centers[0]), float(pct_y), 0.0),
                coord_mode=coord_mode,
                pct_offset_x=pct_offset_x,
                pct_offset_y=pct_offset_y,
                pct_scale_x=pct_scale_x,
                pct_scale_y=pct_scale_y,
            )[1]
            for pct_y in pct_y_centers
        ],
        dtype=np.float64,
    )
    min_x = float(sim_x_centers.min()) - 0.5 * sim_resolution
    min_y = float(sim_y_centers.min()) - 0.5 * sim_resolution
    occupancy = np.ones((dimy, dimx), dtype=bool)
    for xi, sim_x in enumerate(sim_x_centers):
        col = int(math.floor((float(sim_x) - min_x) / sim_resolution))
        col = int(np.clip(col, 0, dimx - 1))
        for yj, sim_y in enumerate(sim_y_centers):
            row_from_bottom = int(math.floor((float(sim_y) - min_y) / sim_resolution))
            row = dimy - 1 - int(np.clip(row_from_bottom, 0, dimy - 1))
            occupancy[row, col] = not bool(walkable_slice[xi, yj])
    return OccupancyGridMap(
        occupancy=occupancy,
        resolution=sim_resolution,
        origin=(min_x, min_y, 0.0),
    )


def add_circular_keepouts(
    grid_map: OccupancyGridMap,
    *,
    centers_xy: Sequence[tuple[float, float]],
    radius_m: float,
) -> OccupancyGridMap:
    """把任务物体等动态障碍投影成 DWA 可见的圆形占用区。"""

    radius = float(radius_m)
    if radius <= 0.0:
        return grid_map
    occupancy = grid_map.occupancy.copy()
    radius_cells = int(math.ceil(radius / float(grid_map.resolution)))
    for center_x, center_y in centers_xy:
        center_row, center_col = grid_map.world_to_grid(float(center_x), float(center_y))
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                row = center_row + dr
                col = center_col + dc
                if not grid_map.in_bounds(row, col):
                    continue
                cell_x, cell_y = grid_map.grid_to_world(row, col)
                if math.hypot(cell_x - float(center_x), cell_y - float(center_y)) <= radius:
                    occupancy[row, col] = True
    return OccupancyGridMap(
        occupancy=occupancy,
        resolution=grid_map.resolution,
        origin=grid_map.origin,
        image_path=grid_map.image_path,
        meta_path=grid_map.meta_path,
    )


def _dilate_free_cells(mask: np.ndarray, *, radius_cells: int) -> np.ndarray:
    """在不依赖 scipy 的情况下扩张可走区域。"""

    radius = int(radius_cells)
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = mask.copy()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            output |= padded[
                radius + dx : radius + dx + mask.shape[0],
                radius + dy : radius + dy + mask.shape[1],
            ]
    return output


def _dilate_obstacle_cells(mask: np.ndarray, *, radius_cells: int) -> np.ndarray:
    """在不依赖 scipy 的情况下扩张障碍区域。"""

    radius = int(radius_cells)
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = mask.copy()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            output |= padded[
                radius + dx : radius + dx + mask.shape[0],
                radius + dy : radius + dy + mask.shape[1],
            ]
    return output


def _read_binary_little_endian_ply_xyz(path: Path) -> np.ndarray:
    """读取当前 collision PLY 的 x/y/z 顶点数组。"""

    vertices, _faces = _read_binary_little_endian_ply_mesh(path)
    return vertices


def _read_binary_little_endian_ply_mesh(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """读取当前 collision PLY 的顶点和三角面。"""

    if not path.is_file():
        raise FileNotFoundError(f"collision PLY 不存在: {path}")
    vertex_count: int | None = None
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    in_vertex_element = False
    face_list_supported = False
    with path.open("rb") as stream:
        first = stream.readline().decode("ascii", errors="ignore").strip()
        if first != "ply":
            raise ValueError(f"不是 PLY 文件: {path}")
        format_seen = False
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError(f"PLY 缺少 end_header: {path}")
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line == "format binary_little_endian 1.0":
                format_seen = True
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex_element = True
            elif line.startswith("element face "):
                face_count = int(line.split()[-1])
                in_vertex_element = False
            elif line.startswith("element "):
                in_vertex_element = False
            elif in_vertex_element and line.startswith("property "):
                parts = line.split()
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError("当前只支持标量 vertex property 的 binary_little_endian PLY。")
                vertex_properties.append((parts[1], parts[2]))
            elif line == "property list uchar int vertex_indices":
                face_list_supported = True
            elif line == "end_header":
                break
        if not format_seen:
            raise ValueError("当前只支持 binary_little_endian 1.0 PLY。")
        if vertex_count is None:
            raise ValueError(f"PLY 缺少 vertex 数量: {path}")
        names = [name for _, name in vertex_properties]
        for required_name in ("x", "y", "z"):
            if required_name not in names:
                raise ValueError(f"PLY vertex 缺少 {required_name} property: {path}")
        dtype_fields = [
            (name, _ply_scalar_dtype(property_type))
            for property_type, name in vertex_properties
        ]
        data = np.fromfile(stream, dtype=np.dtype(dtype_fields), count=vertex_count)
        faces = np.empty((0, 3), dtype=np.int32)
        if face_count:
            if not face_list_supported:
                raise ValueError("当前只支持 uchar/int 三角面 vertex_indices。")
            face_data = np.fromfile(
                stream,
                dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]),
                count=face_count,
            )
            if len(face_data) != face_count or np.any(face_data["count"] != 3):
                raise ValueError("当前 collision PLY 必须只包含三角面。")
            faces = face_data["indices"].astype(np.int32, copy=False)
    vertices = np.column_stack(
        (
            data["x"].astype(np.float32, copy=False),
            data["y"].astype(np.float32, copy=False),
            data["z"].astype(np.float32, copy=False),
        )
    )
    return vertices, faces


def _ply_scalar_dtype(property_type: str) -> str:
    """把 PLY 标量类型映射为 numpy dtype。"""

    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    try:
        return mapping[property_type]
    except KeyError as exc:
        raise ValueError(f"不支持的 PLY property 类型: {property_type}") from exc


def pct_slice_for_z(*, tomogram_path: str | Path, walkable_path: str | Path, z: float) -> int:
    """返回指定 z 在 PCT walkable 中对应的 slice index。"""

    tomogram = _load_tomogram(Path(tomogram_path).expanduser())
    walkable = np.load(Path(walkable_path).expanduser(), mmap_mode="r")
    return _z_to_slice(
        z=float(z),
        slice_h0=float(tomogram["slice_h0"]),
        slice_dh=float(tomogram["slice_dh"]),
        n_slice=int(walkable.shape[0]),
    )


def _load_tomogram(path: Path) -> dict[str, Any]:
    inserted_aliases = _install_numpy_pickle_aliases()
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    finally:
        _remove_numpy_pickle_aliases(inserted_aliases)
    if not isinstance(payload, dict):
        raise ValueError("PCT tomogram pickle 必须包含 dict。")
    for key in ("data", "resolution", "center", "slice_h0", "slice_dh"):
        if key not in payload:
            raise ValueError(f"PCT tomogram 缺少字段: {key}")
    return payload


def _install_numpy_pickle_aliases() -> tuple[str, ...]:
    """兼容 numpy 2 保存、numpy 1 读取的 pickle 模块路径。"""

    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_core_numeric

    aliases: list[tuple[str, object]] = [
        ("numpy._core", numpy_core),
        ("numpy._core.numeric", numpy_core_numeric),
    ]
    inserted: list[str] = []
    for name, module in aliases:
        if name in sys.modules:
            continue
        sys.modules[name] = module
        inserted.append(name)
    return tuple(inserted)


def _remove_numpy_pickle_aliases(inserted_aliases: tuple[str, ...]) -> None:
    """撤销本次 pickle 读取临时加入的 numpy 模块别名。"""

    for name in inserted_aliases:
        sys.modules.pop(name, None)


def _z_to_slice(*, z: float, slice_h0: float, slice_dh: float, n_slice: int) -> int:
    if n_slice < 1:
        raise ValueError("PCT walkable map 没有可用 slice。")
    index = int(round((float(z) - float(slice_h0)) / float(slice_dh)))
    return int(np.clip(index, 0, int(n_slice) - 1))
