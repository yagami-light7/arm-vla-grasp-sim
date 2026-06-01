#!/usr/bin/env python3

"""Export a 2D navigation occupancy map from a USD collision scene."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export a 2D occupancy nav map from a USD scene.")
parser.add_argument("input", nargs="?", help="Legacy positional path to the USD/USDA scene file.")
parser.add_argument("--map", dest="map_path", help="Path to the USD/USDA scene file.")
parser.add_argument(
    "--prim-path",
    type=str,
    default=None,
    help="Prim path containing collision geometry. Defaults to /World/scene_collision when present.",
)
parser.add_argument(
    "--resolution",
    type=float,
    default=0.05,
    help="Map resolution in meters per pixel.",
)
parser.add_argument(
    "--padding",
    type=float,
    default=0.5,
    help="Extra XY padding added around the collision geometry bounds in meters.",
)
parser.add_argument(
    "--min-obstacle-height",
    type=float,
    default=0.15,
    help="Ignore geometry whose top surface stays below this world-frame Z height.",
)
parser.add_argument(
    "--max-obstacle-height",
    type=float,
    default=2.5,
    help="Ignore geometry whose bottom stays above this world-frame Z height.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to write occupancy.pgm, map.json, and preview.ppm.",
)
parser.add_argument(
    "--meta-name",
    type=str,
    default="map.json",
    help="Metadata filename to write in the output directory.",
)
parser.add_argument(
    "--image-name",
    type=str,
    default="occupancy.pgm",
    help="Occupancy raster filename to write in the output directory.",
)
parser.add_argument(
    "--preview-name",
    type=str,
    default="preview.ppm",
    help="Preview raster filename to write in the output directory.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
from pxr import Gf, Usd, UsdGeom

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))
from source.navigation.navlib import OccupancyGridMap, render_plan_preview, write_ppm


def _scene_path() -> Path:
    raw_path = args_cli.map_path or args_cli.input
    if raw_path is None:
        raise ValueError("Pass the scene USD with --map or as the positional input argument.")
    return Path(raw_path).expanduser().resolve()


def _resolve_output_dir() -> Path:
    if args_cli.output_dir is not None:
        return Path(args_cli.output_dir).expanduser().resolve()
    scene_path = _scene_path()
    return scene_path.parent / "nav_maps" / scene_path.stem


def _open_stage(stage_path: str) -> Usd.Stage:
    stage_path = os.path.abspath(os.path.expanduser(stage_path))
    stage = Usd.Stage.Open(stage_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {stage_path}")
    return stage


def _resolve_collision_prim(stage: Usd.Stage, prim_path: str | None):
    candidates: list[str] = []
    if prim_path is not None:
        candidates.append(prim_path)
    candidates.extend(["/World/scene_collision", "/scene_collision"])
    for candidate in candidates:
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid():
            return prim

    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return default_prim
    return stage.GetPseudoRoot()


def _iter_mesh_triangles(root_prim, min_height: float, max_height: float):
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_count = 0
    triangle_count = 0

    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr().Get()
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()
            if points_attr is None or face_counts is None or face_indices is None:
                continue

            mesh_count += 1
            world_transform = xform_cache.GetLocalToWorldTransform(prim)
            points_world = np.array(
                [
                    [
                        float(world_transform.Transform(Gf.Vec3d(point))[0]),
                        float(world_transform.Transform(Gf.Vec3d(point))[1]),
                        float(world_transform.Transform(Gf.Vec3d(point))[2]),
                    ]
                    for point in points_attr
                ],
                dtype=np.float64,
            )

            cursor = 0
            for face_count in face_counts:
                if face_count < 3:
                    cursor += face_count
                    continue
                polygon_indices = face_indices[cursor : cursor + face_count]
                polygon = points_world[np.asarray(polygon_indices, dtype=np.int64)]
                cursor += face_count

                top_z = float(np.max(polygon[:, 2]))
                bottom_z = float(np.min(polygon[:, 2]))
                if top_z < min_height or bottom_z > max_height:
                    continue

                for tri_idx in range(1, face_count - 1):
                    triangle = polygon[[0, tri_idx, tri_idx + 1]]
                    triangle_count += 1
                    yield triangle

    if mesh_count == 0:
        raise RuntimeError(f"No mesh prims found under {root_prim.GetPath()}.")
    if triangle_count == 0:
        raise RuntimeError(
            "No triangles survived the obstacle-height filter. "
            "Try lowering --min-obstacle-height or increasing --max-obstacle-height."
        )


def _collect_bounds(triangles: list[np.ndarray], padding: float) -> tuple[float, float, float, float]:
    min_x = min(float(np.min(triangle[:, 0])) for triangle in triangles) - padding
    max_x = max(float(np.max(triangle[:, 0])) for triangle in triangles) + padding
    min_y = min(float(np.min(triangle[:, 1])) for triangle in triangles) - padding
    max_y = max(float(np.max(triangle[:, 1])) for triangle in triangles) + padding
    return min_x, max_x, min_y, max_y


def _triangle_area_xy(triangle_xy: np.ndarray) -> float:
    a = triangle_xy[1] - triangle_xy[0]
    b = triangle_xy[2] - triangle_xy[0]
    return abs(a[0] * b[1] - a[1] * b[0]) * 0.5


def _rasterize_triangles(
    triangles: list[np.ndarray],
    *,
    resolution: float,
    bounds: tuple[float, float, float, float],
) -> OccupancyGridMap:
    min_x, max_x, min_y, max_y = bounds
    width = max(1, int(math.ceil((max_x - min_x) / resolution)))
    height = max(1, int(math.ceil((max_y - min_y) / resolution)))
    occupancy = np.zeros((height, width), dtype=bool)

    for triangle in triangles:
        tri_xy = triangle[:, :2]
        area = _triangle_area_xy(tri_xy)
        if area < 1.0e-9:
            continue

        tri_min_x = float(np.min(tri_xy[:, 0]))
        tri_max_x = float(np.max(tri_xy[:, 0]))
        tri_min_y = float(np.min(tri_xy[:, 1]))
        tri_max_y = float(np.max(tri_xy[:, 1]))

        col_min = max(0, int(math.floor((tri_min_x - min_x) / resolution)))
        col_max = min(width - 1, int(math.floor((tri_max_x - min_x) / resolution)))
        row_top = max(0, int(math.floor((max_y - tri_max_y) / resolution)))
        row_bottom = min(height - 1, int(math.floor((max_y - tri_min_y) / resolution)))
        if row_top > row_bottom or col_min > col_max:
            continue

        cols = np.arange(col_min, col_max + 1, dtype=np.int32)
        rows = np.arange(row_top, row_bottom + 1, dtype=np.int32)
        if cols.size == 0 or rows.size == 0:
            continue

        xs = min_x + (cols.astype(np.float64) + 0.5) * resolution
        ys = max_y - (rows.astype(np.float64) + 0.5) * resolution
        sample_x, sample_y = np.meshgrid(xs, ys)
        inside = _points_in_triangle(sample_x, sample_y, tri_xy)
        occupancy[row_top : row_bottom + 1, col_min : col_max + 1] |= inside

    return OccupancyGridMap(
        occupancy=occupancy,
        resolution=resolution,
        origin=(min_x, min_y, 0.0),
    )


def _points_in_triangle(sample_x: np.ndarray, sample_y: np.ndarray, triangle_xy: np.ndarray) -> np.ndarray:
    x1, y1 = triangle_xy[0]
    x2, y2 = triangle_xy[1]
    x3, y3 = triangle_xy[2]
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1.0e-12:
        return np.zeros_like(sample_x, dtype=bool)
    a = ((y2 - y3) * (sample_x - x3) + (x3 - x2) * (sample_y - y3)) / denom
    b = ((y3 - y1) * (sample_x - x3) + (x1 - x3) * (sample_y - y3)) / denom
    c = 1.0 - a - b
    eps = 1.0e-9
    return (a >= -eps) & (b >= -eps) & (c >= -eps)


def main():
    scene_path = _scene_path()
    stage = _open_stage(os.fspath(scene_path))
    root_prim = _resolve_collision_prim(stage, args_cli.prim_path)
    print(f"[INFO] Using collision prim: {root_prim.GetPath()}")

    triangles = list(
        _iter_mesh_triangles(
            root_prim,
            min_height=args_cli.min_obstacle_height,
            max_height=args_cli.max_obstacle_height,
        )
    )
    print(f"[INFO] Projecting {len(triangles)} triangles to the XY plane.")

    bounds = _collect_bounds(triangles, padding=args_cli.padding)
    nav_map = _rasterize_triangles(
        triangles,
        resolution=args_cli.resolution,
        bounds=bounds,
    )

    output_dir = _resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / args_cli.image_name
    meta_path = output_dir / args_cli.meta_name
    preview_path = output_dir / args_cli.preview_name

    nav_map.save_pgm(image_path)
    nav_map.save_meta_file(meta_path, image_path=image_path.name)
    preview = render_plan_preview(nav_map)
    write_ppm(preview_path, preview)

    print(f"[INFO] Scene file: {scene_path}")
    print(f"[INFO] Map size: {nav_map.width} x {nav_map.height} cells")
    print(f"[INFO] Resolution: {nav_map.resolution:.3f} m/cell")
    print(f"[INFO] Origin: {nav_map.origin}")
    print(f"[INFO] Occupied cells: {int(np.count_nonzero(nav_map.occupancy))}")
    print(f"[INFO] Wrote occupancy raster: {image_path}")
    print(f"[INFO] Wrote metadata: {meta_path}")
    print(f"[INFO] Wrote preview: {preview_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
