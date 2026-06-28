from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from source.navigation.pct_local_map import (
    add_circular_keepouts,
    load_pct_route_local_map,
    load_pct_slice_local_map,
    load_pct_vertical_obstacle_grid,
    pct_combined_walkable_slice,
    pct_robot_body_obstacle_volume_from_ply,
    pct_vertical_obstacle_mask_from_ply,
    pct_vertical_obstacle_volume_from_ply,
    pct_walkable_slice_to_sim_grid,
)
from source.navigation.navlib import OccupancyGridMap


def test_pct_walkable_slice_maps_obstacles_into_sim_frame() -> None:
    walkable = np.ones((3, 2), dtype=bool)
    walkable[2, 0] = False
    walkable[0, 1] = False

    grid = pct_walkable_slice_to_sim_grid(
        walkable_slice=walkable,
        resolution=1.0,
        center=np.array([0.0, 0.0], dtype=np.float64),
    )

    assert grid.resolution == 1.0
    assert grid.origin == (-1.5, -0.5, 0.0)
    assert grid.occupancy.shape == (2, 3)
    assert grid.is_occupied(*grid.world_to_grid(-1.0, 1.0))
    assert grid.is_occupied(*grid.world_to_grid(1.0, 0.0))
    assert not grid.is_occupied(*grid.world_to_grid(0.0, 0.0))


def test_pct_walkable_slice_can_dilate_free_cells() -> None:
    walkable = np.zeros((5, 5), dtype=bool)
    walkable[2, 2] = True

    grid = pct_walkable_slice_to_sim_grid(
        walkable_slice=walkable,
        resolution=1.0,
        center=np.array([0.0, 0.0], dtype=np.float64),
        free_dilation_radius_cells=1,
    )

    assert not grid.is_occupied(*grid.world_to_grid(0.0, 0.0))
    assert not grid.is_occupied(*grid.world_to_grid(1.0, 0.0))
    assert grid.is_occupied(*grid.world_to_grid(2.0, 2.0))


def test_pct_combined_walkable_slice_matches_server_semantics() -> None:
    traversability = np.full((3, 4, 4), 50.0, dtype=np.float32)
    walkable = np.zeros((3, 4, 4), dtype=bool)
    traversability[1, 1, 1] = 1.0
    walkable[0, 2, 2] = True

    tomogram = {"data": np.stack([traversability], axis=0)}
    combined = pct_combined_walkable_slice(
        tomogram=tomogram,
        walkable=walkable,
        slice_index=1,
        slice_neighbor_radius=1,
    )

    assert combined[1, 1]
    assert combined[2, 2]
    assert not combined[3, 3]


def test_pct_route_local_map_unions_required_height_slices(tmp_path: Path) -> None:
    tomogram_path = tmp_path / "tomogram.pickle"
    walkable_path = tmp_path / "walkable.npy"
    traversability = np.full((4, 5, 5), 50.0, dtype=np.float32)
    walkable = np.zeros((4, 5, 5), dtype=bool)
    walkable[0, 1, 1] = True
    walkable[3, 3, 3] = True
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
        "center": np.array([0.0, 0.0], dtype=np.float64),
        "slice_h0": 0.0,
        "slice_dh": 1.0,
    }
    with tomogram_path.open("wb") as stream:
        pickle.dump(tomogram, stream)
    np.save(walkable_path, walkable)

    route_grid = load_pct_route_local_map(
        tomogram_path=tomogram_path,
        walkable_path=walkable_path,
        z_values=(0.0, 3.0),
        slice_neighbor_radius=0,
    )
    low_grid = load_pct_slice_local_map(
        tomogram_path=tomogram_path,
        walkable_path=walkable_path,
        z=0.0,
        slice_neighbor_radius=0,
    )

    high_only_xy = (-1.0, -1.0)
    assert not route_grid.is_occupied(*route_grid.world_to_grid(*high_only_xy))
    assert low_grid.is_occupied(*low_grid.world_to_grid(*high_only_xy))


def test_pct_vertical_obstacle_mask_uses_collision_ply_height_span(tmp_path: Path) -> None:
    ply_path = tmp_path / "collision.ply"
    points = []
    for z in range(5):
        points.append((0.0, 0.0, float(z)))
    for z in range(2):
        points.append((1.0, 0.0, float(z)))
    _write_binary_xyz_ply(ply_path, points)

    tomogram = {
        "data": np.zeros((5, 6, 7, 7), dtype=np.float32),
        "resolution": 1.0,
        "center": np.array([0.0, 0.0], dtype=np.float64),
        "slice_h0": 0.0,
        "slice_dh": 1.0,
    }

    mask = pct_vertical_obstacle_mask_from_ply(
        collision_ply_path=ply_path,
        tomogram=tomogram,
        vertical_obstacle_min_slices=5,
    )

    offset_x = tomogram["data"].shape[2] // 2
    offset_y = tomogram["data"].shape[3] // 2
    assert mask[offset_x, offset_y]
    assert not mask[offset_x + 1, offset_y]


def test_pct_vertical_obstacle_volume_preserves_floor_specific_cells(
    tmp_path: Path,
) -> None:
    ply_path = tmp_path / "collision.ply"
    points = [(0.0, 0.0, float(z)) for z in range(5)]
    points.extend((1.0, 0.0, float(z)) for z in range(5, 10))
    _write_binary_xyz_ply(ply_path, points)
    tomogram = {
        "data": np.zeros((5, 10, 7, 7), dtype=np.float32),
        "resolution": 1.0,
        "center": np.array([0.0, 0.0], dtype=np.float64),
        "slice_h0": 0.0,
        "slice_dh": 1.0,
    }

    volume = pct_vertical_obstacle_volume_from_ply(
        collision_ply_path=ply_path,
        tomogram=tomogram,
        vertical_obstacle_min_slices=5,
    )

    offset_x = tomogram["data"].shape[2] // 2
    offset_y = tomogram["data"].shape[3] // 2
    assert volume.shape == (10, 7, 7)
    assert volume[0, offset_x, offset_y]
    assert not volume[6, offset_x, offset_y]
    assert volume[6, offset_x + 1, offset_y]
    assert not volume[0, offset_x + 1, offset_y]


def test_pct_robot_body_volume_samples_triangle_furniture(
    tmp_path: Path,
) -> None:
    ply_path = tmp_path / "chair.ply"
    vertices = [
        (0.0, -1.0, 0.30),
        (0.0, 1.0, 0.30),
        (0.0, 0.0, 1.00),
    ]
    _write_binary_triangle_ply(ply_path, vertices, [(0, 1, 2)])
    tomogram = {
        "data": np.zeros((5, 4, 7, 7), dtype=np.float32),
        "resolution": 1.0,
        "center": np.array([0.0, 0.0], dtype=np.float64),
        "slice_h0": 0.0,
        "slice_dh": 1.0,
    }

    volume = pct_robot_body_obstacle_volume_from_ply(
        collision_ply_path=ply_path,
        tomogram=tomogram,
        min_height_m=0.25,
        max_height_m=1.0,
    )

    offset_x = tomogram["data"].shape[2] // 2
    offset_y = tomogram["data"].shape[3] // 2
    assert volume[0, offset_x, offset_y]
    assert not volume[1, offset_x, offset_y]


def test_pct_local_map_can_overlay_vertical_collision_obstacles(tmp_path: Path) -> None:
    tomogram_path = tmp_path / "tomogram.pickle"
    walkable_path = tmp_path / "walkable.npy"
    ply_path = tmp_path / "collision.ply"
    tomogram = {
        "data": np.ones((5, 6, 7, 7), dtype=np.float32),
        "resolution": 1.0,
        "center": np.array([0.0, 0.0], dtype=np.float64),
        "slice_h0": 0.0,
        "slice_dh": 1.0,
    }
    with tomogram_path.open("wb") as stream:
        pickle.dump(tomogram, stream)
    np.save(walkable_path, np.ones((6, 7, 7), dtype=bool))
    _write_binary_xyz_ply(ply_path, [(0.0, 0.0, float(z)) for z in range(5)])

    grid = load_pct_slice_local_map(
        tomogram_path=tomogram_path,
        walkable_path=walkable_path,
        collision_ply_path=ply_path,
        z=0.0,
        vertical_obstacle_min_slices=5,
    )

    assert grid.is_occupied(*grid.world_to_grid(0.0, 0.0))
    assert not grid.is_occupied(*grid.world_to_grid(-1.0, 0.0))


def test_pct_vertical_obstacle_grid_contains_only_hard_obstacles(
    tmp_path: Path,
) -> None:
    tomogram_path = tmp_path / "tomogram.pickle"
    ply_path = tmp_path / "collision.ply"
    tomogram = {
        "data": np.zeros((5, 6, 7, 7), dtype=np.float32),
        "resolution": 1.0,
        "center": np.array([0.0, 0.0], dtype=np.float64),
        "slice_h0": 0.0,
        "slice_dh": 1.0,
    }
    with tomogram_path.open("wb") as stream:
        pickle.dump(tomogram, stream)
    _write_binary_xyz_ply(
        ply_path,
        [(0.0, 0.0, float(z)) for z in range(5)],
    )

    grid = load_pct_vertical_obstacle_grid(
        tomogram_path=tomogram_path,
        collision_ply_path=ply_path,
        vertical_obstacle_min_slices=5,
    )

    assert grid.is_occupied(*grid.world_to_grid(0.0, 0.0))
    assert not grid.is_occupied(*grid.world_to_grid(-1.0, 0.0))


def test_add_circular_keepouts_preserves_nearby_standoff_goal() -> None:
    grid = OccupancyGridMap(
        occupancy=np.zeros((20, 20), dtype=bool),
        resolution=0.1,
        origin=(-1.0, -1.0, 0.0),
    )

    updated = add_circular_keepouts(
        grid,
        centers_xy=((0.0, 0.0),),
        radius_m=0.18,
    )

    assert updated.is_occupied(*updated.world_to_grid(0.0, 0.0))
    assert not updated.is_occupied(*updated.world_to_grid(0.0, -0.5))


def _write_binary_xyz_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    array = np.asarray(points, dtype="<f4")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(array.tobytes())


def _write_binary_triangle_ply(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
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
    vertex_array = np.asarray(vertices, dtype="<f4")
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_array = np.empty(len(faces), dtype=face_dtype)
    face_array["count"] = 3
    face_array["indices"] = np.asarray(faces, dtype="<i4")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(vertex_array.tobytes())
        stream.write(face_array.tobytes())
