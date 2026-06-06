"""Tests for occupancy-grid indexing and rotated ROS-style map origins."""

from __future__ import annotations

import math
import unittest

import numpy as np

from source.navigation.navlib import OccupancyGridMap


class NavMapTransformTest(unittest.TestCase):
    def test_cell_center_round_trip_with_rotated_origin(self) -> None:
        grid = OccupancyGridMap(
            occupancy=np.zeros((2, 3), dtype=bool),
            resolution=1.0,
            origin=(10.0, 20.0, math.pi / 2.0),
        )
        world_xy = grid.grid_to_world(1, 0)
        self.assertTrue(np.allclose(world_xy, (9.5, 20.5)))
        self.assertEqual(grid.world_to_grid(*world_xy), (1, 0))

    def test_out_of_bounds_is_treated_as_occupied(self) -> None:
        grid = OccupancyGridMap(np.zeros((2, 2), dtype=bool), 0.5, (0.0, 0.0, 0.0))
        self.assertTrue(grid.is_occupied(-1, 0))
        self.assertTrue(grid.is_occupied(2, 0))

    def test_obstacle_distance_transform_matches_grid_distance(self) -> None:
        try:
            import scipy.ndimage  # noqa: F401
        except ImportError:
            self.skipTest("SciPy is not installed.")

        occupancy = np.zeros((7, 7), dtype=bool)
        occupancy[3, 3] = True
        grid = OccupancyGridMap(occupancy, 0.1, (0.0, 0.0, 0.0))

        self.assertEqual(grid.distance_to_obstacle(3, 3), 0.0)
        self.assertAlmostEqual(grid.distance_to_obstacle(3, 5), 0.2, places=6)
        self.assertAlmostEqual(grid.distance_to_obstacle(1, 1), math.sqrt(8) * 0.1, places=6)

    def test_inflate_marks_neighbor_cells(self) -> None:
        occupancy = np.zeros((5, 5), dtype=bool)
        occupancy[2, 2] = True
        inflated = OccupancyGridMap(occupancy, 0.1, (0.0, 0.0, 0.0)).inflate(0.1)
        self.assertTrue(inflated.is_occupied(2, 1))
        self.assertTrue(inflated.is_occupied(2, 3))
        self.assertFalse(inflated.is_occupied(1, 1))

    def test_inflate_keeps_clearance_from_map_boundary(self) -> None:
        occupancy = np.zeros((5, 5), dtype=bool)
        inflated = OccupancyGridMap(occupancy, 0.1, (0.0, 0.0, 0.0)).inflate(0.1)
        self.assertTrue(inflated.is_occupied(0, 2))
        self.assertFalse(inflated.is_occupied(1, 2))


if __name__ == "__main__":
    unittest.main()
