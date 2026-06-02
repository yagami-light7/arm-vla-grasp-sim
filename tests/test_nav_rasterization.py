"""Tests for conservative top-down collision rasterization."""

from __future__ import annotations

import unittest

import numpy as np

from source.navigation.navlib import rasterize_triangles_xy


class NavRasterizationTest(unittest.TestCase):
    def test_vertical_wall_projection_is_occupied(self) -> None:
        wall_triangle = np.array(
            [
                [1.0, 0.5, 0.0],
                [1.0, 1.5, 0.0],
                [1.0, 1.5, 1.0],
            ],
            dtype=float,
        )
        grid = rasterize_triangles_xy(
            [wall_triangle],
            resolution=0.1,
            bounds=(0.0, 2.0, 0.0, 2.0),
        )
        for y in (0.5, 1.0, 1.5):
            self.assertTrue(grid.is_occupied(*grid.world_to_grid(1.0, y)))

    def test_triangle_interior_is_occupied(self) -> None:
        triangle = np.array(
            [
                [0.5, 0.5, 0.0],
                [1.5, 0.5, 0.0],
                [0.5, 1.5, 0.0],
            ],
            dtype=float,
        )
        grid = rasterize_triangles_xy(
            [triangle],
            resolution=0.1,
            bounds=(0.0, 2.0, 0.0, 2.0),
        )
        self.assertTrue(grid.is_occupied(*grid.world_to_grid(0.75, 0.75)))


if __name__ == "__main__":
    unittest.main()
