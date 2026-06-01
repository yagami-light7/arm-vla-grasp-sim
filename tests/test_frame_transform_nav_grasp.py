"""Tests for shared navigation-to-grasp frame helpers."""

from __future__ import annotations

import math
import unittest

import numpy as np

from source.navigation.adapters.frame_utils import (
    body_velocity_to_world,
    pose_xyyaw_to_matrix,
    world_point_to_base,
    world_velocity_to_body,
    wrap_yaw,
    yaw_to_quat_wxyz,
)


class FrameTransformNavGraspTest(unittest.TestCase):
    def test_yaw_wrap(self) -> None:
        self.assertAlmostEqual(wrap_yaw(math.pi), -math.pi)
        self.assertAlmostEqual(wrap_yaw(-math.pi - 0.1), math.pi - 0.1)

    def test_body_world_velocity_round_trip(self) -> None:
        velocity_world = body_velocity_to_world(1.0, 0.0, math.pi / 2.0)
        self.assertTrue(np.allclose(velocity_world, (0.0, 1.0)))
        self.assertTrue(np.allclose(world_velocity_to_body(*velocity_world, math.pi / 2.0), (1.0, 0.0)))

    def test_navigation_base_pose_transforms_world_object_point(self) -> None:
        transform_world_base = pose_xyyaw_to_matrix(1.0, 2.0, math.pi / 2.0, z=0.4)
        point_base = world_point_to_base(transform_world_base, (1.0, 3.0, 0.6))
        self.assertTrue(np.allclose(point_base, (1.0, 0.0, 0.2)))

    def test_yaw_quaternion_uses_wxyz(self) -> None:
        self.assertTrue(np.allclose(yaw_to_quat_wxyz(math.pi), (0.0, 0.0, 0.0, 1.0), atol=1.0e-7))


if __name__ == "__main__":
    unittest.main()
