"""Tests for the collision-only Isaac Lab terrain wrapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from source.navigation.adapters.terrain_utils import write_collision_terrain_wrapper


class CollisionTerrainWrapperTest(unittest.TestCase):
    def test_wrapper_references_only_requested_collision_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "scene.usd"
            scene_usd.touch()
            wrapper = write_collision_terrain_wrapper(scene_usd, "/World/scene_collision")
            text = wrapper.read_text(encoding="utf-8")
            self.assertIn('defaultPrim = "scene_collision"', text)
            self.assertIn(f"@{scene_usd.resolve()}@</World/scene_collision>", text)
            self.assertNotIn("</World/go2_x5>", text)


if __name__ == "__main__":
    unittest.main()
