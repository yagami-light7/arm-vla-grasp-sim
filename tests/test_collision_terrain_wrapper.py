"""Tests for the collision-only Isaac Lab terrain wrapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from source.navigation.adapters.terrain_utils import (
    write_collision_terrain_wrapper,
    write_visual_prim_wrapper,
    write_visual_sublayer_wrapper,
)


class CollisionTerrainWrapperTest(unittest.TestCase):
    def test_wrapper_references_only_requested_collision_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "scene.usd"
            scene_usd.touch()
            wrapper = write_collision_terrain_wrapper(scene_usd, "/World/scene_collision")
            text = wrapper.read_text(encoding="utf-8")
            self.assertIn('defaultPrim = "scene_collision"', text)
            self.assertIn('def "scene_collision" (', text)
            self.assertNotIn('def Xform "scene_collision"', text)
            self.assertIn(f"@{scene_usd.resolve()}@</World/scene_collision>", text)
            self.assertNotIn("</World/go2_x5>", text)
            self.assertNotIn("f2_floor_proxy_", text)

    def test_yinluyuan_f2_profile_adds_invisible_smooth_collision_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "scene.usd"
            scene_usd.touch()
            wrapper = write_collision_terrain_wrapper(
                scene_usd,
                "/World/scene_collision",
                floor_proxy_profile="yinluyuan_f2",
            )

            text = wrapper.read_text(encoding="utf-8")
            self.assertEqual(text.count('def Cube "f2_floor_proxy_'), 4)
            self.assertIn("bool physics:collisionEnabled = 1", text)
            self.assertIn("float physxCollision:contactOffset = 0.005", text)
            self.assertIn("float physxCollision:restOffset = 0", text)
            self.assertIn('token visibility = "invisible"', text)
            self.assertIn("3.030000000", text)
            self.assertIn("0.450000000", text)
            self.assertIn("(-2.421488571, -4.973551102, 3.030000000)", text)

    def test_liangzhu_nested_collision_path_does_not_add_old_floor_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "liangzhu.usda"
            scene_usd.touch()
            prim_path = "/World/PhysicsScene/CollisionScene/LiangzhuCollision"

            wrapper = write_collision_terrain_wrapper(
                scene_usd,
                prim_path,
                floor_proxy_profile=None,
                source_prim_is_mesh=True,
            )

            text = wrapper.read_text(encoding="utf-8")
            self.assertIn(f"@{scene_usd.resolve()}@<{prim_path}>", text)
            self.assertIn('def Xform "scene_collision"', text)
            self.assertIn('def Mesh "collision_mesh"', text)
            self.assertNotIn("f2_floor_proxy_", text)

    def test_collision_wrapper_rejects_unknown_floor_proxy_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "scene.usd"
            scene_usd.touch()

            with self.assertRaisesRegex(ValueError, "未知碰撞地面代理"):
                write_collision_terrain_wrapper(
                    scene_usd,
                    floor_proxy_profile="unknown",
                )

    def test_visual_wrapper_references_only_requested_visual_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "scene.usd"
            scene_usd.touch()
            wrapper = write_visual_prim_wrapper(
                scene_usd,
                "/World/gauss",
                excluded_prim_paths=("/World/scene_collision", "/World/go2_x5"),
            )
            text = wrapper.read_text(encoding="utf-8")
            self.assertIn('defaultPrim = "visual_scene"', text)
            self.assertIn(f"@{scene_usd.resolve()}@", text)
            self.assertNotIn(f"@{scene_usd.resolve()}@</World/gauss>", text)
            self.assertIn('over "scene_collision"', text)
            self.assertIn('over "go2_x5"', text)
            self.assertIn("active = false", text)
            self.assertNotIn("</World/go2_x5>", text)

    def test_visual_sublayer_wrapper_preserves_complete_scene_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "scene.usd"
            scene_usd.touch()
            wrapper = write_visual_sublayer_wrapper(
                scene_usd,
                "/World/gauss",
                excluded_prim_paths=("/World/scene_collision", "/World/go2_x5"),
            )
            text = wrapper.read_text(encoding="utf-8")
            self.assertIn('defaultPrim = "World"', text)
            self.assertIn("subLayers = [", text)
            self.assertIn(f"@{scene_usd.resolve()}@", text)
            self.assertNotIn("prepend references", text)
            self.assertIn('over "scene_collision"', text)
            self.assertIn('over "go2_x5"', text)
            self.assertIn("active = false", text)
            self.assertIn('over "gauss"', text)
            self.assertIn('token visibility = "inherited"', text)

    def test_visual_sublayer_wrapper_reveals_nested_liangzhu_visual_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "liangzhu.usda"
            scene_usd.touch()
            wrapper = write_visual_sublayer_wrapper(
                scene_usd,
                "/World/VisualScene/GaussianScene",
                excluded_prim_paths=(
                    "/World/PhysicsScene/CollisionScene/LiangzhuCollision",
                ),
                include_visual_prim=True,
            )

            text = wrapper.read_text(encoding="utf-8")
            assert 'over "VisualScene"' in text
            assert 'over "GaussianScene"' in text
            assert text.count('token visibility = "inherited"') == 2
            assert 'over "PhysicsScene" (' in text
            assert "active = false" in text

    def test_visual_sublayer_wrapper_can_disable_heavy_visual_prim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_usd = Path(tmp_dir) / "scene.usd"
            scene_usd.touch()
            wrapper = write_visual_sublayer_wrapper(
                scene_usd,
                "/World/gauss",
                excluded_prim_paths=("/World/scene_collision",),
                include_visual_prim=False,
            )

            text = wrapper.read_text(encoding="utf-8")
            self.assertIn('over "gauss"', text)
            self.assertIn("active = false", text)


if __name__ == "__main__":
    unittest.main()
