"""GUI viewport 配置的无 Isaac 依赖测试。"""

from __future__ import annotations

import unittest

from source.simulation.viewport import candidate_stage_camera_paths


class SimulationViewportTest(unittest.TestCase):
    def test_default_camera_supports_case_and_reference_fallbacks(self) -> None:
        candidates = candidate_stage_camera_paths("/World/Camera_main")

        self.assertEqual(candidates[0], "/World/Camera_main")
        self.assertIn("/World/camera_main", candidates)
        self.assertIn("/World/Camera1", candidates)
        self.assertIn("/World/Camera_font", candidates)
        self.assertIn("/World/camera1", candidates)
        self.assertIn("/World/nav_visual_scene/Camera_main", candidates)
        self.assertIn("/World/gauss/Camera1", candidates)
        self.assertIn("/World/gauss/Camera_font", candidates)
        self.assertIn("/World/gauss/camera1", candidates)
        self.assertIn("/World/contact_visual_scene/camera1", candidates)
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_lowercase_camera_can_fall_back_to_baseline_path(self) -> None:
        candidates = candidate_stage_camera_paths("/World/camera_main")

        self.assertIn("/World/Camera_main", candidates)
        self.assertIn("/World/contact_visual_scene/camera_main", candidates)

    def test_camera_numbered_names_are_preferred_for_current_scene(self) -> None:
        candidates = candidate_stage_camera_paths("/World/Camera1")

        self.assertEqual(candidates[0], "/World/Camera1")
        self.assertIn("/World/Camera2", candidates)
        self.assertIn("/World/Camera3", candidates)
        self.assertIn("/World/camera1", candidates)
        self.assertIn("/World/gauss/camera1", candidates)
        self.assertIn("/World/gauss/Camera1", candidates)
        self.assertIn("/World/nav_visual_scene/Camera1", candidates)

    def test_camera_font_name_is_supported_for_current_scene(self) -> None:
        candidates = candidate_stage_camera_paths("/World/Camera_font")

        self.assertEqual(candidates[0], "/World/Camera_font")
        self.assertIn("/World/camera_font", candidates)
        self.assertIn("/World/gauss/Camera_font", candidates)
        self.assertIn("/World/contact_visual_scene/camera_font", candidates)


if __name__ == "__main__":
    unittest.main()
