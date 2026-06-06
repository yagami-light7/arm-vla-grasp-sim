"""Tests for terminal yaw-alignment command helpers."""

from __future__ import annotations

import unittest

from source.navigation.adapters.yaw_align import (
    YawAlignConfig,
    YawAlignStallDetector,
    body_goal_components,
    body_goal_forward_projection,
    compute_yaw_align_command,
)


class YawAlignTest(unittest.TestCase):
    def test_command_uses_stronger_yaw_and_activation_vx(self) -> None:
        command = compute_yaw_align_command(
            yaw_error=0.60,
            yaw_tolerance=0.15,
            body_goal_x=0.10,
            config=YawAlignConfig(kp=2.0, min_wz=0.45, max_wz=0.80, activation_vx=0.04),
        )
        self.assertEqual(command[0], 0.04)
        self.assertEqual(command[1], 0.0)
        self.assertEqual(command[2], 0.80)

    def test_command_stops_inside_tolerance(self) -> None:
        command = compute_yaw_align_command(
            yaw_error=0.10,
            yaw_tolerance=0.15,
            body_goal_x=0.10,
            config=YawAlignConfig(),
        )
        self.assertEqual(command, (0.0, 0.0, 0.0))

    def test_command_keeps_gait_active_until_inside_tolerance(self) -> None:
        command = compute_yaw_align_command(
            yaw_error=0.20,
            yaw_tolerance=0.15,
            body_goal_x=0.05,
            config=YawAlignConfig(),
        )
        self.assertEqual(command[0], 0.16)
        self.assertGreater(command[2], 0.0)

    def test_command_keeps_minimum_wz_near_tolerance(self) -> None:
        command = compute_yaw_align_command(
            yaw_error=0.19,
            yaw_tolerance=0.15,
            body_goal_x=0.05,
            config=YawAlignConfig(),
        )
        self.assertEqual(command[0], 0.16)
        self.assertEqual(command[2], 0.55)

    def test_reverse_activation_is_disabled_by_default(self) -> None:
        command = compute_yaw_align_command(
            yaw_error=-0.60,
            yaw_tolerance=0.15,
            body_goal_x=-0.10,
            config=YawAlignConfig(activation_vx=0.04, allow_reverse=False),
        )
        self.assertEqual(command[0], 0.0)
        self.assertLess(command[2], 0.0)

    def test_body_goal_forward_projection(self) -> None:
        self.assertAlmostEqual(body_goal_forward_projection((1.0, 2.0, 0.0), (1.5, 3.0)), 0.5)

    def test_body_goal_components_include_lateral_error(self) -> None:
        body_x, body_y = body_goal_components((1.074, 1.138, 2.194), (1.267, 1.266))
        self.assertAlmostEqual(body_x, -0.009, places=2)
        self.assertAlmostEqual(body_y, -0.232, places=2)

    def test_stall_detector_triggers_when_error_stops_decreasing(self) -> None:
        detector = YawAlignStallDetector(window_steps=3, min_progress_rad=0.05)
        self.assertFalse(detector.update(0.60)[0])
        self.assertFalse(detector.update(0.59)[0])
        stalled, diagnostics = detector.update(0.58)
        self.assertTrue(stalled)
        self.assertAlmostEqual(diagnostics.error_reduction, 0.02)

    def test_stall_detector_accepts_real_yaw_progress(self) -> None:
        detector = YawAlignStallDetector(window_steps=3, min_progress_rad=0.05)
        detector.update(0.60)
        detector.update(0.54)
        stalled, diagnostics = detector.update(0.48)
        self.assertFalse(stalled)
        self.assertAlmostEqual(diagnostics.error_reduction, 0.12)


if __name__ == "__main__":
    unittest.main()
