"""Tests for terminal yaw-alignment command helpers."""

from __future__ import annotations

import math
import unittest

from source.navigation.adapters.yaw_align import (
    TerminalPoseConfig,
    YawAlignConfig,
    YawAlignStallDetector,
    body_goal_components,
    body_goal_forward_projection,
    compute_yaw_align_command,
    compute_terminal_pose_command,
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

    def test_terminal_command_polishes_yaw_inside_position_acceptance(self) -> None:
        command = compute_terminal_pose_command(
            body_goal_x=-0.04,
            body_goal_y=-0.02,
            yaw_error=0.52,
            distance_to_goal=0.06,
            config=TerminalPoseConfig(
                yaw_kp=2.0,
                yaw_min_wz=0.40,
                yaw_max_wz=1.00,
                yaw_slowdown_error=0.65,
                yaw_slowdown_max_wz=0.45,
                yaw_polish_gait_vx=0.08,
                yaw_polish_min_wz=0.45,
                yaw_polish_max_wz=0.55,
            ),
        )
        self.assertEqual(command[0], 0.08)
        self.assertEqual(command[1], 0.0)
        self.assertEqual(command[2], 0.55)

    def test_terminal_command_keeps_arc_motion_for_large_yaw_inside_acceptance(self) -> None:
        command = compute_terminal_pose_command(
            body_goal_x=0.06,
            body_goal_y=-0.16,
            yaw_error=2.0,
            distance_to_goal=0.17,
            config=TerminalPoseConfig(
                position_tolerance=0.08,
                position_acceptance_tolerance=0.18,
                yaw_tolerance=0.08,
                position_kp=0.1,
                max_vx=0.25,
                min_vx=0.20,
                lateral_kp=0.9,
                max_vy=0.45,
                min_vy=0.15,
                lateral_deadband=0.03,
                yaw_kp=1.2,
                yaw_min_wz=0.40,
                yaw_max_wz=0.75,
                yaw_polish_min_wz=0.45,
                yaw_polish_max_wz=0.55,
                large_yaw_error=1.0,
            ),
        )
        self.assertGreaterEqual(command[0], 0.20)
        self.assertLessEqual(command[1], -0.15)
        self.assertEqual(command[2], 0.55)

    def test_terminal_command_keeps_arc_motion_for_small_yaw_polish_outside_tight_position(self) -> None:
        command = compute_terminal_pose_command(
            body_goal_x=0.16,
            body_goal_y=-0.06,
            yaw_error=0.24,
            distance_to_goal=0.16,
            config=TerminalPoseConfig(
                position_tolerance=0.08,
                position_acceptance_tolerance=0.18,
                yaw_tolerance=0.08,
                position_kp=0.8,
                max_vx=0.60,
                min_vx=0.35,
                lateral_kp=0.9,
                max_vy=0.35,
                min_vy=0.15,
                lateral_deadband=0.015,
                yaw_kp=1.4,
                yaw_min_wz=0.50,
                yaw_max_wz=0.85,
                yaw_polish_gait_vx=0.08,
                yaw_polish_min_wz=0.45,
                yaw_polish_max_wz=0.55,
            ),
        )
        self.assertGreater(command[0], 0.08)
        self.assertLessEqual(command[1], -0.15)
        self.assertGreaterEqual(command[2], 0.45)
        self.assertLessEqual(command[2], 0.55)

    def test_terminal_recovery_keeps_polish_yaw_speed_inside_position_acceptance(self) -> None:
        command = compute_terminal_pose_command(
            body_goal_x=0.16,
            body_goal_y=-0.06,
            yaw_error=0.24,
            distance_to_goal=0.16,
            recovery=True,
            config=TerminalPoseConfig(
                position_tolerance=0.08,
                position_acceptance_tolerance=0.18,
                yaw_tolerance=0.08,
                position_kp=0.8,
                max_vx=0.60,
                min_vx=0.35,
                lateral_kp=0.9,
                max_vy=0.35,
                min_vy=0.15,
                lateral_deadband=0.015,
                yaw_kp=1.4,
                yaw_min_wz=0.50,
                yaw_max_wz=0.85,
                recovery_yaw_max_wz=0.32,
                yaw_polish_gait_vx=0.08,
                yaw_polish_min_wz=0.45,
                yaw_polish_max_wz=0.55,
            ),
        )
        self.assertGreater(command[0], 0.08)
        self.assertLessEqual(command[1], -0.15)
        self.assertGreaterEqual(command[2], 0.45)
        self.assertLessEqual(command[2], 0.55)

    def test_terminal_command_allows_small_reverse_recenter(self) -> None:
        command = compute_terminal_pose_command(
            body_goal_x=-0.12,
            body_goal_y=0.02,
            yaw_error=0.20,
            distance_to_goal=0.12,
            config=TerminalPoseConfig(
                position_tolerance=0.08,
                position_acceptance_tolerance=0.10,
                max_vx=0.35,
                min_vx=0.08,
                allow_reverse=True,
                lateral_deadband=0.03,
            ),
        )
        self.assertLess(command[0], 0.0)
        self.assertEqual(command[1], 0.0)

    def test_terminal_recovery_reduces_yaw_and_keeps_gait_active(self) -> None:
        command = compute_terminal_pose_command(
            body_goal_x=0.0,
            body_goal_y=0.0,
            yaw_error=-0.40,
            distance_to_goal=0.04,
            recovery=True,
            config=TerminalPoseConfig(
                position_tolerance=0.01,
                position_acceptance_tolerance=0.02,
                yaw_kp=2.0,
                yaw_max_wz=1.0,
                recovery_yaw_max_wz=0.30,
                recovery_gait_vx=0.07,
            ),
        )
        self.assertEqual(command[0], 0.07)
        self.assertEqual(command[1], 0.0)
        self.assertEqual(command[2], -0.30)

    def test_carry_forward_terminal_reorients_instead_of_strafing_forever(self) -> None:
        """复现真实 place 卡点：最终 yaw 已合格，但目标仍主要位于机身侧方。"""

        command = compute_terminal_pose_command(
            body_goal_x=0.031,
            body_goal_y=0.099,
            yaw_error=0.148,
            distance_to_goal=0.104,
            config=TerminalPoseConfig(
                position_tolerance=0.05,
                yaw_tolerance=0.15,
                max_vx=0.25,
                yaw_max_wz=0.65,
                prefer_forward_translation=True,
            ),
        )

        self.assertEqual(command[0], 0.0)
        self.assertEqual(command[1], 0.0)
        self.assertEqual(command[2], 0.50)

    def test_carry_forward_terminal_advances_after_position_heading_is_aligned(self) -> None:
        command = compute_terminal_pose_command(
            body_goal_x=0.10,
            body_goal_y=0.01,
            yaw_error=-0.40,
            distance_to_goal=math.hypot(0.10, 0.01),
            config=TerminalPoseConfig(
                position_tolerance=0.05,
                yaw_tolerance=0.15,
                max_vx=0.25,
                prefer_forward_translation=True,
            ),
        )

        self.assertEqual(command[0], 0.16)
        self.assertEqual(command[1], 0.0)
        self.assertAlmostEqual(command[2], 2.0 * math.atan2(0.01, 0.10))

    def test_goal_yaw_terminal_translates_laterally_without_turning_toward_xy(self) -> None:
        command = compute_terminal_pose_command(
            body_goal_x=0.031,
            body_goal_y=0.099,
            yaw_error=0.148,
            distance_to_goal=math.hypot(0.031, 0.099),
            config=TerminalPoseConfig(
                position_tolerance=0.05,
                yaw_tolerance=0.15,
                max_vx=0.25,
                max_vy=0.18,
                yaw_max_wz=0.30,
                prefer_goal_yaw_translation=True,
            ),
        )

        self.assertGreater(command[0], 0.0)
        self.assertAlmostEqual(command[1], 0.16)
        self.assertAlmostEqual(command[2], 0.296)
        # 旧 forward-only 模式会先原地朝 atan2(y, x) 大幅转向；新模式
        # 只修正相对最终 yaw 的 0.148 rad 误差。
        self.assertLess(command[2], 0.30)


if __name__ == "__main__":
    unittest.main()
