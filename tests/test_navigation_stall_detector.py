"""Tests for progress-based navigation stall detection."""

from __future__ import annotations

import unittest

from source.navigation.adapters.stall_detector import NavigationStallDetector


class NavigationStallDetectorTest(unittest.TestCase):
    def test_intermittent_slow_command_does_not_reset_stall_history(self) -> None:
        detector = NavigationStallDetector(window_steps=10, min_progress_m=0.05, min_forward_ratio=0.75)
        stalled = False
        for step in range(10):
            cmd_vx = 0.04 if step in {2, 7} else 0.30
            stalled, diagnostics = detector.update(0.0, 0.0, cmd_vx)
        self.assertTrue(stalled)
        self.assertEqual(diagnostics.forward_command_ratio, 0.8)

    def test_real_progress_does_not_trigger_stall(self) -> None:
        detector = NavigationStallDetector(window_steps=10, min_progress_m=0.05)
        for step in range(10):
            stalled, _ = detector.update(step * 0.02, 0.0, 0.30)
        self.assertFalse(stalled)

    def test_rotation_only_window_does_not_trigger_stall(self) -> None:
        detector = NavigationStallDetector(window_steps=10, min_progress_m=0.05)
        for _ in range(10):
            stalled, _ = detector.update(0.0, 0.0, 0.0)
        self.assertFalse(stalled)

    def test_sparse_forward_commands_still_trigger_stall(self) -> None:
        detector = NavigationStallDetector(window_steps=10, min_progress_m=0.05)
        for step in range(10):
            cmd_vx = 0.30 if step in {1, 5, 9} else 0.0
            stalled, diagnostics = detector.update(0.0, 0.0, cmd_vx)
        self.assertTrue(stalled)
        self.assertEqual(diagnostics.forward_command_ratio, 0.3)


if __name__ == "__main__":
    unittest.main()
