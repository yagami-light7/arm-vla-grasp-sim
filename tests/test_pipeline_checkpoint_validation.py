"""Tests for navigation coordinator checkpoint preflight validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.pipeline.run_nav_then_pick import _validate_checkpoint


class PipelineCheckpointValidationTest(unittest.TestCase):
    def test_existing_local_checkpoint_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "model_8500.pt"
            checkpoint.touch()
            self.assertEqual(_validate_checkpoint(str(checkpoint)), str(checkpoint.resolve()))

    def test_missing_checkpoint_fails_before_isaaclab_launch(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Do not use the documentation placeholder"):
            _validate_checkpoint("/你的实际路径/model_8500.pt")


if __name__ == "__main__":
    unittest.main()
