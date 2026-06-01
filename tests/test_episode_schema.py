"""Tests for fixed-schema multi-phase episode recording."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from source.data import EPISODE_COLUMNS, EpisodeRecorder
from source.data.lerobot_converter import validate_episode
from source.manipulation import GraspPipeline


class EpisodeSchemaTest(unittest.TestCase):
    def test_records_multiple_phases_and_validates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = EpisodeRecorder(tmp_dir, 1, 2)
            recorder.save_task({"task_id": 1, "episode_id": 2})
            row = recorder.record("nav", {"cmd_vx": 0.2, "cmd_wz": 0.1}, front_image=b"jpeg")
            recorder.record("yaw_align", {"cmd_wz": -0.1})
            recorder.write_summary({"success": True, "failure_reason": ""})

            self.assertEqual(row["front_image"], "nav/images/front/000000.jpg")
            self.assertTrue((Path(tmp_dir) / "1/2/nav/images/front/000000.jpg").exists())
            with recorder.phase_csv("nav").open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(EPISODE_COLUMNS))
            manifest = validate_episode(recorder.episode_dir)
            self.assertEqual(manifest["phases"]["nav"]["rows"], 1)
            self.assertEqual(manifest["phases"]["yaw_align"]["rows"], 1)

    def test_grasp_execution_logs_are_adapted_to_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = EpisodeRecorder(tmp_dir, 1, 3)
            recorder.save_task({"task_id": 1, "episode_id": 3})
            pipeline = GraspPipeline(recorder=recorder)
            pipeline._record_execution(
                {
                    "execution_logs": [
                        {
                            "type": "motion",
                            "time": [0.0],
                            "target_q_arm": [[1, 2, 3, 4, 5, 6]],
                            "actual_q_arm": [[0, 1, 2, 3, 4, 5]],
                        },
                        {
                            "type": "gripper",
                            "time": [0.0],
                            "target_position": [0.0, 0.0],
                            "actual_q_gripper": [[0.01, 0.01]],
                        },
                    ]
                }
            )
            recorder.write_summary({"success": True})
            rows = recorder.phase_csv("grasp").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertIn("arm_action_joint6", rows[0])

    def test_summary_is_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = EpisodeRecorder(tmp_dir, 4, 5)
            recorder.write_summary({"success": False, "failure_reason": "nav_timeout"})
            payload = json.loads((Path(tmp_dir) / "4/5/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["failure_reason"], "nav_timeout")


if __name__ == "__main__":
    unittest.main()
