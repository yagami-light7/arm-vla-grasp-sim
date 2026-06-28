"""Tests for fixed-schema multi-phase episode recording."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from source.data import EPISODE_COLUMNS, EpisodeRecorder
from source.data.lerobot_converter import validate_episode


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

    def test_summary_is_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = EpisodeRecorder(tmp_dir, 4, 5)
            recorder.write_summary({"success": False, "failure_reason": "nav_timeout"})
            payload = json.loads((Path(tmp_dir) / "4/5/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["failure_reason"], "nav_timeout")

    def test_rejects_grasp_target_far_from_navigation_base(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "grasp_target_unreachable"):
            from source.manipulation.grasp_pipeline import GraspPipeline

            GraspPipeline._validate_target_workspace(
                {
                    "diagnostics": {
                        "target_workspace_base": {
                            "grasp": {"xy_radius_m": 4.6},
                            "pregrasp": {"radius_3d_m": 4.7},
                        }
                    }
                }
            )

    def test_rejects_grasp_target_just_outside_x5_workspace(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "grasp_xy_radius=0.681"):
            from source.manipulation.grasp_pipeline import GraspPipeline

            GraspPipeline._validate_target_workspace(
                {
                    "diagnostics": {
                        "target_workspace_base": {
                            "grasp": {"xy_radius_m": 0.681},
                            "pregrasp": {"radius_3d_m": 0.60},
                        }
                    }
                }
            )

    def test_accepts_grasp_target_inside_x5_workspace(self) -> None:
        from source.manipulation.grasp_pipeline import GraspPipeline

        GraspPipeline._validate_target_workspace(
            {
                "diagnostics": {
                    "target_workspace_base": {
                        "grasp": {"xy_radius_m": 0.62},
                        "pregrasp": {"radius_3d_m": 0.70},
                    }
                }
            }
        )

    def test_grasp_pipeline_checks_workspace_before_starting_curobo(self) -> None:
        from source.manipulation.grasp_pipeline import (
            GraspPipeline,
            GraspPipelineConfig,
            GraspTask,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target_json = root / "pick_target.json"
            target_json.write_text(
                json.dumps(
                    {
                        "diagnostics": {
                            "target_workspace_base": {
                                "grasp": {"xy_radius_m": 0.805},
                                "pregrasp": {"radius_3d_m": 0.783},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            task = GraspTask(
                object_prim_path="/World/apple",
                use_planner_server=False,
                state_json=str(root / "pick_state.json"),
                target_json=str(target_json),
                plan_json=str(root / "pick_plan.json"),
            )

            with self.assertRaisesRegex(RuntimeError, "grasp_xy_radius=0.805"):
                GraspPipeline(GraspPipelineConfig(workspace=root)).plan(task)


if __name__ == "__main__":
    unittest.main()
