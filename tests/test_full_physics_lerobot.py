"""Tests for DWA-compatible full-physics recording and LeRobot export."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from source.interfaces import RobotAction, SimulationState, StepRecord
from source.recording import (
    DWA_CSV_COLUMNS,
    JsonlEpisodeRecorder,
    LeRobotRecordingConfig,
    discover_recorded_episodes,
)


def _state(step_index: int, image: np.ndarray | None) -> SimulationState:
    joint_names = (
        "FR_hip_joint",
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
        "arm_joint7",
        "arm_joint8",
    )
    return SimulationState(
        step_index=step_index,
        timestamp=step_index * 0.02,
        robot_root_pose=(1.0, 2.0, 0.35, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.1, 0.0, 0.0, 0.0, 0.0, 0.2),
        joint_positions=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.02, 0.04),
        joint_velocities=(0.0,) * 9,
        tcp_pose=(1.2, 2.1, 0.8, 1.0, 0.0, 0.0, 0.0),
        object_pose=(0.9, 1.2, 0.82, 1.0, 0.0, 0.0, 0.0),
        object_velocity=(0.0,) * 6,
        camera_images={} if image is None else {"front": image},
        metadata={
            "joint_names": joint_names,
            "body_linear_velocity": (0.3, -0.1, 0.02),
        },
    )


class FullPhysicsLeRobotTest(unittest.TestCase):
    def test_records_dwa_csv_and_jpeg_every_ten_control_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_000000"
            recorder = JsonlEpisodeRecorder(
                episode_dir,
                lerobot_config=LeRobotRecordingConfig(
                    enabled=True,
                    control_dt=0.02,
                    fps=5,
                    image_height=480,
                    image_width=640,
                    jpeg_quality=90,
                ),
            )
            recorder.save_task(
                type(
                    "Spec",
                    (),
                    {
                        "raw_task": {
                            "task_id": 2,
                            "instruction": "Pick and place the apple.",
                        }
                    },
                )()
            )
            image = np.zeros((480, 640, 4), dtype=np.uint8)
            image[..., 0] = 255
            for step_index in (1, 10, 11):
                state = _state(step_index, image)
                recorder.record_step(
                    StepRecord(
                        step_index=step_index,
                        timestamp=state.timestamp,
                        pipeline_state="exec_nav_to_pick",
                        observation=state,
                        action=RobotAction(base_velocity=(0.3, 0.0, 0.1), source="nav"),
                        post_step_observation=state,
                    )
                )

            with (episode_dir / "data.csv").open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(tuple(rows[0]), DWA_CSV_COLUMNS)
            self.assertEqual(rows[0]["前摄像头图像"], "camera0_00000.jpg")
            self.assertAlmostEqual(float(rows[0]["时间戳(秒)"]), 0.0)
            self.assertAlmostEqual(float(rows[1]["时间戳(秒)"]), 0.2)
            self.assertAlmostEqual(float(rows[0]["线速度X"]), 0.3)
            self.assertAlmostEqual(float(rows[0]["线速度Z"]), 0.02)
            self.assertAlmostEqual(float(rows[0]["关节6"]), 6.0)
            self.assertAlmostEqual(float(rows[0]["夹爪"]), 0.03)

            image_path = episode_dir / "images/front/camera0_00000.jpg"
            self.assertTrue(image_path.is_file())
            with Image.open(image_path) as saved_image:
                self.assertEqual(saved_image.size, (640, 480))
            samples = [
                json.loads(line)
                for line in (episode_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(samples[0]["action"]), 10)
            self.assertEqual(len(samples[0]["object_state"]), 13)
            self.assertEqual(len(samples[0]["tcp_pose"]), 7)

            frames = [
                json.loads(line)
                for line in (episode_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(frames[-1]["post_step_observation"]["camera_images"], ["front"])
            self.assertNotIn("255", json.dumps(frames[-1]["post_step_observation"]["camera_images"]))

            export = recorder.prepare_lerobot_export()
            self.assertTrue(export["raw_episode_ready"])
            self.assertEqual(export["sampled_frame_count"], 2)
            self.assertEqual(export["capture_every_n_steps"], 10)
            self.assertIn("dataset_path", export)
            if export["lerobot_exported"]:
                import pyarrow.parquet as pq

                dataset_path = Path(export["dataset_path"])
                parquet_path = dataset_path / "data/chunk-000/episode_000000.parquet"
                video_path = (
                    dataset_path
                    / "videos/chunk-000/observation.images.front/episode_000000.mp4"
                )
                info = json.loads(
                    (dataset_path / "meta/info.json").read_text(encoding="utf-8")
                )
                table = pq.read_table(parquet_path)
                self.assertEqual(table.num_rows, 2)
                self.assertEqual(len(table["observation.state"][0].as_py()), 17)
                self.assertEqual(len(table["observation.base_linear_velocity"][0].as_py()), 3)
                self.assertEqual(len(table["observation.object_state"][0].as_py()), 13)
                self.assertEqual(len(table["observation.tcp_pose"][0].as_py()), 7)
                self.assertEqual(len(table["action"][0].as_py()), 10)
                self.assertFalse(table["next.done"][0].as_py())
                self.assertTrue(table["next.done"][1].as_py())
                self.assertTrue(video_path.is_file())
                self.assertEqual(info["fps"], 5)
                self.assertEqual(info["features"]["observation.images.front"]["shape"], [480, 640, 3])
                self.assertEqual(info["features"]["action"]["shape"], [10])

    def test_discovers_only_successful_raw_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for index, success in enumerate((True, False)):
                episode_dir = root / f"episode_{index:06d}"
                episode_dir.mkdir()
                (episode_dir / "task.json").write_text(
                    json.dumps({"instruction": "task"}),
                    encoding="utf-8",
                )
                (episode_dir / "data.csv").write_text(
                    ",".join(DWA_CSV_COLUMNS) + "\n",
                    encoding="utf-8",
                )
                (episode_dir / "summary.json").write_text(
                    json.dumps({"success": success}),
                    encoding="utf-8",
                )

            episodes = discover_recorded_episodes(root, require_success=True)
            self.assertEqual(episodes, [root / "episode_000000"])


if __name__ == "__main__":
    unittest.main()
