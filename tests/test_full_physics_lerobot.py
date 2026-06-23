"""Tests for DWA-compatible full-physics recording and LeRobot export."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
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
            "body_velocity": (0.3, -0.1, 0.2),
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
                    dataset_fps=5,
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
            self.assertEqual(rows[0]["腕部摄像头图像"], "")
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
            self.assertEqual(samples[0]["pipeline_state"], "exec_nav_to_pick")
            self.assertEqual(samples[0]["base_velocity"], [0.3, -0.1, 0.2])
            self.assertEqual(len(samples[0]["action"]), 11)
            self.assertEqual(len(samples[0]["object_state"]), 13)
            self.assertEqual(len(samples[0]["tcp_pose"]), 7)
            self.assertEqual(
                samples[0]["camera_frames"]["front"]["feature_key"],
                "observation.images.front",
            )

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
            self.assertTrue(export["lerobot_exported"], export)
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
            self.assertEqual(len(table["observation.base_velocity"][0].as_py()), 3)
            self.assertEqual(len(table["observation.object_state"][0].as_py()), 13)
            self.assertEqual(len(table["observation.tcp_pose"][0].as_py()), 7)
            self.assertEqual(table["pipeline_state"][0].as_py(), "exec_nav_to_pick")
            self.assertEqual(len(table["action"][0].as_py()), 11)
            self.assertFalse(table["next.done"][0].as_py())
            self.assertTrue(table["next.done"][1].as_py())
            self.assertTrue(video_path.is_file())
            self.assertEqual(info["fps"], 5)
            self.assertEqual(info["features"]["observation.images.front"]["shape"], [480, 640, 3])
            self.assertEqual(info["features"]["action"]["shape"], [11])
            self.assertEqual(
                info["features"]["observation.base_velocity"]["names"],
                ["vx_body", "vy_body", "wz_body"],
            )
            self.assertIn("pipeline_state", info["features"])
            self.assertIn("next.done", info["features"])
            self.assertEqual(info["camera_keys"], ["front"])

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

    def test_failed_episode_close_removes_lerobot_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_000000"
            recorder = JsonlEpisodeRecorder(
                episode_dir,
                lerobot_config=LeRobotRecordingConfig(
                    enabled=True,
                    dataset_fps=5,
                    image_height=48,
                    image_width=64,
                ),
            )
            recorder.save_task(
                type(
                    "Spec",
                    (),
                    {"raw_task": {"instruction": "Move the apple."}},
                )()
            )
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            state = _state(1, image)
            recorder.record_step(
                StepRecord(
                    step_index=0,
                    timestamp=0.0,
                    pipeline_state="exec_pick",
                    observation=state,
                    action=RobotAction(source="arm"),
                    post_step_observation=state,
                )
            )

            recorder.mark_training_eligible(True, reason="unit_test_force_export")
            export = recorder.prepare_lerobot_export()
            self.assertTrue(export["lerobot_exported"], export)
            self.assertTrue((episode_dir / "lerobot_manifest.json").is_file())
            self.assertTrue((episode_dir / "lerobot_dataset").is_dir())

            summary_path = recorder.close(
                {"success": False, "failure_reason": "place_plan_failed"}
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertFalse((episode_dir / "lerobot_manifest.json").exists())
            self.assertFalse((episode_dir / "lerobot_dataset").exists())
            self.assertFalse(summary["lerobot_training_eligible"])
            self.assertTrue(summary["lerobot_export_skipped"])
            self.assertEqual(summary["lerobot_export_skip_reason"], "place_plan_failed")
            self.assertFalse(summary["lerobot_export"]["lerobot_exported"])
            self.assertFalse(summary["lerobot_export"]["training_eligible"])

    def test_multi_camera_video_export_without_raw_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_000000"
            recorder = JsonlEpisodeRecorder(
                episode_dir,
                lerobot_config=LeRobotRecordingConfig(
                    enabled=True,
                    control_dt=0.02,
                    dataset_fps=10,
                    image_height=48,
                    image_width=64,
                    camera_keys=("front", "wrist", "overview"),
                    save_raw_images=False,
                ),
            )
            recorder.save_task(
                type(
                    "Spec",
                    (),
                    {"raw_task": {"instruction": "Move the apple."}},
                )()
            )
            image = np.full((48, 64, 3), 127, dtype=np.uint8)
            for step_index in (1, 6):
                state = replace(
                    _state(step_index, image),
                    camera_images={"front": image, "overview": image},
                )
                recorder.record_step(
                    StepRecord(
                        step_index=step_index,
                        timestamp=state.timestamp,
                        pipeline_state="exec_pick",
                        observation=state,
                        action=RobotAction(source="arm"),
                        post_step_observation=state,
                    )
                )

            export = recorder.prepare_lerobot_export()

            self.assertTrue(export["lerobot_exported"], export)
            self.assertFalse(export["raw_images_saved"])
            self.assertEqual(export["camera_keys"], ["front", "overview"])
            self.assertIn("wrist", export["missing_camera_keys"])
            self.assertFalse((episode_dir / "images/front").exists())
            info = json.loads(
                (Path(export["dataset_path"]) / "meta/info.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(info["fps"], 10)
            self.assertEqual(info["camera_keys"], ["front", "overview"])
            for camera_key in ("front", "overview"):
                self.assertTrue(
                    (
                        Path(export["dataset_path"])
                        / "videos/chunk-000"
                        / f"observation.images.{camera_key}"
                        / "episode_000000.mp4"
                    ).is_file()
                )

    def test_wrist_camera_is_labeled_in_raw_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_000000"
            recorder = JsonlEpisodeRecorder(
                episode_dir,
                lerobot_config=LeRobotRecordingConfig(
                    enabled=True,
                    control_dt=0.02,
                    dataset_fps=10,
                    image_height=48,
                    image_width=64,
                    camera_keys=("front", "wrist"),
                    save_raw_images=True,
                ),
            )
            recorder.save_task(
                type(
                    "Spec",
                    (),
                    {"raw_task": {"instruction": "Move the apple."}},
                )()
            )
            front = np.full((48, 64, 3), 64, dtype=np.uint8)
            wrist = np.full((48, 64, 3), 192, dtype=np.uint8)
            state = replace(
                _state(1, front),
                camera_images={"front": front, "wrist": wrist},
            )
            recorder.record_step(
                StepRecord(
                    step_index=1,
                    timestamp=state.timestamp,
                    pipeline_state="exec_pick",
                    observation=state,
                    action=RobotAction(source="arm"),
                    post_step_observation=state,
                )
            )

            with (episode_dir / "data.csv").open("r", encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))

            self.assertEqual(row["前摄像头图像"], "camera0_00000.jpg")
            self.assertEqual(row["腕部摄像头图像"], "wrist_00000.jpg")
            self.assertTrue((episode_dir / "images/wrist/wrist_00000.jpg").is_file())

    def test_wrist_camera_generates_lerobot_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_000000"
            recorder = JsonlEpisodeRecorder(
                episode_dir,
                lerobot_config=LeRobotRecordingConfig(
                    enabled=True,
                    control_dt=0.02,
                    dataset_fps=10,
                    image_height=48,
                    image_width=64,
                    camera_keys=("front", "wrist"),
                    save_raw_images=False,
                ),
            )
            recorder.save_task(
                type(
                    "Spec",
                    (),
                    {"raw_task": {"instruction": "Move the apple."}},
                )()
            )
            front = np.full((48, 64, 3), 64, dtype=np.uint8)
            wrist = np.full((48, 64, 3), 192, dtype=np.uint8)
            for step_index in (1, 6):
                state = replace(
                    _state(step_index, front),
                    camera_images={"front": front, "wrist": wrist},
                )
                recorder.record_step(
                    StepRecord(
                        step_index=step_index,
                        timestamp=state.timestamp,
                        pipeline_state="exec_pick",
                        observation=state,
                        action=RobotAction(source="arm"),
                        post_step_observation=state,
                    )
                )

            export = recorder.prepare_lerobot_export()

            self.assertTrue(export["lerobot_exported"], export)
            self.assertEqual(export["camera_keys"], ["front", "wrist"])
            self.assertNotIn("wrist", export["missing_camera_keys"])
            self.assertTrue(
                (
                    Path(export["dataset_path"])
                    / "videos/chunk-000"
                    / "observation.images.wrist"
                    / "episode_000000.mp4"
                ).is_file()
            )

    def test_training_row_uses_pre_step_observation_and_two_gripper_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_000000"
            recorder = JsonlEpisodeRecorder(
                episode_dir,
                lerobot_config=LeRobotRecordingConfig(
                    enabled=True,
                    dataset_fps=5,
                    image_height=48,
                    image_width=64,
                ),
            )
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            before = replace(
                _state(1, image),
                robot_root_pose=(1.0, 2.0, 0.35, 1.0, 0.0, 0.0, 0.0),
                object_pose=(0.9, 1.2, 0.82, 1.0, 0.0, 0.0, 0.0),
            )
            after = replace(
                before,
                robot_root_pose=(9.0, 8.0, 0.35, 1.0, 0.0, 0.0, 0.0),
                object_pose=(7.0, 6.0, 0.82, 1.0, 0.0, 0.0, 0.0),
            )
            recorder.record_step(
                StepRecord(
                    step_index=0,
                    timestamp=0.0,
                    pipeline_state="exec_pick",
                    observation=before,
                    action=RobotAction(
                        base_velocity=(0.1, 0.2, 0.3),
                        arm_joint_positions=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                        metadata={"gripper_joint_positions": (0.01, 0.03)},
                    ),
                    post_step_observation=after,
                )
            )

            with (episode_dir / "data.csv").open("r", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            sample = json.loads(
                (episode_dir / "samples.jsonl").read_text(encoding="utf-8").strip()
            )

            self.assertEqual(float(row["位置X"]), 1.0)
            self.assertEqual(sample["object_state"][0:2], [0.9, 1.2])
            self.assertEqual(sample["action"], [0.1, 0.2, 0.3, 1, 2, 3, 4, 5, 6, 0.01, 0.03])


if __name__ == "__main__":
    unittest.main()
