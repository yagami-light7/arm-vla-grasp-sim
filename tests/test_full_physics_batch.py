"""Full-physics batch launcher tests."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts.pipeline.run_full_physics_batch import (
    BatchEpisodeCommand,
    _build_episode_result,
    _build_child_command,
    _build_parser,
    _color,
    _format_duration,
    _format_progress_suffix,
    _format_result_table,
    _read_episode_progress,
    _read_summary,
    _run_child_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
PICK_PLAN = (
    PROJECT_ROOT
    / "outputs/random_pick_place_dataset/apple_pick_place_contact/episode_0000"
    / "legacy_arm_place_debug/pick_plan.json"
)
PLACE_PLAN = (
    PROJECT_ROOT
    / "outputs/random_pick_place_dataset/apple_pick_place_contact/episode_0000"
    / "legacy_arm_place_debug/arm_place_plan.json"
)


class FullPhysicsBatchTest(unittest.TestCase):
    def test_full_physics_batch_builds_one_episode_command_without_plan_json(self) -> None:
        args = _build_parser().parse_args(
            [
                "--task-json",
                str(TASK_PATH),
                "--output-dir",
                "/tmp/full_physics_batch_test",
                "--num-episodes",
                "3",
                "--seed",
                "100",
                "--pick-plan-json",
                str(PICK_PLAN),
                "--place-plan-json",
                str(PLACE_PLAN),
            ]
        )

        episode = _build_child_command(args, episode_index=2)

        self.assertEqual(episode.seed, 102)
        self.assertEqual(episode.output_dir, Path("/tmp/full_physics_batch_test/episode_000002"))
        self.assertEqual(
            episode.summary_path,
            Path("/tmp/full_physics_batch_test/episode_000002/summary.json"),
        )
        command = episode.command
        self.assertNotIn("--full-physics", command)
        self.assertIn("--num-episodes", command)
        self.assertEqual(command[command.index("--num-episodes") + 1], "1")
        self.assertEqual(command[command.index("--seed") + 1], "102")
        self.assertIn("--randomize-task", command)
        self.assertIn("--randomize-base-goal", command)
        self.assertIn("--headless", command)
        self.assertNotIn("--auto-start-curobo-server", command)
        self.assertNotIn("--lock-base-during-manipulation", command)
        self.assertNotIn("--lock-support-joints-during-manipulation", command)
        self.assertNotIn("--replan-pick-from-current-state", command)
        self.assertNotIn("--viewport-camera-prim", command)
        self.assertNotIn("--pick-plan-json", command)
        self.assertNotIn("--place-plan-json", command)

    def test_batch_forwards_disabled_randomization_and_gui_mode(self) -> None:
        args = _build_parser().parse_args(
            [
                "--task-json",
                str(TASK_PATH),
                "--output-dir",
                "/tmp/full_physics_batch_test",
                "--seed",
                "5",
                "--dry-run",
                "--no-randomize-task",
                "--no-randomize-base-goal",
                "--no-headless",
                "--show-randomization-debug",
            ]
        )

        episode = _build_child_command(args, episode_index=0)
        command = episode.command

        self.assertIn("--dry-run", command)
        self.assertIn("--no-randomize-task", command)
        self.assertIn("--no-randomize-base-goal", command)
        self.assertIn("--no-headless", command)
        self.assertIn("--show-randomization-debug", command)

    def test_progress_format_helpers(self) -> None:
        args = _build_parser().parse_args(
            [
                "--task-json",
                str(TASK_PATH),
                "--output-dir",
                "/tmp/full_physics_batch_test",
            ]
        )
        self.assertEqual(_format_duration(3.2), "3s")
        self.assertEqual(_format_duration(65.0), "1m05s")
        self.assertEqual(_format_duration(3661.0), "1h01m01s")
        self.assertEqual(_color("ok", "green", enabled=False), "ok")
        self.assertIn("\033[32m", _color("ok", "green", enabled=True))
        self.assertEqual(args.progress_interval_s, 5.0)

    def test_result_table_contains_requested_episode_fields_and_colors(self) -> None:
        episode = BatchEpisodeCommand(
            episode_index=2,
            seed=102,
            output_dir=Path("/tmp/batch/episode_000002"),
            summary_path=Path("/tmp/batch/episode_000002/summary.json"),
            command=[],
        )
        success_summary = {
            "success": True,
            "task_config": {
                "randomization": {
                    "object_xy_randomization": {
                        "sampled_xy": {"x": 0.91234, "y": 1.23456},
                    },
                    "place_xy_randomization": {
                        "sampled_xy": {"x": 0.71234, "y": 5.23456},
                    },
                    "base_goal_randomization": {
                        "enabled": True,
                        "pick": {
                            "target_xy": [0.91234, 1.23456],
                            "sampled_base_goal_xyyaw": [1.01234, 1.13456, -3.0],
                        },
                        "place": {
                            "target_xy": [0.71234, 5.23456],
                            "sampled_base_goal_xyyaw": [1.11234, 5.43456, 2.9],
                        },
                    },
                }
            },
            "lerobot_export": {
                "manifest_path": "/tmp/batch/episode_000002/lerobot_manifest.json",
            },
        }
        failed_summary = {
            "success": False,
            "failure_reason": "place_plan_failed",
            "failure_metadata": {"current_state": "plan_place"},
            "task_config": {
                "pick": {"object_pose_world": {"x": 0.91, "y": 1.20}},
                "place": {"place_pose_world": {"x": 0.70, "y": 5.20}},
            },
            "data_output_path": "/tmp/batch/episode_000003",
        }
        success_result = _build_episode_result(
            episode=episode,
            summary=success_summary,
            success=True,
            elapsed_seconds=65.0,
        )
        failed_result = _build_episode_result(
            episode=BatchEpisodeCommand(
                episode_index=3,
                seed=103,
                output_dir=Path("/tmp/batch/episode_000003"),
                summary_path=Path("/tmp/batch/episode_000003/summary.json"),
                command=[],
            ),
            summary=failed_summary,
            success=False,
            elapsed_seconds=7.0,
        )

        plain = _format_result_table(
            [success_result, failed_result],
            color_enabled=False,
        )
        colored = _format_result_table(
            [success_result, failed_result],
            color_enabled=True,
        )

        self.assertIn("Episode", plain)
        self.assertIn("随机化 Pick / Place XY", plain)
        self.assertIn("随机化 BaseGoal / 相对目标", plain)
        self.assertIn("Pipeline 成功", plain)
        self.assertIn("失败 State", plain)
        self.assertIn("LeRobot 数据路径", plain)
        self.assertIn("Episode 耗时", plain)
        self.assertIn("pick=(0.9123,1.2346) place=(0.7123,5.2346)", plain)
        self.assertIn(
            "pick_bg=(1.0123,1.1346,-3.000) Δ=(+0.1000,-0.1000)",
            plain,
        )
        self.assertIn(
            "place_bg=(1.1123,5.4346,2.900) Δ=(+0.4000,+0.2000)",
            plain,
        )
        self.assertIn("plan_place", plain)
        self.assertIn("lerobot_manifest.json", plain)
        self.assertIn("1m05s", plain)
        self.assertIn("\033[36m", colored)
        self.assertIn("\033[35m", colored)
        self.assertIn("\033[32m", colored)
        self.assertIn("\033[31m", colored)
        self.assertIn("\033[33m", colored)
        self.assertIn("\033[34m", colored)
        self.assertIn("\033[37m", colored)

    def test_child_process_output_is_streamed_with_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            summary_path = episode_dir / "summary.json"
            frames_path = episode_dir / "frames.jsonl"
            frames_path.write_text(
                json.dumps(
                    {
                        "pipeline_state": "exec_nav_to_pick",
                        "step_index": 12,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            episode = BatchEpisodeCommand(
                episode_index=0,
                seed=123,
                output_dir=episode_dir,
                summary_path=summary_path,
                command=[
                    sys.executable,
                    "-c",
                    (
                        "import time; "
                        "print('child-start', flush=True); "
                        "time.sleep(0.25); "
                        "print('child-done', flush=True)"
                    ),
                ],
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                returncode = _run_child_process(
                    episode,
                    env=env,
                    progress_interval_s=0.1,
                    color_enabled=False,
                )

        output = stream.getvalue()
        self.assertEqual(returncode, 0)
        self.assertIn("child-start", output)
        self.assertIn("child-done", output)
        self.assertIn("[progress] episode=0 seed=123 running", output)
        self.assertIn("state=launching", output)
        self.assertIn("source=batch", output)

    def test_heartbeat_prints_from_zero_even_when_child_is_chatty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            frames_path = episode_dir / "frames.jsonl"
            frames_path.write_text(
                json.dumps({"pipeline_state": "exec_nav_to_pick", "step_index": 12})
                + "\n",
                encoding="utf-8",
            )
            episode = BatchEpisodeCommand(
                episode_index=0,
                seed=123,
                output_dir=episode_dir,
                summary_path=episode_dir / "summary.json",
                command=[
                    sys.executable,
                    "-c",
                    (
                        "import time; "
                        "[print(f'child-{i}', flush=True) or time.sleep(0.05) "
                        "for i in range(6)]"
                    ),
                ],
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                returncode = _run_child_process(
                    episode,
                    env=env,
                    progress_interval_s=0.1,
                    color_enabled=False,
                )

        output = stream.getvalue()
        self.assertEqual(returncode, 0)
        self.assertIn("child-0", output)
        self.assertIn("[progress] episode=0 seed=123 running elapsed=0s", output)
        self.assertIn("state=launching", output)
        self.assertIn("source=batch", output)

    def test_episode_progress_reads_frames_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            summary_path = episode_dir / "summary.json"
            frames_path = episode_dir / "frames.jsonl"
            frames_path.write_text(
                "\n".join(
                    [
                        json.dumps({"pipeline_state": "build_stage", "step_index": 1}),
                        json.dumps({"pipeline_state": "exec_place", "step_index": 99}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "final_state": "done",
                        "duration_steps": 120,
                        "state_trace": ["build_stage", "done"],
                    }
                ),
                encoding="utf-8",
            )
            episode = BatchEpisodeCommand(
                episode_index=0,
                seed=0,
                output_dir=episode_dir,
                summary_path=summary_path,
                command=[],
            )

            progress = _read_episode_progress(episode)

        self.assertEqual(progress.state, "exec_place")
        self.assertEqual(progress.step_index, 99)
        self.assertEqual(progress.source, "frames")
        self.assertIn("state=exec_place", _format_progress_suffix(progress, color_enabled=False))

    def test_episode_progress_ignores_stale_files_from_previous_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir)
            summary_path = episode_dir / "summary.json"
            frames_path = episode_dir / "frames.jsonl"
            frames_path.write_text(
                json.dumps({"pipeline_state": "exec_nav_to_place", "step_index": 999})
                + "\n",
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps({"final_state": "failed", "duration_steps": 1000}),
                encoding="utf-8",
            )
            stale_mtime = time.time() - 60.0
            os.utime(frames_path, (stale_mtime, stale_mtime))
            os.utime(summary_path, (stale_mtime, stale_mtime))
            episode = BatchEpisodeCommand(
                episode_index=0,
                seed=0,
                output_dir=episode_dir,
                summary_path=summary_path,
                command=[],
            )

            progress = _read_episode_progress(episode, min_mtime=time.time())
            summary = _read_summary(summary_path, min_mtime=time.time())

        self.assertIsNone(summary)
        self.assertIsNone(progress.state)
        self.assertIsNone(progress.step_index)
        self.assertEqual(progress.source, "unavailable")

    def test_summary_reader_accepts_legacy_nested_episode_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_000002"
            legacy_dir = episode_dir / "episode_000000"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "summary.json").write_text(
                json.dumps({"success": True}),
                encoding="utf-8",
            )

            summary = _read_summary(episode_dir / "summary.json")

        self.assertEqual(summary, {"success": True})


if __name__ == "__main__":
    unittest.main()
