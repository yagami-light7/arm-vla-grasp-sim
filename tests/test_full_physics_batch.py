"""Full-physics batch launcher tests."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.pipeline.run_full_physics_batch import (
    BatchEpisodeCommand,
    _build_child_command,
    _build_parser,
    _color,
    _format_duration,
    _format_progress_suffix,
    _read_episode_progress,
    _run_child_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
PICK_PLAN = (
    PROJECT_ROOT
    / "outputs/random_pick_place_dataset/apple_pick_place_contact/episode_0000"
    / "single_stage_arm_place_debug/pick_plan.json"
)
PLACE_PLAN = (
    PROJECT_ROOT
    / "outputs/random_pick_place_dataset/apple_pick_place_contact/episode_0000"
    / "single_stage_arm_place_debug/arm_place_plan.json"
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
                "--full-physics",
                "--viewport-camera-prim",
                "/World/Camera1",
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
            Path("/tmp/full_physics_batch_test/episode_000002/episode_000000/summary.json"),
        )
        command = episode.command
        self.assertIn("--full-physics", command)
        self.assertIn("--num-episodes", command)
        self.assertEqual(command[command.index("--num-episodes") + 1], "1")
        self.assertEqual(command[command.index("--seed") + 1], "102")
        self.assertIn("--randomize-task", command)
        self.assertIn("--headless", command)
        self.assertIn("--auto-start-curobo-server", command)
        self.assertIn("--lock-base-during-manipulation", command)
        self.assertIn("--lock-support-joints-during-manipulation", command)
        self.assertIn("--replan-pick-from-current-state", command)
        self.assertIn("--viewport-camera-prim", command)
        self.assertEqual(command[command.index("--viewport-camera-prim") + 1], "/World/Camera1")
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
                "--no-headless",
                "--show-randomization-debug",
                "--no-auto-start-curobo-server",
                "--keep-window-open",
            ]
        )

        episode = _build_child_command(args, episode_index=0)
        command = episode.command

        self.assertIn("--dry-run", command)
        self.assertIn("--no-randomize-task", command)
        self.assertIn("--no-headless", command)
        self.assertIn("--show-randomization-debug", command)
        self.assertIn("--no-auto-start-curobo-server", command)
        self.assertIn("--keep-window-open", command)

    def test_progress_format_helpers(self) -> None:
        self.assertEqual(_format_duration(3.2), "3s")
        self.assertEqual(_format_duration(65.0), "1m05s")
        self.assertEqual(_format_duration(3661.0), "1h01m01s")
        self.assertEqual(_color("ok", "green", enabled=False), "ok")
        self.assertIn("\033[32m", _color("ok", "green", enabled=True))

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
        self.assertIn("state=exec_nav_to_pick", output)
        self.assertIn("step=12", output)

    def test_heartbeat_waits_for_quiet_child_output(self) -> None:
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
        self.assertNotIn("[progress] episode=0 seed=123 running", output)

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


if __name__ == "__main__":
    unittest.main()
