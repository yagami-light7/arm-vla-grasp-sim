"""Tests for the first-phase full-physics pipeline skeleton."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.pipeline.run_full_physics_pipeline import _build_parser, main
from source.pipeline import FullPhysicsConfig, PipelineState, StateLimits
from source.pipeline.dry_run import create_dry_run_pipeline
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FullPhysicsPipelineTest(unittest.TestCase):
    def _run_task(
        self,
        task_name: str,
        output_dir: Path,
        *,
        limits: StateLimits | None = None,
        seed: int = 7,
    ) -> tuple[dict, object]:
        task_path = PROJECT_ROOT / "tasks" / task_name
        config = FullPhysicsConfig(
            task_json=task_path,
            output_dir=output_dir,
            seed=seed,
            dry_run=True,
            limits=limits or StateLimits(),
        )
        spec = JsonTaskProvider().load(task_path)
        pipeline = create_dry_run_pipeline(
            config=config,
            episode_spec=spec,
            episode_seed=seed,
            episode_dir=output_dir,
        )
        summary = pipeline.run_episode()
        return summary, pipeline

    def test_dry_run_completes_full_state_flow_and_writes_artifacts(self) -> None:
        expected_trace = [state.value for state in PipelineState if state != PipelineState.FAILED]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "episode"
            summary, pipeline = self._run_task(
                "nav_pick_place_apple_contact.json",
                output_dir,
            )

            self.assertTrue(summary["success"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertFalse(summary["execution_provenance_verified"])
            self.assertEqual(summary["success_semantics"], "control_flow_only")
            self.assertEqual(summary["state_trace"], expected_trace)
            self.assertGreater(summary["duration_steps"], len(expected_trace))
            self.assertEqual(pipeline.simulation.step_calls, summary["duration_steps"])
            self.assertEqual(pipeline.simulation.apply_calls, summary["duration_steps"])

            for filename in (
                "task.json",
                "events.jsonl",
                "frames.jsonl",
                "lerobot_manifest.json",
                "summary.json",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)

            manifest = json.loads((output_dir / "lerobot_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["lerobot_exported"])
            event_names = [
                json.loads(line)["name"]
                for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("gripper_close", event_names)
            self.assertIn("gripper_open", event_names)
            self.assertIn("episode_success", event_names)

    def test_pick_only_task_fails_with_structured_place_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary, _pipeline = self._run_task(
                "nav_pick_apple_fast.json",
                Path(tmp_dir) / "episode",
            )

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "place_target_unreachable")
            self.assertEqual(summary["final_state"], PipelineState.FAILED.value)
            self.assertIn(PipelineState.PLAN_NAV_TO_PLACE.value, summary["state_trace"])
            self.assertEqual(summary["state_trace"][-1], PipelineState.FAILED.value)

    def test_navigation_timeout_is_reported_without_exception_exit(self) -> None:
        limits = StateLimits(navigation=1, episode=100)
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary, _pipeline = self._run_task(
                "nav_pick_place_apple_contact.json",
                Path(tmp_dir) / "episode",
                limits=limits,
            )

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "nav_to_pick_timeout")
            self.assertEqual(
                summary["failure_metadata"]["current_state"],
                PipelineState.EXEC_NAV_TO_PICK.value,
            )

    def test_cli_help_is_chinese_and_dry_run_is_explicit(self) -> None:
        help_text = _build_parser().format_help()
        self.assertIn("任务 JSON 路径", help_text)
        self.assertIn("--dry-run", help_text)
        self.assertIn("--no-headless", help_text)

    def test_cli_rejects_real_backend_until_it_is_implemented(self) -> None:
        with self.assertRaisesRegex(SystemExit, "真实纯物理后端尚未接入"):
            main(["--task-json", "tasks/nav_pick_place_apple_contact.json"])

    def test_component_exception_becomes_structured_failure(self) -> None:
        class RaisingPlanner:
            def plan(self, state, goal):
                del state, goal
                raise RuntimeError("planned failure")

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=Path(tmp_dir),
                dry_run=True,
            )
            pipeline = create_dry_run_pipeline(
                config=config,
                episode_spec=JsonTaskProvider().load(task_path),
                episode_seed=0,
                episode_dir=Path(tmp_dir) / "episode",
            )
            pipeline.machine.nav_planner = RaisingPlanner()
            summary = pipeline.run_episode()

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "nav_to_pick_plan_failed")
            self.assertEqual(
                summary["failure_metadata"]["exception_type"],
                "RuntimeError",
            )

    def test_task_provider_preserves_existing_schema_values(self) -> None:
        spec = JsonTaskProvider().load(PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json")
        self.assertEqual(spec.object_prim_path, "/World/apple")
        self.assertIsNotNone(spec.place_goal)
        self.assertIsNotNone(spec.place_target_pose)
        self.assertEqual(spec.raw_task["carry"]["mode"], "contact")


if __name__ == "__main__":
    unittest.main()
