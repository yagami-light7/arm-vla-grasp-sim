"""tick 化 manipulation adapter 测试。"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from source.interfaces import ArmPlan, EpisodeSpec, NavGoal, SimulationState
from source.manipulation.arm_executor import SegmentedArmExecutor, SegmentedArmExecutorConfig
from source.manipulation.current_state_curobo import (
    CurrentStateCuroboPlanner,
    CurrentStateCuroboPlannerConfig,
    build_curobo_state_payload,
    build_arm_place_target_payload,
    build_side_grasp_target_payload,
    pose_to_matrix,
)
from source.manipulation.curobo_adapter import (
    CuroboJsonManipulationPlanner,
    CuroboPlanFormatError,
    arm_plan_from_curobo_payload,
)
from source.manipulation.gripper_controller import BinaryGripperController
from source.manipulation.grasp_pipeline import GraspPipeline, GraspPipelineConfig, GraspTask


class FullPhysicsManipulationTest(unittest.TestCase):
    def test_curobo_state_payload_records_world_collision_metadata(self) -> None:
        payload = build_curobo_state_payload(
            q_arm=[0.0] * 6,
            dq_arm=[0.0] * 6,
            q_full=[0.0] * 8,
            dq_full=[0.0] * 8,
            dof_names=[
                "arm_joint1",
                "arm_joint2",
                "arm_joint3",
                "arm_joint4",
                "arm_joint5",
                "arm_joint6",
                "arm_joint7",
                "arm_joint8",
            ],
            q_gripper=[0.043, 0.043],
            dq_gripper=[0.0, 0.0],
            T_world_base=pose_to_matrix([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            robot_root_path="/World/go2_x5",
            articulation_root_path="/World/go2_x5",
            base_frame_path="/World/go2_x5/arm_base_link",
            tcp_frame_path="/World/go2_x5/grasp_tcp_link",
            tcp_mode="prim",
            world_collision_cuboids=[
                {
                    "name": "table_edge",
                    "pose_base": {
                        "position_xyz": [0.2, 0.0, 0.1],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "dims_xyz": [0.5, 0.1, 0.1],
                }
            ],
            world_collision_metadata={
                "padding_m": 0.05,
                "clearance_margin_m": 0.05,
            },
        )

        world_collision = payload["world_collision"]
        self.assertTrue(world_collision["enabled"])
        self.assertEqual(world_collision["padding_m"], 0.05)
        self.assertEqual(world_collision["clearance_margin_m"], 0.05)
        self.assertEqual(world_collision["cuboids_base"][0]["name"], "table_edge")

    def test_planner_server_request_uses_place_command(self) -> None:
        from unittest.mock import patch

        pipeline = GraspPipeline(GraspPipelineConfig(split_pregrasp_motion=True))
        captured_request = {}

        class FakeSocket:
            def settimeout(self, timeout):
                del timeout

            def sendall(self, payload):
                captured_request.update(json.loads(payload.decode("utf-8")))

            def makefile(self, *args, **kwargs):
                del args, kwargs
                return type("Response", (), {"readline": lambda self: '{"ok": true}\n'})()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                del args

        task = GraspTask(
            object_prim_path="/World/apple",
            curobo_task_mode="place",
            state_json="/tmp/state.json",
            target_json="/tmp/target.json",
            plan_json="/tmp/plan.json",
        )
        with patch(
            "source.manipulation.grasp_pipeline.socket.create_connection",
            return_value=FakeSocket(),
        ):
            pipeline._try_server(task)

        self.assertEqual(captured_request["command"], "plan_place_segments")
        self.assertFalse(captured_request["side_grasp_retreat_to_pregrasp"])
        self.assertTrue(captured_request["split_pregrasp_motion"])

    def test_current_state_planner_uses_baseline_side_retreat(self) -> None:
        class FakeCurrentStateSimulation:
            pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            planner = CurrentStateCuroboPlanner(
                simulation=FakeCurrentStateSimulation(),
                config=CurrentStateCuroboPlannerConfig(
                    output_dir=root / "online",
                    project_root=root,
                    place_plan_json=None,
                    use_planner_server=False,
                ),
            )

        grasp_pipeline = planner._plan_runner.__self__  # type: ignore[attr-defined]
        self.assertFalse(grasp_pipeline.config.side_grasp_plan_vertical_lift)
        self.assertFalse(grasp_pipeline.config.side_grasp_fallback_retreat)
        self.assertFalse(grasp_pipeline.config.side_grasp_retreat_to_pregrasp)
        self.assertTrue(grasp_pipeline.config.split_pregrasp_motion)

    def test_curobo_payload_is_preserved_as_segmented_arm_plan(self) -> None:
        plan = arm_plan_from_curobo_payload(_pick_payload())

        self.assertEqual(plan.operation, "pick")
        self.assertEqual(plan.metadata["joint_names"], _arm_joint_names())
        self.assertEqual(plan.metadata["tool_frame"], "grasp_tcp_link")
        self.assertEqual(
            [segment["name"] for segment in plan.metadata["segments"]],
            [
                "open_gripper",
                "move_to_pregrasp",
                "approach_to_grasp",
                "close_gripper",
                "retreat_object",
                "return_home_after_retreat",
            ],
        )
        self.assertEqual(len(plan.joint_trajectory), 8)

    def test_segmented_executor_strictly_waits_for_pick_clearance_segments(self) -> None:
        strict_segments = SegmentedArmExecutorConfig().strict_post_motion_hold_segments

        self.assertIn("move_to_pregrasp", strict_segments)
        self.assertIn("approach_to_grasp", strict_segments)
        self.assertIn("retreat_object", strict_segments)
        self.assertIn("return_home_after_retreat", strict_segments)

    def test_segmented_executor_holds_gripper_closed_after_close_segment(self) -> None:
        plan = arm_plan_from_curobo_payload(_pick_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.02,
                gripper_move_duration=0.04,
                gripper_hold_duration=0.02,
            ),
        )
        executor.reset(plan)

        actions = _drain_executor(executor)

        close_actions = [
            action
            for action in actions
            if action.metadata.get("segment_name") == "close_gripper"
        ]
        self.assertEqual(close_actions[0].metadata["event_marker"], "gripper_close")
        self.assertEqual(close_actions[0].gripper_command, "close")
        # close_gripper 期间保持上一段 grasp 位姿，避免 TCP 在闭合前漂走。
        self.assertEqual(close_actions[0].arm_joint_positions, (0.2, 0.21, 0.22, 0.23, 0.24, 0.25))

        retreat_actions = [
            action
            for action in actions
            if action.metadata.get("segment_name") == "retreat_object"
        ]
        self.assertTrue(retreat_actions)
        self.assertTrue(all(action.gripper_command == "close" for action in retreat_actions))
        self.assertTrue(
            all(action.metadata.get("gripper_hold_after_close") for action in retreat_actions)
        )
        retreat_settle_actions = [
            action
            for action in actions
            if action.metadata.get("parent_segment_name") == "retreat_object"
            and action.metadata.get("segment_type") == "settle_to_segment_start"
        ]
        self.assertTrue(retreat_settle_actions)
        self.assertTrue(
            all(action.gripper_command == "close" for action in retreat_settle_actions)
        )
        self.assertTrue(
            all(
                action.metadata.get("gripper_hold_after_close_source")
                == "close_target"
                for action in retreat_settle_actions
            )
        )
        self.assertTrue(executor.status()["done"])
        self.assertTrue(executor.status()["close_progress_report"]["close_success"])
        self.assertGreaterEqual(
            executor.status()["close_progress_report"]["close_progress"],
            0.05,
        )
        self.assertTrue(executor.status()["world_step_owned_by_pipeline"])

    def test_segmented_executor_waits_like_baseline_before_close(self) -> None:
        plan = arm_plan_from_curobo_payload(_pick_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.05,
                post_motion_hold_duration=0.10,
                pre_close_arm_hold_duration=0.10,
                gripper_move_duration=0.05,
                gripper_hold_duration=0.0,
            ),
        )
        executor.reset(plan)

        actions = _drain_executor(executor)

        post_hold_actions = [
            action
            for action in actions
            if action.metadata.get("segment_name") == "approach_to_grasp"
            and action.metadata.get("segment_type") == "post_motion_hold"
        ]
        # 测试 runtime 精确跟随目标，因此 strict wait 会在容差检查后提前结束。
        self.assertEqual(post_hold_actions, [])
        wait_reports = executor.status()["strict_post_motion_wait_reports"]
        approach_report = next(
            report
            for report in wait_reports
            if report["segment_name"] == "approach_to_grasp"
        )
        self.assertTrue(approach_report["post_motion_hold_converged"])
        self.assertTrue(approach_report["post_motion_hold_skipped_on_tolerance"])

        pre_close_actions = [
            action
            for action in actions
            if action.metadata.get("segment_type") == "pre_close_arm_hold"
        ]
        self.assertEqual(len(pre_close_actions), 2)
        self.assertTrue(
            all(action.metadata.get("baseline_pre_close_hold") for action in pre_close_actions)
        )
        self.assertTrue(all(action.gripper_command == "hold" for action in pre_close_actions))

        first_pre_close_index = actions.index(pre_close_actions[0])
        first_close_index = next(
            index
            for index, action in enumerate(actions)
            if action.metadata.get("segment_name") == "close_gripper"
        )
        self.assertLess(first_pre_close_index, first_close_index)
        self.assertEqual(actions[first_close_index].gripper_command, "close")
        self.assertEqual(actions[first_close_index].metadata["event_marker"], "gripper_close")

    def test_close_gripper_uses_baseline_smoothstep_from_actual_position(self) -> None:
        plan = arm_plan_from_curobo_payload(_pick_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.10,
                settle_to_segment_start_duration=0.0,
                post_motion_hold_duration=0.0,
                pre_close_arm_hold_duration=0.0,
                gripper_move_duration=0.30,
                gripper_hold_duration=0.0,
            ),
        )
        executor.reset(plan)
        while executor.status()["current_segment"]["name"] != "close_gripper":
            executor.compute_action(_state())

        contact_state = _state_with_all_joints(
            (0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.04, 0.04)
        )
        close_actions = [executor.compute_action(contact_state) for _ in range(3)]
        close_targets = [
            tuple(action.metadata["gripper_joint_positions"])
            for action in close_actions
        ]

        self.assertEqual(close_targets[0], (0.04, 0.04))
        self.assertAlmostEqual(close_targets[1][0], 0.02)
        self.assertAlmostEqual(close_targets[1][1], 0.02)
        self.assertEqual(close_targets[2], (0.0, 0.0))
        self.assertTrue(
            all(action.metadata.get("baseline_gripper_interpolation") for action in close_actions)
        )
        self.assertTrue(
            all(action.metadata.get("gripper_interpolation") == "smoothstep5" for action in close_actions)
        )

    def test_open_gripper_holds_current_arm_pose_before_first_motion(self) -> None:
        plan = arm_plan_from_curobo_payload(_pick_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.10,
                gripper_move_duration=0.20,
                gripper_hold_duration=0.0,
                settle_to_segment_start_duration=0.0,
                post_motion_hold_duration=0.0,
            ),
        )
        executor.reset(plan)

        initial_arm = (0.11, 0.12, 0.13, 0.14, 0.15, 0.16)
        drifted_arm = (0.31, 0.32, 0.33, 0.34, 0.35, 0.36)
        first = executor.compute_action(_state_with_all_joints((*initial_arm, 0.0, 0.0)))
        second = executor.compute_action(_state_with_all_joints((*drifted_arm, 0.01, 0.01)))

        self.assertEqual(first.metadata["segment_name"], "open_gripper")
        self.assertEqual(first.arm_joint_positions, initial_arm)
        self.assertEqual(second.arm_joint_positions, initial_arm)
        self.assertEqual(first.metadata["arm_hold_source"], "current_state_on_segment_entry")
        self.assertEqual(second.metadata["arm_hold_source"], "current_state_on_segment_entry")
        self.assertTrue(first.metadata["arm_hold_during_gripper"])

    def test_close_progress_gate_blocks_motion_when_gripper_barely_moves(self) -> None:
        plan = arm_plan_from_curobo_payload(_pick_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.10,
                settle_to_segment_start_duration=0.0,
                post_motion_hold_duration=0.0,
                pre_close_arm_hold_duration=0.0,
                gripper_move_duration=0.20,
                gripper_hold_duration=0.0,
                final_motion_hold_duration=0.0,
                require_close_progress_for_motion=True,
            ),
        )
        executor.reset(plan)
        joint_positions = [0.0] * 8
        state = _state_with_all_joints(tuple(joint_positions))
        while executor.status()["current_segment"]["name"] != "close_gripper":
            action = executor.compute_action(state)
            if action.arm_joint_positions is not None:
                joint_positions[:6] = action.arm_joint_positions
            gripper_positions = action.metadata.get("gripper_joint_positions")
            if gripper_positions is not None:
                joint_positions[6:8] = gripper_positions
            state = _state_with_all_joints(tuple(joint_positions))

        while executor.status()["current_segment"]["name"] == "close_gripper":
            action = executor.compute_action(state)
            # 模拟夹爪几乎未闭合，实际位置只从 0.040 走到 0.039。
            joint_positions[6:8] = [0.039, 0.039]
            state = _state_with_all_joints(tuple(joint_positions))

        self.assertTrue(executor.is_done(state))
        status = executor.status()
        self.assertTrue(status["failed"])
        self.assertEqual(status["failure_reason"], "gripper_close_progress_insufficient")
        self.assertLess(status["close_progress_report"]["close_progress"], 0.05)

    def test_post_motion_hold_skips_when_joint_state_is_already_converged(self) -> None:
        plan = arm_plan_from_curobo_payload(_pick_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.05,
                post_motion_hold_duration=0.10,
                pre_close_arm_hold_duration=0.0,
                gripper_move_duration=0.05,
                gripper_hold_duration=0.0,
            ),
        )
        executor.reset(plan)

        state = _state_with_all_joints((0.0,) * 8)
        while True:
            current_segment = executor.status()["current_segment"]
            if (
                current_segment["type"] == "post_motion_hold"
                and current_segment["name"] == "approach_to_grasp"
            ):
                break
            action = executor.compute_action(state)
            joint_positions = [0.0] * 8
            if action.arm_joint_positions is not None:
                joint_positions[:6] = action.arm_joint_positions
            gripper_positions = action.metadata.get("gripper_joint_positions")
            if gripper_positions is not None:
                joint_positions[6:8] = gripper_positions
            state = _state_with_all_joints(tuple(joint_positions))
        close_action = executor.compute_action(
            _state_with_joints((0.2, 0.21, 0.22, 0.23, 0.24, 0.25))
        )

        self.assertEqual(close_action.metadata["segment_name"], "close_gripper")
        self.assertEqual(close_action.gripper_command, "close")

    def test_motion_sampling_uses_cubic_hermite_when_qd_is_available(self) -> None:
        plan = ArmPlan(
            operation="pick",
            joint_trajectory=((0.0,) * 6, (1.0,) * 6),
            metadata={
                "joint_names": _arm_joint_names(),
                "segments": [
                    {
                        "name": "hermite_motion",
                        "type": "motion",
                        "trajectory": {
                            "time_from_start": (0.0, 1.0),
                            "q": ((0.0,) * 6, (1.0,) * 6),
                            "qd": ((0.0,) * 6, (0.0,) * 6),
                        },
                    }
                ],
            },
        )
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.25,
                arm_command_dt=0.25,
                settle_to_segment_start_duration=0.0,
                post_motion_hold_duration=0.0,
            ),
        )
        executor.reset(plan)

        executor.compute_action(_state())
        action = executor.compute_action(_state())

        self.assertEqual(action.metadata["interpolation"], "cubic_hermite")
        # 零速度端点的 cubic Hermite 在 t=0.25 处为 0.15625，不是线性 0.25。
        self.assertAlmostEqual(action.arm_joint_positions[0], 0.15625)

    def test_pick_approach_uses_tracking_safe_time_scale(self) -> None:
        plan = arm_plan_from_curobo_payload(_pick_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.02,
                arm_command_dt=0.02,
                motion_time_scale=1.0,
                pick_approach_motion_time_scale=2.0,
                settle_to_segment_start_duration=0.0,
                post_motion_hold_duration=0.0,
                gripper_move_duration=0.02,
                gripper_hold_duration=0.0,
            ),
        )
        executor.reset(plan)

        action = executor.compute_action(_state_with_all_joints((0.0,) * 8))
        while action.metadata.get("segment_name") != "approach_to_grasp":
            action = executor.compute_action(_state_with_all_joints((0.0,) * 8))

        self.assertEqual(action.metadata["motion_time_scale"], 2.0)
        self.assertEqual(action.metadata["segment_ticks"], 3)

    def test_place_carry_segments_keep_baseline_time_scale(self) -> None:
        payload = _place_payload()
        payload["segments"] = [
            _motion_segment(
                "move_to_pre_place",
                [
                    (0.2, 0.21, 0.22, 0.23, 0.24, 0.25),
                    (0.3, 0.31, 0.32, 0.33, 0.34, 0.35),
                ],
            ),
            *payload["segments"],
        ]
        plan = arm_plan_from_curobo_payload(payload)
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.02,
                arm_command_dt=0.02,
                motion_time_scale=0.5,
                place_move_to_pre_place_motion_time_scale=1.0,
                place_approach_motion_time_scale=1.0,
                place_retreat_motion_time_scale=1.0,
                settle_to_segment_start_duration=0.0,
                post_motion_hold_duration=0.0,
                gripper_move_duration=0.02,
                gripper_hold_duration=0.0,
            ),
        )
        executor.reset(plan)

        actions = _drain_executor(executor)
        scale_by_segment = {
            action.metadata["segment_name"]: action.metadata["motion_time_scale"]
            for action in actions
            if action.metadata.get("segment_type") == "motion"
        }

        self.assertEqual(scale_by_segment["move_to_pre_place"], 1.0)
        self.assertEqual(scale_by_segment["approach_to_place"], 1.0)
        self.assertEqual(scale_by_segment["retreat_place"], 1.0)

    def test_place_pre_open_segments_hold_gripper_closed_without_close_segment(self) -> None:
        payload = _place_payload()
        payload["segments"] = [
            _motion_segment(
                "move_to_pre_place",
                [
                    (0.2, 0.21, 0.22, 0.23, 0.24, 0.25),
                    (0.3, 0.31, 0.32, 0.33, 0.34, 0.35),
                ],
            ),
            *payload["segments"],
        ]
        plan = arm_plan_from_curobo_payload(payload)
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.02,
                arm_command_dt=0.02,
                settle_to_segment_start_duration=0.0,
                post_motion_hold_duration=0.0,
                gripper_move_duration=0.02,
                gripper_hold_duration=0.0,
            ),
        )
        executor.reset(plan)

        actions = _drain_executor(executor)
        first_open_index = next(
            index
            for index, action in enumerate(actions)
            if action.metadata.get("segment_name") == "open_gripper"
        )
        pre_open_motion_actions = [
            action
            for action in actions[:first_open_index]
            if action.metadata.get("segment_type") == "motion"
            and action.metadata.get("segment_name") in {"move_to_pre_place", "approach_to_place"}
        ]

        self.assertTrue(pre_open_motion_actions)
        self.assertTrue(all(action.gripper_command == "close" for action in pre_open_motion_actions))
        self.assertTrue(
            all(
                tuple(action.metadata.get("gripper_joint_positions", ())) == (0.0, 0.0)
                for action in pre_open_motion_actions
            )
        )
        self.assertTrue(
            all(action.metadata.get("gripper_hold_after_close") for action in pre_open_motion_actions)
        )

    def test_place_plan_emits_gripper_open_event_without_internal_world_step(self) -> None:
        plan = arm_plan_from_curobo_payload(_place_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.02,
                gripper_move_duration=0.02,
                gripper_hold_duration=0.0,
                post_open_release_settle_duration=0.04,
            ),
        )
        executor.reset(plan)

        actions = _drain_executor(executor)
        open_actions = [
            action
            for action in actions
            if action.metadata.get("segment_name") == "open_gripper"
        ]

        self.assertEqual(open_actions[0].metadata["event_marker"], "gripper_open")
        self.assertEqual(open_actions[0].gripper_command, "open")
        release_settle_actions = [
            action
            for action in actions
            if action.metadata.get("segment_type") == "post_open_release_settle"
        ]
        self.assertEqual(len(release_settle_actions), 2)
        self.assertTrue(
            all(action.metadata.get("baseline_release_settle") for action in release_settle_actions)
        )
        self.assertTrue(all(action.gripper_command == "open" for action in release_settle_actions))
        self.assertTrue(
            all(
                action.arm_joint_positions
                == (0.4, 0.41, 0.42, 0.43, 0.44, 0.45)
                for action in release_settle_actions
            )
        )
        first_release_settle = actions.index(release_settle_actions[0])
        first_retreat = next(
            index
            for index, action in enumerate(actions)
            if action.metadata.get("segment_name") == "retreat_place"
        )
        self.assertLess(first_release_settle, first_retreat)
        self.assertTrue(all(action.metadata["world_step_owned_by_pipeline"] for action in actions))

    def test_open_gripper_uses_baseline_smoothstep_before_release_settle(self) -> None:
        plan = arm_plan_from_curobo_payload(_place_payload())
        executor = SegmentedArmExecutor(
            BinaryGripperController(),
            config=SegmentedArmExecutorConfig(
                sim_dt=0.10,
                settle_to_segment_start_duration=0.0,
                post_motion_hold_duration=0.0,
                gripper_move_duration=0.30,
                gripper_hold_duration=0.0,
                post_open_release_settle_duration=0.20,
            ),
        )
        executor.reset(plan)
        while executor.status()["current_segment"]["name"] != "open_gripper":
            executor.compute_action(_state())

        release_state = _state_with_all_joints(
            (0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.0, 0.0)
        )
        open_actions = [executor.compute_action(release_state) for _ in range(3)]
        open_targets = [
            tuple(action.metadata["gripper_joint_positions"])
            for action in open_actions
        ]

        self.assertEqual(open_targets[0], (0.0, 0.0))
        self.assertAlmostEqual(open_targets[1][0], 0.02)
        self.assertAlmostEqual(open_targets[1][1], 0.02)
        self.assertEqual(open_targets[2], (0.04, 0.04))
        self.assertEqual(
            executor.status()["current_segment"]["type"],
            "post_open_release_settle",
        )

    def test_invalid_curobo_payload_reports_schema_error(self) -> None:
        payload = _pick_payload()
        payload["segments"][1]["trajectory"]["q"][0] = [0.0, 0.1]

        with self.assertRaises(CuroboPlanFormatError):
            arm_plan_from_curobo_payload(payload)

    def test_json_planner_loads_pick_and_place_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pick_path = Path(tmp_dir) / "pick.json"
            place_path = Path(tmp_dir) / "place.json"
            pick_path.write_text(json.dumps(_pick_payload()), encoding="utf-8")
            place_path.write_text(json.dumps(_place_payload()), encoding="utf-8")
            planner = CuroboJsonManipulationPlanner(
                pick_plan_json=pick_path,
                place_plan_json=place_path,
            )

            pick_plan = planner.plan_pick(_state(), None)  # type: ignore[arg-type]
            place_plan = planner.plan_place(_state(), None)  # type: ignore[arg-type]

        self.assertEqual(pick_plan.operation, "pick")
        self.assertEqual(place_plan.operation, "place")
        self.assertEqual(pick_plan.metadata["source_plan_json"], str(pick_path))
        self.assertEqual(place_plan.metadata["source_plan_json"], str(place_path))

    def test_current_state_target_uses_handoff_base_frame(self) -> None:
        # 该数值来自 integrated 失败样例：导航按 xy-only 成功后 yaw 与离线 plan 不一致。
        T_world_base = pose_to_matrix(
            (1.1991218328475952 + 0.12, 0.9471539855003357, 0.33833712339401245 + 0.05),
            _yaw_quat_wxyz(1.4841108421191422),
        )
        bbox_center = (0.9161067008972168, 1.2011691331863403, 0.8165265917778015)
        bbox_size = (0.07, 0.07, 0.07)
        bbox_min = tuple(center - size * 0.5 for center, size in zip(bbox_center, bbox_size))
        bbox_max = tuple(center + size * 0.5 for center, size in zip(bbox_center, bbox_size))

        payload = build_side_grasp_target_payload(
            object_prim_path="/World/apple",
            T_world_base=T_world_base,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            bbox_center=bbox_center,
            bbox_size=bbox_size,
        )

        grasp_position_base = payload["poses"]["grasp"]["position_xyz"]
        self.assertLess(payload["diagnostics"]["target_workspace_base"]["grasp"]["xy_radius_m"], 0.50)
        self.assertLess(abs(grasp_position_base[0]), 0.30)
        self.assertGreater(grasp_position_base[1], 0.20)
        self.assertEqual(payload["source"]["grasp_mode"], "side")
        self.assertAlmostEqual(
            payload["source"]["applied_grasp_center_z_offset_m"],
            0.0075,
        )
        self.assertEqual(payload["gripper"]["close_m"], 0.0)

    def test_side_grasp_uses_baseline_center_height_without_support_lift(self) -> None:
        payload = build_side_grasp_target_payload(
            object_prim_path="/World/apple",
            T_world_base=pose_to_matrix(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            ),
            bbox_min=(0.8, 1.0, 0.783),
            bbox_max=(0.86, 1.06, 0.850),
            bbox_center=(0.83, 1.03, 0.8165),
            bbox_size=(0.06, 0.06, 0.067),
        )

        source = payload["source"]
        grasp_z = source["world_grasp_contact_pose"]["position_xyz"][2]
        self.assertAlmostEqual(grasp_z, 0.824)
        self.assertAlmostEqual(
            source["estimated_open_gripper_bottom_clearance_m"],
            0.005,
        )
        self.assertAlmostEqual(source["applied_grasp_center_z_offset_m"], 0.0075)
        self.assertLess(grasp_z, source["bbox_world"]["max_xyz"][2])
        self.assertAlmostEqual(payload["gripper"]["close_m"], 0.0)

    def test_arm_place_target_aligns_object_center_not_tcp_to_task_xyz(self) -> None:
        payload = build_arm_place_target_payload(
            object_prim_path="/World/apple",
            T_world_base=pose_to_matrix((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            T_world_tcp=pose_to_matrix((1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)),
            bbox_min=(1.08, -0.02, 0.98),
            bbox_max=(1.12, 0.02, 1.02),
            bbox_center=(1.10, 0.0, 1.0),
            bbox_size=(0.04, 0.04, 0.04),
            place_pose_world={"x": 2.0, "y": 3.0, "z": 0.80},
            pick_grasp_quaternion_base=(1.0, 0.0, 0.0, 0.0),
        )

        place_pose = payload["poses"]["place"]

        # baseline 语义：任务 xyz 是物体中心，不是 TCP。当前 TCP 比物体中心小 0.10m，
        # 所以 release 时 TCP x 也应比目标物体中心小 0.10m。
        self.assertAlmostEqual(place_pose["world"]["position_xyz"][0], 1.90)
        self.assertAlmostEqual(place_pose["world"]["position_xyz"][1], 3.0)
        self.assertAlmostEqual(place_pose["world"]["position_xyz"][2], 0.813)
        self.assertEqual(payload["source"]["desired_final_object_center_world"], [2.0, 3.0, 0.8])
        self.assertEqual(payload["source"]["release_object_center_world"], [2.0, 3.0, 0.8130000000000001])
        self.assertEqual(payload["sequence"], ["pre_place", "place", "open_gripper", "retreat"])

    def test_current_state_pick_planner_calls_exporter_and_replans_place(self) -> None:
        exported_place_pick_quats = []

        class FakeCurrentStateSimulation:
            def export_current_curobo_pick_inputs(self, *, output_dir, episode_spec, state):
                del episode_spec, state
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                state_json = output_path / "pick_state.json"
                target_json = output_path / "pick_target.json"
                state_json.write_text("{}", encoding="utf-8")
                target_json.write_text(
                    json.dumps(
                        {
                            "poses": {
                                "grasp": {
                                    "quaternion_wxyz": [
                                        0.9238795,
                                        0.0,
                                        0.0,
                                        0.3826834,
                                    ]
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "state_json": state_json,
                    "target_json": target_json,
                    "object_prim_path": "/World/apple",
                    "target_grasp_position_base": [0.2, 0.3, 0.4],
                }

            def export_current_curobo_place_inputs(
                self,
                *,
                output_dir,
                episode_spec,
                state,
                pick_grasp_quaternion_base,
            ):
                del episode_spec, state
                exported_place_pick_quats.append(pick_grasp_quaternion_base)
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                state_json = output_path / "place_state.json"
                target_json = output_path / "place_target.json"
                state_json.write_text("{}", encoding="utf-8")
                target_json.write_text("{}", encoding="utf-8")
                return {
                    "state_json": state_json,
                    "target_json": target_json,
                    "object_prim_path": "/World/apple",
                    "pick_grasp_quaternion_base": pick_grasp_quaternion_base,
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            calls = []

            def fake_runner(task):
                calls.append(task)
                payload = _place_payload() if task.curobo_task_mode == "place" else _pick_payload()
                Path(task.plan_json).write_text(json.dumps(payload), encoding="utf-8")
                return payload

            planner = CurrentStateCuroboPlanner(
                simulation=FakeCurrentStateSimulation(),
                config=CurrentStateCuroboPlannerConfig(
                    output_dir=root / "online",
                    project_root=root,
                    place_plan_json=None,
                    use_planner_server=False,
                ),
                plan_runner=fake_runner,
            )

            pick_plan = planner.plan_pick(_state(), None)  # type: ignore[arg-type]
            place_plan = planner.plan_place(_state(), _episode_spec())

        self.assertEqual(len(calls), 2)
        self.assertEqual(pick_plan.operation, "pick")
        self.assertTrue(pick_plan.metadata["current_state_replan"]["enabled"])
        self.assertIn("pick_state.json", calls[0].state_json)
        self.assertEqual(calls[0].curobo_task_mode, "grasp")
        self.assertEqual(place_plan.operation, "place")
        self.assertTrue(place_plan.metadata["current_state_replan"]["enabled"])
        self.assertIn("place_state.json", calls[1].state_json)
        self.assertEqual(calls[1].curobo_task_mode, "place")
        self.assertFalse(calls[1].use_planner_server)
        self.assertEqual(exported_place_pick_quats, [None])
        self.assertFalse(
            place_plan.metadata["current_state_place_orientation"][
                "reuse_pick_grasp_orientation_for_place"
            ]
        )
        self.assertIsNone(
            place_plan.metadata["current_state_place_orientation"][
                "exported_pick_grasp_quaternion_base"
            ]
        )
        self.assertIsNotNone(
            place_plan.metadata["current_state_place_orientation"][
                "stored_pick_grasp_quaternion_base"
            ]
        )

    def test_current_state_place_can_reuse_pick_orientation_when_enabled(self) -> None:
        exported_place_pick_quats = []

        class FakeCurrentStateSimulation:
            def export_current_curobo_pick_inputs(self, *, output_dir, episode_spec, state):
                del episode_spec, state
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                state_json = output_path / "pick_state.json"
                target_json = output_path / "pick_target.json"
                state_json.write_text("{}", encoding="utf-8")
                target_json.write_text(
                    json.dumps(
                        {
                            "poses": {
                                "grasp": {
                                    "quaternion_wxyz": [
                                        0.9238795,
                                        0.0,
                                        0.0,
                                        0.3826834,
                                    ]
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "state_json": state_json,
                    "target_json": target_json,
                    "object_prim_path": "/World/apple",
                }

            def export_current_curobo_place_inputs(
                self,
                *,
                output_dir,
                episode_spec,
                state,
                pick_grasp_quaternion_base,
            ):
                del episode_spec, state
                exported_place_pick_quats.append(pick_grasp_quaternion_base)
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                state_json = output_path / "place_state.json"
                target_json = output_path / "place_target.json"
                state_json.write_text("{}", encoding="utf-8")
                target_json.write_text("{}", encoding="utf-8")
                return {
                    "state_json": state_json,
                    "target_json": target_json,
                    "object_prim_path": "/World/apple",
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            def fake_runner(task):
                payload = _place_payload() if task.curobo_task_mode == "place" else _pick_payload()
                Path(task.plan_json).write_text(json.dumps(payload), encoding="utf-8")
                return payload

            planner = CurrentStateCuroboPlanner(
                simulation=FakeCurrentStateSimulation(),
                config=CurrentStateCuroboPlannerConfig(
                    output_dir=root / "online",
                    project_root=root,
                    place_plan_json=None,
                    use_planner_server=False,
                    reuse_pick_grasp_orientation_for_place=True,
                ),
                plan_runner=fake_runner,
            )

            planner.plan_pick(_state(), None)  # type: ignore[arg-type]
            place_plan = planner.plan_place(_state(), _episode_spec())

        self.assertEqual(len(exported_place_pick_quats), 1)
        self.assertIsNotNone(exported_place_pick_quats[0])
        self.assertTrue(
            place_plan.metadata["current_state_place_orientation"][
                "reuse_pick_grasp_orientation_for_place"
            ]
        )
        self.assertIsNotNone(
            place_plan.metadata["current_state_place_orientation"][
                "exported_pick_grasp_quaternion_base"
            ]
        )

    def test_current_state_planner_rejects_lift_pick_when_side_retreat_expected(self) -> None:
        class FakeCurrentStateSimulation:
            def export_current_curobo_pick_inputs(self, *, output_dir, episode_spec, state):
                del episode_spec, state
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                state_json = output_path / "pick_state.json"
                target_json = output_path / "pick_target.json"
                state_json.write_text("{}", encoding="utf-8")
                target_json.write_text("{}", encoding="utf-8")
                return {
                    "state_json": state_json,
                    "target_json": target_json,
                    "object_prim_path": "/World/apple",
                }

        def lift_runner(task):
            payload = _pick_payload()
            payload["segments"][-1] = _motion_segment(
                "lift_object",
                [
                    (0.2, 0.21, 0.22, 0.23, 0.24, 0.25),
                    (0.1, 0.11, 0.12, 0.13, 0.14, 0.15),
                ],
            )
            Path(task.plan_json).write_text(json.dumps(payload), encoding="utf-8")
            return payload

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            planner = CurrentStateCuroboPlanner(
                simulation=FakeCurrentStateSimulation(),
                config=CurrentStateCuroboPlannerConfig(
                    output_dir=root / "online",
                    project_root=root,
                    place_plan_json=None,
                    use_planner_server=False,
                ),
                plan_runner=lift_runner,
            )

            with self.assertRaisesRegex(RuntimeError, "禁止 lift_object"):
                planner.plan_pick(_state(), None)  # type: ignore[arg-type]


def _drain_executor(executor: SegmentedArmExecutor):
    joint_positions = [0.0] * 8
    state = _state_with_all_joints(tuple(joint_positions))
    actions = []
    while not executor.is_done(state):
        action = executor.compute_action(state)
        actions.append(action)
        if action.arm_joint_positions is not None:
            joint_positions[:6] = action.arm_joint_positions
        gripper_positions = action.metadata.get("gripper_joint_positions")
        if gripper_positions is not None:
            joint_positions[6:8] = gripper_positions
        state = _state_with_all_joints(tuple(joint_positions))
    return actions


def _state() -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0),
        robot_root_velocity=(0.0,) * 6,
    )


def _episode_spec() -> EpisodeSpec:
    return EpisodeSpec(
        task_id=0,
        episode_id=0,
        instruction="pick and place apple",
        scene_usd="",
        nav_map="",
        start=NavGoal(0.0, 0.0, 0.0),
        pick_goal=NavGoal(0.0, 0.0, 0.0),
        place_goal=NavGoal(1.0, 1.0, 0.0),
        object_prim_path="/World/apple",
        object_initial_pose=None,
        place_target_pose=(0.9, 1.0, 0.8, 0.0, 0.0, 0.0),
        raw_task={"place": {"enabled": True, "place_pose_world": {"x": 0.9, "y": 1.0, "z": 0.8}}},
    )


def _state_with_joints(joint_positions: tuple[float, ...]) -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0),
        robot_root_velocity=(0.0,) * 6,
        joint_positions=joint_positions,
        joint_velocities=(0.0,) * len(joint_positions),
        metadata={"joint_names": _arm_joint_names()},
    )


def _state_with_all_joints(joint_positions: tuple[float, ...]) -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0),
        robot_root_velocity=(0.0,) * 6,
        joint_positions=joint_positions,
        joint_velocities=(0.0,) * len(joint_positions),
        metadata={
            "joint_names": tuple(f"arm_joint{index}" for index in range(1, 9)),
        },
    )


def _arm_joint_names() -> tuple[str, ...]:
    return (
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
    )


def _yaw_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def _pick_payload() -> dict:
    return {
        "schema_version": 1,
        "robot_name": "go2_x5",
        "planner": "curobo.MotionPlanner.plan_pose merged pregrasp-grasp",
        "joint_names": list(_arm_joint_names()),
        "tool_frame": "grasp_tcp_link",
        "object_prim_path": "/World/apple",
        "grasp_mode": "side",
        "segments": [
            {
                "name": "open_gripper",
                "type": "gripper",
                "joint_names": ["arm_joint7", "arm_joint8"],
                "target_position": [0.04, 0.04],
            },
            _motion_segment(
                "move_to_pregrasp",
                [
                    (0.0, 0.01, 0.02, 0.03, 0.04, 0.05),
                    (0.1, 0.11, 0.12, 0.13, 0.14, 0.15),
                ],
            ),
            _motion_segment(
                "approach_to_grasp",
                [
                    (0.1, 0.11, 0.12, 0.13, 0.14, 0.15),
                    (0.2, 0.21, 0.22, 0.23, 0.24, 0.25),
                ],
            ),
            {
                "name": "close_gripper",
                "type": "gripper",
                "joint_names": ["arm_joint7", "arm_joint8"],
                "target_position": [0.0, 0.0],
            },
            _motion_segment(
                "retreat_object",
                [
                    (0.2, 0.21, 0.22, 0.23, 0.24, 0.25),
                    (0.3, 0.31, 0.32, 0.33, 0.34, 0.35),
                ],
            ),
            _motion_segment(
                "return_home_after_retreat",
                [
                    (0.3, 0.31, 0.32, 0.33, 0.34, 0.35),
                    (0.0, 0.01, 0.02, 0.03, 0.04, 0.05),
                ],
            ),
        ],
        "summary": {"all_motion_segments_success": True},
    }


def _place_payload() -> dict:
    return {
        "schema_version": 1,
        "robot_name": "go2_x5",
        "planner": "curobo.MotionPlanner.plan_pose arm_place",
        "joint_names": list(_arm_joint_names()),
        "tool_frame": "grasp_tcp_link",
        "object_prim_path": "/World/apple",
        "place_mode": "arm_place",
        "segments": [
            _motion_segment(
                "approach_to_place",
                [
                    (0.3, 0.31, 0.32, 0.33, 0.34, 0.35),
                    (0.4, 0.41, 0.42, 0.43, 0.44, 0.45),
                ],
            ),
            {
                "name": "open_gripper",
                "type": "gripper",
                "joint_names": ["arm_joint7", "arm_joint8"],
                "target_position": [0.04, 0.04],
            },
            _motion_segment(
                "retreat_place",
                [
                    (0.4, 0.41, 0.42, 0.43, 0.44, 0.45),
                    (0.1, 0.11, 0.12, 0.13, 0.14, 0.15),
                ],
            ),
        ],
        "summary": {"all_motion_segments_success": True},
    }


def _motion_segment(name: str, q_rows: list[tuple[float, ...]]) -> dict:
    return {
        "name": name,
        "type": "motion",
        "target_name": name,
        "timing": {
            "dt": 0.02,
            "duration_s": 0.02 * (len(q_rows) - 1),
            "num_waypoints": len(q_rows),
        },
        "final_error": {"position_m": 0.0, "orientation_deg": 0.0},
        "plan_info": {"planner_success": True},
        "trajectory": {
            "time_from_start": [0.02 * index for index in range(len(q_rows))],
            "q": [list(row) for row in q_rows],
        },
    }


if __name__ == "__main__":
    unittest.main()
