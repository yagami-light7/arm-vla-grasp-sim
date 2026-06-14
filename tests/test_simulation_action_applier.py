"""simulation action applier 单元测试。"""

from __future__ import annotations

import unittest

from source.interfaces import RobotAction
from source.simulation.action_applier import NamedJointActionApplier
from source.simulation.isaac_runtime import IsaacSimulationRuntime


class FakeArticulationAction:
    def __init__(self, *, joint_positions, joint_indices):
        self.joint_positions = tuple(float(value) for value in joint_positions)
        self.joint_indices = tuple(int(value) for value in joint_indices)


class FakeRobot:
    def __init__(self, dof_names: tuple[str, ...]):
        self.dof_names = dof_names
        self.actions: list[FakeArticulationAction] = []

    def apply_action(self, action: FakeArticulationAction) -> None:
        self.actions.append(action)


class FakeJointApplier:
    def __init__(self):
        self.actions: list[RobotAction] = []

    def apply(self, action: RobotAction) -> dict:
        self.actions.append(action)
        return {"applied": True, "source": action.source}


class SimulationActionApplierTest(unittest.TestCase):
    def test_arm_action_maps_joint_names_to_dof_indices(self) -> None:
        robot = FakeRobot(_dof_names())
        applier = NamedJointActionApplier(
            robot,
            articulation_action_factory=FakeArticulationAction,
        )

        report = applier.apply(
            RobotAction(
                arm_joint_positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                source="arm_pick",
            )
        )

        self.assertTrue(report["applied"])
        self.assertTrue(report["arm_targeted"])
        self.assertFalse(report["uses_direct_joint_state"])
        self.assertEqual(robot.actions[0].joint_indices, (2, 3, 4, 5, 6, 7))
        self.assertEqual(robot.actions[0].joint_positions, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))

    def test_arm_joint_names_can_come_from_action_metadata(self) -> None:
        robot = FakeRobot(_dof_names())
        applier = NamedJointActionApplier(
            robot,
            articulation_action_factory=FakeArticulationAction,
        )

        report = applier.apply(
            RobotAction(
                arm_joint_positions=(0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
                source="arm_pick",
                metadata={
                    "arm_joint_names": (
                        "arm_joint6",
                        "arm_joint5",
                        "arm_joint4",
                        "arm_joint3",
                        "arm_joint2",
                        "arm_joint1",
                    )
                },
            )
        )

        self.assertTrue(report["applied"])
        self.assertEqual(robot.actions[0].joint_indices, (7, 6, 5, 4, 3, 2))
        self.assertEqual(robot.actions[0].joint_positions, (0.6, 0.5, 0.4, 0.3, 0.2, 0.1))

    def test_arm_and_metadata_gripper_targets_are_sent_as_one_action(self) -> None:
        robot = FakeRobot(_dof_names())
        applier = NamedJointActionApplier(
            robot,
            articulation_action_factory=FakeArticulationAction,
        )

        report = applier.apply(
            RobotAction(
                arm_joint_positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                gripper_command="close",
                source="arm_pick",
                metadata={
                    "gripper_joint_names": ("arm_joint7", "arm_joint8"),
                    "gripper_joint_positions": (0.01, 0.02),
                },
            )
        )

        self.assertTrue(report["applied"])
        self.assertTrue(report["gripper_targeted"])
        self.assertEqual(robot.actions[0].joint_indices, (2, 3, 4, 5, 6, 7, 8, 9))
        self.assertEqual(
            robot.actions[0].joint_positions,
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01, 0.02),
        )

    def test_default_gripper_open_close_positions_are_available(self) -> None:
        robot = FakeRobot(_dof_names())
        applier = NamedJointActionApplier(
            robot,
            articulation_action_factory=FakeArticulationAction,
        )

        close_report = applier.apply(RobotAction(gripper_command="close", source="close"))
        open_report = applier.apply(RobotAction(gripper_command="open", source="open"))

        self.assertTrue(close_report["applied"])
        self.assertEqual(robot.actions[0].joint_indices, (8, 9))
        self.assertEqual(robot.actions[0].joint_positions, (0.0, 0.0))
        self.assertTrue(open_report["applied"])
        self.assertEqual(robot.actions[1].joint_positions, (0.04, 0.04))

    def test_hold_without_explicit_target_does_not_apply_joint_action(self) -> None:
        robot = FakeRobot(_dof_names())
        applier = NamedJointActionApplier(
            robot,
            articulation_action_factory=FakeArticulationAction,
        )

        report = applier.apply(RobotAction(gripper_command="hold", source="hold"))

        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "no_joint_targets")
        self.assertEqual(robot.actions, [])

    def test_length_mismatch_is_rejected_before_apply_action(self) -> None:
        robot = FakeRobot(_dof_names())
        applier = NamedJointActionApplier(
            robot,
            articulation_action_factory=FakeArticulationAction,
        )

        with self.assertRaises(RuntimeError):
            applier.apply(RobotAction(arm_joint_positions=(0.1, 0.2), source="bad_arm"))

        self.assertEqual(robot.actions, [])

    def test_missing_joint_name_is_rejected_before_apply_action(self) -> None:
        robot = FakeRobot(tuple(name for name in _dof_names() if name != "arm_joint8"))
        applier = NamedJointActionApplier(
            robot,
            articulation_action_factory=FakeArticulationAction,
        )

        with self.assertRaises(RuntimeError):
            applier.apply(RobotAction(gripper_command="open", source="bad_gripper"))

        self.assertEqual(robot.actions, [])

    def test_isaac_runtime_apply_rejects_base_command(self) -> None:
        runtime = _fake_runtime()

        with self.assertRaises(RuntimeError):
            runtime.apply(RobotAction(base_velocity=(0.1, 0.0, 0.0), source="nav"))

    def test_isaac_runtime_apply_delegates_joint_action_and_records_report(self) -> None:
        runtime = _fake_runtime()
        action = RobotAction(
            arm_joint_positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            gripper_command="close",
            source="arm_pick",
        )

        runtime.apply(action)

        self.assertEqual(runtime._joint_action_applier.actions, [action])  # type: ignore[attr-defined]
        self.assertEqual(runtime._last_action, action)
        self.assertEqual(
            runtime._metadata["last_joint_action_report"],  # type: ignore[attr-defined]
            {"applied": True, "source": "arm_pick"},
        )
        self.assertEqual(runtime._metadata["joint_action_apply_count"], 1)  # type: ignore[attr-defined]
        self.assertEqual(runtime._metadata["arm_joint_action_apply_count"], 0)  # type: ignore[attr-defined]

    def test_arm_tracking_report_is_consumed_after_world_step(self) -> None:
        runtime = _fake_runtime()
        runtime._step_calls = 3  # type: ignore[attr-defined]
        runtime._pending_arm_tracking_target = {  # type: ignore[attr-defined]
            "source": "arm_pick",
            "operation": "pick",
            "segment_index": 0,
            "segment_name": "json_pick_motion",
            "segment_type": "motion",
            "segment_tick": 4,
            "segment_ticks": 10,
            "joint_names": ("arm_joint1", "arm_joint2", "arm_joint3"),
            "joint_indices": (1, 2, 3),
            "target_positions": (0.10, 0.20, 0.30),
            "applied_step_index": 3,
        }

        runtime._consume_pending_arm_tracking((0.0, 0.10, 0.20, 0.30))  # type: ignore[attr-defined]

        self.assertIsNotNone(runtime._pending_arm_tracking_target)  # type: ignore[attr-defined]
        runtime._step_calls = 4  # type: ignore[attr-defined]

        runtime._consume_pending_arm_tracking((0.0, 0.12, 0.17, 0.36))  # type: ignore[attr-defined]

        report = runtime._metadata["last_arm_tracking_report"]  # type: ignore[attr-defined]
        self.assertTrue(report["tracked"])
        self.assertEqual(report["segment_name"], "json_pick_motion")
        self.assertAlmostEqual(report["max_abs_error"], 0.06)
        self.assertAlmostEqual(report["mean_abs_error"], (0.02 + 0.03 + 0.06) / 3)
        aggregate = runtime._metadata["arm_tracking_report"]  # type: ignore[attr-defined]
        self.assertEqual(aggregate["sample_count"], 1)
        self.assertAlmostEqual(aggregate["max_abs_error"], 0.06)
        self.assertEqual(aggregate["peak_report"]["joint_name"], "arm_joint3")
        self.assertEqual(aggregate["peak_report"]["segment_tick"], 4)
        self.assertAlmostEqual(aggregate["peak_report"]["actual_position"], 0.36)
        self.assertAlmostEqual(aggregate["segments"]["json_pick_motion"]["peak_report"]["abs_error"], 0.06)
        self.assertAlmostEqual(aggregate["joints"]["arm_joint2"]["max_abs_error"], 0.03)
        self.assertAlmostEqual(aggregate["joints"]["arm_joint3"]["peak_report"]["abs_error"], 0.06)
        self.assertEqual(aggregate["segments"]["json_pick_motion"]["sample_count"], 1)
        self.assertEqual(
            runtime._metadata["arm_tracking_peak_report"],  # type: ignore[attr-defined]
            aggregate["peak_report"],
        )
        self.assertIsNone(runtime._pending_arm_tracking_target)  # type: ignore[attr-defined]


def _dof_names() -> tuple[str, ...]:
    return (
        "FL_hip_joint",
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


def _fake_runtime() -> IsaacSimulationRuntime:
    runtime = object.__new__(IsaacSimulationRuntime)
    runtime._joint_action_applier = FakeJointApplier()
    runtime._pending_arm_tracking_target = None
    runtime._step_calls = 0
    runtime._metadata = {
        "last_joint_action_report": None,
        "last_arm_tracking_report": None,
        "arm_tracking_peak_report": None,
        "arm_tracking_report": {
            "sample_count": 0,
            "max_abs_error": None,
            "peak_report": None,
            "latest_max_abs_error": None,
            "latest_mean_abs_error": None,
            "segments": {},
            "joints": {},
        },
        "arm_tracking_sample_count": 0,
        "arm_tracking_max_abs_error": None,
        "joint_action_apply_count": 0,
        "arm_joint_action_apply_count": 0,
        "gripper_joint_action_apply_count": 0,
        "gripper_close_apply_count": 0,
        "gripper_open_apply_count": 0,
    }
    runtime._last_action = RobotAction.idle()
    return runtime


if __name__ == "__main__":
    unittest.main()
