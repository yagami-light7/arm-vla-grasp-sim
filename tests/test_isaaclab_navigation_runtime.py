"""IsaacLab navigation runtime action 边界测试。"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np

from source.interfaces import EpisodeSpec, NavGoal, RobotAction
from source.navigation.adapters.isaaclab_go2_adapter import (
    ARM_JOINT_NAMES,
    DOG_JOINT_NAMES,
    Go2LocomotionAdapter,
)
from source.simulation.isaaclab_runtime import (
    IsaacLabNavigationRuntime,
    IsaacLabNavigationRuntimeConfig,
    _collision_candidate_sort_key,
    _dedupe_root_paths,
    _path_is_excluded_by_roots,
    _prim_keyword_match_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakePolicyAction:
    def __init__(self):
        self.to_devices: list[str] = []

    def to(self, device: str) -> "FakePolicyAction":
        self.to_devices.append(device)
        return self


class FakeActionManager:
    def __init__(self):
        self.processed_actions: list[FakePolicyAction] = []

    def process_action(self, action: FakePolicyAction) -> None:
        self.processed_actions.append(action)


class FakeRuntime:
    def __init__(self):
        self.device = "cpu"
        self.action_manager = FakeActionManager()


class FakeAdapter:
    def __init__(self):
        self.arm_joint_ids = (0, 1, 2, 3, 4, 5)
        self.base_commands: list[tuple[float, float, float]] = []
        self.arm_targets: list[tuple[float, ...]] = []
        self.arm_override_flags: list[bool] = []
        self.gripper_targets: list[tuple[float, ...]] = []
        self.base_pose_lock_flags: list[bool] = []
        self.support_joint_lock_flags: list[bool] = []
        self.base_pose_lock_apply_count = 0
        self.support_joint_lock_apply_count = 0
        self.arm_position_target_apply_count = 0
        self.gripper_position_target_apply_count = 0
        self.refresh_flags: list[bool] = []
        self.policy_action = FakePolicyAction()

    def apply_base_command(self, vx: float, vy: float, wz: float) -> None:
        self.base_commands.append((float(vx), float(vy), float(wz)))

    def set_gripper_joint_target(self, target) -> None:
        self.gripper_targets.append(tuple(float(value) for value in target))

    def set_arm_joint_target(self, target) -> None:
        self.arm_targets.append(tuple(float(value) for value in target))

    def set_direct_arm_action_override(self, enabled: bool = True) -> dict:
        self.arm_override_flags.append(bool(enabled))
        return {
            "enabled": bool(enabled),
            "action_term_available": True,
            "arm_action_indices": [12, 13, 14, 15, 16, 17],
            "arm_joint_names": [
                "arm_joint1",
                "arm_joint2",
                "arm_joint3",
                "arm_joint4",
                "arm_joint5",
                "arm_joint6",
            ],
        }

    def set_base_pose_lock(self, enabled: bool = True, pose_xyyaw=None) -> dict:
        del pose_xyyaw
        self.base_pose_lock_flags.append(bool(enabled))
        return {
            "enabled": bool(enabled),
            "pose_xyzyaw": [1.0, 2.0, 0.35, 0.5] if enabled else None,
            "pose_xyyaw": [1.0, 2.0, 0.5] if enabled else None,
        }

    def apply_base_pose_lock(self) -> dict:
        self.base_pose_lock_apply_count += 1
        return {
            "applied": True,
            "pose_xyzyaw": [1.0, 2.0, 0.35, 0.5],
            "uses_direct_root_state": True,
        }

    def set_support_joint_lock(self, enabled: bool = True) -> dict:
        self.support_joint_lock_flags.append(bool(enabled))
        return {
            "enabled": bool(enabled),
            "joint_names": [f"dog_joint{index}" for index in range(12)] if enabled else [],
            "joint_ids": list(range(12)) if enabled else [],
            "action_indices": list(range(12)),
        }

    def apply_support_joint_lock(self) -> dict:
        self.support_joint_lock_apply_count += 1
        return {
            "applied": True,
            "joint_names": [f"dog_joint{index}" for index in range(12)],
            "joint_ids": list(range(12)),
            "action_indices": list(range(12)),
            "uses_direct_joint_state": False,
            "lock_mode": "position_velocity_target_only",
        }

    def apply_arm_joint_target(self) -> dict:
        if not self.arm_targets:
            return {"applied": False, "reason": "arm_joint_target_disabled"}
        self.arm_position_target_apply_count += 1
        return {
            "applied": True,
            "joint_names": [
                "arm_joint1",
                "arm_joint2",
                "arm_joint3",
                "arm_joint4",
                "arm_joint5",
                "arm_joint6",
            ],
            "joint_ids": list(self.arm_joint_ids),
            "target_positions": list(self.arm_targets[-1]),
            "control_mode": "position_target_only",
            "velocity_target_written": False,
            "uses_direct_joint_state": False,
        }

    def apply_gripper_joint_target(self) -> dict:
        if not self.gripper_targets:
            return {"applied": False, "reason": "gripper_joint_target_disabled"}
        self.gripper_position_target_apply_count += 1
        return {
            "applied": True,
            "joint_names": ["arm_joint7", "arm_joint8"],
            "joint_ids": [6, 7],
            "target_positions": list(self.gripper_targets[-1]),
            "uses_direct_joint_state": False,
        }

    def compute_policy_action(self, *, refresh_observations: bool = True) -> FakePolicyAction:
        self.refresh_flags.append(bool(refresh_observations))
        return self.policy_action


class IsaacLabNavigationRuntimeActionTest(unittest.TestCase):
    def test_world_collision_padding_matches_stable_baseline(self) -> None:
        config = IsaacLabNavigationRuntimeConfig()

        self.assertEqual(config.world_collision_padding_m, 0.02)
        self.assertEqual(config.world_collision_vertical_padding_m, 0.02)
        self.assertEqual(config.world_collision_min_size_m, 0.01)
        self.assertEqual(config.world_collision_max_obstacles, 16)
        self.assertEqual(config.world_collision_local_radius_m, 1.25)
        self.assertFalse(config.world_collision_clip_large_support_obstacles)
        self.assertEqual(config.world_collision_large_obstacle_clip_half_extent_m, 0.45)

    def test_place_export_builds_world_collision_metadata_before_payload(self) -> None:
        source = inspect.getsource(
            IsaacLabNavigationRuntime.export_current_curobo_place_inputs
        )
        assignment = (
            "world_collision_metadata = "
            "self._world_collision_export_metadata(collision_cuboids)"
        )

        self.assertIn(assignment, source)
        self.assertLess(
            source.index(assignment),
            source.index("world_collision_metadata=world_collision_metadata"),
        )

    def test_legacy_place_height_fields_do_not_override_baseline_clearances(self) -> None:
        runtime = object.__new__(IsaacLabNavigationRuntime)
        runtime._config = IsaacLabNavigationRuntimeConfig()
        episode_spec = EpisodeSpec(
            task_id=0,
            episode_id=0,
            instruction="place apple",
            scene_usd="",
            nav_map="",
            start=NavGoal(0.0, 0.0, 0.0),
            pick_goal=NavGoal(0.0, 0.0, 0.0),
            place_goal=NavGoal(0.0, 0.0, 0.0),
            object_prim_path="/World/apple",
            object_initial_pose=None,
            place_target_pose=(0.6, 5.0, 0.72, 0.0, 0.0, 0.0),
            raw_task={
                "place": {
                    "place_pose_world": {"x": 0.6, "y": 5.0, "z": 0.72},
                    "release_height": 0.04,
                    "retreat_height": 0.12,
                }
            },
        )

        payload = runtime._place_pose_world_from_episode(episode_spec)

        self.assertEqual(payload["release_clearance"], 0.013)
        self.assertEqual(payload["pre_place_clearance"], 0.06)
        self.assertNotIn("retreat_clearance", payload)

    def test_explicit_place_clearance_fields_are_preserved(self) -> None:
        runtime = object.__new__(IsaacLabNavigationRuntime)
        runtime._config = IsaacLabNavigationRuntimeConfig()
        episode_spec = EpisodeSpec(
            task_id=0,
            episode_id=0,
            instruction="place apple",
            scene_usd="",
            nav_map="",
            start=NavGoal(0.0, 0.0, 0.0),
            pick_goal=NavGoal(0.0, 0.0, 0.0),
            place_goal=NavGoal(0.0, 0.0, 0.0),
            object_prim_path="/World/apple",
            object_initial_pose=None,
            place_target_pose=(0.6, 5.0, 0.72, 0.0, 0.0, 0.0),
            raw_task={
                "place": {
                    "place_pose_world": {"x": 0.6, "y": 5.0, "z": 0.72},
                    "release_clearance": 0.02,
                    "pre_place_clearance": 0.08,
                    "retreat_clearance": 0.10,
                }
            },
        )

        payload = runtime._place_pose_world_from_episode(episode_spec)

        self.assertEqual(payload["release_clearance"], 0.02)
        self.assertEqual(payload["pre_place_clearance"], 0.08)
        self.assertEqual(payload["retreat_clearance"], 0.10)

    def test_distractor_root_paths_match_baseline_asset_metadata(self) -> None:
        class FakePrim:
            def GetPath(self):
                return "/World/Props/object_03"

            def GetName(self):
                return "object_03"

            def GetMetadata(self, name: str):
                if name == "references":
                    return "@/Assets/Fruits/Apple/Apple.usd@"
                return None

        match_text = _prim_keyword_match_text(FakePrim())

        self.assertIn("apple", match_text)
        self.assertEqual(
            _dedupe_root_paths(
                [
                    "/World/apple_03",
                    "/World/apple_03/Apple_M",
                    "/World/orange_01",
                ]
            ),
            ["/World/apple_03", "/World/orange_01"],
        )

    def test_curobo_collision_excludes_only_hidden_distractor_subtrees(self) -> None:
        hidden_roots = ("/World/apple_03", "/World/orange_01")

        self.assertTrue(
            _path_is_excluded_by_roots(
                "/World/apple_03/Apple_M/Apple_0",
                hidden_roots,
            )
        )
        self.assertFalse(_path_is_excluded_by_roots("/World/apple", hidden_roots))
        self.assertFalse(_path_is_excluded_by_roots("/World/Table", hidden_roots))

    def test_curobo_collision_candidates_prioritize_object_near_table(self) -> None:
        candidates = [
            {
                "prim_path": f"/World/nav_collision/terrain_{index:02d}",
                "distance_to_reference_xy_m": 0.4 + index * 0.01,
            }
            for index in range(20)
        ]
        candidates.append(
            {
                "prim_path": "/World/Table/collision",
                "distance_to_reference_xy_m": 0.0,
            }
        )

        selected = sorted(candidates, key=_collision_candidate_sort_key)[:16]

        self.assertEqual(selected[0]["prim_path"], "/World/Table/collision")
        self.assertNotIn("/World/nav_collision/terrain_19", {
            candidate["prim_path"] for candidate in selected
        })

    def test_runtime_defaults_to_current_scene_camera(self) -> None:
        config = IsaacLabNavigationRuntimeConfig()

        self.assertEqual(config.viewport_camera_prim_path, "/World/Camera1")
        self.assertTrue(config.hide_navigation_collision_visual)

    def test_runtime_reads_front_and_wrist_camera_images(self) -> None:
        class FakeSensor:
            def __init__(self, value: int):
                self.data = type(
                    "Data",
                    (),
                    {
                        "output": {
                            "rgb": np.full(
                                (1, 4, 6, 4),
                                value,
                                dtype=np.uint8,
                            )
                        }
                    },
                )()

        runtime = object.__new__(IsaacLabNavigationRuntime)
        runtime._config = IsaacLabNavigationRuntimeConfig(
            enable_front_camera=True,
            enable_wrist_camera=True,
        )
        runtime._metadata = {}
        runtime._runtime = type(
            "Runtime",
            (),
            {
                "scene": {
                    "head_camera": FakeSensor(11),
                    "arm_camera": FakeSensor(22),
                }
            },
        )()

        images = runtime._read_camera_images()

        self.assertEqual(set(images), {"front", "wrist"})
        self.assertEqual(images["front"].shape, (4, 6, 3))
        self.assertEqual(images["wrist"].shape, (4, 6, 3))
        self.assertTrue(np.all(images["wrist"] == 22))
        self.assertEqual(
            runtime._metadata["camera_capture_report"]["available_camera_keys"],
            ["front", "wrist"],
        )
        self.assertEqual(
            runtime._metadata["camera_capture_report"]["missing_camera_keys"],
            [],
        )

    def test_wrist_camera_matches_dwa_ground_pick_mount(self) -> None:
        source_text = (
            PROJECT_ROOT / "source/simulation/isaaclab_runtime.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'prim_path="{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera"',
            source_text,
        )
        self.assertIn("focal_length=18.0", source_text)
        self.assertIn("pos=(0.08657, 0.0, 0.0)", source_text)
        self.assertIn("rot=(0.5, -0.5, 0.5, -0.5)", source_text)

    def test_object_pose_writer_reuses_existing_xform_ops(self) -> None:
        source_text = (
            PROJECT_ROOT / "source/simulation/isaaclab_runtime.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _get_or_add_xform_op", source_text)
        self.assertIn("return UsdGeom.XformOp(attr)", source_text)
        self.assertIn(
            "_get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)",
            source_text,
        )
        self.assertIn(
            "_get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)",
            source_text,
        )

    def test_go2_x5_arm_uses_implicit_drive_for_manipulation_tracking(self) -> None:
        asset_path = PROJECT_ROOT / "source/robot_lab/robot_lab/assets/go2_x5.py"
        text = asset_path.read_text(encoding="utf-8")

        self.assertIn('"arm": ImplicitActuatorCfg(', text)
        self.assertIn('joint_names_expr=["arm_joint[1-6]"]', text)
        self.assertIn("effort_limit_sim=100.0", text)
        self.assertIn("velocity_limit_sim=10.0", text)
        self.assertIn("stiffness=1000.0", text)
        self.assertIn("damping=50.0", text)

    def test_direct_arm_override_bypasses_policy_clip(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not available")

        class FakeBaseCommandTerm:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.vel_command_b = torch.zeros((1, 3), dtype=torch.float32)
                self.is_heading_env = torch.ones((1,), dtype=torch.bool)
                self.is_standing_env = torch.zeros((1,), dtype=torch.bool)
                self.heading_target = torch.ones((1,), dtype=torch.float32)

        class FakeArmCommandTerm:
            def __init__(self) -> None:
                self.command_buffer = torch.zeros((1, 6), dtype=torch.float32)

        class FakeJointPositionActionTerm:
            def __init__(self) -> None:
                self._joint_names = [*DOG_JOINT_NAMES, *ARM_JOINT_NAMES]
                self._scale = torch.tensor([1.0] * 12 + [0.10] * 6, dtype=torch.float32)
                self._offset = torch.zeros((18,), dtype=torch.float32)

        class FakeEnv:
            clip_actions = 1.0

            def get_observations(self):
                return {"policy": torch.zeros((1, 1), dtype=torch.float32)}

        adapter = object.__new__(Go2LocomotionAdapter)
        adapter.env = FakeEnv()
        adapter.policy = lambda _observations: torch.full((1, 18), 2.0, dtype=torch.float32)
        adapter.observations = {}
        adapter.base_cmd_term = FakeBaseCommandTerm()
        adapter.arm_term = FakeArmCommandTerm()
        adapter.joint_pos_action_term = FakeJointPositionActionTerm()
        adapter.dog_action_indices = list(range(12))
        adapter.arm_action_indices = list(range(12, 18))
        adapter.direct_arm_action_override = True
        adapter.gripper_joint_ids = ()
        adapter._base_pose_lock_xyzyaw = None
        adapter._dog_joint_lock_target = None
        adapter._command = (0.0, 0.0, 0.0)
        adapter._arm_joint_target = (1.0, -1.0, 0.5, 0.0, 0.2, -0.3)
        adapter._gripper_joint_target = None
        adapter._last_actions = None

        actions = adapter.compute_policy_action(refresh_observations=True)

        # locomotion policy 仍受 clip_actions 约束；机械臂直接目标必须绕过该裁剪。
        self.assertAlmostEqual(float(actions[0, 0]), 1.0)
        self.assertEqual(
            [round(float(value), 4) for value in actions[0, 12:18]],
            [10.0, -10.0, 5.0, 0.0, 2.0, -3.0],
        )

    def test_support_joint_lock_does_not_write_joint_state(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not available")

        class FakeRobot:
            def __init__(self) -> None:
                self.data = type(
                    "FakeRobotData",
                    (),
                    {"joint_pos": torch.arange(20, dtype=torch.float32).reshape(1, 20)},
                )()
                self.position_target_calls: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
                self.velocity_target_calls: list[tuple[tuple[float, ...], tuple[int, ...]]] = []

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                self.position_target_calls.append(
                    (
                        tuple(float(value) for value in target.reshape(-1).tolist()),
                        tuple(int(index) for index in joint_ids),
                    )
                )

            def set_joint_velocity_target(self, target, *, joint_ids) -> None:
                self.velocity_target_calls.append(
                    (
                        tuple(float(value) for value in target.reshape(-1).tolist()),
                        tuple(int(index) for index in joint_ids),
                    )
                )

            def write_joint_state_to_sim(self, *_args, **_kwargs) -> None:
                raise AssertionError("support lock must not write joint state")

        adapter = object.__new__(Go2LocomotionAdapter)
        adapter.runtime = type("FakeRuntime", (), {"device": "cpu"})()
        adapter.robot = FakeRobot()
        adapter.dog_joint_ids = [1, 6, 11, 0, 5, 10, 3, 8, 13, 2, 7, 12]
        adapter.dog_action_indices = list(range(12))
        adapter._dog_joint_lock_target = None

        enable_report = adapter.set_support_joint_lock(True)
        apply_report = adapter.apply_support_joint_lock()

        self.assertTrue(enable_report["enabled"])
        self.assertTrue(apply_report["applied"])
        self.assertFalse(apply_report["uses_direct_joint_state"])
        self.assertEqual(apply_report["lock_mode"], "position_velocity_target_only")
        self.assertEqual(len(adapter.robot.position_target_calls), 1)
        self.assertEqual(len(adapter.robot.velocity_target_calls), 1)
        self.assertEqual(adapter.robot.position_target_calls[0][1], tuple(adapter.dog_joint_ids))
        self.assertEqual(adapter.robot.velocity_target_calls[0][0], (0.0,) * len(adapter.dog_joint_ids))

    def test_arm_target_does_not_write_zero_velocity_target(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not available")

        class FakeRobot:
            def __init__(self) -> None:
                self.position_target_calls: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
                self.velocity_target_calls: list[tuple[tuple[float, ...], tuple[int, ...]]] = []

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                self.position_target_calls.append(
                    (
                        tuple(float(value) for value in target.reshape(-1).tolist()),
                        tuple(int(index) for index in joint_ids),
                    )
                )

            def set_joint_velocity_target(self, target, *, joint_ids) -> None:
                self.velocity_target_calls.append(
                    (
                        tuple(float(value) for value in target.reshape(-1).tolist()),
                        tuple(int(index) for index in joint_ids),
                    )
                )

        adapter = object.__new__(Go2LocomotionAdapter)
        adapter.runtime = type("FakeRuntime", (), {"device": "cpu"})()
        adapter.robot = FakeRobot()
        adapter.arm_joint_ids = [2, 3, 4, 5, 6, 7]
        adapter._arm_joint_target = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

        report = adapter.apply_arm_joint_target()

        self.assertTrue(report["applied"])
        self.assertEqual(report["control_mode"], "position_target_only")
        self.assertFalse(report["velocity_target_written"])
        self.assertEqual(len(adapter.robot.position_target_calls), 1)
        # 机械臂是运动目标，不能像支撑腿锁一样每拍写 0 速度。
        self.assertEqual(adapter.robot.velocity_target_calls, [])

    def test_apply_accepts_base_and_gripper_close(self) -> None:
        runtime, adapter, fake_runtime = _fake_runtime()
        action = RobotAction(
            base_velocity=(0.1, -0.2, 0.3),
            gripper_command="close",
            source="nav_carry",
        )

        runtime.apply(action)

        self.assertEqual(adapter.base_commands, [(0.1, -0.2, 0.3)])
        self.assertEqual(adapter.gripper_targets, [(0.0, 0.0)])
        self.assertEqual(adapter.refresh_flags, [True])
        self.assertEqual(fake_runtime.action_manager.processed_actions, [adapter.policy_action])
        self.assertEqual(adapter.policy_action.to_devices, ["cpu"])
        self.assertTrue(runtime._action_prepared)  # type: ignore[attr-defined]
        self.assertEqual(runtime._last_action, action)  # type: ignore[attr-defined]
        self.assertEqual(
            runtime._metadata["last_gripper_action_report"],  # type: ignore[attr-defined]
            {
                "target_staged": True,
                "gripper_command": "close",
                "target_source": "command_close",
                "gripper_joint_names": ("arm_joint7", "arm_joint8"),
                "gripper_joint_positions": (0.0, 0.0),
                "world_step_owned_by_pipeline": True,
            },
        )

    def test_apply_accepts_explicit_gripper_hold_target(self) -> None:
        runtime, adapter, _fake_runtime_obj = _fake_runtime()
        action = RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            gripper_command="hold",
            source="nav_hold_gripper",
            metadata={
                "gripper_joint_names": ("arm_joint7", "arm_joint8"),
                "gripper_joint_positions": (0.012, 0.013),
            },
        )

        runtime.apply(action)

        self.assertEqual(adapter.gripper_targets, [(0.012, 0.013)])
        self.assertEqual(
            runtime._metadata["last_gripper_action_report"]["target_source"],  # type: ignore[attr-defined]
            "metadata",
        )

    def test_apply_hold_without_target_keeps_existing_adapter_target(self) -> None:
        runtime, adapter, _fake_runtime_obj = _fake_runtime()

        runtime.apply(RobotAction(gripper_command="hold", source="nav_hold"))

        self.assertEqual(adapter.gripper_targets, [])
        self.assertEqual(
            runtime._metadata["last_gripper_action_report"],  # type: ignore[attr-defined]
            {
                "target_staged": False,
                "gripper_command": "hold",
                "reason": "hold_without_explicit_target",
            },
        )

    def test_apply_accepts_base_arm_home_and_gripper_close_in_one_tick(self) -> None:
        runtime, adapter, fake_runtime = _fake_runtime()
        action = RobotAction(
            base_velocity=(0.1, 0.0, 0.2),
            arm_joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            gripper_command="close",
            source="nav_carry_home",
            metadata={
                "arm_joint_names": (
                    "arm_joint1",
                    "arm_joint2",
                    "arm_joint3",
                    "arm_joint4",
                    "arm_joint5",
                    "arm_joint6",
                ),
            },
        )

        runtime.apply(action)

        self.assertEqual(adapter.base_commands, [(0.1, 0.0, 0.2)])
        self.assertEqual(adapter.arm_targets, [(0.0,) * 6])
        self.assertEqual(adapter.arm_override_flags, [True])
        self.assertEqual(adapter.gripper_targets, [(0.0, 0.0)])
        self.assertEqual(fake_runtime.action_manager.processed_actions, [adapter.policy_action])
        self.assertEqual(
            runtime._metadata["last_arm_action_report"],  # type: ignore[attr-defined]
            {
                "target_staged": True,
                "arm_joint_names": (
                    "arm_joint1",
                    "arm_joint2",
                    "arm_joint3",
                    "arm_joint4",
                    "arm_joint5",
                    "arm_joint6",
                ),
                "arm_joint_positions": (0.0,) * 6,
                "direct_arm_action_override": True,
                "arm_action_indices": (12, 13, 14, 15, 16, 17),
                "uses_direct_joint_state": False,
                "world_step_owned_by_pipeline": True,
            },
        )
        self.assertEqual(runtime._metadata["joint_action_apply_count"], 1)  # type: ignore[attr-defined]
        self.assertEqual(runtime._metadata["arm_joint_action_apply_count"], 1)  # type: ignore[attr-defined]
        self.assertEqual(runtime._metadata["gripper_joint_action_apply_count"], 1)  # type: ignore[attr-defined]
        self.assertEqual(runtime._metadata["gripper_close_apply_count"], 1)  # type: ignore[attr-defined]
        self.assertEqual(runtime._metadata["gripper_open_apply_count"], 0)  # type: ignore[attr-defined]
        self.assertEqual(
            runtime._metadata["last_joint_action_report"]["source"],  # type: ignore[attr-defined]
            "nav_carry_home",
        )

    def test_refreshes_direct_joint_targets_after_action_manager(self) -> None:
        runtime, adapter, _fake_runtime_obj = _fake_runtime()
        runtime.apply(
            RobotAction(
                arm_joint_positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                gripper_command="close",
                source="arm_pick",
                metadata={
                    "arm_joint_names": (
                        "arm_joint1",
                        "arm_joint2",
                        "arm_joint3",
                        "arm_joint4",
                        "arm_joint5",
                        "arm_joint6",
                    ),
                },
            )
        )

        runtime._apply_staged_joint_position_targets(  # type: ignore[attr-defined]
            timing="after_action_manager"
        )

        self.assertEqual(adapter.arm_position_target_apply_count, 1)
        self.assertEqual(adapter.gripper_position_target_apply_count, 1)
        self.assertEqual(
            runtime._metadata["last_arm_joint_position_target_report"],  # type: ignore[attr-defined]
            {
                "applied": True,
                "joint_names": [
                    "arm_joint1",
                    "arm_joint2",
                    "arm_joint3",
                    "arm_joint4",
                    "arm_joint5",
                    "arm_joint6",
                ],
                "joint_ids": [0, 1, 2, 3, 4, 5],
                "target_positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "control_mode": "position_target_only",
                "velocity_target_written": False,
                "uses_direct_joint_state": False,
                "timing": "after_action_manager",
                "apply_count": 1,
                "world_step_owned_by_pipeline": True,
            },
        )
        self.assertEqual(
            runtime._metadata["last_gripper_joint_position_target_report"][  # type: ignore[attr-defined]
                "timing"
            ],
            "after_action_manager",
        )

    def test_manipulation_base_lock_sets_provenance_and_releases_for_navigation(self) -> None:
        runtime, adapter, _fake_runtime_obj = _fake_runtime()

        runtime.apply(
            RobotAction(
                arm_joint_positions=(0.0,) * 6,
                source="arm_pick",
                metadata={
                    "manipulation_base_lock": True,
                    "manipulation_base_lock_phase": "exec_pick",
                    "manipulation_support_joint_lock": True,
                    "manipulation_support_joint_lock_phase": "exec_pick",
                    "arm_joint_names": (
                        "arm_joint1",
                        "arm_joint2",
                        "arm_joint3",
                        "arm_joint4",
                        "arm_joint5",
                        "arm_joint6",
                    ),
                },
            )
        )
        runtime._apply_active_manipulation_base_lock(  # type: ignore[attr-defined]
            timing="before_physics_step"
        )

        self.assertEqual(adapter.base_pose_lock_flags, [True])
        self.assertEqual(adapter.support_joint_lock_flags, [True])
        self.assertEqual(adapter.base_pose_lock_apply_count, 1)
        self.assertEqual(adapter.support_joint_lock_apply_count, 1)
        self.assertTrue(runtime._metadata["used_base_teleport"])  # type: ignore[attr-defined]
        self.assertFalse(runtime._metadata["used_direct_joint_state"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["used_manipulation_base_lock"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["used_manipulation_support_joint_lock"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["manipulation_base_lock_active"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["manipulation_support_joint_lock_active"])  # type: ignore[attr-defined]
        self.assertEqual(runtime._metadata["manipulation_base_lock_apply_count"], 1)  # type: ignore[attr-defined]
        self.assertEqual(runtime._metadata["manipulation_support_joint_lock_apply_count"], 1)  # type: ignore[attr-defined]

        runtime.apply(
            RobotAction(
                base_velocity=(0.1, 0.0, 0.0),
                source="navigation_dwa",
                metadata={"manipulation_base_lock": False},
            )
        )

        self.assertEqual(adapter.base_pose_lock_flags, [True, False])
        self.assertEqual(adapter.support_joint_lock_flags, [True, False])
        self.assertFalse(runtime._metadata["manipulation_base_lock_active"])  # type: ignore[attr-defined]
        self.assertFalse(runtime._metadata["manipulation_support_joint_lock_active"])  # type: ignore[attr-defined]
        self.assertEqual(
            runtime._metadata["last_manipulation_base_lock_report"]["transition"],  # type: ignore[attr-defined]
            "disabled",
        )
        self.assertEqual(
            runtime._metadata["last_manipulation_support_joint_lock_report"]["transition"],  # type: ignore[attr-defined]
            "disabled",
        )

    def test_apply_rejects_unexpected_arm_joint_order(self) -> None:
        runtime, adapter, fake_runtime = _fake_runtime()

        with self.assertRaisesRegex(RuntimeError, "fixed arm joint order"):
            runtime.apply(
                RobotAction(
                    arm_joint_positions=(0.0,) * 6,
                    source="bad_arm_order",
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

        self.assertEqual(adapter.base_commands, [])
        self.assertEqual(adapter.arm_targets, [])
        self.assertEqual(adapter.arm_override_flags, [])
        self.assertEqual(fake_runtime.action_manager.processed_actions, [])

    def test_apply_zeroes_base_when_environment_terminated_but_keeps_gripper_target(self) -> None:
        runtime, adapter, _fake_runtime_obj = _fake_runtime()
        runtime._environment_terminated = True  # type: ignore[attr-defined]

        runtime.apply(
            RobotAction(
                base_velocity=(0.3, 0.0, 0.1),
                gripper_command="open",
                source="terminated",
            )
        )

        self.assertEqual(adapter.base_commands, [(0.0, 0.0, 0.0)])
        self.assertEqual(adapter.gripper_targets, [(0.04, 0.04)])

    def test_arm_tracking_report_uses_post_step_joint_positions(self) -> None:
        runtime, _adapter, _fake_runtime_obj = _fake_runtime()
        runtime.apply(
            RobotAction(
                arm_joint_positions=(0.0,) * 6,
                source="nav_carry_home",
                metadata={
                    "carry_arm_home_phase": "exec_nav_to_place",
                    "arm_joint_names": (
                        "arm_joint1",
                        "arm_joint2",
                        "arm_joint3",
                        "arm_joint4",
                        "arm_joint5",
                        "arm_joint6",
                    ),
                },
            )
        )

        runtime._consume_pending_arm_tracking_target(  # type: ignore[attr-defined]
            FakeRobotJointState((0.01, -0.02, 0.03, -0.04, 0.05, -0.06))
        )

        report = runtime._metadata["last_arm_tracking_report"]  # type: ignore[attr-defined]
        self.assertTrue(report["available"])
        self.assertEqual(report["pipeline_state"], "exec_nav_to_place")
        self.assertAlmostEqual(report["max_abs_error"], 0.06)
        self.assertEqual(report["peak_joint"]["joint_name"], "arm_joint6")
        aggregate = runtime._metadata["arm_tracking_report"]  # type: ignore[attr-defined]
        self.assertEqual(aggregate["sample_count"], 1)
        self.assertAlmostEqual(aggregate["max_abs_error"], 0.06)

    def test_arm_base_link_falls_back_to_base_fixed_offset(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not available")

        class FakeBodyRobot:
            def __init__(self) -> None:
                self.data = type(
                    "FakeBodyData",
                    (),
                    {
                        "body_pos_w": torch.tensor([[[1.0, 2.0, 0.30]]], dtype=torch.float32),
                        "body_quat_w": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
                    },
                )()

            def find_bodies(self, names, preserve_order=True):
                del preserve_order
                if list(names) == ["base"]:
                    return [0], ["base"]
                raise ValueError("body not found")

        runtime, adapter, _fake_runtime_obj = _fake_runtime()
        adapter.robot = FakeBodyRobot()

        matrix, source = runtime._read_body_matrix("arm_base_link")  # type: ignore[attr-defined]

        # IsaacLab 会合并 fixed arm_base_link；这里必须回退到 Go2-X5 URDF 的 arm_base_joint。
        self.assertEqual(source, "isaaclab_body:base+fixed_arm_base_joint(0.12,0,0.05)")
        self.assertAlmostEqual(float(matrix[0, 3]), 1.12)
        self.assertAlmostEqual(float(matrix[1, 3]), 2.0)
        self.assertAlmostEqual(float(matrix[2, 3]), 0.35)

    def test_runtime_tcp_pose_reuses_planner_export_transform(self) -> None:
        import numpy as np

        runtime, _adapter, _fake_runtime_obj = _fake_runtime()
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = (1.0, 2.0, 3.0)
        runtime._read_tcp_export_matrix = lambda: (  # type: ignore[method-assign]
            matrix,
            "isaaclab_body:arm_link6",
            "fallback_parent_link_plus_fixed_offset",
        )

        tcp_pose = runtime._read_tcp_pose()  # type: ignore[attr-defined]

        self.assertEqual(tcp_pose, (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))


class FakeJointPositionMatrix:
    def __init__(self, values: tuple[float, ...]):
        self.values = values

    def __getitem__(self, key):
        row, indices = key
        if row != 0:
            raise IndexError(row)
        return [self.values[int(index)] for index in indices]


class FakeRobotJointState:
    def __init__(self, values: tuple[float, ...]):
        self.data = type(
            "FakeRobotData",
            (),
            {"joint_pos": FakeJointPositionMatrix(values)},
        )()


def _fake_runtime() -> tuple[IsaacLabNavigationRuntime, FakeAdapter, FakeRuntime]:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    adapter = FakeAdapter()
    fake_runtime = FakeRuntime()
    runtime._env = object()
    runtime._runtime = fake_runtime
    runtime._adapter = adapter
    runtime._environment_terminated = False
    runtime._last_action = RobotAction.idle()
    runtime._action_prepared = False
    runtime._step_calls = 0
    runtime._pending_arm_tracking_target = None
    runtime._manipulation_base_lock_active = False
    runtime._manipulation_support_joint_lock_active = False
    runtime._metadata = {
        "used_base_teleport": False,
        "used_direct_joint_state": False,
        "used_manipulation_base_lock": False,
        "used_manipulation_support_joint_lock": False,
        "manipulation_base_lock_active": False,
        "manipulation_base_lock_apply_count": 0,
        "last_manipulation_base_lock_report": None,
        "manipulation_support_joint_lock_active": False,
        "manipulation_support_joint_lock_apply_count": 0,
        "last_manipulation_support_joint_lock_report": None,
        "arm_joint_position_target_apply_count": 0,
        "last_arm_joint_position_target_report": None,
        "gripper_joint_position_target_apply_count": 0,
        "last_gripper_joint_position_target_report": None,
        "last_arm_action_report": None,
        "last_joint_action_report": None,
        "last_arm_tracking_report": None,
        "arm_tracking_peak_report": None,
        "arm_tracking_report": {
            "sample_count": 0,
            "max_abs_error": 0.0,
            "peak_report": None,
        },
        "arm_tracking_sample_count": 0,
        "arm_tracking_max_abs_error": 0.0,
        "last_gripper_action_report": None,
        "joint_action_apply_count": 0,
        "arm_joint_action_apply_count": 0,
        "gripper_joint_action_apply_count": 0,
        "gripper_close_apply_count": 0,
        "gripper_open_apply_count": 0,
    }
    runtime._config = IsaacLabNavigationRuntimeConfig()
    return runtime, adapter, fake_runtime


if __name__ == "__main__":
    unittest.main()
