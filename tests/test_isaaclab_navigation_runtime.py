"""IsaacLab navigation runtime action 边界测试。"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

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
    _retarget_height_scanners,
    _resolve_rigid_body_prim_path,
)
from source.simulation.collision_patch import (
    gripper_collision_patch_report,
    install_gripper_collision_patch_on_spawn,
    keyword_collision_patch_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeSpawnCfg:
    def __init__(self, timeline: list[str]):
        def _spawn(*args, **kwargs):
            del args, kwargs
            timeline.append("spawn")
            return "spawned"

        self.func = _spawn


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
        self.arm_action_indices_for_report = [12, 13, 14, 15, 16, 17]
        self.arm_action_term_available = True
        self.base_commands: list[tuple[float, float, float]] = []
        self.arm_targets: list[tuple[float, ...]] = []
        self.arm_velocity_hold_flags: list[bool] = []
        self.arm_override_flags: list[bool] = []
        self.gripper_targets: list[tuple[float, ...]] = []
        self.base_pose_lock_flags: list[bool] = []
        self.base_pose_lock_targets: list[tuple[float, float, float, float] | None] = []
        self.support_joint_lock_flags: list[bool] = []
        self.navigation_joint_pose_lock_flags: list[bool] = []
        self.navigation_joint_pose_lock_arm_targets: list[tuple[float, ...] | None] = []
        self.base_pose_lock_apply_count = 0
        self.support_joint_lock_apply_count = 0
        self.navigation_joint_pose_lock_apply_count = 0
        self.arm_position_target_apply_count = 0
        self.gripper_position_target_apply_count = 0
        self.refresh_flags: list[bool] = []
        self.policy_action = FakePolicyAction()

    def apply_base_command(self, vx: float, vy: float, wz: float) -> None:
        self.base_commands.append((float(vx), float(vy), float(wz)))

    def set_gripper_joint_target(self, target) -> None:
        self.gripper_targets.append(tuple(float(value) for value in target))

    def set_arm_joint_target(self, target, *, hold_velocity: bool = False) -> None:
        self.arm_targets.append(tuple(float(value) for value in target))
        self.arm_velocity_hold_flags.append(bool(hold_velocity))

    def set_direct_arm_action_override(self, enabled: bool = True) -> dict:
        self.arm_override_flags.append(bool(enabled))
        return {
            "enabled": bool(enabled),
            "action_term_available": self.arm_action_term_available,
            "arm_action_indices": list(self.arm_action_indices_for_report),
            "arm_joint_names": [
                "arm_joint1",
                "arm_joint2",
                "arm_joint3",
                "arm_joint4",
                "arm_joint5",
                "arm_joint6",
            ],
        }

    def set_base_pose_lock(self, enabled: bool = True, pose_xyyaw=None, pose_xyzyaw=None) -> dict:
        del pose_xyyaw
        self.base_pose_lock_flags.append(bool(enabled))
        target = (
            None
            if pose_xyzyaw is None
            else tuple(float(value) for value in pose_xyzyaw)
        )
        self.base_pose_lock_targets.append(target)
        pose_xyzyaw_report = list(target) if target is not None else [1.0, 2.0, 0.35, 0.5]
        return {
            "enabled": bool(enabled),
            "pose_xyzyaw": pose_xyzyaw_report if enabled else None,
            "pose_xyyaw": [pose_xyzyaw_report[0], pose_xyzyaw_report[1], pose_xyzyaw_report[3]] if enabled else None,
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

    def set_navigation_joint_pose_lock(self, enabled: bool = True, *, arm_joint_target=None) -> dict:
        self.navigation_joint_pose_lock_flags.append(bool(enabled))
        target = (
            None
            if arm_joint_target is None
            else tuple(float(value) for value in arm_joint_target)
        )
        self.navigation_joint_pose_lock_arm_targets.append(target)
        return {
            "enabled": bool(enabled),
            "joint_names": [f"joint{index}" for index in range(20)] if enabled else [],
            "joint_ids": list(range(20)) if enabled else [],
            "target_positions": [0.0] * 20 if enabled else [],
            "uses_direct_joint_state": bool(enabled),
            "lock_mode": "stair_float_full_body_pose" if enabled else None,
        }

    def apply_navigation_joint_pose_lock(self) -> dict:
        self.navigation_joint_pose_lock_apply_count += 1
        return {
            "applied": True,
            "joint_names": [f"joint{index}" for index in range(20)],
            "joint_ids": list(range(20)),
            "target_positions": [0.0] * 20,
            "uses_direct_joint_state": True,
            "lock_mode": "stair_float_full_body_pose",
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
    def test_gripper_collision_patch_defaults_match_requested_values(self) -> None:
        config = IsaacLabNavigationRuntimeConfig()

        self.assertTrue(config.patch_gripper_collision)
        self.assertEqual(config.gripper_collision_robot_root, "/World/go2_x5")
        self.assertEqual(
            config.gripper_collision_links,
            ("arm_link7", "arm_link8"),
        )
        self.assertEqual(
            config.gripper_collision_approximation,
            "convexDecomposition",
        )
        self.assertEqual(config.gripper_collision_contact_offset, 0.002)
        self.assertEqual(config.gripper_collision_rest_offset, 0.0)
        self.assertTrue(config.patch_apple_collision)
        self.assertEqual(config.apple_collision_root_path, "/World")
        self.assertEqual(config.apple_collision_keywords, ("apple", "Apple"))
        self.assertEqual(config.apple_collision_approximation, "convexDecomposition")
        self.assertEqual(config.apple_collision_contact_offset, 0.001)
        self.assertEqual(config.apple_collision_rest_offset, 0.0)
        self.assertTrue(config.hide_object_collision_visual)
        self.assertEqual(config.standing_command_threshold, 0.0)
        self.assertEqual(config.policy_action_warmup_steps, 0)
        self.assertEqual(config.object_collision_visual_root_path, "/World")
        self.assertEqual(
            config.object_collision_visual_hide_keywords,
            ("Apple_M_Apple",),
        )
        self.assertEqual(
            config.object_collision_visual_keep_keywords,
            ("visual_video",),
        )

    def test_gripper_collision_spawn_patch_runs_after_spawn_once(self) -> None:
        timeline: list[str] = []
        spawn_cfg = FakeSpawnCfg(timeline)

        def _patch_collision(**kwargs):
            self.assertEqual(kwargs["stage"], "stage")
            timeline.append("patch")
            return {"applied": True, "patch_count": 2}

        def _print_info(**kwargs):
            self.assertEqual(kwargs["stage"], "stage")
            timeline.append("print")
            return []

        def _patch_keyword_collision(**kwargs):
            self.assertEqual(kwargs["stage"], "stage")
            timeline.append("keyword_patch")
            return {"applied": True, "patch_count": 1}

        def _print_keyword_info(**kwargs):
            self.assertEqual(kwargs["stage"], "stage")
            timeline.append("keyword_print")
            return []

        with (
            patch(
                "source.simulation.collision_patch.patch_go2_x5_gripper_collision",
                side_effect=_patch_collision,
            ),
            patch(
                "source.simulation.collision_patch.patch_collision_prims_by_keywords",
                side_effect=_patch_keyword_collision,
            ),
            patch(
                "source.simulation.collision_patch.print_gripper_collision_info",
                side_effect=_print_info,
            ),
            patch(
                "source.simulation.collision_patch.print_collision_info_by_keywords",
                side_effect=_print_keyword_info,
            ),
        ):
            install_gripper_collision_patch_on_spawn(
                spawn_cfg,
                stage_getter=lambda: "stage",
            )
            wrapped_func = spawn_cfg.func
            install_gripper_collision_patch_on_spawn(
                spawn_cfg,
                stage_getter=lambda: "stage",
            )
            self.assertIs(spawn_cfg.func, wrapped_func)
            result = spawn_cfg.func("/World/envs/env_0/Robot", spawn_cfg)

        self.assertEqual(result, "spawned")
        self.assertEqual(
            timeline,
            ["spawn", "patch", "keyword_patch", "print", "keyword_print"],
        )
        self.assertEqual(
            gripper_collision_patch_report(spawn_cfg),
            {"applied": True, "patch_count": 2},
        )
        self.assertEqual(
            keyword_collision_patch_report(spawn_cfg),
            {"applied": True, "patch_count": 1},
        )

    def test_gripper_collision_patch_covers_required_usd_fields(self) -> None:
        source_text = (
            PROJECT_ROOT / "source/simulation/collision_patch.py"
        ).read_text(encoding="utf-8")

        for required_text in (
            "UsdPhysics.CollisionAPI",
            "UsdPhysics.MeshCollisionAPI",
            "PhysxSchema.PhysxCollisionAPI",
            "patch_collision_prims_by_keywords",
            "print_collision_info_by_keywords",
            '"physics:approximation"',
            '"physics:collisionEnabled"',
            '"physxCollision:contactOffset"',
            '"physxCollision:restOffset"',
            '"physics:restOffset"',
            "Sdf.ValueTypeNames.Token",
            "Sdf.ValueTypeNames.Float",
            "Sdf.ValueTypeNames.Bool",
            "apple",
            "PhysxCollisionAPI",
            "prim.SetInstanceable(False)",
        ):
            self.assertIn(required_text, source_text)

    def test_gripper_collision_patch_is_installed_before_wrapper_reset(self) -> None:
        configure_source = inspect.getsource(IsaacLabNavigationRuntime._configure_env)
        build_source = inspect.getsource(IsaacLabNavigationRuntime._build_environment)

        self.assertIn("install_gripper_collision_patch_on_spawn", configure_source)
        self.assertIn("patch_apple_collision", configure_source)
        self.assertIn("apple_collision_patch_report", build_source)
        self.assertLess(
            build_source.index("env = gym.make("),
            build_source.index("wrapped = RslRlVecEnvWrapper("),
        )
        self.assertIn("gripper_collision_patch_report", build_source)

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

        self.assertEqual(config.viewport_camera_prim_path, "/World/Camera0")
        self.assertTrue(config.hide_navigation_collision_visual)

    def test_height_scanners_follow_runtime_navigation_terrain(self) -> None:
        scanner = type("ScannerCfg", (), {"mesh_prim_paths": ["/World/ground"]})()
        scanner_base = type("ScannerCfg", (), {"mesh_prim_paths": ["/World/ground"]})()
        scene_cfg = type(
            "SceneCfg",
            (),
            {
                "height_scanner": scanner,
                "height_scanner_base": scanner_base,
            },
        )()

        updated = _retarget_height_scanners(scene_cfg, "/World/nav_collision")

        self.assertEqual(updated, ("height_scanner", "height_scanner_base"))
        self.assertEqual(scanner.mesh_prim_paths, ["/World/nav_collision"])
        self.assertEqual(scanner_base.mesh_prim_paths, ["/World/nav_collision"])

    def test_height_scanner_retarget_skips_disabled_sensor(self) -> None:
        scene_cfg = type(
            "SceneCfg",
            (),
            {
                "height_scanner": None,
                "height_scanner_base": None,
            },
        )()

        updated = _retarget_height_scanners(scene_cfg, "/World/nav_collision")

        self.assertEqual(updated, ())

    def test_object_reader_resolves_inner_rigid_body_prim(self) -> None:
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("当前 Python 环境没有 OpenUSD pxr")

        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        stage.DefinePrim("/World/apple", "Xform")
        body = stage.DefinePrim("/World/apple/body", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body)

        resolved = _resolve_rigid_body_prim_path(stage, "/World/apple")

        self.assertEqual(resolved, "/World/apple/body")
        self.assertFalse(
            stage.GetPrimAtPath("/World/apple").HasAPI(UsdPhysics.RigidBodyAPI)
        )

    def test_object_collision_visual_hide_runs_after_task_object_visibility(self) -> None:
        source_text = inspect.getsource(IsaacLabNavigationRuntime._load_visual_scene)

        self.assertLess(
            source_text.index("_show_only_task_object"),
            source_text.index("_hide_object_collision_visual"),
        )
        self.assertIn("object_collision_visual_hide_report", source_text)
        builder_source = inspect.getsource(IsaacLabNavigationRuntime._build_environment)
        self.assertLess(
            builder_source.index("object_visibility_after_spawn_report"),
            builder_source.index("object_collision_visual_hide_after_spawn_report"),
        )
        self.assertLess(
            builder_source.index("object_collision_visual_hide_after_spawn_report"),
            builder_source.index("wrapped = RslRlVecEnvWrapper"),
        )

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

    def test_wrist_camera_uses_arm_link6_mount(self) -> None:
        source_text = (
            PROJECT_ROOT / "source/simulation/isaaclab_runtime.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'prim_path="{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera"',
            source_text,
        )
        self.assertIn("focal_length=18.0", source_text)

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
        adapter.standing_command_threshold = 0.08
        adapter.policy_action_warmup_steps = 2
        adapter._policy_action_step = 0
        adapter._policy_action_warmup_scale = 1.0
        adapter._command = (0.04, 0.0, 0.02)
        adapter._effective_command = (0.0, 0.0, 0.0)
        adapter._command_is_standing = True
        adapter._arm_joint_target = (1.0, -1.0, 0.5, 0.0, 0.2, -0.3)
        adapter._gripper_joint_target = None
        adapter._last_actions = None

        actions = adapter.compute_policy_action(refresh_observations=True)

        # locomotion policy 仍受 clip_actions 约束；机械臂直接目标必须绕过该裁剪。
        self.assertAlmostEqual(float(actions[0, 0]), 0.5)
        self.assertAlmostEqual(adapter._policy_action_warmup_scale, 0.5)
        self.assertEqual(adapter._policy_action_step, 1)
        self.assertEqual(adapter._effective_command, (0.0, 0.0, 0.0))
        self.assertTrue(adapter._command_is_standing)
        self.assertEqual(
            tuple(float(value) for value in adapter.base_cmd_term.vel_command_b[0]),
            (0.0, 0.0, 0.0),
        )
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
                "arm_control_mode": "policy_action_override",
                "direct_arm_action_override": True,
                "arm_action_indices": (12, 13, 14, 15, 16, 17),
                "arm_velocity_hold": False,
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

    def test_apply_accepts_dog_only_arm_target_without_policy_action_slots(self) -> None:
        runtime, adapter, fake_runtime = _fake_runtime()
        adapter.arm_action_indices_for_report = []
        action = RobotAction(
            base_velocity=(0.1, 0.0, 0.2),
            arm_joint_positions=(0.2, -0.1, 0.3, -0.2, 0.1, 0.0),
            gripper_command="hold",
            source="arm_pick",
            metadata={
                "arm_joint_names": tuple(ARM_JOINT_NAMES),
            },
        )

        runtime.apply(action)

        self.assertEqual(adapter.arm_targets, [(0.2, -0.1, 0.3, -0.2, 0.1, 0.0)])
        self.assertEqual(adapter.arm_override_flags, [True, False])
        self.assertEqual(fake_runtime.action_manager.processed_actions, [adapter.policy_action])
        self.assertEqual(
            runtime._metadata["last_arm_action_report"],  # type: ignore[attr-defined]
            {
                "target_staged": True,
                "arm_joint_names": tuple(ARM_JOINT_NAMES),
                "arm_joint_positions": (0.2, -0.1, 0.3, -0.2, 0.1, 0.0),
                "arm_control_mode": "independent_position_target",
                "direct_arm_action_override": False,
                "arm_action_indices": (),
                "arm_velocity_hold": False,
                "uses_direct_joint_state": False,
                "world_step_owned_by_pipeline": True,
            },
        )
        self.assertEqual(
            runtime._metadata["last_direct_arm_action_override_disable_report"],  # type: ignore[attr-defined]
            {
                "enabled": False,
                "action_term_available": True,
                "arm_action_indices": [],
                "arm_joint_names": list(ARM_JOINT_NAMES),
            },
        )
        self.assertEqual(runtime._metadata["arm_joint_action_apply_count"], 1)  # type: ignore[attr-defined]

    def test_post_motion_hold_arm_target_requests_velocity_hold(self) -> None:
        runtime, adapter, _fake_runtime_obj = _fake_runtime()
        adapter.arm_action_indices_for_report = []
        action = RobotAction(
            arm_joint_positions=(0.2, -0.1, 0.3, -0.2, 0.1, 0.0),
            gripper_command="hold",
            source="arm_place",
            metadata={
                "arm_joint_names": tuple(ARM_JOINT_NAMES),
                "segment_type": "post_motion_hold",
                "segment_name": "move_to_pre_place",
            },
        )

        runtime.apply(action)

        self.assertEqual(adapter.arm_velocity_hold_flags, [True])
        self.assertTrue(
            runtime._metadata["last_arm_action_report"]["arm_velocity_hold"]  # type: ignore[attr-defined]
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

    def test_navigation_stair_float_updates_root_lock_target(self) -> None:
        runtime, adapter, _fake_runtime_obj = _fake_runtime()
        arm_target = (0.1, 0.2, 0.3, -0.1, -0.2, -0.3)

        runtime.apply(
            RobotAction(
                base_velocity=(0.0, 0.0, 0.0),
                source="navigation_stair_float",
                arm_joint_positions=arm_target,
                metadata={
                    "navigation_base_pose_lock": True,
                    "navigation_base_pose_lock_phase": "pct_stair_float",
                    "navigation_base_pose_lock_xyzyaw": (1.2, 6.3, 1.4, 1.57),
                    "navigation_support_joint_lock": True,
                    "navigation_support_joint_lock_phase": "pct_stair_float",
                    "navigation_full_body_joint_lock": True,
                    "navigation_full_body_joint_lock_phase": "pct_stair_float",
                },
            )
        )

        self.assertEqual(adapter.base_pose_lock_flags, [True])
        self.assertEqual(adapter.base_pose_lock_targets, [(1.2, 6.3, 1.4, 1.57)])
        self.assertEqual(adapter.support_joint_lock_flags, [True])
        self.assertEqual(adapter.navigation_joint_pose_lock_flags, [True])
        self.assertEqual(adapter.navigation_joint_pose_lock_arm_targets, [arm_target])
        self.assertTrue(runtime._metadata["used_navigation_base_lock"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["used_navigation_support_joint_lock"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["used_navigation_joint_pose_lock"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["used_direct_joint_state"])  # type: ignore[attr-defined]
        self.assertEqual(
            runtime._metadata["last_navigation_base_lock_report"]["source"],  # type: ignore[attr-defined]
            "navigation",
        )
        self.assertEqual(
            runtime._metadata["last_navigation_base_lock_report"]["transition"],  # type: ignore[attr-defined]
            "enabled",
        )

        runtime.apply(
            RobotAction(
                base_velocity=(0.1, 0.0, 0.0),
                source="navigation_dwa",
                metadata={},
            )
        )

        self.assertEqual(adapter.base_pose_lock_flags, [True, False])
        self.assertEqual(adapter.support_joint_lock_flags, [True, False])
        self.assertEqual(adapter.navigation_joint_pose_lock_flags, [True, False])
        self.assertFalse(runtime._metadata["manipulation_base_lock_active"])  # type: ignore[attr-defined]
        self.assertFalse(runtime._metadata["manipulation_support_joint_lock_active"])  # type: ignore[attr-defined]
        self.assertFalse(runtime._metadata["navigation_joint_pose_lock_active"])  # type: ignore[attr-defined]

    def test_navigation_stair_float_moves_carried_object_with_root_target(self) -> None:
        runtime, adapter, _fake_runtime_obj = _fake_runtime()

        class FakeRigidView:
            def __init__(self, owner):
                self.owner = owner
                self.velocities = []

            def set_world_poses(self, *, positions, orientations) -> None:
                self.owner.position = np.asarray(
                    positions.detach().cpu().tolist()[0],
                    dtype=np.float64,
                )
                self.owner.orientation = np.asarray(
                    orientations.detach().cpu().tolist()[0],
                    dtype=np.float64,
                )

            def set_velocities(self, velocities) -> None:
                self.velocities.append(velocities.detach().cpu().tolist())

        class FakeObject:
            def __init__(self):
                self.position = np.asarray((0.3, 0.0, 0.5), dtype=np.float64)
                self.orientation = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
                self._rigid_prim_view = FakeRigidView(self)

            def get_world_pose(self):
                return self.position.copy(), self.orientation.copy()

        adapter.robot = type(
            "FakeRobot",
            (),
            {
                "data": type(
                    "FakeRobotData",
                    (),
                    {
                        "root_pos_w": np.asarray(((0.0, 0.0, 0.2),)),
                        "root_quat_w": np.asarray(((1.0, 0.0, 0.0, 0.0),)),
                    },
                )()
            },
        )()
        fake_object = FakeObject()
        runtime._object = fake_object  # type: ignore[attr-defined]
        runtime._read_tcp_pose = lambda: (  # type: ignore[method-assign]
            0.28,
            0.0,
            0.5,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        sleep_calls = []

        def _set_sleeping(*, enabled: bool) -> dict:
            sleep_calls.append(bool(enabled))
            return {"applied": True, "enabled": bool(enabled)}

        runtime._set_object_sleeping = _set_sleeping  # type: ignore[method-assign]
        runtime.apply(
            RobotAction(
                source="navigation_stair_float",
                metadata={
                    "navigation_base_pose_lock": True,
                    "navigation_base_pose_lock_phase": "pct_stair_float",
                    "navigation_base_pose_lock_xyzyaw": (
                        1.0,
                        2.0,
                        0.2,
                        np.pi / 2.0,
                    ),
                    "navigation_support_joint_lock": True,
                    "navigation_carry_object_follow": True,
                },
            )
        )
        runtime._apply_active_manipulation_base_lock(timing="unit_test")  # type: ignore[attr-defined]

        np.testing.assert_allclose(
            fake_object.position,
            np.asarray((1.0, 2.3, 0.5)),
            atol=1.0e-6,
        )
        self.assertTrue(runtime._metadata["used_kinematic_object_follow"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["used_object_teleport"])  # type: ignore[attr-defined]
        self.assertTrue(runtime._metadata["navigation_object_follow_active"])  # type: ignore[attr-defined]

        runtime.apply(RobotAction(source="navigation_dwa"))

        self.assertEqual(sleep_calls, [True, False])
        self.assertFalse(runtime._metadata["navigation_object_follow_active"])  # type: ignore[attr-defined]
        self.assertEqual(
            runtime._metadata["last_navigation_object_follow_report"]["transition"],  # type: ignore[attr-defined]
            "disabled",
        )

    def test_navigation_base_lock_requires_xyzyaw_target(self) -> None:
        runtime, _adapter, _fake_runtime_obj = _fake_runtime()

        with self.assertRaisesRegex(
            RuntimeError,
            "navigation_base_pose_lock_xyzyaw",
        ):
            runtime.apply(
                RobotAction(
                    source="navigation_stair_float",
                    metadata={"navigation_base_pose_lock": True},
                )
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
    runtime._navigation_joint_pose_lock_active = False
    runtime._navigation_object_follow_active = False
    runtime._navigation_object_relative_pose = None
    runtime._navigation_object_follow_root_target = None
    runtime._navigation_object_follow_target_pose = None
    runtime._object = None
    runtime._metadata = {
        "used_base_teleport": False,
        "used_direct_joint_state": False,
        "used_object_teleport": False,
        "used_kinematic_object_follow": False,
        "used_manipulation_base_lock": False,
        "used_manipulation_support_joint_lock": False,
        "used_navigation_base_lock": False,
        "used_navigation_support_joint_lock": False,
        "used_navigation_joint_pose_lock": False,
        "navigation_object_follow_active": False,
        "navigation_object_follow_apply_count": 0,
        "last_navigation_object_follow_report": None,
        "manipulation_base_lock_active": False,
        "manipulation_base_lock_apply_count": 0,
        "last_manipulation_base_lock_report": None,
        "last_navigation_base_lock_report": None,
        "manipulation_support_joint_lock_active": False,
        "manipulation_support_joint_lock_apply_count": 0,
        "last_manipulation_support_joint_lock_report": None,
        "last_navigation_support_joint_lock_report": None,
        "navigation_joint_pose_lock_active": False,
        "navigation_joint_pose_lock_apply_count": 0,
        "last_navigation_joint_pose_lock_report": None,
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
