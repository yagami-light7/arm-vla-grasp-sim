"""纯 CPU 的完整物理导航规划与逐 tick 执行测试。"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.diagnostics import NavigationEpisodeVerifier
from source.interfaces.navigation import NavGoal, NavPlan
from source.interfaces.simulation import SimulationState
from source.navigation.executor import DwaNavExecutor
from source.navigation.navlib import DWAConfig, OccupancyGridMap
from source.navigation.planner_adapter import AStarNavPlanner
from source.pipeline import NavigationSettings
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _state(
    *,
    x: float,
    y: float,
    yaw: float,
    velocity: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    step_index: int = 0,
) -> SimulationState:
    """构造不依赖 Isaac 的最小仿真观测。"""

    half_yaw = 0.5 * yaw
    return SimulationState(
        step_index=step_index,
        timestamp=step_index * 0.05,
        robot_root_pose=(
            x,
            y,
            0.35,
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        ),
        robot_root_velocity=velocity,
    )


class FullPhysicsNavigationTest(unittest.TestCase):
    def test_default_clearance_plans_contact_task_pick_to_place(self) -> None:
        settings = NavigationSettings()
        task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
        spec = JsonTaskProvider().load(task_path)
        self.assertIsNotNone(spec.place_goal)
        occupancy = np.zeros((80, 80), dtype=bool)
        # 在 pick/place 之间放一段带端点的墙，确保测试覆盖 clearance 绕行，
        # 同时不再依赖旧 839920 worktree 中的 ignored nav map。
        occupancy[40, 20:51] = True
        planner = AStarNavPlanner(
            grid_map=OccupancyGridMap(
                occupancy,
                0.1,
                (-3.0, -1.0, 0.0),
            ),
            inflate_radius=settings.global_inflate_radius,
        )
        start = _state(
            x=spec.pick_goal.x,
            y=spec.pick_goal.y,
            yaw=spec.pick_goal.yaw,
        )

        plan = planner.plan(start, spec.place_goal)

        self.assertEqual(settings.global_inflate_radius, 0.20)
        self.assertEqual(settings.local_clearance_radius, 0.20)
        self.assertEqual(plan.waypoints[0], (spec.pick_goal.x, spec.pick_goal.y))
        self.assertEqual(plan.waypoints[-1], (spec.place_goal.x, spec.place_goal.y))
        self.assertGreater(len(plan.waypoints), 2)

    def test_astar_planner_routes_through_gap_and_preserves_exact_endpoints(self) -> None:
        occupancy = np.zeros((40, 40), dtype=bool)
        occupancy[:, 20] = True
        occupancy[8:13, 20] = False
        grid = OccupancyGridMap(occupancy, 0.1, (0.0, 0.0, 0.0))
        planner = AStarNavPlanner(grid_map=grid)
        start = _state(x=0.55, y=1.95, yaw=0.0)
        goal = NavGoal(x=3.45, y=1.95, yaw=0.0)

        plan = planner.plan(start, goal)

        self.assertEqual(plan.waypoints[0], (0.55, 1.95))
        self.assertEqual(plan.waypoints[-1], (3.45, 1.95))
        self.assertGreater(len(plan.waypoints), 2)
        self.assertTrue(any(point[1] > 2.5 for point in plan.waypoints))
        self.assertEqual(plan.metadata["planner"], "astar")
        self.assertGreater(plan.metadata["expanded_nodes"], 0)

    def test_executor_outputs_body_action_and_converts_world_velocity(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        planner = AStarNavPlanner(grid_map=grid)
        start = _state(
            x=0.0,
            y=0.0,
            yaw=math.pi / 2.0,
            velocity=(0.0, 0.30, 0.0, 0.0, 0.0, 0.10),
        )
        plan = planner.plan(
            start,
            NavGoal(x=0.0, y=1.0, yaw=math.pi / 2.0),
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(
                control_dt=0.05,
                goal_tolerance=0.12,
                max_linear_accel=10.0,
            ),
            terminal_start_distance=0.30,
        )
        executor.reset(plan)

        action = executor.compute_action(start)

        self.assertEqual(action.source, "navigation_dwa")
        self.assertIsNone(action.arm_joint_positions)
        self.assertIsNone(action.gripper_command)
        self.assertGreater(action.base_velocity[0], 0.0)
        self.assertEqual(action.base_velocity[1], 0.0)
        body_velocity = action.metadata["measured_body_velocity"]
        self.assertAlmostEqual(body_velocity[0], 0.30, places=6)
        self.assertAlmostEqual(body_velocity[1], 0.0, places=6)
        self.assertAlmostEqual(body_velocity[2], 0.10, places=6)

    def test_executor_reuses_dwa_command_between_recompute_steps(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        state = _state(x=0.0, y=0.0, yaw=0.0)
        plan = AStarNavPlanner(grid_map=grid).plan(
            state,
            NavGoal(x=1.0, y=0.0, yaw=0.0),
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.02, max_linear_accel=10.0),
            command_recompute_interval_steps=3,
            stall_window_steps=20,
        )
        executor.reset(plan)

        actions = [executor.compute_action(state) for _ in range(4)]
        status = executor.status()["dwa_compute"]

        self.assertEqual(actions[1].base_velocity, actions[0].base_velocity)
        self.assertEqual(actions[2].base_velocity, actions[0].base_velocity)
        self.assertEqual(status["recompute_interval_steps"], 3)
        self.assertEqual(status["compute_count"], 2)
        self.assertEqual(status["held_command_count"], 2)
        self.assertGreaterEqual(status["last_duration_s"], 0.0)

    def test_terminal_pose_controller_and_completion_tolerances(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        planner = AStarNavPlanner(grid_map=grid)
        plan = planner.plan(
            _state(x=0.0, y=0.0, yaw=0.0),
            NavGoal(x=0.10, y=0.0, yaw=math.pi / 2.0),
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            terminal_start_distance=0.50,
            position_tolerance=0.05,
            yaw_tolerance=0.10,
            stall_window_steps=4,
        )
        executor.reset(plan)

        action = executor.compute_action(_state(x=0.02, y=0.0, yaw=0.0))
        self.assertEqual(action.source, "navigation_terminal_pose")
        self.assertGreater(action.base_velocity[2], 0.0)
        self.assertFalse(executor.is_done(_state(x=0.02, y=0.0, yaw=0.0)))
        for _ in range(4):
            executor.compute_action(_state(x=0.02, y=0.0, yaw=0.0))
        self.assertFalse(executor.status()["failed"])

        final_state = _state(x=0.08, y=0.0, yaw=math.pi / 2.0 - 0.05)
        self.assertTrue(executor.is_done(final_state))
        stopped = executor.compute_action(final_state)
        self.assertEqual(stopped.base_velocity, (0.0, 0.0, 0.0))
        self.assertTrue(executor.status()["success"])
        self.assertEqual(executor.status()["phase"], "completed")

    def test_carry_terminal_latches_forward_translation_after_final_yaw(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        state = _state(x=0.0, y=0.0, yaw=0.0)
        plan = NavPlan(
            goal=NavGoal(x=0.031, y=0.099, yaw=0.148),
            waypoints=((0.0, 0.0), (0.031, 0.099)),
            metadata={
                "execution_phase": "carry_nav_to_place",
                "require_yaw_alignment": True,
                "yaw_tolerance": 0.15,
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.02, goal_tolerance=0.05),
            terminal_start_distance=0.18,
            position_tolerance=0.05,
            yaw_tolerance=0.15,
            carry_max_linear_velocity=0.25,
            carry_max_angular_velocity=0.30,
        )
        executor.reset(plan)

        action = executor.compute_action(state)

        self.assertEqual(action.source, "navigation_terminal_pose")
        self.assertEqual(action.base_velocity, (0.0, 0.0, 0.30))
        self.assertEqual(
            action.metadata["terminal_control_mode"],
            "carry_forward_translation",
        )
        self.assertAlmostEqual(
            action.metadata["terminal_translation_heading_error"],
            math.atan2(0.099, 0.031),
        )
        self.assertTrue(action.metadata["carry_forward_translation_active"])
        self.assertEqual(
            action.metadata["carry_forward_translation_activation_reason"],
            "final_yaw_aligned_with_xy_residual",
        )

        # Turning toward the XY residual intentionally moves away from final yaw
        # and can briefly increase distance.  Stay in terminal control instead
        # of falling back to DWA and its occupied-cell rejection loop.
        action = executor.compute_action(_state(x=0.0, y=-0.10, yaw=0.40))
        self.assertEqual(action.source, "navigation_terminal_pose")
        self.assertEqual(
            action.metadata["terminal_control_mode"],
            "carry_forward_translation",
        )
        self.assertTrue(action.metadata["carry_forward_translation_active"])

    def test_carry_terminal_keeps_proven_holonomic_alignment_before_final_yaw(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=0.088, y=0.148, yaw=1.25),
            waypoints=((0.0, 0.0), (0.088, 0.148)),
            metadata={
                "execution_phase": "carry_nav_to_place",
                "require_yaw_alignment": True,
                "yaw_tolerance": 0.15,
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.02, goal_tolerance=0.05),
            terminal_start_distance=0.18,
            position_tolerance=0.05,
            yaw_tolerance=0.15,
            carry_max_linear_velocity=0.25,
            carry_max_angular_velocity=0.30,
        )
        executor.reset(plan)

        action = executor.compute_action(_state(x=0.0, y=0.0, yaw=0.0))

        self.assertEqual(action.source, "navigation_terminal_pose")
        self.assertEqual(action.metadata["terminal_control_mode"], "final_pose")
        self.assertFalse(action.metadata["carry_forward_translation_active"])
        self.assertGreater(action.base_velocity[0], 0.0)
        self.assertGreater(action.base_velocity[1], 0.0)
        self.assertGreater(action.base_velocity[2], 0.0)
        self.assertLessEqual(action.base_velocity[2], 0.30)

    def test_carry_terminal_releases_forward_translation_at_place_tolerance(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=0.16, y=0.0, yaw=0.70),
            waypoints=((0.0, 0.0), (0.16, 0.0)),
            metadata={
                "execution_phase": "carry_nav_to_place",
                "require_yaw_alignment": True,
                "yaw_tolerance": 0.15,
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            terminal_start_distance=0.18,
            position_tolerance=0.10,
            carry_position_tolerance=0.15,
            yaw_tolerance=0.15,
            carry_max_linear_velocity=0.25,
            carry_max_angular_velocity=0.30,
        )
        executor.reset(plan)

        translating = executor.compute_action(_state(x=0.0, y=0.0, yaw=0.70))
        self.assertTrue(translating.metadata["carry_forward_translation_active"])
        self.assertEqual(
            translating.metadata["terminal_control_mode"],
            "carry_forward_translation",
        )

        # seed 5000 reached about 0.116 m from the goal while this latch was
        # active.  That is already inside the 0.15 m place tolerance, so
        # final-yaw control must take over instead of orbiting toward 0.08 m.
        polishing = executor.compute_action(_state(x=0.04, y=0.0, yaw=3.0))

        self.assertFalse(polishing.metadata["carry_forward_translation_active"])
        self.assertEqual(polishing.metadata["terminal_control_mode"], "final_pose")
        self.assertAlmostEqual(
            executor._active_terminal_pose_config.position_acceptance_tolerance,
            0.15,
        )
        self.assertLess(polishing.base_velocity[2], 0.0)

    def test_carry_place_tolerance_holds_zero_until_base_is_stable(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=0.10, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (0.10, 0.0)),
            metadata={
                "execution_phase": "carry_nav_to_place",
                "require_yaw_alignment": True,
                "yaw_tolerance": 0.15,
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            terminal_start_distance=0.18,
            position_tolerance=0.05,
            carry_position_tolerance=0.12,
            completion_linear_velocity_tolerance=0.06,
            completion_angular_velocity_tolerance=0.20,
        )
        executor.reset(plan)

        moving = _state(
            x=0.0,
            y=0.0,
            yaw=0.0,
            velocity=(0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        settling = executor.compute_action(moving)

        self.assertEqual(settling.source, "navigation_settling")
        self.assertEqual(settling.base_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(executor.status()["phase"], "settling")
        self.assertEqual(executor.status()["position_tolerance"], 0.12)

        stopped = executor.compute_action(_state(x=0.0, y=0.0, yaw=0.0))
        self.assertEqual(stopped.base_velocity, (0.0, 0.0, 0.0))
        self.assertTrue(executor.status()["success"])

    def test_settling_resumes_control_after_base_drifts_outside_tolerance(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=0.10, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (0.10, 0.0)),
            metadata={"require_yaw_alignment": True},
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            terminal_start_distance=0.18,
            position_tolerance=0.10,
            completion_linear_velocity_tolerance=0.06,
            completion_angular_velocity_tolerance=0.20,
        )
        executor.reset(plan)

        moving_inside = _state(
            x=0.02,
            y=0.0,
            yaw=0.0,
            velocity=(0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        settling = executor.compute_action(moving_inside)
        self.assertEqual(settling.source, "navigation_settling")

        drifted = _state(x=-0.06, y=0.0, yaw=0.0)
        recovery = executor.compute_action(drifted)

        self.assertEqual(recovery.source, "navigation_terminal_pose")
        self.assertGreater(recovery.base_velocity[0], 0.0)
        self.assertFalse(executor.status()["success"])

    def test_navigation_verifier_uses_place_only_position_tolerance(self) -> None:
        spec = JsonTaskProvider().load(
            PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json"
        )
        self.assertIsNotNone(spec.place_goal)
        verifier = NavigationEpisodeVerifier(
            position_tolerance=0.05,
            place_position_tolerance=0.15,
            yaw_tolerance=0.15,
            require_yaw_alignment=True,
        )
        state = _state(
            x=spec.place_goal.x + 0.14,
            y=spec.place_goal.y,
            yaw=spec.place_goal.yaw,
        )

        place_result = verifier.verify_place_reachable(state, spec)
        pick_result = verifier.verify_pick_reachable(state, spec)

        self.assertTrue(place_result.success)
        self.assertEqual(place_result.metadata["position_tolerance"], 0.15)
        self.assertFalse(pick_result.success)
        self.assertEqual(pick_result.metadata["position_tolerance"], 0.05)

    def test_executor_can_accept_xy_without_yaw_alignment(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        planner = AStarNavPlanner(grid_map=grid)
        plan = planner.plan(
            _state(x=0.0, y=0.0, yaw=0.0),
            NavGoal(x=0.10, y=0.0, yaw=math.pi),
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            position_tolerance=0.18,
            yaw_tolerance=0.05,
            require_yaw_alignment=False,
        )
        executor.reset(plan)

        final_state = _state(x=0.10, y=0.0, yaw=0.0)

        self.assertTrue(executor.is_done(final_state))
        self.assertTrue(executor.status()["success"])
        self.assertEqual(executor.status()["acceptance_mode"], "xy_only")

    def test_stall_detector_produces_structured_terminal_status(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((60, 60), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        planner = AStarNavPlanner(grid_map=grid)
        state = _state(x=0.0, y=0.0, yaw=0.0)
        plan = planner.plan(state, NavGoal(x=1.0, y=0.0, yaw=0.0))
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(
                control_dt=0.05,
                goal_tolerance=0.10,
                max_linear_accel=10.0,
            ),
            terminal_start_distance=0.25,
            stall_window_steps=4,
            stall_min_progress_m=0.05,
            stall_min_forward_command=0.05,
            stall_min_forward_ratio=0.75,
        )
        executor.reset(plan)

        actions = [executor.compute_action(state) for _ in range(4)]
        status = executor.status()

        self.assertEqual(actions[-1].base_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(actions[-1].source, "navigation_stalled")
        self.assertTrue(executor.is_done(state))
        self.assertTrue(status["done"])
        self.assertFalse(status["success"])
        self.assertTrue(status["failed"])
        self.assertTrue(status["stall_detected"])
        self.assertEqual(status["failure_reason"], "nav_collision")
        self.assertEqual(status["stall"]["sample_count"], 4)
        self.assertEqual(status["stall"]["max_displacement_m"], 0.0)
        self.assertGreaterEqual(status["stall"]["forward_command_ratio"], 0.75)

    def test_nav_to_pick_near_goal_stall_hands_off_to_pick_planner(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.05,
            (-2.0, -2.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (1.0, 0.0)),
            metadata={"execution_phase": "nav_to_pick"},
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(
                control_dt=0.05,
                goal_tolerance=0.15,
                max_linear_accel=10.0,
                use_command_velocity_window=True,
            ),
            position_tolerance=0.18,
            pick_near_goal_handoff_tolerance_m=0.32,
            stall_window_steps=4,
            stall_min_progress_m=0.05,
            stall_min_forward_command=0.05,
            stall_min_forward_ratio=0.75,
            require_yaw_alignment=False,
        )
        state = _state(x=0.75, y=0.0, yaw=0.0)
        executor.reset(plan)

        actions = [executor.compute_action(state) for _ in range(4)]
        status = executor.status()

        self.assertEqual(actions[-1].source, "navigation_completed")
        self.assertTrue(executor.is_done(state))
        self.assertTrue(status["success"])
        self.assertFalse(status["failed"])
        self.assertFalse(status["stall_detected"])
        self.assertTrue(status["near_goal_stall_handoff"])
        self.assertEqual(status["phase"], "completed_near_goal_stall")
        self.assertAlmostEqual(
            status["near_goal_stall_handoff_tolerance"],
            0.32,
        )

    def test_carry_near_goal_stall_still_fails(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.05,
            (-2.0, -2.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (1.0, 0.0)),
            metadata={"execution_phase": "carry_nav_to_place"},
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(
                control_dt=0.05,
                goal_tolerance=0.15,
                max_linear_accel=10.0,
                use_command_velocity_window=True,
            ),
            position_tolerance=0.18,
            pick_near_goal_handoff_tolerance_m=0.32,
            stall_window_steps=4,
            stall_min_progress_m=0.05,
            stall_min_forward_command=0.05,
            stall_min_forward_ratio=0.75,
            require_yaw_alignment=False,
        )
        state = _state(x=0.75, y=0.0, yaw=0.0)
        executor.reset(plan)

        actions = [executor.compute_action(state) for _ in range(4)]
        status = executor.status()

        self.assertEqual(actions[-1].source, "navigation_stalled")
        self.assertTrue(executor.is_done(state))
        self.assertFalse(status["success"])
        self.assertTrue(status["failed"])
        self.assertTrue(status["stall_detected"])
        self.assertFalse(status["near_goal_stall_handoff"])
        self.assertEqual(status["failure_reason"], "nav_collision")

    def test_executor_can_lazily_load_planner_map_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            np.save(root / "occupancy.npy", np.zeros((40, 40), dtype=bool))
            (root / "map.json").write_text(
                json.dumps(
                    {
                        "image": "occupancy.npy",
                        "resolution": 0.05,
                        "origin": [-1.0, -1.0, 0.0],
                    }
                ),
                encoding="utf-8",
            )
            planner = AStarNavPlanner(root / "map.json")
            state = _state(x=0.0, y=0.0, yaw=0.0)
            plan = planner.plan(state, NavGoal(x=0.5, y=0.0, yaw=0.0))
            executor = DwaNavExecutor(
                None,
                0.0,
                DWAConfig(control_dt=0.05),
                terminal_start_distance=0.20,
            )

            executor.reset(plan)
            action = executor.compute_action(state)

            self.assertEqual(action.source, "navigation_dwa")
            self.assertEqual(executor.map_json, str((root / "map.json").resolve()))


if __name__ == "__main__":
    unittest.main()
