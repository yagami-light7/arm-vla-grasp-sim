"""纯 CPU 的完整物理导航规划与逐 tick 执行测试。"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

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
        planner = AStarNavPlanner(
            PROJECT_ROOT / spec.nav_map,
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
