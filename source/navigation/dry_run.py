"""Deterministic navigation components for control-flow dry runs."""

from __future__ import annotations

import math
from typing import Any

from source.interfaces import NavGoal, NavPlan, RobotAction, SimulationState


class DryRunNavPlanner:
    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        start = state.robot_root_pose
        midpoint = ((start[0] + goal.x) * 0.5, (start[1] + goal.y) * 0.5)
        return NavPlan(
            goal=goal,
            waypoints=((start[0], start[1]), midpoint, (goal.x, goal.y)),
            metadata={"planner": "dry_run_straight_line"},
        )


class DryRunNavExecutor:
    def __init__(self, *, ticks_per_plan: int = 3):
        self.ticks_per_plan = max(2, int(ticks_per_plan))
        self.plan: NavPlan | None = None
        self.tick_index = 0

    def reset(self, plan: NavPlan) -> None:
        self.plan = plan
        self.tick_index = 0

    def compute_action(self, state: SimulationState) -> RobotAction:
        if self.plan is None:
            raise RuntimeError("navigation executor has no plan")
        self.tick_index += 1
        goal = self.plan.goal
        dx = goal.x - state.robot_root_pose[0]
        dy = goal.y - state.robot_root_pose[1]
        distance = math.hypot(dx, dy)
        speed = min(0.5, distance)
        vx = speed if distance > 1.0e-9 else 0.0
        metadata: dict[str, Any] = {
            "progress": self.tick_index / self.ticks_per_plan,
            "goal": (goal.x, goal.y, goal.yaw),
        }
        if self.tick_index >= self.ticks_per_plan:
            metadata["dry_run_effect"] = "nav_reached"
        return RobotAction(
            base_velocity=(vx, 0.0, 0.0),
            source="dry_run_navigation",
            metadata=metadata,
        )

    def is_done(self, state: SimulationState) -> bool:
        del state
        return self.plan is not None and self.tick_index >= self.ticks_per_plan

    def status(self) -> dict[str, Any]:
        return {
            "backend": "dry_run",
            "tick_index": self.tick_index,
            "ticks_per_plan": self.ticks_per_plan,
            "done": self.plan is not None and self.tick_index >= self.ticks_per_plan,
        }
