"""动态障碍任务合同与确定性轨迹的离线测试。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from source.simulation.dynamic_obstacles import resolve_dynamic_obstacle_plan
from source.tasks.task_loader import episode_spec_from_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _valid_task_fragment() -> dict:
    return {
        "dynamic_obstacles": [
            {
                "id": "crossing_cart",
                "shape": "cuboid",
                "floor_id": "F1",
                "surface_class": "flat",
                "size_xyz_m": [0.55, 0.35, 0.75],
                "waypoints_world_xyz": [
                    [-4.50, 4.20, 0.235],
                    [-3.20, 4.20, 0.235],
                ],
                "speed_mps": 0.25,
                "start_delay_s": 8.0,
                "yaw_rad": 0.0,
                "motion": "ping_pong",
                "color_rgb": [0.9, 0.2, 0.1],
                "mass_kg": 20.0,
                "collision_enabled": True,
                "visible": True,
            }
        ],
        "dynamic_obstacle_safety": {
            "minimum_stair_clearance_m": 1.0,
            "stair_exclusion_aabbs_world": [
                {
                    "id": "multifloor_scan_stair_freeze_corridor",
                    "min_xyz": [0.50, 4.50, -0.60],
                    "max_xyz": [3.20, 10.50, 3.80],
                }
            ],
        },
    }


def test_missing_configuration_keeps_dynamic_obstacles_disabled() -> None:
    plan = resolve_dynamic_obstacle_plan({"task_id": 1})

    assert not plan.enabled
    assert plan.obstacles == ()
    assert plan.topology_fingerprint() == []
    assert plan.to_dict()["source"] == "configuration_absent"


def test_dynamic_obstacle_prim_is_direct_child_of_environment_namespace() -> None:
    """动态刚体不依赖 env namespace 下尚未创建的中间 Xform。"""

    obstacle = resolve_dynamic_obstacle_plan(_valid_task_fragment()).obstacles[0]

    assert obstacle.prim_path == "{ENV_REGEX_NS}/DynamicObstacle_crossing_cart"
    assert obstacle.prim_path.rsplit("/", 1)[0] == "{ENV_REGEX_NS}"


def test_ping_pong_trajectory_uses_episode_physics_time_deterministically() -> None:
    raw_task = _valid_task_fragment()
    obstacle = raw_task["dynamic_obstacles"][0]
    obstacle["waypoints_world_xyz"] = [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]]
    obstacle["speed_mps"] = 0.5
    obstacle["start_delay_s"] = 1.0
    plan = resolve_dynamic_obstacle_plan(raw_task)
    spec = plan.obstacles[0]

    waiting = spec.state_at(1.0)
    outbound = spec.state_at(2.0)
    endpoint = spec.state_at(5.0)
    returning = spec.state_at(6.0)
    full_cycle = spec.state_at(9.0)

    assert waiting.position_world_xyz == pytest.approx((0.0, 0.0, 0.5))
    assert waiting.path_direction == 0
    assert waiting.waiting_for_start
    assert outbound.position_world_xyz == pytest.approx((0.5, 0.0, 0.5))
    assert outbound.path_direction == 1
    assert endpoint.position_world_xyz == pytest.approx((2.0, 0.0, 0.5))
    assert returning.position_world_xyz == pytest.approx((1.5, 0.0, 0.5))
    assert returning.path_direction == -1
    assert full_cycle.position_world_xyz == pytest.approx((0.0, 0.0, 0.5))


def test_one_shot_trajectory_waits_moves_then_holds_clear_endpoint() -> None:
    raw_task = _valid_task_fragment()
    obstacle = raw_task["dynamic_obstacles"][0]
    obstacle["waypoints_world_xyz"] = [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]]
    obstacle["speed_mps"] = 0.5
    obstacle["start_delay_s"] = 1.0
    obstacle["motion"] = "one_shot"
    spec = resolve_dynamic_obstacle_plan(raw_task).obstacles[0]

    waiting = spec.state_at(1.0)
    moving = spec.state_at(2.0)
    endpoint = spec.state_at(5.0)
    held = spec.state_at(20.0)

    assert waiting.position_world_xyz == pytest.approx((0.0, 0.0, 0.5))
    assert waiting.waiting_for_start
    assert waiting.path_direction == 0
    assert moving.position_world_xyz == pytest.approx((0.5, 0.0, 0.5))
    assert not moving.waiting_for_start
    assert moving.path_direction == 1
    assert endpoint.position_world_xyz == pytest.approx((2.0, 0.0, 0.5))
    assert endpoint.path_direction == 0
    assert held.position_world_xyz == pytest.approx(endpoint.position_world_xyz)
    assert held.path_distance_m == pytest.approx(spec.total_path_length_m)
    assert held.path_direction == 0


def test_dynamic_obstacle_rejects_unknown_motion_contract() -> None:
    raw_task = _valid_task_fragment()
    raw_task["dynamic_obstacles"][0]["motion"] = "teleport"

    with pytest.raises(ValueError, match="只支持 ping_pong 或 one_shot"):
        resolve_dynamic_obstacle_plan(raw_task)


def test_swept_aabb_is_inflated_by_cuboid_half_extent() -> None:
    plan = resolve_dynamic_obstacle_plan(_valid_task_fragment())
    swept = plan.obstacles[0].swept_aabb_world

    assert swept.minimum_world_xyz == pytest.approx((-4.775, 4.025, -0.140))
    assert swept.maximum_world_xyz == pytest.approx((-2.925, 4.375, 0.610))
    assert plan.to_dict()["stair_corridor_overlap_verified_false"] is True


def test_swept_aabb_projects_nonsquare_cuboid_half_extents_at_right_angle() -> None:
    raw_task = _valid_task_fragment()
    obstacle = raw_task["dynamic_obstacles"][0]
    obstacle["size_xyz_m"] = [0.20, 1.00, 0.40]
    obstacle["waypoints_world_xyz"] = [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]]
    obstacle["yaw_rad"] = math.pi / 2.0
    plan = resolve_dynamic_obstacle_plan(raw_task)
    swept = plan.obstacles[0].swept_aabb_world

    assert swept.minimum_world_xyz == pytest.approx((-0.50, -0.10, 0.30))
    assert swept.maximum_world_xyz == pytest.approx((1.50, 0.10, 0.70))


def test_rotated_swept_aabb_cannot_enter_stair_exclusion_at_diagonal() -> None:
    raw_task = _valid_task_fragment()
    obstacle = raw_task["dynamic_obstacles"][0]
    obstacle["size_xyz_m"] = [0.20, 1.00, 0.40]
    obstacle["waypoints_world_xyz"] = [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]]
    obstacle["yaw_rad"] = math.pi / 4.0
    raw_task["dynamic_obstacle_safety"] = {
        "minimum_stair_clearance_m": 0.0,
        "stair_exclusion_aabbs_world": [
            {
                "id": "diagonal_edge",
                "min_xyz": [0.40, -1.0, 0.0],
                "max_xyz": [2.00, 1.0, 1.0],
            }
        ],
    }

    with pytest.raises(ValueError, match="侵入楼梯排除区 'diagonal_edge'"):
        resolve_dynamic_obstacle_plan(raw_task)


def test_stair_corridor_overlap_fails_closed_with_clearance() -> None:
    raw_task = _valid_task_fragment()
    raw_task["dynamic_obstacles"][0]["waypoints_world_xyz"] = [
        [-0.60, 5.00, 0.235],
        [0.20, 5.00, 0.235],
    ]

    with pytest.raises(ValueError, match="侵入楼梯排除区"):
        resolve_dynamic_obstacle_plan(raw_task)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("floor_id", "stair", "只允许 F1 或 F2"),
        ("surface_class", "stairs", "必须显式为 flat"),
        ("visible", False, "必须同时启用"),
        ("collision_enabled", False, "必须同时启用"),
    ],
)
def test_dynamic_obstacle_rejects_nonflat_or_sensor_collision_mismatch(
    field: str,
    value: object,
    message: str,
) -> None:
    raw_task = _valid_task_fragment()
    raw_task["dynamic_obstacles"][0][field] = value

    with pytest.raises(ValueError, match=message):
        resolve_dynamic_obstacle_plan(raw_task)


def test_checked_in_f1_cart_task_is_default_isolated_from_stair_corridor() -> None:
    task_path = PROJECT_ROOT / "tasks/nav_smoke_scan_multifloor_dynamic_cart_f1.json"
    raw_task = json.loads(task_path.read_text(encoding="utf-8"))

    episode_spec = episode_spec_from_dict(raw_task)
    plan = resolve_dynamic_obstacle_plan(raw_task)

    assert episode_spec.raw_task["dynamic_obstacles"] == raw_task["dynamic_obstacles"]
    assert episode_spec.object_prim_path is None
    assert plan.enabled
    assert len(plan.obstacles) == 1
    assert plan.obstacles[0].floor_id == "F1"
    assert plan.obstacles[0].surface_class == "flat"
    assert plan.obstacles[0].obstacle_id == "crossing_cart"
    assert plan.obstacles[0].motion == "one_shot"
    assert plan.obstacles[0].start_delay_s == pytest.approx(44.0)
    assert plan.obstacles[0].speed_mps == pytest.approx(0.1)
    assert plan.obstacles[0].waypoints_world_xyz[0] == pytest.approx(
        (-4.15, 4.20, 0.235)
    )
    assert plan.obstacles[0].waypoints_world_xyz[1] == pytest.approx(
        (-4.60, 4.20, 0.235)
    )
    assert plan.obstacles[0].total_path_length_m == pytest.approx(0.45)
    assert plan.obstacles[0].state_at(44.0).waiting_for_start
    assert plan.obstacles[0].state_at(46.25).position_world_xyz == pytest.approx(
        (-4.375, 4.20, 0.235)
    )
    assert plan.obstacles[0].state_at(48.5).position_world_xyz == pytest.approx(
        (-4.60, 4.20, 0.235)
    )
    assert plan.obstacles[0].state_at(60.0).position_world_xyz == pytest.approx(
        (-4.60, 4.20, 0.235)
    )
    reference_x = float(raw_task["pick"]["base_goal"]["x"])
    initial_right_face_x = -4.15 + 0.55 / 2.0
    initial_surface_clearance = reference_x - initial_right_face_x
    required_detour_with_optimizer_margin = (
        0.27 + 0.16 - initial_surface_clearance + 0.20
    )
    assert 0.10 < required_detour_with_optimizer_margin < 0.35
    endpoint_right_face_x = -4.60 + 0.55 / 2.0
    assert reference_x - endpoint_right_face_x > 0.27 + 0.16 + 0.20
    assert plan.obstacles[0].swept_aabb_world.minimum_world_xyz == pytest.approx(
        (-4.875, 4.025, -0.140)
    )
    assert plan.obstacles[0].swept_aabb_world.maximum_world_xyz == pytest.approx(
        (-3.875, 4.375, 0.610)
    )
    assert plan.to_dict()["stair_corridor_overlap_verified_false"] is True


def test_checked_in_f1_blocker_task_is_one_shot_and_stair_isolated() -> None:
    task_path = (
        PROJECT_ROOT
        / "tasks/nav_smoke_scan_multifloor_dynamic_blocker_replan_f1.json"
    )
    raw_task = json.loads(task_path.read_text(encoding="utf-8"))

    episode_spec = episode_spec_from_dict(raw_task)
    plan = resolve_dynamic_obstacle_plan(raw_task)
    blocker = plan.obstacles[0]

    assert episode_spec.task_id == 17705
    assert episode_spec.object_prim_path is None
    assert blocker.obstacle_id == "blocking_cart"
    assert blocker.motion == "one_shot"
    assert blocker.start_delay_s == pytest.approx(3.0)
    assert blocker.speed_mps == pytest.approx(0.8)
    assert blocker.state_at(3.0).waiting_for_start
    assert blocker.state_at(5.0).position_world_xyz == pytest.approx(
        blocker.waypoints_world_xyz[-1]
    )
    assert blocker.state_at(5.0).path_direction == 0
    assert blocker.swept_aabb_world.minimum_world_xyz == pytest.approx(
        (-4.65, 3.975, -0.165)
    )
    assert blocker.swept_aabb_world.maximum_world_xyz == pytest.approx(
        (-1.60, 4.425, 0.635)
    )
    assert plan.to_dict()["stair_corridor_overlap_verified_false"] is True


def test_checked_in_f1_blocker_motion_fits_supervisor_replan_budget() -> None:
    """阻断窗口须覆盖首次失败阈值，离场又须早于三轮耗尽。"""

    task_path = (
        PROJECT_ROOT
        / "tasks/nav_smoke_scan_multifloor_dynamic_blocker_replan_f1.json"
    )
    raw_task = json.loads(task_path.read_text(encoding="utf-8"))
    blocker = resolve_dynamic_obstacle_plan(raw_task).obstacles[0]
    retry_period_s = 0.50
    failures_per_replan = 5
    maximum_replan_cycles = 3
    first_replan_budget_s = retry_period_s * failures_per_replan
    exhaustion_budget_s = first_replan_budget_s * maximum_replan_cycles
    endpoint_time_s = (
        blocker.start_delay_s
        + blocker.total_path_length_m / blocker.speed_mps
    )

    assert blocker.start_delay_s > first_replan_budget_s
    assert endpoint_time_s < exhaustion_budget_s
