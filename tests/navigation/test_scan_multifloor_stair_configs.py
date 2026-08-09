"""校验 multi_floor 手工楼梯任务、Path 与 collision PLY 的加载合同。"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import yaml

from scripts.pipeline.run_full_physics_pipeline import (
    _apply_scan_manual_path_goal_override,
    _parse_args,
)
from source.navigation.pct_adapter import sim_to_pct_xyz
from source.scene.placement_support import (
    _vertical_intersections,
    load_binary_triangle_ply,
)
from source.simulation.isaaclab_runtime import (
    FRONT_CAMERA_MOUNT_POS_XYZ_M,
    FRONT_CAMERA_MOUNT_ROT_WXYZ,
)
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCAN_TOOLS_ROOT = PROJECT_ROOT / "ros2_ws/src/scan_navigation_tools"
sys.path.insert(0, str(SCAN_TOOLS_ROOT))

from scan_navigation_tools.path_geometry import prepare_path_points  # noqa: E402


TWO_STEP_TASK = (
    PROJECT_ROOT / "tasks/nav_smoke_scan_multifloor_stair_two_step.json"
)
FIRST_FLIGHT_TASK = (
    PROJECT_ROOT / "tasks/nav_smoke_scan_multifloor_stair_first_flight.json"
)
BASE_TASK = PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
COLLISION_PLY = (
    PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_collision.ply"
)
SCAN_PLANNER_CONFIG = (
    PROJECT_ROOT / "ros2_ws/src/scan_planner/config/planner.yaml"
)
BRIDGE_CONFIG = (
    PROJECT_ROOT / "ros2_ws/src/isaac_navigation_bridge/config/pct_scan.yaml"
)


def _load_path(task_path: Path):
    task = JsonTaskProvider().load(task_path)
    path_path = PROJECT_ROOT / task.raw_task["notes"]["online_reference_path"]
    raw = yaml.safe_load(path_path.read_text(encoding="utf-8"))
    parameters = raw["manual_path_publisher"]["ros__parameters"]
    points = prepare_path_points(
        parameters["points_xyz"],
        min_point_distance_m=parameters["min_point_distance_m"],
    )
    return task, path_path, parameters, points


@pytest.mark.parametrize(
    ("task_path", "expected_point_count"),
    [
        (TWO_STEP_TASK, 6),
        (FIRST_FLIGHT_TASK, 21),
    ],
)
def test_multifloor_stair_task_preserves_scene_and_matches_path_goal(
    task_path: Path,
    expected_point_count: int,
) -> None:
    task, path_path, parameters, points = _load_path(task_path)
    base = JsonTaskProvider().load(BASE_TASK)

    assert task.raw_task["scene_profile"] == "multi_floor"
    assert task.raw_task["scene_runtime"] == base.raw_task["scene_runtime"]
    assert task.scene_usd == base.scene_usd
    assert task.nav_map == base.nav_map
    assert path_path.is_file()
    assert parameters["use_sim_time"] is True
    assert parameters["topic"] == "/initial_path"
    assert parameters["frame_id"] == "world"
    assert len(points) == expected_point_count

    first = points[0]
    last = points[-1]
    first_yaw = math.atan2(
        points[1].y - first.y,
        points[1].x - first.x,
    )
    last_yaw = math.atan2(
        last.y - points[-2].y,
        last.x - points[-2].x,
    )
    assert (task.start.x, task.start.y) == pytest.approx((first.x, first.y))
    assert task.start.z == pytest.approx(first.z + 0.30)
    assert task.start.yaw == pytest.approx(first_yaw)
    assert (task.pick_goal.x, task.pick_goal.y) == pytest.approx(
        (last.x, last.y)
    )
    assert task.pick_goal.z == pytest.approx(last.z + 0.30)
    assert task.pick_goal.yaw == pytest.approx(last_yaw)
    assert task.start.floor_id == task.pick_goal.floor_id == "F1"
    assert task.start.slice_id is task.pick_goal.slice_id is None
    assert task.place_goal is None
    assert task.object_prim_path is None
    assert task.raw_task["pick"]["grasp_mode"] == "none"
    assert task.raw_task["carry"]["arm_posture"] == "stow"

    horizontal_spacings = [
        math.hypot(end.x - start.x, end.y - start.y)
        for start, end in zip(points, points[1:])
    ]
    assert horizontal_spacings
    assert min(horizontal_spacings) >= 0.18
    assert max(horizontal_spacings) <= 0.20


def test_two_step_path_is_exact_prefix_of_first_flight() -> None:
    _, _, _, two_step = _load_path(TWO_STEP_TASK)
    _, _, _, first_flight = _load_path(FIRST_FLIGHT_TASK)

    assert [
        (point.x, point.y, point.z)
        for point in two_step
    ] == pytest.approx(
        [
            (point.x, point.y, point.z)
            for point in first_flight[: len(two_step)]
        ]
    )
    assert two_step[-1].z - two_step[0].z == pytest.approx(
        0.3055225858991939
    )
    assert first_flight[-1].z - first_flight[0].z == pytest.approx(
        1.5676103786288237
    )

    # 两级 Path 只是过程诊断。正式终点必须继续提供至少一个完整双圆柱
    # 半长的支撑面，使前后机身都能落在有定义的地形上。
    required_support_tail_m = 0.27 + 0.16
    support_tail_m = sum(
        math.dist(
            (start.x, start.y),
            (end.x, end.y),
        )
        for start, end in zip(
            first_flight[len(two_step) - 1 :],
            first_flight[len(two_step) :],
        )
    )
    assert support_tail_m >= required_support_tail_m


def test_scan_vertical_inflation_maps_measured_robot_envelope() -> None:
    planner = yaml.safe_load(SCAN_PLANNER_CONFIG.read_text(encoding="utf-8"))
    planner_parameters = planner["scan_planner_node"]["ros__parameters"]
    bridge = yaml.safe_load(BRIDGE_CONFIG.read_text(encoding="utf-8"))
    bridge_parameters = bridge["isaac_navigation_bridge"]["ros__parameters"]

    inflation_up = planner_parameters["grid_map.obstacles_inflation_z_up"]
    inflation_down = planner_parameters[
        "grid_map.obstacles_inflation_z_down"
    ]
    # GridMap 中 obstacle-up 对应机器人下界，obstacle-down 对应上界。
    assert -inflation_up == pytest.approx(
        bridge_parameters["filters.self_z_min_m"]
    )
    assert inflation_down == pytest.approx(
        bridge_parameters["filters.self_z_max_m"]
    )


def test_scan_cloud_ray_origin_matches_isaac_head_camera_mount() -> None:
    planner = yaml.safe_load(SCAN_PLANNER_CONFIG.read_text(encoding="utf-8"))
    parameters = planner["scan_planner_node"]["ros__parameters"]

    assert parameters["grid_map.cloud_is_world"] is True
    assert parameters["grid_map.need_extrinsic"] is True
    assert (
        parameters["grid_map.cloud_sensor_extrinsic_x"],
        parameters["grid_map.cloud_sensor_extrinsic_y"],
        parameters["grid_map.cloud_sensor_extrinsic_z"],
    ) == pytest.approx(FRONT_CAMERA_MOUNT_POS_XYZ_M)
    assert (
        parameters["grid_map.cloud_sensor_extrinsic_qw"],
        parameters["grid_map.cloud_sensor_extrinsic_qx"],
        parameters["grid_map.cloud_sensor_extrinsic_qy"],
        parameters["grid_map.cloud_sensor_extrinsic_qz"],
    ) == pytest.approx(FRONT_CAMERA_MOUNT_ROT_WXYZ)


def test_scan_obstructed_guide_has_separate_bounded_detour_corridor() -> None:
    planner = yaml.safe_load(SCAN_PLANNER_CONFIG.read_text(encoding="utf-8"))
    parameters = planner["scan_planner_node"]["ros__parameters"]

    normal_deviation = parameters["manager.reference_corridor_max_deviation"]
    blocked_deviation = parameters[
        "manager.reference_obstacle_corridor_max_deviation"
    ]
    normal_progress = parameters[
        "manager.reference_corridor_max_progress_lead"
    ]
    blocked_progress = parameters[
        "manager.reference_obstacle_corridor_max_progress_lead"
    ]
    initial_progress_tolerance = parameters[
        "manager.reference_corridor_initial_progress_tolerance"
    ]
    progress_measurement_tolerance = parameters[
        "manager.reference_corridor_progress_measurement_tolerance"
    ]
    optimizer_margin = parameters["optimization.dist0"]

    assert normal_deviation == pytest.approx(0.10)
    assert blocked_deviation == pytest.approx(0.35)
    assert normal_progress == pytest.approx(0.035)
    assert blocked_progress == pytest.approx(0.05)
    assert initial_progress_tolerance == pytest.approx(0.001)
    assert initial_progress_tolerance <= 0.005
    assert progress_measurement_tolerance == pytest.approx(0.001)
    assert progress_measurement_tolerance <= 0.002
    assert normal_progress + progress_measurement_tolerance <= 0.036 + 1.0e-12
    assert blocked_progress + progress_measurement_tolerance <= 0.051 + 1.0e-12
    assert blocked_deviation > normal_deviation + optimizer_margin
    assert blocked_progress >= normal_progress


def test_scan_stable_baseline_does_not_require_active_sensing() -> None:
    """原地转头复观测只作为可选专项能力，不进入稳定基线。"""

    planner = yaml.safe_load(SCAN_PLANNER_CONFIG.read_text(encoding="utf-8"))
    parameters = planner["scan_planner_node"]["ros__parameters"]

    assert parameters["fsm.enable_active_sensing"] is False


def test_scan_reference_target_keeps_bspline_terminal_free_runway() -> None:
    planner = yaml.safe_load(SCAN_PLANNER_CONFIG.read_text(encoding="utf-8"))
    parameters = planner["scan_planner_node"]["ros__parameters"]

    runway = parameters["fsm.reference_target_free_runway"]
    resolution = parameters["grid_map.resolution"]
    control_point_distance = parameters["manager.control_points_distance"]

    assert runway == pytest.approx(0.10)
    assert runway >= 2.0 * resolution
    assert runway >= 0.5 * control_point_distance


def test_first_flight_ground_heights_are_collision_ply_intersections() -> None:
    task, _, _, points = _load_path(FIRST_FLIGHT_TASK)
    vertices, faces = load_binary_triangle_ply(COLLISION_PLY)

    assert task.raw_task["notes"]["coord_mode"] == "sim_to_pct_180deg"
    for point in points:
        pct_x, pct_y, pct_z = sim_to_pct_xyz((point.x, point.y, point.z))
        intersections = _vertical_intersections(
            vertices,
            faces,
            x=pct_x,
            y=pct_y,
        )
        assert intersections
        assert min(
            abs(intersection.z - pct_z)
            for intersection in intersections
        ) <= 1.0e-6


@pytest.mark.parametrize("task_path", [TWO_STEP_TASK, FIRST_FLIGHT_TASK])
def test_navigation_ros2_cli_only_overrides_stair_goal_xyyaw(
    task_path: Path,
) -> None:
    task, _, _, points = _load_path(task_path)
    last = points[-1]
    yaw = math.atan2(
        last.y - points[-2].y,
        last.x - points[-2].x,
    )
    args = _parse_args(
        [
            "--scene-profile",
            "multi_floor",
            "--task-json",
            str(task_path.relative_to(PROJECT_ROOT)),
            "--navigation-smoke",
            "--enable-navigation-ros2-bridge",
            "--scan-manual-path-goal-xyyaw",
            str(last.x),
            str(last.y),
            str(yaw),
        ]
    )

    assert args.scene_profile == "multi_floor"
    assert args.mode == "navigation_smoke"
    assert args.enable_navigation_ros2_bridge is True
    assert args.task_json == str(task_path.relative_to(PROJECT_ROOT))
    overridden = _apply_scan_manual_path_goal_override(
        task,
        tuple(args.scan_manual_path_goal_xyyaw),
    )
    assert overridden.pick_goal.z == task.pick_goal.z
    assert overridden.pick_goal.floor_id == task.pick_goal.floor_id
    assert overridden.pick_goal.slice_id == task.pick_goal.slice_id
    assert (
        overridden.pick_goal.x,
        overridden.pick_goal.y,
        overridden.pick_goal.yaw,
    ) == pytest.approx((last.x, last.y, yaw))
