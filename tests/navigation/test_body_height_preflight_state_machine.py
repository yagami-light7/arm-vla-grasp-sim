"""测试 production PCT goal 发布前的 live body-height 硬门。"""

from __future__ import annotations

from copy import deepcopy
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from source.interfaces import (
    EpisodeSpec,
    NavGoal,
    NavPlan,
    SimulationState,
    VerificationResult,
)
from source.navigation.pct_adapter import sim_to_pct_xyz
from source.pipeline import FullPhysicsConfig, NavigationSettings, PipelineState
from source.pipeline.navigation_smoke import (
    enable_production_pct_goal_body_height_calibration,
)
from source.pipeline.state_machine import FullPhysicsStateMachine
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MULTIFLOOR_COLLISION_PLY = (
    PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_collision.ply"
)
ARM_NAMES = tuple(f"arm_joint{index}" for index in range(1, 7))
ARM_STOW = (0.0,) * 6


def _write_two_floor_ply(path: Path) -> None:
    """写入覆盖 sim (1, 2) 附近的 F1/F2 两层碰撞面。"""

    vertices = (
        (-2.0, -3.0, 0.0),
        (0.0, -3.0, 0.0),
        (-1.0, -1.0, 0.0),
        (-2.0, -3.0, 3.2),
        (0.0, -3.0, 3.2),
        (-1.0, -1.0, 3.2),
    )
    faces = ((0, 1, 2), (3, 4, 5))
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 6\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 2\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        for vertex in vertices:
            stream.write(struct.pack("<fff", *vertex))
        for face in faces:
            stream.write(struct.pack("<Biii", 3, *face))


def _write_sim_support_patches(
    path: Path,
    supports: tuple[tuple[float, float, float], ...],
) -> None:
    """在给定 sim 坐标处写入互不相交的水平碰撞支撑面。"""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    half_extent = 0.40
    for sim_x, sim_y, sim_z in supports:
        pct_x, pct_y, pct_z = sim_to_pct_xyz((sim_x, sim_y, sim_z))
        first_vertex = len(vertices)
        vertices.extend(
            (
                (pct_x - half_extent, pct_y - half_extent, pct_z),
                (pct_x + half_extent, pct_y - half_extent, pct_z),
                (pct_x, pct_y + half_extent, pct_z),
            )
        )
        faces.append((first_vertex, first_vertex + 1, first_vertex + 2))

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        for vertex in vertices:
            stream.write(struct.pack("<fff", *vertex))
        for face in faces:
            stream.write(struct.pack("<Biii", 3, *face))


class _GoalPlanner:
    def __init__(self, *, publish_pct_goal: bool = True) -> None:
        self.publish_pct_goal = publish_pct_goal
        self.goals: list[NavGoal] = []

    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        self.goals.append(goal)
        start = tuple(float(value) for value in state.robot_root_pose[:3])
        return NavPlan(
            goal=goal,
            waypoints=(start, (goal.x, goal.y, float(goal.z or start[2]))),
            metadata={
                "pct_goal_request": {
                    "frame_id": "world",
                    "position_base_xyz": (goal.x, goal.y, goal.z),
                    "yaw": goal.yaw,
                    "height_semantics": "base",
                }
            },
        )


class _Executor:
    def __init__(self) -> None:
        self.plans: list[NavPlan] = []

    def reset(self, plan: NavPlan) -> None:
        self.plans.append(plan)


class _Verifier:
    def __init__(self) -> None:
        self.specs: list[object] = []

    def verify_pick_reachable(self, _state: SimulationState, spec: object):
        self.specs.append(spec)
        return VerificationResult(success=True, failure_reason="", metadata={})


def _machine(
    tmp_path: Path,
    *,
    navigation_body_height_m: float = 0.30,
    body_height_calibration_enabled: bool = True,
    body_height_calibration_quick_enabled: bool = False,
    publish_pct_goal: bool = True,
    navigation_smoke: bool = False,
    navigation_carry_smoke: bool = False,
    stair_locomotion_smoke: bool = True,
    stair_fixed_command_probe: bool = False,
    full_physics: bool = False,
    episode_spec: EpisodeSpec | None = None,
    task_json: Path | None = None,
    place_goal: NavGoal | None = None,
    support_surfaces: tuple[tuple[float, float, float], ...] | None = None,
    collision_ply_path: Path | None = None,
) -> tuple[FullPhysicsStateMachine, _GoalPlanner]:
    if collision_ply_path is not None and support_surfaces is not None:
        raise ValueError("collision_ply_path 与 support_surfaces 不能同时传入")
    collision_ply = collision_ply_path or (tmp_path / "collision.ply")
    if collision_ply_path is not None:
        if not collision_ply.is_file():
            pytest.skip(f"缺少真实 collision PLY: {collision_ply}")
    elif support_surfaces is None:
        _write_two_floor_ply(collision_ply)
    else:
        _write_sim_support_patches(collision_ply, support_surfaces)
    if episode_spec is None:
        base_spec = JsonTaskProvider().load(
            PROJECT_ROOT / "tasks/nav_smoke_example.json"
        )
        episode_spec = replace(
            base_spec,
            start=NavGoal(
                x=1.0,
                y=2.0,
                z=0.37,
                yaw=0.0,
                floor_id="F1",
            ),
            pick_goal=NavGoal(
                x=1.1,
                y=2.0,
                z=3.55,
                yaw=0.4,
                floor_id="F2",
            ),
            place_goal=place_goal,
        )
    elif place_goal is not None:
        raise ValueError("episode_spec 与 place_goal 不能同时传入")
    task_json = task_json or PROJECT_ROOT / "tasks/nav_smoke_example.json"
    navigation = NavigationSettings(
        global_planner=("bypassed" if stair_fixed_command_probe else "pct"),
        pct_enabled=False,
        pct_collision_ply_path=collision_ply,
        navigation_body_height_m=navigation_body_height_m,
        stair_fixed_command_probe=stair_fixed_command_probe,
        body_height_calibration_enabled=body_height_calibration_enabled,
        body_height_calibration_min_samples=2,
        body_height_calibration_min_duration_s=0.1,
        body_height_calibration_timeout_s=2.0,
        body_height_calibration_max_joint_position_dt_s=0.11,
        body_height_calibration_max_hint_error_m=0.60,
        body_height_calibration_max_mad_m=0.01,
        body_height_calibration_max_spread_m=0.02,
        body_height_calibration_quick_enabled=(
            body_height_calibration_quick_enabled
        ),
        body_height_calibration_quick_min_samples=2,
        body_height_calibration_quick_min_duration_s=0.1,
        body_height_calibration_quick_max_mad_m=0.005,
        body_height_calibration_quick_max_spread_m=0.01,
        body_height_calibration_quick_contract_tolerance_m=0.02,
    )
    planner = _GoalPlanner(publish_pct_goal=publish_pct_goal)
    machine = FullPhysicsStateMachine(
        config=FullPhysicsConfig(
            task_json=task_json,
            output_dir=tmp_path,
            navigation=navigation,
            navigation_smoke=navigation_smoke,
            navigation_carry_smoke=navigation_carry_smoke,
            stair_locomotion_smoke=stair_locomotion_smoke,
            full_physics=full_physics,
        ),
        episode_spec=episode_spec,
        episode_seed=0,
        simulation=object(),
        nav_planner=planner,
        nav_executor=_Executor(),
        manipulation_planner=object(),
        arm_executor=object(),
        gripper=object(),
        verifier=object(),
        recorder=object(),
    )
    machine.state = PipelineState.RESET_EPISODE
    machine._body_height_preflight_wait_for_post_reset_frame = True
    return machine, planner


def test_preflight_calibrator_reads_unique_navigation_body_height(
    tmp_path: Path,
) -> None:
    machine, _ = _machine(tmp_path, navigation_body_height_m=0.34)

    assert machine._body_height_calibrator is not None
    assert (
        machine._body_height_calibrator.config.configured_body_height_hint_m
        == pytest.approx(0.34)
    )


def test_state_machine_wires_quick_window_and_reports_certification_mode(
    tmp_path: Path,
) -> None:
    machine, _ = _machine(
        tmp_path,
        body_height_calibration_quick_enabled=True,
    )

    calibrator = machine._body_height_calibrator
    assert calibrator is not None
    assert calibrator.config.quick_minimum_consecutive_samples == 2
    assert calibrator.config.quick_minimum_stable_duration_s == pytest.approx(
        0.1
    )

    events = _complete_preflight(machine)

    assert machine._body_height_preflight_result is not None
    assert (
        machine._body_height_preflight_result.certification_mode
        == "quick_window"
    )
    assert events[0].metadata["calibration_certification_mode"] == (
        "quick_window"
    )
    assert events[0].metadata["calibration_quick_fallback_reason"] is None


@pytest.mark.parametrize(
    "mode",
    (
        {"navigation_smoke": True},
        {"navigation_carry_smoke": True},
        {"full_physics": True},
    ),
    ids=("navigation", "carry", "full_physics"),
)
def test_production_pct_goal_factory_binding_auto_enables_preflight(
    tmp_path: Path,
    mode: dict[str, bool],
) -> None:
    original = FullPhysicsConfig(
        task_json=PROJECT_ROOT / "tasks/nav_smoke_example.json",
        output_dir=tmp_path,
        navigation=NavigationSettings(body_height_calibration_enabled=False),
        **mode,
    )

    bound = enable_production_pct_goal_body_height_calibration(
        original,
        planner=_GoalPlanner(publish_pct_goal=True),
    )

    assert bound is not original
    assert original.navigation.body_height_calibration_enabled is False
    assert bound.navigation.body_height_calibration_enabled is True


@pytest.mark.parametrize(
    "machine_kwargs",
    (
        {
            "navigation_smoke": True,
            "stair_locomotion_smoke": False,
        },
        {
            "stair_locomotion_smoke": True,
            "stair_fixed_command_probe": True,
        },
    ),
    ids=("manual_path", "fixed_command_probe"),
)
def test_non_pct_goal_modes_do_not_enable_preflight(
    tmp_path: Path,
    machine_kwargs: dict[str, bool],
) -> None:
    machine, _ = _machine(
        tmp_path,
        publish_pct_goal=False,
        **machine_kwargs,
    )

    assert machine._body_height_preflight_required is False
    assert machine._body_height_calibrator is None

    disabled = replace(
        machine.config,
        navigation=replace(
            machine.config.navigation,
            body_height_calibration_enabled=False,
        ),
    )
    assert (
        enable_production_pct_goal_body_height_calibration(
            disabled,
            planner=_GoalPlanner(publish_pct_goal=False),
        )
        is disabled
    )


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_joint_position_difference_interval_must_be_finite_positive(
    tmp_path: Path,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_smoke_example.json",
            output_dir=tmp_path,
            navigation=NavigationSettings(
                body_height_calibration_max_joint_position_dt_s=value
            ),
        )


def test_joint_position_difference_interval_cannot_undersample_control(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="smaller than control_dt"):
        FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_smoke_example.json",
            output_dir=tmp_path,
            navigation=NavigationSettings(
                body_height_calibration_max_joint_position_dt_s=0.01,
                control_dt=0.02,
            ),
        )


def _observation(
    step: int,
    *,
    root_xy: tuple[float, float] = (1.0, 2.0),
    root_z: float = 0.30,
    written_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_lock_active: bool = False,
    arm_positions: tuple[float, ...] = ARM_STOW,
    arm_velocities: tuple[float, ...] = (0.0,) * 6,
    odom_timestamp_s: float | None = None,
    used_direct_joint_state: bool = False,
) -> SimulationState:
    timestamp = step * 0.1 if odom_timestamp_s is None else odom_timestamp_s
    return SimulationState(
        step_index=step,
        timestamp=timestamp,
        robot_root_pose=(
            root_xy[0],
            root_xy[1],
            root_z,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
        robot_root_velocity=(0.0,) * 6,
        joint_positions=arm_positions,
        joint_velocities=arm_velocities,
        metadata={
            "joint_names": ARM_NAMES,
            "used_direct_joint_state": used_direct_joint_state,
            "manipulation_base_lock_active": base_lock_active,
            "manipulation_support_joint_lock_active": False,
            "navigation_joint_pose_lock_active": False,
            "navigation_object_follow_active": False,
            "scan_cmd_vel_last_write_report": {
                "write_sequence": step,
                "owner_id": "scan_cmd_vel",
                "written_command": list(written_command),
                "navigation_cmd_vel_inhibited": True,
                "navigation_cmd_vel_inhibit_reason": "body_height_preflight",
            },
            "navigation_ros2_last_publish_report": {
                "completed_control_step": step,
                "timestamp": timestamp,
                "odometry_published": True,
                "odometry_publish_count": step,
            },
        },
    )


def _complete_preflight(
    machine: FullPhysicsStateMachine,
    *,
    root_xy: tuple[float, float] = (1.0, 2.0),
    root_z: float = 0.30,
) -> list[object]:
    """推进最小稳定样本窗口并返回完成事件。"""

    last_events: list[object] = []
    for step in range(1, 8):
        machine.state_ticks = step
        _, last_events, completed = machine._advance_body_height_preflight(
            _observation(step, root_xy=root_xy, root_z=root_z)
        )
        if completed:
            return last_events
        if machine.state == PipelineState.FAILED:
            pytest.fail(f"body-height preflight 意外失败: {machine.failure_reason}")
    pytest.fail("body-height preflight 未在最小稳定窗口内完成")


def test_flat_navigation_smoke_projects_only_pick_and_binds_provenance(
    tmp_path: Path,
) -> None:
    raw_place = NavGoal(
        x=1.2,
        y=2.0,
        z=3.55,
        yaw=-0.3,
        floor_id="F2",
    )
    machine, planner = _machine(
        tmp_path,
        navigation_smoke=True,
        stair_locomotion_smoke=False,
        place_goal=raw_place,
    )
    original_spec = machine.episode_spec
    original_raw_task = deepcopy(machine.episode_spec.raw_task)

    assert machine._body_height_preflight_required is True
    events = _complete_preflight(machine)

    assert events[0].metadata["projected_navigation_phases"] == ["pick"]
    assert machine._effective_pick_goal is not None
    assert machine._effective_pick_goal.z == pytest.approx(3.5)
    assert machine._effective_place_goal is None
    assert machine.episode_spec is original_spec
    assert machine.episode_spec.pick_goal.z == 3.55
    assert machine.episode_spec.place_goal == raw_place
    assert machine.episode_spec.raw_task == original_raw_task

    machine._plan_nav_to_pick(_observation(8, root_z=3.5))
    assert planner.goals == [machine._effective_pick_goal]
    plan = machine.nav_executor.plans[-1]
    request = plan.metadata["pct_goal_request"]
    assert request["effective_goal_provenance_required"] is True
    assert request["effective_goal_provenance"]["navigation_phase"] == "pick"
    assert request["position_base_xyz"][2] == pytest.approx(3.5)


def test_navigation_carry_smoke_projects_only_place_and_binds_provenance(
    tmp_path: Path,
) -> None:
    raw_place = NavGoal(
        x=1.2,
        y=2.0,
        z=3.55,
        yaw=-0.3,
        floor_id="F2",
    )
    machine, planner = _machine(
        tmp_path,
        navigation_carry_smoke=True,
        stair_locomotion_smoke=False,
        place_goal=raw_place,
    )
    original_spec = machine.episode_spec

    events = _complete_preflight(machine)

    assert events[0].metadata["projected_navigation_phases"] == ["place"]
    assert machine._effective_pick_goal is None
    assert machine._effective_place_goal is not None
    assert machine._effective_place_goal.z == pytest.approx(3.5)
    assert machine.episode_spec is original_spec
    assert machine.episode_spec.place_goal == raw_place

    machine._plan_nav_to_place(_observation(8, root_z=3.5))
    assert planner.goals == [machine._effective_place_goal]
    plan = machine.nav_executor.plans[-1]
    request = plan.metadata["pct_goal_request"]
    assert request["effective_goal_provenance_required"] is True
    assert request["effective_goal_provenance"]["navigation_phase"] == "place"
    assert request["position_base_xyz"][2] == pytest.approx(3.5)


def test_legacy_task_1002_real_ply_projects_pick_and_place_without_mutation(
    tmp_path: Path,
) -> None:
    task_json = PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
    spec = JsonTaskProvider().load(task_json)
    original_raw_task = deepcopy(spec.raw_task)
    raw_pick = spec.pick_goal
    raw_place = spec.place_goal
    assert raw_place is not None

    machine, planner = _machine(
        tmp_path,
        stair_locomotion_smoke=False,
        full_physics=True,
        episode_spec=spec,
        task_json=task_json,
        collision_ply_path=MULTIFLOOR_COLLISION_PLY,
    )
    events = _complete_preflight(
        machine,
        root_xy=(spec.start.x, spec.start.y),
        root_z=0.18324563340693345,
    )

    assert events[0].metadata["projected_navigation_phases"] == ["pick", "place"]
    assert machine._effective_pick_goal is not None
    assert machine._effective_place_goal is not None
    # 当前任务的可抓取底盘站位为 y=6.6691；高度必须由同一份已绑定 PLY
    # 投影得到，不能继续沿用旧 y=6.7491 站位的地面高度常量。
    assert machine._effective_pick_goal.z == pytest.approx(0.1254986814737381)
    # place 安全站位从 y=-0.10 移到 y=-0.02 后，真实支撑面降低约 2mm；
    # 必须重新从同一 PLY 投影，不能继续复用旧 XY 的高度常量。
    assert machine._effective_place_goal.z == pytest.approx(3.3019148148024557)
    assert machine.episode_spec is spec
    assert machine.episode_spec.pick_goal is raw_pick
    assert machine.episode_spec.place_goal is raw_place
    assert machine.episode_spec.pick_goal.z == pytest.approx(0.36742)
    assert machine.episode_spec.place_goal.z == pytest.approx(3.62628)
    assert machine.episode_spec.raw_task == original_raw_task

    observation = _observation(
        8,
        root_xy=(spec.start.x, spec.start.y),
        root_z=0.18324563340693345,
    )
    machine._plan_nav_to_pick(observation)
    machine._plan_nav_to_place(observation)
    assert planner.goals == [
        machine._effective_pick_goal,
        machine._effective_place_goal,
    ]
    phases = [
        plan.metadata["pct_goal_request"]["effective_goal_provenance"][
            "navigation_phase"
        ]
        for plan in machine.nav_executor.plans
    ]
    assert phases == ["pick", "place"]


@pytest.mark.parametrize(
    "task_name",
    (
        "nav_smoke_scan_multifloor_dynamic_cart_f1.json",
        "nav_smoke_scan_multifloor_dynamic_blocker_replan_f1.json",
    ),
    ids=("dynamic_cart", "dynamic_replan"),
)
def test_dynamic_f1_correct_z_does_not_drift_or_mutate_spec(
    tmp_path: Path,
    task_name: str,
) -> None:
    task_json = PROJECT_ROOT / "tasks" / task_name
    spec = JsonTaskProvider().load(task_json)
    original_raw_task = deepcopy(spec.raw_task)
    raw_pick = spec.pick_goal

    machine, planner = _machine(
        tmp_path,
        navigation_smoke=True,
        stair_locomotion_smoke=False,
        episode_spec=spec,
        task_json=task_json,
        collision_ply_path=MULTIFLOOR_COLLISION_PLY,
    )
    _complete_preflight(
        machine,
        root_xy=(spec.start.x, spec.start.y),
        root_z=0.18324563340693345,
    )

    assert machine._effective_pick_goal is not None
    assert machine._effective_pick_goal.z == pytest.approx(raw_pick.z, abs=1.0e-9)
    assert machine._effective_pick_goal.x == raw_pick.x
    assert machine._effective_pick_goal.y == raw_pick.y
    assert machine._effective_pick_goal.yaw == raw_pick.yaw
    assert machine.episode_spec is spec
    assert machine.episode_spec.pick_goal is raw_pick
    assert machine.episode_spec.raw_task == original_raw_task

    machine._plan_nav_to_pick(
        _observation(
            8,
            root_xy=(spec.start.x, spec.start.y),
            root_z=0.18324563340693345,
        )
    )
    assert planner.goals == [machine._effective_pick_goal]
    request = machine.nav_executor.plans[-1].metadata["pct_goal_request"]
    assert request["effective_goal_provenance_required"] is True
    assert request["effective_goal_provenance"]["raw_task_goal_z"] == raw_pick.z


def test_preflight_skips_stale_reset_tick_and_builds_effective_goal(
    tmp_path: Path,
) -> None:
    machine, planner = _machine(tmp_path)

    machine.state_ticks = 1
    first_action, first_events, completed = machine._advance_body_height_preflight(
        _observation(1)
    )
    assert completed is False
    assert first_events[0].name == "body_height_preflight_started"
    assert first_events[0].metadata["stale_reset_tick_sampled"] is False
    assert first_action.metadata["navigation_cmd_vel_inhibit"] is True
    assert machine._body_height_calibrator is not None
    assert machine._body_height_calibrator.consecutive_sample_count == 0

    machine.state_ticks = 2
    _, baseline_events, completed = machine._advance_body_height_preflight(
        _observation(2)
    )
    assert completed is False
    assert baseline_events[0].metadata["reason"] == (
        "arm_joint_position_baseline_initialized"
    )
    assert machine._body_height_calibrator.consecutive_sample_count == 0
    assert planner.goals == []

    machine.state_ticks = 3
    _, _, completed = machine._advance_body_height_preflight(_observation(3))
    assert completed is False
    assert machine._body_height_calibrator.consecutive_sample_count == 1

    machine.state_ticks = 4
    _, _, completed = machine._advance_body_height_preflight(_observation(4))
    assert completed is False

    machine.state_ticks = 5
    _, events, completed = machine._advance_body_height_preflight(_observation(5))
    assert completed is True
    assert events[0].name == "body_height_preflight_completed"
    assert machine.episode_spec.start.z == 0.37
    assert machine.episode_spec.pick_goal.z == 3.55
    assert machine._effective_pick_goal is not None
    assert machine._effective_pick_goal.z == pytest.approx(3.5)
    provenance = machine._effective_pick_goal_provenance
    assert provenance["raw_task_goal_z"] == 3.55
    assert provenance["raw_task_z_used_as_height_evidence"] is False
    assert provenance["projection"]["ground_z_m"] == pytest.approx(3.2)

    machine._plan_nav_to_pick(_observation(6))
    assert planner.goals == [machine._effective_pick_goal]
    verifier = _Verifier()
    machine.verifier = verifier
    machine.state = PipelineState.VERIFY_PICK_REACHABLE
    machine._verify_pick_reachable(_observation(7, root_z=3.5))
    assert verifier.specs
    assert verifier.specs[0].pick_goal == machine._effective_pick_goal
    assert machine.episode_spec.pick_goal.z == 3.55


def test_preflight_contract_mismatch_fails_without_planning(tmp_path: Path) -> None:
    machine, planner = _machine(
        tmp_path,
        navigation_smoke=True,
        stair_locomotion_smoke=False,
    )
    machine._body_height_preflight_wait_for_post_reset_frame = False
    machine._body_height_preflight_started_state_tick = 0

    machine.state_ticks = 1
    _, _, completed = machine._advance_body_height_preflight(
        _observation(1, root_z=0.45)
    )
    assert completed is False
    machine.state_ticks = 2
    _, _, completed = machine._advance_body_height_preflight(
        _observation(2, root_z=0.45)
    )
    assert completed is False
    machine.state_ticks = 3
    action, events, completed = machine._advance_body_height_preflight(
        _observation(3, root_z=0.45)
    )

    assert completed is False
    assert machine.state == PipelineState.FAILED
    assert machine.failure_reason == "body_height_contract_mismatch"
    assert action.metadata["navigation_cmd_vel_inhibit"] is True
    assert planner.goals == []
    assert any(event.name == "episode_failed" for event in events)


def test_preflight_runtime_gate_resets_window_and_never_plans(tmp_path: Path) -> None:
    machine, planner = _machine(tmp_path)
    machine._body_height_preflight_wait_for_post_reset_frame = False
    machine._body_height_preflight_started_state_tick = 0

    machine.state_ticks = 1
    machine._advance_body_height_preflight(_observation(1))
    assert machine._body_height_calibrator is not None
    assert machine._body_height_calibrator.consecutive_sample_count == 0

    machine.state_ticks = 2
    machine._advance_body_height_preflight(_observation(2))
    assert machine._body_height_calibrator.consecutive_sample_count == 1

    machine.state_ticks = 3
    _, events, completed = machine._advance_body_height_preflight(
        _observation(3, written_command=(1.0e-9, 0.0, 0.0))
    )

    assert completed is False
    assert machine._body_height_calibrator.consecutive_sample_count == 0
    assert machine._body_height_preflight_rejection_counts[
        "written_command_not_exact_zero"
    ] == 1
    assert events[0].name == "body_height_preflight_sample_rejected"
    assert planner.goals == []


def test_raw_joint_velocity_bias_is_audited_but_position_difference_is_gate(
    tmp_path: Path,
) -> None:
    machine, _ = _machine(tmp_path)
    machine._body_height_preflight_wait_for_post_reset_frame = False
    machine._body_height_preflight_started_state_tick = 0
    raw_velocities = (-0.133, 0.0, 0.0, 0.0, 0.074, 0.0)

    machine.state_ticks = 1
    _, events, completed = machine._advance_body_height_preflight(
        _observation(1, arm_velocities=raw_velocities)
    )
    assert completed is False
    assert events[0].metadata["reason"] == (
        "arm_joint_position_baseline_initialized"
    )
    assert events[0].metadata["raw_arm_joint_speed_exceeded"] is True
    assert events[0].metadata["derived_arm_joint_max_speed_rps"] is None

    first_positions = tuple(value + 1.0e-6 for value in ARM_STOW)
    machine.state_ticks = 2
    _, _, completed = machine._advance_body_height_preflight(
        _observation(
            2,
            arm_positions=first_positions,
            arm_velocities=raw_velocities,
        )
    )
    assert completed is False
    assert machine._body_height_calibrator is not None
    assert machine._body_height_calibrator.consecutive_sample_count == 1
    audit = machine._body_height_preflight_last_update
    assert audit["arm_joint_motion_gate_source"] == (
        "consecutive_position_over_odometry_time"
    )
    assert audit["raw_arm_joint_max_speed_rps"] == pytest.approx(0.133)
    assert audit["raw_arm_joint_speed_exceeded"] is True
    assert audit["derived_arm_joint_max_speed_rps"] == pytest.approx(1.0e-5)
    assert audit["derived_arm_joint_speed_exceeded"] is False
    assert audit["arm_joint_stow_max_error_rad"] == pytest.approx(1.0e-6)

    second_positions = tuple(value + 2.0e-6 for value in ARM_STOW)
    machine.state_ticks = 3
    _, _, completed = machine._advance_body_height_preflight(
        _observation(
            3,
            arm_positions=second_positions,
            arm_velocities=raw_velocities,
        )
    )
    assert completed is True


def test_actual_arm_position_jump_clears_calibration_window(
    tmp_path: Path,
) -> None:
    machine, _ = _machine(tmp_path)
    machine._body_height_preflight_wait_for_post_reset_frame = False
    machine._body_height_preflight_started_state_tick = 0

    machine.state_ticks = 1
    machine._advance_body_height_preflight(_observation(1))
    machine.state_ticks = 2
    machine._advance_body_height_preflight(_observation(2))
    assert machine._body_height_calibrator is not None
    assert machine._body_height_calibrator.consecutive_sample_count == 1

    jumped_positions = (0.02, *ARM_STOW[1:])
    machine.state_ticks = 3
    _, events, completed = machine._advance_body_height_preflight(
        _observation(3, arm_positions=jumped_positions)
    )

    assert completed is False
    assert machine._body_height_calibrator.consecutive_sample_count == 0
    assert events[0].metadata["reason"] == "arm_joint_speed_exceeded"
    assert events[0].metadata["derived_arm_joint_max_speed_rps"] == pytest.approx(
        0.2
    )
    assert events[0].metadata["raw_arm_joint_max_speed_rps"] == 0.0
    assert events[0].metadata["window_reset"] is True
    assert machine._body_height_preflight_arm_motion_baseline is None


def test_arm_position_difference_timestamp_faults_fail_closed(
    tmp_path: Path,
) -> None:
    machine, _ = _machine(tmp_path)
    machine._body_height_preflight_wait_for_post_reset_frame = False
    machine._body_height_preflight_started_state_tick = 0

    machine.state_ticks = 1
    machine._advance_body_height_preflight(
        _observation(1, odom_timestamp_s=1.0)
    )
    machine.state_ticks = 2
    _, rollback_events, completed = machine._advance_body_height_preflight(
        _observation(2, odom_timestamp_s=0.9)
    )
    assert completed is False
    assert rollback_events[0].metadata["reason"] == (
        "arm_joint_position_timestamp_not_strictly_increasing"
    )
    assert rollback_events[0].metadata["arm_joint_position_dt_s"] == pytest.approx(
        -0.1
    )
    assert machine._body_height_preflight_arm_motion_baseline is None

    machine.state_ticks = 3
    _, baseline_events, _ = machine._advance_body_height_preflight(
        _observation(3, odom_timestamp_s=1.1)
    )
    assert baseline_events[0].metadata["reason"] == (
        "arm_joint_position_baseline_initialized"
    )
    machine.state_ticks = 4
    _, gap_events, completed = machine._advance_body_height_preflight(
        _observation(4, odom_timestamp_s=1.3)
    )
    assert completed is False
    assert gap_events[0].metadata["reason"] == (
        "arm_joint_position_interval_exceeded"
    )
    assert gap_events[0].metadata["arm_joint_position_dt_s"] == pytest.approx(
        0.2
    )
    assert machine._body_height_calibrator is not None
    assert machine._body_height_calibrator.consecutive_sample_count == 0
    assert machine._body_height_preflight_arm_motion_baseline is None


def test_direct_joint_state_and_non_stowed_arm_remain_forbidden(
    tmp_path: Path,
) -> None:
    machine, _ = _machine(tmp_path)
    machine._body_height_preflight_wait_for_post_reset_frame = False
    machine._body_height_preflight_started_state_tick = 0

    machine.state_ticks = 1
    _, direct_events, completed = machine._advance_body_height_preflight(
        _observation(1, used_direct_joint_state=True)
    )
    assert completed is False
    assert direct_events[0].metadata["reason"] == "direct_joint_state_forbidden"
    assert machine._body_height_preflight_arm_motion_baseline is None

    non_stowed = (0.06, *ARM_STOW[1:])
    machine.state_ticks = 2
    machine._advance_body_height_preflight(
        _observation(2, arm_positions=non_stowed)
    )
    machine.state_ticks = 3
    _, stow_events, completed = machine._advance_body_height_preflight(
        _observation(3, arm_positions=non_stowed)
    )
    assert completed is False
    assert stow_events[0].metadata["reason"] == "arm_not_stowed"
    assert machine._body_height_calibrator is not None
    assert machine._body_height_calibrator.consecutive_sample_count == 0
    assert machine._body_height_preflight_arm_motion_baseline is None
