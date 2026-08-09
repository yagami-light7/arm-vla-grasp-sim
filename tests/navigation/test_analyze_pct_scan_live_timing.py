from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.navigation import analyze_pct_scan_live_timing as timing


def _frame(
    timestamp: float,
    *,
    pipeline_state: str,
    action_source: str,
    command: tuple[float, float, float],
    motion_allowed: bool,
    pose_xy: tuple[float, float],
    body_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    controller_state: int | None = None,
    controller_is_final: bool | None = None,
) -> dict[str, object]:
    """构造只含测速器必需字段的诊断帧。"""

    metadata: dict[str, object] = {
        "last_action_source": action_source,
        "body_velocity": list(body_velocity),
        "scan_cmd_vel_last_write_report": {
            "motion_allowed": motion_allowed,
            "written_command": list(command),
        },
    }
    if controller_state is not None or controller_is_final is not None:
        status: dict[str, object] = {}
        if controller_state is not None:
            status["state"] = controller_state
        if controller_is_final is not None:
            status["is_final"] = controller_is_final
        metadata["scan_controller_status_last_report"] = status
    return {
        "timestamp": timestamp,
        "pipeline_state": pipeline_state,
        "observation": {
            "robot_root_pose": [pose_xy[0], pose_xy[1], 0.3, 1.0, 0.0, 0.0, 0.0],
            "metadata": metadata,
        },
    }


def _write_run(tmp_path: Path) -> Path:
    """写入同时覆盖握手、仅转向、平移和楼梯边界的最小运行。"""

    run_dir = tmp_path / "run"
    episode_dir = run_dir / "episode_000000"
    episode_dir.mkdir(parents=True)
    frames = [
        _frame(
            0.0,
            pipeline_state="reset_episode",
            action_source="body_height_preflight",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(0.0, 0.0),
        ),
        _frame(
            1.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_pct_goal_waiting_for_path",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(0.0, 0.0),
        ),
        _frame(
            1.5,
            pipeline_state="exec_nav_to_place",
            action_source="scan_ros2_navigation",
            command=(0.0, 0.0, 0.6),
            motion_allowed=True,
            pose_xy=(0.0, 0.0),
            body_velocity=(0.0, 0.0, 0.58),
            controller_state=9,
        ),
        _frame(
            2.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_ros2_navigation",
            command=(0.4, 0.0, 0.1),
            motion_allowed=True,
            pose_xy=(0.1, 0.0),
            body_velocity=(0.3, 0.0, 0.09),
            controller_state=10,
        ),
        _frame(
            3.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_stair_sensor_acquisition_wait",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(0.5, 0.0),
        ),
    ]
    (episode_dir / "frames.jsonl").write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames),
        encoding="utf-8",
    )
    (episode_dir / "summary.json").write_text(
        json.dumps(
            {
                "seed": 7,
                "success": False,
                "failure_reason": "hardware_abort",
                "final_state": "exec_nav_to_place",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "ros2_launch.log").write_text(
        "\n".join(
            (
                "total time:\x1b[42m0.05\x1b[0m,optimize:0.03,refine:0.02",
                "final_plan_success=1",
                "total time:0.07,optimize:0.04,refine:0.03",
                "final_plan_success=1",
                "final_plan_success=0",
                "收到楼梯执行冻结：暂停 SCAN 优化",
                "total time:9.99",
                "final_plan_success=1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _write_successful_crossfloor_run(tmp_path: Path) -> Path:
    """写入可拆分F1、公共楼梯、F2、末端和严格零速门的成功运行。"""

    run_dir = tmp_path / "successful_crossfloor"
    episode_dir = run_dir / "episode_000000"
    episode_dir.mkdir(parents=True)
    frames = [
        _frame(
            0.0,
            pipeline_state="reset_episode",
            action_source="body_height_preflight",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(0.0, 0.0),
        ),
        _frame(
            1.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_pct_goal_waiting_for_path",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(0.0, 0.0),
        ),
        _frame(
            1.5,
            pipeline_state="exec_nav_to_place",
            action_source="scan_ros2_navigation",
            command=(0.0, 0.0, 0.6),
            motion_allowed=True,
            pose_xy=(0.0, 0.0),
            controller_state=9,
            controller_is_final=False,
        ),
        _frame(
            2.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_ros2_navigation",
            command=(0.4, 0.0, 0.1),
            motion_allowed=True,
            pose_xy=(0.2, 0.0),
            controller_state=10,
            controller_is_final=False,
        ),
        _frame(
            3.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_stair_sensor_acquisition_wait",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(0.6, 0.0),
        ),
        _frame(
            4.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_stair_freeze_active",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(1.0, 0.0),
        ),
        _frame(
            5.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_ros2_navigation",
            command=(0.5, 0.0, 0.0),
            motion_allowed=True,
            pose_xy=(1.4, 0.0),
            body_velocity=(0.45, 0.0, 0.0),
            controller_state=10,
            controller_is_final=False,
        ),
        _frame(
            6.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_ros2_navigation",
            command=(0.2, 0.0, 0.1),
            motion_allowed=True,
            pose_xy=(1.85, 0.0),
            body_velocity=(0.18, 0.0, 0.08),
            controller_state=10,
            controller_is_final=True,
        ),
        _frame(
            7.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_ros2_navigation",
            command=(0.0, 0.0, 0.3),
            motion_allowed=True,
            pose_xy=(2.0, 0.0),
            body_velocity=(0.0, 0.0, 0.25),
            controller_state=9,
            controller_is_final=True,
        ),
        _frame(
            8.0,
            pipeline_state="exec_nav_to_place",
            action_source="scan_ros2_navigation",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(2.0, 0.0),
            controller_state=11,
            controller_is_final=True,
        ),
        _frame(
            8.2,
            pipeline_state="done",
            action_source="goal_reached_hold",
            command=(0.0, 0.0, 0.0),
            motion_allowed=False,
            pose_xy=(2.0, 0.0),
        ),
    ]
    (episode_dir / "frames.jsonl").write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames),
        encoding="utf-8",
    )
    (episode_dir / "summary.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "success": True,
                "failure_reason": "",
                "final_state": "done",
                "execution_mode": "navigation_carry_smoke",
                "latest_executor_status": {
                    "done": True,
                    "success": True,
                    "goal_rising_edge_seen": True,
                    "scan_controller_goal_reached_verified": True,
                    "policy_zero_hold_verified": True,
                    "post_goal_nonzero_write_count": 0,
                    "goal_true_receipt_timestamp": 8.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "ros2_launch.log").write_text(
        "\n".join(
            (
                "total time:0.05,optimize:0.03,refine:0.02",
                "final_plan_success=1",
                "final_plan_success=0",
                "收到楼梯执行冻结：暂停 SCAN 优化",
                "楼梯执行冻结解除：等待新传感器",
                "total time:0.08,optimize:0.05,refine:0.03",
                "final_plan_success=1",
                "total time:0.04,optimize:0.02,refine:0.02",
                "final_plan_success=1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_analyze_episode_splits_f1_commands_and_planner_prefix(tmp_path: Path) -> None:
    report = timing.analyze_episode(
        _write_run(tmp_path),
        configured_max_yaw_rate_radps=0.6,
    )

    assert report["seed"] == 7
    assert report["failure_reason"] == "hardware_abort"
    f1 = report["f1"]
    assert f1["navigation_stage_start_s"] == pytest.approx(1.0)
    assert f1["scan_control_start_s"] == pytest.approx(1.5)
    assert f1["stair_takeover_start_s"] == pytest.approx(3.0)
    assert f1["navigation_handshake_s"] == pytest.approx(0.5)
    assert f1["f1_total_from_navigation_stage_s"] == pytest.approx(2.0)
    assert f1["f1_scan_control_s"] == pytest.approx(1.5)
    assert f1["yaw_only_command_s"] == pytest.approx(0.5)
    assert f1["translation_command_s"] == pytest.approx(1.0)
    assert f1["zero_command_s"] == pytest.approx(0.0)
    assert f1["controller_aligning_state_s"] == pytest.approx(0.5)
    assert f1["near_yaw_limit_s"] == pytest.approx(0.5)
    assert f1["mean_command_planar_speed_while_translating_mps"] == pytest.approx(0.4)
    assert f1["mean_measured_planar_speed_while_translating_mps"] == pytest.approx(0.3)
    assert f1["sampled_xy_travel_m"] == pytest.approx(0.5)
    assert f1["sampled_xy_net_displacement_m"] == pytest.approx(0.5)
    assert f1["diagnostic_maximum_gap_s"] == pytest.approx(1.0)

    planner = report["planner_before_first_stair_freeze"]
    assert planner["attempt_count"] == 3
    assert planner["success_count"] == 2
    assert planner["failure_count"] == 1
    assert planner["successful_total_wall_s"] == pytest.approx(0.12)
    assert planner["successful_mean_wall_s"] == pytest.approx(0.06)
    assert planner["successful_p95_wall_s"] == pytest.approx(0.07)


def test_compare_reports_uses_first_run_as_baseline(tmp_path: Path) -> None:
    baseline = timing.analyze_episode(
        _write_run(tmp_path),
        configured_max_yaw_rate_radps=0.6,
    )
    candidate = json.loads(json.dumps(baseline))
    candidate["run_dir"] = "/candidate"
    candidate["f1"]["f1_scan_control_s"] = 1.2
    candidate["f1"]["yaw_only_command_s"] = 0.3

    comparison = timing.compare_reports([baseline, candidate])[0]

    assert comparison["f1_scan_control_delta_s"] == pytest.approx(-0.3)
    assert comparison["f1_scan_control_reduction_ratio"] == pytest.approx(0.2)
    assert comparison["yaw_only_delta_s"] == pytest.approx(-0.2)
    assert comparison["yaw_only_reduction_ratio"] == pytest.approx(0.4)
    assert comparison["full_crossfloor_comparison_available"] is False


def test_successful_crossfloor_splits_common_stair_f2_and_terminal(
    tmp_path: Path,
) -> None:
    report = timing.analyze_episode(
        _write_successful_crossfloor_run(tmp_path),
        configured_max_yaw_rate_radps=0.6,
    )

    assert report["schema"] == "pct_scan_live_timing_v2"
    assert report["crossfloor_unavailable_reason"] is None
    crossfloor = report["crossfloor"]
    assert crossfloor["navigation_handshake_s"] == pytest.approx(0.5)
    assert crossfloor["full_navigation_stage_sim_time_s"] == pytest.approx(7.0)
    assert crossfloor["common_stair_freeze_sim_time_s"] == pytest.approx(2.0)
    assert crossfloor["f1_scan_control_sim_time_s"] == pytest.approx(1.5)
    assert crossfloor["f2_scan_control_sim_time_s"] == pytest.approx(3.0)
    assert crossfloor["planner_controlled_navigation_sim_time_s"] == pytest.approx(
        4.5
    )
    assert crossfloor["f2_preterminal_sim_time_s"] == pytest.approx(1.0)
    assert crossfloor["terminal_capture_sim_time_s"] == pytest.approx(2.0)
    assert crossfloor["translation_command_sim_time_s"] == pytest.approx(3.0)
    assert crossfloor["yaw_only_command_sim_time_s"] == pytest.approx(1.5)
    assert crossfloor["zero_command_sim_time_s"] == pytest.approx(0.0)
    assert crossfloor["primary_metric_excludes_common_stair_freeze"] is True
    assert report["scope"]["full_crossfloor_available"] is True
    planner = report["planner_crossfloor"]
    assert planner["freeze_event_count"] == 1
    assert planner["release_event_count"] == 1
    assert planner["pre_stair"]["attempt_count"] == 2
    assert planner["pre_stair"]["success_count"] == 1
    assert planner["pre_stair"]["failure_count"] == 1
    assert planner["post_stair"]["success_count"] == 2
    assert planner["successful_total_wall_s"] == pytest.approx(0.17)


def test_compare_reports_adds_full_crossfloor_primary_metric(tmp_path: Path) -> None:
    baseline = timing.analyze_episode(
        _write_successful_crossfloor_run(tmp_path),
        configured_max_yaw_rate_radps=0.6,
    )
    candidate = json.loads(json.dumps(baseline))
    candidate["run_dir"] = "/candidate"
    candidate["crossfloor"][
        "planner_controlled_navigation_sim_time_s"
    ] = 3.6
    candidate["crossfloor"]["full_navigation_stage_sim_time_s"] = 6.0

    comparison = timing.compare_reports([baseline, candidate])[0]

    assert comparison["full_crossfloor_comparison_available"] is True
    assert comparison["planner_controlled_navigation_delta_s"] == pytest.approx(-0.9)
    assert comparison[
        "planner_controlled_navigation_reduction_ratio"
    ] == pytest.approx(0.2)
    assert comparison["full_navigation_stage_delta_s"] == pytest.approx(-1.0)


def test_crossfloor_success_requires_final_trajectory_evidence(
    tmp_path: Path,
) -> None:
    run_dir = _write_successful_crossfloor_run(tmp_path)
    frames_path = run_dir / "episode_000000/frames.jsonl"
    frames = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
    ]
    for frame in frames:
        status = frame["observation"]["metadata"].get(
            "scan_controller_status_last_report"
        )
        if isinstance(status, dict):
            status["is_final"] = False
    frames_path.write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames),
        encoding="utf-8",
    )

    with pytest.raises(timing.TimingInputError, match="is_final=true"):
        timing.analyze_episode(
            run_dir,
            configured_max_yaw_rate_radps=0.6,
        )


def test_analyze_episode_prefers_immutable_tuning_snapshot(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    (run_dir / "pct_scan_live_acceptance.json").write_text(
        json.dumps(
            {
                "tuning_config_snapshot": {
                    "selected_parameters": {
                        "scan_controller.limits.max_yaw_rate": 0.75,
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = timing.analyze_episode(run_dir)

    assert report["configured_max_yaw_rate_provenance"] == (
        "immutable_run_tuning_snapshot"
    )
    assert report["f1"]["configured_max_yaw_rate_radps"] == pytest.approx(0.75)
    assert report["f1"]["near_yaw_limit_threshold_radps"] == pytest.approx(0.73)


def test_load_frame_samples_allows_same_tick_state_transitions(tmp_path: Path) -> None:
    path = tmp_path / "frames.jsonl"
    frame = _frame(
        1.0,
        pipeline_state="exec_nav_to_place",
        action_source="scan_ros2_navigation",
        command=(0.1, 0.0, 0.0),
        motion_allowed=True,
        pose_xy=(0.0, 0.0),
    )
    path.write_text(
        json.dumps(frame) + "\n" + json.dumps(frame) + "\n",
        encoding="utf-8",
    )

    samples = timing.load_frame_samples(path)

    assert len(samples) == 2
    assert samples[0].timestamp_s == samples[1].timestamp_s == pytest.approx(1.0)


def test_load_frame_samples_rejects_time_rollback(tmp_path: Path) -> None:
    path = tmp_path / "frames.jsonl"
    first = _frame(
        1.0,
        pipeline_state="exec_nav_to_place",
        action_source="scan_ros2_navigation",
        command=(0.1, 0.0, 0.0),
        motion_allowed=True,
        pose_xy=(0.0, 0.0),
    )
    second = _frame(
        0.9,
        pipeline_state="exec_nav_to_place",
        action_source="scan_ros2_navigation",
        command=(0.1, 0.0, 0.0),
        motion_allowed=True,
        pose_xy=(0.0, 0.0),
    )
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(timing.TimingInputError, match="不得回退"):
        timing.load_frame_samples(path)


def test_analyze_planner_prefix_requires_stair_boundary(tmp_path: Path) -> None:
    log_path = tmp_path / "ros2_launch.log"
    log_path.write_text(
        "total time:0.05\nfinal_plan_success=1\n",
        encoding="utf-8",
    )

    with pytest.raises(timing.TimingInputError, match="楼梯冻结边界"):
        timing.analyze_planner_prefix(log_path)
