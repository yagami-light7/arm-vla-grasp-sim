"""固定命令楼梯 probe 的同拍低层遥测测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.interfaces import RobotAction, SimulationState, StepRecord
from source.navigation.adapters.isaaclab_go2_adapter import (
    DOG_JOINT_NAMES,
    Go2LocomotionAdapter,
)
from source.recording.jsonl_recorder import JsonlEpisodeRecorder
from source.simulation.isaaclab_runtime import IsaacLabNavigationRuntime


def _grid_pattern(cfg, device: str):
    """测试用 GridPattern；顺序与 cfg 决定，不把前区索引写死。"""

    torch = pytest.importorskip("torch")
    x = torch.arange(
        -cfg.size[0] / 2,
        cfg.size[0] / 2 + 1.0e-9,
        cfg.resolution,
        device=device,
    )
    y = torch.arange(
        -cfg.size[1] / 2,
        cfg.size[1] / 2 + 1.0e-9,
        cfg.resolution,
        device=device,
    )
    indexing = cfg.ordering if cfg.ordering == "xy" else "ij"
    grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)
    starts = torch.zeros((grid_x.numel(), 3), device=device)
    starts[:, 0] = grid_x.flatten()
    starts[:, 1] = grid_y.flatten()
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    return starts, directions


def _probe_adapter():
    torch = pytest.importorskip("torch")

    class ObservationManager:
        active_terms = {
            "policy": ["base_state", "height_scan", "velocity_commands"]
        }
        group_obs_term_dim = {"policy": [(2,), (6,), (3,)]}
        group_obs_concatenate = {"policy": True}

    pattern_cfg = SimpleNamespace(
        func=_grid_pattern,
        size=(0.4, 0.2),
        resolution=0.2,
        ordering="yx",
    )
    ray_hits = torch.zeros((1, 6, 3), dtype=torch.float32)
    ray_hits[0, 2, :] = float("inf")
    height_scanner = SimpleNamespace(
        cfg=SimpleNamespace(pattern_cfg=pattern_cfg),
        data=SimpleNamespace(ray_hits_w=ray_hits),
    )
    runtime = SimpleNamespace(
        observation_manager=ObservationManager(),
        scene=SimpleNamespace(sensors={"height_scanner": height_scanner}),
    )
    policy_observation = torch.tensor(
        [[9.0, 8.0, -1.0, -0.5, -1.0, 0.2, 0.3, -1.0, 0.25, 0.0, 0.0]],
        dtype=torch.float32,
    )

    class Env:
        clip_actions = None

        def get_observations(self):
            return {"policy": policy_observation}

    class BaseCommandTerm:
        device = torch.device("cpu")
        vel_command_b = torch.zeros((1, 3), dtype=torch.float32)

    adapter = object.__new__(Go2LocomotionAdapter)
    adapter.runtime = runtime
    adapter.env = Env()
    adapter.policy = lambda _observations: torch.arange(
        12, dtype=torch.float32
    ).reshape(1, 12)
    adapter.observations = {}
    adapter.base_cmd_term = BaseCommandTerm()
    adapter.arm_term = None
    adapter.joint_pos_action_term = None
    adapter.dog_action_indices = list(range(12))
    adapter.arm_action_indices = None
    adapter.direct_arm_action_override = False
    adapter.gripper_joint_ids = ()
    adapter._base_pose_lock_xyzyaw = None
    adapter._dog_joint_lock_target = None
    adapter.standing_command_threshold = 0.0
    adapter.policy_action_warmup_steps = 0
    adapter._policy_action_step = 0
    adapter._policy_action_warmup_scale = 1.0
    adapter._command = (0.25, 0.0, 0.0)
    adapter._effective_command = (0.0, 0.0, 0.0)
    adapter._command_is_standing = False
    adapter._arm_joint_target = None
    adapter._gripper_joint_target = None
    adapter._last_actions = None
    adapter._last_stair_probe_policy_pre_step = None
    return adapter


def test_probe_pre_step_uses_manager_layout_and_captures_command_action_height() -> None:
    adapter = _probe_adapter()

    actions = adapter.compute_policy_action(
        refresh_observations=True,
        capture_stair_probe_telemetry=True,
    )
    report = adapter.get_stair_probe_policy_pre_step()

    assert report["available"] is True
    command = report["command_buffer"]
    assert command["written_effective_command"] == pytest.approx([0.25, 0.0, 0.0])
    assert command["command_buffer_readback"] == pytest.approx([0.25, 0.0, 0.0])
    assert command["write_readback_match"] is True
    assert command["policy_observation_matches_command_buffer"] is True

    terms = report["policy_observation"]["selected_terms"]
    assert terms["height_scan"]["policy_flat_index_start"] == 2
    assert terms["height_scan"]["policy_flat_index_end_exclusive"] == 8
    assert terms["velocity_commands"]["policy_flat_index_start"] == 8
    assert terms["velocity_commands"]["values"] == pytest.approx([0.25, 0.0, 0.0])
    assert terms["height_scan"]["statistics"] == pytest.approx(
        {
            "value_count": 6,
            "finite_count": 6,
            "nonfinite_count": 0,
            "min": -1.0,
            "max": 0.3,
            "mean": -0.5,
        }
    )
    assert terms["height_scan"]["clip_diagnostics"]["clipped_low_count"] == 3
    assert terms["height_scan"]["clip_diagnostics"]["clipped_low_ratio"] == 0.5
    front = terms["height_scan"]["front_subset"]
    assert front["available"] is True
    assert front["relative_height_scan_indices"] == [2, 3, 4, 5]
    assert front["policy_flat_indices"] == [4, 5, 6, 7]
    assert front["values"] == pytest.approx([-1.0, 0.2, 0.3, -1.0])
    assert front["clip_diagnostics"]["clipped_low_count"] == 2
    assert front["front_ray_miss_count"] == 1

    dog_action = report["dog_action"]
    assert dog_action["dog_joint_names"] == DOG_JOINT_NAMES
    assert dog_action["submitted_dog_action"] == pytest.approx(
        [float(index) for index in range(12)]
    )
    assert actions.shape == (1, 12)


def test_probe_pre_step_degrades_without_observation_manager() -> None:
    adapter = _probe_adapter()
    adapter.runtime = SimpleNamespace()

    actions = adapter.compute_policy_action(
        refresh_observations=True,
        capture_stair_probe_telemetry=True,
    )
    report = adapter.get_stair_probe_policy_pre_step()

    assert actions.shape == (1, 12)
    assert report["available"] is False
    assert report["policy_observation"]["available"] is False
    assert "observation_manager_layout_unavailable" in report[
        "policy_observation"
    ]["unavailable_reason"]


def test_probe_post_step_captures_torque_contacts_and_foot_states() -> None:
    torch = pytest.importorskip("torch")
    adapter = object.__new__(Go2LocomotionAdapter)
    body_names = ["base", "FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    robot_data = SimpleNamespace(
        root_pos_w=torch.tensor([[1.0, 2.0, 0.4]]),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_lin_vel_b=torch.tensor([[0.1, 0.0, 0.0]]),
        root_ang_vel_b=torch.tensor([[0.0, 0.0, 0.2]]),
        joint_pos=torch.arange(12, dtype=torch.float32).reshape(1, 12),
        joint_vel=torch.ones((1, 12), dtype=torch.float32),
        applied_torque=torch.arange(12, dtype=torch.float32).reshape(1, 12) + 10.0,
        joint_pos_target=torch.arange(12, dtype=torch.float32).reshape(1, 12) + 20.0,
        body_pos_w=torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        body_lin_vel_w=torch.ones((1, 5, 3), dtype=torch.float32),
    )
    adapter.robot = SimpleNamespace(body_names=body_names, data=robot_data)
    adapter.dog_joint_ids = list(range(12))
    forces = torch.zeros((1, 5, 3), dtype=torch.float32)
    forces[0, 1, 2] = 120.0
    forces[0, 2, 2] = 80.0
    contact_sensor = SimpleNamespace(
        body_names=body_names,
        data=SimpleNamespace(
            net_forces_w=forces,
            current_air_time=torch.tensor([[0.0, 0.0, 0.1, 0.2, 0.0]]),
            current_contact_time=torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.1]]),
        ),
    )
    adapter.runtime = SimpleNamespace(
        scene=SimpleNamespace(sensors={"contact_forces": contact_sensor})
    )

    report = adapter.capture_stair_probe_post_step()

    assert report["available"] is True
    dog_state = report["dog_joint_state"]
    assert dog_state["applied_torque_available"] is True
    assert dog_state["applied_torque_nm"] == pytest.approx(
        [float(index + 10) for index in range(12)]
    )
    assert dog_state["position_targets_rad"] == pytest.approx(
        [float(index + 20) for index in range(12)]
    )
    contacts = report["contacts"]
    assert contacts["foot_contact_force_max_n"] == pytest.approx(120.0)
    assert len(contacts["foot_contacts"]) == 4
    assert contacts["foot_contacts"][1]["current_air_time_s"] == pytest.approx(0.1)
    assert contacts["foot_states"]["available"] is True
    assert len(contacts["foot_states"]["feet"]) == 4


def test_probe_post_step_reports_required_component_unavailable() -> None:
    """子组件缺失时不能只凭 root 状态把整份 post 遥测标为可用。"""

    torch = pytest.importorskip("torch")
    adapter = object.__new__(Go2LocomotionAdapter)
    adapter.get_base_pose_full = lambda: {
        "x": 0.0,
        "y": 0.0,
        "z": 0.3,
        "yaw": 0.0,
        "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
    }
    adapter.get_base_velocity_full = lambda: (0.0, 0.0, 0.0)
    adapter.dog_joint_ids = list(range(12))
    adapter.robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros((1, 12)),
            joint_vel=torch.zeros((1, 12)),
            applied_torque=torch.zeros((1, 12)),
            joint_pos_target=torch.zeros((1, 12)),
        )
    )
    adapter.runtime = SimpleNamespace(scene=SimpleNamespace(sensors={}))

    report = adapter.capture_stair_probe_post_step()

    assert report["available"] is False
    assert report["unavailable_reason"] == "required_post_step_component_unavailable"
    assert report["component_availability"]["root_state"] is True
    assert report["component_availability"]["dog_joint_state"] is True
    assert report["component_availability"]["contacts"] is False
    assert report["component_availability"]["foot_states"] is False


def test_runtime_alignment_binds_pre_action_to_post_contact_pose() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._runtime = SimpleNamespace(  # type: ignore[attr-defined]
        step_dt=0.02,
        cfg=SimpleNamespace(decimation=8),
    )
    runtime._adapter = SimpleNamespace(  # type: ignore[attr-defined]
        get_stair_probe_policy_pre_step=lambda: {
            "available": True,
            "dog_action": {"submitted_dog_action": [0.0] * 12},
        },
        capture_stair_probe_post_step=lambda: {
            "available": True,
            "contacts": {"foot_contact_force_max_n": 123.0},
        },
    )
    runtime._step_calls = 7  # type: ignore[attr-defined]
    runtime._metadata = {}  # type: ignore[attr-defined]
    action = RobotAction(
        base_velocity=(0.25, 0.0, 0.0),
        source="stair_fixed_command_probe",
        metadata={
            "stair_fixed_command_probe": True,
            "stair_probe_phase": "driving",
        },
    )

    runtime._begin_stair_probe_low_level_telemetry(action)
    runtime._complete_stair_probe_low_level_telemetry(
        completed_control_step=8,
    )
    report = runtime._metadata["stair_probe_low_level_telemetry"]  # type: ignore[attr-defined]

    assert report["complete"] is True
    assert report["available"] is True
    assert report["alignment"]["pre_step_state_step_index"] == 7
    assert report["alignment"]["post_step_state_step_index"] == 8
    assert report["alignment"]["pre_step_timestamp_s"] == pytest.approx(0.14)
    assert report["alignment"]["post_step_timestamp_s"] == pytest.approx(0.16)
    assert report["alignment"]["physics_substep_count"] == 8
    assert report["post_step"]["contacts"]["foot_contact_force_max_n"] == 123.0


def _state(step_index: int, metadata: dict | None = None) -> SimulationState:
    return SimulationState(
        step_index=step_index,
        timestamp=step_index * 0.02,
        robot_root_pose=(0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0,) * 6,
        metadata=metadata or {},
    )


def test_jsonl_recorder_writes_probe_telemetry_only_on_matching_action(
    tmp_path: Path,
) -> None:
    telemetry = {
        "schema": "stair_fixed_command_probe_low_level_v1",
        "complete": True,
        "alignment": {
            "pre_step_state_step_index": 4,
            "post_step_state_step_index": 5,
        },
        "pre_step": {"dog_action": {"submitted_dog_action": [0.0] * 12}},
        "post_step": {"contacts": {"foot_contact_force_max_n": 42.0}},
    }
    recorder = JsonlEpisodeRecorder(tmp_path / "episode")
    recorder.record_step(
        StepRecord(
            step_index=4,
            timestamp=0.08,
            pipeline_state="exec_nav_to_pick",
            observation=_state(4),
            action=RobotAction(
                base_velocity=(0.25, 0.0, 0.0),
                source="stair_fixed_command_probe",
                metadata={"stair_fixed_command_probe": True},
            ),
            post_step_observation=_state(
                5,
                {"stair_probe_low_level_telemetry": telemetry},
            ),
        )
    )

    payload = json.loads(recorder.frames_path.read_text(encoding="utf-8"))
    assert payload["stair_probe_low_level_telemetry"] == telemetry
    assert "stair_probe_low_level_telemetry" not in payload["observation"]["metadata"]
    assert "stair_probe_low_level_telemetry" not in payload[
        "post_step_observation"
    ]["metadata"]


def test_jsonl_recorder_sanitizes_nonfinite_probe_values(tmp_path: Path) -> None:
    """遥测原值出现 NaN/Inf 时仍必须生成严格 JSON。"""

    telemetry = {
        "schema": "stair_fixed_command_probe_low_level_v1",
        "complete": True,
        "alignment": {
            "pre_step_state_step_index": 4,
            "post_step_state_step_index": 5,
        },
        "raw_values": [float("nan"), float("inf"), float("-inf")],
    }
    recorder = JsonlEpisodeRecorder(tmp_path / "episode")
    recorder.record_step(
        StepRecord(
            step_index=4,
            timestamp=0.08,
            pipeline_state="exec_nav_to_pick",
            observation=_state(4),
            action=RobotAction(
                base_velocity=(0.25, 0.0, 0.0),
                source="stair_fixed_command_probe",
                metadata={"stair_fixed_command_probe": True},
            ),
            post_step_observation=_state(
                5,
                {"stair_probe_low_level_telemetry": telemetry},
            ),
        )
    )

    raw = recorder.frames_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    payload = json.loads(raw, parse_constant=lambda value: pytest.fail(value))
    assert payload["stair_probe_low_level_telemetry"]["raw_values"] == [
        None,
        None,
        None,
    ]


def test_runtime_clears_probe_metadata_before_nonprobe_skip_action() -> None:
    """即使下一条动作不推进物理，也不能继承上一 probe 的大字段。"""

    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._require_ready = lambda: None
    runtime._metadata = {"stair_probe_low_level_telemetry": {"complete": True}}
    runtime._pending_stair_probe_low_level_telemetry = {"complete": True}

    runtime.apply(
        RobotAction(
            source="verify_pick_reachable",
            metadata={"skip_physics_step": True, "skip_reason": "state_transition"},
        )
    )

    assert "stair_probe_low_level_telemetry" not in runtime._metadata
    assert runtime._pending_stair_probe_low_level_telemetry is None
