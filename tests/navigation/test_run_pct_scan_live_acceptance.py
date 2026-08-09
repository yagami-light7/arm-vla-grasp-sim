from __future__ import annotations

from dataclasses import replace
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.navigation import run_pct_scan_live_acceptance as acceptance


class _FakeProcess:
    """实现编排单测所需的最小 Popen 行为。"""

    def __init__(
        self,
        *,
        pid: int,
        kind: str,
        return_code: int = 0,
        on_pipeline_wait: Any = None,
        early_exit: bool = False,
        pipeline_wait_times_out: bool = False,
    ) -> None:
        self.pid = pid
        self.kind = kind
        self._planned_return_code = return_code
        self.returncode: int | None = return_code if early_exit else None
        self.on_pipeline_wait = on_pipeline_wait
        self.pipeline_wait_times_out = pipeline_wait_times_out
        self.signals: list[int] = []
        self.terminate_count = 0
        self.kill_count = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.kind == "pipeline" and not self.pipeline_wait_times_out:
            if self.on_pipeline_wait is not None:
                self.on_pipeline_wait()
            self.returncode = self._planned_return_code
            return self.returncode
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.kind, timeout)
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        if self.kind == "launch":
            self.returncode = 0

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -signal.SIGKILL


def _config(tmp_path: Path, mode: acceptance.AcceptanceMode) -> acceptance.AcceptanceConfig:
    return acceptance.AcceptanceConfig(
        mode=mode,
        output_dir=tmp_path / f"fresh_{mode}",
        ros_domain_id=217,
        isaac_python=Path(sys.executable).resolve(),
        navigation_body_height_m=acceptance.DEFAULT_NAVIGATION_BODY_HEIGHT_M,
        pipeline_timeout_s=3.0,
        launch_ready_timeout_s=1.0,
        launch_stop_timeout_s=1.0,
        graph_cleanup_timeout_s=1.0,
        require_cuda_preflight=False,
    )


@pytest.fixture(autouse=True)
def _use_small_source_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """单元测试只哈希编排器自身，避免反复扫描整套运行源码。"""

    monkeypatch.setattr(
        acceptance,
        "SOURCE_BUNDLE_ROOTS",
        (Path(acceptance.__file__).resolve(),),
    )


def _write_startup(config: acceptance.AcceptanceConfig, *, status: str = "completed", exit_code: int = 0) -> None:
    expected_mode = acceptance.MODE_SPECS[
        config.mode
    ].pipeline_mode_flag.removeprefix("--").replace("-", "_")
    (config.output_dir / "startup_status.json").write_text(
        json.dumps(
            {
                "status": status,
                "exit_code": exit_code,
                "mode": expected_mode,
                "navigation_body_height_m": config.navigation_body_height_m,
                "scan_stair_freeze_profile_runtime": (
                    acceptance.production_scan_stair_freeze_profile().audit_report()
                ),
            }
        ),
        encoding="utf-8",
    )
    episode_dir = config.output_dir / "episode_000000"
    episode_dir.mkdir(exist_ok=True)
    (episode_dir / "summary.json").write_text(
        json.dumps({"seed": config.seed}) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("mode", "task_name", "task_id", "mode_flag"),
    [
        (
            "static_stair",
            "nav_pick_place_apple_multifloor_pct.json",
            1002,
            "--stair-locomotion-smoke",
        ),
        (
            "flat_policy",
            "nav_pick_place_apple_multifloor_pct.json",
            1002,
            "--navigation-smoke",
        ),
        (
            "crossfloor_carry",
            "nav_pick_place_apple_multifloor_pct.json",
            1002,
            "--navigation-carry-smoke",
        ),
        (
            "dynamic_f1",
            "nav_smoke_scan_multifloor_dynamic_cart_f1.json",
            17704,
            "--navigation-smoke",
        ),
        (
            "dynamic_replan_f1",
            "nav_smoke_scan_multifloor_dynamic_blocker_replan_f1.json",
            17705,
            "--navigation-smoke",
        ),
    ],
)
def test_mode_specs_bind_fixed_tasks_and_pipeline_modes(
    tmp_path: Path,
    mode: acceptance.AcceptanceMode,
    task_name: str,
    task_id: int,
    mode_flag: str,
) -> None:
    config = _config(tmp_path, mode)
    spec = acceptance.MODE_SPECS[mode]
    command = acceptance.build_pipeline_command(config)
    launch_command = acceptance.build_launch_command(config)

    assert spec.task_path.name == task_name
    assert spec.expected_task_id == task_id
    assert mode_flag in command
    assert command[command.index("--task-json") + 1] == str(spec.task_path)
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--ros2-domain-id") + 1] == "217"
    assert command[command.index("--output-dir") + 1] == str(config.output_dir)
    assert "--enable-navigation-ros2-bridge" in command
    assert "--no-pct-stair-float" in command
    assert command[command.index("--navigation-body-height-m") + 1] == "0.338"
    assert command[command.index("--pct-scan-tuning-config") + 1] == str(
        config.tuning_config_file
    )
    assert command[command.index("--diagnostic-frame-stride") + 1] == "10"
    assert "body_height_m:=0.338" in launch_command
    assert f"tuning_config_file:={config.tuning_config_file}" in launch_command
    acceptance.validate_body_height_command_contract(
        config,
        launch_command=launch_command,
        pipeline_command=command,
    )


@pytest.mark.parametrize(
    "mode",
    ["flat_policy", "crossfloor_carry", "dynamic_f1", "dynamic_replan_f1"],
)
def test_navigation_live_tasks_keep_extended_state_budget(
    mode: acceptance.AcceptanceMode,
) -> None:
    """长距离 live 任务不能静默退回默认 100 秒导航预算。"""

    task = json.loads(
        acceptance.MODE_SPECS[mode].task_path.read_text(encoding="utf-8")
    )

    assert task["navigation_execution"]["extended_state_limits"] is True


@pytest.mark.parametrize(
    "mode",
    [
        "static_stair",
        "flat_policy",
        "crossfloor_carry",
        "dynamic_f1",
        "dynamic_replan_f1",
    ],
)
def test_live_modes_do_not_require_optional_active_sensing(
    tmp_path: Path,
    mode: acceptance.AcceptanceMode,
) -> None:
    command = acceptance.build_validator_command(_config(tmp_path, mode))

    assert "--require-active-sensing" not in command


def test_ros_and_isaac_process_environments_split_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROS launch 保留 rclpy 路径，Isaac Python 必须拒绝跨 ABI 路径。"""

    config = _config(tmp_path, "static_stair")
    monkeypatch.setenv(
        "PYTHONPATH",
        "/opt/ros/humble/lib/python3.10/site-packages",
    )
    monkeypatch.setenv("AMENT_PREFIX_PATH", "/tmp/test_ros_overlay")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/test_ros_overlay/lib")

    ros_environment = acceptance.build_process_environment(config)
    isaac_environment = acceptance.build_isaac_process_environment(config)

    assert ros_environment["PYTHONPATH"].endswith("python3.10/site-packages")
    assert "PYTHONPATH" not in isaac_environment
    assert isaac_environment["AMENT_PREFIX_PATH"] == ros_environment[
        "AMENT_PREFIX_PATH"
    ]
    assert isaac_environment["LD_LIBRARY_PATH"] == ros_environment[
        "LD_LIBRARY_PATH"
    ]
    assert isaac_environment["ROS_DOMAIN_ID"] == "217"


def test_ros_workspace_overlay_is_loaded_without_caller_source(
    tmp_path: Path,
) -> None:
    """fresh runner 必须自行加载当前 install，不能依赖调用终端状态。"""

    setup_path = tmp_path / "ros2_ws" / "install" / "setup.bash"
    setup_path.parent.mkdir(parents=True)
    setup_path.write_text("# test overlay\n", encoding="utf-8")
    package_prefix = setup_path.parent / "isaac_navigation_bridge"

    def fake_runner(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[-1] == str(setup_path.resolve())
        assert kwargs["env"]["BASE_SENTINEL"] == "kept"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                f"AMENT_PREFIX_PATH={package_prefix}\0"
                f"LD_LIBRARY_PATH={package_prefix / 'lib'}\0"
                "BASE_SENTINEL=kept\0"
            ),
            stderr="",
        )

    environment = acceptance.load_ros_workspace_environment(
        base_environment={"BASE_SENTINEL": "kept"},
        setup_path=setup_path,
        command_runner=fake_runner,
    )

    assert environment["BASE_SENTINEL"] == "kept"
    assert environment["AMENT_PREFIX_PATH"] == str(package_prefix)


def test_ros_workspace_overlay_rejects_unrelated_ament_prefix(
    tmp_path: Path,
) -> None:
    """source 返回成功也必须证明当前 worktree install 真正进入环境。"""

    setup_path = tmp_path / "ros2_ws" / "install" / "setup.bash"
    setup_path.parent.mkdir(parents=True)
    setup_path.write_text("# test overlay\n", encoding="utf-8")

    def fake_runner(command: tuple[str, ...], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="AMENT_PREFIX_PATH=/opt/ros/humble\0",
            stderr="",
        )

    with pytest.raises(acceptance.AcceptanceError, match="AMENT_PREFIX_PATH"):
        acceptance.load_ros_workspace_environment(
            setup_path=setup_path,
            command_runner=fake_runner,
        )


def test_cuda_preflight_runs_before_ros_and_output_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只有同一 Isaac Python 真正完成 CUDA tensor 运算后才检查 ROS 图。"""

    config = replace(
        _config(tmp_path, "flat_policy"),
        require_cuda_preflight=True,
    )
    monkeypatch.setattr(
        acceptance.shutil,
        "which",
        lambda _: "/opt/ros/humble/bin/ros2",
    )
    events: list[str] = []

    def cuda_runner(
        command: tuple[str, ...],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        assert command == acceptance.build_cuda_preflight_command(config)
        assert kwargs["timeout"] == acceptance.CUDA_PREFLIGHT_TIMEOUT_S
        assert "PYTHONPATH" not in kwargs["env"]
        assert not config.output_dir.exists()
        events.append("cuda_preflight_passed")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "torch_version": "2.7.0+cu128",
                    "torch_cuda_version": "12.8",
                    "cuda_available": True,
                    "cuda_device_count": 1,
                    "cuda_device_name": "test_gpu",
                    "cuda_capability": [8, 9],
                    "tensor_device": "cuda:0",
                    "tensor_value": 1.0,
                }
            ),
            stderr="",
        )

    def node_lister(_: Any) -> set[str]:
        assert events == ["cuda_preflight_passed"]
        events.append("ros_domain_checked")
        return {"/foreign_node"}

    with pytest.raises(acceptance.AcceptanceError, match="不是空 domain"):
        acceptance.run_acceptance(
            config,
            cuda_preflight_runner=cuda_runner,
            node_lister=node_lister,
        )

    assert events == ["cuda_preflight_passed", "ros_domain_checked"]
    assert not config.output_dir.exists()


def test_cuda_preflight_failure_stops_before_ros_and_output_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _config(tmp_path, "flat_policy"),
        require_cuda_preflight=True,
    )
    monkeypatch.setattr(
        acceptance.shutil,
        "which",
        lambda _: "/opt/ros/humble/bin/ros2",
    )
    node_called = False

    def node_lister(_: Any) -> set[str]:
        nonlocal node_called
        node_called = True
        return set()

    def cuda_runner(
        command: tuple[str, ...],
        **_: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="RuntimeError: CUDA unknown error 999",
        )

    with pytest.raises(
        acceptance.AcceptanceError,
        match="CUDA 功能预检失败",
    ):
        acceptance.run_acceptance(
            config,
            cuda_preflight_runner=cuda_runner,
            node_lister=node_lister,
        )

    assert node_called is False
    assert not config.output_dir.exists()


def test_disabled_cuda_preflight_is_explicit_test_only_report(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "flat_policy")

    report = acceptance.run_cuda_preflight(
        config,
        command_runner=lambda *_args, **_kwargs: pytest.fail(
            "disabled preflight must not spawn a subprocess"
        ),
    )

    assert report == {
        "required": False,
        "verified": False,
        "reason": "explicit_test_configuration",
    }


@pytest.mark.parametrize(
    "mode",
    [
        "static_stair",
        "flat_policy",
        "crossfloor_carry",
        "dynamic_f1",
        "dynamic_replan_f1",
    ],
)
def test_run_acceptance_cleans_launch_before_existing_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: acceptance.AcceptanceMode,
) -> None:
    config = _config(tmp_path, mode)
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/opt/ros/humble/bin/ros2")
    events: list[str] = []
    processes: list[_FakeProcess] = []

    def popen_factory(command: tuple[str, ...], **kwargs: Any) -> _FakeProcess:
        child_environment = kwargs["env"]
        if tuple(command) == acceptance.build_launch_command(config):
            assert child_environment.get("PYTHONPATH") == os.environ.get(
                "PYTHONPATH"
            )
            process = _FakeProcess(pid=4100, kind="launch")
            events.append("launch_started")
        else:
            assert tuple(command) == acceptance.build_pipeline_command(config)
            assert "PYTHONPATH" not in child_environment

            def finish_pipeline() -> None:
                _write_startup(config)
                events.append("pipeline_completed")

            process = _FakeProcess(
                pid=4200,
                kind="pipeline",
                on_pipeline_wait=finish_pipeline,
            )
            events.append("pipeline_started")
        processes.append(process)
        return process

    node_calls = 0

    def node_lister(environment: Any) -> set[str]:
        nonlocal node_calls
        assert environment["ROS_DOMAIN_ID"] == "217"
        node_calls += 1
        if node_calls == 1:
            events.append("domain_preflight_empty")
            return set()
        if node_calls == 2:
            events.append("launch_ready")
            return set(acceptance.REQUIRED_LAUNCH_NODES)
        events.append("domain_clean_after_sigint")
        return set()

    def command_runner(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert tuple(command) == acceptance.build_validator_command(config)
        assert "PYTHONPATH" not in kwargs["env"]
        assert processes[0].signals == [signal.SIGINT]
        assert json.loads(
            (config.output_dir / "startup_status.json").read_text(encoding="utf-8")
        )["status"] == "completed"
        events.append("validator_called")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"valid": True, "mode": mode}),
            stderr="",
        )

    result = acceptance.run_acceptance(
        config,
        popen_factory=popen_factory,
        command_runner=command_runner,
        node_lister=node_lister,
    )

    assert result["valid"] is True
    assert result["mode"] == mode
    assert result["launch_pid"] == 4100
    assert result["pipeline_pid"] == 4200
    assert events.index("pipeline_completed") < events.index("domain_clean_after_sigint")
    assert events.index("domain_clean_after_sigint") < events.index("validator_called")
    manifest = json.loads(
        (config.output_dir / "pct_scan_live_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "passed"
    assert manifest["require_active_sensing"] is False
    assert manifest["ros_domain_id"] == 217
    assert manifest["seed"] == 0
    assert manifest["navigation_body_height_m"] == pytest.approx(0.338)
    assert manifest["pipeline_pythonpath_cleared"] is True
    assert "body_height_m:=0.338" in manifest["launch_command"]
    assert manifest["tuning_config_snapshot"]["schema"] == (
        "pct_scan_tuning_snapshot_v1"
    )
    assert manifest["tuning_config_snapshot"]["selected_parameters"][
        "scan_controller.limits.max_yaw_rate"
    ] == pytest.approx(0.60)
    assert manifest["tuning_config_snapshot"]["selected_parameters"][
        "scan_controller.controller.turning_yaw_rate_threshold"
    ] == pytest.approx(0.35)
    assert manifest["tuning_config_snapshot"]["selected_parameters"][
        "scan_controller.controller.turning_max_planar_speed"
    ] == pytest.approx(0.42)
    assert manifest["tuning_config_snapshot"]["selected_parameters"][
        "scan_controller.finish.capture_zero_hold_distance_xy"
    ] == pytest.approx(0.075)
    assert manifest["tuning_config_snapshot"]["selected_parameters"][
        "navigation_supervisor.timeouts.max_yaw_alignment_freeze_sec"
    ] == pytest.approx(12.0)
    assert manifest["tuning_config_verification"]["verified"] is True
    assert manifest["source_bundle_snapshot"]["file_count"] == 1
    assert manifest["source_bundle_verification"]["verified"] is True
    assert result["source_bundle_verification"]["verified"] is True
    assert Path(
        manifest["tuning_config_snapshot"]["snapshot_path"]
    ).read_bytes() == config.tuning_config_file.read_bytes()
    assert manifest["scan_stair_freeze_profile"]["profile_id"] == (
        "go2_x5_multifloor_scan_stair_freeze_v1"
    )
    pipeline_command = manifest["pipeline_command"]
    assert (
        pipeline_command[pipeline_command.index("--navigation-body-height-m") + 1]
        == "0.338"
    )
    assert result["seed"] == 0


def test_tuning_snapshot_detects_source_mutation(tmp_path: Path) -> None:
    source = tmp_path / "pct_scan_tuning.yaml"
    source.write_bytes(acceptance.DEFAULT_TUNING_CONFIG_PATH.read_bytes())
    config = replace(
        _config(tmp_path, "flat_policy"),
        tuning_config_file=source,
    )
    config.output_dir.mkdir()

    snapshot = acceptance.create_tuning_config_snapshot(config)
    verified = acceptance.verify_tuning_config_snapshot(snapshot)

    assert verified["verified"] is True
    assert snapshot["selected_parameters"][
        "scan_controller.finish.min_approach_speed"
    ] == pytest.approx(0.28)
    assert snapshot["selected_parameters"][
        "scan_planner_node.fsm.reference_goal_hold_stable_dwell_sec"
    ] == pytest.approx(0.75)
    assert snapshot["selected_parameters"][
        "scan_planner_node.fsm.reference_goal_hold_distance_xy"
    ] == pytest.approx(0.04)
    assert snapshot["selected_parameters"][
        "scan_planner_node.fsm.reference_goal_hold_yaw_rate"
    ] == pytest.approx(0.10)
    assert snapshot["selected_parameters"][
        "scan_controller.finish.capture_entry_distance_xy"
    ] == pytest.approx(0.055)
    assert snapshot["selected_parameters"][
        "scan_controller.finish.max_yaw_rate"
    ] == pytest.approx(0.10)
    assert snapshot["selected_parameters"][
        "scan_controller.timeouts.max_yaw_alignment_freeze_sec"
    ] == pytest.approx(12.0)
    assert snapshot["selected_parameters"][
        "navigation_supervisor.timeouts.max_yaw_alignment_freeze_sec"
    ] == pytest.approx(12.0)

    source.write_bytes(source.read_bytes() + b"\n# changed during run\n")
    changed = acceptance.verify_tuning_config_snapshot(snapshot)

    assert changed["verified"] is False
    assert changed["error"] == "source_or_snapshot_digest_mismatch"


def test_source_bundle_snapshot_detects_snapshot_tampering(tmp_path: Path) -> None:
    config = _config(tmp_path, "flat_policy")
    config.output_dir.mkdir()

    identity = acceptance.create_source_bundle_snapshot(config)
    verified = acceptance.verify_source_bundle_snapshot(identity)

    assert verified["verified"] is True
    snapshot_path = Path(identity["snapshot_path"])
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b"\n")

    changed = acceptance.verify_source_bundle_snapshot(identity)

    assert changed["verified"] is False
    assert changed["error"] == "source_or_snapshot_digest_mismatch"


def test_source_bundle_snapshot_detects_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "flat_policy")
    config.output_dir.mkdir()
    identity = acceptance.create_source_bundle_snapshot(config)
    original_collect = acceptance._collect_source_bundle

    def changed_collect(source_roots: Any = None) -> dict[str, Any]:
        payload = original_collect(source_roots)
        payload["sha256"] = "0" * 64
        return payload

    monkeypatch.setattr(acceptance, "_collect_source_bundle", changed_collect)

    changed = acceptance.verify_source_bundle_snapshot(identity)

    assert changed["verified"] is False
    assert changed["current_sha256"] == "0" * 64


def test_tuning_snapshot_rejects_planner_hold_before_controller_dwell(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pct_scan_tuning.yaml"
    payload = yaml.safe_load(
        acceptance.DEFAULT_TUNING_CONFIG_PATH.read_text(encoding="utf-8")
    )
    payload["scan_planner_node"]["ros__parameters"][
        "fsm.reference_goal_hold_stable_dwell_sec"
    ] = 0.50
    source.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    config = replace(
        _config(tmp_path, "flat_policy"),
        tuning_config_file=source,
    )
    config.output_dir.mkdir()

    with pytest.raises(
        acceptance.AcceptanceError,
        match="连续驻留必须长于 controller",
    ):
        acceptance.create_tuning_config_snapshot(config)


def test_explicit_seed_is_bound_to_command_manifest_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_config(tmp_path, "flat_policy"), seed=2)
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    launch = _FakeProcess(pid=4300, kind="launch")

    def popen_factory(command: tuple[str, ...], **_: Any) -> _FakeProcess:
        if tuple(command) == acceptance.build_launch_command(config):
            return launch
        assert tuple(command) == acceptance.build_pipeline_command(config)
        return _FakeProcess(
            pid=4400,
            kind="pipeline",
            on_pipeline_wait=lambda: _write_startup(config),
        )

    node_results = iter(
        [set(), set(acceptance.REQUIRED_LAUNCH_NODES), set()]
    )
    result = acceptance.run_acceptance(
        config,
        popen_factory=popen_factory,
        command_runner=lambda command, **_: subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"valid": True}),
            stderr="",
        ),
        node_lister=lambda _: next(node_results),
    )

    manifest = json.loads(
        (config.output_dir / "pct_scan_live_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    assert acceptance.build_pipeline_command(config)[
        acceptance.build_pipeline_command(config).index("--seed") + 1
    ] == "2"
    assert manifest["seed"] == 2
    assert result["seed"] == 2


@pytest.mark.parametrize("value", [True, -1, 2_147_483_648, 1.5])
def test_invalid_seed_is_rejected_before_ros_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
) -> None:
    config = replace(_config(tmp_path, "flat_policy"), seed=value)
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    node_called = False

    def node_lister(_: Any) -> set[str]:
        nonlocal node_called
        node_called = True
        return set()

    with pytest.raises(acceptance.AcceptanceError, match="seed 必须"):
        acceptance.run_acceptance(config, node_lister=node_lister)
    assert node_called is False
    assert not config.output_dir.exists()


def test_summary_seed_mismatch_is_rejected_before_validator(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path, "flat_policy"), seed=2)
    config.output_dir.mkdir()
    episode_dir = config.output_dir / "episode_000000"
    episode_dir.mkdir()
    (episode_dir / "summary.json").write_text(
        json.dumps({"seed": 1}),
        encoding="utf-8",
    )

    with pytest.raises(acceptance.AcceptanceError, match="summary.seed"):
        acceptance.validate_episode_seed(config)


def test_existing_output_is_rejected_before_ros_or_process_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "flat_policy")
    config.output_dir.mkdir()
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    node_called = False

    def node_lister(_: Any) -> set[str]:
        nonlocal node_called
        node_called = True
        return set()

    with pytest.raises(acceptance.AcceptanceError, match="原本不存在"):
        acceptance.run_acceptance(config, node_lister=node_lister)
    assert node_called is False


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_navigation_body_height_is_rejected_before_ros_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    config = replace(
        _config(tmp_path, "flat_policy"),
        navigation_body_height_m=value,
    )
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    node_called = False

    def node_lister(_: Any) -> set[str]:
        nonlocal node_called
        node_called = True
        return set()

    with pytest.raises(acceptance.AcceptanceError, match="有限正数"):
        acceptance.run_acceptance(config, node_lister=node_lister)
    assert node_called is False
    assert not config.output_dir.exists()


def test_body_height_command_drift_is_rejected_before_ros_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "flat_policy")
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    original_builder = acceptance.build_pipeline_command

    def drifted_pipeline_command(
        current: acceptance.AcceptanceConfig,
    ) -> tuple[str, ...]:
        command = list(original_builder(current))
        command[command.index("--navigation-body-height-m") + 1] = "0.35"
        return tuple(command)

    monkeypatch.setattr(
        acceptance,
        "build_pipeline_command",
        drifted_pipeline_command,
    )
    node_called = False

    def node_lister(_: Any) -> set[str]:
        nonlocal node_called
        node_called = True
        return set()

    with pytest.raises(acceptance.AcceptanceError, match="命令合同不一致"):
        acceptance.run_acceptance(config, node_lister=node_lister)
    assert node_called is False
    assert not config.output_dir.exists()


def test_nonempty_domain_is_rejected_without_reserving_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "flat_policy")
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")

    with pytest.raises(acceptance.AcceptanceError, match="不是空 domain"):
        acceptance.run_acceptance(
            config,
            node_lister=lambda _: {"/foreign_node"},
        )
    assert not config.output_dir.exists()


def test_ros_preflight_log_side_effect_does_not_reserve_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模拟 rclpy 自动建日志目录时，正式 fresh 输出仍必须不存在。"""

    config = _config(tmp_path, "flat_policy")
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    observed_log_dirs: list[Path] = []

    def node_lister(environment: Any) -> set[str]:
        log_dir = Path(environment["ROS_LOG_DIR"])
        observed_log_dirs.append(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "python3_query.log").write_text("query\n", encoding="utf-8")
        assert not config.output_dir.exists()
        return {"/foreign_node"}

    with pytest.raises(acceptance.AcceptanceError, match="不是空 domain"):
        acceptance.run_acceptance(config, node_lister=node_lister)

    assert len(observed_log_dirs) == 1
    assert observed_log_dirs[0] != config.output_dir / "ros_logs"
    assert not observed_log_dirs[0].exists()
    assert not config.output_dir.exists()


def test_list_ros_nodes_hides_query_process_from_empty_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认 node list 不请求隐藏节点，避免把本次 ros2cli 查询算作残留。"""

    captured_command: tuple[str, ...] | None = None

    def fake_run(command: tuple[str, ...], **_: Any) -> subprocess.CompletedProcess[str]:
        nonlocal captured_command
        captured_command = tuple(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(acceptance.subprocess, "run", fake_run)

    assert acceptance.list_ros_nodes({"ROS_DOMAIN_ID": "217"}) == set()
    assert captured_command is not None
    assert captured_command[:4] == ("ros2", "node", "list", "--no-daemon")
    assert "--all" not in captured_command


@pytest.mark.parametrize(
    ("status", "exit_code", "message"),
    [
        ("starting", 0, "未完成"),
        ("completed", 1, "未成功"),
    ],
)
def test_startup_status_must_pass_before_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    exit_code: int,
    message: str,
) -> None:
    config = _config(tmp_path, "flat_policy")
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    validator_called = False
    launch = _FakeProcess(pid=5100, kind="launch")

    def popen_factory(command: tuple[str, ...], **_: Any) -> _FakeProcess:
        if tuple(command) == acceptance.build_launch_command(config):
            return launch
        return _FakeProcess(
            pid=5200,
            kind="pipeline",
            on_pipeline_wait=lambda: _write_startup(
                config,
                status=status,
                exit_code=exit_code,
            ),
        )

    node_results = iter(
        [set(), set(acceptance.REQUIRED_LAUNCH_NODES), set()]
    )

    def command_runner(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        nonlocal validator_called
        validator_called = True
        return subprocess.CompletedProcess([], 0, "{}", "")

    with pytest.raises(acceptance.AcceptanceError, match=message):
        acceptance.run_acceptance(
            config,
            popen_factory=popen_factory,
            command_runner=command_runner,
            node_lister=lambda _: next(node_results),
        )
    assert launch.signals == [signal.SIGINT]
    assert validator_called is False


def test_startup_status_must_echo_unique_navigation_body_height(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "flat_policy")
    config.output_dir.mkdir()
    _write_startup(config)
    path = config.output_dir / "startup_status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["navigation_body_height_m"] = 0.35
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(acceptance.AcceptanceError, match="唯一高度合同不一致"):
        acceptance.validate_startup_status(config)


def test_startup_status_must_echo_exact_freeze_profile_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "static_stair")
    config.output_dir.mkdir()
    _write_startup(config)
    path = config.output_dir / "startup_status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scan_stair_freeze_profile_runtime"]["contract_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(acceptance.AcceptanceError, match="生产 profile 不一致"):
        acceptance.validate_startup_status(config)


def test_pipeline_total_timeout_cleans_both_processes_and_records_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _config(tmp_path, "flat_policy"),
        pipeline_timeout_s=0.25,
    )
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    launch = _FakeProcess(pid=5600, kind="launch")
    pipeline = _FakeProcess(
        pid=5700,
        kind="pipeline",
        pipeline_wait_times_out=True,
    )

    def popen_factory(command: tuple[str, ...], **_: Any) -> _FakeProcess:
        if tuple(command) == acceptance.build_launch_command(config):
            return launch
        assert tuple(command) == acceptance.build_pipeline_command(config)
        return pipeline

    node_results = iter(
        [set(), set(acceptance.REQUIRED_LAUNCH_NODES), set()]
    )
    validator_called = False

    def command_runner(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        nonlocal validator_called
        validator_called = True
        return subprocess.CompletedProcess([], 0, "{}", "")

    with pytest.raises(acceptance.AcceptanceError, match="超过总时限"):
        acceptance.run_acceptance(
            config,
            popen_factory=popen_factory,
            command_runner=command_runner,
            node_lister=lambda _: next(node_results),
        )

    assert pipeline.signals == [signal.SIGINT]
    assert pipeline.terminate_count == 1
    assert pipeline.kill_count == 0
    assert launch.signals == [signal.SIGINT]
    assert validator_called is False
    manifest = json.loads(
        (config.output_dir / "pct_scan_live_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "failed"
    assert manifest["pipeline_timeout_s"] == pytest.approx(0.25)
    assert "超过总时限" in manifest["error"]["message"]
    assert manifest["domain_cleaned"] is True


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_invalid_pipeline_timeout_is_rejected_before_ros_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    config = replace(_config(tmp_path, "flat_policy"), pipeline_timeout_s=value)
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    node_called = False

    def node_lister(_: Any) -> set[str]:
        nonlocal node_called
        node_called = True
        return set()

    with pytest.raises(acceptance.AcceptanceError, match="有限正数"):
        acceptance.run_acceptance(config, node_lister=node_lister)
    assert node_called is False
    assert not config.output_dir.exists()


def test_launch_early_exit_prevents_pipeline_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "static_stair")
    monkeypatch.setattr(acceptance.shutil, "which", lambda _: "/usr/bin/ros2")
    process_count = 0

    def popen_factory(command: tuple[str, ...], **_: Any) -> _FakeProcess:
        nonlocal process_count
        process_count += 1
        assert tuple(command) == acceptance.build_launch_command(config)
        return _FakeProcess(pid=6100, kind="launch", early_exit=True)

    nodes = iter([set()])
    with pytest.raises(acceptance.AcceptanceError, match="pipeline 启动前退出"):
        acceptance.run_acceptance(
            config,
            popen_factory=popen_factory,
            node_lister=lambda _: next(nodes),
        )
    assert process_count == 1


def test_stop_launch_timeout_is_forced_but_never_counted_clean() -> None:
    process = _FakeProcess(pid=7100, kind="stubborn")

    report = acceptance.stop_launch_process(process, timeout_s=0.01)

    assert process.signals == [signal.SIGINT]
    assert process.terminate_count == 1
    assert process.kill_count == 0
    assert report.forced_action == "SIGTERM"
    assert report.clean_sigint_exit is False


def test_validator_command_targets_episode_directory(tmp_path: Path) -> None:
    config = _config(tmp_path, "dynamic_replan_f1")

    command = acceptance.build_validator_command(config)

    assert command[3] == str(config.output_dir / "episode_000000")
    assert command[command.index("--mode") + 1] == "dynamic_replan_f1"
    assert command[-1] == "--json"
