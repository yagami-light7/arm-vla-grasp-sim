#!/usr/bin/env python3
"""在 fresh 输出与隔离 ROS domain 中运行五类 PCT→SCAN live 验收。

本脚本只负责进程编排和验收次序，不替代 pipeline、组合 launch 或
``validate_pct_scan_live_summary.py``。输出目录在任何 ROS/Isaac 子进程启动前
必须不存在；ROS domain 在启动前必须为空。pipeline 结束后先用 SIGINT 清理
``ros2 launch`` 父进程并确认图已清空，再检查 ``startup_status.json``，最后才
调用现有 summary 校验器。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.navigation_contract import (  # noqa: E402
    DEFAULT_NAVIGATION_BODY_HEIGHT_M,
)
from source.navigation.scan_stair_freeze_profile import (  # noqa: E402
    ScanStairFreezeProfile,
    ScanStairFreezeProfileError,
    load_scan_stair_freeze_profile,
)


PIPELINE_SCRIPT = PROJECT_ROOT / "scripts/pipeline/run_full_physics_pipeline.py"
VALIDATOR_SCRIPT = PROJECT_ROOT / "scripts/navigation/validate_pct_scan_live_summary.py"
SCAN_STAIR_FREEZE_PROFILE_PATH = (
    PROJECT_ROOT
    / "configs/navigation/scan_stair_freeze_go2_x5_multifloor_v1.json"
)
DEFAULT_TUNING_CONFIG_PATH = (
    PROJECT_ROOT
    / "ros2_ws/src/isaac_navigation_bridge/config/pct_scan_tuning.yaml"
)
SOURCE_BUNDLE_ROOTS = (
    PROJECT_ROOT / "ros2_ws/src",
    PROJECT_ROOT / "source",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "external/PCT_planner",
    PROJECT_ROOT / "external/SCAN-Planner",
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "tasks",
)
SOURCE_BUNDLE_SUFFIXES = frozenset(
    {
        ".action",
        ".c",
        ".cc",
        ".cfg",
        ".cmake",
        ".cpp",
        ".cu",
        ".cuh",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".ini",
        ".json",
        ".launch",
        ".msg",
        ".py",
        ".rviz",
        ".srv",
        ".toml",
        ".txt",
        ".urdf",
        ".xacro",
        ".xml",
        ".yaml",
        ".yml",
    }
)
SOURCE_BUNDLE_FILENAMES = frozenset({"CMakeLists.txt"})
SOURCE_BUNDLE_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "build",
        "install",
        "log",
    }
)
ROS_WORKSPACE_SETUP_PATH = PROJECT_ROOT / "ros2_ws/install/setup.bash"
LAUNCH_COMMAND_PREFIX = (
    "ros2",
    "launch",
    "isaac_navigation_bridge",
    "pct_scan_navigation.launch.py",
    "start_pct:=true",
    "start_scan:=true",
    "start_controller:=true",
    "start_supervisor:=true",
    "start_manual_path:=false",
)
REQUIRED_LAUNCH_NODES = frozenset(
    {
        "/isaac_navigation_bridge",
        "/pct_ros2_adapter",
        "/scan_planner_node",
        "/scan_controller",
        "/navigation_supervisor",
    }
)
CUDA_PREFLIGHT_TIMEOUT_S = 30.0
CUDA_PREFLIGHT_SOURCE = "\n".join(
    (
        "import json",
        "import torch",
        "if not torch.cuda.is_available():",
        "    raise RuntimeError('torch.cuda.is_available() returned false')",
        "tensor = torch.zeros((1,), device='cuda:0', dtype=torch.float32)",
        "tensor.add_(1.0)",
        "torch.cuda.synchronize(0)",
        "properties = torch.cuda.get_device_properties(0)",
        "print(json.dumps({",
        "    'torch_version': torch.__version__,",
        "    'torch_cuda_version': torch.version.cuda,",
        "    'cuda_available': True,",
        "    'cuda_device_count': torch.cuda.device_count(),",
        "    'cuda_device_name': properties.name,",
        "    'cuda_capability': [properties.major, properties.minor],",
        "    'tensor_device': str(tensor.device),",
        "    'tensor_value': float(tensor.item()),",
        "}, sort_keys=True))",
    )
)

AcceptanceMode = Literal[
    "static_stair",
    "flat_policy",
    "crossfloor_carry",
    "dynamic_f1",
    "dynamic_replan_f1",
]


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """描述一种 live 验收对应的 pipeline 入口与固定任务。"""

    mode: AcceptanceMode
    pipeline_mode_flag: str
    task_path: Path
    expected_task_id: int


MODE_SPECS: dict[AcceptanceMode, ModeSpec] = {
    "static_stair": ModeSpec(
        mode="static_stair",
        pipeline_mode_flag="--stair-locomotion-smoke",
        task_path=PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json",
        expected_task_id=1002,
    ),
    "flat_policy": ModeSpec(
        mode="flat_policy",
        pipeline_mode_flag="--navigation-smoke",
        task_path=PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json",
        expected_task_id=1002,
    ),
    "crossfloor_carry": ModeSpec(
        mode="crossfloor_carry",
        pipeline_mode_flag="--navigation-carry-smoke",
        task_path=PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json",
        expected_task_id=1002,
    ),
    "dynamic_f1": ModeSpec(
        mode="dynamic_f1",
        pipeline_mode_flag="--navigation-smoke",
        task_path=PROJECT_ROOT / "tasks/nav_smoke_scan_multifloor_dynamic_cart_f1.json",
        expected_task_id=17704,
    ),
    "dynamic_replan_f1": ModeSpec(
        mode="dynamic_replan_f1",
        pipeline_mode_flag="--navigation-smoke",
        task_path=(
            PROJECT_ROOT
            / "tasks/nav_smoke_scan_multifloor_dynamic_blocker_replan_f1.json"
        ),
        expected_task_id=17705,
    ),
}


class AcceptanceError(RuntimeError):
    """表示运行不能形成可信 live 验收结论。"""


class ManagedProcess(Protocol):
    """声明编排器实际使用的最小子进程接口。"""

    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def send_signal(self, sig: int) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


PopenFactory = Callable[..., ManagedProcess]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
NodeLister = Callable[[Mapping[str, str]], set[str]]


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    """保存一次 fresh live 验收的不可变输入。"""

    mode: AcceptanceMode
    output_dir: Path
    ros_domain_id: int
    isaac_python: Path
    tuning_config_file: Path = DEFAULT_TUNING_CONFIG_PATH
    seed: int = 0
    navigation_body_height_m: float = DEFAULT_NAVIGATION_BODY_HEIGHT_M
    diagnostic_frame_stride: int = 10
    pipeline_timeout_s: float = 1800.0
    launch_ready_timeout_s: float = 20.0
    launch_stop_timeout_s: float = 20.0
    graph_cleanup_timeout_s: float = 10.0
    require_cuda_preflight: bool = True


@dataclass(frozen=True, slots=True)
class StopReport:
    """记录 launch 父进程接收 SIGINT 后的退出方式。"""

    pid: int
    signal_sent: str
    return_code: int
    clean_sigint_exit: bool
    forced_action: str | None


def production_scan_stair_freeze_profile() -> ScanStairFreezeProfile:
    """加载四类 multi-floor live 共用的楼梯冻结生产合同。"""

    try:
        return load_scan_stair_freeze_profile(
            SCAN_STAIR_FREEZE_PROFILE_PATH,
            expected_scene="multi_floor",
            expected_robot="go2_x5",
        )
    except (ScanStairFreezeProfileError, FileNotFoundError) as exc:
        raise AcceptanceError(f"楼梯冻结生产 profile 无效：{exc}") from exc


def _utc_now() -> str:
    """返回便于跨机器比较的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子更新运行清单，避免中断留下半个 JSON。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_bundle_files(
    source_roots: Sequence[Path] | None = None,
) -> list[Path]:
    """枚举会影响本 pipeline 的源码、接口与文本配置。"""

    effective_roots = SOURCE_BUNDLE_ROOTS if source_roots is None else source_roots
    selected: dict[str, Path] = {}
    for raw_root in effective_roots:
        root = raw_root.expanduser().resolve()
        try:
            root.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise AcceptanceError(
                f"源码 bundle 根目录必须位于当前 worktree：{root}"
            ) from exc
        if not root.exists():
            raise AcceptanceError(f"源码 bundle 根目录不存在：{root}")
        candidates = (root,) if root.is_file() else root.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                relative = candidate.resolve().relative_to(PROJECT_ROOT)
            except ValueError as exc:
                raise AcceptanceError(
                    f"源码 bundle 文件越出当前 worktree：{candidate}"
                ) from exc
            if SOURCE_BUNDLE_IGNORED_PARTS.intersection(relative.parts):
                continue
            if (
                candidate.name not in SOURCE_BUNDLE_FILENAMES
                and candidate.suffix.lower() not in SOURCE_BUNDLE_SUFFIXES
            ):
                continue
            selected[relative.as_posix()] = candidate.resolve()
    if not selected:
        raise AcceptanceError("源码 bundle 没有选中任何文件。")
    return [selected[key] for key in sorted(selected)]


def _collect_source_bundle(
    source_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """计算当前导航运行源码的逐文件身份和规范摘要。"""

    effective_roots = SOURCE_BUNDLE_ROOTS if source_roots is None else source_roots
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in _source_bundle_files(effective_roots):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise AcceptanceError(f"无法读取源码 bundle 文件 {path}：{exc}") from exc
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
        )
        total_bytes += len(raw)
    relative_roots = [
        root.expanduser().resolve().relative_to(PROJECT_ROOT).as_posix()
        for root in effective_roots
    ]
    digest_payload = {
        "schema": "pct_scan_source_bundle_digest_v1",
        "source_roots": relative_roots,
        "files": entries,
    }
    canonical = json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": "pct_scan_source_bundle_snapshot_v1",
        "source_roots": relative_roots,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }


def create_source_bundle_snapshot(
    config: AcceptanceConfig,
    *,
    source_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """在 ROS 启动前独占写入本轮导航源码 bundle。"""

    snapshot_path = config.output_dir / "pct_scan_source_bundle_snapshot.json"
    snapshot = _collect_source_bundle(source_roots)
    raw = (
        json.dumps(
            snapshot,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with snapshot_path.open("xb") as handle:
            handle.write(raw)
    except OSError as exc:
        raise AcceptanceError(
            f"无法写入源码 bundle 快照 {snapshot_path}：{exc}"
        ) from exc
    return {
        "schema": "pct_scan_source_bundle_identity_v1",
        "snapshot_path": os.fspath(snapshot_path),
        "snapshot_file_sha256": hashlib.sha256(raw).hexdigest(),
        "sha256": snapshot["sha256"],
        "file_count": snapshot["file_count"],
        "total_bytes": snapshot["total_bytes"],
        "source_roots": snapshot["source_roots"],
    }


def verify_source_bundle_snapshot(
    identity: Mapping[str, Any],
    *,
    source_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """确认源码、根目录范围和已落盘快照在运行期间均未变化。"""

    snapshot_path = Path(str(identity.get("snapshot_path", "")))
    report: dict[str, Any] = {
        "schema": "pct_scan_source_bundle_verification_v1",
        "expected_sha256": identity.get("sha256"),
        "current_sha256": None,
        "expected_snapshot_file_sha256": identity.get("snapshot_file_sha256"),
        "snapshot_file_sha256": None,
        "verified": False,
        "error": None,
    }
    try:
        snapshot_raw = snapshot_path.read_bytes()
        snapshot = json.loads(snapshot_raw.decode("utf-8"))
        current = _collect_source_bundle(source_roots)
    except (OSError, UnicodeError, json.JSONDecodeError, AcceptanceError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    if not isinstance(snapshot, Mapping):
        report["error"] = "snapshot_payload_not_mapping"
        return report
    snapshot_file_sha = hashlib.sha256(snapshot_raw).hexdigest()
    report["snapshot_file_sha256"] = snapshot_file_sha
    report["current_sha256"] = current["sha256"]
    report["verified"] = bool(
        isinstance(identity.get("sha256"), str)
        and snapshot.get("sha256") == identity.get("sha256")
        and current["sha256"] == identity.get("sha256")
        and current["file_count"] == identity.get("file_count")
        and current["total_bytes"] == identity.get("total_bytes")
        and current["source_roots"] == identity.get("source_roots")
        and snapshot_file_sha == identity.get("snapshot_file_sha256")
    )
    if not report["verified"]:
        report["error"] = "source_or_snapshot_digest_mismatch"
    return report


def _finite_tuning_parameter(
    payload: Mapping[str, Any],
    node_name: str,
    parameter_name: str,
) -> float:
    """从统一 YAML 读取用于复现实验的有限数值参数。"""

    try:
        value = float(
            payload[node_name]["ros__parameters"][parameter_name]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AcceptanceError(
            "统一调参文件缺少测速证据参数："
            f"{node_name}.ros__parameters.{parameter_name}。"
        ) from exc
    if not math.isfinite(value):
        raise AcceptanceError(
            "统一调参测速证据参数必须为有限数："
            f"{node_name}.ros__parameters.{parameter_name}。"
        )
    return value


def create_tuning_config_snapshot(config: AcceptanceConfig) -> dict[str, Any]:
    """在启动 ROS 前原样固化本轮完整调参文件及关键物理参数。"""

    source_path = config.tuning_config_file
    snapshot_path = config.output_dir / "pct_scan_tuning_snapshot.yaml"
    try:
        source_bytes = source_path.read_bytes()
        source_text = source_bytes.decode("utf-8")
        raw_payload = yaml.safe_load(source_text)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AcceptanceError(
            f"无法固化统一调参文件 {source_path}：{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(raw_payload, Mapping):
        raise AcceptanceError("统一调参 YAML 顶层必须是对象。")
    body_height = _finite_tuning_parameter(
        raw_payload,
        "navigation_contract",
        "body_height_m",
    )
    if body_height != float(config.navigation_body_height_m):
        raise AcceptanceError(
            "统一 YAML 与 live 命令的 body_height_m 不一致："
            f"yaml={body_height!r}, command={config.navigation_body_height_m!r}。"
        )
    planner_hold_distance_xy = _finite_tuning_parameter(
        raw_payload,
        "scan_planner_node",
        "fsm.reference_goal_hold_distance_xy",
    )
    planner_hold_dwell_sec = _finite_tuning_parameter(
        raw_payload,
        "scan_planner_node",
        "fsm.reference_goal_hold_stable_dwell_sec",
    )
    planner_hold_yaw_rate = _finite_tuning_parameter(
        raw_payload,
        "scan_planner_node",
        "fsm.reference_goal_hold_yaw_rate",
    )
    controller_capture_entry_xy = _finite_tuning_parameter(
        raw_payload,
        "scan_controller",
        "finish.capture_entry_distance_xy",
    )
    controller_capture_zero_hold_xy = _finite_tuning_parameter(
        raw_payload,
        "scan_controller",
        "finish.capture_zero_hold_distance_xy",
    )
    controller_finish_distance_xy = _finite_tuning_parameter(
        raw_payload,
        "scan_controller",
        "finish.distance_xy",
    )
    controller_stable_dwell_sec = _finite_tuning_parameter(
        raw_payload,
        "scan_controller",
        "finish.capture_stable_dwell_sec",
    )
    controller_finish_yaw_rate = _finite_tuning_parameter(
        raw_payload,
        "scan_controller",
        "finish.max_yaw_rate",
    )
    controller_yaw_freeze_sec = _finite_tuning_parameter(
        raw_payload,
        "scan_controller",
        "timeouts.max_yaw_alignment_freeze_sec",
    )
    supervisor_yaw_freeze_sec = _finite_tuning_parameter(
        raw_payload,
        "navigation_supervisor",
        "timeouts.max_yaw_alignment_freeze_sec",
    )
    if not 0.0 < planner_hold_distance_xy < controller_capture_entry_xy:
        raise AcceptanceError(
            "planner stationary final hold 的 XY 门必须严格位于 controller "
            "捕获内门之内。"
        )
    if not (
        controller_capture_entry_xy
        < controller_capture_zero_hold_xy
        < controller_finish_distance_xy
    ):
        raise AcceptanceError(
            "controller 零速保持门必须严格位于捕获内门与完成门之间。"
        )
    if not planner_hold_dwell_sec > controller_stable_dwell_sec > 0.0:
        raise AcceptanceError(
            "planner stationary final hold 的连续驻留必须长于 controller "
            "GOAL_REACHED 驻留，使 moving final 优先完成。"
        )
    if not 0.0 < planner_hold_yaw_rate <= controller_finish_yaw_rate:
        raise AcceptanceError(
            "planner stationary final hold 的 |wz| 门必须为正，且不能宽于 "
            "controller 的最终偏航角速度门。"
        )
    if supervisor_yaw_freeze_sec != controller_yaw_freeze_sec:
        raise AcceptanceError(
            "supervisor 与 controller 的航向对齐冻结硬上限必须完全相等。"
        )
    selected_parameters = {
        "navigation_contract.body_height_m": body_height,
        "pct_ros2_adapter.planner.upstream_astar_step_cost_weight": (
            _finite_tuning_parameter(
                raw_payload,
                "pct_ros2_adapter",
                "planner.upstream_astar_step_cost_weight",
            )
        ),
        "pct_ros2_adapter.planner.path_sample_spacing_m": (
            _finite_tuning_parameter(
                raw_payload,
                "pct_ros2_adapter",
                "planner.path_sample_spacing_m",
            )
        ),
        "scan_planner_node.fsm.planning_horizon": _finite_tuning_parameter(
            raw_payload,
            "scan_planner_node",
            "fsm.planning_horizon",
        ),
        "scan_planner_node.fsm.thresh_replan": _finite_tuning_parameter(
            raw_payload,
            "scan_planner_node",
            "fsm.thresh_replan",
        ),
        "scan_planner_node.fsm.reference_cruise_speed": (
            _finite_tuning_parameter(
                raw_payload,
                "scan_planner_node",
                "fsm.reference_cruise_speed",
            )
        ),
        "scan_planner_node.fsm.reference_goal_hold_distance_xy": (
            planner_hold_distance_xy
        ),
        "scan_planner_node.fsm.reference_goal_hold_stable_dwell_sec": (
            planner_hold_dwell_sec
        ),
        "scan_planner_node.fsm.reference_goal_hold_yaw_rate": (
            planner_hold_yaw_rate
        ),
        "scan_planner_node.manager.max_vel": _finite_tuning_parameter(
            raw_payload,
            "scan_planner_node",
            "manager.max_vel",
        ),
        "scan_planner_node.manager.max_acc": _finite_tuning_parameter(
            raw_payload,
            "scan_planner_node",
            "manager.max_acc",
        ),
        "scan_controller.controller.kp_yaw": _finite_tuning_parameter(
            raw_payload,
            "scan_controller",
            "controller.kp_yaw",
        ),
        "scan_controller.controller.cross_track_heading_error_threshold": (
            _finite_tuning_parameter(
                raw_payload,
                "scan_controller",
                "controller.cross_track_heading_error_threshold",
            )
        ),
        "scan_controller.controller.turning_yaw_rate_threshold": (
            _finite_tuning_parameter(
                raw_payload,
                "scan_controller",
                "controller.turning_yaw_rate_threshold",
            )
        ),
        "scan_controller.controller.turning_max_planar_speed": (
            _finite_tuning_parameter(
                raw_payload,
                "scan_controller",
                "controller.turning_max_planar_speed",
            )
        ),
        "scan_controller.limits.max_vx": _finite_tuning_parameter(
            raw_payload,
            "scan_controller",
            "limits.max_vx",
        ),
        "scan_controller.limits.max_yaw_rate": _finite_tuning_parameter(
            raw_payload,
            "scan_controller",
            "limits.max_yaw_rate",
        ),
        "scan_controller.limits.max_yaw_acc": _finite_tuning_parameter(
            raw_payload,
            "scan_controller",
            "limits.max_yaw_acc",
        ),
        "scan_controller.finish.distance_xy": controller_finish_distance_xy,
        "scan_controller.finish.capture_entry_distance_xy": (
            controller_capture_entry_xy
        ),
        "scan_controller.finish.capture_zero_hold_distance_xy": (
            controller_capture_zero_hold_xy
        ),
        "scan_controller.finish.capture_stable_dwell_sec": (
            controller_stable_dwell_sec
        ),
        "scan_controller.finish.max_yaw_rate": controller_finish_yaw_rate,
        "scan_controller.finish.min_approach_speed": _finite_tuning_parameter(
            raw_payload,
            "scan_controller",
            "finish.min_approach_speed",
        ),
        "scan_controller.timeouts.max_yaw_alignment_freeze_sec": (
            controller_yaw_freeze_sec
        ),
        "navigation_supervisor.timeouts.max_yaw_alignment_freeze_sec": (
            supervisor_yaw_freeze_sec
        ),
    }
    try:
        with snapshot_path.open("xb") as handle:
            handle.write(source_bytes)
    except OSError as exc:
        raise AcceptanceError(
            f"无法写入本轮调参快照 {snapshot_path}：{exc}"
        ) from exc
    digest = hashlib.sha256(source_bytes).hexdigest()
    return {
        "schema": "pct_scan_tuning_snapshot_v1",
        "source_path": os.fspath(source_path),
        "snapshot_path": os.fspath(snapshot_path),
        "sha256": digest,
        "byte_count": len(source_bytes),
        "selected_parameters": selected_parameters,
    }


def verify_tuning_config_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """确认运行使用的源 YAML 与已固化快照在结束时仍逐字节一致。"""

    expected_digest = snapshot.get("sha256")
    source_path = Path(str(snapshot.get("source_path", "")))
    snapshot_path = Path(str(snapshot.get("snapshot_path", "")))
    report: dict[str, Any] = {
        "schema": "pct_scan_tuning_snapshot_verification_v1",
        "expected_sha256": expected_digest,
        "source_path": os.fspath(source_path),
        "snapshot_path": os.fspath(snapshot_path),
        "source_sha256": None,
        "snapshot_sha256": None,
        "verified": False,
        "error": None,
    }
    try:
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        snapshot_digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    except OSError as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    report["source_sha256"] = source_digest
    report["snapshot_sha256"] = snapshot_digest
    report["verified"] = (
        isinstance(expected_digest, str)
        and source_digest == expected_digest
        and snapshot_digest == expected_digest
    )
    if not report["verified"]:
        report["error"] = "source_or_snapshot_digest_mismatch"
    return report


def _validate_config(config: AcceptanceConfig) -> None:
    """在产生任何外部状态前拒绝危险或自相矛盾的输入。"""

    if config.mode not in MODE_SPECS:
        raise AcceptanceError(f"未知验收模式：{config.mode!r}")
    if not 0 <= config.ros_domain_id <= 232:
        raise AcceptanceError("ROS_DOMAIN_ID 必须位于 [0, 232]。")
    if (
        isinstance(config.seed, bool)
        or not isinstance(config.seed, int)
        or not 0 <= config.seed <= 2_147_483_647
    ):
        raise AcceptanceError("seed 必须是 [0, 2147483647] 内的整数。")
    if (
        isinstance(config.navigation_body_height_m, bool)
        or not math.isfinite(config.navigation_body_height_m)
        or config.navigation_body_height_m <= 0.0
    ):
        raise AcceptanceError("navigation_body_height_m 必须是有限正数。")
    if (
        isinstance(config.diagnostic_frame_stride, bool)
        or not isinstance(config.diagnostic_frame_stride, int)
        or config.diagnostic_frame_stride < 1
    ):
        raise AcceptanceError("diagnostic_frame_stride 必须是正整数。")
    if not isinstance(config.require_cuda_preflight, bool):
        raise AcceptanceError("require_cuda_preflight 必须是布尔值。")
    profile = production_scan_stair_freeze_profile()
    try:
        profile.validate_runtime_bindings(
            navigation_body_height_m=config.navigation_body_height_m,
            control_dt_s=0.02,
        )
    except ScanStairFreezeProfileError as exc:
        raise AcceptanceError(
            f"live 参数与楼梯冻结生产 profile 不一致：{exc}"
        ) from exc
    for name, value in (
        ("pipeline_timeout_s", config.pipeline_timeout_s),
        ("launch_ready_timeout_s", config.launch_ready_timeout_s),
        ("launch_stop_timeout_s", config.launch_stop_timeout_s),
        ("graph_cleanup_timeout_s", config.graph_cleanup_timeout_s),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise AcceptanceError(f"{name} 必须为有限正数。")
    if config.output_dir.exists():
        raise AcceptanceError(
            f"输出目录必须原本不存在，拒绝复用：{config.output_dir}"
        )
    if not config.isaac_python.is_file():
        raise AcceptanceError(f"Isaac Python 不存在：{config.isaac_python}")
    if not os.access(config.isaac_python, os.X_OK):
        raise AcceptanceError(f"Isaac Python 不可执行：{config.isaac_python}")
    if not config.tuning_config_file.is_file():
        raise AcceptanceError(
            f"PCT+SCAN 统一调参文件不存在：{config.tuning_config_file}"
        )
    if shutil.which("ros2") is None:
        raise AcceptanceError("找不到 ros2；请先 source ROS 2 与 ros2_ws overlay。")
    spec = MODE_SPECS[config.mode]
    if not spec.task_path.is_file():
        raise AcceptanceError(f"验收任务不存在：{spec.task_path}")
    try:
        task = json.loads(spec.task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"验收任务无法读取：{spec.task_path}: {exc}") from exc
    if not isinstance(task, dict) or task.get("task_id") != spec.expected_task_id:
        raise AcceptanceError(
            "验收任务身份与固定模式不一致："
            f"expected={spec.expected_task_id}, actual="
            f"{task.get('task_id') if isinstance(task, dict) else None}。"
        )


def _body_height_cli_value(config: AcceptanceConfig) -> str:
    """生成跨两个子进程完全相同的机体高度文本。"""

    return repr(float(config.navigation_body_height_m))


def load_navigation_body_height_m(tuning_config_file: Path) -> float:
    """从统一 YAML 读取 live launch 与 host pipeline 共用的机体高度。"""

    try:
        payload = yaml.safe_load(tuning_config_file.read_text(encoding="utf-8"))
        value = float(
            payload["navigation_contract"]["ros__parameters"][
                "body_height_m"
            ]
        )
    except FileNotFoundError as exc:
        raise AcceptanceError(
            f"统一调参文件不存在：{tuning_config_file}"
        ) from exc
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise AcceptanceError(
            "统一调参文件缺少有限正数 "
            "navigation_contract.ros__parameters.body_height_m。"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise AcceptanceError("统一调参 body_height_m 必须是有限正数。")
    return value


def build_launch_command(config: AcceptanceConfig) -> tuple[str, ...]:
    """构造绑定唯一机体高度合同的 ROS 组合 launch 命令。"""

    return (
        *LAUNCH_COMMAND_PREFIX,
        f"body_height_m:={_body_height_cli_value(config)}",
        f"tuning_config_file:={config.tuning_config_file}",
    )


def build_pipeline_command(config: AcceptanceConfig) -> tuple[str, ...]:
    """构造固定 PCT→SCAN live pipeline 命令。"""

    spec = MODE_SPECS[config.mode]
    return (
        os.fspath(config.isaac_python),
        "-B",
        os.fspath(PIPELINE_SCRIPT),
        "--scene-profile",
        "multi_floor",
        "--task-json",
        os.fspath(spec.task_path),
        spec.pipeline_mode_flag,
        "--seed",
        str(config.seed),
        "--num-episodes",
        "1",
        "--no-randomize-task",
        "--no-randomize-base-goal",
        "--enable-navigation-ros2-bridge",
        "--scan-stair-freeze",
        "--no-pct-stair-float",
        "--navigation-body-height-m",
        _body_height_cli_value(config),
        "--pct-scan-tuning-config",
        os.fspath(config.tuning_config_file),
        "--navigation-visual-mode",
        "collision",
        "--headless",
        "--no-keep-window-open",
        "--no-record-dataset",
        "--no-record-video",
        "--diagnostic-frame-stride",
        str(config.diagnostic_frame_stride),
        "--ros2-domain-id",
        str(config.ros_domain_id),
        "--output-dir",
        os.fspath(config.output_dir),
    )


def validate_body_height_command_contract(
    config: AcceptanceConfig,
    *,
    launch_command: Sequence[str],
    pipeline_command: Sequence[str],
) -> None:
    """启动任何进程前证明 ROS 与 host 命令使用同一机体高度。"""

    launch_prefix = "body_height_m:="
    launch_values = [
        value.removeprefix(launch_prefix)
        for value in launch_command
        if value.startswith(launch_prefix)
    ]
    pipeline_indices = [
        index
        for index, value in enumerate(pipeline_command)
        if value == "--navigation-body-height-m"
    ]
    if (
        len(launch_values) != 1
        or len(pipeline_indices) != 1
        or pipeline_indices[0] + 1 >= len(pipeline_command)
    ):
        raise AcceptanceError(
            "机体高度命令合同必须在 launch 与 pipeline 中各出现一次。"
        )
    pipeline_value = pipeline_command[pipeline_indices[0] + 1]
    try:
        launch_height = float(launch_values[0])
        pipeline_height = float(pipeline_value)
    except (TypeError, ValueError) as exc:
        raise AcceptanceError("机体高度命令合同包含非数值参数。") from exc
    expected_height = float(config.navigation_body_height_m)
    if not (
        launch_height == expected_height
        and pipeline_height == expected_height
        and launch_height == pipeline_height
    ):
        raise AcceptanceError(
            "机体高度命令合同不一致："
            f"config={expected_height!r}, launch={launch_height!r}, "
            f"pipeline={pipeline_height!r}。"
        )


def build_validator_command(config: AcceptanceConfig) -> tuple[str, ...]:
    """构造现有 fail-closed summary 校验器命令。"""

    command = (
        os.fspath(config.isaac_python),
        "-B",
        os.fspath(VALIDATOR_SCRIPT),
        os.fspath(config.output_dir / "episode_000000"),
        "--mode",
        config.mode,
        "--json",
    )
    return command


def build_process_environment(
    config: AcceptanceConfig,
    *,
    ros_log_dir: Path | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """生成隔离环境，并允许预检把 ROS 日志放在 fresh 输出之外。"""

    environment = dict(os.environ if base_environment is None else base_environment)
    environment["ROS_DOMAIN_ID"] = str(config.ros_domain_id)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["ROS_LOG_DIR"] = os.fspath(
        config.output_dir / "ros_logs"
        if ros_log_dir is None
        else ros_log_dir
    )
    return environment


def build_isaac_process_environment(
    config: AcceptanceConfig,
    *,
    ros_log_dir: Path | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """生成 Isaac Python 环境，并移除 ROS Humble 的 Python 3.10 路径。

    ROS launch 仍使用 :func:`build_process_environment`，因此能够导入 rclpy；
    Isaac 进程只保留 AMENT/LD_LIBRARY_PATH 以加载 ROS 共享库和自定义消息，
    禁止把 Python 3.10 site-packages 注入 Python 3.11。
    """

    environment = build_process_environment(
        config,
        ros_log_dir=ros_log_dir,
        base_environment=base_environment,
    )
    environment.pop("PYTHONPATH", None)
    return environment


def load_ros_workspace_environment(
    *,
    base_environment: Mapping[str, str] | None = None,
    setup_path: Path = ROS_WORKSPACE_SETUP_PATH,
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, str]:
    """读取当前 worktree 的 ROS overlay，避免依赖调用终端预先 source。"""

    resolved_setup = setup_path.expanduser().resolve()
    if not resolved_setup.is_file():
        raise AcceptanceError(
            "ROS workspace 尚未构建或 setup.bash 不存在："
            f"{resolved_setup}"
        )
    source_environment = dict(
        os.environ if base_environment is None else base_environment
    )
    try:
        result = command_runner(
            (
                "/bin/bash",
                "-c",
                'source "$1" && env -0',
                "pct_scan_ros_overlay",
                os.fspath(resolved_setup),
            ),
            cwd=PROJECT_ROOT,
            env=source_environment,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise AcceptanceError(
            "无法读取 ROS workspace overlay："
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise AcceptanceError(
            "加载 ROS workspace overlay 失败："
            f"return_code={result.returncode}, detail={detail!r}"
        )
    environment: dict[str, str] = {}
    for entry in result.stdout.split("\0"):
        if not entry or "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        environment[name] = value
    ament_prefixes = tuple(
        Path(value).expanduser().resolve()
        for value in environment.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
        if value
    )
    install_root = resolved_setup.parent
    if not any(
        prefix == install_root or install_root in prefix.parents
        for prefix in ament_prefixes
    ):
        raise AcceptanceError(
            "ROS overlay 已执行但 AMENT_PREFIX_PATH 未包含当前 worktree install："
            f"setup={resolved_setup}"
        )
    return environment


def build_cuda_preflight_command(config: AcceptanceConfig) -> tuple[str, ...]:
    """用正式 Isaac Python 构造真实 CUDA tensor 功能预检。"""

    return (
        os.fspath(config.isaac_python),
        "-B",
        "-c",
        CUDA_PREFLIGHT_SOURCE,
    )


def run_cuda_preflight(
    config: AcceptanceConfig,
    *,
    command_runner: CommandRunner = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """要求同一解释器真实分配并同步 CUDA tensor，不能只相信 nvidia-smi。"""

    if not config.require_cuda_preflight:
        return {
            "required": False,
            "verified": False,
            "reason": "explicit_test_configuration",
        }
    command = build_cuda_preflight_command(config)
    child_environment = (
        build_isaac_process_environment(config)
        if environment is None
        else dict(environment)
    )
    try:
        result = command_runner(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment,
            text=True,
            capture_output=True,
            timeout=CUDA_PREFLIGHT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AcceptanceError(
            "Isaac CUDA 功能预检超时："
            f"timeout_s={CUDA_PREFLIGHT_TIMEOUT_S}。"
        ) from exc
    except OSError as exc:
        raise AcceptanceError(
            "无法启动 Isaac CUDA 功能预检："
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise AcceptanceError(
            "Isaac CUDA 功能预检失败；不会启动 ROS/Isaac："
            f"return_code={result.returncode}, detail={detail!r}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AcceptanceError("Isaac CUDA 功能预检未返回合法 JSON。") from exc
    if not isinstance(payload, dict):
        raise AcceptanceError("Isaac CUDA 功能预检结果必须是 JSON 对象。")
    if not (
        payload.get("cuda_available") is True
        and payload.get("tensor_device") == "cuda:0"
        and payload.get("tensor_value") == 1.0
        and isinstance(payload.get("cuda_device_count"), int)
        and payload["cuda_device_count"] >= 1
    ):
        raise AcceptanceError(
            "Isaac CUDA 功能预检合同不完整："
            f"{payload}"
        )
    return {
        "required": True,
        "verified": True,
        "interpreter": os.fspath(config.isaac_python),
        "timeout_s": CUDA_PREFLIGHT_TIMEOUT_S,
        **payload,
    }


def list_ros_nodes(environment: Mapping[str, str]) -> set[str]:
    """绕过 daemon 读取当前 domain 的 ROS 节点集合。"""

    result = subprocess.run(
        (
            "ros2",
            "node",
            "list",
            "--no-daemon",
            "--spin-time",
            "1.0",
        ),
        cwd=PROJECT_ROOT,
        env=dict(environment),
        text=True,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AcceptanceError(f"无法检查隔离 ROS domain：{detail}")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _wait_for_required_nodes(
    process: ManagedProcess,
    environment: Mapping[str, str],
    *,
    timeout_s: float,
    node_lister: NodeLister,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """等待组合 launch 的五个生产节点全部可发现。"""

    deadline = monotonic() + timeout_s
    last_nodes: set[str] = set()
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise AcceptanceError(
                f"ros2 launch 在 pipeline 启动前退出，return_code={return_code}。"
            )
        last_nodes = node_lister(environment)
        if REQUIRED_LAUNCH_NODES <= last_nodes:
            return
        if monotonic() >= deadline:
            missing = sorted(REQUIRED_LAUNCH_NODES - last_nodes)
            raise AcceptanceError(f"等待 ROS 组合 launch 超时，缺少节点：{missing}")
        sleep(0.25)


def _wait_for_empty_domain(
    environment: Mapping[str, str],
    *,
    timeout_s: float,
    node_lister: NodeLister,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """确认 SIGINT 后没有验收节点遗留在隔离 domain。"""

    deadline = monotonic() + timeout_s
    last_nodes: set[str] = set()
    while True:
        last_nodes = node_lister(environment)
        if not last_nodes:
            return
        if monotonic() >= deadline:
            raise AcceptanceError(
                f"SIGINT 后 ROS domain 未清空，遗留节点：{sorted(last_nodes)}"
            )
        sleep(0.25)


def stop_launch_process(
    process: ManagedProcess,
    *,
    timeout_s: float,
) -> StopReport:
    """只对本脚本创建的 launch 父进程执行 SIGINT，并有界回收。"""

    return_code = process.poll()
    if return_code is not None:
        return StopReport(
            pid=process.pid,
            signal_sent="none",
            return_code=return_code,
            clean_sigint_exit=False,
            forced_action=None,
        )

    process.send_signal(signal.SIGINT)
    try:
        return_code = process.wait(timeout=timeout_s)
        return StopReport(
            pid=process.pid,
            signal_sent="SIGINT",
            return_code=return_code,
            clean_sigint_exit=return_code in {0, -signal.SIGINT, 128 + signal.SIGINT},
            forced_action=None,
        )
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return_code = process.wait(timeout=min(timeout_s, 5.0))
            forced_action = "SIGTERM"
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=min(timeout_s, 5.0))
            forced_action = "SIGKILL"
        return StopReport(
            pid=process.pid,
            signal_sent="SIGINT",
            return_code=return_code,
            clean_sigint_exit=False,
            forced_action=forced_action,
        )


def validate_startup_status(config: AcceptanceConfig) -> dict[str, Any]:
    """要求本轮 pipeline 明确完成且退出码为零。"""

    path = config.output_dir / "startup_status.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AcceptanceError(f"缺少本轮 startup_status.json：{path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"startup_status.json 无法读取：{exc}") from exc
    if not isinstance(payload, dict):
        raise AcceptanceError("startup_status.json 顶层必须是 JSON 对象。")
    if payload.get("status") != "completed":
        raise AcceptanceError(
            "pipeline 未完成：startup_status.status="
            f"{payload.get('status')!r}。"
        )
    if payload.get("exit_code") != 0:
        raise AcceptanceError(
            "pipeline 未成功：startup_status.exit_code="
            f"{payload.get('exit_code')!r}。"
        )
    expected_pipeline_mode = MODE_SPECS[config.mode].pipeline_mode_flag.removeprefix(
        "--"
    ).replace("-", "_")
    if payload.get("mode") != expected_pipeline_mode:
        raise AcceptanceError(
            "startup_status.mode 与验收模式不一致："
            f"expected={expected_pipeline_mode!r}, actual={payload.get('mode')!r}。"
        )
    if payload.get("navigation_body_height_m") != config.navigation_body_height_m:
        raise AcceptanceError(
            "startup_status.navigation_body_height_m 与本轮唯一高度合同不一致："
            f"expected={config.navigation_body_height_m!r}, "
            f"actual={payload.get('navigation_body_height_m')!r}。"
        )
    expected_profile = production_scan_stair_freeze_profile().audit_report()
    if payload.get("scan_stair_freeze_profile_runtime") != expected_profile:
        raise AcceptanceError(
            "startup_status.scan_stair_freeze_profile_runtime 与受校验生产 "
            "profile 不一致。"
        )
    return payload


def validate_episode_seed(config: AcceptanceConfig) -> dict[str, Any]:
    """要求唯一 episode 的结果显式回显本轮 seed。"""

    path = config.output_dir / "episode_000000" / "summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AcceptanceError(f"缺少本轮 summary.json：{path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"summary.json 无法读取：{exc}") from exc
    if not isinstance(payload, dict):
        raise AcceptanceError("summary.json 顶层必须是 JSON 对象。")
    if payload.get("seed") != config.seed:
        raise AcceptanceError(
            "summary.seed 与本轮验收 seed 不一致："
            f"expected={config.seed}, actual={payload.get('seed')!r}。"
        )
    return payload


def _interrupt_pipeline(process: ManagedProcess, *, timeout_s: float) -> None:
    """用户中断编排器时，有界停止本脚本创建的 pipeline 子进程。"""

    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=min(timeout_s, 5.0))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=min(timeout_s, 5.0))


def run_acceptance(
    config: AcceptanceConfig,
    *,
    popen_factory: PopenFactory = subprocess.Popen,
    command_runner: CommandRunner = subprocess.run,
    cuda_preflight_runner: CommandRunner = subprocess.run,
    node_lister: NodeLister = list_ros_nodes,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """执行一次 fresh-run 编排，并返回机器可读结果。"""

    _validate_config(config)
    scan_stair_freeze_profile_audit = (
        production_scan_stair_freeze_profile().audit_report()
    )
    launch_command = build_launch_command(config)
    pipeline_command = build_pipeline_command(config)
    validator_command = build_validator_command(config)
    validate_body_height_command_contract(
        config,
        launch_command=launch_command,
        pipeline_command=pipeline_command,
    )
    with tempfile.TemporaryDirectory(prefix="pct_scan_cuda_preflight_") as raw_log_dir:
        cuda_preflight_environment = build_isaac_process_environment(
            config,
            ros_log_dir=Path(raw_log_dir),
            base_environment=base_environment,
        )
        cuda_preflight_report = run_cuda_preflight(
            config,
            command_runner=cuda_preflight_runner,
            environment=cuda_preflight_environment,
        )
    # ``ros2 node list`` 会初始化 rclpy 并自动创建 ROS_LOG_DIR。若预检日志
    # 指向正式 fresh 输出，查询本身会抢先占用 output_dir，使原子 reserve
    # 必然失败；预检只允许写入独立临时目录。
    with tempfile.TemporaryDirectory(prefix="pct_scan_ros_preflight_") as raw_log_dir:
        preflight_environment = build_process_environment(
            config,
            ros_log_dir=Path(raw_log_dir),
            base_environment=base_environment,
        )
        preexisting_nodes = node_lister(preflight_environment)
    if preexisting_nodes:
        raise AcceptanceError(
            "指定 ROS_DOMAIN_ID 不是空 domain，拒绝混入外部图："
            f"{sorted(preexisting_nodes)}"
        )

    try:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise AcceptanceError(
            f"输出目录在预检期间被创建，拒绝复用：{config.output_dir}"
        ) from exc
    (config.output_dir / "ros_logs").mkdir()
    tuning_config_snapshot = create_tuning_config_snapshot(config)
    source_bundle_snapshot = create_source_bundle_snapshot(config)
    ros_environment = build_process_environment(
        config,
        base_environment=base_environment,
    )
    isaac_environment = build_isaac_process_environment(
        config,
        base_environment=base_environment,
    )
    run_id = uuid.uuid4().hex
    manifest_path = config.output_dir / "pct_scan_live_acceptance.json"
    launch_log_path = config.output_dir / "ros2_launch.log"
    manifest: dict[str, Any] = {
        "schema": "pct_scan_live_acceptance_v1",
        "run_id": run_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "status": "reserved",
        "mode": config.mode,
        # 主验收只要求 SCAN 使用实时地图完成局部绕障和路径恢复；原地转头
        # 属于可选感知策略，不能被写成产品链路的强制通过条件。
        "require_active_sensing": False,
        "expected_task_id": MODE_SPECS[config.mode].expected_task_id,
        "seed": config.seed,
        "output_dir": os.fspath(config.output_dir),
        "ros_domain_id": config.ros_domain_id,
        "navigation_body_height_m": config.navigation_body_height_m,
        "tuning_config_file": os.fspath(config.tuning_config_file),
        "tuning_config_snapshot": tuning_config_snapshot,
        "source_bundle_snapshot": source_bundle_snapshot,
        "scan_stair_freeze_profile": scan_stair_freeze_profile_audit,
        "pipeline_timeout_s": config.pipeline_timeout_s,
        "cuda_preflight": cuda_preflight_report,
        "pipeline_pythonpath_cleared": "PYTHONPATH" not in isaac_environment,
        "pipeline_command": list(pipeline_command),
        "launch_command": list(launch_command),
        "validator_command": list(validator_command),
    }
    _write_json(manifest_path, manifest)

    launch_process: ManagedProcess | None = None
    pipeline_process: ManagedProcess | None = None
    stop_report: StopReport | None = None
    pipeline_return_code: int | None = None
    launch_died_before_cleanup = False
    domain_cleaned = False
    try:
        with launch_log_path.open("w", encoding="utf-8") as launch_log:
            launch_process = popen_factory(
                launch_command,
                cwd=PROJECT_ROOT,
                env=ros_environment,
                stdout=launch_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            manifest.update(
                {
                    "status": "launch_starting",
                    "updated_at": _utc_now(),
                    "launch_pid": launch_process.pid,
                }
            )
            _write_json(manifest_path, manifest)
            _wait_for_required_nodes(
                launch_process,
                ros_environment,
                timeout_s=config.launch_ready_timeout_s,
                node_lister=node_lister,
                monotonic=monotonic,
                sleep=sleep,
            )
            manifest.update({"status": "pipeline_running", "updated_at": _utc_now()})
            _write_json(manifest_path, manifest)
            pipeline_process = popen_factory(
                pipeline_command,
                cwd=PROJECT_ROOT,
                env=isaac_environment,
                text=True,
                start_new_session=True,
            )
            manifest.update(
                {
                    "updated_at": _utc_now(),
                    "pipeline_pid": pipeline_process.pid,
                }
            )
            _write_json(manifest_path, manifest)
            try:
                pipeline_return_code = pipeline_process.wait(
                    timeout=config.pipeline_timeout_s
                )
            except subprocess.TimeoutExpired as exc:
                _interrupt_pipeline(
                    pipeline_process,
                    timeout_s=config.launch_stop_timeout_s,
                )
                raise AcceptanceError(
                    "pipeline 超过总时限且已请求有界清理："
                    f"timeout_s={config.pipeline_timeout_s}。"
                ) from exc
            except KeyboardInterrupt:
                _interrupt_pipeline(
                    pipeline_process,
                    timeout_s=config.launch_stop_timeout_s,
                )
                raise
            launch_died_before_cleanup = launch_process.poll() is not None
        if launch_process is not None:
            stop_report = stop_launch_process(
                launch_process,
                timeout_s=config.launch_stop_timeout_s,
            )
        if launch_died_before_cleanup:
            raise AcceptanceError("ros2 launch 在 pipeline 完成前已经退出。")
        if stop_report is None or not stop_report.clean_sigint_exit:
            raise AcceptanceError(
                "ros2 launch 未在 SIGINT 后干净退出："
                f"{None if stop_report is None else asdict(stop_report)}"
            )
        _wait_for_empty_domain(
            ros_environment,
            timeout_s=config.graph_cleanup_timeout_s,
            node_lister=node_lister,
            monotonic=monotonic,
            sleep=sleep,
        )
        domain_cleaned = True
        tuning_config_verification = verify_tuning_config_snapshot(
            tuning_config_snapshot
        )
        source_bundle_verification = verify_source_bundle_snapshot(
            source_bundle_snapshot
        )
        manifest["tuning_config_verification"] = tuning_config_verification
        manifest["source_bundle_verification"] = source_bundle_verification
        _write_json(manifest_path, manifest)
        if tuning_config_verification["verified"] is not True:
            raise AcceptanceError(
                "本轮统一调参源文件或快照在运行期间发生变化，拒绝验收："
                f"{tuning_config_verification}"
            )
        if source_bundle_verification["verified"] is not True:
            raise AcceptanceError(
                "本轮导航源码或bundle快照在运行期间发生变化，拒绝验收："
                f"{source_bundle_verification}"
            )
        if pipeline_return_code != 0:
            raise AcceptanceError(
                f"pipeline 子进程失败，return_code={pipeline_return_code}。"
            )
        startup_status = validate_startup_status(config)
        validate_episode_seed(config)
        manifest.update(
            {
                "status": "validating_summary",
                "updated_at": _utc_now(),
                "pipeline_return_code": pipeline_return_code,
                "launch_stop": asdict(stop_report),
                "startup_status": {
                    "status": startup_status["status"],
                    "exit_code": startup_status["exit_code"],
                    "mode": startup_status["mode"],
                    "navigation_body_height_m": startup_status[
                        "navigation_body_height_m"
                    ],
                },
            }
        )
        _write_json(manifest_path, manifest)
        validation = command_runner(
            validator_command,
            cwd=PROJECT_ROOT,
            env=isaac_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if validation.returncode != 0:
            detail = (validation.stderr or validation.stdout).strip()
            raise AcceptanceError(
                "live summary 校验失败："
                f"return_code={validation.returncode}, detail={detail}"
            )
        try:
            validation_report = json.loads(validation.stdout)
        except json.JSONDecodeError as exc:
            raise AcceptanceError("summary 校验器未返回合法 JSON。") from exc
        if not isinstance(validation_report, dict) or validation_report.get("valid") is not True:
            raise AcceptanceError("summary 校验器返回了非通过报告。")
        result = {
            "schema": "pct_scan_live_acceptance_result_v1",
            "run_id": run_id,
            "mode": config.mode,
            "seed": config.seed,
            "require_active_sensing": False,
            "valid": True,
            "output_dir": os.fspath(config.output_dir),
            "ros_domain_id": config.ros_domain_id,
            "navigation_body_height_m": config.navigation_body_height_m,
            "tuning_config_snapshot": tuning_config_snapshot,
            "tuning_config_verification": tuning_config_verification,
            "source_bundle_snapshot": source_bundle_snapshot,
            "source_bundle_verification": source_bundle_verification,
            "launch_pid": launch_process.pid,
            "pipeline_pid": pipeline_process.pid,
            "launch_stop": asdict(stop_report),
            "validation": validation_report,
        }
        manifest.update(
            {
                "status": "passed",
                "updated_at": _utc_now(),
                "result": result,
            }
        )
        _write_json(manifest_path, manifest)
        return result
    except BaseException as exc:
        cleanup_error: str | None = None
        if launch_process is not None and stop_report is None:
            stop_report = stop_launch_process(
                launch_process,
                timeout_s=config.launch_stop_timeout_s,
            )
        if (
            stop_report is not None
            and stop_report.signal_sent == "SIGINT"
            and not domain_cleaned
        ):
            try:
                _wait_for_empty_domain(
                    ros_environment,
                    timeout_s=config.graph_cleanup_timeout_s,
                    node_lister=node_lister,
                    monotonic=monotonic,
                    sleep=sleep,
                )
                domain_cleaned = True
            except BaseException as cleanup_exc:
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        tuning_config_verification = verify_tuning_config_snapshot(
            tuning_config_snapshot
        )
        source_bundle_verification = verify_source_bundle_snapshot(
            source_bundle_snapshot
        )
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "updated_at": _utc_now(),
                "pipeline_return_code": pipeline_return_code,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "launch_stop": None if stop_report is None else asdict(stop_report),
                "domain_cleaned": domain_cleaned,
                "cleanup_error": cleanup_error,
                "tuning_config_verification": tuning_config_verification,
                "source_bundle_verification": source_bundle_verification,
            }
        )
        _write_json(manifest_path, manifest)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在 fresh 输出目录与空 ROS domain 中运行一组 PCT→SCAN live 验收，"
            "并在 startup_status 成功后调用严格 summary 校验器。"
        ),
    )
    parser.add_argument("--mode", choices=tuple(MODE_SPECS), required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="唯一 episode 的确定性随机种子；会写入命令、清单和结果并强校验。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="必须原本不存在；脚本会原子创建并写入本轮清单。",
    )
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        required=True,
        help="必须是当前没有节点的独立 ROS_DOMAIN_ID。",
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=None,
        help="Isaac 环境 Python；默认读取 ISAAC_PYTHON，最后回退到当前解释器。",
    )
    parser.add_argument(
        "--navigation-body-height-m",
        type=float,
        default=None,
        help=(
            "生产导航唯一机体高度，单位 m；默认读取统一 YAML 的 "
            "navigation_contract.body_height_m，本脚本会把同一值同时传给 "
            "ROS 组合 launch 和 host pipeline。"
        ),
    )
    parser.add_argument(
        "--tuning-config-file",
        type=Path,
        default=DEFAULT_TUNING_CONFIG_PATH,
        help=(
            "ROS PCT/SCAN/controller 与 Isaac cmd_vel→policy 共用的统一 YAML。"
        ),
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=float,
        default=1800.0,
        help="pipeline 子进程总时限，单位秒；超时后执行有界清理。",
    )
    parser.add_argument("--launch-ready-timeout", type=float, default=20.0)
    parser.add_argument("--launch-stop-timeout", type=float, default=20.0)
    parser.add_argument("--graph-cleanup-timeout", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；仅完整验收通过返回零。"""

    arguments = _build_parser().parse_args(argv)
    isaac_python = arguments.isaac_python
    if isaac_python is None:
        isaac_python = Path(os.environ.get("ISAAC_PYTHON", sys.executable))
    tuning_config_file = arguments.tuning_config_file.expanduser().resolve()
    navigation_body_height_m = arguments.navigation_body_height_m
    if navigation_body_height_m is None:
        try:
            navigation_body_height_m = load_navigation_body_height_m(
                tuning_config_file
            )
        except AcceptanceError as exc:
            print(f"FAIL [config]：{exc}", file=sys.stderr)
            return 1
    config = AcceptanceConfig(
        mode=arguments.mode,
        output_dir=arguments.output_dir.expanduser().resolve(),
        ros_domain_id=arguments.ros_domain_id,
        isaac_python=isaac_python.expanduser().resolve(),
        tuning_config_file=tuning_config_file,
        seed=arguments.seed,
        navigation_body_height_m=navigation_body_height_m,
        pipeline_timeout_s=arguments.pipeline_timeout,
        launch_ready_timeout_s=arguments.launch_ready_timeout,
        launch_stop_timeout_s=arguments.launch_stop_timeout,
        graph_cleanup_timeout_s=arguments.graph_cleanup_timeout,
    )
    try:
        ros_workspace_environment = load_ros_workspace_environment()
        result = run_acceptance(
            config,
            base_environment=ros_workspace_environment,
        )
    except KeyboardInterrupt:
        print("INTERRUPTED：pipeline 与 ros2 launch 已请求有界清理。", file=sys.stderr)
        return 130
    except AcceptanceError as exc:
        print(f"FAIL [{config.mode}]：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
