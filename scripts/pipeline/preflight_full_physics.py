#!/usr/bin/env python3
"""在不启动 Isaac Sim 的情况下检查 full-physics 运行前置条件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.tasks import JsonTaskProvider  # noqa: E402
from source.manipulation.planner_server_process import (  # noqa: E402
    planner_server_ping,
    planner_server_supports_required_features,
)
from source.recording.training_action import (  # noqa: E402
    task_requests_vla_training_action,
    validate_vla_training_action_config,
)
from source.simulation.isaaclab_runtime import (  # noqa: E402
    _task_world_collision_cuboids,
)
from source.simulation.receptacle_support import (  # noqa: E402
    resolve_task_receptacle_support_settings,
)
from source.simulation.scene_runtime import (  # noqa: E402
    resolve_scene_runtime_settings,
)


@dataclass(frozen=True)
class PreflightOptions:
    """描述一次只读运行前检查。"""

    task_json: Path
    global_planner: str = "astar"
    policy_profile: str = "flat"
    locomotion_checkpoint: Path | None = None
    pct_server_script: Path | None = None
    pct_tomogram_path: Path | None = None
    pct_walkable_path: Path | None = None
    pct_collision_ply_path: Path | None = None
    required_files: tuple[Path, ...] = ()
    required_prim_paths: tuple[str, ...] = ()
    collision_prim_path: str | None = None
    require_cuda: bool = True
    require_idle_runtime: bool = True
    minimum_free_gb: float = 1.5


def _project_path(raw_path: str | Path | None) -> Path | None:
    """解析相对路径；其他 worktree 只能通过显式绝对路径提供。"""

    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _path_report(path: Path | None) -> dict[str, Any]:
    """返回路径存在性和文件大小，便于报告直接追溯资产。"""

    if path is None:
        return {"configured": False, "path": None, "exists": False}
    exists = path.exists()
    return {
        "configured": True,
        "path": str(path),
        "exists": exists,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _file_sha256(path: Path) -> str:
    """流式计算运行资产哈希，供任务代理来源校验。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_usda_text(path: Path) -> str | None:
    """只读取文本 USDA；二进制 USD 返回 None，不猜测其 prim 结构。"""

    if not path.is_file() or path.suffix.lower() not in {".usd", ".usda"}:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.lstrip().startswith(b"#usda"):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _usda_prim_leaf_marker(text: str, prim_path: str) -> dict[str, Any]:
    """识别 typed/untyped def 与 over；这里只做保守 leaf 级静态检查。"""

    leaf = prim_path.rstrip("/").rsplit("/", 1)[-1]
    pattern = (
        r"(?m)^\s*(?:def|over)(?:\s+[A-Za-z_][A-Za-z0-9_]*)?"
        rf'\s+"{re.escape(leaf)}"(?:\s|\(|$)'
    )
    matched = re.search(pattern, text) is not None
    return {
        "status": "present_text_marker" if matched else "missing_text_marker",
        "leaf": leaf,
        "pattern": pattern,
    }


def _scene_marker_report(
    scene_path: Path,
    *,
    required_prim_paths: Iterable[str],
    collision_prim_path: str | None,
) -> dict[str, Any]:
    """用保守的文本 marker 检查 USDA；无法解析时明确标记 unknown。"""

    text = _read_usda_text(scene_path)
    required = tuple(str(path) for path in required_prim_paths if str(path).strip())
    if text is None:
        return {
            "mode": "unknown_binary_or_unreadable",
            "required_prim_paths": {
                path: {"status": "unknown", "leaf": path.rsplit("/", 1)[-1]}
                for path in required
            },
            "collision_prim": (
                {"path": collision_prim_path, "status": "unknown"}
                if collision_prim_path
                else None
            ),
        }

    prim_reports: dict[str, dict[str, Any]] = {}
    for prim_path in required:
        prim_reports[prim_path] = _usda_prim_leaf_marker(text, prim_path)

    collision_report = None
    if collision_prim_path:
        collision_report = _usda_prim_leaf_marker(text, collision_prim_path)
        collision_report["path"] = collision_prim_path

    return {
        "mode": "usda_text_marker",
        "required_prim_paths": prim_reports,
        "collision_prim": collision_report,
    }


def _task_collision_proxy_report(task_spec: Any) -> dict[str, Any]:
    """离线解析任务 collision proxy，确认 required 配置不会静默丢失。"""

    import numpy as np

    phase_references = {
        "pick": (
            task_spec.object_initial_pose[:3]
            if task_spec.object_initial_pose is not None
            else (task_spec.pick_goal.x, task_spec.pick_goal.y, task_spec.pick_goal.z)
        ),
        "place": (
            task_spec.place_target_pose[:3]
            if task_spec.place_target_pose is not None
            else (
                None
                if task_spec.place_goal is None
                else (
                    task_spec.place_goal.x,
                    task_spec.place_goal.y,
                    task_spec.place_goal.z,
                )
            )
        ),
    }
    phases: dict[str, Any] = {}
    for phase, reference in phase_references.items():
        raw_phase = task_spec.raw_task.get(phase) or {}
        configured = isinstance(raw_phase, dict) and (
            "curobo_world_collision" in raw_phase
        )
        if not configured:
            phases[phase] = {
                "configured": False,
                "obstacle_count": 0,
                "required_count": 0,
            }
            continue
        if reference is None:
            raise RuntimeError(f"task.{phase} 配置了 collision proxy，但缺少参考位姿")
        cuboids = _task_world_collision_cuboids(
            raw_task=task_spec.raw_task,
            phase=phase,
            T_world_base=np.eye(4, dtype=float),
            reference_point=reference,
            padding_xy_m=0.02,
            padding_z_m=0.02,
        )
        phases[phase] = {
            "configured": True,
            "obstacle_count": len(cuboids),
            "required_count": sum(
                1 for cuboid in cuboids if cuboid.get("task_collision_required")
            ),
            "collision_ids": [
                str(cuboid.get("task_collision_id")) for cuboid in cuboids
            ],
            "semantic_roles": [
                str(cuboid.get("semantic_role")) for cuboid in cuboids
            ],
            "padding_modes": [
                str(cuboid.get("padding_mode")) for cuboid in cuboids
            ],
            "source_collision_ply_sha256": sorted(
                {
                    str((cuboid.get("source") or {}).get("collision_ply_sha256"))
                    for cuboid in cuboids
                    if (cuboid.get("source") or {}).get("collision_ply_sha256")
                }
            ),
            "source_prim_paths": sorted(
                {
                    str(cuboid.get("source_prim_path"))
                    for cuboid in cuboids
                    if cuboid.get("source_prim_path")
                }
            ),
            "source_assets": sorted(
                [
                    {
                        "path": str((cuboid.get("source") or {}).get("asset_path")),
                        "sha256": str(
                            (cuboid.get("source") or {}).get("asset_sha256") or ""
                        ),
                    }
                    for cuboid in cuboids
                    if (cuboid.get("source") or {}).get("asset_path")
                ],
                key=lambda item: (item["path"], item["sha256"]),
            ),
        }
    return {
        "runtime_defaults": {
            "padding_xy_m": 0.02,
            "padding_z_m": 0.02,
        },
        "phases": phases,
    }


def _cuda_report() -> dict[str, Any]:
    """读取 CUDA 状态；只做探测，不创建 Isaac Sim 或 CUDA context。"""

    report: dict[str, Any] = {
        "torch_imported": False,
        "torch_version": None,
        "torch_cuda_available": None,
        "torch_device_count": None,
        "torch_cuda_runtime": None,
        "nvidia_smi": None,
    }
    try:
        import torch  # type: ignore

        report.update(
            {
                "torch_imported": True,
                "torch_version": str(torch.__version__),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_device_count": int(torch.cuda.device_count()),
                "torch_cuda_runtime": torch.version.cuda,
            }
        )
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        report["torch_import_error"] = f"{type(exc).__name__}: {exc}"

    executable = shutil.which("nvidia-smi")
    if executable is None:
        report["nvidia_smi"] = {"available": False, "reason": "command_not_found"}
        return report
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        report["nvidia_smi"] = {
            "available": result.returncode == 0,
            "returncode": int(result.returncode),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - 取决于主机状态
        report["nvidia_smi"] = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return report


def _background_process_report() -> dict[str, Any]:
    """报告可见的 Isaac/CuRobo/PCT runtime，绝不执行终止或清理。"""

    try:
        import psutil  # type: ignore
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        return {"available": False, "processes": [], "error": str(exc)}

    processes: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            pid = int(process.info.get("pid") or 0)
            if pid == os.getpid():
                continue
            name = str(process.info.get("name") or "")
            cmdline = " ".join(str(item) for item in (process.info.get("cmdline") or []))
            haystack = f"{name} {cmdline}".lower()
            category = None
            if "grasp_planner_server.py" in haystack:
                category = "curobo_server"
            elif "pct_grid_server.py" in haystack or "pct_server.py" in haystack:
                category = "pct_server"
            elif "run_full_physics_pipeline.py" in haystack:
                category = "isaac_full_physics_pipeline"
            elif (
                name.lower() in {"isaacsim", "kit", "kit.exe"}
                or "omni.kit.app" in haystack
                or "/isaac-sim" in haystack
            ):
                category = "isaac_runtime"
            if category is not None:
                processes.append(
                    {
                        "pid": pid,
                        "name": name,
                        "cmdline": cmdline,
                        "cwd": process.info.get("cwd"),
                        "category": category,
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "available": True,
        "processes": processes,
        "blocking_process_count": len(processes),
    }


def _apply_runtime_process_gate(
    background_report: dict[str, Any],
) -> dict[str, Any]:
    """把能力兼容的单个共享 CuRobo server 从独占 runtime 中排除。"""

    processes = list(background_report.get("processes") or [])
    curobo_processes = [
        process
        for process in processes
        if process.get("category") == "curobo_server"
    ]
    blocking_processes = [
        process
        for process in processes
        if process.get("category") != "curobo_server"
    ]
    allowed_processes: list[dict[str, Any]] = []
    shared_server_report: dict[str, Any] = {
        "detected_process_count": len(curobo_processes),
        "ping_ok": False,
        "required_capabilities_verified": False,
        "allowed": False,
        "ownership": "external_shared_service",
        "shutdown_allowed": False,
    }
    if len(curobo_processes) == 1:
        ping_ok = planner_server_ping()
        capabilities_ok = bool(
            ping_ok and planner_server_supports_required_features()
        )
        shared_server_report.update(
            {
                "ping_ok": ping_ok,
                "required_capabilities_verified": capabilities_ok,
                "allowed": capabilities_ok,
            }
        )
        if capabilities_ok:
            allowed_processes.extend(curobo_processes)
        else:
            blocking_processes.extend(curobo_processes)
    elif curobo_processes:
        shared_server_report["reason"] = "multiple_curobo_server_processes"
        blocking_processes.extend(curobo_processes)

    return {
        **background_report,
        "processes": processes,
        "allowed_processes": allowed_processes,
        "allowed_process_count": len(allowed_processes),
        "blocking_processes": blocking_processes,
        "blocking_process_count": len(blocking_processes),
        "shared_curobo_server": shared_server_report,
    }


def build_preflight_report(options: PreflightOptions) -> dict[str, Any]:
    """构造完整前置检查报告；报告生成过程不会启动仿真。"""

    task_path = _project_path(options.task_json)
    report: dict[str, Any] = {
        "schema_version": 2,
        "project_root": str(PROJECT_ROOT),
        "task_json": str(task_path) if task_path else None,
        "global_planner": options.global_planner,
        "policy_profile": options.policy_profile,
        "checks": {},
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    task_spec = None
    scene_runtime_settings: dict[str, Any] | None = None
    if task_path is None or not task_path.is_file():
        errors.append(f"task JSON 不存在: {task_path}")
    else:
        report["checks"]["task_json"] = _path_report(task_path)
        try:
            task_spec = JsonTaskProvider().load(task_path)
            training_action_config = validate_vla_training_action_config(
                task_spec.raw_task
            )
            report["task"] = {
                "task_id": task_spec.task_id,
                "episode_id": task_spec.episode_id,
                "instruction": task_spec.instruction,
                "object_prim_path": task_spec.object_prim_path,
                "target_receptacle_id": task_spec.raw_task.get(
                    "target_receptacle_id"
                ),
                "target_receptacle_prim_path": task_spec.raw_task.get(
                    "target_receptacle_prim_path"
                ),
                "scene_usd": task_spec.scene_usd,
                "nav_map": task_spec.nav_map,
                "has_place_goal": task_spec.place_goal is not None,
                "has_place_target_pose": task_spec.place_target_pose is not None,
                "vla_training_action_requested": task_requests_vla_training_action(
                    task_spec.raw_task
                ),
                "training_action": training_action_config,
            }
            if training_action_config is None:
                warnings.append("task 未声明 10 维 VLA training_action；只能导出旧控制动作。")
        except Exception as exc:
            errors.append(f"task JSON 无法加载: {type(exc).__name__}: {exc}")

    if task_spec is not None:
        try:
            receptacle_support_settings = (
                resolve_task_receptacle_support_settings(task_spec.raw_task)
            )
            report["checks"]["task_receptacle_support"] = (
                receptacle_support_settings
            )
            report["task"]["receptacle_support_runtime_validation"] = {
                "configured": receptacle_support_settings["configured"],
                "required": receptacle_support_settings[
                    "runtime_validation_required"
                ],
                "support_expected_static": receptacle_support_settings.get(
                    "support_expected_static"
                ),
            }
        except Exception as exc:
            errors.append(
                "task receptacle support 无法解析: "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            scene_runtime_settings = resolve_scene_runtime_settings(
                task_spec.raw_task,
                default_collision_prim_path=(
                    options.collision_prim_path or "/World/scene_collision"
                ),
                default_visual_prim_path="/World/gauss",
                default_collision_floor_proxy_profile=(
                    "yinluyuan_f2"
                    if options.policy_profile == "pct_multifloor"
                    else None
                ),
            )
            report["task"]["scene_runtime"] = scene_runtime_settings
            configured_runtime = task_spec.raw_task.get("scene_runtime")
            has_task_collision_override = (
                isinstance(configured_runtime, dict)
                and "collision_prim_path" in configured_runtime
            )
            resolved_collision_prim_path = str(
                scene_runtime_settings["collision_prim_path"]
            )
            if (
                has_task_collision_override
                and options.collision_prim_path
                and options.collision_prim_path != resolved_collision_prim_path
            ):
                errors.append(
                    "CLI collision prim 与 task.scene_runtime 不一致: "
                    f"cli={options.collision_prim_path} "
                    f"task={resolved_collision_prim_path}"
                )
        except Exception as exc:
            errors.append(
                "task scene_runtime 无法解析: "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            report["checks"]["task_curobo_world_collision"] = (
                _task_collision_proxy_report(task_spec)
            )
            source_asset_checks: dict[str, Any] = {}
            for phase_report in report["checks"][
                "task_curobo_world_collision"
            ]["phases"].values():
                for source_asset in phase_report.get("source_assets", []):
                    source_path = _project_path(source_asset.get("path"))
                    if source_path is None:
                        continue
                    expected_sha256 = str(source_asset.get("sha256") or "")
                    check = _path_report(source_path)
                    check["expected_sha256"] = expected_sha256 or None
                    if source_path.is_file():
                        actual_sha256 = _file_sha256(source_path)
                        check["actual_sha256"] = actual_sha256
                        check["sha256_matches"] = bool(
                            not expected_sha256 or actual_sha256 == expected_sha256
                        )
                        if not check["sha256_matches"]:
                            errors.append(
                                "task collision proxy 来源资产 hash 不一致: "
                                f"{source_path} actual={actual_sha256} "
                                f"expected={expected_sha256}"
                            )
                    else:
                        check["actual_sha256"] = None
                        check["sha256_matches"] = False
                        errors.append(
                            f"task collision proxy 来源资产不存在: {source_path}"
                        )
                    source_asset_checks[str(source_path)] = check
            report["checks"]["task_collision_source_assets"] = (
                source_asset_checks
            )
        except Exception as exc:
            errors.append(
                "task CuRobo world collision 无法解析: "
                f"{type(exc).__name__}: {exc}"
            )

    scene_path: Path | None = None
    nav_map_path: Path | None = None
    if task_spec is not None:
        scene_path = _project_path(task_spec.scene_usd)
        nav_map_path = _project_path(task_spec.nav_map)
        report["checks"]["scene_usd"] = _path_report(scene_path)
        if scene_path is None or not scene_path.is_file():
            errors.append(f"scene USD 不存在: {scene_path}")
        else:
            collision_prim_path = (
                str(scene_runtime_settings["collision_prim_path"])
                if scene_runtime_settings is not None
                else options.collision_prim_path
            )
            runtime_required_prim_paths: tuple[str, ...] = ()
            if (
                scene_runtime_settings is not None
                and scene_runtime_settings["task_override_present"]
            ):
                runtime_required_prim_paths = (
                    str(scene_runtime_settings["visual_prim_path"]),
                )
            raw_place = task_spec.raw_task.get("place") or {}
            receptacle_prim_paths = tuple(
                str(path)
                for path in (
                    task_spec.raw_task.get("target_receptacle_prim_path"),
                    (
                        raw_place.get("target_receptacle_prim_path")
                        if isinstance(raw_place, dict)
                        else None
                    ),
                    (
                        raw_place.get("target_support_prim_path")
                        if isinstance(raw_place, dict)
                        else None
                    ),
                )
                if path
            )
            scene_report = _scene_marker_report(
                scene_path,
                required_prim_paths=(
                    *((task_spec.object_prim_path,) if task_spec.object_prim_path else ()),
                    *runtime_required_prim_paths,
                    *receptacle_prim_paths,
                    *options.required_prim_paths,
                ),
                collision_prim_path=collision_prim_path,
            )
            report["checks"]["scene_structure"] = scene_report
            for prim_path, prim_report in scene_report["required_prim_paths"].items():
                if prim_report["status"] == "missing_text_marker":
                    errors.append(f"scene 中未找到目标 prim marker: {prim_path}")
                elif prim_report["status"] == "unknown":
                    warnings.append(f"无法静态检查二进制 USD 的 prim: {prim_path}")
            collision_report = scene_report.get("collision_prim")
            if collision_report and collision_report["status"] == "missing_text_marker":
                errors.append(
                    f"scene 中未找到 collision prim marker: {collision_prim_path}"
                )
            elif collision_report and collision_report["status"] == "unknown":
                warnings.append(
                    f"无法静态检查二进制 USD 的 collision prim: {collision_prim_path}"
                )

        report["checks"]["task_nav_map"] = _path_report(nav_map_path)
        if options.global_planner == "astar" and (
            nav_map_path is None or not nav_map_path.is_file()
        ):
            errors.append(f"A* planner 需要有效 nav_map: {nav_map_path}")
        elif nav_map_path is None or not nav_map_path.is_file():
            warnings.append("PCT 模式没有 flat nav_map；将不能启用 A* fallback。")

        if not task_spec.object_prim_path:
            errors.append("task 没有 object_prim_path")
        if task_spec.place_goal is None or task_spec.place_target_pose is None:
            errors.append("full-physics task 必须同时提供 place goal 和 place target pose")

    resolved_required_files = tuple(
        path for path in (_project_path(item) for item in options.required_files) if path is not None
    )
    report["checks"]["required_files"] = {
        str(path): _path_report(path) for path in resolved_required_files
    }
    for path in resolved_required_files:
        if not path.is_file():
            errors.append(f"要求的资产文件不存在: {path}")

    if options.global_planner == "pct":
        pct_paths = {
            "server_script": _project_path(options.pct_server_script),
            "tomogram": _project_path(options.pct_tomogram_path),
            "walkable": _project_path(options.pct_walkable_path),
            "collision_ply": _project_path(options.pct_collision_ply_path),
        }
        report["checks"]["pct_assets"] = {
            name: _path_report(path) for name, path in pct_paths.items()
        }
        for name in ("server_script", "tomogram", "walkable"):
            path = pct_paths[name]
            if path is None or not path.is_file():
                errors.append(f"PCT {name} 不存在: {path}")
        if pct_paths["collision_ply"] is None or not pct_paths["collision_ply"].is_file():
            warnings.append("未提供 PCT collision PLY；局部障碍层只能使用 tomogram/walkable。")
        else:
            actual_hash = _file_sha256(pct_paths["collision_ply"])
            task_collision_report = report["checks"].get(
                "task_curobo_world_collision",
                {},
            )
            expected_hashes = sorted(
                {
                    str(expected_hash)
                    for phase_report in (
                        task_collision_report.get("phases", {}) or {}
                    ).values()
                    for expected_hash in phase_report.get(
                        "source_collision_ply_sha256",
                        [],
                    )
                }
            )
            hash_matches = not expected_hashes or actual_hash in expected_hashes
            report["checks"]["pct_collision_ply_hash"] = {
                "path": str(pct_paths["collision_ply"]),
                "actual_sha256": actual_hash,
                "task_proxy_expected_sha256": expected_hashes,
                "matches_task_proxy_source": hash_matches,
            }
            if not hash_matches:
                errors.append(
                    "PCT collision PLY hash 与 task CuRobo collision proxy 来源不一致: "
                    f"actual={actual_hash} expected={expected_hashes}"
                )

    checkpoint_path = _project_path(options.locomotion_checkpoint)
    report["checks"]["locomotion_checkpoint"] = _path_report(checkpoint_path)
    if options.policy_profile == "pct_multifloor":
        if checkpoint_path is None or not checkpoint_path.is_file():
            errors.append(f"pct_multifloor checkpoint 不存在: {checkpoint_path}")

    cuda = _cuda_report()
    report["checks"]["cuda"] = cuda
    if options.require_cuda and not bool(cuda.get("torch_cuda_available")):
        errors.append("当前 Python/主机没有可用 CUDA device，不能运行真实 Isaac full-physics。")

    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = float(usage.free) / (1024.0**3)
    report["checks"]["disk"] = {
        "free_bytes": int(usage.free),
        "free_gb": round(free_gb, 3),
        "minimum_free_gb": float(options.minimum_free_gb),
    }
    if free_gb < float(options.minimum_free_gb):
        errors.append(
            f"磁盘剩余空间不足: {free_gb:.2f} GiB < {float(options.minimum_free_gb):.2f} GiB"
        )

    background_report = _apply_runtime_process_gate(_background_process_report())
    report["checks"]["background_processes"] = background_report
    runtime_processes = background_report.get("processes") or []
    allowed_runtime_processes = background_report.get("allowed_processes") or []
    blocking_runtime_processes = background_report.get("blocking_processes") or []
    asset_checks_passed = not errors
    if allowed_runtime_processes:
        warnings.append(
            "检测到能力兼容的共享 CuRobo server；允许复用，pipeline 不拥有且不会关闭该服务。"
        )
    if blocking_runtime_processes:
        warnings.append(
            "检测到阻塞型 Isaac/PCT 或不兼容 CuRobo runtime；本检查未执行任何终止操作。"
        )
        if options.require_idle_runtime:
            errors.append(
                "检测到其他 Isaac/PCT 或不兼容 CuRobo runtime；按启动门禁禁止并发启动新仿真。"
            )
    elif runtime_processes and not allowed_runtime_processes:
        warnings.append("检测到未分类 runtime；按保守策略不允许启动。")
        if options.require_idle_runtime:
            errors.append("检测到未分类 runtime；按启动门禁禁止并发启动新仿真。")

    report["asset_checks_passed"] = asset_checks_passed
    report["runtime_launch_safe"] = bool(
        asset_checks_passed and not blocking_runtime_processes
    )
    report["ready_for_real_full_physics"] = not errors
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读检查 full-physics pipeline 的任务、资产、CUDA、磁盘和后台进程。",
    )
    parser.add_argument("--task-json", required=True, help="任务 JSON 路径。")
    parser.add_argument("--global-planner", choices=("astar", "pct"), default="astar")
    parser.add_argument("--policy-profile", choices=("flat", "pct_multifloor"), default="flat")
    parser.add_argument("--locomotion-checkpoint", help="locomotion checkpoint 路径。")
    parser.add_argument("--pct-server-script", help="PCT server 脚本路径。")
    parser.add_argument("--pct-tomogram-path", help="PCT tomogram pickle 路径。")
    parser.add_argument("--pct-walkable-path", help="PCT walkable npy 路径。")
    parser.add_argument("--pct-collision-ply-path", help="PCT collision PLY 路径。")
    parser.add_argument(
        "--required-file",
        action="append",
        default=[],
        help="额外要求存在的资产文件；可重复传入。",
    )
    parser.add_argument(
        "--required-prim-path",
        action="append",
        default=[],
        help="要求 scene 中出现的 prim leaf marker；可重复传入。",
    )
    parser.add_argument("--collision-prim-path", help="要求 scene 中出现的 collision prim leaf marker。")
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="真实运行时要求 CUDA；CPU 资产检查可用 --no-require-cuda。",
    )
    parser.add_argument(
        "--require-idle-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "默认要求没有其他 Isaac/PCT 或不兼容 CuRobo runtime；能力兼容的单个共享 "
            "CuRobo server 可复用。只做资产盘点可用 --no-require-idle-runtime。"
        ),
    )
    parser.add_argument("--minimum-free-gb", type=float, default=1.5)
    parser.add_argument("--output-json", help="可选的 JSON 报告路径。")
    return parser


def _print_summary(report: dict[str, Any]) -> None:
    """打印短摘要，同时保留完整 JSON 供 CI 或人工检查。"""

    status = "READY" if report["ready_for_real_full_physics"] else "BLOCKED"
    print(f"[preflight] {status}")
    for item in report["errors"]:
        print(f"[ERROR] {item}")
    for item in report["warnings"]:
        print(f"[WARN] {item}")
    cuda = report["checks"].get("cuda", {})
    disk = report["checks"].get("disk", {})
    print(
        "[preflight] "
        f"cuda={cuda.get('torch_cuda_available')} "
        f"free_gb={disk.get('free_gb')} "
        f"background_processes={len(report['checks'].get('background_processes', {}).get('processes', []))} "
        f"blocking_processes={report['checks'].get('background_processes', {}).get('blocking_process_count')} "
        f"assets_ready={report.get('asset_checks_passed')} "
        f"runtime_launch_safe={report.get('runtime_launch_safe')}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    options = PreflightOptions(
        task_json=Path(args.task_json),
        global_planner=str(args.global_planner),
        policy_profile=str(args.policy_profile),
        locomotion_checkpoint=_project_path(args.locomotion_checkpoint),
        pct_server_script=_project_path(args.pct_server_script),
        pct_tomogram_path=_project_path(args.pct_tomogram_path),
        pct_walkable_path=_project_path(args.pct_walkable_path),
        pct_collision_ply_path=_project_path(args.pct_collision_ply_path),
        required_files=tuple(Path(item) for item in args.required_file),
        required_prim_paths=tuple(str(item) for item in args.required_prim_path),
        collision_prim_path=args.collision_prim_path,
        require_cuda=bool(args.require_cuda),
        require_idle_runtime=bool(args.require_idle_runtime),
        minimum_free_gb=float(args.minimum_free_gb),
    )
    report = build_preflight_report(options)
    if args.output_json:
        output_path = _project_path(args.output_json)
        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[preflight] report={output_path}")
    _print_summary(report)
    return 0 if report["ready_for_real_full_physics"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
