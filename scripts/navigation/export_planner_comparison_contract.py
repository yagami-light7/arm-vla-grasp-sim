#!/usr/bin/env python3
"""从一次 PCT→SCAN fresh run 导出隔离 DWA 公平对照的不可变输入合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODE = "crossfloor_carry"
EXPECTED_POLICY_TASK = "RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0"
EXPECTED_TERMINATION = {
    "final_position_tolerance": 0.08,
    "place_position_tolerance": 0.08,
    "final_yaw_tolerance": 0.20,
    "stable_linear_velocity": 0.05,
    "stable_angular_velocity": 0.10,
    "require_yaw_alignment": True,
    "require_stable_base": True,
}
EXPECTED_FINISH_PARAMETERS = {
    "finish.distance_xy": 0.08,
    "finish.distance_z": 0.12,
    "finish.capture_stable_dwell_sec": 0.50,
    "finish.max_yaw_error": 0.20,
    "finish.max_planar_speed": 0.05,
    "finish.max_vertical_speed": 0.05,
    "finish.max_yaw_rate": 0.10,
    "finish.max_angular_speed": 0.10,
}


class ComparisonContractError(ValueError):
    """表示某次运行不足以形成可审计的 SCAN/DWA 公平输入。"""


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonContractError(
            f"无法读取{label} {path}：{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ComparisonContractError(f"{label}顶层必须是对象：{path}")
    return payload


def _load_yaml_mapping(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ComparisonContractError(
            f"无法读取{label} {path}：{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ComparisonContractError(f"{label}顶层必须是对象：{path}")
    return raw, payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonContractError(f"{label}必须是对象。")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonContractError(f"{label}必须是有限数。")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonContractError(f"{label}必须是有限数。")
    return result


def _finite_vector(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ComparisonContractError(f"{label}必须包含{size}个有限数。")
    return tuple(
        _finite_number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _path_points_sha256(points: Sequence[Sequence[float]]) -> str:
    """复用 ROS Path identity 的网络字节序双精度哈希语义。"""

    digest = hashlib.sha256()
    for index, point in enumerate(points):
        x, y, z = _finite_vector(point, 3, f"path[{index}]")
        digest.update(struct.pack("!ddd", x, y, z))
    return digest.hexdigest()


def _file_identity(path_value: object, label: str) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ComparisonContractError(f"{label}路径必须是非空字符串。")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ComparisonContractError(f"无法读取{label} {path}：{exc}") from exc
    return {
        "path": os.fspath(path),
        "sha256": _sha256_bytes(raw),
        "byte_count": len(raw),
    }


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_root_pose(root_pose: Sequence[float]) -> float:
    x, y, z, w, qx, qy, qz = _finite_vector(
        root_pose,
        7,
        "robot_root_pose",
    )
    del x, y, z
    return math.atan2(
        2.0 * (w * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _path_length_3d(points: Sequence[Sequence[float]]) -> float:
    return sum(
        math.dist(
            _finite_vector(left, 3, "path point"),
            _finite_vector(right, 3, "path point"),
        )
        for left, right in zip(points, points[1:])
    )


def _integer_number(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    """读取 JSON 整数，拒绝布尔值、分数和越界值。"""

    number = _finite_number(value, label)
    if not number.is_integer():
        raise ComparisonContractError(f"{label}必须是整数。")
    result = int(number)
    if minimum is not None and result < minimum:
        raise ComparisonContractError(f"{label}不得小于{minimum}。")
    return result


def _append_path_report(
    report: Mapping[str, Any],
    *,
    label: str,
    path_variants: dict[tuple[str, float], dict[str, Any]],
    require_complete_points: bool,
) -> bool:
    """校验并聚合一个完整 Path 报告；清空代际不计入路线几何。"""

    points_raw = report.get("points_ground_xyz")
    if "points_ground_xyz" not in report:
        if require_complete_points:
            raise ComparisonContractError(f"{label}缺少完整points_ground_xyz。")
        return False
    if report.get("cleared") is True:
        if points_raw != []:
            raise ComparisonContractError(
                f"{label}声明cleared=true但点列不是空数组。"
            )
        if report.get("points_sha256") != _path_points_sha256([]):
            raise ComparisonContractError(f"{label}的清空Path哈希不匹配。")
        return False
    if points_raw == []:
        raise ComparisonContractError(
            f"{label}点列为空但cleared没有明确为true。"
        )
    if not isinstance(points_raw, (list, tuple)) or len(points_raw) < 2:
        raise ComparisonContractError(f"{label}的非空参考Path必须至少包含两个点。")
    if report.get("cleared") is not False:
        raise ComparisonContractError(f"{label}.cleared必须明确为false。")
    points = tuple(
        _finite_vector(point, 3, f"{label}.points[{index}]")
        for index, point in enumerate(points_raw)
    )
    declared_sha = report.get("points_sha256")
    actual_sha = _path_points_sha256(points)
    if declared_sha != actual_sha:
        raise ComparisonContractError(
            f"{label}哈希不匹配："
            f"declared={declared_sha!r}, actual={actual_sha}。"
        )
    terminal_yaw = _finite_number(
        report.get("terminal_yaw"),
        f"{label}.terminal_yaw",
    )
    if report.get("source") != "ros2_nav_msgs_path":
        raise ComparisonContractError(f"{label}来源不是ros2_nav_msgs_path。")
    if report.get("topic") != "/pct/global_path":
        raise ComparisonContractError(f"{label} topic不是/pct/global_path。")
    if report.get("frame_id") != "world":
        raise ComparisonContractError(f"{label} frame_id不是world。")
    sequence = _integer_number(
        report.get("sequence"),
        f"{label}.sequence",
        minimum=0,
    )
    stamp = _mapping(report.get("stamp"), f"{label}.stamp")
    stamp_sec = _integer_number(
        stamp.get("sec"),
        f"{label}.stamp.sec",
        minimum=0,
    )
    stamp_nanosec = _integer_number(
        stamp.get("nanosec"),
        f"{label}.stamp.nanosec",
        minimum=0,
    )
    if stamp_nanosec >= 1_000_000_000:
        raise ComparisonContractError(f"{label}.stamp.nanosec超出合法范围。")
    stamp_ns = stamp_sec * 1_000_000_000 + stamp_nanosec
    key = (actual_sha, terminal_yaw)
    existing = path_variants.get(key)
    if existing is None:
        path_variants[key] = {
            "points_ground_xyz": [list(point) for point in points],
            "points_sha256": actual_sha,
            "terminal_yaw_rad": terminal_yaw,
            "source_topic": "/pct/global_path",
            "frame_id": "world",
            "source_stamp_ns_values": [stamp_ns],
            "source_sequence_values": [sequence],
        }
    else:
        if stamp_ns not in existing["source_stamp_ns_values"]:
            existing["source_stamp_ns_values"].append(stamp_ns)
        if sequence not in existing["source_sequence_values"]:
            existing["source_sequence_values"].append(sequence)
    return True


def _extract_path_and_initial_pose(
    frames_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """提取唯一完整 Path 与导航起姿；新运行优先读取单次快照文件。"""

    path_variants: dict[tuple[str, float], dict[str, Any]] = {}
    initial_pose: dict[str, Any] | None = None
    compact_path_report_count = 0
    snapshots_path = frames_path.parent / "navigation_path_snapshots.jsonl"
    snapshots_available = snapshots_path.is_file()
    try:
        stream = frames_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ComparisonContractError(f"无法读取诊断帧 {frames_path}：{exc}") from exc
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ComparisonContractError(
                    f"诊断帧第{line_number}行不是合法JSON：{exc}"
                ) from exc
            if not isinstance(frame, Mapping):
                raise ComparisonContractError(
                    f"诊断帧第{line_number}行顶层必须是对象。"
                )
            observation = _mapping(
                frame.get("observation"),
                f"frames[{line_number}].observation",
            )
            if (
                initial_pose is None
                and frame.get("pipeline_state") == "exec_nav_to_place"
            ):
                root_pose = _finite_vector(
                    observation.get("robot_root_pose"),
                    7,
                    f"frames[{line_number}].robot_root_pose",
                )
                initial_pose = {
                    "diagnostic_timestamp_s": _finite_number(
                        frame.get("timestamp"),
                        f"frames[{line_number}].timestamp",
                    ),
                    "base_xyz": list(root_pose[:3]),
                    "base_yaw_rad": _yaw_from_root_pose(root_pose),
                }
            if snapshots_available:
                continue
            metadata = _mapping(
                observation.get("metadata"),
                f"frames[{line_number}].observation.metadata",
            )
            raw_report = metadata.get("scan_reference_path_last_report")
            if raw_report is None:
                continue
            report = _mapping(
                raw_report,
                f"frames[{line_number}].scan_reference_path_last_report",
            )
            if "points_ground_xyz" not in report:
                compact_path_report_count += 1
                continue
            _append_path_report(
                report,
                label=f"frames[{line_number}].reference_path",
                path_variants=path_variants,
                require_complete_points=False,
            )
    if initial_pose is None:
        raise ComparisonContractError("诊断帧没有exec_nav_to_place起始姿态。")

    evidence_source: Path
    evidence_kind: str
    if snapshots_available:
        evidence_source = snapshots_path
        evidence_kind = "dedicated_navigation_path_snapshots_jsonl"
        try:
            snapshot_stream = snapshots_path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ComparisonContractError(
                f"无法读取Path快照 {snapshots_path}：{exc}"
            ) from exc
        with snapshot_stream:
            for line_number, raw_line in enumerate(snapshot_stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    snapshot = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ComparisonContractError(
                        f"Path快照第{line_number}行不是合法JSON：{exc}"
                    ) from exc
                snapshot = _mapping(
                    snapshot,
                    f"navigation_path_snapshots[{line_number}]",
                )
                if snapshot.get("schema") != "navigation_path_snapshot_v1":
                    raise ComparisonContractError(
                        f"Path快照第{line_number}行schema无效。"
                    )
                report = _mapping(
                    snapshot.get("report"),
                    f"navigation_path_snapshots[{line_number}].report",
                )
                declared_payload_sha = snapshot.get("report_payload_sha256")
                actual_payload_sha = _canonical_payload_sha256(report)
                if declared_payload_sha != actual_payload_sha:
                    raise ComparisonContractError(
                        f"Path快照第{line_number}行报告摘要不匹配："
                        f"declared={declared_payload_sha!r}, "
                        f"actual={actual_payload_sha}。"
                    )
                _append_path_report(
                    report,
                    label=f"navigation_path_snapshots[{line_number}].report",
                    path_variants=path_variants,
                    require_complete_points=True,
                )
    else:
        evidence_source = frames_path
        evidence_kind = "legacy_full_diagnostic_frame"
    if not path_variants:
        if snapshots_available:
            raise ComparisonContractError(
                "Path快照文件没有非空/pct/global_path完整点快照。"
            )
        if compact_path_report_count:
            raise ComparisonContractError(
                "历史运行没有完整Path点快照，只保留了Path哈希摘要；"
                "请使用带navigation_path_snapshots.jsonl的新运行重新验收。"
            )
        raise ComparisonContractError("诊断帧没有非空/pct/global_path。")
    if len(path_variants) != 1:
        hashes = sorted(key[0] for key in path_variants)
        raise ComparisonContractError(
            "公平静态对照要求唯一Path几何，本轮包含重规划或多代不同Path："
            f"{hashes}"
        )
    path = next(iter(path_variants.values()))
    path["source_stamp_ns_values"].sort()
    path["source_sequence_values"].sort()
    path["point_count"] = len(path["points_ground_xyz"])
    path["length_3d_m"] = _path_length_3d(path["points_ground_xyz"])
    path["height_semantics"] = "ground"
    path["evidence"] = {
        "kind": evidence_kind,
        **_file_identity(os.fspath(evidence_source), "Path证据文件"),
    }
    return path, initial_pose


def _read_tuning_snapshot(
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = manifest.get("tuning_config_snapshot")
    if isinstance(snapshot, Mapping):
        snapshot_path = Path(str(snapshot.get("snapshot_path", ""))).expanduser()
        if not snapshot_path.is_absolute():
            snapshot_path = run_dir / snapshot_path
        snapshot_path = snapshot_path.resolve()
        raw, payload = _load_yaml_mapping(snapshot_path, "调参快照")
        actual_sha = _sha256_bytes(raw)
        declared_sha = snapshot.get("sha256")
        if declared_sha != actual_sha:
            raise ComparisonContractError(
                "调参快照哈希不匹配："
                f"declared={declared_sha!r}, actual={actual_sha}。"
            )
        return payload, {
            "immutable_run_snapshot": True,
            "path": os.fspath(snapshot_path),
            "sha256": actual_sha,
            "byte_count": len(raw),
        }
    source_path_raw = manifest.get("tuning_config_file")
    if not isinstance(source_path_raw, str):
        raise ComparisonContractError("运行清单没有调参快照或源YAML路径。")
    source_path = Path(source_path_raw).expanduser().resolve()
    raw, payload = _load_yaml_mapping(source_path, "当前调参源文件")
    return payload, {
        "immutable_run_snapshot": False,
        "path": os.fspath(source_path),
        "sha256": _sha256_bytes(raw),
        "byte_count": len(raw),
        "warning": "当前源文件状态不是该历史运行的不可变参数证据",
    }


def _read_source_bundle_snapshot(
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """读取并自校验 fresh-run 固化的导航源码 bundle。"""

    identity = manifest.get("source_bundle_snapshot")
    verification = manifest.get("source_bundle_verification")
    if not isinstance(identity, Mapping) or not isinstance(verification, Mapping):
        return {
            "immutable_run_snapshot": False,
            "warning": "历史运行没有不可变导航源码bundle或结束复核",
        }
    snapshot_path = Path(str(identity.get("snapshot_path", ""))).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = run_dir / snapshot_path
    snapshot_path = snapshot_path.resolve()
    try:
        raw = snapshot_path.read_bytes()
        snapshot = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonContractError(
            f"无法读取源码bundle快照 {snapshot_path}：{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(snapshot, Mapping):
        raise ComparisonContractError("源码bundle快照顶层必须是对象。")
    if snapshot.get("schema") != "pct_scan_source_bundle_snapshot_v1":
        raise ComparisonContractError("源码bundle快照schema不受支持。")
    actual_snapshot_file_sha = _sha256_bytes(raw)
    if actual_snapshot_file_sha != identity.get("snapshot_file_sha256"):
        raise ComparisonContractError("源码bundle快照文件哈希不匹配。")
    files = snapshot.get("files")
    roots = snapshot.get("source_roots")
    if not isinstance(files, list) or not files:
        raise ComparisonContractError("源码bundle快照必须包含非空文件清单。")
    if not isinstance(roots, list) or not roots:
        raise ComparisonContractError("源码bundle快照必须包含非空根目录清单。")
    digest_payload = {
        "schema": "pct_scan_source_bundle_digest_v1",
        "source_roots": roots,
        "files": files,
    }
    actual_bundle_sha = _canonical_payload_sha256(digest_payload)
    if actual_bundle_sha != snapshot.get("sha256"):
        raise ComparisonContractError("源码bundle文件清单摘要不匹配。")
    for key in ("sha256", "file_count", "total_bytes", "source_roots"):
        if snapshot.get(key) != identity.get(key):
            raise ComparisonContractError(f"源码bundle身份字段{key}不匹配。")
    if snapshot.get("file_count") != len(files):
        raise ComparisonContractError("源码bundle文件数量不匹配。")
    byte_count = 0
    for index, entry in enumerate(files):
        mapping = _mapping(entry, f"source_bundle.files[{index}]")
        if not isinstance(mapping.get("path"), str) or not mapping.get("path"):
            raise ComparisonContractError("源码bundle文件路径无效。")
        sha = mapping.get("sha256")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(character not in "0123456789abcdef" for character in sha)
        ):
            raise ComparisonContractError("源码bundle逐文件SHA256无效。")
        byte_count += _integer_number(
            mapping.get("byte_count"),
            f"source_bundle.files[{index}].byte_count",
            minimum=0,
        )
    if byte_count != snapshot.get("total_bytes"):
        raise ComparisonContractError("源码bundle总字节数不匹配。")
    if (
        verification.get("verified") is not True
        or verification.get("expected_sha256") != actual_bundle_sha
        or verification.get("current_sha256") != actual_bundle_sha
        or verification.get("snapshot_file_sha256") != actual_snapshot_file_sha
    ):
        raise ComparisonContractError("源码bundle没有通过运行结束复核。")
    return {
        "immutable_run_snapshot": True,
        "path": os.fspath(snapshot_path),
        "sha256": actual_bundle_sha,
        "snapshot_file_sha256": actual_snapshot_file_sha,
        "file_count": len(files),
        "total_bytes": byte_count,
        "source_roots": roots,
    }


def _tuning_parameter(
    tuning: Mapping[str, Any],
    node: str,
    key: str,
) -> float:
    node_mapping = _mapping(tuning.get(node), f"tuning.{node}")
    parameters = _mapping(
        node_mapping.get("ros__parameters"),
        f"tuning.{node}.ros__parameters",
    )
    return _finite_number(parameters.get(key), f"tuning.{node}.{key}")


def _assert_expected_contract(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise ComparisonContractError(f"{label}缺少{key}。")
        actual_value = actual[key]
        if isinstance(expected_value, bool):
            matches = actual_value is expected_value
        else:
            try:
                matches = math.isclose(
                    _finite_number(actual_value, f"{label}.{key}"),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            raise ComparisonContractError(
                f"{label}.{key}不符合公平基准："
                f"expected={expected_value!r}, actual={actual_value!r}。"
            )


def _stair_contract(
    manifest: Mapping[str, Any],
    startup: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_stair = _mapping(
        manifest.get("scan_stair_freeze_profile"),
        "acceptance.scan_stair_freeze_profile",
    )
    startup_stair = _mapping(
        startup.get("scan_stair_freeze_profile_runtime"),
        "startup.scan_stair_freeze_profile_runtime",
    )
    required = (
        "profile_id",
        "contract_sha256",
        "source_sha256",
        "source_branch",
        "baseline_behavior",
        "non_physical_root_lock_workaround",
    )
    for key in required:
        if manifest_stair.get(key) != startup_stair.get(key):
            raise ComparisonContractError(
                f"acceptance与startup的楼梯冻结合同{key}不一致。"
            )
    if manifest_stair.get("baseline_behavior") != "chassis_root_lock":
        raise ComparisonContractError("公平对照只接受共同chassis_root_lock楼梯行为。")
    return {
        key: manifest_stair[key]
        for key in required
    }


def build_comparison_contract(
    run_dir_value: str | Path,
    *,
    allow_incomplete_source: bool = False,
) -> dict[str, Any]:
    """构造合同；默认只接受完整通过、参数不可变的跨层 SCAN 来源。"""

    run_dir = Path(run_dir_value).expanduser().resolve()
    manifest = _load_json_mapping(
        run_dir / "pct_scan_live_acceptance.json",
        "fresh验收清单",
    )
    startup = _load_json_mapping(run_dir / "startup_status.json", "启动状态")
    episode_dir = run_dir / "episode_000000"
    summary = _load_json_mapping(episode_dir / "summary.json", "episode汇总")
    path, initial_pose = _extract_path_and_initial_pose(
        episode_dir / "frames.jsonl"
    )
    tuning, tuning_identity = _read_tuning_snapshot(run_dir, manifest)
    source_revision_identity = _read_source_bundle_snapshot(run_dir, manifest)

    mode = manifest.get("mode")
    if mode != EXPECTED_MODE:
        raise ComparisonContractError(
            f"公平跨层对照要求mode={EXPECTED_MODE}，actual={mode!r}。"
        )
    if summary.get("seed") != manifest.get("seed"):
        raise ComparisonContractError("manifest与summary的seed不一致。")
    planner = _mapping(summary.get("latest_planner_result"), "latest_planner_result")
    navigation_execution = _mapping(
        planner.get("navigation_execution"),
        "latest_planner_result.navigation_execution",
    )
    _assert_expected_contract(
        navigation_execution,
        EXPECTED_TERMINATION,
        "navigation_execution",
    )
    finish_parameters = {
        key: _tuning_parameter(tuning, "scan_controller", key)
        for key in EXPECTED_FINISH_PARAMETERS
    }
    _assert_expected_contract(
        finish_parameters,
        EXPECTED_FINISH_PARAMETERS,
        "scan_controller.finish",
    )
    body_height = _tuning_parameter(
        tuning,
        "navigation_contract",
        "body_height_m",
    )
    if not math.isclose(
        body_height,
        _finite_number(
            manifest.get("navigation_body_height_m"),
            "manifest.navigation_body_height_m",
        ),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ComparisonContractError("YAML与运行清单的body_height_m不一致。")

    goal_request = _mapping(planner.get("pct_goal_request"), "pct_goal_request")
    goal_base_xyz = _finite_vector(
        goal_request.get("position_base_xyz"),
        3,
        "pct_goal_request.position_base_xyz",
    )
    goal_yaw = _finite_number(goal_request.get("yaw"), "pct_goal_request.yaw")
    if goal_request.get("height_semantics") != "base":
        raise ComparisonContractError("PCT goal必须使用base高度语义。")
    endpoint_ground = _finite_vector(
        path["points_ground_xyz"][-1],
        3,
        "reference_path.endpoint",
    )
    endpoint_base = (
        endpoint_ground[0],
        endpoint_ground[1],
        endpoint_ground[2] + body_height,
    )
    endpoint_xy_error = math.dist(endpoint_base[:2], goal_base_xyz[:2])
    endpoint_z_error = abs(endpoint_base[2] - goal_base_xyz[2])
    endpoint_yaw_error = abs(
        _wrap_angle(float(path["terminal_yaw_rad"]) - goal_yaw)
    )
    if endpoint_xy_error > 1.0e-6:
        raise ComparisonContractError("PCT Path末点XY没有精确绑定任务目标。")
    if endpoint_z_error > 1.0e-4:
        raise ComparisonContractError("PCT Path末点只加一次body height后不匹配目标。")
    if endpoint_yaw_error > 1.0e-5:
        raise ComparisonContractError("PCT Path末端yaw不匹配任务目标。")

    startup_defaults = _mapping(
        startup.get("scene_profile_defaults_applied"),
        "startup.scene_profile_defaults_applied",
    )
    policy_task = startup.get("locomotion_task")
    if policy_task != EXPECTED_POLICY_TASK:
        raise ComparisonContractError(
            "公平对照必须使用原Go2-X5 mobile-manipulation checkpoint任务："
            f"actual={policy_task!r}。"
        )
    if startup_defaults.get("locomotion_task") != policy_task:
        raise ComparisonContractError("scene默认与startup的locomotion_task不一致。")
    checkpoint_identity = _file_identity(
        startup.get("locomotion_checkpoint"),
        "locomotion checkpoint",
    )
    task_identity = _file_identity(startup.get("task_json"), "任务JSON")
    scene_profile_identity = _file_identity(
        startup.get("scene_profile_config_path"),
        "场景profile",
    )
    stair = _stair_contract(manifest, startup)
    executor = _mapping(summary.get("latest_executor_status"), "latest_executor_status")

    ineligibility_reasons: list[str] = []
    if manifest.get("status") != "passed":
        ineligibility_reasons.append("acceptance_manifest_not_passed")
    result = manifest.get("result")
    if not isinstance(result, Mapping) or result.get("valid") is not True:
        ineligibility_reasons.append("acceptance_result_not_valid")
    if startup.get("status") != "completed" or startup.get("exit_code") != 0:
        ineligibility_reasons.append("pipeline_startup_status_not_completed")
    if summary.get("success") is not True or summary.get("final_state") != "done":
        ineligibility_reasons.append("episode_not_successful")
    if executor.get("done") is not True or executor.get("success") is not True:
        ineligibility_reasons.append("navigation_executor_not_successful")
    if executor.get("scan_controller_goal_reached_verified") is not True:
        ineligibility_reasons.append("scan_goal_reached_not_verified")
    if executor.get("policy_zero_hold_verified") is not True:
        ineligibility_reasons.append("post_goal_zero_hold_not_verified")
    if executor.get("post_goal_nonzero_write_count") != 0:
        ineligibility_reasons.append("post_goal_nonzero_policy_write")
    if tuning_identity["immutable_run_snapshot"] is not True:
        ineligibility_reasons.append("missing_immutable_tuning_snapshot")
    if source_revision_identity["immutable_run_snapshot"] is not True:
        ineligibility_reasons.append("missing_immutable_source_bundle")
    eligible = not ineligibility_reasons
    if not allow_incomplete_source and not eligible:
        raise ComparisonContractError(
            "来源运行不具备公平对照资格：" + ",".join(ineligibility_reasons)
        )

    root_base = _finite_vector(initial_pose["base_xyz"], 3, "initial.base_xyz")
    path_start_ground = _finite_vector(
        path["points_ground_xyz"][0],
        3,
        "reference_path.start",
    )
    path_start_base = (
        path_start_ground[0],
        path_start_ground[1],
        path_start_ground[2] + body_height,
    )
    calibration = _mapping(
        _mapping(
            goal_request.get("effective_goal_provenance"),
            "goal.effective_goal_provenance",
        ).get("calibration"),
        "goal.effective_goal_provenance.calibration",
    )
    collision_ply_sha = calibration.get("collision_ply_sha256")
    if not isinstance(collision_ply_sha, str) or len(collision_ply_sha) != 64:
        raise ComparisonContractError("运行没有有效collision PLY SHA256。")

    contract: dict[str, Any] = {
        "schema": "pct_local_planner_comparison_contract_v1",
        "eligible_as_crossplanner_navigation_source": eligible,
        "ineligibility_reasons": ineligibility_reasons,
        "source_run": {
            "planner": "scan",
            "run_dir": os.fspath(run_dir),
            "run_id": manifest.get("run_id"),
            "mode": mode,
            "seed": summary.get("seed"),
            "task_id": summary.get("task_id"),
            "acceptance_status": manifest.get("status"),
            "episode_success": summary.get("success"),
            "episode_final_state": summary.get("final_state"),
        },
        "global_path": path,
        "initial_condition": {
            "seed": summary.get("seed"),
            "nav_stage_first_diagnostic_base_xyzyaw": [
                root_base[0],
                root_base[1],
                root_base[2],
                initial_pose["base_yaw_rad"],
            ],
            "nav_stage_first_diagnostic_timestamp_s": initial_pose[
                "diagnostic_timestamp_s"
            ],
            "serialized_path_start_base_xyz": list(path_start_base),
            "path_start_to_diagnostic_pose_xy_error_m": math.dist(
                path_start_base[:2],
                root_base[:2],
            ),
        },
        "goal": {
            "position_base_xyz": list(goal_base_xyz),
            "yaw_rad": goal_yaw,
            "serialized_path_endpoint_ground_xyz": list(endpoint_ground),
            "serialized_path_endpoint_base_xyz": list(endpoint_base),
            "endpoint_goal_xy_error_m": endpoint_xy_error,
            "endpoint_goal_z_error_m": endpoint_z_error,
            "endpoint_goal_yaw_error_rad": endpoint_yaw_error,
            "height_semantics": "path_ground_plus_body_height_once",
        },
        "termination_contract": {
            **{key: navigation_execution[key] for key in EXPECTED_TERMINATION},
            "finish_distance_z": finish_parameters["finish.distance_z"],
            "stable_dwell_s": finish_parameters[
                "finish.capture_stable_dwell_sec"
            ],
            "post_goal_zero_write_ticks": executor.get(
                "required_zero_write_ticks"
            ),
            "goal_reached_requires_fresh_odometry_and_point_cloud": True,
        },
        "time_accounting_contract": {
            "primary_metric": "planner_controlled_navigation_sim_time_s",
            "planner_controlled_start": "exec_nav_to_place_enter",
            "planner_controlled_intervals": [
                "scan_or_dwa_controls_policy_before_stair_takeover",
                "scan_or_dwa_controls_policy_after_stair_release_until_goal_reached",
            ],
            "excluded_from_primary": [
                "common_chassis_root_lock_stair_interval",
                "pipeline_startup_and_scene_build",
                "pick_and_place_manipulation",
            ],
            "required_secondary_metrics": [
                "full_navigation_stage_sim_time_s",
                "common_stair_freeze_sim_time_s",
                "planner_compute_wall_time_s",
                "yaw_only_command_sim_time_s",
                "terminal_capture_sim_time_s",
            ],
            "full_pipeline_must_still_reach_place_handoff": True,
        },
        "shared_runtime_contract": {
            "policy_task": policy_task,
            "policy_checkpoint": checkpoint_identity,
            "body_height_m": body_height,
            "tuning": tuning_identity,
            "source_revision": source_revision_identity,
            "task_json": task_identity,
            "scene_profile": scene_profile_identity,
            "collision_ply": {
                "path": calibration.get("collision_ply"),
                "sha256": collision_ply_sha,
                "sha256_provenance": "live_body_height_calibration",
            },
            "stair_freeze": stair,
            "mechanical_arm_navigation_posture": "stow_fixed",
            "dwa_must_run_in_isolated_legacy_worktree": True,
            "dwa_must_not_be_added_to_pct_scan_runtime": True,
        },
        "comparison_rules": {
            "same_serialized_path_points_sha256_required": True,
            "same_seed_required": True,
            "same_checkpoint_sha256_required": True,
            "same_collision_ply_sha256_required": True,
            "same_stair_freeze_contract_sha256_required": True,
            "same_termination_contract_required": True,
            "minimum_seed_count_for_claim": 3,
            # 导航比较必须先看能否到达，再比较成功episode的耗时。把失败前
            # 39s 当成比成功完成47s更快，会奖励撞停并扭曲用户要求的稳定导航。
            "evaluation_order": "success_rate_then_successful_completion_time",
            "higher_success_rate_is_superior": True,
            "equal_success_rate_requires_lower_primary_mean": True,
            "failed_run_time_is_never_completion_time": True,
        },
    }
    contract["contract_payload_sha256"] = _canonical_payload_sha256(contract)
    return contract


def verify_contract_payload(contract: Mapping[str, Any]) -> bool:
    """验证合同自身的规范JSON摘要，防止复制后静默改Path或门限。"""

    expected = contract.get("contract_payload_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    payload = dict(contract)
    payload.pop("contract_payload_sha256", None)
    return _canonical_payload_sha256(payload) == expected


def write_comparison_contract(
    contract: Mapping[str, Any],
    output_value: str | Path,
) -> Path:
    """以独占创建方式写合同，拒绝覆盖已有基准证据。"""

    if not verify_contract_payload(contract):
        raise ComparisonContractError("待写入合同的payload SHA256无效。")
    output = Path(output_value).expanduser().resolve()
    if output.exists():
        raise ComparisonContractError(f"输出文件必须原本不存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(
                contract,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    except OSError as exc:
        raise ComparisonContractError(f"无法写入公平对照合同 {output}：{exc}") from exc
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete-source",
        action="store_true",
        help="仅供诊断历史失败运行；输出会明确标记不可用于胜负结论。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI入口。"""

    arguments = _build_parser().parse_args(argv)
    try:
        contract = build_comparison_contract(
            arguments.run_dir,
            allow_incomplete_source=arguments.allow_incomplete_source,
        )
        output = write_comparison_contract(contract, arguments.output)
    except ComparisonContractError as exc:
        print(f"FAIL：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": os.fspath(output),
                "contract_payload_sha256": contract[
                    "contract_payload_sha256"
                ],
                "eligible": contract[
                    "eligible_as_crossplanner_navigation_source"
                ],
                "ineligibility_reasons": contract["ineligibility_reasons"],
                "path_points_sha256": contract["global_path"][
                    "points_sha256"
                ],
                "path_point_count": contract["global_path"]["point_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
