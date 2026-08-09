#!/usr/bin/env python3
"""从真实 PCT→SCAN episode 中提取可复核的导航分段耗时。

分析器只读取 ``frames.jsonl``、``summary.json`` 和同一 fresh-run 的
``ros2_launch.log``，不导入 ROS 2、Isaac Sim 或 GPU 运行时。第一阶段 F1
定义为 ``exec_nav_to_place`` 开始至首次楼梯接管；其中 SCAN 控制段从唯一
policy writer 首次写入非零 ``scan_ros2_navigation`` 命令开始计算。

输出中的平移、仅转向和零命令时间按相邻诊断帧的仿真时间差积分。结果会同时
报告最大采样间隔，不能把低频诊断近似冒充逐控制拍精确统计。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PLANNER_TOTAL_TIME = re.compile(
    r"total\s+time:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_FINAL_PLAN_SUCCESS = re.compile(r"final_plan_success\s*=\s*([01])")
_STAIR_TAKEOVER_PREFIXES = (
    "scan_stair_sensor_acquisition",
    "scan_stair_freeze",
)
_COMMAND_EPSILON = 1.0e-9
# 低于 0.02m/s 的斜向残量不足以形成原 Go2-X5 的可辨平移，统计为转向过渡。
_TRANSLATION_ACTIVITY_THRESHOLD_MPS = 0.02
# 诊断帧以 0.2s 抽样，角速度限幅前后会出现一个加减速样本；留 0.02rad/s
# 余量统计“受限幅约束的转向时间”，而不是只数浮点上恰好等于上限的样本。
_YAW_SATURATION_MARGIN_RADPS = 0.02


class TimingInputError(ValueError):
    """表示运行产物不足以形成可信的分段耗时。"""


@dataclass(frozen=True, slots=True)
class FrameSample:
    """保留分段统计所需的最小诊断帧字段。"""

    timestamp_s: float
    pipeline_state: str
    action_source: str
    motion_allowed: bool
    command: tuple[float, float, float] | None
    body_planar_speed_mps: float | None
    body_yaw_rate_radps: float | None
    pose_xy: tuple[float, float] | None
    controller_state: int | None
    controller_is_final: bool | None


@dataclass(frozen=True, slots=True)
class PlannerPrefixReport:
    """汇总首次楼梯冻结前的 SCAN 规划日志。"""

    attempt_count: int
    success_count: int
    failure_count: int
    successful_total_wall_s: float
    successful_mean_wall_s: float | None
    successful_p95_wall_s: float | None
    log_truncated_at_first_stair_freeze: bool


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingInputError(f"{path} 必须是有限实数。")
    result = float(value)
    if not math.isfinite(result):
        raise TimingInputError(f"{path} 必须是有限实数。")
    return result


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TimingInputError(f"{path} 必须是 JSON 对象。")
    return value


def _vector(
    value: object,
    path: str,
    *,
    minimum_length: int,
) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
        or len(value) < minimum_length
    ):
        raise TimingInputError(f"{path} 至少需要 {minimum_length} 个有限数。")
    return tuple(_finite_number(item, f"{path}[{index}]") for index, item in enumerate(value))


def resolve_episode_path(raw_path: str | Path) -> tuple[Path, Path]:
    """解析 fresh-run 根目录或单个 episode 目录。"""

    path = Path(raw_path).expanduser().resolve()
    if (path / "frames.jsonl").is_file():
        episode_dir = path
        run_dir = path.parent
    elif (path / "episode_000000" / "frames.jsonl").is_file():
        run_dir = path
        episode_dir = path / "episode_000000"
    else:
        raise TimingInputError(
            "输入必须包含 frames.jsonl，或包含 episode_000000/frames.jsonl："
            f"{path}"
        )
    if not (episode_dir / "summary.json").is_file():
        raise TimingInputError(f"缺少 summary.json：{episode_dir}")
    return run_dir, episode_dir


def _optional_command(metadata: Mapping[str, Any], line_number: int) -> tuple[float, float, float] | None:
    raw_report = metadata.get("scan_cmd_vel_last_write_report")
    if raw_report is None:
        return None
    report = _mapping(raw_report, f"frames line {line_number}.scan_cmd_vel_last_write_report")
    raw_command = report.get("written_command")
    if raw_command is None:
        return None
    command = _vector(
        raw_command,
        f"frames line {line_number}.written_command",
        minimum_length=3,
    )
    return command[0], command[1], command[2]


def _optional_controller_state(metadata: Mapping[str, Any]) -> int | None:
    raw_status = metadata.get("scan_controller_status_last_report")
    if not isinstance(raw_status, Mapping):
        return None
    state = raw_status.get("state")
    if isinstance(state, bool) or not isinstance(state, int):
        return None
    return state


def _optional_controller_is_final(metadata: Mapping[str, Any]) -> bool | None:
    raw_status = metadata.get("scan_controller_status_last_report")
    if not isinstance(raw_status, Mapping):
        return None
    value = raw_status.get("is_final")
    if not isinstance(value, bool):
        return None
    return value


def load_frame_samples(frames_path: str | Path) -> list[FrameSample]:
    """流式读取大型 frames.jsonl，只保留测速所需字段。"""

    path = Path(frames_path)
    samples: list[FrameSample] = []
    previous_timestamp: float | None = None
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise TimingInputError(f"无法读取 {path}：{exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                raw_frame = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TimingInputError(
                    f"frames.jsonl 第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc
            frame = _mapping(raw_frame, f"frames line {line_number}")
            timestamp = _finite_number(
                frame.get("timestamp"),
                f"frames line {line_number}.timestamp",
            )
            # 同一个仿真 tick 内可能连续记录两次 pipeline 状态切换；相同时间戳
            # 只会形成零时长样本，不会抬高任何耗时。只有时钟真实回退才说明
            # episode 数据不能按时间积分。
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise TimingInputError(
                    "frames 时间戳不得回退："
                    f"line={line_number}, previous={previous_timestamp}, current={timestamp}"
                )
            previous_timestamp = timestamp
            pipeline_state = frame.get("pipeline_state")
            if not isinstance(pipeline_state, str):
                raise TimingInputError(
                    f"frames line {line_number}.pipeline_state 必须是字符串。"
                )
            observation = _mapping(
                frame.get("observation"),
                f"frames line {line_number}.observation",
            )
            metadata = _mapping(
                observation.get("metadata"),
                f"frames line {line_number}.observation.metadata",
            )
            action_source = metadata.get("last_action_source", "")
            if action_source is None:
                action_source = ""
            if not isinstance(action_source, str):
                raise TimingInputError(
                    f"frames line {line_number}.last_action_source 必须是字符串或 null。"
                )
            raw_report = metadata.get("scan_cmd_vel_last_write_report")
            motion_allowed = False
            if isinstance(raw_report, Mapping):
                raw_motion_allowed = raw_report.get("motion_allowed", False)
                if not isinstance(raw_motion_allowed, bool):
                    raise TimingInputError(
                        f"frames line {line_number}.motion_allowed 必须是布尔值。"
                    )
                motion_allowed = raw_motion_allowed
            command = _optional_command(metadata, line_number)

            body_planar_speed: float | None = None
            body_yaw_rate: float | None = None
            raw_body_velocity = metadata.get("body_velocity")
            if raw_body_velocity is not None:
                body_velocity = _vector(
                    raw_body_velocity,
                    f"frames line {line_number}.body_velocity",
                    minimum_length=3,
                )
                body_planar_speed = math.hypot(body_velocity[0], body_velocity[1])
                body_yaw_rate = body_velocity[2]

            pose_xy: tuple[float, float] | None = None
            raw_pose = observation.get("robot_root_pose")
            if raw_pose is not None:
                pose = _vector(
                    raw_pose,
                    f"frames line {line_number}.robot_root_pose",
                    minimum_length=2,
                )
                pose_xy = pose[0], pose[1]

            samples.append(
                FrameSample(
                    timestamp_s=timestamp,
                    pipeline_state=pipeline_state,
                    action_source=action_source,
                    motion_allowed=motion_allowed,
                    command=command,
                    body_planar_speed_mps=body_planar_speed,
                    body_yaw_rate_radps=body_yaw_rate,
                    pose_xy=pose_xy,
                    controller_state=_optional_controller_state(metadata),
                    controller_is_final=_optional_controller_is_final(metadata),
                )
            )
    if len(samples) < 2:
        raise TimingInputError("frames.jsonl 至少需要两个有效诊断帧。")
    return samples


def _is_stair_takeover(source: str) -> bool:
    return source.startswith(_STAIR_TAKEOVER_PREFIXES)


def _first_index(samples: Sequence[FrameSample], predicate: Any, description: str) -> int:
    for index, sample in enumerate(samples):
        if predicate(sample):
            return index
    raise TimingInputError(f"未找到{description}。")


def _weighted_mean(weighted_values: Sequence[tuple[float, float]]) -> float | None:
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0.0:
        return None
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _summarize_scan_control_interval(
    samples: Sequence[FrameSample],
    *,
    start_index: int,
    end_timestamp_s: float,
    configured_max_yaw_rate_radps: float,
    label: str,
) -> dict[str, Any]:
    """按诊断帧近似积分一个由 SCAN 实际控制 policy 的连续区间。"""

    if not 0 <= start_index < len(samples) - 1:
        raise TimingInputError(f"{label}起点索引无效。")
    start_timestamp = samples[start_index].timestamp_s
    end_timestamp = _finite_number(end_timestamp_s, f"{label}.end_timestamp_s")
    if end_timestamp <= start_timestamp:
        raise TimingInputError(f"{label}终点必须晚于起点。")
    if samples[-1].timestamp_s + 1.0e-12 < end_timestamp:
        raise TimingInputError(
            f"{label}诊断帧没有覆盖终点："
            f"last={samples[-1].timestamp_s}, end={end_timestamp}。"
        )
    max_yaw_rate = _finite_number(
        configured_max_yaw_rate_radps,
        "configured_max_yaw_rate_radps",
    )
    if max_yaw_rate <= 0.0:
        raise TimingInputError("configured_max_yaw_rate_radps 必须为正数。")

    translation_duration = 0.0
    yaw_only_duration = 0.0
    zero_duration = 0.0
    aligning_duration = 0.0
    near_yaw_limit_duration = 0.0
    sample_gaps: list[float] = []
    command_speed_samples: list[tuple[float, float]] = []
    measured_speed_samples: list[tuple[float, float]] = []
    maximum_command_yaw_rate = 0.0
    maximum_measured_yaw_rate = 0.0
    xy_travel = 0.0
    saturation_threshold = max(
        _COMMAND_EPSILON,
        max_yaw_rate - _YAW_SATURATION_MARGIN_RADPS,
    )

    for index in range(start_index, len(samples) - 1):
        current = samples[index]
        following = samples[index + 1]
        if current.timestamp_s >= end_timestamp - 1.0e-12:
            break
        if current.action_source != "scan_ros2_navigation":
            raise TimingInputError(
                f"{label}中出现非SCAN控制源："
                f"timestamp={current.timestamp_s}, source={current.action_source!r}。"
            )
        raw_gap = following.timestamp_s - current.timestamp_s
        duration = min(following.timestamp_s, end_timestamp) - current.timestamp_s
        if raw_gap <= 0.0 or duration <= 0.0:
            raise TimingInputError(f"{label}诊断帧时间差必须为正。")
        sample_gaps.append(raw_gap)
        command = current.command or (0.0, 0.0, 0.0)
        translation_speed = math.hypot(command[0], command[1])
        yaw_rate = abs(command[2])
        maximum_command_yaw_rate = max(maximum_command_yaw_rate, yaw_rate)
        if current.body_yaw_rate_radps is not None:
            maximum_measured_yaw_rate = max(
                maximum_measured_yaw_rate,
                abs(current.body_yaw_rate_radps),
            )
        if translation_speed > _TRANSLATION_ACTIVITY_THRESHOLD_MPS:
            translation_duration += duration
            command_speed_samples.append((translation_speed, duration))
            if current.body_planar_speed_mps is not None:
                measured_speed_samples.append(
                    (current.body_planar_speed_mps, duration)
                )
        elif yaw_rate > _COMMAND_EPSILON:
            yaw_only_duration += duration
        else:
            zero_duration += duration
        if current.controller_state == 9:
            aligning_duration += duration
        if yaw_rate + 1.0e-12 >= saturation_threshold:
            near_yaw_limit_duration += duration
        if current.pose_xy is not None and following.pose_xy is not None:
            xy_travel += (
                math.hypot(
                    following.pose_xy[0] - current.pose_xy[0],
                    following.pose_xy[1] - current.pose_xy[1],
                )
                * duration
                / raw_gap
            )

    control_duration = end_timestamp - start_timestamp
    categorized_duration = translation_duration + yaw_only_duration + zero_duration
    if abs(control_duration - categorized_duration) > 1.0e-8:
        raise TimingInputError(
            f"{label}命令分类没有覆盖完整控制段："
            f"control={control_duration}, categorized={categorized_duration}。"
        )
    if not sample_gaps:
        raise TimingInputError(f"{label}没有可积分的诊断区间。")
    return {
        "start_s": start_timestamp,
        "end_s": end_timestamp,
        "control_s": control_duration,
        "translation_command_s": translation_duration,
        "yaw_only_command_s": yaw_only_duration,
        "zero_command_s": zero_duration,
        "yaw_only_fraction": yaw_only_duration / control_duration,
        "controller_aligning_state_s": aligning_duration,
        "near_yaw_limit_s": near_yaw_limit_duration,
        "near_yaw_limit_fraction": near_yaw_limit_duration / control_duration,
        "configured_max_yaw_rate_radps": max_yaw_rate,
        "near_yaw_limit_threshold_radps": saturation_threshold,
        "translation_activity_threshold_mps": (
            _TRANSLATION_ACTIVITY_THRESHOLD_MPS
        ),
        "maximum_command_yaw_rate_radps": maximum_command_yaw_rate,
        "maximum_measured_yaw_rate_radps": maximum_measured_yaw_rate,
        "mean_command_planar_speed_while_translating_mps": _weighted_mean(
            command_speed_samples
        ),
        "mean_measured_planar_speed_while_translating_mps": _weighted_mean(
            measured_speed_samples
        ),
        "sampled_xy_travel_m": xy_travel,
        "diagnostic_interval_count": len(sample_gaps),
        "diagnostic_maximum_gap_s": max(sample_gaps),
        "diagnostic_median_gap_s": statistics.median(sample_gaps),
    }


def analyze_f1_samples(
    samples: Sequence[FrameSample],
    *,
    configured_max_yaw_rate_radps: float,
) -> dict[str, Any]:
    """计算一楼 F1 阶段的控制、位移和转向占时。"""

    max_yaw_rate = _finite_number(
        configured_max_yaw_rate_radps,
        "configured_max_yaw_rate_radps",
    )
    if max_yaw_rate <= 0.0:
        raise TimingInputError("configured_max_yaw_rate_radps 必须为正数。")
    nav_index = _first_index(
        samples,
        lambda sample: sample.pipeline_state == "exec_nav_to_place",
        "exec_nav_to_place 起点",
    )
    scan_index = _first_index(
        samples[nav_index:],
        lambda sample: sample.action_source == "scan_ros2_navigation",
        "SCAN 执行源首个控制拍",
    ) + nav_index
    stair_index = _first_index(
        samples[scan_index + 1 :],
        lambda sample: _is_stair_takeover(sample.action_source),
        "首次楼梯接管",
    ) + scan_index + 1
    if stair_index <= scan_index:
        raise TimingInputError("楼梯接管必须晚于 SCAN 首个非零命令。")

    translation_duration = 0.0
    yaw_only_duration = 0.0
    zero_duration = 0.0
    aligning_duration = 0.0
    near_yaw_limit_duration = 0.0
    sample_gaps: list[float] = []
    command_speed_samples: list[tuple[float, float]] = []
    measured_speed_samples: list[tuple[float, float]] = []
    maximum_command_yaw_rate = 0.0
    maximum_measured_yaw_rate = 0.0
    xy_travel = 0.0

    saturation_threshold = max(
        _COMMAND_EPSILON,
        max_yaw_rate - _YAW_SATURATION_MARGIN_RADPS,
    )
    for index in range(scan_index, stair_index):
        current = samples[index]
        following = samples[index + 1]
        duration = following.timestamp_s - current.timestamp_s
        if duration <= 0.0:
            raise TimingInputError("控制段诊断帧时间差必须为正。")
        sample_gaps.append(duration)
        command = current.command or (0.0, 0.0, 0.0)
        translation_speed = math.hypot(command[0], command[1])
        yaw_rate = abs(command[2])
        maximum_command_yaw_rate = max(maximum_command_yaw_rate, yaw_rate)
        if current.body_yaw_rate_radps is not None:
            maximum_measured_yaw_rate = max(
                maximum_measured_yaw_rate,
                abs(current.body_yaw_rate_radps),
            )
        if translation_speed > _TRANSLATION_ACTIVITY_THRESHOLD_MPS:
            translation_duration += duration
            command_speed_samples.append((translation_speed, duration))
            if current.body_planar_speed_mps is not None:
                measured_speed_samples.append((current.body_planar_speed_mps, duration))
        elif yaw_rate > _COMMAND_EPSILON:
            yaw_only_duration += duration
        else:
            zero_duration += duration
        if current.controller_state == 9:
            aligning_duration += duration
        if yaw_rate + 1.0e-12 >= saturation_threshold:
            near_yaw_limit_duration += duration
        if current.pose_xy is not None and following.pose_xy is not None:
            xy_travel += math.hypot(
                following.pose_xy[0] - current.pose_xy[0],
                following.pose_xy[1] - current.pose_xy[1],
            )

    control_duration = samples[stair_index].timestamp_s - samples[scan_index].timestamp_s
    categorized_duration = translation_duration + yaw_only_duration + zero_duration
    if abs(control_duration - categorized_duration) > 1.0e-9:
        raise TimingInputError("控制命令分类时长没有覆盖完整 F1 SCAN 控制段。")
    start_pose = samples[scan_index].pose_xy
    end_pose = samples[stair_index].pose_xy
    net_displacement = None
    if start_pose is not None and end_pose is not None:
        net_displacement = math.hypot(
            end_pose[0] - start_pose[0],
            end_pose[1] - start_pose[1],
        )

    return {
        "navigation_stage_start_s": samples[nav_index].timestamp_s,
        "scan_control_start_s": samples[scan_index].timestamp_s,
        "stair_takeover_start_s": samples[stair_index].timestamp_s,
        "navigation_handshake_s": samples[scan_index].timestamp_s
        - samples[nav_index].timestamp_s,
        "f1_total_from_navigation_stage_s": samples[stair_index].timestamp_s
        - samples[nav_index].timestamp_s,
        "f1_scan_control_s": control_duration,
        "translation_command_s": translation_duration,
        "yaw_only_command_s": yaw_only_duration,
        "zero_command_s": zero_duration,
        "yaw_only_fraction": yaw_only_duration / control_duration,
        "controller_aligning_state_s": aligning_duration,
        "near_yaw_limit_s": near_yaw_limit_duration,
        "near_yaw_limit_fraction": near_yaw_limit_duration / control_duration,
        "configured_max_yaw_rate_radps": max_yaw_rate,
        "near_yaw_limit_threshold_radps": saturation_threshold,
        "translation_activity_threshold_mps": (
            _TRANSLATION_ACTIVITY_THRESHOLD_MPS
        ),
        "maximum_command_yaw_rate_radps": maximum_command_yaw_rate,
        "maximum_measured_yaw_rate_radps": maximum_measured_yaw_rate,
        "mean_command_planar_speed_while_translating_mps": _weighted_mean(
            command_speed_samples
        ),
        "mean_measured_planar_speed_while_translating_mps": _weighted_mean(
            measured_speed_samples
        ),
        "sampled_xy_travel_m": xy_travel,
        "sampled_xy_net_displacement_m": net_displacement,
        "diagnostic_interval_count": len(sample_gaps),
        "diagnostic_maximum_gap_s": max(sample_gaps),
        "diagnostic_median_gap_s": statistics.median(sample_gaps),
    }


def analyze_crossfloor_samples(
    samples: Sequence[FrameSample],
    *,
    goal_reached_timestamp_s: float,
    configured_max_yaw_rate_radps: float,
) -> dict[str, Any]:
    """拆分一次完整跨层成功运行的F1、公共楼梯、F2与末端捕获。"""

    goal_timestamp = _finite_number(
        goal_reached_timestamp_s,
        "goal_reached_timestamp_s",
    )
    nav_index = _first_index(
        samples,
        lambda sample: sample.pipeline_state == "exec_nav_to_place",
        "exec_nav_to_place 起点",
    )
    f1_start_index = _first_index(
        samples[nav_index:],
        lambda sample: sample.action_source == "scan_ros2_navigation",
        "F1 SCAN 控制起点",
    ) + nav_index
    stair_start_index = _first_index(
        samples[f1_start_index + 1 :],
        lambda sample: _is_stair_takeover(sample.action_source),
        "公共楼梯接管起点",
    ) + f1_start_index + 1
    stair_resume_index = _first_index(
        samples[stair_start_index + 1 :],
        lambda sample: sample.action_source == "scan_ros2_navigation",
        "楼梯释放后的F2 SCAN恢复点",
    ) + stair_start_index + 1
    if goal_timestamp <= samples[stair_resume_index].timestamp_s:
        raise TimingInputError("GOAL_REACHED 必须晚于楼梯释放后的F2恢复。")

    terminal_start_index: int | None = None
    for index in range(stair_resume_index, len(samples)):
        sample = samples[index]
        if sample.timestamp_s >= goal_timestamp:
            break
        if (
            sample.action_source == "scan_ros2_navigation"
            and sample.controller_is_final is True
        ):
            terminal_start_index = index
            break
    if terminal_start_index is None:
        raise TimingInputError(
            "完整跨层成功运行没有记录首个is_final=true的末端轨迹。"
        )

    f1 = _summarize_scan_control_interval(
        samples,
        start_index=f1_start_index,
        end_timestamp_s=samples[stair_start_index].timestamp_s,
        configured_max_yaw_rate_radps=configured_max_yaw_rate_radps,
        label="F1 SCAN控制段",
    )
    f2 = _summarize_scan_control_interval(
        samples,
        start_index=stair_resume_index,
        end_timestamp_s=goal_timestamp,
        configured_max_yaw_rate_radps=configured_max_yaw_rate_radps,
        label="F2 SCAN控制段",
    )
    terminal = _summarize_scan_control_interval(
        samples,
        start_index=terminal_start_index,
        end_timestamp_s=goal_timestamp,
        configured_max_yaw_rate_radps=configured_max_yaw_rate_radps,
        label="末端捕获段",
    )

    planner_controlled = f1["control_s"] + f2["control_s"]
    translation_duration = (
        f1["translation_command_s"] + f2["translation_command_s"]
    )
    yaw_only_duration = f1["yaw_only_command_s"] + f2["yaw_only_command_s"]
    zero_duration = f1["zero_command_s"] + f2["zero_command_s"]
    categorized = translation_duration + yaw_only_duration + zero_duration
    if not math.isclose(
        categorized,
        planner_controlled,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise TimingInputError("跨层SCAN命令分类没有覆盖完整受控导航时间。")

    def combined_translation_mean(key: str) -> float | None:
        weighted: list[tuple[float, float]] = []
        for phase in (f1, f2):
            value = phase[key]
            weight = phase["translation_command_s"]
            if value is not None and weight > 0.0:
                weighted.append((float(value), float(weight)))
        return _weighted_mean(weighted)

    return {
        "navigation_stage_start_s": samples[nav_index].timestamp_s,
        "f1_scan_control_start_s": samples[f1_start_index].timestamp_s,
        "stair_takeover_start_s": samples[stair_start_index].timestamp_s,
        "stair_release_f2_resume_s": samples[stair_resume_index].timestamp_s,
        "terminal_final_trajectory_start_s": samples[
            terminal_start_index
        ].timestamp_s,
        "goal_reached_s": goal_timestamp,
        "navigation_handshake_s": (
            samples[f1_start_index].timestamp_s
            - samples[nav_index].timestamp_s
        ),
        "full_navigation_stage_sim_time_s": (
            goal_timestamp - samples[nav_index].timestamp_s
        ),
        "common_stair_freeze_sim_time_s": (
            samples[stair_resume_index].timestamp_s
            - samples[stair_start_index].timestamp_s
        ),
        "planner_controlled_navigation_sim_time_s": planner_controlled,
        "f1_scan_control_sim_time_s": f1["control_s"],
        "f2_scan_control_sim_time_s": f2["control_s"],
        "f2_preterminal_sim_time_s": (
            samples[terminal_start_index].timestamp_s
            - samples[stair_resume_index].timestamp_s
        ),
        "terminal_capture_sim_time_s": terminal["control_s"],
        "translation_command_sim_time_s": translation_duration,
        "yaw_only_command_sim_time_s": yaw_only_duration,
        "zero_command_sim_time_s": zero_duration,
        "yaw_only_fraction_of_planner_controlled": (
            yaw_only_duration / planner_controlled
        ),
        "mean_command_planar_speed_while_translating_mps": (
            combined_translation_mean(
                "mean_command_planar_speed_while_translating_mps"
            )
        ),
        "mean_measured_planar_speed_while_translating_mps": (
            combined_translation_mean(
                "mean_measured_planar_speed_while_translating_mps"
            )
        ),
        "sampled_xy_travel_planner_controlled_m": (
            f1["sampled_xy_travel_m"] + f2["sampled_xy_travel_m"]
        ),
        "configured_max_yaw_rate_radps": configured_max_yaw_rate_radps,
        "diagnostic_maximum_gap_s": max(
            f1["diagnostic_maximum_gap_s"],
            f2["diagnostic_maximum_gap_s"],
        ),
        "f1": f1,
        "f2": f2,
        "terminal": terminal,
        "primary_metric_excludes_common_stair_freeze": True,
        "place_manipulation_excluded": True,
    }


def analyze_planner_prefix(log_path: str | Path) -> PlannerPrefixReport:
    """统计首次楼梯冻结日志前的局部规划成功、失败和耗时。"""

    path = Path(log_path)
    success_count = 0
    failure_count = 0
    successful_wall_times: list[float] = []
    truncated = False
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise TimingInputError(f"无法读取 {path}：{exc}") from exc
    with handle:
        for raw_line in handle:
            line = _ANSI_ESCAPE.sub("", raw_line)
            if "收到楼梯执行冻结" in line:
                truncated = True
                break
            total_match = _PLANNER_TOTAL_TIME.search(line)
            if total_match is not None:
                successful_wall_times.append(float(total_match.group(1)))
            success_match = _FINAL_PLAN_SUCCESS.search(line)
            if success_match is not None:
                if success_match.group(1) == "1":
                    success_count += 1
                else:
                    failure_count += 1
    if not truncated:
        raise TimingInputError("ros2_launch.log 未出现首次楼梯冻结边界。")
    if len(successful_wall_times) != success_count:
        raise TimingInputError(
            "成功规划数与 total time 日志数不一致："
            f"success={success_count}, total_time={len(successful_wall_times)}"
        )
    p95 = None
    if successful_wall_times:
        ordered = sorted(successful_wall_times)
        rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
        p95 = ordered[rank]
    return PlannerPrefixReport(
        attempt_count=success_count + failure_count,
        success_count=success_count,
        failure_count=failure_count,
        successful_total_wall_s=sum(successful_wall_times),
        successful_mean_wall_s=(
            statistics.fmean(successful_wall_times)
            if successful_wall_times
            else None
        ),
        successful_p95_wall_s=p95,
        log_truncated_at_first_stair_freeze=truncated,
    )


def _planner_segment_statistics(
    *,
    success_wall_times: Sequence[float],
    failure_count: int,
) -> dict[str, Any]:
    """生成一个规划阶段的稳定统计，不把失败尝试伪装成零耗时成功。"""

    ordered = sorted(success_wall_times)
    p95 = None
    if ordered:
        rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
        p95 = ordered[rank]
    return {
        "attempt_count": len(success_wall_times) + failure_count,
        "success_count": len(success_wall_times),
        "failure_count": failure_count,
        "successful_total_wall_s": sum(success_wall_times),
        "successful_mean_wall_s": (
            statistics.fmean(success_wall_times)
            if success_wall_times
            else None
        ),
        "successful_p95_wall_s": p95,
    }


def analyze_planner_crossfloor_log(log_path: str | Path) -> dict[str, Any]:
    """按首次楼梯冻结/释放拆分完整成功运行的SCAN规划墙钟耗时。"""

    path = Path(log_path)
    segment = "pre_stair"
    success_wall_times: dict[str, list[float]] = {
        "pre_stair": [],
        "post_stair": [],
    }
    failure_counts = {"pre_stair": 0, "post_stair": 0}
    pending_success_wall_time: tuple[str, float] | None = None
    freeze_count = 0
    release_count = 0
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise TimingInputError(f"无法读取 {path}：{exc}") from exc
    with handle:
        for raw_line in handle:
            line = _ANSI_ESCAPE.sub("", raw_line)
            if "收到楼梯执行冻结" in line:
                if pending_success_wall_time is not None:
                    raise TimingInputError("楼梯冻结前存在未配对的规划耗时日志。")
                freeze_count += 1
                segment = "stair_paused"
                continue
            if "楼梯执行冻结解除" in line:
                if pending_success_wall_time is not None:
                    raise TimingInputError("楼梯释放前存在未配对的规划耗时日志。")
                release_count += 1
                segment = "post_stair"
                continue
            total_match = _PLANNER_TOTAL_TIME.search(line)
            if total_match is not None:
                if segment == "stair_paused":
                    raise TimingInputError("楼梯冻结期间不应执行SCAN规划。")
                if pending_success_wall_time is not None:
                    raise TimingInputError("连续出现未配对的SCAN total time日志。")
                pending_success_wall_time = (
                    segment,
                    float(total_match.group(1)),
                )
            success_match = _FINAL_PLAN_SUCCESS.search(line)
            if success_match is None:
                continue
            if segment == "stair_paused":
                raise TimingInputError("楼梯冻结期间不应出现SCAN规划结果。")
            if success_match.group(1) == "1":
                if pending_success_wall_time is None:
                    raise TimingInputError("成功SCAN规划缺少配对total time日志。")
                pending_segment, wall_time = pending_success_wall_time
                if pending_segment != segment:
                    raise TimingInputError("SCAN规划耗时跨越了楼梯阶段边界。")
                success_wall_times[segment].append(wall_time)
                pending_success_wall_time = None
            else:
                if pending_success_wall_time is not None:
                    raise TimingInputError("失败SCAN规划不应带成功total time日志。")
                failure_counts[segment] += 1
    if pending_success_wall_time is not None:
        raise TimingInputError("日志结束时仍有未配对的SCAN total time。")
    if freeze_count != 1 or release_count != 1:
        raise TimingInputError(
            "完整公平跨层运行必须恰好出现一次楼梯冻结和一次释放："
            f"freeze={freeze_count}, release={release_count}。"
        )
    pre = _planner_segment_statistics(
        success_wall_times=success_wall_times["pre_stair"],
        failure_count=failure_counts["pre_stair"],
    )
    post = _planner_segment_statistics(
        success_wall_times=success_wall_times["post_stair"],
        failure_count=failure_counts["post_stair"],
    )
    return {
        "pre_stair": pre,
        "post_stair": post,
        "freeze_event_count": freeze_count,
        "release_event_count": release_count,
        "attempt_count": pre["attempt_count"] + post["attempt_count"],
        "success_count": pre["success_count"] + post["success_count"],
        "failure_count": pre["failure_count"] + post["failure_count"],
        "successful_total_wall_s": (
            pre["successful_total_wall_s"]
            + post["successful_total_wall_s"]
        ),
        "planner_compute_excludes_stair_pause": True,
    }


def _load_summary(summary_path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TimingInputError(f"无法读取 summary：{summary_path}: {exc}") from exc
    return _mapping(payload, "summary")


def resolve_configured_max_yaw_rate(
    run_dir: Path,
    legacy_override: float | None,
) -> tuple[float, str]:
    """优先从不可变运行快照读取角速度上限，旧产物才使用显式参数。"""

    manifest_path = run_dir / "pct_scan_live_acceptance.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TimingInputError(f"无法读取运行清单 {manifest_path}：{exc}") from exc
        if isinstance(manifest, Mapping):
            snapshot = manifest.get("tuning_config_snapshot")
            if isinstance(snapshot, Mapping):
                selected = snapshot.get("selected_parameters")
                if isinstance(selected, Mapping) and (
                    "scan_controller.limits.max_yaw_rate" in selected
                ):
                    value = _finite_number(
                        selected["scan_controller.limits.max_yaw_rate"],
                        "tuning_config_snapshot.selected_parameters.max_yaw_rate",
                    )
                    if value <= 0.0:
                        raise TimingInputError("运行快照中的 max_yaw_rate 必须为正数。")
                    return value, "immutable_run_tuning_snapshot"
    if legacy_override is None:
        raise TimingInputError(
            "旧运行没有不可变调参快照；请显式提供 "
            "--configured-max-yaw-rate。"
        )
    value = _finite_number(legacy_override, "configured_max_yaw_rate_radps")
    if value <= 0.0:
        raise TimingInputError("configured_max_yaw_rate_radps 必须为正数。")
    return value, "explicit_legacy_cli_value"


def _successful_crossfloor_goal_timestamp(
    summary: Mapping[str, Any],
) -> float | None:
    """成功carry才返回受严格零速门确认的GOAL_REACHED仿真时间。"""

    if summary.get("execution_mode") != "navigation_carry_smoke":
        return None
    if summary.get("success") is not True:
        return None
    if summary.get("final_state") != "done":
        raise TimingInputError("成功crossfloor carry的final_state必须为done。")
    executor = _mapping(
        summary.get("latest_executor_status"),
        "summary.latest_executor_status",
    )
    required_true = (
        "done",
        "success",
        "goal_rising_edge_seen",
        "scan_controller_goal_reached_verified",
        "policy_zero_hold_verified",
    )
    missing = [key for key in required_true if executor.get(key) is not True]
    if missing:
        raise TimingInputError(
            "成功crossfloor carry缺少严格终点证据：" + ",".join(missing)
        )
    if executor.get("post_goal_nonzero_write_count") != 0:
        raise TimingInputError("GOAL_REACHED后仍存在非零policy写入。")
    timestamp = _finite_number(
        executor.get("goal_true_receipt_timestamp"),
        "summary.latest_executor_status.goal_true_receipt_timestamp",
    )
    if timestamp <= 0.0:
        raise TimingInputError("GOAL_REACHED时间必须为正数。")
    return timestamp


def analyze_episode(
    raw_path: str | Path,
    *,
    configured_max_yaw_rate_radps: float | None = None,
) -> dict[str, Any]:
    """分析一个 fresh-run，并返回机器可读报告。"""

    run_dir, episode_dir = resolve_episode_path(raw_path)
    samples = load_frame_samples(episode_dir / "frames.jsonl")
    summary = _load_summary(episode_dir / "summary.json")
    resolved_yaw_rate, yaw_rate_provenance = resolve_configured_max_yaw_rate(
        run_dir,
        configured_max_yaw_rate_radps,
    )
    f1 = analyze_f1_samples(
        samples,
        configured_max_yaw_rate_radps=resolved_yaw_rate,
    )
    planner = analyze_planner_prefix(run_dir / "ros2_launch.log")
    goal_timestamp = _successful_crossfloor_goal_timestamp(summary)
    crossfloor: dict[str, Any] | None = None
    planner_crossfloor: dict[str, Any] | None = None
    if goal_timestamp is not None:
        crossfloor = analyze_crossfloor_samples(
            samples,
            goal_reached_timestamp_s=goal_timestamp,
            configured_max_yaw_rate_radps=resolved_yaw_rate,
        )
        planner_crossfloor = analyze_planner_crossfloor_log(
            run_dir / "ros2_launch.log"
        )
    if summary.get("execution_mode") != "navigation_carry_smoke":
        crossfloor_unavailable_reason = "execution_mode_is_not_crossfloor_carry"
    elif summary.get("success") is not True:
        crossfloor_unavailable_reason = "crossfloor_episode_not_successful"
    else:
        crossfloor_unavailable_reason = None
    return {
        "schema": "pct_scan_live_timing_v2",
        "run_dir": str(run_dir),
        "episode_dir": str(episode_dir),
        "seed": summary.get("seed"),
        "success": summary.get("success"),
        "failure_reason": summary.get("failure_reason"),
        "final_state": summary.get("final_state"),
        "configured_max_yaw_rate_provenance": yaw_rate_provenance,
        "f1": f1,
        "crossfloor": crossfloor,
        "crossfloor_unavailable_reason": crossfloor_unavailable_reason,
        "planner_before_first_stair_freeze": asdict(planner),
        "planner_crossfloor": planner_crossfloor,
        "scope": {
            "stair_freeze_excluded": True,
            "f2_excluded": crossfloor is None,
            "terminal_excluded": crossfloor is None,
            "place_excluded": True,
            "diagnostic_frame_approximation": True,
            "full_crossfloor_available": crossfloor is not None,
        },
    }


def compare_reports(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """以首个运行作基准，计算后续运行的单阶段变化。"""

    if len(reports) < 2:
        return []
    baseline = _mapping(reports[0].get("f1"), "reports[0].f1")
    baseline_control = _finite_number(
        baseline.get("f1_scan_control_s"),
        "reports[0].f1.f1_scan_control_s",
    )
    baseline_yaw = _finite_number(
        baseline.get("yaw_only_command_s"),
        "reports[0].f1.yaw_only_command_s",
    )
    comparisons: list[dict[str, Any]] = []
    for report in reports[1:]:
        current = _mapping(report.get("f1"), "report.f1")
        control = _finite_number(current.get("f1_scan_control_s"), "report.f1_scan_control_s")
        yaw = _finite_number(current.get("yaw_only_command_s"), "report.yaw_only_command_s")
        comparison: dict[str, Any] = {
            "baseline_run_dir": reports[0].get("run_dir"),
            "candidate_run_dir": report.get("run_dir"),
            "f1_scan_control_delta_s": control - baseline_control,
            "f1_scan_control_reduction_ratio": (
                (baseline_control - control) / baseline_control
                if baseline_control > 0.0
                else None
            ),
            "yaw_only_delta_s": yaw - baseline_yaw,
            "yaw_only_reduction_ratio": (
                (baseline_yaw - yaw) / baseline_yaw
                if baseline_yaw > 0.0
                else None
            ),
            "full_crossfloor_comparison_available": False,
        }
        baseline_crossfloor = reports[0].get("crossfloor")
        candidate_crossfloor = report.get("crossfloor")
        if isinstance(baseline_crossfloor, Mapping) and isinstance(
            candidate_crossfloor,
            Mapping,
        ):
            baseline_primary = _finite_number(
                baseline_crossfloor.get(
                    "planner_controlled_navigation_sim_time_s"
                ),
                "reports[0].crossfloor.planner_controlled_navigation_sim_time_s",
            )
            candidate_primary = _finite_number(
                candidate_crossfloor.get(
                    "planner_controlled_navigation_sim_time_s"
                ),
                "report.crossfloor.planner_controlled_navigation_sim_time_s",
            )
            baseline_full = _finite_number(
                baseline_crossfloor.get("full_navigation_stage_sim_time_s"),
                "reports[0].crossfloor.full_navigation_stage_sim_time_s",
            )
            candidate_full = _finite_number(
                candidate_crossfloor.get("full_navigation_stage_sim_time_s"),
                "report.crossfloor.full_navigation_stage_sim_time_s",
            )
            comparison.update(
                {
                    "full_crossfloor_comparison_available": True,
                    "planner_controlled_navigation_delta_s": (
                        candidate_primary - baseline_primary
                    ),
                    "planner_controlled_navigation_reduction_ratio": (
                        (baseline_primary - candidate_primary)
                        / baseline_primary
                        if baseline_primary > 0.0
                        else None
                    ),
                    "full_navigation_stage_delta_s": (
                        candidate_full - baseline_full
                    ),
                    "full_navigation_stage_reduction_ratio": (
                        (baseline_full - candidate_full) / baseline_full
                        if baseline_full > 0.0
                        else None
                    ),
                }
            )
        comparisons.append(comparison)
    return comparisons


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episodes",
        type=Path,
        nargs="+",
        help="fresh-run 根目录或 episode_000000 目录；多个输入时以第一个作基准。",
    )
    parser.add_argument(
        "--configured-max-yaw-rate",
        type=float,
        default=None,
        help=(
            "仅供没有不可变调参快照的旧产物使用：实际 "
            "scan_controller limits.max_yaw_rate，单位 rad/s。"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；输入不完整时拒绝输出半可信统计。"""

    arguments = _build_parser().parse_args(argv)
    try:
        reports = [
            analyze_episode(
                path,
                configured_max_yaw_rate_radps=arguments.configured_max_yaw_rate,
            )
            for path in arguments.episodes
        ]
        payload = {
            "schema": "pct_scan_live_timing_collection_v2",
            "reports": reports,
            "comparisons_to_first": compare_reports(reports),
        }
    except TimingInputError as exc:
        print(f"FAIL：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
