"""为 SCAN 三维参考路径提供独立的楼梯底盘冻结协调器。

本模块只负责识别楼梯段、按弧长生成连续 root 目标，以及描述分阶段锁定和
释放动作。它不依赖 DWA、ROS 2 或 Isaac Sim；真正的 ``/cmd_vel`` 清零仍由
runtime 中既有的唯一 policy 命令安全门执行。
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from source.interfaces.simulation import RobotAction, SimulationState


SCAN_STAIR_FREEZE_DOG_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
SCAN_STAIR_FREEZE_DOG_STAND_JOINT_POSITIONS = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)

_CONTROLLER_STATUS_SOURCE = "ros2_scan_planner_msgs_controller_status"
_CONTROLLER_EVENT_ACCEPTED = 1
_CONTROLLER_EVENT_REJECTED = 2
_CONTROLLER_EVENT_INVALIDATED = 3
_CONTROLLER_EVENT_STATE_CHANGED = 4
_CONTROLLER_EVENT_DUPLICATE = 5
_CONTROLLER_EXECUTION_STATES = frozenset({9, 10})
_CONTROLLER_EVENTS = frozenset(range(6))
_CONTROLLER_STATES = frozenset((*range(13), 255))
_ROOT_LOCK_PHASES = frozenset(
    {
        "active",
        "full_lock_settle",
        "root_release_settle",
        "release_action_pending",
        "terminal_hold",
    }
)
_FULL_BODY_LOCK_PHASES = frozenset(
    {"active", "full_lock_settle", "terminal_hold"}
)
_SENSOR_FRESHNESS_STOP_REASONS = frozenset(
    {
        "missing_odometry",
        "odometry_from_future",
        "odometry_timeout",
        "missing_point_cloud",
        "point_cloud_from_future",
        "point_cloud_timeout",
    }
)
_NAVIGATION_STATE_EMERGENCY_STOP = 5
_NAVIGATION_STATE_TRACKING = 3
_NAVIGATION_STATE_GOAL_REACHED = 6
_CONTROLLER_STATE_GOAL_REACHED = 12
_NAVIGATION_STAIR_INHIBIT_REASON = "scan_stair_execution_inhibited"
_NAVIGATION_STAIR_ALLOWED_STALE_INPUTS = frozenset({"bspline"})
_NAVIGATION_TERMINAL_CONTROLLER_STATES = frozenset({9, 10, 12})


@dataclass(frozen=True, slots=True)
class ScanReferencePath:
    """与手工 Path 发布器共享的一份已校验地面高度路径。"""

    points_ground_xyz: tuple[tuple[float, float, float], ...]
    source_path: str
    sha256: str
    points_sha256: str
    topic: str
    frame_id: str
    use_sim_time: bool
    min_point_distance_m: float
    stair_segment_indices: tuple[tuple[int, int], ...] | None


@dataclass(frozen=True, slots=True)
class ScanStairFreezeConfig:
    """定义台阶识别、root 运动和分阶段解锁参数。"""

    enabled: bool = True
    speed_mps: float = 0.18
    activation_radius_m: float = 0.12
    min_component_z_delta_m: float = 0.10
    min_step_z_delta_m: float = 0.05
    min_step_grade: float = 0.30
    min_riser_grade_variation: float = 0.15
    max_inter_step_gap_m: float = 0.45
    # Go2-X5 携臂收纳姿态前向碰撞边界约 0.38 m；必须在 SCAN 把第一阶
    # 判为双圆柱碰撞并急停之前接管，因此默认向楼梯入口前扩展 0.40 m。
    approach_distance_m: float = 0.40
    exit_distance_m: float = 0.40
    activation_lookahead_m: float = 0.50
    activation_timeout_s: float = 8.00
    activation_passed_margin_m: float = 0.05
    full_lock_settle_time_s: float = 1.20
    root_release_settle_time_s: float = 1.00
    post_release_stable_time_s: float = 0.50
    post_release_stabilization_timeout_s: float = 5.00
    resume_wait_fresh_cmd_timeout_s: float = 3.00
    terminal_goal_hold_timeout_s: float = 8.00
    post_release_max_linear_speed_mps: float = 0.08
    # 解锁后 SCAN 只需要航向角速度稳定；四足站姿正常的 roll/pitch 周期摆动
    # 由独立的 tilt 门约束，不能再把三轴角速度范数误当成 wz。
    post_release_max_angular_speed_rps: float = 0.20
    post_release_max_z_error_m: float = 0.10
    post_release_max_tilt_rad: float = 0.35
    yaw_lookahead_m: float = 0.35
    body_height_m: float = 0.30
    # 这些是跨模块协议容差，不是机器人物理到达容差。PCT backend 会把
    # 请求终点精确追加到 Path，并只对 collision PLY 投影引入浮点误差。
    terminal_goal_xy_tolerance_m: float = 1.0e-6
    terminal_goal_z_tolerance_m: float = 1.0e-4
    terminal_goal_yaw_tolerance_rad: float = 1.0e-5
    min_measured_body_height_m: float = 0.15
    max_measured_body_height_m: float = 0.60
    certified_progress_m: float = 0.02
    # 生产链同时校验 supervisor 实际收到的 Odometry/点云；纯单元调用可关闭。
    require_supervisor_sensor_status: bool = False
    supervisor_sensor_status_timeout_s: float = 0.25
    default_control_dt_s: float = 0.02
    max_control_dt_s: float = 0.20

    def __post_init__(self) -> None:
        for name in ("enabled", "require_supervisor_sensor_status"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须是布尔值。")
        positive = (
            "speed_mps",
            "min_component_z_delta_m",
            "min_step_z_delta_m",
            "min_step_grade",
            "max_inter_step_gap_m",
            "activation_timeout_s",
            "post_release_stabilization_timeout_s",
            "resume_wait_fresh_cmd_timeout_s",
            "terminal_goal_hold_timeout_s",
            "body_height_m",
            "max_measured_body_height_m",
            "certified_progress_m",
            "supervisor_sensor_status_timeout_s",
            "default_control_dt_s",
            "max_control_dt_s",
            "post_release_max_linear_speed_mps",
            "post_release_max_angular_speed_rps",
            "post_release_max_z_error_m",
            "post_release_max_tilt_rad",
        )
        nonnegative = (
            "activation_radius_m",
            "min_riser_grade_variation",
            "approach_distance_m",
            "exit_distance_m",
            "activation_lookahead_m",
            "activation_passed_margin_m",
            "full_lock_settle_time_s",
            "root_release_settle_time_s",
            "post_release_stable_time_s",
            "yaw_lookahead_m",
            "min_measured_body_height_m",
            "terminal_goal_xy_tolerance_m",
            "terminal_goal_z_tolerance_m",
            "terminal_goal_yaw_tolerance_rad",
        )
        for name in (*positive, *nonnegative):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} 必须是有限数值。")
            object.__setattr__(self, name, float(value))
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} 必须大于零。")
        for name in nonnegative:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} 不能为负数。")
        if self.min_measured_body_height_m >= self.max_measured_body_height_m:
            raise ValueError("实测 body height 上限必须大于下限。")
        if self.body_height_m > self.max_measured_body_height_m:
            raise ValueError("body_height_m 不能超过实测 body height 上限。")
        if self.max_control_dt_s < self.default_control_dt_s:
            raise ValueError("max_control_dt_s 不能小于 default_control_dt_s。")


@dataclass(frozen=True, slots=True)
class _StairComponent:
    """一段同方向、可连续冻结执行的楼梯地面折线。"""

    points: tuple[tuple[float, float, float], ...]
    source_start_index: int
    source_end_index: int
    direction: str


@dataclass(frozen=True, slots=True)
class _ResumePolicyWriteReport:
    """保存解锁后由新 Twist 驱动的一次 policy 实写证据。"""

    write_sequence: int
    write_timestamp: float
    source_sequence: int
    source_receipt_timestamp: float
    drain_sequence: int | None
    drain_receipt_timestamp: float | None


@dataclass(frozen=True, slots=True)
class _ControllerStatusEvidence:
    """保存一条不损失纳秒精度的 typed controller 状态证据。"""

    receipt_timestamp: float
    rx_sequence: int
    header_stamp_ns: int
    status_sequence: int
    acceptance_sequence: int
    event: int
    state: int
    accepted: bool
    trajectory_valid: bool
    is_final: bool
    emergency_stop: bool
    identity: tuple[int, int, int, int]


def load_scan_reference_path(path: str | Path) -> ScanReferencePath:
    """读取手工 Path ROS 参数 YAML，并保持 z 为地面高度。"""

    import yaml

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SCAN 手工参考路径不存在：{source}")
    raw_bytes = source.read_bytes()
    try:
        document = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法解析 SCAN 手工参考路径：{source}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("SCAN 手工参考路径 YAML 顶层必须是对象。")
    node = document.get("manual_path_publisher")
    if not isinstance(node, Mapping):
        raise ValueError("参考路径缺少 manual_path_publisher 节点配置。")
    parameters = node.get("ros__parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("参考路径缺少 ros__parameters。")

    topic = parameters.get("topic")
    frame_id = parameters.get("frame_id")
    use_sim_time = parameters.get("use_sim_time")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("参考路径 topic 必须是非空字符串。")
    if frame_id != "world":
        raise ValueError("底盘冻结当前只接受 world frame 的参考路径。")
    if use_sim_time is not True:
        raise ValueError("参考路径必须显式配置 use_sim_time=true。")
    minimum_distance = _finite_number(
        parameters.get("min_point_distance_m", 0.02),
        field_name="min_point_distance_m",
    )
    if minimum_distance < 0.0:
        raise ValueError("min_point_distance_m 不能为负数。")
    points = _prepare_flattened_points(
        parameters.get("points_xyz"),
        min_point_distance_m=minimum_distance,
    )
    points_sha256 = hash_ground_path_points(points)
    raw_freeze_config = document.get("scan_stair_freeze")
    if raw_freeze_config is None and parameters.get(
        "scan_stair_freeze_points_sha256"
    ) is not None:
        flat_indices = parameters.get(
            "scan_stair_freeze_stair_segment_indices"
        )
        if flat_indices == [-1, -1]:
            raw_segments: list[dict[str, int]] = []
        elif (
            isinstance(flat_indices, Sequence)
            and not isinstance(flat_indices, (str, bytes))
            and len(flat_indices) % 2 == 0
        ):
            raw_segments = [
                {
                    "start_index": flat_indices[index],
                    "end_index": flat_indices[index + 1],
                }
                for index in range(0, len(flat_indices), 2)
            ]
        else:
            raise ValueError(
                "scan_stair_freeze_stair_segment_indices 必须是成对整数数组。"
            )
        raw_freeze_config = {
            "points_sha256": parameters.get(
                "scan_stair_freeze_points_sha256"
            ),
            "stair_segments": raw_segments,
        }
    stair_segment_indices = _load_stair_segment_indices(
        raw_freeze_config,
        point_count=len(points),
        points_sha256=points_sha256,
    )
    return ScanReferencePath(
        points_ground_xyz=points,
        source_path=str(source),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        points_sha256=points_sha256,
        topic=topic,
        frame_id=frame_id,
        use_sim_time=True,
        min_point_distance_m=minimum_distance,
        stair_segment_indices=stair_segment_indices,
    )


def hash_ground_path_points(
    points_ground_xyz: Sequence[Sequence[float]],
) -> str:
    """按 IEEE-754 双精度几何值计算跨进程稳定的 Path 哈希。"""

    points = _coerce_points(points_ground_xyz)
    digest = hashlib.sha256()
    for point in points:
        digest.update(struct.pack("!ddd", *point))
    return digest.hexdigest()


def _load_stair_segment_indices(
    raw_config: Any,
    *,
    point_count: int,
    points_sha256: str,
) -> tuple[tuple[int, int], ...] | None:
    """读取可选显式楼梯区间，并校验其绑定的点几何哈希。"""

    if raw_config is None:
        return None
    if not isinstance(raw_config, Mapping):
        raise ValueError("scan_stair_freeze 配置必须是对象。")
    declared_sha256 = raw_config.get("points_sha256")
    if declared_sha256 != points_sha256:
        raise ValueError("scan_stair_freeze.points_sha256 与 Path 几何不匹配。")
    raw_segments = raw_config.get("stair_segments")
    if not isinstance(raw_segments, Sequence) or isinstance(
        raw_segments,
        (str, bytes),
    ):
        raise ValueError("scan_stair_freeze.stair_segments 必须是数组。")
    output: list[tuple[int, int]] = []
    previous_end = -1
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise ValueError(f"stair_segments[{index}] 必须是对象。")
        start = raw_segment.get("start_index")
        end = raw_segment.get("end_index")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError("stair_segments 的 start_index/end_index 必须是整数。")
        if not (0 <= start < end < point_count):
            raise ValueError(f"stair_segments[{index}] 点索引超出 Path。")
        if start <= previous_end:
            raise ValueError("stair_segments 必须严格递增且不能重叠。")
        output.append((start, end))
        previous_end = end
    return tuple(output)


def extract_stair_components(
    path_ground_xyz: Sequence[Sequence[float]],
    config: ScanStairFreezeConfig,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """按台阶高度和坡度提取楼梯段，连续缓坡不会被判为楼梯。"""

    points = _coerce_points(path_ground_xyz)
    if len(points) < 2:
        return ()
    candidates: list[tuple[int, int, float]] = []
    for index, (start, end) in enumerate(zip(points, points[1:])):
        dz = end[2] - start[2]
        direction = 1 if dz > 0.0 else -1
        horizontal = math.hypot(end[0] - start[0], end[1] - start[1])
        grade = float("inf") if horizontal <= 1.0e-9 else abs(dz) / horizontal
        if (
            abs(dz) >= config.min_step_z_delta_m
            and grade >= config.min_step_grade
        ):
            candidates.append((index, direction, grade))
    if not candidates:
        return ()

    groups: list[list[int]] = []
    group_directions: list[int] = []
    candidate_grades: dict[int, float] = {}
    for segment_index, direction, grade in candidates:
        candidate_grades[segment_index] = grade
        if not groups:
            groups.append([segment_index])
            group_directions.append(direction)
            continue
        previous_segment = groups[-1][-1]
        gap = _polyline_distance(
            points,
            previous_segment + 1,
            segment_index,
        )
        if direction == group_directions[-1] and gap <= config.max_inter_step_gap_m:
            groups[-1].append(segment_index)
        else:
            groups.append([segment_index])
            group_directions.append(direction)

    components: list[_StairComponent] = []
    for group, direction in zip(groups, group_directions, strict=True):
        raw_start = group[0]
        raw_end = group[-1] + 1
        grades = [candidate_grades[index] for index in group]
        tread_evidence = any(
            index not in candidate_grades
            for index in range(group[0], group[-1] + 1)
        )
        # 稀疏 Path 无法仅凭坡度完美区分陡坡与楼梯。启发式 fallback
        # 至少要求多个疑似 riser，并观察到踏面间隔或明显不均匀的 riser；
        # 手工验收路径优先使用与点几何哈希绑定的显式区间。
        if len(group) < 2 or not (
            tread_evidence
            or max(grades) - min(grades)
            >= config.min_riser_grade_variation
        ):
            continue
        z_values = [point[2] for point in points[raw_start : raw_end + 1]]
        if max(z_values) - min(z_values) < config.min_component_z_delta_m:
            continue
        start_index = _extend_start_index(
            points,
            raw_start,
            config.approach_distance_m,
        )
        end_index = _extend_end_index(
            points,
            raw_end,
            config.exit_distance_m,
        )
        component_points = tuple(points[start_index : end_index + 1])
        if len(component_points) < 2:
            continue
        components.append(
            _StairComponent(
                points=component_points,
                source_start_index=start_index,
                source_end_index=end_index,
                direction="up" if direction > 0 else "down",
            )
        )

    # pct_scene 的工程基线从同一方向的首个 riser 一直锁到最后一个 riser，
    # 中间平台也不能释放 root。这里先用局部证据排除连续斜坡，再把同方向且
    # 没有明显反向高差的有效楼梯组连成一次锁定。
    merged = _merge_continuous_stair_components(points, components, config)
    return tuple(component.points for component in merged)


def components_from_stair_segment_indices(
    path_ground_xyz: Sequence[Sequence[float]],
    stair_segment_indices: Sequence[Sequence[int]],
    config: ScanStairFreezeConfig,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """从与 Path 几何哈希绑定的显式点索引构造冻结组件。"""

    points = _coerce_points(path_ground_xyz)
    components: list[_StairComponent] = []
    previous_end = -1
    for raw_index, raw_segment in enumerate(stair_segment_indices):
        if (
            not isinstance(raw_segment, Sequence)
            or isinstance(raw_segment, (str, bytes))
            or len(raw_segment) != 2
        ):
            raise ValueError(
                f"stair_segment_indices[{raw_index}] 必须是 [start, end]。"
            )
        start, end = raw_segment
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError("显式楼梯区间索引必须是整数。")
        if not (0 <= start < end < len(points)):
            raise ValueError(
                f"显式楼梯区间 [{start}, {end}] 超出 Path 点索引。"
            )
        if start <= previous_end:
            raise ValueError("显式楼梯区间必须严格递增且不能重叠。")
        previous_end = end
        z_delta = points[end][2] - points[start][2]
        if abs(z_delta) < config.min_component_z_delta_m:
            raise ValueError("显式楼梯区间总高度变化不足。")
        start_index = _extend_start_index(
            points,
            start,
            config.approach_distance_m,
        )
        end_index = _extend_end_index(
            points,
            end,
            config.exit_distance_m,
        )
        components.append(
            _StairComponent(
                points=tuple(points[start_index : end_index + 1]),
                source_start_index=start_index,
                source_end_index=end_index,
                direction="up" if z_delta > 0.0 else "down",
            )
        )

    merged = _merge_continuous_stair_components(points, components, config)
    return tuple(component.points for component in merged)


def _merge_continuous_stair_components(
    points: tuple[tuple[float, float, float], ...],
    components: Sequence[_StairComponent],
    config: ScanStairFreezeConfig,
) -> list[_StairComponent]:
    """把同一方向楼梯井的多跑楼梯和中间平台合成一次 root 锁定。"""

    merged: list[_StairComponent] = []
    for component in components:
        if not merged:
            merged.append(component)
            continue
        previous = merged[-1]
        overlaps = component.source_start_index <= previous.source_end_index
        same_direction_landing = (
            previous.direction == component.direction
            and previous.direction in {"up", "down"}
            and _landing_preserves_stair_direction(
                points,
                start_index=previous.source_end_index,
                end_index=component.source_start_index,
                direction=previous.direction,
                reverse_tolerance_m=config.min_step_z_delta_m,
            )
        )
        if not overlaps and not same_direction_landing:
            merged.append(component)
            continue
        start_index = previous.source_start_index
        end_index = max(previous.source_end_index, component.source_end_index)
        merged[-1] = _StairComponent(
            points=tuple(points[start_index : end_index + 1]),
            source_start_index=start_index,
            source_end_index=end_index,
            direction=(
                previous.direction
                if previous.direction == component.direction
                else "mixed"
            ),
        )
    return merged


def _landing_preserves_stair_direction(
    points: tuple[tuple[float, float, float], ...],
    *,
    start_index: int,
    end_index: int,
    direction: str,
    reverse_tolerance_m: float,
) -> bool:
    """确认两跑之间只有平台起伏，没有另一段反向楼梯。"""

    if end_index <= start_index:
        return True
    sign = 1.0 if direction == "up" else -1.0
    tolerance = float(reverse_tolerance_m)
    for start, end in zip(
        points[start_index:end_index],
        points[start_index + 1 : end_index + 1],
    ):
        if (end[2] - start[2]) * sign <= -tolerance:
            return False
    return True


class ScanStairFreezeController:
    """把 SCAN 地面参考路径中的楼梯段转换为逐 tick root 锁定动作。"""

    def __init__(self, config: ScanStairFreezeConfig | None = None) -> None:
        self.config = config or ScanStairFreezeConfig()
        self._reset_runtime_state()

    @property
    def certified_progress_seen(self) -> bool:
        """返回是否已有足够 root 弧长进展，可作为本轮执行活动证据。"""

        return self._certified_progress_seen

    @property
    def finish_ready(self) -> bool:
        """返回楼梯执行已可参与导航完成验收。

        普通楼梯组件只有在释放并收到新鲜 SCAN policy 写入后才就绪；若最后
        一个冻结组件本身就是参考 Path 终点，则保持 pct_scene 风格的底盘与
        全身锁定直到导航完成，避免为了验收而在目标处强制释放物理。
        """

        if self._phase in {"not_applicable", "completed"}:
            return True
        return bool(
            self._phase == "terminal_hold"
            and (
                not self.config.require_supervisor_sensor_status
                or self._terminal_supervisor_goal_acknowledged
            )
        )

    @property
    def terminal_supervisor_transition_pending(self) -> bool:
        """返回终点 typed 轨迹已确认、但 supervisor ACK 尚未对齐。"""

        evidence = self._last_controller_status_evidence
        anchor = self._terminal_controller_status_anchor
        return bool(
            self._phase == "terminal_hold"
            and self.config.require_supervisor_sensor_status
            and not self._terminal_supervisor_goal_acknowledged
            and not self._sensor_safety_fault_reasons
            and not self._policy_freeze_write_fault_reasons
            and self._controller_status_is_current_terminal_evidence(evidence)
            and self._controller_status_is_current_terminal_evidence(anchor)
            and evidence is not None
            and anchor is not None
            and evidence.identity == anchor.identity
        )

    @property
    def command_inhibit_active(self) -> bool:
        """返回当前动作是否必须让唯一 cmd_vel owner 写精确零速。"""

        return self._phase in {
            "active",
            "full_lock_settle",
            "root_release_settle",
            "release_action_pending",
            "post_release_stabilizing",
            "terminal_hold",
        }

    def reset(
        self,
        path_ground_xyz: Sequence[Sequence[float]] | None,
        *,
        path_source: str | None = None,
        path_sha256: str | None = None,
        path_points_sha256: str | None = None,
        path_stamp_ns: int | None = None,
        path_terminal_yaw: float | None = None,
        terminal_goal_base_xyzyaw: Sequence[float] | None = None,
        stair_segment_indices: Sequence[Sequence[int]] | None = None,
        carry_object_follow: bool = False,
    ) -> None:
        """载入新代参考路径；传入 z 时禁止预先增加 body height。"""

        self._reset_runtime_state()
        self._carry_object_follow = bool(carry_object_follow)
        self._path_source = path_source
        self._path_sha256 = path_sha256
        self._path_points_sha256 = path_points_sha256
        self._path_stamp_ns = _optional_positive_int(
            path_stamp_ns,
            field_name="path_stamp_ns",
        )
        self._terminal_goal_base_xyzyaw = _optional_xyzyaw(
            terminal_goal_base_xyzyaw,
            field_name="terminal_goal_base_xyzyaw",
        )
        self._stair_segment_indices = (
            None
            if stair_segment_indices is None
            else tuple(tuple(segment) for segment in stair_segment_indices)
        )
        if not self.config.enabled:
            self._phase = "not_applicable"
            self._reason = "disabled"
            return
        if path_ground_xyz is None:
            self._phase = "not_applicable"
            self._reason = "reference_path_unavailable"
            return
        points = _coerce_points(path_ground_xyz)
        self._reference_path = points
        self._path_terminal_yaw = (
            _terminal_path_yaw(points)
            if path_terminal_yaw is None
            else _finite_number(
                path_terminal_yaw,
                field_name="path_terminal_yaw",
            )
        )
        actual_points_sha256 = hash_ground_path_points(points)
        if (
            path_points_sha256 is not None
            and path_points_sha256 != actual_points_sha256
        ):
            raise ValueError("参考路径点几何哈希与载入内容不一致。")
        self._path_points_sha256 = actual_points_sha256
        self._components = (
            extract_stair_components(points, self.config)
            if stair_segment_indices is None
            else components_from_stair_segment_indices(
                points,
                stair_segment_indices,
                self.config,
            )
        )
        if not self._components:
            self._phase = "not_applicable"
            self._reason = "no_stair_component"
            return
        self._bind_terminal_goal_if_required()
        self._phase = "approach"
        self._reason = "ready"

    def compute_action(self, state: SimulationState) -> RobotAction | None:
        """为当前 observation 返回幂等的锁定、解冻或无接管动作。"""

        observation_key = (int(state.step_index), float(state.timestamp))
        if observation_key == self._last_action_observation_key:
            return self._last_action
        self._last_action_observation_key = observation_key

        if self._sensor_acquisition_is_pending():
            self._check_sensor_acquisition_timeout(state)
        if (
            self._policy_freeze_write_fault_reasons
            and self._phase in _ROOT_LOCK_PHASES
        ):
            origin_phase = self._phase
            self._emergency_hold_latched = True
            self._emergency_hold_reason = "stair_policy_freeze_write_fault"
            self._emergency_hold_origin_phase = origin_phase
            self._emergency_hold_full_body_lock = (
                origin_phase in _FULL_BODY_LOCK_PHASES
            )
            self._phase = "failed"
            self._reason = "stair_policy_freeze_write_fault"
            raise RuntimeError(
                "楼梯底盘冻结期间 policy 写零协议非法："
                + ",".join(self._policy_freeze_write_fault_reasons)
            )
        if (
            self._sensor_safety_fault_reasons
            and self._phase in _ROOT_LOCK_PHASES
        ):
            # policy 安全门在底盘锁期间仍必须独立检查 Odometry/点云。
            # 一旦失鲜，先锁存最后 root/关节目标再转入失败态，禁止继续按
            # 仿真时钟盲走；外层执行器随后保持该目标并请求全局重规划。
            origin_phase = self._phase
            self._emergency_hold_latched = True
            self._emergency_hold_reason = "stair_sensor_freshness_fault"
            self._emergency_hold_origin_phase = origin_phase
            self._emergency_hold_full_body_lock = (
                origin_phase in _FULL_BODY_LOCK_PHASES
            )
            self._phase = "failed"
            self._reason = "stair_sensor_freshness_fault"
            raise RuntimeError(
                "楼梯底盘冻结期间 ROS 传感器失鲜："
                + ",".join(self._sensor_safety_fault_reasons)
            )

        if self._phase in {"not_applicable", "completed"}:
            self._last_action = None
            return None
        if self._phase == "terminal_hold":
            self._check_terminal_hold_timeout(state)
            self._last_action = self._locked_action(
                state,
                include_full_body_lock=True,
                source="scan_stair_freeze_terminal_hold",
                inhibit_reason="scan_stair_terminal_hold",
            )
            return self._last_action
        if self._phase == "resume_wait_fresh_cmd":
            self._check_resume_wait_timeout(state)
            self._last_action = None
            return None
        if self._phase == "approach":
            if not self._should_activate(state):
                self._last_action = None
                return None
            self._activate(state)
            self._last_action = self._locked_action(
                state,
                include_full_body_lock=True,
                source="scan_stair_freeze_activated",
            )
            return self._last_action
        if self._phase == "active":
            if self._sensor_acquisition_is_pending():
                # 屏障等待不属于轨迹执行时间。持续刷新积分基准，避免传感器
                # 恢复后的第一拍把整个等待时长一次性积分进 root 目标。
                self._last_timestamp = _finite_number(
                    state.timestamp,
                    field_name="state.timestamp",
                )
                self._reason = "waiting_for_active_generation_sensors"
                self._last_action = self._locked_action(
                    state,
                    include_full_body_lock=True,
                    source="scan_stair_sensor_acquisition_wait",
                )
                return self._last_action
            self._advance_active_target(state)
            self._last_action = self._locked_action(
                state,
                include_full_body_lock=True,
                source=(
                    "scan_stair_freeze_full_lock_settle"
                    if self._phase == "full_lock_settle"
                    else "scan_stair_freeze_active"
                ),
            )
            return self._last_action
        if self._phase == "full_lock_settle":
            self._advance_settle_timer(state, full_body=True)
            if self._phase == "terminal_hold":
                self._last_action = self._locked_action(
                    state,
                    include_full_body_lock=True,
                    source="scan_stair_freeze_terminal_hold",
                    inhibit_reason="scan_stair_terminal_hold",
                )
                return self._last_action
            self._last_action = self._locked_action(
                state,
                include_full_body_lock=(self._phase == "full_lock_settle"),
                source=(
                    "scan_stair_freeze_full_lock_settle"
                    if self._phase == "full_lock_settle"
                    else "scan_stair_freeze_root_release_settle"
                ),
            )
            return self._last_action
        if self._phase == "root_release_settle":
            self._advance_settle_timer(state, full_body=False)
            if self._phase == "release_action_pending":
                self._last_action = self._release_action(state)
            else:
                self._last_action = self._locked_action(
                    state,
                    include_full_body_lock=False,
                    source="scan_stair_freeze_root_release_settle",
                )
            return self._last_action
        if self._phase == "release_action_pending":
            self._last_action = self._release_action(state)
            return self._last_action
        if self._phase == "post_release_stabilizing":
            self._last_action = self._post_release_stabilizing_action(state)
            return self._last_action
        if self._phase == "failed":
            raise RuntimeError(self._reason)
        raise RuntimeError(f"未知 SCAN 楼梯冻结阶段：{self._phase}")

    def emergency_hold_action(
        self,
        state: SimulationState,
        *,
        reason: str,
    ) -> RobotAction | None:
        """在已进入楼梯锁定后锁存最后目标，供急停帧继续保持底盘。"""

        if not isinstance(reason, str) or not reason:
            raise ValueError("楼梯急停保持原因必须是非空字符串。")
        if not self._emergency_hold_latched:
            if self._hold_xyzyaw is None or self._phase not in {
                "active",
                "full_lock_settle",
                "root_release_settle",
                "release_action_pending",
                "terminal_hold",
            }:
                return None
            self._emergency_hold_latched = True
            self._emergency_hold_reason = reason
            self._emergency_hold_origin_phase = self._phase
            self._emergency_hold_full_body_lock = self._phase in {
                "active",
                "full_lock_settle",
                "terminal_hold",
            }
        action = self._locked_action(
            state,
            include_full_body_lock=self._emergency_hold_full_body_lock,
            source="scan_stair_emergency_hold",
        )
        return RobotAction(
            base_velocity=action.base_velocity,
            arm_joint_positions=action.arm_joint_positions,
            gripper_command=action.gripper_command,
            source=action.source,
            metadata={
                **action.metadata,
                "navigation_stair_emergency_hold": True,
                "navigation_stair_emergency_hold_reason": (
                    self._emergency_hold_reason
                ),
                "navigation_stair_emergency_hold_origin_phase": (
                    self._emergency_hold_origin_phase
                ),
                "navigation_cmd_vel_inhibit_reason": (
                    "scan_stair_emergency_hold"
                ),
            },
        )

    def observe_policy_write(
        self,
        raw_report: Any,
        *,
        owner_id: str = "scan_cmd_vel",
    ) -> None:
        """记录接近阶段活动，并只接受解锁后同 owner 的新鲜正常写入。"""

        if not isinstance(raw_report, Mapping):
            return
        sequence = raw_report.get("write_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            if (
                self._phase in _ROOT_LOCK_PHASES
                and self._sensor_acquisition_complete
            ):
                self._latch_policy_freeze_write_fault(
                    ("invalid_policy_write_sequence",),
                    write_sequence=None,
                    timestamp=raw_report.get("timestamp"),
                )
            return
        previous_sequence = self._last_policy_write_sequence_observed
        if previous_sequence is not None:
            if sequence == previous_sequence:
                # executor 的 compute_action/is_done 可能在同一 observation
                # 重复消费同一份 metadata；它不是一次新的 policy 实写。
                return
            if sequence < previous_sequence:
                if (
                    self._phase in _ROOT_LOCK_PHASES
                    and self._sensor_acquisition_complete
                ):
                    self._latch_policy_freeze_write_fault(
                        ("policy_write_sequence_regressed",),
                        write_sequence=sequence,
                        timestamp=raw_report.get("timestamp"),
                    )
                return
        self._last_policy_write_sequence_observed = sequence
        if self._sensor_acquisition_is_pending():
            self._observe_sensor_acquisition_write(
                raw_report,
                sequence=sequence,
                owner_id=owner_id,
            )
            return
        if self._phase in _ROOT_LOCK_PHASES:
            expected_inhibit_reason = self._expected_policy_freeze_inhibit_reason
            if (
                expected_inhibit_reason is None
                or not _policy_write_is_exact_stair_zero_report(
                    raw_report,
                    owner_id=owner_id,
                    inhibit_reason=expected_inhibit_reason,
                )
            ):
                self._latch_policy_freeze_write_fault(
                    ("invalid_stair_freeze_policy_write",),
                    write_sequence=sequence,
                    timestamp=raw_report.get("timestamp"),
                )
                return
            sensor_faults = tuple(
                dict.fromkeys(
                    (
                        *_policy_sensor_freshness_faults(raw_report),
                        *self._supervisor_sensor_freshness_faults(
                            raw_report,
                            require_stair_inhibit_ack=False,
                        ),
                    )
                )
            )
            if sensor_faults:
                self._latch_sensor_safety_fault(
                    sensor_faults,
                    write_sequence=sequence,
                    timestamp=raw_report.get("timestamp"),
                )
                return
        if self._phase == "approach" and _policy_write_has_execution_activity(
            raw_report,
            owner_id=owner_id,
        ):
            self._approach_execution_activity_seen = True
            self._approach_activity_write_sequence = sequence
            raw_timestamp = raw_report.get("timestamp")
            if (
                isinstance(raw_timestamp, (int, float))
                and not isinstance(raw_timestamp, bool)
                and math.isfinite(float(raw_timestamp))
            ):
                self._approach_activity_timestamp = float(raw_timestamp)
        if self._phase != "resume_wait_fresh_cmd":
            return
        resume_report = _parse_resume_policy_write_report(
            raw_report,
            owner_id=owner_id,
        )
        if resume_report is None:
            return
        sequence = resume_report.write_sequence
        timestamp = resume_report.write_timestamp
        if (
            self._release_write_sequence is None
            or self._release_write_timestamp is None
            or self._resume_wait_started_timestamp is None
        ):
            return
        if sequence <= self._release_write_sequence:
            return
        if timestamp <= self._release_write_timestamp:
            return
        release_source_sequences = tuple(
            value
            for value in (
                self._release_cmd_vel_source_sequence,
                self._release_cmd_vel_drain_sequence,
            )
            if value is not None
        )
        if (
            release_source_sequences
            and resume_report.source_sequence <= max(release_source_sequences)
        ):
            return
        release_source_timestamps = tuple(
            value
            for value in (
                self._release_cmd_vel_source_receipt_timestamp,
                self._release_cmd_vel_drain_receipt_timestamp,
            )
            if value is not None
        )
        if (
            release_source_timestamps
            and resume_report.source_receipt_timestamp
            <= max(release_source_timestamps)
        ):
            return
        if (
            resume_report.source_receipt_timestamp
            <= self._release_write_timestamp
        ):
            return
        if (
            self._fresh_controller_execution_identity is None
            or self._fresh_controller_status_receipt_timestamp is None
            or resume_report.source_receipt_timestamp
            <= self._fresh_controller_status_receipt_timestamp
        ):
            return
        self._resume_write_sequence = sequence
        self._resume_write_timestamp = timestamp
        self._resume_cmd_vel_source_sequence = resume_report.source_sequence
        self._resume_cmd_vel_source_receipt_timestamp = (
            resume_report.source_receipt_timestamp
        )
        self._completed_component_count += 1
        if self._component_index + 1 < len(self._components):
            self._component_index += 1
            self._phase = "approach"
            self._reason = "next_component_ready"
            self._clear_active_component_state()
        else:
            self._phase = "completed"
            self._reason = "completed"

    def _supervisor_sensor_freshness_faults(
        self,
        raw_report: Mapping[str, Any],
        *,
        require_stair_inhibit_ack: bool,
    ) -> tuple[str, ...]:
        """校验 supervisor 对本 Path 的状态确认及传感器新鲜度。"""

        if not self.config.require_supervisor_sensor_status:
            return ()
        diagnostics = raw_report.get("navigation_status_observed_report")
        if not isinstance(diagnostics, Mapping):
            return ("missing_supervisor_navigation_status",)
        if (
            diagnostics.get("schema")
            != "navigation_status_observed_diagnostics_v1"
        ):
            return ("invalid_supervisor_navigation_status",)
        if diagnostics.get("status_error") is not None:
            return ("supervisor_navigation_status_error",)
        status = diagnostics.get("status")
        if not isinstance(status, Mapping):
            return ("missing_supervisor_navigation_status",)
        if status.get("identity_valid") is not True:
            return ("supervisor_navigation_identity_invalid",)
        expected_path_stamp_ns = self._path_stamp_ns
        if expected_path_stamp_ns is not None and (
            status.get("active_path_stamp_ns") != expected_path_stamp_ns
            or diagnostics.get("local_active_path_stamp_ns")
            != expected_path_stamp_ns
        ):
            return ("supervisor_navigation_path_mismatch",)
        write_timestamp = raw_report.get("timestamp")
        receipt_timestamp = status.get("receipt_timestamp")
        if (
            isinstance(write_timestamp, bool)
            or not isinstance(write_timestamp, (int, float))
            or not math.isfinite(float(write_timestamp))
            or isinstance(receipt_timestamp, bool)
            or not isinstance(receipt_timestamp, (int, float))
            or not math.isfinite(float(receipt_timestamp))
        ):
            return ("invalid_supervisor_navigation_status_time",)
        status_age = float(write_timestamp) - float(receipt_timestamp)
        if status_age < -1.0e-9:
            return ("supervisor_navigation_status_from_future",)
        if status_age > self.config.supervisor_sensor_status_timeout_s:
            return ("supervisor_navigation_status_timeout",)
        if (
            status.get("global_replan_requested") is not False
            or status.get("global_replan_in_flight") is not False
        ):
            return ("supervisor_stair_freeze_not_acknowledged",)
        stair_inhibit_ack = bool(
            status.get("state") == _NAVIGATION_STATE_EMERGENCY_STOP
            and status.get("reason") == _NAVIGATION_STAIR_INHIBIT_REASON
            and status.get("allow_tracking_command") is False
            and status.get("force_zero_velocity") is True
            and status.get("stop_confirmed") is True
        )
        terminal_transition_causal = bool(
            not require_stair_inhibit_ack
            and self._controller_status_allows_terminal_supervisor_transition(
                write_timestamp=float(write_timestamp),
                supervisor_receipt_timestamp=float(receipt_timestamp),
            )
        )
        terminal_tracking_status = bool(
            not require_stair_inhibit_ack
            and status.get("state") == _NAVIGATION_STATE_TRACKING
            and status.get("reason") == "tracking_inputs_ready"
            and status.get("allow_tracking_command") is True
            and status.get("force_zero_velocity") is False
            and status.get("stop_confirmed") is False
        )
        terminal_tracking_ack = bool(
            terminal_transition_causal and terminal_tracking_status
        )
        terminal_goal_status = bool(
            not require_stair_inhibit_ack
            and status.get("state") == _NAVIGATION_STATE_GOAL_REACHED
            and status.get("reason") == "goal_reached"
            and status.get("allow_tracking_command") is False
            and status.get("force_zero_velocity") is True
            and status.get("stop_confirmed") is True
        )
        terminal_goal_ack = bool(
            terminal_transition_causal
            and terminal_goal_status
            and self._last_controller_status_evidence is not None
            and self._last_controller_status_evidence.state
            == _CONTROLLER_STATE_GOAL_REACHED
            and self._last_controller_status_evidence.receipt_timestamp
            <= float(receipt_timestamp) + 1.0e-9
        )
        terminal_controller_ack = terminal_tracking_ack or terminal_goal_ack
        terminal_tracking_waiting_for_controller = bool(
            terminal_tracking_status
            and not terminal_tracking_ack
            and self._controller_status_can_wait_for_terminal_transition(
                write_timestamp=float(write_timestamp),
                supervisor_receipt_timestamp=float(receipt_timestamp),
            )
        )
        terminal_goal_waiting_for_controller = bool(
            terminal_goal_status and not terminal_goal_ack
        )
        terminal_transition_waiting_for_controller = bool(
            terminal_tracking_waiting_for_controller
            or terminal_goal_waiting_for_controller
        )
        # 代际屏障只能由 typed freeze 的精确 EMERGENCY_STOP ACK 放行。
        # 屏障通过后，末端 final B-spline 可能让 supervisor 合法切换到
        # TRACKING/GOAL_REACHED；该例外仍严格绑定同一 Path 的 typed 状态。
        if (
            not stair_inhibit_ack
            and not terminal_controller_ack
            and not terminal_transition_waiting_for_controller
        ):
            return ("supervisor_stair_freeze_not_acknowledged",)
        pct_plan_id = status.get("pct_plan_id")
        consecutive_scan_failures = status.get("consecutive_scan_failures")
        if (
            isinstance(pct_plan_id, bool)
            or not isinstance(pct_plan_id, int)
            or pct_plan_id < 1
            or isinstance(consecutive_scan_failures, bool)
            or not isinstance(consecutive_scan_failures, int)
            or consecutive_scan_failures < 0
        ):
            return ("invalid_supervisor_stair_freeze_acknowledgement",)
        # ``consecutive_scan_failures`` 是进入冻结前的 SCAN 历史计数，不是
        # Odometry/点云新鲜度。只要 supervisor 已用同 Path 的精确楼梯 ACK
        # 停车，且没有请求或执行 PCT 重规划，非零但未触发重规划的历史计数
        # 不应阻塞 root-lock；否则楼梯入口前一次滚动重规划失败会永久死锁。
        stale_inputs = status.get("stale_inputs")
        if not isinstance(stale_inputs, (list, tuple)) or not all(
            isinstance(value, str) for value in stale_inputs
        ):
            return ("invalid_supervisor_navigation_stale_inputs",)
        faults: list[str] = []
        if "odometry" in stale_inputs:
            faults.append("supervisor_odometry_stale")
        if "point_cloud" in stale_inputs:
            faults.append("supervisor_point_cloud_stale")
        if faults:
            return tuple(faults)
        unexpected_stale_inputs = tuple(
            value
            for value in stale_inputs
            if value not in _NAVIGATION_STAIR_ALLOWED_STALE_INPUTS
        )
        if unexpected_stale_inputs:
            return ("unexpected_supervisor_stair_stale_input",)
        if terminal_transition_waiting_for_controller:
            pending_started = (
                self._terminal_supervisor_goal_pending_started_timestamp
            )
            if pending_started is None:
                self._terminal_supervisor_goal_pending_started_timestamp = (
                    float(write_timestamp)
                )
                return ()
            if (
                float(write_timestamp) - pending_started
                > self.config.supervisor_sensor_status_timeout_s
            ):
                return ("supervisor_terminal_controller_evidence_timeout",)
            return ()
        self._terminal_supervisor_goal_pending_started_timestamp = None
        latest_controller = self._last_controller_status_evidence
        if (
            not require_stair_inhibit_ack
            and self._controller_status_is_current_terminal_evidence(
                latest_controller
            )
            and latest_controller is not None
            and latest_controller.state == _CONTROLLER_STATE_GOAL_REACHED
            and status.get("state") != _NAVIGATION_STATE_GOAL_REACHED
            and float(write_timestamp) - latest_controller.receipt_timestamp
            > self.config.supervisor_sensor_status_timeout_s
        ):
            return ("supervisor_terminal_transition_timeout",)
        if (
            terminal_goal_ack
        ):
            self._terminal_supervisor_goal_acknowledged = True
        return tuple(faults)

    def _controller_status_can_wait_for_terminal_transition(
        self,
        *,
        write_timestamp: float,
        supervisor_receipt_timestamp: float,
    ) -> bool:
        """只为同 Path final 接受状态保留一次跨 topic 有界等待。"""

        evidence = self._last_controller_status_evidence
        if evidence is None or not evidence.accepted:
            return True
        if not bool(
            evidence.trajectory_valid
            and evidence.is_final
            and not evidence.emergency_stop
            and self._path_stamp_ns is not None
            and evidence.identity[0] == self._path_stamp_ns
        ):
            return False
        if evidence.state == 0 and evidence.event in {
            _CONTROLLER_EVENT_ACCEPTED,
            _CONTROLLER_EVENT_DUPLICATE,
        }:
            return True
        anchor = self._terminal_controller_status_anchor
        if (
            evidence.state not in _NAVIGATION_TERMINAL_CONTROLLER_STATES
            or anchor is None
            or anchor.identity != evidence.identity
        ):
            return False
        future_lead = evidence.receipt_timestamp - min(
            write_timestamp,
            supervisor_receipt_timestamp,
        )
        return bool(
            future_lead > 1.0e-9
            and future_lead
            <= self.config.supervisor_sensor_status_timeout_s + 1.0e-9
        )

    def _controller_status_allows_terminal_supervisor_transition(
        self,
        *,
        write_timestamp: float,
        supervisor_receipt_timestamp: float,
    ) -> bool:
        """确认最新 typed controller 是本 Path 的有效 final 轨迹。"""

        evidence = self._last_controller_status_evidence
        anchor = self._terminal_controller_status_anchor
        return bool(
            self._controller_status_is_current_terminal_evidence(evidence)
            and self._controller_status_is_current_terminal_evidence(anchor)
            and evidence is not None
            and anchor is not None
            and evidence.identity == anchor.identity
            and anchor.receipt_timestamp <= write_timestamp + 1.0e-9
            and anchor.receipt_timestamp
            <= supervisor_receipt_timestamp + 1.0e-9
        )

    def _controller_status_is_current_terminal_evidence(
        self,
        evidence: _ControllerStatusEvidence | None,
    ) -> bool:
        """确认 controller 快照属于当前 Path 的有效 final 轨迹。"""

        return bool(
            evidence is not None
            and evidence.accepted
            and evidence.trajectory_valid
            and evidence.is_final
            and not evidence.emergency_stop
            and evidence.state in _NAVIGATION_TERMINAL_CONTROLLER_STATES
            and self._path_stamp_ns is not None
            and evidence.identity[0] == self._path_stamp_ns
        )

    def _update_terminal_controller_status_anchor(
        self,
        evidence: _ControllerStatusEvidence,
    ) -> None:
        """保存同一 final 轨迹最早可被 supervisor 因果确认的快照。"""

        if not self._controller_status_is_current_terminal_evidence(evidence):
            self._terminal_controller_status_anchor = None
            return
        anchor = self._terminal_controller_status_anchor
        if anchor is None:
            self._terminal_controller_status_anchor = evidence
            return
        if anchor.identity == evidence.identity:
            return
        # identity 替换必须伴随 acceptance_sequence 严格递增；否则保留旧锚点，
        # 让后续 supervisor TRACKING 无法被旧 final 证据掩盖。
        if evidence.acceptance_sequence > anchor.acceptance_sequence:
            self._terminal_controller_status_anchor = evidence

    def _sensor_acquisition_is_pending(self) -> bool:
        """返回当前楼梯锁是否仍在等待本 Path 代次的传感器证据。"""

        return bool(
            self._sensor_acquisition_is_required()
            and self._phase in _ROOT_LOCK_PHASES
            and not self._sensor_acquisition_complete
        )

    def _sensor_acquisition_is_required(self) -> bool:
        """返回当前参考 Path 是否需要生产传感器代际屏障。"""

        return bool(
            self.config.require_supervisor_sensor_status and self._components
        )

    def _observe_sensor_acquisition_write(
        self,
        raw_report: Mapping[str, Any],
        *,
        sequence: int,
        owner_id: str,
    ) -> None:
        """消费冻结激活后的新 policy 写入，建立本 Path 代次传感器屏障。"""

        self._sensor_acquisition_last_write_sequence = sequence
        if sequence < 1:
            self._sensor_acquisition_pending_reasons = (
                "awaiting_new_active_generation_policy_write",
            )
            return
        raw_timestamp = raw_report.get("timestamp")
        if (
            isinstance(raw_timestamp, bool)
            or not isinstance(raw_timestamp, (int, float))
            or not math.isfinite(float(raw_timestamp))
            or float(raw_timestamp) < 0.0
        ):
            self._sensor_acquisition_pending_reasons = (
                "invalid_active_generation_policy_write_time",
            )
            return
        timestamp = float(raw_timestamp)
        self._sensor_acquisition_last_write_timestamp = timestamp
        sequence_floor = self._sensor_acquisition_write_sequence_floor
        if sequence_floor is not None and sequence <= sequence_floor:
            self._sensor_acquisition_pending_reasons = (
                "awaiting_new_active_generation_policy_write",
            )
            return
        started_timestamp = self._sensor_acquisition_started_timestamp
        if started_timestamp is None or timestamp <= started_timestamp:
            self._sensor_acquisition_pending_reasons = (
                "awaiting_new_active_generation_policy_write",
            )
            return
        if self._path_stamp_ns is None:
            self._sensor_acquisition_pending_reasons = (
                "missing_active_generation_path_stamp",
            )
            return
        if (
            timestamp - started_timestamp
            > self.config.activation_timeout_s
        ):
            self._sensor_acquisition_pending_reasons = (
                "active_generation_sensor_acquisition_timeout",
            )
            return
        if not _policy_write_is_exact_stair_zero_report(
            raw_report,
            owner_id=owner_id,
            inhibit_reason="scan_stair_freeze",
        ):
            self._sensor_acquisition_pending_reasons = (
                "awaiting_active_generation_freeze_write",
            )
            return

        diagnostics = raw_report.get("navigation_status_observed_report")
        self._sensor_acquisition_last_navigation_status_observed_report = (
            copy.deepcopy(diagnostics)
            if isinstance(diagnostics, Mapping)
            else None
        )
        local_sensor_faults = _policy_sensor_freshness_faults(
            raw_report,
        )
        supervisor_sensor_faults = self._supervisor_sensor_freshness_faults(
            raw_report,
            require_stair_inhibit_ack=True,
        )
        self._sensor_acquisition_local_ready = not local_sensor_faults
        self._sensor_acquisition_supervisor_ready = not supervisor_sensor_faults
        sensor_faults = tuple(
            dict.fromkeys((*local_sensor_faults, *supervisor_sensor_faults))
        )
        if sensor_faults:
            self._sensor_acquisition_pending_reasons = sensor_faults
            return

        status = (
            diagnostics.get("status")
            if isinstance(diagnostics, Mapping)
            else None
        )
        receipt_timestamp = (
            status.get("receipt_timestamp")
            if isinstance(status, Mapping)
            else None
        )
        if (
            isinstance(receipt_timestamp, bool)
            or not isinstance(receipt_timestamp, (int, float))
            or not math.isfinite(float(receipt_timestamp))
            or float(receipt_timestamp) <= started_timestamp
        ):
            self._sensor_acquisition_supervisor_ready = False
            self._sensor_acquisition_pending_reasons = (
                "awaiting_new_active_generation_supervisor_status",
            )
            return

        self._sensor_acquisition_complete = True
        self._sensor_acquisition_completed_timestamp = timestamp
        self._sensor_acquisition_write_sequence = sequence
        self._sensor_acquisition_progress_m_at_completion = float(
            self._progress_m
        )
        self._sensor_acquisition_policy_write_report = copy.deepcopy(raw_report)
        self._sensor_acquisition_navigation_status_observed_report = (
            copy.deepcopy(diagnostics)
        )
        self._sensor_acquisition_pending_reasons = ()
        self._reason = "active_generation_sensors_acquired"

    def _check_sensor_acquisition_timeout(self, state: SimulationState) -> None:
        """在有限等待后把未完成的代际屏障升级为可重规划故障。"""

        timestamp = _finite_number(
            state.timestamp,
            field_name="state.timestamp",
        )
        started_timestamp = self._sensor_acquisition_started_timestamp
        if started_timestamp is None:
            self._sensor_acquisition_started_timestamp = timestamp
            return
        if timestamp < started_timestamp:
            self._latch_sensor_safety_fault(
                ("active_generation_sensor_clock_regressed",),
                write_sequence=self._sensor_acquisition_last_write_sequence,
                timestamp=timestamp,
            )
            return
        if (
            timestamp - started_timestamp
            <= self.config.activation_timeout_s
        ):
            return
        reasons = self._sensor_acquisition_pending_reasons or (
            "active_generation_sensor_acquisition_timeout",
        )
        self._latch_sensor_safety_fault(
            reasons,
            write_sequence=self._sensor_acquisition_last_write_sequence,
            timestamp=timestamp,
        )

    def _latch_sensor_safety_fault(
        self,
        reasons: Sequence[str],
        *,
        write_sequence: int | None,
        timestamp: Any,
    ) -> None:
        """锁存传感器故障证据；调用方负责在下一动作边界急停。"""

        self._sensor_safety_fault_reasons = tuple(dict.fromkeys(reasons))
        self._sensor_safety_fault_write_sequence = write_sequence
        if (
            isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and math.isfinite(float(timestamp))
            and float(timestamp) >= 0.0
        ):
            self._sensor_safety_fault_timestamp = float(timestamp)

    def _latch_policy_freeze_write_fault(
        self,
        reasons: Sequence[str],
        *,
        write_sequence: int | None,
        timestamp: Any,
    ) -> None:
        """锁存唯一 policy owner 的楼梯写零协议故障。"""

        self._policy_freeze_write_fault_reasons = tuple(
            dict.fromkeys(reasons)
        )
        self._policy_freeze_write_fault_sequence = write_sequence
        if (
            isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and math.isfinite(float(timestamp))
            and float(timestamp) >= 0.0
        ):
            self._policy_freeze_write_fault_timestamp = float(timestamp)

    def observe_controller_status(
        self,
        raw_report: Any,
        *,
        expected_topic: str = "/planning/controller_status",
        expected_frame_id: str = "world",
    ) -> None:
        """观察 typed controller 快照，并认证解锁后的新 B-spline。

        只有当前 Path 代次上的全新轨迹接受事件，以及随后处于 yaw 对齐或
        TRACKING 的有效快照，才建立恢复 cmd_vel 所需的控制器证据。拒绝候选
        使用独立 identity，因此不会覆盖已认证的当前轨迹。
        """

        if raw_report is None:
            return
        try:
            evidence = _parse_controller_status_report(
                raw_report,
                expected_topic=expected_topic,
                expected_frame_id=expected_frame_id,
            )
        except (TypeError, ValueError):
            self._invalid_controller_status_count += 1
            return
        previous = self._last_controller_status_evidence
        if previous is not None:
            if (
                evidence.rx_sequence == previous.rx_sequence
                and evidence.status_sequence == previous.status_sequence
                and evidence.identity == previous.identity
                and evidence.event == previous.event
                and evidence.state == previous.state
                and evidence.accepted == previous.accepted
                and evidence.trajectory_valid == previous.trajectory_valid
                and evidence.is_final == previous.is_final
                and evidence.emergency_stop == previous.emergency_stop
            ):
                return
            if (
                evidence.rx_sequence <= previous.rx_sequence
                or evidence.status_sequence <= previous.status_sequence
                or evidence.acceptance_sequence
                < previous.acceptance_sequence
            ):
                self._controller_status_sequence_reset_count += 1
                self._invalid_controller_status_count += 1
                self._clear_fresh_controller_acceptance()
                self._terminal_controller_status_anchor = None
                self._last_controller_status_evidence = evidence
                if self._phase == "resume_wait_fresh_cmd":
                    self._phase = "failed"
                    self._reason = "stair_controller_status_sequence_regressed"
                return
        self._last_controller_status_evidence = evidence
        self._update_terminal_controller_status_anchor(evidence)
        self._controller_status_observation_count += 1
        if self._phase != "resume_wait_fresh_cmd":
            return

        if (
            evidence.identity
            in {
                self._fresh_controller_pending_identity,
                self._fresh_controller_execution_identity,
            }
            and (
                evidence.event == _CONTROLLER_EVENT_INVALIDATED
                or not evidence.accepted
                or not evidence.trajectory_valid
                or evidence.emergency_stop
            )
        ):
            self._clear_fresh_controller_acceptance()
            return
        if (
            evidence.identity == self._fresh_controller_execution_identity
            and evidence.state not in _CONTROLLER_EXECUTION_STATES
        ):
            self._clear_fresh_controller_execution_state()
        if evidence.event in {
            _CONTROLLER_EVENT_REJECTED,
            _CONTROLLER_EVENT_INVALIDATED,
            _CONTROLLER_EVENT_DUPLICATE,
        }:
            return
        if evidence.event not in {
            _CONTROLLER_EVENT_ACCEPTED,
            _CONTROLLER_EVENT_STATE_CHANGED,
        }:
            return
        if not self._controller_status_is_fresh_acceptance(evidence):
            return

        self._fresh_controller_pending_identity = evidence.identity
        self._fresh_controller_acceptance_sequence = (
            evidence.acceptance_sequence
        )
        if evidence.state in _CONTROLLER_EXECUTION_STATES:
            self._fresh_controller_execution_identity = evidence.identity
            self._fresh_controller_status_sequence = evidence.status_sequence
            self._fresh_controller_status_receipt_timestamp = (
                evidence.receipt_timestamp
            )
            self._fresh_controller_status_header_stamp_ns = (
                evidence.header_stamp_ns
            )
        elif self._fresh_controller_execution_identity == evidence.identity:
            self._clear_fresh_controller_execution_state()

    def status(self) -> dict[str, Any]:
        """返回可写入 episode summary 的冻结 provenance 与时序。"""

        target = self._hold_xyzyaw
        return {
            "enabled": bool(self.config.enabled),
            "applicable": bool(self._components),
            "phase": self._phase,
            "reason": self._reason,
            "path_source": self._path_source,
            "path_sha256": self._path_sha256,
            "path_points_sha256": self._path_points_sha256,
            "path_stamp_ns": self._path_stamp_ns,
            "path_terminal_yaw": self._path_terminal_yaw,
            "stair_segment_indices": (
                None
                if self._stair_segment_indices is None
                else [list(segment) for segment in self._stair_segment_indices]
            ),
            "component_source": (
                "geometry_heuristic"
                if self._stair_segment_indices is None
                else "explicit_hash_bound_indices"
            ),
            "height_semantics": "ground_plus_one_body_height_in_freeze_adapter",
            "reference_path_point_count": len(self._reference_path),
            "component_count": len(self._components),
            "component_index": self._component_index,
            "completed_component_count": self._completed_component_count,
            "terminal_component": self._current_component_reaches_path_end(),
            "terminal_hold": self._phase == "terminal_hold",
            "terminal_goal_bound": self._terminal_goal_bound,
            "terminal_goal_base_xyzyaw": (
                None
                if self._terminal_goal_base_xyzyaw is None
                else list(self._terminal_goal_base_xyzyaw)
            ),
            "terminal_path_endpoint_ground_xyz": (
                None
                if self._terminal_path_endpoint_ground_xyz is None
                else list(self._terminal_path_endpoint_ground_xyz)
            ),
            "terminal_goal_xy_error_m": self._terminal_goal_xy_error_m,
            "terminal_goal_z_contract_error_m": (
                self._terminal_goal_z_contract_error_m
            ),
            "terminal_goal_yaw_error_rad": self._terminal_goal_yaw_error_rad,
            "terminal_hold_started_timestamp": (
                self._terminal_hold_started_timestamp
            ),
            "terminal_goal_hold_timeout_s": (
                float(self.config.terminal_goal_hold_timeout_s)
            ),
            "active": self.command_inhibit_active,
            "finish_ready": self.finish_ready,
            "certified_progress_seen": self._certified_progress_seen,
            "progress_m": float(self._progress_m),
            "total_length_m": float(self._total_length_m),
            "progress_ratio": (
                float(self._progress_m / self._total_length_m)
                if self._total_length_m > 1.0e-9
                else 0.0
            ),
            "measured_body_height_m": self._measured_body_height_m,
            "configured_body_height_m": float(self.config.body_height_m),
            "target_body_height_m": (
                float(self.config.body_height_m)
                if self._measured_body_height_m is None
                else max(
                    float(self.config.body_height_m),
                    float(self._measured_body_height_m),
                )
            ),
            "hold_xyzyaw": None if target is None else list(target),
            "settle_remaining_s": float(self._settle_remaining_s),
            "approach_window_entered_timestamp": (
                self._approach_window_entered_timestamp
            ),
            "approach_execution_activity_seen": (
                self._approach_execution_activity_seen
            ),
            "approach_activity_write_sequence": (
                self._approach_activity_write_sequence
            ),
            "approach_activity_timestamp": self._approach_activity_timestamp,
            "approach_started_timestamp": self._approach_started_timestamp,
            "post_release_stable_elapsed_s": float(
                self._post_release_stable_elapsed_s
            ),
            "post_release_started_timestamp": (
                self._post_release_started_timestamp
            ),
            "resume_wait_started_timestamp": self._resume_wait_started_timestamp,
            "post_release_last_linear_speed_mps": (
                self._post_release_last_linear_speed_mps
            ),
            "post_release_last_angular_speed_rps": (
                self._post_release_last_angular_speed_rps
            ),
            "post_release_last_z_error_m": self._post_release_last_z_error_m,
            "post_release_last_tilt_rad": self._post_release_last_tilt_rad,
            "release_write_sequence": self._release_write_sequence,
            "release_write_timestamp": self._release_write_timestamp,
            "release_cmd_vel_drain_sequence": (
                self._release_cmd_vel_drain_sequence
            ),
            "release_cmd_vel_drain_receipt_timestamp": (
                self._release_cmd_vel_drain_receipt_timestamp
            ),
            "release_cmd_vel_source_sequence": (
                self._release_cmd_vel_source_sequence
            ),
            "release_cmd_vel_source_receipt_timestamp": (
                self._release_cmd_vel_source_receipt_timestamp
            ),
            "resume_write_sequence": self._resume_write_sequence,
            "resume_write_timestamp": self._resume_write_timestamp,
            "resume_cmd_vel_source_sequence": (
                self._resume_cmd_vel_source_sequence
            ),
            "resume_cmd_vel_source_receipt_timestamp": (
                self._resume_cmd_vel_source_receipt_timestamp
            ),
            "controller_status_observation_count": (
                self._controller_status_observation_count
            ),
            "invalid_controller_status_count": (
                self._invalid_controller_status_count
            ),
            "controller_status_sequence_reset_count": (
                self._controller_status_sequence_reset_count
            ),
            "last_controller_status_sequence": (
                None
                if self._last_controller_status_evidence is None
                else self._last_controller_status_evidence.status_sequence
            ),
            "last_controller_acceptance_sequence": (
                None
                if self._last_controller_status_evidence is None
                else self._last_controller_status_evidence.acceptance_sequence
            ),
            "last_controller_identity": (
                None
                if self._last_controller_status_evidence is None
                else list(self._last_controller_status_evidence.identity)
            ),
            "last_controller_is_final": (
                None
                if self._last_controller_status_evidence is None
                else self._last_controller_status_evidence.is_final
            ),
            "terminal_supervisor_goal_acknowledged": (
                self._terminal_supervisor_goal_acknowledged
            ),
            "terminal_supervisor_goal_pending_started_timestamp": (
                self._terminal_supervisor_goal_pending_started_timestamp
            ),
            "release_controller_status_sequence": (
                self._release_controller_status_sequence
            ),
            "release_controller_acceptance_sequence": (
                self._release_controller_acceptance_sequence
            ),
            "release_controller_identity": (
                None
                if self._release_controller_identity is None
                else list(self._release_controller_identity)
            ),
            "release_controller_status_receipt_timestamp": (
                self._release_controller_status_receipt_timestamp
            ),
            "release_controller_status_header_stamp_ns": (
                self._release_controller_status_header_stamp_ns
            ),
            "fresh_controller_pending_identity": (
                None
                if self._fresh_controller_pending_identity is None
                else list(self._fresh_controller_pending_identity)
            ),
            "fresh_controller_execution_identity": (
                None
                if self._fresh_controller_execution_identity is None
                else list(self._fresh_controller_execution_identity)
            ),
            "fresh_controller_status_sequence": (
                self._fresh_controller_status_sequence
            ),
            "fresh_controller_acceptance_sequence": (
                self._fresh_controller_acceptance_sequence
            ),
            "fresh_controller_status_receipt_timestamp": (
                self._fresh_controller_status_receipt_timestamp
            ),
            "fresh_controller_status_header_stamp_ns": (
                self._fresh_controller_status_header_stamp_ns
            ),
            "carry_object_follow": self._carry_object_follow,
            "emergency_hold_latched": self._emergency_hold_latched,
            "emergency_hold_reason": self._emergency_hold_reason,
            "emergency_hold_origin_phase": self._emergency_hold_origin_phase,
            "emergency_hold_full_body_lock": (
                self._emergency_hold_full_body_lock
            ),
            "sensor_safety_fault_reasons": list(
                self._sensor_safety_fault_reasons
            ),
            "sensor_safety_fault_write_sequence": (
                self._sensor_safety_fault_write_sequence
            ),
            "sensor_safety_fault_timestamp": (
                self._sensor_safety_fault_timestamp
            ),
            "policy_freeze_write_fault_reasons": list(
                self._policy_freeze_write_fault_reasons
            ),
            "policy_freeze_write_fault_sequence": (
                self._policy_freeze_write_fault_sequence
            ),
            "policy_freeze_write_fault_timestamp": (
                self._policy_freeze_write_fault_timestamp
            ),
            "sensor_acquisition_required": self._sensor_acquisition_is_required(),
            "sensor_acquisition_pending": self._sensor_acquisition_is_pending(),
            "sensor_acquisition_complete": self._sensor_acquisition_complete,
            "sensor_acquisition_started_timestamp": (
                self._sensor_acquisition_started_timestamp
            ),
            "sensor_acquisition_completed_timestamp": (
                self._sensor_acquisition_completed_timestamp
            ),
            "sensor_acquisition_timeout_s": float(
                self.config.activation_timeout_s
            ),
            "sensor_acquisition_write_sequence_floor": (
                self._sensor_acquisition_write_sequence_floor
            ),
            "sensor_acquisition_write_sequence": (
                self._sensor_acquisition_write_sequence
            ),
            "sensor_acquisition_last_write_sequence": (
                self._sensor_acquisition_last_write_sequence
            ),
            "sensor_acquisition_last_write_timestamp": (
                self._sensor_acquisition_last_write_timestamp
            ),
            "sensor_acquisition_pending_reasons": list(
                self._sensor_acquisition_pending_reasons
            ),
            "sensor_acquisition_barrier": {
                "required": self._sensor_acquisition_is_required(),
                "passed": bool(
                    not self._sensor_acquisition_is_required()
                    or self._sensor_acquisition_complete
                ),
                "pending": self._sensor_acquisition_is_pending(),
                "path_stamp_ns": self._path_stamp_ns,
                "activation_timestamp": (
                    self._sensor_acquisition_started_timestamp
                ),
                "timeout_s": float(self.config.activation_timeout_s),
                "status_freshness_timeout_s": float(
                    self.config.supervisor_sensor_status_timeout_s
                ),
                "write_sequence": self._sensor_acquisition_write_sequence,
                "write_timestamp": (
                    self._sensor_acquisition_completed_timestamp
                ),
                "progress_m_at_pass": (
                    self._sensor_acquisition_progress_m_at_completion
                ),
                "local_sensors_fresh": (
                    self._sensor_acquisition_local_ready
                ),
                "supervisor_sensors_fresh": (
                    self._sensor_acquisition_supervisor_ready
                ),
                "last_write_sequence": (
                    self._sensor_acquisition_last_write_sequence
                ),
                "last_write_timestamp": (
                    self._sensor_acquisition_last_write_timestamp
                ),
                "pending_reasons": list(
                    self._sensor_acquisition_pending_reasons
                ),
                "navigation_status_observed_report": copy.deepcopy(
                    self._sensor_acquisition_navigation_status_observed_report
                ),
                "last_navigation_status_observed_report": copy.deepcopy(
                    self._sensor_acquisition_last_navigation_status_observed_report
                ),
                "policy_write_report": copy.deepcopy(
                    self._sensor_acquisition_policy_write_report
                ),
            },
            "non_physical_root_lock_workaround": bool(self._components),
        }

    def _reset_runtime_state(self) -> None:
        self._reference_path: tuple[tuple[float, float, float], ...] = ()
        self._components: tuple[tuple[tuple[float, float, float], ...], ...] = ()
        self._component_index = 0
        self._completed_component_count = 0
        self._phase = "not_applicable"
        self._reason = "not_configured"
        self._path_source: str | None = None
        self._path_sha256: str | None = None
        self._path_points_sha256: str | None = None
        self._path_stamp_ns: int | None = None
        self._path_terminal_yaw: float | None = None
        self._terminal_goal_base_xyzyaw: (
            tuple[float, float, float, float] | None
        ) = None
        self._terminal_goal_bound = False
        self._terminal_path_endpoint_ground_xyz: (
            tuple[float, float, float] | None
        ) = None
        self._terminal_goal_xy_error_m: float | None = None
        self._terminal_goal_z_contract_error_m: float | None = None
        self._terminal_goal_yaw_error_rad: float | None = None
        self._terminal_hold_started_timestamp: float | None = None
        self._stair_segment_indices: tuple[tuple[int, ...], ...] | None = None
        self._carry_object_follow = False
        self._emergency_hold_latched = False
        self._emergency_hold_reason: str | None = None
        self._emergency_hold_origin_phase: str | None = None
        self._emergency_hold_full_body_lock = False
        self._sensor_safety_fault_reasons: tuple[str, ...] = ()
        self._sensor_safety_fault_write_sequence: int | None = None
        self._sensor_safety_fault_timestamp: float | None = None
        self._policy_freeze_write_fault_reasons: tuple[str, ...] = ()
        self._policy_freeze_write_fault_sequence: int | None = None
        self._policy_freeze_write_fault_timestamp: float | None = None
        self._last_policy_write_sequence_observed: int | None = None
        self._expected_policy_freeze_inhibit_reason: str | None = None
        self._sensor_acquisition_complete = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_started_timestamp: float | None = None
        self._sensor_acquisition_completed_timestamp: float | None = None
        self._sensor_acquisition_write_sequence_floor: int | None = None
        self._sensor_acquisition_write_sequence: int | None = None
        self._sensor_acquisition_progress_m_at_completion: float | None = None
        self._sensor_acquisition_local_ready = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_supervisor_ready = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_last_write_sequence: int | None = None
        self._sensor_acquisition_last_write_timestamp: float | None = None
        self._sensor_acquisition_pending_reasons: tuple[str, ...] = ()
        self._sensor_acquisition_navigation_status_observed_report: (
            dict[str, Any] | None
        ) = None
        self._sensor_acquisition_last_navigation_status_observed_report: (
            dict[str, Any] | None
        ) = None
        self._sensor_acquisition_policy_write_report: dict[str, Any] | None = None
        self._active_path: tuple[tuple[float, float, float], ...] = ()
        self._segment_lengths: tuple[float, ...] = ()
        self._total_length_m = 0.0
        self._progress_m = 0.0
        self._measured_body_height_m: float | None = None
        self._hold_xyzyaw: tuple[float, float, float, float] | None = None
        self._last_timestamp: float | None = None
        self._settle_remaining_s = 0.0
        self._approach_window_entered_timestamp: float | None = None
        self._approach_execution_activity_seen = False
        self._approach_activity_write_sequence: int | None = None
        self._approach_activity_timestamp: float | None = None
        self._approach_started_timestamp: float | None = None
        self._post_release_stable_elapsed_s = 0.0
        self._post_release_started_timestamp: float | None = None
        self._resume_wait_started_timestamp: float | None = None
        self._post_release_last_linear_speed_mps: float | None = None
        self._post_release_last_angular_speed_rps: float | None = None
        self._post_release_last_z_error_m: float | None = None
        self._post_release_last_tilt_rad: float | None = None
        self._certified_progress_seen = False
        self._release_write_sequence: int | None = None
        self._release_write_timestamp: float | None = None
        self._release_cmd_vel_drain_sequence: int | None = None
        self._release_cmd_vel_drain_receipt_timestamp: float | None = None
        self._release_cmd_vel_source_sequence: int | None = None
        self._release_cmd_vel_source_receipt_timestamp: float | None = None
        self._resume_write_sequence: int | None = None
        self._resume_write_timestamp: float | None = None
        self._resume_cmd_vel_source_sequence: int | None = None
        self._resume_cmd_vel_source_receipt_timestamp: float | None = None
        self._last_controller_status_evidence: (
            _ControllerStatusEvidence | None
        ) = None
        self._terminal_controller_status_anchor: (
            _ControllerStatusEvidence | None
        ) = None
        self._terminal_supervisor_goal_acknowledged = False
        self._terminal_supervisor_goal_pending_started_timestamp: (
            float | None
        ) = None
        self._controller_status_observation_count = 0
        self._invalid_controller_status_count = 0
        self._controller_status_sequence_reset_count = 0
        self._release_controller_status_sequence: int | None = None
        self._release_controller_acceptance_sequence: int | None = None
        self._release_controller_identity: tuple[int, int, int, int] | None = None
        self._release_controller_status_receipt_timestamp: float | None = None
        self._release_controller_status_header_stamp_ns: int | None = None
        self._fresh_controller_pending_identity: (
            tuple[int, int, int, int] | None
        ) = None
        self._fresh_controller_execution_identity: (
            tuple[int, int, int, int] | None
        ) = None
        self._fresh_controller_status_sequence: int | None = None
        self._fresh_controller_acceptance_sequence: int | None = None
        self._fresh_controller_status_receipt_timestamp: float | None = None
        self._fresh_controller_status_header_stamp_ns: int | None = None
        self._last_action_observation_key: tuple[int, float] | None = None
        self._last_action: RobotAction | None = None

    def _clear_active_component_state(self) -> None:
        self._active_path = ()
        self._segment_lengths = ()
        self._total_length_m = 0.0
        self._progress_m = 0.0
        self._measured_body_height_m = None
        self._hold_xyzyaw = None
        self._last_timestamp = None
        self._settle_remaining_s = 0.0
        self._approach_window_entered_timestamp = None
        self._approach_execution_activity_seen = False
        self._approach_activity_write_sequence = None
        self._approach_activity_timestamp = None
        self._approach_started_timestamp = None
        self._post_release_stable_elapsed_s = 0.0
        self._post_release_started_timestamp = None
        self._resume_wait_started_timestamp = None
        self._post_release_last_linear_speed_mps = None
        self._post_release_last_angular_speed_rps = None
        self._post_release_last_z_error_m = None
        self._post_release_last_tilt_rad = None
        self._release_write_sequence = None
        self._release_write_timestamp = None
        self._release_cmd_vel_drain_sequence = None
        self._release_cmd_vel_drain_receipt_timestamp = None
        self._release_cmd_vel_source_sequence = None
        self._release_cmd_vel_source_receipt_timestamp = None
        self._resume_write_sequence = None
        self._resume_write_timestamp = None
        self._resume_cmd_vel_source_sequence = None
        self._resume_cmd_vel_source_receipt_timestamp = None
        self._release_controller_status_sequence = None
        self._release_controller_acceptance_sequence = None
        self._release_controller_identity = None
        self._release_controller_status_receipt_timestamp = None
        self._release_controller_status_header_stamp_ns = None
        self._clear_fresh_controller_acceptance()
        self._terminal_hold_started_timestamp = None
        self._terminal_supervisor_goal_pending_started_timestamp = None
        self._emergency_hold_latched = False
        self._emergency_hold_reason = None
        self._emergency_hold_origin_phase = None
        self._emergency_hold_full_body_lock = False
        self._sensor_safety_fault_reasons = ()
        self._sensor_safety_fault_write_sequence = None
        self._sensor_safety_fault_timestamp = None
        self._policy_freeze_write_fault_reasons = ()
        self._policy_freeze_write_fault_sequence = None
        self._policy_freeze_write_fault_timestamp = None
        self._expected_policy_freeze_inhibit_reason = None
        self._sensor_acquisition_complete = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_started_timestamp = None
        self._sensor_acquisition_completed_timestamp = None
        self._sensor_acquisition_write_sequence_floor = None
        self._sensor_acquisition_write_sequence = None
        self._sensor_acquisition_progress_m_at_completion = None
        self._sensor_acquisition_local_ready = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_supervisor_ready = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_last_write_sequence = None
        self._sensor_acquisition_last_write_timestamp = None
        self._sensor_acquisition_pending_reasons = ()
        self._sensor_acquisition_navigation_status_observed_report = None
        self._sensor_acquisition_last_navigation_status_observed_report = None
        self._sensor_acquisition_policy_write_report = None

    def _current_component(self) -> tuple[tuple[float, float, float], ...]:
        return self._components[self._component_index]

    def _current_component_reaches_path_end(self) -> bool:
        """确认当前冻结组件是参考 Path 的严格末尾子序列。"""

        if not self._components or not self._reference_path:
            return False
        return self._component_reaches_path_end(self._current_component())

    def _component_reaches_path_end(
        self,
        component: Sequence[Sequence[float]],
    ) -> bool:
        """确认指定冻结组件是当前参考 Path 的严格末尾子序列。"""

        if len(component) > len(self._reference_path):
            return False
        reference_tail = self._reference_path[-len(component) :]
        return all(
            math.dist(component_point, reference_point) <= 1.0e-6
            for component_point, reference_point in zip(
                component,
                reference_tail,
                strict=True,
            )
        )

    def _bind_terminal_goal_if_required(self) -> None:
        """把末端楼梯组件严格绑定到本代 base 高度导航目标。"""

        terminal_components = tuple(
            component
            for component in self._components
            if self._component_reaches_path_end(component)
        )
        if not terminal_components:
            return
        goal = self._terminal_goal_base_xyzyaw
        if goal is None:
            raise ValueError("末端楼梯 Path 缺少本代 terminal NavGoal 绑定。")
        endpoint = self._reference_path[-1]
        self._terminal_path_endpoint_ground_xyz = endpoint
        xy_error = math.hypot(endpoint[0] - goal[0], endpoint[1] - goal[1])
        expected_base_z = endpoint[2] + float(self.config.body_height_m)
        z_error = abs(expected_base_z - goal[2])
        assert self._path_terminal_yaw is not None
        yaw_error = abs(
            _normalize_angle(self._path_terminal_yaw - goal[3])
        )
        self._terminal_goal_xy_error_m = xy_error
        self._terminal_goal_z_contract_error_m = z_error
        self._terminal_goal_yaw_error_rad = yaw_error
        if xy_error > self.config.terminal_goal_xy_tolerance_m:
            raise ValueError(
                "末端楼梯 Path XY 与本代 NavGoal 不一致："
                f"{xy_error:.9f} m"
            )
        if z_error > self.config.terminal_goal_z_tolerance_m:
            raise ValueError(
                "末端楼梯 Path ground+body_height 与本代 NavGoal.z 不一致："
                f"{z_error:.9f} m"
            )
        if yaw_error > self.config.terminal_goal_yaw_tolerance_rad:
            raise ValueError(
                "末端楼梯 Path terminal yaw 与本代 NavGoal.yaw 不一致："
                f"{yaw_error:.9f} rad"
            )
        self._terminal_goal_bound = True

    def _should_activate(self, state: SimulationState) -> bool:
        pose_xy = _root_xy(state)
        component = self._current_component()
        progress, distance, segment_index, segment_ratio = _project_xy_to_polyline(
            component,
            pose_xy,
        )
        timestamp = _finite_number(state.timestamp, field_name="state.timestamp")
        projected_ground = _interpolate_segment(
            component[segment_index],
            component[segment_index + 1],
            segment_ratio,
        )
        root_z = _finite_number(
            state.robot_root_pose[2],
            field_name="robot_root_pose.z",
        )
        measured_height = root_z - projected_ground[2]
        # 回形楼梯或上下楼层可能在 XY 上完全重叠。先用当前有序段的地面
        # 高度约束接管候选，避免在另一个楼层上因纯 XY 投影误触发冻结。
        if not (
            self.config.min_measured_body_height_m
            <= measured_height
            <= self.config.max_measured_body_height_m
        ):
            self._reason = "activation_height_mismatch"
            return False
        start_tangent = _planar_unit_tangent(component[0], component[1])
        end_tangent = _planar_unit_tangent(component[-2], component[-1])
        along_start = (
            (pose_xy[0] - component[0][0]) * start_tangent[0]
            + (pose_xy[1] - component[0][1]) * start_tangent[1]
        )
        along_end = (
            (pose_xy[0] - component[-1][0]) * end_tangent[0]
            + (pose_xy[1] - component[-1][1]) * end_tangent[1]
        )
        total = sum(_segment_lengths_3d(component))
        if (
            progress >= total - 1.0e-6
            and along_end >= -self.config.activation_passed_margin_m
        ):
            self._phase = "failed"
            self._reason = "stair_activation_missed"
            raise RuntimeError(self._reason)
        if (
            along_start >= -self.config.activation_lookahead_m
            and along_end <= self.config.activation_passed_margin_m
            and distance
            <= max(
                self.config.activation_radius_m,
                self.config.activation_lookahead_m,
            )
        ):
            if self._approach_window_entered_timestamp is None:
                self._approach_window_entered_timestamp = timestamp
            # Path 可能在 planner 等待首帧地图时就进入 Isaac。只有唯一
            # policy owner 已实际写入非零命令，才说明楼梯接近执行真正开始；
            # 否则把建图/规划启动时间算入这里会在机器人尚未运动时误超时。
            if self._approach_execution_activity_seen:
                if self._approach_started_timestamp is None:
                    self._approach_started_timestamp = timestamp
                elif (
                    timestamp - self._approach_started_timestamp
                    > self.config.activation_timeout_s
                ):
                    self._phase = "failed"
                    self._reason = "stair_activation_timeout"
                    raise RuntimeError(self._reason)
        self._reason = "approaching_stair"
        if distance > self.config.activation_radius_m:
            return False
        if progress >= max(0.0, total - 1.0e-6):
            return False
        self._reason = "activation_corridor"
        return True
    def _activate(self, state: SimulationState) -> None:
        component = self._current_component()
        pose_xy = _root_xy(state)
        projected_progress, _, segment_index, segment_ratio = _project_xy_to_polyline(
            component,
            pose_xy,
        )
        del projected_progress
        projected_ground = _interpolate_segment(
            component[segment_index],
            component[segment_index + 1],
            segment_ratio,
        )
        root_z = _finite_number(state.robot_root_pose[2], field_name="robot_root_pose.z")
        measured_height = root_z - projected_ground[2]
        if not (
            self.config.min_measured_body_height_m
            <= measured_height
            <= self.config.max_measured_body_height_m
        ):
            raise ValueError(
                "楼梯冻结激活时实测 root-ground 高度超出安全范围："
                f"{measured_height:.6f} m"
            )
        suffix = (
            projected_ground,
            *component[segment_index + 1 :],
        )
        current_ground = (pose_xy[0], pose_xy[1], root_z - measured_height)
        if math.dist(current_ground, suffix[0]) > 1.0e-6:
            suffix = (current_ground, *suffix)
        suffix = _deduplicate_points(suffix)
        lengths = tuple(_segment_lengths_3d(suffix))
        total = float(sum(lengths))
        if len(suffix) < 2 or total <= 1.0e-9:
            raise ValueError("楼梯冻结激活后的剩余路径长度为零。")
        yaw = _root_yaw(state)
        self._active_path = suffix
        self._segment_lengths = lengths
        self._total_length_m = total
        self._progress_m = 0.0
        self._measured_body_height_m = measured_height
        self._hold_xyzyaw = (pose_xy[0], pose_xy[1], root_z, yaw)
        self._last_timestamp = float(state.timestamp)
        self._sensor_acquisition_complete = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_started_timestamp = float(state.timestamp)
        self._sensor_acquisition_completed_timestamp = None
        self._sensor_acquisition_write_sequence_floor = (
            self._last_policy_write_sequence_observed
        )
        self._sensor_acquisition_write_sequence = None
        self._sensor_acquisition_progress_m_at_completion = None
        self._sensor_acquisition_local_ready = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_supervisor_ready = bool(
            not self.config.require_supervisor_sensor_status
        )
        self._sensor_acquisition_last_write_sequence = None
        self._sensor_acquisition_last_write_timestamp = None
        self._sensor_acquisition_pending_reasons = (
            ("awaiting_active_generation_freeze_write",)
            if self.config.require_supervisor_sensor_status
            else ()
        )
        self._sensor_acquisition_navigation_status_observed_report = None
        self._sensor_acquisition_last_navigation_status_observed_report = None
        self._sensor_acquisition_policy_write_report = None
        self._phase = "active"
        self._reason = (
            "waiting_for_active_generation_sensors"
            if self.config.require_supervisor_sensor_status
            else "active"
        )

    def _advance_active_target(self, state: SimulationState) -> None:
        dt = self._control_dt(state)
        self._progress_m = min(
            self._total_length_m,
            self._progress_m + self.config.speed_mps * dt,
        )
        if self._progress_m >= self.config.certified_progress_m:
            self._certified_progress_seen = True
        ground = _interpolate_polyline_3d(
            self._active_path,
            self._segment_lengths,
            self._progress_m,
        )
        lookahead_progress = min(
            self._total_length_m,
            self._progress_m + self.config.yaw_lookahead_m,
        )
        lookahead = _interpolate_polyline_3d(
            self._active_path,
            self._segment_lengths,
            lookahead_progress,
        )
        ratio = min(1.0, self._progress_m / self._total_length_m)
        smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        assert self._measured_body_height_m is not None
        target_body_height = max(
            float(self.config.body_height_m),
            float(self._measured_body_height_m),
        )
        body_height = self._measured_body_height_m + (
            target_body_height - self._measured_body_height_m
        ) * smooth_ratio
        fallback_yaw = (
            _root_yaw(state) if self._hold_xyzyaw is None else self._hold_xyzyaw[3]
        )
        yaw = _path_yaw(ground, lookahead, fallback_yaw=fallback_yaw)
        if self._current_component_reaches_path_end():
            if not self._terminal_goal_bound:
                raise RuntimeError("末端楼梯冻结缺少已验证 NavGoal 绑定。")
            assert self._terminal_goal_base_xyzyaw is not None
            remaining = max(0.0, self._total_length_m - self._progress_m)
            blend_distance = float(self.config.yaw_lookahead_m)
            blend_ratio = (
                1.0
                if blend_distance <= 1.0e-9
                else 1.0 - min(1.0, remaining / blend_distance)
            )
            smooth_blend = blend_ratio * blend_ratio * (3.0 - 2.0 * blend_ratio)
            yaw = _interpolate_yaw_shortest(
                yaw,
                self._terminal_goal_base_xyzyaw[3],
                smooth_blend,
            )
            if self._progress_m >= self._total_length_m - 1.0e-9:
                yaw = _normalize_angle(self._terminal_goal_base_xyzyaw[3])
        self._hold_xyzyaw = (
            ground[0],
            ground[1],
            ground[2] + body_height,
            yaw,
        )
        if self._progress_m >= self._total_length_m - 1.0e-9:
            self._phase = "full_lock_settle"
            self._reason = "full_lock_settle"
            self._settle_remaining_s = self.config.full_lock_settle_time_s

    def _advance_settle_timer(self, state: SimulationState, *, full_body: bool) -> None:
        dt = self._control_dt(state)
        self._settle_remaining_s = max(0.0, self._settle_remaining_s - dt)
        if self._settle_remaining_s > 0.0:
            return
        if full_body:
            if self._current_component_reaches_path_end():
                if not self._terminal_goal_bound:
                    raise RuntimeError("末端楼梯冻结缺少已验证 NavGoal 绑定。")
                # pct_scene 的冻结动作优先于普通局部控制。本代 Path 已在冻结
                # 末端结束时没有后续平地需要接管，因此继续保持最后认证的
                # root/support/full-body 目标，并让 ROS goal + policy 零速完成
                # 验收；此处释放只会重新引入目标点物理倾倒和角速度毛刺。
                self._phase = "terminal_hold"
                self._reason = "terminal_path_hold"
                self._settle_remaining_s = 0.0
                self._terminal_hold_started_timestamp = _finite_number(
                    state.timestamp,
                    field_name="state.timestamp",
                )
                self._completed_component_count += 1
                return
            self._phase = "root_release_settle"
            self._reason = "root_release_settle"
            self._settle_remaining_s = self.config.root_release_settle_time_s
        else:
            self._phase = "release_action_pending"
            self._reason = "release_action_pending"

    def _check_terminal_hold_timeout(self, state: SimulationState) -> None:
        """终点 Bool/零速证据长期缺失时保持最后锁定目标并失败关闭。"""

        timestamp = _finite_number(state.timestamp, field_name="state.timestamp")
        started = self._terminal_hold_started_timestamp
        if started is None:
            raise RuntimeError("terminal_hold 缺少开始时间。")
        elapsed = timestamp - started
        if elapsed < -1.0e-9:
            raise ValueError("terminal_hold 期间仿真时钟发生回退。")
        if elapsed <= self.config.terminal_goal_hold_timeout_s:
            return
        self._emergency_hold_latched = True
        self._emergency_hold_reason = "stair_terminal_goal_hold_timeout"
        self._emergency_hold_origin_phase = "terminal_hold"
        self._emergency_hold_full_body_lock = True
        self._phase = "failed"
        self._reason = "stair_terminal_goal_hold_timeout"
        raise RuntimeError(self._reason)

    def _control_dt(self, state: SimulationState) -> float:
        timestamp = _finite_number(state.timestamp, field_name="state.timestamp")
        if self._last_timestamp is None:
            dt = self.config.default_control_dt_s
        else:
            dt = timestamp - self._last_timestamp
        self._last_timestamp = timestamp
        if dt < -1.0e-9:
            raise ValueError("楼梯冻结期间仿真时钟发生回退。")
        if dt <= 0.0:
            # 首次控制可使用配置周期；一旦已有时间基准，暂停的仿真时钟
            # 必须冻结 root 进度，不能把重复时间戳误当成一个新物理 tick。
            return 0.0
        return min(dt, self.config.max_control_dt_s)

    def _locked_action(
        self,
        state: SimulationState,
        *,
        include_full_body_lock: bool,
        source: str,
        inhibit_reason: str = "scan_stair_freeze",
    ) -> RobotAction:
        del state
        if self._hold_xyzyaw is None:
            raise RuntimeError("楼梯冻结阶段缺少 root hold 目标。")
        # policy 实写出现在下一 observation；保存“刚下发动作”的期望原因，
        # 避免 active→terminal_hold 等阶段切换把上一动作报告误判为伪造。
        self._expected_policy_freeze_inhibit_reason = inhibit_reason
        metadata: dict[str, Any] = {
            "navigation_base_pose_lock": True,
            "navigation_base_pose_lock_phase": self._phase,
            "navigation_base_pose_lock_xyzyaw": self._hold_xyzyaw,
            "navigation_support_joint_lock": True,
            "navigation_support_joint_lock_phase": self._phase,
            "navigation_dog_joint_names": SCAN_STAIR_FREEZE_DOG_JOINT_NAMES,
            "navigation_dog_joint_positions": (
                SCAN_STAIR_FREEZE_DOG_STAND_JOINT_POSITIONS
            ),
            "navigation_scan_stair_freeze": True,
            "navigation_scan_stair_freeze_phase": self._phase,
            "navigation_scan_stair_freeze_progress_m": float(self._progress_m),
            "navigation_scan_stair_freeze_progress_ratio": (
                float(self._progress_m / self._total_length_m)
                if self._total_length_m > 1.0e-9
                else 0.0
            ),
            "navigation_cmd_vel_inhibit": True,
            "navigation_cmd_vel_inhibit_reason": inhibit_reason,
        }
        if include_full_body_lock:
            metadata.update(
                {
                    "navigation_full_body_joint_lock": True,
                    "navigation_full_body_joint_lock_phase": self._phase,
                }
            )
        if self._sensor_acquisition_is_pending():
            metadata.update(
                {
                    "navigation_scan_stair_sensor_acquisition_pending": True,
                    "navigation_scan_stair_sensor_acquisition_path_stamp_ns": (
                        self._path_stamp_ns
                    ),
                    "navigation_scan_stair_sensor_acquisition_timeout_s": float(
                        self.config.activation_timeout_s
                    ),
                    "navigation_scan_stair_sensor_acquisition_pending_reasons": (
                        self._sensor_acquisition_pending_reasons
                    ),
                }
            )
        if self._carry_object_follow:
            metadata["navigation_carry_object_follow"] = True
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            source=source,
            metadata=metadata,
        )

    def _release_action(self, state: SimulationState) -> RobotAction:
        timestamp = _finite_number(
            state.timestamp,
            field_name="state.timestamp",
        )
        self._capture_release_write_marker(state)
        self._phase = "post_release_stabilizing"
        self._reason = "post_release_stabilizing"
        self._post_release_stable_elapsed_s = 0.0
        self._post_release_started_timestamp = timestamp
        self._resume_wait_started_timestamp = None
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            gripper_command="hold" if self._carry_object_follow else None,
            source="scan_stair_freeze_released",
            metadata={
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": "released",
                "navigation_scan_stair_freeze_released": True,
                # 解锁帧与随后自由稳定窗口仍由唯一安全门写零并清空旧命令。
                "navigation_cmd_vel_inhibit": True,
                "navigation_cmd_vel_inhibit_reason": "scan_stair_freeze_release",
            },
        )

    def _post_release_stabilizing_action(
        self,
        state: SimulationState,
    ) -> RobotAction:
        """解除全部锁后用实测速度、姿态和 z 连续验证机体稳定。"""

        timestamp = _finite_number(
            state.timestamp,
            field_name="state.timestamp",
        )
        if self._post_release_started_timestamp is None:
            raise RuntimeError("自由稳定窗口缺少开始时间。")
        post_release_elapsed = timestamp - self._post_release_started_timestamp
        if post_release_elapsed < -1.0e-9:
            raise ValueError("楼梯释放稳定期间仿真时钟发生回退。")
        if (
            post_release_elapsed
            > self.config.post_release_stabilization_timeout_s
        ):
            self._phase = "failed"
            self._reason = "stair_post_release_stabilization_timeout"
            raise RuntimeError(self._reason)

        self._capture_release_write_marker(state)
        dt = self._control_dt(state)
        velocity = tuple(
            _finite_number(value, field_name="robot_root_velocity")
            for value in state.robot_root_velocity
        )
        if len(velocity) != 6:
            raise ValueError("robot_root_velocity 必须包含六个分量。")
        linear_speed = math.sqrt(sum(value * value for value in velocity[:3]))
        # SCAN 闭环接管关心的是平面航向角速度 wz。四足落脚会产生短时
        # roll/pitch 角速度，即使机体倾角始终安全；把三轴范数用于此门会让
        # 解锁稳定计时永久清零。翻滚风险仍由连续 tilt 和线速度门约束。
        angular_speed = abs(velocity[5])
        if self._hold_xyzyaw is None:
            raise RuntimeError("自由稳定窗口缺少楼梯末端 root 目标。")
        root_z = _finite_number(
            state.robot_root_pose[2],
            field_name="robot_root_pose.z",
        )
        z_error = abs(root_z - self._hold_xyzyaw[2])
        roll, pitch = _root_roll_pitch(state)
        tilt = max(abs(roll), abs(pitch))
        self._post_release_last_linear_speed_mps = linear_speed
        self._post_release_last_angular_speed_rps = angular_speed
        self._post_release_last_z_error_m = z_error
        self._post_release_last_tilt_rad = tilt
        stable = (
            linear_speed <= self.config.post_release_max_linear_speed_mps
            and angular_speed <= self.config.post_release_max_angular_speed_rps
            and z_error <= self.config.post_release_max_z_error_m
            and tilt <= self.config.post_release_max_tilt_rad
        )
        if stable:
            self._post_release_stable_elapsed_s += dt
        else:
            self._post_release_stable_elapsed_s = 0.0
        if (
            self._post_release_stable_elapsed_s
            >= self.config.post_release_stable_time_s
        ):
            self._phase = "resume_wait_fresh_cmd"
            self._reason = "released_stable_waiting_for_fresh_cmd"
            self._resume_wait_started_timestamp = timestamp
            self._capture_controller_status_release_baseline()
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            gripper_command="hold" if self._carry_object_follow else None,
            source="scan_stair_freeze_post_release_stabilizing",
            metadata={
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": (
                    "post_release_stabilizing"
                    if self._phase == "post_release_stabilizing"
                    else "released_stable"
                ),
                "navigation_scan_stair_freeze_released": True,
                "navigation_scan_stair_freeze_stable": stable,
                "navigation_scan_stair_freeze_stable_elapsed_s": (
                    self._post_release_stable_elapsed_s
                ),
                "navigation_cmd_vel_inhibit": True,
                "navigation_cmd_vel_inhibit_reason": "scan_stair_freeze_release",
            },
        )

    def _check_resume_wait_timeout(self, state: SimulationState) -> None:
        """在有限时间内等待解锁后的新鲜正常 policy 实写。"""

        timestamp = _finite_number(
            state.timestamp,
            field_name="state.timestamp",
        )
        if self._resume_wait_started_timestamp is None:
            raise RuntimeError("恢复命令等待阶段缺少开始时间。")
        elapsed = timestamp - self._resume_wait_started_timestamp
        if elapsed < -1.0e-9:
            raise ValueError("楼梯恢复命令等待期间仿真时钟发生回退。")
        if elapsed > self.config.resume_wait_fresh_cmd_timeout_s:
            self._phase = "failed"
            self._reason = "stair_resume_fresh_cmd_timeout"
            raise RuntimeError(self._reason)

    def _capture_controller_status_release_baseline(self) -> None:
        """在自由稳定结束时冻结控制器序列下界与当前轨迹 identity。"""

        evidence = self._last_controller_status_evidence
        self._release_controller_status_sequence = (
            None if evidence is None else evidence.status_sequence
        )
        self._release_controller_acceptance_sequence = (
            None if evidence is None else evidence.acceptance_sequence
        )
        self._release_controller_identity = (
            None if evidence is None else evidence.identity
        )
        self._release_controller_status_receipt_timestamp = (
            None if evidence is None else evidence.receipt_timestamp
        )
        self._release_controller_status_header_stamp_ns = (
            None if evidence is None else evidence.header_stamp_ns
        )
        self._clear_fresh_controller_acceptance()

    def _controller_status_is_fresh_acceptance(
        self,
        evidence: _ControllerStatusEvidence,
    ) -> bool:
        """证明当前有效轨迹在释放后新建且绑定同一 Path 代次。"""

        if (
            self._path_stamp_ns is None
            or self._release_write_timestamp is None
            or self._resume_wait_started_timestamp is None
            or not evidence.accepted
            or not evidence.trajectory_valid
            or evidence.emergency_stop
        ):
            return False
        if (
            self._release_controller_status_sequence is not None
            and evidence.status_sequence
            <= self._release_controller_status_sequence
        ):
            return False
        if (
            self._release_controller_acceptance_sequence is not None
            and evidence.acceptance_sequence
            <= self._release_controller_acceptance_sequence
        ):
            return False
        if (
            self._release_controller_identity is not None
            and evidence.identity == self._release_controller_identity
        ):
            return False
        reference_path_stamp_ns, bspline_stamp_ns, start_time_ns, _ = (
            evidence.identity
        )
        if reference_path_stamp_ns != self._path_stamp_ns:
            return False
        release_stamp_ns = int(round(self._release_write_timestamp * 1.0e9))
        if (
            evidence.header_stamp_ns <= release_stamp_ns
            or bspline_stamp_ns <= release_stamp_ns
            or start_time_ns <= release_stamp_ns
            or evidence.receipt_timestamp
            <= self._release_write_timestamp
        ):
            return False
        return True

    def _clear_fresh_controller_acceptance(self) -> None:
        """清除非终点交接中的新轨迹证明，但保留 release 下界。"""

        self._fresh_controller_pending_identity = None
        self._clear_fresh_controller_execution_state()
        self._fresh_controller_acceptance_sequence = None

    def _clear_fresh_controller_execution_state(self) -> None:
        """撤销 TRACKING/ALIGNING 证明，同时保留已接受 identity。"""

        self._fresh_controller_execution_identity = None
        self._fresh_controller_status_sequence = None
        self._fresh_controller_status_receipt_timestamp = None
        self._fresh_controller_status_header_stamp_ns = None

    def _capture_release_write_marker(self, state: SimulationState) -> None:
        """记录冻结写入的递增序号和时间，作为恢复命令排他下界。"""

        raw_report = state.metadata.get("scan_cmd_vel_last_write_report")
        if not isinstance(raw_report, Mapping):
            return
        sequence = raw_report.get("write_sequence")
        raw_timestamp = raw_report.get("timestamp")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or isinstance(raw_timestamp, bool)
            or not isinstance(raw_timestamp, (int, float))
            or not math.isfinite(float(raw_timestamp))
            or float(raw_timestamp) < 0.0
            or raw_report.get("navigation_cmd_vel_inhibited") is not True
        ):
            return
        timestamp = float(raw_timestamp)
        source_marker = _policy_source_marker(
            raw_report,
            sequence_field="cmd_vel_source_sequence",
            timestamp_field="cmd_vel_source_receipt_timestamp",
        )
        drain_marker = _policy_source_marker(
            raw_report,
            sequence_field="last_cmd_vel_drain_sequence",
            timestamp_field="last_cmd_vel_drain_receipt_timestamp",
        )
        if source_marker is not None:
            source_sequence, source_timestamp = source_marker
            if (
                self._release_cmd_vel_source_sequence is None
                or source_sequence > self._release_cmd_vel_source_sequence
                or (
                    source_sequence == self._release_cmd_vel_source_sequence
                    and (
                        self._release_cmd_vel_source_receipt_timestamp is None
                        or source_timestamp
                        > self._release_cmd_vel_source_receipt_timestamp
                    )
                )
            ):
                self._release_cmd_vel_source_sequence = source_sequence
                self._release_cmd_vel_source_receipt_timestamp = (
                    source_timestamp
                )
        if drain_marker is not None:
            drain_sequence, drain_timestamp = drain_marker
            if (
                self._release_cmd_vel_drain_sequence is None
                or drain_sequence > self._release_cmd_vel_drain_sequence
                or (
                    drain_sequence == self._release_cmd_vel_drain_sequence
                    and (
                        self._release_cmd_vel_drain_receipt_timestamp is None
                        or drain_timestamp
                        > self._release_cmd_vel_drain_receipt_timestamp
                    )
                )
            ):
                self._release_cmd_vel_drain_sequence = drain_sequence
                self._release_cmd_vel_drain_receipt_timestamp = drain_timestamp
        if self._release_write_sequence is None:
            self._release_write_sequence = sequence
            self._release_write_timestamp = timestamp
            return
        assert self._release_write_timestamp is not None
        if (
            sequence > self._release_write_sequence
            and timestamp > self._release_write_timestamp
        ):
            self._release_write_sequence = sequence
            self._release_write_timestamp = timestamp


def _policy_sensor_freshness_faults(
    raw_report: Mapping[str, Any],
) -> tuple[str, ...]:
    """从已通过写零协议校验的报告提取独立传感器故障。"""

    stop_reasons = raw_report.get("stop_reasons")
    assert isinstance(stop_reasons, (list, tuple))
    return tuple(
        dict.fromkeys(
            reason
            for reason in stop_reasons
            if reason in _SENSOR_FRESHNESS_STOP_REASONS
        )
    )


def _policy_write_is_exact_stair_zero_report(
    raw_report: Mapping[str, Any],
    *,
    owner_id: str,
    inhibit_reason: str,
) -> bool:
    """确认唯一 owner 按当前楼梯锁阶段执行了精确零速写入。"""

    if raw_report.get("owner_id") != owner_id:
        return False
    if raw_report.get("motion_allowed") is not False:
        return False
    if raw_report.get("navigation_cmd_vel_inhibited") is not True:
        return False
    if raw_report.get("navigation_cmd_vel_inhibit_reason") != inhibit_reason:
        return False
    written = raw_report.get("written_command")
    if (
        not isinstance(written, (list, tuple))
        or len(written) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or abs(float(value)) > 1.0e-9
            for value in written
        )
    ):
        return False
    stop_reasons = raw_report.get("stop_reasons")
    if not isinstance(stop_reasons, (list, tuple)) or not all(
        isinstance(reason, str) for reason in stop_reasons
    ):
        return False
    if len(set(stop_reasons)) != len(stop_reasons):
        return False
    if inhibit_reason not in stop_reasons:
        return False
    allowed_reasons = {
        inhibit_reason,
        *_SENSOR_FRESHNESS_STOP_REASONS,
    }
    return all(reason in allowed_reasons for reason in stop_reasons)


def _policy_write_has_execution_activity(
    raw_report: Mapping[str, Any],
    *,
    owner_id: str,
) -> bool:
    """确认同一唯一 owner 已把有限非零速度实际写入 policy。"""

    if raw_report.get("owner_id") != owner_id:
        return False
    if raw_report.get("motion_allowed") is not True:
        return False
    stop_reasons = raw_report.get("stop_reasons")
    if not isinstance(stop_reasons, (list, tuple)) or stop_reasons:
        return False
    written = raw_report.get("written_command")
    if (
        not isinstance(written, (list, tuple))
        or len(written) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in written
        )
    ):
        return False
    return any(abs(float(value)) > 1.0e-6 for value in written)


def _parse_controller_status_report(
    raw_report: Any,
    *,
    expected_topic: str,
    expected_frame_id: str,
) -> _ControllerStatusEvidence:
    """再次校验 runtime metadata 中的 typed controller 自包含快照。"""

    if not isinstance(raw_report, Mapping):
        raise TypeError("controller status report 必须是映射。")
    required_fields = {
        "source",
        "topic",
        "receipt_timestamp",
        "rx_sequence",
        "header",
        "status_sequence",
        "acceptance_sequence",
        "event",
        "state",
        "reason",
        "accepted",
        "trajectory_valid",
        "is_final",
        "emergency_stop",
        "active_sensing_yaw_only",
        "command_aggregate",
        "identity",
        "candidate",
    }
    if set(raw_report) != required_fields:
        raise ValueError("controller status report 字段集合不完整。")
    if raw_report.get("source") != _CONTROLLER_STATUS_SOURCE:
        raise ValueError("controller status source 不匹配。")
    if raw_report.get("topic") != expected_topic:
        raise ValueError("controller status topic 不匹配。")
    receipt_timestamp = _strict_report_timestamp(
        raw_report.get("receipt_timestamp"),
        field_name="controller_status.receipt_timestamp",
        require_positive=True,
    )
    rx_sequence = _strict_report_integer(
        raw_report.get("rx_sequence"),
        field_name="controller_status.rx_sequence",
        minimum=0,
    )
    header = raw_report.get("header")
    if not isinstance(header, Mapping) or set(header) != {
        "frame_id",
        "stamp",
        "stamp_ns",
    }:
        raise ValueError("controller status header 非法。")
    if header.get("frame_id") != expected_frame_id:
        raise ValueError("controller status frame_id 不匹配。")
    header_stamp_ns = _strict_status_stamp(
        header,
        stamp_field="stamp",
        stamp_ns_field="stamp_ns",
        field_name="controller_status.header",
        require_positive=True,
    )
    status_sequence = _strict_report_integer(
        raw_report.get("status_sequence"),
        field_name="controller_status.status_sequence",
        minimum=1,
    )
    acceptance_sequence = _strict_report_integer(
        raw_report.get("acceptance_sequence"),
        field_name="controller_status.acceptance_sequence",
        minimum=0,
    )
    if acceptance_sequence > status_sequence:
        raise ValueError("controller acceptance_sequence 超过 status_sequence。")
    event = _strict_report_integer(
        raw_report.get("event"),
        field_name="controller_status.event",
        minimum=0,
    )
    state = _strict_report_integer(
        raw_report.get("state"),
        field_name="controller_status.state",
        minimum=0,
    )
    if event not in _CONTROLLER_EVENTS or state not in _CONTROLLER_STATES:
        raise ValueError("controller status 枚举值非法。")
    reason = raw_report.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("controller status reason 必须非空。")
    bool_fields = {
        name: raw_report.get(name)
        for name in (
            "accepted",
            "trajectory_valid",
            "is_final",
            "emergency_stop",
        )
    }
    if any(not isinstance(value, bool) for value in bool_fields.values()):
        raise TypeError("controller status 布尔字段类型非法。")
    accepted = bool_fields["accepted"]
    trajectory_valid = bool_fields["trajectory_valid"]
    is_final = bool_fields["is_final"]
    emergency_stop = bool_fields["emergency_stop"]
    active_sensing_yaw_only = raw_report.get("active_sensing_yaw_only")
    if not isinstance(active_sensing_yaw_only, bool):
        raise TypeError("controller status 主动感知标志类型非法。")
    (
        command_sample_count,
        first_command,
        command_maxima,
    ) = _parse_controller_command_aggregate(
        raw_report.get("command_aggregate")
    )
    identity = _parse_controller_identity(
        raw_report.get("identity"),
        field_name="controller_status.identity",
        require_positive_stamps=accepted,
    )
    candidate = raw_report.get("candidate")
    if event == _CONTROLLER_EVENT_REJECTED:
        if candidate is None:
            raise ValueError("REJECTED 事件缺少 candidate identity。")
        _parse_controller_identity(
            candidate,
            field_name="controller_status.candidate",
            require_positive_stamps=False,
        )
    elif candidate is not None:
        raise ValueError("非 REJECTED 事件不能携带 candidate identity。")
    if trajectory_valid and not accepted:
        raise ValueError("有效轨迹必须已接受。")
    if accepted:
        if acceptance_sequence < 1 or any(value <= 0 for value in identity[:3]):
            raise ValueError("已接受轨迹缺少完整 identity。")
    elif (
        acceptance_sequence != 0
        or identity != (0, 0, 0, 0)
        or trajectory_valid
        or bool_fields["is_final"]
        or emergency_stop
        or active_sensing_yaw_only
        or command_sample_count != 0
    ):
        raise ValueError("未接受状态携带了轨迹 identity 或语义。")
    if active_sensing_yaw_only and (is_final or emergency_stop):
        raise ValueError("主动感知 yaw-only 轨迹不能是 final/emergency_stop。")
    if active_sensing_yaw_only:
        if command_sample_count < 1:
            raise ValueError("主动感知 accepted 状态缺少首条零命令。")
        if any(value != 0.0 for value in first_command):
            raise ValueError("主动感知 first_command 必须严格为零 Twist。")
        if command_maxima[0] != 0.0 or command_maxima[1] != 0.0:
            raise ValueError("主动感知实际命令的 vx/vy 必须严格为零。")
        if command_maxima[2] > 0.20 + 1.0e-12:
            raise ValueError("主动感知实际命令的 |wz| 不得超过 0.20 rad/s。")
    if event in {_CONTROLLER_EVENT_ACCEPTED, _CONTROLLER_EVENT_DUPLICATE} and (
        not accepted or not trajectory_valid
    ):
        raise ValueError("ACCEPTED/DUPLICATE 必须指向有效轨迹。")
    if event == _CONTROLLER_EVENT_INVALIDATED and (
        not accepted or trajectory_valid
    ):
        raise ValueError("INVALIDATED 必须指向已失效轨迹。")
    return _ControllerStatusEvidence(
        receipt_timestamp=receipt_timestamp,
        rx_sequence=rx_sequence,
        header_stamp_ns=header_stamp_ns,
        status_sequence=status_sequence,
        acceptance_sequence=acceptance_sequence,
        event=event,
        state=state,
        accepted=accepted,
        trajectory_valid=trajectory_valid,
        is_final=is_final,
        emergency_stop=emergency_stop,
        identity=identity,
    )


def _parse_controller_command_aggregate(
    raw_aggregate: Any,
) -> tuple[int, tuple[float, ...], tuple[float, float, float]]:
    """严格校验 controller 实际发布的速度命令聚合证据。"""

    if not isinstance(raw_aggregate, Mapping) or set(raw_aggregate) != {
        "sample_count",
        "first_command",
        "max_abs_vx",
        "max_abs_vy",
        "max_abs_wz",
        "violation_count",
    }:
        raise ValueError("controller status command_aggregate 字段集合非法。")
    sample_count = _strict_report_integer(
        raw_aggregate.get("sample_count"),
        field_name="controller_status.command_aggregate.sample_count",
        minimum=0,
    )
    violation_count = _strict_report_integer(
        raw_aggregate.get("violation_count"),
        field_name="controller_status.command_aggregate.violation_count",
        minimum=0,
    )
    if violation_count > sample_count:
        raise ValueError("controller status violation_count 超过 sample_count。")
    first_command = raw_aggregate.get("first_command")
    if (
        not isinstance(first_command, (list, tuple))
        or len(first_command) != 6
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in first_command
        )
    ):
        raise ValueError("controller status first_command 必须是有限六维 Twist。")
    maxima = tuple(
        _finite_number(
            raw_aggregate.get(field_name),
            field_name=f"controller_status.command_aggregate.{field_name}",
        )
        for field_name in ("max_abs_vx", "max_abs_vy", "max_abs_wz")
    )
    if any(value < 0.0 for value in maxima):
        raise ValueError("controller status command max_abs 不能为负数。")
    if sample_count == 0:
        if (
            any(float(value) != 0.0 for value in first_command)
            or any(value != 0.0 for value in maxima)
            or violation_count != 0
        ):
            raise ValueError("零 command sample 要求聚合证据全部为默认值。")
    elif (
        abs(float(first_command[0])) > maxima[0] + 1.0e-12
        or abs(float(first_command[1])) > maxima[1] + 1.0e-12
        or abs(float(first_command[5])) > maxima[2] + 1.0e-12
    ):
        raise ValueError("controller status first_command 超出 max_abs 聚合。")
    if any(float(first_command[index]) != 0.0 for index in (2, 3, 4)):
        raise ValueError("controller status 非平面命令轴必须严格为零。")
    return (
        sample_count,
        tuple(float(value) for value in first_command),
        (float(maxima[0]), float(maxima[1]), float(maxima[2])),
    )


def _parse_controller_identity(
    raw_identity: Any,
    *,
    field_name: str,
    require_positive_stamps: bool = True,
) -> tuple[int, int, int, int]:
    """解析 controller 的四元 B-spline identity。"""

    if not isinstance(raw_identity, Mapping) or set(raw_identity) != {
        "reference_path_stamp",
        "reference_path_stamp_ns",
        "bspline_header_stamp",
        "bspline_header_stamp_ns",
        "start_time",
        "start_time_ns",
        "traj_id",
    }:
        raise ValueError(f"{field_name} 字段集合非法。")
    reference_stamp_ns = _strict_status_stamp(
        raw_identity,
        stamp_field="reference_path_stamp",
        stamp_ns_field="reference_path_stamp_ns",
        field_name=f"{field_name}.reference_path_stamp",
        require_positive=require_positive_stamps,
    )
    bspline_stamp_ns = _strict_status_stamp(
        raw_identity,
        stamp_field="bspline_header_stamp",
        stamp_ns_field="bspline_header_stamp_ns",
        field_name=f"{field_name}.bspline_header_stamp",
        require_positive=require_positive_stamps,
    )
    start_time_ns = _strict_status_stamp(
        raw_identity,
        stamp_field="start_time",
        stamp_ns_field="start_time_ns",
        field_name=f"{field_name}.start_time",
        require_positive=require_positive_stamps,
    )
    traj_id = _strict_report_integer(
        raw_identity.get("traj_id"),
        field_name=f"{field_name}.traj_id",
    )
    return reference_stamp_ns, bspline_stamp_ns, start_time_ns, traj_id


def _strict_status_stamp(
    raw_container: Mapping[str, Any],
    *,
    stamp_field: str,
    stamp_ns_field: str,
    field_name: str,
    require_positive: bool,
) -> int:
    """交叉校验 ROS sec/nanosec 对象与显式整数纳秒。"""

    raw_stamp = raw_container.get(stamp_field)
    if not isinstance(raw_stamp, Mapping) or set(raw_stamp) != {
        "sec",
        "nanosec",
    }:
        raise ValueError(f"{field_name} 必须包含 sec/nanosec。")
    sec = _strict_report_integer(
        raw_stamp.get("sec"),
        field_name=f"{field_name}.sec",
        minimum=0,
    )
    nanosec = _strict_report_integer(
        raw_stamp.get("nanosec"),
        field_name=f"{field_name}.nanosec",
        minimum=0,
    )
    if nanosec >= 1_000_000_000:
        raise ValueError(f"{field_name}.nanosec 超出范围。")
    expected_ns = sec * 1_000_000_000 + nanosec
    explicit_ns = _strict_report_integer(
        raw_container.get(stamp_ns_field),
        field_name=f"{field_name}_ns",
        minimum=0,
    )
    if explicit_ns != expected_ns:
        raise ValueError(f"{field_name} 的整数纳秒交叉校验失败。")
    if require_positive and expected_ns <= 0:
        raise ValueError(f"{field_name} 必须非零。")
    return expected_ns


def _strict_report_integer(
    value: Any,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    """读取 metadata 中不接受 bool 的整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} 必须是整数。")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} 小于允许下限。")
    return value


def _strict_report_timestamp(
    value: Any,
    *,
    field_name: str,
    require_positive: bool,
) -> float:
    """读取连续 ROS 时钟的有限浮点接收时间。"""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{field_name} 必须是有限数值。")
    timestamp = float(value)
    if require_positive and timestamp <= 0.0:
        raise ValueError(f"{field_name} 必须为正。")
    return timestamp


def _parse_resume_policy_write_report(
    raw_report: Mapping[str, Any],
    *,
    owner_id: str,
) -> _ResumePolicyWriteReport | None:
    """严格解析解锁后可恢复 SCAN 接管的完整 policy 实写报告。"""

    sequence = raw_report.get("write_sequence")
    raw_timestamp = raw_report.get("timestamp")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or isinstance(raw_timestamp, bool)
        or not isinstance(raw_timestamp, (int, float))
        or not math.isfinite(float(raw_timestamp))
        or float(raw_timestamp) < 0.0
    ):
        return None
    if raw_report.get("owner_id") != owner_id:
        return None
    if raw_report.get("motion_allowed") is not True:
        return None
    if raw_report.get("navigation_cmd_vel_inhibited") is not False:
        return None
    if raw_report.get("navigation_cmd_vel_inhibit_reason") is not None:
        return None
    if raw_report.get("cmd_vel_sample_received_this_tick") is not True:
        return None
    if raw_report.get("cmd_vel_sample_drained_this_tick") is not False:
        return None
    stop_reasons = raw_report.get("stop_reasons")
    if not isinstance(stop_reasons, (list, tuple)) or stop_reasons:
        return None
    requested = _finite_planar_command(raw_report.get("requested_command"))
    written = _finite_planar_command(raw_report.get("written_command"))
    if requested is None or written is None:
        return None
    source_marker = _policy_source_marker(
        raw_report,
        sequence_field="cmd_vel_source_sequence",
        timestamp_field="cmd_vel_source_receipt_timestamp",
    )
    if source_marker is None:
        return None
    source_sequence, source_receipt_timestamp = source_marker
    if source_sequence < 1:
        return None
    write_timestamp = float(raw_timestamp)
    if source_receipt_timestamp > write_timestamp + 1.0e-9:
        return None
    drain_marker = _policy_source_marker(
        raw_report,
        sequence_field="last_cmd_vel_drain_sequence",
        timestamp_field="last_cmd_vel_drain_receipt_timestamp",
    )
    if drain_marker is not None:
        drain_sequence, drain_receipt_timestamp = drain_marker
        if source_sequence <= drain_sequence:
            return None
        if source_receipt_timestamp <= drain_receipt_timestamp:
            return None
    else:
        drain_sequence = None
        drain_receipt_timestamp = None
    return _ResumePolicyWriteReport(
        write_sequence=sequence,
        write_timestamp=write_timestamp,
        source_sequence=source_sequence,
        source_receipt_timestamp=source_receipt_timestamp,
        drain_sequence=drain_sequence,
        drain_receipt_timestamp=drain_receipt_timestamp,
    )


def _policy_source_marker(
    raw_report: Mapping[str, Any],
    *,
    sequence_field: str,
    timestamp_field: str,
) -> tuple[int, float] | None:
    """严格读取 cmd_vel counter 与连续 ROS 接收时间的成对标记。"""

    sequence = raw_report.get(sequence_field)
    raw_timestamp = raw_report.get(timestamp_field)
    if sequence is None and raw_timestamp is None:
        return None
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or isinstance(raw_timestamp, bool)
        or not isinstance(raw_timestamp, (int, float))
        or not math.isfinite(float(raw_timestamp))
        or float(raw_timestamp) <= 0.0
    ):
        return None
    return sequence, float(raw_timestamp)


def _finite_planar_command(raw_command: Any) -> tuple[float, float, float] | None:
    """把完整的有限三轴命令转成 tuple；缺字段和非法值均拒绝。"""

    if (
        not isinstance(raw_command, (list, tuple))
        or len(raw_command) != 3
    ):
        return None
    command: list[float] = []
    for value in raw_command:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        command.append(float(value))
    return command[0], command[1], command[2]


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是有限数值。")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数值。")
    return number


def _prepare_flattened_points(
    raw_points: Any,
    *,
    min_point_distance_m: float,
) -> tuple[tuple[float, float, float], ...]:
    if (
        not isinstance(raw_points, Sequence)
        or isinstance(raw_points, (str, bytes))
        or len(raw_points) < 6
        or len(raw_points) % 3 != 0
    ):
        raise ValueError("points_xyz 必须包含至少两个完整 xyz 点。")
    points = tuple(
        (
            _finite_number(raw_points[index], field_name=f"points_xyz[{index}]"),
            _finite_number(
                raw_points[index + 1],
                field_name=f"points_xyz[{index + 1}]",
            ),
            _finite_number(
                raw_points[index + 2],
                field_name=f"points_xyz[{index + 2}]",
            ),
        )
        for index in range(0, len(raw_points), 3)
    )
    deduplicated: list[tuple[float, float, float]] = []
    for point in points:
        if deduplicated and math.dist(deduplicated[-1], point) <= min_point_distance_m:
            continue
        deduplicated.append(point)
    if len(deduplicated) < 2:
        raise ValueError("移除相邻近重复点后，参考路径不足两个点。")
    return tuple(deduplicated)


def _coerce_points(
    raw_points: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    if isinstance(raw_points, (str, bytes)):
        raise ValueError("参考路径必须是 xyz 点列。")
    points: list[tuple[float, float, float]] = []
    for index, raw_point in enumerate(raw_points):
        if (
            not isinstance(raw_point, Sequence)
            or isinstance(raw_point, (str, bytes))
            or len(raw_point) < 3
        ):
            raise ValueError(f"reference_path[{index}] 必须是 xyz 点。")
        point = (
            _finite_number(raw_point[0], field_name=f"reference_path[{index}].x"),
            _finite_number(raw_point[1], field_name=f"reference_path[{index}].y"),
            _finite_number(raw_point[2], field_name=f"reference_path[{index}].z"),
        )
        if points and math.dist(points[-1], point) <= 1.0e-6:
            continue
        points.append(point)
    return tuple(points)


def _polyline_distance(
    points: Sequence[tuple[float, float, float]],
    start_index: int,
    end_index: int,
) -> float:
    if end_index <= start_index:
        return 0.0
    return sum(
        math.dist(points[index], points[index + 1])
        for index in range(start_index, end_index)
    )


def _extend_start_index(
    points: Sequence[tuple[float, float, float]],
    start_index: int,
    distance_m: float,
) -> int:
    remaining = distance_m
    index = start_index
    while index > 0 and remaining > 0.0:
        remaining -= math.dist(points[index - 1], points[index])
        index -= 1
    return index


def _extend_end_index(
    points: Sequence[tuple[float, float, float]],
    end_index: int,
    distance_m: float,
) -> int:
    remaining = distance_m
    index = end_index
    while index < len(points) - 1 and remaining > 0.0:
        remaining -= math.dist(points[index], points[index + 1])
        index += 1
    return index


def _deduplicate_points(
    points: Sequence[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], ...]:
    output: list[tuple[float, float, float]] = []
    for point in points:
        if output and math.dist(output[-1], point) <= 1.0e-9:
            continue
        output.append(point)
    return tuple(output)


def _segment_lengths_3d(
    path: Sequence[tuple[float, float, float]],
) -> list[float]:
    return [math.dist(start, end) for start, end in zip(path, path[1:])]


def _interpolate_segment(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    ratio: float,
) -> tuple[float, float, float]:
    clamped = max(0.0, min(1.0, float(ratio)))
    return tuple(
        start[index] + clamped * (end[index] - start[index])
        for index in range(3)
    )  # type: ignore[return-value]


def _interpolate_polyline_3d(
    path: Sequence[tuple[float, float, float]],
    lengths: Sequence[float],
    progress_m: float,
) -> tuple[float, float, float]:
    if not path:
        raise ValueError("楼梯冻结路径不能为空。")
    remaining = max(0.0, float(progress_m))
    for index, length in enumerate(lengths):
        if remaining <= length or index == len(lengths) - 1:
            ratio = 0.0 if length <= 1.0e-9 else remaining / length
            return _interpolate_segment(path[index], path[index + 1], ratio)
        remaining -= length
    return path[-1]


def _project_xy_to_polyline(
    path: Sequence[tuple[float, float, float]],
    point_xy: tuple[float, float],
) -> tuple[float, float, int, float]:
    """返回 3D 弧长进度、XY 横距、段索引和段内比例。"""

    if len(path) < 2:
        raise ValueError("投影路径至少需要两个点。")
    lengths = _segment_lengths_3d(path)
    best = (0.0, float("inf"), 0, 0.0)
    accumulated = 0.0
    for index, (start, end) in enumerate(zip(path, path[1:])):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        planar_sq = dx * dx + dy * dy
        if planar_sq <= 1.0e-12:
            ratio = 0.0
        else:
            ratio = (
                (point_xy[0] - start[0]) * dx
                + (point_xy[1] - start[1]) * dy
            ) / planar_sq
            ratio = max(0.0, min(1.0, ratio))
        projected_x = start[0] + ratio * dx
        projected_y = start[1] + ratio * dy
        distance = math.hypot(point_xy[0] - projected_x, point_xy[1] - projected_y)
        if distance < best[1]:
            best = (
                accumulated + ratio * lengths[index],
                distance,
                index,
                ratio,
            )
        accumulated += lengths[index]
    return best


def _path_yaw(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    fallback_yaw: float,
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if math.hypot(dx, dy) <= 1.0e-9:
        return float(fallback_yaw)
    return math.atan2(dy, dx)


def _terminal_path_yaw(
    path: Sequence[tuple[float, float, float]],
) -> float:
    """返回与手工 Path publisher 一致的最后有效平面切向。"""

    if len(path) < 2:
        raise ValueError("参考 Path 至少需要两个点才能确定 terminal yaw。")
    terminal = path[-1]
    for previous in reversed(path[:-1]):
        if math.hypot(terminal[0] - previous[0], terminal[1] - previous[1]) > 1.0e-9:
            return math.atan2(
                terminal[1] - previous[1],
                terminal[0] - previous[0],
            )
    raise ValueError("参考 Path 末端没有可用的平面方向。")


def _normalize_angle(value: float) -> float:
    """把角度归一化到 ``[-pi, pi)``。"""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _interpolate_yaw_shortest(start: float, end: float, ratio: float) -> float:
    """沿最短角差插值，避免跨越 ``±pi`` 时绕远路。"""

    clamped = max(0.0, min(1.0, float(ratio)))
    delta = _normalize_angle(float(end) - float(start))
    return _normalize_angle(float(start) + clamped * delta)


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    """严格解析可选正整数。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} 必须是正整数。")
    return value


def _optional_xyzyaw(
    value: Sequence[float] | None,
    *,
    field_name: str,
) -> tuple[float, float, float, float] | None:
    """严格解析可选的 base ``x、y、z、yaw`` 目标。"""

    if value is None:
        return None
    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{field_name} 必须包含四个有限数值。")
    parsed = tuple(
        _finite_number(component, field_name=f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )
    return parsed  # type: ignore[return-value]


def _planar_unit_tangent(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> tuple[float, float]:
    """返回非退化 Path 段的 XY 单位切向。"""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm <= 1.0e-9:
        raise ValueError("楼梯组件包含 XY 退化段，无法判定入口方向。")
    return (dx / norm, dy / norm)


def _root_xy(state: SimulationState) -> tuple[float, float]:
    return (
        _finite_number(state.robot_root_pose[0], field_name="robot_root_pose.x"),
        _finite_number(state.robot_root_pose[1], field_name="robot_root_pose.y"),
    )


def _root_yaw(state: SimulationState) -> float:
    w, x, y, z = (
        _finite_number(value, field_name="robot_root_pose.quaternion")
        for value in state.robot_root_pose[3:7]
    )
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("robot_root_pose 四元数范数为零。")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _root_roll_pitch(state: SimulationState) -> tuple[float, float]:
    """从 WXYZ 四元数返回归一化后的 roll、pitch。"""

    w, x, y, z = (
        _finite_number(value, field_name="robot_root_pose.quaternion")
        for value in state.robot_root_pose[3:7]
    )
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("robot_root_pose 四元数范数为零。")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(pitch_sine)
