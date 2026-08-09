"""PCT 多楼层全局规划器适配器。"""

from __future__ import annotations

import json
import itertools
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.interfaces.navigation import NavGoal, NavPlan
from source.interfaces.simulation import SimulationState


@dataclass(frozen=True)
class PCTPlannerConfig:
    enabled: bool = False
    planner_root: Path | None = None
    server_script: Path | None = None
    server_python: Path | None = None
    tomogram_name: str = "mutifloor"
    tomogram_path: Path | None = None
    walkable_path: Path | None = None
    collision_ply_path: Path | None = None
    global_vertical_obstacle_min_slices: int = 7
    cross_floor_vertical_obstacle_min_slices: int = 9
    cross_floor_gateway_points: tuple[tuple[float, float, float], ...] = (
        (1.5, 5.7, 0.6),
    )
    cross_floor_stair_exit_points: tuple[tuple[float, float, float], ...] = (
        (2.70, 7.05, 3.0),
    )
    cross_floor_stair_midpoint_points: tuple[tuple[float, float, float], ...] = (
        (1.51822, 6.27683, 0.29486),
        (2.74512, 9.14634, 1.64666),
        (1.9202, 9.52807, 1.71919),
        (2.69841, 7.79872, 2.61031),
    )
    cross_floor_gateway_radius_m: float = 0.6
    robot_root_to_floor_m: float = 0.45
    body_obstacle_min_height_m: float = 0.30
    body_obstacle_max_height_m: float = 1.0
    stair_min_horizontal_per_slice_m: float = 0.40
    stair_max_horizontal_per_slice_m: float = 0.90
    stair_vertical_radius_m: float = 0.60
    stair_progress_tolerance: float = 0.35
    stair_progress_cost_weight: float = 20.0
    obstacle_clearance_radius_m: float = 0.60
    obstacle_clearance_cost_weight: float = 2.0
    startup_timeout_s: float = 30.0
    request_timeout_s: float = 10.0
    coord_mode: str = "sim_to_pct_180deg"
    pct_offset_x: float = 0.0
    pct_offset_y: float = 0.0
    pct_offset_z: float = 0.0
    pct_scale_x: float = 1.0
    pct_scale_y: float = 1.0
    pct_scale_z: float = 1.0
    pct_rotation_x_rad: float = 0.0
    pct_rotation_y_rad: float = 0.0
    pct_rotation_z_rad: float = 0.0
    fallback_to_astar: bool = False

    def __post_init__(self) -> None:
        """拒绝旧 PCT→A* 回退开关，避免规划失败后静默换链。"""

        if self.fallback_to_astar:
            raise ValueError("pct-scan 分支禁止 PCT→A* fallback")


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def _xyz(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) < 3:
        raise ValueError("expected an xyz sequence with at least three values")
    xyz = (float(values[0]), float(values[1]), float(values[2]))
    if not all(math.isfinite(value) for value in xyz):
        raise ValueError("xyz coordinates must be finite")
    return xyz


def _refine_cross_floor_stair_centerline(
    path_3d: tuple[tuple[float, float, float], ...],
    config: PCTPlannerConfig,
) -> tuple[tuple[tuple[float, float, float], ...], dict[str, Any]]:
    """用实测楼梯中心线替换 PCT 切片路径中的锯齿和扶手横切段。"""

    report: dict[str, Any] = {
        "applied": False,
        "reason": "stair_anchors_unavailable",
        "raw_point_count": len(path_3d),
        "sample_spacing_m": 0.20,
    }
    if (
        len(path_3d) < 2
        or not config.cross_floor_gateway_points
        or not config.cross_floor_stair_exit_points
    ):
        return path_3d, report

    gateway = config.cross_floor_gateway_points[0]
    stair_exit = config.cross_floor_stair_exit_points[0]
    midpoints = _ordered_stair_centerline_midpoints(
        gateway,
        stair_exit,
        config.cross_floor_stair_midpoint_points,
    )
    anchors = _normalized_stair_centerline_anchors(
        (gateway, *midpoints, stair_exit)
    )
    if len(anchors) < 2:
        return path_3d, report
    if float(path_3d[-1][2]) < float(path_3d[0][2]) - 0.25:
        report["reason"] = "descending_refinement_unavailable"
        return path_3d, report

    anchors, partial_route = _anchors_for_observed_stair_extent(
        path_3d,
        anchors,
    )
    if len(anchors) < 2:
        report["reason"] = "partial_stair_anchor_unavailable"
        return path_3d, report

    start_index = min(
        range(len(path_3d) - 1),
        key=lambda index: _stair_anchor_path_score(path_3d[index], anchors[0]),
    )
    end_index = min(
        range(start_index + 1, len(path_3d)),
        key=lambda index: _stair_anchor_path_score(path_3d[index], anchors[-1]),
    )
    if end_index <= start_index:
        report["reason"] = "stair_path_indices_invalid"
        return path_3d, report

    adjusted_anchors = list(anchors)
    adjusted_anchors[0] = (
        float(adjusted_anchors[0][0]),
        float(adjusted_anchors[0][1]),
        float(path_3d[start_index][2]),
    )
    adjusted_anchors[-1] = (
        float(adjusted_anchors[-1][0]),
        float(adjusted_anchors[-1][1]),
        max(float(adjusted_anchors[-1][2]), float(path_3d[end_index][2])),
    )
    approach_start_index = _stair_approach_start_index(
        path_3d,
        stair_start_index=start_index,
        gateway=adjusted_anchors[0],
    )
    approach = _sample_stair_approach_curve(
        path_3d,
        start_index=approach_start_index,
        gateway=adjusted_anchors[0],
        spacing_m=0.15,
    )
    stair_centerline = _sample_stair_centerline(
        tuple(adjusted_anchors),
        spacing_m=0.20,
    )
    centerline = _deduplicate_path_3d((*approach, *stair_centerline))
    refined = _deduplicate_path_3d(
        (
            *path_3d[:approach_start_index],
            *centerline,
            *path_3d[end_index + 1 :],
        )
    )
    report.update(
        {
            "applied": True,
            "reason": (
                "calibrated_stair_centerline_partial"
                if partial_route
                else "calibrated_stair_centerline"
            ),
            "raw_start_index": int(start_index),
            "raw_end_index": int(end_index),
            "raw_start": list(path_3d[start_index]),
            "raw_end": list(path_3d[end_index]),
            "approach_start_index": int(approach_start_index),
            "approach_start": list(path_3d[approach_start_index]),
            "approach_point_count": len(approach),
            "centerline_anchors": [list(point) for point in adjusted_anchors],
            "centerline_point_count": len(centerline),
            "refined_point_count": len(refined),
        }
    )
    return refined, report


def _anchors_for_observed_stair_extent(
    path_3d: tuple[tuple[float, float, float], ...],
    anchors: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[tuple[float, float, float], ...], bool]:
    """部分上楼目标只保留当前路径真正到达的标定锚点。

    旧逻辑会把落在一层半平台的目标先扩展到整段楼梯出口，再沿原始网格路径
    倒退到目标。这里用原始路径最高切片与末点 XY 选择最后一个可达锚点，避免
    生成越过目标、穿过楼板或随后回头的全局路径。
    """

    if len(anchors) < 2 or len(path_3d) < 2:
        return anchors, False
    maximum_path_z = max(float(point[2]) for point in path_3d)
    terminal_anchor_z = float(anchors[-1][2])
    if maximum_path_z >= terminal_anchor_z - 0.30:
        return anchors, False

    candidate_indices = [
        index
        for index, anchor in enumerate(anchors[1:], start=1)
        if float(anchor[2]) <= maximum_path_z + 0.30
    ]
    if not candidate_indices:
        return anchors[:1], True
    raw_terminal = path_3d[-1]
    terminal_index = min(
        candidate_indices,
        key=lambda index: _stair_anchor_path_score(
            raw_terminal,
            anchors[index],
        ),
    )
    return anchors[: terminal_index + 1], terminal_index < len(anchors) - 1


def _ordered_stair_centerline_midpoints(
    gateway: tuple[float, float, float],
    stair_exit: tuple[float, float, float],
    midpoints: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    """按高度和平台拓扑排列无序实测点，避免平台两端顺序反转。"""

    if not midpoints:
        return ()
    ascending = float(stair_exit[2]) >= float(gateway[2])
    sorted_points = sorted(
        midpoints,
        key=lambda point: float(point[2]),
        reverse=not ascending,
    )
    clusters: list[list[tuple[float, float, float]]] = []
    for point in sorted_points:
        if not clusters or abs(float(point[2]) - float(clusters[-1][-1][2])) > 0.25:
            clusters.append([point])
        else:
            clusters[-1].append(point)

    ordered: list[tuple[float, float, float]] = []
    previous = gateway
    for cluster_index, cluster in enumerate(clusters):
        next_anchor = (
            clusters[cluster_index + 1][0]
            if cluster_index + 1 < len(clusters)
            else stair_exit
        )
        candidates = itertools.permutations(cluster)
        plateau = min(
            candidates,
            key=lambda candidate: (
                _xy_distance(previous, candidate[0])
                + sum(
                    _xy_distance(start, end)
                    for start, end in zip(candidate, candidate[1:])
                )
                + _xy_distance(candidate[-1], next_anchor)
            ),
        )
        ordered.extend(plateau)
        previous = plateau[-1]
    return tuple(ordered)


def _normalized_stair_centerline_anchors(
    anchors: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    """保持中心线高度沿行走顺序单调，吸收平台测量的厘米级高度差。"""

    if len(anchors) < 2:
        return anchors
    points = [list(map(float, point)) for point in anchors]
    ascending = points[-1][2] >= points[0][2]
    if ascending:
        points[0][2] = min(points[0][2], points[1][2])
        for index in range(1, len(points)):
            points[index][2] = max(points[index][2], points[index - 1][2])
    else:
        points[0][2] = max(points[0][2], points[1][2])
        for index in range(1, len(points)):
            points[index][2] = min(points[index][2], points[index - 1][2])
    return tuple(tuple(point) for point in points)


def _sample_stair_centerline(
    anchors: tuple[tuple[float, float, float], ...],
    *,
    spacing_m: float,
) -> tuple[tuple[float, float, float], ...]:
    """逐段线性采样楼梯中心线，并保留每个平台控制点。"""

    output: list[tuple[float, float, float]] = [anchors[0]]
    for start, end in zip(anchors, anchors[1:]):
        horizontal = _xy_distance(start, end)
        steps = max(1, int(math.ceil(horizontal / max(spacing_m, 1.0e-3))))
        for step in range(1, steps + 1):
            alpha = step / steps
            output.append(
                (
                    float(start[0]) + alpha * (float(end[0]) - float(start[0])),
                    float(start[1]) + alpha * (float(end[1]) - float(start[1])),
                    float(start[2]) + alpha * (float(end[2]) - float(start[2])),
                )
            )
    return tuple(output)


def _stair_approach_start_index(
    path_3d: tuple[tuple[float, float, float], ...],
    *,
    stair_start_index: int,
    gateway: tuple[float, float, float],
) -> int:
    """在扶手区域之前选择足够长的平层点，用于提前完成楼梯入口转向。"""

    # 楼梯 smoke 会直接从 gateway 出发，此时入口可能就是路径第 0 点。
    selected = max(0, stair_start_index - 1)
    gateway_z = float(gateway[2])
    for index in range(stair_start_index - 1, -1, -1):
        point = path_3d[index]
        if abs(float(point[2]) - gateway_z) > 0.10:
            continue
        selected = index
        if _xy_distance(point, gateway) >= 1.20:
            break
    return selected


def _sample_stair_approach_curve(
    path_3d: tuple[tuple[float, float, float], ...],
    *,
    start_index: int,
    gateway: tuple[float, float, float],
    spacing_m: float,
) -> tuple[tuple[float, float, float], ...]:
    """以平层来向和楼梯朝向为切线，生成入口前的三次 Bezier 缓弯。"""

    start = path_3d[start_index]
    previous = path_3d[max(0, start_index - 1)]
    incoming_x = float(start[0]) - float(previous[0])
    incoming_y = float(start[1]) - float(previous[1])
    incoming_norm = math.hypot(incoming_x, incoming_y)
    if incoming_norm <= 1.0e-6:
        # smoke 从 gateway 附近起步时没有上一段来向，改用入口方向避免横向回钩。
        incoming_x = float(gateway[0]) - float(start[0])
        incoming_y = float(gateway[1]) - float(start[1])
        incoming_norm = math.hypot(incoming_x, incoming_y)
    if incoming_norm <= 1.0e-6:
        incoming_x, incoming_y, incoming_norm = 1.0, 0.0, 1.0
    incoming_x /= incoming_norm
    incoming_y /= incoming_norm

    distance = _xy_distance(start, gateway)
    # 控制柄不能长于短入口本身，否则 snap 点附近会产生先远离再折返的路径。
    handle = min(0.55, 0.40 * distance)
    control_1 = (
        float(start[0]) + handle * incoming_x,
        float(start[1]) + handle * incoming_y,
    )
    # 当前实测楼梯从 gateway 沿 +Y 上升，终点控制柄放在入口下方。
    control_2 = (float(gateway[0]), float(gateway[1]) - handle)
    sample_count = max(2, int(math.ceil(distance / max(spacing_m, 1.0e-3))))
    output: list[tuple[float, float, float]] = []
    for sample_index in range(sample_count + 1):
        t = sample_index / sample_count
        one_minus_t = 1.0 - t
        x = (
            one_minus_t**3 * float(start[0])
            + 3.0 * one_minus_t**2 * t * control_1[0]
            + 3.0 * one_minus_t * t**2 * control_2[0]
            + t**3 * float(gateway[0])
        )
        y = (
            one_minus_t**3 * float(start[1])
            + 3.0 * one_minus_t**2 * t * control_1[1]
            + 3.0 * one_minus_t * t**2 * control_2[1]
            + t**3 * float(gateway[1])
        )
        z = float(start[2]) + t * (float(gateway[2]) - float(start[2]))
        output.append((x, y, z))
    return tuple(output)


def _stair_anchor_path_score(
    point: tuple[float, float, float],
    anchor: tuple[float, float, float],
) -> float:
    return _xy_distance(point, anchor) + 0.20 * abs(float(point[2]) - float(anchor[2]))


def _xy_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _deduplicate_path_3d(
    points: Sequence[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], ...]:
    output: list[tuple[float, float, float]] = []
    for point in points:
        normalized = tuple(float(value) for value in point[:3])
        if output and math.dist(output[-1], normalized) <= 1.0e-6:
            continue
        output.append(normalized)
    return tuple(output)


def _validate_coord_config(
    *,
    coord_mode: str,
    pct_offset_x: float,
    pct_offset_y: float,
    pct_offset_z: float,
    pct_scale_x: float,
    pct_scale_y: float,
    pct_scale_z: float,
    pct_rotation_x_rad: float,
    pct_rotation_y_rad: float,
    pct_rotation_z_rad: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    if coord_mode not in {"sim_to_pct_180deg", "identity"}:
        raise ValueError(f"unsupported PCT coord_mode: {coord_mode}")
    offset = (float(pct_offset_x), float(pct_offset_y), float(pct_offset_z))
    scale = (float(pct_scale_x), float(pct_scale_y), float(pct_scale_z))
    rotation = (
        float(pct_rotation_x_rad),
        float(pct_rotation_y_rad),
        float(pct_rotation_z_rad),
    )
    if not all(
        math.isfinite(value)
        for value in (*offset, *scale, *rotation)
    ):
        raise ValueError("PCT coordinate transform values must be finite")
    if any(value == 0.0 for value in scale):
        raise ValueError("PCT coordinate scales must be non-zero")
    return offset, scale, rotation


def _rotation_matrix_xyz(
    rotation_xyz_rad: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """返回固定轴 XYZ 欧拉旋转矩阵 ``Rz @ Ry @ Rx``。"""

    rx, ry, rz = rotation_xyz_rad
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    return (
        (
            cz * cy,
            cz * sy * sx - sz * cx,
            cz * sy * cx + sz * sx,
        ),
        (
            sz * cy,
            sz * sy * sx + cz * cx,
            sz * sy * cx - cz * sx,
        ),
        (-sy, cy * sx, cy * cx),
    )


def _matrix_vector(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        sum(matrix[0][index] * vector[index] for index in range(3)),
        sum(matrix[1][index] * vector[index] for index in range(3)),
        sum(matrix[2][index] * vector[index] for index in range(3)),
    )


def _transpose(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _coord_mode_signs(coord_mode: str) -> tuple[float, float, float]:
    if coord_mode == "identity":
        return (1.0, 1.0, 1.0)
    return (-1.0, -1.0, 1.0)


def _finite_transform_result(
    xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    if not all(math.isfinite(value) for value in xyz):
        raise ValueError("PCT coordinate transform produced non-finite output")
    return xyz


def sim_to_pct_xyz(
    xyz: Sequence[float],
    *,
    coord_mode: str = "sim_to_pct_180deg",
    pct_offset_x: float = 0.0,
    pct_offset_y: float = 0.0,
    pct_offset_z: float = 0.0,
    pct_scale_x: float = 1.0,
    pct_scale_y: float = 1.0,
    pct_scale_z: float = 1.0,
    pct_rotation_x_rad: float = 0.0,
    pct_rotation_y_rad: float = 0.0,
    pct_rotation_z_rad: float = 0.0,
) -> tuple[float, float, float]:
    """将 Sim 世界坐标转换到 PCT。

    固定顺序为：``coord_mode`` 基础轴向、逐轴 scale、固定轴 X/Y/Z
    欧拉旋转（矩阵 ``Rz @ Ry @ Rx``）、最后 XYZ offset。默认配置严格保持
    ``pct=(-sim_x, -sim_y, sim_z)``。
    """

    offset, scale, rotation = _validate_coord_config(
        coord_mode=coord_mode,
        pct_offset_x=pct_offset_x,
        pct_offset_y=pct_offset_y,
        pct_offset_z=pct_offset_z,
        pct_scale_x=pct_scale_x,
        pct_scale_y=pct_scale_y,
        pct_scale_z=pct_scale_z,
        pct_rotation_x_rad=pct_rotation_x_rad,
        pct_rotation_y_rad=pct_rotation_y_rad,
        pct_rotation_z_rad=pct_rotation_z_rad,
    )
    sim = _xyz(xyz)
    signs = _coord_mode_signs(coord_mode)
    scaled = (
        sim[0] * signs[0] * scale[0],
        sim[1] * signs[1] * scale[1],
        sim[2] * signs[2] * scale[2],
    )
    rotated = _matrix_vector(_rotation_matrix_xyz(rotation), scaled)
    return _finite_transform_result(
        (
            rotated[0] + offset[0],
            rotated[1] + offset[1],
            rotated[2] + offset[2],
        )
    )


def pct_to_sim_xyz(
    xyz: Sequence[float],
    *,
    coord_mode: str = "sim_to_pct_180deg",
    pct_offset_x: float = 0.0,
    pct_offset_y: float = 0.0,
    pct_offset_z: float = 0.0,
    pct_scale_x: float = 1.0,
    pct_scale_y: float = 1.0,
    pct_scale_z: float = 1.0,
    pct_rotation_x_rad: float = 0.0,
    pct_rotation_y_rad: float = 0.0,
    pct_rotation_z_rad: float = 0.0,
) -> tuple[float, float, float]:
    """按 ``sim_to_pct_xyz`` 的精确逆序恢复 Sim 世界坐标。"""

    offset, scale, rotation = _validate_coord_config(
        coord_mode=coord_mode,
        pct_offset_x=pct_offset_x,
        pct_offset_y=pct_offset_y,
        pct_offset_z=pct_offset_z,
        pct_scale_x=pct_scale_x,
        pct_scale_y=pct_scale_y,
        pct_scale_z=pct_scale_z,
        pct_rotation_x_rad=pct_rotation_x_rad,
        pct_rotation_y_rad=pct_rotation_y_rad,
        pct_rotation_z_rad=pct_rotation_z_rad,
    )
    pct = _xyz(xyz)
    translated = (
        pct[0] - offset[0],
        pct[1] - offset[1],
        pct[2] - offset[2],
    )
    unrotated = _matrix_vector(
        _transpose(_rotation_matrix_xyz(rotation)),
        translated,
    )
    signs = _coord_mode_signs(coord_mode)
    sim = (
        unrotated[0] / scale[0] * signs[0],
        unrotated[1] / scale[1] * signs[1],
        unrotated[2] / scale[2] * signs[2],
    )
    return _finite_transform_result(sim)


class PCTPlannerClient:
    """通过子进程封装 PCT stdin/stdout JSON 规划协议。"""

    def __init__(self, config: PCTPlannerConfig):
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._recent_lines: deque[str] = deque(maxlen=40)

    def plan(
        self,
        *,
        start: Sequence[float],
        end: Sequence[float],
    ) -> dict[str, Any]:
        self.start()
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError("PCT server stdin is unavailable")
        request = {"start": list(_xyz(start)), "end": list(_xyz(end))}
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("PCT server pipe closed while sending request") from exc

        deadline = time.monotonic() + float(self.config.request_timeout_s)
        while time.monotonic() < deadline:
            self._raise_if_exited()
            remaining = max(0.01, deadline - time.monotonic())
            try:
                line = self._stdout_queue.get(timeout=min(0.10, remaining))
            except queue.Empty:
                continue
            payload = self._decode_json_line(line)
            if payload is None:
                continue
            if payload.get("status") == "ok" and isinstance(payload.get("traj"), list):
                return payload
            raise RuntimeError(f"PCT planner returned non-ok response: {payload}")
        raise TimeoutError(
            "PCT planner request timed out after "
            f"{float(self.config.request_timeout_s):.1f}s; recent server output: "
            f"{self._recent_output()}"
        )

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        server_python = self._server_python()
        server_script = self._server_script()
        env = self._server_env()
        cwd = self._server_cwd(server_script)
        self._stdout_queue = queue.Queue()
        self._recent_lines.clear()
        self._process = subprocess.Popen(
            [str(server_python), str(server_script)],
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process,),
            name="pct-planner-stdout",
            daemon=True,
        )
        self._stdout_thread.start()
        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        self._process = None

    def _server_python(self) -> Path:
        path = self.config.server_python or _path_from_env("PCT_SERVER_PYTHON")
        if path is None:
            return Path(sys.executable)
        return Path(path).expanduser()

    def _server_script(self) -> Path:
        if self.config.server_script is not None:
            return Path(self.config.server_script).expanduser()
        root = self.config.planner_root or _path_from_env("PCT_PLANNER_ROOT")
        if root is None:
            raise ValueError(
                "PCT server script is not configured; pass pct_server_script "
                "or use the pipeline factory default scripts/navigation/pct_grid_server.py"
            )
        return Path(root).expanduser() / "scripts/navigation/pct_server.py"

    def _server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        root = self.config.planner_root or _path_from_env("PCT_PLANNER_ROOT")
        tomogram_path = self.config.tomogram_path or _path_from_env("PCT_TOMOGRAM_PATH")
        walkable_path = self.config.walkable_path or _path_from_env("PCT_WALKABLE_PATH")
        collision_ply_path = (
            self.config.collision_ply_path
            or _path_from_env("PCT_COLLISION_PLY_PATH")
        )
        if root is not None:
            env["PCT_PLANNER_ROOT"] = str(Path(root).expanduser())
        env["PCT_TOMOGRAM_NAME"] = str(self.config.tomogram_name)
        if tomogram_path is not None:
            env["PCT_TOMOGRAM_PATH"] = str(Path(tomogram_path).expanduser())
        if walkable_path is not None:
            env["PCT_WALKABLE_PATH"] = str(Path(walkable_path).expanduser())
        if collision_ply_path is not None:
            env["PCT_COLLISION_PLY_PATH"] = str(
                Path(collision_ply_path).expanduser()
            )
        env["PCT_GLOBAL_VERTICAL_OBSTACLE_MIN_SLICES"] = str(
            int(self.config.global_vertical_obstacle_min_slices)
        )
        env["PCT_CROSS_FLOOR_VERTICAL_OBSTACLE_MIN_SLICES"] = str(
            int(self.config.cross_floor_vertical_obstacle_min_slices)
        )
        if self.config.cross_floor_gateway_points:
            gateways_pct = [
                list(self._sim_to_pct_gateway(gateway))
                for gateway in self.config.cross_floor_gateway_points
            ]
            env["PCT_CROSS_FLOOR_GATEWAYS_PCT"] = json.dumps(
                gateways_pct,
                separators=(",", ":"),
            )
            env["PCT_CROSS_FLOOR_GATEWAY_RADIUS_M"] = str(
                float(self.config.cross_floor_gateway_radius_m)
            )
        if self.config.cross_floor_stair_exit_points:
            stair_exits_pct = [
                list(self._sim_to_pct_gateway(stair_exit))
                for stair_exit in self.config.cross_floor_stair_exit_points
            ]
            env["PCT_CROSS_FLOOR_STAIR_EXITS_PCT"] = json.dumps(
                stair_exits_pct,
                separators=(",", ":"),
            )
        if self.config.cross_floor_stair_midpoint_points:
            stair_midpoints_pct = [
                list(self._sim_to_pct_gateway(stair_midpoint))
                for stair_midpoint in self.config.cross_floor_stair_midpoint_points
            ]
            env["PCT_CROSS_FLOOR_STAIR_MIDPOINTS_PCT"] = json.dumps(
                stair_midpoints_pct,
                separators=(",", ":"),
            )
        env["PCT_ROBOT_ROOT_TO_FLOOR_M"] = str(
            float(self.config.robot_root_to_floor_m)
        )
        env["PCT_BODY_OBSTACLE_MIN_HEIGHT_M"] = str(
            float(self.config.body_obstacle_min_height_m)
        )
        env["PCT_BODY_OBSTACLE_MAX_HEIGHT_M"] = str(
            float(self.config.body_obstacle_max_height_m)
        )
        env["PCT_STAIR_MIN_HORIZONTAL_PER_SLICE_M"] = str(
            float(self.config.stair_min_horizontal_per_slice_m)
        )
        env["PCT_STAIR_MAX_HORIZONTAL_PER_SLICE_M"] = str(
            float(self.config.stair_max_horizontal_per_slice_m)
        )
        env["PCT_STAIR_VERTICAL_RADIUS_M"] = str(
            float(self.config.stair_vertical_radius_m)
        )
        env["PCT_STAIR_PROGRESS_TOLERANCE"] = str(
            float(self.config.stair_progress_tolerance)
        )
        env["PCT_STAIR_PROGRESS_COST_WEIGHT"] = str(
            float(self.config.stair_progress_cost_weight)
        )
        env["PCT_OBSTACLE_CLEARANCE_RADIUS_M"] = str(
            float(self.config.obstacle_clearance_radius_m)
        )
        env["PCT_OBSTACLE_CLEARANCE_COST_WEIGHT"] = str(
            float(self.config.obstacle_clearance_cost_weight)
        )
        return env

    def _server_cwd(self, server_script: Path) -> Path | None:
        if self.config.planner_root is not None:
            return Path(self.config.planner_root).expanduser()
        root = _path_from_env("PCT_PLANNER_ROOT")
        if root is not None:
            return root
        return server_script.parent

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + float(self.config.startup_timeout_s)
        while time.monotonic() < deadline:
            self._raise_if_exited()
            try:
                line = self._stdout_queue.get(timeout=0.10)
            except queue.Empty:
                continue
            stripped = line.strip()
            if "READY" in stripped.upper():
                return
            payload = self._decode_json_line(line)
            if payload is not None and payload.get("status") in {"ready", "ok"}:
                return
        raise TimeoutError(
            "PCT server did not report READY within "
            f"{float(self.config.startup_timeout_s):.1f}s; recent server output: "
            f"{self._recent_output()}"
        )

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self._recent_lines.append(line.rstrip("\n"))
            self._stdout_queue.put(line)

    def _decode_json_line(self, line: str) -> dict[str, Any] | None:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RuntimeError("PCT server process has not been started")
        self._raise_if_exited()
        return self._process

    def _raise_if_exited(self) -> None:
        process = self._process
        if process is None:
            return
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"PCT server exited with code {returncode}; recent server output: "
                f"{self._recent_output()}"
            )

    def _recent_output(self) -> str:
        if not self._recent_lines:
            return "<none>"
        return " | ".join(self._recent_lines)

    def __enter__(self) -> "PCTPlannerClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _sim_to_pct_gateway(
        self,
        xyz: Sequence[float],
    ) -> tuple[float, float, float]:
        return sim_to_pct_xyz(
            xyz,
            coord_mode=self.config.coord_mode,
            pct_offset_x=self.config.pct_offset_x,
            pct_offset_y=self.config.pct_offset_y,
            pct_offset_z=self.config.pct_offset_z,
            pct_scale_x=self.config.pct_scale_x,
            pct_scale_y=self.config.pct_scale_y,
            pct_scale_z=self.config.pct_scale_z,
            pct_rotation_x_rad=self.config.pct_rotation_x_rad,
            pct_rotation_y_rad=self.config.pct_rotation_y_rad,
            pct_rotation_z_rad=self.config.pct_rotation_z_rad,
        )


class PCTNavPlanner:
    """将 PCT server 适配为当前 pipeline 使用的全局导航规划器。"""

    def __init__(
        self,
        config: PCTPlannerConfig,
        *,
        client: PCTPlannerClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or PCTPlannerClient(config)

    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        try:
            return self._plan_with_pct(state, goal)
        except Exception as exc:
            raise RuntimeError(f"PCT global planning failed: {exc}") from exc

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _plan_with_pct(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        if not self.config.enabled:
            raise RuntimeError("PCT planner is disabled")
        start_sim = (
            float(state.robot_root_pose[0]),
            float(state.robot_root_pose[1]),
            float(state.robot_root_pose[2]),
        )
        goal_z_missing = goal.z is None
        end_z = start_sim[2] if goal.z is None else float(goal.z)
        end_sim = (float(goal.x), float(goal.y), end_z)
        start_pct = self._sim_to_pct(start_sim)
        end_pct = self._sim_to_pct(end_sim)
        response = self.client.plan(start=start_pct, end=end_pct)
        raw_traj = response.get("traj")
        if not isinstance(raw_traj, list) or len(raw_traj) < 2:
            raise RuntimeError("PCT planner returned fewer than two trajectory points")
        raw_path_3d = tuple(self._pct_to_sim(point) for point in raw_traj)
        if response.get("cross_floor") is True:
            path_3d, stair_centerline_refinement = (
                _refine_cross_floor_stair_centerline(raw_path_3d, self.config)
            )
        else:
            path_3d = raw_path_3d
            stair_centerline_refinement = {
                "applied": False,
                "reason": "same_floor_plan",
                "raw_point_count": len(raw_path_3d),
            }
        waypoints_xy = tuple((float(x), float(y)) for x, y, _z in path_3d)
        metadata = {
            "planner": "pct",
            "path_3d": path_3d,
            "pct_raw_path_3d": raw_path_3d,
            "stair_centerline_refinement": stair_centerline_refinement,
            "sim_start": start_sim,
            "slice_start": _first_present(response, "slice_start", "start_slice", "slice_id_start"),
            "slice_end": _first_present(response, "slice_end", "end_slice", "slice_id_end"),
            "snap_start_dist": _first_present(response, "snap_start_dist", "start_snap_dist"),
            "snap_end_dist": _first_present(response, "snap_end_dist", "end_snap_dist"),
            "snap_start_distance_m": response.get("snap_start_distance_m"),
            "snap_end_distance_m": response.get("snap_end_distance_m"),
            "snapped_start_pct_xyz": response.get("snapped_start_xyz"),
            "snapped_end_pct_xyz": response.get("snapped_end_xyz"),
            "snapped_start_slice": response.get("snapped_start_slice"),
            "snapped_end_slice": response.get("snapped_end_slice"),
            "snap_start_slice_delta": response.get(
                "snap_start_slice_delta"
            ),
            "snap_end_slice_delta": response.get("snap_end_slice_delta"),
            "goal_z_missing": goal_z_missing,
            "goal_z_source": "robot_root_pose" if goal_z_missing else "goal",
            "coord_mode": self.config.coord_mode,
            "coordinate_transform_order": (
                "coord_mode_then_scale_then_fixed_xyz_rotation_then_offset"
            ),
            "pct_start": start_pct,
            "pct_end": end_pct,
            "pct_status": response.get("status"),
            "pct_path_mode": response.get("path_mode"),
            "hard_obstacle_cells": response.get("hard_obstacle_cells"),
            "hard_obstacle_mode": response.get("hard_obstacle_mode"),
            "hard_obstacle_min_slices": response.get(
                "hard_obstacle_min_slices"
            ),
            "cross_floor": response.get("cross_floor"),
            "default_hard_obstacle_min_slices": response.get(
                "default_hard_obstacle_min_slices"
            ),
            "cross_floor_hard_obstacle_min_slices": response.get(
                "cross_floor_hard_obstacle_min_slices"
            ),
            "cross_floor_gateway_count": response.get("cross_floor_gateway_count"),
            "cross_floor_stair_exit_count": response.get(
                "cross_floor_stair_exit_count"
            ),
            "cross_floor_stair_midpoint_count": response.get(
                "cross_floor_stair_midpoint_count"
            ),
            "cross_floor_gateway_radius_m": response.get(
                "cross_floor_gateway_radius_m"
            ),
            "cross_floor_gateway_cells": response.get("cross_floor_gateway_cells"),
            "cross_floor_stair_vertical_cells": response.get(
                "cross_floor_stair_vertical_cells"
            ),
            "cross_floor_gateway_mode": response.get("cross_floor_gateway_mode"),
            "robot_root_to_floor_m": response.get("robot_root_to_floor_m"),
            "planning_start_z": response.get("planning_start_z"),
            "planning_end_z": response.get("planning_end_z"),
            "stair_vertical_radius_m": response.get("stair_vertical_radius_m"),
            "stair_constraint_mode": response.get("stair_constraint_mode"),
            "stair_progress_tolerance": response.get("stair_progress_tolerance"),
            "stair_progress_cost_weight": response.get(
                "stair_progress_cost_weight"
            ),
        }
        return NavPlan(goal=goal, waypoints=waypoints_xy, metadata=metadata)

    def _sim_to_pct(self, xyz: Sequence[float]) -> tuple[float, float, float]:
        return sim_to_pct_xyz(
            xyz,
            coord_mode=self.config.coord_mode,
            pct_offset_x=self.config.pct_offset_x,
            pct_offset_y=self.config.pct_offset_y,
            pct_offset_z=self.config.pct_offset_z,
            pct_scale_x=self.config.pct_scale_x,
            pct_scale_y=self.config.pct_scale_y,
            pct_scale_z=self.config.pct_scale_z,
            pct_rotation_x_rad=self.config.pct_rotation_x_rad,
            pct_rotation_y_rad=self.config.pct_rotation_y_rad,
            pct_rotation_z_rad=self.config.pct_rotation_z_rad,
        )

    def _pct_to_sim(self, xyz: Sequence[float]) -> tuple[float, float, float]:
        return pct_to_sim_xyz(
            xyz,
            coord_mode=self.config.coord_mode,
            pct_offset_x=self.config.pct_offset_x,
            pct_offset_y=self.config.pct_offset_y,
            pct_offset_z=self.config.pct_offset_z,
            pct_scale_x=self.config.pct_scale_x,
            pct_scale_y=self.config.pct_scale_y,
            pct_scale_z=self.config.pct_scale_z,
            pct_rotation_x_rad=self.config.pct_rotation_x_rad,
            pct_rotation_y_rad=self.config.pct_rotation_y_rad,
            pct_rotation_z_rad=self.config.pct_rotation_z_rad,
        )


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


__all__ = [
    "PCTNavPlanner",
    "PCTPlannerClient",
    "PCTPlannerConfig",
    "pct_to_sim_xyz",
    "sim_to_pct_xyz",
]
