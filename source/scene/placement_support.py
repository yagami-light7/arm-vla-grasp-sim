"""用 collision PLY 的垂直交点检查固定放置目标是否有几何支撑。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class VerticalIntersection:
    """一次从目标 XY 穿过三角面的垂直射线交点。"""

    z: float
    face_index: int


@dataclass(frozen=True)
class PlacementSupportResult:
    """固定物体中心位姿与其下方碰撞面的静态关系。"""

    collision_ply: Path
    position_xyz: tuple[float, float, float]
    intersections: tuple[VerticalIntersection, ...]
    support: VerticalIntersection | None
    center_to_support_m: float | None
    minimum_clearance_m: float
    maximum_clearance_m: float

    @property
    def geometry_verified(self) -> bool:
        """只有支撑面位于目标下方且中心净空落入阈值时才通过。"""

        return (
            self.support is not None
            and self.center_to_support_m is not None
            and self.minimum_clearance_m
            <= self.center_to_support_m
            <= self.maximum_clearance_m
        )

    def to_dict(self, *, include_sha256: bool = False) -> dict[str, Any]:
        """转换为可持久化的检查报告。"""

        report: dict[str, Any] = {
            "collision_ply": str(self.collision_ply),
            "position_xyz": list(self.position_xyz),
            "vertical_intersections": [
                {"z": item.z, "face_index": item.face_index}
                for item in self.intersections
            ],
            "support_surface": (
                None
                if self.support is None
                else {"z": self.support.z, "face_index": self.support.face_index}
            ),
            "center_to_support_m": self.center_to_support_m,
            "minimum_clearance_m": self.minimum_clearance_m,
            "maximum_clearance_m": self.maximum_clearance_m,
            "geometry_verified": self.geometry_verified,
        }
        if include_sha256:
            report["collision_ply_sha256"] = _sha256(self.collision_ply)
        return report


@dataclass(frozen=True)
class NearestGroundSupportResult:
    """按楼层高度提示选出的 collision PLY 支撑面。"""

    collision_ply: Path
    query_pct_xyz: tuple[float, float, float]
    intersections: tuple[VerticalIntersection, ...]
    support: VerticalIntersection
    hint_error_m: float
    maximum_hint_error_m: float | None

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 episode 元数据的高度来源报告。"""

        return {
            "collision_ply": str(self.collision_ply),
            "query_pct_xyz": list(self.query_pct_xyz),
            "support_surface": {
                "z": self.support.z,
                "face_index": self.support.face_index,
            },
            "hint_error_m": self.hint_error_m,
            "maximum_hint_error_m": self.maximum_hint_error_m,
            "vertical_intersection_count": len(self.intersections),
            "selection": "nearest_z_hint_then_z_then_face_index",
        }


@dataclass(frozen=True)
class _BinaryTrianglePlyLayout:
    """当前碰撞资产使用的 binary little-endian triangle PLY 布局。"""

    vertex_count: int
    face_count: int
    vertex_offset: int
    face_offset: int


def inspect_placement_support(
    collision_ply: str | Path,
    position_xyz: Sequence[float],
    *,
    minimum_clearance_m: float = 0.01,
    maximum_clearance_m: float = 0.20,
) -> PlacementSupportResult:
    """寻找目标中心正下方最高的碰撞三角面。"""

    path = Path(collision_ply).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"collision PLY 不存在: {path}")
    if len(position_xyz) < 3:
        raise ValueError("position_xyz 至少需要三个数值")
    if minimum_clearance_m < 0.0 or maximum_clearance_m <= minimum_clearance_m:
        raise ValueError("放置支撑净空阈值无效")

    position = tuple(float(value) for value in position_xyz[:3])
    vertices, face_indices = load_binary_triangle_ply(path)

    intersections = _vertical_intersections(
        vertices,
        face_indices,
        x=position[0],
        y=position[1],
    )
    support_candidates = [
        item
        for item in intersections
        if item.z <= position[2] - float(minimum_clearance_m)
    ]
    support = max(support_candidates, key=lambda item: item.z, default=None)
    clearance = None if support is None else float(position[2] - support.z)
    return PlacementSupportResult(
        collision_ply=path,
        position_xyz=position,
        intersections=intersections,
        support=support,
        center_to_support_m=clearance,
        minimum_clearance_m=float(minimum_clearance_m),
        maximum_clearance_m=float(maximum_clearance_m),
    )


def inspect_nearest_ground_support(
    collision_ply: str | Path,
    query_pct_xyz: Sequence[float],
    *,
    maximum_hint_error_m: float | None = None,
) -> NearestGroundSupportResult:
    """在固定 PCT XY 上选择最接近楼层 z 提示的真实支撑面。"""

    path = Path(collision_ply).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"collision PLY 不存在: {path}")
    if len(query_pct_xyz) < 3:
        raise ValueError("query_pct_xyz 至少需要三个数值")
    query = tuple(float(value) for value in query_pct_xyz[:3])
    if not all(np.isfinite(value) for value in query):
        raise ValueError("query_pct_xyz 不能包含 NaN 或 Inf")
    maximum_error = None
    if maximum_hint_error_m is not None:
        maximum_error = float(maximum_hint_error_m)
        if not np.isfinite(maximum_error) or maximum_error <= 0.0:
            raise ValueError("maximum_hint_error_m 必须是有限正数")

    vertices, face_indices = load_binary_triangle_ply(path)
    intersections = _vertical_intersections(
        vertices,
        face_indices,
        x=query[0],
        y=query[1],
    )
    if not intersections:
        raise ValueError(
            "collision PLY 在 "
            f"XY=({query[0]:.6f},{query[1]:.6f}) 没有支撑面"
        )
    support = min(
        intersections,
        key=lambda item: (
            abs(float(item.z) - query[2]),
            float(item.z),
            int(item.face_index),
        ),
    )
    hint_error = abs(float(support.z) - query[2])
    if maximum_error is not None and hint_error > maximum_error:
        raise ValueError(
            "collision PLY 最近支撑面与楼层提示相差 "
            f"{hint_error:.3f} m，超过 {maximum_error:.3f} m"
        )
    return NearestGroundSupportResult(
        collision_ply=path,
        query_pct_xyz=query,
        intersections=intersections,
        support=support,
        hint_error_m=hint_error,
        maximum_hint_error_m=maximum_error,
    )


def load_binary_triangle_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """以内存映射读取标准 XYZ + triangle binary PLY。"""

    resolved = Path(path).expanduser().resolve()
    layout = _read_binary_triangle_ply_layout(resolved)
    vertices = np.memmap(
        resolved,
        dtype="<f4",
        mode="r",
        offset=layout.vertex_offset,
        shape=(layout.vertex_count, 3),
    )
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    faces = np.memmap(
        resolved,
        dtype=face_dtype,
        mode="r",
        offset=layout.face_offset,
        shape=(layout.face_count,),
    )
    counts = np.asarray(faces["count"])
    if not bool(np.all(counts == 3)):
        raise ValueError("collision PLY 含有非三角面，当前检查器不会猜测可变 face 布局")
    return vertices, np.asarray(faces["indices"])


def _read_binary_triangle_ply_layout(path: Path) -> _BinaryTrianglePlyLayout:
    """读取并严格验证本工具支持的 PLY header。"""

    vertex_count: int | None = None
    face_count: int | None = None
    active_element: str | None = None
    vertex_properties: list[str] = []
    face_properties: list[str] = []
    with path.open("rb") as stream:
        first = stream.readline()
        if first != b"ply\n":
            raise ValueError(f"不是 PLY 文件: {path}")
        format_line = stream.readline()
        if format_line != b"format binary_little_endian 1.0\n":
            raise ValueError("只支持 format binary_little_endian 1.0")
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError("PLY header 缺少 end_header")
            line = raw_line.decode("ascii").strip()
            if line == "end_header":
                vertex_offset = stream.tell()
                break
            if line.startswith("element "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"无法解析 PLY element: {line}")
                active_element = parts[1]
                if active_element == "vertex":
                    vertex_count = int(parts[2])
                elif active_element == "face":
                    face_count = int(parts[2])
                continue
            if line.startswith("property "):
                if active_element == "vertex":
                    vertex_properties.append(line)
                elif active_element == "face":
                    face_properties.append(line)

    expected_vertex = ["property float x", "property float y", "property float z"]
    expected_face = ["property list uchar int vertex_indices"]
    if vertex_count is None or face_count is None:
        raise ValueError("PLY header 缺少 vertex 或 face 计数")
    if vertex_properties != expected_vertex:
        raise ValueError(f"不支持的 vertex properties: {vertex_properties}")
    if face_properties != expected_face:
        raise ValueError(f"不支持的 face properties: {face_properties}")
    return _BinaryTrianglePlyLayout(
        vertex_count=vertex_count,
        face_count=face_count,
        vertex_offset=vertex_offset,
        face_offset=vertex_offset + vertex_count * 3 * np.dtype("<f4").itemsize,
    )


def _vertical_intersections(
    vertices: np.ndarray,
    face_indices: np.ndarray,
    *,
    x: float,
    y: float,
) -> tuple[VerticalIntersection, ...]:
    """计算固定 XY 与所有非退化三角面投影的交点。"""

    triangles = np.asarray(vertices[face_indices], dtype=np.float64)
    xy = triangles[:, :, :2]
    lower = xy.min(axis=1)
    upper = xy.max(axis=1)
    candidate_indices = np.flatnonzero(
        (lower[:, 0] <= x)
        & (x <= upper[:, 0])
        & (lower[:, 1] <= y)
        & (y <= upper[:, 1])
    )
    query = np.array([float(x), float(y)], dtype=np.float64)
    output: list[VerticalIntersection] = []
    for face_index in candidate_indices.tolist():
        triangle = triangles[face_index]
        first, second, third = triangle[:, :2]
        denominator = (
            (second[1] - third[1]) * (first[0] - third[0])
            + (third[0] - second[0]) * (first[1] - third[1])
        )
        if abs(float(denominator)) <= 1.0e-12:
            continue
        weight_first = (
            (second[1] - third[1]) * (query[0] - third[0])
            + (third[0] - second[0]) * (query[1] - third[1])
        ) / denominator
        weight_second = (
            (third[1] - first[1]) * (query[0] - third[0])
            + (first[0] - third[0]) * (query[1] - third[1])
        ) / denominator
        weight_third = 1.0 - weight_first - weight_second
        if min(weight_first, weight_second, weight_third) < -1.0e-7:
            continue
        z = float(
            weight_first * triangle[0, 2]
            + weight_second * triangle[1, 2]
            + weight_third * triangle[2, 2]
        )
        output.append(VerticalIntersection(z=z, face_index=int(face_index)))
    return tuple(sorted(output, key=lambda item: (item.z, item.face_index)))


def _sha256(path: Path) -> str:
    """流式计算资产哈希，避免把大 PLY 一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
