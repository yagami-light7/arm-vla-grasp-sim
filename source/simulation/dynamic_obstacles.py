"""解析并推进 Isaac Lab 导航任务中的确定性动态障碍。

本模块只描述任务合同和世界系运动学，不导入 Isaac Sim、Isaac Lab 或 ROS 2。
运行时负责把这里生成的 WXYZ 位姿写入可见且可碰撞的 kinematic rigid body。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


_OBSTACLE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SUPPORTED_FLOORS = frozenset({"F1", "F2"})
_MAX_OBSTACLE_COUNT = 8
_MAX_WAYPOINT_COUNT = 128
_MAX_FLAT_Z_SPAN_M = 0.03


def _finite_float(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """把配置值校验为有限实数，并应用可选闭区间门限。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} 必须是实数。")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} 必须是有限值。")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field_name} 不能小于 {minimum}。")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field_name} 不能大于 {maximum}。")
    return normalized


def _finite_vector(
    value: object,
    *,
    field_name: str,
    length: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    """校验固定长度的有限实数向量。"""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != length
    ):
        raise TypeError(f"{field_name} 必须是长度为 {length} 的数组。")
    return tuple(
        _finite_float(
            item,
            field_name=f"{field_name}[{index}]",
            minimum=minimum,
            maximum=maximum,
        )
        for index, item in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class DynamicObstacleAabb:
    """世界系轴对齐包围盒，用于楼梯冻结走廊的静态排除。"""

    identifier: str
    minimum_world_xyz: tuple[float, float, float]
    maximum_world_xyz: tuple[float, float, float]

    @classmethod
    def from_mapping(
        cls,
        raw: object,
        *,
        field_name: str,
    ) -> "DynamicObstacleAabb":
        if not isinstance(raw, Mapping):
            raise TypeError(f"{field_name} 必须是对象。")
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{field_name}.id 必须是非空字符串。")
        minimum = _finite_vector(
            raw.get("min_xyz"),
            field_name=f"{field_name}.min_xyz",
            length=3,
        )
        maximum = _finite_vector(
            raw.get("max_xyz"),
            field_name=f"{field_name}.max_xyz",
            length=3,
        )
        if any(left >= right for left, right in zip(minimum, maximum, strict=True)):
            raise ValueError(f"{field_name} 每个 min_xyz 都必须严格小于 max_xyz。")
        return cls(
            identifier=identifier.strip(),
            minimum_world_xyz=minimum,
            maximum_world_xyz=maximum,
        )

    def inflated(self, margin_m: float) -> "DynamicObstacleAabb":
        """在三个世界轴上等量膨胀包围盒。"""

        margin = _finite_float(
            margin_m,
            field_name="margin_m",
            minimum=0.0,
        )
        return DynamicObstacleAabb(
            identifier=self.identifier,
            minimum_world_xyz=tuple(value - margin for value in self.minimum_world_xyz),
            maximum_world_xyz=tuple(value + margin for value in self.maximum_world_xyz),
        )

    def intersects(self, other: "DynamicObstacleAabb") -> bool:
        """判断两个闭包围盒是否相交；边界接触也按不安全处理。"""

        return all(
            self.minimum_world_xyz[axis] <= other.maximum_world_xyz[axis]
            and self.maximum_world_xyz[axis] >= other.minimum_world_xyz[axis]
            for axis in range(3)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "min_xyz": list(self.minimum_world_xyz),
            "max_xyz": list(self.maximum_world_xyz),
        }


@dataclass(frozen=True, slots=True)
class DynamicObstacleState:
    """某一仿真时刻的动态障碍世界系目标状态。"""

    obstacle_id: str
    scene_asset_name: str
    elapsed_time_s: float
    position_world_xyz: tuple[float, float, float]
    orientation_world_wxyz: tuple[float, float, float, float]
    path_distance_m: float
    path_direction: int
    waiting_for_start: bool

    def root_pose_wxyz(self) -> tuple[float, ...]:
        return (*self.position_world_xyz, *self.orientation_world_wxyz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.obstacle_id,
            "scene_asset_name": self.scene_asset_name,
            "elapsed_time_s": self.elapsed_time_s,
            "position_world_xyz": list(self.position_world_xyz),
            "orientation_world_wxyz": list(self.orientation_world_wxyz),
            "path_distance_m": self.path_distance_m,
            "path_direction": self.path_direction,
            "waiting_for_start": self.waiting_for_start,
        }


@dataclass(frozen=True, slots=True)
class DynamicObstacleSpec:
    """一个可见、可碰撞的平地 kinematic cuboid 任务合同。"""

    obstacle_id: str
    scene_asset_name: str
    floor_id: str
    surface_class: str
    size_xyz_m: tuple[float, float, float]
    waypoints_world_xyz: tuple[tuple[float, float, float], ...]
    speed_mps: float
    start_delay_s: float
    yaw_rad: float
    color_rgb: tuple[float, float, float]
    mass_kg: float
    motion: str = "ping_pong"
    shape: str = "cuboid"
    collision_enabled: bool = True
    visible: bool = True

    @classmethod
    def from_mapping(
        cls,
        raw: object,
        *,
        index: int,
    ) -> "DynamicObstacleSpec":
        field_name = f"dynamic_obstacles[{index}]"
        if not isinstance(raw, Mapping):
            raise TypeError(f"{field_name} 必须是对象。")

        obstacle_id = raw.get("id")
        if (
            not isinstance(obstacle_id, str)
            or _OBSTACLE_ID_PATTERN.fullmatch(obstacle_id) is None
        ):
            raise ValueError(
                f"{field_name}.id 必须匹配 {_OBSTACLE_ID_PATTERN.pattern!r}。"
            )
        shape = raw.get("shape", "cuboid")
        if shape != "cuboid":
            raise ValueError(f"{field_name}.shape 第一阶段只支持 cuboid。")
        motion = raw.get("motion", "ping_pong")
        if motion not in {"ping_pong", "one_shot"}:
            raise ValueError(
                f"{field_name}.motion 只支持 ping_pong 或 one_shot。"
            )

        floor_id = raw.get("floor_id")
        if floor_id not in _SUPPORTED_FLOORS:
            raise ValueError(f"{field_name}.floor_id 只允许 F1 或 F2。")
        surface_class = raw.get("surface_class")
        if surface_class != "flat":
            raise ValueError(f"{field_name}.surface_class 必须显式为 flat。")

        collision_enabled = raw.get("collision_enabled", True)
        visible = raw.get("visible", True)
        if collision_enabled is not True or visible is not True:
            raise ValueError(
                f"{field_name} 必须同时启用 collision_enabled 与 visible，"
                "避免 PhysX 和 RTX 观测不一致。"
            )

        size = _finite_vector(
            raw.get("size_xyz_m"),
            field_name=f"{field_name}.size_xyz_m",
            length=3,
            minimum=0.10,
            maximum=2.0,
        )
        if size[2] < 0.20:
            raise ValueError(f"{field_name}.size_xyz_m[2] 不能小于 0.20 m。")

        raw_waypoints = raw.get("waypoints_world_xyz")
        if (
            not isinstance(raw_waypoints, Sequence)
            or isinstance(raw_waypoints, (str, bytes, bytearray))
            or not 2 <= len(raw_waypoints) <= _MAX_WAYPOINT_COUNT
        ):
            raise ValueError(
                f"{field_name}.waypoints_world_xyz 必须包含 2..{_MAX_WAYPOINT_COUNT} 个点。"
            )
        waypoints = tuple(
            _finite_vector(
                waypoint,
                field_name=f"{field_name}.waypoints_world_xyz[{waypoint_index}]",
                length=3,
            )
            for waypoint_index, waypoint in enumerate(raw_waypoints)
        )
        segment_lengths = tuple(
            math.dist(left, right)
            for left, right in zip(waypoints, waypoints[1:])
        )
        if any(length <= 1.0e-6 for length in segment_lengths):
            raise ValueError(f"{field_name} 不允许连续重复 waypoint。")
        waypoint_z_span = max(point[2] for point in waypoints) - min(
            point[2] for point in waypoints
        )
        if waypoint_z_span > _MAX_FLAT_Z_SPAN_M:
            raise ValueError(
                f"{field_name} 平地轨迹的中心 z 跨度不能超过 {_MAX_FLAT_Z_SPAN_M} m。"
            )

        return cls(
            obstacle_id=obstacle_id,
            scene_asset_name=f"dynamic_obstacle_{index:02d}",
            floor_id=str(floor_id),
            surface_class=str(surface_class),
            size_xyz_m=(size[0], size[1], size[2]),
            waypoints_world_xyz=tuple(
                (point[0], point[1], point[2]) for point in waypoints
            ),
            speed_mps=_finite_float(
                raw.get("speed_mps"),
                field_name=f"{field_name}.speed_mps",
                minimum=0.01,
                maximum=1.0,
            ),
            start_delay_s=_finite_float(
                raw.get("start_delay_s", 0.0),
                field_name=f"{field_name}.start_delay_s",
                minimum=0.0,
                maximum=600.0,
            ),
            yaw_rad=_finite_float(
                raw.get("yaw_rad", 0.0),
                field_name=f"{field_name}.yaw_rad",
            ),
            color_rgb=_finite_vector(
                raw.get("color_rgb", (0.9, 0.2, 0.1)),
                field_name=f"{field_name}.color_rgb",
                length=3,
                minimum=0.0,
                maximum=1.0,
            ),
            mass_kg=_finite_float(
                raw.get("mass_kg", 20.0),
                field_name=f"{field_name}.mass_kg",
                minimum=0.1,
                maximum=500.0,
            ),
            motion=str(motion),
            shape=str(shape),
            collision_enabled=True,
            visible=True,
        )

    @property
    def prim_path(self) -> str:
        # InteractiveScene 在解析带 env regex 的刚体 spawn 路径时，要求
        # regex 后的直接父 Prim 已存在。env namespace 由场景先创建；额外的
        # ``DynamicObstacles`` 中间 Xform 却不会自动创建，因此把障碍放在
        # env 根下；该 spawn 路径兼容多环境克隆，也不依赖预先改写当前
        # USD stage。当前位姿推进与证据链仍由 runtime 限定为单环境。
        return f"{{ENV_REGEX_NS}}/DynamicObstacle_{self.obstacle_id}"

    @property
    def total_path_length_m(self) -> float:
        return sum(
            math.dist(left, right)
            for left, right in zip(
                self.waypoints_world_xyz,
                self.waypoints_world_xyz[1:],
            )
        )

    @property
    def orientation_world_wxyz(self) -> tuple[float, float, float, float]:
        half_yaw = 0.5 * self.yaw_rad
        return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))

    @property
    def swept_aabb_world(self) -> DynamicObstacleAabb:
        half_x, half_y, half_z = (
            0.5 * value for value in self.size_xyz_m
        )
        abs_cos_yaw = abs(math.cos(self.yaw_rad))
        abs_sin_yaw = abs(math.sin(self.yaw_rad))
        # cuboid 尺寸定义在本体系；楼梯排除区是世界系 AABB，必须先投影
        # 固定 yaw 后的 XY 半径，否则非方形推车会在斜角下低估 swept volume。
        half_size_world = (
            abs_cos_yaw * half_x + abs_sin_yaw * half_y,
            abs_sin_yaw * half_x + abs_cos_yaw * half_y,
            half_z,
        )
        return DynamicObstacleAabb(
            identifier=f"{self.obstacle_id}_swept_volume",
            minimum_world_xyz=tuple(
                min(point[axis] for point in self.waypoints_world_xyz)
                - half_size_world[axis]
                for axis in range(3)
            ),
            maximum_world_xyz=tuple(
                max(point[axis] for point in self.waypoints_world_xyz)
                + half_size_world[axis]
                for axis in range(3)
            ),
        )

    def state_at(self, elapsed_time_s: float) -> DynamicObstacleState:
        """按 physics 仿真时间计算确定性的折线往返或单程位姿。"""

        elapsed = _finite_float(
            elapsed_time_s,
            field_name="elapsed_time_s",
            minimum=0.0,
        )
        waiting = elapsed <= self.start_delay_s
        total_length = self.total_path_length_m
        if waiting:
            path_distance = 0.0
            path_direction = 0
        elif self.motion == "one_shot":
            travelled = (elapsed - self.start_delay_s) * self.speed_mps
            path_distance = min(travelled, total_length)
            # 单程障碍到达末点后保持可见和可碰撞，但不再伪造运动方向。
            path_direction = 1 if travelled < total_length else 0
        else:
            travelled = (elapsed - self.start_delay_s) * self.speed_mps
            wrapped = math.fmod(travelled, 2.0 * total_length)
            if wrapped <= total_length:
                path_distance = wrapped
                path_direction = 1
            else:
                path_distance = 2.0 * total_length - wrapped
                path_direction = -1

        position = self._position_at_path_distance(path_distance)
        return DynamicObstacleState(
            obstacle_id=self.obstacle_id,
            scene_asset_name=self.scene_asset_name,
            elapsed_time_s=elapsed,
            position_world_xyz=position,
            orientation_world_wxyz=self.orientation_world_wxyz,
            path_distance_m=path_distance,
            path_direction=path_direction,
            waiting_for_start=waiting,
        )

    def _position_at_path_distance(
        self,
        path_distance_m: float,
    ) -> tuple[float, float, float]:
        remaining = min(max(float(path_distance_m), 0.0), self.total_path_length_m)
        for left, right in zip(
            self.waypoints_world_xyz,
            self.waypoints_world_xyz[1:],
        ):
            segment_length = math.dist(left, right)
            if remaining <= segment_length:
                alpha = remaining / segment_length
                return tuple(
                    left[axis] + alpha * (right[axis] - left[axis])
                    for axis in range(3)
                )
            remaining -= segment_length
        return self.waypoints_world_xyz[-1]

    def topology_fingerprint(self) -> dict[str, Any]:
        """返回必须重建 Isaac stage 才能改变的 spawn 属性。"""

        return {
            "id": self.obstacle_id,
            "scene_asset_name": self.scene_asset_name,
            "prim_path": self.prim_path,
            "shape": self.shape,
            "size_xyz_m": list(self.size_xyz_m),
            "color_rgb": list(self.color_rgb),
            "mass_kg": self.mass_kg,
            "collision_enabled": self.collision_enabled,
            "visible": self.visible,
            "kinematic_enabled": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.topology_fingerprint(),
            "floor_id": self.floor_id,
            "surface_class": self.surface_class,
            "waypoints_world_xyz": [list(point) for point in self.waypoints_world_xyz],
            "speed_mps": self.speed_mps,
            "start_delay_s": self.start_delay_s,
            "yaw_rad": self.yaw_rad,
            "motion": self.motion,
            "total_path_length_m": self.total_path_length_m,
            "swept_aabb_world": self.swept_aabb_world.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DynamicObstaclePlan:
    """单个 episode 的动态障碍集合及楼梯排除证据。"""

    obstacles: tuple[DynamicObstacleSpec, ...] = ()
    stair_exclusion_aabbs_world: tuple[DynamicObstacleAabb, ...] = ()
    minimum_stair_clearance_m: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.obstacles)

    def topology_fingerprint(self) -> list[dict[str, Any]]:
        return [obstacle.topology_fingerprint() for obstacle in self.obstacles]

    def state_at(self, elapsed_time_s: float) -> tuple[DynamicObstacleState, ...]:
        return tuple(
            obstacle.state_at(elapsed_time_s) for obstacle in self.obstacles
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": "task_json.dynamic_obstacles" if self.enabled else "configuration_absent",
            "obstacle_count": len(self.obstacles),
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "minimum_stair_clearance_m": self.minimum_stair_clearance_m,
            "stair_exclusion_aabbs_world": [
                bounds.to_dict() for bounds in self.stair_exclusion_aabbs_world
            ],
            "stair_corridor_overlap_verified_false": self.enabled,
            "physics_time_source": "episode_physics_step_index_x_physics_dt",
        }


def resolve_dynamic_obstacle_plan(raw_task: object) -> DynamicObstaclePlan:
    """从 task JSON 解析动态障碍；字段缺失时保持默认关闭。"""

    if not isinstance(raw_task, Mapping):
        raise TypeError("raw_task 必须是对象。")
    if "dynamic_obstacles" not in raw_task:
        return DynamicObstaclePlan()

    raw_obstacles = raw_task.get("dynamic_obstacles")
    if (
        not isinstance(raw_obstacles, Sequence)
        or isinstance(raw_obstacles, (str, bytes, bytearray))
    ):
        raise TypeError("dynamic_obstacles 必须是数组。")
    if len(raw_obstacles) == 0:
        return DynamicObstaclePlan()
    if len(raw_obstacles) > _MAX_OBSTACLE_COUNT:
        raise ValueError(f"dynamic_obstacles 不能超过 {_MAX_OBSTACLE_COUNT} 个。")

    obstacles = tuple(
        DynamicObstacleSpec.from_mapping(raw, index=index)
        for index, raw in enumerate(raw_obstacles)
    )
    ids = [obstacle.obstacle_id for obstacle in obstacles]
    if len(ids) != len(set(ids)):
        raise ValueError("dynamic_obstacles.id 必须唯一。")

    raw_safety = raw_task.get("dynamic_obstacle_safety")
    if not isinstance(raw_safety, Mapping):
        raise ValueError(
            "启用 dynamic_obstacles 时必须配置 dynamic_obstacle_safety。"
        )
    clearance = _finite_float(
        raw_safety.get("minimum_stair_clearance_m"),
        field_name="dynamic_obstacle_safety.minimum_stair_clearance_m",
        minimum=0.0,
        maximum=10.0,
    )
    raw_exclusions = raw_safety.get("stair_exclusion_aabbs_world")
    if (
        not isinstance(raw_exclusions, Sequence)
        or isinstance(raw_exclusions, (str, bytes, bytearray))
        or len(raw_exclusions) == 0
    ):
        raise ValueError(
            "dynamic_obstacle_safety.stair_exclusion_aabbs_world 必须是非空数组。"
        )
    exclusions = tuple(
        DynamicObstacleAabb.from_mapping(
            raw,
            field_name=(
                "dynamic_obstacle_safety.stair_exclusion_aabbs_world"
                f"[{index}]"
            ),
        )
        for index, raw in enumerate(raw_exclusions)
    )

    for obstacle in obstacles:
        swept = obstacle.swept_aabb_world
        for exclusion in exclusions:
            if swept.intersects(exclusion.inflated(clearance)):
                raise ValueError(
                    f"动态障碍 {obstacle.obstacle_id!r} 的 swept AABB 侵入楼梯"
                    f"排除区 {exclusion.identifier!r}（含 {clearance:.3f} m 安全距）。"
                )

    return DynamicObstaclePlan(
        obstacles=obstacles,
        stair_exclusion_aabbs_world=exclusions,
        minimum_stair_clearance_m=clearance,
    )
