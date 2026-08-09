"""把 ROS 2 ``cmd_vel`` 安全写入 Go2-X5 locomotion policy 命令入口。

本模块不依赖 ROS 2、Torch 或 Isaac Sim。调用方应把 ``node.get_clock().now()``
得到的仿真时间显式传入，并把
``Go2LocomotionAdapter.apply_base_command`` 作为 ``command_sink``。这样既能在
纯 Python 中测试安全门，也不会绕过既有的 policy observation 写入路径。
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any


_ZERO_COMMAND = (0.0, 0.0, 0.0)


class PolicyCommandOwnershipError(RuntimeError):
    """表示调用者没有当前 policy 命令入口的独占所有权。"""


@dataclass(frozen=True, slots=True)
class BodyVelocityCommand:
    """机体坐标系平面速度命令。"""

    vx: float
    vy: float
    wz: float

    def __post_init__(self) -> None:
        values = (self.vx, self.vy, self.wz)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise TypeError("vx、vy、wz 必须是实数。")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("vx、vy、wz 必须是有限值。")
        object.__setattr__(self, "vx", float(self.vx))
        object.__setattr__(self, "vy", float(self.vy))
        object.__setattr__(self, "wz", float(self.wz))

    @classmethod
    def zero(cls) -> BodyVelocityCommand:
        """返回零速度命令。"""

        return cls(*_ZERO_COMMAND)

    def as_tuple(self) -> tuple[float, float, float]:
        """返回 ``apply_base_command`` 所需的三元组。"""

        return self.vx, self.vy, self.wz


@dataclass(frozen=True, slots=True)
class NavigationSafetyPermit:
    """由 ``NavigationStatus`` 严格转换得到的执行层安全许可。

    ``identity_valid`` 由接收桥按本进程实际发布的目标和实际接收的 Path
    交叉校验。这里保留 supervisor 的原始 allow/force 位，最终是否允许运动
    只由唯一 policy writer 在控制 tick 内判定。
    """

    header_stamp_ns: int
    received_at: float
    status_sequence: int
    state_revision: int
    goal_id: int
    active_path_stamp_ns: int
    state: int
    allow_tracking_command: bool
    force_zero_velocity: bool
    identity_valid: bool
    reason: str

    def __post_init__(self) -> None:
        uint64_fields = (
            "header_stamp_ns",
            "status_sequence",
            "state_revision",
            "goal_id",
            "active_path_stamp_ns",
        )
        for field_name in uint64_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} 必须是整数。")
            if not 0 <= value <= (1 << 64) - 1:
                raise ValueError(f"{field_name} 必须位于 uint64 范围内。")
        if self.header_stamp_ns <= 0:
            raise ValueError("header_stamp_ns 必须非零。")
        if self.status_sequence <= 0:
            raise ValueError("status_sequence 必须非零。")
        if isinstance(self.state, bool) or not isinstance(self.state, int):
            raise TypeError("state 必须是整数。")
        if not 0 <= self.state <= 255:
            raise ValueError("state 必须位于 uint8 范围内。")
        received_at = _time_to_seconds(self.received_at, "received_at")
        object.__setattr__(self, "received_at", received_at)
        for field_name in (
            "allow_tracking_command",
            "force_zero_velocity",
            "identity_valid",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} 必须是布尔值。")
        if self.allow_tracking_command == self.force_zero_velocity:
            raise ValueError("allow_tracking_command 与 force_zero_velocity 必须互反。")
        if self.allow_tracking_command and self.state != 3:
            raise ValueError("只有 TRACKING 状态可以允许跟踪命令。")
        if self.allow_tracking_command and (
            self.goal_id <= 0 or self.active_path_stamp_ns <= 0
        ):
            raise ValueError("允许跟踪时 goal_id 与 active_path_stamp 必须非零。")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason 必须是非空字符串。")
        object.__setattr__(self, "reason", self.reason.strip())

    @property
    def source_timestamp(self) -> float:
        """返回不用于 identity 的 ROS Header 秒值。"""

        return self.header_stamp_ns * 1.0e-9

    @property
    def command_identity(self) -> tuple[int, int, int]:
        """返回速度命令必须绑定的目标、路径和状态代际。"""

        return self.goal_id, self.active_path_stamp_ns, self.state_revision

    @property
    def signature(self) -> tuple[object, ...]:
        """返回检测同序号冲突所需的完整不可变签名。"""

        return (
            self.header_stamp_ns,
            self.status_sequence,
            self.state_revision,
            self.goal_id,
            self.active_path_stamp_ns,
            self.state,
            self.allow_tracking_command,
            self.force_zero_velocity,
            self.reason,
        )


@dataclass(frozen=True, slots=True)
class PolicyCommandInput:
    """把速度与同一接收周期观察到的 supervisor 许可送入唯一 writer。"""

    command: object | None = None
    navigation_permit: NavigationSafetyPermit | None = None
    navigation_status_error: str | None = None

    def __post_init__(self) -> None:
        if self.navigation_status_error is not None:
            if (
                not isinstance(self.navigation_status_error, str)
                or not self.navigation_status_error.strip()
            ):
                raise ValueError("navigation_status_error 必须是非空字符串或 None。")
            object.__setattr__(
                self,
                "navigation_status_error",
                self.navigation_status_error.strip(),
            )
        if self.navigation_permit is not None and not isinstance(
            self.navigation_permit,
            NavigationSafetyPermit,
        ):
            raise TypeError("navigation_permit 必须是 NavigationSafetyPermit 或 None。")
        if self.navigation_permit is not None and self.navigation_status_error is not None:
            raise ValueError("有效许可与 navigation_status_error 不能同时存在。")
        if (
            self.command is None
            and self.navigation_permit is None
            and self.navigation_status_error is None
        ):
            raise ValueError("PolicyCommandInput 不能是空包络。")


@dataclass(frozen=True, slots=True)
class CmdVelToPolicyConfig:
    """定义 policy 能力边界、变化率和输入新鲜度要求。"""

    max_vx: float = 0.30
    max_vy: float = 0.15
    max_wz: float = 0.45
    max_vx_rate: float = 0.50
    max_vy_rate: float = 0.40
    max_wz_rate: float = 1.00
    cmd_vel_timeout_s: float = 0.25
    odometry_timeout_s: float = 0.25
    point_cloud_timeout_s: float = 0.50
    navigation_status_timeout_s: float = 0.25
    control_lease_timeout_s: float = 0.50
    future_tolerance_s: float = 0.02
    clock_rewind_tolerance_s: float = 1.0e-6
    require_odometry: bool = True
    require_point_cloud: bool = True
    require_navigation_status: bool = True

    def __post_init__(self) -> None:
        positive_fields = (
            "max_vx",
            "max_vy",
            "max_wz",
            "max_vx_rate",
            "max_vy_rate",
            "max_wz_rate",
            "cmd_vel_timeout_s",
            "odometry_timeout_s",
            "point_cloud_timeout_s",
            "navigation_status_timeout_s",
            "control_lease_timeout_s",
        )
        nonnegative_fields = ("future_tolerance_s", "clock_rewind_tolerance_s")
        for field_name in (*positive_fields, *nonnegative_fields):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} 必须是实数。")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} 必须是有限值。")
            object.__setattr__(self, field_name, float(value))
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} 必须大于零。")
        for field_name in nonnegative_fields:
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} 不能为负数。")
        for field_name in (
            "require_odometry",
            "require_point_cloud",
            "require_navigation_status",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} 必须是布尔值。")


@dataclass(frozen=True, slots=True)
class PolicyCommandWriteReport:
    """记录一个控制 tick 最终写入 policy 的命令及安全判定。"""

    timestamp: float
    owner_id: str
    requested_command: BodyVelocityCommand | None
    limited_target: BodyVelocityCommand
    written_command: BodyVelocityCommand
    motion_allowed: bool
    stop_reasons: tuple[str, ...]
    clipped_axes: tuple[str, ...]
    rate_limited_axes: tuple[str, ...]


@dataclass(slots=True)
class _OwnershipLease:
    """进程内一个 policy 命令入口的独占租约。"""

    token: object
    owner_id: str
    last_renewed: float
    expires_at: float


class _PolicyCommandOwnershipRegistry:
    """用进程级锁保证同一 command sink 同时只有一个写入者。"""

    _lock = threading.RLock()
    _leases: dict[Hashable, _OwnershipLease] = {}

    @classmethod
    def claim(
        cls,
        resource: Hashable,
        token: object,
        owner_id: str,
        now: float,
        timeout_s: float,
        rewind_tolerance_s: float,
        sink: Callable[[float, float, float], None],
    ) -> bool:
        """获取或续租所有权；新接管者在返回前先写一次零速度。"""

        with cls._lock:
            current = cls._leases.get(resource)
            same_token = current is not None and current.token is token
            clock_rewound = (
                current is not None
                and now < current.last_renewed - rewind_tolerance_s
            )
            expired = current is not None and now > current.expires_at
            if current is not None and not same_token and not (expired or clock_rewound):
                raise PolicyCommandOwnershipError(
                    f"policy 命令入口已由 {current.owner_id!r} 持有。"
                )

            restart_same_owner = same_token and (expired or clock_rewound)
            is_new_owner = not same_token
            previous = current
            cls._leases[resource] = _OwnershipLease(
                token=token,
                owner_id=owner_id,
                last_renewed=now,
                expires_at=now + timeout_s,
            )
            if is_new_owner or restart_same_owner:
                try:
                    sink(*_ZERO_COMMAND)
                except Exception:
                    if previous is None:
                        cls._leases.pop(resource, None)
                    else:
                        cls._leases[resource] = previous
                    raise
            return is_new_owner or restart_same_owner

    @classmethod
    def verify(cls, resource: Hashable, token: object) -> _OwnershipLease:
        """校验调用实例仍持有入口，不把过期租约误当成写入权限丢失。"""

        with cls._lock:
            current = cls._leases.get(resource)
            if current is None or current.token is not token:
                owner = None if current is None else current.owner_id
                raise PolicyCommandOwnershipError(
                    f"当前实例不持有 policy 命令入口；当前 owner={owner!r}。"
                )
            return current

    @classmethod
    def renew(
        cls,
        resource: Hashable,
        token: object,
        now: float,
        timeout_s: float,
    ) -> None:
        """续租当前实例的控制心跳。"""

        with cls._lock:
            current = cls.verify(resource, token)
            current.last_renewed = now
            current.expires_at = now + timeout_s

    @classmethod
    def lease_status(
        cls,
        resource: Hashable,
        token: object,
        now: float,
        future_tolerance_s: float,
    ) -> str:
        """返回 ``fresh``、``expired`` 或 ``from_future``。"""

        with cls._lock:
            current = cls.verify(resource, token)
            if current.last_renewed > now + future_tolerance_s:
                return "from_future"
            if now > current.expires_at:
                return "expired"
            return "fresh"

    @classmethod
    def write(
        cls,
        resource: Hashable,
        token: object,
        command: BodyVelocityCommand,
        sink: Callable[[float, float, float], None],
    ) -> None:
        """在所有权校验与写入之间保持同一把锁，避免接管竞争。"""

        with cls._lock:
            cls.verify(resource, token)
            sink(*command.as_tuple())

    @classmethod
    def release(
        cls,
        resource: Hashable,
        token: object,
        sink: Callable[[float, float, float], None],
    ) -> None:
        """写入零速度后释放所有权。"""

        with cls._lock:
            cls.verify(resource, token)
            sink(*_ZERO_COMMAND)
            cls._leases.pop(resource, None)


class _IdentityResource:
    """用对象身份而不是可能复用的裸 ``id`` 表示默认 command sink。"""

    __slots__ = ("_kind", "_objects")

    def __init__(self, kind: str, *objects: object) -> None:
        self._kind = kind
        self._objects = objects

    def __hash__(self) -> int:
        return hash((self._kind, *(id(value) for value in self._objects)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _IdentityResource)
            and self._kind == other._kind
            and len(self._objects) == len(other._objects)
            and all(
                left is right
                for left, right in zip(self._objects, other._objects, strict=True)
            )
        )


def _time_to_seconds(value: Any, field_name: str) -> float:
    """把 ROS Time、builtin_interfaces/Time 或实数转换为秒。"""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} 不能是布尔值。")
    if isinstance(value, Real):
        seconds = float(value)
    elif hasattr(value, "nanoseconds"):
        seconds = float(value.nanoseconds) * 1.0e-9
    elif hasattr(value, "seconds_nanoseconds"):
        sec, nanosec = value.seconds_nanoseconds()
        seconds = float(sec) + float(nanosec) * 1.0e-9
    elif hasattr(value, "sec") and hasattr(value, "nanosec"):
        seconds = float(value.sec) + float(value.nanosec) * 1.0e-9
    else:
        raise TypeError(
            f"{field_name} 必须是秒数、rclpy.time.Time 或 builtin_interfaces/Time。"
        )
    if not math.isfinite(seconds):
        raise ValueError(f"{field_name} 必须是有限值。")
    if seconds < 0.0:
        raise ValueError(f"{field_name} 不能为负数。")
    return seconds


def _nested_mapping_value(
    value: Mapping[str, Any],
    group: str,
    axis: str,
) -> Any:
    nested = value.get(group)
    if not isinstance(nested, Mapping) or axis not in nested:
        raise TypeError(f"命令缺少 {group}.{axis}。")
    return nested[axis]


def body_velocity_from_input(value: Any) -> BodyVelocityCommand:
    """解析 ROS Twist、三元组、映射或带 ``base_velocity`` 的纯数据对象。"""

    if isinstance(value, BodyVelocityCommand):
        return value
    if hasattr(value, "twist"):
        value = value.twist
    if hasattr(value, "linear") and hasattr(value, "angular"):
        return BodyVelocityCommand(
            value.linear.x,
            value.linear.y,
            value.angular.z,
        )
    if hasattr(value, "base_velocity"):
        value = value.base_velocity
    if isinstance(value, Mapping):
        if all(axis in value for axis in ("vx", "vy", "wz")):
            return BodyVelocityCommand(value["vx"], value["vy"], value["wz"])
        return BodyVelocityCommand(
            _nested_mapping_value(value, "linear", "x"),
            _nested_mapping_value(value, "linear", "y"),
            _nested_mapping_value(value, "angular", "z"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 3:
            raise ValueError("速度序列必须恰好包含 vx、vy、wz 三个值。")
        return BodyVelocityCommand(value[0], value[1], value[2])
    raise TypeError(
        "命令必须是 ROS Twist、BodyVelocityCommand、三元组或速度映射。"
    )


def _resolve_sink(
    command_sink: Any,
) -> tuple[Callable[[float, float, float], None], Hashable]:
    """解析既有 adapter 的命令入口，并生成默认所有权资源键。"""

    apply_method = getattr(command_sink, "apply_base_command", None)
    if callable(apply_method):
        return apply_method, _IdentityResource("apply_base_command", command_sink)
    if not callable(command_sink):
        raise TypeError(
            "command_sink 必须可调用，或提供 apply_base_command(vx, vy, wz)。"
        )
    bound_instance = getattr(command_sink, "__self__", None)
    bound_function = getattr(command_sink, "__func__", None)
    if bound_instance is not None and bound_function is not None:
        return command_sink, _IdentityResource(
            "bound_command_sink",
            bound_instance,
            bound_function,
        )
    return command_sink, _IdentityResource("command_sink", command_sink)


def _validated_resource_key(value: Hashable) -> Hashable:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("ownership_resource 必须可哈希。") from exc
    return value


class CmdVelToPolicyAdapter:
    """对 ``cmd_vel → policy`` 唯一写入路径实施安全门。

    调用顺序通常为：

    1. ``claim(owner_id, now)`` 获取唯一写入权；
    2. ROS 回调分别调用 ``accept_cmd_vel``、``mark_odometry`` 和
       ``mark_point_cloud``；
    3. 活跃控制器周期性调用 ``renew_control_lease``；
    4. 每个 Isaac 控制 tick 调用 ``write``。

    所有时间都来自调用方传入的 ROS/仿真时钟，模块内部不读取 wall clock。
    """

    def __init__(
        self,
        command_sink: Any,
        config: CmdVelToPolicyConfig | None = None,
        *,
        ownership_resource: Hashable | None = None,
    ) -> None:
        self.config = config or CmdVelToPolicyConfig()
        self._sink, default_resource = _resolve_sink(command_sink)
        self._resource = _validated_resource_key(
            default_resource if ownership_resource is None else ownership_resource
        )
        self._token = object()
        self._owner_id: str | None = None
        self._requested_command: BodyVelocityCommand | None = None
        self._cmd_vel_stamp: float | None = None
        self._odometry_stamp: float | None = None
        self._point_cloud_stamp: float | None = None
        self._last_clock_time: float | None = None
        self._last_output_time: float | None = None
        self._last_output = BodyVelocityCommand.zero()
        self._invalid_command_pending = False
        self._clock_rewind_pending = False
        self._navigation_permit: NavigationSafetyPermit | None = None
        self._navigation_status_fault: str | None = None
        self._command_navigation_identity: tuple[int, int, int] | None = None
        self._last_navigation_status_sequence: int | None = None
        self._last_navigation_status_signature: tuple[object, ...] | None = None

    @property
    def owner_id(self) -> str | None:
        """返回本实例声明的 owner；不代表租约仍然新鲜。"""

        return self._owner_id

    @property
    def last_output(self) -> BodyVelocityCommand:
        """返回最近一次成功写入的命令。"""

        return self._last_output

    def claim(self, owner_id: str, now: Any) -> bool:
        """获取独占写入权；返回值表示是否发生了新接管。"""

        owner = self._validate_owner_id(owner_id)
        if self._owner_id is not None and self._owner_id != owner:
            raise PolicyCommandOwnershipError(
                "同一实例不能直接更换 owner_id；请先 release()。"
            )
        timestamp = _time_to_seconds(now, "now")
        already_claimed = self._owner_id == owner
        if already_claimed:
            self._observe_clock(timestamp)
        is_new_owner = _PolicyCommandOwnershipRegistry.claim(
            self._resource,
            self._token,
            owner,
            timestamp,
            self.config.control_lease_timeout_s,
            self.config.clock_rewind_tolerance_s,
            self._sink,
        )
        if is_new_owner or not already_claimed:
            self._clear_inputs()
            self._last_output = BodyVelocityCommand.zero()
            self._last_output_time = timestamp
            self._last_clock_time = timestamp
            self._clock_rewind_pending = False
        self._owner_id = owner
        return is_new_owner

    def reset(self, *, owner_id: str, now: Any) -> None:
        """在 episode/timeline 重置边界写零并清空所有旧输入。

        所有权保持不变，但控制租约从新的仿真时间重新计算；调用方随后必须重新
        提供 ``cmd_vel``、Odometry 和点云，才会恢复非零速度。
        """

        timestamp = _time_to_seconds(now, "now")
        self._require_owner(owner_id)
        self._observe_clock(timestamp)
        _PolicyCommandOwnershipRegistry.renew(
            self._resource,
            self._token,
            timestamp,
            self.config.control_lease_timeout_s,
        )
        zero = BodyVelocityCommand.zero()
        _PolicyCommandOwnershipRegistry.write(
            self._resource,
            self._token,
            zero,
            self._sink,
        )
        self._clear_inputs()
        self._last_output = zero
        self._last_output_time = timestamp
        self._last_clock_time = timestamp
        self._clock_rewind_pending = False

    def renew_control_lease(self, owner_id: str, now: Any) -> None:
        """续租当前控制器的仿真时间心跳。"""

        timestamp = _time_to_seconds(now, "now")
        self._require_owner(owner_id)
        self._observe_clock(timestamp)
        _PolicyCommandOwnershipRegistry.renew(
            self._resource,
            self._token,
            timestamp,
            self.config.control_lease_timeout_s,
        )

    def accept_cmd_vel(
        self,
        command: Any,
        *,
        owner_id: str,
        received_at: Any,
    ) -> BodyVelocityCommand | None:
        """接收速度及可选许可包络；非法输入会使旧命令立即失效。

        只有状态变化而没有新 ``Twist`` 时，桥会传入 ``command=None`` 的
        ``PolicyCommandInput``。方法此时只更新许可并返回 ``None``，同一控制
        周期随后的 ``tick`` 会通过本实例真实写入零速度。
        """

        timestamp = _time_to_seconds(received_at, "received_at")
        self._require_owner(owner_id)
        self._observe_clock(timestamp)
        raw_command = command
        if isinstance(command, PolicyCommandInput):
            if command.navigation_status_error is not None:
                self.invalidate_navigation_status(
                    command.navigation_status_error,
                    owner_id=owner_id,
                )
            elif command.navigation_permit is not None:
                self.accept_navigation_status(
                    command.navigation_permit,
                    owner_id=owner_id,
                )
            raw_command = command.command
            if raw_command is None:
                return None
        try:
            parsed = body_velocity_from_input(raw_command)
        except (TypeError, ValueError):
            self._requested_command = None
            self._cmd_vel_stamp = None
            self._command_navigation_identity = None
            self._invalid_command_pending = True
            raise
        self._requested_command = parsed
        self._cmd_vel_stamp = timestamp
        if not self.config.require_navigation_status:
            self._command_navigation_identity = None
        elif not self._navigation_permit_stop_reasons(
            timestamp,
            check_command_identity=False,
        ):
            assert self._navigation_permit is not None
            self._command_navigation_identity = (
                self._navigation_permit.command_identity
            )
        else:
            # 拒绝态收到的速度不能在后续许可恢复时自动复活。
            self._command_navigation_identity = None
        self._invalid_command_pending = False
        return parsed

    def receive(
        self,
        command: Any,
        stamp: Any,
        owner_id: str,
    ) -> BodyVelocityCommand | None:
        """提供便于 runtime 轮询接线的 ``receive(command, stamp, owner)`` 别名。"""

        return self.accept_cmd_vel(
            command,
            owner_id=owner_id,
            received_at=stamp,
        )

    def accept_navigation_status(
        self,
        permit: NavigationSafetyPermit,
        *,
        owner_id: str,
    ) -> bool:
        """接收一条严格许可；重复或回退不能刷新新鲜度。"""

        self._require_owner(owner_id)
        if not isinstance(permit, NavigationSafetyPermit):
            raise TypeError("permit 必须是 NavigationSafetyPermit。")
        previous_sequence = self._last_navigation_status_sequence
        previous_signature = self._last_navigation_status_signature
        if previous_sequence is not None:
            if permit.status_sequence < previous_sequence:
                self._navigation_permit = None
                self._command_navigation_identity = None
                self._navigation_status_fault = (
                    "navigation_status_sequence_regression"
                )
                return False
            if permit.status_sequence == previous_sequence:
                if permit.signature != previous_signature:
                    self._navigation_permit = None
                    self._command_navigation_identity = None
                    self._navigation_status_fault = (
                        "conflicting_navigation_status_sequence"
                    )
                    return False
                if (
                    self._navigation_permit is not None
                    and permit.identity_valid
                    != self._navigation_permit.identity_valid
                ):
                    # 本地目标或 Path identity 可能晚于同一 status 到达，也可能
                    # 被新目标/tombstone 立即撤销；这不是 supervisor 同序冲突。
                    self._navigation_permit = permit
                    self._navigation_status_fault = None
                    self._command_navigation_identity = None
                    return True
                # 完全重复不能借新的 Twist 接收时间刷新 status freshness。
                return False
        previous_permit = self._navigation_permit
        had_fault = self._navigation_status_fault is not None
        self._last_navigation_status_sequence = permit.status_sequence
        self._last_navigation_status_signature = permit.signature
        self._navigation_permit = permit
        self._navigation_status_fault = None
        if (
            had_fault
            or previous_permit is None
            or previous_permit.command_identity != permit.command_identity
            or not permit.allow_tracking_command
            or permit.force_zero_velocity
            or not permit.identity_valid
        ):
            self._command_navigation_identity = None
        return True

    def invalidate_navigation_status(
        self,
        reason: str,
        *,
        owner_id: str,
    ) -> None:
        """锁存接收或身份协议故障，直到更高序号有效状态到达。"""

        self._require_owner(owner_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason 必须是非空字符串。")
        self._navigation_permit = None
        self._command_navigation_identity = None
        self._navigation_status_fault = reason.strip()

    def navigation_gate_diagnostics(self) -> dict[str, object]:
        """返回唯一 policy writer 当前执行许可的可序列化只读证据。"""

        permit = self._navigation_permit
        command_identity = self._command_navigation_identity
        return {
            "schema": "navigation_policy_gate_diagnostics_v1",
            "required": bool(self.config.require_navigation_status),
            "timeout_s": float(self.config.navigation_status_timeout_s),
            "status_fault": self._navigation_status_fault,
            "permit_received": permit is not None,
            "permit": (
                None
                if permit is None
                else {
                    "header_stamp_ns": int(permit.header_stamp_ns),
                    "received_at": float(permit.received_at),
                    "status_sequence": int(permit.status_sequence),
                    "state_revision": int(permit.state_revision),
                    "goal_id": int(permit.goal_id),
                    "active_path_stamp_ns": int(
                        permit.active_path_stamp_ns
                    ),
                    "state": int(permit.state),
                    "allow_tracking_command": bool(
                        permit.allow_tracking_command
                    ),
                    "force_zero_velocity": bool(
                        permit.force_zero_velocity
                    ),
                    "identity_valid": bool(permit.identity_valid),
                    "reason": permit.reason,
                }
            ),
            "command_identity": (
                None
                if command_identity is None
                else [int(value) for value in command_identity]
            ),
            "command_identity_matches_permit": bool(
                permit is not None
                and command_identity == permit.command_identity
            ),
        }

    def mark_odometry(self, *, owner_id: str, received_at: Any) -> None:
        """记录一帧 Odometry 的 ROS/仿真时钟接收时间。"""

        timestamp = _time_to_seconds(received_at, "received_at")
        self._require_owner(owner_id)
        self._observe_clock(timestamp)
        self._odometry_stamp = timestamp

    def mark_point_cloud(self, *, owner_id: str, received_at: Any) -> None:
        """记录一帧点云的 ROS/仿真时钟接收时间。"""

        timestamp = _time_to_seconds(received_at, "received_at")
        self._require_owner(owner_id)
        self._observe_clock(timestamp)
        self._point_cloud_stamp = timestamp

    def write(self, *, owner_id: str, now: Any) -> PolicyCommandWriteReport:
        """执行一个安全控制 tick，并唯一写入最终速度命令。"""

        timestamp = _time_to_seconds(now, "now")
        owner = self._require_owner(owner_id)
        self._observe_clock(timestamp)
        reasons = self._stop_reasons(timestamp)
        requested = self._requested_command

        if reasons:
            if any(
                reason == "missing_navigation_status"
                or reason.startswith("navigation_")
                for reason in reasons
            ):
                # 许可曾失效后即使 heartbeat 恢复，也不能复用失效窗口前的
                # Twist；必须在恢复许可下重新绑定一条新命令。
                self._command_navigation_identity = None
            limited_target = BodyVelocityCommand.zero()
            written = BodyVelocityCommand.zero()
            clipped_axes: tuple[str, ...] = ()
            rate_limited_axes: tuple[str, ...] = ()
            motion_allowed = False
        else:
            assert requested is not None
            limited_target, clipped_axes = self._clip_command(requested)
            written, rate_limited_axes = self._rate_limit(limited_target, timestamp)
            motion_allowed = True

        _PolicyCommandOwnershipRegistry.write(
            self._resource,
            self._token,
            written,
            self._sink,
        )
        self._last_output = written
        self._last_output_time = timestamp
        self._clock_rewind_pending = False
        return PolicyCommandWriteReport(
            timestamp=timestamp,
            owner_id=owner,
            requested_command=requested,
            limited_target=limited_target,
            written_command=written,
            motion_allowed=motion_allowed,
            stop_reasons=tuple(reasons),
            clipped_axes=clipped_axes,
            rate_limited_axes=rate_limited_axes,
        )

    def tick(self, now: Any, owner_id: str) -> PolicyCommandWriteReport:
        """提供便于仿真控制循环接线的 ``tick(now, owner)`` 别名。"""

        return self.write(owner_id=owner_id, now=now)

    def emergency_stop(
        self,
        *,
        owner_id: str,
        now: Any,
        reason: str = "emergency_stop",
    ) -> PolicyCommandWriteReport:
        """立即写零并使当前 ``cmd_vel`` 失效，不对停车应用变化率限制。"""

        timestamp = _time_to_seconds(now, "now")
        owner = self._require_owner(owner_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason 必须是非空字符串。")
        self._observe_clock(timestamp)
        requested = self._requested_command
        self._requested_command = None
        self._cmd_vel_stamp = None
        self._command_navigation_identity = None
        zero = BodyVelocityCommand.zero()
        _PolicyCommandOwnershipRegistry.write(
            self._resource,
            self._token,
            zero,
            self._sink,
        )
        self._last_output = zero
        self._last_output_time = timestamp
        return PolicyCommandWriteReport(
            timestamp=timestamp,
            owner_id=owner,
            requested_command=requested,
            limited_target=zero,
            written_command=zero,
            motion_allowed=False,
            stop_reasons=(reason.strip(),),
            clipped_axes=(),
            rate_limited_axes=(),
        )

    def inhibit(
        self,
        *,
        owner_id: str,
        now: Any,
        reason: str = "temporary_inhibit",
    ) -> PolicyCommandWriteReport:
        """非锁存写零，并保留 Odometry/点云失鲜的独立停车原因。"""

        report = self.emergency_stop(
            owner_id=owner_id,
            now=now,
            reason=reason,
        )
        sensor_reasons = self._sensor_stop_reasons(report.timestamp)
        if not sensor_reasons:
            return report
        return replace(
            report,
            stop_reasons=tuple(
                dict.fromkeys((*report.stop_reasons, *sensor_reasons))
            ),
        )

    def release(self, *, owner_id: str, now: Any) -> None:
        """立即写零并释放 policy 命令入口。"""

        timestamp = _time_to_seconds(now, "now")
        self._require_owner(owner_id)
        self._observe_clock(timestamp)
        _PolicyCommandOwnershipRegistry.release(
            self._resource,
            self._token,
            self._sink,
        )
        self._clear_inputs()
        self._last_output = BodyVelocityCommand.zero()
        self._last_output_time = timestamp
        self._owner_id = None

    @staticmethod
    def _validate_owner_id(owner_id: Any) -> str:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id 必须是非空字符串。")
        return owner_id.strip()

    def _require_owner(self, owner_id: Any) -> str:
        owner = self._validate_owner_id(owner_id)
        if self._owner_id != owner:
            raise PolicyCommandOwnershipError(
                f"本实例 owner={self._owner_id!r}，调用 owner={owner!r}。"
            )
        _PolicyCommandOwnershipRegistry.verify(self._resource, self._token)
        return owner

    def _observe_clock(self, timestamp: float) -> None:
        if (
            self._last_clock_time is not None
            and timestamp
            < self._last_clock_time - self.config.clock_rewind_tolerance_s
        ):
            self._clear_inputs()
            self._last_output = BodyVelocityCommand.zero()
            self._last_output_time = timestamp
            self._clock_rewind_pending = True
        self._last_clock_time = timestamp

    def _clear_inputs(self) -> None:
        self._requested_command = None
        self._cmd_vel_stamp = None
        self._odometry_stamp = None
        self._point_cloud_stamp = None
        self._invalid_command_pending = False
        self._navigation_permit = None
        self._navigation_status_fault = None
        self._command_navigation_identity = None
        self._last_navigation_status_sequence = None
        self._last_navigation_status_signature = None

    def _freshness_reason(
        self,
        *,
        name: str,
        stamp: float | None,
        now: float,
        timeout_s: float,
    ) -> str | None:
        if stamp is None:
            return f"missing_{name}"
        if stamp > now + self.config.future_tolerance_s:
            return f"{name}_from_future"
        if now - stamp > timeout_s:
            return f"{name}_timeout"
        return None

    def _stop_reasons(self, now: float) -> list[str]:
        reasons: list[str] = []
        if self._clock_rewind_pending:
            reasons.append("clock_rewind")
        lease_status = _PolicyCommandOwnershipRegistry.lease_status(
            self._resource,
            self._token,
            now,
            self.config.future_tolerance_s,
        )
        if lease_status != "fresh":
            reasons.append(f"control_lease_{lease_status}")
        if self._invalid_command_pending:
            reasons.append("invalid_cmd_vel")
        cmd_reason = self._freshness_reason(
            name="cmd_vel",
            stamp=self._cmd_vel_stamp,
            now=now,
            timeout_s=self.config.cmd_vel_timeout_s,
        )
        if cmd_reason is not None:
            reasons.append(cmd_reason)
        reasons.extend(self._sensor_stop_reasons(now))
        reasons.extend(self._navigation_permit_stop_reasons(now))
        return reasons

    def _sensor_stop_reasons(self, now: float) -> list[str]:
        """返回不能被临时底盘锁掩盖的 ROS 传感器新鲜度故障。"""

        reasons: list[str] = []
        if self.config.require_odometry:
            odom_reason = self._freshness_reason(
                name="odometry",
                stamp=self._odometry_stamp,
                now=now,
                timeout_s=self.config.odometry_timeout_s,
            )
            if odom_reason is not None:
                reasons.append(odom_reason)
        if self.config.require_point_cloud:
            cloud_reason = self._freshness_reason(
                name="point_cloud",
                stamp=self._point_cloud_stamp,
                now=now,
                timeout_s=self.config.point_cloud_timeout_s,
            )
            if cloud_reason is not None:
                reasons.append(cloud_reason)
        return reasons

    def _navigation_permit_stop_reasons(
        self,
        now: float,
        *,
        check_command_identity: bool = True,
    ) -> list[str]:
        """返回 supervisor 许可、身份绑定与新鲜度的 fail-closed 原因。"""

        if not self.config.require_navigation_status:
            return []
        if self._navigation_status_fault is not None:
            return [
                "navigation_status_invalid:"
                f"{self._navigation_status_fault}"
            ]
        permit = self._navigation_permit
        if permit is None:
            return ["missing_navigation_status"]
        reasons: list[str] = []
        receipt_reason = self._freshness_reason(
            name="navigation_status",
            stamp=permit.received_at,
            now=now,
            timeout_s=self.config.navigation_status_timeout_s,
        )
        if receipt_reason is not None:
            reasons.append(receipt_reason)
        source_reason = self._freshness_reason(
            name="navigation_status_source",
            stamp=permit.source_timestamp,
            now=now,
            timeout_s=self.config.navigation_status_timeout_s,
        )
        if source_reason is not None:
            reasons.append(source_reason)
        if not permit.identity_valid:
            reasons.append("navigation_status_identity_invalid")
        if permit.force_zero_velocity:
            reasons.append("navigation_status_force_zero")
        if not permit.allow_tracking_command:
            reasons.append("navigation_tracking_not_allowed")
        if (
            check_command_identity
            and permit.allow_tracking_command
            and self._command_navigation_identity != permit.command_identity
        ):
            reasons.append("navigation_command_identity_mismatch")
        return reasons

    def _clip_command(
        self,
        command: BodyVelocityCommand,
    ) -> tuple[BodyVelocityCommand, tuple[str, ...]]:
        limits = (
            ("vx", command.vx, self.config.max_vx),
            ("vy", command.vy, self.config.max_vy),
            ("wz", command.wz, self.config.max_wz),
        )
        clipped_values: list[float] = []
        clipped_axes: list[str] = []
        for axis, value, limit in limits:
            clipped = min(limit, max(-limit, value))
            clipped_values.append(clipped)
            if clipped != value:
                clipped_axes.append(axis)
        return BodyVelocityCommand(*clipped_values), tuple(clipped_axes)

    def _rate_limit(
        self,
        target: BodyVelocityCommand,
        now: float,
    ) -> tuple[BodyVelocityCommand, tuple[str, ...]]:
        # 上游的严格零命令表示停车；若仍按普通减速斜率输出残余速度，
        # 超时、到达或急停会在若干控制周期内继续驱动机器人。
        if target == BodyVelocityCommand.zero():
            return BodyVelocityCommand.zero(), ()
        if self._last_output_time is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_output_time)
        rotate_in_place = target.vx == 0.0 and target.vy == 0.0
        axes = (
            ("vx", self._last_output.vx, target.vx, self.config.max_vx_rate),
            ("vy", self._last_output.vy, target.vy, self.config.max_vy_rate),
            ("wz", self._last_output.wz, target.wz, self.config.max_wz_rate),
        )
        values: list[float] = []
        limited_axes: list[str] = []
        for axis, previous, desired, rate in axes:
            # 航向误差过大时 controller 发布 (0, 0, wz)。policy 必须真正
            # 原地转向，不能把上一拍平移速度再拖行一个减速周期。
            if rotate_in_place and axis in {"vx", "vy"}:
                values.append(0.0)
                continue
            max_delta = rate * dt
            delta = desired - previous
            applied_delta = min(max_delta, max(-max_delta, delta))
            values.append(previous + applied_delta)
            if applied_delta != delta:
                limited_axes.append(axis)
        return BodyVelocityCommand(*values), tuple(limited_axes)
