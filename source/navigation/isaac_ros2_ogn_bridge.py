"""Isaac Sim 内部用于发布导航观测的 ROS 2 OmniGraph 桥接器。"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import numbers
import re
import struct
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .cmd_vel_to_policy import NavigationSafetyPermit, PolicyCommandInput


OdometrySource = Literal["direct", "compute"]

_USD_PRIM_PATH_RE = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$")
_ROS_TOPIC_RE = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$")
_FRAME_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*$")

# 与 Isaac Sim 5.1 ROS2QoSProfile 的 “Sensor Data” 预设逐字段一致。
SENSOR_DATA_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 5,
        "reliability": "bestEffort",
        "durability": "volatile",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

CLOCK_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "bestEffort",
        "durability": "volatile",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

CMD_VEL_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "reliable",
        "durability": "volatile",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

# ``goal_reached`` 是持续发布的状态量。订阅端使用 volatile，避免新建或重绑
# OGN graph 时把 transient-local 的上一轮 ``true`` 误当成本轮到达事件。
GOAL_REACHED_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "reliable",
        "durability": "volatile",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

# ``/initial_path`` 是一份可被晚加入订阅端读取的参考状态，而不是瞬时命令。
# 与 ROS 2 Path 发布端保持 reliable + transient-local，确保 Isaac 在 SCAN launch
# 之后创建 OGN graph 时仍能收到最新一代路径。
REFERENCE_PATH_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "reliable",
        "durability": "transientLocal",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

# ``/planning/controller_status`` 是 SCAN controller 的自包含生命周期快照。
# 与发布端保持 reliable + transient-local + keep-last(64)，既支持晚加入，也
# 防止同一 controller 回调连续发布“旧 active 终态 + 新轨迹接受态”时被覆盖。
CONTROLLER_STATUS_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 64,
        "reliability": "reliable",
        "durability": "transientLocal",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

# GridMap 与 B-spline 诊断是最终动态验收的有序证据流。必须可靠接收短时
# burst，并允许 Isaac 晚加入恢复最近窗口；消息自身 sequence 负责发现丢包。
PLANNING_DIAGNOSTICS_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 64,
        "reliability": "reliable",
        "durability": "transientLocal",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

# ``/navigation/status`` 是唯一 policy writer 的执行许可快照。晚加入的 Isaac
# 必须先读到当前拒绝/允许状态，但快照仍由 writer 的新鲜度租约约束，不能把
# transient-local 历史值无限期当作许可。
NAVIGATION_STATUS_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "reliable",
        "durability": "transientLocal",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

# ``/planning/stair_execution_frozen`` 是 Isaac 楼梯执行器对 SCAN planner
# 发布的带 Path 代际与 writer epoch 的类型化规划抑制快照。它与 controller 自身的
# ``/planning/go2_execution_frozen`` 含义和 writer 完全独立；晚加入的
# planner 必须立即读到最近一次冻结状态，因此使用 transient-local。
STAIR_EXECUTION_FROZEN_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "reliable",
        "durability": "transientLocal",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)

# ``/pct/goal`` 是单次目标命令。发布端使用 reliable + volatile，避免同一
# ROS 图重启后把上一轮任务目标自动重放给新的 PCT adapter。
PCT_GOAL_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "reliable",
        "durability": "volatile",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    },
    separators=(",", ":"),
    sort_keys=True,
)


@dataclass(frozen=True, slots=True)
class IsaacRos2OgnBridgeConfig:
    """定义 Isaac 侧 ROS 2 发布图及其接口名称。"""

    graph_path: str = "/World/PCTScanNavigationROS2Bridge"
    graph_backed_by_usd: bool = True
    robot_prim_path: str = "/World/envs/env_0/Robot"
    clock_topic: str = "/clock"
    odometry_topic: str = "/isaac/body_pose_raw"
    point_cloud_topic: str = "/isaac/cloud_registered_raw"
    command_topic: str = "/cmd_vel"
    goal_reached_topic: str = "/planning/goal_reached"
    controller_status_topic: str = "/planning/controller_status"
    grid_map_diagnostics_topic: str = (
        "/planning/grid_map_observation_diagnostics"
    )
    bspline_diagnostics_topic: str = "/planning/bspline_diagnostics"
    navigation_status_topic: str = "/navigation/status"
    stair_execution_frozen_topic: str = "/planning/stair_execution_frozen"
    stair_execution_frozen_writer_id: str = "isaac_ros2_ogn_bridge"
    reference_path_topic: str = "/initial_path"
    pct_goal_topic: str = "/pct/goal"
    odom_frame_id: str = "world"
    base_frame_id: str = "base_link"
    point_cloud_frame_id: str = "world"
    clock_qos_profile: str = CLOCK_QOS_PROFILE
    sensor_qos_profile: str = SENSOR_DATA_QOS_PROFILE
    command_qos_profile: str = CMD_VEL_QOS_PROFILE
    goal_reached_qos_profile: str = GOAL_REACHED_QOS_PROFILE
    controller_status_qos_profile: str = CONTROLLER_STATUS_QOS_PROFILE
    planning_diagnostics_qos_profile: str = (
        PLANNING_DIAGNOSTICS_QOS_PROFILE
    )
    navigation_status_qos_profile: str = NAVIGATION_STATUS_QOS_PROFILE
    stair_execution_frozen_qos_profile: str = (
        STAIR_EXECUTION_FROZEN_QOS_PROFILE
    )
    reference_path_qos_profile: str = REFERENCE_PATH_QOS_PROFILE
    pct_goal_qos_profile: str = PCT_GOAL_QOS_PROFILE
    domain_id: int | None = None
    odometry_source: OdometrySource = "direct"
    enable_command_subscription: bool = False
    enable_goal_reached_subscription: bool = False
    enable_controller_status_subscription: bool = False
    enable_grid_map_diagnostics_subscription: bool = False
    enable_bspline_diagnostics_subscription: bool = False
    enable_stair_execution_frozen_publisher: bool = False
    enable_reference_path_subscription: bool = True
    enable_pct_goal_publisher: bool = False
    reset_sim_time_on_stop: bool = False

    def __post_init__(self) -> None:
        _validate_prim_path(self.graph_path, "graph_path")
        if not isinstance(self.graph_backed_by_usd, bool):
            raise TypeError("graph_backed_by_usd 必须是布尔值。")
        _validate_prim_path(self.robot_prim_path, "robot_prim_path")
        _validate_topic(self.clock_topic, "clock_topic")
        _validate_topic(self.odometry_topic, "odometry_topic")
        _validate_topic(self.point_cloud_topic, "point_cloud_topic")
        _validate_topic(self.command_topic, "command_topic")
        _validate_topic(self.goal_reached_topic, "goal_reached_topic")
        _validate_topic(self.controller_status_topic, "controller_status_topic")
        _validate_topic(
            self.grid_map_diagnostics_topic,
            "grid_map_diagnostics_topic",
        )
        _validate_topic(
            self.bspline_diagnostics_topic,
            "bspline_diagnostics_topic",
        )
        _validate_topic(self.navigation_status_topic, "navigation_status_topic")
        _validate_topic(
            self.stair_execution_frozen_topic,
            "stair_execution_frozen_topic",
        )
        _validate_nonempty_text(
            self.stair_execution_frozen_writer_id,
            "stair_execution_frozen_writer_id",
        )
        _validate_topic(self.reference_path_topic, "reference_path_topic")
        _validate_topic(self.pct_goal_topic, "pct_goal_topic")
        _validate_frame_id(self.odom_frame_id, "odom_frame_id")
        _validate_frame_id(self.base_frame_id, "base_frame_id")
        _validate_frame_id(self.point_cloud_frame_id, "point_cloud_frame_id")
        if self.point_cloud_frame_id != self.odom_frame_id:
            raise ValueError(
                "未提供 TF 变换时 point_cloud_frame_id 必须与 odom_frame_id 相同。"
            )
        _validate_qos_profile(self.clock_qos_profile, "clock_qos_profile")
        _validate_qos_profile(self.sensor_qos_profile, "sensor_qos_profile")
        _validate_qos_profile(self.command_qos_profile, "command_qos_profile")
        _validate_qos_profile(
            self.goal_reached_qos_profile,
            "goal_reached_qos_profile",
        )
        _validate_qos_profile(
            self.controller_status_qos_profile,
            "controller_status_qos_profile",
        )
        _validate_qos_profile(
            self.planning_diagnostics_qos_profile,
            "planning_diagnostics_qos_profile",
        )
        _validate_qos_profile(
            self.navigation_status_qos_profile,
            "navigation_status_qos_profile",
        )
        _validate_qos_profile(
            self.stair_execution_frozen_qos_profile,
            "stair_execution_frozen_qos_profile",
        )
        _validate_qos_profile(
            self.reference_path_qos_profile,
            "reference_path_qos_profile",
        )
        _validate_qos_profile(
            self.pct_goal_qos_profile,
            "pct_goal_qos_profile",
        )
        reference_path_qos = json.loads(self.reference_path_qos_profile)
        if (
            reference_path_qos["reliability"] != "reliable"
            or reference_path_qos["durability"] != "transientLocal"
        ):
            raise ValueError(
                "reference_path_qos_profile 必须使用 reliable + transientLocal。"
            )
        controller_status_qos = json.loads(self.controller_status_qos_profile)
        if (
            controller_status_qos["reliability"] != "reliable"
            or controller_status_qos["durability"] != "transientLocal"
            or controller_status_qos["history"] != "keepLast"
            or controller_status_qos["depth"] != 64
        ):
            raise ValueError(
                "controller_status_qos_profile 必须使用 "
                "reliable + transientLocal + keepLast(64)。"
            )
        navigation_status_qos = json.loads(self.navigation_status_qos_profile)
        if (
            navigation_status_qos["reliability"] != "reliable"
            or navigation_status_qos["durability"] != "transientLocal"
            or navigation_status_qos["history"] != "keepLast"
            or navigation_status_qos["depth"] != 1
        ):
            raise ValueError(
                "navigation_status_qos_profile 必须使用 "
                "reliable + transientLocal + keepLast(1)。"
            )
        diagnostics_qos = json.loads(self.planning_diagnostics_qos_profile)
        if (
            diagnostics_qos["reliability"] != "reliable"
            or diagnostics_qos["durability"] != "transientLocal"
            or diagnostics_qos["history"] != "keepLast"
            or diagnostics_qos["depth"] != 64
        ):
            raise ValueError(
                "planning_diagnostics_qos_profile 必须使用 "
                "reliable + transientLocal + keepLast(64)。"
            )
        stair_execution_frozen_qos = json.loads(
            self.stair_execution_frozen_qos_profile
        )
        if (
            stair_execution_frozen_qos["reliability"] != "reliable"
            or stair_execution_frozen_qos["durability"] != "transientLocal"
            or stair_execution_frozen_qos["history"] != "keepLast"
            or stair_execution_frozen_qos["depth"] != 1
        ):
            raise ValueError(
                "stair_execution_frozen_qos_profile 必须使用 "
                "reliable + transientLocal + keepLast(1)。"
            )
        pct_goal_qos = json.loads(self.pct_goal_qos_profile)
        if (
            pct_goal_qos["reliability"] != "reliable"
            or pct_goal_qos["durability"] != "volatile"
        ):
            raise ValueError(
                "pct_goal_qos_profile 必须使用 reliable + volatile。"
            )
        if not isinstance(self.enable_command_subscription, bool):
            raise TypeError("enable_command_subscription 必须是布尔值。")
        if not isinstance(self.enable_goal_reached_subscription, bool):
            raise TypeError("enable_goal_reached_subscription 必须是布尔值。")
        if not isinstance(self.enable_controller_status_subscription, bool):
            raise TypeError("enable_controller_status_subscription 必须是布尔值。")
        if not isinstance(
            self.enable_grid_map_diagnostics_subscription,
            bool,
        ):
            raise TypeError(
                "enable_grid_map_diagnostics_subscription 必须是布尔值。"
            )
        if not isinstance(
            self.enable_bspline_diagnostics_subscription,
            bool,
        ):
            raise TypeError(
                "enable_bspline_diagnostics_subscription 必须是布尔值。"
            )
        if not isinstance(self.enable_stair_execution_frozen_publisher, bool):
            raise TypeError(
                "enable_stair_execution_frozen_publisher 必须是布尔值。"
            )
        if not isinstance(self.enable_reference_path_subscription, bool):
            raise TypeError("enable_reference_path_subscription 必须是布尔值。")
        if not isinstance(self.enable_pct_goal_publisher, bool):
            raise TypeError("enable_pct_goal_publisher 必须是布尔值。")
        if self.domain_id is not None:
            if isinstance(self.domain_id, bool) or not isinstance(self.domain_id, int):
                raise TypeError("domain_id 必须是整数或 None。")
            if not 0 <= self.domain_id <= 232:
                raise ValueError("domain_id 必须位于 [0, 232]。")
        if self.odometry_source not in ("direct", "compute"):
            raise ValueError("odometry_source 只能是 'direct' 或 'compute'。")

    @property
    def use_domain_id_environment(self) -> bool:
        """未显式配置 domain 时让 ROS2Context 读取 ROS_DOMAIN_ID。"""

        return self.domain_id is None


@dataclass(frozen=True, slots=True)
class OgnGraphSpec:
    """不依赖 Isaac 模块、可直接单元测试的 OmniGraph 描述。"""

    graph_path: str
    evaluator_name: str
    create_nodes: tuple[tuple[str, str], ...]
    set_values: tuple[tuple[str, object], ...]
    connections: tuple[tuple[str, str], ...]

    def node_types(self) -> dict[str, str]:
        """返回按节点名称索引的 OGN 类型。"""

        return dict(self.create_nodes)

    def values(self) -> dict[str, object]:
        """返回按属性路径索引的静态值。"""

        return dict(self.set_values)


@dataclass(frozen=True, slots=True)
class OgnOdometrySample:
    """已经转换为 OGN 端口顺序的单帧里程计数据。"""

    position: tuple[float, float, float]
    orientation_ijkr: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    timestamp: float


@dataclass(frozen=True, slots=True)
class OgnTwistSample:
    """一条由 OGN 收到并按仿真时钟标记的机体系速度命令。"""

    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    receipt_timestamp: float
    sequence: int
    command_present: bool = True
    navigation_permit: NavigationSafetyPermit | None = None
    navigation_status_error: str | None = None

    @property
    def planar_command(
        self,
    ) -> tuple[float, float, float] | PolicyCommandInput:
        """返回 policy 输入；生产链把安全许可与 Twist 送入同一 writer。"""

        command = (
            self.linear_velocity[0],
            self.linear_velocity[1],
            self.angular_velocity[2],
        )
        if self.navigation_permit is None and self.navigation_status_error is None:
            return command
        return PolicyCommandInput(
            command=command if self.command_present else None,
            navigation_permit=self.navigation_permit,
            navigation_status_error=self.navigation_status_error,
        )


@dataclass(frozen=True, slots=True)
class OgnNavigationStatusSample:
    """一条经过严格校验的 supervisor 执行许可快照。"""

    source_topic: str
    receipt_timestamp: float
    rx_sequence: int
    frame_id: str
    header_stamp_sec: int
    header_stamp_nanosec: int
    status_sequence: int
    state_revision: int
    goal_id: int
    state: int
    allow_tracking_command: bool
    force_zero_velocity: bool
    stop_confirmed: bool
    global_replan_requested: bool
    global_replan_in_flight: bool
    global_replan_request_id: int
    pct_plan_id: int
    active_path_stamp_sec: int
    active_path_stamp_nanosec: int
    consecutive_scan_failures: int
    stale_inputs: tuple[str, ...]
    reason: str

    @property
    def header_stamp_ns(self) -> int:
        """返回状态 Header 的精确整数纳秒。"""

        return self.header_stamp_sec * 1_000_000_000 + self.header_stamp_nanosec

    @property
    def active_path_stamp_ns(self) -> int:
        """返回 supervisor 当前 Path identity 的精确整数纳秒。"""

        return (
            self.active_path_stamp_sec * 1_000_000_000
            + self.active_path_stamp_nanosec
        )

    def to_safety_permit(self, *, identity_valid: bool) -> NavigationSafetyPermit:
        """转换为不依赖 ROS/OGN 的 policy writer 许可。"""

        return NavigationSafetyPermit(
            header_stamp_ns=self.header_stamp_ns,
            received_at=self.receipt_timestamp,
            status_sequence=self.status_sequence,
            state_revision=self.state_revision,
            goal_id=self.goal_id,
            active_path_stamp_ns=self.active_path_stamp_ns,
            state=self.state,
            allow_tracking_command=self.allow_tracking_command,
            force_zero_velocity=self.force_zero_velocity,
            identity_valid=identity_valid,
            reason=self.reason,
        )


@dataclass(frozen=True, slots=True)
class OgnBoolSample:
    """一条由 OGN 收到并按仿真时钟标记的布尔状态。"""

    value: bool
    receipt_timestamp: float
    sequence: int


@dataclass(frozen=True, slots=True)
class OgnStairExecutionFreezePublicationReport:
    """一条由 generic OGN publisher 成功触发的类型化楼梯冻结快照。"""

    frozen: bool
    source_topic: str
    publish_timestamp: float
    header_stamp_sec: int
    header_stamp_nanosec: int
    reference_path_stamp_sec: int
    reference_path_stamp_nanosec: int
    writer_id: str
    writer_epoch: str
    sequence: int

    @property
    def value(self) -> bool:
        """保留 runtime metadata 的只读命名，不代表 ROS 消息仍是裸 Bool。"""

        return self.frozen

    @property
    def reference_path_stamp_ns(self) -> int:
        """返回快照绑定的精确 Path identity。"""

        return (
            self.reference_path_stamp_sec * 1_000_000_000
            + self.reference_path_stamp_nanosec
        )


@dataclass(frozen=True, slots=True)
class OgnControllerStatusSample:
    """一条经过结构与生命周期交叉校验的 SCAN controller 状态。"""

    source_topic: str
    receipt_timestamp: float
    rx_sequence: int
    frame_id: str
    header_stamp_sec: int
    header_stamp_nanosec: int
    status_sequence: int
    acceptance_sequence: int
    event: int
    reference_path_stamp_sec: int
    reference_path_stamp_nanosec: int
    bspline_header_stamp_sec: int
    bspline_header_stamp_nanosec: int
    start_time_sec: int
    start_time_nanosec: int
    traj_id: int
    accepted: bool
    trajectory_valid: bool
    is_final: bool
    emergency_stop: bool
    state: int
    reason: str
    candidate_present: bool
    candidate_reference_path_stamp_sec: int
    candidate_reference_path_stamp_nanosec: int
    candidate_bspline_header_stamp_sec: int
    candidate_bspline_header_stamp_nanosec: int
    candidate_start_time_sec: int
    candidate_start_time_nanosec: int
    candidate_traj_id: int
    active_sensing_yaw_only: bool = False
    command_sample_count: int = 0
    first_command: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    max_abs_vx: float = 0.0
    max_abs_vy: float = 0.0
    max_abs_wz: float = 0.0
    command_violation_count: int = 0

    @staticmethod
    def _stamp(sec: int, nanosec: int) -> dict[str, int]:
        return {"sec": sec, "nanosec": nanosec}

    @staticmethod
    def _stamp_ns(sec: int, nanosec: int) -> int:
        return sec * 1_000_000_000 + nanosec

    @property
    def header_stamp(self) -> dict[str, int]:
        """返回状态发布时刻，不经浮点数转换。"""

        return self._stamp(self.header_stamp_sec, self.header_stamp_nanosec)

    @property
    def header_stamp_ns(self) -> int:
        return self._stamp_ns(self.header_stamp_sec, self.header_stamp_nanosec)

    @property
    def reference_path_stamp(self) -> dict[str, int]:
        return self._stamp(
            self.reference_path_stamp_sec,
            self.reference_path_stamp_nanosec,
        )

    @property
    def reference_path_stamp_ns(self) -> int:
        return self._stamp_ns(
            self.reference_path_stamp_sec,
            self.reference_path_stamp_nanosec,
        )

    @property
    def bspline_header_stamp(self) -> dict[str, int]:
        return self._stamp(
            self.bspline_header_stamp_sec,
            self.bspline_header_stamp_nanosec,
        )

    @property
    def bspline_header_stamp_ns(self) -> int:
        return self._stamp_ns(
            self.bspline_header_stamp_sec,
            self.bspline_header_stamp_nanosec,
        )

    @property
    def start_time(self) -> dict[str, int]:
        return self._stamp(self.start_time_sec, self.start_time_nanosec)

    @property
    def start_time_ns(self) -> int:
        return self._stamp_ns(self.start_time_sec, self.start_time_nanosec)

    @property
    def candidate_reference_path_stamp(self) -> dict[str, int]:
        return self._stamp(
            self.candidate_reference_path_stamp_sec,
            self.candidate_reference_path_stamp_nanosec,
        )

    @property
    def candidate_reference_path_stamp_ns(self) -> int:
        return self._stamp_ns(
            self.candidate_reference_path_stamp_sec,
            self.candidate_reference_path_stamp_nanosec,
        )

    @property
    def candidate_bspline_header_stamp(self) -> dict[str, int]:
        return self._stamp(
            self.candidate_bspline_header_stamp_sec,
            self.candidate_bspline_header_stamp_nanosec,
        )

    @property
    def candidate_bspline_header_stamp_ns(self) -> int:
        return self._stamp_ns(
            self.candidate_bspline_header_stamp_sec,
            self.candidate_bspline_header_stamp_nanosec,
        )

    @property
    def candidate_start_time(self) -> dict[str, int]:
        return self._stamp(
            self.candidate_start_time_sec,
            self.candidate_start_time_nanosec,
        )

    @property
    def candidate_start_time_ns(self) -> int:
        return self._stamp_ns(
            self.candidate_start_time_sec,
            self.candidate_start_time_nanosec,
        )


@dataclass(frozen=True, slots=True)
class OgnGridMapObservationDiagnosticsSample:
    """一帧过滤后点云在 GridMap 中实际融合和清障的 typed 证据。"""

    source_topic: str
    receipt_timestamp: float
    rx_sequence: int
    frame_id: str
    header_stamp_sec: int
    header_stamp_nanosec: int
    observation_sequence: int
    sensor_pose_stamp_sec: int
    sensor_pose_stamp_nanosec: int
    sensor_origin: tuple[float, float, float]
    canonical_empty: bool
    map_fusion_performed: bool
    map_resolution: float
    input_point_count: int
    accepted_endpoint_count: int
    hit_endpoint_count: int
    explicit_free_endpoint_count: int
    hit_endpoint_samples_truncated: bool
    hit_endpoint_samples: tuple[tuple[float, float, float], ...]
    hit_endpoint_sample_voxel_indices: tuple[tuple[int, int, int], ...]
    free_to_occupied_transition_count: int
    free_to_occupied_transition_samples_truncated: bool
    free_to_occupied_transition_hit_samples: tuple[
        tuple[float, float, float], ...
    ]
    free_to_occupied_transition_voxel_indices: tuple[
        tuple[int, int, int], ...
    ]
    explicit_free_miss_voxel_count: int
    occupied_to_free_by_explicit_miss_count: int
    occupied_to_free_samples_truncated: bool
    occupied_to_free_by_explicit_miss_samples: tuple[
        tuple[float, float, float], ...
    ]
    occupied_to_free_sample_voxel_indices: tuple[
        tuple[int, int, int], ...
    ]
    occupied_to_free_transition_hit_observation_sequences: tuple[int, ...]
    occupied_to_free_transition_hit_samples: tuple[
        tuple[float, float, float], ...
    ]
    occupied_to_free_transition_hit_header_stamp_ns: tuple[int, ...]
    occupied_removed_by_sliding_reset_count: int

    @property
    def header_stamp_ns(self) -> int:
        return self.header_stamp_sec * 1_000_000_000 + self.header_stamp_nanosec

    @property
    def sensor_pose_stamp_ns(self) -> int:
        return (
            self.sensor_pose_stamp_sec * 1_000_000_000
            + self.sensor_pose_stamp_nanosec
        )


@dataclass(frozen=True, slots=True)
class OgnBsplineDiagnosticsSample:
    """一条已发布 B-spline 的 identity、ordered corridor 与有界几何证据。"""

    source_topic: str
    receipt_timestamp: float
    rx_sequence: int
    frame_id: str
    header_stamp_sec: int
    header_stamp_nanosec: int
    diagnostic_sequence: int
    start_time_sec: int
    start_time_nanosec: int
    reference_path_stamp_sec: int
    reference_path_stamp_nanosec: int
    traj_id: int
    is_final: bool
    emergency_stop: bool
    stationary: bool
    ordered_reference_checked: bool
    ordered_reference_safe: bool
    maximum_trajectory_deviation: float
    maximum_guide_anchor_deviation: float
    maximum_guide_progress_lead: float
    maximum_deviation_limit: float
    maximum_progress_lead_limit: float
    trajectory_duration: float
    maximum_velocity_upper_bound: float
    double_cylinder_radius: float
    double_cylinder_offset: float
    trajectory_sample_count_total: int
    trajectory_samples_truncated: bool
    trajectory_samples: tuple[tuple[float, float, float], ...]
    ordered_reference_sample_count_total: int
    ordered_reference_samples_truncated: bool
    ordered_reference_samples: tuple[tuple[float, float, float], ...]
    active_sensing: bool = False
    active_sensing_event: int = 0
    active_sensing_start_yaw: float = 0.0
    active_sensing_target_yaw: float = 0.0
    active_sensing_yaw_offset: float = 0.0
    active_sensing_yaw_rate: float = 0.0
    active_sensing_settle_stamp_sec: int = 0
    active_sensing_settle_stamp_nanosec: int = 0
    active_sensing_settle_yaw_error: float = 0.0
    active_sensing_settle_angular_speed: float = 0.0
    active_sensing_stable_duration: float = 0.0
    active_sensing_fusion_baseline: int = 0
    active_sensing_fusion_current: int = 0
    active_sensing_fusion_distinct: int = 0
    active_sensing_fusion_required: int = 0
    active_sensing_completed: bool = False
    active_sensing_failed: bool = False
    active_sensing_reason: str = ""

    @property
    def header_stamp_ns(self) -> int:
        return self.header_stamp_sec * 1_000_000_000 + self.header_stamp_nanosec

    @property
    def start_time_ns(self) -> int:
        return self.start_time_sec * 1_000_000_000 + self.start_time_nanosec

    @property
    def reference_path_stamp_ns(self) -> int:
        return (
            self.reference_path_stamp_sec * 1_000_000_000
            + self.reference_path_stamp_nanosec
        )

    @property
    def active_sensing_settle_stamp_ns(self) -> int:
        """返回主动感知稳定窗起点的精确整数纳秒。"""

        return (
            self.active_sensing_settle_stamp_sec * 1_000_000_000
            + self.active_sensing_settle_stamp_nanosec
        )


@dataclass(frozen=True, slots=True)
class OgnPathSample:
    """一代经过严格校验的 ``nav_msgs/msg/Path`` 地面高度参考路径。"""

    points_ground_xyz: tuple[tuple[float, float, float], ...]
    terminal_yaw: float | None
    source_topic: str
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int
    sequence: int
    points_sha256: str

    @property
    def stamp(self) -> dict[str, int]:
        """返回不损失纳秒精度、可直接写入 metadata 的 ROS 时间戳。"""

        return {
            "sec": self.stamp_sec,
            "nanosec": self.stamp_nanosec,
        }

    @property
    def stamp_ns(self) -> int:
        """返回 Path identity 的精确整数纳秒。"""

        return self.stamp_sec * 1_000_000_000 + self.stamp_nanosec


@dataclass(frozen=True, slots=True)
class OgnPCTGoalSample:
    """一条已经由 Isaac OGN 发布的 PCT base 高度目标。"""

    position_base_xyz: tuple[float, float, float]
    yaw: float
    source_topic: str
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int
    sequence: int

    @property
    def stamp(self) -> dict[str, int]:
        """返回不损失纳秒精度的 ROS 时间戳。"""

        return {
            "sec": self.stamp_sec,
            "nanosec": self.stamp_nanosec,
        }

    @property
    def stamp_ns(self) -> int:
        """返回与 PCT ``goal_id`` 完全一致的精确整数纳秒。"""

        return self.stamp_sec * 1_000_000_000 + self.stamp_nanosec


def build_graph_spec(
    config: IsaacRos2OgnBridgeConfig | None = None,
) -> OgnGraphSpec:
    """构造 Isaac Sim 5.1 发布图；此函数不会导入 OmniGraph。"""

    cfg = config or IsaacRos2OgnBridgeConfig()
    nodes: list[tuple[str, str]] = [
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("StateTx", "omni.graph.action.OnImpulseEvent"),
        ("CloudTx", "omni.graph.action.OnImpulseEvent"),
        ("IsaacReadSimulationTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("ROS2PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("ROS2PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ("ROS2PublishPointCloud", "isaacsim.ros2.bridge.ROS2PublishPointCloud"),
    ]
    if cfg.odometry_source == "compute":
        nodes.append(("IsaacComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"))
    if cfg.enable_command_subscription:
        nodes.extend(
            [
                ("CommandRxTick", "omni.graph.action.OnImpulseEvent"),
                (
                    "ROS2SubscribeTwist",
                    "isaacsim.ros2.bridge.ROS2SubscribeTwist",
                ),
                ("CommandRxCounter", "omni.graph.action.Counter"),
                ("NavigationStatusRxTick", "omni.graph.action.OnImpulseEvent"),
                (
                    "ROS2SubscribeNavigationStatus",
                    "isaacsim.ros2.bridge.ROS2Subscriber",
                ),
                ("NavigationStatusRxCounter", "omni.graph.action.Counter"),
            ]
        )
    if cfg.enable_goal_reached_subscription:
        nodes.extend(
            [
                ("GoalReachedRxTick", "omni.graph.action.OnImpulseEvent"),
                (
                    "ROS2SubscribeGoalReached",
                    "isaacsim.ros2.bridge.ROS2Subscriber",
                ),
                ("GoalReachedRxCounter", "omni.graph.action.Counter"),
            ]
        )
    if cfg.enable_controller_status_subscription:
        nodes.extend(
            [
                ("ControllerStatusRxTick", "omni.graph.action.OnImpulseEvent"),
                (
                    "ROS2SubscribeControllerStatus",
                    "isaacsim.ros2.bridge.ROS2Subscriber",
                ),
                ("ControllerStatusRxCounter", "omni.graph.action.Counter"),
            ]
        )
    if cfg.enable_grid_map_diagnostics_subscription:
        nodes.extend(
            [
                ("GridMapDiagnosticsRxTick", "omni.graph.action.OnImpulseEvent"),
                (
                    "ROS2SubscribeGridMapDiagnostics",
                    "isaacsim.ros2.bridge.ROS2Subscriber",
                ),
                ("GridMapDiagnosticsRxCounter", "omni.graph.action.Counter"),
            ]
        )
    if cfg.enable_bspline_diagnostics_subscription:
        nodes.extend(
            [
                ("BsplineDiagnosticsRxTick", "omni.graph.action.OnImpulseEvent"),
                (
                    "ROS2SubscribeBsplineDiagnostics",
                    "isaacsim.ros2.bridge.ROS2Subscriber",
                ),
                ("BsplineDiagnosticsRxCounter", "omni.graph.action.Counter"),
            ]
        )
    if cfg.enable_reference_path_subscription:
        nodes.extend(
            [
                ("ReferencePathRxTick", "omni.graph.action.OnImpulseEvent"),
                (
                    "ROS2SubscribeReferencePath",
                    "isaacsim.ros2.bridge.ROS2Subscriber",
                ),
                ("ReferencePathRxCounter", "omni.graph.action.Counter"),
            ]
        )
    if cfg.enable_pct_goal_publisher:
        nodes.extend(
            [
                ("PCTGoalTx", "omni.graph.action.OnImpulseEvent"),
                (
                    "ROS2PublishPCTGoal",
                    "isaacsim.ros2.bridge.ROS2Publisher",
                ),
            ]
        )
    if cfg.enable_stair_execution_frozen_publisher:
        nodes.extend(
            [
                (
                    "StairExecutionFrozenTx",
                    "omni.graph.action.OnImpulseEvent",
                ),
                (
                    "ROS2PublishStairExecutionFrozen",
                    "isaacsim.ros2.bridge.ROS2Publisher",
                ),
            ]
        )

    values: list[tuple[str, object]] = [
        ("ROS2Context.inputs:domain_id", 0 if cfg.domain_id is None else cfg.domain_id),
        ("ROS2Context.inputs:useDomainIDEnvVar", cfg.use_domain_id_environment),
        ("StateTx.inputs:onlyPlayback", True),
        ("CloudTx.inputs:onlyPlayback", True),
        (
            "IsaacReadSimulationTime.inputs:resetOnStop",
            cfg.reset_sim_time_on_stop,
        ),
        ("ROS2PublishClock.inputs:topicName", cfg.clock_topic),
        ("ROS2PublishClock.inputs:qosProfile", cfg.clock_qos_profile),
        ("ROS2PublishClock.inputs:queueSize", 1),
        ("ROS2PublishOdometry.inputs:topicName", cfg.odometry_topic),
        ("ROS2PublishOdometry.inputs:odomFrameId", cfg.odom_frame_id),
        ("ROS2PublishOdometry.inputs:chassisFrameId", cfg.base_frame_id),
        (
            "ROS2PublishOdometry.inputs:publishRawVelocities",
            cfg.odometry_source == "direct",
        ),
        ("ROS2PublishOdometry.inputs:qosProfile", cfg.sensor_qos_profile),
        ("ROS2PublishPointCloud.inputs:topicName", cfg.point_cloud_topic),
        ("ROS2PublishPointCloud.inputs:frameId", cfg.point_cloud_frame_id),
        ("ROS2PublishPointCloud.inputs:qosProfile", cfg.sensor_qos_profile),
        ("ROS2PublishPointCloud.inputs:cudaDeviceIndex", -1),
        ("ROS2PublishPointCloud.inputs:dataPtr", 0),
        ("ROS2PublishPointCloud.inputs:bufferSize", 0),
    ]
    if cfg.enable_command_subscription:
        values.extend(
            [
                ("CommandRxTick.inputs:onlyPlayback", True),
                ("ROS2SubscribeTwist.inputs:topicName", cfg.command_topic),
                (
                    "ROS2SubscribeTwist.inputs:qosProfile",
                    cfg.command_qos_profile,
                ),
                ("ROS2SubscribeTwist.inputs:queueSize", 1),
                ("NavigationStatusRxTick.inputs:onlyPlayback", True),
                (
                    "ROS2SubscribeNavigationStatus.inputs:messagePackage",
                    "scan_planner_msgs",
                ),
                (
                    "ROS2SubscribeNavigationStatus.inputs:messageSubfolder",
                    "msg",
                ),
                (
                    "ROS2SubscribeNavigationStatus.inputs:messageName",
                    "NavigationStatus",
                ),
                (
                    "ROS2SubscribeNavigationStatus.inputs:topicName",
                    cfg.navigation_status_topic,
                ),
                (
                    "ROS2SubscribeNavigationStatus.inputs:qosProfile",
                    cfg.navigation_status_qos_profile,
                ),
                ("ROS2SubscribeNavigationStatus.inputs:queueSize", 1),
            ]
        )
    if cfg.enable_goal_reached_subscription:
        values.extend(
            [
                ("GoalReachedRxTick.inputs:onlyPlayback", True),
                (
                    "ROS2SubscribeGoalReached.inputs:messagePackage",
                    "std_msgs",
                ),
                (
                    "ROS2SubscribeGoalReached.inputs:messageSubfolder",
                    "msg",
                ),
                ("ROS2SubscribeGoalReached.inputs:messageName", "Bool"),
                (
                    "ROS2SubscribeGoalReached.inputs:topicName",
                    cfg.goal_reached_topic,
                ),
                (
                    "ROS2SubscribeGoalReached.inputs:qosProfile",
                    cfg.goal_reached_qos_profile,
                ),
                ("ROS2SubscribeGoalReached.inputs:queueSize", 1),
            ]
        )
    if cfg.enable_controller_status_subscription:
        values.extend(
            [
                ("ControllerStatusRxTick.inputs:onlyPlayback", True),
                (
                    "ROS2SubscribeControllerStatus.inputs:messagePackage",
                    "scan_planner_msgs",
                ),
                (
                    "ROS2SubscribeControllerStatus.inputs:messageSubfolder",
                    "msg",
                ),
                (
                    "ROS2SubscribeControllerStatus.inputs:messageName",
                    "ControllerStatus",
                ),
                (
                    "ROS2SubscribeControllerStatus.inputs:topicName",
                    cfg.controller_status_topic,
                ),
                (
                    "ROS2SubscribeControllerStatus.inputs:qosProfile",
                    cfg.controller_status_qos_profile,
                ),
                ("ROS2SubscribeControllerStatus.inputs:queueSize", 64),
            ]
        )
    if cfg.enable_grid_map_diagnostics_subscription:
        values.extend(
            [
                ("GridMapDiagnosticsRxTick.inputs:onlyPlayback", True),
                (
                    "ROS2SubscribeGridMapDiagnostics.inputs:messagePackage",
                    "scan_planner_msgs",
                ),
                (
                    "ROS2SubscribeGridMapDiagnostics.inputs:messageSubfolder",
                    "msg",
                ),
                (
                    "ROS2SubscribeGridMapDiagnostics.inputs:messageName",
                    "GridMapObservationDiagnostics",
                ),
                (
                    "ROS2SubscribeGridMapDiagnostics.inputs:topicName",
                    cfg.grid_map_diagnostics_topic,
                ),
                (
                    "ROS2SubscribeGridMapDiagnostics.inputs:qosProfile",
                    cfg.planning_diagnostics_qos_profile,
                ),
                ("ROS2SubscribeGridMapDiagnostics.inputs:queueSize", 64),
            ]
        )
    if cfg.enable_bspline_diagnostics_subscription:
        values.extend(
            [
                ("BsplineDiagnosticsRxTick.inputs:onlyPlayback", True),
                (
                    "ROS2SubscribeBsplineDiagnostics.inputs:messagePackage",
                    "scan_planner_msgs",
                ),
                (
                    "ROS2SubscribeBsplineDiagnostics.inputs:messageSubfolder",
                    "msg",
                ),
                (
                    "ROS2SubscribeBsplineDiagnostics.inputs:messageName",
                    "BsplineDiagnostics",
                ),
                (
                    "ROS2SubscribeBsplineDiagnostics.inputs:topicName",
                    cfg.bspline_diagnostics_topic,
                ),
                (
                    "ROS2SubscribeBsplineDiagnostics.inputs:qosProfile",
                    cfg.planning_diagnostics_qos_profile,
                ),
                ("ROS2SubscribeBsplineDiagnostics.inputs:queueSize", 64),
            ]
        )
    if cfg.enable_reference_path_subscription:
        values.extend(
            [
                ("ReferencePathRxTick.inputs:onlyPlayback", True),
                (
                    "ROS2SubscribeReferencePath.inputs:messagePackage",
                    "nav_msgs",
                ),
                (
                    "ROS2SubscribeReferencePath.inputs:messageSubfolder",
                    "msg",
                ),
                ("ROS2SubscribeReferencePath.inputs:messageName", "Path"),
                (
                    "ROS2SubscribeReferencePath.inputs:topicName",
                    cfg.reference_path_topic,
                ),
                (
                    "ROS2SubscribeReferencePath.inputs:qosProfile",
                    cfg.reference_path_qos_profile,
                ),
                ("ROS2SubscribeReferencePath.inputs:queueSize", 1),
            ]
        )
    if cfg.enable_pct_goal_publisher:
        values.extend(
            [
                ("PCTGoalTx.inputs:onlyPlayback", True),
                (
                    "ROS2PublishPCTGoal.inputs:messagePackage",
                    "geometry_msgs",
                ),
                (
                    "ROS2PublishPCTGoal.inputs:messageSubfolder",
                    "msg",
                ),
                ("ROS2PublishPCTGoal.inputs:messageName", "PoseStamped"),
                (
                    "ROS2PublishPCTGoal.inputs:topicName",
                    cfg.pct_goal_topic,
                ),
                (
                    "ROS2PublishPCTGoal.inputs:qosProfile",
                    cfg.pct_goal_qos_profile,
                ),
                ("ROS2PublishPCTGoal.inputs:queueSize", 1),
            ]
        )
    if cfg.enable_stair_execution_frozen_publisher:
        values.extend(
            [
                ("StairExecutionFrozenTx.inputs:onlyPlayback", True),
                (
                    "ROS2PublishStairExecutionFrozen.inputs:messagePackage",
                    "scan_planner_msgs",
                ),
                (
                    "ROS2PublishStairExecutionFrozen.inputs:messageSubfolder",
                    "msg",
                ),
                (
                    "ROS2PublishStairExecutionFrozen.inputs:messageName",
                    "StairExecutionFreeze",
                ),
                (
                    "ROS2PublishStairExecutionFrozen.inputs:topicName",
                    cfg.stair_execution_frozen_topic,
                ),
                (
                    "ROS2PublishStairExecutionFrozen.inputs:qosProfile",
                    cfg.stair_execution_frozen_qos_profile,
                ),
                ("ROS2PublishStairExecutionFrozen.inputs:queueSize", 1),
            ]
        )
    if cfg.odometry_source == "compute":
        # setup() 会把纯字符串转换成 Isaac 5.1 要求的 usdrt.Sdf.Path target。
        values.append(("IsaacComputeOdometry.inputs:chassisPrim", cfg.robot_prim_path))

    connections: list[tuple[str, str]] = [
        ("ROS2Context.outputs:context", "ROS2PublishClock.inputs:context"),
        ("ROS2Context.outputs:context", "ROS2PublishOdometry.inputs:context"),
        ("ROS2Context.outputs:context", "ROS2PublishPointCloud.inputs:context"),
        ("StateTx.outputs:execOut", "ROS2PublishClock.inputs:execIn"),
        ("CloudTx.outputs:execOut", "ROS2PublishPointCloud.inputs:execIn"),
    ]
    if cfg.odometry_source == "direct":
        connections.append(("StateTx.outputs:execOut", "ROS2PublishOdometry.inputs:execIn"))
    else:
        connections.extend(
            [
                ("StateTx.outputs:execOut", "IsaacComputeOdometry.inputs:execIn"),
                (
                    "IsaacComputeOdometry.outputs:execOut",
                    "ROS2PublishOdometry.inputs:execIn",
                ),
                (
                    "IsaacComputeOdometry.outputs:position",
                    "ROS2PublishOdometry.inputs:position",
                ),
                (
                    "IsaacComputeOdometry.outputs:orientation",
                    "ROS2PublishOdometry.inputs:orientation",
                ),
                (
                    "IsaacComputeOdometry.outputs:linearVelocity",
                    "ROS2PublishOdometry.inputs:linearVelocity",
                ),
                (
                    "IsaacComputeOdometry.outputs:angularVelocity",
                    "ROS2PublishOdometry.inputs:angularVelocity",
                ),
                (
                    "IsaacReadSimulationTime.outputs:simulationTime",
                    "ROS2PublishClock.inputs:timeStamp",
                ),
                (
                    "IsaacReadSimulationTime.outputs:simulationTime",
                    "ROS2PublishOdometry.inputs:timeStamp",
                ),
                (
                    "IsaacReadSimulationTime.outputs:simulationTime",
                    "ROS2PublishPointCloud.inputs:timeStamp",
                ),
            ]
        )
    if cfg.enable_command_subscription:
        connections.extend(
            [
                (
                    "ROS2Context.outputs:context",
                    "ROS2SubscribeTwist.inputs:context",
                ),
                (
                    "CommandRxTick.outputs:execOut",
                    "ROS2SubscribeTwist.inputs:execIn",
                ),
                (
                    "ROS2SubscribeTwist.outputs:execOut",
                    "CommandRxCounter.inputs:execIn",
                ),
                (
                    "ROS2Context.outputs:context",
                    "ROS2SubscribeNavigationStatus.inputs:context",
                ),
                (
                    "NavigationStatusRxTick.outputs:execOut",
                    "ROS2SubscribeNavigationStatus.inputs:execIn",
                ),
                (
                    "ROS2SubscribeNavigationStatus.outputs:execOut",
                    "NavigationStatusRxCounter.inputs:execIn",
                ),
            ]
        )
    if cfg.enable_goal_reached_subscription:
        connections.extend(
            [
                (
                    "ROS2Context.outputs:context",
                    "ROS2SubscribeGoalReached.inputs:context",
                ),
                (
                    "GoalReachedRxTick.outputs:execOut",
                    "ROS2SubscribeGoalReached.inputs:execIn",
                ),
                (
                    "ROS2SubscribeGoalReached.outputs:execOut",
                    "GoalReachedRxCounter.inputs:execIn",
                ),
            ]
        )
    if cfg.enable_controller_status_subscription:
        connections.extend(
            [
                (
                    "ROS2Context.outputs:context",
                    "ROS2SubscribeControllerStatus.inputs:context",
                ),
                (
                    "ControllerStatusRxTick.outputs:execOut",
                    "ROS2SubscribeControllerStatus.inputs:execIn",
                ),
                (
                    "ROS2SubscribeControllerStatus.outputs:execOut",
                    "ControllerStatusRxCounter.inputs:execIn",
                ),
            ]
        )
    if cfg.enable_grid_map_diagnostics_subscription:
        connections.extend(
            [
                (
                    "ROS2Context.outputs:context",
                    "ROS2SubscribeGridMapDiagnostics.inputs:context",
                ),
                (
                    "GridMapDiagnosticsRxTick.outputs:execOut",
                    "ROS2SubscribeGridMapDiagnostics.inputs:execIn",
                ),
                (
                    "ROS2SubscribeGridMapDiagnostics.outputs:execOut",
                    "GridMapDiagnosticsRxCounter.inputs:execIn",
                ),
            ]
        )
    if cfg.enable_bspline_diagnostics_subscription:
        connections.extend(
            [
                (
                    "ROS2Context.outputs:context",
                    "ROS2SubscribeBsplineDiagnostics.inputs:context",
                ),
                (
                    "BsplineDiagnosticsRxTick.outputs:execOut",
                    "ROS2SubscribeBsplineDiagnostics.inputs:execIn",
                ),
                (
                    "ROS2SubscribeBsplineDiagnostics.outputs:execOut",
                    "BsplineDiagnosticsRxCounter.inputs:execIn",
                ),
            ]
        )
    if cfg.enable_reference_path_subscription:
        connections.extend(
            [
                (
                    "ROS2Context.outputs:context",
                    "ROS2SubscribeReferencePath.inputs:context",
                ),
                (
                    "ReferencePathRxTick.outputs:execOut",
                    "ROS2SubscribeReferencePath.inputs:execIn",
                ),
                (
                    "ROS2SubscribeReferencePath.outputs:execOut",
                    "ReferencePathRxCounter.inputs:execIn",
                ),
            ]
        )
    if cfg.enable_pct_goal_publisher:
        connections.extend(
            [
                (
                    "ROS2Context.outputs:context",
                    "ROS2PublishPCTGoal.inputs:context",
                ),
                (
                    "PCTGoalTx.outputs:execOut",
                    "ROS2PublishPCTGoal.inputs:execIn",
                ),
            ]
        )
    if cfg.enable_stair_execution_frozen_publisher:
        connections.extend(
            [
                (
                    "ROS2Context.outputs:context",
                    "ROS2PublishStairExecutionFrozen.inputs:context",
                ),
                (
                    "StairExecutionFrozenTx.outputs:execOut",
                    "ROS2PublishStairExecutionFrozen.inputs:execIn",
                ),
            ]
        )

    return OgnGraphSpec(
        graph_path=cfg.graph_path,
        evaluator_name="execution",
        create_nodes=tuple(nodes),
        set_values=tuple(values),
        connections=tuple(connections),
    )


def validate_point_cloud(points: Any) -> Any:
    """校验非空 Nx3 有限浮点原始点云。"""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 项目运行环境固定包含 NumPy
        raise RuntimeError("校验点云需要 NumPy。") from exc

    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError("点云必须是形状为 (N, 3) 的二维数组。")
    if array.shape[0] == 0:
        raise ValueError("原始点云不能为空。")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("点云元素必须使用浮点 dtype。")
    if not bool(np.isfinite(array).all()):
        raise ValueError("点云不能包含 NaN 或无穷值。")
    return np.array(array, dtype=np.float32, order="C", copy=True)


def prepare_odometry_sample(
    position: Sequence[float],
    orientation_wxyz: Sequence[float],
    linear_velocity: Sequence[float],
    angular_velocity: Sequence[float],
    timestamp: float,
) -> OgnOdometrySample:
    """校验 IsaacLab 根状态，并把 WXYZ 四元数改排为 OGN 的 IJKR/XYZW。"""

    position_xyz = _finite_vector(position, 3, "position")
    orientation = _finite_vector(orientation_wxyz, 4, "orientation_wxyz")
    orientation_norm = math.sqrt(
        sum(component * component for component in orientation)
    )
    if orientation_norm <= 1.0e-12:
        raise ValueError("orientation_wxyz 不能是零四元数。")
    linear_xyz = _finite_vector(linear_velocity, 3, "linear_velocity")
    angular_xyz = _finite_vector(angular_velocity, 3, "angular_velocity")
    timestamp_value = _finite_scalar(timestamp, "timestamp")
    if timestamp_value <= 0.0:
        raise ValueError("timestamp 必须是正数。")
    w, x, y, z = (
        component / orientation_norm for component in orientation
    )
    return OgnOdometrySample(
        position=(position_xyz[0], position_xyz[1], position_xyz[2]),
        orientation_ijkr=(x, y, z, w),
        linear_velocity=(linear_xyz[0], linear_xyz[1], linear_xyz[2]),
        angular_velocity=(angular_xyz[0], angular_xyz[1], angular_xyz[2]),
        timestamp=timestamp_value,
    )


def parse_reference_path_outputs(
    header_stamp_sec: object,
    header_stamp_nanosec: object,
    header_frame_id: object,
    poses_json: object,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    str,
    int,
    int,
    str,
    float | None,
]:
    """严格解析 generic OGN subscriber 的 Path 动态输出。

    Isaac 5.1 会把固定嵌套字段展开为独立动态端口，因此 Header 分别来自
    ``header:stamp:sec``、``header:stamp:nanosec`` 和
    ``header:frame_id``；只有 PoseStamped 数组元素被编码为 JSON token。
    这里仅接受 world 坐标和有效 ROS 时间戳。零点 Path 是 PCT 用来清除旧
    代际的 tombstone；一点 Path 没有可执行几何，必须拒绝；非空 Path 至少
    包含两个有限 XYZ 点。非空 Path 的末 Pose 四元数必须能确定 world 平面
    terminal yaw；零点 tombstone 则明确返回 ``None``。Path 的 Z 保持地面
    高度语义，不在桥内叠加 ``body_height``。
    """

    frame_id = header_frame_id
    if frame_id != "world":
        raise ValueError("initial_path.header.frame_id 必须严格等于 'world'。")
    stamp_sec = _json_integer(
        header_stamp_sec,
        "initial_path.header.stamp.sec",
    )
    stamp_nanosec = _json_integer(
        header_stamp_nanosec,
        "initial_path.header.stamp.nanosec",
    )
    if stamp_sec < 0:
        raise ValueError("initial_path.header.stamp.sec 不能为负数。")
    if not 0 <= stamp_nanosec < 1_000_000_000:
        raise ValueError(
            "initial_path.header.stamp.nanosec 必须位于 [0, 1000000000)。"
        )
    if stamp_sec == 0 and stamp_nanosec == 0:
        raise ValueError("initial_path.header.stamp 必须非零。")

    if isinstance(poses_json, (str, bytes, bytearray, dict)):
        raise TypeError("initial_path.poses 必须是 PoseStamped JSON 字符串数组。")
    try:
        pose_items = tuple(poses_json)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "initial_path.poses 必须是 PoseStamped JSON 字符串数组。"
        ) from exc
    if len(pose_items) == 1:
        raise ValueError("initial_path.poses 不能只有一个路径点。")

    points: list[tuple[float, float, float]] = []
    terminal_yaw: float | None = None
    for index, raw_pose in enumerate(pose_items):
        pose_stamped = _parse_json_object(
            raw_pose,
            f"initial_path.poses[{index}]",
        )
        pose = pose_stamped.get("pose")
        if not isinstance(pose, dict):
            raise ValueError(f"initial_path.poses[{index}].pose 必须是 JSON 对象。")
        position = pose.get("position")
        if not isinstance(position, dict):
            raise ValueError(
                f"initial_path.poses[{index}].pose.position 必须是 JSON 对象。"
            )
        points.append(
            (
                _json_finite_number(
                    position.get("x"),
                    f"initial_path.poses[{index}].pose.position.x",
                ),
                _json_finite_number(
                    position.get("y"),
                    f"initial_path.poses[{index}].pose.position.y",
                ),
                _json_finite_number(
                    position.get("z"),
                    f"initial_path.poses[{index}].pose.position.z",
                ),
            )
        )
        if index == len(pose_items) - 1:
            orientation = pose.get("orientation")
            if not isinstance(orientation, dict):
                raise ValueError(
                    f"initial_path.poses[{index}].pose.orientation "
                    "必须是 JSON 对象。"
                )
            terminal_yaw = _terminal_yaw_from_json_orientation(
                orientation,
                field_name=(
                    f"initial_path.poses[{index}].pose.orientation"
                ),
            )
    points_tuple = tuple(points)
    return (
        points_tuple,
        frame_id,
        stamp_sec,
        stamp_nanosec,
        _path_points_sha256(points_tuple),
        terminal_yaw,
    )


_NAVIGATION_STATUS_OUTPUT_FIELDS = frozenset(
    {
        "header_stamp_sec",
        "header_stamp_nanosec",
        "header_frame_id",
        "status_sequence",
        "state_revision",
        "goal_id",
        "state",
        "allow_tracking_command",
        "force_zero_velocity",
        "stop_confirmed",
        "global_replan_requested",
        "global_replan_in_flight",
        "global_replan_request_id",
        "pct_plan_id",
        "active_path_stamp_sec",
        "active_path_stamp_nanosec",
        "consecutive_scan_failures",
        "stale_inputs",
        "reason",
    }
)
_NAVIGATION_STATUS_OUTPUT_PORTS = {
    "header_stamp_sec": "outputs:header:stamp:sec",
    "header_stamp_nanosec": "outputs:header:stamp:nanosec",
    "header_frame_id": "outputs:header:frame_id",
    "status_sequence": "outputs:status_sequence",
    "state_revision": "outputs:state_revision",
    "goal_id": "outputs:goal_id",
    "state": "outputs:state",
    "allow_tracking_command": "outputs:allow_tracking_command",
    "force_zero_velocity": "outputs:force_zero_velocity",
    "stop_confirmed": "outputs:stop_confirmed",
    "global_replan_requested": "outputs:global_replan_requested",
    "global_replan_in_flight": "outputs:global_replan_in_flight",
    "global_replan_request_id": "outputs:global_replan_request_id",
    "pct_plan_id": "outputs:pct_plan_id",
    "active_path_stamp_sec": "outputs:active_path_stamp:sec",
    "active_path_stamp_nanosec": "outputs:active_path_stamp:nanosec",
    "consecutive_scan_failures": "outputs:consecutive_scan_failures",
    "stale_inputs": "outputs:stale_inputs",
    "reason": "outputs:reason",
}
_NAVIGATION_STATUS_STATES = frozenset((*range(7), 255))
_NAVIGATION_STATE_TRACKING = 3


def parse_navigation_status_outputs(
    outputs: Mapping[str, object],
    *,
    source_topic: str,
    receipt_timestamp: float,
    rx_sequence: int,
) -> OgnNavigationStatusSample:
    """严格解析 ``NavigationStatus`` 并校验执行许可交叉合同。"""

    if not isinstance(outputs, Mapping):
        raise TypeError("navigation_status outputs 必须是映射。")
    if set(outputs) != _NAVIGATION_STATUS_OUTPUT_FIELDS:
        missing = sorted(_NAVIGATION_STATUS_OUTPUT_FIELDS.difference(outputs))
        extra = sorted(set(outputs).difference(_NAVIGATION_STATUS_OUTPUT_FIELDS))
        raise ValueError(
            "navigation_status outputs 字段集合不完整："
            f"missing={missing}, extra={extra}"
        )
    _validate_topic(source_topic, "navigation_status source_topic")
    timestamp = _finite_scalar(
        receipt_timestamp,
        "navigation_status receipt_timestamp",
    )
    if timestamp <= 0.0:
        raise ValueError("navigation_status receipt_timestamp 必须是正数。")
    rx = _bounded_integer(
        rx_sequence,
        "navigation_status rx_sequence",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    frame_id = outputs["header_frame_id"]
    if frame_id != "world":
        raise ValueError("navigation_status.header.frame_id 必须严格等于 'world'。")
    header_sec, header_nanosec = _ros_time_parts(
        outputs["header_stamp_sec"],
        outputs["header_stamp_nanosec"],
        "navigation_status.header.stamp",
        require_nonzero=True,
    )
    status_sequence = _bounded_integer(
        outputs["status_sequence"],
        "navigation_status.status_sequence",
        minimum=1,
        maximum=(1 << 64) - 1,
    )
    state_revision = _bounded_integer(
        outputs["state_revision"],
        "navigation_status.state_revision",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    goal_id = _bounded_integer(
        outputs["goal_id"],
        "navigation_status.goal_id",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    state = _bounded_integer(
        outputs["state"],
        "navigation_status.state",
        minimum=0,
        maximum=255,
    )
    if state not in _NAVIGATION_STATUS_STATES:
        raise ValueError("navigation_status.state 不是已定义状态。")
    allow_tracking = _strict_bool(
        outputs["allow_tracking_command"],
        "navigation_status.allow_tracking_command",
    )
    force_zero = _strict_bool(
        outputs["force_zero_velocity"],
        "navigation_status.force_zero_velocity",
    )
    stop_confirmed = _strict_bool(
        outputs["stop_confirmed"],
        "navigation_status.stop_confirmed",
    )
    replan_requested = _strict_bool(
        outputs["global_replan_requested"],
        "navigation_status.global_replan_requested",
    )
    replan_in_flight = _strict_bool(
        outputs["global_replan_in_flight"],
        "navigation_status.global_replan_in_flight",
    )
    replan_request_id = _bounded_integer(
        outputs["global_replan_request_id"],
        "navigation_status.global_replan_request_id",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    pct_plan_id = _bounded_integer(
        outputs["pct_plan_id"],
        "navigation_status.pct_plan_id",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    active_path_sec, active_path_nanosec = _ros_time_parts(
        outputs["active_path_stamp_sec"],
        outputs["active_path_stamp_nanosec"],
        "navigation_status.active_path_stamp",
    )
    consecutive_failures = _bounded_integer(
        outputs["consecutive_scan_failures"],
        "navigation_status.consecutive_scan_failures",
        minimum=0,
        maximum=(1 << 32) - 1,
    )
    raw_stale_inputs = outputs["stale_inputs"]
    if isinstance(raw_stale_inputs, (str, bytes, bytearray, Mapping)):
        raise TypeError("navigation_status.stale_inputs 必须是字符串数组。")
    try:
        stale_inputs = tuple(raw_stale_inputs)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("navigation_status.stale_inputs 必须是字符串数组。") from exc
    if any(not isinstance(item, str) or not item.strip() for item in stale_inputs):
        raise ValueError("navigation_status.stale_inputs 只能包含非空字符串。")
    if len(set(stale_inputs)) != len(stale_inputs):
        raise ValueError("navigation_status.stale_inputs 不能包含重复项。")
    reason = outputs["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("navigation_status.reason 必须是非空字符串。")

    active_path_stamp_ns = active_path_sec * 1_000_000_000 + active_path_nanosec
    if allow_tracking == force_zero:
        raise ValueError("allow_tracking_command 与 force_zero_velocity 必须互反。")
    if allow_tracking and (
        state != _NAVIGATION_STATE_TRACKING
        or goal_id <= 0
        or pct_plan_id <= 0
        or active_path_stamp_ns <= 0
        or stale_inputs
        or replan_requested
        or replan_in_flight
    ):
        raise ValueError("允许跟踪的 NavigationStatus 含有拒绝态语义。")
    if replan_in_flight and not replan_requested:
        raise ValueError("global_replan_in_flight 要求已请求全局重规划。")
    if (replan_requested or replan_in_flight) and replan_request_id <= 0:
        raise ValueError("全局重规划状态要求非零 request_id。")

    return OgnNavigationStatusSample(
        source_topic=source_topic,
        receipt_timestamp=timestamp,
        rx_sequence=rx,
        frame_id=frame_id,
        header_stamp_sec=header_sec,
        header_stamp_nanosec=header_nanosec,
        status_sequence=status_sequence,
        state_revision=state_revision,
        goal_id=goal_id,
        state=state,
        allow_tracking_command=allow_tracking,
        force_zero_velocity=force_zero,
        stop_confirmed=stop_confirmed,
        global_replan_requested=replan_requested,
        global_replan_in_flight=replan_in_flight,
        global_replan_request_id=replan_request_id,
        pct_plan_id=pct_plan_id,
        active_path_stamp_sec=active_path_sec,
        active_path_stamp_nanosec=active_path_nanosec,
        consecutive_scan_failures=consecutive_failures,
        stale_inputs=stale_inputs,
        reason=reason.strip(),
    )


_CONTROLLER_STATUS_OUTPUT_FIELDS = frozenset(
    {
        "header_stamp_sec",
        "header_stamp_nanosec",
        "header_frame_id",
        "status_sequence",
        "acceptance_sequence",
        "event",
        "reference_path_stamp_sec",
        "reference_path_stamp_nanosec",
        "bspline_header_stamp_sec",
        "bspline_header_stamp_nanosec",
        "start_time_sec",
        "start_time_nanosec",
        "traj_id",
        "accepted",
        "trajectory_valid",
        "is_final",
        "emergency_stop",
        "state",
        "reason",
        "candidate_present",
        "candidate_reference_path_stamp_sec",
        "candidate_reference_path_stamp_nanosec",
        "candidate_bspline_header_stamp_sec",
        "candidate_bspline_header_stamp_nanosec",
        "candidate_start_time_sec",
        "candidate_start_time_nanosec",
        "candidate_traj_id",
        "active_sensing_yaw_only",
        "command_sample_count",
        "first_command_linear_x",
        "first_command_linear_y",
        "first_command_linear_z",
        "first_command_angular_x",
        "first_command_angular_y",
        "first_command_angular_z",
        "max_abs_vx",
        "max_abs_vy",
        "max_abs_wz",
        "command_violation_count",
    }
)
_CONTROLLER_STATUS_OUTPUT_PORTS = {
    "header_stamp_sec": "outputs:header:stamp:sec",
    "header_stamp_nanosec": "outputs:header:stamp:nanosec",
    "header_frame_id": "outputs:header:frame_id",
    "status_sequence": "outputs:status_sequence",
    "acceptance_sequence": "outputs:acceptance_sequence",
    "event": "outputs:event",
    "reference_path_stamp_sec": "outputs:reference_path_stamp:sec",
    "reference_path_stamp_nanosec": "outputs:reference_path_stamp:nanosec",
    "bspline_header_stamp_sec": "outputs:bspline_header_stamp:sec",
    "bspline_header_stamp_nanosec": "outputs:bspline_header_stamp:nanosec",
    "start_time_sec": "outputs:start_time:sec",
    "start_time_nanosec": "outputs:start_time:nanosec",
    "traj_id": "outputs:traj_id",
    "accepted": "outputs:accepted",
    "trajectory_valid": "outputs:trajectory_valid",
    "is_final": "outputs:is_final",
    "emergency_stop": "outputs:emergency_stop",
    "state": "outputs:state",
    "reason": "outputs:reason",
    "candidate_present": "outputs:candidate_present",
    "candidate_reference_path_stamp_sec": (
        "outputs:candidate_reference_path_stamp:sec"
    ),
    "candidate_reference_path_stamp_nanosec": (
        "outputs:candidate_reference_path_stamp:nanosec"
    ),
    "candidate_bspline_header_stamp_sec": (
        "outputs:candidate_bspline_header_stamp:sec"
    ),
    "candidate_bspline_header_stamp_nanosec": (
        "outputs:candidate_bspline_header_stamp:nanosec"
    ),
    "candidate_start_time_sec": "outputs:candidate_start_time:sec",
    "candidate_start_time_nanosec": "outputs:candidate_start_time:nanosec",
    "candidate_traj_id": "outputs:candidate_traj_id",
    "active_sensing_yaw_only": "outputs:active_sensing_yaw_only",
    "command_sample_count": "outputs:command_sample_count",
    "first_command_linear_x": "outputs:first_command:linear:x",
    "first_command_linear_y": "outputs:first_command:linear:y",
    "first_command_linear_z": "outputs:first_command:linear:z",
    "first_command_angular_x": "outputs:first_command:angular:x",
    "first_command_angular_y": "outputs:first_command:angular:y",
    "first_command_angular_z": "outputs:first_command:angular:z",
    "max_abs_vx": "outputs:max_abs_vx",
    "max_abs_vy": "outputs:max_abs_vy",
    "max_abs_wz": "outputs:max_abs_wz",
    "command_violation_count": "outputs:command_violation_count",
}
_CONTROLLER_STATUS_EVENTS = frozenset(range(6))
_CONTROLLER_STATUS_STATES = frozenset((*range(13), 255))
_CONTROLLER_EVENT_INITIAL = 0
_CONTROLLER_EVENT_ACCEPTED = 1
_CONTROLLER_EVENT_REJECTED = 2
_CONTROLLER_EVENT_INVALIDATED = 3
_CONTROLLER_EVENT_STATE_CHANGED = 4
_CONTROLLER_EVENT_DUPLICATE = 5


def parse_controller_status_outputs(
    outputs: Mapping[str, object],
    *,
    source_topic: str,
    receipt_timestamp: float,
    rx_sequence: int,
) -> OgnControllerStatusSample:
    """严格解析 generic OGN subscriber 的 ``ControllerStatus`` 输出。

    当前已接受轨迹与被拒候选使用不同 identity；本函数同时校验事件、布尔
    快照和两组 identity 的交叉合同，避免候选包伪装成可恢复轨迹。所有 ROS
    时间戳始终保留 ``sec/nanosec``，不经过浮点秒转换。
    """

    if not isinstance(outputs, Mapping):
        raise TypeError("controller_status outputs 必须是映射。")
    if set(outputs) != _CONTROLLER_STATUS_OUTPUT_FIELDS:
        missing = sorted(_CONTROLLER_STATUS_OUTPUT_FIELDS.difference(outputs))
        extra = sorted(set(outputs).difference(_CONTROLLER_STATUS_OUTPUT_FIELDS))
        raise ValueError(
            "controller_status outputs 字段集合不完整："
            f"missing={missing}, extra={extra}"
        )
    _validate_topic(source_topic, "controller_status source_topic")
    timestamp = _finite_scalar(receipt_timestamp, "controller_status receipt_timestamp")
    if timestamp <= 0.0:
        raise ValueError("controller_status receipt_timestamp 必须是正数。")
    rx = _bounded_integer(
        rx_sequence,
        "controller_status rx_sequence",
        minimum=0,
        maximum=(1 << 64) - 1,
    )

    frame_id = outputs["header_frame_id"]
    if frame_id != "world":
        raise ValueError("controller_status.header.frame_id 必须严格等于 'world'。")
    header_sec, header_nanosec = _ros_time_parts(
        outputs["header_stamp_sec"],
        outputs["header_stamp_nanosec"],
        "controller_status.header.stamp",
        require_nonzero=True,
    )
    status_sequence = _bounded_integer(
        outputs["status_sequence"],
        "controller_status.status_sequence",
        minimum=1,
        maximum=(1 << 64) - 1,
    )
    acceptance_sequence = _bounded_integer(
        outputs["acceptance_sequence"],
        "controller_status.acceptance_sequence",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    if acceptance_sequence > status_sequence:
        raise ValueError(
            "controller_status.acceptance_sequence 不能大于 status_sequence。"
        )
    event = _bounded_integer(
        outputs["event"],
        "controller_status.event",
        minimum=0,
        maximum=255,
    )
    if event not in _CONTROLLER_STATUS_EVENTS:
        raise ValueError("controller_status.event 不是已定义事件。")
    state = _bounded_integer(
        outputs["state"],
        "controller_status.state",
        minimum=0,
        maximum=255,
    )
    if state not in _CONTROLLER_STATUS_STATES:
        raise ValueError("controller_status.state 不是已定义状态。")

    reference_path_sec, reference_path_nanosec = _ros_time_parts(
        outputs["reference_path_stamp_sec"],
        outputs["reference_path_stamp_nanosec"],
        "controller_status.reference_path_stamp",
    )
    bspline_header_sec, bspline_header_nanosec = _ros_time_parts(
        outputs["bspline_header_stamp_sec"],
        outputs["bspline_header_stamp_nanosec"],
        "controller_status.bspline_header_stamp",
    )
    start_sec, start_nanosec = _ros_time_parts(
        outputs["start_time_sec"],
        outputs["start_time_nanosec"],
        "controller_status.start_time",
    )
    traj_id = _bounded_integer(
        outputs["traj_id"],
        "controller_status.traj_id",
        minimum=-(1 << 63),
        maximum=(1 << 63) - 1,
    )
    accepted = _strict_bool(outputs["accepted"], "controller_status.accepted")
    trajectory_valid = _strict_bool(
        outputs["trajectory_valid"],
        "controller_status.trajectory_valid",
    )
    is_final = _strict_bool(outputs["is_final"], "controller_status.is_final")
    emergency_stop = _strict_bool(
        outputs["emergency_stop"],
        "controller_status.emergency_stop",
    )
    reason = outputs["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("controller_status.reason 必须是非空字符串。")
    candidate_present = _strict_bool(
        outputs["candidate_present"],
        "controller_status.candidate_present",
    )
    candidate_reference_sec, candidate_reference_nanosec = _ros_time_parts(
        outputs["candidate_reference_path_stamp_sec"],
        outputs["candidate_reference_path_stamp_nanosec"],
        "controller_status.candidate_reference_path_stamp",
    )
    candidate_header_sec, candidate_header_nanosec = _ros_time_parts(
        outputs["candidate_bspline_header_stamp_sec"],
        outputs["candidate_bspline_header_stamp_nanosec"],
        "controller_status.candidate_bspline_header_stamp",
    )
    candidate_start_sec, candidate_start_nanosec = _ros_time_parts(
        outputs["candidate_start_time_sec"],
        outputs["candidate_start_time_nanosec"],
        "controller_status.candidate_start_time",
    )
    candidate_traj_id = _bounded_integer(
        outputs["candidate_traj_id"],
        "controller_status.candidate_traj_id",
        minimum=-(1 << 63),
        maximum=(1 << 63) - 1,
    )
    active_sensing_yaw_only = _strict_bool(
        outputs["active_sensing_yaw_only"],
        "controller_status.active_sensing_yaw_only",
    )
    command_sample_count = _bounded_integer(
        outputs["command_sample_count"],
        "controller_status.command_sample_count",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    first_command = tuple(
        _finite_scalar(
            outputs[field_name],
            f"controller_status.{field_name}",
        )
        for field_name in (
            "first_command_linear_x",
            "first_command_linear_y",
            "first_command_linear_z",
            "first_command_angular_x",
            "first_command_angular_y",
            "first_command_angular_z",
        )
    )
    max_abs_vx = _finite_scalar(
        outputs["max_abs_vx"], "controller_status.max_abs_vx"
    )
    max_abs_vy = _finite_scalar(
        outputs["max_abs_vy"], "controller_status.max_abs_vy"
    )
    max_abs_wz = _finite_scalar(
        outputs["max_abs_wz"], "controller_status.max_abs_wz"
    )
    if min(max_abs_vx, max_abs_vy, max_abs_wz) < 0.0:
        raise ValueError("controller_status command max_abs 不能为负数。")
    command_violation_count = _bounded_integer(
        outputs["command_violation_count"],
        "controller_status.command_violation_count",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    if command_violation_count > command_sample_count:
        raise ValueError("controller_status violation_count 不能超过 sample_count。")
    if command_sample_count == 0:
        if any(value != 0.0 for value in first_command) or any(
            value != 0.0 for value in (max_abs_vx, max_abs_vy, max_abs_wz)
        ) or command_violation_count != 0:
            raise ValueError("零 command_sample_count 要求命令聚合全部为默认值。")
    elif (
        abs(first_command[0]) > max_abs_vx + 1.0e-12
        or abs(first_command[1]) > max_abs_vy + 1.0e-12
        or abs(first_command[5]) > max_abs_wz + 1.0e-12
    ):
        raise ValueError("first_command 超出 controller_status max_abs 聚合。")
    if any(first_command[index] != 0.0 for index in (2, 3, 4)):
        raise ValueError("controller_status first_command 非平面轴必须严格为零。")

    current_stamp_ns = (
        reference_path_sec * 1_000_000_000 + reference_path_nanosec,
        bspline_header_sec * 1_000_000_000 + bspline_header_nanosec,
        start_sec * 1_000_000_000 + start_nanosec,
    )
    if trajectory_valid and not accepted:
        raise ValueError("trajectory_valid=true 要求 accepted=true。")
    if accepted:
        if acceptance_sequence < 1 or any(stamp <= 0 for stamp in current_stamp_ns):
            raise ValueError("accepted=true 要求非零 identity 与 acceptance_sequence。")
    elif (
        acceptance_sequence != 0
        or any(stamp != 0 for stamp in current_stamp_ns)
        or traj_id != 0
        or trajectory_valid
        or is_final
        or emergency_stop
        or active_sensing_yaw_only
        or command_sample_count != 0
    ):
        raise ValueError(
            "accepted=false 时当前 identity 与轨迹语义必须全部为默认值。"
        )
    if active_sensing_yaw_only:
        if is_final or emergency_stop:
            raise ValueError("主动感知 yaw-only 轨迹不能是 final/emergency_stop。")
        if command_sample_count < 1:
            raise ValueError("主动感知 accepted 状态必须包含首条零命令。")
        if any(value != 0.0 for value in first_command):
            raise ValueError("主动感知 first_command 必须严格为零 Twist。")
        if max_abs_vx != 0.0 or max_abs_vy != 0.0:
            raise ValueError("主动感知实际命令的 vx/vy 必须严格为零。")
        if max_abs_wz > 0.20 + 1.0e-12:
            raise ValueError("主动感知实际命令的 |wz| 不得超过 0.20 rad/s。")

    candidate_stamp_ns = (
        candidate_reference_sec * 1_000_000_000 + candidate_reference_nanosec,
        candidate_header_sec * 1_000_000_000 + candidate_header_nanosec,
        candidate_start_sec * 1_000_000_000 + candidate_start_nanosec,
    )
    if not candidate_present and (
        any(stamp != 0 for stamp in candidate_stamp_ns) or candidate_traj_id != 0
    ):
        raise ValueError("candidate_present=false 时候选 identity 必须全部为默认值。")
    if event == _CONTROLLER_EVENT_REJECTED:
        if not candidate_present:
            raise ValueError("EVENT_REJECTED 必须提供独立 candidate identity。")
    elif candidate_present:
        raise ValueError("只有 EVENT_REJECTED 可以携带 candidate identity。")
    if event in {_CONTROLLER_EVENT_ACCEPTED, _CONTROLLER_EVENT_DUPLICATE} and (
        not accepted or not trajectory_valid
    ):
        raise ValueError("ACCEPTED/DUPLICATE 事件要求当前轨迹已接受且有效。")
    if event == _CONTROLLER_EVENT_INVALIDATED and (
        not accepted or trajectory_valid
    ):
        raise ValueError("EVENT_INVALIDATED 要求已接受轨迹处于无效状态。")
    if event in {_CONTROLLER_EVENT_INITIAL, _CONTROLLER_EVENT_STATE_CHANGED} and (
        candidate_present
    ):
        raise ValueError("INITIAL/STATE_CHANGED 事件不能携带候选 identity。")

    return OgnControllerStatusSample(
        source_topic=source_topic,
        receipt_timestamp=timestamp,
        rx_sequence=rx,
        frame_id=frame_id,
        header_stamp_sec=header_sec,
        header_stamp_nanosec=header_nanosec,
        status_sequence=status_sequence,
        acceptance_sequence=acceptance_sequence,
        event=event,
        reference_path_stamp_sec=reference_path_sec,
        reference_path_stamp_nanosec=reference_path_nanosec,
        bspline_header_stamp_sec=bspline_header_sec,
        bspline_header_stamp_nanosec=bspline_header_nanosec,
        start_time_sec=start_sec,
        start_time_nanosec=start_nanosec,
        traj_id=traj_id,
        accepted=accepted,
        trajectory_valid=trajectory_valid,
        is_final=is_final,
        emergency_stop=emergency_stop,
        state=state,
        reason=reason,
        candidate_present=candidate_present,
        candidate_reference_path_stamp_sec=candidate_reference_sec,
        candidate_reference_path_stamp_nanosec=candidate_reference_nanosec,
        candidate_bspline_header_stamp_sec=candidate_header_sec,
        candidate_bspline_header_stamp_nanosec=candidate_header_nanosec,
        candidate_start_time_sec=candidate_start_sec,
        candidate_start_time_nanosec=candidate_start_nanosec,
        candidate_traj_id=candidate_traj_id,
        active_sensing_yaw_only=active_sensing_yaw_only,
        command_sample_count=command_sample_count,
        first_command=first_command,
        max_abs_vx=max_abs_vx,
        max_abs_vy=max_abs_vy,
        max_abs_wz=max_abs_wz,
        command_violation_count=command_violation_count,
    )


_GRID_MAP_DIAGNOSTICS_OUTPUT_PORTS = {
    "header_stamp_sec": "outputs:header:stamp:sec",
    "header_stamp_nanosec": "outputs:header:stamp:nanosec",
    "header_frame_id": "outputs:header:frame_id",
    "observation_sequence": "outputs:observation_sequence",
    "sensor_pose_stamp_sec": "outputs:sensor_pose_stamp:sec",
    "sensor_pose_stamp_nanosec": "outputs:sensor_pose_stamp:nanosec",
    "sensor_origin_x": "outputs:sensor_origin:x",
    "sensor_origin_y": "outputs:sensor_origin:y",
    "sensor_origin_z": "outputs:sensor_origin:z",
    "canonical_empty": "outputs:canonical_empty",
    "map_fusion_performed": "outputs:map_fusion_performed",
    "map_resolution": "outputs:map_resolution",
    "input_point_count": "outputs:input_point_count",
    "accepted_endpoint_count": "outputs:accepted_endpoint_count",
    "hit_endpoint_count": "outputs:hit_endpoint_count",
    "explicit_free_endpoint_count": "outputs:explicit_free_endpoint_count",
    "hit_endpoint_samples_truncated": (
        "outputs:hit_endpoint_samples_truncated"
    ),
    "hit_endpoint_samples": "outputs:hit_endpoint_samples",
    "hit_endpoint_sample_voxel_indices_xyz": (
        "outputs:hit_endpoint_sample_voxel_indices_xyz"
    ),
    "free_to_occupied_transition_count": (
        "outputs:free_to_occupied_transition_count"
    ),
    "free_to_occupied_transition_samples_truncated": (
        "outputs:free_to_occupied_transition_samples_truncated"
    ),
    "free_to_occupied_transition_hit_samples": (
        "outputs:free_to_occupied_transition_hit_samples"
    ),
    "free_to_occupied_transition_voxel_indices_xyz": (
        "outputs:free_to_occupied_transition_voxel_indices_xyz"
    ),
    "explicit_free_miss_voxel_count": (
        "outputs:explicit_free_miss_voxel_count"
    ),
    "occupied_to_free_by_explicit_miss_count": (
        "outputs:occupied_to_free_by_explicit_miss_count"
    ),
    "occupied_to_free_samples_truncated": (
        "outputs:occupied_to_free_samples_truncated"
    ),
    "occupied_to_free_by_explicit_miss_samples": (
        "outputs:occupied_to_free_by_explicit_miss_samples"
    ),
    "occupied_to_free_sample_voxel_indices_xyz": (
        "outputs:occupied_to_free_sample_voxel_indices_xyz"
    ),
    "occupied_to_free_transition_hit_observation_sequences": (
        "outputs:occupied_to_free_transition_hit_observation_sequences"
    ),
    "occupied_to_free_transition_hit_samples": (
        "outputs:occupied_to_free_transition_hit_samples"
    ),
    "occupied_to_free_transition_hit_header_stamp_ns": (
        "outputs:occupied_to_free_transition_hit_header_stamp_ns"
    ),
    "occupied_removed_by_sliding_reset_count": (
        "outputs:occupied_removed_by_sliding_reset_count"
    ),
}

_BSPLINE_DIAGNOSTICS_OUTPUT_PORTS = {
    "header_stamp_sec": "outputs:header:stamp:sec",
    "header_stamp_nanosec": "outputs:header:stamp:nanosec",
    "header_frame_id": "outputs:header:frame_id",
    "diagnostic_sequence": "outputs:diagnostic_sequence",
    "start_time_sec": "outputs:start_time:sec",
    "start_time_nanosec": "outputs:start_time:nanosec",
    "reference_path_stamp_sec": "outputs:reference_path_stamp:sec",
    "reference_path_stamp_nanosec": (
        "outputs:reference_path_stamp:nanosec"
    ),
    "traj_id": "outputs:traj_id",
    "is_final": "outputs:is_final",
    "emergency_stop": "outputs:emergency_stop",
    "stationary": "outputs:stationary",
    "ordered_reference_checked": "outputs:ordered_reference_checked",
    "ordered_reference_safe": "outputs:ordered_reference_safe",
    "maximum_trajectory_deviation": (
        "outputs:maximum_trajectory_deviation"
    ),
    "maximum_guide_anchor_deviation": (
        "outputs:maximum_guide_anchor_deviation"
    ),
    "maximum_guide_progress_lead": (
        "outputs:maximum_guide_progress_lead"
    ),
    "maximum_deviation_limit": "outputs:maximum_deviation_limit",
    "maximum_progress_lead_limit": (
        "outputs:maximum_progress_lead_limit"
    ),
    "trajectory_duration": "outputs:trajectory_duration",
    "maximum_velocity_upper_bound": (
        "outputs:maximum_velocity_upper_bound"
    ),
    "double_cylinder_radius": "outputs:double_cylinder_radius",
    "double_cylinder_offset": "outputs:double_cylinder_offset",
    "trajectory_sample_count_total": (
        "outputs:trajectory_sample_count_total"
    ),
    "trajectory_samples_truncated": "outputs:trajectory_samples_truncated",
    "trajectory_samples": "outputs:trajectory_samples",
    "ordered_reference_sample_count_total": (
        "outputs:ordered_reference_sample_count_total"
    ),
    "ordered_reference_samples_truncated": (
        "outputs:ordered_reference_samples_truncated"
    ),
    "ordered_reference_samples": "outputs:ordered_reference_samples",
    "active_sensing": "outputs:active_sensing",
    "active_sensing_event": "outputs:active_sensing_event",
    "active_sensing_start_yaw": "outputs:active_sensing_start_yaw",
    "active_sensing_target_yaw": "outputs:active_sensing_target_yaw",
    "active_sensing_yaw_offset": "outputs:active_sensing_yaw_offset",
    "active_sensing_yaw_rate": "outputs:active_sensing_yaw_rate",
    "active_sensing_settle_stamp_sec": (
        "outputs:active_sensing_settle_stamp:sec"
    ),
    "active_sensing_settle_stamp_nanosec": (
        "outputs:active_sensing_settle_stamp:nanosec"
    ),
    "active_sensing_settle_yaw_error": (
        "outputs:active_sensing_settle_yaw_error"
    ),
    "active_sensing_settle_angular_speed": (
        "outputs:active_sensing_settle_angular_speed"
    ),
    "active_sensing_stable_duration": (
        "outputs:active_sensing_stable_duration"
    ),
    "active_sensing_fusion_baseline": (
        "outputs:active_sensing_fusion_baseline"
    ),
    "active_sensing_fusion_current": "outputs:active_sensing_fusion_current",
    "active_sensing_fusion_distinct": (
        "outputs:active_sensing_fusion_distinct"
    ),
    "active_sensing_fusion_required": (
        "outputs:active_sensing_fusion_required"
    ),
    "active_sensing_completed": "outputs:active_sensing_completed",
    "active_sensing_failed": "outputs:active_sensing_failed",
    "active_sensing_reason": "outputs:active_sensing_reason",
}


def _parse_bounded_point_samples(
    raw_samples: object,
    field_name: str,
) -> tuple[tuple[float, float, float], ...]:
    """解析 generic subscriber 输出的 bounded geometry_msgs/Point 数组。"""

    if isinstance(raw_samples, (str, bytes, bytearray, dict)):
        raise TypeError(f"{field_name} 必须是 Point JSON 字符串数组。")
    try:
        samples = tuple(raw_samples)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{field_name} 必须是 Point JSON 字符串数组。"
        ) from exc
    if len(samples) > 64:
        raise ValueError(f"{field_name} 不能超过 64 个有界样本。")
    points: list[tuple[float, float, float]] = []
    for index, raw_sample in enumerate(samples):
        point = _parse_json_object(raw_sample, f"{field_name}[{index}]")
        if set(point) != {"x", "y", "z"}:
            raise ValueError(f"{field_name}[{index}] 必须只包含 x/y/z。")
        points.append(
            tuple(
                _json_finite_number(
                    point[axis],
                    f"{field_name}[{index}].{axis}",
                )
                for axis in ("x", "y", "z")
            )
        )
    return tuple(points)


def _parse_voxel_index_triples(
    raw_values: object,
    field_name: str,
    *,
    sample_count: int,
) -> tuple[tuple[int, int, int], ...]:
    """解析 C++ 直接发布的 canonical global voxel index 三元组。"""

    if isinstance(raw_values, (str, bytes, bytearray, dict)):
        raise TypeError(f"{field_name} 必须是 int64 数组。")
    try:
        values = tuple(raw_values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} 必须是 int64 数组。") from exc
    if len(values) != 3 * sample_count:
        raise ValueError(f"{field_name} 必须与 Point 样本逐点三轴对齐。")
    parsed = tuple(
        _bounded_integer(
            value,
            f"{field_name}[{index}]",
            minimum=-(1 << 63),
            maximum=(1 << 63) - 1,
        )
        for index, value in enumerate(values)
    )
    return tuple(
        (parsed[index], parsed[index + 1], parsed[index + 2])
        for index in range(0, len(parsed), 3)
    )


def _parse_transition_sequences(
    raw_values: object,
    field_name: str,
    *,
    sample_count: int,
    current_observation_sequence: int,
) -> tuple[int, ...]:
    """解析 clear 样本对应的占据阈值穿越来源 sequence。"""

    if isinstance(raw_values, (str, bytes, bytearray, dict)):
        raise TypeError(f"{field_name} 必须是 uint64 数组。")
    try:
        values = tuple(raw_values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} 必须是 uint64 数组。") from exc
    if len(values) != sample_count:
        raise ValueError(f"{field_name} 必须与 clear Point 样本逐点对齐。")
    sequences = tuple(
        _bounded_integer(
            value,
            f"{field_name}[{index}]",
            minimum=0,
            maximum=(1 << 64) - 1,
        )
        for index, value in enumerate(values)
    )
    if any(
        sequence != 0 and sequence >= current_observation_sequence
        for sequence in sequences
    ):
        raise ValueError("clear provenance 必须引用严格更早的 hit observation。")
    return sequences


def _parse_transition_hit_header_stamps(
    raw_values: object,
    field_name: str,
    *,
    sample_count: int,
    current_header_stamp_ns: int,
) -> tuple[int, ...]:
    """解析 occupied epoch 来源 hit 的 ROS Header 纳秒值。"""

    if isinstance(raw_values, (str, bytes, bytearray, dict)):
        raise TypeError(f"{field_name} 必须是 int64 数组。")
    try:
        values = tuple(raw_values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} 必须是 int64 数组。") from exc
    if len(values) != sample_count:
        raise ValueError(f"{field_name} 必须与 clear Point 样本逐点对齐。")
    stamps = tuple(
        _bounded_integer(
            value,
            f"{field_name}[{index}]",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        for index, value in enumerate(values)
    )
    if any(
        stamp != 0 and stamp >= current_header_stamp_ns
        for stamp in stamps
    ):
        raise ValueError("clear provenance hit stamp 必须严格早于 clear Header。")
    return stamps


def _validate_bounded_sample_count(
    *,
    total_count: int,
    samples: tuple[tuple[float, float, float], ...],
    truncated: bool,
    field_name: str,
) -> None:
    """交叉校验总数、64 点数组和截断标志，禁止不完整样本冒充全量。"""

    expected_sample_count = min(total_count, 64)
    if len(samples) != expected_sample_count:
        raise ValueError(
            f"{field_name} 必须保留 min(total_count, 64) 个样本。"
        )
    expected_truncated = total_count > 64
    if truncated != expected_truncated:
        raise ValueError(f"{field_name} 截断标志必须严格表示 total_count > 64。")


def parse_grid_map_observation_diagnostics_outputs(
    outputs: Mapping[str, object],
    *,
    source_topic: str,
    receipt_timestamp: float,
    rx_sequence: int,
) -> OgnGridMapObservationDiagnosticsSample:
    """严格解析 GridMap 对过滤后点云和 explicit-free p_miss 的证据。"""

    expected_fields = frozenset(_GRID_MAP_DIAGNOSTICS_OUTPUT_PORTS)
    if not isinstance(outputs, Mapping) or set(outputs) != expected_fields:
        raise ValueError("grid_map diagnostics outputs 字段集合不完整。")
    _validate_topic(source_topic, "grid_map diagnostics source_topic")
    timestamp = _finite_scalar(receipt_timestamp, "grid_map receipt_timestamp")
    if timestamp <= 0.0:
        raise ValueError("grid_map receipt_timestamp 必须是正数。")
    rx = _bounded_integer(
        rx_sequence,
        "grid_map rx_sequence",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    if outputs["header_frame_id"] != "world":
        raise ValueError("grid_map diagnostics frame 必须严格等于 world。")
    header_sec, header_nanosec = _ros_time_parts(
        outputs["header_stamp_sec"],
        outputs["header_stamp_nanosec"],
        "grid_map.header.stamp",
        require_nonzero=True,
    )
    pose_sec, pose_nanosec = _ros_time_parts(
        outputs["sensor_pose_stamp_sec"],
        outputs["sensor_pose_stamp_nanosec"],
        "grid_map.sensor_pose_stamp",
        require_nonzero=True,
    )
    observation_sequence = _bounded_integer(
        outputs["observation_sequence"],
        "grid_map.observation_sequence",
        minimum=1,
        maximum=(1 << 64) - 1,
    )
    sensor_origin = tuple(
        _finite_scalar(outputs[f"sensor_origin_{axis}"], f"sensor_origin.{axis}")
        for axis in ("x", "y", "z")
    )
    canonical_empty = _strict_bool(
        outputs["canonical_empty"],
        "grid_map.canonical_empty",
    )
    fusion = _strict_bool(
        outputs["map_fusion_performed"],
        "grid_map.map_fusion_performed",
    )
    map_resolution = _finite_scalar(
        outputs["map_resolution"],
        "grid_map.map_resolution",
    )
    if map_resolution <= 0.0:
        raise ValueError("grid_map.map_resolution 必须是有限正数。")
    count_names = (
        "input_point_count",
        "accepted_endpoint_count",
        "hit_endpoint_count",
        "explicit_free_endpoint_count",
        "free_to_occupied_transition_count",
        "explicit_free_miss_voxel_count",
        "occupied_to_free_by_explicit_miss_count",
        "occupied_removed_by_sliding_reset_count",
    )
    counts = {
        name: _bounded_integer(
            outputs[name],
            f"grid_map.{name}",
            minimum=0,
            maximum=(1 << 32) - 1,
        )
        for name in count_names
    }
    hit_truncated = _strict_bool(
        outputs["hit_endpoint_samples_truncated"],
        "grid_map.hit_endpoint_samples_truncated",
    )
    hit_samples = _parse_bounded_point_samples(
        outputs["hit_endpoint_samples"],
        "grid_map.hit_endpoint_samples",
    )
    hit_voxel_indices = _parse_voxel_index_triples(
        outputs["hit_endpoint_sample_voxel_indices_xyz"],
        "grid_map.hit_endpoint_sample_voxel_indices_xyz",
        sample_count=len(hit_samples),
    )
    transition_truncated = _strict_bool(
        outputs["free_to_occupied_transition_samples_truncated"],
        "grid_map.free_to_occupied_transition_samples_truncated",
    )
    transition_hit_samples = _parse_bounded_point_samples(
        outputs["free_to_occupied_transition_hit_samples"],
        "grid_map.free_to_occupied_transition_hit_samples",
    )
    transition_voxel_indices = _parse_voxel_index_triples(
        outputs["free_to_occupied_transition_voxel_indices_xyz"],
        "grid_map.free_to_occupied_transition_voxel_indices_xyz",
        sample_count=len(transition_hit_samples),
    )
    clear_truncated = _strict_bool(
        outputs["occupied_to_free_samples_truncated"],
        "grid_map.occupied_to_free_samples_truncated",
    )
    clear_samples = _parse_bounded_point_samples(
        outputs["occupied_to_free_by_explicit_miss_samples"],
        "grid_map.occupied_to_free_by_explicit_miss_samples",
    )
    clear_voxel_indices = _parse_voxel_index_triples(
        outputs["occupied_to_free_sample_voxel_indices_xyz"],
        "grid_map.occupied_to_free_sample_voxel_indices_xyz",
        sample_count=len(clear_samples),
    )
    transition_hit_sequences = _parse_transition_sequences(
        outputs[
            "occupied_to_free_transition_hit_observation_sequences"
        ],
        "grid_map.occupied_to_free_transition_hit_observation_sequences",
        sample_count=len(clear_samples),
        current_observation_sequence=observation_sequence,
    )
    clear_transition_hit_samples = _parse_bounded_point_samples(
        outputs["occupied_to_free_transition_hit_samples"],
        "grid_map.occupied_to_free_transition_hit_samples",
    )
    if len(clear_transition_hit_samples) != len(clear_samples):
        raise ValueError(
            "occupied→free 来源 hit samples 必须与 clear samples 对齐。"
        )
    header_stamp_ns = header_sec * 1_000_000_000 + header_nanosec
    clear_transition_hit_stamps = _parse_transition_hit_header_stamps(
        outputs["occupied_to_free_transition_hit_header_stamp_ns"],
        "grid_map.occupied_to_free_transition_hit_header_stamp_ns",
        sample_count=len(clear_samples),
        current_header_stamp_ns=header_stamp_ns,
    )
    if any(
        (sequence == 0) != (stamp == 0)
        for sequence, stamp in zip(
            transition_hit_sequences,
            clear_transition_hit_stamps,
            strict=True,
        )
    ):
        raise ValueError("clear provenance sequence 与 hit stamp 的有效性不一致。")
    _validate_bounded_sample_count(
        total_count=counts["hit_endpoint_count"],
        samples=hit_samples,
        truncated=hit_truncated,
        field_name="grid_map.hit_endpoint_samples",
    )
    _validate_bounded_sample_count(
        total_count=counts["free_to_occupied_transition_count"],
        samples=transition_hit_samples,
        truncated=transition_truncated,
        field_name="grid_map.free_to_occupied_transition_hit_samples",
    )
    if counts["free_to_occupied_transition_count"] > counts["hit_endpoint_count"]:
        raise ValueError("free→occupied transition 数不能超过 hit endpoint 数。")
    _validate_bounded_sample_count(
        total_count=counts["occupied_to_free_by_explicit_miss_count"],
        samples=clear_samples,
        truncated=clear_truncated,
        field_name="grid_map.occupied_to_free_samples",
    )
    if counts["accepted_endpoint_count"] > counts["input_point_count"]:
        raise ValueError("GridMap 接纳端点不能超过输入点数。")
    if (
        counts["hit_endpoint_count"]
        + counts["explicit_free_endpoint_count"]
        != counts["accepted_endpoint_count"]
    ):
        raise ValueError("GridMap hit/free 端点数与接纳总数不一致。")
    if (
        counts["occupied_to_free_by_explicit_miss_count"]
        > counts["explicit_free_miss_voxel_count"]
    ):
        raise ValueError("occupied→free 数不能超过 explicit-free miss 更新数。")
    if fusion != (counts["accepted_endpoint_count"] > 0):
        raise ValueError("map_fusion_performed 与接纳端点数不一致。")
    if canonical_empty and any(counts.values()):
        raise ValueError("canonical empty 不能携带端点、融合或清障计数。")

    return OgnGridMapObservationDiagnosticsSample(
        source_topic=source_topic,
        receipt_timestamp=timestamp,
        rx_sequence=rx,
        frame_id="world",
        header_stamp_sec=header_sec,
        header_stamp_nanosec=header_nanosec,
        observation_sequence=observation_sequence,
        sensor_pose_stamp_sec=pose_sec,
        sensor_pose_stamp_nanosec=pose_nanosec,
        sensor_origin=sensor_origin,
        canonical_empty=canonical_empty,
        map_fusion_performed=fusion,
        map_resolution=map_resolution,
        input_point_count=counts["input_point_count"],
        accepted_endpoint_count=counts["accepted_endpoint_count"],
        hit_endpoint_count=counts["hit_endpoint_count"],
        explicit_free_endpoint_count=counts["explicit_free_endpoint_count"],
        hit_endpoint_samples_truncated=hit_truncated,
        hit_endpoint_samples=hit_samples,
        hit_endpoint_sample_voxel_indices=hit_voxel_indices,
        free_to_occupied_transition_count=counts[
            "free_to_occupied_transition_count"
        ],
        free_to_occupied_transition_samples_truncated=(
            transition_truncated
        ),
        free_to_occupied_transition_hit_samples=transition_hit_samples,
        free_to_occupied_transition_voxel_indices=(
            transition_voxel_indices
        ),
        explicit_free_miss_voxel_count=counts[
            "explicit_free_miss_voxel_count"
        ],
        occupied_to_free_by_explicit_miss_count=counts[
            "occupied_to_free_by_explicit_miss_count"
        ],
        occupied_to_free_samples_truncated=clear_truncated,
        occupied_to_free_by_explicit_miss_samples=clear_samples,
        occupied_to_free_sample_voxel_indices=clear_voxel_indices,
        occupied_to_free_transition_hit_observation_sequences=(
            transition_hit_sequences
        ),
        occupied_to_free_transition_hit_samples=(
            clear_transition_hit_samples
        ),
        occupied_to_free_transition_hit_header_stamp_ns=(
            clear_transition_hit_stamps
        ),
        occupied_removed_by_sliding_reset_count=counts[
            "occupied_removed_by_sliding_reset_count"
        ],
    )


def parse_bspline_diagnostics_outputs(
    outputs: Mapping[str, object],
    *,
    source_topic: str,
    receipt_timestamp: float,
    rx_sequence: int,
) -> OgnBsplineDiagnosticsSample:
    """严格解析已发布 B-spline 的有序参考门与全时域抽样证据。"""

    expected_fields = frozenset(_BSPLINE_DIAGNOSTICS_OUTPUT_PORTS)
    if not isinstance(outputs, Mapping) or set(outputs) != expected_fields:
        raise ValueError("bspline diagnostics outputs 字段集合不完整。")
    _validate_topic(source_topic, "bspline diagnostics source_topic")
    timestamp = _finite_scalar(receipt_timestamp, "bspline receipt_timestamp")
    if timestamp <= 0.0:
        raise ValueError("bspline receipt_timestamp 必须是正数。")
    rx = _bounded_integer(
        rx_sequence,
        "bspline diagnostics rx_sequence",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    if outputs["header_frame_id"] != "world":
        raise ValueError("bspline diagnostics frame 必须严格等于 world。")
    header_sec, header_nanosec = _ros_time_parts(
        outputs["header_stamp_sec"],
        outputs["header_stamp_nanosec"],
        "bspline diagnostics header.stamp",
        require_nonzero=True,
    )
    start_sec, start_nanosec = _ros_time_parts(
        outputs["start_time_sec"],
        outputs["start_time_nanosec"],
        "bspline diagnostics start_time",
        require_nonzero=True,
    )
    reference_sec, reference_nanosec = _ros_time_parts(
        outputs["reference_path_stamp_sec"],
        outputs["reference_path_stamp_nanosec"],
        "bspline diagnostics reference_path_stamp",
        require_nonzero=True,
    )
    diagnostic_sequence = _bounded_integer(
        outputs["diagnostic_sequence"],
        "bspline diagnostics sequence",
        minimum=1,
        maximum=(1 << 64) - 1,
    )
    traj_id = _bounded_integer(
        outputs["traj_id"],
        "bspline diagnostics traj_id",
        minimum=-(1 << 63),
        maximum=(1 << 63) - 1,
    )
    booleans = {
        name: _strict_bool(outputs[name], f"bspline diagnostics {name}")
        for name in (
            "is_final",
            "emergency_stop",
            "stationary",
            "ordered_reference_checked",
            "ordered_reference_safe",
            "trajectory_samples_truncated",
            "ordered_reference_samples_truncated",
        )
    }
    metrics = {
        name: _finite_scalar(outputs[name], f"bspline diagnostics {name}")
        for name in (
            "maximum_trajectory_deviation",
            "maximum_guide_anchor_deviation",
            "maximum_guide_progress_lead",
            "maximum_deviation_limit",
            "maximum_progress_lead_limit",
            "trajectory_duration",
            "maximum_velocity_upper_bound",
            "double_cylinder_radius",
            "double_cylinder_offset",
        )
    }
    if any(value < 0.0 for value in metrics.values()):
        raise ValueError("B-spline corridor 指标与门限不能为负数。")
    if metrics["trajectory_duration"] <= 0.0:
        raise ValueError("B-spline trajectory_duration 必须为正数。")
    if metrics["double_cylinder_radius"] <= 0.0:
        raise ValueError("B-spline double_cylinder_radius 必须为正数。")
    trajectory_total = _bounded_integer(
        outputs["trajectory_sample_count_total"],
        "trajectory_sample_count_total",
        minimum=2,
        maximum=(1 << 32) - 1,
    )
    reference_total = _bounded_integer(
        outputs["ordered_reference_sample_count_total"],
        "ordered_reference_sample_count_total",
        minimum=0,
        maximum=(1 << 32) - 1,
    )
    trajectory_samples = _parse_bounded_point_samples(
        outputs["trajectory_samples"],
        "bspline diagnostics trajectory_samples",
    )
    reference_samples = _parse_bounded_point_samples(
        outputs["ordered_reference_samples"],
        "bspline diagnostics ordered_reference_samples",
    )
    _validate_bounded_sample_count(
        total_count=trajectory_total,
        samples=trajectory_samples,
        truncated=booleans["trajectory_samples_truncated"],
        field_name="bspline diagnostics trajectory_samples",
    )
    _validate_bounded_sample_count(
        total_count=reference_total,
        samples=reference_samples,
        truncated=booleans["ordered_reference_samples_truncated"],
        field_name="bspline diagnostics ordered_reference_samples",
    )
    expected_trajectory_total = (
        math.ceil(metrics["trajectory_duration"] / 0.01) + 1
    )
    if trajectory_total != expected_trajectory_total:
        raise ValueError(
            "B-spline trajectory_sample_count_total 与 0.01 s 全时域合同不一致。"
        )
    checked = booleans["ordered_reference_checked"]
    safe = booleans["ordered_reference_safe"]
    if safe and not checked:
        raise ValueError("ordered_reference_safe=true 要求已执行有序参考检查。")
    if checked and not safe:
        raise ValueError("未通过有序参考门的 B-spline 不得发布诊断。")
    if checked:
        if reference_total < 2:
            raise ValueError("有序参考检查要求至少两个 reference 样本。")
        if (
            metrics["maximum_trajectory_deviation"]
            > metrics["maximum_deviation_limit"] + 1.0e-9
            or metrics["maximum_guide_anchor_deviation"]
            > metrics["maximum_deviation_limit"] + 1.0e-9
            or metrics["maximum_guide_progress_lead"]
            > metrics["maximum_progress_lead_limit"] + 1.0e-9
        ):
            raise ValueError("已发布 B-spline 的 ordered corridor 指标超过门限。")
    elif not (booleans["stationary"] or booleans["emergency_stop"]):
        raise ValueError("普通运动 B-spline 必须执行 ordered reference 检查。")
    if (
        not booleans["stationary"]
        and metrics["maximum_velocity_upper_bound"] <= 0.0
    ):
        raise ValueError("运动 B-spline 的连续速度上界必须为正数。")

    active_sensing = _strict_bool(
        outputs["active_sensing"],
        "bspline diagnostics active_sensing",
    )
    active_event = _bounded_integer(
        outputs["active_sensing_event"],
        "bspline diagnostics active_sensing_event",
        minimum=0,
        maximum=6,
    )
    active_metrics = {
        name: _finite_scalar(
            outputs[name],
            f"bspline diagnostics {name}",
        )
        for name in (
            "active_sensing_start_yaw",
            "active_sensing_target_yaw",
            "active_sensing_yaw_offset",
            "active_sensing_yaw_rate",
            "active_sensing_settle_yaw_error",
            "active_sensing_settle_angular_speed",
            "active_sensing_stable_duration",
        )
    }
    settle_sec, settle_nanosec = _ros_time_parts(
        outputs["active_sensing_settle_stamp_sec"],
        outputs["active_sensing_settle_stamp_nanosec"],
        "bspline diagnostics active_sensing_settle_stamp",
    )
    fusion_baseline = _bounded_integer(
        outputs["active_sensing_fusion_baseline"],
        "bspline diagnostics active_sensing_fusion_baseline",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    fusion_current = _bounded_integer(
        outputs["active_sensing_fusion_current"],
        "bspline diagnostics active_sensing_fusion_current",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    fusion_distinct = _bounded_integer(
        outputs["active_sensing_fusion_distinct"],
        "bspline diagnostics active_sensing_fusion_distinct",
        minimum=0,
        maximum=(1 << 32) - 1,
    )
    fusion_required = _bounded_integer(
        outputs["active_sensing_fusion_required"],
        "bspline diagnostics active_sensing_fusion_required",
        minimum=0,
        maximum=(1 << 32) - 1,
    )
    active_completed = _strict_bool(
        outputs["active_sensing_completed"],
        "bspline diagnostics active_sensing_completed",
    )
    active_failed = _strict_bool(
        outputs["active_sensing_failed"],
        "bspline diagnostics active_sensing_failed",
    )
    active_reason = outputs["active_sensing_reason"]
    if not isinstance(active_reason, str):
        raise TypeError(
            "bspline diagnostics active_sensing_reason 必须是字符串。"
        )

    active_values_default = (
        active_event == 0
        and all(value == 0.0 for value in active_metrics.values())
        and settle_sec == 0
        and settle_nanosec == 0
        and fusion_baseline == 0
        and fusion_current == 0
        and fusion_distinct == 0
        and fusion_required == 0
        and not active_completed
        and not active_failed
        and active_reason == ""
    )
    if not active_sensing:
        if not active_values_default:
            raise ValueError(
                "普通 B-spline 的主动感知字段必须全部为默认值。"
            )
    else:
        if active_event == 0:
            raise ValueError("active_sensing=true 不能使用 EVENT_NONE。")
        if not booleans["stationary"] or booleans["is_final"] or booleans[
            "emergency_stop"
        ]:
            raise ValueError(
                "主动感知轨迹必须是非 final、非急停的 stationary 轨迹。"
            )
        if not active_reason.strip():
            raise ValueError("主动感知事件必须提供非空 reason。")
        yaw_offset = active_metrics["active_sensing_yaw_offset"]
        yaw_rate = active_metrics["active_sensing_yaw_rate"]
        if abs(yaw_offset) > 0.22 + 1.0e-12:
            raise ValueError("主动感知 yaw_offset 超过 0.22 rad。")
        if yaw_rate <= 0.0 or yaw_rate > 0.20 + 1.0e-12:
            raise ValueError("主动感知 yaw_rate 必须位于 (0, 0.20] rad/s。")
        expected_target = math.atan2(
            math.sin(active_metrics["active_sensing_start_yaw"] + yaw_offset),
            math.cos(active_metrics["active_sensing_start_yaw"] + yaw_offset),
        )
        target_error = math.atan2(
            math.sin(
                active_metrics["active_sensing_target_yaw"] - expected_target
            ),
            math.cos(
                active_metrics["active_sensing_target_yaw"] - expected_target
            ),
        )
        if abs(target_error) > 1.0e-9:
            raise ValueError(
                "主动感知 target_yaw 与 start_yaw+yaw_offset 不一致。"
            )
        if fusion_required != 3:
            raise ValueError("主动感知 fusion_required 必须严格等于 3。")

        settle_stamp_ns = settle_sec * 1_000_000_000 + settle_nanosec
        settle_error = active_metrics["active_sensing_settle_yaw_error"]
        settle_speed = active_metrics["active_sensing_settle_angular_speed"]
        stable_duration = active_metrics["active_sensing_stable_duration"]
        pre_settle = active_event in (1, 2)
        if pre_settle:
            if (
                settle_stamp_ns != 0
                or settle_error != 0.0
                or settle_speed != 0.0
                or stable_duration != 0.0
                or fusion_baseline != 0
                or fusion_current != 0
                or fusion_distinct != 0
                or active_completed
                or active_failed
            ):
                raise ValueError(
                    "主动感知 STARTED/ACCEPTED 快照必须处于 settle 前默认态。"
                )
        else:
            failed_before_settle = active_event == 6 and settle_stamp_ns == 0
            if failed_before_settle:
                if (
                    settle_error != 0.0
                    or settle_speed != 0.0
                    or stable_duration != 0.0
                    or fusion_baseline != 0
                    or fusion_current != 0
                    or fusion_distinct != 0
                ):
                    raise ValueError(
                        "settle 前 FAILED 快照的稳定与融合字段必须为零。"
                    )
            else:
                if settle_stamp_ns <= 0:
                    raise ValueError(
                        "settle 后主动感知事件必须提供非零 settle_stamp。"
                    )
                if settle_error < 0.0 or settle_error > 0.02 + 1.0e-12:
                    raise ValueError("主动感知 settle_yaw_error 超过 0.02 rad。")
                if settle_speed < 0.0 or settle_speed > 0.05 + 1.0e-12:
                    raise ValueError(
                        "主动感知 settle_angular_speed 超过 0.05 rad/s。"
                    )
                if stable_duration + 1.0e-12 < 0.10:
                    raise ValueError("主动感知 stable_duration 小于 0.10 s。")
                if fusion_current < fusion_baseline:
                    raise ValueError(
                        "主动感知 fusion_current 不能小于 baseline。"
                    )
                if fusion_distinct > fusion_current - fusion_baseline:
                    raise ValueError(
                        "主动感知 distinct fusion 超过融合序列增量。"
                    )
                if active_event == 3 and (
                    fusion_current != fusion_baseline or fusion_distinct != 0
                ):
                    raise ValueError(
                        "YAW_STABLE 快照必须以当前 fusion 序列为 baseline。"
                    )

        if active_event == 5:
            if (
                not active_completed
                or active_failed
                or fusion_distinct < fusion_required
            ):
                raise ValueError(
                    "COMPLETED 快照必须满足三帧融合且仅置 completed。"
                )
        elif active_event == 6:
            if active_completed or not active_failed:
                raise ValueError("FAILED 快照必须仅置 failed。")
        elif active_completed or active_failed:
            raise ValueError(
                "非终态主动感知事件不能设置 completed/failed。"
            )

    return OgnBsplineDiagnosticsSample(
        source_topic=source_topic,
        receipt_timestamp=timestamp,
        rx_sequence=rx,
        frame_id="world",
        header_stamp_sec=header_sec,
        header_stamp_nanosec=header_nanosec,
        diagnostic_sequence=diagnostic_sequence,
        start_time_sec=start_sec,
        start_time_nanosec=start_nanosec,
        reference_path_stamp_sec=reference_sec,
        reference_path_stamp_nanosec=reference_nanosec,
        traj_id=traj_id,
        is_final=booleans["is_final"],
        emergency_stop=booleans["emergency_stop"],
        stationary=booleans["stationary"],
        ordered_reference_checked=checked,
        ordered_reference_safe=safe,
        maximum_trajectory_deviation=metrics[
            "maximum_trajectory_deviation"
        ],
        maximum_guide_anchor_deviation=metrics[
            "maximum_guide_anchor_deviation"
        ],
        maximum_guide_progress_lead=metrics["maximum_guide_progress_lead"],
        maximum_deviation_limit=metrics["maximum_deviation_limit"],
        maximum_progress_lead_limit=metrics[
            "maximum_progress_lead_limit"
        ],
        trajectory_duration=metrics["trajectory_duration"],
        maximum_velocity_upper_bound=metrics[
            "maximum_velocity_upper_bound"
        ],
        double_cylinder_radius=metrics["double_cylinder_radius"],
        double_cylinder_offset=metrics["double_cylinder_offset"],
        trajectory_sample_count_total=trajectory_total,
        trajectory_samples_truncated=booleans[
            "trajectory_samples_truncated"
        ],
        trajectory_samples=trajectory_samples,
        ordered_reference_sample_count_total=reference_total,
        ordered_reference_samples_truncated=booleans[
            "ordered_reference_samples_truncated"
        ],
        ordered_reference_samples=reference_samples,
        active_sensing=active_sensing,
        active_sensing_event=active_event,
        active_sensing_start_yaw=active_metrics[
            "active_sensing_start_yaw"
        ],
        active_sensing_target_yaw=active_metrics[
            "active_sensing_target_yaw"
        ],
        active_sensing_yaw_offset=active_metrics[
            "active_sensing_yaw_offset"
        ],
        active_sensing_yaw_rate=active_metrics["active_sensing_yaw_rate"],
        active_sensing_settle_stamp_sec=settle_sec,
        active_sensing_settle_stamp_nanosec=settle_nanosec,
        active_sensing_settle_yaw_error=active_metrics[
            "active_sensing_settle_yaw_error"
        ],
        active_sensing_settle_angular_speed=active_metrics[
            "active_sensing_settle_angular_speed"
        ],
        active_sensing_stable_duration=active_metrics[
            "active_sensing_stable_duration"
        ],
        active_sensing_fusion_baseline=fusion_baseline,
        active_sensing_fusion_current=fusion_current,
        active_sensing_fusion_distinct=fusion_distinct,
        active_sensing_fusion_required=fusion_required,
        active_sensing_completed=active_completed,
        active_sensing_failed=active_failed,
        active_sensing_reason=active_reason,
    )


class IsaacRos2OgnBridge:
    """创建并驱动 Isaac 内部的 ROS 2 状态与点云发布图。"""

    def __init__(self, config: IsaacRos2OgnBridgeConfig | None = None) -> None:
        self.config = config or IsaacRos2OgnBridgeConfig()
        self._graph: object | None = None
        self._setup_failed = False
        self._attribute_helpers: dict[str, object] = {}
        self._last_state_timestamp: float | None = None
        self._last_cloud_timestamp: float | None = None
        self._last_command_sequence: int | None = 0
        self._last_navigation_status_rx_sequence: int | None = 0
        self._last_navigation_status_status_sequence: int | None = None
        self._last_navigation_status_state_revision: int | None = None
        self._last_navigation_status_header_stamp_ns: int | None = None
        self._last_navigation_status_sample: OgnNavigationStatusSample | None = None
        self._navigation_status_fault: str | None = None
        self._navigation_gate_dirty = False
        self._last_goal_reached_sequence: int | None = 0
        self._last_controller_status_rx_sequence: int | None = 0
        self._last_controller_status_status_sequence: int | None = None
        self._last_controller_status_acceptance_sequence: int | None = None
        self._last_grid_map_diagnostics_rx_sequence: int | None = 0
        self._last_grid_map_observation_sequence: int | None = None
        self._last_bspline_diagnostics_rx_sequence: int | None = 0
        self._last_bspline_diagnostic_sequence: int | None = None
        self._last_reference_path_sequence: int | None = 0
        self._latest_reference_path_stamp_ns = 0
        self._latest_reference_path_signature: tuple[bool, str] | None = None
        self._active_reference_path_stamp_ns = 0
        self._reference_path_identity_fault: str | None = None
        self._pct_goal_publish_sequence = 0
        self._last_pct_goal_stamp_ns = 0
        self._last_pct_goal_sample: OgnPCTGoalSample | None = None
        self._pct_goal_transport_attempt_count = 0
        self._stair_execution_frozen_writer_epoch = uuid.uuid4().hex
        self._stair_execution_frozen_publish_sequence = 0
        self._last_stair_execution_frozen_publish_timestamp: float | None = (
            None
        )
        self._last_stair_execution_frozen_report: (
            OgnStairExecutionFreezePublicationReport | None
        ) = None

    def graph_spec(self) -> OgnGraphSpec:
        """返回当前配置对应的纯 Python 图描述。"""

        return build_graph_spec(self.config)

    @property
    def is_setup(self) -> bool:
        """是否持有图句柄；每次发布仍会复核当前 stage 中的 graph。"""

        return self._graph is not None and not self._setup_failed

    @property
    def pct_goal_transport_attempt_count(self) -> int:
        """返回当前 PCT goal 代际已经触发的 DDS 传输次数。

        新目标首发计为一次；同一 stamp/payload 的传输重试只增加本计数，
        不会推进业务 ``sequence`` 或 ROS 时间戳。发布下一代目标后重新从
        一开始计数。
        """

        return self._pct_goal_transport_attempt_count

    @property
    def last_stair_execution_frozen_report(
        self,
    ) -> OgnStairExecutionFreezePublicationReport | None:
        """返回最后一次成功发布的楼梯规划抑制快照。"""

        return self._last_stair_execution_frozen_report

    @property
    def active_reference_path_stamp_ns(self) -> int:
        """返回可用于发布冻结快照的当前精确非空 Path identity。"""

        if self._reference_path_identity_fault is not None:
            return 0
        return int(self._active_reference_path_stamp_ns)

    def setup(self) -> object:
        """在 SimulationApp 启动后创建发布图，并返回 OmniGraph 对象。"""

        og = _import_omni_graph()
        if not _ros2_bridge_extension_is_enabled():
            raise RuntimeError(
                "isaacsim.ros2.bridge extension 尚未启用；"
                "请在 SimulationApp 创建后、setup() 前启用。"
            )
        current_graph = og.Controller.graph(self.config.graph_path)
        if self._graph is not None and current_graph is not None:
            self._graph = current_graph
            if self._setup_failed:
                # 动态端口生成可能在 Kit update 中途失败。保留本对象对图的
                # 所有权并允许显式 setup() 重试，但在成功前禁止任何发布。
                self._attribute_helpers.clear()
                self._configure_dynamic_interfaces(og)
                self._setup_failed = False
            return current_graph
        if self._graph is None and current_graph is not None:
            raise RuntimeError(
                f"graph_path 已被现有图占用：{self.config.graph_path}"
            )
        self._clear_runtime_state()

        spec = self.graph_spec()
        runtime_values = list(spec.set_values)
        if self.config.odometry_source == "compute":
            usdrt = _import_usdrt()
            runtime_values = [
                (
                    attribute,
                    [usdrt.Sdf.Path(value)]
                    if attribute == "IsaacComputeOdometry.inputs:chassisPrim"
                    else value,
                )
                for attribute, value in runtime_values
            ]

        # 该图属于仿真运行时资源，不是 GUI 中需要撤销/重做的用户编辑。
        # 使用即时命令可避免 headless Kit 在已启动仿真后通过 undo stack
        # 创建 graph wrapper prim 失败。
        graph_controller = og.Controller(
            update_usd=self.config.graph_backed_by_usd,
            undoable=False,
        )
        graph, _, _, _ = graph_controller.edit(
            {
                "graph_path": spec.graph_path,
                "evaluator_name": spec.evaluator_name,
            },
            {
                og.Controller.Keys.CREATE_NODES: list(spec.create_nodes),
                og.Controller.Keys.SET_VALUES: runtime_values,
                og.Controller.Keys.CONNECT: list(spec.connections),
            },
        )
        self._graph = graph
        self._setup_failed = True
        self._configure_dynamic_interfaces(og)
        self._setup_failed = False
        return graph

    def _configure_dynamic_interfaces(self, og: Any) -> None:
        """生成当前配置启用的 generic publisher/subscriber 动态端口。"""

        if self.config.enable_pct_goal_publisher:
            self._configure_pct_goal_dynamic_inputs(og)
        if self.config.enable_stair_execution_frozen_publisher:
            self._configure_stair_execution_frozen_dynamic_input(og)
        if self.config.enable_reference_path_subscription:
            self._configure_reference_path_dynamic_outputs(og)
        if self.config.enable_command_subscription:
            self._configure_navigation_status_dynamic_outputs(og)
        if self.config.enable_controller_status_subscription:
            self._configure_controller_status_dynamic_outputs(og)
        if self.config.enable_grid_map_diagnostics_subscription:
            self._configure_planning_diagnostics_dynamic_outputs(
                og,
                node_name="ROS2SubscribeGridMapDiagnostics",
                message_name="GridMapObservationDiagnostics",
                output_ports=_GRID_MAP_DIAGNOSTICS_OUTPUT_PORTS,
            )
        if self.config.enable_bspline_diagnostics_subscription:
            self._configure_planning_diagnostics_dynamic_outputs(
                og,
                node_name="ROS2SubscribeBsplineDiagnostics",
                message_name="BsplineDiagnostics",
                output_ports=_BSPLINE_DIAGNOSTICS_OUTPUT_PORTS,
            )

    def _configure_stair_execution_frozen_dynamic_input(self, og: Any) -> None:
        """生成带 Header、Path identity 与 writer epoch 的动态输入。"""

        self._set_attribute(
            og,
            "ROS2PublishStairExecutionFrozen.inputs:messageName",
            "",
        )
        _update_kit_once()
        self._set_attribute(
            og,
            "ROS2PublishStairExecutionFrozen.inputs:messagePackage",
            "scan_planner_msgs",
        )
        self._set_attribute(
            og,
            "ROS2PublishStairExecutionFrozen.inputs:messageSubfolder",
            "msg",
        )
        self._set_attribute(
            og,
            "ROS2PublishStairExecutionFrozen.inputs:messageName",
            "StairExecutionFreeze",
        )
        _update_kit_once()
        self._attribute_helpers.clear()
        defaults: tuple[tuple[str, object], ...] = (
            ("inputs:header:stamp:sec", 0),
            ("inputs:header:stamp:nanosec", 0),
            ("inputs:header:frame_id", self.config.odom_frame_id),
            ("inputs:reference_path_stamp:sec", 0),
            ("inputs:reference_path_stamp:nanosec", 0),
            (
                "inputs:writer_id",
                self.config.stair_execution_frozen_writer_id,
            ),
            ("inputs:writer_epoch", self._stair_execution_frozen_writer_epoch),
            ("inputs:sequence", 0),
            # 首次脉冲前保持 fail-closed true，但绝不自动触发发布。
            ("inputs:frozen", True),
        )
        node_name = "ROS2PublishStairExecutionFrozen"
        for input_path, default in defaults:
            relative_path = f"{node_name}.{input_path}"
            try:
                attribute = self._lookup_attribute(og, relative_path)
                if hasattr(attribute, "is_valid") and not attribute.is_valid():
                    raise RuntimeError("动态输入端口句柄无效")
                helper = og.AttributeValueHelper(attribute)
                helper.set(default)
                self._attribute_helpers[relative_path] = helper
            except Exception as exc:  # pragma: no cover - 仅真实 Kit 可触发
                raise RuntimeError(
                    "ROS2Publisher 未生成 scan_planner_msgs/"
                    "StairExecutionFreeze 动态输入端口："
                    f"{relative_path}"
                ) from exc

    def _configure_pct_goal_dynamic_inputs(self, og: Any) -> None:
        """让 generic publisher 生成 PoseStamped 的嵌套输入端口。"""

        self._set_attribute(
            og,
            "ROS2PublishPCTGoal.inputs:messageName",
            "",
        )
        _update_kit_once()
        self._set_attribute(
            og,
            "ROS2PublishPCTGoal.inputs:messagePackage",
            "geometry_msgs",
        )
        self._set_attribute(
            og,
            "ROS2PublishPCTGoal.inputs:messageSubfolder",
            "msg",
        )
        self._set_attribute(
            og,
            "ROS2PublishPCTGoal.inputs:messageName",
            "PoseStamped",
        )
        _update_kit_once()
        self._attribute_helpers.clear()
        for relative_path in (
            "ROS2PublishPCTGoal.inputs:header:stamp:sec",
            "ROS2PublishPCTGoal.inputs:header:stamp:nanosec",
            "ROS2PublishPCTGoal.inputs:header:frame_id",
            "ROS2PublishPCTGoal.inputs:pose:position:x",
            "ROS2PublishPCTGoal.inputs:pose:position:y",
            "ROS2PublishPCTGoal.inputs:pose:position:z",
            "ROS2PublishPCTGoal.inputs:pose:orientation:x",
            "ROS2PublishPCTGoal.inputs:pose:orientation:y",
            "ROS2PublishPCTGoal.inputs:pose:orientation:z",
            "ROS2PublishPCTGoal.inputs:pose:orientation:w",
        ):
            try:
                attribute = self._lookup_attribute(og, relative_path)
                if hasattr(attribute, "is_valid") and not attribute.is_valid():
                    raise RuntimeError("动态输入端口句柄无效")
                self._attribute_helpers[relative_path] = (
                    og.AttributeValueHelper(attribute)
                )
            except Exception as exc:  # pragma: no cover - 仅真实 Kit 可触发
                raise RuntimeError(
                    "ROS2Publisher 未生成 geometry_msgs/PoseStamped 动态输入端口："
                    f"{relative_path}"
                ) from exc

    def _configure_reference_path_dynamic_outputs(self, og: Any) -> None:
        """分两次 Kit update 生成 generic Path subscriber 的动态端口。

        Isaac 5.1 的 ``ROS2Subscriber`` 不能在创建节点的同一次 graph edit 中
        同时可靠地产生复杂消息输出。官方 subscriber 测试同样先清空
        ``messageName``，等待一次 Kit update，再写入完整类型并等待下一次
        update；否则真实运行中只有静态输入端口而没有 ``header/poses``。
        """

        self._set_attribute(
            og,
            "ROS2SubscribeReferencePath.inputs:messageName",
            "",
        )
        _update_kit_once()
        self._set_attribute(
            og,
            "ROS2SubscribeReferencePath.inputs:messagePackage",
            "nav_msgs",
        )
        self._set_attribute(
            og,
            "ROS2SubscribeReferencePath.inputs:messageSubfolder",
            "msg",
        )
        self._set_attribute(
            og,
            "ROS2SubscribeReferencePath.inputs:messageName",
            "Path",
        )
        _update_kit_once()
        self._attribute_helpers.clear()
        for relative_path in (
            "ROS2SubscribeReferencePath.outputs:header:stamp:sec",
            "ROS2SubscribeReferencePath.outputs:header:stamp:nanosec",
            "ROS2SubscribeReferencePath.outputs:header:frame_id",
            "ROS2SubscribeReferencePath.outputs:poses",
        ):
            try:
                attribute = self._lookup_attribute(og, relative_path)
                if hasattr(attribute, "is_valid") and not attribute.is_valid():
                    raise RuntimeError("动态输出端口句柄无效")
                self._attribute_helpers[relative_path] = (
                    og.AttributeValueHelper(attribute)
                )
            except Exception as exc:  # pragma: no cover - 仅真实 Kit 可触发
                raise RuntimeError(
                    "ROS2Subscriber 未生成 nav_msgs/Path 动态输出端口："
                    f"{relative_path}"
                ) from exc

    def _configure_controller_status_dynamic_outputs(self, og: Any) -> None:
        """分两次 Kit update 生成自定义 ControllerStatus 的动态输出。"""

        self._set_attribute(
            og,
            "ROS2SubscribeControllerStatus.inputs:messageName",
            "",
        )
        _update_kit_once()
        self._set_attribute(
            og,
            "ROS2SubscribeControllerStatus.inputs:messagePackage",
            "scan_planner_msgs",
        )
        self._set_attribute(
            og,
            "ROS2SubscribeControllerStatus.inputs:messageSubfolder",
            "msg",
        )
        self._set_attribute(
            og,
            "ROS2SubscribeControllerStatus.inputs:messageName",
            "ControllerStatus",
        )
        _update_kit_once()
        self._attribute_helpers.clear()
        for output_path in _CONTROLLER_STATUS_OUTPUT_PORTS.values():
            relative_path = f"ROS2SubscribeControllerStatus.{output_path}"
            try:
                attribute = self._lookup_attribute(og, relative_path)
                if hasattr(attribute, "is_valid") and not attribute.is_valid():
                    raise RuntimeError("动态输出端口句柄无效")
                self._attribute_helpers[relative_path] = (
                    og.AttributeValueHelper(attribute)
                )
            except Exception as exc:  # pragma: no cover - 仅真实 Kit 可触发
                raise RuntimeError(
                    "ROS2Subscriber 未生成 scan_planner_msgs/ControllerStatus "
                    f"动态输出端口：{relative_path}"
                ) from exc

    def _configure_planning_diagnostics_dynamic_outputs(
        self,
        og: Any,
        *,
        node_name: str,
        message_name: str,
        output_ports: Mapping[str, str],
    ) -> None:
        """生成两个 typed 规划诊断 subscriber 的嵌套和 bounded-array 端口。"""

        self._set_attribute(og, f"{node_name}.inputs:messageName", "")
        _update_kit_once()
        self._set_attribute(
            og,
            f"{node_name}.inputs:messagePackage",
            "scan_planner_msgs",
        )
        self._set_attribute(og, f"{node_name}.inputs:messageSubfolder", "msg")
        self._set_attribute(
            og,
            f"{node_name}.inputs:messageName",
            message_name,
        )
        _update_kit_once()
        self._attribute_helpers.clear()
        for output_path in output_ports.values():
            relative_path = f"{node_name}.{output_path}"
            try:
                attribute = self._lookup_attribute(og, relative_path)
                if hasattr(attribute, "is_valid") and not attribute.is_valid():
                    raise RuntimeError("动态输出端口句柄无效")
                self._attribute_helpers[relative_path] = (
                    og.AttributeValueHelper(attribute)
                )
            except Exception as exc:  # pragma: no cover - 仅真实 Kit 可触发
                raise RuntimeError(
                    "ROS2Subscriber 未生成 "
                    f"scan_planner_msgs/{message_name} 动态输出端口："
                    f"{relative_path}"
                ) from exc

    def _configure_navigation_status_dynamic_outputs(self, og: Any) -> None:
        """分两次 Kit update 生成自定义 NavigationStatus 动态输出。"""

        self._set_attribute(
            og,
            "ROS2SubscribeNavigationStatus.inputs:messageName",
            "",
        )
        _update_kit_once()
        self._set_attribute(
            og,
            "ROS2SubscribeNavigationStatus.inputs:messagePackage",
            "scan_planner_msgs",
        )
        self._set_attribute(
            og,
            "ROS2SubscribeNavigationStatus.inputs:messageSubfolder",
            "msg",
        )
        self._set_attribute(
            og,
            "ROS2SubscribeNavigationStatus.inputs:messageName",
            "NavigationStatus",
        )
        _update_kit_once()
        self._attribute_helpers.clear()
        for output_path in _NAVIGATION_STATUS_OUTPUT_PORTS.values():
            relative_path = f"ROS2SubscribeNavigationStatus.{output_path}"
            try:
                attribute = self._lookup_attribute(og, relative_path)
                if hasattr(attribute, "is_valid") and not attribute.is_valid():
                    raise RuntimeError("动态输出端口句柄无效")
                self._attribute_helpers[relative_path] = (
                    og.AttributeValueHelper(attribute)
                )
            except Exception as exc:  # pragma: no cover - 仅真实 Kit 可触发
                raise RuntimeError(
                    "ROS2Subscriber 未生成 scan_planner_msgs/NavigationStatus "
                    f"动态输出端口：{relative_path}"
                ) from exc

    def invalidate_after_stage_reload(self) -> None:
        """stage 被替换后清除旧 Graph/属性句柄与时间状态。"""

        self._clear_runtime_state()

    def refresh_after_timeline_reset(self) -> None:
        """同一 stage 硬重置后重绑 Graph，并保留连续时间门禁。

        Isaac Lab 的 ``SimulationContext.reset(soft=False)`` 会停止并重启
        timeline，但不会替换 USD stage。Graph 本身仍归本对象所有，只需丢弃
        可能随 Fabric/PhysX 重建失效的属性 helper；上一帧时间戳必须保留。
        """

        if self._graph is None:
            return
        og = _import_omni_graph()
        current_graph = og.Controller.graph(self.config.graph_path)
        if current_graph is None:
            self._clear_runtime_state()
            raise RuntimeError(
                "timeline 重置后 OmniGraph 已从当前 stage 消失；"
                "必须在继续发布前重新 setup()。"
            )
        self._graph = current_graph
        self._attribute_helpers.clear()
        # Counter 的 stage state 是否跨 hard reset 保留取决于 Fabric 生命周期。
        # 下一次 poll 先建立安全基线，绝不把 reset 前输出误当成新命令。
        self._last_command_sequence = None
        self._last_navigation_status_rx_sequence = None
        self._last_navigation_status_sample = None
        self._navigation_status_fault = "timeline_reset_requires_fresh_status"
        self._navigation_gate_dirty = True
        self._last_goal_reached_sequence = None
        self._last_controller_status_rx_sequence = None
        self._last_grid_map_diagnostics_rx_sequence = None
        self._last_grid_map_observation_sequence = None
        self._last_bspline_diagnostics_rx_sequence = None
        self._last_bspline_diagnostic_sequence = None
        self._last_reference_path_sequence = None
        self._active_reference_path_stamp_ns = 0
        self._reference_path_identity_fault = (
            "timeline_reset_requires_fresh_path"
        )

    def update_odometry(
        self,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
        linear_velocity: Sequence[float],
        angular_velocity: Sequence[float],
        timestamp: float,
    ) -> None:
        """写入根状态并同步发布。

        速度必须来自 ``root_lin_vel_b`` 和 ``root_ang_vel_b``，即
        ``base_link`` 机体系；direct 图使用 ``publishRawVelocities=true``。
        """

        if self.config.odometry_source != "direct":
            raise RuntimeError("compute 模式不能直接写入里程计端口。")
        sample = prepare_odometry_sample(
            position,
            orientation_wxyz,
            linear_velocity,
            angular_velocity,
            timestamp,
        )
        self._require_monotonic_timestamp(
            sample.timestamp,
            previous=self._last_state_timestamp,
            field_name="Odometry timestamp",
        )
        og = self._require_runtime()
        self._set_attribute(og, "ROS2PublishOdometry.inputs:position", sample.position)
        self._set_attribute(
            og,
            "ROS2PublishOdometry.inputs:orientation",
            sample.orientation_ijkr,
        )
        self._set_attribute(
            og,
            "ROS2PublishOdometry.inputs:linearVelocity",
            sample.linear_velocity,
        )
        self._set_attribute(
            og,
            "ROS2PublishOdometry.inputs:angularVelocity",
            sample.angular_velocity,
        )
        self._set_attribute(
            og,
            "ROS2PublishOdometry.inputs:timeStamp",
            sample.timestamp,
        )
        self._set_attribute(og, "ROS2PublishClock.inputs:timeStamp", sample.timestamp)
        self._trigger_and_evaluate(og, "StateTx")
        self._last_state_timestamp = sample.timestamp

    def update_point_cloud(
        self,
        points: Any,
        *,
        timestamp: float | None = None,
    ) -> None:
        """写入有限 Nx3 点云，并以状态同源的仿真时间发布。

        direct 模式必须传入与 ``update_odometry()`` 同一连续仿真时钟的时间戳，
        不能传每个 episode 归零的局部计时。
        """

        point_cloud = validate_point_cloud(points)
        if self.config.odometry_source == "direct":
            if timestamp is None:
                raise ValueError("direct 模式发布点云必须提供 timestamp。")
            timestamp_value = _finite_scalar(timestamp, "timestamp")
            if timestamp_value <= 0.0:
                raise ValueError("timestamp 必须是正数。")
            self._require_monotonic_timestamp(
                timestamp_value,
                previous=self._last_cloud_timestamp,
                field_name="PointCloud timestamp",
            )
        elif timestamp is not None:
            raise ValueError(
                "compute 模式的点云时间来自 IsaacReadSimulationTime，"
                "不能显式提供 timestamp。"
            )
        else:
            timestamp_value = None

        og = self._require_runtime()
        self._set_attribute(og, "ROS2PublishPointCloud.inputs:data", point_cloud)
        if timestamp_value is not None:
            self._set_attribute(
                og,
                "ROS2PublishPointCloud.inputs:timeStamp",
                timestamp_value,
            )
        self._trigger_and_evaluate(og, "CloudTx")
        self._last_cloud_timestamp = timestamp_value

    def publish_stair_execution_frozen(
        self,
        value: bool,
        *,
        timestamp: float,
    ) -> OgnStairExecutionFreezePublicationReport:
        """发布绑定当前精确 Path identity 的类型化楼梯冻结快照。

        Header 使用与 ``/clock``、Odometry 和 cmd_vel 安全门相同的连续
        仿真时钟；writer epoch 在 bridge 生命周期内固定，sequence 仅在
        OGN publisher 成功求值后递增。没有已确认的非空 Path 或 Path 身份
        冲突时禁止发布，不能用无身份的 ``false`` 解锁 SCAN。
        """

        if not self.config.enable_stair_execution_frozen_publisher:
            raise RuntimeError(
                "当前 OGN 配置未启用 /planning/stair_execution_frozen 发布。"
            )
        if not isinstance(value, bool):
            raise TypeError("stair_execution_frozen value 必须是布尔值。")
        timestamp_value = _finite_scalar(timestamp, "timestamp")
        if timestamp_value <= 0.0:
            raise ValueError("timestamp 必须为正数。")
        self._require_monotonic_timestamp(
            timestamp_value,
            previous=self._last_stair_execution_frozen_publish_timestamp,
            field_name="stair_execution_frozen publish timestamp",
        )
        if self._reference_path_identity_fault is not None:
            raise RuntimeError(
                "当前 reference Path identity 无效，禁止发布楼梯冻结快照："
                f"{self._reference_path_identity_fault}"
            )
        reference_path_stamp_ns = self._active_reference_path_stamp_ns
        if reference_path_stamp_ns <= 0:
            raise RuntimeError(
                "尚未收到精确的非空 current reference Path identity，"
                "禁止发布楼梯冻结快照。"
            )
        # 不能 round 到尚未发布的下一纳秒；SCAN 对 future Header 必须
        # fail-closed，因此与连续仿真时钟取不晚于当前值的整数纳秒。
        header_stamp_ns = max(1, int(timestamp_value * 1_000_000_000))
        header_stamp_sec, header_stamp_nanosec = divmod(
            header_stamp_ns,
            1_000_000_000,
        )
        reference_path_stamp_sec, reference_path_stamp_nanosec = divmod(
            reference_path_stamp_ns,
            1_000_000_000,
        )
        next_sequence = self._stair_execution_frozen_publish_sequence + 1
        og = self._require_runtime()
        values = {
            "ROS2PublishStairExecutionFrozen.inputs:header:stamp:sec": (
                header_stamp_sec
            ),
            "ROS2PublishStairExecutionFrozen.inputs:header:stamp:nanosec": (
                header_stamp_nanosec
            ),
            "ROS2PublishStairExecutionFrozen.inputs:header:frame_id": (
                self.config.odom_frame_id
            ),
            "ROS2PublishStairExecutionFrozen.inputs:reference_path_stamp:sec": (
                reference_path_stamp_sec
            ),
            "ROS2PublishStairExecutionFrozen.inputs:reference_path_stamp:nanosec": (
                reference_path_stamp_nanosec
            ),
            "ROS2PublishStairExecutionFrozen.inputs:writer_id": (
                self.config.stair_execution_frozen_writer_id
            ),
            "ROS2PublishStairExecutionFrozen.inputs:writer_epoch": (
                self._stair_execution_frozen_writer_epoch
            ),
            "ROS2PublishStairExecutionFrozen.inputs:sequence": next_sequence,
            "ROS2PublishStairExecutionFrozen.inputs:frozen": value,
        }
        for attribute, field_value in values.items():
            self._set_attribute(og, attribute, field_value)
        self._trigger_and_evaluate(og, "StairExecutionFrozenTx")
        self._stair_execution_frozen_publish_sequence = next_sequence
        report = OgnStairExecutionFreezePublicationReport(
            frozen=value,
            source_topic=self.config.stair_execution_frozen_topic,
            publish_timestamp=timestamp_value,
            header_stamp_sec=header_stamp_sec,
            header_stamp_nanosec=header_stamp_nanosec,
            reference_path_stamp_sec=reference_path_stamp_sec,
            reference_path_stamp_nanosec=reference_path_stamp_nanosec,
            writer_id=self.config.stair_execution_frozen_writer_id,
            writer_epoch=self._stair_execution_frozen_writer_epoch,
            sequence=next_sequence,
        )
        self._last_stair_execution_frozen_publish_timestamp = timestamp_value
        self._last_stair_execution_frozen_report = report
        return report

    def publish_pct_goal(
        self,
        position_base_xyz: Sequence[float],
        yaw: float,
        *,
        stamp_ns: int,
        frame_id: str = "world",
    ) -> OgnPCTGoalSample:
        """发布一条带精确代际的 world-frame base 高度 PCT 目标。

        ``stamp_ns`` 来自与 Odometry、点云和 ``/clock`` 相同的连续仿真
        时钟。若调用方在同一纳秒提交下一代目标，这里只向前推进一纳秒，
        保证 PCT adapter 不会把两个任务阶段折叠为同一代。
        """

        if not self.config.enable_pct_goal_publisher:
            raise RuntimeError("当前 OGN 配置未启用 /pct/goal 发布。")
        if frame_id != self.config.odom_frame_id:
            raise ValueError(
                "PCT goal frame 必须与 Odometry world frame 完全一致。"
            )
        position = _finite_vector(
            position_base_xyz,
            3,
            "position_base_xyz",
        )
        yaw_value = _finite_scalar(yaw, "yaw")
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int):
            raise TypeError("stamp_ns 必须是整数纳秒。")
        if stamp_ns <= 0:
            raise ValueError("stamp_ns 必须为正数。")
        effective_stamp_ns = max(
            int(stamp_ns),
            self._last_pct_goal_stamp_ns + 1,
        )
        stamp_sec, stamp_nanosec = divmod(
            effective_stamp_ns,
            1_000_000_000,
        )
        half_yaw = 0.5 * yaw_value
        og = self._require_runtime()
        values = {
            "ROS2PublishPCTGoal.inputs:header:stamp:sec": stamp_sec,
            "ROS2PublishPCTGoal.inputs:header:stamp:nanosec": stamp_nanosec,
            "ROS2PublishPCTGoal.inputs:header:frame_id": frame_id,
            "ROS2PublishPCTGoal.inputs:pose:position:x": position[0],
            "ROS2PublishPCTGoal.inputs:pose:position:y": position[1],
            "ROS2PublishPCTGoal.inputs:pose:position:z": position[2],
            "ROS2PublishPCTGoal.inputs:pose:orientation:x": 0.0,
            "ROS2PublishPCTGoal.inputs:pose:orientation:y": 0.0,
            "ROS2PublishPCTGoal.inputs:pose:orientation:z": math.sin(half_yaw),
            "ROS2PublishPCTGoal.inputs:pose:orientation:w": math.cos(half_yaw),
        }
        for attribute, value in values.items():
            self._set_attribute(og, attribute, value)
        self._trigger_and_evaluate(og, "PCTGoalTx")
        self._last_pct_goal_stamp_ns = effective_stamp_ns
        # 新目标立即使上一代 status/Twist 身份失配；下一次 policy tick 即使
        # 没有新 Twist，也必须把这条边沿送给唯一 writer 写零。
        self._navigation_gate_dirty = True
        self._pct_goal_publish_sequence += 1
        sample = OgnPCTGoalSample(
            position_base_xyz=(position[0], position[1], position[2]),
            yaw=yaw_value,
            source_topic=self.config.pct_goal_topic,
            frame_id=frame_id,
            stamp_sec=stamp_sec,
            stamp_nanosec=stamp_nanosec,
            sequence=self._pct_goal_publish_sequence,
        )
        self._last_pct_goal_sample = sample
        self._pct_goal_transport_attempt_count = 1
        return sample

    def republish_last_pct_goal(self) -> OgnPCTGoalSample:
        """重触发最后一条 PCT goal，并严格复用原 stamp 与 payload。

        Isaac 5.1 的 generic ROS 2 publisher 在首次执行时才懒创建 DDS
        writer；``reliable + volatile`` 的第一包可能早于 discovery 匹配而
        丢失。本方法只用于跨过该传输竞态。调用前会逐字段核对当前 OGN
        动态输入，任何缺失或篡改都会失败关闭，绝不把另一份 payload 冒充
        为同一业务代际。
        """

        if not self.config.enable_pct_goal_publisher:
            raise RuntimeError("当前 OGN 配置未启用 /pct/goal 发布。")
        sample = self._last_pct_goal_sample
        if sample is None or self._pct_goal_transport_attempt_count < 1:
            raise RuntimeError("尚无可重试的 PCT goal 首发样本。")
        og = self._require_runtime()
        half_yaw = 0.5 * sample.yaw
        expected_values = {
            "ROS2PublishPCTGoal.inputs:header:stamp:sec": sample.stamp_sec,
            "ROS2PublishPCTGoal.inputs:header:stamp:nanosec": (
                sample.stamp_nanosec
            ),
            "ROS2PublishPCTGoal.inputs:header:frame_id": sample.frame_id,
            "ROS2PublishPCTGoal.inputs:pose:position:x": (
                sample.position_base_xyz[0]
            ),
            "ROS2PublishPCTGoal.inputs:pose:position:y": (
                sample.position_base_xyz[1]
            ),
            "ROS2PublishPCTGoal.inputs:pose:position:z": (
                sample.position_base_xyz[2]
            ),
            "ROS2PublishPCTGoal.inputs:pose:orientation:x": 0.0,
            "ROS2PublishPCTGoal.inputs:pose:orientation:y": 0.0,
            "ROS2PublishPCTGoal.inputs:pose:orientation:z": math.sin(half_yaw),
            "ROS2PublishPCTGoal.inputs:pose:orientation:w": math.cos(half_yaw),
        }
        for attribute, expected in expected_values.items():
            actual = self._get_attribute(og, attribute)
            if actual != expected:
                raise RuntimeError(
                    "拒绝重试已被篡改的 PCT goal 动态输入："
                    f"{attribute}"
                )
        self._trigger_and_evaluate(og, "PCTGoalTx")
        self._pct_goal_transport_attempt_count += 1
        return sample

    def publish_computed_odometry(self) -> None:
        """在兼容 compute 模式下触发一次里程计与时钟发布。"""

        if self.config.odometry_source != "compute":
            raise RuntimeError("publish_computed_odometry() 仅用于 compute 模式。")
        og = self._require_runtime()
        self._trigger_and_evaluate(og, "StateTx")

    def poll_navigation_status(
        self,
        *,
        receipt_timestamp: float,
    ) -> OgnNavigationStatusSample | None:
        """轮询一条新的 supervisor 快照并执行语义序列门禁。"""

        if not self.config.enable_command_subscription:
            raise RuntimeError(
                "只有启用 /cmd_vel 唯一 writer 时才启用 /navigation/status 订阅。"
            )
        timestamp = _finite_scalar(receipt_timestamp, "receipt_timestamp")
        if timestamp <= 0.0:
            raise ValueError("receipt_timestamp 必须是正数。")
        og = self._require_runtime()
        self._trigger_and_evaluate(og, "NavigationStatusRxTick")
        raw_rx_sequence = self._get_attribute(
            og,
            "NavigationStatusRxCounter.outputs:count",
        )
        if raw_rx_sequence is None:
            rx_sequence = 0
        else:
            try:
                rx_sequence = _bounded_integer(
                    raw_rx_sequence,
                    "OGN navigation_status Counter",
                    minimum=0,
                    maximum=(1 << 64) - 1,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "OGN navigation_status Counter 未返回整数序号。"
                ) from exc

        previous_rx = self._last_navigation_status_rx_sequence
        if previous_rx is None:
            self._last_navigation_status_rx_sequence = rx_sequence
            return None
        if rx_sequence == previous_rx:
            return None
        if rx_sequence < previous_rx:
            self._last_navigation_status_rx_sequence = rx_sequence
            self._last_navigation_status_sample = None
            self._navigation_status_fault = (
                "navigation_status_rx_counter_regression"
            )
            self._navigation_gate_dirty = True
            return None

        outputs = {
            field_name: self._get_attribute(
                og,
                f"ROS2SubscribeNavigationStatus.{output_path}",
            )
            for field_name, output_path in _NAVIGATION_STATUS_OUTPUT_PORTS.items()
        }
        try:
            sample = parse_navigation_status_outputs(
                outputs,
                source_topic=self.config.navigation_status_topic,
                receipt_timestamp=timestamp,
                rx_sequence=rx_sequence,
            )
            previous_status = self._last_navigation_status_status_sequence
            if previous_status is not None and sample.status_sequence <= previous_status:
                raise RuntimeError(
                    "navigation_status.status_sequence 未严格递增："
                    f"当前 {sample.status_sequence}，上一条 {previous_status}。"
                )
            previous_revision = self._last_navigation_status_state_revision
            if previous_revision is not None and sample.state_revision < previous_revision:
                raise RuntimeError(
                    "navigation_status.state_revision 发生回退："
                    f"当前 {sample.state_revision}，上一条 {previous_revision}。"
                )
            previous_header = self._last_navigation_status_header_stamp_ns
            if previous_header is not None and sample.header_stamp_ns < previous_header:
                raise RuntimeError(
                    "navigation_status.header.stamp 发生回退。"
                )
        except (TypeError, ValueError, RuntimeError) as exc:
            self._last_navigation_status_rx_sequence = rx_sequence
            self._last_navigation_status_sample = None
            self._navigation_status_fault = str(exc)
            self._navigation_gate_dirty = True
            raise

        self._last_navigation_status_rx_sequence = rx_sequence
        self._last_navigation_status_status_sequence = sample.status_sequence
        self._last_navigation_status_state_revision = sample.state_revision
        self._last_navigation_status_header_stamp_ns = sample.header_stamp_ns
        self._last_navigation_status_sample = sample
        self._navigation_status_fault = None
        self._navigation_gate_dirty = True
        return sample

    def _navigation_gate_payload(
        self,
    ) -> tuple[NavigationSafetyPermit | None, str | None]:
        """将当前 status 与本地目标/Path identity 交叉绑定。"""

        if self._navigation_status_fault is not None:
            return None, self._navigation_status_fault
        sample = self._last_navigation_status_sample
        if sample is None:
            return None, "missing_navigation_status"
        identity_valid = (
            self._last_pct_goal_stamp_ns > 0
            and sample.goal_id == self._last_pct_goal_stamp_ns
            and self._reference_path_identity_fault is None
            and self._active_reference_path_stamp_ns > 0
            and sample.active_path_stamp_ns
            == self._active_reference_path_stamp_ns
        )
        return sample.to_safety_permit(identity_valid=identity_valid), None

    def navigation_status_observed_diagnostics(self) -> dict[str, object]:
        """返回 supervisor→OGN 已观测状态，不冒充 policy 已消费许可。"""

        sample = self._last_navigation_status_sample
        permit, error = self._navigation_gate_payload()
        return {
            "schema": "navigation_status_observed_diagnostics_v1",
            "topic": self.config.navigation_status_topic,
            "status_error": error,
            "local_pct_goal_stamp_ns": int(self._last_pct_goal_stamp_ns),
            "local_active_path_stamp_ns": int(
                self._active_reference_path_stamp_ns
            ),
            "local_reference_path_identity_fault": (
                self._reference_path_identity_fault
            ),
            "status": (
                None
                if sample is None
                else {
                    "receipt_timestamp": float(sample.receipt_timestamp),
                    "rx_sequence": int(sample.rx_sequence),
                    "header_stamp_ns": int(sample.header_stamp_ns),
                    "status_sequence": int(sample.status_sequence),
                    "state_revision": int(sample.state_revision),
                    "goal_id": int(sample.goal_id),
                    "state": int(sample.state),
                    "allow_tracking_command": bool(
                        sample.allow_tracking_command
                    ),
                    "force_zero_velocity": bool(
                        sample.force_zero_velocity
                    ),
                    "stop_confirmed": bool(sample.stop_confirmed),
                    "global_replan_requested": bool(
                        sample.global_replan_requested
                    ),
                    "global_replan_in_flight": bool(
                        sample.global_replan_in_flight
                    ),
                    "global_replan_request_id": int(
                        sample.global_replan_request_id
                    ),
                    "pct_plan_id": int(sample.pct_plan_id),
                    "active_path_stamp_ns": int(
                        sample.active_path_stamp_ns
                    ),
                    "consecutive_scan_failures": int(
                        sample.consecutive_scan_failures
                    ),
                    "stale_inputs": list(sample.stale_inputs),
                    "reason": sample.reason,
                    "identity_valid": bool(
                        permit is not None and permit.identity_valid
                    ),
                }
            ),
        }

    def _policy_input_sample(
        self,
        *,
        receipt_timestamp: float,
        sequence: int,
        linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        command_present: bool,
    ) -> OgnTwistSample:
        """构造同一 writer 消费的速度/许可或纯许可输入。"""

        permit, error = self._navigation_gate_payload()
        return OgnTwistSample(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            receipt_timestamp=receipt_timestamp,
            sequence=sequence,
            command_present=command_present,
            navigation_permit=permit,
            navigation_status_error=error,
        )

    def poll_twist(self, *, receipt_timestamp: float) -> OgnTwistSample | None:
        """轮询一条新 ``Twist``，重复值也由 OGN Counter 正确识别。

        ``geometry_msgs/Twist`` 本身没有 Header，因此接收时间必须由调用方传入
        同一个连续仿真时钟。没有新消息时返回 ``None``，调用方据此执行超时停车。
        """

        if not self.config.enable_command_subscription:
            raise RuntimeError("当前 OGN 配置未启用 /cmd_vel 订阅。")
        timestamp = _finite_scalar(receipt_timestamp, "receipt_timestamp")
        if timestamp <= 0.0:
            raise ValueError("receipt_timestamp 必须是正数。")

        # runtime 仍只调用本方法；这里先读安全快照，使拒绝边沿不依赖第二个
        # ``/cmd_vel`` publisher，也不需要新增 policy sink writer。
        try:
            self.poll_navigation_status(receipt_timestamp=timestamp)
        except (TypeError, ValueError, RuntimeError):
            # 故障已经锁存在 _navigation_status_fault。继续到返回包络，确保
            # 调用方的同一 policy tick 先真实写零，而不是在写入前抛出。
            pass
        og = self._require_runtime()
        self._trigger_and_evaluate(og, "CommandRxTick")
        raw_sequence = self._get_attribute(og, "CommandRxCounter.outputs:count")
        if raw_sequence is None:
            # Counter 在第一次 execIn 前可能保持未赋值状态，等价于零条消息。
            sequence = 0
        else:
            try:
                sequence = int(raw_sequence)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("OGN cmd_vel Counter 未返回整数序号。") from exc
        if sequence < 0:
            raise RuntimeError("OGN cmd_vel Counter 返回了负序号。")

        if self._last_command_sequence is None:
            self._last_command_sequence = sequence
            if self._navigation_gate_dirty:
                self._navigation_gate_dirty = False
                return self._policy_input_sample(
                    receipt_timestamp=timestamp,
                    sequence=sequence,
                    command_present=False,
                )
            return None
        if sequence == self._last_command_sequence:
            if self._navigation_gate_dirty:
                self._navigation_gate_dirty = False
                return self._policy_input_sample(
                    receipt_timestamp=timestamp,
                    sequence=sequence,
                    command_present=False,
                )
            return None
        if sequence < self._last_command_sequence:
            # Counter 随 timeline/Fabric 重建归零时先丢弃旧输出，等待下一帧。
            self._last_command_sequence = sequence
            if self._navigation_gate_dirty:
                self._navigation_gate_dirty = False
                return self._policy_input_sample(
                    receipt_timestamp=timestamp,
                    sequence=sequence,
                    command_present=False,
                )
            return None

        linear = _finite_vector(
            self._get_attribute(
                og,
                "ROS2SubscribeTwist.outputs:linearVelocity",
            ),
            3,
            "cmd_vel.linear",
        )
        angular = _finite_vector(
            self._get_attribute(
                og,
                "ROS2SubscribeTwist.outputs:angularVelocity",
            ),
            3,
            "cmd_vel.angular",
        )
        self._last_command_sequence = sequence
        self._navigation_gate_dirty = False
        return self._policy_input_sample(
            linear_velocity=(linear[0], linear[1], linear[2]),
            angular_velocity=(angular[0], angular[1], angular[2]),
            receipt_timestamp=timestamp,
            sequence=sequence,
            command_present=True,
        )

    def poll_goal_reached(
        self,
        *,
        receipt_timestamp: float,
    ) -> OgnBoolSample | None:
        """轮询一条新的 ``goal_reached`` 布尔状态。

        ``std_msgs/Bool`` 没有 Header，因此接收时间由调用方使用连续仿真时钟
        提供。Counter 能区分内容相同的连续消息；timeline 重置后的第一次轮询
        只建立新基线，避免复用 OGN 动态输出中残留的上一轮状态。
        """

        if not self.config.enable_goal_reached_subscription:
            raise RuntimeError("当前 OGN 配置未启用 /planning/goal_reached 订阅。")
        timestamp = _finite_scalar(receipt_timestamp, "receipt_timestamp")
        if timestamp <= 0.0:
            raise ValueError("receipt_timestamp 必须是正数。")

        og = self._require_runtime()
        self._trigger_and_evaluate(og, "GoalReachedRxTick")
        raw_sequence = self._get_attribute(
            og,
            "GoalReachedRxCounter.outputs:count",
        )
        if raw_sequence is None:
            sequence = 0
        else:
            try:
                sequence = int(raw_sequence)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "OGN goal_reached Counter 未返回整数序号。"
                ) from exc
        if sequence < 0:
            raise RuntimeError("OGN goal_reached Counter 返回了负序号。")

        if self._last_goal_reached_sequence is None:
            self._last_goal_reached_sequence = sequence
            return None
        if sequence == self._last_goal_reached_sequence:
            return None
        if sequence < self._last_goal_reached_sequence:
            self._last_goal_reached_sequence = sequence
            return None

        raw_value = self._get_attribute(
            og,
            "ROS2SubscribeGoalReached.outputs:data",
        )
        if not isinstance(raw_value, bool):
            raise RuntimeError("OGN goal_reached 输出不是布尔值。")
        self._last_goal_reached_sequence = sequence
        return OgnBoolSample(
            value=raw_value,
            receipt_timestamp=timestamp,
            sequence=sequence,
        )

    def poll_controller_status(
        self,
        *,
        receipt_timestamp: float,
    ) -> OgnControllerStatusSample | None:
        """轮询一条新的 typed controller 生命周期快照。"""

        if not self.config.enable_controller_status_subscription:
            raise RuntimeError(
                "当前 OGN 配置未启用 /planning/controller_status 订阅。"
            )
        timestamp = _finite_scalar(receipt_timestamp, "receipt_timestamp")
        if timestamp <= 0.0:
            raise ValueError("receipt_timestamp 必须是正数。")

        og = self._require_runtime()
        self._trigger_and_evaluate(og, "ControllerStatusRxTick")
        raw_rx_sequence = self._get_attribute(
            og,
            "ControllerStatusRxCounter.outputs:count",
        )
        if raw_rx_sequence is None:
            rx_sequence = 0
        else:
            try:
                rx_sequence = _bounded_integer(
                    raw_rx_sequence,
                    "OGN controller_status Counter",
                    minimum=0,
                    maximum=(1 << 64) - 1,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "OGN controller_status Counter 未返回整数序号。"
                ) from exc

        if self._last_controller_status_rx_sequence is None:
            self._last_controller_status_rx_sequence = rx_sequence
            return None
        if rx_sequence == self._last_controller_status_rx_sequence:
            return None
        if rx_sequence < self._last_controller_status_rx_sequence:
            self._last_controller_status_rx_sequence = rx_sequence
            return None
        if rx_sequence != self._last_controller_status_rx_sequence + 1:
            raise RuntimeError(
                "controller_status OGN 接收序列出现缺口："
                f"当前 {rx_sequence}，上一条 "
                f"{self._last_controller_status_rx_sequence}。"
            )

        outputs = {
            field_name: self._get_attribute(
                og,
                f"ROS2SubscribeControllerStatus.{output_path}",
            )
            for field_name, output_path in _CONTROLLER_STATUS_OUTPUT_PORTS.items()
        }
        sample = parse_controller_status_outputs(
            outputs,
            source_topic=self.config.controller_status_topic,
            receipt_timestamp=timestamp,
            rx_sequence=rx_sequence,
        )
        previous_status = self._last_controller_status_status_sequence
        if (
            previous_status is not None
            and sample.status_sequence != previous_status + 1
        ):
            raise RuntimeError(
                "controller_status.status_sequence 未严格递增（要求连续）："
                f"当前 {sample.status_sequence}，上一条 {previous_status}。"
            )
        previous_acceptance = self._last_controller_status_acceptance_sequence
        if (
            previous_acceptance is not None
            and sample.acceptance_sequence < previous_acceptance
        ):
            raise RuntimeError(
                "controller_status.acceptance_sequence 发生回退："
                f"当前 {sample.acceptance_sequence}，上一条 {previous_acceptance}。"
            )
        self._last_controller_status_rx_sequence = rx_sequence
        self._last_controller_status_status_sequence = sample.status_sequence
        self._last_controller_status_acceptance_sequence = (
            sample.acceptance_sequence
        )
        return sample

    def poll_grid_map_observation_diagnostics(
        self,
        *,
        receipt_timestamp: float,
    ) -> OgnGridMapObservationDiagnosticsSample | None:
        """轮询过滤后点云的 GridMap 融合、hit 和 explicit-miss 证据。"""

        if not self.config.enable_grid_map_diagnostics_subscription:
            raise RuntimeError("当前 OGN 配置未启用 GridMap diagnostics 订阅。")
        timestamp = _finite_scalar(receipt_timestamp, "receipt_timestamp")
        if timestamp <= 0.0:
            raise ValueError("receipt_timestamp 必须是正数。")
        og = self._require_runtime()
        self._trigger_and_evaluate(og, "GridMapDiagnosticsRxTick")
        raw_rx_sequence = self._get_attribute(
            og,
            "GridMapDiagnosticsRxCounter.outputs:count",
        )
        rx_sequence = (
            0
            if raw_rx_sequence is None
            else _bounded_integer(
                raw_rx_sequence,
                "OGN GridMap diagnostics Counter",
                minimum=0,
                maximum=(1 << 64) - 1,
            )
        )
        previous_rx = self._last_grid_map_diagnostics_rx_sequence
        if previous_rx is None:
            self._last_grid_map_diagnostics_rx_sequence = rx_sequence
            return None
        if rx_sequence == previous_rx:
            return None
        if rx_sequence < previous_rx:
            self._last_grid_map_diagnostics_rx_sequence = rx_sequence
            return None
        outputs = {
            field_name: self._get_attribute(
                og,
                f"ROS2SubscribeGridMapDiagnostics.{output_path}",
            )
            for field_name, output_path in (
                _GRID_MAP_DIAGNOSTICS_OUTPUT_PORTS.items()
            )
        }
        sample = parse_grid_map_observation_diagnostics_outputs(
            outputs,
            source_topic=self.config.grid_map_diagnostics_topic,
            receipt_timestamp=timestamp,
            rx_sequence=rx_sequence,
        )
        previous_sequence = self._last_grid_map_observation_sequence
        if (
            previous_sequence is not None
            and sample.observation_sequence <= previous_sequence
        ):
            raise RuntimeError(
                "GridMap observation_sequence 未严格递增："
                f"当前 {sample.observation_sequence}，上一条 {previous_sequence}。"
            )
        self._last_grid_map_diagnostics_rx_sequence = rx_sequence
        self._last_grid_map_observation_sequence = sample.observation_sequence
        return sample

    def poll_bspline_diagnostics(
        self,
        *,
        receipt_timestamp: float,
    ) -> OgnBsplineDiagnosticsSample | None:
        """轮询一条与已发布 B-spline identity 完全绑定的几何诊断。"""

        if not self.config.enable_bspline_diagnostics_subscription:
            raise RuntimeError("当前 OGN 配置未启用 B-spline diagnostics 订阅。")
        timestamp = _finite_scalar(receipt_timestamp, "receipt_timestamp")
        if timestamp <= 0.0:
            raise ValueError("receipt_timestamp 必须是正数。")
        og = self._require_runtime()
        self._trigger_and_evaluate(og, "BsplineDiagnosticsRxTick")
        raw_rx_sequence = self._get_attribute(
            og,
            "BsplineDiagnosticsRxCounter.outputs:count",
        )
        rx_sequence = (
            0
            if raw_rx_sequence is None
            else _bounded_integer(
                raw_rx_sequence,
                "OGN B-spline diagnostics Counter",
                minimum=0,
                maximum=(1 << 64) - 1,
            )
        )
        previous_rx = self._last_bspline_diagnostics_rx_sequence
        if previous_rx is None:
            self._last_bspline_diagnostics_rx_sequence = rx_sequence
            return None
        if rx_sequence == previous_rx:
            return None
        if rx_sequence < previous_rx:
            self._last_bspline_diagnostics_rx_sequence = rx_sequence
            return None
        outputs = {
            field_name: self._get_attribute(
                og,
                f"ROS2SubscribeBsplineDiagnostics.{output_path}",
            )
            for field_name, output_path in (
                _BSPLINE_DIAGNOSTICS_OUTPUT_PORTS.items()
            )
        }
        sample = parse_bspline_diagnostics_outputs(
            outputs,
            source_topic=self.config.bspline_diagnostics_topic,
            receipt_timestamp=timestamp,
            rx_sequence=rx_sequence,
        )
        previous_sequence = self._last_bspline_diagnostic_sequence
        if (
            previous_sequence is not None
            and sample.diagnostic_sequence <= previous_sequence
        ):
            raise RuntimeError(
                "B-spline diagnostic_sequence 未严格递增："
                f"当前 {sample.diagnostic_sequence}，上一条 {previous_sequence}。"
            )
        self._last_bspline_diagnostics_rx_sequence = rx_sequence
        self._last_bspline_diagnostic_sequence = sample.diagnostic_sequence
        return sample

    def poll_reference_path(self) -> OgnPathSample | None:
        """轮询一代新的 ``/initial_path`` 三维地面高度参考路径。

        Generic subscriber 会把 Header 和 PoseStamped 嵌套消息分别编码成
        JSON 输出。Counter 仅用来判断消息代际，不用点值变化猜测新消息；同一
        timeline reset 后第一次轮询只建立基线，下一代 Path 仍会正常返回。
        """

        if not self.config.enable_reference_path_subscription:
            raise RuntimeError("当前 OGN 配置未启用 /initial_path 订阅。")
        og = self._require_runtime()
        self._trigger_and_evaluate(og, "ReferencePathRxTick")
        raw_sequence = self._get_attribute(
            og,
            "ReferencePathRxCounter.outputs:count",
        )
        if raw_sequence is None:
            sequence = 0
        else:
            try:
                sequence = int(raw_sequence)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "OGN initial_path Counter 未返回整数序号。"
                ) from exc
        if sequence < 0:
            raise RuntimeError("OGN initial_path Counter 返回了负序号。")

        if self._last_reference_path_sequence is None:
            self._last_reference_path_sequence = sequence
            return None
        if sequence == self._last_reference_path_sequence:
            return None
        if sequence < self._last_reference_path_sequence:
            self._last_reference_path_sequence = sequence
            return None

        try:
            parsed = parse_reference_path_outputs(
                self._get_attribute(
                    og,
                    "ROS2SubscribeReferencePath.outputs:header:stamp:sec",
                ),
                self._get_attribute(
                    og,
                    "ROS2SubscribeReferencePath.outputs:header:stamp:nanosec",
                ),
                self._get_attribute(
                    og,
                    "ROS2SubscribeReferencePath.outputs:header:frame_id",
                ),
                self._get_attribute(
                    og,
                    "ROS2SubscribeReferencePath.outputs:poses",
                ),
            )
        except (TypeError, ValueError) as exc:
            self._last_reference_path_sequence = sequence
            self._active_reference_path_stamp_ns = 0
            self._reference_path_identity_fault = f"invalid_reference_path:{exc}"
            self._navigation_gate_dirty = True
            raise
        (
            points,
            frame_id,
            stamp_sec,
            stamp_nanosec,
            points_sha256,
            terminal_yaw,
        ) = parsed
        self._last_reference_path_sequence = sequence
        sample = OgnPathSample(
            points_ground_xyz=points,
            terminal_yaw=terminal_yaw,
            source_topic=self.config.reference_path_topic,
            frame_id=frame_id,
            stamp_sec=stamp_sec,
            stamp_nanosec=stamp_nanosec,
            sequence=sequence,
            points_sha256=points_sha256,
        )
        signature = (bool(points), points_sha256)
        if sample.stamp_ns < self._latest_reference_path_stamp_ns:
            # 乱序旧 Path 不得覆盖当前 identity，也不刷新执行许可。
            return sample
        if sample.stamp_ns == self._latest_reference_path_stamp_ns:
            if (
                self._latest_reference_path_signature is not None
                and signature != self._latest_reference_path_signature
            ):
                self._active_reference_path_stamp_ns = 0
                self._reference_path_identity_fault = (
                    "conflicting_same_stamp_reference_path"
                )
                self._navigation_gate_dirty = True
            elif (
                points
                and self._latest_reference_path_signature == signature
                and self._active_reference_path_stamp_ns == 0
            ):
                # timeline hard reset 会主动失效 active identity，但
                # transient-local Path 可能以同 stamp、同 payload 重放。只有
                # 完整签名一致时才能恢复该精确代际，随后 typed freeze 才可发。
                self._active_reference_path_stamp_ns = sample.stamp_ns
                self._reference_path_identity_fault = None
                self._navigation_gate_dirty = True
            return sample
        self._latest_reference_path_stamp_ns = sample.stamp_ns
        self._latest_reference_path_signature = signature
        self._active_reference_path_stamp_ns = sample.stamp_ns if points else 0
        self._reference_path_identity_fault = None
        self._navigation_gate_dirty = True
        return sample

    def _require_runtime(self) -> Any:
        """确认图已创建后再延迟导入 OmniGraph。"""

        if self._graph is None:
            raise RuntimeError("必须先调用 setup() 创建 ROS 2 发布图。")
        if self._setup_failed:
            raise RuntimeError(
                "ROS 2 发布图的动态端口 setup() 尚未完成；"
                "必须先重试 setup()。"
            )
        og = _import_omni_graph()
        current_graph = og.Controller.graph(self.config.graph_path)
        if current_graph is None:
            self._clear_runtime_state()
            raise RuntimeError(
                "OmniGraph 已随 stage 消失；请重新调用 setup()。"
            )
        if not _timeline_is_playing():
            raise RuntimeError("timeline 未播放，拒绝触发 ROS 2 发布。")
        self._graph = current_graph
        return og

    def _set_attribute(self, og: Any, relative_path: str, value: object) -> None:
        """通过 Isaac 5.1 AttributeValueHelper 写入单个端口。"""

        helper = self._attribute_helpers.get(relative_path)
        if helper is None:
            attribute = self._lookup_attribute(og, relative_path)
            helper = og.AttributeValueHelper(attribute)
            self._attribute_helpers[relative_path] = helper
        helper.set(value)

    def _get_attribute(self, og: Any, relative_path: str) -> object:
        """读取一个 OGN 端口，并复用与写端一致的属性 helper。"""

        helper = self._attribute_helpers.get(relative_path)
        if helper is None:
            attribute = self._lookup_attribute(og, relative_path)
            helper = og.AttributeValueHelper(attribute)
            self._attribute_helpers[relative_path] = helper
        return helper.get()

    def _lookup_attribute(self, og: Any, relative_path: str) -> object:
        """查找静态端口或 generic subscriber 的动态输出端口。

        Isaac 5.1 的 generic ``ROS2Subscriber`` 会把复杂消息字段创建成
        运行期动态属性。插件日志虽已显示端口创建成功，这些属性仍不会进入
        ``Controller.attribute(绝对路径)`` 使用的 USD 对象索引；NVIDIA 自带
        subscriber 测试使用节点句柄读取动态输出。因此这里只对 Path 与
        ControllerStatus 动态输出复用同一节点句柄接口，其他静态端口继续按
        绝对路径访问。
        """

        dynamic_nodes = {
            "ROS2SubscribeReferencePath.outputs:": (
                "ROS2SubscribeReferencePath"
            ),
            "ROS2SubscribeControllerStatus.outputs:": (
                "ROS2SubscribeControllerStatus"
            ),
            "ROS2SubscribeNavigationStatus.outputs:": (
                "ROS2SubscribeNavigationStatus"
            ),
            "ROS2SubscribeGridMapDiagnostics.outputs:": (
                "ROS2SubscribeGridMapDiagnostics"
            ),
            "ROS2SubscribeBsplineDiagnostics.outputs:": (
                "ROS2SubscribeBsplineDiagnostics"
            ),
            "ROS2PublishPCTGoal.inputs:": "ROS2PublishPCTGoal",
            "ROS2PublishStairExecutionFrozen.inputs:": (
                "ROS2PublishStairExecutionFrozen"
            ),
        }
        dynamic_node = next(
            (
                node_name
                for prefix, node_name in dynamic_nodes.items()
                if relative_path.startswith(prefix)
            ),
            None,
        )
        if dynamic_node is not None:
            node_path = f"{self.config.graph_path}/{dynamic_node}"
            node = og.Controller.node(node_path)
            if hasattr(node, "is_valid") and not node.is_valid():
                raise RuntimeError(
                    f"{dynamic_node} 节点句柄无效。"
                )
            attribute_name = relative_path.split(".", 1)[1]
            return og.Controller.attribute(attribute_name, node)
        return og.Controller.attribute(
            f"{self.config.graph_path}/{relative_path}"
        )

    def _trigger_and_evaluate(self, og: Any, impulse_node: str) -> None:
        """置位独立脉冲并同步求值，避免发布频率绑定到物理 tick。"""

        self._set_attribute(og, f"{impulse_node}.state:enableImpulse", True)
        # OnImpulseEvent 会在本次 compute 内自行消费并复位脉冲；额外写 false
        # 会再请求一次图求值，不能用来处理过滤后合法为空的点云。
        og.Controller.evaluate_sync(graph_id=self._graph)

    def _clear_runtime_state(self) -> None:
        self._graph = None
        self._setup_failed = False
        self._attribute_helpers.clear()
        self._last_state_timestamp = None
        self._last_cloud_timestamp = None
        self._last_command_sequence = 0
        self._last_navigation_status_rx_sequence = 0
        self._last_navigation_status_status_sequence = None
        self._last_navigation_status_state_revision = None
        self._last_navigation_status_header_stamp_ns = None
        self._last_navigation_status_sample = None
        self._navigation_status_fault = None
        self._navigation_gate_dirty = False
        self._last_goal_reached_sequence = 0
        self._last_controller_status_rx_sequence = 0
        self._last_controller_status_status_sequence = None
        self._last_controller_status_acceptance_sequence = None
        self._last_grid_map_diagnostics_rx_sequence = 0
        self._last_grid_map_observation_sequence = None
        self._last_bspline_diagnostics_rx_sequence = 0
        self._last_bspline_diagnostic_sequence = None
        self._last_reference_path_sequence = 0
        self._latest_reference_path_stamp_ns = 0
        self._latest_reference_path_signature = None
        self._active_reference_path_stamp_ns = 0
        self._reference_path_identity_fault = None
        self._pct_goal_publish_sequence = 0
        self._last_pct_goal_stamp_ns = 0
        self._last_pct_goal_sample = None
        self._pct_goal_transport_attempt_count = 0
        self._stair_execution_frozen_writer_epoch = uuid.uuid4().hex
        self._stair_execution_frozen_publish_sequence = 0
        self._last_stair_execution_frozen_publish_timestamp = None
        self._last_stair_execution_frozen_report = None

    @staticmethod
    def _require_monotonic_timestamp(
        value: float,
        *,
        previous: float | None,
        field_name: str,
    ) -> None:
        if previous is not None and value < previous:
            raise ValueError(
                f"{field_name} 不能回退：当前 {value}，上一帧 {previous}。"
            )


def _validate_prim_path(value: str, field_name: str) -> None:
    """校验可用于 USD prim 的绝对路径。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串。")
    if not _USD_PRIM_PATH_RE.fullmatch(value):
        raise ValueError(f"{field_name} 必须是合法的绝对 USD prim 路径。")


def _validate_nonempty_text(value: str, field_name: str) -> None:
    """校验协议身份字段为无首尾空白的非空字符串。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串。")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} 必须是无首尾空白的非空字符串。")


def _validate_topic(value: str, field_name: str) -> None:
    """校验本桥接器要求的绝对 ROS 2 topic。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串。")
    if not _ROS_TOPIC_RE.fullmatch(value):
        raise ValueError(f"{field_name} 必须是合法的绝对 ROS 2 topic。")


def _validate_frame_id(value: str, field_name: str) -> None:
    """校验无前导斜杠的 ROS frame_id。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串。")
    if not _FRAME_ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} 必须是合法且无前导斜杠的 frame_id。")


def _validate_qos_profile(value: str, field_name: str) -> None:
    """校验 Isaac ROS2QoSProfile 要求的八字段 JSON。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是 JSON 字符串。")
    try:
        profile = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} 不是合法 JSON。") from exc
    expected_fields = {
        "history",
        "depth",
        "reliability",
        "durability",
        "deadline",
        "lifespan",
        "liveliness",
        "leaseDuration",
    }
    if not isinstance(profile, dict) or set(profile) != expected_fields:
        raise ValueError(f"{field_name} 必须恰好包含 Isaac QoS 八个字段。")
    if (
        isinstance(profile["depth"], bool)
        or not isinstance(profile["depth"], int)
        or profile["depth"] < 1
    ):
        raise ValueError(f"{field_name}.depth 必须是正整数。")
    for duration_name in ("deadline", "lifespan", "leaseDuration"):
        if not isinstance(profile[duration_name], float):
            raise ValueError(f"{field_name}.{duration_name} 必须是浮点数。")


def _finite_vector(
    values: Sequence[float],
    expected_size: int,
    field_name: str,
) -> tuple[float, ...]:
    """把一维数值序列转换为固定长度的有限浮点元组。"""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} 必须是数值序列。")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} 必须是数值序列。") from exc
    if len(result) != expected_size:
        raise ValueError(f"{field_name} 必须包含 {expected_size} 个元素。")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} 不能包含 NaN 或无穷值。")
    return result


def _finite_scalar(value: float, field_name: str) -> float:
    """把单个数值转换成有限浮点数。"""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是数值。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} 必须是数值。") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} 不能是 NaN 或无穷值。")
    return result


def _parse_json_object(value: object, field_name: str) -> dict[str, object]:
    """把 OGN nested-message 动态输出严格解析为 JSON 对象。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是 JSON 字符串。")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} 不是合法 JSON。") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} 必须解析为 JSON 对象。")
    return parsed


def _json_integer(value: object, field_name: str) -> int:
    """校验 JSON 字段是整数且不是布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} 必须是整数。")
    return value


def _bounded_integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """校验 OGN 标量是指定闭区间内的真整数。"""

    # OmniGraph 的动态 int64/uint64 数组在 Isaac Sim 中以 NumPy 整数标量
    # 暴露；它们仍是精确整数，先规范化为 Python int 再做位宽边界校验。
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{field_name} 必须是整数。")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{field_name} 必须位于 [{minimum}, {maximum}]。")
    return normalized


def _ros_time_parts(
    raw_sec: object,
    raw_nanosec: object,
    field_name: str,
    *,
    require_nonzero: bool = False,
) -> tuple[int, int]:
    """校验 builtin_interfaces/Time 并保留整数纳秒。"""

    sec = _bounded_integer(
        raw_sec,
        f"{field_name}.sec",
        minimum=0,
        maximum=(1 << 31) - 1,
    )
    nanosec = _bounded_integer(
        raw_nanosec,
        f"{field_name}.nanosec",
        minimum=0,
        maximum=999_999_999,
    )
    if require_nonzero and sec == 0 and nanosec == 0:
        raise ValueError(f"{field_name} 必须非零。")
    return sec, nanosec


def _strict_bool(value: object, field_name: str) -> bool:
    """拒绝会被 Python 当作布尔值的整数或字符串。"""

    if not isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是布尔值。")
    return value


def _json_finite_number(value: object, field_name: str) -> float:
    """校验 JSON 字段是有限实数且不是布尔值或数字字符串。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须是 JSON 数值。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} 不能是 NaN 或无穷值。")
    return result


def _terminal_yaw_from_json_orientation(
    orientation: dict[str, object],
    *,
    field_name: str,
) -> float:
    """单位化末 Pose 四元数，并提取可判定的 world 平面朝向。"""

    x = _json_finite_number(orientation.get("x"), f"{field_name}.x")
    y = _json_finite_number(orientation.get("y"), f"{field_name}.y")
    z = _json_finite_number(orientation.get("z"), f"{field_name}.z")
    w = _json_finite_number(orientation.get("w"), f"{field_name}.w")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError(f"{field_name} 不能是零四元数。")
    x, y, z, w = (component / norm for component in (x, y, z, w))
    heading_x = 1.0 - 2.0 * (y * y + z * z)
    heading_y = 2.0 * (x * y + w * z)
    if math.hypot(heading_x, heading_y) <= 1.0e-6:
        raise ValueError(f"{field_name} 无法确定 world 平面 terminal yaw。")
    yaw = math.atan2(heading_y, heading_x)
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def _path_points_sha256(
    points: Sequence[Sequence[float]],
) -> str:
    """按网络字节序双精度几何值计算跨进程稳定的路径摘要。"""

    digest = hashlib.sha256()
    for point in points:
        x, y, z = _finite_vector(point, 3, "path point")
        digest.update(struct.pack("!ddd", x, y, z))
    return digest.hexdigest()


def _update_kit_once() -> None:
    """同步推进一次 Kit 消息循环，使 generic OGN 动态端口完成重建。"""

    app_module = importlib.import_module("omni.kit.app")
    app_module.get_app().update()


def _import_omni_graph() -> Any:
    """仅在 Isaac runtime 路径中导入 OmniGraph。"""

    try:
        return importlib.import_module("omni.graph.core")
    except ImportError as exc:
        raise RuntimeError(
            "OmniGraph 不可用；请在 Isaac Sim 5.1 SimulationApp 启动后调用。"
        ) from exc


def _import_usdrt() -> Any:
    """仅在 compute 模式创建 target 时导入 usdrt。"""

    try:
        return importlib.import_module("usdrt")
    except ImportError as exc:
        raise RuntimeError(
            "usdrt 不可用；请在 Isaac Sim 5.1 SimulationApp 启动后调用。"
        ) from exc


def _ros2_bridge_extension_is_enabled() -> bool:
    """查询 Isaac ROS 2 bridge extension 是否已启用。"""

    try:
        kit_app = importlib.import_module("omni.kit.app")
    except ImportError as exc:
        raise RuntimeError(
            "omni.kit.app 不可用；请先启动 SimulationApp。"
        ) from exc
    manager = kit_app.get_app().get_extension_manager()
    return bool(manager.is_extension_enabled("isaacsim.ros2.bridge"))


def enable_ros2_bridge_extension() -> dict[str, object]:
    """在 ``SimulationApp`` 建立后显式启用 ROS 2 bridge。

    extension 自身声明 OmniGraph Action 依赖，这里只管理顶层
    ``isaacsim.ros2.bridge``，避免与 Kit 的依赖解析产生两套状态。
    """

    extension_name = "isaacsim.ros2.bridge"
    try:
        kit_app = importlib.import_module("omni.kit.app")
        manager = kit_app.get_app().get_extension_manager()
        enabled_before = bool(manager.is_extension_enabled(extension_name))
        if not enabled_before:
            manager.set_extension_enabled_immediate(extension_name, True)
        enabled_after = bool(manager.is_extension_enabled(extension_name))
        if not enabled_after:
            raise RuntimeError(f"扩展启用请求未生效：{extension_name}")
    except Exception as exc:
        root_cause = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "无法启用 Isaac ROS 2 OGN 必需扩展；请检查 extension 初始化顺序、"
            "Isaac ROS 2 库路径、ROS_DISTRO、RMW_IMPLEMENTATION 与 "
            f"ROS_DOMAIN_ID。原始错误：{root_cause}"
        ) from exc
    return {
        "extension": extension_name,
        "enabled_before": enabled_before,
        "enabled": enabled_after,
    }


def _timeline_is_playing() -> bool:
    """查询 Isaac timeline 是否处于播放状态。"""

    try:
        timeline_module = importlib.import_module("omni.timeline")
    except ImportError as exc:
        raise RuntimeError(
            "omni.timeline 不可用；请先启动 SimulationApp。"
        ) from exc
    return bool(timeline_module.get_timeline_interface().is_playing())


# 兼容常见的全大写 ROS 缩写命名。
IsaacROS2OgnBridge = IsaacRos2OgnBridge
IsaacROS2OgnBridgeConfig = IsaacRos2OgnBridgeConfig


__all__ = [
    "IsaacROS2OgnBridge",
    "IsaacROS2OgnBridgeConfig",
    "IsaacRos2OgnBridge",
    "IsaacRos2OgnBridgeConfig",
    "OdometrySource",
    "OgnGraphSpec",
    "OgnStairExecutionFreezePublicationReport",
    "OgnBoolSample",
    "OgnControllerStatusSample",
    "OgnGridMapObservationDiagnosticsSample",
    "OgnBsplineDiagnosticsSample",
    "OgnNavigationStatusSample",
    "OgnOdometrySample",
    "OgnTwistSample",
    "CLOCK_QOS_PROFILE",
    "CMD_VEL_QOS_PROFILE",
    "CONTROLLER_STATUS_QOS_PROFILE",
    "PLANNING_DIAGNOSTICS_QOS_PROFILE",
    "NAVIGATION_STATUS_QOS_PROFILE",
    "GOAL_REACHED_QOS_PROFILE",
    "PCT_GOAL_QOS_PROFILE",
    "REFERENCE_PATH_QOS_PROFILE",
    "SENSOR_DATA_QOS_PROFILE",
    "STAIR_EXECUTION_FROZEN_QOS_PROFILE",
    "OgnPCTGoalSample",
    "OgnPathSample",
    "build_graph_spec",
    "enable_ros2_bridge_extension",
    "parse_controller_status_outputs",
    "parse_grid_map_observation_diagnostics_outputs",
    "parse_bspline_diagnostics_outputs",
    "parse_navigation_status_outputs",
    "prepare_odometry_sample",
    "validate_point_cloud",
]
