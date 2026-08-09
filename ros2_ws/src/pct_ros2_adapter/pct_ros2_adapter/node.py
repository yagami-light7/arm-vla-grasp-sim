"""PCT 三维全局路径 ROS 2 adapter。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Callable

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMessage
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scan_planner_msgs.msg import PCTPlanningStatus
from scan_planner_msgs.srv import PCTPlanningCommand

from .backend import (
    GlobalPlannerBackend,
    PCTBackendConfig,
    PCTBackendError,
    PCTBackendPlan,
    PCTNoPathError,
    create_global_planner_backend,
    resolve_project_path,
    resolve_project_root,
    xyz_triples,
)
from .path_geometry import (
    normalize_frame_id,
    prepare_ground_path,
    quaternion_xyzw_to_yaw,
)


@dataclass(frozen=True)
class _OdometrySnapshot:
    position_xyz: tuple[float, float, float]
    stamp_ns: int


@dataclass(frozen=True)
class _NavigationGoal:
    """跨多次 PCT 重规划持续存在的导航目标快照。"""

    goal_id: int
    position_xyz: tuple[float, float, float]
    yaw: float
    stamp_ns: int
    message: PoseStamped

    @property
    def payload(self) -> tuple[float, ...]:
        """返回用于严格同目标比较的规范化位姿。"""

        return (*self.position_xyz, self.yaw)

    @property
    def identity(self) -> tuple[object, ...]:
        """返回完整 wire PoseStamped 的不可变目标身份。

        PCT backend 只消费位置和 yaw，但重规划事务必须保留原始四元数与
        frame 快照；不能把 ``q`` 与 ``-q`` 等几何等价 payload 当作同一
        wire identity，否则跨节点精确重试无法审计。
        """

        pose = self.message.pose
        return (
            self.stamp_ns,
            str(self.message.header.frame_id).strip().lstrip("/"),
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )


@dataclass(frozen=True)
class _PlanRequest:
    """一次会产生新 Path 代际的 PLAN 或 REPLAN 请求。"""

    plan_id: int
    request_id: int
    command: int
    command_stamp_ns: int
    invalidation_stamp_ns: int
    goal: _NavigationGoal


@dataclass(frozen=True)
class _PlanningJob:
    request: _PlanRequest
    start_position_xyz: tuple[float, float, float]
    start_stamp_ns: int
    cancel_event: threading.Event


@dataclass(frozen=True)
class _CommandAck:
    """缓存最近一次已接受命令的幂等 ACK。"""

    plan_id: int
    goal_id: int
    request_id: int
    tombstone_stamp_ns: int
    message: str


class _FutureTimestampError(ValueError):
    """消息时间明显超前于当前 ROS 仿真时间。"""


class PCTROS2Adapter(Node):
    """把 Odometry 和 PoseStamped 目标转换为 PCT 地面高度 Path。"""

    def __init__(
        self,
        *,
        backend: GlobalPlannerBackend | None = None,
        backend_factory: Callable[[PCTBackendConfig], GlobalPlannerBackend] = (
            create_global_planner_backend
        ),
    ) -> None:
        super().__init__("pct_ros2_adapter")
        self._enable_sim_time()
        self._declare_parameters()
        self._world_frame = normalize_frame_id(
            self.get_parameter("frames.world").value,
            field_name="frames.world",
        )
        self._base_frame = normalize_frame_id(
            self.get_parameter("frames.base").value,
            field_name="frames.base",
        )
        self._odometry_timeout_sec = self._positive_parameter(
            "planning.odometry_timeout_sec"
        )
        self._future_tolerance_sec = self._nonnegative_parameter(
            "planning.odometry_future_tolerance_sec"
        )
        self._goal_future_tolerance_sec = self._nonnegative_parameter(
            "planning.goal_future_tolerance_sec"
        )
        self._maximum_start_drift_m = self._positive_parameter(
            "planning.maximum_start_drift_m"
        )
        self._minimum_path_spacing_m = self._positive_parameter(
            "path.minimum_point_spacing_m"
        )
        qos_depth = self._positive_int_parameter("qos.depth")
        path_depth = self._positive_int_parameter("qos.path_depth")
        status_depth = self._positive_int_parameter("qos.status_depth")

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        goal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=path_depth,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=status_depth,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        path_output_topic = str(self.get_parameter("topics.path_output").value)
        scan_path_output_topic = str(
            self.get_parameter("topics.scan_path_output").value
        )
        if self.resolve_topic_name(path_output_topic) == self.resolve_topic_name(
            scan_path_output_topic
        ):
            raise ValueError(
                "topics.path_output 与 topics.scan_path_output 不能相同"
            )
        self._path_publisher = self.create_publisher(
            PathMessage,
            path_output_topic,
            path_qos,
        )
        self._scan_path_publisher = self.create_publisher(
            PathMessage,
            scan_path_output_topic,
            path_qos,
        )
        self._status_publisher = self.create_publisher(
            PCTPlanningStatus,
            str(self.get_parameter("topics.status_output").value),
            status_qos,
        )
        self._odometry_subscription = self.create_subscription(
            Odometry,
            str(self.get_parameter("topics.odometry_input").value),
            self._odometry_callback,
            sensor_qos,
        )
        self._goal_subscription = self.create_subscription(
            PoseStamped,
            str(self.get_parameter("topics.goal_input").value),
            self._goal_callback,
            goal_qos,
        )
        self._command_service = self.create_service(
            PCTPlanningCommand,
            str(self.get_parameter("topics.command_service").value),
            self._command_callback,
        )

        self._backend = backend
        self._backend_factory = backend_factory
        self._backend_config = (
            None if backend is not None else self._read_backend_config()
        )
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pct-global-planner",
        )
        # upstream backend 的 tomogram、collision PLY 与 native core 都是
        # 只读静态资产。节点启动后立即在同一个串行 worker 中预热，使约两秒
        # 的加载与 Isaac 场景启动重叠；真正的起终点搜索仍只在 PLAN 后执行。
        self._backend_preload_future: Future[GlobalPlannerBackend] | None = None
        if self._backend is None:
            if self._backend_config is None:
                raise PCTBackendError("PCT backend 配置未初始化")
            self._backend_preload_future = self._worker.submit(
                self._backend_factory,
                self._backend_config,
            )
        self._future: Future[PCTBackendPlan] | None = None
        self._future_job: _PlanningJob | None = None
        self._latest_odometry: _OdometrySnapshot | None = None
        self._navigation_goal: _NavigationGoal | None = None
        self._pending_plan: _PlanRequest | None = None
        self._latest_goal_stamp_ns = 0
        self._latest_goal_payload: tuple[float, ...] | None = None
        self._latest_command_request_id = 0
        self._last_command_signature: tuple[object, ...] | None = None
        self._last_command_ack: _CommandAck | None = None
        self._plan_id = 0
        self._last_path_stamp_ns = 0
        self._active_path_stamp_ns = 0
        self._last_status_key: tuple[object, ...] | None = None
        self._startup_generation_published = False
        self._startup_timer = self.create_timer(
            0.02,
            self._publish_startup_generation_if_ready,
        )
        self._poll_timer = self.create_timer(0.02, self._poll_planning_result)

        self.get_logger().info(
            "PCT ROS 2 adapter 已启动：输入为 base 位姿，"
            "输出 Path z 为 collision PLY 地面高度"
        )

    def _enable_sim_time(self) -> None:
        """强制所有规划代际使用 ROS 仿真时钟。"""

        result = self.set_parameters(
            [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
        )[0]
        if not result.successful:
            raise RuntimeError(f"无法启用 use_sim_time：{result.reason}")

    def _declare_parameters(self) -> None:
        """声明 topic、frame、资产、坐标和规划安全参数。"""

        self.declare_parameter("topics.odometry_input", "/body_pose")
        self.declare_parameter("topics.goal_input", "/pct/goal")
        self.declare_parameter(
            "topics.command_service",
            "/pct/planning_command",
        )
        self.declare_parameter("topics.path_output", "/pct/global_path")
        self.declare_parameter("topics.scan_path_output", "/initial_path")
        self.declare_parameter("topics.status_output", "/pct/planning_status")
        self.declare_parameter("frames.world", "world")
        self.declare_parameter("frames.base", "base_link")
        self.declare_parameter("qos.depth", 5)
        self.declare_parameter("qos.path_depth", 1)
        self.declare_parameter("qos.status_depth", 1)
        self.declare_parameter("planning.odometry_timeout_sec", 0.50)
        self.declare_parameter("planning.odometry_future_tolerance_sec", 0.20)
        self.declare_parameter("planning.goal_future_tolerance_sec", 0.20)
        self.declare_parameter("planning.maximum_start_drift_m", 0.20)
        self.declare_parameter("path.minimum_point_spacing_m", 0.05)
        self.declare_parameter("planner.project_root", "")
        # 直接构造节点的测试默认保留 compatible；生产 YAML 与组合 launch
        # 显式选择 upstream，任何初始化错误都由 selector 原样失败。
        self.declare_parameter("planner.backend_kind", "compatible")
        self.declare_parameter(
            "planner.upstream_source_root",
            "external/PCT_planner",
        )
        self.declare_parameter("planner.upstream_use_quintic", True)
        self.declare_parameter("planner.upstream_max_heading_rate", 10.0)
        self.declare_parameter("planner.upstream_astar_step_cost_weight", 0.20)
        self.declare_parameter(
            "planner.upstream_body_clearance_enabled",
            False,
        )
        self.declare_parameter(
            "planner.upstream_body_clearance_radius_m",
            0.80,
        )
        self.declare_parameter(
            "planner.upstream_body_clearance_maximum_cost",
            20.0,
        )
        self.declare_parameter(
            "planner.upstream_body_clearance_power",
            2.0,
        )
        self.declare_parameter(
            "planner.upstream_endpoint_layer_max_z_error_m",
            0.25,
        )
        self.declare_parameter(
            "planner.upstream_same_layer_shortcut_clearance_m",
            0.27,
        )
        self.declare_parameter(
            "planner.upstream_same_layer_shortcut_max_segment_m",
            10.0,
        )
        self.declare_parameter("planner.upstream_stair_profile_path", "")
        self.declare_parameter(
            "planner.upstream_stair_profile_match_tolerance_m",
            0.60,
        )
        self.declare_parameter(
            "planner.tomogram_path",
            "source/scene/multifloor/mutifloor.pickle",
        )
        self.declare_parameter(
            "planner.upstream_tomogram_path",
            "source/scene/multifloor/mutifloor_upstream.pickle",
        )
        self.declare_parameter(
            "planner.walkable_path",
            "source/scene/multifloor/mutifloor_ply_walkable.npy",
        )
        self.declare_parameter(
            "planner.collision_ply_path",
            "source/scene/multifloor/ply/3dgs_collision.ply",
        )
        self.declare_parameter("planner.coord_mode", "sim_to_pct_180deg")
        self.declare_parameter("planner.pct_offset_x", 0.0)
        self.declare_parameter("planner.pct_offset_y", 0.0)
        self.declare_parameter("planner.pct_offset_z", 0.0)
        self.declare_parameter("planner.pct_scale_x", 1.0)
        self.declare_parameter("planner.pct_scale_y", 1.0)
        self.declare_parameter("planner.pct_scale_z", 1.0)
        self.declare_parameter("planner.pct_rotation_x_rad", 0.0)
        self.declare_parameter("planner.pct_rotation_y_rad", 0.0)
        self.declare_parameter("planner.pct_rotation_z_rad", 0.0)
        # slice 查询和端点落地默认共享当前待实跑标定的高度合同。
        self.declare_parameter("planner.slice_query_root_to_floor_m", 0.30)
        self.declare_parameter("planner.goal_base_to_ground_m", 0.30)
        self.declare_parameter("planner.global_vertical_obstacle_min_slices", 7)
        self.declare_parameter(
            "planner.cross_floor_vertical_obstacle_min_slices",
            9,
        )
        self.declare_parameter(
            "planner.cross_floor_gateway_points_xyz",
            [1.5, 5.7, 0.6],
        )
        self.declare_parameter(
            "planner.cross_floor_stair_exit_points_xyz",
            [2.70, 7.05, 3.0],
        )
        self.declare_parameter(
            "planner.cross_floor_stair_midpoint_points_xyz",
            [
                1.51822,
                6.27683,
                0.29486,
                2.74512,
                9.14634,
                1.64666,
                1.9202,
                9.52807,
                1.71919,
                2.69841,
                7.79872,
                2.61031,
            ],
        )
        self.declare_parameter("planner.cross_floor_gateway_radius_m", 0.60)
        self.declare_parameter("planner.body_obstacle_min_height_m", 0.30)
        self.declare_parameter("planner.body_obstacle_max_height_m", 1.00)
        self.declare_parameter(
            "planner.stair_min_horizontal_per_slice_m",
            0.40,
        )
        self.declare_parameter(
            "planner.stair_max_horizontal_per_slice_m",
            0.90,
        )
        self.declare_parameter("planner.stair_vertical_radius_m", 0.60)
        self.declare_parameter("planner.stair_progress_tolerance", 0.35)
        self.declare_parameter("planner.stair_progress_cost_weight", 20.0)
        self.declare_parameter("planner.obstacle_clearance_radius_m", 0.60)
        self.declare_parameter("planner.obstacle_clearance_cost_weight", 2.0)
        self.declare_parameter("planner.ground_projection_max_z_error_m", 0.60)
        self.declare_parameter(
            "planner.start_ground_projection_max_z_error_m",
            0.08,
        )
        self.declare_parameter(
            "planner.terminal_ground_projection_max_z_error_m",
            0.08,
        )
        self.declare_parameter("planner.path_sample_spacing_m", 0.20)
        self.declare_parameter("planner.maximum_snap_distance_m", 0.25)
        self.declare_parameter("planner.grid_max_expansions", 1_500_000)
        self.declare_parameter("planner.grid_compress_max_segment_m", 0.80)
        self.declare_parameter("planner.grid_timeout_sec", 10.0)

    def _read_backend_config(self) -> PCTBackendConfig:
        """解析仓库相对资产；大地图仍延迟到规划 worker 中加载。"""

        project_root = resolve_project_root(
            str(self.get_parameter("planner.project_root").value)
        )
        backend_kind = str(
            self.get_parameter("planner.backend_kind").value
        ).strip().lower()
        tomogram_parameter = (
            "planner.upstream_tomogram_path"
            if backend_kind == "upstream"
            else "planner.tomogram_path"
        )
        tomogram = resolve_project_path(
            project_root,
            str(self.get_parameter(tomogram_parameter).value),
            field_name=tomogram_parameter,
        )
        walkable = resolve_project_path(
            project_root,
            str(self.get_parameter("planner.walkable_path").value),
            field_name="planner.walkable_path",
        )
        collision = resolve_project_path(
            project_root,
            str(self.get_parameter("planner.collision_ply_path").value),
            field_name="planner.collision_ply_path",
        )
        upstream_source_value = str(
            self.get_parameter("planner.upstream_source_root").value
        ).strip()
        if not upstream_source_value:
            raise ValueError("planner.upstream_source_root 不能为空")
        upstream_source_path = Path(upstream_source_value).expanduser()
        if not upstream_source_path.is_absolute():
            upstream_source_path = project_root / upstream_source_path
        stair_profile_value = str(
            self.get_parameter("planner.upstream_stair_profile_path").value
        ).strip()
        stair_profile_path = None
        if stair_profile_value:
            stair_profile_path = resolve_project_path(
                project_root,
                stair_profile_value,
                field_name="planner.upstream_stair_profile_path",
            )
        return PCTBackendConfig(
            project_root=project_root,
            tomogram_path=tomogram,
            walkable_path=walkable,
            collision_ply_path=collision,
            backend_kind=backend_kind,
            upstream_source_root=upstream_source_path.resolve(),
            upstream_use_quintic=bool(
                self.get_parameter("planner.upstream_use_quintic").value
            ),
            upstream_max_heading_rate=self._positive_parameter(
                "planner.upstream_max_heading_rate"
            ),
            upstream_astar_step_cost_weight=self._positive_parameter(
                "planner.upstream_astar_step_cost_weight"
            ),
            upstream_body_clearance_enabled=bool(
                self.get_parameter(
                    "planner.upstream_body_clearance_enabled"
                ).value
            ),
            upstream_body_clearance_radius_m=self._positive_parameter(
                "planner.upstream_body_clearance_radius_m"
            ),
            upstream_body_clearance_maximum_cost=self._positive_parameter(
                "planner.upstream_body_clearance_maximum_cost"
            ),
            upstream_body_clearance_power=self._positive_parameter(
                "planner.upstream_body_clearance_power"
            ),
            upstream_endpoint_layer_max_z_error_m=self._positive_parameter(
                "planner.upstream_endpoint_layer_max_z_error_m"
            ),
            upstream_same_layer_shortcut_clearance_m=(
                self._nonnegative_parameter(
                    "planner.upstream_same_layer_shortcut_clearance_m"
                )
            ),
            upstream_same_layer_shortcut_max_segment_m=(
                self._positive_parameter(
                    "planner.upstream_same_layer_shortcut_max_segment_m"
                )
            ),
            upstream_stair_profile_path=stair_profile_path,
            upstream_stair_profile_match_tolerance_m=(
                self._positive_parameter(
                    "planner.upstream_stair_profile_match_tolerance_m"
                )
            ),
            coord_mode=str(self.get_parameter("planner.coord_mode").value),
            pct_offset_x=self._finite_parameter("planner.pct_offset_x"),
            pct_offset_y=self._finite_parameter("planner.pct_offset_y"),
            pct_offset_z=self._finite_parameter("planner.pct_offset_z"),
            pct_scale_x=self._nonzero_parameter("planner.pct_scale_x"),
            pct_scale_y=self._nonzero_parameter("planner.pct_scale_y"),
            pct_scale_z=self._nonzero_parameter("planner.pct_scale_z"),
            pct_rotation_x_rad=self._finite_parameter(
                "planner.pct_rotation_x_rad"
            ),
            pct_rotation_y_rad=self._finite_parameter(
                "planner.pct_rotation_y_rad"
            ),
            pct_rotation_z_rad=self._finite_parameter(
                "planner.pct_rotation_z_rad"
            ),
            slice_query_root_to_floor_m=self._nonnegative_parameter(
                "planner.slice_query_root_to_floor_m"
            ),
            goal_base_to_ground_m=self._positive_parameter(
                "planner.goal_base_to_ground_m"
            ),
            global_vertical_obstacle_min_slices=self._positive_int_parameter(
                "planner.global_vertical_obstacle_min_slices"
            ),
            cross_floor_vertical_obstacle_min_slices=(
                self._positive_int_parameter(
                    "planner.cross_floor_vertical_obstacle_min_slices"
                )
            ),
            cross_floor_gateway_points=xyz_triples(
                self.get_parameter(
                    "planner.cross_floor_gateway_points_xyz"
                ).value,
                field_name="planner.cross_floor_gateway_points_xyz",
            ),
            cross_floor_stair_exit_points=xyz_triples(
                self.get_parameter(
                    "planner.cross_floor_stair_exit_points_xyz"
                ).value,
                field_name="planner.cross_floor_stair_exit_points_xyz",
            ),
            cross_floor_stair_midpoint_points=xyz_triples(
                self.get_parameter(
                    "planner.cross_floor_stair_midpoint_points_xyz"
                ).value,
                field_name="planner.cross_floor_stair_midpoint_points_xyz",
            ),
            cross_floor_gateway_radius_m=self._nonnegative_parameter(
                "planner.cross_floor_gateway_radius_m"
            ),
            body_obstacle_min_height_m=self._nonnegative_parameter(
                "planner.body_obstacle_min_height_m"
            ),
            body_obstacle_max_height_m=self._positive_parameter(
                "planner.body_obstacle_max_height_m"
            ),
            stair_min_horizontal_per_slice_m=self._positive_parameter(
                "planner.stair_min_horizontal_per_slice_m"
            ),
            stair_max_horizontal_per_slice_m=self._positive_parameter(
                "planner.stair_max_horizontal_per_slice_m"
            ),
            stair_vertical_radius_m=self._positive_parameter(
                "planner.stair_vertical_radius_m"
            ),
            stair_progress_tolerance=self._nonnegative_parameter(
                "planner.stair_progress_tolerance"
            ),
            stair_progress_cost_weight=self._nonnegative_parameter(
                "planner.stair_progress_cost_weight"
            ),
            obstacle_clearance_radius_m=self._nonnegative_parameter(
                "planner.obstacle_clearance_radius_m"
            ),
            obstacle_clearance_cost_weight=self._nonnegative_parameter(
                "planner.obstacle_clearance_cost_weight"
            ),
            ground_projection_max_z_error_m=self._positive_parameter(
                "planner.ground_projection_max_z_error_m"
            ),
            start_ground_projection_max_z_error_m=self._positive_parameter(
                "planner.start_ground_projection_max_z_error_m"
            ),
            terminal_ground_projection_max_z_error_m=(
                self._positive_parameter(
                    "planner.terminal_ground_projection_max_z_error_m"
                )
            ),
            path_sample_spacing_m=self._positive_parameter(
                "planner.path_sample_spacing_m"
            ),
            maximum_snap_distance_m=self._nonnegative_parameter(
                "planner.maximum_snap_distance_m"
            ),
            grid_max_expansions=self._positive_int_parameter(
                "planner.grid_max_expansions"
            ),
            grid_compress_max_segment_m=self._positive_parameter(
                "planner.grid_compress_max_segment_m"
            ),
            grid_timeout_sec=self._positive_parameter(
                "planner.grid_timeout_sec"
            ),
        )

    def _odometry_callback(self, message: Odometry) -> None:
        """缓存单调、新鲜且 frame 正确的 base Odometry。"""

        try:
            snapshot = self._normalize_odometry(message)
        except ValueError as exc:
            self.get_logger().warning(f"拒绝非法 PCT Odometry：{exc}")
            return
        now_ns = self._clock_now_ns()
        maximum_future_ns = int(
            self._future_tolerance_sec * 1_000_000_000
        )
        if now_ns <= 0 or snapshot.stamp_ns > now_ns + maximum_future_ns:
            self.get_logger().warning(
                "拒绝超前于当前 ROS 仿真时间的 PCT Odometry"
            )
            return
        if (
            self._latest_odometry is not None
            and snapshot.stamp_ns < self._latest_odometry.stamp_ns
        ):
            self.get_logger().warning("拒绝时间倒退的 PCT Odometry")
            return
        self._latest_odometry = snapshot
        self._schedule_active_goal_if_idle()

    def _goal_callback(self, message: PoseStamped) -> None:
        """保留旧 PoseStamped 入口，并映射为 request_id=0 的 PLAN。"""

        try:
            position, yaw, stamp_ns = self._normalize_goal(message)
        except _FutureTimestampError as exc:
            # 仍需撤销旧运动意图，但新代际只取可信 ROS 当前时间，绝不使用
            # wall-time goal stamp。
            self._invalidate_rejected_goal(f"未来时间域目标：{exc}")
            return
        except ValueError as exc:
            self._invalidate_rejected_goal(f"非法目标：{exc}")
            return
        payload = (*position, yaw)
        if stamp_ns < self._latest_goal_stamp_ns:
            self.get_logger().warning("忽略时间戳早于当前代际的 PCT goal")
            return
        if stamp_ns == self._latest_goal_stamp_ns:
            if payload == self._latest_goal_payload:
                self.get_logger().info("忽略重复的 PCT goal 消息")
                return
            # 相同 stamp 是跨节点唯一代际；同代不同 payload 必须使旧路径
            # 失效，且该 stamp 后续不能靠任一 payload 重新复活。
            self._latest_goal_payload = None
            self._invalidate_rejected_goal("同 stamp 的 PCT goal payload 冲突")
            return
        self._latest_goal_stamp_ns = stamp_ns
        self._latest_goal_payload = payload
        goal = _NavigationGoal(
            goal_id=stamp_ns,
            position_xyz=position,
            yaw=yaw,
            stamp_ns=stamp_ns,
            message=deepcopy(message),
        )
        self._navigation_goal = goal
        self._accept_plan_request(
            goal=goal,
            request_id=0,
            command=PCTPlanningStatus.COMMAND_PLAN,
            command_stamp_ns=stamp_ns,
        )

    def _command_callback(
        self,
        request: PCTPlanningCommand.Request,
        response: PCTPlanningCommand.Response,
    ) -> PCTPlanningCommand.Response:
        """快速接收 typed PLAN/REPLAN/CANCEL，规划结果继续异步发布。"""

        command = int(request.command)
        goal_id = int(request.goal_id)
        request_id = int(request.request_id)
        raw_signature = self._raw_command_signature(request)

        if request_id > 0 and request_id < self._latest_command_request_id:
            return self._fill_command_response(
                response,
                disposition=PCTPlanningCommand.Response.DISPOSITION_STALE,
                goal_id=goal_id,
                request_id=request_id,
                message="request_id 早于最近已接受命令",
            )
        if request_id > 0 and request_id == self._latest_command_request_id:
            if raw_signature == self._last_command_signature:
                ack = self._last_command_ack
                if ack is None:
                    return self._fill_command_response(
                        response,
                        disposition=(
                            PCTPlanningCommand.Response.DISPOSITION_CONFLICT
                        ),
                        goal_id=goal_id,
                        request_id=request_id,
                        message="命令缓存缺失，拒绝不确定的同 request_id 请求",
                    )
                return self._fill_command_response(
                    response,
                    disposition=(
                        PCTPlanningCommand.Response.DISPOSITION_DUPLICATE
                    ),
                    goal_id=ack.goal_id,
                    request_id=ack.request_id,
                    plan_id=ack.plan_id,
                    tombstone_stamp_ns=ack.tombstone_stamp_ns,
                    message=ack.message,
                )
            return self._fill_command_response(
                response,
                disposition=PCTPlanningCommand.Response.DISPOSITION_CONFLICT,
                goal_id=goal_id,
                request_id=request_id,
                message="同 request_id 出现不同 command payload",
            )

        try:
            command_stamp_ns = self._normalize_command_header(request)
            if goal_id <= 0 or request_id <= 0:
                raise ValueError("goal_id 与 request_id 必须为正整数")
            expected_path_stamp_ns = _optional_stamp_to_nanoseconds(
                request.expected_path_stamp
            )
            if command not in {
                PCTPlanningCommand.Request.COMMAND_PLAN,
                PCTPlanningCommand.Request.COMMAND_REPLAN,
                PCTPlanningCommand.Request.COMMAND_CANCEL,
            }:
                raise ValueError("command 不是 PLAN、REPLAN 或 CANCEL")
            normalized_goal = None
            if command != PCTPlanningCommand.Request.COMMAND_CANCEL:
                position, yaw, goal_stamp_ns = self._normalize_goal(request.goal)
                normalized_goal = _NavigationGoal(
                    goal_id=goal_id,
                    position_xyz=position,
                    yaw=yaw,
                    stamp_ns=goal_stamp_ns,
                    message=deepcopy(request.goal),
                )
        except (ValueError, _FutureTimestampError) as exc:
            return self._fill_command_response(
                response,
                disposition=PCTPlanningCommand.Response.DISPOSITION_REJECTED,
                goal_id=goal_id,
                request_id=request_id,
                message=f"拒绝非法 PCT command：{exc}",
            )

        if command == PCTPlanningCommand.Request.COMMAND_PLAN:
            if expected_path_stamp_ns != 0:
                return self._fill_command_response(
                    response,
                    disposition=(
                        PCTPlanningCommand.Response.DISPOSITION_REJECTED
                    ),
                    goal_id=goal_id,
                    request_id=request_id,
                    message="PLAN 的 expected_path_stamp 必须为零",
                )
            assert normalized_goal is not None
            active = self._navigation_goal
            if active is not None and active.goal_id == goal_id:
                if active.identity != normalized_goal.identity:
                    disposition = (
                        PCTPlanningCommand.Response.DISPOSITION_CONFLICT
                    )
                    text = "同 goal_id 不能绑定不同目标位姿"
                else:
                    disposition = (
                        PCTPlanningCommand.Response.DISPOSITION_REJECTED
                    )
                    text = "同一活动目标重算必须使用 REPLAN"
                return self._fill_command_response(
                    response,
                    disposition=disposition,
                    goal_id=goal_id,
                    request_id=request_id,
                    message=text,
                )
            self._navigation_goal = normalized_goal
            plan = self._accept_plan_request(
                goal=normalized_goal,
                request_id=request_id,
                command=PCTPlanningStatus.COMMAND_PLAN,
                command_stamp_ns=command_stamp_ns,
            )
            ack_text = "PCT PLAN 已接受并发布 Path tombstone"
            ack = _CommandAck(
                plan_id=plan.plan_id,
                goal_id=goal_id,
                request_id=request_id,
                tombstone_stamp_ns=plan.invalidation_stamp_ns,
                message=ack_text,
            )
        else:
            active = self._navigation_goal
            if active is None:
                return self._fill_command_response(
                    response,
                    disposition=(
                        PCTPlanningCommand.Response.DISPOSITION_REJECTED
                    ),
                    goal_id=goal_id,
                    request_id=request_id,
                    message="当前没有可重规划或取消的活动目标",
                )
            if active.goal_id != goal_id:
                return self._fill_command_response(
                    response,
                    disposition=(
                        PCTPlanningCommand.Response.DISPOSITION_REJECTED
                    ),
                    goal_id=goal_id,
                    request_id=request_id,
                    message="goal_id 与当前活动目标不一致",
                )
            if (
                expected_path_stamp_ns <= 0
                or expected_path_stamp_ns != self._active_path_stamp_ns
            ):
                return self._fill_command_response(
                    response,
                    disposition=PCTPlanningCommand.Response.DISPOSITION_STALE,
                    goal_id=goal_id,
                    request_id=request_id,
                    message="expected_path_stamp 不是当前活动 Path 代际",
                )

            if command == PCTPlanningCommand.Request.COMMAND_REPLAN:
                assert normalized_goal is not None
                if normalized_goal.identity != active.identity:
                    return self._fill_command_response(
                        response,
                        disposition=(
                            PCTPlanningCommand.Response.DISPOSITION_CONFLICT
                        ),
                        goal_id=goal_id,
                        request_id=request_id,
                        message="REPLAN 必须保持同一 goal_id 的完整目标快照",
                    )
                plan = self._accept_plan_request(
                    goal=active,
                    request_id=request_id,
                    command=PCTPlanningStatus.COMMAND_REPLAN,
                    command_stamp_ns=command_stamp_ns,
                )
                ack_text = "PCT REPLAN 已接受并发布 Path tombstone"
                ack = _CommandAck(
                    plan_id=plan.plan_id,
                    goal_id=goal_id,
                    request_id=request_id,
                    tombstone_stamp_ns=plan.invalidation_stamp_ns,
                    message=ack_text,
                )
            else:
                tombstone_stamp_ns = self._accept_cancel_command(
                    goal_id=goal_id,
                    request_id=request_id,
                    command_stamp_ns=command_stamp_ns,
                )
                ack_text = "PCT CANCEL 已接受，活动目标与 Path 已清除"
                ack = _CommandAck(
                    plan_id=self._plan_id,
                    goal_id=goal_id,
                    request_id=request_id,
                    tombstone_stamp_ns=tombstone_stamp_ns,
                    message=ack_text,
                )

        self._latest_command_request_id = request_id
        self._last_command_signature = raw_signature
        self._last_command_ack = ack
        return self._fill_command_response(
            response,
            disposition=PCTPlanningCommand.Response.DISPOSITION_ACCEPTED,
            goal_id=ack.goal_id,
            request_id=ack.request_id,
            plan_id=ack.plan_id,
            tombstone_stamp_ns=ack.tombstone_stamp_ns,
            message=ack.message,
        )

    def _raw_command_signature(
        self,
        request: PCTPlanningCommand.Request,
    ) -> tuple[object, ...]:
        """保留完整 wire payload，用于辨别精确重试和同 ID 冲突。"""

        goal = request.goal
        pose = goal.pose
        return (
            int(request.header.stamp.sec),
            int(request.header.stamp.nanosec),
            str(request.header.frame_id),
            int(request.command),
            int(request.goal_id),
            int(request.request_id),
            int(goal.header.stamp.sec),
            int(goal.header.stamp.nanosec),
            str(goal.header.frame_id),
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
            int(request.expected_path_stamp.sec),
            int(request.expected_path_stamp.nanosec),
            str(request.reason),
        )

    def _normalize_command_header(
        self,
        request: PCTPlanningCommand.Request,
    ) -> int:
        """校验 command header 的 world frame 与连续仿真时间。"""

        frame = normalize_frame_id(
            request.header.frame_id,
            field_name="command header.frame_id",
        )
        if frame != self._world_frame:
            raise ValueError(
                f"command frame 必须是 {self._world_frame}，收到 {frame}"
            )
        stamp_ns = _stamp_to_nanoseconds(request.header.stamp)
        now_ns = self._clock_now_ns()
        maximum_future_ns = int(
            self._goal_future_tolerance_sec * 1_000_000_000
        )
        if now_ns <= 0:
            raise _FutureTimestampError("ROS 仿真时间尚未开始")
        if stamp_ns > now_ns + maximum_future_ns:
            raise _FutureTimestampError(
                "command stamp 超过当前 ROS 时间允许的 future tolerance"
            )
        return stamp_ns

    def _fill_command_response(
        self,
        response: PCTPlanningCommand.Response,
        *,
        disposition: int,
        goal_id: int,
        request_id: int,
        message: str,
        plan_id: int = 0,
        tombstone_stamp_ns: int = 0,
    ) -> PCTPlanningCommand.Response:
        """填写 ACK 与当前活动目标快照，不改变规划状态。"""

        event_stamp_ns = max(self._clock_now_ns(), int(tombstone_stamp_ns))
        if event_stamp_ns > 0:
            response.header.stamp = _time_from_nanoseconds(event_stamp_ns)
        response.header.frame_id = self._world_frame
        response.disposition = int(disposition)
        response.plan_id = int(plan_id)
        response.goal_id = int(goal_id)
        response.request_id = int(request_id)
        response.tombstone_stamp = _optional_time_from_nanoseconds(
            tombstone_stamp_ns
        )
        active = self._navigation_goal
        response.has_active_goal = active is not None
        if active is not None:
            response.active_goal = deepcopy(active.message)
        response.active_path_stamp = _optional_time_from_nanoseconds(
            self._active_path_stamp_ns if active is not None else 0
        )
        response.message = str(message)
        return response

    def _finish_new_command_response(
        self,
        response: PCTPlanningCommand.Response,
        *,
        raw_signature: tuple[object, ...],
        disposition: int,
        goal_id: int,
        request_id: int,
        message: str,
        plan_id: int = 0,
        tombstone_stamp_ns: int = 0,
    ) -> PCTPlanningCommand.Response:
        """缓存新 request 的结果，使拒绝请求的精确重试同样幂等。"""

        if request_id > 0:
            self._latest_command_request_id = int(request_id)
            self._last_command_signature = raw_signature
            self._last_command_ack = _CommandAck(
                plan_id=int(plan_id),
                goal_id=int(goal_id),
                request_id=int(request_id),
                tombstone_stamp_ns=int(tombstone_stamp_ns),
                message=str(message),
            )
        return self._fill_command_response(
            response,
            disposition=disposition,
            goal_id=goal_id,
            request_id=request_id,
            plan_id=plan_id,
            tombstone_stamp_ns=tombstone_stamp_ns,
            message=message,
        )

    def _accept_plan_request(
        self,
        *,
        goal: _NavigationGoal,
        request_id: int,
        command: int,
        command_stamp_ns: int,
    ) -> _PlanRequest:
        """先发布严格更新 tombstone，再排队执行一次 PCT 规划。"""

        self._plan_id += 1
        invalidation_stamp_ns = self._next_path_stamp_ns(
            max(
                self._clock_now_ns(),
                command_stamp_ns,
                goal.stamp_ns,
                self._active_path_stamp_ns,
            )
        )
        plan = _PlanRequest(
            plan_id=self._plan_id,
            request_id=int(request_id),
            command=int(command),
            command_stamp_ns=int(command_stamp_ns),
            invalidation_stamp_ns=invalidation_stamp_ns,
            goal=goal,
        )
        self._pending_plan = plan
        self._active_path_stamp_ns = invalidation_stamp_ns
        self._publish_empty_path(invalidation_stamp_ns)
        self._mark_startup_generation_published()
        if self._future is not None:
            self._request_backend_cancellation()
            self._publish_plan_status(
                plan,
                state=PCTPlanningStatus.PLANNING,
                text="新命令已覆盖旧规划，等待当前 PCT 调用退出",
                path_stamp_ns=invalidation_stamp_ns,
                point_count=0,
                event_stamp_ns=invalidation_stamp_ns,
            )
        else:
            self._schedule_active_goal_if_idle()
        return plan

    def _accept_cancel_command(
        self,
        *,
        goal_id: int,
        request_id: int,
        command_stamp_ns: int,
    ) -> int:
        """立即取消活动目标并发布不可被旧 worker 复活的 tombstone。"""

        goal = self._navigation_goal
        goal_stamp_ns = 0 if goal is None else goal.stamp_ns
        tombstone_stamp_ns = self._next_path_stamp_ns(
            max(
                self._clock_now_ns(),
                command_stamp_ns,
                goal_stamp_ns,
                self._active_path_stamp_ns,
            )
        )
        self._pending_plan = None
        self._navigation_goal = None
        self._active_path_stamp_ns = 0
        self._request_backend_cancellation()
        self._publish_empty_path(tombstone_stamp_ns)
        self._mark_startup_generation_published()
        self._publish_status(
            plan_id=self._plan_id,
            goal_id=goal_id,
            request_id=request_id,
            command=PCTPlanningStatus.COMMAND_CANCEL,
            state=PCTPlanningStatus.IDLE,
            text="PCT 活动目标已取消",
            goal_stamp_ns=goal_stamp_ns,
            path_stamp_ns=tombstone_stamp_ns,
            point_count=0,
            event_stamp_ns=tombstone_stamp_ns,
        )
        return tombstone_stamp_ns

    def _invalidate_rejected_goal(self, reason: str) -> None:
        """用可信 ROS 当前时间撤销旧目标，并拒绝发布旧 worker 结果。"""

        event_stamp_ns = self._clock_now_ns()
        if event_stamp_ns <= 0 and self._last_path_stamp_ns <= 0:
            self.get_logger().error(
                f"拒绝 PCT goal，仿真时间仍为零：{reason}"
            )
            return
        self._plan_id += 1
        plan_id = self._plan_id
        invalidation_stamp_ns = self._next_path_stamp_ns(
            max(
                event_stamp_ns,
                self._last_path_stamp_ns,
                self._latest_goal_stamp_ns,
            )
        )
        # 非法新目标同样代表上游撤销了旧意图；清除签名后允许发送方用
        # 修正后的相同几何和 stamp 重新断言目标。
        self._pending_plan = None
        self._navigation_goal = None
        self._active_path_stamp_ns = 0
        self._request_backend_cancellation()
        self._publish_empty_path(invalidation_stamp_ns)
        self._mark_startup_generation_published()
        self._publish_status(
            plan_id=plan_id,
            goal_id=0,
            request_id=0,
            command=PCTPlanningStatus.COMMAND_PLAN,
            state=PCTPlanningStatus.ERROR,
            text=f"拒绝 {reason}",
            goal_stamp_ns=0,
            path_stamp_ns=invalidation_stamp_ns,
            point_count=0,
            event_stamp_ns=invalidation_stamp_ns,
        )
        self.get_logger().error(f"拒绝 PCT goal：{reason}")

    def _schedule_active_goal_if_idle(self) -> None:
        """在 worker 空闲且 Odometry 新鲜时提交当前待规划请求。"""

        plan = self._pending_plan
        if plan is None or self._future is not None:
            return
        goal = plan.goal
        if (
            self._navigation_goal is None
            or self._navigation_goal.goal_id != goal.goal_id
        ):
            self._pending_plan = None
            return
        odometry = self._latest_odometry
        if odometry is None or not self._odometry_is_fresh(odometry, goal):
            self._publish_plan_status(
                plan,
                state=PCTPlanningStatus.WAITING_FOR_ODOMETRY,
                text="等待同一 ROS 时间域的新鲜 Odometry",
                goal_stamp_ns=goal.stamp_ns,
                path_stamp_ns=plan.invalidation_stamp_ns,
                point_count=0,
                event_stamp_ns=max(
                    plan.invalidation_stamp_ns,
                    self._clock_now_ns(),
                ),
            )
            return
        job = _PlanningJob(
            request=plan,
            start_position_xyz=odometry.position_xyz,
            start_stamp_ns=odometry.stamp_ns,
            cancel_event=threading.Event(),
        )
        self._future_job = job
        self._future = self._worker.submit(self._run_backend, job)
        self._publish_plan_status(
            plan,
            state=PCTPlanningStatus.PLANNING,
            text="PCT 正在规划三维全局路径",
            goal_stamp_ns=goal.stamp_ns,
            path_stamp_ns=plan.invalidation_stamp_ns,
            point_count=0,
            event_stamp_ns=max(plan.invalidation_stamp_ns, self._clock_now_ns()),
        )

    def _run_backend(self, job: _PlanningJob) -> PCTBackendPlan:
        """只在单 worker 中初始化并调用非并发 PCT backend。"""

        if job.cancel_event.is_set():
            raise PCTBackendError(
                f"PCT plan_id={job.request.plan_id} 在 worker 启动前已取消"
            )
        if self._backend is None:
            preload = self._backend_preload_future
            if preload is not None:
                self._backend = preload.result()
                self._backend_preload_future = None
            else:
                # 保留测试注入和显式运行时替换所需的惰性初始化路径。
                if self._backend_config is None:
                    raise PCTBackendError("PCT backend 配置未初始化")
                self._backend = self._backend_factory(self._backend_config)
        if job.cancel_event.is_set():
            cancel = getattr(self._backend, "cancel_current_plan", None)
            if callable(cancel):
                cancel()
            raise PCTBackendError(
                f"PCT plan_id={job.request.plan_id} 在 backend 初始化期间已取消"
            )
        prepare_plan = getattr(self._backend, "prepare_plan", None)
        if callable(prepare_plan):
            prepare_plan(job.cancel_event)
        if job.cancel_event.is_set():
            cancel = getattr(self._backend, "cancel_current_plan", None)
            if callable(cancel):
                cancel()
            raise PCTBackendError(
                f"PCT plan_id={job.request.plan_id} 在 backend 准备期间已取消"
            )
        return self._backend.plan(
            start_base_xyz=job.start_position_xyz,
            goal_base_xyz=job.request.goal.position_xyz,
            goal_yaw=job.request.goal.yaw,
        )

    def _poll_planning_result(self) -> None:
        """在 ROS executor 线程发布结果，并丢弃被新目标覆盖的旧结果。"""

        future = self._future
        job = self._future_job
        if future is None or job is None or not future.done():
            if future is None:
                self._schedule_active_goal_if_idle()
            return
        self._future = None
        self._future_job = None
        active_plan = self._pending_plan
        if active_plan is None or active_plan.plan_id != job.request.plan_id:
            self.get_logger().info(
                f"丢弃已被覆盖的 PCT plan_id={job.request.plan_id} 结果"
            )
            self._schedule_active_goal_if_idle()
            return
        active_goal = active_plan.goal
        try:
            plan = future.result()
        except PCTNoPathError as exc:
            self._finish_failure(
                active_plan,
                state=PCTPlanningStatus.NO_PATH,
                text=str(exc),
            )
            return
        except Exception as exc:
            self._finish_failure(
                active_plan,
                state=PCTPlanningStatus.ERROR,
                text=str(exc),
            )
            return

        current_odometry = self._latest_odometry
        if (
            current_odometry is None
            or not self._odometry_is_fresh(current_odometry, active_goal)
        ):
            self._publish_plan_status(
                active_plan,
                state=PCTPlanningStatus.WAITING_FOR_ODOMETRY,
                text="PCT 完成时 Odometry 已超时，等待新鲜位姿后重规划",
                goal_stamp_ns=active_goal.stamp_ns,
                path_stamp_ns=active_plan.invalidation_stamp_ns,
                point_count=0,
                event_stamp_ns=max(
                    active_plan.invalidation_stamp_ns,
                    self._clock_now_ns(),
                ),
            )
            return
        drift = math.dist(
            current_odometry.position_xyz,
            job.start_position_xyz,
        )
        if drift > self._maximum_start_drift_m:
            self.get_logger().warning(
                f"PCT 规划期间 base 漂移 {drift:.3f} m，按当前位姿重规划"
            )
            self._schedule_active_goal_if_idle()
            return

        path_stamp_ns = self._next_path_stamp_ns(
            self._clock_now_ns()
        )
        try:
            path = self._path_message(plan, active_goal, path_stamp_ns)
        except ValueError as exc:
            self._finish_failure(
                active_plan,
                state=PCTPlanningStatus.ERROR,
                text=f"PCT Path 合同失败：{exc}",
            )
            return
        self._publish_path_generation(path)
        self._active_path_stamp_ns = path_stamp_ns
        self._publish_plan_status(
            active_plan,
            state=PCTPlanningStatus.SUCCEEDED,
            text="PCT 三维全局路径规划成功",
            goal_stamp_ns=active_goal.stamp_ns,
            path_stamp_ns=path_stamp_ns,
            point_count=len(path.poses),
            event_stamp_ns=path_stamp_ns,
        )
        self.get_logger().info(
            f"PCT plan_id={active_plan.plan_id} 成功发布 "
            f"{len(path.poses)} 点地面 Path"
        )
        self._pending_plan = None

    def _finish_failure(
        self,
        plan: _PlanRequest,
        *,
        state: int,
        text: str,
    ) -> None:
        """失败时再次发布更新代际的空 Path，并给出类型化状态。"""

        path_stamp_ns = self._next_path_stamp_ns(
            self._clock_now_ns()
        )
        self._publish_empty_path(path_stamp_ns)
        self._active_path_stamp_ns = path_stamp_ns
        self._publish_plan_status(
            plan,
            state=state,
            text=text,
            path_stamp_ns=path_stamp_ns,
            point_count=0,
            event_stamp_ns=path_stamp_ns,
        )
        level = (
            self.get_logger().warning
            if state == PCTPlanningStatus.NO_PATH
            else self.get_logger().error
        )
        level(f"PCT plan_id={plan.plan_id} 失败：{text}")
        self._pending_plan = None

    def _path_message(
        self,
        plan: PCTBackendPlan,
        goal: _NavigationGoal,
        stamp_ns: int,
    ) -> PathMessage:
        """构造 header、frame、单位四元数和地面 z 均有效的 Path。"""

        if plan.metadata.get("height_semantics") != "ground_height":
            raise ValueError(
                "PCT backend 必须声明 height_semantics=ground_height"
            )
        points = prepare_ground_path(
            plan.points_xyz,
            terminal_yaw=goal.yaw,
            minimum_point_spacing_m=self._minimum_path_spacing_m,
        )
        stamp = _time_from_nanoseconds(stamp_ns)
        message = PathMessage()
        message.header.stamp = stamp
        message.header.frame_id = self._world_frame
        for point in points:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self._world_frame
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = point.z
            pose.pose.orientation.z = math.sin(0.5 * point.yaw)
            pose.pose.orientation.w = math.cos(0.5 * point.yaw)
            message.poses.append(pose)
        return message

    def _publish_empty_path(self, stamp_ns: int) -> None:
        """发布可靠缓存空 Path，使 SCAN 立即清除旧参考代际。"""

        message = PathMessage()
        message.header.stamp = _time_from_nanoseconds(stamp_ns)
        message.header.frame_id = self._world_frame
        self._publish_path_generation(message)

    def _publish_path_generation(self, message: PathMessage) -> None:
        """把同一 Path 代际原样发布到 PCT 源 topic 与 SCAN 输入 topic。"""

        # 两个 publisher 接收同一个消息对象，不重写 header、pose 或 z。
        # /pct/global_path 保留 PCT 规划结果的可审计来源；/initial_path 是
        # SCAN 的稳定输入接口。成功 Path 与空 tombstone 都必须同时发布，
        # 否则下游可能继续执行已经被 PCT 撤销的旧路径。
        self._path_publisher.publish(message)
        self._scan_path_publisher.publish(message)

    def _publish_startup_generation_if_ready(self) -> None:
        """仿真时钟生效后清空下游旧路径，覆盖 adapter 单独重启场景。"""

        if self._startup_generation_published:
            self._startup_timer.cancel()
            return
        now_ns = self._clock_now_ns()
        if now_ns <= 0:
            return
        path_stamp_ns = self._next_path_stamp_ns(now_ns)
        self._publish_empty_path(path_stamp_ns)
        self._publish_status(
            plan_id=0,
            goal_id=0,
            request_id=0,
            command=PCTPlanningStatus.COMMAND_NONE,
            state=PCTPlanningStatus.IDLE,
            text="PCT adapter 已就绪，等待目标",
            goal_stamp_ns=0,
            path_stamp_ns=path_stamp_ns,
            point_count=0,
            event_stamp_ns=path_stamp_ns,
        )
        self._mark_startup_generation_published()

    def _mark_startup_generation_published(self) -> None:
        """标记已有空 Path 代际，避免启动定时器反向覆盖新目标状态。"""

        self._startup_generation_published = True
        self._startup_timer.cancel()

    def _publish_plan_status(
        self,
        plan: _PlanRequest,
        *,
        state: int,
        text: str,
        path_stamp_ns: int,
        point_count: int,
        event_stamp_ns: int,
        goal_stamp_ns: int | None = None,
    ) -> None:
        """发布与一次 PLAN/REPLAN 精确绑定的 typed 状态。"""

        self._publish_status(
            plan_id=plan.plan_id,
            goal_id=plan.goal.goal_id,
            request_id=plan.request_id,
            command=plan.command,
            state=state,
            text=text,
            goal_stamp_ns=(
                plan.goal.stamp_ns
                if goal_stamp_ns is None
                else int(goal_stamp_ns)
            ),
            path_stamp_ns=path_stamp_ns,
            point_count=point_count,
            event_stamp_ns=event_stamp_ns,
        )

    def _publish_status(
        self,
        *,
        plan_id: int,
        goal_id: int,
        request_id: int,
        command: int,
        state: int,
        text: str,
        goal_stamp_ns: int,
        path_stamp_ns: int,
        point_count: int,
        event_stamp_ns: int,
    ) -> None:
        """发布只由枚举驱动、文本仅供人读的 PCT 状态。"""

        if event_stamp_ns <= 0:
            self.get_logger().warning("ROS 仿真时间为零，暂不发布 PCT 状态")
            return
        key = (
            int(plan_id),
            int(goal_id),
            int(request_id),
            int(command),
            int(state),
            str(text),
            int(goal_stamp_ns),
            int(path_stamp_ns),
            int(point_count),
            int(self._active_path_stamp_ns),
            0 if self._navigation_goal is None else self._navigation_goal.identity,
        )
        if key == self._last_status_key:
            return
        self._last_status_key = key
        message = PCTPlanningStatus()
        message.header.stamp = _time_from_nanoseconds(event_stamp_ns)
        message.header.frame_id = self._world_frame
        message.plan_id = int(plan_id)
        message.goal_id = int(goal_id)
        message.request_id = int(request_id)
        message.command = int(command)
        message.state = int(state)
        message.goal_stamp = _optional_time_from_nanoseconds(goal_stamp_ns)
        message.path_stamp = _optional_time_from_nanoseconds(path_stamp_ns)
        active = self._navigation_goal
        message.has_active_goal = active is not None
        if active is not None:
            message.active_goal = deepcopy(active.message)
        message.active_path_stamp = _optional_time_from_nanoseconds(
            self._active_path_stamp_ns if active is not None else 0
        )
        message.message = str(text)
        message.path_point_count = int(point_count)
        self._status_publisher.publish(message)

    def _normalize_odometry(self, message: Odometry) -> _OdometrySnapshot:
        frame = normalize_frame_id(
            message.header.frame_id,
            field_name="Odometry header.frame_id",
        )
        child = normalize_frame_id(
            message.child_frame_id,
            field_name="Odometry child_frame_id",
        )
        if frame != self._world_frame or child != self._base_frame:
            raise ValueError(
                f"Odometry frame 必须是 {self._world_frame}/{self._base_frame}"
            )
        stamp_ns = _stamp_to_nanoseconds(message.header.stamp)
        position = message.pose.pose.position
        point = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Odometry 位置包含 NaN 或 Inf")
        orientation = message.pose.pose.orientation
        quaternion_xyzw_to_yaw(
            (orientation.x, orientation.y, orientation.z, orientation.w)
        )
        return _OdometrySnapshot(position_xyz=point, stamp_ns=stamp_ns)

    def _normalize_goal(
        self,
        message: PoseStamped,
    ) -> tuple[tuple[float, float, float], float, int]:
        frame = normalize_frame_id(
            message.header.frame_id,
            field_name="goal header.frame_id",
        )
        if frame != self._world_frame:
            raise ValueError(
                f"goal frame 必须是 {self._world_frame}，收到 {frame}"
            )
        stamp_ns = _stamp_to_nanoseconds(message.header.stamp)
        now_ns = self._clock_now_ns()
        maximum_future_ns = int(
            self._goal_future_tolerance_sec * 1_000_000_000
        )
        if now_ns <= 0:
            raise _FutureTimestampError("ROS 仿真时间尚未开始")
        if stamp_ns > now_ns + maximum_future_ns:
            raise _FutureTimestampError(
                "goal stamp 超过当前 ROS 时间允许的 future tolerance"
            )
        position = message.pose.position
        point = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("goal 位置包含 NaN 或 Inf")
        orientation = message.pose.orientation
        yaw = quaternion_xyzw_to_yaw(
            (orientation.x, orientation.y, orientation.z, orientation.w)
        )
        return point, yaw, stamp_ns

    def _odometry_is_fresh(
        self,
        odometry: _OdometrySnapshot,
        goal: _NavigationGoal,
    ) -> bool:
        now_ns = max(self._clock_now_ns(), goal.stamp_ns)
        delta_sec = (now_ns - odometry.stamp_ns) / 1_000_000_000.0
        return (
            -self._future_tolerance_sec
            <= delta_sec
            <= self._odometry_timeout_sec
        )

    def _next_path_stamp_ns(self, candidate_ns: int) -> int:
        """即使同一仿真 tick 连续清空/成功，也保证 Path 代际严格递增。"""

        next_stamp = max(int(candidate_ns), self._last_path_stamp_ns + 1)
        if next_stamp <= 0:
            raise ValueError("无法在零仿真时间生成 Path")
        self._last_path_stamp_ns = next_stamp
        return next_stamp

    def _clock_now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _finite_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值")
        return value

    def _positive_parameter(self, name: str) -> float:
        value = self._finite_parameter(name)
        if value <= 0.0:
            raise ValueError(f"{name} 必须为正数")
        return value

    def _nonzero_parameter(self, name: str) -> float:
        value = self._finite_parameter(name)
        if value == 0.0:
            raise ValueError(f"{name} 不能为零")
        return value

    def _nonnegative_parameter(self, name: str) -> float:
        value = self._finite_parameter(name)
        if value < 0.0:
            raise ValueError(f"{name} 不能为负数")
        return value

    def _positive_int_parameter(self, name: str) -> int:
        value = int(self.get_parameter(name).value)
        if value < 1:
            raise ValueError(f"{name} 必须至少为 1")
        return value

    def destroy_node(self) -> bool:
        """停止规划 worker，避免 launch 退出后残留后台线程。"""

        self._startup_timer.cancel()
        self._poll_timer.cancel()
        self._request_backend_cancellation()
        self._worker.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _request_backend_cancellation(self) -> None:
        """若 backend 支持抢占，则请求当前同步搜索尽快退出。"""

        job = self._future_job
        if job is not None:
            job.cancel_event.set()
        backend = self._backend
        cancel = getattr(backend, "cancel_current_plan", None)
        if callable(cancel):
            cancel()


def _stamp_to_nanoseconds(stamp: Time) -> int:
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("时间戳范围非法")
    value = sec * 1_000_000_000 + nanosec
    if value <= 0:
        raise ValueError("时间戳必须非零")
    return value


def _optional_stamp_to_nanoseconds(stamp: Time) -> int:
    """解析允许全零的前置 Path stamp，其余范围仍严格校验。"""

    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if sec == 0 and nanosec == 0:
        return 0
    return _stamp_to_nanoseconds(stamp)


def _time_from_nanoseconds(value: int) -> Time:
    nanoseconds = int(value)
    if nanoseconds <= 0:
        raise ValueError("时间戳必须为正数")
    return Time(
        sec=nanoseconds // 1_000_000_000,
        nanosec=nanoseconds % 1_000_000_000,
    )


def _optional_time_from_nanoseconds(value: int) -> Time:
    return Time() if int(value) <= 0 else _time_from_nanoseconds(value)


def main(args: list[str] | None = None) -> None:
    """启动 PCT ROS 2 adapter。"""

    rclpy.init(args=args)
    node: PCTROS2Adapter | None = None
    try:
        node = PCTROS2Adapter()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
