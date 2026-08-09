"""Isaac OmniGraph 原始导航 topic 到 SCAN 标准 topic 的桥接节点。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2

from .geometry import OrderedGroundPath, PointCloudFilterConfig
from .messages import (
    base_transform_from_odometry,
    convert_point_cloud,
    matching_frame_id,
    normalize_frame_id,
    normalize_odometry,
    normalized_stamp,
    ordered_ground_path_from_message,
    stamp_is_valid,
    stamp_to_nanoseconds,
)
from .qos import make_reliable_transient_local_qos, make_sensor_data_qos


@dataclass
class _GroundPathCache:
    """带代际、时间戳和单调进度的地面 Path 缓存。"""

    path: OrderedGroundPath
    stamp_ns: int
    generation: int
    progress_m: float | None = None


class IsaacNavigationBridge(Node):
    """规范化 Isaac 里程计和世界系点云，并执行局部安全过滤。"""

    def __init__(self) -> None:
        super().__init__("isaac_navigation_bridge")
        self._enable_sim_time_by_default()
        self._declare_parameters()

        self._warning_throttle_sec = self._float_parameter(
            "warning_throttle_sec"
        )
        if self._warning_throttle_sec < 0.0:
            raise ValueError("warning_throttle_sec 不能为负数")
        self._last_warning_monotonic: dict[str, float] = {}

        self._drop_cloud_without_odom = self._bool_parameter(
            "filters.drop_cloud_without_odom"
        )
        self._drop_cloud_without_ground_path = self._bool_parameter(
            "filters.drop_cloud_without_ground_path"
        )
        self._minimum_valid_input_points = self._int_parameter(
            "filters.minimum_valid_input_points"
        )
        if self._minimum_valid_input_points < 1:
            raise ValueError(
                "filters.minimum_valid_input_points 必须是正整数"
            )
        self._filter_config = self._load_filter_config()
        self._latest_base_pose: tuple[
            tuple[float, float, float],
            tuple[float, float, float, float],
        ] | None = None
        self._ground_path_cache: _GroundPathCache | None = None
        self._ground_path_cache_generation = 0
        self._latest_ground_path_stamp_ns: int | None = None
        self._latest_odom_stamp_ns: int | None = None
        self._latest_odom_received_monotonic: float | None = None
        self._odom_timeout_sec = self._float_parameter(
            "filters.odom_timeout_sec"
        )
        self._max_cloud_odom_skew_sec = self._float_parameter(
            "filters.max_cloud_odom_skew_sec"
        )
        if self._odom_timeout_sec <= 0.0:
            raise ValueError("filters.odom_timeout_sec 必须为正数")
        if self._max_cloud_odom_skew_sec < 0.0:
            raise ValueError("filters.max_cloud_odom_skew_sec 不能为负数")

        self._odom_frame_id = normalize_frame_id(
            self._string_parameter("frames.odom"),
            field_name="frames.odom",
        )
        self._base_frame_id = normalize_frame_id(
            self._string_parameter("frames.base"),
            field_name="frames.base",
        )
        self._cloud_frame_id = normalize_frame_id(
            self._string_parameter("frames.cloud"),
            field_name="frames.cloud",
        )
        if self._cloud_frame_id != self._odom_frame_id:
            raise ValueError(
                "未启用 TF 变换时 frames.cloud 必须与 frames.odom 相同"
            )
        qos = make_sensor_data_qos(self._int_parameter("qos.depth"))
        path_qos = make_reliable_transient_local_qos(
            self._int_parameter("qos.path_depth")
        )

        self._body_pose_publisher = self.create_publisher(
            Odometry,
            self._string_parameter("topics.body_pose_output"),
            qos,
        )
        self._cloud_publisher = self.create_publisher(
            PointCloud2,
            self._string_parameter("topics.cloud_output"),
            qos,
        )
        self._body_pose_subscription = self.create_subscription(
            Odometry,
            self._string_parameter("topics.body_pose_input"),
            self._body_pose_callback,
            qos,
        )
        self._cloud_subscription = self.create_subscription(
            PointCloud2,
            self._string_parameter("topics.cloud_input"),
            self._cloud_callback,
            qos,
        )
        self._path_subscription = self.create_subscription(
            Path,
            self._string_parameter("topics.initial_path_input"),
            self._path_callback,
            path_qos,
        )

        self.get_logger().info(
            "Isaac navigation bridge ready: "
            f"{self._string_parameter('topics.body_pose_input')} -> "
            f"{self._string_parameter('topics.body_pose_output')}, "
            f"{self._string_parameter('topics.cloud_input')} -> "
            f"{self._string_parameter('topics.cloud_output')}"
        )

    def _enable_sim_time_by_default(self) -> None:
        """强制使用 Isaac Sim 发布的仿真时钟。"""

        result = self.set_parameters(
            [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
        )[0]
        if not result.successful:
            raise RuntimeError(f"无法启用 use_sim_time：{result.reason}")

    def _declare_parameters(self) -> None:
        """声明启动期可配置的 topic、frame、QoS 和过滤参数。"""

        defaults: dict[str, Any] = {
            "topics.body_pose_input": "/isaac/body_pose_raw",
            "topics.cloud_input": "/isaac/cloud_registered_raw",
            "topics.body_pose_output": "/body_pose",
            "topics.cloud_output": "/cloud_registered",
            "topics.initial_path_input": "/initial_path",
            "frames.odom": "world",
            "frames.base": "base_link",
            "frames.cloud": "world",
            "qos.depth": 5,
            "qos.path_depth": 1,
            "warning_throttle_sec": 5.0,
            "filters.drop_cloud_without_odom": True,
            "filters.drop_cloud_without_ground_path": True,
            "filters.minimum_valid_input_points": 64,
            "filters.odom_timeout_sec": 0.50,
            "filters.max_cloud_odom_skew_sec": 0.20,
            "filters.range_min_m": 0.0,
            "filters.range_max_m": 8.0,
            "filters.crop_min_xyz_m": [-5.0, -5.0, -0.60],
            "filters.crop_max_xyz_m": [5.0, 5.0, 2.00],
            "filters.filter_ground": True,
            "filters.body_height_m": 0.30,
            "filters.ground_clearance_m": 0.03,
            "filters.ground_band_down_m": 0.03,
            "filters.filter_path_ground": True,
            "filters.path_ground_corridor_radius_m": 0.70,
            "filters.path_ground_clearance_m": 0.05,
            "filters.path_ground_stair_minimum_slope": 0.20,
            "filters.path_ground_stair_clearance_m": 0.09,
            "filters.path_ground_band_down_m": 0.05,
            "filters.path_ground_stair_band_down_m": 0.09,
            "filters.path_min_point_spacing_m": 0.05,
            "filters.path_ground_backward_arc_m": 1.0,
            "filters.path_ground_forward_arc_m": 3.0,
            "filters.filter_self": True,
            "filters.double_cylinder_radius_m": 0.27,
            "filters.double_cylinder_offset_m": 0.16,
            "filters.self_z_min_m": -0.40,
            "filters.self_z_max_m": 0.50,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_filter_config(self) -> PointCloudFilterConfig:
        return PointCloudFilterConfig(
            range_min_m=self._float_parameter("filters.range_min_m"),
            range_max_m=self._float_parameter("filters.range_max_m"),
            crop_min_xyz_m=self._float_array_parameter(
                "filters.crop_min_xyz_m"
            ),
            crop_max_xyz_m=self._float_array_parameter(
                "filters.crop_max_xyz_m"
            ),
            filter_ground=self._bool_parameter("filters.filter_ground"),
            body_height_m=self._float_parameter("filters.body_height_m"),
            ground_clearance_m=self._float_parameter(
                "filters.ground_clearance_m"
            ),
            ground_band_down_m=self._float_parameter(
                "filters.ground_band_down_m"
            ),
            filter_path_ground=self._bool_parameter(
                "filters.filter_path_ground"
            ),
            path_ground_corridor_radius_m=self._float_parameter(
                "filters.path_ground_corridor_radius_m"
            ),
            path_ground_clearance_m=self._float_parameter(
                "filters.path_ground_clearance_m"
            ),
            path_ground_stair_minimum_slope=self._float_parameter(
                "filters.path_ground_stair_minimum_slope"
            ),
            path_ground_stair_clearance_m=self._float_parameter(
                "filters.path_ground_stair_clearance_m"
            ),
            path_ground_band_down_m=self._float_parameter(
                "filters.path_ground_band_down_m"
            ),
            path_ground_stair_band_down_m=self._float_parameter(
                "filters.path_ground_stair_band_down_m"
            ),
            path_min_point_spacing_m=self._float_parameter(
                "filters.path_min_point_spacing_m"
            ),
            path_ground_backward_arc_m=self._float_parameter(
                "filters.path_ground_backward_arc_m"
            ),
            path_ground_forward_arc_m=self._float_parameter(
                "filters.path_ground_forward_arc_m"
            ),
            filter_self=self._bool_parameter("filters.filter_self"),
            double_cylinder_radius_m=self._float_parameter(
                "filters.double_cylinder_radius_m"
            ),
            double_cylinder_offset_m=self._float_parameter(
                "filters.double_cylinder_offset_m"
            ),
            self_z_min_m=self._float_parameter("filters.self_z_min_m"),
            self_z_max_m=self._float_parameter("filters.self_z_max_m"),
        )

    def _string_parameter(self, name: str) -> str:
        value = str(self.get_parameter(name).value).strip()
        if not value:
            raise ValueError(f"{name} 不能为空")
        return value

    def _float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值")
        return value

    def _int_parameter(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _bool_parameter(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _float_array_parameter(
        self,
        name: str,
    ) -> tuple[float, float, float]:
        values = tuple(float(value) for value in self.get_parameter(name).value)
        if len(values) != 3:
            raise ValueError(f"{name} 必须包含 3 个数值")
        return values

    def _warn_throttled(self, key: str, message: str) -> None:
        """使用单调时钟节流，避免仿真时钟暂停时刷屏。"""

        now = time.monotonic()
        last = self._last_warning_monotonic.get(key)
        if (
            last is not None
            and now - last < self._warning_throttle_sec
        ):
            return
        self._last_warning_monotonic[key] = now
        self.get_logger().warning(message)

    def _body_pose_callback(self, message: Odometry) -> None:
        fallback_stamp = self.get_clock().now().to_msg()
        try:
            normalized = normalize_odometry(
                message,
                fallback_stamp=fallback_stamp,
                frame_id=self._odom_frame_id,
                child_frame_id=self._base_frame_id,
            )
        except ValueError as exc:
            self._warn_throttled("invalid_odom", f"丢弃非法 Odometry：{exc}")
            return
        self._latest_base_pose = base_transform_from_odometry(normalized)
        self._latest_odom_stamp_ns = stamp_to_nanoseconds(
            normalized.header.stamp
        )
        self._latest_odom_received_monotonic = time.monotonic()
        self._body_pose_publisher.publish(normalized)

    def _path_callback(self, message: Path) -> None:
        """按 SCAN 合同缓存 Path；空 Path 显式清除旧代际。"""

        try:
            matching_frame_id(
                message.header.frame_id,
                self._cloud_frame_id,
                field_name="Path header.frame_id",
            )
            if not stamp_is_valid(message.header.stamp):
                raise ValueError("Path header.stamp 必须是合法非零时间")
            stamp_ns = stamp_to_nanoseconds(message.header.stamp)
        except ValueError as exc:
            self._warn_throttled(
                "invalid_ground_path",
                f"忽略无效地面 Path：{exc}",
            )
            return

        latest_stamp_ns = self._latest_ground_path_stamp_ns
        if latest_stamp_ns is not None and stamp_ns < latest_stamp_ns:
            self._warn_throttled(
                "stale_ground_path",
                "忽略时间戳早于当前缓存代际的地面 Path",
            )
            return

        if not message.poses:
            self._ground_path_cache_generation += 1
            self._latest_ground_path_stamp_ns = stamp_ns
            self._ground_path_cache = None
            self.get_logger().info(
                "收到空 initial_path，已显式清除地面 Path 缓存"
            )
            return

        try:
            path = ordered_ground_path_from_message(
                message,
                frame_id=self._cloud_frame_id,
                min_point_spacing_m=(
                    self._filter_config.path_min_point_spacing_m
                ),
            )
        except ValueError as exc:
            self._warn_throttled(
                "invalid_ground_path",
                f"忽略无效地面 Path：{exc}",
            )
            return

        self._ground_path_cache_generation += 1
        generation = self._ground_path_cache_generation
        self._latest_ground_path_stamp_ns = stamp_ns
        self._ground_path_cache = _GroundPathCache(
            path=path,
            stamp_ns=stamp_ns,
            generation=generation,
        )

    def _cloud_callback(self, message: PointCloud2) -> None:
        base_pose = self._latest_base_pose
        if base_pose is None and self._drop_cloud_without_odom:
            self._warn_throttled(
                "cloud_without_odom",
                "尚无有效 Odometry，丢弃 PointCloud2 以避免错误地面/自点过滤",
            )
            return
        if base_pose is None:
            self._warn_throttled(
                "cloud_without_odom_passthrough",
                "尚无有效 Odometry；当前仅移除非有限点，"
                "不执行位姿相关过滤",
            )
            base_position = None
            base_orientation = None
        else:
            base_position, base_orientation = base_pose

        fallback_stamp = self.get_clock().now().to_msg()
        try:
            cloud_stamp = normalized_stamp(
                message.header.stamp,
                fallback_stamp,
            )
        except ValueError as exc:
            self._warn_throttled(
                "invalid_cloud_stamp",
                f"丢弃非法 PointCloud2：{exc}",
            )
            return

        cloud_stamp_ns = stamp_to_nanoseconds(cloud_stamp)
        if base_pose is not None:
            received_monotonic = self._latest_odom_received_monotonic
            odom_stamp_ns = self._latest_odom_stamp_ns
            if received_monotonic is None or odom_stamp_ns is None:
                self._warn_throttled(
                    "incomplete_odom_cache",
                    "Odometry 缓存不完整，丢弃 PointCloud2",
                )
                return
            odom_age_sec = time.monotonic() - received_monotonic
            if odom_age_sec > self._odom_timeout_sec:
                self._warn_throttled(
                    "stale_odom",
                    "Odometry 已超时，丢弃 PointCloud2 以避免错误自点过滤",
                )
                return
            skew_sec = abs(cloud_stamp_ns - odom_stamp_ns) / 1_000_000_000.0
            if skew_sec > self._max_cloud_odom_skew_sec:
                self._warn_throttled(
                    "cloud_odom_skew",
                    "PointCloud2 与 Odometry 时间差超限，丢弃该点云",
                )
                return

        current_path_cache = self._ground_path_cache
        if (
            self._drop_cloud_without_ground_path
            and self._filter_config.filter_path_ground
        ):
            if current_path_cache is None:
                self._warn_throttled(
                    "cloud_without_ground_path",
                    "尚无有效 initial_path，丢弃点云以避免坡面支撑点提前污染占据地图",
                )
                return
            if cloud_stamp_ns < current_path_cache.stamp_ns:
                self._warn_throttled(
                    "cloud_before_ground_path",
                    "点云时间戳早于当前 initial_path，丢弃旧帧以避免跨 Path 代际误过滤",
                )
                return

        path_cache_snapshot: _GroundPathCache | None = None
        local_ground_path_segments = None
        if (
            base_position is not None
            and current_path_cache is not None
            and cloud_stamp_ns >= current_path_cache.stamp_ns
        ):
            path_cache_snapshot = current_path_cache
            ground_query = (
                base_position[0],
                base_position[1],
                base_position[2] - self._filter_config.body_height_m,
            )
            progress = current_path_cache.path.project_progress(
                ground_query,
                previous_progress_m=current_path_cache.progress_m,
                backward_arc_m=(
                    self._filter_config.path_ground_backward_arc_m
                ),
                forward_arc_m=(
                    self._filter_config.path_ground_forward_arc_m
                ),
            )
            if (
                self._ground_path_cache is not current_path_cache
                or current_path_cache.generation
                != self._ground_path_cache_generation
            ):
                self._warn_throttled(
                    "ground_path_generation_changed",
                    "点云处理期间 Path 代际已变化，丢弃该点云",
                )
                return
            current_path_cache.progress_m = progress
            local_ground_path_segments = (
                current_path_cache.path.local_segments(
                    progress,
                    backward_arc_m=(
                        self._filter_config.path_ground_backward_arc_m
                    ),
                    forward_arc_m=(
                        self._filter_config.path_ground_forward_arc_m
                    ),
                )
            )

        try:
            normalized = convert_point_cloud(
                message,
                fallback_stamp=fallback_stamp,
                frame_id=self._cloud_frame_id,
                base_position_world_xyz=base_position,
                base_yaw_rad=None,
                filter_config=self._filter_config,
                base_orientation_world_xyzw=base_orientation,
                local_ground_path_segments=local_ground_path_segments,
                minimum_valid_input_points=(
                    self._minimum_valid_input_points
                ),
            )
        except ValueError as exc:
            self._warn_throttled(
                "invalid_cloud",
                f"丢弃非法 PointCloud2：{exc}",
            )
            return
        if (
            path_cache_snapshot is not None
            and (
                self._ground_path_cache is not path_cache_snapshot
                or path_cache_snapshot.generation
                != self._ground_path_cache_generation
            )
        ):
            self._warn_throttled(
                "ground_path_generation_changed",
                "点云转换完成前 Path 代际已变化，丢弃该点云",
            )
            return
        self._cloud_publisher.publish(normalized)


def main(args: list[str] | None = None) -> None:
    """启动桥接节点。"""

    rclpy.init(args=args)
    node: IsaacNavigationBridge | None = None
    try:
        node = IsaacNavigationBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
