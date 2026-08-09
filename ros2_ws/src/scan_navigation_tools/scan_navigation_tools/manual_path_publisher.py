"""向 SCAN Planner 发布可配置的手工三维参考路径。"""

from __future__ import annotations

import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
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

from .path_geometry import (
    PathPoint,
    prepare_path_points,
    validate_frame_id,
    validate_topic_name,
)


class ManualPathPublisher(Node):
    """发布一次可靠且可缓存的手工地面高度路径。"""

    def __init__(self) -> None:
        super().__init__("manual_path_publisher")
        self._enable_sim_time()
        self._declare_parameters()

        self._topic = validate_topic_name(
            self.get_parameter("topic").value,
        )
        self._frame_id = validate_frame_id(
            self.get_parameter("frame_id").value,
        )
        raw_points = tuple(self.get_parameter("points_xyz").value)
        self._points = prepare_path_points(
            raw_points,
            min_point_distance_m=self.get_parameter(
                "min_point_distance_m"
            ).value,
        )
        self._startup_delay_sec = self._finite_parameter(
            "startup_delay_sec"
        )
        if self._startup_delay_sec < 0.0:
            raise ValueError("startup_delay_sec 不能为负数")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(Path, self._topic, qos)
        current_time_ns = self.get_clock().now().nanoseconds
        self._publish_not_before_ns = current_time_ns + round(
            self._startup_delay_sec * 1_000_000_000
        )
        self._clock_warning_emitted = False
        self._timer = self.create_timer(0.05, self._publish_when_ready)

        removed_count = len(raw_points) // 3 - len(self._points)
        self.get_logger().info(
            f"手工路径已加载：topic={self._topic}，"
            f"frame={self._frame_id}，有效点={len(self._points)}，"
            f"移除近重复点={removed_count}；Path z 表示地面高度"
        )

    def _enable_sim_time(self) -> None:
        """强制使用仿真时钟，防止与 Isaac 消息时间域分裂。"""

        result = self.set_parameters(
            [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
        )[0]
        if not result.successful:
            raise RuntimeError(f"无法启用 use_sim_time：{result.reason}")

    def _declare_parameters(self) -> None:
        """声明路径 topic、frame、点列和发布时机参数。"""

        self.declare_parameter("topic", "/initial_path")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter(
            "points_xyz",
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
            ],
        )
        self.declare_parameter("min_point_distance_m", 0.02)
        self.declare_parameter("startup_delay_sec", 1.0)
        # 仅供 pipeline 侧把冻结区间与 live Path 几何绑定；发布的 ROS
        # 消息仍保持标准 nav_msgs/Path，不引入私有传输格式。
        self.declare_parameter("scan_stair_freeze_points_sha256", "")
        self.declare_parameter(
            "scan_stair_freeze_stair_segment_indices",
            [-1, -1],
        )

    def _finite_parameter(self, name: str) -> float:
        """读取一个有限浮点参数。"""

        value = float(self.get_parameter(name).value)
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值")
        return value

    def _path_message(self, stamp: object) -> Path:
        """构造所有 header 和四元数均有效的 Path 消息。"""

        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = self._frame_id
        for point in self._points:
            message.poses.append(self._pose_message(point, stamp))
        return message

    def _pose_message(self, point: PathPoint, stamp: object) -> PoseStamped:
        """把一个地面路径点转换为带单位 yaw 四元数的 PoseStamped。"""

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self._frame_id
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z
        pose.pose.orientation.z = math.sin(point.yaw * 0.5)
        pose.pose.orientation.w = math.cos(point.yaw * 0.5)
        return pose

    def _publish_when_ready(self) -> None:
        """仿真时间有效且启动延时结束后发布一次路径。"""

        now = self.get_clock().now()
        if now.nanoseconds <= 0:
            if not self._clock_warning_emitted:
                self.get_logger().warning(
                    "仿真时间尚未生效，暂不发布零时间戳 Path"
                )
                self._clock_warning_emitted = True
            return
        if now.nanoseconds < self._publish_not_before_ns:
            return
        self._publisher.publish(self._path_message(now.to_msg()))
        self._timer.cancel()
        self.get_logger().info(
            f"已向 {self._topic} 发布 {len(self._points)} 点手工三维 Path"
        )


def main(args: list[str] | None = None) -> None:
    """启动手工路径发布节点。"""

    rclpy.init(args=args)
    node: ManualPathPublisher | None = None
    try:
        node = ManualPathPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # ros2 launch 关闭时可能在 spin() 返回后再次送达 SIGINT；清理阶段也要
        # 幂等吞掉该中断，避免成功联调以 exit -2 和 traceback 收尾。
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
