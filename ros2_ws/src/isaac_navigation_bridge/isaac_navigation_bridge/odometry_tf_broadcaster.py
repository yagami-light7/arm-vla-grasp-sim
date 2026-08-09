"""将规范化 Odometry 广播为动态 TF。"""

from __future__ import annotations

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import TransformBroadcaster

from .messages import (
    normalize_frame_id,
    normalize_odometry,
    stamp_is_valid,
)
from .qos import make_sensor_data_qos


class OdometryTfBroadcaster(Node):
    """订阅机器人 Odometry，并广播 world 到 base_link 的动态变换。"""

    def __init__(self) -> None:
        super().__init__("odometry_tf_broadcaster")
        self._enable_sim_time()
        self._declare_parameters()

        self._body_pose_topic = self._string_parameter(
            "topics.body_pose"
        )
        self._world_frame = normalize_frame_id(
            self._string_parameter("frames.world"),
            field_name="frames.world",
        )
        self._base_frame = normalize_frame_id(
            self._string_parameter("frames.base"),
            field_name="frames.base",
        )

        if self._world_frame == self._base_frame:
            raise ValueError("frames.world 与 frames.base 不能相同")

        qos_depth = self._positive_int_parameter("qos.depth")
        qos = make_sensor_data_qos(qos_depth)

        # TransformBroadcaster 会把动态坐标变换发布到 /tf。
        self._tf_broadcaster = TransformBroadcaster(self)

        # 保存 subscription 对象，避免它被 Python 垃圾回收。
        self._body_pose_subscription = self.create_subscription(
            Odometry,
            self._body_pose_topic,
            self._body_pose_callback,
            qos,
        )

        self._invalid_message_active = False
        self.get_logger().info(
            f"Odometry TF broadcaster ready: "
            f"{self._body_pose_topic} -> "
            f"{self._world_frame} -> {self._base_frame}"
        )

    def _enable_sim_time(self) -> None:
        """强制使用仿真时间，保证 TF 与 Isaac 消息处于同一时间域。"""

        result = self.set_parameters(
            [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
        )[0]
        if not result.successful:
            raise RuntimeError(f"无法启用 use_sim_time：{result.reason}")

    def _declare_parameters(self) -> None:
        """声明 Topic、坐标系和 QoS 参数。"""

        self.declare_parameter("topics.body_pose", "/body_pose")
        self.declare_parameter("frames.world", "world")
        self.declare_parameter("frames.base", "base_link")
        self.declare_parameter("qos.depth", 5)

    def _string_parameter(self, name: str) -> str:
        """读取并校验字符串参数。"""

        value = self.get_parameter(name).value

        # isinstance 检查后，Pylance 能确定 value 是 str。
        if not isinstance(value, str):
            raise TypeError(
                f"{name} 必须是字符串，实际类型为 {type(value).__name__}"
            )

        value = value.strip()
        if not value:
            raise ValueError(f"{name} 不能为空")

        return value

    def _positive_int_parameter(self, name: str) -> int:
        """读取并校验正整数参数。"""

        value = self.get_parameter(name).value

        # Python 中 bool 是 int 的子类，因此必须单独拒绝。
        if isinstance(value, bool):
            raise TypeError(f"{name} 必须是整数，不能是布尔值")

        # 此检查之后，Pylance 能确定 value 是 int。
        if not isinstance(value, int):
            raise TypeError(
                f"{name} 必须是整数，实际类型为 {type(value).__name__}"
            )

        if value < 1:
            raise ValueError(f"{name} 必须至少为 1")

        return value

    def _body_pose_callback(self, message: Odometry) -> None:
        """把一帧合法 Odometry 转换成 TransformStamped。"""

        # 动态 TF 必须具有有效时间戳，不能把零时间戳静默改成当前时间。
        if not stamp_is_valid(message.header.stamp):
            self._warn_invalid("Odometry 时间戳为零或非法")
            return

        try:
            normalized = normalize_odometry(
                message,
                # 前面已确认时间戳有效，因此这里只会保留原始时间戳。
                fallback_stamp=message.header.stamp,
                frame_id=self._world_frame,
                child_frame_id=self._base_frame,
            )
        except ValueError as exc:
            self._warn_invalid(f"Odometry 不符合 TF 合同：{exc}")
            return

        transform = TransformStamped()
        transform.header.stamp = normalized.header.stamp
        transform.header.frame_id = normalized.header.frame_id
        transform.child_frame_id = normalized.child_frame_id

        position = normalized.pose.pose.position
        orientation = normalized.pose.pose.orientation

        transform.transform.translation.x = position.x
        transform.transform.translation.y = position.y
        transform.transform.translation.z = position.z

        transform.transform.rotation.x = orientation.x
        transform.transform.rotation.y = orientation.y
        transform.transform.rotation.z = orientation.z
        transform.transform.rotation.w = orientation.w

        self._tf_broadcaster.sendTransform(transform)
        self._invalid_message_active = False

    def _warn_invalid(self, reason: str) -> None:
        """同一段连续非法输入只警告一次，避免高频刷屏。"""

        if self._invalid_message_active:
            return
        self._invalid_message_active = True
        self.get_logger().warning(reason)


def main(args: list[str] | None = None) -> None:
    """启动 Odometry 到 TF 的广播节点。"""

    rclpy.init(args=args)
    node: OdometryTfBroadcaster | None = None
    try:
        node = OdometryTfBroadcaster()
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