"""ROS 2 QoS 合同。"""

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


def make_sensor_data_qos(depth: int = 5) -> QoSProfile:
    """返回显式、可测试的 SensorData QoS。"""

    queue_depth = int(depth)
    if queue_depth < 1:
        raise ValueError("SensorData QoS depth 必须至少为 1")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=queue_depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def make_reliable_transient_local_qos(depth: int = 1) -> QoSProfile:
    """返回 Path 等低频状态消息使用的可靠缓存 QoS。"""

    queue_depth = int(depth)
    if queue_depth < 1:
        raise ValueError("可靠缓存 QoS depth 必须至少为 1")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=queue_depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
