"""SensorData QoS 合同测试。"""

import pytest
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    ReliabilityPolicy,
)

from isaac_navigation_bridge.qos import (
    make_reliable_transient_local_qos,
    make_sensor_data_qos,
)


def test_sensor_data_qos_contract() -> None:
    qos = make_sensor_data_qos(depth=7)

    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 7
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.durability == DurabilityPolicy.VOLATILE


def test_sensor_data_qos_rejects_empty_queue() -> None:
    with pytest.raises(ValueError):
        make_sensor_data_qos(depth=0)


def test_reliable_transient_local_qos_contract() -> None:
    qos = make_reliable_transient_local_qos(depth=2)

    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 2
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_reliable_transient_local_qos_rejects_empty_queue() -> None:
    with pytest.raises(ValueError):
        make_reliable_transient_local_qos(depth=0)
