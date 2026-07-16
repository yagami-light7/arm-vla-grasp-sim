#!/usr/bin/env python3
"""Plot commanded and measured vx/vy/wz from the ROS2 benchmark bridge."""

from __future__ import annotations

import time
from collections import deque

import matplotlib.pyplot as plt
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


AXES = (
    ("vx", "linear.x", "m/s"),
    ("vy", "linear.y", "m/s"),
    ("wz", "angular.z", "rad/s"),
)


def _component(message: TwistStamped, field: str) -> float:
    group, name = field.split(".", maxsplit=1)
    return float(getattr(getattr(message.twist, group), name))


class VelocityTrackingPlotNode(Node):
    def __init__(self) -> None:
        super().__init__("loco_velocity_tracking_plot")
        self.declare_parameter("command_topic", "/loco_velocity_tracking_bridge/cmd_vel")
        self.declare_parameter("measured_topic", "/loco_velocity_tracking_bridge/measured_twist")
        self.declare_parameter("window_seconds", 30.0)
        self.declare_parameter("refresh_hz", 20.0)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.measured_topic = str(self.get_parameter("measured_topic").value)
        self.window_seconds = max(2.0, float(self.get_parameter("window_seconds").value))
        self.refresh_period_s = 1.0 / max(1.0, float(self.get_parameter("refresh_hz").value))
        self.latest_command: TwistStamped | None = None
        self.latest_measured: TwistStamped | None = None
        self.start_time = time.monotonic()
        self.last_sample_time = -1.0
        max_samples = max(1000, int(self.window_seconds / self.refresh_period_s * 3.0))
        self.times: deque[float] = deque(maxlen=max_samples)
        self.command_values = {axis: deque(maxlen=max_samples) for axis, _, _ in AXES}
        self.measured_values = {axis: deque(maxlen=max_samples) for axis, _, _ in AXES}
        self.create_subscription(TwistStamped, self.command_topic, self._on_command, 20)
        self.create_subscription(TwistStamped, self.measured_topic, self._on_measured, 20)
        self.get_logger().info(
            f"plotting command={self.command_topic} measured={self.measured_topic} "
            f"window={self.window_seconds:.1f}s"
        )

    def _on_command(self, message: TwistStamped) -> None:
        self.latest_command = message

    def _on_measured(self, message: TwistStamped) -> None:
        self.latest_measured = message

    def sample(self) -> bool:
        if self.latest_command is None or self.latest_measured is None:
            return False
        now = time.monotonic() - self.start_time
        if self.last_sample_time >= 0.0 and now - self.last_sample_time < self.refresh_period_s:
            return False
        self.last_sample_time = now
        self.times.append(now)
        for axis, field, _ in AXES:
            self.command_values[axis].append(_component(self.latest_command, field))
            self.measured_values[axis].append(_component(self.latest_measured, field))
        return True


class VelocityTrackingFigure:
    def __init__(self, node: VelocityTrackingPlotNode) -> None:
        plt.ion()
        self.node = node
        self.figure, self.axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        try:
            self.figure.canvas.manager.set_window_title("Go2-X5 velocity tracking")
        except AttributeError:
            pass
        self.lines = {}
        for plot_axis, (axis, _, unit) in zip(self.axes, AXES):
            command_line, = plot_axis.plot([], [], "k--", linewidth=1.8, label="command")
            measured_line, = plot_axis.plot([], [], color="#1f77b4", linewidth=1.4, label="measured")
            self.lines[axis] = (command_line, measured_line)
            plot_axis.set_ylabel(f"{axis} ({unit})")
            plot_axis.grid(True, alpha=0.3)
            plot_axis.legend(loc="upper right")
        self.axes[-1].set_xlabel("time (s)")
        self.figure.suptitle("Low-level RL policy: commanded vs measured body velocity")
        self.figure.tight_layout()
        self.figure.show()

    @property
    def is_open(self) -> bool:
        return plt.fignum_exists(self.figure.number)

    def refresh(self) -> None:
        if not self.node.sample() or not self.node.times:
            plt.pause(0.001)
            return
        times = list(self.node.times)
        right = times[-1]
        left = max(0.0, right - self.node.window_seconds)
        for plot_axis, (axis, _, _) in zip(self.axes, AXES):
            command_line, measured_line = self.lines[axis]
            command_line.set_data(times, list(self.node.command_values[axis]))
            measured_line.set_data(times, list(self.node.measured_values[axis]))
            plot_axis.set_xlim(left, max(self.node.window_seconds, right))
            plot_axis.relim()
            plot_axis.autoscale_view(scalex=False, scaley=True)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)


def main() -> None:
    rclpy.init()
    node = VelocityTrackingPlotNode()
    figure = VelocityTrackingFigure(node)
    try:
        while rclpy.ok() and figure.is_open:
            rclpy.spin_once(node, timeout_sec=0.01)
            figure.refresh()
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()
