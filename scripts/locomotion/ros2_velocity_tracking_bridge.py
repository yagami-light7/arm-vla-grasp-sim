#!/usr/bin/env python3
"""Tail benchmark samples and publish actual/commanded motion for RViz2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _yaw_quaternion(yaw: float):
    from geometry_msgs.msg import Quaternion

    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


class VelocityTrackingBridge(Node):
    def __init__(self) -> None:
        super().__init__("loco_velocity_tracking_bridge")
        self.declare_parameter("samples_path", "")
        self.declare_parameter("rows_per_poll", 1)
        samples_path_value = str(self.get_parameter("samples_path").value).strip()
        if not samples_path_value:
            raise ValueError("set -p samples_path:=/path/to/samples.jsonl")
        self.samples_path = Path(samples_path_value).expanduser()
        self.rows_per_poll = max(1, int(self.get_parameter("rows_per_poll").value))
        self.frame_id = "map"
        self.offset = 0
        self.actual_path = PathMsg()
        self.command_path = PathMsg()
        self.actual_path.header.frame_id = self.frame_id
        self.command_path.header.frame_id = self.frame_id
        self.command_x = self.command_y = self.command_yaw = 0.0
        self.last_time_s: float | None = None
        self.odom_pub = self.create_publisher(Odometry, "~/odom_actual", 10)
        self.actual_path_pub = self.create_publisher(PathMsg, "~/path_actual", 10)
        self.command_path_pub = self.create_publisher(PathMsg, "~/path_command_integrated", 10)
        self.command_pub = self.create_publisher(TwistStamped, "~/cmd_vel", 10)
        self.measured_pub = self.create_publisher(TwistStamped, "~/measured_twist", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/markers", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.02, self._poll)
        self.get_logger().info(f"tailing {self.samples_path}")

    def _poll(self) -> None:
        if not self.samples_path.exists():
            return
        with self.samples_path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            for _ in range(self.rows_per_poll):
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    self._publish(json.loads(line))
                except json.JSONDecodeError:
                    handle.seek(line_start)
                    break
                self.offset = handle.tell()

    def _pose(self, x: float, y: float, yaw: float, stamp) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = _yaw_quaternion(yaw)
        return pose

    def _twist(self, row: dict, stamp, prefix: str) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(row[f"{prefix}_vx"])
        msg.twist.linear.y = float(row[f"{prefix}_vy"])
        msg.twist.angular.z = float(row[f"{prefix}_wz"])
        return msg

    def _arrow(self, marker_id: int, x: float, y: float, vx: float, vy: float, color: ColorRGBA, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "velocity"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [Point(x=x, y=y, z=0.45), Point(x=x + vx, y=y + vy, z=0.45)]
        marker.scale = Vector3(x=0.025, y=0.05, z=0.05)
        marker.color = color
        return marker

    def _publish(self, row: dict) -> None:
        now = self.get_clock().now().to_msg()
        x, y, yaw = float(row["base_x"]), float(row["base_y"]), float(row["base_yaw"])
        time_s = float(row["time_s"])
        dt = 0.0 if self.last_time_s is None else max(0.0, time_s - self.last_time_s)
        self.last_time_s = time_s
        self.command_yaw += float(row["cmd_wz"]) * dt
        c, s = math.cos(self.command_yaw), math.sin(self.command_yaw)
        self.command_x += (c * float(row["cmd_vx"]) - s * float(row["cmd_vy"])) * dt
        self.command_y += (s * float(row["cmd_vx"]) + c * float(row["cmd_vy"])) * dt
        actual_pose = self._pose(x, y, yaw, now)
        command_pose = self._pose(self.command_x, self.command_y, self.command_yaw, now)
        self.actual_path.header.stamp = now
        self.command_path.header.stamp = now
        self.actual_path.poses.append(actual_pose)
        self.command_path.poses.append(command_pose)
        self.actual_path.poses = self.actual_path.poses[-10000:]
        self.command_path.poses = self.command_path.poses[-10000:]
        self.actual_path_pub.publish(self.actual_path)
        self.command_path_pub.publish(self.command_path)
        odom = Odometry()
        odom.header = actual_pose.header
        odom.child_frame_id = "base_link"
        odom.pose.pose = actual_pose.pose
        odom.twist.twist.linear.x = float(row["measured_vx"])
        odom.twist.twist.linear.y = float(row["measured_vy"])
        odom.twist.twist.angular.z = float(row["measured_wz"])
        self.odom_pub.publish(odom)
        self.command_pub.publish(self._twist(row, now, "cmd"))
        self.measured_pub.publish(self._twist(row, now, "measured"))
        green = ColorRGBA(r=0.1, g=1.0, b=0.1, a=1.0)
        red = ColorRGBA(r=1.0, g=0.15, b=0.1, a=1.0)
        markers = [
            self._arrow(0, x, y, 1.5 * float(row["cmd_vx"]), 1.5 * float(row["cmd_vy"]), green, now),
            self._arrow(1, x, y, 1.5 * float(row["measured_vx"]), 1.5 * float(row["measured_vy"]), red, now),
        ]
        text = Marker()
        text.header.frame_id = self.frame_id
        text.header.stamp = now
        text.ns = "velocity"
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = Point(x=x, y=y, z=0.75)
        text.scale.z = 0.12
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = (
            f"cmd [{row['cmd_vx']:.2f}, {row['cmd_vy']:.2f}, {row['cmd_wz']:.2f}]\n"
            f"meas [{row['measured_vx']:.2f}, {row['measured_vy']:.2f}, {row['measured_wz']:.2f}]"
        )
        markers.append(text)
        self.marker_pub.publish(MarkerArray(markers=markers))
        transform = TransformStamped()
        transform.header = actual_pose.header
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = float(row["base_z"])
        transform.transform.rotation = actual_pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)


def main() -> None:
    rclpy.init()
    node = VelocityTrackingBridge()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
