"""验证 Isaac ROS 2 自定义消息环境的启动前门禁。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from source.navigation.isaac_ros2_environment import (
    DEFAULT_CUSTOM_MESSAGE_LIBRARIES,
    IsaacRos2EnvironmentError,
    validate_isaac_ros2_custom_message_environment,
)


class IsaacRos2EnvironmentTest(unittest.TestCase):
    """覆盖 overlay、共享库、RMW 与 Python ABI 的失败关闭语义。"""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.prefix = Path(self._temporary_directory.name) / "scan_planner_msgs"
        resource = (
            self.prefix
            / "share/ament_index/resource_index/rosidl_interfaces"
            / "scan_planner_msgs"
        )
        resource.parent.mkdir(parents=True)
        resource.write_text(
            (
                "msg/Bspline.msg\n"
                "msg/BsplineDiagnostics.msg\n"
                "msg/ControllerStatus.msg\n"
                "msg/GridMapObservationDiagnostics.msg\n"
                "msg/NavigationStatus.msg\n"
                "msg/StairExecutionFreeze.msg\n"
            ),
            encoding="utf-8",
        )
        library_dir = self.prefix / "lib"
        library_dir.mkdir(parents=True)
        for library_name in DEFAULT_CUSTOM_MESSAGE_LIBRARIES:
            (library_dir / library_name).write_bytes(b"test")
        self.environment = {
            "ROS_DISTRO": "humble",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "AMENT_PREFIX_PATH": str(self.prefix),
            "LD_LIBRARY_PATH": str(library_dir),
            "PYTHONPATH": "",
        }

    def test_valid_overlay_returns_auditable_report(self) -> None:
        report = validate_isaac_ros2_custom_message_environment(
            environ=self.environment,
            python_version=(3, 11),
        )

        self.assertTrue(report["verified"])
        self.assertEqual(report["ros_distro"], "humble")
        self.assertEqual(report["rmw_implementation"], "rmw_fastrtps_cpp")
        self.assertEqual(report["rmw_source"], "environment")
        self.assertEqual(report["python_version"], "3.11")
        self.assertTrue(report["pythonpath_cleared"])
        self.assertEqual(report["interface_prefix"], str(self.prefix))
        self.assertEqual(report["message_library_dir"], str(self.prefix / "lib"))
        self.assertEqual(
            report["message_names"],
            [
                "ControllerStatus",
                "NavigationStatus",
                "GridMapObservationDiagnostics",
                "BsplineDiagnostics",
                "StairExecutionFreeze",
            ],
        )
        self.assertIn(
            "msg/StairExecutionFreeze.msg",
            report["interface_entries"],
        )

    def test_unset_rmw_uses_verified_ros_default(self) -> None:
        environment = dict(self.environment)
        environment.pop("RMW_IMPLEMENTATION")

        report = validate_isaac_ros2_custom_message_environment(
            environ=environment,
            python_version=(3, 11),
        )

        self.assertEqual(report["rmw_implementation"], "rmw_fastrtps_cpp")
        self.assertEqual(report["rmw_source"], "ros_default")

    def test_wrong_ros_or_rmw_is_rejected_with_startup_hint(self) -> None:
        environment = dict(self.environment)
        environment["ROS_DISTRO"] = "jazzy"
        environment["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"

        with self.assertRaises(IsaacRos2EnvironmentError) as context:
            validate_isaac_ros2_custom_message_environment(
                environ=environment,
                python_version=(3, 11),
            )

        message = str(context.exception)
        self.assertIn("ROS_DISTRO", message)
        self.assertIn("RMW_IMPLEMENTATION", message)
        self.assertIn("source /opt/ros/humble/setup.zsh", message)
        self.assertIn("unset PYTHONPATH", message)

    def test_ros_python_abi_path_is_rejected_for_isaac_python(self) -> None:
        environment = dict(self.environment)
        environment["PYTHONPATH"] = (
            "/opt/ros/humble/lib/python3.10/site-packages"
        )

        with self.assertRaisesRegex(
            IsaacRos2EnvironmentError,
            "Python 3.11 ABI 不匹配",
        ):
            validate_isaac_ros2_custom_message_environment(
                environ=environment,
                python_version=(3, 11),
            )

    def test_missing_interface_resource_is_rejected(self) -> None:
        environment = dict(self.environment)
        environment["AMENT_PREFIX_PATH"] = str(
            Path(self._temporary_directory.name) / "missing"
        )

        with self.assertRaisesRegex(
            IsaacRos2EnvironmentError,
            "rosidl interface resource",
        ):
            validate_isaac_ros2_custom_message_environment(
                environ=environment,
                python_version=(3, 11),
            )

    def test_missing_navigation_status_interface_is_rejected(self) -> None:
        resource = (
            self.prefix
            / "share/ament_index/resource_index/rosidl_interfaces"
            / "scan_planner_msgs"
        )
        resource.write_text(
            (
                "msg/Bspline.msg\n"
                "msg/BsplineDiagnostics.msg\n"
                "msg/ControllerStatus.msg\n"
                "msg/GridMapObservationDiagnostics.msg\n"
                "msg/StairExecutionFreeze.msg\n"
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            IsaacRos2EnvironmentError,
            "NavigationStatus",
        ):
            validate_isaac_ros2_custom_message_environment(
                environ=self.environment,
                python_version=(3, 11),
            )

    def test_missing_planning_diagnostics_interface_is_rejected(self) -> None:
        resource = (
            self.prefix
            / "share/ament_index/resource_index/rosidl_interfaces"
            / "scan_planner_msgs"
        )
        resource.write_text(
            (
                "msg/Bspline.msg\n"
                "msg/ControllerStatus.msg\n"
                "msg/NavigationStatus.msg\n"
                "msg/StairExecutionFreeze.msg\n"
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            IsaacRos2EnvironmentError,
            "BsplineDiagnostics|GridMapObservationDiagnostics",
        ):
            validate_isaac_ros2_custom_message_environment(
                environ=self.environment,
                python_version=(3, 11),
            )

    def test_missing_stair_execution_freeze_interface_is_rejected(self) -> None:
        resource = (
            self.prefix
            / "share/ament_index/resource_index/rosidl_interfaces"
            / "scan_planner_msgs"
        )
        resource.write_text(
            (
                "msg/Bspline.msg\n"
                "msg/BsplineDiagnostics.msg\n"
                "msg/ControllerStatus.msg\n"
                "msg/GridMapObservationDiagnostics.msg\n"
                "msg/NavigationStatus.msg\n"
            ),
            encoding="utf-8",
        )

        with self.assertRaises(IsaacRos2EnvironmentError) as context:
            validate_isaac_ros2_custom_message_environment(
                environ=self.environment,
                python_version=(3, 11),
            )

        message = str(context.exception)
        self.assertIn(
            "rosidl interface resource 缺少：msg/StairExecutionFreeze.msg",
            message,
        )
        self.assertNotIn("msg/ControllerStatus.msg", message.split("缺少：", 1)[1])

    def test_missing_or_invisible_message_library_is_rejected(self) -> None:
        missing_library = self.prefix / "lib" / DEFAULT_CUSTOM_MESSAGE_LIBRARIES[0]
        missing_library.unlink()
        with self.assertRaisesRegex(
            IsaacRos2EnvironmentError,
            "缺少自定义消息共享库",
        ):
            validate_isaac_ros2_custom_message_environment(
                environ=self.environment,
                python_version=(3, 11),
            )

        missing_library.write_bytes(b"test")
        environment = dict(self.environment)
        environment["LD_LIBRARY_PATH"] = str(
            Path(self._temporary_directory.name) / "unrelated"
        )
        with self.assertRaisesRegex(
            IsaacRos2EnvironmentError,
            "LD_LIBRARY_PATH 未包含",
        ):
            validate_isaac_ros2_custom_message_environment(
                environ=environment,
                python_version=(3, 11),
            )


if __name__ == "__main__":
    unittest.main()
