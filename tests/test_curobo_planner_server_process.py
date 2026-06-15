"""cuRobo planner server 自动启动边界测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from source.manipulation.planner_server_process import (
    CuroboPlannerServerProcess,
    CuroboPlannerServerProcessConfig,
    planner_server_supports_required_features,
)


class CuroboPlannerServerProcessTest(unittest.TestCase):
    def test_capabilities_require_single_retime_split_pregrasp(self) -> None:
        def supports(features: dict[str, bool]) -> bool:
            socket_context = Mock()
            socket_client = Mock()
            socket_context.__enter__ = Mock(return_value=socket_client)
            socket_context.__exit__ = Mock(return_value=False)
            socket_client.makefile.return_value.readline.return_value = json.dumps(
                {
                    "ok": True,
                    "features": features,
                }
            )
            with patch(
                "source.manipulation.planner_server_process.socket.create_connection",
                return_value=socket_context,
            ):
                return planner_server_supports_required_features()

        legacy_features = {
            "side_grasp_retreat_to_pregrasp": True,
            "split_pregrasp_motion": True,
        }
        self.assertFalse(supports(legacy_features))
        self.assertTrue(
            supports(
                {
                    **legacy_features,
                    "single_retime_split_pregrasp": True,
                }
            )
        )

    def test_existing_server_is_reused_and_not_owned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CuroboPlannerServerProcess(
                CuroboPlannerServerProcessConfig(project_root=Path(tmp_dir))
            )
            with patch(
                "source.manipulation.planner_server_process.planner_server_ping",
                return_value=True,
            ), patch(
                "source.manipulation.planner_server_process.planner_server_supports_required_features",
                return_value=True,
            ), patch(
                "source.manipulation.planner_server_process.subprocess.Popen"
            ) as popen:
                manager.start()
                manager.close()

        popen.assert_not_called()
        self.assertTrue(manager.reused_existing)
        self.assertTrue(manager.start_report["ready"])

    def test_spawned_server_waits_until_ready_and_is_cleaned_up(self) -> None:
        process = Mock()
        process.pid = 123
        process.poll.return_value = None
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            server_script = root / "scripts/curobo/grasp_planner_server.py"
            server_script.parent.mkdir(parents=True)
            server_script.write_text("", encoding="utf-8")
            manager = CuroboPlannerServerProcess(
                CuroboPlannerServerProcessConfig(
                    project_root=root,
                    startup_timeout_s=1.0,
                    log_path=root / "planner.log",
                    python_executable="/tmp/python",
                )
            )
            with patch(
                "source.manipulation.planner_server_process.planner_server_ping",
                side_effect=[False, True],
            ), patch(
                "source.manipulation.planner_server_process.subprocess.Popen",
                return_value=process,
            ) as popen, patch(
                "source.manipulation.planner_server_process._request_shutdown",
                return_value=True,
            ) as shutdown:
                manager.start()
                self.assertTrue(manager.wait_until_ready())
                manager.close()

        command = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertEqual(command[0], "/tmp/python")
        self.assertIn("grasp_planner_server.py", command[-1])
        self.assertEqual(env["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"], "0")
        self.assertEqual(env["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"], "0")
        self.assertEqual(env["GO2_X5_SIDE_GRASP_RETREAT_TO_PREGRASP"], "0")
        self.assertEqual(env["GO2_X5_SPLIT_PREGRASP_MOTION"], "1")
        shutdown.assert_called_once()
        process.wait.assert_called_once_with(timeout=5.0)

    def test_incompatible_existing_server_is_restarted(self) -> None:
        process = Mock()
        process.pid = 456
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            server_script = root / "scripts/curobo/grasp_planner_server.py"
            server_script.parent.mkdir(parents=True)
            server_script.write_text("", encoding="utf-8")
            manager = CuroboPlannerServerProcess(
                CuroboPlannerServerProcessConfig(
                    project_root=root,
                    startup_timeout_s=1.0,
                    log_path=root / "planner.log",
                    python_executable="/tmp/python",
                )
            )
            with patch(
                "source.manipulation.planner_server_process.planner_server_ping",
                side_effect=[True, False],
            ), patch(
                "source.manipulation.planner_server_process.planner_server_supports_required_features",
                return_value=False,
            ), patch(
                "source.manipulation.planner_server_process._request_shutdown",
                return_value=True,
            ) as shutdown, patch(
                "source.manipulation.planner_server_process.subprocess.Popen",
                return_value=process,
            ) as popen:
                manager.start()

        shutdown.assert_called_once()
        popen.assert_called_once()
        self.assertTrue(manager.start_report["restarted_incompatible_existing"])


if __name__ == "__main__":
    unittest.main()
