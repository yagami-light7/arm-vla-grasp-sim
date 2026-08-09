from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import source.navigation as navigation
import source.navigation.adapters as adapters
import source.navigation.executor as legacy_executor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_public_navigation_api_does_not_default_to_dwa() -> None:
    """公共导航包不得再把通用名称或执行器默认绑定到 DWA。"""

    assert not hasattr(navigation, "NavPlanner")
    assert not hasattr(navigation, "DwaNavExecutor")
    assert not hasattr(navigation, "DWAExecutor")
    assert not hasattr(legacy_executor, "NavExecutor")
    assert not hasattr(legacy_executor, "DWAExecutor")
    assert legacy_executor.__all__ == ["DwaNavExecutor"]
    assert not hasattr(adapters, "NavPlanner")
    assert adapters.__all__ == []


def test_legacy_run_nav_only_entrypoint_fails_before_isaac_startup() -> None:
    """旧独立 DWA 脚本必须在导入 Isaac Lab 前给出迁移提示。"""

    result = subprocess.run(
        [
            sys.executable,
            "scripts/navigation/run_nav_only.py",
            "--task-json",
            "tasks/nav_smoke_scan_ramp.json",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )

    assert result.returncode == 2
    assert "run_nav_only.py 已在 pct-scan 分支退役" in result.stderr
    assert "PCT→SCAN ROS 2 主链" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
