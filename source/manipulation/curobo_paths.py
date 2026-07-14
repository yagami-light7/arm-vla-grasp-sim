"""cuRobo 项目路径解析工具。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root_from_env(default: Path = PROJECT_ROOT) -> Path:
    """读取项目根目录；未设置时根据当前文件位置推导。"""

    raw_path = os.environ.get("GO2_X5_WORKSPACE")
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return default.resolve()


def curobo_source_root_from_env(workspace: Path) -> Path:
    """读取 cuRobo 源码目录；默认使用仓库内的 external/curobo。"""

    raw_path = os.environ.get("GO2_X5_CUROBO_SOURCE_ROOT")
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return (workspace / "external/curobo").resolve()


def load_project_robot_config(robot_yaml: Path) -> dict[str, Any]:
    """加载项目 cuRobo YAML，并按配置文件位置补齐运行时路径。

    cuRobo 的配置格式要求 asset_root_path 和 urdf_path 在运行时可直接访问。
    YAML 文件只保存项目内相对路径；真正创建 planner 时根据当前仓库位置解析，
    因此复制仓库或更换部署目录后无需修改配置文件。
    """

    robot_yaml = robot_yaml.expanduser().resolve()
    with robot_yaml.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"cuRobo robot YAML 顶层必须是字典: {robot_yaml}")

    robot_config = data.get("robot_cfg", data)
    if not isinstance(robot_config, dict):
        raise ValueError(f"cuRobo robot_cfg 必须是字典: {robot_yaml}")
    kinematics = robot_config.get("kinematics", robot_config)
    if not isinstance(kinematics, dict):
        raise ValueError(f"cuRobo kinematics 必须是字典: {robot_yaml}")

    asset_root = robot_yaml.parent.parent
    urdf_path = robot_yaml.parent / f"{robot_yaml.stem}.urdf"
    kinematics["asset_root_path"] = str(asset_root)
    kinematics["urdf_path"] = str(urdf_path)
    return data
