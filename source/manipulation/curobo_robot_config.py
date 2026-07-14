"""Workspace-aware loading for the Go2-X5 CuRobo robot config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROBOT_ASSET_ROOT = Path("source/robot/go2_x5")
ROBOT_CONFIG_PATH = ROBOT_ASSET_ROOT / "curobo/go2_x5_arm.yml"
ROBOT_URDF_PATH = ROBOT_ASSET_ROOT / "curobo/go2_x5_arm.urdf"


def load_workspace_robot_config(
    workspace: str | Path,
    *,
    robot_yaml: str | Path | None = None,
) -> dict[str, Any]:
    """Load the robot YAML and resolve repository-local assets for CuRobo.

    CuRobo interprets relative kinematics paths against its own package asset
    directory, not against the YAML file.  The project keeps portable relative
    paths in source control and resolves them from ``GO2_X5_WORKSPACE`` at the
    planner process boundary.
    """

    workspace_path = Path(workspace).expanduser().resolve()
    yaml_path = (
        Path(robot_yaml).expanduser().resolve()
        if robot_yaml is not None
        else workspace_path / ROBOT_CONFIG_PATH
    )
    asset_root = (workspace_path / ROBOT_ASSET_ROOT).resolve()
    urdf_path = (workspace_path / ROBOT_URDF_PATH).resolve()

    if not yaml_path.is_file():
        raise FileNotFoundError(f"CuRobo robot config does not exist: {yaml_path}")
    if not asset_root.is_dir():
        raise FileNotFoundError(f"CuRobo robot asset root does not exist: {asset_root}")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"CuRobo robot URDF does not exist: {urdf_path}")

    with yaml_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"CuRobo robot config must contain a mapping: {yaml_path}")

    robot_config = config.get("robot_cfg", config)
    if not isinstance(robot_config, dict):
        raise ValueError(f"CuRobo robot_cfg must contain a mapping: {yaml_path}")
    kinematics = robot_config.get("kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError(f"CuRobo kinematics config is missing: {yaml_path}")

    kinematics["asset_root_path"] = str(asset_root)
    kinematics["urdf_path"] = str(urdf_path)
    return config
