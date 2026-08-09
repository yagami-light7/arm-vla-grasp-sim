"""从生产任务与统一调参文件生成 multi-floor ROS 探针合同。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import yaml


@dataclass(frozen=True, slots=True)
class MultifloorProbeContract:
    """保存探针实际消费的终点、终端朝向和机体高度。"""

    goal_base_xyz: tuple[float, float, float]
    goal_yaw: float
    body_height_m: float
    raw_goal_base_xyz: tuple[float, float, float]
    ground_surface_z_m: float
    ground_face_index: int
    task_path: Path
    tuning_path: Path


def load_multifloor_probe_contract(
    project_root: str | Path,
) -> MultifloorProbeContract:
    """按 full pipeline 的唯一高度语义解析 place 导航终点。

    任务中的 z 只用于选择楼层；真正发布给 PCT 的 base z 必须和生产
    preflight 一样，由 collision PLY 支撑面加统一配置的 body height 得到。
    """

    root = Path(project_root).resolve()
    task_path = root / "tasks/nav_pick_place_apple_multifloor_pct.json"
    tuning_path = (
        root
        / "ros2_ws/src/isaac_navigation_bridge/config/pct_scan_tuning.yaml"
    )
    collision_ply = root / "source/scene/multifloor/ply/3dgs_collision.ply"

    task = json.loads(task_path.read_text(encoding="utf-8"))
    tuning = yaml.safe_load(tuning_path.read_text(encoding="utf-8"))
    raw_goal = task["place"]["base_goal"]
    raw_xyz = tuple(float(raw_goal[axis]) for axis in ("x", "y", "z"))
    goal_yaw = float(raw_goal["yaw"])
    body_height_m = float(
        tuning["navigation_contract"]["ros__parameters"]["body_height_m"]
    )
    if not all(
        math.isfinite(value) for value in (*raw_xyz, goal_yaw, body_height_m)
    ):
        raise ValueError("multi-floor 任务终点与 body height 必须全部为有限数")
    if body_height_m <= 0.0:
        raise ValueError("统一调参 body_height_m 必须为正数")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from source.navigation.body_height_calibration import (  # noqa: PLC0415
        BodyHeightCalibrationConfig,
        LiveBodyHeightCalibrator,
    )

    calibrator = LiveBodyHeightCalibrator(
        BodyHeightCalibrationConfig(
            collision_ply=collision_ply,
            configured_body_height_hint_m=body_height_m,
            arm_stow_joint_positions=(0.0,) * 6,
            # 历史 task z 只是粗楼层提示，与真实碰撞面可相差约 0.3m。
            maximum_ground_hint_error_m=0.60,
        )
    )
    projection = calibrator.project_ground_surface(
        (raw_xyz[0], raw_xyz[1], raw_xyz[2] - body_height_m)
    )
    effective_xyz = tuple(
        float(value) for value in projection.projected_base_sim_xyz
    )
    return MultifloorProbeContract(
        goal_base_xyz=effective_xyz,
        goal_yaw=goal_yaw,
        body_height_m=body_height_m,
        raw_goal_base_xyz=raw_xyz,
        ground_surface_z_m=float(projection.ground_surface_sim_xyz[2]),
        ground_face_index=int(projection.ground_face_index),
        task_path=task_path,
        tuning_path=tuning_path,
    )
