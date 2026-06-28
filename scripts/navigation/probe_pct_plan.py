"""在不启动 Isaac Sim 的情况下探测 PCT 多楼层规划链路。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.interfaces.navigation import NavGoal
from source.interfaces.simulation import SimulationState
from source.navigation.pct_adapter import PCTNavPlanner, PCTPlannerConfig
from source.pipeline.config import NavigationSettings


DEFAULT_TASK_JSON = PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
DEFAULT_SERVER_SCRIPT = PROJECT_ROOT / "scripts/navigation/pct_grid_server.py"
DEFAULT_TOMOGRAM = PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle"
DEFAULT_WALKABLE = PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/pct_plan_probe.json"


def main() -> int:
    args = _parse_args()
    task_json = _project_path(args.task_json)
    tomogram_path = _project_path(args.pct_tomogram_path)
    walkable_path = _project_path(args.pct_walkable_path)
    server_script = _optional_project_path(args.pct_server_script)
    planner_root = _optional_project_path(args.pct_planner_root)
    server_python = _optional_project_path(args.pct_server_python)
    output_json = _project_path(args.output_json)

    report: dict[str, Any] = {
        "task_json": str(task_json),
        "pct_planner_root": str(planner_root) if planner_root else None,
        "pct_server_script": str(server_script) if server_script else None,
        "pct_server_python": str(server_python) if server_python else None,
        "pct_tomogram_path": str(tomogram_path),
        "pct_walkable_path": str(walkable_path),
        "coord_mode": args.pct_coord_mode,
        "cross_floor_gateway_points": _parse_xyz_points(
            args.pct_cross_floor_gateway,
            default=NavigationSettings().pct_cross_floor_gateway_points,
        ),
        "cross_floor_stair_exit_points": _parse_xyz_points(
            args.pct_cross_floor_stair_exit,
            default=NavigationSettings().pct_cross_floor_stair_exit_points,
        ),
        "cross_floor_stair_midpoint_points": _parse_xyz_points(
            args.pct_cross_floor_stair_midpoint,
            default=NavigationSettings().pct_cross_floor_stair_midpoint_points,
        ),
        "cross_floor_gateway_radius_m": float(args.pct_cross_floor_gateway_radius),
        "stair_vertical_radius_m": float(args.pct_stair_vertical_radius),
        "stair_progress_tolerance": float(args.pct_stair_progress_tolerance),
        "stair_progress_cost_weight": float(args.pct_stair_progress_cost_weight),
        "segments": [],
        "checks": {},
    }
    task = _load_task(task_json, report)
    missing = _collect_missing_inputs(
        report=report,
        task_json=task_json,
        task=task,
        planner_root=planner_root,
        server_script=server_script,
        tomogram_path=tomogram_path,
        walkable_path=walkable_path,
    )
    report["ready_for_pct_server_plan"] = not missing
    report["missing"] = missing

    if args.dry_run:
        _write_report(output_json, report)
        _print_summary(report)
        return 0
    if missing:
        _write_report(output_json, report)
        _print_summary(report)
        return 2

    planner = PCTNavPlanner(
        PCTPlannerConfig(
            enabled=True,
            planner_root=planner_root,
            server_script=server_script,
            server_python=server_python,
            tomogram_path=tomogram_path,
            walkable_path=walkable_path,
            startup_timeout_s=float(args.startup_timeout_s),
            request_timeout_s=float(args.request_timeout_s),
            coord_mode=args.pct_coord_mode,
            pct_offset_x=float(args.pct_offset_x),
            pct_offset_y=float(args.pct_offset_y),
            pct_scale_x=float(args.pct_scale_x),
            pct_scale_y=float(args.pct_scale_y),
            cross_floor_gateway_points=tuple(
                tuple(point)
                for point in report["cross_floor_gateway_points"]
            ),
            cross_floor_stair_exit_points=tuple(
                tuple(point)
                for point in report["cross_floor_stair_exit_points"]
            ),
            cross_floor_stair_midpoint_points=tuple(
                tuple(point)
                for point in report["cross_floor_stair_midpoint_points"]
            ),
            cross_floor_gateway_radius_m=float(args.pct_cross_floor_gateway_radius),
            stair_vertical_radius_m=float(args.pct_stair_vertical_radius),
            stair_progress_tolerance=float(args.pct_stair_progress_tolerance),
            stair_progress_cost_weight=float(args.pct_stair_progress_cost_weight),
            fallback_to_astar=False,
        )
    )
    try:
        for segment in _segments_from_task(task):
            plan = planner.plan(segment["state"], segment["goal"])
            path_3d = plan.metadata.get("path_3d", ())
            report["segments"].append(
                {
                    "name": segment["name"],
                    "goal": _goal_report(segment["goal"]),
                    "waypoint_count": len(plan.waypoints),
                    "path_3d_count": len(path_3d),
                    "start_z": segment["state"].robot_root_pose[2],
                    "end_z": segment["goal"].z,
                    "z_delta": None
                    if segment["goal"].z is None
                    else float(segment["goal"].z) - float(segment["state"].robot_root_pose[2]),
                    "slice_start": plan.metadata.get("slice_start"),
                    "slice_end": plan.metadata.get("slice_end"),
                    "snap_start_dist": plan.metadata.get("snap_start_dist"),
                    "snap_end_dist": plan.metadata.get("snap_end_dist"),
                    "path_3d": path_3d,
                    "waypoints_xy": plan.waypoints,
                    "metadata": {
                        key: plan.metadata.get(key)
                        for key in (
                            "planner",
                            "pct_status",
                            "pct_path_mode",
                            "cross_floor",
                            "slice_start",
                            "slice_end",
                            "snap_start_dist",
                            "snap_end_dist",
                            "hard_obstacle_cells",
                            "hard_obstacle_mode",
                            "hard_obstacle_min_slices",
                            "default_hard_obstacle_min_slices",
                            "cross_floor_hard_obstacle_min_slices",
                            "cross_floor_gateway_count",
                            "cross_floor_stair_exit_count",
                            "cross_floor_stair_midpoint_count",
                            "cross_floor_gateway_radius_m",
                            "cross_floor_gateway_cells",
                            "cross_floor_stair_vertical_cells",
                            "cross_floor_gateway_mode",
                            "robot_root_to_floor_m",
                            "planning_start_z",
                            "planning_end_z",
                            "stair_vertical_radius_m",
                            "stair_constraint_mode",
                            "stair_progress_tolerance",
                            "stair_progress_cost_weight",
                        )
                    },
                    "pct_start": plan.metadata.get("pct_start"),
                    "pct_end": plan.metadata.get("pct_end"),
                }
            )
    finally:
        planner.close()

    report["ready_for_pct_server_plan"] = True
    report["planned"] = True
    _write_report(output_json, report)
    _print_summary(report)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查并探测 PCT 多楼层规划链路，不启动 Isaac Sim。",
    )
    parser.add_argument("--task-json", default=str(DEFAULT_TASK_JSON), help="多楼层任务 JSON。")
    parser.add_argument(
        "--pct-planner-root",
        default=None,
        help="兼容旧外部入口的 PCT 根目录；默认不依赖 external/PCT。",
    )
    parser.add_argument(
        "--pct-server-script",
        default=str(DEFAULT_SERVER_SCRIPT),
        help="PCT server 脚本路径，默认使用本仓库本地迁移版本。",
    )
    parser.add_argument("--pct-server-python", default=None, help="运行 PCT server 的 Python。")
    parser.add_argument("--pct-tomogram-path", default=str(DEFAULT_TOMOGRAM), help="PCT tomogram pickle。")
    parser.add_argument("--pct-walkable-path", default=str(DEFAULT_WALKABLE), help="PCT walkable npy。")
    parser.add_argument("--pct-coord-mode", default="sim_to_pct_180deg", choices=["sim_to_pct_180deg", "identity"])
    parser.add_argument("--pct-offset-x", type=float, default=0.0)
    parser.add_argument("--pct-offset-y", type=float, default=0.0)
    parser.add_argument("--pct-scale-x", type=float, default=1.0)
    parser.add_argument("--pct-scale-y", type=float, default=1.0)
    parser.add_argument(
        "--pct-cross-floor-gateway",
        action="append",
        default=None,
        help="允许跨楼层换 slice 的楼梯/坡道中心点，Isaac Sim 坐标 x,y,z；可重复传入。",
    )
    parser.add_argument(
        "--pct-cross-floor-gateway-radius",
        type=float,
        default=NavigationSettings().pct_cross_floor_gateway_radius_m,
        help="跨楼层 gateway 的 XY 半径，单位米。",
    )
    parser.add_argument(
        "--pct-cross-floor-stair-exit",
        action="append",
        default=None,
        help="楼梯上层出口的 Isaac Sim 坐标 x,y,z；与 gateway 按顺序配对。",
    )
    parser.add_argument(
        "--pct-cross-floor-stair-midpoint",
        action="append",
        default=None,
        help="楼梯中间拐角/平台控制点的 Isaac Sim 坐标 x,y,z；可重复传入。",
    )
    parser.add_argument(
        "--pct-stair-vertical-radius",
        type=float,
        default=NavigationSettings().pct_stair_vertical_radius_m,
        help="楼梯跨 slice 换层允许的中心带半径，单位米。",
    )
    parser.add_argument(
        "--pct-stair-progress-tolerance",
        type=float,
        default=NavigationSettings().pct_stair_progress_tolerance,
        help="楼梯高度 slice 与入口到出口进度匹配的容差。",
    )
    parser.add_argument(
        "--pct-stair-progress-cost-weight",
        type=float,
        default=NavigationSettings().pct_stair_progress_cost_weight,
        help="楼梯高度 slice 与折线进度不匹配时的软代价权重。",
    )
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--request-timeout-s", type=float, default=10.0)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT), help="探针报告输出路径。")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入是否齐全，不启动 PCT server。")
    return parser.parse_args()


def _parse_xyz_points(
    raw_values: list[str] | None,
    *,
    default: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    """解析 CLI 中重复传入的 x,y,z 点列表。"""

    if raw_values is None:
        return default
    points: list[tuple[float, float, float]] = []
    for raw_value in raw_values:
        text = raw_value.strip()
        if text.lower() in {"", "none", "off", "disable", "disabled"}:
            return ()
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"坐标点必须使用 x,y,z 格式: {raw_value}")
        points.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return tuple(points)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _optional_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return _project_path(text)


def _load_task(task_json: Path, report: dict[str, Any]) -> dict[str, Any] | None:
    if not task_json.is_file():
        report["task_error"] = f"task JSON not found: {task_json}"
        return None
    try:
        payload = json.loads(task_json.read_text(encoding="utf-8"))
    except Exception as exc:
        report["task_error"] = f"failed to read task JSON: {exc}"
        return None
    return payload if isinstance(payload, dict) else None


def _collect_missing_inputs(
    *,
    report: dict[str, Any],
    task_json: Path,
    task: dict[str, Any] | None,
    planner_root: Path | None,
    server_script: Path | None,
    tomogram_path: Path,
    walkable_path: Path,
) -> list[str]:
    checks = report["checks"]
    checks["task_json_exists"] = task_json.is_file()
    checks["task_loaded"] = task is not None
    checks["tomogram_exists"] = tomogram_path.is_file()
    checks["walkable_exists"] = walkable_path.is_file()
    resolved_server = _resolve_server_script(planner_root, server_script)
    checks["server_script"] = str(resolved_server) if resolved_server else None
    checks["server_script_exists"] = resolved_server is not None and resolved_server.is_file()
    checks["using_external_pct_server"] = _is_under_root(resolved_server, planner_root)
    checks["pct_planner_root_exists"] = None if planner_root is None else planner_root.is_dir()
    checks["pct_mutifloor_dir_exists"] = (
        None if planner_root is None else (planner_root / "mutifloor").is_dir()
    )
    checks["pct_mutifloor_assets"] = (
        {} if planner_root is None else _pct_mutifloor_asset_checks(planner_root)
    )
    checks["pct_server_has_home_y_paths"] = (
        _server_has_home_y_paths(resolved_server) if checks["using_external_pct_server"] else False
    )
    checks["planner_wrapper_available_under_pct_root"] = (
        None if planner_root is None else _planner_wrapper_exists(planner_root)
    )
    missing: list[str] = []
    if not checks["task_json_exists"]:
        missing.append("task_json")
    if task is None:
        missing.append("task_loaded")
    if not checks["tomogram_exists"]:
        missing.append("pct_tomogram_path")
    if not checks["walkable_exists"]:
        missing.append("pct_walkable_path")
    if not checks["server_script_exists"]:
        missing.append("pct_server_script_or_pct_planner_root")
    warnings: list[str] = report.setdefault("warnings", [])
    if checks["using_external_pct_server"]:
        if not checks["pct_mutifloor_dir_exists"]:
            warnings.append("external/PCT 仅作为参考，当前外部入口缺少 mutifloor/ 场景资产")
        missing_pct_assets = [
            name for name, exists in checks["pct_mutifloor_assets"].items() if not exists
        ]
        if missing_pct_assets:
            warnings.append(f"external/PCT mutifloor/ 缺少 README 资产: {', '.join(missing_pct_assets)}")
        if checks["pct_server_has_home_y_paths"]:
            warnings.append("外部 pct_server.py 含 /home/y 硬编码；请先迁移到本仓库本地脚本后再运行")
        if not checks["planner_wrapper_available_under_pct_root"]:
            warnings.append("外部 pct_planner_root 下没有找到 planner_wrapper backend")
    return missing


def _resolve_server_script(planner_root: Path | None, server_script: Path | None) -> Path | None:
    if server_script is not None:
        return server_script
    if planner_root is None:
        return None
    return planner_root / "scripts/navigation/pct_server.py"


def _server_has_home_y_paths(server_script: Path | None) -> bool:
    if server_script is None or not server_script.is_file():
        return False
    try:
        return "/home/y/" in server_script.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def _planner_wrapper_exists(planner_root: Path | None) -> bool:
    if planner_root is None or not planner_root.is_dir():
        return False
    patterns = ("planner_wrapper.py", "planner_wrapper*.so", "planner_wrapper*.pyd")
    for pattern in patterns:
        if any(planner_root.rglob(pattern)):
            return True
    return False


def _is_under_root(path: Path | None, root: Path | None) -> bool:
    if path is None or root is None:
        return False
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _pct_mutifloor_asset_checks(planner_root: Path | None) -> dict[str, bool]:
    if planner_root is None:
        return {
            "3dgs_collision_cropped.ply": False,
            "3dgs_cropped.ply": False,
            "mutifloor_collision_cropped.usd": False,
            "mutifloor_cropped.usda": False,
        }
    mutifloor_dir = planner_root / "mutifloor"
    return {
        "3dgs_collision_cropped.ply": (mutifloor_dir / "3dgs_collision_cropped.ply").is_file(),
        "3dgs_cropped.ply": (mutifloor_dir / "3dgs_cropped.ply").is_file(),
        "mutifloor_collision_cropped.usd": (mutifloor_dir / "mutifloor_collision_cropped.usd").is_file(),
        "mutifloor_cropped.usda": (mutifloor_dir / "mutifloor_cropped.usda").is_file(),
    }


def _segments_from_task(task: dict[str, Any] | None) -> list[dict[str, Any]]:
    if task is None:
        return []
    start = _pose_dict(task.get("start"), fallback_z=0.0)
    pick = _pose_dict(_nested(task, "pick", "base_goal"), fallback_z=start["z"])
    place = _pose_dict(_nested(task, "place", "base_goal"), fallback_z=pick["z"])
    return [
        {
            "name": "start_to_pick",
            "state": _state_from_pose(start),
            "goal": _goal_from_pose(pick),
        },
        {
            "name": "pick_to_place",
            "state": _state_from_pose(pick),
            "goal": _goal_from_pose(place),
        },
    ]


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _pose_dict(value: Any, *, fallback_z: float) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    return {
        "x": float(payload.get("x", 0.0)),
        "y": float(payload.get("y", 0.0)),
        "z": float(payload.get("z", fallback_z)),
        "yaw": float(payload.get("yaw", 0.0)),
        "floor_id": payload.get("floor_id"),
        "slice_id": payload.get("slice_id"),
    }


def _state_from_pose(pose: dict[str, Any]) -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(float(pose["x"]), float(pose["y"]), float(pose["z"]), 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metadata={"floor_id": pose.get("floor_id"), "slice_id": pose.get("slice_id")},
    )


def _goal_from_pose(pose: dict[str, Any]) -> NavGoal:
    slice_id = pose.get("slice_id")
    return NavGoal(
        x=float(pose["x"]),
        y=float(pose["y"]),
        z=float(pose["z"]),
        yaw=float(pose["yaw"]),
        floor_id=pose.get("floor_id"),
        slice_id=int(slice_id) if slice_id is not None else None,
    )


def _goal_report(goal: NavGoal) -> dict[str, Any]:
    return {
        "x": goal.x,
        "y": goal.y,
        "z": goal.z,
        "yaw": goal.yaw,
        "floor_id": goal.floor_id,
        "slice_id": goal.slice_id,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
