#!/usr/bin/env python3
"""直接验证官方 PCT backend 能生成真实多楼层地面路径。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


EXPECTED_PYTHON = (3, 10)
START_BASE_XYZ = (-3.50493, 6.74910, 0.1251194919198741)
GOAL_BASE_XYZ = (0.50, 0.30, 3.309128817237528)
BODY_HEIGHT_M = 0.30
UPSTREAM_COMMIT = "35cd73fd82bcd51bc538429294af7646b2a09815"
UPSTREAM_ARCHIVE_SHA256 = (
    "daf5f90b29c76cfa5fc6bf10d6dcfd200c1077778b22671c98aa51f9adb06d64"
)
UPSTREAM_PATCH_ID = (
    "pct-scan-native-astar-cancel-cost-aware-no-corner-cut-gateway-v3"
)
UPSTREAM_PATCH_SHA256 = (
    "fa6f9364b7bdf07a9d698e40083617646829608e64727bf5c68df391589b4e1b"
)
NATIVE_MODULE_NAMES = ("a_star", "ele_planner", "py_map_manager", "traj_opt")
EXPECTED_ASSET_KIND = "upstream_official_semantics_with_stair_profile_v1"
EXPECTED_CORE_MODE = "offline_ele_planner_native_astar_ground"


def _parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description="调用固定提交的官方 PCT TomogramPlanner 规划真实跨层路径",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="pct_scan worktree 根目录",
    )
    return parser.parse_args()


def _maximum_segment_length(
    points: tuple[tuple[float, float, float], ...],
) -> float:
    return max(
        math.dist(first, second)
        for first, second in zip(points, points[1:])
    )


def _validate_native_modules(source_root: Path) -> dict[str, str]:
    module_paths: dict[str, str] = {}
    planner_lib = (source_root / "planner/lib").resolve()
    for name in NATIVE_MODULE_NAMES:
        matches = tuple(planner_lib.glob(f"{name}.cpython-310-*.so"))
        if len(matches) != 1:
            raise AssertionError(f"官方扩展 {name} 的 ABI 产物数量不是 1")
        resolved = matches[0].resolve()
        if ".cpython-310-" not in resolved.name:
            raise AssertionError(
                f"官方扩展 {name} 不是 CPython 3.10 ABI：{resolved.name}"
            )
        module_paths[name] = str(resolved)
    for name in ("a_star", "ele_planner", "traj_opt"):
        module = sys.modules.get(f"lib.{name}")
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != Path(
            module_paths[name]
        ):
            raise AssertionError(f"官方 wrapper 未从固定目录载入 lib.{name}")
    return module_paths


def _run(project_root: Path) -> dict[str, object]:
    project_root = project_root.expanduser().resolve()
    package_source = project_root / "ros2_ws/src/pct_ros2_adapter"
    sys.path.insert(0, str(package_source))

    from pct_ros2_adapter.backend import (  # noqa: PLC0415
        PCTBackendConfig,
        create_global_planner_backend,
    )

    upstream_root = project_root / "external/PCT_planner"
    tomogram = (
        project_root / "source/scene/multifloor/mutifloor_upstream.pickle"
    )
    walkable = (
        project_root / "source/scene/multifloor/mutifloor_ply_walkable.npy"
    )
    collision_ply = (
        project_root / "source/scene/multifloor/ply/3dgs_collision.ply"
    )
    stair_profile = (
        project_root / "configs/navigation/pct_multifloor_stair_profile.json"
    )
    required_paths = (
        upstream_root,
        tomogram,
        walkable,
        collision_ply,
        stair_profile,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"官方 PCT 探针缺少输入：{missing}")

    config = PCTBackendConfig(
        project_root=project_root,
        backend_kind="upstream",
        upstream_source_root=upstream_root,
        tomogram_path=tomogram,
        walkable_path=walkable,
        collision_ply_path=collision_ply,
        upstream_stair_profile_path=stair_profile,
        upstream_body_clearance_enabled=True,
        upstream_body_clearance_radius_m=0.80,
        upstream_body_clearance_maximum_cost=20.0,
        upstream_body_clearance_power=2.0,
        slice_query_root_to_floor_m=BODY_HEIGHT_M,
        goal_base_to_ground_m=BODY_HEIGHT_M,
        path_sample_spacing_m=0.20,
    )
    started_at = time.perf_counter()
    backend = create_global_planner_backend(config)
    initialized_sec = time.perf_counter() - started_at
    native_modules = _validate_native_modules(upstream_root)

    planning_started_at = time.perf_counter()
    plan = backend.plan(
        start_base_xyz=START_BASE_XYZ,
        goal_base_xyz=GOAL_BASE_XYZ,
        goal_yaw=0.0,
    )
    planning_sec = time.perf_counter() - planning_started_at
    metadata = plan.metadata
    points = plan.points_xyz

    if metadata.get("backend_kind") != "upstream":
        raise AssertionError(f"误用了非官方 backend：{metadata.get('backend_kind')}")
    if metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise AssertionError("官方 PCT commit 身份不符")
    if metadata.get("upstream_archive_sha256") != UPSTREAM_ARCHIVE_SHA256:
        raise AssertionError("官方 PCT 归档哈希不符")
    if metadata.get("upstream_patch_id") != UPSTREAM_PATCH_ID:
        raise AssertionError("pct-scan 原生补丁身份不符")
    if metadata.get("upstream_patch_sha256") != UPSTREAM_PATCH_SHA256:
        raise AssertionError("pct-scan 原生补丁哈希不符")
    if metadata.get("upstream_native_cancel_supported") is not True:
        raise AssertionError("官方 PCT 原生 A* 没有声明可中断取消")
    if metadata.get("upstream_native_gil_released") is not True:
        raise AssertionError("官方 PCT 原生调用没有声明释放 Python GIL")
    if metadata.get("transport") != "direct_in_process_ros2":
        raise AssertionError("官方 PCT 未使用 ROS 2 进程内调用边界")
    if metadata.get("height_semantics") != "ground_height":
        raise AssertionError("官方 PCT 输出不是统一的 ground_height 语义")
    if metadata.get("start_layer") == metadata.get("goal_layer"):
        raise AssertionError("真实多楼层端点被错误映射到同一逻辑层")
    if metadata.get("start_layer_height_error_m", math.inf) > 0.25:
        raise AssertionError("起点逻辑层与真实地面高度不匹配")
    if metadata.get("goal_layer_height_error_m", math.inf) > 0.25:
        raise AssertionError("终点逻辑层与真实地面高度不匹配")
    if metadata.get("upstream_gateway_up_count", 0) < 1:
        raise AssertionError("官方多楼层资产没有向上 gateway")
    if metadata.get("upstream_gateway_down_count", 0) < 1:
        raise AssertionError("官方多楼层资产没有向下 gateway")
    if metadata.get("upstream_core_invoked") is not True:
        raise AssertionError("真实跨层路径没有调用官方 native core")
    if metadata.get("upstream_core_mode") != EXPECTED_CORE_MODE:
        raise AssertionError("PCT 主线没有使用只含原生 A* 的全局规划边界")
    if not math.isclose(
        float(metadata.get("upstream_astar_step_cost_weight", math.nan)),
        0.20,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("PCT A* 没有使用已验证的 0.20 地形代价权重")
    clearance_report = metadata.get("upstream_body_clearance_overlay", {})
    if clearance_report.get("enabled") is not True:
        raise AssertionError("真实跨层路径没有启用 Go2-X5 机身软净空层")
    if clearance_report.get("changed_cell_count", 0) < 1:
        raise AssertionError("机身软净空层没有改变任何可通行单元")
    if clearance_report.get("gateway_source") != "original_traversability":
        raise AssertionError("楼梯 gateway 没有从原始 tomogram 独立推导")
    if metadata.get("upstream_tomogram_asset_kind") != EXPECTED_ASSET_KIND:
        raise AssertionError("PCT 没有加载固定楼梯 profile 的官方语义资产")
    stair_report = metadata.get("upstream_stair_centerline_report", {})
    if stair_report.get("applied") is not True:
        raise AssertionError(f"跨层路径没有归一到标定楼梯中心线：{stair_report}")
    if stair_report.get("reason") != "calibrated_stair_centerline":
        raise AssertionError(f"楼梯中心线归一原因异常：{stair_report}")
    approach_report = metadata.get(
        "upstream_stair_approach_clearance_report",
        {},
    )
    if approach_report.get("enabled") is not True:
        raise AssertionError("跨层路径没有启用一楼楼梯接近段净空审计")
    if approach_report.get("collision_free") is not True:
        raise AssertionError(
            f"一楼楼梯接近段没有通过双圆柱净空审计：{approach_report}"
        )
    if float(approach_report.get("minimum_surface_clearance_m", 0.0)) < (
        float(
            approach_report.get(
                "minimum_required_surface_clearance_m",
                math.inf,
            )
        )
    ):
        raise AssertionError(
            f"一楼楼梯接近段净空低于合同：{approach_report}"
        )
    shortcut_report = metadata.get(
        "upstream_same_layer_shortcut_report",
        {},
    )
    if shortcut_report.get("applied") is not True:
        raise AssertionError(f"跨层平地折线没有安全压缩：{shortcut_report}")
    if shortcut_report.get("reason") != (
        "cross_layer_floor_segments_clearance_shortcut"
    ):
        raise AssertionError(f"跨层平地压缩原因异常：{shortcut_report}")
    if shortcut_report.get("blocked_cell_source") != (
        "native_traversability_plus_body_obstacle_surface"
    ):
        raise AssertionError(f"跨层平地压缩没有使用机身障碍面：{shortcut_report}")
    required_shortcut_center_clearance_m = float(
        shortcut_report["clearance_m"]
    ) + float(shortcut_report["grid_cell_cover_margin_m"])
    for floor_name in ("start_floor_report", "goal_floor_report"):
        floor_report = shortcut_report.get(floor_name, {})
        verified_clearance = floor_report.get(
            "minimum_verified_segment_clearance_m"
        )
        if verified_clearance is None or float(verified_clearance) + 1.0e-9 < (
            required_shortcut_center_clearance_m
        ):
            raise AssertionError(
                f"跨层平地捷径没有满足离散净空门：{floor_report}"
            )
    profile_payload = json.loads(stair_profile.read_text(encoding="utf-8"))
    approach_anchors = tuple(
        tuple(float(value) for value in point)
        for point in profile_payload["lower_floor_approach"][
            "anchors_sim_ground_xyz"
        ]
    )
    maximum_approach_xy_error_m = max(
        min(math.dist(anchor[:2], point[:2]) for point in points)
        for anchor in approach_anchors
    )
    if maximum_approach_xy_error_m > 1.0e-6:
        raise AssertionError(
            "发布 Path 没有保持标定的一楼接近曲线 XY："
            f"{maximum_approach_xy_error_m:.9f} m"
        )
    profile_anchors = tuple(
        tuple(float(value) for value in point)
        for point in profile_payload["anchors_sim_ground_xyz"]
    )
    maximum_anchor_xy_error_m = max(
        min(math.dist(anchor[:2], point[:2]) for point in points)
        for anchor in profile_anchors
    )
    if maximum_anchor_xy_error_m > 1.0e-6:
        raise AssertionError(
            "发布 Path 没有保持标定楼梯中心线 XY："
            f"{maximum_anchor_xy_error_m:.9f} m"
        )
    if len(points) < 2:
        raise AssertionError("官方 PCT 返回的路径点不足")
    if not all(math.isfinite(value) for point in points for value in point):
        raise AssertionError("官方 PCT 路径包含非有限值")
    first_floor_wall_corridor = tuple(
        point
        for point in points
        if -2.30 <= point[0] <= -1.50 and point[2] < 0.0
    )
    if not first_floor_wall_corridor:
        raise AssertionError("官方 PCT 路径没有经过一楼墙边回归区间")
    maximum_wall_corridor_y = max(
        point[1] for point in first_floor_wall_corridor
    )
    if maximum_wall_corridor_y > 4.95:
        raise AssertionError(
            "官方 PCT A* 重新退化为一楼贴墙最短路："
            f"maximum_y={maximum_wall_corridor_y:.3f} m"
        )

    start_xy_error = math.dist(points[0][:2], START_BASE_XYZ[:2])
    goal_xy_error = math.dist(points[-1][:2], GOAL_BASE_XYZ[:2])
    start_base_z_error = abs(
        points[0][2] + BODY_HEIGHT_M - START_BASE_XYZ[2]
    )
    goal_base_z_error = abs(
        points[-1][2] + BODY_HEIGHT_M - GOAL_BASE_XYZ[2]
    )
    if start_xy_error > 1.0e-6 or goal_xy_error > 1.0e-6:
        raise AssertionError(
            "官方路径没有保持请求端点 XY："
            f"start={start_xy_error:.9f}, goal={goal_xy_error:.9f}"
        )
    if start_base_z_error > 0.08 or goal_base_z_error > 0.08:
        raise AssertionError(
            "ground_height 加一次 body_height 后不能还原请求 base z："
            f"start={start_base_z_error:.6f}, goal={goal_base_z_error:.6f}"
        )
    vertical_span = max(point[2] for point in points) - min(
        point[2] for point in points
    )
    if vertical_span < 2.50:
        raise AssertionError(f"官方路径没有形成跨楼层高度变化：{vertical_span:.3f} m")
    maximum_segment_m = _maximum_segment_length(points)
    route_length_m = sum(
        math.dist(start, end) for start, end in zip(points, points[1:])
    )
    if maximum_segment_m > 0.30:
        raise AssertionError(
            f"官方路径重采样间距异常：最大段长 {maximum_segment_m:.3f} m"
        )

    return {
        "result": "PASS",
        "backend_kind": metadata["backend_kind"],
        "upstream_commit": metadata["upstream_commit"],
        "upstream_archive_sha256": metadata["upstream_archive_sha256"],
        "upstream_patch_id": metadata["upstream_patch_id"],
        "upstream_patch_sha256": metadata["upstream_patch_sha256"],
        "upstream_native_cancel_supported": metadata[
            "upstream_native_cancel_supported"
        ],
        "upstream_native_gil_released": metadata[
            "upstream_native_gil_released"
        ],
        "upstream_license": metadata["upstream_license"],
        "transport": metadata["transport"],
        "height_semantics": metadata["height_semantics"],
        "start_layer": metadata["start_layer"],
        "goal_layer": metadata["goal_layer"],
        "start_layer_height_error_m": metadata[
            "start_layer_height_error_m"
        ],
        "goal_layer_height_error_m": metadata["goal_layer_height_error_m"],
        "gateway_up_count": metadata["upstream_gateway_up_count"],
        "gateway_down_count": metadata["upstream_gateway_down_count"],
        "tomogram_asset_kind": metadata["upstream_tomogram_asset_kind"],
        "upstream_core_mode": metadata["upstream_core_mode"],
        "upstream_astar_step_cost_weight": metadata[
            "upstream_astar_step_cost_weight"
        ],
        "upstream_body_clearance_overlay": clearance_report,
        "maximum_first_floor_wall_corridor_y_m": maximum_wall_corridor_y,
        "cross_layer_floor_shortcut_report": shortcut_report,
        "stair_centerline_report": stair_report,
        "point_count": len(points),
        "vertical_span_m": vertical_span,
        "maximum_segment_m": maximum_segment_m,
        "route_length_m": route_length_m,
        "maximum_stair_anchor_xy_error_m": maximum_anchor_xy_error_m,
        "start_xy_error_m": start_xy_error,
        "goal_xy_error_m": goal_xy_error,
        "start_base_z_error_m": start_base_z_error,
        "goal_base_z_error_m": goal_base_z_error,
        "initialization_sec": initialized_sec,
        "planning_sec": planning_sec,
        "native_modules": native_modules,
    }


def main() -> int:
    """执行探针并以机器可读 JSON 汇报证据。"""

    if sys.version_info[:2] != EXPECTED_PYTHON:
        print(
            json.dumps(
                {
                    "result": "BLOCKED",
                    "reason": (
                        "官方扩展固定使用 CPython 3.10 ABI；"
                        "请用 /usr/bin/python3 执行"
                    ),
                    "actual_python": sys.version,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    args = _parse_args()
    try:
        report = _run(args.project_root)
    except Exception as exc:  # 探针必须把启动硬失败明确交给自动化。
        print(
            json.dumps(
                {
                    "result": "FAIL",
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
