#!/usr/bin/env python3
"""Run the full-physics pipeline dry run or real Isaac simulation smoke."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.pipeline import (  # noqa: E402
    BaseGoalRandomizationSettings,
    DEFAULT_OVERVIEW_CAMERA_PRIM_PATH,
    FullPhysicsConfig,
    LocomotionPolicySettings,
    ManipulationSettings,
    NavigationSettings,
    PCT_MULTIFLOOR_LOCOMOTION_TASK,
    RandomizationSettings,
    RecordingSettings,
    SceneLightingSettings,
    VideoRecordingSettings,
)
from source.pipeline.dry_run import create_dry_run_pipeline  # noqa: E402
from source.pipeline.isaac_compat import patch_numpy_for_isaacsim  # noqa: E402
from source.scene.profiles import (  # noqa: E402
    SceneProfileError,
    apply_scene_profile_defaults,
    check_scene_profile_assets,
    list_scene_profiles,
    load_scene_profile,
)
from source.scene.runtime_assets import (  # noqa: E402
    materialize_scene_asset_bindings,
    write_scene_binding_report,
)
from source.tasks import JsonTaskProvider, prepare_episode_spec  # noqa: E402


DEFAULT_SCENE_PROFILE = "liangzhu"

# NuRec 体渲染必须在 Kit 首次 Hydra 同步之前锁定为单 GPU；运行期再改设置已太晚。
NUREC_KIT_ARGS = (
    "--/renderer/multiGpu/enabled=false",
    "--/renderer/multiGpu/autoEnable=false",
    "--/renderer/multiGpu/maxGpuCount=1",
    "--/rtx/post/aa/op=1",
    "--/rtx-defaults/post/aa/op=1",
    "--/rtx/rtpt/gaussian/skipTonemapping/enabled=false",
)


def _scene_isaac_kit_args(scene_profile: Any) -> tuple[str, ...]:
    """返回必须在 Isaac App 创建前应用的场景渲染参数。"""

    if scene_profile.supports("nurec_visual"):
        return NUREC_KIT_ARGS
    return ()


def _scene_isaac_app_overrides(scene_profile: Any) -> dict[str, Any]:
    """返回 SimulationApp 会在启动后再次写入的场景专用配置。"""

    if scene_profile.supports("nurec_visual"):
        return {
            "multi_gpu": False,
            # Isaac Sim 5.1 的 NuRec volume 在 DLSS render product 路径会触发 CUDA 700。
            "anti_aliasing": 1,
        }
    return {}


def _scene_isaac_runtime_overrides(scene_profile: Any) -> dict[str, Any]:
    """返回环境创建时应用、晚于 SimulationApp 的场景渲染配置。"""

    if scene_profile.supports("nurec_visual"):
        return {"render_antialiasing_mode": "TAA"}
    return {}


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _optional_project_path(raw_path: str | Path | None) -> Path | None:
    return None if raw_path is None else _project_path(raw_path)


def _parse_xyz_points(
    raw_values: Sequence[str] | None,
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


def _utc_now_iso() -> str:
    return _datetime.datetime.now(_datetime.UTC).isoformat()


def _json_safe(value: Any) -> Any:
    """把启动诊断中的 Path 等对象转成可写入 JSON 的基础类型。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_startup_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _record_startup_phase(
    startup_status: dict[str, Any] | None,
    startup_status_path: Path | None,
    phase: str,
    **metadata: Any,
) -> None:
    """记录 episode 之前的启动进度，便于定位 GUI 打开后立即退出的问题。"""

    if startup_status is None or startup_status_path is None:
        return
    event = {"time": _utc_now_iso(), "phase": phase}
    event.update(metadata)
    startup_status.setdefault("phases", []).append(event)
    startup_status["updated_at"] = event["time"]
    startup_status["last_phase"] = phase
    _write_startup_json(startup_status_path, startup_status)


def _record_startup_failure(
    startup_status: dict[str, Any] | None,
    startup_status_path: Path | None,
    output_dir: Path,
    exc: BaseException,
) -> None:
    """启动阶段异常也要落盘，否则 Isaac 关闭后只剩 Kit warning。"""

    if startup_status is None:
        startup_status = {
            "schema_version": 1,
            "created_at": _utc_now_iso(),
            "status": "failed",
            "output_dir": str(output_dir),
            "phases": [],
        }
    startup_status["status"] = "failed"
    startup_status["updated_at"] = _utc_now_iso()
    startup_status["exception"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    failure_path = output_dir / "startup_failure.json"
    _write_startup_json(failure_path, startup_status)
    if startup_status_path is not None:
        _write_startup_json(startup_status_path, startup_status)
    print(
        "[full-physics] startup failed before/around episode creation; "
        f"details={failure_path}",
        flush=True,
    )


def _locomotion_runtime_kwargs(config: FullPhysicsConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if config.locomotion.locomotion_task:
        kwargs["task_name"] = config.locomotion.locomotion_task
    if config.locomotion.locomotion_checkpoint is not None:
        kwargs["checkpoint"] = config.locomotion.locomotion_checkpoint
    if config.locomotion.policy_profile == "pct_multifloor":
        # DogOnly checkpoint 的 gait 奖励从 0.08 开始；更小命令按站立处理，
        # 避免 DWA 起步爬升阶段触发原地换脚。
        kwargs["standing_command_threshold"] = 0.08
        kwargs["policy_action_warmup_steps"] = 50
    return kwargs


def _navigation_visual_runtime_kwargs(
    policy_profile: str,
    requested_mode: str,
    *,
    recording_visual_required: bool = False,
) -> dict[str, object]:
    """选择物理验收的视觉负载，隔离高质量渲染与导航执行。"""

    mode = requested_mode
    if mode == "auto":
        mode = (
            "full"
            if recording_visual_required
            else ("collision" if policy_profile == "pct_multifloor" else "full")
        )
    return {
        "enable_scene_visual": mode == "full",
        "hide_navigation_collision_visual": mode != "collision",
        "hide_object_collision_visual": mode != "collision",
    }


def _navigation_smoke_viewport_runtime_kwargs(
    *,
    headless: bool,
    stair_locomotion_smoke: bool,
    overview_camera_mode: str,
    overview_camera_prim_path: str,
) -> dict[str, object]:
    """按场景 profile 设置导航 smoke 的初始视口相机。"""

    return {
        "viewport_camera_prim_path": str(overview_camera_prim_path),
        "auto_manage_viewport_camera": bool(
            headless or stair_locomotion_smoke or overview_camera_mode == "fixed"
        ),
    }


def _create_full_physics_runtime(
    *,
    config: FullPhysicsConfig,
    args: argparse.Namespace,
    simulation_app: object,
    scene_isaac_runtime_overrides: dict[str, Any] | None = None,
):
    """Create the one IsaacLab runtime optionally shared by all episodes."""

    from source.simulation import (
        IsaacLabNavigationRuntime,
        IsaacLabNavigationRuntimeConfig,
    )

    video_modes = set(config.video.modes) if config.video.enabled else set()
    runtime_overrides = dict(scene_isaac_runtime_overrides or {})
    return IsaacLabNavigationRuntime(
        simulation_app=simulation_app,
        project_root=PROJECT_ROOT,
        config=IsaacLabNavigationRuntimeConfig(
            **_locomotion_runtime_kwargs(config),
            **_navigation_visual_runtime_kwargs(
                config.locomotion.policy_profile,
                args.navigation_visual_mode,
                recording_visual_required=(
                    config.recording.enabled and bool(config.recording.camera_keys)
                ),
            ),
            **runtime_overrides,
            auto_manage_viewport_camera=bool(
                config.headless or config.video.overview_camera_mode == "fixed"
            ),
            enable_front_camera=(
                (
                    config.recording.enabled
                    and "front" in config.recording.camera_keys
                )
                or bool({"front", "composite"} & video_modes)
            ),
            front_camera_height=config.recording.image_height,
            front_camera_width=config.recording.image_width,
            enable_wrist_camera=(
                (
                    config.recording.enabled
                    and "wrist" in config.recording.camera_keys
                )
                or bool({"wrist", "composite"} & video_modes)
            ),
            wrist_camera_height=config.recording.image_height,
            wrist_camera_width=config.recording.image_width,
            enable_overview_camera=(
                (
                    config.recording.enabled
                    and "overview" in config.recording.camera_keys
                )
                or "composite" in video_modes
            ),
            overview_camera_prim_path=config.recording.overview_camera_prim_path,
            overview_camera_height=config.recording.image_height,
            overview_camera_width=config.recording.image_width,
            camera_render_interval_control_steps=(
                1
                if (config.video.enabled or not config.headless)
                else max(
                    1,
                    int(round(1.0 / (config.recording.dataset_fps * 0.02))),
                )
            ),
            enable_relocatable_episode_supports=bool(
                config.reuse_isaac_stage and config.num_episodes > 1
            ),
            viewport_camera_prim_path=config.recording.overview_camera_prim_path,
            scene_light_mode=config.lighting.scene_light_mode,
            camera_light_intensity=config.lighting.camera_light_intensity,
            camera_light_radius=config.lighting.camera_light_radius,
            place_release_clearance_min_m=(
                config.manipulation.place_release_clearance_min_m
            ),
            place_pre_clearance_min_m=(
                config.manipulation.place_pre_clearance_min_m
            ),
            show_randomization_debug=config.randomization.show_debug_region,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行单进程、单 World 的纯物理 nav-pick-place pipeline。",
    )
    parser.add_argument(
        "--scene-profile",
        default=DEFAULT_SCENE_PROFILE,
        help="场景 profile 名称或别名；默认 liangzhu，可用 --list-scene-profiles 查看。",
    )
    parser.add_argument(
        "--list-scene-profiles",
        action="store_true",
        help="列出当前仓库可用场景并退出。",
    )
    parser.add_argument(
        "--check-scene-assets",
        action="store_true",
        help="检查所选场景的运行资产并退出。",
    )
    parser.add_argument(
        "--pct-multifloor",
        action="store_const",
        const="multi_floor",
        dest="scene_profile",
        help="兼容旧命令：等价于 --scene-profile multi_floor。",
    )
    parser.add_argument(
        "--task-json",
        default=None,
        help="任务 JSON 路径；默认由所选 scene profile 提供。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "episode、事件、帧和 summary 的输出目录；stair-locomotion-smoke "
            "默认写入 outputs/stair_locomotion_smoke。"
        ),
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="运行的 episode 数量。")
    parser.add_argument(
        "--reuse-isaac-stage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "多 episode 时复用同一 Isaac 进程和已构建 stage，只重置 episode 位姿；"
            "默认开启。"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="首个 episode 的随机种子；相同 seed 会严格复现同一任务布局。",
    )
    parser.add_argument(
        "--randomize-task",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="按 episode seed 随机采样任务布局；默认由 scene profile 决定。",
    )
    parser.add_argument(
        "--show-randomization-debug",
        action="store_true",
        help="显示 pick/place 随机区域和采样点的非物理 USD guide；默认关闭。",
    )
    parser.add_argument(
        "--show-planned-trajectories",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "显示 PCT 路径和 cuRobo TCP 轨迹；stair-locomotion-smoke 和 "
            "PCT preview 默认开启，完整 pipeline 默认关闭。"
        ),
    )
    parser.add_argument(
        "--randomize-base-goal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="开启 pick/place 导航交接 base_goal 随机化；默认由 scene profile 决定。",
    )
    parser.add_argument(
        "--keep-window-open",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "pipeline 结束后保持 GUI 窗口，便于检查场景和调试标记；"
            "stair-locomotion-smoke 的 GUI 默认开启。"
        ),
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否以无界面模式运行；使用 --headless 关闭GUI渲染。",
    )
    parser.add_argument(
        "--navigation-visual-mode",
        choices=("auto", "collision", "full"),
        default=None,
        help=(
            "物理验收视觉模式；full 加载 GaussianScene，collision 仅显示碰撞场景；"
            "默认由 scene profile 决定。"
        ),
    )
    parser.add_argument(
        "--scene-light-mode",
        choices=("auto", "camera", "stage"),
        default=SceneLightingSettings().scene_light_mode,
        help=(
            "真实 Isaac stage 灯光模式；默认 auto：full 视觉使用 USD 原场景灯光，"
            "collision 视觉使用相机补光；camera/stage 可显式覆盖。"
        ),
    )
    parser.add_argument(
        "--camera-light-intensity",
        type=float,
        default=SceneLightingSettings().camera_light_intensity,
        help="camera light 的 SphereLight 强度。",
    )
    parser.add_argument(
        "--camera-light-radius",
        type=float,
        default=SceneLightingSettings().camera_light_radius,
        help="camera light 的 SphereLight 半径。",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "启用展示视频；完整 pipeline 默认录制 overview/front/wrist "
            "三视角拼接视频，可用 --no-record-video 关闭。"
        ),
    )
    parser.add_argument(
        "--record-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存 LeRobot dataset 图像和数据；物理验收可用 --no-record-dataset 节省空间。",
    )
    parser.add_argument(
        "--dataset-camera-keys",
        nargs="+",
        choices=("front", "wrist", "overview"),
        default=list(RecordingSettings().camera_keys),
        help=(
            "训练数据相机流，默认 front wrist overview；至少包含 front。"
            "可用于隔离特定渲染后端问题。"
        ),
    )
    parser.add_argument(
        "--video-mode",
        choices=("overview", "front", "font", "wrist", "composite", "all"),
        default=None,
        help=(
            "录制视频类型：overview 为第三人称展示视角，front/font 为前视 observation "
            "camera，wrist 为腕部 observation camera，composite 将同一仿真 step 的 "
            "overview/front/wrist 拼成单个视频，all 同时导出三路独立视频。"
        ),
    )
    parser.add_argument(
        "--video-out",
        help="overview 视频输出目录或 .mp4 文件路径；默认写到 episode_dir/overview_videos。",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1280,
        help="overview 捕获宽度，默认 1280；front/wrist observation 不受影响。",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=720,
        help="overview 捕获高度，默认 720；front/wrist observation 不受影响。",
    )
    parser.add_argument(
        "--overview-camera-mode",
        choices=("fixed", "auto"),
        default=None,
        help="overview camera 选择模式；默认由 scene profile 决定。",
    )
    parser.add_argument(
        "--overview-camera-prim-path",
        default=None,
        help="image/video/GUI 共用的 overview Camera prim；默认由 scene profile 决定。",
    )
    parser.add_argument(
        "--overview-camera-schedule",
        default=None,
        help=(
            "headless overview Camera0-8 切换规则 JSON；可按 pipeline state 和机器人 XYZ 调整。"
        ),
    )
    parser.add_argument(
        "--overview-capture-backend",
        choices=("viewport", "render_product", "auto"),
        default="viewport",
        help=(
            "overview 取帧后端；viewport 抓取最终视口画面，最接近 GUI；"
            "render_product 使用 Replicator RGB；auto 先尝试 viewport 再回退。"
        ),
    )
    parser.add_argument(
        "--overview-initial-hold-frames",
        type=int,
        default=160,
        help="overview 初始全局镜头最少保持帧数，默认 160，避免 reset 后立即切走 third_person1。",
    )
    parser.add_argument(
        "--overview-exposure",
        type=float,
        default=0.0,
        help="overview 线性 RGB 转视频前的曝光补偿，单位 EV stops，默认 0。",
    )
    parser.add_argument(
        "--overview-gamma",
        type=float,
        default=2.2,
        help="overview 线性 RGB 转 sRGB 的 gamma，默认 2.2；设为 1.0 可关闭 gamma 提亮。",
    )
    parser.add_argument(
        "--export-video-camera-trajectory",
        action="store_true",
        help="录制 overview 视频时同步导出相机轨迹 JSONL，供离线 3DGS 背景渲染使用。",
    )
    parser.add_argument(
        "--video-camera-trajectory-out",
        help="overview 相机轨迹 JSONL 输出路径；默认跟随 overview mp4 文件名。",
    )
    parser.add_argument(
        "--pick-plan-json",
        help="预先生成的 pick cuRobo 分段计划 JSON；仅用于 manipulation apply smoke。",
    )
    parser.add_argument(
        "--place-plan-json",
        help="预先生成的 place cuRobo 分段计划 JSON；仅用于 manipulation apply smoke。",
    )
    parser.add_argument(
        "--global-planner",
        choices=("astar", "pct"),
        default=None,
        help="全局导航规划器；默认由 scene profile 决定。",
    )
    parser.add_argument("--pct-planner-root", help="兼容旧外部入口的 PCT 根目录；默认不依赖 external/PCT。")
    parser.add_argument(
        "--pct-server-script",
        default=None,
        help="PCT server 脚本路径；默认由 scene profile 决定。",
    )
    parser.add_argument("--pct-server-python", help="运行 PCT server 的 Python 解释器。")
    parser.add_argument(
        "--pct-tomogram-path",
        default=None,
        help="PCT tomogram pickle 路径；默认由 scene profile 决定。",
    )
    parser.add_argument(
        "--pct-walkable-path",
        default=None,
        help="PCT walkable map；默认由 scene profile 决定。",
    )
    parser.add_argument(
        "--pct-collision-ply-path",
        default=None,
        help="PCT DWA collision PLY；默认由 scene profile 决定。",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--pct-no-fallback",
        action="store_true",
        dest="pct_no_fallback",
        default=None,
        help="PCT 规划失败时不回退 A*。",
    )
    fallback_group.add_argument(
        "--pct-allow-fallback",
        "--pct-fallback-to-astar",
        action="store_false",
        dest="pct_no_fallback",
        help="允许 PCT 失败时回退 A*。",
    )
    parser.add_argument(
        "--pct-coord-mode",
        choices=("sim_to_pct_180deg", "identity"),
        default=None,
        help="Isaac 世界坐标到 PCT/PLY 的变换模式；默认由 scene profile 决定。",
    )
    parser.add_argument("--pct-offset-x", type=float, default=0.0, help="PCT 坐标 X 偏移。")
    parser.add_argument("--pct-offset-y", type=float, default=0.0, help="PCT 坐标 Y 偏移。")
    parser.add_argument("--pct-scale-x", type=float, default=1.0, help="PCT 坐标 X 缩放。")
    parser.add_argument("--pct-scale-y", type=float, default=1.0, help="PCT 坐标 Y 缩放。")
    parser.add_argument(
        "--pct-vertical-obstacle-min-slices",
        type=int,
        default=0,
        help="实验性 collision PLY 垂直障碍层阈值；0 表示关闭，避免楼梯和稀疏结构形成假障碍。",
    )
    parser.add_argument(
        "--pct-vertical-obstacle-dilation-radius-cells",
        type=int,
        default=0,
        help="DWA 垂直障碍层的栅格膨胀半径。",
    )
    parser.add_argument(
        "--pct-global-vertical-obstacle-min-slices",
        type=int,
        default=NavigationSettings().pct_global_vertical_obstacle_min_slices,
        help="PCT 全局路径判定硬墙所需的最小 PLY 高度切片数。",
    )
    parser.add_argument(
        "--pct-cross-floor-vertical-obstacle-min-slices",
        type=int,
        default=NavigationSettings().pct_cross_floor_vertical_obstacle_min_slices,
        help="PCT 跨楼层路径判定硬墙所需的最小 PLY 高度切片数。",
    )
    parser.add_argument(
        "--pct-cross-floor-gateway",
        action="append",
        default=None,
        help=(
            "允许跨楼层换 slice 的楼梯/坡道中心点，Isaac Sim 坐标 x,y,z；"
            "可重复传入，传 none 可关闭该约束。"
        ),
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
        "--pct-robot-root-to-floor",
        type=float,
        default=NavigationSettings().pct_robot_root_to_floor_m,
        help="机器人 root z 到 PCT 地面 slice 的高度偏移。",
    )
    parser.add_argument(
        "--pct-body-obstacle-min-height",
        type=float,
        default=NavigationSettings().pct_body_obstacle_min_height_m,
        help="相对地面开始计入身体碰撞的最小高度。",
    )
    parser.add_argument(
        "--pct-body-obstacle-max-height",
        type=float,
        default=NavigationSettings().pct_body_obstacle_max_height_m,
        help="相对地面计入身体碰撞的最大高度。",
    )
    parser.add_argument(
        "--pct-stair-min-horizontal-per-slice",
        type=float,
        default=NavigationSettings().pct_stair_min_horizontal_per_slice_m,
        help="楼梯每升高一个 slice 所需的最小水平行程。",
    )
    parser.add_argument(
        "--pct-stair-max-horizontal-per-slice",
        type=float,
        default=NavigationSettings().pct_stair_max_horizontal_per_slice_m,
        help="楼梯每升高一个 slice 允许的最大水平行程。",
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
        help="楼梯高度 slice 与入口到出口进度匹配的容差，0 到 1。",
    )
    parser.add_argument(
        "--pct-stair-progress-cost-weight",
        type=float,
        default=NavigationSettings().pct_stair_progress_cost_weight,
        help="楼梯高度 slice 与折线进度不匹配时的软代价权重。",
    )
    parser.add_argument(
        "--pct-obstacle-clearance-radius",
        type=float,
        default=NavigationSettings().pct_obstacle_clearance_radius_m,
        help="PCT 全局路径对墙体和家具施加软代价的净空半径。",
    )
    parser.add_argument(
        "--pct-obstacle-clearance-cost",
        type=float,
        default=NavigationSettings().pct_obstacle_clearance_cost_weight,
        help="PCT 障碍净空软代价权重。",
    )
    parser.add_argument(
        "--pct-multifloor-vertical-obstacle-min-slices",
        type=int,
        default=5,
        help="跨楼层 carry 地图中判定墙体所需的最小 PLY 高度切片数。",
    )
    parser.add_argument(
        "--pct-multifloor-obstacle-inflate-radius",
        type=float,
        default=0.12,
        help="跨楼层 carry 地图障碍膨胀半径。",
    )
    parser.add_argument(
        "--pct-multifloor-route-corridor-radius",
        type=float,
        default=NavigationSettings().pct_multifloor_route_corridor_radius,
        help="沿 PCT 三维路径投影保留的可行走走廊半径。",
    )
    parser.add_argument(
        "--pct-carry-max-linear-velocity",
        type=float,
        default=NavigationSettings().pct_carry_max_linear_velocity,
        help="携物跨楼层导航最大前进速度。",
    )
    parser.add_argument(
        "--pct-carry-max-angular-velocity",
        type=float,
        default=NavigationSettings().pct_carry_max_angular_velocity,
        help="携物跨楼层导航最大角速度。",
    )
    parser.add_argument(
        "--pct-stair-float",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "携物上楼阶段冻结底盘并沿 PCT 3D 路径漂移；"
            "PCT 稳定完整 pipeline 默认开启，可用 --no-pct-stair-float 关闭。"
        ),
    )
    parser.add_argument(
        "--pct-stair-float-speed",
        type=float,
        default=NavigationSettings().pct_stair_float_speed_mps,
        help="PCT 楼梯漂移速度，单位 m/s。",
    )
    parser.add_argument(
        "--pct-stair-float-activation-radius",
        type=float,
        default=NavigationSettings().pct_stair_float_activation_radius_m,
        help="机器人接近楼梯段首点多少米内启动漂移。",
    )
    parser.add_argument(
        "--pct-stair-float-completion-radius",
        type=float,
        default=NavigationSettings().pct_stair_float_completion_radius_m,
        help="距离楼梯漂移段末端多少米内视为完成并恢复普通导航。",
    )
    parser.add_argument(
        "--pct-stair-float-approach-distance",
        type=float,
        default=NavigationSettings().pct_stair_float_approach_distance_m,
        help="PCT 楼梯漂移向入口前方扩展的距离，用于覆盖窄通道。",
    )
    parser.add_argument(
        "--pct-stair-float-exit-distance",
        type=float,
        default=NavigationSettings().pct_stair_float_exit_distance_m,
        help="PCT 楼梯漂移向二楼出口方向继续扩展的距离。",
    )
    parser.add_argument(
        "--pct-stair-float-settle-time",
        type=float,
        default=NavigationSettings().pct_stair_float_settle_time_s,
        help="PCT 楼梯漂移完成后保持 root 与全身关节锁的稳定时间。",
    )
    parser.add_argument(
        "--pct-stair-float-release-settle-time",
        type=float,
        default=NavigationSettings().pct_stair_float_release_settle_time_s,
        help="解除 PCT 楼梯漂移 root 锁后保持零速站稳的时间。",
    )
    parser.add_argument(
        "--pct-stair-float-min-root-z-offset",
        type=float,
        default=NavigationSettings().pct_stair_float_min_root_z_offset_m,
        help="PCT 楼梯漂移期间 root 高度相对 PCT 地面路径的最小偏移。",
    )
    parser.add_argument(
        "--pct-stair-float-release-root-z-offset",
        type=float,
        default=(
            NavigationSettings().pct_stair_float_release_root_z_offset_m
        ),
        help="PCT 楼梯漂移结束时 root 相对二楼地面的释放高度。",
    )
    parser.add_argument(
        "--goal-z-tolerance",
        type=float,
        default=0.35,
        help="多楼层导航目标 z 到达容差；旧 XY-only 任务不触发 z 检查。",
    )
    parser.add_argument(
        "--locomotion-task",
        help="Isaac Lab locomotion task 名称；pct_multifloor 默认使用本地 DogOnly rough task。",
    )
    parser.add_argument(
        "--locomotion-checkpoint",
        default=None,
        help="RSL-RL locomotion checkpoint；默认由 scene profile 决定。",
    )
    parser.add_argument(
        "--policy-profile",
        choices=("flat", "pct_multifloor"),
        default=None,
        help="底层 locomotion policy profile；默认由 scene profile 决定。",
    )
    parser.add_argument(
        "--require-locomotion-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="要求 locomotion checkpoint 存在；默认开启。",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_const",
        const="dry_run",
        dest="mode",
        help="使用无 Isaac 依赖的内存后端验证完整状态流。",
    )
    mode_group.add_argument(
        "--simulation-smoke",
        action="store_const",
        const="simulation_smoke",
        dest="mode",
        help="启动真实 Isaac Sim，仅验证单 Stage、单 World 的场景构建和 episode reset。",
    )
    mode_group.add_argument(
        "--navigation-smoke",
        action="store_const",
        const="navigation_smoke",
        dest="mode",
        help="启动真实 Isaac Lab locomotion policy，验证 pipeline 驱动的物理导航。",
    )
    mode_group.add_argument(
        "--navigation-carry-smoke",
        action="store_const",
        const="navigation_carry_smoke",
        dest="mode",
        help="从 pick 导航终点出发，验证导航到 place 时机械臂保持全零 home 且夹爪闭合。",
    )
    mode_group.add_argument(
        "--stair-locomotion-smoke",
        action="store_const",
        const="stair_locomotion_smoke",
        dest="mode",
        help=(
            "从楼梯入口重置并沿 PCT 在线规划的 path_3d 纯物理上楼；"
            "禁用 Float 和 DWA，默认开启 GUI、保留窗口并录制 overview 视频和数据集，"
            "越过楼梯出口并进入 F2 平台后结束。"
        ),
    )
    mode_group.add_argument(
        "--pct-plan-preview",
        action="store_const",
        const="pct_plan_preview",
        dest="mode",
        help="只启动 GUI、规划并绘制 PCT 多楼层路线，不执行 DWA/RL/机械臂。",
    )
    mode_group.add_argument(
        "--pick-smoke",
        action="store_const",
        const="pick_smoke",
        dest="mode",
        help="真实执行 nav_to_pick 和 pick，pick 成功后停止，不进入跨楼层 place。",
    )
    mode_group.add_argument(
        "--manipulation-smoke",
        action="store_const",
        const="manipulation_smoke",
        dest="mode",
        help="跳过导航，使用分段机械臂计划验证 manipulation action 合同。",
    )
    mode_group.add_argument(
        "--manipulation-apply-smoke",
        action="store_const",
        const="manipulation_apply_smoke",
        dest="mode",
        help="启动真实 Isaac Sim，跳过导航并验证机械臂/夹爪 action 能下发到 articulation。",
    )
    parser.set_defaults(mode="full_physics")
    return parser


def _resolve_runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """按 scene profile 补齐稳定默认值，同时保留显式 CLI 覆盖。"""

    if args.list_scene_profiles:
        try:
            profiles = list_scene_profiles(PROJECT_ROOT)
        except SceneProfileError as exc:
            raise SystemExit(str(exc)) from exc
        for profile in profiles:
            aliases = ", ".join(profile.aliases) if profile.aliases else "-"
            print(
                f"{profile.name:12s} aliases=[{aliases}] {profile.description}",
                flush=True,
            )
        raise SystemExit(0)

    try:
        profile = load_scene_profile(args.scene_profile, PROJECT_ROOT)
        applied_defaults = apply_scene_profile_defaults(
            args,
            profile,
            mode=str(args.mode),
        )
    except SceneProfileError as exc:
        raise SystemExit(str(exc)) from exc

    args.scene_profile = profile.name
    args.scene_profile_config_path = str(profile.config_path)
    args.scene_profile_task_name = profile.task_scene_profile
    args.scene_profile_aliases = profile.all_names
    args.scene_runtime_asset_manifest = profile.runtime_asset_manifest
    args.scene_profile_defaults_applied = applied_defaults

    stair_locomotion_smoke = str(args.mode) == "stair_locomotion_smoke"
    generic_defaults = {
        "output_dir": f"outputs/{profile.name}",
        "global_planner": "pct",
        "pct_server_script": "scripts/navigation/pct_grid_server.py",
        "pct_no_fallback": True,
        "pct_coord_mode": "identity",
        "policy_profile": "pct_multifloor",
        "locomotion_task": PCT_MULTIFLOOR_LOCOMOTION_TASK,
        "locomotion_checkpoint": (
            "checkpoints/go2_x5/pct_multifloor/model_26000.pt"
        ),
        "randomize_task": False,
        "randomize_base_goal": False,
        "show_planned_trajectories": bool(
            stair_locomotion_smoke or str(args.mode) == "pct_plan_preview"
        ),
        "pct_stair_float": False,
        "navigation_visual_mode": "full",
        "overview_camera_mode": "fixed",
        "overview_camera_prim_path": DEFAULT_OVERVIEW_CAMERA_PRIM_PATH,
        "video_mode": "composite",
    }
    for key, value in generic_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.task_json is None:
        raise SystemExit(
            f"场景 profile {profile.name!r} 未提供 task_json，请显式传 --task-json。"
        )
    if stair_locomotion_smoke and not profile.supports(
        "stair_locomotion_smoke"
    ):
        raise SystemExit(
            f"场景 profile {profile.name!r} 未声明 stair_locomotion_smoke 能力。"
        )
    if stair_locomotion_smoke and str(args.global_planner) != "pct":
        raise SystemExit("--stair-locomotion-smoke 只支持 PCT 全局规划器。")
    if args.keep_window_open is None:
        args.keep_window_open = bool(stair_locomotion_smoke and not args.headless)
    if args.record_video is None:
        args.record_video = bool(
            str(args.mode) == "full_physics" or stair_locomotion_smoke
        )
    args.runtime_preset = f"scene_profile:{profile.name}"

    if args.check_scene_assets:
        report = check_scene_profile_assets(profile, PROJECT_ROOT)
        print(
            f"scene_profile={profile.name} available={len(report.available)} "
            f"missing={len(report.missing)}",
            flush=True,
        )
        for path in report.missing:
            print(f"MISSING {path}", flush=True)
        if report.success:
            print("PASS", flush=True)
        raise SystemExit(0 if report.success else 2)
    return args


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _resolve_runtime_defaults(_build_parser().parse_args(argv))


def _validate_task_scene_profile(
    args: argparse.Namespace,
    raw_task: dict[str, Any],
) -> None:
    """防止显式 task 覆盖后静默加载到错误场景。"""

    actual = raw_task.get("scene_profile")
    if actual is None:
        return

    def _normalized(value: object) -> str:
        return str(value).strip().lower().replace("-", "_")

    accepted = {
        _normalized(args.scene_profile_task_name),
        *(_normalized(value) for value in args.scene_profile_aliases),
    }
    if _normalized(actual) not in accepted:
        raise SystemExit(
            "任务与场景 profile 不匹配："
            f"task.scene_profile={actual!r}，CLI scene_profile={args.scene_profile!r}。"
        )


def _validate_external_plan_paths(config: FullPhysicsConfig) -> None:
    if (config.full_physics or config.pick_smoke) and (
        config.pick_plan_json is not None or config.place_plan_json is not None
    ):
        raise SystemExit("full-physics / pick-smoke 模式按当前仿真状态在线规划，不接受 --pick-plan-json/--place-plan-json。")
    missing_paths = [
        (label, path)
        for label, path in (
            ("pick", config.pick_plan_json),
            ("place", config.place_plan_json),
        )
        if path is not None and not path.is_file()
    ]
    if not missing_paths:
        return
    details = "\n".join(f"  {label}: {path}" for label, path in missing_paths)
    raise SystemExit(f"外部 cuRobo plan JSON 不存在，请检查路径：\n{details}")


def _keep_gui_open(simulation_app: object, retained_simulation: object | None = None) -> None:
    """物理 runtime 已暂停后只维持 Kit GUI，不再调用 pipeline world.step。"""

    if retained_simulation is not None and hasattr(retained_simulation, "refresh_viewport"):
        retained_simulation.refresh_viewport(reason="keep_gui_open_before_loop")
    print(
        "[full-physics] pipeline 已结束，GUI 保持打开；关闭窗口或按 Ctrl+C 退出。",
        flush=True,
    )
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("[full-physics] 收到 Ctrl+C，正在关闭 GUI。", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = str(args.mode)
    dry_run = mode == "dry_run"
    simulation_smoke = mode == "simulation_smoke"
    stair_locomotion_smoke = mode == "stair_locomotion_smoke"
    navigation_smoke = mode == "navigation_smoke"
    navigation_carry_smoke = mode == "navigation_carry_smoke"
    pct_plan_preview = mode == "pct_plan_preview"
    pick_smoke = mode == "pick_smoke"
    manipulation_smoke = mode == "manipulation_smoke"
    manipulation_apply_smoke = mode == "manipulation_apply_smoke"
    full_physics = mode == "full_physics"
    flat_episode_output = os.environ.get("FULL_PHYSICS_FLAT_EPISODE_OUTPUT") == "1"
    if (full_physics or pick_smoke) and (args.pick_plan_json or args.place_plan_json):
        raise SystemExit("full-physics / pick-smoke 模式禁止使用离线 plan JSON；pick/place 必须按当前仿真状态在线规划。")
    if args.keep_window_open and args.headless:
        raise SystemExit("--keep-window-open 只能与 --no-headless 一起使用。")
    if stair_locomotion_smoke and args.pct_stair_float:
        raise SystemExit("--stair-locomotion-smoke 固定禁用 Float，请不要传 --pct-stair-float。")
    if args.record_video and dry_run:
        raise SystemExit("--record-video 需要真实 Isaac stage / camera images，不能与 --dry-run 一起使用。")
    if args.export_video_camera_trajectory and not args.record_video:
        raise SystemExit("--export-video-camera-trajectory 需要同时启用 --record-video。")
    if args.export_video_camera_trajectory and str(args.video_mode).lower() not in {"overview", "all"}:
        raise SystemExit("--export-video-camera-trajectory 只支持 --video-mode overview 或 all。")
    if args.record_video and args.video_out and Path(args.video_out).suffix.lower() == ".mp4" and args.num_episodes != 1:
        raise SystemExit("--video-out 指向单个 .mp4 文件时只支持 --num-episodes 1；多 episode 请传输出目录。")
    multi_episode_real_supported = bool(
        full_physics and args.reuse_isaac_stage and args.headless
    )
    if (
        simulation_smoke
        or navigation_smoke
        or navigation_carry_smoke
        or stair_locomotion_smoke
        or pct_plan_preview
        or pick_smoke
        or manipulation_apply_smoke
        or full_physics
    ) and args.num_episodes != 1 and not multi_episode_real_supported:
        raise SystemExit(
            "真实 Isaac 多 episode 仅支持 headless full_physics，并要求 "
            "--reuse-isaac-stage。"
        )
    if flat_episode_output and args.num_episodes != 1:
        raise SystemExit("batch 扁平输出模式只支持单 episode 子进程。")

    locomotion_checkpoint = _optional_project_path(args.locomotion_checkpoint)
    locomotion_task = args.locomotion_task
    if args.policy_profile == "pct_multifloor" and not locomotion_task:
        locomotion_task = PCT_MULTIFLOOR_LOCOMOTION_TASK
    if args.policy_profile == "pct_multifloor" and not pct_plan_preview and (
        locomotion_checkpoint is None or not locomotion_checkpoint.is_file()
    ):
        raise SystemExit(
            "PCT multi-floor policy checkpoint missing. "
            "Please train or pass --locomotion-checkpoint."
        )
    pct_cross_floor_gateway_points = _parse_xyz_points(
        args.pct_cross_floor_gateway,
        default=(),
    )
    pct_cross_floor_stair_exit_points = _parse_xyz_points(
        args.pct_cross_floor_stair_exit,
        default=(),
    )
    pct_cross_floor_stair_midpoint_points = _parse_xyz_points(
        args.pct_cross_floor_stair_midpoint,
        default=(),
    )

    config = FullPhysicsConfig(
        task_json=_project_path(args.task_json),
        output_dir=_project_path(args.output_dir),
        num_episodes=args.num_episodes,
        seed=args.seed,
        reuse_isaac_stage=bool(args.reuse_isaac_stage),
        headless=args.headless,
        keep_window_open=args.keep_window_open,
        show_planned_trajectories=bool(args.show_planned_trajectories),
        pick_plan_json=_project_path(args.pick_plan_json) if args.pick_plan_json else None,
        place_plan_json=_project_path(args.place_plan_json) if args.place_plan_json else None,
        dry_run=dry_run,
        simulation_smoke=simulation_smoke,
        navigation_smoke=navigation_smoke,
        navigation_carry_smoke=navigation_carry_smoke,
        stair_locomotion_smoke=stair_locomotion_smoke,
        pct_plan_preview=pct_plan_preview,
        pick_smoke=pick_smoke,
        manipulation_smoke=manipulation_smoke,
        manipulation_apply_smoke=manipulation_apply_smoke,
        full_physics=full_physics,
        navigation=NavigationSettings(
            global_planner=str(args.global_planner),
            pct_enabled=str(args.global_planner) == "pct",
            pct_planner_root=_optional_project_path(args.pct_planner_root),
            pct_server_script=_optional_project_path(args.pct_server_script),
            pct_server_python=_optional_project_path(args.pct_server_python),
            pct_tomogram_path=_optional_project_path(args.pct_tomogram_path),
            pct_walkable_path=_optional_project_path(args.pct_walkable_path),
            pct_collision_ply_path=_optional_project_path(args.pct_collision_ply_path),
            pct_fallback_to_astar=not bool(args.pct_no_fallback),
            pct_coord_mode=str(args.pct_coord_mode),
            pct_offset_x=float(args.pct_offset_x),
            pct_offset_y=float(args.pct_offset_y),
            pct_scale_x=float(args.pct_scale_x),
            pct_scale_y=float(args.pct_scale_y),
            pct_vertical_obstacle_min_slices=int(args.pct_vertical_obstacle_min_slices),
            pct_vertical_obstacle_dilation_radius_cells=int(
                args.pct_vertical_obstacle_dilation_radius_cells
            ),
            pct_global_vertical_obstacle_min_slices=int(
                args.pct_global_vertical_obstacle_min_slices
            ),
            pct_cross_floor_vertical_obstacle_min_slices=int(
                args.pct_cross_floor_vertical_obstacle_min_slices
            ),
            pct_cross_floor_gateway_points=pct_cross_floor_gateway_points,
            pct_cross_floor_stair_exit_points=(
                pct_cross_floor_stair_exit_points
            ),
            pct_cross_floor_stair_midpoint_points=(
                pct_cross_floor_stair_midpoint_points
            ),
            pct_cross_floor_gateway_radius_m=float(
                args.pct_cross_floor_gateway_radius
            ),
            pct_robot_root_to_floor_m=float(args.pct_robot_root_to_floor),
            pct_body_obstacle_min_height_m=float(
                args.pct_body_obstacle_min_height
            ),
            pct_body_obstacle_max_height_m=float(
                args.pct_body_obstacle_max_height
            ),
            pct_stair_min_horizontal_per_slice_m=float(
                args.pct_stair_min_horizontal_per_slice
            ),
            pct_stair_max_horizontal_per_slice_m=float(
                args.pct_stair_max_horizontal_per_slice
            ),
            pct_stair_vertical_radius_m=float(args.pct_stair_vertical_radius),
            pct_stair_progress_tolerance=float(
                args.pct_stair_progress_tolerance
            ),
            pct_stair_progress_cost_weight=float(
                args.pct_stair_progress_cost_weight
            ),
            pct_obstacle_clearance_radius_m=float(
                args.pct_obstacle_clearance_radius
            ),
            pct_obstacle_clearance_cost_weight=float(
                args.pct_obstacle_clearance_cost
            ),
            pct_multifloor_vertical_obstacle_min_slices=int(
                args.pct_multifloor_vertical_obstacle_min_slices
            ),
            pct_multifloor_obstacle_inflate_radius=float(
                args.pct_multifloor_obstacle_inflate_radius
            ),
            pct_multifloor_route_corridor_radius=float(
                args.pct_multifloor_route_corridor_radius
            ),
            pct_carry_max_linear_velocity=float(
                args.pct_carry_max_linear_velocity
            ),
            pct_carry_max_angular_velocity=float(
                args.pct_carry_max_angular_velocity
            ),
            pct_stair_float_enabled=bool(args.pct_stair_float),
            pct_stair_float_speed_mps=float(args.pct_stair_float_speed),
            pct_stair_float_activation_radius_m=float(
                args.pct_stair_float_activation_radius
            ),
            pct_stair_float_completion_radius_m=float(
                args.pct_stair_float_completion_radius
            ),
            pct_stair_float_approach_distance_m=float(
                args.pct_stair_float_approach_distance
            ),
            pct_stair_float_exit_distance_m=float(
                args.pct_stair_float_exit_distance
            ),
            pct_stair_float_settle_time_s=float(
                args.pct_stair_float_settle_time
            ),
            pct_stair_float_release_settle_time_s=float(
                args.pct_stair_float_release_settle_time
            ),
            pct_stair_float_min_root_z_offset_m=float(
                args.pct_stair_float_min_root_z_offset
            ),
            pct_stair_float_release_root_z_offset_m=float(
                args.pct_stair_float_release_root_z_offset
            ),
            goal_z_tolerance=float(args.goal_z_tolerance),
        ),
        locomotion=LocomotionPolicySettings(
            locomotion_task=locomotion_task,
            locomotion_checkpoint=locomotion_checkpoint,
            locomotion_checkpoint_required=bool(args.require_locomotion_checkpoint),
            policy_profile=str(args.policy_profile),
        ),
        manipulation=ManipulationSettings(
            settle_object_before_navigation=bool(
                str(args.policy_profile) == "pct_multifloor"
                and (pick_smoke or full_physics)
            ),
            settle_base_before_navigation=bool(
                str(args.policy_profile) == "pct_multifloor"
                and (pick_smoke or full_physics)
            ),
            initialization_base_lock_steps=(
                20
                if (
                    str(args.policy_profile) == "pct_multifloor"
                    and (pick_smoke or full_physics)
                )
                else 0
            ),
        ),
        randomization=RandomizationSettings(
            enabled=args.randomize_task,
            show_debug_region=args.show_randomization_debug,
            collision_ply_path=_optional_project_path(
                args.pct_collision_ply_path
            ),
            base_goal=BaseGoalRandomizationSettings(
                enabled=args.randomize_base_goal,
            ),
        ),
        recording=RecordingSettings(
            enabled=bool(args.record_dataset),
            camera_keys=tuple(args.dataset_camera_keys),
            debug_per_episode_lerobot=(
                os.environ.get("FULL_PHYSICS_DEFER_LEROBOT_EXPORT") != "1"
            ),
            overview_camera_prim_path=str(args.overview_camera_prim_path),
        ),
        lighting=SceneLightingSettings(
            scene_light_mode=str(args.scene_light_mode),
            camera_light_intensity=float(args.camera_light_intensity),
            camera_light_radius=float(args.camera_light_radius),
        ),
        video=VideoRecordingSettings(
            enabled=bool(args.record_video),
            mode=str(args.video_mode),
            output_path=_project_path(args.video_out) if args.video_out else None,
            overview_camera_mode=str(args.overview_camera_mode),
            overview_camera_prim_path=str(args.overview_camera_prim_path),
            overview_camera_schedule_path=(
                _project_path(args.overview_camera_schedule)
                if args.overview_camera_schedule
                else None
            ),
            width=int(args.video_width),
            height=int(args.video_height),
            overview_capture_backend=str(args.overview_capture_backend),
            overview_initial_hold_frames=int(args.overview_initial_hold_frames),
            overview_exposure=float(args.overview_exposure),
            overview_gamma=float(args.overview_gamma),
            export_camera_trajectory=bool(args.export_video_camera_trajectory),
            camera_trajectory_path=(
                _project_path(args.video_camera_trajectory_out)
                if args.video_camera_trajectory_out
                else None
            ),
        ),
    )
    _validate_external_plan_paths(config)
    base_spec = JsonTaskProvider().load(config.task_json)
    _validate_task_scene_profile(args, base_spec.raw_task)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scene_profile = load_scene_profile(args.scene_profile, PROJECT_ROOT)
    scene_isaac_kit_args = _scene_isaac_kit_args(scene_profile)
    scene_isaac_app_overrides = _scene_isaac_app_overrides(scene_profile)
    scene_isaac_runtime_overrides = _scene_isaac_runtime_overrides(scene_profile)
    binding_profile = scene_profile
    if base_spec.raw_task.get("scene_profile") is None:
        # 兼容没有 scene_profile 字段的旧任务：它们不应继承默认场景的大资产绑定。
        binding_profile = replace(scene_profile, usd_asset_bindings=())
    scene_binding_report = materialize_scene_asset_bindings(
        binding_profile,
        base_spec.scene_usd,
        config.output_dir / ".runtime" / f"{scene_profile.name}_scene_bound.usda",
        project_root=PROJECT_ROOT,
    )
    runtime_scene_usd = str(scene_binding_report["runtime_scene_usd"])
    bound_raw_task = {
        **base_spec.raw_task,
        "scene_usd": runtime_scene_usd,
        "scene_asset_binding_runtime": scene_binding_report,
    }
    base_spec = replace(
        base_spec,
        scene_usd=runtime_scene_usd,
        raw_task=bound_raw_task,
    )
    scene_binding_report_path = write_scene_binding_report(
        scene_binding_report,
        config.output_dir / ".runtime" / "scene_asset_bindings.json",
    )
    batch_summary_path = None
    if not flat_episode_output:
        batch_summary_name = os.environ.get(
            "FULL_PHYSICS_BATCH_SUMMARY_NAME",
            "batch_summary.jsonl",
        )
        if Path(batch_summary_name).name != batch_summary_name:
            raise SystemExit("FULL_PHYSICS_BATCH_SUMMARY_NAME must be a file name")
        batch_summary_path = config.output_dir / batch_summary_name
        batch_summary_path.write_text("", encoding="utf-8")
    stale_startup_failure_path = config.output_dir / "startup_failure.json"
    if stale_startup_failure_path.exists():
        stale_startup_failure_path.unlink()

    startup_status_path = config.output_dir / "startup_status.json"
    startup_status: dict[str, Any] | None = {
        "schema_version": 1,
        "status": "starting",
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "mode": mode,
        "runtime_preset": str(args.runtime_preset),
        "scene_profile": str(args.scene_profile),
        "scene_profile_config_path": str(args.scene_profile_config_path),
        "scene_runtime_asset_manifest": str(args.scene_runtime_asset_manifest),
        "scene_asset_binding_report": scene_binding_report,
        "scene_asset_binding_report_path": str(scene_binding_report_path),
        "scene_profile_defaults_applied": dict(
            args.scene_profile_defaults_applied
        ),
        "scene_isaac_kit_args": list(scene_isaac_kit_args),
        "scene_isaac_app_overrides": dict(scene_isaac_app_overrides),
        "scene_isaac_runtime_overrides": dict(scene_isaac_runtime_overrides),
        "task_json": str(config.task_json),
        "output_dir": str(config.output_dir),
        "headless": bool(config.headless),
        "keep_window_open": bool(config.keep_window_open),
        "record_video": bool(config.video.enabled),
        "reuse_isaac_stage": bool(config.reuse_isaac_stage),
        "navigation_visual_mode": str(args.navigation_visual_mode),
        "overview_camera_mode": config.video.overview_camera_mode,
        "overview_camera_prim_path": config.recording.overview_camera_prim_path,
        "scene_light_mode": config.lighting.scene_light_mode,
        "global_planner": config.navigation.global_planner,
        "pct_coord_mode": config.navigation.pct_coord_mode,
        "policy_profile": config.locomotion.policy_profile,
        "pct_plan_preview_auto_keep_window_open": bool(
            pct_plan_preview and not config.headless
        ),
        "locomotion_task": config.locomotion.locomotion_task,
        "locomotion_checkpoint": (
            None
            if config.locomotion.locomotion_checkpoint is None
            else str(config.locomotion.locomotion_checkpoint)
        ),
        "phases": [],
    }
    _record_startup_phase(
        startup_status,
        startup_status_path,
        "config_ready",
        flat_episode_output=flat_episode_output,
        batch_summary_path=str(batch_summary_path) if batch_summary_path else None,
    )

    app_launcher = None
    planner_server = None
    retained_simulation = None
    shared_full_physics_simulation = None
    try:
        if config.full_physics or config.pick_smoke:
            from source.manipulation import (
                CuroboPlannerServerProcess,
                CuroboPlannerServerProcessConfig,
            )

            planner_server = CuroboPlannerServerProcess(
                CuroboPlannerServerProcessConfig(project_root=PROJECT_ROOT)
            )
            _record_startup_phase(
                startup_status,
                startup_status_path,
                "curobo_server_starting",
            )
            planner_server.start()
            _record_startup_phase(
                startup_status,
                startup_status_path,
                "curobo_server_started",
                start_report=planner_server.start_report,
            )
        if (
            simulation_smoke
            or navigation_smoke
            or navigation_carry_smoke
            or stair_locomotion_smoke
            or pct_plan_preview
            or pick_smoke
            or manipulation_apply_smoke
            or full_physics
        ):
            compat_report = patch_numpy_for_isaacsim()
            if compat_report["patched_broadcast_to"]:
                print(
                    "[full-physics] patched NumPy for Isaac Sim: "
                    f"broadcast_to restored; numpy={compat_report['numpy_version']} "
                    f"file={compat_report['numpy_file']}",
                    flush=True,
                )
            from isaaclab.app import AppLauncher

            _record_startup_phase(
                startup_status,
                startup_status_path,
                "isaac_app_starting",
                enable_cameras=bool(
                    ((config.full_physics or config.pick_smoke) and config.recording.enabled)
                    or config.video.enabled
                    or config.randomization.show_debug_region
                    or not config.headless
                ),
                kit_args=list(scene_isaac_kit_args),
                simulation_app_overrides=dict(scene_isaac_app_overrides),
            )
            app_launcher_config = {
                "headless": config.headless,
                "enable_cameras": bool(
                    ((config.full_physics or config.pick_smoke) and config.recording.enabled)
                    or config.video.enabled
                    or config.randomization.show_debug_region
                    or not config.headless
                ),
            }
            if scene_isaac_kit_args:
                app_launcher_config["kit_args"] = " ".join(scene_isaac_kit_args)
            app_launcher_config.update(scene_isaac_app_overrides)
            app_launcher = AppLauncher(app_launcher_config)
            _record_startup_phase(
                startup_status,
                startup_status_path,
                "isaac_app_started",
            )
            if planner_server is not None:
                ready = planner_server.wait_until_ready()
                _record_startup_phase(
                    startup_status,
                    startup_status_path,
                    "curobo_server_ready_checked",
                    ready=ready,
                    start_report=planner_server.start_report,
                )

        all_success = True
        for episode_index in range(config.num_episodes):
            episode_seed = config.episode_seed(episode_index)
            _record_startup_phase(
                startup_status,
                startup_status_path,
                "episode_spec_preparing",
                episode_index=episode_index,
                episode_seed=episode_seed,
            )
            episode_spec = prepare_episode_spec(
                base_spec,
                episode_id=base_spec.episode_id + episode_index,
                seed=episode_seed,
                settings=config.randomization,
            )
            if config.randomization.enabled or config.randomization.base_goal.enabled:
                pick_xy = (
                    "unavailable"
                    if episode_spec.object_initial_pose is None
                    else (
                        f"({episode_spec.object_initial_pose[0]:.6f},"
                        f"{episode_spec.object_initial_pose[1]:.6f})"
                    )
                )
                place_xy = (
                    "unavailable"
                    if episode_spec.place_target_pose is None
                    else (
                        f"({episode_spec.place_target_pose[0]:.6f},"
                        f"{episode_spec.place_target_pose[1]:.6f})"
                    )
                )
                print(
                    "[full-physics] randomization "
                    f"episode={episode_index} seed={episode_seed} "
                    f"pick_xy={pick_xy} place_xy={place_xy}",
                    flush=True,
                )
            base_goal_randomization = (
                episode_spec.raw_task.get("randomization", {})
                .get("base_goal_randomization")
            )
            if isinstance(base_goal_randomization, dict) and base_goal_randomization.get("enabled"):
                print(
                    "[base-goal-randomization] "
                    f"enabled=True seed={base_goal_randomization.get('seed')}",
                    flush=True,
                )
                for label in ("pick", "place"):
                    sample = base_goal_randomization.get(label)
                    if not isinstance(sample, dict) or not sample.get("valid"):
                        continue
                    if sample.get("fallback_used"):
                        print(
                            "[base-goal-randomization][warning] "
                            f"{label} sampling fallback used: {sample.get('fallback_reason')}",
                            flush=True,
                        )
                    goal = sample.get("sampled_base_goal_xyyaw") or ()
                    target = sample.get("target_xy") or ()
                    if len(goal) >= 3 and len(target) >= 2:
                        print(
                            f"[{label}-base-goal] "
                            f"target=({float(target[0]):.4f},{float(target[1]):.4f}) "
                            f"sampled=({float(goal[0]):.4f},{float(goal[1]):.4f},"
                            f"{float(goal[2]):.4f}) "
                            f"radius={float(sample.get('radius_m', 0.0)):.4f} "
                            f"attempt={sample.get('attempt')}",
                            flush=True,
                        )
            episode_dir = (
                config.output_dir
                if flat_episode_output
                else config.output_dir / f"episode_{episode_index:06d}"
            )
            _record_startup_phase(
                startup_status,
                startup_status_path,
                "pipeline_creating",
                episode_index=episode_index,
                episode_dir=str(episode_dir),
            )
            if dry_run:
                pipeline = create_dry_run_pipeline(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                )
            elif manipulation_smoke:
                from source.pipeline.manipulation_smoke import (
                    create_manipulation_smoke_pipeline,
                )

                pipeline = create_manipulation_smoke_pipeline(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                )
            elif manipulation_apply_smoke:
                from source.pipeline.manipulation_apply_smoke import (
                    create_manipulation_apply_smoke_pipeline,
                )
                from source.simulation import IsaacSimulationConfig, IsaacSimulationRuntime

                pipeline = create_manipulation_apply_smoke_pipeline(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                    simulation=IsaacSimulationRuntime(
                        simulation_app=app_launcher.app,
                        project_root=PROJECT_ROOT,
                        config=IsaacSimulationConfig(
                            show_randomization_debug=(
                                config.randomization.show_debug_region
                            ),
                        ),
                    ),
                )
            elif full_physics or pick_smoke:
                from source.pipeline.factory import (
                    create_full_physics_pipeline,
                )

                share_runtime = bool(
                    full_physics
                    and config.reuse_isaac_stage
                    and config.num_episodes > 1
                )
                if share_runtime:
                    if shared_full_physics_simulation is None:
                        shared_full_physics_simulation = _create_full_physics_runtime(
                            config=config,
                            args=args,
                            simulation_app=app_launcher.app,
                            scene_isaac_runtime_overrides=(
                                scene_isaac_runtime_overrides
                            ),
                        )
                    simulation = shared_full_physics_simulation
                else:
                    simulation = _create_full_physics_runtime(
                        config=config,
                        args=args,
                        simulation_app=app_launcher.app,
                        scene_isaac_runtime_overrides=scene_isaac_runtime_overrides,
                    )
                pipeline = create_full_physics_pipeline(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                    simulation=simulation,
                    close_simulation_on_exit=not share_runtime,
                )
            elif pct_plan_preview:
                from source.pipeline.pct_plan_preview import (
                    create_pct_plan_preview_pipeline,
                )

                pipeline = create_pct_plan_preview_pipeline(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                    simulation_app=app_launcher.app,
                    project_root=PROJECT_ROOT,
                )
            elif simulation_smoke:
                from source.pipeline.simulation_smoke import (
                    create_simulation_smoke_pipeline,
                )
                from source.simulation import IsaacSimulationConfig, IsaacSimulationRuntime

                pipeline = create_simulation_smoke_pipeline(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                    simulation=IsaacSimulationRuntime(
                        simulation_app=app_launcher.app,
                        project_root=PROJECT_ROOT,
                        config=IsaacSimulationConfig(
                            show_randomization_debug=(
                                config.randomization.show_debug_region
                            ),
                        ),
                    ),
                )
            elif navigation_smoke or navigation_carry_smoke or stair_locomotion_smoke:
                from source.pipeline.navigation_smoke import (
                    create_navigation_carry_smoke_pipeline,
                    create_navigation_smoke_pipeline,
                    create_stair_locomotion_smoke_pipeline,
                )
                from source.simulation import (
                    IsaacLabNavigationRuntime,
                    IsaacLabNavigationRuntimeConfig,
                )

                if stair_locomotion_smoke:
                    pipeline_factory = create_stair_locomotion_smoke_pipeline
                elif navigation_carry_smoke:
                    pipeline_factory = create_navigation_carry_smoke_pipeline
                else:
                    pipeline_factory = create_navigation_smoke_pipeline
                pipeline = pipeline_factory(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                    simulation=IsaacLabNavigationRuntime(
                        simulation_app=app_launcher.app,
                        project_root=PROJECT_ROOT,
                        config=IsaacLabNavigationRuntimeConfig(
                            **_locomotion_runtime_kwargs(config),
                            **_navigation_visual_runtime_kwargs(
                                config.locomotion.policy_profile,
                                args.navigation_visual_mode,
                            ),
                            **_navigation_smoke_viewport_runtime_kwargs(
                                headless=config.headless,
                                stair_locomotion_smoke=stair_locomotion_smoke,
                                overview_camera_mode=(
                                    config.video.overview_camera_mode
                                ),
                                overview_camera_prim_path=(
                                    config.recording.overview_camera_prim_path
                                ),
                            ),
                            show_velocity_command_debug=bool(
                                stair_locomotion_smoke
                                and config.show_planned_trajectories
                            ),
                            scene_light_mode=config.lighting.scene_light_mode,
                            camera_light_intensity=(
                                config.lighting.camera_light_intensity
                            ),
                            camera_light_radius=config.lighting.camera_light_radius,
                            show_randomization_debug=(
                                config.randomization.show_debug_region
                            ),
                        ),
                    ),
                )
            else:
                raise RuntimeError("未识别的 pipeline 执行模式")
            _record_startup_phase(
                startup_status,
                startup_status_path,
                "pipeline_created",
                episode_index=episode_index,
                pipeline_type=type(pipeline).__name__,
            )
            summary = pipeline.run_episode()
            if config.keep_window_open:
                retained_simulation = pipeline.simulation
            all_success = all_success and bool(summary["success"])
            _record_startup_phase(
                startup_status,
                startup_status_path,
                "episode_finished",
                episode_index=episode_index,
                success=bool(summary["success"]),
                final_state=summary.get("final_state"),
                failure_reason=summary.get("failure_reason"),
            )
            if batch_summary_path is not None:
                with batch_summary_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
                    )
                    stream.write("\n")
            print(
                f"[full-physics] episode={episode_index} seed={episode_seed} "
                f"mode={summary['execution_mode']} success={summary['success']} "
                f"pure_physics_success={summary['pure_physics_success']}",
                flush=True,
            )
            print(
                "[full-physics] states:",
                " -> ".join(summary["state_trace"]),
                flush=True,
            )
        if startup_status is not None:
            startup_status["status"] = "completed"
            startup_status["exit_code"] = 0 if all_success else 1
            _record_startup_phase(
                startup_status,
                startup_status_path,
                "completed",
                all_success=all_success,
            )
        if (
            (args.keep_window_open or (pct_plan_preview and not config.headless))
            and app_launcher is not None
        ):
            _keep_gui_open(app_launcher.app, retained_simulation)
        return 0 if all_success else 1
    except BaseException as exc:
        _record_startup_failure(
            startup_status,
            startup_status_path,
            config.output_dir,
            exc,
        )
        raise
    finally:
        try:
            if shared_full_physics_simulation is not None:
                shared_full_physics_simulation.close()
            elif retained_simulation is not None:
                retained_simulation.close()
            if app_launcher is not None:
                app_launcher.app.close()
        finally:
            if planner_server is not None:
                planner_server.close()


if __name__ == "__main__":
    raise SystemExit(main())
