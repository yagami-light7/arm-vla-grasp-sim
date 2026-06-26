#!/usr/bin/env python3
"""Run the full-physics pipeline dry run or real Isaac simulation smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.pipeline import (  # noqa: E402
    BaseGoalRandomizationSettings,
    FullPhysicsConfig,
    LocomotionPolicySettings,
    ManipulationSettings,
    NavigationSettings,
    RandomizationSettings,
    RecordingSettings,
    VideoRecordingSettings,
)
from source.pipeline.dry_run import create_dry_run_pipeline  # noqa: E402
from source.pipeline.isaac_compat import patch_numpy_for_isaacsim  # noqa: E402
from source.tasks import JsonTaskProvider, prepare_episode_spec  # noqa: E402


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _optional_project_path(raw_path: str | Path | None) -> Path | None:
    return None if raw_path is None else _project_path(raw_path)


def _locomotion_runtime_kwargs(config: FullPhysicsConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if config.locomotion.locomotion_task:
        kwargs["task_name"] = config.locomotion.locomotion_task
    if config.locomotion.locomotion_checkpoint is not None:
        kwargs["checkpoint"] = config.locomotion.locomotion_checkpoint
    return kwargs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行单进程、单 World 的纯物理 nav-pick-place pipeline。",
    )
    parser.add_argument("--task-json", required=True, help="任务 JSON 路径。")
    parser.add_argument(
        "--output-dir",
        default="outputs/full_physics_pipeline",
        help="episode、事件、帧和 summary 的输出目录。",
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="运行的 episode 数量。")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="首个 episode 的随机种子；相同 seed 会严格复现相同的随机 XY。",
    )
    parser.add_argument(
        "--randomize-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按 episode seed 随机采样 pick/place XY；默认开启，可用 --no-randomize-task 关闭。",
    )
    parser.add_argument(
        "--show-randomization-debug",
        action="store_true",
        help="显示 pick/place 随机区域和采样点的非物理 USD guide；默认关闭。",
    )
    parser.add_argument(
        "--randomize-base-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="开启 pick/place 导航交接 base_goal 极坐标随机化；默认开启，可用 --no-randomize-base-goal 关闭。",
    )
    parser.add_argument(
        "--keep-window-open",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pipeline 结束后保持 GUI 窗口，便于检查场景和调试标记；默认关闭。",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否以无界面模式运行；使用 --headless 关闭GUI渲染。",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="启用展示用 overview 视频录制；默认关闭。",
    )
    parser.add_argument(
        "--video-mode",
        choices=("overview", "front", "font", "wrist", "all"),
        default="overview",
        help=(
            "录制视频类型：overview 为第三人称展示视角，front/font 为前视 observation "
            "camera，wrist 为腕部 observation camera，all 同时导出 overview/front/wrist。"
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
        choices=("auto",),
        default="auto",
        help="overview camera 选择模式；auto 会遍历 USD stage 中已有 Camera 并按任务阶段切换。",
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
        default="astar",
        help="全局导航规划器；默认 astar，pct 使用多楼层 PCT server。",
    )
    parser.add_argument("--pct-planner-root", help="兼容旧外部入口的 PCT 根目录；默认不依赖 external/PCT。")
    parser.add_argument("--pct-server-script", help="PCT server 脚本路径；PCT 模式默认使用本仓库本地版本。")
    parser.add_argument("--pct-server-python", help="运行 PCT server 的 Python 解释器。")
    parser.add_argument("--pct-tomogram-path", help="PCT tomogram pickle 路径。")
    parser.add_argument("--pct-walkable-path", help="PCT walkable map .npy 路径。")
    parser.add_argument(
        "--pct-no-fallback",
        action="store_true",
        help="PCT 规划失败时不回退 A*，直接报错。",
    )
    parser.add_argument("--pct-offset-x", type=float, default=0.0, help="PCT 坐标 X 偏移。")
    parser.add_argument("--pct-offset-y", type=float, default=0.0, help="PCT 坐标 Y 偏移。")
    parser.add_argument("--pct-scale-x", type=float, default=1.0, help="PCT 坐标 X 缩放。")
    parser.add_argument("--pct-scale-y", type=float, default=1.0, help="PCT 坐标 Y 缩放。")
    parser.add_argument(
        "--goal-z-tolerance",
        type=float,
        default=0.35,
        help="多楼层导航目标 z 到达容差；旧 XY-only 任务不触发 z 检查。",
    )
    parser.add_argument("--locomotion-task", help="Isaac Lab locomotion task 名称。")
    parser.add_argument("--locomotion-checkpoint", help="RSL-RL locomotion checkpoint 路径。")
    parser.add_argument(
        "--policy-profile",
        choices=("flat", "pct_multifloor"),
        default="flat",
        help="底层 locomotion policy profile；pct_multifloor 要求提供 checkpoint。",
    )
    parser.add_argument(
        "--require-locomotion-checkpoint",
        action="store_true",
        help="要求显式 locomotion checkpoint 存在。",
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _validate_external_plan_paths(config: FullPhysicsConfig) -> None:
    if config.full_physics and (config.pick_plan_json is not None or config.place_plan_json is not None):
        raise SystemExit("full-physics 模式按当前仿真状态在线规划 pick/place，不接受 --pick-plan-json/--place-plan-json。")
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
    navigation_smoke = mode == "navigation_smoke"
    navigation_carry_smoke = mode == "navigation_carry_smoke"
    manipulation_smoke = mode == "manipulation_smoke"
    manipulation_apply_smoke = mode == "manipulation_apply_smoke"
    full_physics = mode == "full_physics"
    flat_episode_output = os.environ.get("FULL_PHYSICS_FLAT_EPISODE_OUTPUT") == "1"
    if full_physics and (args.pick_plan_json or args.place_plan_json):
        raise SystemExit("默认 full-physics 模式禁止使用离线 plan JSON；pick/place 必须按当前仿真状态在线规划。")
    if args.keep_window_open and args.headless:
        raise SystemExit("--keep-window-open 只能与 --no-headless 一起使用。")
    if args.record_video and dry_run:
        raise SystemExit("--record-video 需要真实 Isaac stage / camera images，不能与 --dry-run 一起使用。")
    if args.export_video_camera_trajectory and not args.record_video:
        raise SystemExit("--export-video-camera-trajectory 需要同时启用 --record-video。")
    if args.export_video_camera_trajectory and str(args.video_mode).lower() not in {"overview", "all"}:
        raise SystemExit("--export-video-camera-trajectory 只支持 --video-mode overview 或 all。")
    if args.record_video and args.video_out and Path(args.video_out).suffix.lower() == ".mp4" and args.num_episodes != 1:
        raise SystemExit("--video-out 指向单个 .mp4 文件时只支持 --num-episodes 1；多 episode 请传输出目录。")
    if (
        simulation_smoke
        or navigation_smoke
        or navigation_carry_smoke
        or manipulation_apply_smoke
        or full_physics
    ) and args.num_episodes != 1:
        raise SystemExit("真实 Isaac smoke 目前只支持 --num-episodes 1。")
    if flat_episode_output and args.num_episodes != 1:
        raise SystemExit("batch 扁平输出模式只支持单 episode 子进程。")

    locomotion_checkpoint = _optional_project_path(args.locomotion_checkpoint)
    if args.policy_profile == "pct_multifloor" and (
        locomotion_checkpoint is None or not locomotion_checkpoint.is_file()
    ):
        raise SystemExit(
            "PCT multi-floor policy checkpoint missing. "
            "Please train or pass --locomotion-checkpoint."
        )

    config = FullPhysicsConfig(
        task_json=_project_path(args.task_json),
        output_dir=_project_path(args.output_dir),
        num_episodes=args.num_episodes,
        seed=args.seed,
        headless=args.headless,
        keep_window_open=args.keep_window_open,
        pick_plan_json=_project_path(args.pick_plan_json) if args.pick_plan_json else None,
        place_plan_json=_project_path(args.place_plan_json) if args.place_plan_json else None,
        dry_run=dry_run,
        simulation_smoke=simulation_smoke,
        navigation_smoke=navigation_smoke,
        navigation_carry_smoke=navigation_carry_smoke,
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
            pct_fallback_to_astar=not bool(args.pct_no_fallback),
            pct_offset_x=float(args.pct_offset_x),
            pct_offset_y=float(args.pct_offset_y),
            pct_scale_x=float(args.pct_scale_x),
            pct_scale_y=float(args.pct_scale_y),
            goal_z_tolerance=float(args.goal_z_tolerance),
        ),
        locomotion=LocomotionPolicySettings(
            locomotion_task=args.locomotion_task,
            locomotion_checkpoint=locomotion_checkpoint,
            locomotion_checkpoint_required=bool(args.require_locomotion_checkpoint),
            policy_profile=str(args.policy_profile),
        ),
        manipulation=ManipulationSettings(),
        randomization=RandomizationSettings(
            enabled=args.randomize_task,
            show_debug_region=args.show_randomization_debug,
            base_goal=BaseGoalRandomizationSettings(
                enabled=args.randomize_base_goal,
            ),
        ),
        recording=RecordingSettings(
            debug_per_episode_lerobot=(
                os.environ.get("FULL_PHYSICS_DEFER_LEROBOT_EXPORT") != "1"
            ),
        ),
        video=VideoRecordingSettings(
            enabled=bool(args.record_video),
            mode=str(args.video_mode),
            output_path=_project_path(args.video_out) if args.video_out else None,
            overview_camera_mode=str(args.overview_camera_mode),
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
    config.output_dir.mkdir(parents=True, exist_ok=True)
    batch_summary_path = None
    if not flat_episode_output:
        batch_summary_path = config.output_dir / "batch_summary.jsonl"
        batch_summary_path.write_text("", encoding="utf-8")

    app_launcher = None
    planner_server = None
    retained_simulation = None
    try:
        if (
            config.full_physics
        ):
            from source.manipulation import (
                CuroboPlannerServerProcess,
                CuroboPlannerServerProcessConfig,
            )

            planner_server = CuroboPlannerServerProcess(
                CuroboPlannerServerProcessConfig(project_root=PROJECT_ROOT)
            )
            planner_server.start()
        if (
            simulation_smoke
            or navigation_smoke
            or navigation_carry_smoke
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

            app_launcher = AppLauncher(
                {
                    "headless": config.headless,
                    "enable_cameras": bool(
                        (config.full_physics and config.recording.enabled)
                        or config.video.enabled
                        or config.randomization.show_debug_region
                        or not config.headless
                    ),
                }
            )
            if planner_server is not None:
                planner_server.wait_until_ready()

        all_success = True
        for episode_index in range(config.num_episodes):
            episode_seed = config.episode_seed(episode_index)
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
            elif full_physics:
                from source.pipeline.factory import (
                    create_full_physics_pipeline,
                )
                from source.simulation import (
                    IsaacLabNavigationRuntime,
                    IsaacLabNavigationRuntimeConfig,
                )

                video_modes = set(config.video.modes) if config.video.enabled else set()
                pipeline = create_full_physics_pipeline(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                    simulation=IsaacLabNavigationRuntime(
                        simulation_app=app_launcher.app,
                        project_root=PROJECT_ROOT,
                        config=IsaacLabNavigationRuntimeConfig(
                            **_locomotion_runtime_kwargs(config),
                            enable_front_camera=(
                                (
                                    config.recording.enabled
                                    and "front" in config.recording.camera_keys
                                )
                                or "front" in video_modes
                            ),
                            front_camera_height=config.recording.image_height,
                            front_camera_width=config.recording.image_width,
                            enable_wrist_camera=(
                                (
                                    config.recording.enabled
                                    and "wrist" in config.recording.camera_keys
                                )
                                or "wrist" in video_modes
                            ),
                            wrist_camera_height=config.recording.image_height,
                            wrist_camera_width=config.recording.image_width,
                            place_release_clearance_min_m=(
                                config.manipulation.place_release_clearance_min_m
                            ),
                            place_pre_clearance_min_m=(
                                config.manipulation.place_pre_clearance_min_m
                            ),
                            show_randomization_debug=(
                                config.randomization.show_debug_region
                            ),
                        ),
                    ),
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
            elif navigation_smoke or navigation_carry_smoke:
                from source.pipeline.navigation_smoke import (
                    create_navigation_carry_smoke_pipeline,
                    create_navigation_smoke_pipeline,
                )
                from source.simulation import (
                    IsaacLabNavigationRuntime,
                    IsaacLabNavigationRuntimeConfig,
                )

                pipeline_factory = (
                    create_navigation_carry_smoke_pipeline
                    if navigation_carry_smoke
                    else create_navigation_smoke_pipeline
                )
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
                            show_randomization_debug=(
                                config.randomization.show_debug_region
                            ),
                        ),
                    ),
                )
            else:
                raise RuntimeError("未识别的 pipeline 执行模式")
            summary = pipeline.run_episode()
            if config.keep_window_open:
                retained_simulation = pipeline.simulation
            all_success = all_success and bool(summary["success"])
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
        if args.keep_window_open and app_launcher is not None:
            _keep_gui_open(app_launcher.app, retained_simulation)
        return 0 if all_success else 1
    finally:
        try:
            if retained_simulation is not None:
                retained_simulation.close()
            if app_launcher is not None:
                app_launcher.app.close()
        finally:
            if planner_server is not None:
                planner_server.close()


if __name__ == "__main__":
    raise SystemExit(main())
