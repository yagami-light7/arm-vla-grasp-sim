#!/usr/bin/env python3
"""Run the full-physics pipeline dry run or real Isaac simulation smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.pipeline import (  # noqa: E402
    FullPhysicsConfig,
    ManipulationSettings,
    RandomizationSettings,
)
from source.pipeline.dry_run import create_dry_run_pipeline  # noqa: E402
from source.tasks import JsonTaskProvider, prepare_episode_spec  # noqa: E402


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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
        "--pick-plan-json",
        help="预先生成的 pick cuRobo 分段计划 JSON；仅用于 manipulation apply smoke。",
    )
    parser.add_argument(
        "--place-plan-json",
        help="预先生成的 place cuRobo 分段计划 JSON；仅用于 manipulation apply smoke。",
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
    if (
        config.integrated_apply_smoke
        and config.place_plan_json is None
        and config.manipulation.replan_pick_from_current_state
    ):
        raise SystemExit(
            "integrated apply smoke 当前仍需要 --place-plan-json；pick 默认会按当前状态重规划。"
        )
    if (
        config.integrated_apply_smoke
        and not config.manipulation.replan_pick_from_current_state
        and (config.pick_plan_json is None or config.place_plan_json is None)
    ):
        raise SystemExit(
            "关闭 current-state pick replan 时，integrated apply smoke 必须同时提供 "
            "--pick-plan-json 和 --place-plan-json。"
        )
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
    if full_physics and (args.pick_plan_json or args.place_plan_json):
        raise SystemExit("默认 full-physics 模式禁止使用离线 plan JSON；pick/place 必须按当前仿真状态在线规划。")
    if args.keep_window_open and args.headless:
        raise SystemExit("--keep-window-open 只能与 --no-headless 一起使用。")
    if (
        simulation_smoke
        or navigation_smoke
        or navigation_carry_smoke
        or manipulation_apply_smoke
        or full_physics
    ) and args.num_episodes != 1:
        raise SystemExit("真实 Isaac smoke 目前只支持 --num-episodes 1。")

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
        manipulation=ManipulationSettings(),
        randomization=RandomizationSettings(
            enabled=args.randomize_task,
            show_debug_region=args.show_randomization_debug,
        ),
    )
    _validate_external_plan_paths(config)
    base_spec = JsonTaskProvider().load(config.task_json)
    config.output_dir.mkdir(parents=True, exist_ok=True)
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
            from isaaclab.app import AppLauncher

            app_launcher = AppLauncher(
                {
                    "headless": config.headless,
                    "enable_cameras": bool(
                        (config.full_physics and config.recording.enabled)
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
            if config.randomization.enabled:
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
            episode_dir = config.output_dir / f"episode_{episode_index:06d}"
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
                from source.pipeline.integrated_apply_smoke import (
                    create_full_physics_pipeline,
                )
                from source.simulation import (
                    IsaacLabNavigationRuntime,
                    IsaacLabNavigationRuntimeConfig,
                )

                pipeline = create_full_physics_pipeline(
                    config=config,
                    episode_spec=episode_spec,
                    episode_seed=episode_seed,
                    episode_dir=episode_dir,
                    simulation=IsaacLabNavigationRuntime(
                        simulation_app=app_launcher.app,
                        project_root=PROJECT_ROOT,
                        config=IsaacLabNavigationRuntimeConfig(
                            enable_front_camera=config.recording.enabled,
                            front_camera_height=config.recording.image_height,
                            front_camera_width=config.recording.image_width,
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
            with batch_summary_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
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
