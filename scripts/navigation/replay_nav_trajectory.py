#!/usr/bin/env python3
"""Replay a recorded Go2-X5 navigation trajectory in Isaac Lab for video capture."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_LAB_SOURCE = PROJECT_ROOT / "source/robot_lab"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROBOT_LAB_SOURCE))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-json", required=True, help="Task JSON used to resolve scene assets.")
parser.add_argument("--trajectory", required=True, help="Recorded replay trajectory JSONL.")
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0")
parser.add_argument("--map", dest="scene_usd", default=None, help="Override task scene USD.")
parser.add_argument("--terrain-prim-path", default="/World/scene_collision")
parser.add_argument("--ground-height", type=float, default=0.0)
parser.add_argument("--add-nav-ground", action="store_true")
parser.add_argument("--load-visual-scene", action="store_true")
parser.add_argument("--visual-load-mode", choices=("sublayer", "reference"), default="sublayer")
parser.add_argument("--visual-prim-path", default="/World/gauss")
parser.add_argument("--hide-nav-collision-visual", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--loop", action="store_true")
parser.add_argument("--follow-camera-mode", choices=("fixed", "overhead", "chase", "front", "stage"), default="fixed")
parser.add_argument(
    "--viewport-camera-prim",
    default="/World/camera_main",
    help="USD Camera prim used when --follow-camera-mode stage is selected.",
)
parser.add_argument("--fixed-camera-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--fixed-camera-lookat", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from pxr import Tf, Usd, UsdGeom, UsdPhysics

import robot_lab.tasks  # noqa: F401
from source.data import load_task
from source.navigation.adapters.terrain_utils import (
    write_collision_terrain_wrapper,
    write_visual_prim_wrapper,
    write_visual_sublayer_wrapper,
)


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_trajectory(path: Path) -> list[dict]:
    """Load a JSONL replay trajectory."""

    if not path.exists():
        raise RuntimeError(f"Replay trajectory does not exist: {path}")
    frames: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            for key in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel"):
                if key not in frame:
                    raise RuntimeError(f"Replay frame {line_number} is missing required key: {key}")
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"Replay trajectory is empty: {path}")
    return frames


def _open_scene_stage(scene_usd: Path) -> Usd.Stage:
    """Open a scene USD with actionable diagnostics."""

    if not scene_usd.exists():
        raise RuntimeError(f"Scene USD does not exist: {scene_usd}")
    size_bytes = scene_usd.stat().st_size
    if size_bytes <= 0:
        raise RuntimeError(f"Scene USD is empty: {scene_usd}")
    try:
        stage = Usd.Stage.Open(str(scene_usd))
    except Tf.ErrorException as exc:
        raise RuntimeError(
            f"Failed to open scene USD: {scene_usd} ({size_bytes} bytes). "
            "Check that external SAGE assets are mounted."
        ) from exc
    if stage is None:
        raise RuntimeError(f"Failed to open scene USD: {scene_usd} ({size_bytes} bytes).")
    return stage


def _validate_scene_collision(scene_usd: Path, prim_path: str) -> None:
    """Fail fast if the scene collision payload is missing or empty."""

    stage = _open_scene_stage(scene_usd)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Scene collision prim does not exist: {prim_path} in {scene_usd}")
    mesh_count = 0
    collision_api_count = 0
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            mesh_count += 1
        if child.HasAPI(UsdPhysics.CollisionAPI):
            collision_api_count += 1
    print(
        f"[INFO] Scene collision preflight: prim={prim_path} "
        f"meshes={mesh_count} collision_api={collision_api_count}"
    )
    if mesh_count == 0:
        raise RuntimeError(f"Scene collision prim {prim_path} has no mesh geometry.")


def _validate_scene_prim(scene_usd: Path, prim_path: str, label: str) -> None:
    """Fail fast if an optional scene prim path is not present."""

    stage = _open_scene_stage(scene_usd)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Scene {label} prim does not exist: {prim_path} in {scene_usd}")
    print(f"[INFO] Scene {label} preflight: prim={prim_path} type={prim.GetTypeName() or 'typeless'}")


def _load_visual_scene_sublayer(
    scene_usd: Path,
    visual_prim_path: str,
    visual_exclude_prim_paths: list[str] | tuple[str, ...],
) -> bool:
    """Add the complete SAGE scene as a display-only sublayer before env creation."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] Cannot load full-scene visual sublayer before an Isaac stage exists.")
        return False
    wrapper = write_visual_sublayer_wrapper(
        scene_usd,
        visual_prim_path,
        excluded_prim_paths=visual_exclude_prim_paths,
    )
    root_layer = stage.GetRootLayer()
    wrapper_path = str(wrapper)
    if wrapper_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(wrapper_path)
    print(f"[INFO] Replay visual sublayer: {wrapper} -> {scene_usd} with visible prim {visual_prim_path}")
    return True


def _hide_nav_collision_visual() -> None:
    """Hide navigation collision geometry in the viewport while keeping physics active."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] Cannot hide navigation collision visual: no active USD stage.")
        return
    hidden_any = False
    for prim_path in ("/World/nav_collision/terrain", "/World/nav_collision"):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN] Navigation collision visual prim not found: {prim_path}")
            continue
        if not prim.IsA(UsdGeom.Imageable):
            print(f"[WARN] Navigation collision visual prim is not imageable: {prim_path}")
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        print(f"[INFO] Hid navigation collision visual: {prim_path}")
        hidden_any = True
    if not hidden_any:
        print("[WARN] No navigation collision visual prim found to hide.")


def _disable_event(env_cfg, name: str) -> None:
    if hasattr(env_cfg.events, name):
        setattr(env_cfg.events, name, None)


def _configure_env(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    *,
    scene_usd: Path,
    start_pose: tuple[float, float, float],
    visual_exclude_prim_paths: list[str] | tuple[str, ...],
) -> None:
    """Configure a one-env replay scene without policy randomization."""

    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    terrain_usd = write_collision_terrain_wrapper(scene_usd, args_cli.terrain_prim_path)
    print(f"[INFO] Replay terrain wrapper: {terrain_usd} -> {scene_usd}<{args_cli.terrain_prim_path}>")
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/nav_collision",
        terrain_type="usd",
        usd_path=str(terrain_usd),
        debug_vis=False,
    )
    if args_cli.add_nav_ground:
        env_cfg.scene.nav_ground = AssetBaseCfg(
            prim_path="/World/nav_ground",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.ground_height)),
            spawn=sim_utils.GroundPlaneCfg(),
        )
    if args_cli.load_visual_scene and args_cli.visual_load_mode == "reference":
        visual_usd = write_visual_prim_wrapper(
            scene_usd,
            args_cli.visual_prim_path,
            excluded_prim_paths=visual_exclude_prim_paths,
        )
        print(f"[INFO] Replay visual wrapper: {visual_usd} -> {scene_usd} with visible prim {args_cli.visual_prim_path}")
        env_cfg.scene.visual_scene = AssetBaseCfg(
            prim_path="/World/nav_visual_scene",
            spawn=sim_utils.UsdFileCfg(usd_path=str(visual_usd)),
        )
    start_x, start_y, start_yaw = start_pose
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {
            "x": (start_x, start_x),
            "y": (start_y, start_y),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (start_yaw, start_yaw),
        },
        "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
    }
    env_cfg.observations.policy.enable_corruption = False
    for event_name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "push_robot",
        "randomize_push_robot",
        "randomize_actuator_gains",
    ):
        _disable_event(env_cfg, event_name)
    for curriculum_name in ("terrain_levels", "command_levels_lin_vel", "command_levels_ang_vel"):
        if hasattr(env_cfg.curriculum, curriculum_name):
            setattr(env_cfg.curriculum, curriculum_name, None)
    env_cfg.terminations.time_out = None
    env_cfg.terminations.illegal_contact = None
    env_cfg.terminations.terrain_out_of_bounds = None


def _yaw_from_quat_wxyz(quat: list[float]) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _default_camera_from_frame(frame: dict) -> tuple[list[float], list[float]]:
    root = [float(value) for value in frame["root_pos_w"]]
    yaw = _yaw_from_quat_wxyz([float(value) for value in frame["root_quat_w"]])
    forward = (math.cos(yaw), math.sin(yaw), 0.0)
    left = (-math.sin(yaw), math.cos(yaw), 0.0)
    eye = [
        root[0] - 2.2 * forward[0] - 0.75 * left[0],
        root[1] - 2.2 * forward[1] - 0.75 * left[1],
        root[2] + 1.35,
    ]
    lookat = [root[0] + 0.35 * forward[0], root[1] + 0.35 * forward[1], root[2] + 0.25]
    return eye, lookat


def _candidate_stage_camera_paths(camera_prim_path: str) -> list[str]:
    """Return camera prim path candidates for sublayer and referenced visual modes."""

    candidates = [camera_prim_path]
    if camera_prim_path == "/World/camera_main":
        candidates.append("/World/Camera_main")
    if camera_prim_path.startswith("/World/"):
        for candidate in tuple(candidates):
            candidates.append("/World/nav_visual_scene/" + candidate.removeprefix("/World/"))
    return list(dict.fromkeys(candidates))


def _set_viewport_stage_camera(camera_prim_path: str) -> bool:
    """Switch the active viewport to a USD camera prim authored in the stage."""

    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] Cannot set stage camera: no active USD stage.")
            return False

        selected_path = None
        for candidate in _candidate_stage_camera_paths(camera_prim_path):
            prim = stage.GetPrimAtPath(candidate)
            if prim.IsValid() and prim.IsA(UsdGeom.Camera):
                selected_path = candidate
                break
        if selected_path is None:
            print(
                "[WARN] Cannot set stage camera: no valid Camera prim found. "
                f"tried={_candidate_stage_camera_paths(camera_prim_path)}"
            )
            return False

        viewport = get_active_viewport()
        if viewport is None:
            print("[WARN] Cannot set stage camera: no active viewport.")
            return False
        sdf_path = Sdf.Path(selected_path)
        try:
            viewport.camera_path = sdf_path
        except Exception:
            if hasattr(viewport, "set_active_camera"):
                viewport.set_active_camera(sdf_path)
            else:
                raise
        print(f"[INFO] Viewport camera set to stage camera: {selected_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to set stage camera {camera_prim_path}: {exc}")
        return False


def _update_view_camera(env, frame: dict) -> None:
    """Update the viewport camera for replay."""

    runtime = env.unwrapped
    controller = getattr(runtime, "viewport_camera_controller", None)
    mode = args_cli.follow_camera_mode
    if mode == "stage":
        if getattr(_update_view_camera, "_stage_camera_applied", False):
            return
        if _set_viewport_stage_camera(args_cli.viewport_camera_prim):
            _update_view_camera._stage_camera_applied = True
        return
    if controller is None:
        return
    root_pos = torch.tensor(frame["root_pos_w"], dtype=torch.float32, device=runtime.device)
    root_quat = torch.tensor(frame["root_quat_w"], dtype=torch.float32, device=runtime.device)
    if mode == "fixed":
        if getattr(_update_view_camera, "_fixed_camera_applied", False):
            return
        if args_cli.fixed_camera_eye is None or args_cli.fixed_camera_lookat is None:
            eye_values, lookat_values = _default_camera_from_frame(frame)
        else:
            eye_values, lookat_values = args_cli.fixed_camera_eye, args_cli.fixed_camera_lookat
        eye = torch.tensor(eye_values, dtype=torch.float32, device=runtime.device)
        lookat = torch.tensor(lookat_values, dtype=torch.float32, device=runtime.device)
    elif mode == "overhead":
        eye = root_pos + torch.tensor([-0.8, 0.0, 2.8], dtype=torch.float32, device=runtime.device)
        lookat = root_pos + torch.tensor([0.0, 0.0, 0.20], dtype=torch.float32, device=runtime.device)
    else:
        direction = 1.0 if mode == "front" else -1.0
        offset = torch.tensor([direction * 2.4, 0.0, 0.9], dtype=torch.float32, device=runtime.device)
        eye = math_utils.transform_points(offset.unsqueeze(0), pos=root_pos.unsqueeze(0), quat=root_quat.unsqueeze(0)).squeeze(0)
        lookat = root_pos + torch.tensor([0.0, 0.0, 0.20], dtype=torch.float32, device=runtime.device)
    controller.set_view_env_index(env_index=0)
    controller.update_view_location(
        eye=eye.detach().cpu().numpy(),
        lookat=lookat.detach().cpu().numpy(),
    )
    if mode == "fixed":
        _update_view_camera._fixed_camera_applied = True


def _validate_joint_schema(robot, frame: dict) -> None:
    """Validate trajectory joint count and warn on joint-name mismatches."""

    actual_joint_names = list(robot.joint_names)
    recorded_joint_names = list(frame.get("joint_names") or [])
    recorded_count = len(frame["joint_pos"])
    actual_count = int(robot.data.joint_pos.shape[1])
    if recorded_count != actual_count:
        raise RuntimeError(
            "Replay joint count mismatch: "
            f"trajectory has {recorded_count}, robot has {actual_count}. "
            f"recorded_names={recorded_joint_names}, actual_names={actual_joint_names}"
        )
    if recorded_joint_names and recorded_joint_names != actual_joint_names:
        print(
            "[WARN] Replay joint_names differ from current robot joint order. "
            "Lengths match, so replay will continue using recorded index order."
        )
        print(f"[WARN] recorded_joint_names={recorded_joint_names}")
        print(f"[WARN] actual_joint_names={actual_joint_names}")


def _write_frame_to_robot(env, frame: dict) -> None:
    """Write one replay frame into the robot articulation and advance the simulation."""

    runtime = env.unwrapped
    robot = runtime.scene["robot"]
    device = runtime.device
    root_pose = torch.tensor(
        [[*frame["root_pos_w"], *frame["root_quat_w"]]],
        dtype=torch.float32,
        device=device,
    )
    root_velocity = torch.tensor(
        [[*frame["root_lin_vel_w"], *frame["root_ang_vel_w"]]],
        dtype=torch.float32,
        device=device,
    )
    joint_pos = torch.tensor([frame["joint_pos"]], dtype=torch.float32, device=device)
    joint_vel = torch.tensor([frame["joint_vel"]], dtype=torch.float32, device=device)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    runtime.scene.write_data_to_sim()
    runtime.sim.step(render=not args_cli.headless)
    runtime.scene.update(float(runtime.physics_dt))


def _sleep_for_frame(previous_frame: dict | None, frame: dict, fallback_dt: float) -> None:
    if not args_cli.real_time:
        return
    speed = max(float(args_cli.speed), 1.0e-6)
    if previous_frame is None:
        delay = fallback_dt / speed
    else:
        delay = max(0.0, float(frame.get("timestamp", 0.0)) - float(previous_frame.get("timestamp", 0.0))) / speed
    time.sleep(delay)


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg) -> None:
    task_path = _project_path(args_cli.task_json)
    task = load_task(task_path)
    trajectory_path = _project_path(args_cli.trajectory)
    frames = _read_trajectory(trajectory_path)
    scene_usd = _project_path(args_cli.scene_usd or task.scene_usd)
    _validate_scene_collision(scene_usd, args_cli.terrain_prim_path)
    if args_cli.load_visual_scene:
        _validate_scene_prim(scene_usd, args_cli.visual_prim_path, "visual")

    visual_exclude_prim_paths = [
        args_cli.terrain_prim_path,
        "/World/go2_x5",
        "/World/mec_arm_6dof",
    ]
    object_prim_path = getattr(task.pick, "object_prim_path", None)
    if object_prim_path:
        visual_exclude_prim_paths.append(object_prim_path)
    if args_cli.load_visual_scene and args_cli.visual_load_mode == "sublayer":
        if not _load_visual_scene_sublayer(scene_usd, args_cli.visual_prim_path, visual_exclude_prim_paths):
            print("[WARN] Falling back to stable visual reference mode.")
            args_cli.visual_load_mode = "reference"

    first_frame = frames[0]
    start_pose = (
        float(first_frame["root_pos_w"][0]),
        float(first_frame["root_pos_w"][1]),
        _yaw_from_quat_wxyz([float(value) for value in first_frame["root_quat_w"]]),
    )
    _configure_env(
        env_cfg,
        scene_usd=scene_usd,
        start_pose=start_pose,
        visual_exclude_prim_paths=visual_exclude_prim_paths,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env.reset()
    if args_cli.hide_nav_collision_visual:
        _hide_nav_collision_visual()
    robot = env.unwrapped.scene["robot"]
    _validate_joint_schema(robot, first_frame)
    fallback_dt = float(getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt))
    print(f"[INFO] Replaying trajectory: {trajectory_path} frames={len(frames)} speed={args_cli.speed}")
    try:
        while simulation_app.is_running():
            previous_frame = None
            for frame in frames:
                if not simulation_app.is_running():
                    break
                _write_frame_to_robot(env, frame)
                _update_view_camera(env, frame)
                _sleep_for_frame(previous_frame, frame, fallback_dt)
                previous_frame = frame
            if not args_cli.loop:
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
