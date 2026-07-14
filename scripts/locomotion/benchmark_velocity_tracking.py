#!/usr/bin/env python3
"""Benchmark a Go2-X5 locomotion checkpoint's body-velocity tracking on a flat plane."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "source/robot_lab"))

from scripts.navigation import isaaclab_cli_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--profile", choices=("quick", "full"), default="quick")
parser.add_argument("--settle-seconds", type=float, default=2.0)
parser.add_argument("--hold-seconds", type=float, default=3.0)
parser.add_argument("--stop-seconds", type=float, default=1.5)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--policy-action-warmup-steps", type=int, default=50)
parser.add_argument("--standing-command-threshold", type=float, default=0.0)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--print-every", type=int, default=50)
isaaclab_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401
from source.locomotion_benchmark import build_schedule, write_benchmark_artifacts
from source.navigation.adapters.isaaclab_go2_adapter import Go2LocomotionAdapter


def _set_if_present(owner, name: str, value) -> None:
    if owner is not None and hasattr(owner, name):
        setattr(owner, name, value)


def _configure_flat_deterministic(env_cfg: ManagerBasedRLEnvCfg, total_duration_s: float) -> None:
    """Use a plane while retaining the DogOnly checkpoint's height-scan observations."""

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 4.0
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.episode_length_s = max(float(total_duration_s) + 10.0, 60.0)
    _set_if_present(env_cfg.curriculum, "terrain_levels", None)
    _set_if_present(env_cfg.observations.policy, "enable_corruption", False)
    if hasattr(env_cfg, "sim2sim_action_delay_range"):
        env_cfg.sim2sim_action_delay_range = (0, 0)
    if hasattr(env_cfg, "sim2sim_action_hold_prob"):
        env_cfg.sim2sim_action_hold_prob = 0.0
    if hasattr(env_cfg, "sim2sim_action_noise_std"):
        env_cfg.sim2sim_action_noise_std = 0.0
    if hasattr(env_cfg, "sim2sim_obs_delay_steps"):
        env_cfg.sim2sim_obs_delay_steps = 0
    events = getattr(env_cfg, "events", None)
    for name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_actuator_gains",
        "randomize_push_robot",
        "randomize_reset_joints",
    ):
        _set_if_present(events, name, None)
    reset_event = getattr(events, "randomize_reset_base", None)
    if reset_event is not None:
        reset_event.params = {
            "pose_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
            "velocity_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        }


def _step(adapter: Go2LocomotionAdapter):
    with torch.inference_mode():
        actions = adapter.compute_policy_action()
        command_diagnostics = adapter.diagnostics()
        observations, _, dones, extras = adapter.env.step(actions)
        adapter.update_observations(observations)
    return bool(dones[0].item()), extras, command_diagnostics


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule(
        args_cli.profile,
        settle_s=args_cli.settle_seconds,
        hold_s=args_cli.hold_seconds,
        stop_s=args_cli.stop_seconds,
        repeats=args_cli.repeats,
    )
    total_duration_s = sum(segment.duration_s for segment in schedule)
    _configure_flat_deterministic(env_cfg, total_duration_s)

    agent_cfg = isaaclab_cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    adapter = Go2LocomotionAdapter(
        env,
        policy,
        env.get_observations(),
        standing_command_threshold=args_cli.standing_command_threshold,
        policy_action_warmup_steps=args_cli.policy_action_warmup_steps,
    )

    dt = float(env.unwrapped.step_dt)
    samples: list[dict] = []
    samples_path = output_dir / "samples.jsonl"
    benchmark_time_s = 0.0
    global_step = 0
    terminated_early = False
    with samples_path.open("w", encoding="utf-8", buffering=1) as stream:
        for segment_index, segment in enumerate(schedule):
            segment_steps = max(1, round(segment.duration_s / dt))
            print(
                f"[benchmark] segment={segment.name} steps={segment_steps} "
                f"cmd=({segment.vx:.3f}, {segment.vy:.3f}, {segment.wz:.3f})"
            )
            for segment_step in range(segment_steps):
                wall_start = time.perf_counter()
                adapter.apply_base_command(segment.vx, segment.vy, segment.wz)
                done, _, command_diagnostics = _step(adapter)
                diagnostics = adapter.diagnostics()
                pose = adapter.get_base_pose_full()
                row = {
                    "time_s": benchmark_time_s,
                    "wall_time_s": time.time(),
                    "global_step": global_step,
                    "segment_index": segment_index,
                    "segment_name": segment.name,
                    "segment_time_s": (segment_step + 1) * dt,
                    "evaluate": segment.evaluate,
                    "cmd_vx": segment.vx,
                    "cmd_vy": segment.vy,
                    "cmd_wz": segment.wz,
                    "command_seen_vx": command_diagnostics["command_seen_vx"],
                    "command_seen_vy": command_diagnostics["command_seen_vy"],
                    "command_seen_wz": command_diagnostics["command_seen_wz"],
                    "measured_vx": diagnostics["measured_vx"],
                    "measured_vy": diagnostics["measured_vy"],
                    "measured_wz": diagnostics["measured_wz"],
                    "base_x": pose["x"],
                    "base_y": pose["y"],
                    "base_z": pose["z"],
                    "base_yaw": pose["yaw"],
                    "base_roll": diagnostics["base_roll"],
                    "base_pitch": diagnostics["base_pitch"],
                    "action_abs_max": diagnostics.get("action_abs_max"),
                    "dog_action_abs_mean": diagnostics.get("dog_action_abs_mean"),
                    "done": done,
                }
                samples.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if global_step % max(1, args_cli.print_every) == 0:
                    print(
                        f"[tracking] t={benchmark_time_s:.2f}s cmd=({segment.vx:.2f},{segment.vy:.2f},{segment.wz:.2f}) "
                        f"meas=({row['measured_vx']:.2f},{row['measured_vy']:.2f},{row['measured_wz']:.2f})"
                    )
                benchmark_time_s += dt
                global_step += 1
                if args_cli.real_time:
                    time.sleep(max(0.0, dt - (time.perf_counter() - wall_start)))
                if done:
                    terminated_early = True
                    print(f"[benchmark] environment terminated during {segment.name}")
                    break
            if terminated_early:
                break

    metadata = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "task": args_cli.task,
        "profile": args_cli.profile,
        "seed": args_cli.seed,
        "terrain": "plane",
        "deterministic": True,
        "control_dt_s": dt,
        "policy_action_warmup_steps": args_cli.policy_action_warmup_steps,
        "standing_command_threshold": args_cli.standing_command_threshold,
        "terminated_early": terminated_early,
        "schedule": [segment.to_dict() for segment in schedule],
    }
    summary = write_benchmark_artifacts(output_dir, samples, metadata)
    print(f"[benchmark] report={output_dir / 'report.md'} pass_rate={summary['pass_rate']:.1%}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
