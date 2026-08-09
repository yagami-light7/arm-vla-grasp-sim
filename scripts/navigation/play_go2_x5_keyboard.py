"""在确定性楼梯上用键盘测试仓库内的 Go2-X5 DogOnly policy。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_LAB_SOURCE = PROJECT_ROOT / "source" / "robot_lab"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "go2_x5"
    / "pct_multifloor"
    / "model_26000.pt"
)
EXPECTED_POLICY_OBSERVATION_DIM = 260
EXPECTED_POLICY_ACTION_DIM = 12


# ----------------------------------------------------------------------
# 必须先启动 Isaac Sim，之后才能导入 torch、Omniverse 和 Isaac Lab 环境模块。
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="键盘测试仓库内的 Go2-X5 DogOnly 楼梯 locomotion policy"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=str(DEFAULT_CHECKPOINT),
    help="Go2-X5 DogOnly RSL-RL checkpoint 路径",
)
parser.add_argument(
    "--stair-height",
    type=float,
    default=0.10,
    help="固定测试楼梯的台阶高度，单位 m",
)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=50,
    help="reset 后把 policy action 从零渐入的控制步数",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="最多运行的控制步数；0 表示一直运行，主要用于有限步 smoke test",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    help="尽量按照真实时间运行，方便键盘操作",
)
parser.add_argument(
    "--vx",
    type=float,
    default=0.30,
    help="前后移动速度，单位 m/s",
)
parser.add_argument(
    "--vy",
    type=float,
    default=0.0,
    help="横向移动速度，单位 m/s",
)
parser.add_argument(
    "--wz",
    type=float,
    default=0.40,
    help="转向角速度，单位 rad/s",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.stair_height <= 0.0:
    parser.error("--stair-height 必须大于 0")
if args_cli.warmup_steps < 0:
    parser.error("--warmup-steps 不能为负数")
if args_cli.max_steps < 0:
    parser.error("--max-steps 不能为负数")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ----------------------------------------------------------------------
# Isaac Sim 启动之后再导入这些模块。
# ----------------------------------------------------------------------

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

if str(ROBOT_LAB_SOURCE) not in sys.path:
    sys.path.insert(0, str(ROBOT_LAB_SOURCE))

import robot_lab.tasks  # noqa: E402, F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.go2_x5.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    Go2X5DogOnlyRoughPPORunnerCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.go2_x5.train_route_env_cfg import (  # noqa: E402
    Go2X5DogOnlyRoughEnvCfg,
)


def resolve_checkpoint() -> str:
    """检查并返回 Go2-X5 checkpoint 的绝对路径。"""

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")
    return str(checkpoint)


def _disable_evaluation_randomization(env_cfg: Go2X5DogOnlyRoughEnvCfg) -> None:
    """关闭训练期随机化，使不同 checkpoint 的楼梯结果可重复比较。"""

    env_cfg.observations.policy.enable_corruption = False

    for event_name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_push_robot",
        "randomize_actuator_gains",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    # 固定机器人初始位置、姿态和速度。
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    env_cfg.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)

    for curriculum_name in (
        "terrain_levels",
        "command_levels_lin_vel",
        "command_levels_ang_vel",
        "reward_weights",
    ):
        if hasattr(env_cfg.curriculum, curriculum_name):
            setattr(env_cfg.curriculum, curriculum_name, None)


def _configure_deterministic_stairs(env_cfg: Go2X5DogOnlyRoughEnvCfg) -> None:
    """只生成一块固定尺寸的上楼梯地形。"""

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise RuntimeError("Go2-X5 环境没有启用 terrain generator")

    # inverted stairs 的中心是低平台；机器人从中心向外走时会上楼。
    stair_cfg = terrain_generator.sub_terrains["pyramid_stairs_inv"]
    stair_cfg.proportion = 1.0
    stair_cfg.step_height_range = (
        args_cli.stair_height,
        args_cli.stair_height,
    )
    stair_cfg.step_width = 0.30

    terrain_generator.sub_terrains = {"pyramid_stairs_inv": stair_cfg}
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    terrain_generator.curriculum = False
    terrain_generator.difficulty_range = (0.0, 0.0)
    env_cfg.scene.terrain.max_init_terrain_level = 0


def create_environment() -> tuple[
    RslRlVecEnvWrapper,
    Go2X5DogOnlyRoughPPORunnerCfg,
]:
    """创建匹配 model_26000.pt 的单机器人确定性楼梯环境。"""

    env_cfg = Go2X5DogOnlyRoughEnvCfg()
    agent_cfg = Go2X5DogOnlyRoughPPORunnerCfg()

    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1_000_000.0

    _configure_deterministic_stairs(env_cfg)
    _disable_evaluation_randomization(env_cfg)

    # 禁止环境随机生成速度，运行时每个控制周期由键盘覆写 command buffer。
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.debug_vis = True
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.ranges.heading = None

    # 保留跌倒后的物理状态用于观察，只关闭 episode 时间到期。
    env_cfg.terminations.time_out = None

    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    raw_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(
        raw_env,
        clip_actions=agent_cfg.clip_actions,
    )
    return env, agent_cfg


def _verify_policy_interface(env: RslRlVecEnvWrapper) -> None:
    """在加载 checkpoint 前验证观测和 action 接口，避免静默错配。"""

    observations = env.get_observations()
    policy_observation_dim = int(observations["policy"].shape[-1])
    policy_action_dim = int(env.unwrapped.action_manager.total_action_dim)

    if policy_observation_dim != EXPECTED_POLICY_OBSERVATION_DIM:
        raise RuntimeError(
            "Go2-X5 policy 观测维度不匹配："
            f"期望 {EXPECTED_POLICY_OBSERVATION_DIM}，实际 {policy_observation_dim}"
        )
    if policy_action_dim != EXPECTED_POLICY_ACTION_DIM:
        raise RuntimeError(
            "Go2-X5 policy action 维度不匹配："
            f"期望 {EXPECTED_POLICY_ACTION_DIM}，实际 {policy_action_dim}"
        )

    print(
        "[INFO] policy 接口验证通过："
        f"observation={policy_observation_dim}, action={policy_action_dim}"
    )


def main() -> None:
    """加载 checkpoint 并运行键盘控制循环。"""

    env, agent_cfg = create_environment()
    _verify_policy_interface(env)
    checkpoint = resolve_checkpoint()
    print(f"[INFO] 加载 checkpoint：{checkpoint}")

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(checkpoint, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_network = runner.alg.policy

    # headless 模式没有窗口键盘，只用于有限步启动检查。
    keyboard = None
    if not args_cli.headless:
        keyboard = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=args_cli.vx,
                v_y_sensitivity=args_cli.vy,
                omega_z_sensitivity=args_cli.wz,
                sim_device=env.unwrapped.device,
            )
        )
        print(keyboard)
        print("[INFO] 请点击 Isaac Sim 的 Viewport，然后按住控制键。")
        print("[INFO] 如果命令没有归零，按 L 紧急清零。")
    else:
        print("[INFO] headless 模式使用零键盘命令，仅用于启动 smoke test。")

    velocity_command = env.unwrapped.command_manager.get_command("base_velocity")
    zero_keyboard_command = torch.zeros(3, device=env.unwrapped.device)
    warmup_step = 0
    completed_steps = 0

    print(
        f"[INFO] 前 {args_cli.warmup_steps} 个控制步保持零速度并渐入 policy action。"
    )

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                keyboard_command = (
                    zero_keyboard_command
                    if keyboard is None
                    else keyboard.advance()
                )
                in_warmup = warmup_step < args_cli.warmup_steps

                if in_warmup:
                    velocity_command.zero_()
                else:
                    velocity_command[:] = keyboard_command

                # 必须在写入速度之后重新计算观测，policy 才能读取当前键盘命令。
                observations = env.get_observations()
                actions = policy(observations)

                if in_warmup and args_cli.warmup_steps > 0:
                    warmup_scale = min(
                        1.0,
                        float(warmup_step + 1) / float(args_cli.warmup_steps),
                    )
                    actions = actions * warmup_scale

                _, _, dones, _ = env.step(actions)
                policy_network.reset(dones)

                if bool(torch.any(dones).item()):
                    warmup_step = 0
                    if keyboard is not None:
                        keyboard.reset()
                else:
                    warmup_step += 1

            completed_steps += 1
            if args_cli.max_steps > 0 and completed_steps >= args_cli.max_steps:
                print(f"[INFO] 已完成有限步测试：{completed_steps} steps")
                break

            if args_cli.real_time:
                remaining_time = env.unwrapped.step_dt - (
                    time.time() - start_time
                )
                if remaining_time > 0.0:
                    time.sleep(remaining_time)

    finally:
        velocity_command.zero_()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
