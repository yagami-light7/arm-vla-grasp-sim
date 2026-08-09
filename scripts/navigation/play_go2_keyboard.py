"""使用键盘速度指令控制 Isaac Lab 官方 Go2 locomotion policy。"""

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher


# ----------------------------------------------------------------------
# 必须先启动 Isaac Sim，之后才能导入 torch、Omniverse 和 Isaac Lab 环境模块
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(description="键盘控制 Isaac Lab 官方 Go2 policy")

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="可选的 checkpoint 路径；不填写时使用 Isaac Lab 官方 checkpoint",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    help="尽量按照真实时间运行，方便键盘操作",
)
parser.add_argument(
    "--vx",
    type=float,
    default=0.6,
    help="前后移动速度，单位 m/s",
)
parser.add_argument(
    "--vy",
    type=float,
    default=0.3,
    help="横向移动速度，单位 m/s",
)
parser.add_argument(
    "--wz",
    type=float,
    default=0.8,
    help="转向角速度，单位 rad/s",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ----------------------------------------------------------------------
# Isaac Sim 启动之后再导入这些模块
# ----------------------------------------------------------------------

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_published_pretrained_checkpoint,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg_PLAY,
)


TRAIN_TASK = "Isaac-Velocity-Rough-Unitree-Go2-v0"


def resolve_checkpoint() -> str:
    """返回需要加载的 checkpoint 路径。"""

    if args_cli.checkpoint is not None:
        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()

        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")

        return str(checkpoint)

    checkpoint = get_published_pretrained_checkpoint(
        "rsl_rl",
        TRAIN_TASK,
    )

    if checkpoint is None:
        raise RuntimeError("没有找到 Isaac Lab 官方 Go2 checkpoint")

    return checkpoint


def create_environment() -> tuple[RslRlVecEnvWrapper, UnitreeGo2RoughPPORunnerCfg]:
    """创建单机器狗的 Go2 rough-terrain 推理环境。"""

    env_cfg = UnitreeGo2RoughEnvCfg_PLAY()
    agent_cfg = UnitreeGo2RoughPPORunnerCfg()

    # 键盘只控制一条机器狗。
    env_cfg.scene.num_envs = 1

    # 使用较容易的地形等级，先验证手动控制链路。
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # 避免运行一段时间后因为 episode 超时而自动重置。
    # 如果机器人摔倒，base_contact 等安全终止条件仍然有效。
    env_cfg.episode_length_s = 1_000_000.0
    env_cfg.curriculum = None

    # 固定初始朝向，方便判断“前进”方向。
    pose_range = env_cfg.events.reset_base.params["pose_range"]
    pose_range["x"] = (0.0, 0.0)
    pose_range["y"] = (0.0, 0.0)
    pose_range["yaw"] = (0.0, 0.0)

    command_cfg = env_cfg.commands.base_velocity

    # 关闭环境原本的随机速度指令。
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False

    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.ranges.heading = None

    # 仿真和 policy 使用同一块设备。
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    raw_env = ManagerBasedRLEnv(cfg=env_cfg)

    env = RslRlVecEnvWrapper(
        raw_env,
        clip_actions=agent_cfg.clip_actions,
    )

    return env, agent_cfg


def main() -> None:
    """运行键盘控制循环。"""

    env, agent_cfg = create_environment()
    checkpoint = resolve_checkpoint()

    print(f"[INFO] 加载 checkpoint：{checkpoint}")

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(checkpoint)

    # 获得只用于推理的 policy 函数。
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_network = runner.alg.policy

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

    # 这是 command manager 内部真正被 policy 读取的三维速度张量。
    # 形状为 [num_envs, 3]，当前就是 [1, 3]。
    velocity_command = (
        env.unwrapped.command_manager.get_command("base_velocity")
    )

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # 1. 从键盘获得 [vx, vy, wz]。
                keyboard_command = keyboard.advance()

                # 2. 写入 command manager。
                #    keyboard_command 是 [3]，会广播到 [1, 3]。
                velocity_command[:] = keyboard_command

                # 3. 写入速度后重新生成观测。
                #    这样 policy 当前帧读到的就是键盘命令。
                observations = env.get_observations()

                # 4. RL policy 将速度指令转换成关节动作。
                actions = policy(observations)

                # 5. 将关节动作交给物理仿真。
                _, _, dones, _ = env.step(actions)

                # 机器人摔倒或环境重置时，清理 policy 的内部状态。
                policy_network.reset(dones)

            if args_cli.real_time:
                remaining_time = env.unwrapped.step_dt - (
                    time.time() - start_time
                )
                if remaining_time > 0.0:
                    time.sleep(remaining_time)

    finally:
        # 退出前显式清零速度。
        velocity_command.zero_()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()