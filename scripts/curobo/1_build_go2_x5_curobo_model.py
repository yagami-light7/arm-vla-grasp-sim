#!/usr/bin/env python3
"""
调用 cuRobo 官方 CLI 生成 Go2-X5 机械臂的 yml / xrdf 机器人模型配置。

用途：
    本文件是项目级“官方 CLI 包装脚本”，不是重新实现 cuRobo RobotBuilder。

    它等价于手动运行：

    python -m curobo.examples.getting_started.build_robot_model \
        --urdf source/robot/go2_x5/curobo/go2_x5_arm.urdf \
        --asset-path source/robot/go2_x5 \
        --output source/robot/go2_x5/curobo/go2_x5_arm.yml \
        --export-xrdf \
        --tool-frames arm_eef_link \
        --sphere-density 1.0 \
        --num-collision-samples 1000 \
        --compute-metrics \
        --seed 42

为什么保留这种写法：
    1. 和 cuRobo 官方 build_robot_model 教程完全对齐。
    2. 生成逻辑仍由官方 CLI 负责，便于对照文档排错。
    3. 项目脚本只管理路径、参数、环境变量和输出位置。

输入：
    source/robot/go2_x5/curobo/go2_x5_arm.urdf

输出：
    source/robot/go2_x5/curobo/go2_x5_arm.yml
    source/robot/go2_x5/curobo/go2_x5_arm.xrdf

运行：
    cd /home/light/workspace/arm_vla

    PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
    /data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
    scripts/curobo/1_build_go2_x5_curobo_model.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path("/home/light/workspace/arm_vla")
CUROBO_SOURCE_ROOT = Path("/home/light/workspace/curobo")

DEFAULT_ARM_URDF = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.urdf"
DEFAULT_ASSET_PATH = WORKSPACE / "source/robot/go2_x5"
DEFAULT_OUTPUT_YML = WORKSPACE / "source/robot/go2_x5/curobo/go2_x5_arm.yml"

DEFAULT_TOOL_FRAME = "arm_eef_link"
DEFAULT_SPHERE_DENSITY = 1.0
DEFAULT_NUM_COLLISION_SAMPLES = 1000
DEFAULT_SEED = 42


def check_required_file(path: Path, description: str) -> None:
    """检查输入文件是否存在。"""
    if not path.exists():
        raise FileNotFoundError(f"{description} 不存在: {path}")


def check_required_dir(path: Path, description: str) -> None:
    """检查输入目录是否存在。"""
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{description} 不存在或不是目录: {path}")


def build_official_cli_command(args: argparse.Namespace) -> list[str]:
    """构造 cuRobo 官方 build_robot_model CLI 命令。"""
    command = [
        sys.executable,
        "-m",
        "curobo.examples.getting_started.build_robot_model",
        "--urdf",
        str(args.urdf),
        "--asset-path",
        str(args.asset_path),
        "--output",
        str(args.output_yml),
        "--export-xrdf",
        "--tool-frames",
        args.tool_frame,
        "--sphere-density",
        str(args.sphere_density),
        "--num-collision-samples",
        str(args.num_collision_samples),
        "--seed",
        str(args.seed),
    ]

    if args.compute_metrics:
        command.append("--compute-metrics")

    if args.visualize:
        command.append("--visualize")

    return command


def build_subprocess_env() -> dict[str, str]:
    """
    构造子进程环境变量。

    重点是把 /home/light/workspace/curobo 放到 PYTHONPATH 前面，
    让官方 CLI 使用当前本地 cuRobo 源码。
    """
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")

    if CUROBO_SOURCE_ROOT.exists():
        if old_pythonpath:
            env["PYTHONPATH"] = f"{CUROBO_SOURCE_ROOT}:{old_pythonpath}"
        else:
            env["PYTHONPATH"] = str(CUROBO_SOURCE_ROOT)

    return env


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Call official cuRobo build_robot_model CLI for Go2-X5 arm.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_ARM_URDF,
        help="arm-only URDF 路径。",
    )
    parser.add_argument(
        "--asset-path",
        type=Path,
        default=DEFAULT_ASSET_PATH,
        help="mesh asset 根目录。URDF 中的 ./meshes/... 会相对这个目录解析。",
    )
    parser.add_argument(
        "--output-yml",
        type=Path,
        default=DEFAULT_OUTPUT_YML,
        help="输出 cuRobo YAML 路径。XRDF 会由官方 CLI 自动输出到同名 .xrdf。",
    )
    parser.add_argument(
        "--tool-frame",
        default=DEFAULT_TOOL_FRAME,
        help="cuRobo tool frame 名称。",
    )
    parser.add_argument(
        "--sphere-density",
        type=float,
        default=DEFAULT_SPHERE_DENSITY,
        help="碰撞球密度。越大 sphere 越多，碰撞近似更细但更慢。",
    )
    parser.add_argument(
        "--num-collision-samples",
        type=int,
        default=DEFAULT_NUM_COLLISION_SAMPLES,
        help="生成 self-collision ignore matrix 的随机采样数量。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="随机种子，保证 sphere fitting / collision sampling 尽量可复现。",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        default=True,
        help="打印每个 link 的 sphere fitting 指标。",
    )
    parser.add_argument(
        "--no-compute-metrics",
        action="store_false",
        dest="compute_metrics",
        help="不打印 sphere fitting 指标。",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="打开 cuRobo 官方 viser 可视化服务。默认关闭，避免脚本一直阻塞。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()

    check_required_file(args.urdf, "arm-only URDF")
    check_required_dir(args.asset_path, "asset path")
    args.output_yml.parent.mkdir(parents=True, exist_ok=True)

    command = build_official_cli_command(args)
    env = build_subprocess_env()

    print("========== Call Official cuRobo build_robot_model ==========")
    print("command:")
    print(" ".join(command))
    print()
    print(f"PYTHONPATH={env.get('PYTHONPATH', '')}")
    print("============================================================")

    subprocess.run(command, env=env, check=True)

    output_xrdf = args.output_yml.with_suffix(".xrdf")
    print()
    print("========== Go2-X5 cuRobo Model Build Finished ==========")
    print(f"YAML: {args.output_yml}")
    print(f"XRDF: {output_xrdf}")
    print("下一步请检查 YAML 中的 base_link / tool_frames / joint_names。")


if __name__ == "__main__":
    main()
