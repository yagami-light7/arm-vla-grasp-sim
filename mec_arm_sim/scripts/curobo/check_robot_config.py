from pathlib import Path

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg


ROBOT_CONFIG = Path("/home/light/workspace/arm_vla/mec_arm_sim/configs/curobo/mec_arm.yml")
WORLD_CONFIG = Path(
    "/home/light/workspace/arm_vla/mec_arm_sim/configs/curobo/worlds/smoke_cuboid.yml"
)

EXPECTED_JOINT_ORDER = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
EXPECTED_TOOL_FRAME = "TCP_link"


def main():
    print("=== cuRobo Robot Config 检查 ===")
    print("robot config:", ROBOT_CONFIG)
    print("world config:", WORLD_CONFIG)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device 0:", torch.cuda.get_device_name(0))

    if not ROBOT_CONFIG.exists():
        raise FileNotFoundError(f"robot config 不存在: {ROBOT_CONFIG}")
    if not WORLD_CONFIG.exists():
        raise FileNotFoundError(f"world config 不存在: {WORLD_CONFIG}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 当前不可用，cuRobo MotionPlanner 不能运行。")

    cfg = MotionPlannerCfg.create(
        robot=str(ROBOT_CONFIG),
        scene_model=str(WORLD_CONFIG),
        use_cuda_graph=False,
        num_ik_seeds=4,
        num_trajopt_seeds=1,
    )

    print("\n=== MotionPlannerCfg 加载成功 ===")

    planner = MotionPlanner(cfg)
    print("=== MotionPlanner 创建成功 ===")

    joint_names = list(planner.joint_names)
    tool_frames = list(planner.tool_frames)
    default_joint_position = planner.default_joint_state.position.detach().cpu().tolist()

    print("\n=== Planner 信息 ===")
    print("joint_names:", joint_names)
    print("tool_frames:", tool_frames)
    print("default_joint_state.position:", default_joint_position)

    assert joint_names == EXPECTED_JOINT_ORDER, (
        f"joint order 不一致: {joint_names} != {EXPECTED_JOINT_ORDER}"
    )
    assert EXPECTED_TOOL_FRAME in tool_frames, (
        f"没有找到 TCP tool frame: {EXPECTED_TOOL_FRAME}, planner.tool_frames={tool_frames}"
    )

    print("\n=== 检查通过 ===")
    print("cuRobo 可以加载 mec_arm robot config 和 smoke cuboid world。")


if __name__ == "__main__":
    main()
