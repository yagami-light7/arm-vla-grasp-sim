"""
验证生成的机器人模型是否正确
mec_arm_from_urdf.yml
-> cuRobo 能加载
-> joint_names 正确
-> tool_frames 正确
-> default q 能做 FK
-> TCP_link 位姿能被算出来
-> collision spheres 能被生成
"""

from pathlib import Path
import sys
import torch

# 当前项目路径
WORKSPACE = Path("/home/light/workspace/arm_vla")

# cuRobo路径
CUROBO_SOURCE_ROOT = Path("/home/light/workspace/curobo")

# yml路径
ROBOT_CONFIG = WORKSPACE / "mec_arm_sim/configs/curobo/mec_arm_from_urdf.yml"

# Isaac Sim DOF顺序
EXPECTED_JOINT_ORDER = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]

# TCP frame name
EXPECTED_TOOL_FRAME = "TCP_link"

# 把本地 cuRobo 源码放到 Python 搜索路径最前面
# 这样 import curobo 时，会优先使用 /home/light/workspace/curobo 里的源码。
if CUROBO_SOURCE_ROOT.exists():
    sys.path.insert(0, str(CUROBO_SOURCE_ROOT))

# MotionPlannerCfg cuRobo 的规划器配置对象。它负责读取 robot yml、world yml、IK 配置、trajopt 配置等。

# MotionPlanner cuRobo 的主规划器对象。后面做 FK、IK、plan_pose 都从它进入。

# JointState cuRobo 的关节状态类型。它不是普通 list，而是带 joint_names、position、velocity 等信息的结构。
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState


def print_header(title:str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def tensor_to_list(x):
    # 把 PyTorch tensor 转成普通的 Python list，方便打印和比较。
    return x.detach().cpu().tolist()

def check_file_exists(path: Path):
    # 检查文件是否存在
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    

# 创建规划器
def create_planner():
    print_header("1. 创建 MotionPlannerCfg")

    check_file_exists(ROBOT_CONFIG)

    print("robot config:", ROBOT_CONFIG)
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError("CUDA 不可用。cuRobo 需要 CUDA 才能正常运行。")
    
    cfg = MotionPlannerCfg.create(
        robot = str(ROBOT_CONFIG), # 机器人配置文件
        scene_model = None, # 不加载场景模型
        self_collision_check = True, # 启动自碰撞检查
        use_cuda_graph = False, # 关闭cuda graph
        num_ik_seeds = 4, # IK 种子数量
        num_trajopt_seeds = 1, # 轨迹优化种子数量
    )

    print("MotionPlannerCfg 创建成功")

    print_header("2. 创建 MotionPlanner")
    planner = MotionPlanner(cfg)
    print("MotionPlanner 创建成功")

    return planner


# 检查关节名称和TCP frame
def check_robot_metadata(planner: MotionPlanner):
    print_header("3. 检查 robot metadata")

    joint_names = list(planner.joint_names)
    tool_frames = list(planner.tool_frames)

    print("planner.joint_names:", joint_names)
    print("planner.tool_frames:", tool_frames)

    if joint_names != EXPECTED_JOINT_ORDER:
        raise RuntimeError(
            f"joint order 不一致:\n"
            f"  planner:  {joint_names}\n"
            f"  expected: {EXPECTED_JOINT_ORDER}"
        )

    if EXPECTED_TOOL_FRAME not in tool_frames:
        raise RuntimeError(
            f"没有找到 TCP tool frame: {EXPECTED_TOOL_FRAME}, 当前 tool_frames={tool_frames}"
        )

    print("joint order 检查通过")
    print("tool frame 检查通过")


# 读取joint_state
def get_default_joint_state(planner:MotionPlanner) -> JointState:
    print_header("4. 获取default joint state")

    default_joint_state = planner.default_joint_state # 来自 yml 中 cspace.default_joint_position
    q_default = default_joint_state.position.detach().clone()

    print("raw default_joint_state.position shape:", tuple(q_default.shape))
    print("raw default_joint_state.position:", tensor_to_list(q_default))

    # cuRobo FK 期望 position 形状是 [batch, dof]
    # 如果读出来是 [dof]，就手动加 batch 维度变成 [1, dof]。
    if q_default.ndim == 1:
        q_default = q_default.unsqueeze(0)

    joint_state = JointState.from_position(
        position=q_default,
        joint_names=list(planner.joint_names),
    )

    print("用于 FK 的 q shape:", tuple(joint_state.position.shape))
    print("用于 FK 的 q:", tensor_to_list(joint_state.position))

    return joint_state

# FK
def run_fk(planner:MotionPlanner, joint_state:JointState):
    print_header("5. 运行 cuRobo FK")

    kin_state = planner.compute_kinematics(joint_state)

    print("kin_state.tool_frames:", kin_state.tool_frames)

    #kin_state.tool_poses 保存所有 tool frame 的位姿 从tool_frames里找 EXPECTED_TOOL_FRAME 的位姿
    tcp_pose = kin_state.tool_poses.get_link_pose(
        EXPECTED_TOOL_FRAME,
        make_contiguous=True,
    )

    # tcp_pose 包含 position 和 quaternion，都是 tensor，把它们转成 numpy array 打印出来
    tcp_position = tcp_pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
    tcp_quaternion = tcp_pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]

    print(f"{EXPECTED_TOOL_FRAME} position in base_link frame:")
    print(tcp_position)

    print(f"{EXPECTED_TOOL_FRAME} quaternion in base_link frame, wxyz:")
    print(tcp_quaternion)

    # 当前 q 下，所有 collision spheres 在 base_link 下的位置和半径
    if kin_state.robot_spheres is not None:
        spheres = kin_state.robot_spheres.detach()
        print("robot_spheres shape:", tuple(spheres.shape))
        print("robot_spheres 含义: [batch, horizon, num_spheres, 4], 最后一维是 x,y,z,r")
    else:
        print("robot_spheres: None")

    return tcp_position, tcp_quaternion


def main():
    print_header("cuRobo 生成机器人模型检查")

    planner = None

    try:
        planner = create_planner()
        check_robot_metadata(planner)
        joint_state = get_default_joint_state(planner)
        run_fk(planner, joint_state)

        print_header("检查通过")
        print("mec_arm_from_urdf.yml 可以被 cuRobo 加载，并且可以计算 TCP_link FK。")

    finally:
        # MotionPlanner 内部可能持有 CUDA / collision / optimizer 资源。
        # 显式 destroy 可以减少反复调试时的 CUDA 资源残留。
        if planner is not None:
            planner.destroy()


if __name__ == "__main__":
    main()