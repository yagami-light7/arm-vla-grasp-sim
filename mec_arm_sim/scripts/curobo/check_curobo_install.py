import importlib.metadata as metadata
import importlib.util
import sys

print("=== Python 环境 ===")
print("python:", sys.executable)

print("\n=== 包导入检查 ===")
for name in ["curobo", "torch", "trimesh", "yaml", "packaging", "websockets"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", spec.origin if spec else "NOT_FOUND")

print("\n=== 版本检查 ===")
for dist in ["nvidia-curobo", "torch", "packaging", "websockets", "isaacsim-core", "isaacsim-kernel"]:
    try:
        print(f"{dist}:", metadata.version(dist))
    except Exception as exc:
        print(f"{dist}: unavailable ({exc})")

print("\n=== cuRobo API 检查 ===")
try:
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState, Pose

    print("新 MotionPlanner API: OK")
except Exception as exc:
    print("新 MotionPlanner API: FAIL")
    print(type(exc).__name__, exc)

try:
    from curobo.wrap.reacher.motion_gen import MotionGen

    print("旧 MotionGen API: OK")
except Exception as exc:
    print("旧 MotionGen API: 不可用，这是当前新版 cuRobo 的预期现象")
    print(type(exc).__name__, exc)

print("\n=== CUDA 检查 ===")
import torch

print("torch version:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
    x = torch.zeros((2, 2), device="cuda")
    print("cuda tensor:", x.device, x.shape)
else:
    print("CUDA 当前不可用：cuRobo MotionPlanner 还不能运行。")