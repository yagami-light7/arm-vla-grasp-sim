"""允许 colcon build 前直接导入 supervisor 两个 Python 包。"""

from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# 允许在尚未重新 colcon build 时复用工作区内已经生成的 ROS 2 消息。
for generated_messages in sorted(
    (PROJECT_ROOT / "ros2_ws" / "install").glob(
        "*/local/lib/python*/dist-packages"
    )
):
    sys.path.insert(0, str(generated_messages))
