"""兼容旧导入路径的 PCT + SCAN 导航状态机入口。"""

try:
    # 已 source ROS workspace 时优先使用正式安装包，避免依赖当前 cwd。
    from navigation_core.supervisor import (
        NavigationState,
        NavigationSupervisor,
        NavigationSupervisorConfig,
        SupervisorDecision,
        ZERO_BODY_VELOCITY,
    )
except ModuleNotFoundError:
    # 仓库内轻量 pytest 尚未 colcon build 时使用同一源码文件；这里不是
    # 第二份实现，只是兼容原有 source.navigation 导入路径。
    from ros2_ws.src.navigation_supervisor.navigation_core.supervisor import (
        NavigationState,
        NavigationSupervisor,
        NavigationSupervisorConfig,
        SupervisorDecision,
        ZERO_BODY_VELOCITY,
    )

__all__ = [
    "NavigationState",
    "NavigationSupervisor",
    "NavigationSupervisorConfig",
    "SupervisorDecision",
    "ZERO_BODY_VELOCITY",
]
