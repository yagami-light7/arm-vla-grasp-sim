"""保存跨进程生产导航必须共享的轻量合同常量。"""

from __future__ import annotations


# ROS 组合 launch 与 host pipeline 都从各自 CLI 接收本值；这里仅定义
# 两个 CLI 未显式覆盖时共同使用的默认值，避免默认配置再次发生漂移。
# Go2-X5 携臂收纳姿态在 multi_floor collision PLY 上的 51 帧静止实测
# 中位数为 0.337957 m；统一调参 YAML 是主运行链的首选来源，本值仅作为
# 不经过该入口时的安全默认值。
DEFAULT_NAVIGATION_BODY_HEIGHT_M = 0.338


__all__ = ["DEFAULT_NAVIGATION_BODY_HEIGHT_M"]
