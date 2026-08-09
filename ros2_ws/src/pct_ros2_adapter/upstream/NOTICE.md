# PCT Planner 来源与许可证说明

生产 `upstream` backend 在运行时加载独立目录 `external/PCT_planner` 中的
PCT Planner。固定来源如下：

- 仓库：`https://github.com/byangw/PCT_planner`
- 提交：`35cd73fd82bcd51bc538429294af7646b2a09815`
- 源码归档 SHA256：`daf5f90b29c76cfa5fc6bf10d6dcfd200c1077778b22671c98aa51f9adb06d64`
- 上游许可证：GNU General Public License v2 或更高版本
- 上游版权：Copyright (c) 2024 Bowen Yang、Jie Cheng

本 ROS 2 adapter 没有把上游源码复制进自身 Python 包；运行时会核对固定提交
中五个核心文件的 SHA256，并在扩展缺失、Python ABI 不匹配、共享库不可加载或
source pin 不一致时失败关闭，不会切换到 `compatible` backend。

上游完整许可证和第三方声明分别保留在
`external/PCT_planner/LICENSE` 与 `external/PCT_planner/NOTICE`。重新分发源码、
二进制扩展或包含该依赖的镜像前，必须一并保留这些文件并复核 GPL 与第三方
依赖的分发义务。本说明不替代法律审查。
