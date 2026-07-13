# Multifloor 导航地图占位说明

PCT 多楼层导航主要依赖 tomogram 和 walkable map，不依赖旧 flat-world 2D A*
occupancy map。

当前 pipeline 的 DWA executor 仍需要一个 2D local map 执行避障。真实多楼层
执行前有两种选择：

1. 从 multifloor 场景生成临时 2D local map，作为 DWA 过渡执行器输入。
2. 后续将局部执行器升级为直接消费 PCT 3D path / slice 信息的 controller。

本目录当前不提交占位 `map.json`，避免误把不真实的 2D map 当作可用地图。

当前仓库已提供 PCT 地图资产，运行 pipeline 不依赖 PLY 文件。重新建图属于离线维护流程，
不纳入默认部署步骤。

推荐 PCT 输出路径：

```text
source/scene/multifloor/mutifloor.pickle
source/scene/multifloor/mutifloor_ply_walkable.npy
```
