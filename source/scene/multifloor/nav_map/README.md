# Multifloor 导航地图占位说明

PCT 多楼层导航主要依赖 tomogram 和 walkable map，不依赖旧 flat-world 2D A*
occupancy map。

当前 pipeline 的 DWA executor 仍需要一个 2D local map 执行避障。真实多楼层
执行前有两种选择：

1. 从 multifloor 场景生成临时 2D local map，作为 DWA 过渡执行器输入。
2. 后续将局部执行器升级为直接消费 PCT 3D path / slice 信息的 controller。

本目录当前不提交占位 `map.json`，避免误把不真实的 2D map 当作可用地图。

PLY 转 USD 只解决 Isaac Sim 场景加载问题，不会生成 PCT 需要的 tomogram。
请先用 `tools/scene/rebuild_multifloor_sage_assets.sh` 生成主场景 USDA，再用外部 PCT
建图脚本基于 `source/scene/multifloor/ply/3dgs_collision.ply` 生成 PCT 地图资产。

推荐 PCT 输出路径：

```text
source/scene/multifloor/mutifloor.pickle
source/scene/multifloor/mutifloor_ply_walkable.npy
```
