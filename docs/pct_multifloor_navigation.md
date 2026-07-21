# PCT 多楼层导航

## 架构

```text
task JSON
-> EpisodeSpec
-> PCTNavPlanner
-> PCT server
-> NavPlan
-> DwaNavExecutor / RL policy
-> Isaac Sim
```

`PCTNavPlanner` 是全局规划器 adapter，不替换 pipeline 状态机。它通过
stdin/stdout JSON 协议调用本仓库内迁移的 PCT server，把返回的 3D 轨迹转换回 Isaac
Sim 世界坐标，写入 `NavPlan.metadata["path_3d"]`，并继续向当前
`DwaNavExecutor` 暴露兼容旧执行器的 XY waypoints。

## 为什么替换 A*

当前 A* 基于 2D occupancy grid，适合 flat single-floor 场景。PCT 基于
tomogram slice 表达地图，可以描述多楼层路线、楼梯、坡道和上下重叠结构。

PCT 只负责多楼层全局路径。DWA 或 locomotion RL policy 仍负责局部执行。
真实楼梯、坡道和跨楼层 traversal 需要训练好的 `pct_multifloor` policy。

## 本地资产约定

大型资产和 checkpoint 不提交到 git。本地目录约定如下：

- `checkpoints/go2_x5/pct_multifloor/`：本地 RL policy checkpoint、导出 policy 和训练参数，均为实体文件。
- `source/scene/multifloor/ply/`：真实 PLY 源文件，不使用软链接。
- `source/scene/multifloor/usdz/`：真实 NuRec/3DGS visual USDZ。
- `source/scene/multifloor/usd/`：真实 collision USD。
- `source/scene/multifloor/usda/`：Isaac Sim 主场景 USDA。

最终 Isaac Sim 应打开：

```text
source/scene/multifloor/usda/multifloor.usda
```

当前 `multifloor.usda` 是唯一保留的高质量 Isaac Sim 场景入口：`/World/gauss`
引用 clean NuRec visual，`/World/scene_collision` 引用隐藏的 collision mesh。两层都按
PCT README 场景约定使用 Z 轴 180 度旋转，避免 visual 和 collision 相对错位。

当前 `multifloor.usdz` 由原始 `3dgs_visual.ply` 裁剪清理后导出：按 collision mesh
bounds 裁掉离群点，过滤 `exp(max(scale_*)) > 0.5m` 的超大 splat，并保留
`opacity >= 0.2` 的 Gaussian。最终版本保留约 1676 万个 Gaussian，`.usdz` 约
1.98GB，低于 2GiB，避免早期全量 24.5M 版本包体过大导致的 USDZ 包成员解析问题。

当前 PLY 命名按实际用途整理为：

```text
source/scene/multifloor/ply/3dgs_collision.ply   碰撞 mesh，包含 face
source/scene/multifloor/ply/3dgs_visual.ply      3DGS Gaussian PLY，无 face
```

推荐 checkpoint 引用方式：

```bash
--locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt
```

当前 `model_26000.pt` 是 dog-only rough policy：RSL-RL policy 输出 12 维腿部
action，机械臂通过命令 term 固定姿态跟随。运行 CLI 时只要传
`--policy-profile pct_multifloor`，pipeline 会默认使用本仓库注册的
`RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0`。不要把该 checkpoint 加载到
18 维 arm-locomotion task，否则会出现 `std` / `actor.6` / `critic.0` size mismatch。
多楼层任务会把高度扫描射线限制在当前层高以内，避免命中上层楼板；DWA 命令也会
限制在该 checkpoint 的训练范围内。两项修改仅作用于 `pct_multifloor` profile，
原单楼层 profile 保持原配置。

`pct_multifloor` 初始化时会在前 50 个控制步把 policy action 从零平滑渐入，避免
默认关节姿态切换到 policy target 时产生底盘弹跳。nav-to-place 携物阶段使用独立的
保守控制上限：默认 `vx<=0.30 m/s`、`|wz|<=0.35 rad/s`。全局 server 会读取
collision PLY 的三角面，用顶点、边中点和面中心生成逐 slice 的机器人身体净空障碍体。
相对地面 `0.30-1.00 m` 内的墙面、桌椅和其他家具都会参与规划，不再只识别贯穿多个
高度切片的高墙。距离障碍 `0.60 m` 内还会增加软代价，使路径远离家具，同时避免硬
膨胀封死真实门洞。DWA 局部地图使用相同身体净空语义，并只在已通过 3D 校验的 PCT
中心线附近清理 `0.16 m` 走廊。

Yinluyuan 当前楼梯入口是 Isaac Sim 坐标 `(1.5, 5.7, 0.6)`，楼梯平台/拐角是
`(1.9202, 9.52807, 1.71919)`，上层出口是 `(1.9, 8.0, 3.0)`。PCT server 会按
入口 -> 拐角 -> 出口生成半宽 `0.60 m` 的折线 stair corridor，而不是用入口到出口的
直线段。每升高一个 `0.50 m` slice 必须水平移动 `0.40-0.90 m`，因此路径不能在楼梯井内
垂直穿楼板。
真正发生跨 slice 的边还会被限制到更窄的楼梯中心带内，默认半径为 `0.30 m`，并要求
当前高度 slice 与入口到出口的归一化进度相匹配，默认容差为 `0.12`。这样黄色预览线会
保留每个 `0.50 m` PCT slice 的上楼节点，避免从扶手或楼梯井边缘快速斜跨到上层。
task 的 z 是机器人 root 高度；server 默认减去 `0.45 m` 后再选择地面 slice，
避免到达 F2 房间后额外升层。可用
`--pct-cross-floor-gateway x,y,z` 指定楼梯/坡道入口；重复传入可配置多个候选点，传
`--pct-cross-floor-stair-exit x,y,z` 指定对应上层出口，
`--pct-cross-floor-stair-midpoint x,y,z` 指定中间平台/拐角。`--pct-stair-vertical-radius`
控制换层中心带宽度，`--pct-stair-progress-tolerance` 控制高度与楼梯进度的匹配容差。
严格 corridor 路线失败时返回
`no_path`，不会再退回可能穿墙的 relaxed 三维搜索。成功路线应记录
`cross_floor_gateway_mode="strict_monotonic"` 和
`hard_obstacle_mode="body_clearance_volume"`。

当前本地 `source/scene/multifloor/` 已生成 PCT 建图结果；运行时可以显式把对应文件
传给：

```bash
--pct-tomogram-path source/scene/multifloor/mutifloor.pickle
--pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy
--pct-collision-ply-path source/scene/multifloor/ply/3dgs_collision.ply
```

本仓库已经迁移出一套可复现的 PCT-compatible 离线建图和探针链路。运行时默认使用
本仓库内脚本，不直接调用 `external/PCT` 的 API：

```bash
/data/conda_envs/sage/bin/python scripts/navigation/build_pct_multifloor_assets.py \
  --collision-ply source/scene/multifloor/ply/3dgs_collision.ply \
  --output-tomogram source/scene/multifloor/mutifloor.pickle \
  --output-walkable source/scene/multifloor/mutifloor_ply_walkable.npy \
  --report-output outputs/pct_multifloor_asset_build_report.json
```

该脚本输出 PCT-compatible `mutifloor.pickle` 和 `mutifloor_ply_walkable.npy`。本地
portable server 使用同样的 stdin/stdout JSON 协议：

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python scripts/navigation/probe_pct_plan.py \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-server-python /data/conda_envs/sage/bin/python \
  --pct-tomogram-path source/scene/multifloor/mutifloor.pickle \
  --pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy \
  --output-json outputs/pct_plan_probe.json
```

`pct_grid_server.py` 是本仓库内迁移的 PCT-compatible grid server，用于验证
NavPlanner adapter、tomogram/walkable 坐标和多楼层 path metadata。`external/PCT`
仅作为参考 codebase；如果后续需要上游 C++ `planner_wrapper` 的能力，应先把对应
backend 迁移或封装进本仓库本地脚本，再接入 pipeline。

PCT navigation smoke 中，`DwaNavExecutor` 仍然消费 XY waypoint。为了让局部执行器和
PCT 全局规划使用一致的地图语义，pipeline 会从 tomogram traversability、
`mutifloor_ply_walkable.npy` 和 collision mesh 身体净空体合并生成 DWA local map。
root z 会先减去 `0.45 m` 再选择 floor slice。不要只用
`walkable.npy` 的单一 slice 替代该 local map，否则 PCT 返回的路径点可能在 DWA 中被误判
为障碍，表现为机器狗顶住障碍物后原地旋转或滑动逃离。

相邻一个 slice 通常只是 base z 与 0.5 m 切片边界的量化差异，不代表跨楼层。同层规划会先
检查 start 到 goal 的机器人宽度走廊；整条走廊在相邻 slice 中均可走时，PCT server 返回
`path_mode=same_floor_direct`，executor 使用两点直连轨迹。只有 `path_3d` 的 z 跨度超过
0.35 m，或缺少 3D path 时 slice 相差至少 2，才按跨楼层路径处理。

旧 `--pct-vertical-obstacle-min-slices` 仍保留用于兼容诊断，但 PCT pipeline 默认
使用逐 floor slice 的身体净空体，不再把所有高度压成同一张二维硬墙图。

task JSON 中的 `pick.base_goal` / `place.base_goal` 必须是机器人底盘站位，不是苹果中心或
place object 中心。苹果自身不属于 PCT 静态建图资产；PCT smoke 会把
`pick.object_pose_world` 投影为一个小的 DWA keepout，避免底盘把苹果当作可穿越空间。

一键重建 SAGE-3D / Isaac Sim 资产：

```bash
conda activate /data/conda_envs/sage
cd ~/workspace/arm_vla_pct
bash tools/scene/rebuild_multifloor_sage_assets.sh
```

该脚本会依次执行：

```text
tools/scene/check_multifloor_ply.py
tools/scene/sample_gaussian_ply.py --min-opacity 0.2 --max-scale 0.5
python -m threedgrut.export.scripts.ply_to_usd
tools/scene/build_multifloor_collision_usd.py
tools/scene/build_multifloor_usda.py --visual-mode nurec
tools/scene/validate_multifloor_usd_assets.py
```

如果打开 `multifloor.usda` 时 Isaac Sim 中已经存在 `/World/gauss`，但渲染报
`HydraEngine::render failed to end the compute graph`，不要继续改资产路径。这说明
composition 已经进入 NuRec/RTX 渲染阶段，下一步应排查 Isaac Sim 渲染设置、NuRec
扩展、驱动和 RTX 模式。

## 资产目录

当前 `source/scene/` 只保留脚本和三个资产目录：

```text
source/scene/839920/       单楼层 baseline 场景、USDZ、collision USD 和 nav_map
source/scene/multifloor/   多楼层 PLY、NuRec USDZ、collision USD、主 USDA 和 PCT 地图占位
source/scene/objects/      apple / orange / bottle 等可操作物体资产
```

`source/scene/839920/839920_go2_x5.usd` 已改为只引用当前仓库内的相对路径，不再依赖
`/mnt/sage_data` 或其他 worktree。现有 flat-scene task 仍默认使用 839920 单楼层场景。

当前 mutifloor 资产来自两个语义化 PLY：

```text
source/scene/multifloor/ply/3dgs_collision.ply
  碰撞 mesh PLY，包含 face，用于 PCT 建图和 collision USD

source/scene/multifloor/ply/3dgs_visual.ply
  3DGS Gaussian PLY，包含 opacity / scale_* / rot_*，没有 mesh face
```

注意：历史文件名 `3dgs_collision_cropped.ply` 容易误导。当前仓库统一使用
`3dgs_collision.ply` 和 `3dgs_visual.ply`，禁止在业务逻辑中按旧文件名猜测用途。

如需给外部 PCT README 兼容脚本准备 `mutifloor/` 命名，可以只做检查：

```bash
python scripts/scene/setup_pct_mutifloor_assets.py
```

如果确实需要在 `external/PCT/mutifloor/` 下生成 README 兼容文件，显式传
`--copy-assets`。该脚本只复制，不创建软链接：

```bash
python scripts/scene/setup_pct_mutifloor_assets.py --copy-assets --force
```

## SAGE-3D USD 场景生成流程

当前推荐用 SAGE-3D/NuRec 路线从 `3dgs_visual.ply` 生成 visual USDZ，从
`3dgs_collision.ply` 生成 collision USD，再组合为 Isaac Sim 主 USDA：

```bash
bash tools/scene/rebuild_multifloor_sage_assets.sh
```

输出：

```text
source/scene/multifloor/usdz/multifloor.usdz
source/scene/multifloor/usd/multifloor_collision.usd
source/scene/multifloor/usda/multifloor.usda
```

主 stage 会把碰撞几何挂到 `/World/scene_collision`，这是当前
`IsaacLabSimulationRuntime` 和 navigation 脚本默认读取的 terrain prim。主 stage
会同时把 `/World/gauss` visual 和 `/World/scene_collision` collision 施加 180 度
Z 轴旋转，使 Isaac Sim 世界坐标和 PLY/PCT 坐标保持：

```text
PCT_x = -sim_x
PCT_y = -sim_y
PCT_z =  sim_z
```

`multifloor.usda` 中 `/World/gauss` 是可见 NuRec visual，
`/World/scene_collision` 是不可见 physics collision payload。`tools/scene/validate_multifloor_usd_assets.py`
会检查 NuRec field 是否存在，并确认 `/World/gauss` 的 `rotateXYZ` 是 `(0, 0, 180)`。

当前仓库还提供了一个 `gsplat` 离线渲染入口，用于验证 multifloor 的真实 3DGS 视觉
质量。它不会修改 Isaac Sim stage，而是直接读取 `3dgs_visual.ply` 并输出 PNG：

```bash
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_EXTENSIONS_DIR=/tmp/arm_vla_pct_torch_extensions
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=1
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python scripts/scene/render_yinluyuan_3dgs.py \
  --max-gaussians 2000000 \
  --width 960 \
  --height 540 \
  --output-image outputs/multifloor_3dgs/preview.png
```

如果 `gsplat` 首次运行时编译失败并提示缺少 `cuda_runtime_api.h`，说明 C++ 编译阶段
没有找到 CUDA headers。补齐上面的 `CPATH` / `CPLUS_INCLUDE_PATH` 后重新运行。8GB
显存机器不建议一开始加载全部 2455 万个 Gaussian；先用 100 万到 400 万个点验证
相机、颜色和坐标，最终离线视频再按显存余量提高 `--max-gaussians`。

该离线渲染路径可以达到比 `UsdGeom.Points` 更接近最终视频的视觉质量，但它还不是
Isaac Sim viewport 原生渲染。正式录制时需要让 Isaac 相机轨迹和该脚本使用同一组
camera pose，再把 Isaac 中的机器人 / 物体前景与 3DGS 背景合成；这属于后续的视频
合成集成工作。

当前下一阶段已经预留相机同步接口。运行 full-physics / smoke 时，可以让 overview
recorder 在写 MP4 的同时导出相机轨迹：

```bash
python scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --record-video \
  --video-mode overview \
  --export-video-camera-trajectory \
  --video-camera-trajectory-out outputs/multifloor_3dgs/overview_camera_trajectory.jsonl
```

然后用同一条相机轨迹离线批量渲染 3DGS 背景帧：

```bash
python scripts/scene/render_yinluyuan_3dgs.py \
  --camera-trajectory-jsonl outputs/multifloor_3dgs/overview_camera_trajectory.jsonl \
  --frame-output-dir outputs/multifloor_3dgs/background_frames \
  --max-gaussians 4000000 \
  --trajectory-resolution json \
  --radius-clip 0.0
```

这一步输出的是与 Isaac overview 视频同 camera pose 的 3DGS 背景帧。下一步还需要
让 Isaac 导出机器人 / 物体前景的 alpha 或 depth，再做逐帧合成。

早期 `scripts/scene/build_yinluyuan_usd.py` 和
`scripts/scene/build_yinluyuan_visual_points_usd.py` 只保留为 legacy/诊断工具；默认输出
带 `legacy` 后缀，避免覆盖当前 SAGE/NuRec 主资产。正式 Isaac Sim 场景入口仍是
`source/scene/multifloor/usda/multifloor.usda`。

生成的 `.usd` / `.usda` / `.usdz` 属于本地大型资产，按 `.gitignore` 不提交。

## 资产准备流程

1. 确认当前资产位于 `source/scene/multifloor/ply/3dgs_collision.ply` 和
   `source/scene/multifloor/ply/3dgs_visual.ply`，并且不是软链接。
2. 运行 `bash tools/scene/rebuild_multifloor_sage_assets.sh`，生成 clean NuRec USDZ、
   collision USD 和唯一主 USDA。
3. 使用本仓库 `scripts/navigation/build_pct_multifloor_assets.py` 从 collision PLY 构建
   `source/scene/multifloor/mutifloor.pickle` 和
   `source/scene/multifloor/mutifloor_ply_walkable.npy`。
4. 用 `scripts/navigation/probe_pct_plan.py` 启动本仓库 `pct_grid_server.py`，先在
   不启动 Isaac Sim 的情况下验证 `start -> pick -> place` 的 3D path。
5. 使用 `scripts/scene/render_yinluyuan_3dgs.py` 验证真实 3DGS 离线渲染质量。
6. 录制视频时传 `--export-video-camera-trajectory`，导出 overview 相机轨迹。
7. 使用 `--camera-trajectory-jsonl` 批量渲染 3DGS 背景帧。
8. 训练或准备 Go2-X5 多楼层 locomotion checkpoint。

`external/PCT` 仅作为参考 codebase，不是当前 pipeline 的默认运行依赖。如确实需要给
上游 README 兼容脚本准备旧文件名，运行
   `python scripts/scene/setup_pct_mutifloor_assets.py --copy-assets --force` 复制到
   `external/PCT/mutifloor/`；不要使用软链接。

手动重建当前推荐的 16.76M clean NuRec 版本：

```bash
python tools/scene/sample_gaussian_ply.py \
  --input-ply source/scene/multifloor/ply/3dgs_visual.ply \
  --output-ply source/scene/multifloor/ply/3dgs_visual_final_17m_clean.ply \
  --max-points 0 \
  --clip-reference-ply source/scene/multifloor/ply/3dgs_collision.ply \
  --clip-margin 5 5 3 \
  --max-scale 0.5 \
  --min-opacity 0.2 \
  --report-output outputs/multifloor_nurec_final_17m_clean_ply.json \
  --force
```

```bash
PATH=/data/conda_envs/sage/bin:$PATH \
/data/conda_envs/sage/bin/python -m threedgrut.export.scripts.ply_to_usd \
  source/scene/multifloor/ply/3dgs_visual_final_17m_clean.ply \
  --output_file source/scene/multifloor/usdz/multifloor.usdz
```

```bash
/data/conda_envs/sage/bin/python tools/scene/build_multifloor_usda.py \
  --visual-mode nurec \
  --usdz source/scene/multifloor/usdz/multifloor.usdz \
  --collision-usd source/scene/multifloor/usd/multifloor_collision.usd \
  --output-usda source/scene/multifloor/usda/multifloor.usda \
  --force
```

注意：当前 `external/PCT/scripts/navigation/pct_server.py` 仍包含 `/home/y/...`
硬编码路径，并依赖 clone 中没有出现的 `planner_wrapper` backend，因此不作为当前
pipeline 的运行入口。需要上游 backend 时，应先把相关代码迁移到本仓库本地脚本，
并保留 stdin/stdout JSON 协议。

示例 task `tasks/nav_pick_place_apple_multifloor_pct.json` 是模板。当前 `scene_usd`
指向 `source/scene/multifloor/usda/multifloor.usda`。真实运行前仍需要根据场景
坐标更新目标、楼层、slice 和 PCT tomogram / walkable 路径。

## 运行命令

使用 PCT 运行 navigation smoke：

```bash
python scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --navigation-smoke \
  --global-planner pct \
  --pct-no-fallback \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-server-python /data/conda_envs/sage/bin/python \
  --pct-tomogram-path source/scene/multifloor/mutifloor.pickle \
  --pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --policy-profile pct_multifloor \
  --navigation-visual-mode collision \
  --no-randomize-task \
  --no-randomize-base-goal
```

如果使用当前仓库默认路径，`--global-planner pct` 会自动选择
`scripts/navigation/pct_grid_server.py`、
`source/scene/multifloor/mutifloor.pickle` 和
`source/scene/multifloor/mutifloor_ply_walkable.npy`。上面的命令显式写出这些参数，
是为了调试时能一眼确认实际使用的运行入口。
如果要严格使用 task JSON 中按 Isaac Sim Transform 记录的 start/pick/place 位置，
必须保留 `--no-randomize-task --no-randomize-base-goal`；否则默认随机化会替换
pick/place base goal。

当前多楼层场景已按 Camera0-8 布置 stage camera。pipeline 的默认第三人称相机为
`/World/Camera0`，overview recorder 也会优先选择 `Camera0`。如果需要检查前视或腕部
observation，相机仍由 robot task 内的 sensor config 决定，不等同于 stage Camera0。

`--navigation-visual-mode auto` 是默认值：`pct_multifloor` navigation smoke 使用
轻量 collision mesh，避免 1676 万 Gaussian NuRec visual 与 PhysX/CUDA 同卡运行；
flat profile 仍使用完整视觉。需要单独检查高质量视觉时可传
`--navigation-visual-mode full`，但它不适合作为当前 8 GB GPU 上的物理导航验收模式。

navigation smoke 成功后，建议先运行 pick-only 真实机械臂验收。该模式会执行
`nav_to_pick -> pick`，pick 成功后进入 cleanup，不会跨楼层去 place：

```bash
python scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --pick-smoke \
  --global-planner pct \
  --pct-no-fallback \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-server-python /data/conda_envs/sage/bin/python \
  --pct-tomogram-path source/scene/multifloor/mutifloor.pickle \
  --pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --policy-profile pct_multifloor \
  --navigation-visual-mode collision \
  --no-randomize-task \
  --no-randomize-base-goal \
  --record-video \
  --video-mode overview \
  --overview-capture-backend viewport
```

当前多楼层 task 使用 `/World/apple_01` 作为 pick 物体；`/World/apple` 是楼上 place
位置的场景占位苹果，会在运行时隐藏。`pct_multifloor` 的 pick-smoke/full-physics
会在 reset 后先让动态苹果自然沉降，连续 20 个控制步满足低线速度和低角速度后，
记录并冻结当前 PhysX 位姿，再开始 PCT 导航。稳定结果写入 summary 的
`simulation.object_settle_final_report`；若苹果滚出 task 初始位置 0.25 m 或 240
步内仍未稳定，pipeline 会分别报告 `object_settle_out_of_bounds` 或
`object_settle_timeout`，不会把晃动中的瞬时 pose 当作抓取基准。
当前 task 的 `pick.object_pose_world` 已更新为一次真实运行中连续 20 帧低速后的稳定
PhysX pose；其中 world quaternion 已反解为 `rotateX:unitsResolve=90°` 之前的根 RPY。
运行时仍保留沉降检查，以适应碰撞求解器和初始接触的微小变化。

`/World/apple_01/Apple_M_Apple_0/Apple_M_Apple_0` 是带
`PhysicsCollisionAPI` 的碰撞 Mesh，保持 `visibility=invisible`；实际渲染 Mesh
是同级的 `/World/apple_01/Apple_M_Apple_0/visual`。运行时 collision visual
过滤器只隐藏带碰撞 schema 或 `physics:collisionEnabled` 属性的 Mesh，不能仅因
完整 prim path 含有 `Apple_M_Apple` 就隐藏节点，否则会同时隐藏 `visual` 并让
CuRobo 导出得到零尺寸 bbox。若再次出现 `authored object bbox 必须具有正尺寸`，
应先检查 summary 中 `last_current_state_curobo_pick_export.bbox_world.size_xyz`
以及 visibility filter report，不要直接把碰撞代理改为可见。

pick-smoke 通过后，再去掉 `--pick-smoke` 运行完整 `nav_to_pick -> pick ->
nav_to_place -> place`。当前该命令会进入跨楼层 place 验收：

```bash
python scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --global-planner pct \
  --pct-no-fallback \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-server-python /data/conda_envs/sage/bin/python \
  --pct-tomogram-path source/scene/multifloor/mutifloor.pickle \
  --pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --policy-profile pct_multifloor \
  --navigation-visual-mode collision \
  --no-randomize-task \
  --no-randomize-base-goal \
  --record-video \
  --video-mode overview \
  --overview-capture-backend viewport
```

调试 PCT 时关闭 A* fallback：

```bash
python scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --navigation-smoke \
  --global-planner pct \
  --pct-no-fallback \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-server-python /data/conda_envs/sage/bin/python \
  --pct-tomogram-path source/scene/multifloor/mutifloor.pickle \
  --pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --policy-profile pct_multifloor \
  --no-randomize-task \
  --no-randomize-base-goal
```

在启动 Isaac Sim 前，先用 PCT 规划探针验证 server、tomogram 和 walkable 是否能给
`start -> pick -> place` 返回 3D path：

```bash
python scripts/navigation/probe_pct_plan.py \
  --dry-run \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-server-python /data/conda_envs/sage/bin/python \
  --pct-tomogram-path source/scene/multifloor/mutifloor.pickle \
  --pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy
```

`--dry-run` 只检查输入是否齐全，不启动 server。资产齐全后去掉 `--dry-run`：

```bash
python scripts/navigation/probe_pct_plan.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-server-python /data/conda_envs/sage/bin/python \
  --pct-tomogram-path source/scene/multifloor/mutifloor.pickle \
  --pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy \
  --output-json outputs/pct_plan_probe.json
```

探针报告会记录每段路径的 `path_3d`、`slice_start`、`slice_end`、
`snap_start_dist`、`snap_end_dist`、`cross_floor`、`hard_obstacle_min_slices` 和
`cross_floor_hard_obstacle_min_slices`。`hard_obstacle_mode="body_clearance_volume"`
表示三角面墙体和家具按当前 floor slice 参与搜索。跨楼层时还会记录
`cross_floor_gateway_count`、`cross_floor_gateway_radius_m`、
`cross_floor_gateway_cells` 和 `cross_floor_gateway_mode`。如果 `pick_to_place` 的
z 或 slice 没有变化，说明还没有真正验证到跨楼层规划。当前 Yinluyuan 默认跨层路线
应先从 pick 区域经过中间大门，再经过 `(1.5, 5.7)` 附近楼梯入口；如果黄色路径没有
进入该 gateway 半径，优先检查 `--pct-cross-floor-gateway` 和
`--pct-cross-floor-vertical-obstacle-min-slices`。

只想在 GUI 中观察规划路线，不执行 DWA/RL/机械臂时，使用 PCT preview 模式：

```bash
python scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --pct-plan-preview \
  --global-planner pct \
  --pct-no-fallback \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-server-python /data/conda_envs/sage/bin/python \
  --pct-tomogram-path source/scene/multifloor/mutifloor.pickle \
  --pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy \
  --no-randomize-task \
  --no-randomize-base-goal
```

该模式会打开 task 场景，在 `/World/PCTPlanPreview` 下绘制两条路线：
`start_to_pick` 为青色，`pick_to_place` 为黄色，并写出
`outputs/full_physics_pipeline/episode_000000/pct_plan_preview.json`。非 headless
运行时窗口会自动保持打开；用 `--headless` 可以只生成 JSON 报告。

训练脚手架 dry-run：

```bash
python scripts/navigation/train_pct_multifloor_policy.py --dry-run
```

## 坐标系

默认 adapter 模式为 `sim_to_pct_180deg`：

```text
PCT_x = -sim_x
PCT_y = -sim_y
PCT_z =  sim_z
```

如果 tomogram 原点或单位与 Isaac Sim 不一致，使用 `--pct-offset-x`、
`--pct-offset-y`、`--pct-scale-x` 和 `--pct-scale-y` 调整。路径镜像时优先检查
x / y 符号是否反了。

## 故障排查

- 当前目录不是 `arm_vla_pct`：从 `/home/light/workspace/arm_vla_pct` 运行。
- 当前分支不是 `pct`：停止当前流程，不要自动切换分支。
- PCT server 没有输出 `READY`：检查 `--pct-server-script`、
  `--pct-server-python` 和 PCT 运行环境。
- tomogram 路径错误：检查 `--pct-tomogram-path` 和 `PCT_TOMOGRAM_PATH`。
- walkable map 路径错误：检查 `--pct-walkable-path` 和 `PCT_WALKABLE_PATH`。
- 坐标系 x / y 反了：调整 `--pct-offset-*`、`--pct-scale-*` 或 coord mode。
- goal 被 snap 到错误楼层：检查 `metadata["slice_start"]`、
  `metadata["slice_end"]`、`metadata["snap_start_dist"]` 和
  `metadata["snap_end_dist"]`。
- 跨楼层路线走侧门而不是中间大门：先运行 `--pct-plan-preview`，检查黄色
  `pick_to_place` 路线是否从 pick 区域直接沿 `x≈-3.28` 下行。如果是，检查
  `metadata["cross_floor"]` 是否为 true、`metadata["hard_obstacle_min_slices"]`
  是否为 9；必要时临时调大 `--pct-cross-floor-vertical-obstacle-min-slices`，
  但不要同时放宽同层 `--pct-global-vertical-obstacle-min-slices`。
- 跨楼层路线直接在远离楼梯的位置换层：检查
  `metadata["cross_floor_gateway_count"]` 是否大于 0，以及黄色路线是否先经过
  `(1.5, 5.7)` 附近。默认 gateway 可通过
  `--pct-cross-floor-gateway 1.5,5.7,0.6` 和
  `--pct-cross-floor-stair-exit 1.9,8.0,3.0` 显式指定。正常 metadata 应为
  `cross_floor_gateway_mode="strict_monotonic"`、
  `hard_obstacle_mode="body_clearance_volume"`；如果返回 `no_gateway_path`，应修补
  tomogram/walkable 或调整真实楼梯 gateway，不能恢复 relaxed fallback。
- 上楼轨迹贴到扶手或快速斜跨上层：先运行 `--pct-plan-preview`，检查
  `metadata["cross_floor_stair_vertical_cells"]`、`stair_vertical_radius_m` 和
  `stair_constraint_mode`。正常应为 `surface_3d`：server 按每个高度 slice 构造
  三维楼梯支撑走廊，换层边不能只因 XY 投影重叠就从扶手处上升。默认表面容错半径
  为 `0.60 m`，并使用 `stair_progress_cost_weight=20.0` 优先选择三维中心线。
  `0.60 m` 用于容忍扫描可走格孔洞，不是允许任意高度在同一二维走廊内换层。
  `--pct-cross-floor-stair-midpoint` 不是任务 waypoint，而是地图级楼梯表面采样；
  可以无序传入。server 会按高度聚类同层平台，并自动选择连接上下楼梯段最短的
  平台顺序。若真实楼梯中心线仍有偏差，先修正采样或 tomogram，不要恢复二维
  progress fallback。
- U 形楼梯在 XY 投影上自重叠：不要用单张 `progress[x,y]` 决定换层顺序。同一 XY
  可能同时属于楼梯下段和上段；当前 `surface_3d` 使用 `(slice,x,y)` 区分它们。
- F1/F2 路径穿墙或穿过椅子：确认 metadata 为
  `hard_obstacle_mode="body_clearance_volume"`。如仍过近，可提高
  `--pct-obstacle-clearance-cost`，不要直接增加硬膨胀导致门洞不可达。
- 机器人顶住障碍后旋转或滑动逃离：先确认 PCT local map 是由 tomogram
  traversability 与 walkable 合并生成，而不是只读取单层 `walkable.npy`。如果
  PCT 返回的 `path_3d` 点在 DWA inflated map 中是 occupied，说明局部地图和 PCT
  server 的可走判定不一致。
- 同层目标明明可直达却先绕路：检查 `metadata["pct_path_mode"]` 和
  `metadata["local_refinement"]["mode"]`，正常应分别为 `same_floor_direct` 和
  `pct_same_floor_direct`。确认没有显式传入正数
  `--pct-vertical-obstacle-min-slices`，否则 collision PLY 启发式可能制造假障碍。
- carry 阶段多次撞墙但 DWA 始终报告 `clearance=1.0`：检查 executor metadata
  中 `map_selection.multifloor=true`、`route_cells_cleared>0`，并确认
  `protected_cells_preserved` 字段存在、planner metadata 中
  `hard_obstacle_min_slices=7`，以及
  `dwa_limits.max_linear_velocity=0.30`、`max_angular_velocity=0.35`。
  全局硬障碍阈值与跨楼层 DWA 障碍阈值相互独立；不要通过关闭 carry 墙体投影解决
  楼梯 gateway 的假占用。若路径仍穿墙，先检查
  `--pct-global-vertical-obstacle-min-slices`，不要继续增大走廊清理半径。
- 接近 pick 点时原地摔倒：确认 `pick.base_goal` 是机器人底盘站位，且与
  `pick.object_pose_world` 保持安全距离。`base_goal` 不应写成苹果中心；苹果会被
  PCT smoke 投影为一个小 keepout，防止底盘把任务物体当成可穿越空间。
- `exec_nav_to_pick` 在目标前约 `0.25 m` 报 `nav_collision`：检查 summary 中
  `stall.max_displacement_m` 和 `distance_to_goal`。如果 DWA 候选均可行但机器人在
  桌边连续前进无位移，通常是底盘站位落入物理不可达边界，而不是 PCT 无路径。
  当前多楼层示例把 pick 底盘站位设为 `(-3.50493, 6.50)`；不要通过放宽全局 A*
  到达容差掩盖 task 坐标错误。
- 能规划但机器人上不去楼梯：需要训练或传入有效的 `pct_multifloor`
  locomotion checkpoint。
- 高度扫描 187 维全部为 `-1`：检查是否误用了单层任务的 20 m 射线起点；
  `pct_multifloor` task 应使用层高以内的射线起点。
- 机器人在急转时失稳：检查 episode 中的 `velocity_commands` 和
  `policy_observation_report`；当前 profile 会把命令限制为
  `vx<=0.45 m/s`、`|wz|<=0.50 rad/s`，仍低于 checkpoint 训练快照的
  `vx<=0.55 m/s`、`|wz|<=0.60 rad/s`。近目标速度仍限制为 `0.18 m/s`。
- `carb.cudainterop` 报 `CUDA error 700` / external semaphore 失败：检查 Kit
  日志是否使用 `isaaclab.python.rendering.kit` 且同时加载 `/World/gauss`。导航
  验收改用 `--navigation-visual-mode collision`；关闭崩溃后的 Isaac Sim 进程后
  再启动，因为 error 700 发生后当前 CUDA context 不可继续使用。
- `gsplat` 报 `cuda_runtime_api.h` 不存在：设置 `CPATH` 和
  `CPLUS_INCLUDE_PATH` 指向当前 CUDA toolkit 的 `targets/x86_64-linux/include`。
- 3DGS 离线图像仍是黑的：先降低 `--radius-clip`，或显式传
  `--camera-eye` / `--camera-target`，并检查 `preview_report.json` 中的
  `render_bounds_min` / `render_bounds_max`。
- Isaac Sim / RViz Vulkan 冲突：避免同时启动多个 Vulkan 消费者，或拆到不同
  display / GPU 环境。
