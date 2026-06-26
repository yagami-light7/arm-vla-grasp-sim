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

- `Rough/`：本地 RL policy checkpoint、导出 policy 和训练参数。
- `checkpoints/go2_x5/pct_multifloor/`：整理后的新 RL policy 运行目录。
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

当前 `source/scene/multifloor/` 中尚未包含 `.pickle` / `.npy` 建图结果。生成后可以
把对应文件传给：

```bash
--pct-tomogram-path source/scene/multifloor/mutifloor.pickle
--pct-walkable-path source/scene/multifloor/mutifloor_ply_walkable.npy
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
  --mode full_physics \
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
  --policy-profile pct_multifloor
```

如果使用当前仓库默认路径，`--global-planner pct` 会自动选择
`scripts/navigation/pct_grid_server.py`、
`source/scene/multifloor/mutifloor.pickle` 和
`source/scene/multifloor/mutifloor_ply_walkable.npy`。上面的命令显式写出这些参数，
是为了调试时能一眼确认实际使用的运行入口。

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
  --policy-profile pct_multifloor
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
`snap_start_dist` 和 `snap_end_dist`。如果 `pick_to_place` 的 z 或 slice 没有变化，
说明还没有真正验证到跨楼层规划。

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
- 能规划但机器人上不去楼梯：需要训练或传入有效的 `pct_multifloor`
  locomotion checkpoint。
- `gsplat` 报 `cuda_runtime_api.h` 不存在：设置 `CPATH` 和
  `CPLUS_INCLUDE_PATH` 指向当前 CUDA toolkit 的 `targets/x86_64-linux/include`。
- 3DGS 离线图像仍是黑的：先降低 `--radius-clip`，或显式传
  `--camera-eye` / `--camera-target`，并检查 `preview_report.json` 中的
  `render_bounds_min` / `render_bounds_max`。
- Isaac Sim / RViz Vulkan 冲突：避免同时启动多个 Vulkan 消费者，或拆到不同
  display / GPU 环境。
