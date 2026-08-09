# PCT Scene: Go2-X5 多场景移动操作与数据采集

PCT Scene 用一套代码运行良渚单层任务和别墅多楼层任务。使用`--scene-profile` 选择场景后，程序会加载对应的任务、PCT 地图、坐标变换、随机化、楼梯配置、视觉层和 overview 相机。切换场景不需要更换 worktree，也不需要重复输入整组参数。

`pct-scan` 分支已把主运行链从旧 PCT + DWA 替换为 PCT + SCAN ROS 2。
架构边界、ROS/Isaac 进程关系和当前阶段见
[PCT + SCAN 跨楼层导航重构](docs/pct_scan_navigation.md)。

截至 2026-08-09，发布主线固定使用
`ros2_ws/src/isaac_navigation_bridge/config/pct_scan_tuning.yaml`：SCAN 参考巡航
`0.60 m/s`、纵向硬上限 `0.65 m/s`、偏航角速度上限 `0.60 rad/s`。同一代码、
同一参数、同一 146 点 PCT Path 和原 Go2-X5 checkpoint 的跨楼层携物导航
seeds 0、1、2 已严格通过 `3/3`，均到达目标并在到达后持续输出零速度。生产
参数还完成了一次
`nav → pick → 携物跨层 nav → place → LeRobot export` 全流程实测，最终状态为
`done`，抓取和放置验证均通过。

这里的“稳定跑通”有明确边界：静态别墅场景中的 PCT→SCAN 平地闭环、楼层交接、
抓放交接和最终停车已经跑通；楼梯段按用户确认的方案冻结底盘并沿已绑定的 PCT
楼梯段切换楼层。因此它是可复现的工程主线，不是纯物理爬楼，运行摘要会如实保持
`physical_navigation_success=false` 和 `pure_physics_success=false`。移动推车
在线绕障和触发 PCT 全局重规划尚未列入本次发布验收。

本次发布不再继续 DWA 对比，也不声称 SCAN 算法快于 DWA。仓库保留已有实验记录
用于审计，但默认运行链中没有 DWA、fallback 或局部规划器切换开关。`0.75 rad/s`
偏航候选也没有进入生产配置：它虽然通过了 3 次独立携物导航，却在完整 pipeline
的放置质量门失败。

## 场景与流程


| Profile       | 任务与默认行为                                                                                                                                        |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `liangzhu`    | 良渚单层 box1 → box2 任务：拿起 box1 上的可乐并放到 box2。开启联合随机化，使用`identity` PCT 坐标、collision 量产视觉和固定 `/World/overview` 相机。 |
| `multi_floor` | 别墅多楼层任务，将苹果从 F1 搬到 F2。使用`sim_to_pct_180deg` PCT 坐标、楼梯锚点和 collision 视觉；关闭任务随机化，overview 相机按阶段切换。           |

兼容别名：`liangzhu_single_floor` 对应 `liangzhu`；`multifloor`、`pct_multifloor` 和`villa` 对应 `multi_floor`。

场景配置位于 `configs/scenes/*.json`。CLI 参数的优先级高于 profile 默认值，调试时可以只覆盖需要修改的选项。兼容参数 `--pct-multifloor` 等价于`--scene-profile multi_floor`。

一个 episode 依次执行导航、抓取、携物导航、放置和数据导出：

```mermaid
graph LR
    A["随机化"] --> B["nav2pick"]
    B --> C["pick"]
    C --> D["nav2place"]
    D --> E["place"]
    E --> F["LeRobot 数据导出"]
```

良渚默认随机化模式是 `liangzhu_box_pair_xy_v1`，一次样本使用同一个
`random.Random(seed)` 同步产生所有位姿：

- box1/box2 只在 authored 位姿基础上分别采样 XY `Uniform(-0.12m, 0.12m)`；Z、姿态、scale 和 xform op 不变，两桌中心距至少 `2.4m`。
- 机器人位于两桌中心连线的 `0.4..0.6` 段，横向偏移 `±0.18m`，yaw 在 `[-180°, 180°]` 全范围采样，根 Z 由 collision PLY 实时求地面。
- 可乐在 box1 中心局部 `0.08m × 0.06m` 安全区中采样 XY，yaw 在 `[-180°, 180°]`，Z 由 box1 顶面加可乐 bbox 半高求得。
- box2 放置区为中心局部 `0.10m × 0.05m`；pick standoff 为 `0.50..0.54m`，place standoff 为 `0.48..0.51m`。后者为 `0.08m` 导航交接容差预留了机械臂可达裕量。
- 采样同步更新 robot/object/box pose、支撑 bbox、CuRobo proxy、placement/base goal、PCT/DWA keepout 和 rejection metadata；最多尝试 300 次。

良渚 box1 任务还启用 `supported_upright_v1` 可乐初始化策略：前 8 个控制步允许
Z 向沉降但保持随机化请求的 XY/直立姿态，接触箱面后 sleep，抓取接触再由 PhysX
自动唤醒。导航前强制检查 XY/Z 偏差均不超过 `0.02m`、姿态角误差不超过
`0.10rad`；超限以 `object_initialization_pose_invalid` 拒绝。历史 seed 7 的
旧结果曾倾倒 `96.03°` 并漂移 `84.07mm`，修复后同 seed 倾角约
`0.000005°`、XY 漂移约 `0.00019mm`，且完整 pipeline 与 251 行 LeRobot 数据均
通过验收。输出为
`/mnt/sage_data/outputs/pct_scene/seed7_object_init_upright_fix_v2_20260719/episode_000000`。

当前属于 Phase 1 几何随机化：不随机光照、材质或相机，抓放目标使用
live Mesh/PhysX 真值，RGB 用于数据记录而非当前控制定位。

## 文档导航

- [1. 环境部署](#1-环境部署)
- [2. 仿真运行与数据导出](#2-仿真运行与数据导出)
- [3. 常用命令与 CLI 参数](#3-常用命令与-cli-参数)

## 1. 环境部署

### 1.1 系统要求

已验证环境为 Ubuntu Linux、支持 CUDA 12.x 的 NVIDIA 驱动、Python 3.11、Isaac Sim 5.1、Isaac Lab 2.3.x 和 cuRobo 0.8.x。显存少于 12 GB 时，建议使用collision 视觉模式采集数据。

安装系统依赖并确认驱动可用：

```bash
sudo apt update
sudo apt install -y git ffmpeg build-essential cmake ninja-build libgl1 libglib2.0-0

nvidia-smi
conda --version
```

如果系统尚未安装 conda，请先安装 Miniforge、Miniconda 或 Anaconda，然后重新打开终端。

实际显存需求取决于场景和视觉模式。良渚 full/Gaussian 模式使用 NuRec 资产，8 GB RTX 4060 Laptop 已知会在 RGB render product 启动后触发 CUDA illegal address 700。

### 1.2 获取代码

```bash
git clone --branch pct-scan --single-branch \
  https://github.com/yagami-light7/arm-vla-grasp-sim.git \
  pct_scan
cd pct_scan
```

场景和物体等运行时资产不在 Git 仓库中，请按第 1.5 节的说明从网盘下载并放到指定目录。

### 1.3 创建 Isaac 仿真环境

```bash
conda create -n isaac_locomani python=3.11 -y
conda activate isaac_locomani

python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
git -C ../IsaacLab checkout 2f91d7dd2994246505602526b32ac67ff758d472
../IsaacLab/isaaclab.sh -i

git clone https://github.com/NVlabs/curobo.git ../curobo
git -C ../curobo checkout 87260212b9ad5ebe486427cbf168611145232884
python -m pip install -e "../curobo[cu12]"

python -m pip install -r requirements/isaacsim51_runtime.txt
```

requirements 文件只安装已验证的常规 Python 依赖。Isaac Sim、Isaac Lab 和 cuRobo 需要按上述步骤单独安装。Go2-X5 task 包已放在 `source/robot_lab`，不用重复安装。

验证 Python 环境：

```bash
python -c "import isaacsim, isaaclab, curobo, torch; print(torch.cuda.is_available())"
```

输出应为 `True`。后续命令默认使用当前 conda 环境中的 Python：

```bash
export ISAAC_PYTHON="$(command -v python)"
export PCT_SCENE_ROOT="$(pwd)"
export PCT_SCENE_OUTPUT="${PCT_SCENE_OUTPUT:-$HOME/pct_scene_outputs}"
mkdir -p "$PCT_SCENE_OUTPUT"
```

### 1.4 创建 LeRobot / Rerun 工具环境

数据校验和 Rerun 导出使用独立的 Python 3.10 环境，避免与 Isaac Sim 依赖冲突：

```bash
conda create -n lerobot_rerun python=3.10 -y
conda activate lerobot_rerun
python -m pip install -r requirements/lerobot_rerun.txt
```

### 1.5 准备场景资产

此前记录的 73/169 点仿真路径由仓库内 PCT-compatible 3D A* 实现
`scripts/navigation/pct_grid_server.py` 生成；生产 YAML 与 PCT→SCAN 组合 launch
现已切为官方 backend，扩展不可用时会启动失败而不会静默回退。官方 PCT core 位于
`byangw/PCT_planner`，必须固定到下列提交；其中 `planner/` 和 `tomography/`
分别提供真实规划与层析建图核心，许可证为 GPLv2-or-later：

```bash
git clone https://github.com/byangw/PCT_planner.git external/PCT_planner
git -C external/PCT_planner checkout 35cd73fd82bcd51bc538429294af7646b2a09815
```

本机已通过相同固定提交的 GitHub codeload 归档准备 ignored 本地副本，归档
SHA256 为 `daf5f90b29c76cfa5fc6bf10d6dcfd200c1077778b22671c98aa51f9adb06d64`。
本机现已完成 GTSAM 4.1.1、OSQP 与四个 CPython 3.10 planner 扩展的 Release
构建，RUNPATH/`ldd`、固定源码树/补丁哈希和真实加载均通过。运行身份是固定官方
commit 加受跟踪的 native A* 补丁；补丁关闭 `march=native`，使用相对 RUNPATH，
并增加触达节点清理、原子取消和 GIL 释放。生产 ROS 2 backend 只使用
官方 `OfflineElePlanner` native A* 做全局三维选路，SCAN 负责连续局部优化；
upstream 离线探针得到 190 点、logical layer `8→15` 的真实多楼层地面 Path，
隔离 DDS 生命周期探针发布 189 点 typed Path 并干净退出。若扩展、共享库或固定
资产缺失，生产入口仍会失败关闭，不会切回 compatible backend。

同一 189 点 upstream Path 已在隔离 CPU ROS 2 图中继续贯通 SCAN 与 controller：
144/144 个显式 free 端点完成 typed 地图融合，随后发布 19 控制点三阶正常
B-spline 和非零 `/cmd_vel`，局部规划约 `67.6 ms`。时间分配与最终发布门现统一
按三维速度/加速度向量模长执行，斜向运动不会再因“分量通过、模长超限”反复急停。
这项结果只验收首段消息与规划链；后续章节记录的 Isaac 静态跨层导航已经完成，
动态绕障恢复和 PCT 在线全局重规划仍不属于本次发布验收。

multi-floor 官方语义 tomogram 通过下列命令由 collision PLY 与从 `pct-scene`
提取的楼梯局部 profile 重建；profile 只覆盖楼梯拓扑，不硬编码完整任务路线：

```bash
/usr/bin/python3 scripts/navigation/build_pct_multifloor_assets.py \
  --tomogram-kind upstream \
  --output-tomogram source/scene/multifloor/mutifloor_upstream.pickle \
  --output-walkable /tmp/mutifloor_upstream_walkable.npy \
  --report-output outputs/pct_multifloor_upstream_asset_build_report.json
```

场景和物体资产通过网盘提供，不随 Git 仓库发布。下载后，将网盘中的三个目录完整复制到下表所示位置。目录名和内部层级不要改动。


| 网盘目录      | 放置位置（相对于仓库根目录） | 内容                                                 |
| ------------- | ---------------------------- | ---------------------------------------------------- |
| `liangzhu/`   | `source/scene/liangzhu/`     | 良渚主场景、视觉与碰撞资产、PCT 地图、PLY 和运行清单 |
| `multifloor/` | `source/scene/multifloor/`   | 别墅主场景、视觉与碰撞资产、PCT 地图、PLY 和运行清单 |
| `objects/`    | `source/scene/objects/`      | 任务物体与支撑体                                     |

完成后的关键目录应为：

```text
source/scene/
├── liangzhu/                            # 整个目录通过网盘提供
│   ├── liangzhu.usda
│   ├── usd/
│   │   └── liangzhu_collision.usda
│   └── usdz/
│       └── liangzhu.usdz
├── multifloor/                          # 整个目录通过网盘提供
│   ├── usda/
│   │   └── multifloor.usda
│   ├── usd/
│   │   └── multifloor_collision.usd
│   └── usdz/
│       └── multifloor.usdz
└── objects/                              # 整个目录通过网盘提供
    ├── box/
    │   └── box.usd
    ├── box2/
    │   └── box2.usd
    ├── cola/
    ├── apple/
    └── bottle/
```

假设网盘解压目录为 `/path/to/assets/`，并且该目录下直接包含 `liangzhu/`、
`multifloor/` 和 `objects/`，请在仓库根目录执行：

```bash
mkdir -p source/scene/liangzhu source/scene/multifloor source/scene/objects

cp -a /path/to/assets/liangzhu/. source/scene/liangzhu/
cp -a /path/to/assets/multifloor/. source/scene/multifloor/
cp -a /path/to/assets/objects/. source/scene/objects/
```

命令中的 `/.` 表示复制目录内容，可以避免生成
`source/scene/liangzhu/liangzhu/` 这类重复目录。

良渚场景还支持将大型资产放在仓库外。只有采用这种布局时，才需要设置环境变量：

```bash
export LIANGZHU_VISUAL_USDZ=/absolute/path/to/liangzhu.usdz
export LIANGZHU_COLLISION_USD=/absolute/path/to/liangzhu_collision.usda
```

环境变量的优先级高于仓库默认路径。如果以前设置过这些变量，现在想使用仓库内的
相对路径，请先执行：

```bash
unset LIANGZHU_VISUAL_USDZ LIANGZHU_COLLISION_USD
```

可以用以下命令检查所有默认路径：

```bash
ls -lh \
  source/scene/liangzhu/usdz/liangzhu.usdz \
  source/scene/liangzhu/usd/liangzhu_collision.usda \
  source/scene/objects/box/box.usd \
  source/scene/objects/box2/box2.usd \
  source/scene/multifloor/usdz/multifloor.usdz \
  source/scene/multifloor/usd/multifloor_collision.usd
```

文件名区分大小写。以上命令应在仓库根目录执行，也就是能看到 `README.md`、`configs/` 和 `source/` 的目录。每个场景的完整清单位于`source/scene/<scene>/runtime_asset_manifest.json`。

### 1.6 检查部署

运行仿真前，先执行只读检查：

```bash
# 先列出 profile，然后只检查自己需要的场景。
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --list-scene-profiles

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --check-scene-assets

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile multi_floor \
  --check-scene-assets
```

所需 profile 的检查全部通过后，再启动仿真。Python import 成功不代表场景资产齐全。

## 2. 仿真运行与数据导出

新终端中先激活环境并设置项目路径。将第二行替换为自己的仓库路径：

```bash
conda activate isaac_locomani
cd /path/to/pct_scene

export PCT_SCENE_ROOT="$PWD"
export PCT_SCENE_OUTPUT="${PCT_SCENE_OUTPUT:-$HOME/pct_scene_outputs}"
export ISAAC_PYTHON="$(command -v python)"
mkdir -p "$PCT_SCENE_OUTPUT"
```

### 2.1 先运行 dry-run

`--dry-run` 不运行物理仿真，它用于检查 profile、task、随机化和状态机是否能到达 `done`：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --dry-run \
  --output-dir /tmp/pct_scene_liangzhu_dry

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile multi_floor \
  --dry-run \
  --output-dir /tmp/pct_scene_multi_floor_dry
```

### 2.2 运行单个 episode

导航类模式现在固定走 PCT→SCAN ROS 2 主线。两个终端必须使用相同的 DDS
domain 和 Fast DDS。ROS 终端保留 ROS Python 3.10 的 `PYTHONPATH`：

```zsh
cd /path/to/pct_scan
export ROS_DOMAIN_ID=189
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py
```

组合 launch 默认同时启动 typed `navigation_supervisor`，并以 reliable +
transient-local KeepLast(64) 发布 GridMap/B-spline 诊断。若自定义状态、诊断
topic 或 frame，Isaac 命令必须同步传入
`--ros2-navigation-status-topic`、`--ros2-grid-map-diagnostics-topic`、
`--ros2-bspline-diagnostics-topic`、`--ros2-world-frame` 和
`--ros2-base-frame`；否则 policy 安全门或证据门会因接口不匹配失败关闭。
动态验收会把同一障碍的 GridMap free→occupied 来源、ordered detour、controller
accepted/TRACKING、clear 前 policy 写入和 clear 后恢复按 identity/时间顺序连接；
B-spline 净距使用速度上界推导的连续曲线下界，不以少量离散点代替全轨迹安全。

Isaac 终端要先固定 conda 环境中的 Python 3.11 路径，再 source ROS overlay 以
保留自定义消息共享库搜索路径，最后清除 ROS Python 3.10 路径：

```zsh
conda activate isaac_locomani
cd /path/to/pct_scan
export ISAAC_PYTHON="$(command -v python)"
source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh
export ROS_DOMAIN_ID=189
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset PYTHONPATH
```

`ControllerStatus`、`NavigationStatus`、`GridMapObservationDiagnostics` 和
`BsplineDiagnostics` 默认启用；若 overlay、相关 rosidl 共享库、消息接口或
Python ABI 不正确，pipeline 会在创建 CuRobo server、SimulationApp 和 RTX
场景之前失败，并打印上述修复命令。不要尝试在已经启动的 Python 进程内
source。

良渚 GUI：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --simulation-smoke \
  --seed 7000 \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_gui_seed7000" \
  --navigation-visual-mode full \
  --no-record-dataset \
  --no-record-video \
  --no-headless \
  --keep-window-open
```

该 scene-only 命令使用 `full` Gaussian/NuRec 视觉，适合在采集数据前检查场景。
PCT→SCAN 在线导航必须使用 profile 默认的 `collision` 视觉，保证前视 RTX 深度
能看到碰撞场景。在 8 GB RTX 4060 Laptop 与 Isaac Sim 5.1 的已验证环境中，
使用以下命令采集数据：

灯光模式默认是 `auto`。只要最终视觉模式为 `full`，runtime 就会自动显示该场景
USDA 中编写的 `DomeLight`、`SphereLight`、`RectLight` 等 stage lights，并关闭
相机补光；`collision` 模式自动使用相机补光。因此上述命令无需再添加
`--scene-light-mode stage`。需要覆盖自动行为时仍可显式指定 `camera` 或 `stage`。

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --seed 7000 \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_seed7000" \
  --headless
```

默认会导出 LeRobot 三路相机与同步`overview + front + wrist` composite MP4。当前 8 GB RTX 4060 Laptop 实测中，
`full` 能够加载 Gaussian/NUREC 并启动 PhysX，但首帧 headless 三相机渲染会触发 `cudaErrorIllegalAddress (700)`，因此 profile 的量产默认保持`collision`。这不是 CUDA 不可用；GUI 可显式使用 `full` 调试，两种视觉来源不应无标记混合。


别墅多楼层 GUI：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile multi_floor \
  --seed 0 \
  --output-dir "$PCT_SCENE_OUTPUT/multi_floor_gui" \
  --no-headless \
  --keep-window-open
```

只验证别墅楼梯 locomotion：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile multi_floor \
  --stair-locomotion-smoke \
  --output-dir "$PCT_SCENE_OUTPUT/multi_floor_stair_smoke" \
  --no-headless \
  --keep-window-open
```

该 smoke 会关闭 stair-float，并通过 SCAN Path 在楼梯段启用用户接受的 root/
关节冻结 workaround；它不代表低层 policy 纯物理爬楼。若要隔离验证 checkpoint
的纯物理能力，应使用下面的固定速度探针，且不得把失败或 root-lock 结果写成
纯物理跨层 locomotion 成功。

仓库内 `model_26000.pt` 的纯低层键盘测试使用独立入口。该入口加载与 checkpoint
一致的 Go2-X5 DogOnly 配置（260 维 policy 观测、12 维腿部动作），只生成一块
固定高度的确定性楼梯，并关闭训练期随机化。此测试不需要启动 ROS 2、PCT 或
SCAN：

```zsh
conda activate /data/conda_envs/isaacsim51_3dgs_grasp
cd /home/light/workspace/IsaacLab

./isaaclab.sh -p \
  /mnt/sage_data/workspace/pct_scan/scripts/navigation/play_go2_x5_keyboard.py \
  --checkpoint /mnt/sage_data/workspace/pct_scan/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --stair-height 0.10 \
  --warmup-steps 50 \
  --vx 0.30 \
  --vy 0.0 \
  --wz 0.40 \
  --real-time
```

窗口出现后先点击 Viewport。方向键上/下控制前进和后退，`Z`/`X` 控制左右
转向，`L` 立即将速度命令清零。初始位置位于低平台中心，朝外直行就是上楼；
前 50 个控制步保持零速度并渐入 action，用于避免刚 reset 时动作突变。默认
`vy=0`，因此左右方向键不会产生横移。这个入口只回答 checkpoint 是否具备纯
物理爬楼能力，并不加载别墅 PLY，也不能替代后续 PCT→SCAN 完整场景验收。

2026-08-02 用户在该确定性楼梯入口中完成了一次交互测试：设置
`vx=1.0 m/s`、`wz=1.0 rad/s` 时表现尚可，继续提高命令会发生翻倒。该结果是
定性单次观察，没有记录完成率、姿态峰值或多 seed，因此只作为 policy 能力边界
线索，不直接作为导航生产限幅。PCT→SCAN 生产主线使用
`max_vx=0.65 m/s`、`max_yaw_rate=0.60 rad/s`；权威值以统一调参 YAML 为准。

新的 `go2_rl_robotlab` MoE-CTS policy 使用导出的 TorchScript，而不是 RSL-RL
训练 checkpoint。它接收 45 维当前帧观测，在模型内部维护 10 帧、共 450 维
历史，并输出 12 个腿部关节动作。专用入口会从外部仓库读取源码和模型，不需要
把该仓库的定制 `rsl_rl` 安装进当前环境：

```zsh
conda activate /data/conda_envs/isaacsim51_3dgs_grasp
cd /home/light/workspace/IsaacLab

./isaaclab.sh -p \
  /mnt/sage_data/workspace/pct_scan/scripts/navigation/play_go2_moe_cts_keyboard.py \
  --robotlab-repo /mnt/sage_data/workspace/go2_rl_robotlab \
  --stair-height 0.19 \
  --stair-width 0.30 \
  --vx 0.80 \
  --vy 0.60 \
  --wz 0.80 \
  --warmup-steps 50 \
  --real-time
```

窗口出现后点击 Viewport。方向键控制前后和横向移动，`Z`/`X` 控制偏航，`L`
清零命令。该楼梯尺寸与仓库 `stairs_and_slope.xml` 的 0.19m 高、0.30m 深台阶
一致；机器人位于低平台中央，沿 `+X` 前进即上楼。2026-08-02 的 seed=0
headless 探针以 `vx=0.8m/s` 运行 300 个 50Hz 控制步，未触发 reset，前进约
2.26m、相对升高约 0.82m，已经跨过约四级台阶；同时出现约 0.33m 横向漂移，
所以这只是初步爬楼通过，不是多 seed 稳定验收。

该入口使用新仓库原生的标准 Go2 模型。2026-08-02 用户完成 GUI 交互测试，
确认该 policy 可以稳定上楼梯。这里必须区分两条用途不同的运行链：

- **无机械臂导航实验链**：标准 Go2 + MoE-CTS，只验证 PCT + SCAN Planner、
  RViz 和 Isaac 导航能力；不要求适配 X5 机械臂载荷。
- **mobile-manipulation 主 pipeline**：Go2-X5 + 原有 checkpoint，继续负责
  `navigate / pick / carry / place / LeRobot export` 完整流程，不替换为 MoE-CTS。

因此，下文的 MoE-CTS adapter 只服务无机械臂导航实验，不会修改
`run_full_physics_pipeline.py` 的 Go2-X5 policy 选择。

该结果只说明“底层 policy 独立爬楼”已经通过，尚不等于 PCT + SCAN 完整导航
通过。无机械臂导航实验的下一阶段会把键盘命令替换为 ROS 2 `/cmd_vel`，并
严格复用唯一写入者与超时停车安全门，数据流固定为：

```text
/cmd_vel
→ CmdVelToPolicyAdapter（限幅、变化率、超时与 supervisor 许可）
→ 标准 Go2 base_velocity command buffer
→ single_obs[45]
→ MoE-CTS TorchScript（内部历史 450）
→ 12 维腿部动作
→ Isaac Sim / PhysX
```

实验链接入后先用手工三维 Path 验证
`SCAN → /planning/bspline → /cmd_vel → MoE-CTS` 的平地、转弯与楼梯，
再启用 PCT 发布的 `/pct/global_path` 做完整跨楼层验收。这样若出现问题，可以
明确区分是全局路径、局部轨迹、速度跟踪还是 locomotion policy 导致的。

模型加载与接口检查集中在
`source/navigation/adapters/moe_cts_policy_adapter.py`：

- `MoeCtsPolicyContract` 固定 `single_obs=45`、内部历史 `450`、腿部动作 `12`；
- `MoeCtsPolicyAdapter` 延迟导入 Torch，并使用 `torch.jit.load()` 加载部署模型；
- 每次推理都拒绝错误 shape、NaN 和 Inf；
- `reset()` 会检查 10 帧历史确实清零，避免两个 episode 互相污染；
- `verify()` 会执行一次真实 forward，再清空历史后返回可记录的接口报告。

### 用 ROS 2 `/cmd_vel` 驱动标准 Go2 MoE-CTS

`play_go2_moe_cts_keyboard.py` 现在保留 `keyboard` 模式，同时增加了 `ros2`
模式。ROS 模式直接复用项目已有的 Isaac OGN DDS bridge，不会在 Isaac 的
Python 3.11 进程中导入 ROS Humble 的 Python 3.10 `rclpy`。ROS 模式还会发布
Odometry、临时 height-scanner 世界系点云，并订阅当前 `/initial_path`。标准
Go2 + MoE-CTS 支线不使用 Go2-X5 root-lock，而是持续发布与当前 Path 精确
绑定的 `StairExecutionFreeze.frozen=false` 心跳，使 SCAN 区分“明确未冻结”
和“状态发布器失活”。命令限幅、变化率限制、0.25s 超时停车、唯一写入权和
退出清零均已生效；supervisor、Odometry 和点云尚未作为该实验入口的 policy
运动许可前置条件，接入完整 controller 时仍须逐项收紧。

终端 1 启动 Isaac。必须先记住 Conda Python 的绝对路径，再 source ROS，最后
清掉 ROS Python 3.10 的 `PYTHONPATH`：

```zsh
cd /mnt/sage_data/workspace/pct_scan
conda activate /data/conda_envs/isaacsim51_3dgs_grasp

export PCT_SCAN_ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python
source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

$PCT_SCAN_ISAAC_PYTHON -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --command-source ros2 \
  --robotlab-repo /mnt/sage_data/workspace/go2_rl_robotlab \
  --cmd-vel-topic /cmd_vel \
  --reference-path-topic /initial_path \
  --stair-execution-frozen-topic /planning/stair_execution_frozen \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.80 \
  --policy-max-vy 0.50 \
  --policy-max-wz 0.80 \
  --stair-height 0.19 \
  --stair-width 0.30 \
  --warmup-steps 50 \
  --real-time
```

终端 2 使用同一个 domain 连续发布低速命令。先从 `vx=0.30m/s` 开始，不要一
上来使用 policy 的上限：

```zsh
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 topic pub --rate 10 \
  /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.30, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

终端 2 按 `Ctrl-C` 后，不需要再发一条零速；安全门会在 0.25s 内把 command
buffer 清零。终端 1 中应依次看到 `policy_warmup`、`missing_cmd_vel`、
`允许运动`，停止发布后看到 `cmd_vel_timeout`。若 Isaac 看不到 topic，先检查：

```zsh
ros2 topic info /cmd_vel --verbose
```

发布端与 Isaac 必须使用相同的 `ROS_DOMAIN_ID`。同时只启动一个向 `/cmd_vel`
发布非零速度的节点。

2026-08-03 的真实 DDS/GPU smoke 先启动发布端，再启动 Isaac：OGN
共接收 14 次 `Twist`，安全门向 command buffer 写入 501 次；从
`missing_cmd_vel` 进入允许运动，停止发布后进入 `cmd_vel_timeout` 并
清零。机器人在有限步数内沿 X 方向位移约 0.238m，证明这条边界已经
通过真实 DDS，但尚未代表 SCAN 轨迹跟踪或 PCT 跨层导航通过。

#### 手工 Path → SCAN B-spline 的完整五终端验收

该验收只打通 planner，不启动 `scan_controller`，因此 `/cmd_vel` 必须保持
零发布者，机器人不会运动。终端 1～4 是常驻进程，终端 5 只用于检查。所有
ROS 终端必须使用同一个 `ROS_DOMAIN_ID=71` 和 Fast DDS；除终端 1 外不要激活
Isaac Conda 环境。

终端 1 启动标准 Go2、MoE-CTS、原始 Odometry/点云以及 Path 绑定的未冻结
心跳：

```zsh
cd /mnt/sage_data/workspace/pct_scan
conda activate /data/conda_envs/isaacsim51_3dgs_grasp

export PCT_SCAN_ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python
source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

$PCT_SCAN_ISAAC_PYTHON -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --command-source ros2 \
  --robotlab-repo /mnt/sage_data/workspace/go2_rl_robotlab \
  --cmd-vel-topic /cmd_vel \
  --reference-path-topic /initial_path \
  --stair-execution-frozen-topic /planning/stair_execution_frozen \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.80 \
  --policy-max-vy 0.50 \
  --policy-max-wz 0.80 \
  --stair-height 0.19 \
  --stair-width 0.30 \
  --warmup-steps 50 \
  --real-time
```

终端 2 只启动接口归一化 bridge 和 `world → base_link` 动态 TF；此时禁止
组合 launch 自动启动 PCT、SCAN、controller 或 supervisor：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_scan:=false \
  start_controller:=false \
  start_pct:=false \
  start_supervisor:=false \
  start_manual_path:=false \
  start_odometry_tf:=true \
  initial_path_topic:=/initial_path \
  body_height_m:=0.342
```

终端 3 发布当前低平台上的三点地面高度 Path：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 run scan_navigation_tools \
  manual_path_publisher \
  --ros-args \
  --params-file \
  /mnt/sage_data/workspace/pct_scan/ros2_ws/src/scan_navigation_tools/config/go2_moe_cts_flat_path.yaml
```

终端 4 只启动 SCAN Planner。当前确定性楼梯低平台地面为 `z=-2.47m`，
height scanner 不是头部相机，因此覆盖地图地面高度、机体高度并禁用头相机
外参：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 run scan_planner scan_planner_node \
  --ros-args \
  --params-file \
  /mnt/sage_data/workspace/pct_scan/ros2_ws/src/scan_planner/config/planner.yaml \
  -p use_sim_time:=true \
  -p grid_map.body_height:=0.342 \
  -p grid_map.ground_height:=-2.47 \
  -p grid_map.need_extrinsic:=false \
  -r body_pose:=/body_pose \
  -r sensor_pose:=/body_pose \
  -r cloud:=/cloud_registered \
  -r initial_path:=/initial_path \
  -r planning/bspline:=/planning/bspline
```

终端 5 初始化一次环境后反复执行检查命令：

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
```

先确认未冻结心跳为单发布者、单订阅者，并分别读取当前 Path 与冻结状态的
identity：

```zsh
ros2 topic info /planning/stair_execution_frozen --verbose

ros2 topic echo \
  /initial_path \
  nav_msgs/msg/Path \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --field header \
  --once

ros2 topic echo \
  /planning/stair_execution_frozen \
  scan_planner_msgs/msg/StairExecutionFreeze \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once
```

预期 `frozen: false`、`sequence>0`，并且 `reference_path_stamp` 的
`sec/nanosec` 与 `/initial_path.header.stamp` 完全相同。随后读取 SCAN 输出：

```zsh
ros2 topic echo \
  /planning/bspline \
  scan_planner_msgs/msg/Bspline \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --no-arr \
  --once

ros2 topic info /cmd_vel --verbose
```

本阶段要求 B-spline 的 `order=3`、`emergency_stop=false`，同时 `/cmd_vel`
保持 `Publisher count: 0`。若 B-spline 仍未发布，再读取
`/planning/scan_status`，不要通过手工发布 `/cmd_vel` 绕过规划失败：

```zsh
ros2 topic echo \
  /planning/scan_status \
  scan_planner_msgs/msg/ScanPlanningStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once
```

正常 planner 隔离验收应看到 `event: 3`、`state: 2`、
`trajectory_present: true`、`stop_required: false`。这一步通过后再启动
`scan_controller`，避免把 planner 问题和速度跟踪问题混在同一次调试中。

#### 加入 SCAN controller 驱动 MoE-CTS

planner 隔离验收通过后，先在所有进程停止时构建一次新增的标准 Go2 profile：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh

colcon build \
  --symlink-install \
  --packages-select scan_controller

source install/setup.zsh
```

重新启动上节的终端 1～4，并保持它们运行。标准 Go2 profile 只覆盖
`body_height=0.342m`、50Hz 发布频率和保守速度包络；Path、B-spline、Odometry、
点云超时和 typed controller status 均继续复用主 `scan_controller`。

终端 5 在启动 controller 前记录初始位置。第二条命令一旦启动成功，机器人就会
开始沿低平台短 Path 运动；不要同时运行手工 `/cmd_vel` publisher：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 topic echo \
  /body_pose \
  nav_msgs/msg/Odometry \
  --qos-profile sensor_data \
  --field pose.pose.position \
  --once

ros2 launch scan_controller \
  go2_moe_cts_flat_controller.launch.py
```

终端 6 检查唯一写入者、控制器状态和实际位移：

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 topic info /cmd_vel --verbose

ros2 topic echo \
  /planning/controller_status \
  scan_planner_msgs/msg/ControllerStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /cmd_vel \
  geometry_msgs/msg/Twist \
  --once

ros2 topic echo \
  /body_pose \
  nav_msgs/msg/Odometry \
  --qos-profile sensor_data \
  --field pose.pose.position \
  --once
```

这里的 CLI reader 必须使用 `depth=1`。controller publisher 为审计保留最近
64 条 transient-local 状态；若使用 `--qos-depth 64 --once`，late join reader
会先收到缓存中最旧的启动快照并立即退出，看起来会停在
`WAITING_FOR_TRAJECTORY`，但这不代表 controller 当前仍在等待。

运动期间 `/cmd_vel` 必须恰好一个发布者、一个订阅者；最新 controller status
应有 `accepted: true`、`command_sample_count>0`、
`command_violation_count: 0`，状态在运动中为 `10`（TRACKING），完成后为
`12`（GOAL_REACHED），且三轴峰值不超过 `0.30/0.15/0.45`。机器人完成短 Path
后应接近 `x=0.75m` 并持续零速；可读取最终完成状态：

```zsh
ros2 topic echo \
  /planning/goal_reached \
  std_msgs/msg/Bool \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once
```

预期最终为 `data: true`。若机器人姿态异常，立即在终端 5 按 `Ctrl-C`；controller
发布者退出后，Isaac 侧独立的 `0.25s` 命令超时门会把 policy command buffer
清零。

2026-08-05 的真实闭环验收中，`/cmd_vel` 为唯一 `1→1` 链，机器人从此前约
`x=-0.04887m` 前进到 `(0.68562, -0.00304, -2.12145)m`，X 净位移约
`0.73449m`。相对 Path 地面终点 `(0.75, 0.02, -2.47)m`，XY 误差约
`0.06838m`，base 高度误差约 `0.00655m`，均满足 controller 的完成门；
`/planning/goal_reached=true`，到达后的 `/cmd_vel` 三轴严格为零。因此手工平地
`Path → SCAN → B-spline → controller → cmd_vel → MoE-CTS → PhysX` 已真实
通过。首次读取 controller status 得到 `status_sequence=1` 是上述 depth=64
历史读取行为，已修正文档命令，不是运行链故障。

#### 标准 Go2 + MoE-CTS 单跑楼梯闭环验收

平地闭环通过后，下一步使用同一确定性高度场完成一整跑纯物理楼梯。当前
Isaac 高度场的低平台地面为 `z=-2.47m`；live 原始点云确认第一处抬升约在
`x=1.40m`，每级升高 `0.19m`。需要注意：上游 HfTerrain 的水平网格为
`0.10m`，当前 `--stair-width=0.30` 经 `int()` 离散后实际踏面深度为
`0.20m`。`go2_moe_cts_stair_path.yaml` 按这个已经完成 policy 独立测试的真实
几何编写；本轮不同时修改楼梯和导航控制器。

开始前先在旧终端 1～6 中逐个按 `Ctrl-C`，确认旧的 `manual_path_publisher`、
`scan_planner_node` 和 `scan_controller` 全部退出。同一 ROS 图中不允许保留两套
Path 或 `/cmd_vel` 发布者。普通 `ros2 node list` 可能读取 daemon 中的旧图，
因此清场必须绕过 daemon 直接做一次 DDS 发现：

```zsh
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
```

节点列表应为空，`/cmd_vel` 应报告 `Unknown topic`。随后在普通 ROS 终端构建一次：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh

colcon build \
  --symlink-install \
  --packages-select \
  scan_navigation_tools \
  isaac_navigation_bridge \
  scan_planner \
  scan_controller

source install/setup.zsh
```

楼梯验收只需要两个常驻终端和一个检查终端。终端 1 启动 Isaac、MoE-CTS、
原始 Odometry/点云、`/cmd_vel` 订阅以及绑定 Path 的 `frozen=false` 心跳：

```zsh
cd /mnt/sage_data/workspace/pct_scan
conda activate /data/conda_envs/isaacsim51_3dgs_grasp

export PCT_SCAN_ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python
source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

$PCT_SCAN_ISAAC_PYTHON -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --command-source ros2 \
  --robotlab-repo /mnt/sage_data/workspace/go2_rl_robotlab \
  --cmd-vel-topic /cmd_vel \
  --reference-path-topic /initial_path \
  --stair-execution-frozen-topic /planning/stair_execution_frozen \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.80 \
  --policy-max-vy 0.50 \
  --policy-max-wz 0.80 \
  --stair-height 0.19 \
  --stair-width 0.30 \
  --warmup-steps 50 \
  --status-every 50 \
  --real-time
```

终端 3 先初始化检查环境，并在终端 2 尚未启动时记录起点。此终端不是发布者，
后续所有检查命令都继续在这里执行：

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 topic echo \
  /isaac/body_pose_raw \
  nav_msgs/msg/Odometry \
  --qos-profile sensor_data \
  --field pose.pose.position \
  --once
```

这里读取的是终端 1 直接发布的 Isaac 原始里程计；归一化后的 `/body_pose`
要等终端 2 中的 bridge 启动后才会出现。起点应接近
`(-0.05, 0.02, -2.128)`。现在才在终端 2 启动 ROS 导航侧。这个
专用 launch 一次启动 bridge、手工 33 点楼梯 Path、SCAN Planner、controller
和动态 TF；它明确不启动 PCT 与 supervisor：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  go2_moe_cts_stair_navigation.launch.py
```

机器人会先沿低平台接近楼梯，再连续上完整一跑，并在外侧顶平台
`ground=(4.40, 0.02, 0.0)m` 附近停车。若机器人失稳、Path 明显错位或即将
离开地形，立即在**终端 2**按一次 `Ctrl-C`；唯一 `/cmd_vel` 发布者退出后，
Isaac 侧独立超时门会在 `0.25s` 内清零。ROS 模式没有创建键盘控制器，因此
不要依赖 Viewport 的 `L` 键急停。

运动期间在终端 3 检查唯一写入者、物理楼梯未冻结状态以及 SCAN 输出：

```zsh
ros2 topic info /cmd_vel --verbose

ros2 topic echo \
  /planning/stair_execution_frozen \
  scan_planner_msgs/msg/StairExecutionFreeze \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /planning/scan_status \
  scan_planner_msgs/msg/ScanPlanningStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /planning/controller_status \
  scan_planner_msgs/msg/ControllerStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once
```

必须看到 `/cmd_vel` 恰好 `1 publisher → 1 subscription`、
`stair_execution_frozen.frozen=false`、SCAN 没有 emergency stop；运动中的
controller 状态应为 `TRACKING`。Isaac 终端的周期状态中 `resets` 必须始终
为 `0`。

到达顶平台后仍在终端 3 执行最终验收：

```zsh
ros2 topic echo \
  /planning/goal_reached \
  std_msgs/msg/Bool \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /planning/controller_status \
  scan_planner_msgs/msg/ControllerStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /body_pose \
  nav_msgs/msg/Odometry \
  --qos-profile sensor_data \
  --field pose.pose.position \
  --once

ros2 topic echo \
  /cmd_vel \
  geometry_msgs/msg/Twist \
  --once
```

通过条件为：`goal_reached=true`、controller `state=12`、
`accepted=true`、`trajectory_valid=true`、`emergency_stop=false`、
`command_violation_count=0`；最终 base 位置接近 `(4.40, 0.02, 0.342)m`，
XY 误差小于 `0.08m`、Z 误差小于 `0.12m`，到达后的 `/cmd_vel` 三轴严格为
零。只有这些条件和 Isaac `resets=0` 同时满足，才能把本轮记为
“手工 Path 下的纯物理单跑楼梯通过”；它仍不等于 PCT 跨楼层最终链通过。

#### 标准 Go2 + MoE-CTS 快速单跑楼梯复验

保守楼梯运行中若发现攀爬命令偏慢，先检查 `/cmd_vel` 的发布者数量。2026-08-05
首次准备提速时，普通 ROS 2 CLI daemon 曾报告两个同名 `scan_controller`；
随后 `--no-daemon` 直接发现证明所有进程退出后 daemon 仍会显示旧节点和 GID，
因此那次输出不能证明两个 controller 曾同时实时写入。性能验收必须用
`--no-daemon` 核对唯一发布者，不能依据 daemon 缓存判断重复写入。运行中实测
`vx≈0.236m/s` 仍证明保守轨迹偏慢，所以继续进行独立快速 profile A/B。

后续一次“能够上楼但不流畅、偶尔卡住”的运行已经由 `--no-daemon` 直接 DDS
发现确认存在**两套真实在线链**：`/cmd_vel` 有 2 个 `scan_controller` 发布者，
`/initial_path`、`/planning/bspline`、`/body_pose` 和 `/cloud_registered` 也各有
2 个发布者。这不是 daemon 缓存。两个 planner 的不同时间基准会被两个
controller 交叉接收，最终两路 Twist 同时进入一个 Isaac policy 订阅端，因此
该次表现不能作为 fast 或 redline 参数的有效结果。快速 launch、基础楼梯 launch
以及“基础 launch + redline YAML”三者在一次实验中只能选择一个，绝不能并行启动。
两个实例使用完全相同的节点名时，`ros2 param get /scan_planner_node ...` 也没有
唯一目标：每条命令可能由不同实例响应，因而会拼出一组实际上不存在的混合参数。
必须先把发布者数量恢复为 1，再读取参数和记录性能。

清场并只重启 fast launch 后，直接 DDS 复验已恢复为 5 个唯一导航节点：
`/cmd_vel` 为 `1→1`、`/initial_path` 为 1 个 publisher、
`/planning/bspline` 为 `1→1`。planner 参数为
`0.60/0.65/1.20/0.75/0.0075`，controller 参数为
`0.65/0.15/0.45/1.20`；这组单链运行才是有效的 fast profile 基线。

快速配置保留原来的 33 点 Path、点云过滤、碰撞包络、终点门、QoS 和超时停车，
只做以下 A/B 调整：巡航 `0.40→0.60m/s`、纵向硬上限
`0.45→0.65m/s`、纵向加速度 `0.80→1.20m/s²`、轨迹初始化有效加速度
`0.40→0.90m/s²`、速度滤波时间常数 `0.20→0.12s`、末段最低有效速度
`0.22→0.28m/s`。横移和转向包络不变；所有值仍低于 Isaac policy 的
`0.80m/s`、`2.0m/s²` 安全门。

先在当前两个 ROS launch 终端和 Isaac 终端分别按一次 `Ctrl-C`，等到每个进程
都返回 shell。随后用检查终端确认旧节点已经消失；不要直接叠加启动：

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
ros2 topic info /initial_path --verbose --no-daemon --spin-time 3
ros2 topic info /planning/bspline --verbose --no-daemon --spin-time 3
```

清场时不应再看到 `manual_path_publisher`、`scan_planner_node`、
`scan_controller`、`isaac_navigation_bridge` 或 `odometry_tf_broadcaster`；
`/cmd_vel`、`/initial_path` 和 `/planning/bspline` 都应报告 `Unknown topic`。
如果普通、不带 `--no-daemon` 的命令仍显示旧节点或 GID，只是 CLI daemon
缓存；可以执行 `ros2 daemon stop`，后续 CLI 会按需启动新的 daemon。然后在
一次性构建终端执行：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh

colcon build \
  --symlink-install \
  --packages-select \
  scan_navigation_tools \
  isaac_navigation_bridge \
  scan_planner \
  scan_controller

source install/setup.zsh
```

终端 1 重新启动 Isaac、MoE-CTS、原始 Odometry/点云和唯一 `/cmd_vel`
订阅者：

```zsh
cd /mnt/sage_data/workspace/pct_scan
conda activate /data/conda_envs/isaacsim51_3dgs_grasp

export PCT_SCAN_ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python
source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

$PCT_SCAN_ISAAC_PYTHON -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --command-source ros2 \
  --robotlab-repo /mnt/sage_data/workspace/go2_rl_robotlab \
  --cmd-vel-topic /cmd_vel \
  --reference-path-topic /initial_path \
  --stair-execution-frozen-topic /planning/stair_execution_frozen \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.80 \
  --policy-max-vy 0.50 \
  --policy-max-wz 0.80 \
  --stair-height 0.19 \
  --stair-width 0.30 \
  --warmup-steps 50 \
  --status-every 50 \
  --real-time
```

终端 3 在导航启动前确认新的 Isaac episode 回到原始起点：

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 topic echo \
  /isaac/body_pose_raw \
  nav_msgs/msg/Odometry \
  --qos-profile sensor_data \
  --field pose.pose.position \
  --once
```

起点应接近 `(-0.05,0.02,-2.128)m`。终端 2 只启动一次快速楼梯组合 launch：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  go2_moe_cts_stair_fast_navigation.launch.py
```

运动开始后在终端 3 验证快速参数真正生效，而且 `/cmd_vel` 恰好为 `1→1`：

```zsh
ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
ros2 topic info /initial_path --verbose --no-daemon --spin-time 3
ros2 topic info /planning/bspline --verbose --no-daemon --spin-time 3

ros2 param get /scan_planner_node fsm.reference_cruise_speed
ros2 param get /scan_planner_node manager.max_vel
ros2 param get /scan_planner_node manager.max_acc
ros2 param get /scan_planner_node manager.reference_profile_acceleration_scale
ros2 param get /scan_controller limits.max_vx
ros2 param get /scan_controller limits.max_ax

ros2 topic echo \
  /planning/stair_execution_frozen \
  scan_planner_msgs/msg/StairExecutionFreeze \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /planning/controller_status \
  scan_planner_msgs/msg/ControllerStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /cmd_vel \
  geometry_msgs/msg/Twist \
  --once
```

参数依次应为 `0.60、0.65、1.20、0.75、0.65、1.20`；直接发现必须同时看到
每种导航节点各 1 个、`/cmd_vel` 1 个 controller publisher → 1 个 Isaac
subscriber、`/initial_path` 1 个 publisher、`/planning/bspline` 1 个 publisher、`frozen=false`、
`emergency_stop=false`，Isaac 周期状态中的 `resets` 始终为 0。若出现两个
publisher、机器人俯仰失稳或即将越界，立即在终端 2 按一次 `Ctrl-C`。

到达顶平台后仍在终端 3 执行最终验收：

```zsh
ros2 topic echo \
  /planning/goal_reached \
  std_msgs/msg/Bool \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /planning/controller_status \
  scan_planner_msgs/msg/ControllerStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /body_pose \
  nav_msgs/msg/Odometry \
  --qos-profile sensor_data \
  --field pose.pose.position \
  --once

ros2 topic echo \
  /cmd_vel \
  geometry_msgs/msg/Twist \
  --once
```

最终合同不因提速而放宽：`goal_reached=true`、controller `state=12`、
`accepted=true`、`trajectory_valid=true`、`emergency_stop=false`、
`command_violation_count=0`，base 接近 `(4.40,0.02,0.342)m`，到达后 Twist
六轴严格为零，并且 Isaac `resets=0`。

#### 实际 0.30 m 踏面与用户高速包络楼梯验证

当唯一 fast 运行链仍在楼梯边缘卡住时，先验证楼梯几何，不继续盲目
提高速度。Isaac Lab 当前高度场的 `horizontal_scale=0.10m`，而楼梯
生成器使用 `int(step_width / horizontal_scale)` 离散踏面。因此请求
`--stair-width 0.30` 会因浮点表示向下截断为 2 格，实际只有 `0.20m`；
本 A/B 必须请求 `0.31m`，才会生成 3 格、实际 `0.30m` 踏面。启动
脚本会打印请求值、格数和实际值，不再依赖目测。

踏面变宽后，固定 8 m 子地形由 13 次抬升变为 9 次抬升；低平台
地面也从 `-2.47m` 变为 `-1.71m`。专用
`go2_moe_cts_stair_wide_path.yaml` 已用 25 个地面高度点表示新几何。
`go2_moe_cts_stair_wide_fast_tuning.yaml` 现保留用户选定的高速包络：
planner 速度/加速度上限为 `1.00/1.50`，controller 上限为
`vx=1.50`、`vy=0.50`、`wz=0.80`、`ax=2.50`。常规 reference
巡航目标也使用用户当前值 `1.00m/s`，航向修复不会暗中回退速度。

本配置还只在当前 Path 进度前后各 `0.45m` 的窗口内检测坡角不小于
`0.45rad` 的陡升段。检测到楼梯后，机体 yaw 跟随完整 Path 的水平合成
切向；SCAN B-spline 的世界系 XY 速度保持不变，所以几厘米横向重接由
`vy` 完成，不再把近 `±90°` 的人工回接短弦转换成饱和 `wz`。该开关只在
wide-fast 楼梯覆盖层启用，不是全局 `wz=0`，平地与 Path 真实转弯仍由
B-spline/Path 朝向正常控制。

将 planner 上限改为 `1.00/1.50` 后，旧
`manager.feasibility_tolerance=0.0075` 会令速度余量变为
`0.0076>0.005`，导致 SCAN 在运动前报错“连续可行性容差不得宽于
最终动态采样门”。现改为 `0.0045`：速度余量为
`1.00×0.0045+0.0001=0.0046≤0.005`，加速度余量为
`1.50×0.0045+0.0001=0.00685≤0.01`。不关闭安全门，只缩紧与新上限
匹配的相对容差。

2026-08-06 实机闭环进度：用户已按本节参数完成 Isaac Sim/PhysX 复验，
确认 `0.30m` 有效踏面上楼通过且过程较流畅。由此，标准 Go2 MoE-CTS
实验链的“手工三维 Path → SCAN B-spline → 唯一 `/cmd_vel` → RL policy →
PhysX 楼梯”已通过。该结论不扩展到 PCT 在线全局 Path、动态障碍或带机械臂
Go2-X5 主 pipeline；下一阶段将停用 `manual_path_publisher`，接入
`/pct/global_path → /initial_path` 并统一在 RViz 中验收完整跨楼层路线。

先在所有旧 ROS launch 和 Isaac 终端按 `Ctrl-C`，等待每个进程真正返回
shell。终端 0 用直接 DDS 发现检查清场：

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
ros2 topic info /initial_path --verbose --no-daemon --spin-time 3
ros2 topic info /planning/bspline --verbose --no-daemon --spin-time 3
```

不应再看到 `manual_path_publisher`、`scan_planner_node`、`scan_controller`、
`isaac_navigation_bridge` 或 `odometry_tf_broadcaster`；三个 topic 应为
`Unknown topic`。随后在一次性构建终端执行：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh

colcon build \
  --symlink-install \
  --packages-select \
  scan_navigation_tools \
  isaac_navigation_bridge \
  scan_planner \
  scan_controller

source install/setup.zsh
```

终端 1 启动 Isaac、MoE-CTS、仿真 Odometry/点云与唯一 `/cmd_vel`
订阅者。注意这里必须是 `--stair-width 0.31`：

```zsh
cd /mnt/sage_data/workspace/pct_scan
conda activate /data/conda_envs/isaacsim51_3dgs_grasp

export PCT_SCAN_ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python
source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

$PCT_SCAN_ISAAC_PYTHON -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --command-source ros2 \
  --robotlab-repo /mnt/sage_data/workspace/go2_rl_robotlab \
  --cmd-vel-topic /cmd_vel \
  --reference-path-topic /initial_path \
  --stair-execution-frozen-topic /planning/stair_execution_frozen \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 1.50 \
  --policy-max-vy 0.50 \
  --policy-max-wz 0.80 \
  --policy-max-vx-rate 2.50 \
  --policy-max-vy-rate 0.80 \
  --policy-max-wz-rate 1.50 \
  --stair-height 0.19 \
  --stair-width 0.31 \
  --warmup-steps 50 \
  --status-every 50 \
  --real-time
```

终端 1 必须出现以下关键日志；若 `cells` 或 `effective` 不符，不启动
导航：

```text
[INFO] 楼梯踏面离散：requested=0.310m, horizontal_scale=0.100m, cells=3, effective=0.300m
```

终端 3 先确认新 episode 的起点。由于 base 高度约为 `0.342m`，预期
`z≈-1.71+0.342=-1.368m`：

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 topic echo \
  /isaac/body_pose_raw \
  nav_msgs/msg/Odometry \
  --qos-profile sensor_data \
  --field pose.pose.position \
  --once
```

起点应接近 `(-0.05,0.02,-1.368)m`。终端 2 只启动一次宽踏面 fast
组合 launch：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  go2_moe_cts_stair_wide_fast_navigation.launch.py
```

运动开始后，终端 3 检查“新几何 + 用户高速参数 + 唯一写入者”：

```zsh
ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
ros2 topic info /initial_path --verbose --no-daemon --spin-time 3
ros2 topic info /planning/bspline --verbose --no-daemon --spin-time 3

ros2 param get /scan_planner_node grid_map.ground_height
ros2 param get /scan_planner_node fsm.reference_cruise_speed
ros2 param get /scan_planner_node manager.max_vel
ros2 param get /scan_planner_node manager.max_acc
ros2 param get /scan_planner_node manager.reference_profile_acceleration_scale
ros2 param get /scan_planner_node manager.feasibility_tolerance
ros2 param get /scan_controller limits.max_vx
ros2 param get /scan_controller limits.max_ax
ros2 param get /scan_controller controller.stair_heading_lock_enabled
ros2 param get /scan_controller controller.stair_heading_lock_half_window_arc_m
ros2 param get /scan_controller controller.stair_heading_lock_min_pitch_rad
ros2 param get /scan_controller controller.stair_forward_speed_floor

ros2 topic echo \
  /planning/controller_status \
  scan_planner_msgs/msg/ControllerStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /cmd_vel \
  geometry_msgs/msg/Twist \
  --once
```

参数依次应为
`-1.71、1.00、1.00、1.50、0.75、0.0045、1.50、2.50、true、0.45、0.45、1.00`；
`/cmd_vel` 必须是 1 个 controller publisher → 1 个 Isaac subscriber，
`/initial_path` 和 `/planning/bspline` 各只有 1 个 publisher。Isaac 终端中
`resets` 必须始终为 0。宽踏面 Path 用 `0.02m` 水平跨距描述
`0.19m` 立面，三维 B-spline 在该段的大部分速度因此落在 z 轴，而 Go2
policy 只消费平面 `vx/vy/wz`。楼梯牵引门会在完整 Path 的已认证陡升窗口内，
仅当 B-spline 仍明确要求向前、机器人未进入横向恢复且未进入终点制动时，
把 Path 切向速度补到 `1.00m/s`。零速、反向、急停、输入超时和终点制动
保持原有严格行为，因此不能另起固定速度 publisher。

运行过程中可让下面命令持续输出每条轨迹的峰值；
修复前多条轨迹会达到 `max_abs_wz=0.8`，本轮重点检查该饱和是否消失：

```zsh
ros2 topic echo \
  /planning/controller_status \
  scan_planner_msgs/msg/ControllerStatus \
  --qos-depth 64 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --field max_abs_wz
```

另开同一检查终端观察实际楼梯命令。进入楼梯窗口并完成加速度爬升后，正常
前向轨迹的 `linear.x` 应接近 `1.00m/s`；短暂低于该值只允许出现在限加速、
横向恢复、明确制动或终点收口期间：

```zsh
ros2 topic echo \
  /cmd_vel \
  geometry_msgs/msg/Twist
```

到达顶平台后，终端 3 执行最终验收：

```zsh
ros2 topic echo \
  /planning/goal_reached \
  std_msgs/msg/Bool \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /planning/controller_status \
  scan_planner_msgs/msg/ControllerStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once

ros2 topic echo \
  /body_pose \
  nav_msgs/msg/Odometry \
  --qos-profile sensor_data \
  --field pose.pose.position \
  --once

ros2 topic echo \
  /cmd_vel \
  geometry_msgs/msg/Twist \
  --once
```

通过条件是：机器人不在踏面边缘长时卡住，能连续完成 9 次抬升；
`goal_reached=true`、controller `state=12`、`accepted=true`、
`trajectory_valid=true`、`emergency_stop=false`、`command_violation_count=0`；base
最终接近 `(4.40,0.02,0.342)m`，到达后 Twist 六轴为零，Isaac
`resets=0`。若这组通过而实际 0.20 m 踏面组仍卡住，就有较强证据
说明主要瓶颈是地形几何，而不是继续放大 SCAN 速度上限。

#### PCT 双 Topic 只规划与 RViz 验收

手工三维 Path 的物理楼梯基线通过后，下一步先只接入真实 PCT，不立刻让
controller 驱动机器狗。当前 MoE-CTS 楼梯脚本使用的是 RobotLab 确定性楼梯，
而 PCT 使用 `source/scene/multifloor` 的真实 PLY/tomogram；两者尚未放在同一
仿真场景和坐标原点中。若现在同时启动运动，Path 接口问题、地图配准问题和
低层跟踪问题会混在一起。

本阶段链路为：

```text
真实 upstream PCT 资产 + 测试起点/跨层目标
→ pct_ros2_adapter
→ /pct/global_path（PCT 原始结果，供审计和 RViz 显示）
→ /initial_path（SCAN 的稳定输入接口）
```

两个 Topic 不是由两个规划器生成。`pct_ros2_adapter` 把**同一个 Path 消息
对象**发布到两个 reliable + transient-local publisher，成功 Path 和撤销旧
路径的空 tombstone 都保持相同 stamp、frame、点列和 ground-height 语义。
这样既能看见 PCT 的原始输出，也不会因为重新构造或转发消息破坏 SCAN 的
Path 代际身份。若两个 Topic 配成同名，节点会拒绝启动。

先在所有旧 Isaac 和 ROS launch 终端按 `Ctrl-C`，确认各进程已经返回 shell。
终端 0 做清场检查：

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /pct/global_path --verbose --no-daemon --spin-time 3
ros2 topic info /initial_path --verbose --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
```

不应再看到旧的 `manual_path_publisher`、`scan_planner_node`、
`scan_controller` 或 `pct_ros2_adapter`。上述三个 Topic 此时都应为
`Unknown topic`。终端 0 随后完成一次性构建：

```zsh
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh

colcon build \
  --symlink-install \
  --packages-select \
  scan_planner_msgs \
  pct_ros2_adapter \
  navigation_visualization

source install/setup.zsh
```

终端 1 只启动静态 PLY 和 RViz，不启动手工 Path 或 Go2 模型：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  start_manual_path:=false \
  start_go2_model:=false \
  use_sim_time:=true
```

终端 2 启动真实 upstream PCT adapter。这里的 `0.30m` 是现有真实资产探针
起点/终点的历史高度合同，仅用于本次只规划验收；后续把真实 Go2 放进
multifloor 场景时仍使用 live 标定的生产值约 `0.338m`：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch pct_ros2_adapter \
  pct_ros2_adapter.launch.py \
  body_height_m:=0.30 \
  path_topic:=/pct/global_path \
  scan_path_topic:=/initial_path
```

终端 3 运行一次真实资产生命周期探针。它临时发布 `/clock`、`/body_pose`
和带有正确二楼 z 的 `/pct/goal`，等待规划完成后自动退出；本步骤不需要另开
Isaac 终端：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_upstream_pct_ros_lifecycle.py
```

当前资产的通过输出应包含：

```text
UPSTREAM_PCT_ROS_LIFECYCLE_OK ... points=141 scan_alias_points=141 ...
vertical_span_m=3.209145 ... stair_anchor_xy_error_m=0.000000000
```

探针不仅比较点数，还会序列化比较两个 Path 的完整 payload，并检查起终点、
地面高度、跨层高度跨度和标定楼梯中心线。不要用 RViz 的 `2D Goal Pose` 做
本次跨楼层验收：该工具默认给出 `z=0`，无法表达二楼 base 高度。

终端 4 在探针退出后检查持久缓存的结果与安全边界：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 topic info /pct/global_path --verbose --no-daemon --spin-time 3
ros2 topic info /initial_path --verbose --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3

ros2 topic echo \
  /pct/global_path \
  nav_msgs/msg/Path \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --no-arr \
  --once

ros2 topic echo \
  /initial_path \
  nav_msgs/msg/Path \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --no-arr \
  --once

ros2 topic echo \
  /pct/planning_status \
  scan_planner_msgs/msg/PCTPlanningStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once
```

通过条件是两个 Path Topic 都只有一个名为 `pct_ros2_adapter` 的 publisher，
QoS 都是 reliable + transient local，Path header/stamp 和 141 点长度一致，
planning status 为成功；`/cmd_vel` 必须仍是 `Unknown topic`。RViz 中青色粗线
`PCT Global Path` 与暗红色 `SCAN Input Path` 应完全重合。此时只证明真实
PCT 已安全接到 SCAN 的输入边界，尚未证明 SCAN B-spline、机器人/地图配准
或 PCT 跨层物理执行；下一阶段才会把标准 Go2 与在线点云放进同一个
multifloor 场景并启动 SCAN，controller 仍先保持关闭。

#### PCT → SCAN 只规划 B-spline 验收

在上一节只验证双 Path Topic 之后，本节再启动生产
`scan_planner_node`，但仍然不启动 `scan_controller`、supervisor 或机器人。
本阶段链路是：

```text
真实 multifloor PCT tomogram / collision PLY
→ pct_ros2_adapter
→ /pct/global_path + /initial_path
→ scan_planner_node
→ /planning/bspline + /optimal_list
```

验收探针会临时直接提供已符合桥输出合同的 `/body_pose`、
`/cloud_registered`、`/clock` 和 `/pct/goal`。因此组合 launch 必须使用
`start_bridge:=false`，否则 bridge 与探针会成为同一规范化 Topic 的两组
publisher。这个探针使用合成自由空间射线和理想 Odometry，所以它验证的是
“真实 PCT 路线能否进入生产 SCAN 并生成合法局部轨迹”，不是 Isaac
场景点云或物理跟踪的替代。

先在所有旧 Isaac 和 ROS 2 launch 终端按 `Ctrl-C`。终端 0 清场并构建：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3

colcon build \
  --symlink-install \
  --packages-select \
  scan_planner_msgs \
  pct_ros2_adapter \
  scan_planner \
  isaac_navigation_bridge \
  navigation_visualization

source install/setup.zsh
```

终端 1 只启动静态 PLY 和 RViz：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  start_manual_path:=false \
  start_go2_model:=false \
  use_sim_time:=true
```

终端 2 只启动 PCT 和 SCAN Planner：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_bridge:=false \
  start_scan:=true \
  start_controller:=false \
  start_manual_path:=false \
  start_pct:=true \
  start_supervisor:=false \
  start_odometry_tf:=false \
  body_height_m:=0.338 \
  pct_backend_kind:=upstream
```

终端 2 的节点列表中应只增加 `/pct_ros2_adapter` 和
`/scan_planner_node`；不应出现 `isaac_navigation_bridge`、`scan_controller` 或
`navigation_supervisor`。

终端 3 运行一次确定性探针，通过或失败后会自动退出：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_real_pct_to_scan_chain.py \
  --planning-only
```

当前真实资产的通过输出为：

```text
PCT_SCAN_PLANNING_ONLY_OK path_points=142 scan_alias_points=142
spline_points=35 spline_order=3 spline_final=False
spline_duration_s=3.002090 spline_velocity_upper_bound_mps=0.544554
cmd_nonzero=0 cmd_publishers=0 ...
```

`142` 与上一节生命周期探针的 `141` 不是丢点：两个探针使用了不同的
已标定终点合同。本探针还会逐字段比较 `/pct/global_path` 和
`/initial_path`，检查 B-spline 的 Path stamp、轨迹诊断、地图融合证据与跨层
z 跨度，并严格要求 `/cmd_vel` 的 publisher 数为 0。

终端 4 在终端 3 退出后检查持久化 Path/B-spline 和安全边界：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3

ros2 topic info \
  /pct/global_path \
  --verbose \
  --no-daemon \
  --spin-time 3

ros2 topic info \
  /initial_path \
  --verbose \
  --no-daemon \
  --spin-time 3

ros2 topic info \
  /planning/bspline \
  --verbose \
  --no-daemon \
  --spin-time 3

ros2 topic info \
  /cmd_vel \
  --verbose \
  --no-daemon \
  --spin-time 3

ros2 topic echo \
  /planning/bspline \
  scan_planner_msgs/msg/Bspline \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --no-arr \
  --once

ros2 topic echo \
  /planning/scan_status \
  scan_planner_msgs/msg/ScanPlanningStatus \
  --qos-depth 1 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  --once
```

通过条件是：`/initial_path` 只由 PCT 发布并被 SCAN 订阅；
`/planning/bspline` 只由 SCAN 发布；`/cmd_vel` 是 `Unknown topic`。RViz 中
青色 PCT Path 和暗红色 SCAN Input Path 完全重合，`/optimal_list` 显示的
SCAN 局部轨迹从起点沿全局路线向前展开。验收后先在终端 2 按
`Ctrl-C`，再关闭 RViz。

如果需要用图而不是长篇 `ros2 topic info` 查看连接，保持终端 1、2 运行，
另开终端 5 启动系统已安装的 `rqt_graph`：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 run rqt_graph rqt_graph
```

在 `rqt_graph` 窗口中：

1. 左上角选择 `Nodes/Topics (all)`，因为当前
   `/planning/bspline` 没有 controller subscriber，只看 active 连接可能把它隐藏。
2. 取消 `Hide: Dead sinks` 和 `Hide: Leaf topics`，然后点左上角刷新按钮。
3. Humble 版的两个过滤框**没有可见文字标签**。实时窗口顶部从左向右是：

   ```text
   [刷新] [Nodes/Topics (all)] [Node/namespace filter] [Topic filter] [保存按钮]
   ```

   也就是下拉框右边第一个输入框是 Node/namespace filter，紧挨着的
   第二个空白输入框才是 Topic filter。鼠标悬停后，两者分别显示
   `Namespace filter...` 和 `Topic filter...` 提示。保存出来的 DOT、SVG、PNG
   只包含图画布，不包含这行 GUI 控件；过滤必须回到实时 `rqt_graph`
   窗口执行。
4. 先不设过滤观察全图；若要只看当前主链，Node/namespace filter 输入：

   ```text
   /pct_ros2_adapter,/scan_planner_node,/navigation_rviz
   ```

   Topic filter 输入：

   ```text
   /pct/global_path,/initial_path,/planning/bspline,/optimal_list
   ```

   输入后按 Enter 或点刷新。
5. 把鼠标放到节点上可高亮它的入边和出边；工具栏可将当前图保存为
   DOT、SVG 或图片。

如果在**实时窗口**中连顶部这行控件都看不到，先最大化窗口。仍然不显示时
关闭它，然后用下面命令只重置 `rqt` GUI 布局与过滤偏好；它不会停止或修改
ROS 节点、Topic 或本项目文件：

```zsh
ros2 run rqt_graph rqt_graph --clear-config
```

本次输出中 `/planning/bspline` 的 subscription count 为 0 是预期行为：
`scan_controller` 被刻意关闭，RViz 也不直接解析自定义 `Bspline` 消息，而是订阅
SCAN 另外发布的可视化 Marker Topic `/optimal_list`。因此 `rqt_graph` 中会同时看到
“原始 B-spline 数据接口”和“RViz Marker 显示接口”两条不同的边。

#### Isaac multifloor 真实观测 → PCT → SCAN 只规划验收

本阶段已经把上一节的合成 Odometry/自由空间点云替换为真实 Isaac Lab
物理状态和 RayCaster 点云，同时继续关闭 controller。完整链路为：

```text
multifloor collision USD + 标准 Go2 + MoE-CTS 零速站立
→ /isaac/body_pose_raw + /isaac/cloud_registered_raw
→ isaac_navigation_bridge
→ /body_pose + /cloud_registered
→ upstream PCT
→ /pct/global_path + /initial_path
→ SCAN Planner
→ /planning/bspline
```

这里有两个重要实现细节：

1. RobotLab 单层高度场原本从 base 上方 `20 m` 向下投射高度射线。在封闭
   multifloor mesh 中会先命中 `z≈6~7 m` 的屋顶，而不是当前楼层。multifloor
   模式现在把射线原点改为 base 上方 `1.0 m`，确定性楼梯模式仍保留官方原值。
2. 创建 OGN ROS 图会推进若干 Kit 帧。脚本在 OGN 初始化后重新执行一次
   environment reset 并清空 CTS 历史，保证首帧 Odometry、PCT 起点和物理
   episode 都从同一个标定位姿开始。

先在所有旧 ROS launch 和 Isaac 终端按 `Ctrl-C`，等待它们真正返回 shell。
终端 0 做清场与一次性构建：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /initial_path --verbose --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3

colcon build \
  --symlink-install \
  --packages-select \
  scan_planner_msgs \
  pct_ros2_adapter \
  scan_planner \
  isaac_navigation_bridge \
  navigation_visualization \
  go2_description

source install/setup.zsh
```

清场时不应看到旧的 `pct_ros2_adapter`、`scan_planner_node`、
`scan_controller` 或 Isaac OGN 节点。若 `ros2 node list` 对同名节点给出
warning，不要继续启动另一份；回到对应旧终端按 `Ctrl-C`。本阶段严格要求
`/cmd_vel` 没有 publisher，也没有 subscriber。

终端 1 启动 bridge、PCT、SCAN 和 Odometry TF；controller 与 supervisor
保持关闭：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_bridge:=true \
  start_scan:=true \
  start_controller:=false \
  start_manual_path:=false \
  start_pct:=true \
  start_supervisor:=false \
  start_odometry_tf:=true \
  body_height_m:=0.338 \
  pct_backend_kind:=upstream
```

终端 2 启动 PLY、RViz 和 Go2 模型。这里的关节姿态是
`joint_state_publisher` 提供的站姿，但 `world → base_link` 位姿来自终端 1 的
真实 Isaac Odometry：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  start_manual_path:=false \
  start_go2_model:=true \
  go2_use_gui:=false \
  use_sim_time:=true
```

终端 3 启动真实 multifloor collision、标准 Go2 和 MoE-CTS。必须先保存
Isaac Python 路径，再 source ROS 环境，最后 `unset PYTHONPATH`；否则 Python
3.11 会误加载 ROS Humble 的 Python 3.10 包：

```zsh
cd /mnt/sage_data/workspace/pct_scan

export ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python

source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

"$ISAAC_PYTHON" -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --terrain-mode multifloor \
  --multifloor-ray-origin-offset-z 1.0 \
  --navigation-planning-only \
  --command-source ros2 \
  --ros-domain-id 71 \
  --real-time \
  --warmup-steps 100 \
  --status-every 100
```

`--navigation-planning-only` 不只是把速度设为零：它根本不创建 `/cmd_vel`
订阅，并且每个 policy 控制拍显式清空 command buffer。启动后脚本会发布真实
Odometry/点云，等待 100 个控制步，再向 `/pct/goal` 发布已标定二楼目标；若
尚未收到 Path，会保持同一 goal stamp 做有限重试。

终端 4 运行只读整链探针。它不发布 `/clock`、Odometry、点云、Path、goal 或
Twist，因此不会替换任何真实输入：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_live_multifloor_pct_scan_chain.py \
  --timeout-sec 45
```

当前实测通过输出为：

```text
LIVE_MULTIFLOOR_PCT_SCAN_OK elapsed_s=0.015 raw_points=187
canonical_points=147 map_accepted_endpoints=147 path_points=142
vertical_span_m=3.208873 spline_points=35 spline_order=3 traj_id=1
start_xy_error_m=0.019410 start_z_error_m=0.005308
cmd_publishers=0 cmd_subscriptions=0
```

探针会检查 raw/canonical Odometry 同 stamp、raw/canonical 点云同 stamp、
`ray_endpoint_type`、单一 publisher、PCT 与 `/initial_path` 逐字段相等、跨层
高度、B-spline 代际、GridMap 融合、`frozen=false` 和 `/cmd_vel` 零端点。
点数从 `187` 变成 `147` 是 bridge 移除机器人自身点并分类支撑面后的正常
结果，不是点云丢失故障。

若希望同时看 ROS 图，终端 5 可选运行：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 run rqt_graph rqt_graph
```

主链过滤值可使用：

```text
Node/namespace filter:
/isaac_navigation_bridge,/pct_ros2_adapter,/scan_planner_node,/odometry_tf_broadcaster,/navigation_rviz

Topic filter:
/isaac/body_pose_raw,/isaac/cloud_registered_raw,/body_pose,/cloud_registered,/pct/goal,/pct/global_path,/initial_path,/planning/bspline
```

验收结束按终端 3（Isaac）→终端 1（规划 launch）→终端 2（RViz）的顺序
`Ctrl-C`。本阶段已经证明真实场景、真实物理位姿、真实 RayCaster 点云、PCT
和 SCAN 的几何/时间/Topic 合同成立；机器人仍未执行 PCT 路线。下一小步才
会启用唯一 `scan_controller` 和 `/cmd_vel` subscriber，先设置低速与急停门，
执行首段平地，再逐步放行楼梯段。

#### PCT → SCAN → MoE-CTS 首段平地真实执行

跨层只规划探针通过后，先执行同一条生产 PCT 路线的前 `1.507 m` 平地段，
不要直接放行完整楼梯。短目标取自已审计 142 点跨层 Path 的第 9 个地面点，
base 目标为 `(-3.103225, 5.063706, 0.184312)`；PCT 仍负责生成该同层路线，
SCAN 仍负责在线 B-spline 和闭环控制。执行探针只订阅证据，不发布 Twist；它
必须同时看到唯一 `/cmd_vel` 一发一收、真实位移、无命令违规和严格
`GOAL_REACHED` 才通过。

先在上一轮终端按 `Ctrl-C`，等待所有进程返回。终端 0 清场并构建：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3

colcon build \
  --symlink-install \
  --packages-select \
  scan_planner_msgs \
  pct_ros2_adapter \
  scan_planner \
  scan_controller \
  isaac_navigation_bridge \
  navigation_visualization \
  go2_description

source install/setup.zsh
```

终端 1 启动唯一 bridge、PCT、SCAN 和 controller；supervisor 仍关闭，避免在
首段运动验收中同时引入全局重规划状态机：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_bridge:=true \
  start_scan:=true \
  start_controller:=true \
  start_manual_path:=false \
  start_pct:=true \
  start_supervisor:=false \
  start_odometry_tf:=true \
  body_height_m:=0.338 \
  pct_backend_kind:=upstream
```

终端 2 启动 PLY、RViz 和 Go2 模型：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  voxel_leaf_size_m:=0.08 \
  start_manual_path:=false \
  start_go2_model:=true \
  go2_use_gui:=false \
  use_sim_time:=true
```

默认 RViz 现在只显示机器狗附近的实时点云：`Online SCAN Cloud` 订阅
`/cloud_registered`，`Decay Time=0` 表示只保留最新一帧；整张
`Static PLY Map`（`/map/ply`）默认关闭，但 Topic 和显示项仍保留，需要检查全局
地图配准时可在 Displays 中临时勾选。已经打开的 RViz 不会自动重载配置，修改后
只需重启本终端 2。`voxel_leaf_size_m` 只改变整张 PLY 的采样密度，不能形成随
机器人移动的局部窗口；不要为了减少 RViz 点数而改动 SCAN 实际使用的
`/cloud_registered`。在有效 `/initial_path` 建立前，bridge 会按支撑面安全合同
暂不发布 canonical 局部点云，此时局部显示短暂无数据是正常现象。

终端 3 必须在 Isaac 之前启动只读执行探针，确保它捕获本轮 episode 的真实
标定起点，而不是机器人已经移动后的中途位置：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_live_multifloor_pct_scan_chain.py \
  --expect-flat-execution \
  --minimum-planar-displacement-m 0.25 \
  --timeout-sec 120
```

终端 4 最后启动 Isaac。这里刻意删除 `--navigation-planning-only`，并让 policy
适配器的三轴速度/变化率门与当前生产 controller 完全一致：

```zsh
cd /mnt/sage_data/workspace/pct_scan

export ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python

source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

"$ISAAC_PYTHON" -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --terrain-mode multifloor \
  --multifloor-ray-origin-offset-z 1.0 \
  --command-source ros2 \
  --ros-domain-id 71 \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.55 \
  --policy-max-vy 0.15 \
  --policy-max-wz 0.60 \
  --policy-max-vx-rate 0.80 \
  --policy-max-vy-rate 0.40 \
  --policy-max-wz-rate 1.50 \
  --multifloor-goal-x -3.1032249450683596 \
  --multifloor-goal-y 5.063706207275391 \
  --multifloor-goal-z 0.18431236715521436 \
  --multifloor-goal-yaw -1.3217018692542983 \
  --real-time \
  --warmup-steps 100 \
  --status-every 100
```

执行模式与只规划模式都必须在约第 100 个控制步打印：

```text
[PCT_GOAL] 已发布 multifloor base 目标
```

随后 `/pct/planning_status` 应从 `IDLE` 进入规划成功，`/initial_path` 不再是
空 tombstone，bridge 才开始发布依赖有效 Path 支撑面语义的
`/cloud_registered`。phase257 首次执行曾把 goal 发布错误限制在
`--navigation-planning-only` 分支，表现为 raw Odometry/点云持续存在、
canonical 点云为 0、PCT 永远 IDLE、controller 永远等待轨迹且 policy 命令
全零；现已改为以 `terrain_mode == multifloor` 为条件，两种模式共享目标发布，
只有只规划模式清零 command buffer。出现上述旧症状时必须结束完整终端 1～4
并重新启动，不能只重启探针复用旧 transient 状态。

严格通过标志为：

```text
LIVE_MULTIFLOOR_PCT_SCAN_FLAT_EXECUTION_OK
```

若机器人已经到达短目标，而探针仅因在线重规划期间的
`SCAN typed 状态与 B-spline traj_id 不一致` 退出，可保持终端 1、2、4 继续运行，
另开终端 5 做一次只读晚加入补验：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_live_multifloor_pct_scan_chain.py \
  --expect-flat-execution \
  --allow-late-join \
  --minimum-planar-displacement-m 0.25 \
  --timeout-sec 30
```

晚加入模式仍要求同一 Path 的完整 B-spline identity、移动命令历史、严格
`GOAL_REACHED`、真实终点位移和唯一 `/cmd_vel` 一发一收；它只把已知标定起点
作为位移原点，成功行会明确打印 `start_observation_verified=false`，不能替代
下一轮“探针先于 Isaac”启动的严格起点采集。正常运行时不要添加该参数。

controller 确认 `GOAL_REACHED` 后，Isaac 会停止为已完成 Path 继续发布
`frozen=false` 心跳，避免 SCAN 在清除目标后持续打印
`reference_path_identity_mismatch`；收到新 Path 时心跳会自动重新启用。

该标志只会在 controller 已发布 `GOAL_REACHED` 后出现，所以机器人此时应持续
收到零速度。若需要人工急停，优先在终端 1 按 `Ctrl-C` 停止唯一 controller，
Isaac 入口会在 `0.25 s` 后因 `cmd_vel_timeout` 自动清零；随后再停止终端 4。
首段通过后才把目标恢复为默认二楼终点，进行完整 PCT 跨层执行。

#### PCT → SCAN → MoE-CTS 完整跨层真实执行

同层短目标和 RViz 局部点云通过后，恢复默认二楼 base 目标
`(0.4, -0.02, 3.339914814802456)`。本轮不仅检查跨层 Path 和
B-spline，还要等待机器人完成真实平面运动、至少 `2.5 m` 的机体爬升、
到达二楼目标并进入 `GOAL_REACHED`。这是动态障碍引入前的静态跨层主验收。

先结束上一轮所有终端，等待 Isaac 和 ROS 2 launch 完全退出。终端 0
只做清场检查；本步只修改了源码目录中的只读探针，不需要重新 `colcon build`：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
```

此时不应还存在上一轮的 `scan_controller` 或 `/cmd_vel` publisher。如果仍然
看到，回到对应旧终端按 `Ctrl-C`，不要在旧 transient-local 代际上直接开新
episode。

终端 1 启动唯一 bridge、PCT、SCAN 和 controller。静态跨层验收仍不启动
supervisor，避免同时引入动态重规划状态机：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_bridge:=true \
  start_scan:=true \
  start_controller:=true \
  start_manual_path:=false \
  start_pct:=true \
  start_supervisor:=false \
  start_odometry_tf:=true \
  body_height_m:=0.338 \
  pct_backend_kind:=upstream
```

终端 2 启动 RViz、Go2 模型和 PLY 发布器。RViz 默认仍只显示
`/cloud_registered` 的机器狗附近最新点云；`Static PLY Map` 只在需要观察全局
配准时手动勾选：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  voxel_leaf_size_m:=0.08 \
  start_manual_path:=false \
  start_go2_model:=true \
  go2_use_gui:=false \
  use_sim_time:=true
```

终端 3 必须在 Isaac 之前启动严格跨层执行探针。`5.0 m` 是平面位移下限，
`2.5 m` 是机体真实高度增量下限；探针只读，不会发布 Twist：

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_live_multifloor_pct_scan_chain.py \
  --expect-crossfloor-execution \
  --minimum-planar-displacement-m 5.0 \
  --minimum-vertical-displacement-m 2.5 \
  --timeout-sec 420
```

终端 4 最后启动 Isaac 和 MoE-CTS。二楼目标在命令中显式写出，避免旧默认值
或短目标残留；不得添加 `--navigation-planning-only`：

```zsh
cd /mnt/sage_data/workspace/pct_scan

export ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python

source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

"$ISAAC_PYTHON" -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --terrain-mode multifloor \
  --multifloor-ray-origin-offset-z 1.0 \
  --command-source ros2 \
  --ros-domain-id 71 \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.55 \
  --policy-max-vy 0.15 \
  --policy-max-wz 0.60 \
  --policy-max-vx-rate 0.80 \
  --policy-max-vy-rate 0.40 \
  --policy-max-wz-rate 1.50 \
  --multifloor-goal-x 0.4 \
  --multifloor-goal-y -0.02 \
  --multifloor-goal-z 3.339914814802456 \
  --multifloor-goal-yaw -1.5707963267948966 \
  --real-time \
  --warmup-steps 100 \
  --status-every 100
```

本轮唯一严格通过标志为：

```text
LIVE_MULTIFLOOR_PCT_SCAN_CROSSFLOOR_EXECUTION_OK
```

该行还会打印 `vertical_span_m`、`planar_displacement_m`、
`vertical_displacement_m`、最大三轴速度、终点 xy/z 误差和唯一 `/cmd_vel` 端点证据。
看到该标志后，机器人应在二楼持续零速。按终端 4（Isaac）→终端 1
（导航 launch）→终端 2（RViz）的顺序 `Ctrl-C`，终端 3 会在成功后自动退出。
跨层静态执行通过后，下一阶段才启动 supervisor 并加入移动人/推车，验收 SCAN
局部绕障、回归 PCT 全局路径和必要时的 PCT 重规划。

##### phase261：一楼楼梯入口前转角紧急停车诊断

首次完整跨层执行在一楼转角处进入 `EMERGENCY_STOP`。这不是楼梯踏面或
MoE-CTS policy 失效：当时轨迹起点约为 `(1.02, 4.86, 0.235)`，而 PCT Path
直到索引 44 仍是一楼平面，真正楼梯上升从索引 45、`y≈6.28` 才开始。

typed 状态证明 SCAN 在同一 Path 上发布了 `event=EMERGENCY_STOP`、
`global_replan_recommended=true`。碰撞审计在第 `5/130` 个样本、
`(1.0549, 4.8631, 0.2263)` 检出占据。GridMap raw occupancy 最近对应体素为
`(1.325, 4.625, 0.125)`；它与 PCT 一楼中心线的 xy 间距只有约
`0.330 m`。对 `radius=0.27 m, offset=0.16 m` 的双圆柱包络，沿当时轨迹
初始切向旋转后该体素已进入前圆柱；膨胀地图最近体素距碰撞样本仅
`0.043 m`。因此 L-BFGS `-1008` 是“起点近旁已无可行局部绕行”的结果，
不是优化器无缘无故崩溃。

离线对照已证明：只把
`planner.upstream_same_layer_shortcut_clearance_m` 从 `0.27` 改为 `0.43/0.48`，或把
`planner.upstream_astar_step_cost_weight` 从 `0.20` 改为 `1.0`，都不会移动这个
关键转角点，因此不要盲目改 YAML、放宽碰撞门或继续提速。当前 launch
明确使用 `start_supervisor=false`，所以安全停车后不会自动消费这次全局重规划建议。
下一修复应为楼梯 profile 加入经连续双圆柱审计的一楼接近段，让机器人在进入
这个窄转角前提前变向；安全门和紧急停车逻辑保持不变。

##### phase262：一楼楼梯入口接近段修复与重新验收

phase261 定位的入口几何缺陷已经按“先离线证明，再恢复真实执行”的顺序修复。
`pct_multifloor_stair_profile.json` 现在包含一条独立的
`lower_floor_approach`：17 个地面高度锚点让 PCT Path 在一楼窄口前连续转向，
最后一个接近锚点与原楼梯第一个拓扑锚点严格重合。原来的 7 个楼梯锚点仍单独
负责 tomogram 的跨层拓扑修补；接近段不会被写入或保护为 tomogram 楼梯拓扑，
避免把场景标定的执行几何误当成层析连通关系。

接近段在 backend 初始化时必须通过真实 collision PLY 三角面审计。审计以
`0.01 m` 空间步长和不大于 `0.02 rad` 的航向步长连续采样，并在每个折点扫过
最短航向变化；每个姿态同时检查 `radius=0.27 m、offset=0.16 m` 的前后圆柱。
旧晚转路径的同口径表面净空约为 `0.213 m`，小于圆柱半径；新接近段最小净空为
`0.293162 m`，通过 `0.285 m` 合同并保留约 `0.023162 m` 的半径外余量。
任何 PLY、接近锚点或包络参数变化导致审计不通过时，PCT backend 会直接拒绝
启动，不会退回旧路径。

生产端点的完整 PCT Path 由 142 点更新为 146 点。CPU ROS 2 全链离线验收得到
27 个 SCAN 窗口，其中 22 个为非终点窗口；最小速度上界 `0.469881 m/s`，最大
轨迹时长 `3.002090 s`，最后 5 个窗口均为 final，最大终点捕获 xy 误差
`0.050594 m`。这些结果证明入口几何和 PCT→SCAN 数学链已通过，但不等于
Isaac Sim 中机器狗已经重新爬到二楼；严格 live 标志仍需下面的终端 0～4 验收。

新 profile 的 SHA256 会写入本地 ignored upstream tomogram。更新代码后必须先
结束上一轮 Isaac、导航 launch 和 RViz，再运行终端 0 重建资产和 adapter；若跳过
重建，backend 会因 profile hash 不一致而失败关闭。

终端 0：重建本地 upstream 资产、编译 adapter，并确认旧 ROS 图已清空。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan

source /opt/ros/humble/setup.zsh

/usr/bin/python3 scripts/navigation/build_pct_multifloor_assets.py \
  --tomogram-kind upstream \
  --output-tomogram source/scene/multifloor/mutifloor_upstream.pickle \
  --output-walkable /tmp/mutifloor_upstream_walkable.npy \
  --report-output outputs/pct_multifloor_upstream_asset_build_report.json

cd ros2_ws

colcon build \
  --symlink-install \
  --packages-select pct_ros2_adapter

source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
```

终端 0 的最后两条命令不应看到旧 `scan_controller` 或 `/cmd_vel` publisher。
若仍存在，回到对应旧终端按 `Ctrl-C` 并重新检查，不要直接启动下一代。

终端 1：启动唯一的 bridge、PCT、SCAN、controller 和 odometry TF。本轮继续
关闭 supervisor，先验收修复后的静态跨层几何，不让自动重规划掩盖问题。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_bridge:=true \
  start_scan:=true \
  start_controller:=true \
  start_manual_path:=false \
  start_pct:=true \
  start_supervisor:=false \
  start_odometry_tf:=true \
  body_height_m:=0.338 \
  pct_backend_kind:=upstream
```

终端 2：启动 RViz、Go2 模型和 PLY 发布器。默认只显示机器人附近最新一帧
`/cloud_registered`；需要查看全局配准时再手动勾选 `Static PLY Map`。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  voxel_leaf_size_m:=0.08 \
  start_manual_path:=false \
  start_go2_model:=true \
  go2_use_gui:=false \
  use_sim_time:=true
```

终端 3：在 Isaac 之前启动严格只读探针。它不发布 Twist，只采集本轮起点、同代
Path/B-spline/controller identity、真实位移和终点状态。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_live_multifloor_pct_scan_chain.py \
  --expect-crossfloor-execution \
  --minimum-planar-displacement-m 5.0 \
  --minimum-vertical-displacement-m 2.5 \
  --timeout-sec 420
```

终端 4：最后启动 Isaac Sim 和 MoE-CTS。目标必须保持为二楼 base 目标，不要添加
`--navigation-planning-only`。

```zsh
cd /mnt/sage_data/workspace/pct_scan

export ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python

source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

"$ISAAC_PYTHON" -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --terrain-mode multifloor \
  --multifloor-ray-origin-offset-z 1.0 \
  --command-source ros2 \
  --ros-domain-id 71 \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.55 \
  --policy-max-vy 0.15 \
  --policy-max-wz 0.60 \
  --policy-max-vx-rate 0.80 \
  --policy-max-vy-rate 0.40 \
  --policy-max-wz-rate 1.50 \
  --multifloor-goal-x 0.4 \
  --multifloor-goal-y -0.02 \
  --multifloor-goal-z 3.339914814802456 \
  --multifloor-goal-yaw -1.5707963267948966 \
  --real-time \
  --warmup-steps 100 \
  --status-every 100
```

启动顺序必须是终端 0 完成后，再依次启动 1 → 2 → 3 → 4。只有终端 3 打印

```text
LIVE_MULTIFLOOR_PCT_SCAN_CROSSFLOOR_EXECUTION_OK
```

才表示这次入口修复获得真实跨层验收。成功后终端 3 自动退出，按终端 4
（Isaac）→终端 1（导航）→终端 2（RViz）的顺序停止。若再次急停，请保留终端 1
从第一条“局部轨迹碰撞”到 `EMERGENCY_STOP` 的完整日志，并同时回传终端 3 的
异常；不要先放宽半径、膨胀、速度门或 feasibility tolerance。

##### phase263：楼梯踏面支撑点误入占据图修复

phase262 的提前转向已生效：本轮 146 点 PCT Path 成功通过一楼窄口，
机器人也实际进入并爬上了第一段楼梯。新的首个稳定失败点是
`(1.6310, 7.1887, 0.8229)`，随后在 `(1.6812, 7.5334, 0.9875)`
连轨迹起点都被 SCAN 判定为占据，最终正确进入
`REPLAN_TRAJ -> EMERGENCY_STOP`。`L-BFGS -1008` 和
`The robot is inside an obstacle` 是占据起点之后的二次结果，不是根因。

对实际 collision PLY 踏面、生产 Path 和前后双圆柱包络的密集离线扫描表明：
第一段楼梯踏面相对稀疏 Path 折线内插高度的上方偏差最大为
`0.080362 m`，而 bridge 原本的 Path 支撑面上边界只有 `0.05 m`。
偏差介于 5–8 cm 的真实踏面因此被发布为占据端点，再经 GridMap
障碍膨胀进入机体双圆柱，与日志的重复碰撞位置一致。

修复不是全局放宽地面过滤。平地和缓坡仍使用 `0.05 m`上边界；
只当当前 Path 线段的 `|dz|/dxy >= 0.20` 时，才使用 `0.09 m`
楼梯支撑面上边界。这覆盖实测最大偏差并保留约 `9.6 mm` 余量，
同时平地上高于 5 cm 的低矮障碍仍会进入占据图。本修复没有改动 SCAN
双圆柱半径、障碍膨胀、速度限制、可行性门或紧急停车逻辑。

上一轮 GridMap 已保存错误占据体素，SCAN 也已锁存 `EMERGENCY_STOP`；
不能在原进程中只改参数继续跑。先分别在 Isaac、导航 launch 和 RViz 终端按
`Ctrl-C`，再按下面 0 → 1 → 2 → 3 → 4 的顺序启动新一代。

终端 0：只重编本轮修改的 bridge 包，并确认旧 ROS 图已清空。
phase262 已重建 upstream PCT 资产，本轮没有改 Path profile，无需再生成 tomogram。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh

colcon build \
  --symlink-install \
  --packages-select isaac_navigation_bridge

source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
```

终端 1：启动唯一的 bridge、PCT、SCAN、controller 和 odometry TF。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_bridge:=true \
  start_scan:=true \
  start_controller:=true \
  start_manual_path:=false \
  start_pct:=true \
  start_supervisor:=false \
  start_odometry_tf:=true \
  body_height_m:=0.338 \
  pct_backend_kind:=upstream
```

终端 1 启动完后，在已空闲的终端 0 确认新参数确实被运行节点读取：

```zsh
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 param get /isaac_navigation_bridge \
  filters.path_ground_clearance_m
ros2 param get /isaac_navigation_bridge \
  filters.path_ground_stair_minimum_slope
ros2 param get /isaac_navigation_bridge \
  filters.path_ground_stair_clearance_m
```

三条输出应依次为 `0.05`、`0.20`、`0.09`。

终端 2：启动 RViz、Go2 模型和 PLY 发布器。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  voxel_leaf_size_m:=0.08 \
  start_manual_path:=false \
  start_go2_model:=true \
  go2_use_gui:=false \
  use_sim_time:=true
```

终端 3：在 Isaac 之前启动严格只读跨层探针。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_live_multifloor_pct_scan_chain.py \
  --expect-crossfloor-execution \
  --minimum-planar-displacement-m 5.0 \
  --minimum-vertical-displacement-m 2.5 \
  --timeout-sec 420
```

终端 4：最后启动 Isaac Sim 和 MoE-CTS。

```zsh
cd /mnt/sage_data/workspace/pct_scan

export ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python

source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

"$ISAAC_PYTHON" -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --terrain-mode multifloor \
  --multifloor-ray-origin-offset-z 1.0 \
  --command-source ros2 \
  --ros-domain-id 71 \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.55 \
  --policy-max-vy 0.15 \
  --policy-max-wz 0.60 \
  --policy-max-vx-rate 0.80 \
  --policy-max-vy-rate 0.40 \
  --policy-max-wz-rate 1.50 \
  --multifloor-goal-x 0.4 \
  --multifloor-goal-y -0.02 \
  --multifloor-goal-z 3.339914814802456 \
  --multifloor-goal-yaw -1.5707963267948966 \
  --real-time \
  --warmup-steps 100 \
  --status-every 100
```

这次的必要验收条件是：终端 1 不再在
`(1.63, 7.19, 0.82)` 或 `(1.68, 7.53, 0.99)` 附近重复报占据，
机器人继续通过第一段楼梯，且终端 3 最终打印：

```text
LIVE_MULTIFLOOR_PCT_SCAN_CROSSFLOOR_EXECUTION_OK
```

若仍然急停，回传新一轮终端 1 从第一条“局部轨迹碰撞”到
`EMERGENCY_STOP` 的连续日志和终端 3 异常。不要复用本轮已锁存的
GridMap，也不要先改速度、碰撞半径或 feasibility tolerance。

若要进一步排除 PCT、SCAN 和航向控制器，可运行固定机体系速度探针：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile multi_floor \
  --task-json tasks/nav_smoke_scan_multifloor_stair_two_step.json \
  --stair-locomotion-smoke \
  --stair-fixed-command-probe \
  --stair-probe-vx 0.25 \
  --stair-probe-duration 3.84 \
  --navigation-visual-mode collision \
  --headless \
  --no-record-dataset \
  --no-record-video \
  --output-dir "$PCT_SCENE_OUTPUT/multi_floor_stair_fixed_vx025"
```

探针保持任务原始起点和收纳姿态，不创建 PCT client 或 SCAN，不启用 Float、
root pose lock 或第二路底盘命令。`vx` 只允许 `0.20/0.25/0.30 m/s`，驱动
时长必须位于 3–5 秒；到期后的第一条状态转换动作严格清零。

当前 SCAN 主线另有一条已经实际验收的楼梯工程路径：ROS 2 发布的完整
地面 Path（生产 `/pct/global_path`，手工 smoke `/initial_path`）通过几何哈希
绑定楼梯段；机器人进入该段时先清零唯一 policy 速度入口，再锁定 root、支撑
关节和导航收纳姿态，沿同一三维 Path 推进，离开楼梯后分阶段释放并等待 SCAN
controller 的终点稳定驻留。楼梯内发生 Path/时钟/规划异常时会锁存最后 root
目标和关节锁，并保持零速急停。该结果明确记为
`scan_stair_root_lock_workaround`，不是旧 PCT+DWA stair-float，也不是 RL policy
纯物理爬楼。完整状态、证据和剩余边界见
[`docs/pct_scan_navigation.md`](docs/pct_scan_navigation.md)。

full-physics 默认保存 LeRobot 数据，并录制 profile 指定的视频流。如果只做物理诊断，可以添加 `--no-record-dataset --no-record-video` 减少磁盘占用。

##### phase268：楼梯下踏面过滤、快速巡航与分层 RViz

本轮对新急停点 `(1.7437, 8.0479, 1.2694)` 做了 collision PLY 数值审计。
唯一落入双圆柱碰撞体且未被旧过滤器识别的三角面位于 PCT 地面折线下方
`0.064994 m`；旧 `filters.path_ground_band_down_m=0.05` 因而漏掉它，随后
0.05 m GridMap 栅格和竖直膨胀把当前轨迹起点判为 occupied。现在平地/缓坡
仍只使用上下 0.05 m 窄带；仅当 Path 线段斜率 `|dz|/dxy >= 0.20` 时，采用
楼梯专用上下 0.09 m 窄带。这个修改用于吸收折线路径跨越离散踏面时的量化误差，
不会全局放宽障碍碰撞门。

生产速度配置也统一切到已有单独测试覆盖的快速楼梯包络：巡航
`0.60 m/s`、轨迹和 policy 的 `vx` 上限 `0.65 m/s`、纵向变化率
`1.20 m/s²`。横向偏差进入 Path 恢复模式时，前进速度由 `0.18` 提到
`0.30 m/s`，避免机器狗因正常步态横摆短时越线后失去爬楼动力。横向速度和
yaw 上限仍保持 `0.15 m/s`、`0.60 rad/s`。

RViz 新增的 `navigation_marker_publisher` 只负责显示，不参与规划和控制。默认
图层现包括 `Ground Grid`、`Go2`、`Robot Path`、`PCT Reference Path`、
`PCT Path Cylinders`、`PCT Start and Goal`、`Sensor Cloud`、
`Inflated Occupancy`、`Sliding Map Bounds`、`SCAN Optimized B-spline` 和
`SCAN Goal`。整张 `PCT Source PCD` 与原始 `Occupancy` 默认关闭，以免遮住
机器人附近的在线地图。当前 PCT adapter 还没有发布独立 tomogram 可视化消息，
因此没有用 PLY 冒充截图中的 `PCT Tomogram`；后续若需要再接真实层析切片 Marker。

所有旧进程和 RViz 都必须停止后，从终端 0 开始完整重启。旧 GridMap 已锁存的
occupied 体素不会因为只重启 Isaac 或只刷新 RViz 自动消失。

终端 0：构建并确认 ROS domain 中没有上一轮残留节点。若最后一条命令仍列出
节点，回到对应旧终端 `Ctrl-C`，直到输出为空再继续。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh

colcon build \
  --symlink-install \
  --packages-select \
  navigation_visualization \
  isaac_navigation_bridge \
  pct_ros2_adapter \
  scan_controller

source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 node list --no-daemon --spin-time 3
```

终端 1：启动唯一的 bridge、PCT、SCAN、controller 和 odometry TF。该 launch
会从 `pct_scan.yaml` 读取新的楼梯窄带，并从 `pct_scan_tuning.yaml` 读取统一
快速速度合同。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_bridge:=true \
  start_scan:=true \
  start_controller:=true \
  start_manual_path:=false \
  start_pct:=true \
  start_supervisor:=false \
  start_odometry_tf:=true \
  body_height_m:=0.338 \
  pct_backend_kind:=upstream
```

终端 2：启动 RViz、Go2 模型、静态 PLY 发布器和纯显示 Marker 节点。打开后
不需要手工逐项添加图层；若使用的是修改前已打开的 RViz 窗口，必须关闭重启。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  voxel_leaf_size_m:=0.08 \
  start_manual_path:=false \
  start_go2_model:=true \
  go2_use_gui:=false \
  use_sim_time:=true
```

终端 3：必须在 Isaac 之前启动严格只读跨层探针。它不发布 `/cmd_vel`。

```zsh
conda deactivate 2>/dev/null || true
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_live_multifloor_pct_scan_chain.py \
  --expect-crossfloor-execution \
  --minimum-planar-displacement-m 5.0 \
  --minimum-vertical-displacement-m 2.5 \
  --timeout-sec 420
```

终端 4：最后启动 Isaac Sim 和 MoE-CTS。这里的 policy 限幅必须与终端 1
controller 完全一致，不能继续沿用 phase263 的 `0.55/0.80` 旧值。

```zsh
cd /mnt/sage_data/workspace/pct_scan

export ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python

source /opt/ros/humble/setup.zsh
source ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71
unset PYTHONPATH

"$ISAAC_PYTHON" -B \
  scripts/navigation/play_go2_moe_cts_keyboard.py \
  --terrain-mode multifloor \
  --multifloor-ray-origin-offset-z 1.0 \
  --command-source ros2 \
  --ros-domain-id 71 \
  --cmd-vel-timeout 0.25 \
  --policy-max-vx 0.65 \
  --policy-max-vy 0.15 \
  --policy-max-wz 0.60 \
  --policy-max-vx-rate 1.20 \
  --policy-max-vy-rate 0.40 \
  --policy-max-wz-rate 1.50 \
  --multifloor-goal-x 0.4 \
  --multifloor-goal-y -0.02 \
  --multifloor-goal-z 3.339914814802456 \
  --multifloor-goal-yaw -1.5707963267948966 \
  --real-time \
  --warmup-steps 100 \
  --status-every 100
```

可选终端 5：只读确认新参数与 RViz Marker 链。前三个参数应输出
`0.09`、`0.60`、`0.65`，两个 Marker topic 均应有一个 publisher 和一个
RViz subscription，`/cmd_vel` 必须始终只有一个 publisher。

```zsh
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=71

ros2 param get /isaac_navigation_bridge \
  filters.path_ground_stair_band_down_m
ros2 param get /scan_planner_node \
  fsm.reference_cruise_speed
ros2 param get /scan_controller \
  limits.max_vx

ros2 topic info /visualization/pct_path_cylinders --verbose
ros2 topic info /visualization/robot_path --verbose
ros2 topic info /cmd_vel --verbose --no-daemon --spin-time 3
```

本轮严格通过标志仍是：

```text
LIVE_MULTIFLOOR_PCT_SCAN_CROSSFLOOR_EXECUTION_OK
```

同时要求终端 1 不再在 `(1.74, 8.05, 1.27)` 附近反复出现
`The robot is inside an obstacle`，并且终端 4 的爬楼有效前进命令不再长期被
限制在约 `0.18 m/s`。停止顺序为终端 4 → 1 → 2；终端 3 成功后会自动退出。

##### phase269：GPU 恢复后的生产跨层与完整抓放验收

系统重启后，新 boot id、`nvidia-smi` 和正式 Isaac Python 的 CUDA 张量分配/
同步均通过。生产配置保持 `body_height=0.338 m`、巡航 `0.60 m/s`、
`max_vx=0.65 m/s`、`max_ax=1.20 m/s²` 和 `max_yaw_rate=0.60 rad/s`。
原 Go2-X5 checkpoint 的 `crossfloor_carry` seeds 0、1 已分别严格通过：两次都
消费同一条 146 点 PCT Path，点列 SHA256 为
`25fa7c170cc6c4561680f51fbe5de4359cc9458867923dbdf7f2b97fafeb6f32`；均完成
一楼 SCAN、用户确认的楼梯底盘冻结、二楼 SCAN 恢复、严格
`GOAL_REACHED` 和到达后连续零速。两次完整导航仿真时间分别为 `107.80 s`
和 `109.24 s`；这里包含约一分钟的共同楼梯冻结，不等于 SCAN 规划计算时间。

随后运行了真实 `nav → pick → carry nav → place`。第一次放置已经落在目标附近，
但苹果释放瞬间角速度 `12.3421 rad/s` 比不变的 `12.0 rad/s` 质量门高
`2.85%`。没有放宽质量门，而是在本任务的 `manipulation_execution` 中把圆形苹果
的计划释放净空从通用 `10 mm` 改为 `4 mm`。复验完整 pipeline 成功：导航交接
XY 误差 `0.0521 m`、yaw 误差 `0.1306 rad`，苹果最终放置 XY 误差
`0.00592 m`、Z 误差 `0.00050 m`，释放峰值角速度降到 `3.9462 rad/s`。
这次结果明确使用楼梯 root lock，因此 `stable_physics_success=true`，但
`physical_navigation_success=false`、`pure_physics_success=false`；不能写成 RL
policy 已经纯物理爬楼。

记录器现在真正应用 `--diagnostic-frame-stride`。同一完整任务把逐帧诊断文件从
约 `1.010 GB` 降到 `135.6 MB`，记录帧从每个控制拍缩减为 700 帧；状态转换、
Path 快照和逐拍质量峰值仍完整保留。首次完整 carry 成功还导出了不可变局部规划器
对照合同；DWA 尚未按这份合同隔离实跑，所以当前仍不能宣称 SCAN 已超过 DWA。

`max_yaw_rate=0.75 rad/s` 的严格单变量实验已被否决：同一 seed 和同一 Path 下，
更快转向使机器人在楼梯入口的支撑占据区先达到连续规划失败阈值，随后 PCT Path
被重规划 tombstone 清除，运行以
`scan_reference_path_cleared_during_stair_freeze` 失败。生产 YAML 从未改成
`0.75`；速度与稳定性现阶段选择 `0.60 rad/s`。

##### phase270–273：末端连续驻留与楼梯恢复身份修复

seed 2 暴露了两个独立的末端竞态：controller 曾在移动 final 轨迹名义结束后凭
单拍到点立即锁存成功，planner 也曾凭单拍到点发布静止 hold。现在移动 final
必须在终点范围内连续稳定 `0.50 s`，planner 必须观察真实 Odometry 连续稳定
`0.75 s` 才能发布静止 hold。为给原 Go2-X5 步态回弹留下物理余量，planner
末端安全目标由 `0.06 m` 收紧到 `0.04 m`，controller 捕获入口由 `0.075 m`
收紧到 `0.055 m`；严格完成门仍是 `0.08 m`，没有放宽碰撞或到达标准。

最新配置的 cross-floor seed 2 已严格通过：终点 XY 距离约 `0.0657 m`，summary
导航时间 `122.66 s`。分段统计中，共同的楼梯底盘冻结为 `64.60 s`，SCAN 实际
控制一楼和二楼为 `55.74 s`，21 次成功局部优化累计墙钟仅 `1.80127 s`。这说明
本次慢项主要仍是 RL policy 的平移、原地转向和末端稳定，不是 SCAN 求解器持续
计算。

最新配置的完整 seed 0 复验已完成第一段导航和抓取，但在楼梯入口安全停止为
`stair_sensor_freshness_fault`。记录证明点云和 Odometry 当时均新鲜；真正原因是
故障前同一条 B-spline 恰好从 `TRACKING` 切到 `ALIGNING_YAW`，supervisor 把它
的新状态序号误当成一条恢复轨迹，提前清除了楼梯停车状态。修复后，恢复轨迹
identity 必须与故障前 identity 不同；同一旧轨迹的 `TRACKING/ALIGNING_YAW`
心跳都不能解除停车。supervisor 包级 `87 passed, 4 skipped`，相关主线回归
`538 passed`。该修复尚待下一次 fresh full Isaac 复跑，当前不能把最新配置写成
完整 pipeline 已通过，也仍不能宣称 SCAN 超过 DWA。该次运行退出后 GPU 计算接口
曾短暂不可用：`nvidia-smi` 无法连接驱动、Isaac Python 报 CUDA 设备数为 0；运行
时间窗没有新的内核 Xid。`23:24` 复检时管理查询和同一 Isaac Python 的真实 CUDA
张量分配/同步均重新通过，设备数为 1；但遵照本轮结束后不自动开启第二个 Isaac
的约定，修复后的完整实跑仍待用户明确继续。

### 2.3 批量采集当前边界

PCT→SCAN 尚未实现跨 episode 的 ROS 2 epoch/reset/ack，因此 batch 导航当前
失败关闭：`--num-episodes` 只能为 1。下面命令只验证 batch 包装器和单条输出；
不得用多个独立 Isaac 子进程共享同一个持续运行的 ROS 图来规避该门禁。

良渚单 episode：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_seed7000_n1" \
  --num-episodes 1 \
  --seed 7000 \
  --no-record-video
```

别墅多楼层单 episode（profile 默认使用固定任务）：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile multi_floor \
  --output-dir "$PCT_SCENE_OUTPUT/multi_floor_seed0_n1" \
  --num-episodes 1 \
  --seed 0 \
  --no-record-video
```

batch 的 stage 复用机制仍保留在实现中，待 ROS 2 epoch 合同完成后再重新开放。
历史设计会在一个 IsaacLab env / stage 中连续执行 episode，并重置机器人、任务物体、记录器和状态机；良渚 box1/box2
会在 stage 初建时转成“episode 内不可移动、episode 间可重定位”的 kinematic support，
避免运行中热改 USD 静态 collider 后 PhysX/Fabric 仍使用上一条位姿。

复用 stage 不等于复用上一条 episode 的 PhysX 求解器状态。每个后续 episode
会先写入本次 box/可乐 USD 位姿，再执行 `SimulationContext.reset(soft=False)`：
USD stage 不重开、不重建，但 articulation/contact solver、PhysX tensor view 会重新创建；
随后重新绑定可乐/支撑物 reader，并重新应用 front/wrist D436 内参。硬重置本身不推进
physics/control step。episode reset 的第一条审计 action 也严格使用
`skip_physics_step`，不会提前消耗 RL warmup 或推进 action history。

机器人初始化采用两阶段交接：前 20 个 control tick 同时固定 reset root，并用
actuator target 保持支撑腿；这段人为静止不计入稳定步数。随后同时解除 root/支撑腿锁，
由 `pct_multifloor` RL policy 在真实接触下平衡，并重新满足速度、roll/pitch 和位姿门后
才进入导航。这样既避免 reset 首步冲击，又不会让固定站姿在不同地面/yaw 下缓慢侧翻。

batch heartbeat 会按子进程启动时间筛选本轮新写入的进度文件，再从
`frames.jsonl -> events.jsonl -> summary.json` 读取状态。复用一个已经存在的输出根目录时，
旧的高编号 `episode_XXXXXX` 不会再覆盖当前 episode；frames 处于部分写入或无法解析时，
会回退到轻量的 `events.jsonl`。真实状态仍按 `--progress-interval-s` 打印；尚未生成本轮
进度文件的 startup/pending 信息最多每 30 秒打印一次，不再每 5 秒连续刷
`state=unknown source=unavailable`。正式采集仍应使用新的输出目录，避免历史数据混放。

`--no-reuse-isaac-process` 不能绕过当前 epoch 门禁；它只保留给非导航模式和未来
协议完成后的隔离诊断。

默认使用 headless 模式，episode seed 为 `seed + episode_index`。失败的 episode 会保留
诊断文件；通过物理来源和训练质量检查的 episode 会合并到
`<output-dir>/lerobot_dataset`。

#### Headless batch 性能与图像/状态同步契约

量产建议使用 `--no-record-video`。该模式只降低训练相机渲染和诊断 I/O 频率，不改变
物理或控制时序：PhysX 仍为 400Hz（`physics_dt=0.0025s`），control/locomotion 仍为
50Hz（`control_dt=0.02s`），decimation 仍为 8。headless 数据相机与 LeRobot 数据格点
为 5Hz，`frames.jsonl` 记录 5Hz 格点及所有状态切换；GUI 和展示视频仍逐 control step
渲染，本优化不降低 composite 视频频率。

图像编码和写盘默认异步，但取样是同一 control tick 内的同步事务：一次性读取
`front/wrist/overview`、state 和 action，把 GPU image 立即冻结为独立 CPU buffer，
再生成一个 `SynchronizedSamplePacket`。后台单 worker 只按 FIFO 顺序编码和写盘，
不再访问仿真状态。每个样本必须满足：

```text
simulation_step == camera_capture_step == state_step
simulation_timestamp == camera_capture_timestamp == state_timestamp
```

step 必须严格相等，timestamp 只保留 `1e-9s` 浮点容差；任一不一致都会拒绝数据。
队列满时主线程 backpressure，episode 结束前必须 drain
全部 packet，并审计连续 frame index 和 `sampling_coverage`。

2026-07-19 在同一台 RTX 4060 Laptop、相同良渚任务、相同 seed 7/8/9、
`--no-record-video` 下做了严格逐 seed 对照。当前列使用包含上述 PhysX 隔离修复的
20-seed 运行结果，不使用更早但存在 `base_settle_timeout` 的复用结果：

| Seed | 未优化 pipeline wall | 当前 pipeline wall | 每条节省 | 降幅 |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 303.06s | 205.85s | 97.21s | 32.1% |
| 8 | 275.29s | 182.33s | 92.96s | 33.8% |
| 9 | 282.20s | 177.89s | 104.31s | 37.0% |
| 平均 | 286.85s | 188.69s | 98.16s | 34.2% |

同三条 aggregate real-time factor 从 `0.1526` 提升到 `0.2136`，吞吐约为旧版
`1.52x`。先前独立 I/O benchmark 中，RTX render 从 48.11s/条降到 5.51s/条，
`frames.jsonl` 写入从 20.71s/条降到 0.37s/条，3 条完整输出从 1.173GB 降到
0.162GB；安全修复没有撤销这些 5Hz/异步导出优化。

`arm_vla_liangzhu` 最终唯一 seeds 7..26 的 20 条真实 full-physics 结果为
`20/20` 成功、`20/20` 训练质量门通过、`base_settle_timeout=0`。其中最长连续
单进程 stage 复用为 15 条；外部工具中断后补齐剩余 seed，并让 seed 26 再次作为
复用后的第二条验证。20 条 pipeline 内部 wall time 平均 191.94s（约 3m12s）、
中位数 189.10s。按旧版 3-seed 均值作吞吐估算，每条约省 94.91s（33.1%），
20 条约省 31m38s；该 20 条估算不是逐 seed 旧版配对，严格配对结论只使用上表。

同步完成后，`pct_scene --scene-profile liangzhu` 又独立运行 seeds 27/28：两条均在
同一 Isaac 进程/stage 内完成完整 pipeline、通过训练质量门，第二条明确经过 PhysX
hard reset 和 stage reuse，`base_settle_timeout=0`。这证明修复在统一 scene-profile
代码中真实生效；`multi_floor` 仍需单独做 stage-reuse GPU gate，不能用良渚结果代替。

#### 2026-07-18 双 worktree 小规模验收


| worktree / seeds            | 尝试 | 完整成功并进入统一数据集 | 数据集帧数 | validator           | composite     |
| --------------------------- | ---: | -----------------------: | ---------: | ------------------- | ------------- |
| `pct_scene` / `0..19`       |   20 |                       18 |       4096 | 0 error / 0 warning | 20/20，零丢帧 |
| `arm_vla_liangzhu` / `0..4` |    5 |                        5 |       1004 | 0 error / 0 warning | 5/5，零丢帧   |

两边合计实际尝试 25 条，23 条通过训练质量门并进入各自统一 LeRobot 数据集。`pct_scene`
seed 7 和 seed 19 分别在最终放置验证中以 `place_release_pose_error` 和
`place_release_ejected` 被拒绝；两条都已完成导航、抓取和放置动作，原始诊断与
composite 视频保留，但未混入统一训练数据集。验证阈值没有为提高表面成功率而放宽。

两个 worktree 的 seed 0..4 随机化采样，以及派生出的 start、pick/place、动态
keepout、空间约束和 mesh-truth 目标逐 JSON 哈希一致；比较时只排除了 worktree 自身的
绝对 collision PLY 路径。数据证据位于：

```text
/mnt/sage_data/outputs/pct_scene/
liangzhu_headless_batch20_seed0_standofffix_20260718_v2

/mnt/sage_data/outputs/arm_vla_liangzhu/
alignment_batch5_seed0_composite_20260718
```

### 2.4 校验数据并导出 Rerun

校验 batch 时，`--dataset-root` 应指向合并后的 `lerobot_dataset`，而不是 batch
根目录或未成功导出的 episode：

```bash
$ISAAC_PYTHON -B scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root "$PCT_SCENE_OUTPUT/liangzhu_seed7000_n20/lerobot_dataset"
```

如果磁盘空间有限，可以只保留 `lerobot_dataset/meta`、目标 episode 的 Parquet
chunk 和相机视频，也可以直接在服务器上转换。以下命令导出第 0 个 episode 的前 200 帧：

```bash
conda activate lerobot_rerun

python tools/lerobot_to_rerun.py \
  --repo-id pct_scene_dataset \
  --root "$PCT_SCENE_OUTPUT/liangzhu_seed7000_n20/lerobot_dataset" \
  --episode-index 0 \
  --max-frames 200 \
  --out "$PCT_SCENE_OUTPUT/liangzhu_seed7000_n20/episode_000000.rrd"
```

验收时检查以下结果：


| 阶段         | 通过条件                                              |
| ------------ | ----------------------------------------------------- |
| 资产检查     | 命令退出码为 0，所需 profile 没有 missing asset       |
| dry-run      | 状态机到达`done`                                      |
| full-physics | `summary.json` 中 `success=true`                      |
| 数据集校验   | `validation_report.json` 中 `valid=true` 且没有 error |

### 2.5 输出内容

full-physics 成功后会保留运行诊断文件，并导出 LeRobot v2.1 数据集。默认情况下，`--output-dir` 指向本次运行的输出根目录，单个 episode 位于其下的`episode_000000/`：

```text
<output-dir>/
├── .runtime/                              # 运行时场景绑定
├── startup_status.json                    # 启动状态
├── batch_summary.jsonl                    # episode 结果索引
└── episode_000000/
    ├── task.json                          # 任务、seed 和随机化结果
    ├── events.jsonl                      # 状态机事件
    ├── frames.jsonl                      # 原始帧级记录
    ├── summary.json                      # 运行结果与数据质量检查
    ├── data.csv                          # 同步采样记录
    ├── samples.jsonl                     # LeRobot 转换所需的帧数据
    ├── lerobot_manifest.json             # 本 episode 的导出清单
    ├── images/                           # front、wrist、overview 原始图像
    ├── recording_videos/                 # LeRobot 转换前的相机视频
    ├── overview_videos/                  # 展示视频，仅 --record-video 时生成
    └── lerobot_dataset/
        ├── data/chunk-000/
        │   └── episode_000000.parquet    # 标准 LeRobot episode
        ├── videos/chunk-000/
        │   ├── observation.images.front/
        │   ├── observation.images.wrist/
        │   └── observation.images.overview/
        ├── meta/                         # info、tasks、episodes 和 subtask 元数据
        ├── validation_report.json        # 数据集校验结果
        └── episodes/
            └── <task_id>/
                └── <episode_id>/
                    ├── task.csv
                    ├── <episode_id>-1/   # nav_straight
                    ├── <episode_id>-2/   # nav_turn
                    ├── <episode_id>-3/   # nav_stop
                    ├── <episode_id>-4/   # arm_approach
                    ├── <episode_id>-5/   # arm_contact
                    └── <episode_id>-6/   # arm_retreat
```

六个 subtask 目录都包含 `data.csv`、`images/front/` 和 `images/wrist/`。某类subtask 没有有效帧时，对应目录仍会保留，但 `data.csv` 只有表头。标准 LeRobotParquet 同时保存 `task_stage`、`subtask` 和 `subtask_segment_index` 字段，因此既可按完整 episode 训练，也可直接使用 `episodes/` 下已经切分的数据。

使用 `--no-record-dataset` 时，只保留任务和运行诊断文件，不生成 `data.csv`、`samples.jsonl`、相机图像、相机视频或 `lerobot_dataset/`。

batch 会在输出根目录中保留每个源 episode，并将通过质量检查的数据合并到`lerobot_dataset/`。`batch_summary.jsonl` 记录每个 episode 的结果，`lerobot_export_manifest.json` 记录合并来源。

数据集包含以下主要训练字段：


| Feature                     | 内容                                |
| --------------------------- | ----------------------------------- |
| `observation.state`         | 17 维底盘、TCP、机械臂和夹爪状态    |
| `observation.base_velocity` | 3 维机体系底盘速度                  |
| `observation.object_state`  | 13 维物体位姿与速度                 |
| `observation.tcp_pose`      | 7 维 TCP 世界系位姿                 |
| `observation.images.front`  | 640 × 480 前视 RGB 视频            |
| `observation.images.wrist`  | 640 × 480 腕部 RGB 视频            |
| `action`                    | 10 维 VLA 训练动作                  |
| `control.action`            | 11 维原始底盘、机械臂和夹爪控制目标 |
| `task_stage` / `subtask`    | 任务阶段与六类统一子任务标签        |

图像作为 LeRobot video feature 存储，不直接写入 Parquet。数据集的完整 schema、
坐标系、单位和夹爪约定会写入 `meta/info.json` 和每个 episode 的 `task.csv`。

front 与 wrist 均使用 D436 的 640×480 标定内参：`fx=383.44608095`、
`fy=383.52724198`、`cx=324.33479864`、`cy=238.90275478`，OpenCV
pinhole 的 12 个畸变系数均为 0。runtime 会尝试启用
`OmniLensDistortionOpenCvPinholeAPI`；当前 IsaacLab 5.1 headless 实测回退到标准
USD pinhole，实际渲染 K 为 `fx=fy=383.486661465`、`cx=320`、`cy=240`，最大
偏差约 4.3 px，并记录在 `camera_runtime_intrinsics_report`。wrist camera 挂载在
`arm_link6`。原始 `arm_link6_T_camera_color_optical` 手眼标定为位置
`(0.0559054476, 0.0026732239, 0.0767149320)` m、wxyz 四元数
`(0.3377891849, -0.6214992221, 0.6185057335, -0.3421810063)`。由于标定板弯曲，
当前在 ROS optical frame 沿相机 `-Y` 平移 `0.02 m` 作为可追溯的视觉对齐
修正，最终仿真安装位置为 `(0.0666580792, 0.0028071889, 0.0935779972)` m，
旋转保持不变。该平移把夹爪近端移到图像下方，同时保持相机到 TCP 的光轴深度
约 `0.1270 m`。禁止通过 optical `+Z` 前移和近裁剪隐藏底座：`+0.04 m` 已在
良渚真实 pipeline 中复现 near clipping 切入可乐 mesh。box1→box2 任务启用逐帧
wrist/目标表面间距门禁，要求可见目标表面至少位于 near clipping 之后 `0.01 m`；
违规 episode 不进入训练集。历史 metadata 中若出现
`hand_eye_calibration_with_visual_alignment_v2`，pipeline 重新处理该 summary 时会直接拒绝，
对应输出不得用于 VLA 训练。该值来自实机画面约束，不是新的精密手眼标定。front camera
仅更新为同一套内参，安装外参仍沿用现有
`base/head_cam` 配置。front/wrist 请求非 640×480 分辨率时 runtime 会直接报错。

## 3. 常用命令与 CLI 参数

### 3.1 常用命令

以下命令都从仓库根目录运行。每个示例都写明 `--scene-profile`，命令历史和日志中
因此会保留场景信息。

```bash
cd "$PCT_SCENE_ROOT"
```

#### GUI 单次 full-physics

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_gui_seed7000" \
  --seed 7000 \
  --navigation-visual-mode full \
  --no-record-dataset \
  --no-record-video \
  --no-headless \
  --keep-window-open
```

GUI 适合观察单个 episode。`--keep-window-open` 会在 pipeline 结束后保留窗口。
判断运行结果时，以 `summary.json` 和 `events.jsonl` 为准。该命令使用良渚默认的
full/NuRec 场景并自动启用 USDA authored stage lights，但不创建训练相机 render
product。

#### Headless 单次 full-physics

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_single_seed7000" \
  --seed 7000 \
  --headless
```

#### Headless batch 数据采集

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_batch_seed7000_n20" \
  --num-episodes 20 \
  --seed 7000 \
  --no-record-video
```

`liangzhu` profile 会加载良渚任务、PCT 单层地图、locomotion checkpoint 和
随机化配置。机器人 yaw 由 task JSON 在 `[-180°, 180°]` 内采样，不由 CLI 设置。
上述命令使用 profile 的 collision 量产视觉；`--no-record-video` 只关闭展示视频，
LeRobot 的 front/wrist/overview 三路 5Hz 数据仍会完整导出。
运行别墅场景时，将 profile 改为 `multi_floor`。

#### 复现固定任务

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_fixed_baseline" \
  --seed 0 \
  --no-randomize-task \
  --no-randomize-base-goal \
  --navigation-visual-mode collision \
  --no-record-video \
  --headless
```

关闭任务和 base goal 随机化后，程序使用 task JSON 中的固定布局。`--seed` 仍会
写入输出文件，但不会改变机器人、box1、box2、可乐或 base goal 的位置。

#### 显示随机化区域

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_randomization_debug_seed7000" \
  --seed 7000 \
  --show-randomization-debug \
  --show-planned-trajectories \
  --no-record-dataset \
  --no-record-video \
  --no-headless \
  --keep-window-open
```

#### 验证 LeRobot 数据集并导出 Rerun

```bash
conda activate isaac_locomani

python -B \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root "$PCT_SCENE_OUTPUT/liangzhu_batch_seed7000_n20/lerobot_dataset"

conda activate lerobot_rerun

python \
  tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root "$PCT_SCENE_OUTPUT/liangzhu_batch_seed7000_n20/lerobot_dataset" \
  --episode-index 0 \
  --max-frames 200 \
  --out "$PCT_SCENE_OUTPUT/liangzhu_batch_seed7000_n20/episode_000000.rrd"
```

### 3.2 CLI 参数表

#### `scripts/pipeline/run_full_physics_pipeline.py`

默认模式是 full-physics。下表列出日常运行和验收常用的参数。PCT 和楼梯的完整
试验参数请查看 `python -B scripts/pipeline/run_full_physics_pipeline.py --help`。
机器人 yaw 范围、扇形半径和物体间距属于 task 配置，不是 CLI 参数。良渚使用
`robot_yaw_range_deg=[-180, 180]`，别墅 profile 默认使用固定任务。模式参数主要用于
smoke 测试和调试。


| 参数                                                 | 类型 / 默认            | 说明                                                                                                                    |
| ---------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `--scene-profile`                                    | `liangzhu`             | 选择场景；可用`--list-scene-profiles` 查看，别墅使用 `multi_floor`                                                      |
| `--list-scene-profiles` / `--check-scene-assets`     | 只读检查               | 列出动态发现的 profile，或检查所选场景资产后退出                                                                        |
| `--task-json`                                        | 由 profile 提供        | 良渚可乐任务或别墅苹果任务；使用 CLI 覆盖时会校验 scene_profile                                                         |
| `--output-dir`                                       | `outputs/<profile>`    | 输出目录；数据采集建议使用空间充足的独立磁盘                                                                            |
| `--num-episodes`                                     | `1`                    | 导航主线在 epoch/reset/ack 完成前只允许单 episode                                                                        |
| `--reuse-isaac-stage` / `--no-reuse-isaac-stage`     | 默认开启               | 多 episode 复用 IsaacLab env/stage；排查隔离问题时可关闭                                                               |
| `--seed`                                             | `0`                    | episode seed；相同 task/config/seed 复现同一布局                                                                        |
| `--randomize-task` / `--no-randomize-task`           | 由 profile 提供        | 良渚默认开启；别墅默认关闭；CLI 开关会覆盖 profile 设置                                                                 |
| `--show-randomization-debug`                         | 默认关闭               | 显示矩形/前向扇区和采样点 USD guide                                                                                     |
| `--show-planned-trajectories`                        | 默认关闭               | 显示 PCT 路径和 CuRobo TCP 轨迹 guide                                                                                   |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 由 profile 提供        | 良渚默认开启；别墅默认关闭                                                                                              |
| `--keep-window-open` / `--no-keep-window-open`       | 默认关闭               | 结束后保留 GUI；必须配合`--no-headless`                                                                                 |
| `--headless` / `--no-headless`                       | 默认`--no-headless`    | 是否无界面运行                                                                                                          |
| `--navigation-visual-mode`                           | `collision`            | PCT→SCAN 导航强制 collision；`full` 只用于不执行导航的场景检查                                                           |
| `--scene-light-mode`                                 | `auto`                 | 最终为 `full` 时自动使用当前 profile 的 USD 原场景灯光，`collision` 自动使用相机补光；可用 `camera`/`stage` 覆盖       |
| `--global-planner`                                   | `pct`                  | 导航主线固定为 PCT；A* 仅保留为隔离历史测试                                                                             |
| `--pct-collision-ply-path`                           | 由 profile 提供        | 每个场景必须声明自己的 collision PLY，禁止静默借用别墅地图                                                              |
| `--pct-no-fallback`                                  | 强制开启               | PCT 失败即拒绝 episode；导航主线拒绝 PCT→A* fallback                                                                    |
| `--enable-navigation-ros2-bridge`                    | 导航模式强制开启       | 启用 OGN 传感器、PCT goal、Path、cmd_vel 与完成事件链；仅固定速度楼梯探针可关闭                                          |
| `--ros2-reference-path-topic`                        | `/pct/global_path`     | 生产参考 Path；手工 Path smoke 才显式改为 `/initial_path`                                                               |
| `--ros2-pct-goal-topic`                              | `/pct/goal`            | OGN 每个 pipeline generation 发布一次 base 高度 PoseStamped 目标                                                        |
| `--pct-coord-mode`                                   | 由 profile 提供        | 良渚为`identity`；别墅为 `sim_to_pct_180deg`                                                                            |
| `--policy-profile`                                   | `pct_multifloor`       | 复用已验证的 RL locomotion profile                                                                                      |
| `--locomotion-checkpoint`                            | Go2-X5 model_26000     | 默认使用仓库 checkpoint                                                                                                 |
| `--require-locomotion-checkpoint`                    | 默认开启               | checkpoint 缺失时立即失败                                                                                               |
| `--record-video`                                     | full-physics 默认开启  | 可用`--no-record-video` 关闭；展示视频固定为 25 FPS                                                                     |
| `--record-dataset`                                   | 默认开启               | 保存同步帧与 LeRobot 数据；GUI 检查可用`--no-record-dataset`                                                            |
| `--dataset-camera-keys`                              | `front wrist overview` | 选择训练数据相机流；主要用于渲染后端诊断，至少包含 front                                                                |
| `--video-mode`                                       | 由 profile 提供        | 两个 profile 默认均为`composite`；同步拼接 overview/front/wrist，也可只选单路或 `all`                                   |
| `--video-out`                                        | 可选                   | 视频输出目录或单个`.mp4`；多路或多 episode 需要传目录                                                                   |
| `--video-width` / `--video-height`                   | `1280` / `720`         | overview 捕获分辨率；不改变 front/wrist observation                                                                     |
| `--overview-camera-mode`                             | 由 profile 提供        | 良渚为`fixed`；别墅为 `auto`，并按 schedule 切换                                                                        |
| `--overview-camera-prim-path`                        | 由 profile 提供        | 良渚为`/World/overview`；别墅从 `/World/Camera0` 开始                                                                   |
| `--overview-capture-backend`                         | `viewport`             | overview 取帧后端；`viewport` 抓最终视口画面最接近 GUI，`render_product` 使用 Replicator RGB，`auto` 先 viewport 后回退 |
| `--overview-initial-hold-frames`                     | `160`                  | 初始`third_person1` 的最少保持帧数，避免 reset 后立即切到导航镜头                                                       |
| `--overview-exposure`                                | `0.0`                  | overview 线性 RGB 转视频前曝光补偿，单位 EV stops                                                                       |
| `--overview-gamma`                                   | `2.2`                  | overview 线性 RGB 转 sRGB 的 gamma；设为`1.0` 可关闭 gamma 提亮                                                         |
| `--pick-plan-json`                                   | 可选                   | 仅 manipulation apply smoke 使用；full-physics 禁止                                                                     |
| `--place-plan-json`                                  | 可选                   | 仅 manipulation apply smoke 使用；full-physics 禁止                                                                     |
| `--dry-run`                                          | mode                   | 无 Isaac 内存后端状态机验证                                                                                             |
| `--simulation-smoke`                                 | mode                   | 只验证真实 Isaac stage/reset                                                                                            |
| `--navigation-smoke`                                 | mode                   | 只验证 nav 到 pick                                                                                                      |
| `--navigation-carry-smoke`                           | mode                   | 验证 carry 姿态下 nav 到 place                                                                                          |
| `--pct-plan-preview`                                 | mode                   | GUI 中只预览 PCT 路径，不执行 locomotion                                                                                |
| `--pick-smoke`                                       | mode                   | 只运行到 pick 成功验证                                                                                                  |
| `--manipulation-smoke`                               | mode                   | 使用假后端验证 manipulation action 合同                                                                                 |
| `--manipulation-apply-smoke`                         | mode                   | 真实 Isaac 中验证 arm/gripper action 下发                                                                               |

#### `scripts/pipeline/run_full_physics_batch.py`

默认模式是 full-physics，默认 headless，默认继续执行失败后的 episode。batch
默认只启动一个 Isaac Sim 子进程并复用 stage，在结束时只合并通过质量检查的数据。


| 参数                                                 | 类型 / 默认            | 说明                                                                         |
| ---------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| `--scene-profile`                                    | `liangzhu`             | 选择`liangzhu` 或 `multi_floor`，其余参数由 profile 提供                     |
| `--task-json`                                        | 由 profile 提供        | 使用 CLI 覆盖时，单 episode 入口会校验 task 与场景兼容性                     |
| `--output-dir`                                       | 必填                   | batch 输出目录；必须使用新目录，避免混入旧摘要                               |
| `--num-episodes`                                     | `1`                    | episode 数量                                                                 |
| `--reuse-isaac-process` / `--no-reuse-isaac-process` | 默认开启               | 复用单个 Isaac 进程/stage；关闭后每条独立进程                                |
| `--seed`                                             | `0`                    | 首个 seed，后续使用`seed + episode_index`                                    |
| `--randomize-task` / `--no-randomize-task`           | 由 profile 提供        | 良渚默认开启；别墅默认关闭                                                   |
| `--show-randomization-debug`                         | 默认关闭               | 显示矩形/前向扇区；通常只用于 GUI 单 episode                                 |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 由 profile 提供        | 良渚默认开启；别墅默认关闭                                                   |
| `--headless` / `--no-headless`                       | 默认 headless          | batch 是否无界面运行；批量采集建议使用 headless                              |
| `--navigation-visual-mode`                           | 由 profile 提供        | 两个 profile 量产默认均为`collision`                                         |
| `--global-planner`                                   | 由 profile 提供        | 两个 profile 均使用 PCT                                                      |
| `--pct-server-script`                                | legacy batch 参数      | 仅供旧 compatible/full-physics 诊断；生产 ROS 2 主线使用进程内 upstream backend |
| `--pct-tomogram-path` / `--pct-walkable-path`        | 由 profile 提供        | 自动选择良渚单层或别墅多楼层地图                                             |
| `--pct-collision-ply-path`                           | 由 profile 提供        | 自动选择对应场景 collision PLY                                               |
| `--pct-no-fallback` / `--pct-allow-fallback`         | 默认禁止回退           | 默认 PCT 失败即拒绝 episode                                                  |
| `--pct-coord-mode`                                   | 由 profile 提供        | 良渚为`identity`；别墅为 `sim_to_pct_180deg`                                 |
| `--policy-profile`                                   | `pct_multifloor`       | 复用已验证的 RL locomotion profile                                           |
| `--locomotion-checkpoint`                            | Go2-X5 model_26000     | 默认使用仓库 checkpoint                                                      |
| `--require-locomotion-checkpoint`                    | 默认开启               | checkpoint 缺失时立即失败                                                    |
| `--continue-on-failure` / `--no-continue-on-failure` | 默认继续               | 单 episode 失败后是否继续                                                    |
| `--pick-plan-json`                                   | 可选                   | 非 full-physics smoke 可转发离线 pick plan                                   |
| `--place-plan-json`                                  | 可选                   | 非 full-physics smoke 可转发离线 place plan                                  |
| `--progress-interval-s`                              | `5.0`                  | 真实状态 heartbeat 间隔；startup/pending 低信息状态最多每 30 秒打印一次      |
| `--color` / `--no-color`                             | 默认开启               | 是否使用 ANSI 彩色输出；保存 CI 日志时建议关闭                               |
| `--record-video`                                     | full-physics 默认开启  | 沿用单 episode/profile 默认；可用`--no-record-video` 关闭                    |
| `--dataset-camera-keys`                              | `front wrist overview` | 转发训练数据相机流；可用`--dataset-camera-keys front wrist` 做诊断           |
| `--video-mode`                                       | 由 profile 提供        | profile 默认为`all`；`font` 是 `front` 的兼容别名                            |
| `--video-out`                                        | 可选                   | 视频输出根目录；batch 写入其下的`episode_XXXXXX/` 子目录，不支持单个 `.mp4`  |
| `--video-width` / `--video-height`                   | `1280` / `720`         | overview 捕获分辨率；不改变 front/wrist observation                          |
| `--overview-camera-mode`                             | 由 profile 提供        | 良渚固定相机；别墅按 schedule 自动切换                                       |
| `--overview-camera-prim-path`                        | 由 profile 提供        | image/video/GUI 共用的初始 overview Camera prim                              |
| `--overview-capture-backend`                         | `viewport`             | overview 取帧后端；`viewport` 最接近 GUI，`render_product` 用于排查 fallback |
| `--overview-initial-hold-frames`                     | `160`                  | 初始`third_person1` 的最少保持帧数                                           |
| `--overview-exposure`                                | `0.0`                  | overview 曝光补偿，单位 EV stops                                             |
| `--overview-gamma`                                   | `2.2`                  | overview 线性 RGB 转 sRGB gamma                                              |
| `--dry-run`                                          | mode                   | 子进程 dry-run                                                               |
| `--simulation-smoke`                                 | mode                   | 子进程 simulation smoke                                                      |
| `--navigation-smoke`                                 | mode                   | 子进程 navigation smoke                                                      |
| `--navigation-carry-smoke`                           | mode                   | 子进程 navigation carry smoke                                                |
| `--manipulation-apply-smoke`                         | mode                   | 子进程 manipulation apply smoke                                              |

#### `tools/lerobot_to_rerun.py`

该脚本必须在 `lerobot_rerun` 环境中运行。


| 参数              | 类型 / 默认   | 说明                                |
| ----------------- | ------------- | ----------------------------------- |
| `--repo-id`       | 必填          | LeRobot repo_id 或本地数据集名称    |
| `--root`          | 可选          | 本地 LeRobot dataset root           |
| `--episode-index` | `0`           | 要转换的 episode 编号               |
| `--max-frames`    | `-1`          | 最大转换帧数，`-1` 表示完整 episode |
| `--out`           | `episode.rrd` | 输出 Rerun`.rrd` 路径               |
| `--spawn`         | 默认关闭      | 转换时直接打开 Rerun Viewer         |

#### `scripts/pipeline/validate_lerobot_episode.py`


| 参数             | 类型 / 默认 | 说明                                                    |
| ---------------- | ----------- | ------------------------------------------------------- |
| `--episode-dir`  | 二选一      | 校验单个 full-physics episode 目录中的`lerobot_dataset` |
| `--dataset-root` | 二选一      | 校验合并后的 LeRobot dataset root                       |
