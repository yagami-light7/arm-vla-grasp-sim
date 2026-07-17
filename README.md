# PCT Scene: Go2-X5 多场景移动操作与数据采集

PCT Scene 用一套代码运行良渚单层任务和别墅多楼层任务。使用`--scene-profile` 选择场景后，程序会加载对应的任务、PCT 地图、坐标变换、随机化、楼梯配置、视觉层和 overview 相机。切换场景不需要更换 worktree，也不需要重复输入整组参数。

## 场景与流程


| Profile           | 任务与默认行为                                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `liangzhu`              | 良渚单层任务，将可乐搬到鼠标垫。开启任务随机化，使用`identity` PCT 坐标和 Gaussian 视觉层；overview 相机固定为 `/World/overview`。          |
| `multi_floor`           | 别墅多楼层任务，将苹果从 F1 搬到 F2。使用`sim_to_pct_180deg` PCT 坐标、楼梯锚点和 collision 视觉；关闭任务随机化，overview 相机按阶段切换。 |

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

## 文档导航

- [1. 环境部署](#1-环境部署)
- [2. 仿真运行与数据导出](#2-仿真运行与数据导出)
- [3. 常用命令与 CLI 参数](#3-常用命令与-cli-参数)

## 1. 环境部署

### 1.1 系统要求

已验证环境为 Ubuntu Linux、支持 CUDA 12.x 的 NVIDIA 驱动、Python 3.11、Isaac Sim 5.1、Isaac Lab 2.3.x 和 cuRobo 0.8.x。显存少于 12 GB 时，建议使用
collision 视觉模式采集数据。

安装系统依赖并确认驱动可用：

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg build-essential cmake ninja-build libgl1 libglib2.0-0

nvidia-smi
git lfs install
conda --version
```

如果系统尚未安装 conda，请先安装 Miniforge、Miniconda 或 Anaconda，然后重新打开终端。

实际显存需求取决于场景和视觉模式。良渚 full/Gaussian 模式使用 NuRec 资产，8 GB RTX 4060 Laptop 已知会在 RGB render product 启动后触发 CUDA illegal address 700。

### 1.2 获取代码

```bash
git clone https://github.com/yagami-light7/arm-vla-grasp-sim.git pct_scene
cd pct_scene
git checkout pct_scene
git lfs pull
```

Git LFS 只会下载已纳入版本控制的大文件。其他运行时资产请按`source/scene/<scene>/runtime_asset_manifest.json` 中的清单准备。

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

仿真运行使用 `scripts/navigation/pct_grid_server.py`，不依赖 `external/PCT`。
只有建图、查阅 PCT 原实现或扩展导航功能时才需要克隆外部仓库：

```bash
git clone https://github.com/BoZhiStudying233/PCT.git external/PCT
```

良渚场景的 visual 和 collision 资产可以放在任意磁盘，然后通过环境变量绑定：

```bash
export LIANGZHU_VISUAL_USDZ=/path/to/liangzhu_cropped_nozip64.usdz
export LIANGZHU_COLLISION_USD=/path/to/liangzhu_collision.usda
```

别墅场景的 visual 和 collision 资产使用相对路径。运行前请按 manifest 补齐
`source/scene/multifloor` 下的 runtime 资产。其他场景的地图不能替代这些文件。

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

良渚 GUI：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --seed 7000 \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_gui_seed7000" \
  --no-record-dataset \
  --no-record-video \
  --no-headless \
  --keep-window-open
```

该命令使用 `full` Gaussian/NuRec 视觉，但不创建 RGB render product，
适合在采集数据前检查场景、随机化和轨迹。在 8 GB RTX 4060 Laptop 与
Isaac Sim 5.1 的已验证环境中，使用以下命令采集数据：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --seed 7000 \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_seed7000" \
  --navigation-visual-mode collision \
  --no-record-video \
  --headless
```

`--no-record-video` 只关闭额外的展示视频。LeRobot 仍会从同步数据帧生成
front、wrist 和 overview 三路 MP4。显存充足时，可以去掉
`--navigation-visual-mode collision` 并重新验证 full 视觉采集。训练数据应注明视觉模式，
不要直接混合 collision 和 full 数据。

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

该 smoke 会关闭 stair-float，只测试低层 policy 的纯物理楼梯执行。
dog-only policy 可能报告 `stair_locomotion_stalled`。默认多楼层 pipeline 启用
stair-float，该模式的结果不等同于纯物理跨层 locomotion 成功。

full-physics 默认保存 LeRobot 数据，并录制 profile 指定的视频流。如果只做物理诊断，
可以添加 `--no-record-dataset --no-record-video` 减少磁盘占用。

### 2.3 批量采集

良渚随机 batch：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_seed7000_n20" \
  --num-episodes 20 \
  --seed 7000 \
  --navigation-visual-mode collision \
  --no-record-video
```

别墅多楼层 batch（profile 默认使用固定任务）：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile multi_floor \
  --output-dir "$PCT_SCENE_OUTPUT/multi_floor_seed0_n5" \
  --num-episodes 5 \
  --seed 0
```

batch 为每个 episode 启动独立的 Isaac 子进程，默认使用 headless 模式。
episode seed 为 `seed + episode_index`。失败的 episode 会保留诊断文件；通过物理来源和
训练质量检查的 episode 会合并到 `<output-dir>/lerobot_dataset`。

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

full-physics 成功后会保留运行诊断文件，并导出 LeRobot v2 数据集。单个 episode
的主要文件如下：

```text
episode_000000/
├── task.json                  # 任务、seed 和随机化结果
├── events.jsonl              # 状态机事件
├── summary.json              # 运行结果与数据质量检查
├── frames.jsonl              # 原始帧级记录
├── images/                     # 相机图像
├── videos/                     # 相机视频
└── lerobot_dataset/
    ├── data/                   # Parquet episode
    ├── videos/                 # front / wrist 视频
    ├── meta/                   # LeRobot metadata
    └── episodes/               # VLA 训练兼容目录
```

batch 会在输出根目录中保留每个源 episode，并将通过质量检查的数据合并到
`lerobot_dataset/`。`batch_summary.jsonl` 记录每个 episode 的结果，
`lerobot_export_manifest.json` 记录合并来源。

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
  --no-record-dataset \
  --no-record-video \
  --no-headless \
  --keep-window-open
```

GUI 适合观察单个 episode。`--keep-window-open` 会在 pipeline 结束后保留窗口。
判断运行结果时，以 `summary.json` 和 `events.jsonl` 为准。该命令使用良渚默认的
full/NuRec 场景，但不创建训练相机 render product。

#### Headless 单次 full-physics

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_single_seed7000" \
  --seed 7000 \
  --navigation-visual-mode collision \
  --no-record-video \
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
  --navigation-visual-mode collision \
  --no-record-video
```

`liangzhu` profile 会加载良渚任务、PCT 单层地图、locomotion checkpoint 和
随机化配置。机器人 yaw 由 task JSON 在 `[-180°, 180°]` 内采样，不由 CLI 设置。
上述命令使用 collision 视觉，适合显存较小的机器。运行别墅场景时，将 profile 改为
`multi_floor`。

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
写入输出文件，但不会改变机器人、可乐、鼠标垫或 base goal 的位置。

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


| 参数                                         | 类型 / 默认            | 说明                                                                                                                    |
| ---------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `--scene-profile`                                    | `liangzhu`             | 选择场景；可用`--list-scene-profiles` 查看，别墅使用 `multi_floor`                                                      |
| `--list-scene-profiles` / `--check-scene-assets`     | 只读检查               | 列出动态发现的 profile，或检查所选场景资产后退出                                                                        |
| `--task-json`                                        | 由 profile 提供        | 良渚可乐任务或别墅苹果任务；使用 CLI 覆盖时会校验 scene_profile                                                         |
| `--output-dir`                                       | `outputs/<profile>`    | 输出目录；数据采集建议使用空间充足的独立磁盘                                                                            |
| `--num-episodes`                                     | `1`                    | episode 数量；真实 Isaac 模式只支持 1                                                                                   |
| `--seed`                                             | `0`                    | episode seed；相同 task/config/seed 复现同一布局                                                                        |
| `--randomize-task` / `--no-randomize-task`           | 由 profile 提供        | 良渚默认开启；别墅默认关闭；CLI 开关会覆盖 profile 设置                                                                 |
| `--show-randomization-debug`                         | 默认关闭               | 显示矩形/前向扇区和采样点 USD guide                                                                                     |
| `--show-planned-trajectories`                        | 默认关闭               | 显示 PCT 路径和 CuRobo TCP 轨迹 guide                                                                                   |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 由 profile 提供        | 良渚默认开启；别墅默认关闭                                                                                              |
| `--keep-window-open` / `--no-keep-window-open`       | 默认关闭               | 结束后保留 GUI；必须配合`--no-headless`                                                                                 |
| `--headless` / `--no-headless`                       | 默认`--no-headless`    | 是否无界面运行                                                                                                          |
| `--navigation-visual-mode`                           | 由 profile 提供        | 良渚为`full`；别墅为 `collision`；可用 CLI 覆盖                                                                         |
| `--scene-light-mode`                                 | `camera`               | `camera` 用于保存图像；`stage` 使用 USD 原场景灯光                                                                      |
| `--global-planner`                                   | `pct`                  | 良渚默认使用 PCT；可切换为`astar`                                                                                       |
| `--pct-collision-ply-path`                           | 由 profile 提供        | 每个场景必须声明自己的 collision PLY，禁止静默借用别墅地图                                                              |
| `--pct-no-fallback` / `--pct-allow-fallback`         | 默认禁止回退           | 默认 PCT 失败即拒绝 episode                                                                                             |
| `--pct-coord-mode`                                   | 由 profile 提供        | 良渚为`identity`；别墅为 `sim_to_pct_180deg`                                                                            |
| `--policy-profile`                                   | `pct_multifloor`       | 复用已验证的 RL locomotion profile                                                                                      |
| `--locomotion-checkpoint`                            | Go2-X5 model_26000     | 默认使用仓库 checkpoint                                                                                                 |
| `--require-locomotion-checkpoint`                    | 默认开启               | checkpoint 缺失时立即失败                                                                                               |
| `--record-video`                                     | full-physics 默认开启  | 可用`--no-record-video` 关闭；展示视频固定为 25 FPS                                                                     |
| `--record-dataset`                                   | 默认开启               | 保存同步帧与 LeRobot 数据；GUI 检查可用`--no-record-dataset`                                                            |
| `--dataset-camera-keys`                              | `front wrist overview` | 选择训练数据相机流；主要用于渲染后端诊断，至少包含 front                                                                |
| `--video-mode`                                       | 由 profile 提供        | 两个 profile 默认均为`all`；也可只选 overview/front/wrist                                                               |
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
会为每个 episode 启动独立的 Isaac Sim 子进程，并在结束时只合并通过质量检查的数据。


| 参数                                                 | 类型 / 默认            | 说明                                                                         |
| ---------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| `--scene-profile`                                    | `liangzhu`             | 选择`liangzhu` 或 `multi_floor`，其余参数由 profile 提供                     |
| `--task-json`                                        | 由 profile 提供        | 使用 CLI 覆盖时，单 episode 入口会校验 task 与场景兼容性                     |
| `--output-dir`                                       | 必填                   | batch 输出目录；必须使用新目录，避免混入旧摘要                               |
| `--num-episodes`                                     | `1`                    | episode 数量                                                                 |
| `--seed`                                             | `0`                    | 首个 seed，后续使用`seed + episode_index`                                    |
| `--randomize-task` / `--no-randomize-task`           | 由 profile 提供        | 良渚默认开启；别墅默认关闭                                                   |
| `--show-randomization-debug`                         | 默认关闭               | 显示矩形/前向扇区；通常只用于 GUI 单 episode                                 |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 由 profile 提供        | 良渚默认开启；别墅默认关闭                                                   |
| `--headless` / `--no-headless`                       | 默认 headless          | batch 是否无界面运行；批量采集建议使用 headless                              |
| `--navigation-visual-mode`                           | 由 profile 提供        | 良渚为`full`；别墅为 `collision`                                             |
| `--global-planner`                                   | 由 profile 提供        | 两个 profile 均使用 PCT                                                      |
| `--pct-server-script`                                | 由 profile 提供        | 两个 profile 均使用仓库内的`pct_grid_server.py`                              |
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
| `--progress-interval-s`                              | `5.0`                  | heartbeat 进度打印间隔                                                       |
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
