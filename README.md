# PCT Scene: Go2-X5 多场景移动操作与数据采集

PCT Scene 用一套代码运行良渚单层任务和别墅多楼层任务。使用`--scene-profile` 选择场景后，程序会加载对应的任务、PCT 地图、坐标变换、随机化、楼梯配置、视觉层和 overview 相机。切换场景不需要更换 worktree，也不需要重复输入整组参数。

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
git clone --branch pct_scene --single-branch \
  https://github.com/yagami-light7/arm-vla-grasp-sim.git \
  pct_scene
cd pct_scene
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

仿真运行使用 `scripts/navigation/pct_grid_server.py`，不依赖 `external/PCT`。只有建图、查阅 PCT 原实现或扩展导航功能时才需要克隆外部仓库：

```bash
git clone https://github.com/BoZhiStudying233/PCT.git external/PCT
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

良渚 GUI：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --seed 7000 \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_gui_seed7000" \
  --navigation-visual-mode full \
  --no-record-dataset \
  --no-record-video \
  --no-headless \
  --keep-window-open
```

该命令使用 `full` Gaussian/NuRec 视觉，但不创建 RGB render product，适合在采集数据前检查场景、随机化和轨迹。在 8 GB RTX 4060 Laptop 与Isaac Sim 5.1 的已验证环境中，使用以下命令采集数据：

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

该 smoke 会关闭 stair-float，只测试低层 policy 的纯物理楼梯执行。dog-only policy 可能报告 `stair_locomotion_stalled`。默认多楼层 pipeline 启用stair-float，该模式的结果不等同于纯物理跨层 locomotion 成功。

full-physics 默认保存 LeRobot 数据，并录制 profile 指定的视频流。如果只做物理诊断，可以添加 `--no-record-dataset --no-record-video` 减少磁盘占用。

### 2.3 批量采集

良渚随机 batch：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_seed7000_n20" \
  --num-episodes 20 \
  --seed 7000
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
  --seed 7000
```

`liangzhu` profile 会加载良渚任务、PCT 单层地图、locomotion checkpoint 和
随机化配置。机器人 yaw 由 task JSON 在 `[-180°, 180°]` 内采样，不由 CLI 设置。
上述命令使用 profile 的 collision 量产视觉和 composite 视频默认。
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
| `--num-episodes`                                     | `1`                    | episode 数量；真实 Isaac 模式只支持 1                                                                                   |
| `--seed`                                             | `0`                    | episode seed；相同 task/config/seed 复现同一布局                                                                        |
| `--randomize-task` / `--no-randomize-task`           | 由 profile 提供        | 良渚默认开启；别墅默认关闭；CLI 开关会覆盖 profile 设置                                                                 |
| `--show-randomization-debug`                         | 默认关闭               | 显示矩形/前向扇区和采样点 USD guide                                                                                     |
| `--show-planned-trajectories`                        | 默认关闭               | 显示 PCT 路径和 CuRobo TCP 轨迹 guide                                                                                   |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 由 profile 提供        | 良渚默认开启；别墅默认关闭                                                                                              |
| `--keep-window-open` / `--no-keep-window-open`       | 默认关闭               | 结束后保留 GUI；必须配合`--no-headless`                                                                                 |
| `--headless` / `--no-headless`                       | 默认`--no-headless`    | 是否无界面运行                                                                                                          |
| `--navigation-visual-mode`                           | 由 profile 提供        | 两个 profile 量产默认均为`collision`；良渚 GUI 可显式用 `full` 调试 Gaussian/NUREC                                      |
| `--scene-light-mode`                                 | `auto`                 | 最终为 `full` 时自动使用当前 profile 的 USD 原场景灯光，`collision` 自动使用相机补光；可用 `camera`/`stage` 覆盖       |
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
| `--navigation-visual-mode`                           | 由 profile 提供        | 两个 profile 量产默认均为`collision`                                         |
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
