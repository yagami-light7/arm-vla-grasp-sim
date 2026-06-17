# Go2-X5 loco-manipulation pipeline

**本仓库实现loco-manipulation的数据采集流程，导出用于VLA模型训练的数据集**

大致流程为：



```mermaid
graph LR
    A["随机化"] --> B["nav2pick"]
    B --> C["pick"]
    C --> D["nav2place"]
    D --> E["place"]
    E --> F["LeRobot 数据导出"]
```


## 一、环境依赖

### Isaac Sim 5.1 / Isaac Lab 环境


| Package                  | Version        | 用途                            |
| ------------------------ | -------------- | ------------------------------- |
| Python                   | `3.11.15`      | Isaac Sim 5.1 当前环境 Python   |
| `isaacsim`               | `5.1.0.0`      | Isaac Sim Kit / runtime         |
| `isaacsim-core`          | `5.1.0.0`      | Isaac Sim core API              |
| `isaacsim-kernel`        | `5.1.0.0`      | Isaac Sim kernel 依赖           |
| `isaaclab`               | `0.54.3`       | Isaac Lab runtime               |
| `isaaclab-rl`            | `0.4.7`        | Isaac Lab RL wrapper            |
| `rsl-rl-lib`             | `3.1.2`        | Go2-X5 locomotion policy runner |
| `rl-games`               | `1.6.1`        | Isaac Lab 依赖                  |
| `nvidia-curobo`          | `0.0.0`        | cuRobo planner                  |
| `warp-lang`              | `1.13.0`       | cuRobo / NVIDIA Warp            |
| `torch`                  | `2.7.0+cu128`  | CUDA tensor / policy / planner  |
| `torchvision`            | `0.22.0+cu128` | 图像工具                        |
| `torchaudio`             | `2.7.0+cu128`  | torch 环境配套包                |
| `numpy`                  | `1.26.0`       | Isaac Sim 兼容 NumPy            |
| `scipy`                  | `1.15.3`       | 数值工具                        |
| `packaging`              | `23.0`         | Isaac Sim 版本约束              |
| `psutil`                 | `5.9.8`        | Isaac Sim kernel 版本约束       |
| `websockets`             | `12.0`         | Isaac Sim kernel 版本约束       |
| `pillow`                 | `11.3.0`       | 图像保存                        |
| `opencv-python-headless` | `4.11.0.86`    | MP4 编码和帧处理                |
| `pyarrow`                | `24.0.0`       | LeRobot parquet 物化            |
| `pandas`                 | `3.0.3`        | 表格处理                        |
| `tqdm`                   | `4.67.3`       | 进度条                          |
| `imageio`                | `2.37.0`       | 视频/图像 I/O                   |
| `imageio-ffmpeg`         | `0.6.0`        | ffmpeg 后端                     |
| `gymnasium`              | `1.2.1`        | Isaac Lab env API               |
| `hydra-core`             | `1.3.2`        | Isaac Lab 配置                  |
| `omegaconf`              | `2.3.0`        | Hydra 配置                      |
| `trimesh`                | `4.5.1`        | mesh / collision 诊断           |
| `networkx`               | `3.3`          | 图搜索辅助                      |
| `matplotlib`             | `3.10.3`       | debug 可视化                    |

可以使用以下命令创建一个conda虚拟环境

```bash
conda create -n isaac_locomani python=3.11 -y
conda activate isaac_locomani
python -m pip install -r requirements/isaacsim51_runtime.txt
```

### LeRobot / Rerun 环境


| Package           | Version        | 用途                           |
| ----------------- | -------------- | ------------------------------ |
| Python            | `3.10.20`      | LeRobot/Rerun 普通 Python 环境 |
| `lerobot`         | `0.4.4`        | LeRobot v2 dataset API         |
| `rerun-sdk`       | `0.26.2`       | `.rrd` 可视化记录              |
| `numpy`           | `2.2.6`        | 数组处理                       |
| `pandas`          | `2.3.3`        | parquet / metadata 检查        |
| `pyarrow`         | `24.0.0`       | LeRobot parquet 读取           |
| `pillow`          | `12.2.0`       | 图片读取                       |
| `opencv-python`   | `4.13.0.92`    | 视频帧处理                     |
| `tqdm`            | `4.68.2`       | 转换进度                       |
| `imageio`         | `2.37.3`       | 视频/图像 I/O                  |
| `imageio-ffmpeg`  | `0.6.0`        | ffmpeg 后端                    |
| `torch`           | `2.10.0+cu128` | LeRobot tensor 数据            |
| `torchvision`     | `0.25.0+cu128` | 图像 tensor 工具               |
| `pyyaml`          | `6.0.3`        | metadata 配置                  |
| `huggingface-hub` | `0.35.3`       | LeRobot/HF 数据集工具          |
| `datasets`        | `4.8.5`        | HF dataset 工具                |
| `safetensors`     | `0.8.0`        | torch/模型数据依赖             |
| `av`              | `15.1.0`       | 视频解码依赖                   |
| `packaging`       | `25.0`         | 版本解析                       |

可以使用以下命令创建一个conda虚拟环境

```bash
conda create -n lerobot_rerun python=3.10 -y
conda activate lerobot_rerun
python -m pip install -r requirements/lerobot_rerun.txt
```

## 二、文件结构

当前主要结构：

```text
scripts/
└── pipeline/
    ├── run_full_physics_pipeline.py    # 单 episode / smoke 入口
    ├── run_full_physics_batch.py       # 批量自动化入口
    └── validate_lerobot_episode.py     # LeRobot 数据集校验入口

tools/
└── lerobot_to_rerun.py                 # LeRobot episode -> Rerun .rrd

source/
├── interfaces/                         # navigation / manipulation / simulation / recording 协议
├── pipeline/                           # config、状态机、pipeline 主循环、工厂函数
├── simulation/                         # IsaacLab runtime、viewport、collision patch
├── navigation/                         # A* / DWA / Go2 locomotion adapter
├── manipulation/                       # current-state cuRobo、planner server、arm executor
├── diagnostics/                        # 成功判据、随机化可视化、报告
├── recording/                          # raw episode、LeRobot v2 写出与校验
└── tasks/                              # task loader、随机化、episode spec

tasks/
└── nav_pick_place_apple_contact.json   # 当前主任务

checkpoints/
└── go2_x5/flat/model_8500.pt           # 本地 locomotion checkpoint，通常不提交 git
```

## 三、Pipeline 流程

![pipeline](requirements/image_video/pipeline.png)!

状态流：

```text
build_stage
-> reset_episode
-> plan_nav_to_pick
-> exec_nav_to_pick
-> verify_pick_reachable
-> plan_pick
-> exec_pick
-> verify_pick_success
-> plan_nav_to_place
-> exec_nav_to_place
-> verify_place_reachable
-> plan_place
-> exec_place
-> verify_place_success
-> export_lerobot
-> cleanup_episode
-> done
```


| 组件              | 实现方式                                                      |
| ----------------- | ------------------------------------------------------------- |
| 单进程 / 单 World | 单 episode 内由`FullPhysicsPipeline` 持有唯一 step loop       |
| nav               | A* + DWA + Isaac Lab Go2 locomotion policy                    |
| pick/place 规划   | 当前仿真状态导出为 cuRobo state/target JSON，在线规划         |
| 执行              | 机械臂通过逐 step position target / gripper target 物理执行   |
| carry             | pick 后 return home，carry 阶段保持 arm home 和 gripper close |
| 稳定模式          | 机械臂阶段默认锁 base root 和 support joints                  |

### Batch 流程

`run_full_physics_batch.py` 当前按 episode 启动子进程。

优点：每个 episode 隔离，失败不会污染后续 episode

缺点：是每个 episode 都要重复启动 Isaac Sim、加载 USD、创建 env 和相机，耗时时间较长。

batch 结束会打印以下表格：


| 列                           | 内容                                        |
| ---------------------------- | ------------------------------------------- |
| `Episode`                    | episode 编号和 seed                         |
| `随机化 Pick / Place XY`     | 本次随机化后的 pick/place 目标 XY           |
| `随机化 BaseGoal / 相对目标` | pick/place base_goal 和相对目标的 XY 偏移   |
| `Pipeline 成功`              | 成功 / 失败                                 |
| `失败 State`                 | 失败时状态机 state                          |
| `LeRobot 数据路径`           | 成功 episode 的 LeRobot manifest 或数据路径 |
| `Episode 耗时`               | 单 episode 墙钟时间                         |

![print_batch](requirements/image_video/print_batch.png)

失败 episode 会保留诊断原始文件，但不会导出 LeRobot 训练入口：

```text
frames.jsonl / events.jsonl / data.csv / samples.jsonl / images    保留，用于排查
lerobot_manifest.json / lerobot_dataset/                           失败时删除或不生成
```

batch 合并统一数据集时只合并成功 episode。

## 四、LeRobot 数据导出

full-physics 成功后会导出 raw episode 和 LeRobot v2 数据。

单 episode 典型输出：

```text
outputs/run_name/episode_000000/
├── task.json
├── events.jsonl
├── frames.jsonl
├── summary.json
├── data.csv
├── samples.jsonl
├── images/
├── videos/
├── lerobot_manifest.json
└── lerobot_dataset/
    ├── data/chunk-000/episode_000000.parquet
    ├── videos/chunk-000/observation.images.front/episode_000000.mp4
    ├── videos/chunk-000/observation.images.wrist/episode_000000.mp4
    └── meta/
```

batch 典型输出：

```text
outputs/batch_run_name/
├── episode_000000/
├── episode_000001/
├── ...
├── batch_summary.jsonl
├── lerobot_export_manifest.json
└── lerobot_dataset/
```

关键 feature：


| Feature                     | 内容                                                     |
| --------------------------- | -------------------------------------------------------- |
| `observation.state`         | base pose、TCP、arm joints、gripper、object 等 17 维状态 |
| `observation.base_velocity` | body-frame base velocity                                 |
| `observation.object_state`  | object pose / velocity                                   |
| `observation.tcp_pose`      | TCP pose                                                 |
| `observation.images.front`  | front camera MP4                                         |
| `observation.images.wrist`  | wrist camera MP4                                         |
| `pipeline_state`            | 当前状态机阶段                                           |
| `action`                    | base3 + arm6 + gripper2，共 11 维                        |
| `next.done`                 | episode 末帧标记                                         |

校验单 episode：

```bash
conda activate isaac_locomani

python scripts/pipeline/validate_lerobot_episode.py \
  --episode-dir outputs/.../episode_000000
```

校验 batch 统一数据集：

```bash
conda activate isaac_locomani

python scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root outputs/.../lerobot_dataset
```

## Rerun 检查

![rerun](requirements/image_video/rerun.png)

Rerun 转换脚本必须在 `lerobot_rerun` 环境中运行：

```bash
conda activate lerobot_rerun

cd /path/to/project

python tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root outputs/full_physics_batch/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out outputs/full_physics_batch/episode_000000.rrd
```

打开：

```bash
conda activate lerobot_rerun

python -m rerun \
  outputs/full_physics_batch/episode_000000.rrd
```

或转换时直接打开 Viewer：

```bash
conda activate lerobot_rerun

python \
  tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root outputs/full_physics_batch/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out outputs/full_physics_batch/episode_000000.rrd \
  --spawn
```

Rerun 路径内容：


| 路径                          | 内容                                        |
| ----------------------------- | ------------------------------------------- |
| `cameras/front/image`         | front camera                                |
| `cameras/wrist/image`         | wrist camera                                |
| `observation/state/*`         | observation.state 逐维 scalar               |
| `observation/base_velocity/*` | base velocity                               |
| `observation/object_state/*`  | object state                                |
| `observation/tcp_pose/*`      | TCP pose                                    |
| `action/*`                    | action 逐维 scalar                          |
| `meta/*`                      | episode/frame/dataset index、pipeline state |
| `robot/ee`                    | 可选末端位姿轨迹                            |

## 五、常见运行命令

以下命令中的 `/path/to/project` 表示本仓库根目录。

### GUI 单次 full-physics

```bash
conda activate isaac_locomani
cd /path/to/project

PYTHONDONTWRITEBYTECODE=1 python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_gui \
  --seed 0 \
  --no-headless \
  --keep-window-open
```

### Headless 单次 full-physics

```bash
conda activate isaac_locomani
cd /path/to/project

PYTHONDONTWRITEBYTECODE=1 python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_headless \
  --seed 0 \
  --headless
```

### Headless batch 数据采集

```bash
conda activate isaac_locomani
cd /path/to/project

PYTHONDONTWRITEBYTECODE=1 python -B \
  scripts/pipeline/run_full_physics_batch.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_batch \
  --num-episodes 20 \
  --seed 0
```

### 显示随机化区域

```bash
conda activate isaac_locomani
cd /path/to/project

PYTHONDONTWRITEBYTECODE=1 python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/randomization_debug_gui \
  --seed 0 \
  --show-randomization-debug \
  --no-headless \
  --keep-window-open
```

```###

```bash
conda activate isaac_locomani
cd /path/to/project

python -B \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root outputs/full_physics_batch/lerobot_dataset

conda activate lerobot_rerun

python \
  tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root outputs/full_physics_batch/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out outputs/full_physics_batch/episode_000000.rrd
```

## 附录:CLI 参数表

### `scripts/pipeline/run_full_physics_pipeline.py`

默认模式是 full-physics。只在需要 smoke/debug 时传模式参数。


| 参数                                                 | 类型 / 默认                     | 说明                                                |
| ---------------------------------------------------- | ------------------------------- | --------------------------------------------------- |
| `--task-json`                                        | 必填                            | 任务 JSON 路径                                      |
| `--output-dir`                                       | `outputs/full_physics_pipeline` | 输出目录                                            |
| `--num-episodes`                                     | `1`                             | episode 数量；真实 Isaac 模式当前只支持 1           |
| `--seed`                                             | `0`                             | 首个 episode seed                                   |
| `--randomize-task` / `--no-randomize-task`           | 默认开启                        | 是否随机化 pick/place 目标 XY                       |
| `--show-randomization-debug`                         | 默认关闭                        | 显示随机区域和采样点 USD guide                      |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 默认开启                        | 是否随机化 pick/place 导航交接 base_goal            |
| `--keep-window-open` / `--no-keep-window-open`       | 默认关闭                        | 结束后保留 GUI；必须配合`--no-headless`             |
| `--headless` / `--no-headless`                       | 默认`--no-headless`             | 是否无界面运行                                      |
| `--pick-plan-json`                                   | 可选                            | 仅 manipulation apply smoke 使用；full-physics 禁止 |
| `--place-plan-json`                                  | 可选                            | 仅 manipulation apply smoke 使用；full-physics 禁止 |
| `--dry-run`                                          | mode                            | 无 Isaac 内存后端状态机验证                         |
| `--simulation-smoke`                                 | mode                            | 只验证真实 Isaac stage/reset                        |
| `--navigation-smoke`                                 | mode                            | 只验证 nav 到 pick                                  |
| `--navigation-carry-smoke`                           | mode                            | 验证 carry 姿态下 nav 到 place                      |
| `--manipulation-smoke`                               | mode                            | 使用假后端验证 manipulation action 合同             |
| `--manipulation-apply-smoke`                         | mode                            | 真实 Isaac 中验证 arm/gripper action 下发           |

### `scripts/pipeline/run_full_physics_batch.py`

默认模式是 full-physics，默认 headless，默认继续执行失败后的 episode。


| 参数                                                 | 类型 / 默认   | 说明                                        |
| ---------------------------------------------------- | ------------- | ------------------------------------------- |
| `--task-json`                                        | 必填          | 任务 JSON 路径                              |
| `--output-dir`                                       | 必填          | batch 输出目录                              |
| `--num-episodes`                                     | `1`           | episode 数量                                |
| `--seed`                                             | `0`           | 首个 seed，后续使用`seed + episode_index`   |
| `--randomize-task` / `--no-randomize-task`           | 默认开启      | 是否随机化 pick/place 目标 XY               |
| `--show-randomization-debug`                         | 默认关闭      | 显示随机区域；通常只用于 GUI 单 episode     |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 默认开启      | 是否随机化导航交接 base_goal                |
| `--headless` / `--no-headless`                       | 默认 headless | batch 是否无界面运行                        |
| `--continue-on-failure` / `--no-continue-on-failure` | 默认继续      | 单 episode 失败后是否继续                   |
| `--pick-plan-json`                                   | 可选          | 非 full-physics smoke 可转发离线 pick plan  |
| `--place-plan-json`                                  | 可选          | 非 full-physics smoke 可转发离线 place plan |
| `--progress-interval-s`                              | `5.0`         | heartbeat 进度打印间隔                      |
| `--color` / `--no-color`                             | 默认开启      | 是否使用 ANSI 彩色输出                      |
| `--dry-run`                                          | mode          | 子进程 dry-run                              |
| `--simulation-smoke`                                 | mode          | 子进程 simulation smoke                     |
| `--navigation-smoke`                                 | mode          | 子进程 navigation smoke                     |
| `--navigation-carry-smoke`                           | mode          | 子进程 navigation carry smoke               |
| `--manipulation-apply-smoke`                         | mode          | 子进程 manipulation apply smoke             |

### `tools/lerobot_to_rerun.py`

该脚本必须在 `lerobot_rerun` 环境中运行。


| 参数              | 类型 / 默认   | 说明                                |
| ----------------- | ------------- | ----------------------------------- |
| `--repo-id`       | 必填          | LeRobot repo_id 或本地数据集名称    |
| `--root`          | 可选          | 本地 LeRobot dataset root           |
| `--episode-index` | `0`           | 要转换的 episode 编号               |
| `--max-frames`    | `-1`          | 最大转换帧数，`-1` 表示完整 episode |
| `--out`           | `episode.rrd` | 输出 Rerun`.rrd` 路径               |
| `--spawn`         | 默认关闭      | 转换时直接打开 Rerun Viewer         |

### `scripts/pipeline/validate_lerobot_episode.py`


| 参数             | 类型 / 默认 | 说明                                                    |
| ---------------- | ----------- | ------------------------------------------------------- |
| `--episode-dir`  | 二选一      | 校验单个 full-physics episode 目录中的`lerobot_dataset` |
| `--dataset-root` | 二选一      | 校验合并后的 LeRobot dataset root                       |
