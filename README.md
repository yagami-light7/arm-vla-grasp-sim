# Arm VLA Full Physics Pipeline

本仓库当前分支是 `exp/full-physics-pipeline`，目标是在 Isaac Sim / Isaac Lab 中运行 Go2-X5 四足机器人 + X5 机械臂 + 双指夹爪的单进程、单 World、物理 nav-pick-place pipeline，并导出 LeRobot v2/v2.1 训练数据。

稳定 video baseline 保留在独立工作区：

```text
/home/light/workspace/arm_vla
```

本仓库工作区：

```text
/home/light/workspace/arm_vla_full_physics
```

不要在本分支修改 `/home/light/workspace/arm_vla`。

## 环境依赖

### Isaac Sim / Isaac Lab 环境

full-physics 仿真运行在 Isaac Sim 5.1 + Isaac Lab 环境中。当前默认 Python：

```text
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python
```

关键依赖：

| 组件 | 用途 | 当前约定 |
| --- | --- | --- |
| Isaac Sim 5.1 | Kit、USD、PhysX、RTX、相机渲染 | 由 `/data/conda_envs/isaacsim51_3dgs_grasp` 提供 |
| Isaac Lab | ManagerBasedRLEnv、AppLauncher、RSL-RL wrapper | `/home/light/workspace/IsaacLab` |
| RobotLab Go2-X5 policy | 四足 locomotion policy | `checkpoints/go2_x5/flat/model_8500.pt` |
| cuRobo | pick/place 在线规划 | 通过 planner server 或 one-shot 子进程调用 |
| SAGE scene payload | 场景 collision / visual USD | 需要挂载 `/mnt/sage_data` |

运行前建议检查：

```bash
cd /home/light/workspace/arm_vla_full_physics

test -x /data/conda_envs/isaacsim51_3dgs_grasp/bin/python
test -f checkpoints/go2_x5/flat/model_8500.pt
test -d /home/light/workspace/IsaacLab
test -d /mnt/sage_data
```

Isaac 环境中不要安装 `rerun-sdk` 和新版 LeRobot。此前 `rerun-sdk` 可能会把 `numpy` 升到 2.x，破坏 Isaac Sim 依赖。Isaac 环境建议保持：

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python - <<'PY'
import numpy
import numpy.lib.stride_tricks as st
print(numpy.__version__)
print(hasattr(st, "broadcast_to"))
PY
```

如果 Isaac 环境被污染，可按实际环境重新固定 NumPy / SciPy 等 Isaac 依赖。

常用环境变量：

| 环境变量 | 默认 | 作用 |
| --- | --- | --- |
| `GO2_X5_CHECKPOINT` | `checkpoints/go2_x5/flat/model_8500.pt` | 覆盖 locomotion checkpoint |
| `GO2_X5_CUROBO_PYTHON` | `/data/conda_envs/isaacsim51_3dgs_grasp/bin/python` | 覆盖 cuRobo planner 使用的 Python |
| `FULL_PHYSICS_DEFER_LEROBOT_EXPORT` | 未设置 | 设为 `1` 时，单 episode 只保留 raw episode，由 batch 统一 materialize |
| `FULL_PHYSICS_FLAT_EPISODE_OUTPUT` | batch 内部使用 | batch 子进程扁平输出，普通用户不需要手动设置 |

### LeRobot / Rerun 环境

LeRobot/Rerun 检查使用普通 Python 环境，不依赖 Isaac、Omni、PXR。

推荐单独环境：

```text
/data/conda_envs/lerobot_rerun
```

需要包：

```text
rerun-sdk
lerobot
numpy
pandas
pyarrow
pillow
opencv-python
tqdm
imageio
imageio-ffmpeg
torch
torchvision
pyyaml
```

检查：

```bash
/data/conda_envs/lerobot_rerun/bin/python - <<'PY'
import rerun as rr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
import numpy as np
import pandas as pd
import pyarrow
from PIL import Image
print("lerobot/rerun env ok")
PY
```

## 文件结构

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
├── recording/                          # raw episode、LeRobot v2/v2.1 写出与校验
└── tasks/                              # task loader、随机化、episode spec

tasks/
└── nav_pick_place_apple_contact.json   # 当前主任务

checkpoints/
└── go2_x5/flat/model_8500.pt           # 本地 locomotion checkpoint，通常不提交 git
```

## Pipeline 流程

默认执行模式是 full-physics，不需要显式传 `--full-physics`。

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

核心约束：

| 约束 | 当前实现 |
| --- | --- |
| 单进程 / 单 World | 单 episode 内由 `FullPhysicsPipeline` 持有唯一 step loop |
| nav | A* + DWA + Isaac Lab Go2 locomotion policy，yaw 到达判定默认不强制 |
| pick/place 规划 | 当前仿真状态导出为 cuRobo state/target JSON，在线规划 |
| 执行 | 机械臂通过逐 step position target / gripper target 物理执行 |
| carry | pick 后 return home，carry 阶段保持 arm home 和 gripper close |
| 稳定模式 | 机械臂阶段默认锁 base root 和 support joints，因此 `pure_physics_success=false`，`stable_physics_success=true` |
| 禁止事项 | full-physics 不接受离线 `--pick-plan-json/--place-plan-json`，不通过 object teleport/TCP clamp 伪造成功 |

### Batch 流程

`run_full_physics_batch.py` 当前按 episode 启动子进程。优点是每个 episode 隔离，失败不会污染后续 episode；缺点是每个 episode 都要重复启动 Isaac Sim、加载 USD、创建 env 和相机管线，墙钟时间较长。

batch 结束会打印表格：

| 列 | 内容 |
| --- | --- |
| `Episode` | episode 编号和 seed |
| `随机化 Pick / Place XY` | 本次随机化后的 pick/place 目标 XY |
| `随机化 BaseGoal / 相对目标` | pick/place base_goal 和相对目标的 XY 偏移 |
| `Pipeline 成功` | 成功 / 失败 |
| `失败 State` | 失败时状态机 state |
| `LeRobot 数据路径` | 成功 episode 的 LeRobot manifest 或数据路径 |
| `Episode 耗时` | 单 episode 墙钟时间 |

失败 episode 会保留诊断原始文件，但不会导出 LeRobot 训练入口：

```text
frames.jsonl / events.jsonl / data.csv / samples.jsonl / images    保留，用于排查
lerobot_manifest.json / lerobot_dataset/                           失败时删除或不生成
```

batch 合并统一数据集时只合并成功 episode。

## LeRobot 数据导出

full-physics 成功后会导出 raw episode 和 LeRobot v2/v2.1 数据。

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
outputs/full_physics_batch/
├── episode_000000/
├── episode_000001/
├── ...
├── batch_summary.jsonl
├── lerobot_export_manifest.json
└── lerobot_dataset/
```

关键 feature：

| Feature | 内容 |
| --- | --- |
| `observation.state` | base pose、TCP、arm joints、gripper、object 等 17 维状态 |
| `observation.base_velocity` | body-frame base velocity |
| `observation.object_state` | object pose / velocity |
| `observation.tcp_pose` | TCP pose |
| `observation.images.front` | front camera MP4 |
| `observation.images.wrist` | wrist camera MP4 |
| `pipeline_state` | 当前状态机阶段 |
| `action` | base3 + arm6 + gripper2，共 11 维 |
| `next.done` | episode 末帧标记 |

校验单 episode：

```bash
cd /home/light/workspace/arm_vla_full_physics

/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/validate_lerobot_episode.py \
  --episode-dir outputs/full_physics_pipeline/episode_000000
```

校验 batch 统一数据集：

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root outputs/full_physics_batch/lerobot_dataset
```

## Rerun 检查

Rerun 转换脚本必须在 `lerobot_rerun` 环境中运行：

```bash
cd /home/light/workspace/arm_vla_full_physics

/data/conda_envs/lerobot_rerun/bin/python \
  tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root outputs/full_physics_batch/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out outputs/full_physics_batch/episode_000000.rrd
```

打开：

```bash
/data/conda_envs/lerobot_rerun/bin/python -m rerun \
  outputs/full_physics_batch/episode_000000.rrd
```

或转换时直接打开 Viewer：

```bash
/data/conda_envs/lerobot_rerun/bin/python \
  tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root outputs/full_physics_batch/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out outputs/full_physics_batch/episode_000000.rrd \
  --spawn
```

Rerun 路径：

| 路径 | 内容 |
| --- | --- |
| `cameras/front/image` | front camera |
| `cameras/wrist/image` | wrist camera |
| `observation/state/*` | observation.state 逐维 scalar |
| `observation/base_velocity/*` | base velocity |
| `observation/object_state/*` | object state |
| `observation/tcp_pose/*` | TCP pose |
| `action/*` | action 逐维 scalar |
| `meta/*` | episode/frame/dataset index、pipeline state |
| `robot/ee` | 可选末端位姿轨迹 |

## 常见运行命令

### GUI 单次 full-physics

```bash
cd /home/light/workspace/arm_vla_full_physics

PYTHONDONTWRITEBYTECODE=1 /data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_gui \
  --seed 0 \
  --no-headless \
  --keep-window-open
```

### Headless 单次 full-physics

```bash
PYTHONDONTWRITEBYTECODE=1 /data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_headless \
  --seed 0 \
  --headless
```

### Headless batch 数据采集

```bash
PYTHONDONTWRITEBYTECODE=1 /data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_batch.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_batch \
  --num-episodes 20 \
  --seed 0
```

### 显示随机化区域

```bash
PYTHONDONTWRITEBYTECODE=1 /data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/randomization_debug_gui \
  --seed 0 \
  --show-randomization-debug \
  --no-headless \
  --keep-window-open
```

### 只跑 dry-run 状态机

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir /tmp/full_physics_dry_run \
  --dry-run
```

### LeRobot 校验与 Rerun

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root outputs/full_physics_batch/lerobot_dataset

/data/conda_envs/lerobot_rerun/bin/python \
  tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root outputs/full_physics_batch/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out outputs/full_physics_batch/episode_000000.rrd
```

## CLI 参数表

### `scripts/pipeline/run_full_physics_pipeline.py`

默认模式是 full-physics。只在需要 smoke/debug 时传模式参数。

| 参数 | 类型 / 默认 | 说明 |
| --- | --- | --- |
| `--task-json` | 必填 | 任务 JSON 路径 |
| `--output-dir` | `outputs/full_physics_pipeline` | 输出目录 |
| `--num-episodes` | `1` | episode 数量；真实 Isaac 模式当前只支持 1 |
| `--seed` | `0` | 首个 episode seed |
| `--randomize-task` / `--no-randomize-task` | 默认开启 | 是否随机化 pick/place 目标 XY |
| `--show-randomization-debug` | 默认关闭 | 显示随机区域和采样点 USD guide |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 默认开启 | 是否随机化 pick/place 导航交接 base_goal |
| `--keep-window-open` / `--no-keep-window-open` | 默认关闭 | 结束后保留 GUI；必须配合 `--no-headless` |
| `--headless` / `--no-headless` | 默认 `--no-headless` | 是否无界面运行 |
| `--pick-plan-json` | 可选 | 仅 manipulation apply smoke 使用；full-physics 禁止 |
| `--place-plan-json` | 可选 | 仅 manipulation apply smoke 使用；full-physics 禁止 |
| `--dry-run` | mode | 无 Isaac 内存后端状态机验证 |
| `--simulation-smoke` | mode | 只验证真实 Isaac stage/reset |
| `--navigation-smoke` | mode | 只验证 nav 到 pick |
| `--navigation-carry-smoke` | mode | 验证 carry 姿态下 nav 到 place |
| `--manipulation-smoke` | mode | 使用假后端验证 manipulation action 合同 |
| `--manipulation-apply-smoke` | mode | 真实 Isaac 中验证 arm/gripper action 下发 |

### `scripts/pipeline/run_full_physics_batch.py`

默认模式是 full-physics，默认 headless，默认继续执行失败后的 episode。

| 参数 | 类型 / 默认 | 说明 |
| --- | --- | --- |
| `--task-json` | 必填 | 任务 JSON 路径 |
| `--output-dir` | 必填 | batch 输出目录 |
| `--num-episodes` | `1` | episode 数量 |
| `--seed` | `0` | 首个 seed，后续使用 `seed + episode_index` |
| `--randomize-task` / `--no-randomize-task` | 默认开启 | 是否随机化 pick/place 目标 XY |
| `--show-randomization-debug` | 默认关闭 | 显示随机区域；通常只用于 GUI 单 episode |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 默认开启 | 是否随机化导航交接 base_goal |
| `--headless` / `--no-headless` | 默认 headless | batch 是否无界面运行 |
| `--continue-on-failure` / `--no-continue-on-failure` | 默认继续 | 单 episode 失败后是否继续 |
| `--pick-plan-json` | 可选 | 非 full-physics smoke 可转发离线 pick plan |
| `--place-plan-json` | 可选 | 非 full-physics smoke 可转发离线 place plan |
| `--progress-interval-s` | `5.0` | heartbeat 进度打印间隔 |
| `--color` / `--no-color` | 默认开启 | 是否使用 ANSI 彩色输出 |
| `--dry-run` | mode | 子进程 dry-run |
| `--simulation-smoke` | mode | 子进程 simulation smoke |
| `--navigation-smoke` | mode | 子进程 navigation smoke |
| `--navigation-carry-smoke` | mode | 子进程 navigation carry smoke |
| `--manipulation-apply-smoke` | mode | 子进程 manipulation apply smoke |

### `tools/lerobot_to_rerun.py`

该脚本必须在 `lerobot_rerun` 环境中运行。

| 参数 | 类型 / 默认 | 说明 |
| --- | --- | --- |
| `--repo-id` | 必填 | LeRobot repo_id 或本地数据集名称 |
| `--root` | 可选 | 本地 LeRobot dataset root |
| `--episode-index` | `0` | 要转换的 episode 编号 |
| `--max-frames` | `-1` | 最大转换帧数，`-1` 表示完整 episode |
| `--out` | `episode.rrd` | 输出 Rerun `.rrd` 路径 |
| `--spawn` | 默认关闭 | 转换时直接打开 Rerun Viewer |

### `scripts/pipeline/validate_lerobot_episode.py`

| 参数 | 类型 / 默认 | 说明 |
| --- | --- | --- |
| `--episode-dir` | 二选一 | 校验单个 full-physics episode 目录中的 `lerobot_dataset` |
| `--dataset-root` | 二选一 | 校验合并后的 LeRobot dataset root |

## 常见问题

### 为什么有些 episode 看起来卡在 build_stage 很久？

`frames.jsonl` 中真实状态机 `build_stage` 通常只占 1 个 tick。batch 进度是读取最后一条 frame 状态；如果 Isaac 子进程启动、env reset、渲染、相机读回或 IO 阻塞，没有及时写出下一帧，progress 会继续显示上一次状态，例如 `build_stage`。这不是 seed 让 build_stage 逻辑执行了几分钟。

当前 batch 每个 episode 都新开 Isaac 子进程，墙钟时间会受 Kit/IsaacLab 初始化、USD/payload 加载、相机/RTX、视频编码和磁盘 IO 波动影响。真正要提速需要做单进程多 episode 复用 Isaac App/World，而不是继续调 DWA 或 cuRobo 参数。

### 失败 episode 会进入训练集吗？

不会。失败 episode 会保留诊断文件，但 `JsonlEpisodeRecorder.close()` 会删除 `lerobot_manifest.json` 和 `lerobot_dataset/`，summary 中写：

```json
{
  "lerobot_training_eligible": false,
  "lerobot_export_skipped": true
}
```

batch 合并统一数据集时只使用成功 episode。

### full-physics 为什么 `pure_physics_success=false`？

默认稳定模式会在机械臂执行期间锁定 base root 和四足支撑关节，避免当前 locomotion policy 因机械臂重心变化侧翻。因此它是稳定物理执行 `stable_physics_success=true`，但不是 strict pure physics。后续若重新训练 locomotion，可再关闭锁定做 strict 验证。

### 为什么不在 full-physics 使用离线 pick/place plan JSON？

full-physics 必须按当前仿真状态在线规划。离线 JSON 只用于 manipulation apply smoke。full-physics 传 `--pick-plan-json` 或 `--place-plan-json` 会直接报错。

### 相机 / wrist image 没有写出怎么办？

检查：

1. `summary.json` 中 `simulation_report.front_camera_report` / `wrist_camera_report`
2. `lerobot_manifest.json` 中 `camera_keys` 和 `missing_camera_keys`
3. `samples.jsonl` 中 `camera_frames`
4. `lerobot_dataset/videos/chunk-000/observation.images.<camera>/episode_000000.mp4`

### Rerun 中没有图像怎么办？

确认数据集 root 是 LeRobot v2/v2.1 格式，并且 `meta/info.json` 中存在 `observation.images.front` 或 `observation.images.wrist`。转换脚本会打印 `Detected image keys`，没有检测到时先检查视频文件和 feature key。
