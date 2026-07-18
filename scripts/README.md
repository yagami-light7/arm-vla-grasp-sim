# PCT Scene 脚本入口

日常部署、完整 CLI 表、数据 schema 和资产清单以根目录
[`README.md`](../README.md) 为准。当前只维护以下入口：

1. `pipeline/run_full_physics_pipeline.py`：单 episode、GUI 和各类 smoke。
2. `pipeline/run_full_physics_batch.py`：独立 Isaac 子进程的 headless batch 与统一 LeRobot 物化。
3. `pipeline/preflight_full_physics.py`：不启动仿真的资产/runtime 检查。
4. `pipeline/validate_lerobot_episode.py`：单 episode 或统一 LeRobot v2.2 数据集检查。
5. `pipeline/validate_full_pipeline_acceptance.py`：状态机、物理来源、数据与视频联合验收。
6. `navigation/probe_pct_plan.py`：不启动 Isaac 的 PCT 路径检查。
7. `curobo/03_plan_grasp_trajectory.py` 与 `curobo/grasp_planner_server.py`：CuRobo one-shot 和常驻规划服务。

## 场景选择

```bash
export ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --list-scene-profiles

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu --check-scene-assets

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile multi_floor --check-scene-assets
```

- `liangzhu`：从 box1 拿起可乐并放到 box2；PCT identity 坐标，默认联合随机化。
- `multi_floor`：别墅 F1 到 F2 苹果搬运；保留楼梯锚点、阶段相机和原 locomotion 逻辑。

两个 profile 均默认开启 `composite` 展示视频：overview 位于左侧 2/3，front
和 wrist 分别在右上/右下，三路来自同一 simulation step，默认 1280×720、25 fps。

## 良渚随机化

`tasks/liangzhu_placement_target.json` 是 box1 → box2 标注和随机化的单一来源。
`liangzhu_box_pair_xy_v1` 使用 seed 同步采样：

- box1/box2 authored XY 各加 `±0.12m`，其他 transform 不变；
- 机器人在两桌中间生成，yaw 为 `[-180°, 180°]`；
- 可乐在 box1 中央安全区采样 XY/yaw，放置区跟随 box2；
- pick standoff 为 `0.50..0.54m`，place standoff 为 `0.48..0.51m`；
- robot/object/boxes、CuRobo proxy、PCT/DWA keepout、base goal 和 metadata 使用同一份样本。

当前不随机光照、材质或相机。控制使用 live Mesh/PhysX 真值，RGB 为训练数据记录。

## 常用命令

良渚单条 headless：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir /mnt/sage_data/outputs/pct_scene/liangzhu_single_seed0 \
  --seed 0 \
  --headless
```

良渚 20 条 batch：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir /mnt/sage_data/outputs/pct_scene/liangzhu_batch_seed0_n20 \
  --num-episodes 20 \
  --seed 0
```

别墅单条 GUI：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile multi_floor \
  --output-dir /mnt/sage_data/outputs/pct_scene/multi_floor_gui \
  --no-headless \
  --keep-window-open
```

batch 默认 headless，每条使用 `seed + episode_index`，失败条保留诊断；只有最终
验证和训练质量门同时通过的 episode 会进入 `<output-dir>/lerobot_dataset`。

校验统一数据集：

```bash
$ISAAC_PYTHON -B scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root /mnt/sage_data/outputs/pct_scene/liangzhu_batch_seed0_n20/lerobot_dataset
```

## 视觉模式限制

headless 三相机量产当前使用 `collision`。在已验证的 8 GB RTX 4060 Laptop、
Isaac Sim 5.1 环境中，良渚 `full` Gaussian/NUREC 能加载并启动 PhysX，但首帧
headless 三相机渲染会触发 `cudaErrorIllegalAddress (700)`。这是渲染路径问题，
不是 CUDA 设备不可用。GUI 视觉调试可显式使用：

```bash
$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --navigation-visual-mode full \
  --no-record-dataset \
  --no-record-video \
  --no-headless \
  --keep-window-open
```

`--scene-light-mode auto` 是默认值：`full` 会自动启用所选 scene profile 的 USDA
stage lights，`collision` 会自动使用相机补光；通常无需额外传
`--scene-light-mode stage`，但仍可显式指定 `camera` 或 `stage` 覆盖。

不要在无标记的情况下混合 collision 与 full 视觉数据。
