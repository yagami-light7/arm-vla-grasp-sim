# Go2-X5 script entrypoints

本目录只保留当前 full-physics nav-pick-place pipeline 及其检查工具。更完整的数据
schema、历史任务和 CLI 参数说明见仓库根目录 `README.md`。

## 当前良渚默认任务

默认任务为：

```text
Pick up the coke can on box1 and place it on box2.
```

任务入口是 `tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json`。以下内容集中
维护在 `tasks/liangzhu_placement_target.json`：

- box1 抓取支撑、box2 放置支撑和 CuRobo proxy；
- box1/box2 XY-only 随机化；
- 两个 box 之间的机器人位置与 `[-180°, 180°]` yaw；
- box1 中央安全区内的可乐 XY/yaw；
- 六类英文 subtask 与四类分段英文 instruction 模板。

box 随机化只改根 `translate` 的 X/Y。Z、orientation、scale、`unitsResolve` 和 xform
op 顺序均保留资产 authored 值。runtime 在 physics 初始化前为缺少 authored
collision 的 box2 添加静态 mesh collision，不会写回 USD。

seed 5000 已完成真实 headless 全链路验证：最终 `success=true`、状态到达 `done`，
LeRobot 、六类 subtask 与 front/wrist 图像通过 validator。2026-07-18 又完成了
overview/front/wrist composite 实测；具体多 seed 成功率以最新 batch 摘要为准。

## 维护入口

1. `pipeline/run_full_physics_pipeline.py`：单 episode、GUI 或 smoke。
2. `pipeline/run_full_physics_batch.py`：隔离子进程的批量采集与统一 LeRobot 物化。
3. `pipeline/preflight_full_physics.py`：只读资产/runtime 检查。
4. `pipeline/validate_lerobot_episode.py`：LeRobot v2.2 数据检查。
5. `navigation/probe_pct_plan.py`：不启动 Isaac 的 PCT 两段路径检查。
6. `curobo/03_plan_grasp_trajectory.py`：CuRobo one-shot fallback。
7. `curobo/grasp_planner_server.py`：常驻 CuRobo planner server。

## 常用命令

以下命令在 `/home/light/workspace/arm_vla_liangzhu` 下运行。

### 只读 preflight

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/preflight_full_physics.py \
  --task-json tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json \
  --global-planner pct \
  --policy-profile pct_multifloor \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-tomogram-path source/scene/liangzhu/pct/liangzhu_single_floor.pickle \
  --pct-walkable-path source/scene/liangzhu/pct/liangzhu_single_floor_walkable.npy \
  --pct-collision-ply-path source/scene/liangzhu/ply/liangzhu_collision.ply \
  --required-file source/scene/objects/box/box.usd \
  --required-file source/scene/objects/box2/box2.usd \
  --required-file source/robot/go2_x5/urdf/go2_x5/go2_x5.usd \
  --required-prim-path /World/box1 \
  --required-prim-path /World/box2 \
  --required-prim-path /World/cola \
  --collision-prim-path /World/PhysicsScene/CollisionScene/LiangzhuCollision \
  --output-json /tmp/liangzhu_box_pair_preflight.json
```

preflight 不启动或终止进程。默认要求没有冲突的 Isaac/PCT runtime；能正常响应 ping
并声明全部能力的共享 CuRobo server 可以复用，pipeline 不拥有其生命周期。

### PCT 两段路径检查

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/navigation/probe_pct_plan.py \
  --task-json tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-tomogram-path source/scene/liangzhu/pct/liangzhu_single_floor.pickle \
  --pct-walkable-path source/scene/liangzhu/pct/liangzhu_single_floor_walkable.npy \
  --pct-collision-ply-path source/scene/liangzhu/ply/liangzhu_collision.ply \
  --pct-coord-mode identity \
  --pct-cross-floor-gateway off \
  --pct-cross-floor-stair-exit off \
  --pct-cross-floor-stair-midpoint off \
  --output-json /tmp/liangzhu_box_pair_pct_probe.json
```

PCT probe 验证全局可达性；真实控制端不会直接追踪 PCT 吸附后的栅格中心。执行器会用
机器人实时起点和 task 精确目标恢复连续端点，在已膨胀 occupancy map 上先做
supercover line-of-sight 检查；仅在直线被阻塞时运行 local A*，随后做可见性路径精简。
DWA 首帧还会把实时位姿投影到整条路径并跳过身后的 prefix。携物阶段先按支撑物 bbox
和机器人扫掠半径执行安全退出、停稳和实时重锚定；同层末端控制保持最终 yaw，并使用
body-frame `vx/vy` 修正剩余 XY，避免“先朝向数厘米位置误差、再转回最终 yaw”的往返
旋转。运行结果可在每个导航帧的 `action.metadata.local_refinement`、
`action.metadata.carry_departure`、`action.metadata.dwa.path_anchor_*` 和
`action.metadata.carry_goal_yaw_translation_active` 中审计。

真实回归 seed 5002 已完成完整 pipeline，LeRobot 为 196 rows、三路相机各 196 帧、
0 error / 0 warning。seed 5003 的携物导航由 1423 ticks / 407.71° 实际累计 yaw 降至
930 ticks / 233.06°；该条随后在 CuRobo `plan_place` 的机械臂可达边界失败，不能记为
完整 pipeline 成功，也不应归因于 PCT/DWA。

### 单 episode

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_seed5000 \
  --seed 5000 \
  --headless
```

GUI 调试时将 `--headless` 改为 `--no-headless --keep-window-open`。默认已包含当前
task、PCT 单层资产、identity 坐标、仓库 collision PLY、`pct_multifloor`
checkpoint、禁止 A* fallback 和 collision 视觉模式。

headless 三相机量产保持 collision 视觉。`--navigation-visual-mode full` 在当前
Isaac Sim 5.1 + NUREC 路径的首帧渲染会触发 CUDA 700，因此只用于 GUI
视觉调试，不是 batch 稳定默认。

### 三视角拼接与完整验收

完整 single/batch pipeline 已默认启用 `composite`；以下显式参数用于强调验收配置，
需要节省磁盘空间时可传 `--no-record-video`。

```bash
set -euo pipefail
OUT=/mnt/sage_data/outputs/arm_vla_liangzhu/acceptance_composite_seed5002_$(date +%Y%m%d_%H%M%S)

/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir "$OUT" \
  --seed 5002 \
  --headless \
  --record-video \
  --video-mode composite

/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/validate_full_pipeline_acceptance.py \
  --episode-dir "$OUT/episode_000000"
```

输出 `overview_videos/episode_000001_composite.mp4`。布局为 overview 左侧主视图、
front 右上、wrist 右下；三路来自同一 simulation step，默认 1280×720、25fps。

### 批量采集

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_batch.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_batch_seed5000_n20 \
  --num-episodes 20 \
  --seed 5000
```

batch 每个 episode 使用 `seed + episode_index`，并在独立 Isaac 子进程中运行。只有
`training_quality_gate_passed=true` 且最终验证通过的 episode 会被合并到统一
`lerobot_dataset/`。扩大随机范围前，应先运行单 episode 和 PCT seed sweep。

### 固定 nominal 布局

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_nominal \
  --no-randomize-task \
  --no-randomize-base-goal \
  --headless
```

### 校验统一 LeRobot 数据集

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_batch_seed5000_n20/lerobot_dataset
```

## Subtask 与 instruction

目录仍固定为六类：`nav_straight`、`nav_turn`、`nav_stop`、`arm_approach`、
`arm_contact`、`arm_retreat`。每个目录都有 `data.csv` 和
`images/front`、`images/wrist`；同类的多个连续片段聚合到同一目录。

逐帧/连续 segment instruction 字段为：

```text
instruction
instruction_id
instruction_target_id
instruction_direction
instruction_relative_bearing_rad
instruction_pose_source
instruction_annotation_schema
```

寻找 box1/box2 的 `nav_turn` 使用 segment 首帧机器人 pose 计算八方向英文标签；其余
pick-side 和 place-side segment 分别使用固定英文抓取/放置指令。schema 是
`relative_direction_segment_instruction_v1`，语言为 `en`。

## 运行边界

- `perception_mode=sim_ground_truth`：RGB 用于 VLA 记录，不代表 RGB-D 定位成功。
- 默认 `--navigation-visual-mode collision`；显式使用 `full` 才加载 GaussianScene。
- `--scene-light-mode auto` 为默认值；`full` 自动启用 USDA 内编写的 stage lights，
  `collision` 自动使用相机补光，也可显式指定 `camera` 或 `stage` 覆盖。
- `/World/overview` 是 GUI/image/video 的默认 overview 相机。
- full-physics 成败以 `summary.json`、`events.jsonl` 和 LeRobot validator 为准，不以
  GUI 最后一帧判断。
- 旧 `tasks/nav_pick_place_cola_liangzhu_pct.json` 是可乐到鼠标垫兼容任务，不再是
  pipeline/batch 默认入口。
