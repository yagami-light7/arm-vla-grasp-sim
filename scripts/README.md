# Go2-X5 Script Layout

This repository now keeps only the full-physics nav-pick-place pipeline and its
supporting tools.  The old Script Editor/video-baseline handoff scripts have
been removed from this branch.

Current maintained entrypoints:

1. `pipeline/run_full_physics_pipeline.py` for one episode.
2. `pipeline/run_full_physics_batch.py` for automation.
3. `pipeline/validate_lerobot_episode.py` for exported data checks.
4. `curobo/03_plan_grasp_trajectory.py` for cuRobo one-shot planning fallback.
5. `curobo/grasp_planner_server.py` for persistent online planning.

日常部署、双场景选择、batch 数据导出和完整 CLI 表以仓库根目录
[`README.md`](../README.md) 为准。统一入口不再要求切换 worktree：

```bash
export ISAAC_PYTHON=/data/conda_envs/isaacsim51_3dgs_grasp/bin/python

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --list-scene-profiles

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu --check-scene-assets

$ISAAC_PYTHON -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile multi_floor --check-scene-assets
```

开始真实 full-physics 运行前，先使用只读 preflight：

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/preflight_full_physics.py \
  --task-json tasks/nav_pick_place_cola_liangzhu_pct.json \
  --global-planner pct \
  --policy-profile pct_multifloor \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-tomogram-path source/scene/liangzhu/pct/liangzhu_single_floor.pickle \
  --pct-walkable-path source/scene/liangzhu/pct/liangzhu_single_floor_walkable.npy \
  --pct-collision-ply-path source/scene/liangzhu/ply/liangzhu_collision.ply \
  --required-file "$LIANGZHU_COLLISION_USD" \
  --required-file source/robot/go2_x5/urdf/go2_x5/go2_x5.usd \
  --required-file source/scene/objects/carpet.usd \
  --required-file source/scene/objects/cola/MesaTask-10K/MesaTask_Assets/can/0364ab96f338493c972248102b462aa4/usd/0364ab96f338493c972248102b462aa4.usd \
  --required-file source/scene/objects/cola/MesaTask-10K/MesaTask_Assets/can/0364ab96f338493c972248102b462aa4/usd/textures/0364ab96f338493c972248102b462aa4_texture0.png \
  --required-prim-path /World/cola \
  --collision-prim-path /World/PhysicsScene/CollisionScene/LiangzhuCollision \
  --output-json /tmp/full_physics_preflight.json
```

该检查只读取任务、场景 marker、PCT 资产、locomotion checkpoint、任务级 CuRobo
碰撞代理、CUDA、磁盘和后台 Isaac/CuRobo/PCT runtime。默认
`--require-idle-runtime`：其他 Isaac/PCT 或能力不兼容的 CuRobo runtime 会令
`runtime_launch_safe=false` 并返回 `BLOCKED`。单个能正常响应 ping 且声明全部必需
capabilities 的常驻 CuRobo server 是可复用共享服务，不占用 Isaac 独占门禁；pipeline
不拥有且不会关闭它。preflight 始终不会启动或终止任何进程。
`--no-require-idle-runtime` 只用于资产盘点，不代表允许并发启动；同样，
`--no-require-cuda` 只适合静态资产检查，不能作为真实 Isaac 运行验收。

良渚 task 的 `scene_runtime` 是运行时 prim 的唯一配置来源：collision 使用
`/World/PhysicsScene/CollisionScene/LiangzhuCollision`，可选 Gaussian visual prim 为
`/World/VisualScene/GaussianScene`，并显式关闭旧 Yinluyuan F2 地面代理。pipeline
profile 默认 `--navigation-visual-mode full` 并加载 GaussianScene；显式传入
`--navigation-visual-mode collision` 才会关闭 Gaussian 视觉。当前 8GB RTX 4060
Laptop + Isaac Sim 5.1 在 full/NuRec 与 RGB render product 同时启用时会复现 CUDA
illegal address 700，因此本机量产数据使用 collision 兼容模式；两种视觉来源不能
无标记混合。
preflight 中的 `--collision-prim-path` 只作为一致性断言；与 task 不一致会直接失败。
collision terrain wrapper 不写死根 prim 类型，必须让引用源保持实际的 `Mesh` 或
`Xform`；IsaacLab 在创建 `TerrainImporterCfg` 前会重新打开 wrapper，并要求其中至少
存在一个 mesh。良渚当前离线组合结果应为 `composed_root_type=Mesh`、
`mesh_count=1`，真实 smoke 还需核对 metadata 中的
`collision_terrain_wrapper_report`。

当前任务目标是把现有可乐放到 `/World/carpet` 垫子上，人工标注的固定
基准点记录在 `tasks/liangzhu_placement_target.json`。垫子实际碰撞 Mesh 为
`/World/carpet/material`，世界范围约 `0.17635m × 0.13775m`，顶面
`z=-0.15077093m`；任务在 PhysX 初始化前将地垫根位姿落到 collision PLY 地面。
真实运行时不会直接把该静态点送给 CuRobo：目标 XY 来自当前
stage 中安全 placement region 的中心，目标 Z 来自当前支撑 Mesh 顶面加抓取前可乐
实时 bbox 的 center-to-min-z；人工点只作为漂移审计基准。目标使用碰撞包围盒中心，
而不是位于边角的 prim translate。
安全区域按可乐 `0.03m` 足迹半径和 `0.01m` 边缘余量向内收缩，成功判定使用
`0.025m` XY 与 `0.02m` Z 容差并强制检查该区域。固定 baseline 中地垫底面比最高
地面采样点高 `0.0002m`。pick 代理来自良渚 PLY，place 代理来自
实际垫子碰撞世界包围盒；真实运行必须检查
`world_collision_export.task_collision_ids` 同时包含 pick floor 与 mat support，并验证
释放后可乐稳定留在垫子安全区域。

机器人根平移固定为
`(-1.4849319648011197, 5.126136502764003, 0.29281728532721385)`。默认 Phase 1
随机化会在 `[-180°, 180°]` 内采样机器人 yaw，再分别在机器人前向 `±35°` 扇区内
独立采样可乐和地垫；可乐半径为 `[0.70m, 1.15m]`，地垫半径为
`[0.85m, 1.30m]`。两者位置和 yaw 独立随机，并通过旋转地垫足迹距离门禁保证可乐
初始不在地垫上。抓取/放置 TCP 目标
直接使用 Mesh/PhysX 真值，不以相机目标定位为前提；相机仍记录 VLA RGB 数据。
不启动 Isaac 时，可使用 SAGE OpenUSD 对任务支撑体做同构检查：

```bash
/data/conda_envs/sage/bin/python -B \
  scripts/scene/validate_task_receptacle_support.py \
  --task-json tasks/nav_pick_place_cola_liangzhu_pct.json \
  --output-json /tmp/liangzhu_mat_receptacle_support.json
```

该检查要求 `/World/carpet/material` 在组合场景中同时具备 Mesh、启用的
CollisionAPI、静态属性、匹配的世界包围盒和安全 placement region；还会核对任务给
CuRobo 的 mat-support proxy 未相对实际 USD 几何漂移。

良渚 PCT 地图和任务可以在不启动 Isaac Sim 的情况下单独验证：

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/navigation/probe_pct_plan.py \
  --task-json tasks/nav_pick_place_cola_liangzhu_pct.json \
  --pct-server-script scripts/navigation/pct_grid_server.py \
  --pct-tomogram-path source/scene/liangzhu/pct/liangzhu_single_floor.pickle \
  --pct-walkable-path source/scene/liangzhu/pct/liangzhu_single_floor_walkable.npy \
  --pct-collision-ply-path source/scene/liangzhu/ply/liangzhu_collision.ply \
  --pct-coord-mode identity \
  --pct-cross-floor-gateway off \
  --pct-cross-floor-stair-exit off \
  --pct-cross-floor-stair-midpoint off \
  --output-json /tmp/liangzhu_pct_plan_probe.json
```

有其他 Isaac/PCT runtime 或不兼容 CuRobo runtime 时只运行上述 preflight、PCT probe
和 `--dry-run`。单个能力兼容的常驻 CuRobo server 可以跨 worktree 复用；兼容性由
ping 与必需 capabilities 检查，不以源码目录名判断。共享 server 会被保留且永远不会
收到 shutdown。独占 runtime 空闲后，Phase 0 先运行不启动 CuRobo/PCT 的 scene/reset
smoke：

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_cola_liangzhu_pct.json \
  --output-dir /tmp/liangzhu_scene_reset_smoke \
  --simulation-smoke \
  --headless \
  --policy-profile pct_multifloor \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --require-locomotion-checkpoint \
  --no-randomize-task \
  --no-randomize-base-goal
```

该 smoke 只验收 stage/collision/object/articulation/reset，不是抓放成功，也不会进入训练集。
报告中必须看到任务级 collision prim、`units_resolve_preserved=true` 和
`collision_floor_proxy_profile=null`，并且
`task_receptacle_support_runtime_stage_report.geometry_verified=true`。该 runtime
支撑报告也是 LeRobot 训练质量门禁；缺失或不一致的垫子碰撞不会进入主训练集。
完整 episode 还必须同时生成
`last_current_state_curobo_pick_export.mesh_truth_pick_target_report` 和
`last_current_state_curobo_place_export.mesh_truth_place_target_report`：抓取 bbox 中心与
放置时物体中心必须来自 live PhysX pose，放置支撑必须来自当前 stage Mesh，且实际
CuRobo final object center 必须与 Mesh 推导结果一致。缺少任一证据时会以
`mesh_truth_manipulation_targets_not_verified` 拒绝训练导出。
通过后可运行一个随机化完整 episode：

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir outputs/liangzhu_cola_to_mat_randomized \
  --seed 1000 \
  --navigation-visual-mode collision \
  --no-record-video \
  --no-headless
```

`liangzhu` profile 使用良渚可乐任务、PCT 单层地图、identity 坐标、禁止 fallback、
无跨楼层约束、`pct_multifloor` policy/checkpoint 和仓库内
`source/scene/liangzhu/ply/liangzhu_collision.ply`；视觉默认是 full，上述命令只为当前
8GB 主机显式覆盖为 collision。

默认 image/video/GUI overview 均优先使用 USDA 中已有的 `/World/overview`。该相机
位于 GaussianScene 子树之外，因此禁用 Gaussian 后仍可采集。默认
`--dataset-camera-keys front wrist overview`；质量门要求配置中选中的每一路都有同步帧，
且至少包含 front。`--dataset-camera-keys` 主要用于渲染诊断，不会解决当前 NuRec
CUDA 700；`auto` 仅作为旧自适应行为保留。

`pick.object_pose_world` 定义当前 episode 初始生成位姿；CuRobo 侧向抓取 TCP 在 handoff
时由可乐当前 Mesh 尺寸和 live PhysX 中心生成。放置 TCP 同样由当前垫子支撑 Mesh、
抓取前可乐 bbox 和当前 TCP-to-object offset 生成。任务为 pick floor 和 place mat 各
配置了一个强制局部 CuRobo 支撑代理；配置与坐标变换已经离线测试，但实际 planner
world 更新、垫子 PhysX collision 和轨迹避碰仍需真实 Isaac/CuRobo handoff 报告确认。
`perception_mode=sim_ground_truth` 明确表示控制不使用 RGB-D 定位；RGB 只用于数据记录，
当前结果不能描述为视觉定位成功。

随机化单次通过后，批采集入口复用同一组默认 PCT 与 policy 参数：

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir outputs/liangzhu_cola_to_mat_batch \
  --num-episodes 10 \
  --seed 1000 \
  --navigation-visual-mode collision \
  --no-record-video
```

batch 只会把 `training_quality_gate_passed=true` 的物理 episode 合并到统一
LeRobot 数据集；普通 `success=true` 但执行来源或数据门禁未通过的轨迹会被拒绝。
每个随机 episode 默认复用仓库内 collision PLY 探测可乐与地垫支撑面；需要使用其他
点云时可显式传 `--pct-collision-ply-path`。随后同步
更新机器人 yaw、可乐位姿、地垫 stage 位姿、placement region、pick/place base goal
和两阶段 CuRobo 支撑代理。任一同步或地面几何门禁失败都会在 Isaac 启动前拒绝该
episode。扩大随机几何范围或加入新物体后，应先用单 episode 验收，再恢复批量采集。

当前已对 seed 4013..4032 运行 20 条 headless full-physics 回归，20/20
pipeline 成功且 20/20 通过训练质量门。携物 nav-to-place 使用不含“可乐初始
位置 keepout”的阶段地图；静态 collision PLY 障碍仍保留，避免抓取后已移动
的可乐在原地留下过期障碍。

统一数据集中每个 episode 恰好有 6 个子任务目录：
`1-1` 到 `1-6` 分别对应 `nav_straight`、`nav_turn`、
`nav_stop`、`arm_approach`、`arm_contact`、`arm_retreat`。每个目录均保留
`data.csv` 以及 `images/front` / `images/wrist`；同类多次出现会按原始帧序合并，
不再按状态切换重复建目录。

Setup, inspection, FK checks, and single-purpose diagnostics live under
`dev_tools/` so they do not look like required demo steps.

## Full-physics refactor

The experimental single-process pipeline starts at:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_dry_run \
  --num-episodes 1 \
  --seed 0 \
  --dry-run
```

The dependency-free control-flow dry run always sets
`pure_physics_success=false`; legacy video-baseline paths are not maintained in this branch.

Episode-level task-profile randomization is enabled by default. Generic tasks
sample pick/place XY; the Liangzhu task jointly samples robot yaw, cola, mat,
navigation goals and support proxies. Use `--no-randomize-task` when
reproducing a fixed baseline:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_randomized_dry_run \
  --num-episodes 3 \
  --seed 100 \
  --randomize-task \
  --show-randomization-debug \
  --dry-run
```

`--show-randomization-debug` defaults to off. In a real GUI run it creates
green pick and blue place guides plus sampled-position markers; the Liangzhu
profile additionally draws the two robot-forward sector bands. These display
prims use the viewport-visible default USD purpose and have no collision or
rigid-body API. Seeds advance as `seed + episode_index`.

Real Isaac modes are intentionally one episode per process in
`run_full_physics_pipeline.py`. Use the batch launcher below for automation; it
starts one child process per episode so each run still owns exactly one Isaac
World lifecycle:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_batch.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir /tmp/full_physics_random_batch \
  --num-episodes 3 \
  --seed 100
```

The batch launcher writes per-run summaries under
`episode_000000/episode_000000/summary.json` and a top-level
`batch_summary.jsonl`.

`--simulation-smoke` intentionally exits immediately after stage build and
episode reset. Add `--keep-window-open --no-headless` when the purpose is to
inspect the generated stage or randomization guides. The runtime pauses before
the GUI hold loop, so this option does not introduce a second physics loop.

The second-stage real scene/reset smoke check is:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_simulation_smoke \
  --simulation-smoke
```

`--simulation-smoke` launches Isaac Sim, opens one Stage, creates one World,
checks the robot, collision scene, task object, and camera, then performs one
episode reset. It does not invoke navigation, cuRobo, or arm control, and it
never reports `pure_physics_success=true`.

The third-stage physical navigation smoke check is:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_smoke_example.json \
  --output-dir outputs/full_physics_navigation_smoke \
  --seed 31 \
  --navigation-smoke
```

`--navigation-smoke` launches Isaac Lab, loads the locomotion policy, plans an
A* route, and lets the pipeline drive DWA velocity commands one tick at a time.
It exits after `nav_to_pick_success`, so it validates physical navigation
without claiming full pick/place or `pure_physics_success`. Navigation handoff
matches the latest video baseline: success requires base XY to enter the
position tolerance; yaw and base velocity are recorded for diagnostics but do
not reject the episode because the arm planner can absorb remaining yaw error.

The full online nav-pick-place physics check is:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir /tmp/full_physics_online_place_gui \
  --seed 100 \
  --show-randomization-debug \
  --no-headless \
  --keep-window-open
```

Full physics is the default mode. It uses one IsaacLab stage/runtime for nav-to-pick, online
current-state cuRobo pick planning, physical pick execution, closed-gripper
carry navigation, online current-state cuRobo place planning, physical place
execution, and LeRobot export. It does not accept `--pick-plan-json` or
`--place-plan-json`; those offline plan files are only for
`--manipulation-apply-smoke`.

Mechanical-arm execution locks the floating base root pose and support joints
by default because the current locomotion policy was not trained for large
arm-induced center of mass changes. These stable defaults are fixed in
`FullPhysicsConfig` rather than exposed as production CLI switches. The lock
applies only in manipulation and terminal hold phases; navigation remains physically driven.
Any lock use is recorded as `used_manipulation_base_lock=true` and
`used_manipulation_support_joint_lock=true`, so successful default runs report
`stable_physics_success=true` and `pure_physics_success=false`.

Full-physics data recording follows `/home/light/workspace/DWA`:

- physics `dt=0.0025` (400 Hz in the current DWA source), control 50 Hz;
- `RecordingSettings.dataset_fps` defaults to 5 Hz and samples on a fixed
  dataset-time grid; it can be changed to 10/15 Hz without changing physics;
- `480x640` RGB JPEG at quality 90, named `camera0_00000.jpg`;
- raw files: `data.csv`, `samples.jsonl`, and optionally `images/<camera>/`;
- LeRobot v2.1 files: `data/chunk-*/*.parquet`,
  `videos/chunk-*/observation.images.<camera>/*.mp4`, and `meta/*`.

原始 `data.csv` 保留 DWA 的 17 维机器人状态和实测底盘速度列。
`samples.jsonl` 额外保存世界系底盘完整位姿、世界系 TCP 四元数、物体状态、
实测夹爪开度、pipeline 阶段和原始 11 维控制目标。对于显式声明
`base_xyyaw_tcp_base_rpy_gripper_v1` 的任务，LeRobot `action` 使用下一同步
采样帧的 10 维实际执行位姿：世界系底盘 `[x,y,yaw]`、base 系 TCP
`[x,y,z,roll,pitch,yaw]` 和归一化夹爪标量。原始 11 维底盘速度、机械臂关节
和双指目标继续保存为 `control.action`，不会被静默改名为 VLA action。末帧
显式使用 `hold_current_pose`。Parquet 同时保存 `observation.base_pose`、body 系
`observation.base_velocity=[vx,vy,wz]` 和 `pipeline_state`；图像 feature 仍由视频
承载，并在 `meta/info.json` 中声明。

只有物理执行完成、执行来源验证通过、相机帧同步、LeRobot 转换与校验成功，
并且任务要求时已生成有效 10 维 VLA action，才会写入
`training_eligible=true`。dry-run 和 smoke 输出只作为诊断工件，绝不会标记为
训练数据。

Batch episode files are written directly under
`<output-dir>/episode_000000/`, `<output-dir>/episode_000001/`, and so on.
There is no second nested `episode_000000` directory. Batch runs merge all
successful episodes into `<output-dir>/lerobot_dataset`.
The merger only uses successful episodes from the current batch invocation, so
an existing output directory cannot silently inject older episodes.
Existing raw episodes can be converted again without rerunning simulation:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -m source.data.lerobot_converter \
  --episodes-root outputs/full_physics_batch
```

该转换入口默认只发现通过最终物理来源门禁的 episode。只有诊断旧数据时才可显式
传 `--allow-unverified-success`；这种输出不能据此标记为当前 VLA 主训练数据。

Validate either a single episode or the unified dataset:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root outputs/full_physics_batch/lerobot_dataset
```
