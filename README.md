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
    ├── run_full_physics_pipeline.py        # 单 episode / smoke 入口
    ├── run_full_physics_batch.py           # 批量自动化入口
    └── validate_lerobot_episode.py         # LeRobot 数据集校验入口

tools/
└── lerobot_to_rerun.py                     # LeRobot episode -> Rerun .rrd

source/
├── interfaces/                             # 跨模块协议层，只定义数据结构和抽象接口
│   ├── navigation.py                       # NavGoal / NavPlan / NavigationPlanner / NavigationExecutor
│   ├── manipulation.py                     # ArmPlan / RobotAction / ManipulationPlanner / ArmExecutor
│   ├── simulation.py                       # SimulationState / SimulationRuntime / action apply 协议
│   ├── recording.py                        # EpisodeRecorder / 数据记录接口
│   ├── task.py                             # EpisodeSpec / TaskProvider 任务协议
│   └── verification.py                     # VerificationResult / 成功判据接口
├── pipeline/                               # full-physics 状态机和主循环
│   ├── config.py                           # FullPhysicsConfig 及 nav/manip/randomization/recording 参数
│   ├── states.py                           # PipelineState 枚举
│   ├── events.py                           # PipelineEvent 事件结构
│   ├── state_machine.py                    # nav -> pick -> carry nav -> place -> export 状态机
│   ├── full_physics_pipeline.py            # 单 World step loop、summary、LeRobot export 调度
│   ├── factory.py                          # 组装 nav planner、manip planner、executor、runtime、recorder
│   ├── dry_run.py                          # 无 Isaac 依赖的控制流 dry-run factory
│   ├── simulation_smoke.py                 # stage/reset smoke factory
│   ├── navigation_smoke.py                 # nav-only / carry-nav smoke factory
│   ├── manipulation_smoke.py               # manipulation contract smoke factory
│   ├── manipulation_apply_smoke.py         # 真实 Isaac action apply smoke factory
│   └── isaac_compat.py                     # Isaac/Kit 兼容辅助
├── simulation/                             # Isaac Sim / Isaac Lab runtime 与 USD patch
│   ├── isaaclab_runtime.py                 # 当前 full-physics 主 runtime，负责场景、传感器、状态导出、step
│   ├── isaac_runtime.py                    # 轻量 Isaac Sim runtime，保留给 apply/smoke 路径
│   ├── action_applier.py                   # arm/gripper/base action 下发与 tracking report
│   ├── collision_patch.py                  # 夹爪/苹果 collision approximation 与 offset patch
│   ├── visibility_patch.py                 # 视觉层隐藏/显示 patch，例如隐藏碰撞苹果 mesh
│   ├── viewport.py                         # GUI viewport camera 选择与同步
│   └── in_memory.py                        # 测试用内存 simulation backend
├── navigation/                             # A* / DWA / Go2 locomotion policy 封装
│   ├── planner_adapter.py                  # 将 navlib A*/DWA 包装为 NavigationPlanner
│   ├── executor.py                         # waypoint / velocity command 执行器
│   ├── dry_run.py                          # 无 Isaac 的导航 dry-run
│   ├── adapters/
│   │   ├── dwa_nav_adapter.py              # DWA 局部规划 adapter
│   │   ├── isaaclab_go2_adapter.py         # IsaacLab Go2-X5 locomotion policy adapter
│   │   ├── frame_utils.py                  # world/map/body 坐标变换
│   │   ├── stall_detector.py               # 导航卡住检测
│   │   ├── terrain_utils.py                # terrain / collision 辅助
│   │   └── yaw_align.py                    # 终端 yaw 对齐控制
│   └── navlib/
│       ├── astar.py                        # 全局 A* 路径规划
│       ├── dwa.py                          # DWA 速度采样与 rollout
│       ├── grid_map.py                     # occupancy grid 数据结构
│       ├── path_tracking.py                # waypoint lookahead / tracking
│       ├── rasterization.py                # USD collision -> occupancy rasterization
│       └── serialization.py                # nav map JSON/PGM 读写
├── manipulation/                           # cuRobo 在线规划、机械臂执行、夹爪控制
│   ├── current_state_curobo.py             # 从当前仿真状态导出 cuRobo pick/place state + target
│   ├── grasp_pipeline.py                   # cuRobo planner-only wrapper，server 或 one-shot fallback
│   ├── planner_server_process.py           # 自动启动/复用 grasp_planner_server.py
│   ├── curobo_adapter.py                   # cuRobo JSON plan -> ArmPlan segment 转换
│   ├── arm_executor.py                     # 分段轨迹 tracking、strict wait、return-home、open/close 时序
│   ├── gripper_controller.py               # 二值夹爪 open/close target 生成
│   ├── smoke.py                            # manipulation smoke planner
│   └── dry_run.py                          # manipulation dry-run planner/executor
├── diagnostics/                            # 成功判据、调试可视化、报告
│   ├── full_physics.py                     # pick/carry/place success verifier
│   ├── navigation.py                       # navigation verifier
│   ├── manipulation_apply.py               # action apply smoke verifier
│   ├── integrated_apply.py                 # 历史诊断兼容；不作为主入口
│   ├── randomization_debug.py              # pick/place/base_goal 随机区域可视化
│   └── dry_run.py                          # dry-run verifier
├── recording/                              # full-physics raw frames 与 LeRobot v2 写出
│   ├── jsonl_recorder.py                   # frames/events/samples/jsonl 与 LeRobot export 调度
│   ├── lerobot_dataset.py                  # LeRobot v2 parquet/video/meta 写出
│   └── lerobot_validator.py                # LeRobot v2 结构、时间戳、feature 校验
├── data/                                   # 旧数据 schema / converter 兼容层
│   ├── task_schema.py                      # task JSON dataclass/schema helper
│   ├── random_task.py                      # 旧随机任务生成 helper
│   ├── episode_recorder.py                 # 旧多 phase CSV recorder
│   ├── vla_episode_recorder.py             # VLA episode recorder 兼容接口
│   └── lerobot_converter.py                # 旧 LeRobot converter 兼容接口
├── tasks/                                  # task loader 与随机化逻辑
│   ├── task_loader.py                      # JSON task -> EpisodeSpec
│   ├── randomizer.py                       # 通用 pick/place XY 和 base_goal 随机化
│   └── forward_sector_randomization.py     # 良渚机器人前向扇区联合随机化
├── scene/                                  # USD 场景、物体资产和导航地图
│   └── liangzhu/                           # 良渚 USDA、PCT 单层地图、collision PLY 和 manifest
├── robot/                                  # Go2-X5 URDF / robot 资产源文件
└── robot_lab/                              # Isaac Lab extension / Go2-X5 task registration

tasks/
└── nav_pick_place_cola_liangzhu_pct.json   # 当前默认良渚可乐到鼠标垫任务

checkpoints/
└── go2_x5/pct_multifloor/model_26000.pt    # 当前默认 locomotion checkpoint
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

良渚量产回归默认使用 headless batch。GUI 只用于单 episode 可视化调试；
如果 viewport 看起来停在 pick，但没有生成 `state_failed` 或最终 `summary.json`，
应先按渲染 timeline 未继续推进处理，不要把画面最后一帧当作 pick 逻辑失败。
同一 seed 应用 headless 入口复验，并以 `events.jsonl` 和 `summary.json` 为准。

失败 episode 会保留诊断原始文件，但不会导出 LeRobot 训练入口：

```text
frames.jsonl / events.jsonl / data.csv / samples.jsonl / images    保留，用于排查
lerobot_manifest.json / lerobot_dataset/                           失败时删除或不生成
```

batch 合并统一数据集时只合并成功并通过最终物理来源质量门的 episode。

### 良渚 Phase 1 联合随机化

良渚可乐到鼠标垫任务使用任务级随机化模式：

```text
robot_forward_sector_v1
```

这不是分别给几个固定坐标增加噪声，而是以一个 episode seed 联合生成机器人朝向、
可乐、鼠标垫、导航交接点、放置区域和 CuRobo 碰撞代理。完整顺序为：

```text
episode seed
-> 采样机器人 yaw
-> 在机器人前向扇区采样可乐和鼠标垫
-> 拒绝重叠或底盘交接点冲突的布局
-> 使用 collision PLY 探测地面支撑
-> 计算可乐 Z、鼠标垫根位姿和放置高度
-> 重建 pick/place base_goal 与 CuRobo 支撑代理
-> 写入统一 EpisodeSpec
-> Isaac/PhysX、PCT、CuRobo 和 recorder 消费同一份结果
```

#### 当前随机变量和固定量

配置来源为 `tasks/nav_pick_place_cola_liangzhu_pct.json` 的
`randomization.forward_sector`：

| 项目 | 当前分布或取值 | 说明 |
| --- | --- | --- |
| 机器人 XYZ | `(-1.4849319648, 5.1261365028, 0.2928172853)` | 当前固定，不随机平移 |
| 机器人 yaw | `Uniform(-180°, 180°)` | 每个 episode 重新采样，覆盖全向朝向 |
| 前向扇区 | 机器人 yaw 左右各 `35°` | 可乐和鼠标垫都相对采样后的 yaw 定义 |
| 可乐半径 | `[0.70m, 1.15m]` | 在前向扇形内按面积均匀采样 |
| 鼠标垫半径 | `[0.85m, 1.30m]` | 与可乐分别采样 |
| 可乐 yaw | `Uniform(-180°, 180°)` | roll/pitch 固定为 0 |
| 鼠标垫 yaw | `Uniform(-180°, 180°)` | roll/pitch 固定为 0 |
| 放置后可乐 yaw | `Uniform(-180°, 180°)` | 与初始可乐 yaw 独立采样 |
| pick base standoff | `[0.35m, 0.39m]` | 底盘最终正面朝向可乐 |
| place base standoff | `[0.35m, 0.39m]` | 底盘最终正面朝向鼠标垫 |
| base approach angle noise | `0°` | 当前不加入侧向接近噪声 |
| placement region | 鼠标垫中心 `0.06m × 0.06m` | 当前不在安全区域内部二次采样 XY |

可乐与鼠标垫的无约束极坐标样本来自同一个确定性随机流。角度均匀采样，半径使用：

```text
radius = sqrt(Uniform(radius_min², radius_max²))
bearing_world = robot_yaw + Uniform(-sector_half_angle, +sector_half_angle)
x = robot_x + radius * cos(bearing_world)
y = robot_y + radius * sin(bearing_world)
```

使用平方半径采样可以使点在扇形面积内近似均匀，避免大量样本聚集在机器人附近。
可乐和鼠标垫先分别采样，随后经过联合约束过滤，因此最终接受分布不是完全独立分布。

#### 布局拒绝条件

鼠标垫被建模为带 yaw 的旋转矩形足迹，可乐被建模为半径 `0.03m` 的圆形足迹。
随机化器把可乐中心转换到鼠标垫局部坐标系，并计算点到旋转矩形的最短距离。
候选布局满足以下任一条件时会被拒绝：

```text
可乐与鼠标垫中心距离 < 0.30m
可乐足迹到鼠标垫足迹的净空 < 0.06m
pick base_goal 落入鼠标垫足迹外扩 0.30m 的区域
place base_goal 距离可乐初始位置 < 0.30m
```

每次几何布局最多尝试 `300` 次。被拒绝的 attempt、原因和测量距离都会写入：

```text
task.randomization.sample.rejected_layout_samples
```

同一个 seed 会复现相同的接受布局和拒绝历史。

#### 正面抓取和放置的底盘目标

pick base goal 从机器人初始位置朝可乐生成：

```text
pick_goal_xy = cola_xy - pick_standoff * unit(robot_xy -> cola_xy)
pick_goal_yaw = bearing(pick_goal_xy -> cola_xy)
```

place base goal 从已经生成的 pick base goal 朝鼠标垫生成：

```text
place_goal_xy = mat_xy - place_standoff * unit(pick_goal_xy -> mat_xy)
place_goal_yaw = bearing(place_goal_xy -> mat_xy)
```

因此携物导航从实际抓取交接位置朝鼠标垫收敛，而不是仍以 episode 初始位置为接近
原点。两段目标都写入：

```text
target_region_in_base = front
final_alignment_mode = face_target
target_bearing_base_rad = 0
```

这对应当前 top-down、正面抓放策略，不再要求机器狗到达目标附近后额外旋转 90°。

#### collision PLY 地面支撑

XY 布局通过后，随机化器使用与 PCT/DWA 同源的 collision PLY 从
`ground_query_ceiling_z=2.0m` 向下寻找最高支撑三角面。路径解析顺序为：

```text
--pct-collision-ply-path
-> task.randomization.forward_sector.collision_ply_path
-> task 中 collision_ply_env 指定的环境变量
```

缺少 PLY 或路径无效属于配置错误，会立即失败，不会通过重复采样掩盖。

可乐只探测中心 XY：

```text
cola_center_z = cola_support_z + 0.0537467943m
```

鼠标垫探测中心和旋转后的四个角，共五个点。五点最高与最低地面高度差必须不超过
`0.006m`；否则整套布局重新采样，最多进行 `20` 次支撑重采样。鼠标垫使用最高支撑
点计算根位姿，使所有角都位于地面之上：

```text
mat_top_z = max(mat_ground_probe_z) + 0.000598875m
place_object_center_z = mat_top_z + live_or_calibrated_object_bbox_half_height
```

鼠标垫 prim 原点不等于碰撞矩形中心。代码会旋转已标定的
`mat_root_to_support_center_xyz`，再反算 `/World/carpet` 根位姿，禁止把采样中心直接当成
prim translate。鼠标垫当前保持 roll/pitch 为 0；高度变化门禁只允许近似水平地面。

#### PhysX、PCT 与 CuRobo 同步

接受布局会一次性更新以下字段：

```text
task.start
task.pick.object_pose_world
task.pick.base_goal
task.place.receptacle_pose_world
task.place.place_pose_world
task.place.placement_region
task.place.base_goal
task.pick.curobo_world_collision
task.place.curobo_world_collision
```

运行时的对应关系为：

- 机器人 reset 使用单点 pose range，精确采用随机化后的固定 XYZ 和 yaw，不会再次随机。
- `/World/cola` 在 PhysX 初始化前写入随机 pose，随后以动态刚体自然 settle；抓取目标使用
  settle 后的 live PhysX/Mesh bbox。
- `/World/carpet` 是静态 CollisionAPI，随机根位姿必须在 PhysX 初始化前写入组合 stage；
  运行时会再次验证实际支撑 bbox、placement region 和任务配置一致。
- PCT 状态机直接使用随机化后的 `EpisodeSpec.pick_goal/place_goal`，并保留 PCT path、
  snap distance 和 planner metadata。
- pick CuRobo proxy 是可乐下方 `0.45m × 0.30m × 0.04m` 的地面 cuboid，顶面与 PLY
  支撑面重合，局部 Z 轴与支撑三角面法向对齐。
- place CuRobo proxy 随鼠标垫 XY/yaw 重建，顶面与当前 episode 鼠标垫碰撞顶面一致。
- place XYZ 在真实运行时由当前 stage 鼠标垫顶面和抓取前可乐 live bbox 推导；配置中的
  place pose 同时作为严格漂移审计基准。

生成阶段会在 `task.randomization.synchronization` 中记录上述同步项。该字段证明任务结构
已统一写回；真实执行是否一致还要以 runtime support report、CuRobo world export、PCT
metadata 和最终 validator 为准。

#### Seed、CLI 和 batch 语义

单 episode 和 batch CLI 的 `--randomize-task`、`--randomize-base-goal` 默认均开启：

| 参数组合 | 良渚前向扇区 profile 的实际行为 |
| --- | --- |
| `--randomize-task --randomize-base-goal` | 完整联合随机化，standoff 也在配置区间内随机 |
| `--randomize-task --no-randomize-base-goal` | 目标和机器人 yaw 仍随机；standoff 固定为区间中点，但 base goal 仍随目标移动 |
| `--no-randomize-task --no-randomize-base-goal` | 完全使用 task JSON 固定 baseline |
| `--no-randomize-task --randomize-base-goal` | 良渚专用 profile 不转入旧通用 sampler，保持固定任务 |

实际运行开关由 CLI 的 `RandomizationSettings` 控制；task JSON 中的
`randomization.mode` 选择具体随机化算法。batch 为每个 episode 启动独立进程，并使用：

```text
episode_seed = batch_seed + episode_index
```

单个 episode 失败后默认继续。batch 只把同时满足以下条件的源 episode 交给统一
LeRobot 物化器：

```text
success = true
training_quality_gate_passed = true
training_quality_success_verified(summary) = true
```

源 episode 的即时 `lerobot_manifest.json` 主要用于调试，可能因为尚未物化训练 action 而显示
`lerobot_training_eligible=false`。这不会替代上述最终物理来源门禁。统一物化完成后，以 batch
根目录 `lerobot_export_manifest.json` 的 `vla_training_action_available`、
`vla_training_eligible` 和 `validation_report` 为最终训练入口判据。

每个源 episode 的 `task.json` 和 `summary.json` 保存完整随机化配置、seed、接受布局、
拒绝历史、地面探测、base goal 和同步状态。合并后的 `task.csv` 至少保留 seed、机器人
起点和 pick/place base goal；因此训练数据可以按 seed 追溯到源布局。

#### 当前随机化边界

`robot_forward_sector_v1` 当前只属于 Phase 1 连续空间随机化。以下项目尚未随机：

- 机器人初始 XYZ、roll、pitch。
- 目标物体种类、尺寸、质量、摩擦、材质和纹理。
- 鼠标垫种类以及 placement region 内的局部放置点。
- 光照、相机内外参、曝光、RGB/depth 噪声和遮挡。
- 场景局部障碍和家具布置。
- 机器人质量、惯量、COM、执行器增益和外力。
- 任务指令、任务组合和多目标实例选择。

full-physics 数据采集会显式关闭 locomotion 训练环境中的观测 corruption、质量/惯量/COM、
执行器增益、随机外力和 push event，避免把未审计的训练期 domain randomization 混入专家
数据。当前 `perception_mode=sim_ground_truth`；RGB 会被记录用于 VLA，但不能把当前结果描述
为 RGB-D 检测或视觉定位成功。

前向扇区 sampler 本身执行几何与支撑门禁，不在采样函数内部预跑 PCT 或 CuRobo。实际
PCT/CuRobo 规划失败会使该 episode 失败并保留诊断；当前范围已经通过 50 个 seed 的离线
PCT sweep，但扩大 yaw、半径或局部场景范围时必须重新运行 sweep 和真实 batch。

#### 导航执行稳定性保护

真实多 seed 运行暴露出一个低层 locomotion policy 与局部速度命令之间的死区：DWA 或末端
P 控制器持续输出很小的线速度/角速度时，策略可能只踏步而没有足够的位姿进展。当前保护为：

- PCT profile 的非零线速度候选不得低于 locomotion gait floor；非零角速度候选同样设置下限。
- 角速度换向必须先经过零命令，避免一步内直接反向导致振荡。
- 携物阶段所有末端 yaw 命令统一受 carry yaw-rate 上限约束，末端 P 控制器不能绕过该限制。
- stall detector 同时检查 XY 平移和 yaw 旋转；只有命令占比高且实测进展低时才判定卡住。
- `navigation_settling` 期间若机器人漂回容差外，会恢复末端控制，而不是永久保持零命令。
- 良渚单层任务使用默认 `5000` 导航 step 上限；`12000` 只保留给真实多楼层长路径。

当前良渚 task 的 pick 底盘末端位置容差为 `0.10m`，place 底盘交接使用独立的 `0.15m`，
place 目标物体 XY 容差为 `0.035m`，Mesh truth 物体竖直 extent 审计容差为 `0.010m`。
这些数值来自本轮真实随机 batch，而不是用于隐藏失败；
导航碰撞、不可达、掉落和放置验证失败仍会明确拒绝该 episode。

#### 2026-07-15 历史 ±30° yaw 随机 batch 基线

使用 `seed=4013..4032` 连续运行 20 个 full-physics episode：

> 这组 20/20 使用的是扩展到全向 yaw 之前的 `[-30°, 30°]` 配置，保留用于
> 格式和窄角度稳定性对照，不能作为当前 `[-180°, 180°]` 配置的成功率结论。

| 指标 | 结果 |
| --- | --- |
| 尝试数 | 20 |
| 质量门通过 / 纳入统一数据集 | 20 |
| 失败并隔离 | 0 |
| 实测成功率 | 100% |
| 统一数据集帧数 | 2476，5 Hz |
| 视觉 | front / overview / wrist，60 个 mp4；子任务 front/wrist 各 2476 JPG |
| parquet | 20 个，每个 accepted episode 一个 |
| subtask | 每 episode 固定 6 目录，共 120 目录；保留 189 个原始连续 segment 编号 |
| action | 10D VLA action + 11D `control.action` |
| validator | 20 episodes，2476 rows，0 error，0 warning |
| 磁盘占用 | 统一 LeRobot 数据集约 176 MB |

统一数据集位于：

```text
/mnt/sage_data/outputs/arm_vla_liangzhu/
validation_seed4013_n20_final_carry_map_v2/lerobot_dataset
```

成功 seed 为 `4013..4032`。修复前的 phase145/146/147 数据仍保留作为历史
对照，但其 front-only 或按连续 segment 建目录的布局不是当前 v3 格式，不应
与本数据集直接混用。

每个 `episodes/2001/<episode_id>/task.csv` 保存 seed、机器人起点、pick/place 目标和
base goal；`lerobot_export_manifest.json` 保存源 episode 目录，可反查完整
`task.json` 和原始随机布局。原时间连续分段编号仍在 CSV/Parquet 中，因此将
同类帧合并到六个固定目录不会丢失阶段边界。

这些 episode 使用 manipulation base/support joint lock，因此
`stable_physics_success=true`、`training_quality_gate_passed=true`，但
`pure_physics_success=false`。它们可以进入当前稳定物理专家数据集，不能报告为“无辅助锁定的
严格纯物理成功”。此外，当前仍是 `perception_mode=sim_ground_truth`，不能报告为 RGB-D
视觉定位成功。

## 四、LeRobot 数据导出

full-physics 成功后会导出 raw episode 和 LeRobot v2 数据。

单 episode 输出结构：

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
    ├── meta/
    │   └── subtasks.jsonl
    └── episodes/
        └── 2001/                         # task_id
            └── 1/                        # episode_id
                ├── task.csv
                ├── 1-1/
                │   ├── data.csv              # nav_straight
                │   └── images/
                │       ├── front/camera0_00000.jpg
                │       └── wrist/camera0_00000.jpg
                ├── 1-2/                      # nav_turn
                ├── 1-3/                      # nav_stop
                ├── 1-4/                      # arm_approach
                ├── 1-5/                      # arm_contact
                └── 1-6/                      # arm_retreat
```

batch 输出结构：

```text
outputs/batch_run_name/
├── episode_000000/
├── episode_000001/
├── ...
├── batch_summary.jsonl
├── lerobot_export_manifest.json
└── lerobot_dataset/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    ├── videos/
    │   └── chunk-000/
    │       ├── observation.images.front/
    │       │   ├── episode_000000.mp4
    │       │   ├── episode_000001.mp4
    │       │   └── ...
    │       └── observation.images.wrist/
    │           ├── episode_000000.mp4
    │           ├── episode_000001.mp4
    │           └── ...
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── episodes_stats.jsonl
    │   ├── stats.jsonl
    │   ├── subtasks.jsonl
    │   ├── task_index_map.json
    │   └── tasks.jsonl
    ├── episodes/
    │   └── <task_id>/<episode_id>/
    │       ├── task.csv
    │       ├── <episode_id>-1/data.csv
    │       ├── <episode_id>-1/images/front/camera0_00000.jpg
    │       ├── <episode_id>-1/images/wrist/camera0_00000.jpg
    │       ├── <episode_id>-2/          # nav_turn
    │       ├── <episode_id>-3/          # nav_stop
    │       ├── <episode_id>-4/          # arm_approach
    │       ├── <episode_id>-5/          # arm_contact
    │       └── <episode_id>-6/          # arm_retreat
    └── validation_report.json
```

`episodes/<task_id>/<episode_id>/` 是供当前 VLA 训练流程直接读取的逐图片兼容层，官方 LeRobot parquet/video/meta 仍完整保留。当前目录布局版本为 `episodes_task_episode_subtask_front_wrist_v3`。每个已采集 episode 固定且仅有 6 个 subtask 目录；每个目录的 `images/` 必须且只能包含 `front/` 和 `wrist/`，两路均逐帧导出；`data.csv` 分别使用 `image_front_path` 和 `image_wrist_path` 指向对应图像。`task.csv` 使用 `subtask_directory_count`、`subtask_directories_json` 和 `source_subtask_segment_count` 分别记录固定类别目录与原始连续片段数量；尚未采集的 episode 仍只创建 `task.csv`，不会伪造六个轨迹目录。

目录编号不再按状态变化先后生成，而是固定映射为：`1=nav_straight`、`2=nav_turn`、`3=nav_stop`、`4=arm_approach`、`5=arm_contact`、`6=arm_retreat`。同一标签在 episode 中多次出现时全部合并到同一目录，并按原始 `episode_frame_index` 排序；某类未出现时仍保留空 `data.csv` 及空的 `images/front`、`images/wrist`。目录内两路图像各自使用 `camera0_00000.jpg` 从 0 连续编号。CSV 中的 `segment_index` 继续保存原始时间连续片段编号，以便恢复阶段边界；校验器会按 `episode_frame_index` 还原完整时序，并检查每帧恰好出现一次。

batch 子进程可能继承相同的 task `episode_id`。合并数据集检测到同一 `task_id` 下有重复值时，会按数据集顺序统一重编号为 1、2、3……，并在 `task.csv` 的 `source_task_episode_id` 中保留原始值，防止目录覆盖。

当前统一标签为：

```text
task_stage: nav_to_pick / pick / nav_to_place / place
subtask:    nav_straight / nav_turn / nav_stop /
            arm_approach / arm_contact / arm_retreat
```

切分默认使用 3 帧最短片段和 2 帧迟滞；必要的短接触或终端对齐片段会保留并记录原因。当前 `arm_contact` 来源为动作语义与运动学启发式，不会冒充真实接触传感器标签。

每个 `episode_XXXXXX.parquet` 的列：

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `index` | `int64` | 全局帧编号，跨 episode 单调递增。 |
| `episode_index` | `int64` | episode 编号。 |
| `frame_index` | `int64` | episode 内帧序号，从 0 开始。 |
| `timestamp` | `float32` | episode 内时间戳，单位为秒，当前数据集 `fps=5.0`。 |
| `task_index` | `int64` | 指向 LeRobot `meta/tasks.jsonl` / task metadata 的任务编号。 |
| `observation.state` | `list[float32] × 17` | 机器人主状态向量，维度顺序见下表。 |
| `observation.base_velocity` | `list[float32] × 3` | 机体系底盘速度 `[vx_body, vy_body, wz_body]`。 |
| `observation.object_state` | `list[float32] × 13` | 目标物体 pose 和速度，维度顺序见下表。 |
| `observation.tcp_pose` | `list[float32] × 7` | TCP 位姿 `[x, y, z, quat_w, quat_x, quat_y, quat_z]`。 |
| `pipeline_state` | `string` | 当前 full-physics 状态机阶段，例如 `exec_nav_to_pick`、`exec_pick`。 |
| `task_stage` | `string` | 统一任务阶段：`nav_to_pick`、`pick`、`nav_to_place` 或 `place`。 |
| `subtask` | `string` | 当前连续动作标签，取值为上面的六类之一。 |
| `subtask_segment_index` | `int64` | episode 内连续片段编号，从 1 开始。 |
| `action` | `list[float32] × 10` | VLA 训练动作：下一采样时刻实际执行到的底盘/TCP/夹爪位姿。 |
| `control.action` | `list[float32] × 11` | 同步保存的原始底盘、机械臂和夹爪控制目标。 |
| `next.done` | `bool` | episode 末帧为 `True`，其余帧为 `False`。 |

图像数据不直接写入 parquet 列。LeRobot v2 中图像作为 video feature 存储：

| Feature | 类型 | 文件位置 | 说明 |
| --- | --- | --- | --- |
| `observation.images.front` | `video[480, 640, 3]` | `videos/chunk-000/observation.images.front/episode_XXXXXX.mp4` | 前视相机 RGB 视频。 |
| `observation.images.wrist` | `video[480, 640, 3]` | `videos/chunk-000/observation.images.wrist/episode_XXXXXX.mp4` | 腕部相机 RGB 视频。 |

`observation.state` 17 维顺序：

| 维度 | 名称 | 说明 |
| --- | --- | --- |
| 0 | `base_x` | 底盘世界系 x。 |
| 1 | `base_y` | 底盘世界系 y。 |
| 2 | `base_z` | 底盘世界系 z。 |
| 3 | `base_yaw` | 底盘 yaw。 |
| 4 | `tcp_x` | TCP 世界系 x。 |
| 5 | `tcp_y` | TCP 世界系 y。 |
| 6 | `tcp_z` | TCP 世界系 z。 |
| 7 | `tcp_roll` | TCP roll。 |
| 8 | `tcp_pitch` | TCP pitch。 |
| 9 | `tcp_yaw` | TCP yaw。 |
| 10 | `arm_joint1` | 机械臂第 1 关节位置。 |
| 11 | `arm_joint2` | 机械臂第 2 关节位置。 |
| 12 | `arm_joint3` | 机械臂第 3 关节位置。 |
| 13 | `arm_joint4` | 机械臂第 4 关节位置。 |
| 14 | `arm_joint5` | 机械臂第 5 关节位置。 |
| 15 | `arm_joint6` | 机械臂第 6 关节位置。 |
| 16 | `gripper_joint7_joint8_mean` | 两个夹爪关节位置均值。 |

`observation.base_velocity` 3 维顺序：

| 维度 | 名称 | 说明 |
| --- | --- | --- |
| 0 | `vx_body` | 机体系前向线速度。 |
| 1 | `vy_body` | 机体系横向线速度。 |
| 2 | `wz_body` | 机体系 yaw 角速度。 |

`observation.object_state` 13 维顺序：

| 维度 | 名称 | 说明 |
| --- | --- | --- |
| 0 | `object_x` | 物体世界系 x。 |
| 1 | `object_y` | 物体世界系 y。 |
| 2 | `object_z` | 物体世界系 z。 |
| 3 | `object_quat_w` | 物体姿态四元数 w。 |
| 4 | `object_quat_x` | 物体姿态四元数 x。 |
| 5 | `object_quat_y` | 物体姿态四元数 y。 |
| 6 | `object_quat_z` | 物体姿态四元数 z。 |
| 7 | `object_vx` | 物体世界系线速度 x。 |
| 8 | `object_vy` | 物体世界系线速度 y。 |
| 9 | `object_vz` | 物体世界系线速度 z。 |
| 10 | `object_wx` | 物体世界系角速度 x。 |
| 11 | `object_wy` | 物体世界系角速度 y。 |
| 12 | `object_wz` | 物体世界系角速度 z。 |

`observation.tcp_pose` 7 维顺序：

| 维度 | 名称 | 说明 |
| --- | --- | --- |
| 0 | `tcp_x` | TCP 世界系 x。 |
| 1 | `tcp_y` | TCP 世界系 y。 |
| 2 | `tcp_z` | TCP 世界系 z。 |
| 3 | `tcp_quat_w` | TCP 姿态四元数 w。 |
| 4 | `tcp_quat_x` | TCP 姿态四元数 x。 |
| 5 | `tcp_quat_y` | TCP 姿态四元数 y。 |
| 6 | `tcp_quat_z` | TCP 姿态四元数 z。 |

`action` 10 维顺序：

| 维度 | 名称 | 说明 |
| --- | --- | --- |
| 0 | `base_x_world` | 下一采样时刻底盘世界系 x。 |
| 1 | `base_y_world` | 下一采样时刻底盘世界系 y。 |
| 2 | `base_yaw_world` | 下一采样时刻底盘世界系 yaw。 |
| 3 | `tcp_x_base` | 下一采样时刻 TCP 在底盘系中的 x。 |
| 4 | `tcp_y_base` | 下一采样时刻 TCP 在底盘系中的 y。 |
| 5 | `tcp_z_base` | 下一采样时刻 TCP 在底盘系中的 z。 |
| 6 | `tcp_roll_base` | 下一采样时刻 TCP 在底盘系中的 roll。 |
| 7 | `tcp_pitch_base` | 下一采样时刻 TCP 在底盘系中的 pitch。 |
| 8 | `tcp_yaw_base` | 下一采样时刻 TCP 在底盘系中的 yaw。 |
| 9 | `gripper_normalized` | 夹爪归一化值，0 为闭合、1 为张开。 |

episode 末帧没有下一采样时刻，因此使用当前姿态保持动作。`action` 的坐标系、单位、对齐方式和夹爪约定同时写入 `meta/info.json` 与 `task.csv`。

`control.action` 11 维顺序：

| 维度 | 名称 | 说明 |
| --- | --- | --- |
| 0 | `base_cmd_vx` | 底盘前向速度指令。 |
| 1 | `base_cmd_vy` | 底盘横向速度指令。 |
| 2 | `base_cmd_wz` | 底盘 yaw 角速度指令。 |
| 3 | `arm_joint1_target` | 机械臂第 1 关节目标位置。 |
| 4 | `arm_joint2_target` | 机械臂第 2 关节目标位置。 |
| 5 | `arm_joint3_target` | 机械臂第 3 关节目标位置。 |
| 6 | `arm_joint4_target` | 机械臂第 4 关节目标位置。 |
| 7 | `arm_joint5_target` | 机械臂第 5 关节目标位置。 |
| 8 | `arm_joint6_target` | 机械臂第 6 关节目标位置。 |
| 9 | `gripper_joint7_target` | 第 7 夹爪关节目标位置。 |
| 10 | `gripper_joint8_target` | 第 8 夹爪关节目标位置。 |

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
cd /home/light/workspace/arm_vla_liangzhu

python tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root /mnt/sage_data/outputs/arm_vla_liangzhu/batch_run/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out /mnt/sage_data/outputs/arm_vla_liangzhu/batch_run/episode_000000.rrd
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

以下命令均在当前良渚 worktree 中运行。入口已默认使用良渚可乐任务、
PCT 单层地图、`pct_multifloor` policy/checkpoint、identity 坐标、仓库内 collision
PLY 和 `/World/overview`，不需重复传入整组稳定参数。

```bash
cd /home/light/workspace/arm_vla_liangzhu
```

### GUI 单次 full-physics

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/gui_seed7000 \
  --seed 7000 \
  --no-headless \
  --keep-window-open
```

GUI 只建议用于单条观察。`--keep-window-open` 会在 pipeline 结束后保留窗口；
实际成败仍以 `summary.json` 和 `events.jsonl` 为准。

### Headless 单次 full-physics

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/single_seed7000 \
  --seed 7000 \
  --headless
```

### Headless batch 数据采集

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_batch.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/batch_seed7000_n20 \
  --num-episodes 20 \
  --seed 7000
```

batch 默认使用已经验证的良渚任务、联合随机化、PCT 单层地图、identity 坐标、禁止 A*
fallback、`pct_multifloor` locomotion policy/checkpoint、headless 模式、固定 `/World/overview`
相机、仓库内 collision PLY 和 collision 视觉模式。除输出目录、episode 数和 seed 外，
无需重复传入稳定参数。当前机器人 yaw 由 task JSON 在 `[-180°, 180°]`
内采样；这不是 CLI 参数。

### 复现固定任务

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/fixed_baseline \
  --seed 0 \
  --no-randomize-task \
  --no-randomize-base-goal \
  --headless
```

关闭 task randomization 后使用 task JSON 中的 nominal 机器人、可乐、鼠标垫和
base goal；`--seed` 仍会记录，但不会改变该固定布局。

### 显示随机化区域

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/randomization_debug_seed7000 \
  --seed 7000 \
  --show-randomization-debug \
  --show-planned-trajectories \
  --no-headless \
  --keep-window-open
```

### 显式加载 Gaussian 视觉场景

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/gui_gaussian_seed7000 \
  --seed 7000 \
  --navigation-visual-mode full \
  --no-headless \
  --keep-window-open
```

稳定默认是 `collision`。`full` 会加载 GaussianScene；在 8 GB GPU 上不建议将
`full` 与多路相机批量采集同时开启。

### 验证 LeRobot 数据集并导出 Rerun

```bash
conda activate isaac_locomani

python -B \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root /mnt/sage_data/outputs/arm_vla_liangzhu/batch_seed7000_n20/lerobot_dataset

conda activate lerobot_rerun

python \
  tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root /mnt/sage_data/outputs/arm_vla_liangzhu/batch_seed7000_n20/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out /mnt/sage_data/outputs/arm_vla_liangzhu/batch_seed7000_n20/episode_000000.rrd
```

## 附录:CLI 参数表

### `scripts/pipeline/run_full_physics_pipeline.py`

默认模式是 full-physics。下表只列日常运行和验收需要的参数；完整试验性 PCT/
楼梯参数以 `python -B scripts/pipeline/run_full_physics_pipeline.py --help` 为准。
机器人 yaw 范围、扇形半径和物体间距属于 task 配置，不是 CLI 参数；当前
`robot_yaw_range_deg=[-180, 180]`。只在需要 smoke/debug 时传模式参数。


| 参数                                                 | 类型 / 默认                     | 说明                                                |
| ---------------------------------------------------- | ------------------------------- | --------------------------------------------------- |
| `--task-json`                                        | 良渚可乐任务                    | 默认 `tasks/nav_pick_place_cola_liangzhu_pct.json`  |
| `--output-dir`                                       | `outputs/full_physics_pipeline` | 输出目录；真实采集建议使用 `/mnt/sage_data`       |
| `--num-episodes`                                     | `1`                             | episode 数量；真实 Isaac 模式当前只支持 1           |
| `--seed`                                             | `0`                             | episode seed；相同 task/config/seed 复现同一布局       |
| `--randomize-task` / `--no-randomize-task`           | 默认开启                        | 随机化机器人 yaw、可乐、鼠标垫及同步目标             |
| `--show-randomization-debug`                         | 默认关闭                        | 显示矩形/前向扇区和采样点 USD guide                 |
| `--show-planned-trajectories`                        | 默认关闭                        | 显示 PCT 路径和 CuRobo TCP 轨迹 guide                   |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 默认开启                        | 是否随机化 pick/place 导航交接 base_goal            |
| `--keep-window-open` / `--no-keep-window-open`       | 默认关闭                        | 结束后保留 GUI；必须配合`--no-headless`             |
| `--headless` / `--no-headless`                       | 默认`--no-headless`             | 是否无界面运行                                      |
| `--navigation-visual-mode`                           | `collision`                     | 稳定默认不加载 GaussianScene；`full` 显式加载，`auto` 保留兼容 |
| `--scene-light-mode`                                 | `camera`                        | `camera` 用于保存图像；`stage` 使用 USD 原场景灯光          |
| `--global-planner`                                   | `pct`                           | 良渚默认使用 PCT；可显式切换 `astar`                 |
| `--pct-collision-ply-path`                           | 仓库内良渚 PLY                  | 默认 `source/scene/liangzhu/ply/liangzhu_collision.ply` |
| `--pct-no-fallback` / `--pct-allow-fallback`         | 默认禁止回退                    | 默认 PCT 失败即拒绝 episode                          |
| `--pct-coord-mode`                                   | `identity`                      | 良渚 PLY 与 Isaac 使用同一坐标方向                   |
| `--policy-profile`                                   | `pct_multifloor`                | 复用已验证的 RL locomotion profile                   |
| `--locomotion-checkpoint`                            | Go2-X5 model_26000              | 默认使用仓库 checkpoint                              |
| `--require-locomotion-checkpoint`                    | 默认开启                        | checkpoint 缺失时立即失败                            |
| `--record-video`                                     | 默认关闭                        | 启用 episode 展示/observation MP4 录制；展示视频固定 25fps |
| `--video-mode`                                       | `overview`                      | `overview` 使用第三人称视角；`front`/`font` 使用前视 observation；`wrist` 使用腕部 observation；`all` 同时导出三路 |
| `--video-out`                                        | 可选                            | 视频输出目录或单个`.mp4`；多路/多 episode 请传目录  |
| `--video-width` / `--video-height`                   | `1280` / `720`                  | overview 捕获分辨率；不改变 front/wrist observation |
| `--overview-camera-mode`                             | `fixed`                         | 默认固定使用指定 overview；`auto` 才按阶段发现相机  |
| `--overview-camera-prim-path`                        | `/World/overview`               | image/video/GUI 共用的 overview Camera prim          |
| `--overview-capture-backend`                         | `viewport`                      | overview 取帧后端；`viewport` 抓最终视口画面最接近 GUI，`render_product` 使用 Replicator RGB，`auto` 先 viewport 后回退 |
| `--overview-initial-hold-frames`                     | `160`                           | 初始`third_person1`最少保持帧数，避免刚 reset 后立即切到导航镜头 |
| `--overview-exposure`                                | `0.0`                           | overview 线性 RGB 转视频前曝光补偿，单位 EV stops  |
| `--overview-gamma`                                   | `2.2`                           | overview 线性 RGB 转 sRGB 的 gamma；设为`1.0`可关闭 gamma 提亮 |
| `--pick-plan-json`                                   | 可选                            | 仅 manipulation apply smoke 使用；full-physics 禁止 |
| `--place-plan-json`                                  | 可选                            | 仅 manipulation apply smoke 使用；full-physics 禁止 |
| `--dry-run`                                          | mode                            | 无 Isaac 内存后端状态机验证                         |
| `--simulation-smoke`                                 | mode                            | 只验证真实 Isaac stage/reset                        |
| `--navigation-smoke`                                 | mode                            | 只验证 nav 到 pick                                  |
| `--navigation-carry-smoke`                           | mode                            | 验证 carry 姿态下 nav 到 place                      |
| `--pct-plan-preview`                                 | mode                            | GUI 中只预览 PCT 路径，不执行 locomotion             |
| `--pick-smoke`                                       | mode                            | 只运行到 pick 成功验证                              |
| `--manipulation-smoke`                               | mode                            | 使用假后端验证 manipulation action 合同             |
| `--manipulation-apply-smoke`                         | mode                            | 真实 Isaac 中验证 arm/gripper action 下发           |

### `scripts/pipeline/run_full_physics_batch.py`

默认模式是 full-physics，默认 headless，默认继续执行失败后的 episode。batch
会为每个 episode 启动独立 Isaac Sim 子进程，并在结束时只合并通过质量门的数据。


| 参数                                                 | 类型 / 默认   | 说明                                        |
| ---------------------------------------------------- | ------------- | ------------------------------------------- |
| `--task-json`                                        | 良渚可乐任务  | 默认 `tasks/nav_pick_place_cola_liangzhu_pct.json` |
| `--output-dir`                                       | 必填          | batch 输出目录；必须使用新目录，避免混入旧摘要       |
| `--num-episodes`                                     | `1`           | episode 数量                                |
| `--seed`                                             | `0`           | 首个 seed，后续使用`seed + episode_index`   |
| `--randomize-task` / `--no-randomize-task`           | 默认开启      | 是否按 task profile 随机化完整 episode 布局 |
| `--show-randomization-debug`                         | 默认关闭      | 显示矩形/前向扇区；通常只用于 GUI 单 episode |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 默认开启      | 是否随机化导航交接 base_goal                |
| `--headless` / `--no-headless`                       | 默认 headless | batch 是否无界面运行；量产不建议 `--no-headless`     |
| `--navigation-visual-mode`                           | `collision`   | 稳定默认不加载 GaussianScene；`full` 可显式启用    |
| `--global-planner`                                   | `pct`         | 良渚 batch 默认使用 PCT                     |
| `--pct-server-script`                                | 良渚 grid server | 默认使用仓库内 `pct_grid_server.py`       |
| `--pct-tomogram-path` / `--pct-walkable-path`        | 良渚单层资产  | 默认使用 `source/scene/liangzhu/pct/` 下资产 |
| `--pct-collision-ply-path`                           | 仓库内良渚 PLY | 默认 `source/scene/liangzhu/ply/liangzhu_collision.ply` |
| `--pct-no-fallback` / `--pct-allow-fallback`         | 默认禁止回退  | 默认 PCT 失败即拒绝 episode                 |
| `--pct-coord-mode`                                   | `identity`    | 良渚 PLY 与 Isaac 使用同一坐标方向          |
| `--policy-profile`                                   | `pct_multifloor` | 复用已验证的 RL locomotion profile       |
| `--locomotion-checkpoint`                            | Go2-X5 model_26000 | 默认使用仓库 checkpoint                  |
| `--require-locomotion-checkpoint`                    | 默认开启      | checkpoint 缺失时立即失败                   |
| `--continue-on-failure` / `--no-continue-on-failure` | 默认继续      | 单 episode 失败后是否继续                   |
| `--pick-plan-json`                                   | 可选          | 非 full-physics smoke 可转发离线 pick plan  |
| `--place-plan-json`                                  | 可选          | 非 full-physics smoke 可转发离线 place plan |
| `--progress-interval-s`                              | `5.0`         | heartbeat 进度打印间隔                      |
| `--color` / `--no-color`                             | 默认开启      | 是否使用 ANSI 彩色输出；保存 CI 日志时建议关闭       |
| `--record-video`                                     | 默认关闭      | 转发给单 episode pipeline，启用 MP4 录制；展示视频固定 25fps |
| `--video-mode`                                       | `overview`    | `overview`/`front`/`font`/`wrist`/`all`；`font` 是 `front` 兼容别名 |
| `--video-out`                                        | 可选          | 视频输出根目录；batch 会写入其下的`episode_XXXXXX/`子目录，不支持单个`.mp4` |
| `--video-width` / `--video-height`                   | `1280` / `720`| overview 捕获分辨率；不改变 front/wrist observation |
| `--overview-camera-mode`                             | `fixed`       | 默认固定使用指定 overview；`auto` 保留动态发现 |
| `--overview-camera-prim-path`                        | `/World/overview` | image/video/GUI 共用的 overview Camera prim |
| `--overview-capture-backend`                         | `viewport`    | overview 取帧后端；`viewport` 最接近 GUI，`render_product` 用于排查 fallback |
| `--overview-initial-hold-frames`                     | `160`         | 初始`third_person1`最少保持帧数              |
| `--overview-exposure`                                | `0.0`         | overview 曝光补偿，单位 EV stops            |
| `--overview-gamma`                                   | `2.2`         | overview 线性 RGB 转 sRGB gamma             |
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
