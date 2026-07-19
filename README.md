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
│   ├── box_pair_randomization.py           # 良渚 box1 -> box2 联合随机化
│   └── forward_sector_randomization.py     # 良渚机器人前向扇区联合随机化
├── scene/                                  # USD 场景、物体资产和导航地图
│   └── liangzhu/                           # 良渚 USDA、PCT 单层地图、collision PLY 和 manifest
├── robot/                                  # Go2-X5 URDF / robot 资产源文件
└── robot_lab/                              # Isaac Lab extension / Go2-X5 task registration

tasks/
├── nav_pick_place_cola_box1_to_box2_liangzhu_pct.json # 当前默认良渚 box1 -> box2 任务
├── liangzhu_placement_target.json          # 抓放、随机化、subtask、instruction 单一配置源
└── nav_pick_place_cola_liangzhu_pct.json   # 旧可乐到鼠标垫任务，仅兼容保留

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

`run_full_physics_batch.py` 默认只启动一个 Isaac 子进程，并在同一个
IsaacLab env / stage 中连续执行所有 episode。每条 episode 仍会重置机器人、
可乐、box1/box2、记录器和状态机；box1/box2 在 stage 初建时转为
“episode 内不可移动、episode 间可重定位”的 kinematic support，避免运行中热改
USD 静态 collider 时 PhysX/Fabric 仍使用上一条位姿。

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
旧的高编号 `episode_XXXXXX` 不会再覆盖当前 episode 的状态；帧诊断文件处于部分写入或
无法解析时，会回退到轻量的 `events.jsonl`。真实状态仍按
`--progress-interval-s` 打印；尚未生成本轮进度文件的 startup/pending 信息最多每 30 秒
打印一次，不再每 5 秒连续刷 `state=unknown source=unavailable`。正式采集仍应使用新的
输出目录，避免旧数据文件与本轮产物在磁盘上混放。

这一默认模式避免重复启动 Isaac Sim、加载 USD、创建 env/相机和加载
locomotion policy。若要排查跨 episode 状态污染，可显式使用
`--no-reuse-isaac-process` 回退到每条独立子进程。

batch 结束会打印以下表格：


| 列                           | 内容                                        |
| ---------------------------- | ------------------------------------------- |
| `Episode`                    | episode 编号和 seed                         |
| `随机化 Pick / Place XY`     | 本次随机化后的可乐/box2 目标 XY             |
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

#### Headless batch 性能与图像/状态同步契约

量产建议使用 `--no-record-video`。该模式下只降低与训练数据相关的渲染和
诊断 I/O 频率，不改物理或控制时序：

| 时序 / I/O | 当前值 |
| --- | --- |
| PhysX | `physics_dt=0.0025s`，400Hz |
| locomotion/control | `control_dt=0.02s`，50Hz |
| decimation | `8` 次物理 step / control step |
| headless 数据相机 | 每 10 个 control step 渲染一次，5Hz |
| `frames.jsonl` | 5Hz 数据格点 + 所有状态切换 |
| GUI / 展示视频 | 仍按每个 control step 渲染，本优化不降低 composite 频率 |

图像编码和写盘默认异步，但“取样”仍是同步事务：在同一 control tick
内读取 `front/wrist/overview`、state 和 action，立即把 GPU image 冻结为独立 CPU
buffer，并生成一个 `SynchronizedSamplePacket`。后台单 worker 只按 FIFO 顺序执行
JPEG/MP4/CSV/JSONL 编码写入，不再读仿真状态。每帧同时保存：

```text
simulation_step == camera_capture_step == state_step
simulation_timestamp == camera_capture_timestamp == state_timestamp
```

step 必须严格相等，timestamp 只保留 `1e-9s` 浮点容差；任一不一致都会立即拒绝数据。
队列满时主线程会 backpressure，episode 结束前
必须 drain 全部 packet 并审计连续 frame index。另外会根据仿真时长检查
`sampling_coverage`，防止只有一帧却被误判为有效数据集。

2026-07-19 在同一台 RTX 4060 Laptop、相同任务、相同 seed 7/8/9、
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

最终唯一 seeds 7..26 的 20 条真实 full-physics 结果为 `20/20` 成功、
`20/20` 训练质量门通过、`base_settle_timeout=0`。其中最长连续单进程 stage 复用为
15 条；外部工具中断后补齐剩余 seed，并让 seed 26 再次作为复用后的第二条验证。
20 条 pipeline 内部 wall time 平均 191.94s（约 3m12s）、中位数 189.10s。
按旧版 3-seed 均值作吞吐估算，每条约省 94.91s（33.1%），20 条约省 31m38s；
该 20 条估算不是逐 seed 旧版配对，严格配对结论只使用上表 seeds 7/8/9。

### 当前良渚 box1 -> box2 任务

默认任务是：

```text
Pick up the coke can on box1 and place it on box2.
```

入口文件为 `tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json`。抓取点、
放置点、box 几何、随机化、六类 subtask 和英文 instruction 模板统一维护在
`tasks/liangzhu_placement_target.json`；task loader 在运行前将其中的
`task_overrides` 合并到 EpisodeSpec，不再要求在多个 JSON 中同步修改同一套标注。

当前随机化模式为 `liangzhu_box_pair_xy_v1`：

| 项目 | 当前分布 | 不变量 |
| --- | --- | --- |
| box1 根位姿 | nominal XY 各加 `Uniform(-0.12m, 0.12m)` | 根 Z、authored orient、scale、xform op 顺序不变 |
| box2 根位姿 | nominal XY 各加 `Uniform(-0.12m, 0.12m)` | 根 Z、authored orient、scale、xform op 顺序不变 |
| 机器人位置 | 两个 box 中心连线的 `0.4..0.6` 区间，另加 `±0.18m` 横向偏移 | 使用 collision PLY 重新求地面高度；与两个 box 保持净空 |
| 机器人 yaw | `Uniform(-180°, 180°)` | roll/pitch 不随机 |
| 可乐位置 | box1 中心局部安全区，半宽 `0.08m × 0.06m` | 由 box1 顶面和可乐 bbox 半高求 Z，并保留桌边余量 |
| 可乐 yaw | `Uniform(-180°, 180°)` | roll/pitch 为 0 |
| box2 placement region | box2 中心半宽 `0.10m × 0.05m` | 保留桌边余量 |
| pick standoff | `Uniform(0.50m, 0.54m)` | 底盘最终正面朝向可乐 |
| place standoff | `Uniform(0.48m, 0.51m)` | 底盘最终正面朝向 box2，并为 `0.08m` 导航交接容差预留机械臂可达裕量 |

每次采样会同步更新 robot reset、box1/box2 stage translate、可乐 pose、box2
placement region、pick/place base goal、两个 CuRobo 支撑 proxy 和 PCT 局部动态
keepout。box2 原资产没有 authored collision，runtime 会在 physics 初始化前为
`/World/box2/node_0` 添加静态 mesh collision；不会保存或修改 USD 资产。

#### 可乐初始化落稳

box1 任务启用 `supported_upright_v1` 初始化策略。episode reset 后，runtime 通过
PhysX tensor API 恢复随机化请求的可乐世界位姿，不改写 USD 的 `xformOp` 或
`unitsResolve`。前 8 个控制步只允许 Z 向沉降，并保持 XY 与直立姿态；接触箱面后
将可乐置于 sleep，抓取接触仍会由 PhysX 自动唤醒。状态机在进入导航前独立检查：

- XY 漂移不超过 `0.02m`；
- Z 沉降不超过 `0.02m`；
- 与请求姿态的四元数角误差不超过 `0.10rad`。

任何一项超限都会以 `object_initialization_pose_invalid` 拒绝 episode，不能再把
已经摔倒的可乐当作有效初始状态。历史 batch 的 seed 7 曾把倾倒
`96.03°`、XY 漂移 `84.07mm` 的可乐误判为落稳；修复后同一 seed 的真实
`pct_scene` 复跑在 47 步完成落稳，倾角约 `0.000005°`、XY 漂移约
`0.00019mm`、Z 沉降 `4.95mm`，随后完整完成 pick、携物导航、place 和数据导出。
验收输出位于：

```text
/mnt/sage_data/outputs/pct_scene/
seed7_object_init_upright_fix_v2_20260719/episode_000000
```

布局最多尝试 300 次，并拒绝桌间距不足、机器人/导航交接点离桌过近、可乐越出
box1 安全区或 collision PLY 地面支撑不满足约束的样本。同一个 task/config/seed 会
复现相同布局。除 20 seed 静态随机化和 40 次 PCT 请求检查外，2026-07-18 已完成两份
worktree 的真实 headless full-physics batch：

| worktree / seeds | 尝试 | 完整成功并进入统一数据集 | 数据集帧数 | validator | composite |
| --- | ---: | ---: | ---: | --- | --- |
| `arm_vla_liangzhu` / `0..4` | 5 | 5 | 1004 | 0 error / 0 warning | 5/5，零丢帧 |
| `pct_scene` / `0..19` | 20 | 18 | 4096 | 0 error / 0 warning | 20/20，零丢帧 |

两边合计尝试 25 条、通过训练质量门并导出 23 条。历史 `pct_scene` batch 的 seed 7
在初始化时已经摔倒但被旧门禁误接收，最终又以 `place_release_pose_error` 被拒绝；
seed 19 以 `place_release_ejected` 被拒绝。两条历史失败原始数据保留但没有混入统一
训练数据集。seed 7 的初始化 bug 已按上文修复并定向复跑通过，但这不会回写或改称原
batch 为 19/20。

两边 seed 0..4 的 `randomization` 采样及派生出的 start、pick/place、动态 keepout、
空间约束和 mesh-truth 目标逐 JSON 哈希一致；比较时只排除了必然不同的 worktree 绝对
collision PLY 路径。对应输出证据为：

```text
/mnt/sage_data/outputs/arm_vla_liangzhu/
alignment_batch5_seed0_composite_20260718

/mnt/sage_data/outputs/pct_scene/
liangzhu_headless_batch20_seed0_standofffix_20260718_v2
```

当前默认任务的 pick 导航容差为 `0.10m`，place 交接容差为 `0.08m`；box2
`pre_place_clearance` 与 `retreat_clearance` 均为 `0.08m`。完成通用导航接口修正后，
seed 5000 已在宿主机 RTX 4060 上真实跑通：PCT + RL locomotion + CuRobo + PhysX
状态机到达 `done`，抓取最大抬升 `0.2389m`，放置 XY/Z 误差分别为
`0.0080m / 0.0077m`。LeRobot 导出 291 帧、6 个固定 subtask 目录，front/wrist
各 291 张，validator 为 0 error / 0 warning。输出证据位于：

```text
/mnt/sage_data/outputs/arm_vla_liangzhu/
nav_structure_seed5000_20260717_v1
```

这条 seed 5000 结果是早期单条基线；上面的 2026-07-18 双 worktree batch 是当前
多 seed 成功率和数据格式的权威验收结果。

headless 量产当前保持 `navigation_visual_mode=collision`。2026-07-18 的
Gaussian/NUREC 实测能够加载资产并启动 PhysX，但首帧三相机渲染会触发
`cudaErrorIllegalAddress (700)`。这是 Isaac Sim 5.1 headless 渲染路径的运行时限制，
不是 CUDA 设备不可用。在该路径修复前，不得把 `full` 设为 batch 默认；GUI
可用 `--navigation-visual-mode full` 做单条视觉调试。

#### 地图无关的导航路径接口

seed 5000 的 PCT 两段全局路线本来都是两点直线。旧实现出现额外转弯的触发链是：

1. PCT 将连续世界坐标吸附到 `0.2m` 栅格中心；旧局部层把这个吸附点误当成机器人
   的真实控制起点。
2. 局部层额外使用 `max(0.4m, 2 * resolution)` 的隐藏 clearance 门禁；合法的
   `0.2m` 末端操作通道也会被判定为需要重新 A*。
3. A* 再次输出栅格中心，首个 waypoint 可能落在机器人后方；DWA 会忠实地先追这个
   人造点，形成反向首转、折返和原地踏步。

低层 RL policy 对异常角速度命令的相关系数为 `0.989`，说明它执行了局部层给出的
命令；根因不在 locomotion policy，也不需要逐地图重调 DWA/PCT 参数。当前统一策略为：

```text
PCT 全局拓扑路径
-> 用机器人实时 XY / 精确目标 XY 恢复连续端点
-> 在已膨胀 occupancy configuration space 中做 supercover 线段检查
-> 直线可通行时直接使用两点路径
-> 直线阻塞时才运行 local A*，并对结果做 line-of-sight string pulling
-> DWA 首帧把机器人投影到整条路径，跳过已经位于身后的 path prefix
-> 携物离开支撑物时先按支撑物 bbox 与机器人扫掠半径计算安全退出距离
-> 退出后停稳并使用实时位姿重新锚定剩余路径
-> 末端 P 控制器在保持最终 yaw 的同时用 body-frame vx/vy 修正剩余 XY
```

仅当机器人所在单个栅格因离散化被标记占用时，局部层才临时恢复该一个 start cell；
目标占用、越界或任一输出线段碰撞都会结构化失败，不再静默清障或退回原始路径。PCT
吸附同时使用目标方向作为同层候选的 tie-break，并导出吸附前后坐标与距离，避免固定
网格轴顺序造成场景相关偏置。

修正后的 seed 5000 两段局部路径均为精确两点直线，`turn_count=0`。与旧实现相比，
导航控制步数从 1773 降到 1504，角速度命令绝对积分从 `692.7°` 降到 `603.0°`，
转向方向反复切换从 18 次降到 6 次。剩余旋转来自随机初始 yaw、目标位于身后以及
操作位姿最终 yaw 对齐，而不是栅格 waypoint 绕行。

box1 抓取后的携物导航还增加了两个与地图无关的执行约束：安全退出距离由当前支撑物
世界 bbox、机器人扫掠半径和配置余量实时计算，而不是写死 box1 坐标；同层 DWA 的
原地旋转门使用进入/退出迟滞和停稳窗口，避免航向误差在阈值附近反复切换。所有退出
进度、重锚定位姿、旋转门和末端控制模式都写入导航 frame metadata，可区分全局路径、
局部跟踪和操作交接问题。

旧末端控制器在“XY 已进入较宽交接区、但仍有数厘米误差”时，会先转向这个微小位置
误差，前进后再转回操作最终 yaw。seed 5003 因此额外执行了约 `95° + 95°` 的往返旋转。
当前同层携物收尾直接在最终 yaw 下生成 body-frame `vx/vy`，只用连续 P 控制修正 yaw；
跨楼层 profile 保留旧的非完整约束兼容路径。真实回归结果如下：

| seed / 版本 | 携物导航 ticks | 实际 yaw 累计 | 角速度命令累计 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 5003 / 旧末端控制 | 1423 | 407.71° | 417.87° | pipeline 成功，但末端有往返旋转 |
| 5003 / 最终 yaw 平移收尾 | 930 | 233.06° | 237.64° | 导航成功；后续 CuRobo place 规划失败 |
| 5002 / 最终 yaw 平移收尾 | 897 | 254.41° | 243.01° | 完整 pipeline 到 `done` |

seed 5003 的新导航比旧版本少 493 ticks、少约 `174.65°` 实际旋转，且离开 box1 后
机器人中心与 box1 中心的最小距离保持 `0.6264m`。它随后失败在 CuRobo
`plan_place`：实际底盘误差把 pre-place 目标推到 arm-base 半径约 `0.4746m` 的边界；
这不是 PCT 或 DWA 失败。它暴露了下一项跨层契约：base-goal 候选和导航验收应显式
预留机械臂可达裕量，必要时触发小范围底盘二次对位，而不是继续按地图重调局部规划器。

seed 5002 已验证上述改动不破坏完整闭环：抓取、携物导航、CuRobo 放置、释放验证和
LeRobot 导出全部成功；数据集为 196 行，front/wrist/overview 各 196 帧，validator
为 0 error / 0 warning，`training_eligible=true`。证据目录为：

```text
/mnt/sage_data/outputs/arm_vla_liangzhu/
nav_goal_yaw_terminal_seed5002_20260717_v4
```

换场景时仍需要提供正确的 PCT tomogram/walkable、坐标标定、机器人半径膨胀和动态
keepout；这些是场景输入，不是需要重新调一套控制器。机器人 footprint、RL policy
速度响应/死区和执行延迟属于 robot profile，应标定一次后跨地图复用；支撑物尺寸、
操作目标和机械臂可达裕量属于 task/manipulation contract，应由几何与可达性检查派生。
跨不同地图原点、地图旋转、起点栅格相位和 17 个初始 yaw 的回归测试由
`tests/navigation/test_path_refinement.py` 覆盖。

#### 分段英文 instruction annotation

subtask 标签和六目录结构保持不变：

```text
nav_straight / nav_turn / nav_stop /
arm_approach / arm_contact / arm_retreat
```

每个连续 segment 在首帧绑定一条固定 instruction。`nav_turn` 用首帧机器人 yaw 与
当前目标的相对方位生成八方向标签；其他抓取侧/放置侧 segment 使用固定动作指令：

| 阶段 | 触发条件 | `instruction_id` | 英文模板 |
| --- | --- | --- | --- |
| 寻找 box1 | `nav_to_pick + nav_turn` | `find_pick_box` | `Turn toward your {direction} to find the box with the coke can.` |
| 前进并抓取 | 其余 `nav_to_pick` 与全部 `pick` | `pick_from_front_box` | `Pick up the coke can from the box in front of you.` |
| 寻找 box2 | `nav_to_place + nav_turn` | `find_place_box` | `Turn toward your {direction} to find the box where you can place the coke can.` |
| 前进并放置 | 其余 `nav_to_place` 与全部 `place` | `place_on_front_box` | `Place the coke can on the box in front of you.` |

八方向标签统一为 `front`、`front-left`、`left`、`back-left`、`back`、
`back-right`、`right`、`front-right`。annotation schema 为
`relative_direction_segment_instruction_v1`，语言固定为 `en`。同一固定类别目录内
可能同时出现 pick-side 与 place-side 行，因此 instruction 按帧/连续 segment 保存，
不能只根据六个目录名推断。

### 旧良渚可乐到鼠标垫随机化（兼容/历史）

以下 `robot_forward_sector_v1`、鼠标垫参数和 2026-07-15/16 实测只描述旧任务
`tasks/nav_pick_place_cola_liangzhu_pct.json`，不再是 CLI 默认，也不能作为当前
box1 -> box2 真实运行成功率。

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
`randomization.mode` 选择具体随机化算法。batch 默认复用同一 Isaac 进程和
stage，但每条仍使用独立 seed 和记录目录：

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

旧可乐到鼠标垫 task 的 pick 底盘末端位置容差为 `0.10m`，place 底盘交接使用独立的 `0.15m`，
place 目标物体中心 XY 容差为 `0.040m`，Mesh truth 物体竖直 extent 审计容差为 `0.010m`。
按 `0.3 x 0.2m` 鼠标垫与约 `0.0343m` 可乐足迹半径计算，即使落在该中心容差边界，
短边仍保留约 `0.0257m` 的实体边缘余量。
这些数值来自本轮真实随机 batch，而不是用于隐藏失败；
导航碰撞、不可达、掉落和放置验证失败仍会明确拒绝该 episode。

#### 2026-07-16 全向 yaw 随机 batch 实测

使用当前 `robot_yaw_range_deg=[-180, 180]` 配置连续运行
`seed=7000..7019` 共 20 个真实 headless full-physics episode。20 个实际采样 yaw
覆盖 `-170.50°..157.95°`，四个 90° 象限分别包含 `4 / 5 / 6 / 5` 条，不是只在
原来的正前方小角度附近采样。

| 指标 | 结果 |
| --- | --- |
| 连续 batch 尝试数 | 20 |
| pipeline 成功 / 质量门通过 | 19 / 19 |
| 隔离失败 | 1，seed 7018 的旧 `0.035m` placement center 边界拒绝 |
| 连续 batch 实测成功率 | 95% |
| 统一数据集 | 19 episodes，2419 rows，5 Hz |
| 视觉 | front / overview / wrist，57 个 mp4；子任务 front/wrist 各 2419 JPG |
| subtask | 每 episode 固定 6 目录，共 114 目录 |
| action | 10D VLA action + 11D `control.action` |
| validator | valid=true，19 episodes，2419 rows，0 error，0 warning |

本批统一数据集位于：

```text
/mnt/sage_data/outputs/arm_vla_liangzhu/
validation_full_yaw_seed7000_n20_final_v2_20260715/lerobot_dataset
```

seed 7018 的最终可乐中心相对目标误差为 `0.03580m`，且仅越过旧 placement region
边界约 `0.00024m`；物体已经稳定释放在鼠标垫上，不是抓取、导航、掉落或物理放置失败。
将中心容差调整为 `0.040m` 后，定向真实复跑的最终 XY 误差为 `0.03633m`、Z 误差为
`0.00032m`、线速度为 `0.00063m/s`，完整状态机成功到达 `done`，结果位于：

```text
/mnt/sage_data/outputs/arm_vla_liangzhu/
revalidate_full_yaw_seed7018_region40_20260716
```

因此这组 20 个全向 seed 均已有成功执行证据，但严格统计仍应表述为“连续 batch 19/20，
唯一失败 seed 修正后定向通过”，不能写成同一次连续 batch 20/20。当前结果说明全向 yaw
下 PCT、局部导航、locomotion、top-down 抓取和放置链路保持稳定；继续扩大随机范围时仍需
重新统计连续 batch 成功率。

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
        └── 2002/                         # 当前 box1 -> box2 task_id
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

当前 box1 -> box2 任务还会把分段 instruction 写入 parquet 和每个固定类别目录的
`data.csv`。`task.csv` 保存 annotation schema/language；同一连续 segment 内这些字段
保持不变：

```text
instruction
instruction_id
instruction_target_id
instruction_direction
instruction_relative_bearing_rad
instruction_pose_source
instruction_annotation_schema
```

其中 `instruction_direction` 只在寻找 box 的 `nav_turn` segment 中取八方向英文标签；
动作 segment 为空值。episode 级 `task_index` 和总任务 instruction 继续保留，用于兼容
标准 LeRobot task metadata，不替代上述逐帧 instruction。

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
| `instruction` | `string` | 当前连续 segment 的英文自然语言指令。 |
| `instruction_id` | `string` | 四类模板 ID，例如 `find_pick_box`。 |
| `instruction_target_id` | `string` | 当前目标实例 ID，寻找/抓取为可乐，寻找/放置为 box2。 |
| `instruction_direction` | `string` | 寻找指令的八方向标签；非寻找 segment 为空。 |
| `instruction_relative_bearing_rad` | `float32` | segment 首帧机器人到目标的相对方位；非寻找 segment 为 null。 |
| `instruction_pose_source` | `string` | 用于生成方位的机器人 pose 来源。 |
| `instruction_annotation_schema` | `string` | 当前为 `relative_direction_segment_instruction_v1`。 |
| `action` | `list[float32] × 10` | VLA 训练动作：下一采样时刻实际执行到的底盘/TCP/夹爪位姿。 |
| `control.action` | `list[float32] × 11` | 同步保存的原始底盘、机械臂和夹爪控制目标。 |
| `next.done` | `bool` | episode 末帧为 `True`，其余帧为 `False`。 |

图像数据不直接写入 parquet 列。LeRobot v2 中图像作为 video feature 存储：

| Feature | 类型 | 文件位置 | 说明 |
| --- | --- | --- | --- |
| `observation.images.front` | `video[480, 640, 3]` | `videos/chunk-000/observation.images.front/episode_XXXXXX.mp4` | 前视相机 RGB 视频。 |
| `observation.images.wrist` | `video[480, 640, 3]` | `videos/chunk-000/observation.images.wrist/episode_XXXXXX.mp4` | 腕部相机 RGB 视频。 |

front 与 wrist 均使用 D436 的 640×480 标定内参：`fx=383.44608095`、
`fy=383.52724198`、`cx=324.33479864`、`cy=238.90275478`，OpenCV
pinhole 的 12 个畸变系数均为 0。runtime 会尝试启用
`OmniLensDistortionOpenCvPinholeAPI`；当前 IsaacLab 5.1 headless 实测未把该 API
应用到 render camera，因此会显式回退到标准 USD pinhole，实际渲染 K 为
`fx=fy=383.486661465`、`cx=320`、`cy=240`。该近似与标定 K 最大相差约 4.3 px，
实际生效值记录在 `camera_runtime_intrinsics_report`。wrist camera 挂载在
`arm_link6`，prim 为
`{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera`；其
原始 `arm_link6_T_camera_color_optical` 手眼标定为位置
`(0.0559054476, 0.0026732239, 0.0767149320)` m、wxyz 四元数
`(0.3377891849, -0.6214992221, 0.6185057335, -0.3421810063)`。由于标定板弯曲，
当前另加一层可追溯的视觉对齐修正：在 ROS optical frame 沿相机 `-Y` 平移
`0.02 m`，不修改旋转；最终仿真安装位置为
`(0.0666580792, 0.0028071889, 0.0935779972)` m。该平移把夹爪近端移到图像下方，
同时保持相机到 TCP 的光轴深度约 `0.1270 m`。禁止通过 optical `+Z` 前移和近裁剪
隐藏夹爪底座：已验证 `+0.04 m` 会使 `0.03 m` near clipping 切入抓持中的可乐
mesh。box1→box2 任务还启用了逐帧 wrist/目标表面间距门禁，要求可见可乐表面至少
位于 near clipping 之后 `0.01 m`；违规 episode 会被标记为不可训练。历史 metadata 中若出现
`hand_eye_calibration_with_visual_alignment_v2`，pipeline 重新处理该 summary 时会直接拒绝，
对应输出不得用于 VLA 训练。该修正来自实机画面约束，不应表述为新的精密手眼标定。
front camera 仅更新为同一套内参，
安装外参仍沿用现有
`base/head_cam` 配置。front/wrist 请求非 640×480 分辨率时 runtime 会直接报错，
避免静默套用不匹配的标定参数。

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

以下命令均在当前良渚 worktree 中运行。入口已默认使用良渚 box1 可乐到 box2 任务、
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
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_gui_seed5000 \
  --seed 5000 \
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
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_single_seed5000 \
  --seed 5000 \
  --headless
```

### 三视角视频完整验收

`composite` 使用同一个 simulation step 的 `overview/front/wrist` 图像，overview 位于
左侧 2/3，front/wrist 分别位于右上和右下。三路都保持原始宽高比并带英文标签，默认
输出 1280×720、25fps。以下命令同时验收状态机、物理导航/操作、LeRobot 数据和拼接视频：

```bash
set -euo pipefail
cd /home/light/workspace/arm_vla_liangzhu
export PYTHONDONTWRITEBYTECODE=1

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

拼接视频位于：

```text
$OUT/episode_000000/overview_videos/episode_000001_composite.mp4
```

验收脚本要求 `final_state=done`、导航/操作/稳定物理/训练质量门全部通过、LeRobot
`error_count=0` 且 `warning_count=0`，并校验 MP4 帧数、25fps 和 1280×720 分辨率。

### Headless batch 数据采集

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_batch.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_batch_seed5000_n20 \
  --num-episodes 20 \
  --seed 5000 \
  --no-record-video
```

batch 默认使用当前 box1 -> box2 任务、联合随机化、PCT 单层地图、identity 坐标、禁止 A*
fallback、`pct_multifloor` locomotion policy/checkpoint、headless 模式、固定 `/World/overview`
相机、仓库内 collision PLY、collision 视觉模式，并默认复用同一 Isaac 进程/
stage。除输出目录、episode 数和 seed 外，无需重复传入稳定参数。量产时建议
显式传 `--no-record-video`，LeRobot 的 front/wrist/overview 三路 5Hz 数据仍会完整导出。
box1/box2 只随机 XY；机器人在两箱之间生成且 yaw 在
`[-180°, 180°]` 内采样；可乐在 box1 中央安全区随机 XY/yaw。这些范围由
`tasks/liangzhu_placement_target.json` 管理，不是 CLI 参数。2026-07-19 已用该默认入口
完成唯一 seeds 7..26 的 20 条真实 headless full-physics 验证：20/20 到达 `done`，
20/20 通过训练质量门，`base_settle_timeout=0`。更换场景资产、randomization 范围、
locomotion checkpoint 或控制参数后仍需重新建立小批次门禁，不能沿用本轮成功率。

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

关闭 task randomization 后使用 annotation JSON 中的 nominal 机器人、box1、box2、可乐和
base goal；`--seed` 仍会记录，但不会改变该固定布局。

### 显示随机化区域

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_debug_seed5000 \
  --seed 5000 \
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
  --output-dir /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_gaussian_seed5000 \
  --seed 5000 \
  --navigation-visual-mode full \
  --no-headless \
  --keep-window-open
```

稳定默认是 `collision`。`full` 会加载 GaussianScene；在 8 GB GPU 上不建议将
`full` 与多路相机批量采集同时开启。

灯光模式默认是 `auto`：使用 `--navigation-visual-mode full` 时会自动显示 USDA
中编写的 `DomeLight`、`SphereLight`、`RectLight` 等 stage lights，并关闭相机
补光灯；`collision` 模式则继续使用相机补光。通常无需再追加
`--scene-light-mode stage`，只有需要覆盖自动行为时才显式传 `camera` 或 `stage`。

### 验证 LeRobot 数据集并导出 Rerun

```bash
conda activate isaac_locomani

python -B \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_batch_seed5000_n20/lerobot_dataset

conda activate lerobot_rerun

python \
  tools/lerobot_to_rerun.py \
  --repo-id full_physics_dataset \
  --root /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_batch_seed5000_n20/lerobot_dataset \
  --episode-index 0 \
  --max-frames 200 \
  --out /mnt/sage_data/outputs/arm_vla_liangzhu/box_pair_batch_seed5000_n20/episode_000000.rrd
```

## 附录：CLI 参数表

### `scripts/pipeline/run_full_physics_pipeline.py`

默认模式是 full-physics。下表只列日常运行和验收需要的参数；完整试验性 PCT/
楼梯参数以 `python -B scripts/pipeline/run_full_physics_pipeline.py --help` 为准。
box XY 范围、机器人生成区间/yaw、可乐中央安全区和物体间距属于 annotation
配置，不是 CLI 参数；当前 `robot_yaw_range_deg=[-180, 180]`。只在需要
smoke/debug 时传模式参数。


| 参数                                                 | 类型 / 默认                     | 说明                                                |
| ---------------------------------------------------- | ------------------------------- | --------------------------------------------------- |
| `--task-json`                                        | 良渚 box1 -> box2 任务          | 默认 `tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json` |
| `--output-dir`                                       | `outputs/full_physics_pipeline` | 输出目录；真实采集建议使用 `/mnt/sage_data`       |
| `--num-episodes`                                     | `1`                             | episode 数量；headless full-physics 可在同一 stage 连续执行 |
| `--reuse-isaac-stage` / `--no-reuse-isaac-stage`     | 默认开启                        | 复用 USD env/stage；每条仍硬重置 PhysX view/solver，排查隔离问题时可关闭 |
| `--seed`                                             | `0`                             | episode seed；相同 task/config/seed 复现同一布局       |
| `--randomize-task` / `--no-randomize-task`           | 默认开启                        | 联合随机化 box XY、机器人 pose、可乐 pose 及同步目标 |
| `--show-randomization-debug`                         | 默认关闭                        | 显示 box/目标/导航交接点 USD guide                   |
| `--show-planned-trajectories`                        | 默认关闭                        | 显示 PCT 路径和 CuRobo TCP 轨迹 guide                   |
| `--randomize-base-goal` / `--no-randomize-base-goal` | 默认开启                        | 是否随机化 pick/place 导航交接 base_goal            |
| `--keep-window-open` / `--no-keep-window-open`       | 默认关闭                        | 结束后保留 GUI；必须配合`--no-headless`             |
| `--headless` / `--no-headless`                       | 默认`--no-headless`             | 是否无界面运行                                      |
| `--navigation-visual-mode`                           | `collision`                     | 稳定默认不加载 GaussianScene；`full` 显式加载，`auto` 保留兼容 |
| `--scene-light-mode`                                 | `auto`                          | `full` 自动使用 USD 原场景灯光，`collision` 自动使用相机补光；可用 `camera`/`stage` 覆盖 |
| `--global-planner`                                   | `pct`                           | 良渚默认使用 PCT；可显式切换 `astar`                 |
| `--pct-collision-ply-path`                           | 仓库内良渚 PLY                  | 默认 `source/scene/liangzhu/ply/liangzhu_collision.ply` |
| `--pct-no-fallback` / `--pct-allow-fallback`         | 默认禁止回退                    | 默认 PCT 失败即拒绝 episode                          |
| `--pct-coord-mode`                                   | `identity`                      | 良渚 PLY 与 Isaac 使用同一坐标方向                   |
| `--policy-profile`                                   | `pct_multifloor`                | 复用已验证的 RL locomotion profile                   |
| `--locomotion-checkpoint`                            | Go2-X5 model_26000              | 默认使用仓库 checkpoint                              |
| `--require-locomotion-checkpoint`                    | 默认开启                        | checkpoint 缺失时立即失败                            |
| `--record-video` / `--no-record-video`               | 完整 pipeline 默认开启          | 默认录制 episode 三视角展示 MP4；批量空间敏感时可显式关闭，展示视频固定 25fps |
| `--video-mode`                                       | `composite`                     | 将同步 overview/front/wrist 拼成单个视频；其余支持 `overview`/`front`/`font`/`wrist`/`all` |
| `--video-out`                                        | 可选                            | 视频输出目录或单个`.mp4`；多路/多 episode 请传目录  |
| `--video-width` / `--video-height`                   | `1280` / `720`                  | overview 捕获或 composite 输出分辨率；不改变 observation 原始分辨率 |
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
默认只启动一个 Isaac Sim 子进程并复用 stage，在结束时只合并通过质量门的数据。


| 参数                                                 | 类型 / 默认   | 说明                                        |
| ---------------------------------------------------- | ------------- | ------------------------------------------- |
| `--task-json`                                        | 良渚 box1 -> box2 任务 | 默认 `tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json` |
| `--output-dir`                                       | 必填          | batch 输出目录；必须使用新目录，避免混入旧摘要       |
| `--num-episodes`                                     | `1`           | episode 数量                                |
| `--reuse-isaac-process` / `--no-reuse-isaac-process` | 默认开启      | 复用单个 Isaac 进程/USD stage；episode 间硬重置 PhysX，关闭后每条独立进程 |
| `--seed`                                             | `0`           | 首个 seed，后续使用`seed + episode_index`   |
| `--randomize-task` / `--no-randomize-task`           | 默认开启      | 是否按 task profile 随机化完整 episode 布局 |
| `--show-randomization-debug`                         | 默认关闭      | 显示 box/目标/交接点；通常只用于 GUI 单 episode |
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
| `--progress-interval-s`                              | `5.0`         | 真实状态 heartbeat 间隔；startup/pending 低信息状态最多每 30 秒打印一次 |
| `--color` / `--no-color`                             | 默认开启      | 是否使用 ANSI 彩色输出；保存 CI 日志时建议关闭       |
| `--record-video` / `--no-record-video`               | 完整 pipeline 默认开启 | 默认转发 composite MP4 录制；空间敏感时可显式关闭，展示视频固定 25fps |
| `--video-mode`                                       | `composite`   | 三视角拼接；也支持 `overview`/`front`/`font`/`wrist`/`all` |
| `--video-out`                                        | 可选          | 视频输出根目录；batch 会写入其下的`episode_XXXXXX/`子目录，不支持单个`.mp4` |
| `--video-width` / `--video-height`                   | `1280` / `720`| overview 捕获或 composite 输出分辨率；不改变 observation 原始分辨率 |
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
