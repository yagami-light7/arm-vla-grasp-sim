# Go2-X5 Catch Pipeline

当前阶段目标为实现Isaac Sim 中的 Go2-X5 固定底座抓取 demo，主要流程如下：

```text
Isaac Sim 当前 stage
-> 导出机器人状态和局部环境碰撞体
-> 根据当前选中的物体 bbox 生成抓取目标
-> 外部 cuRobo Python 进程规划机械臂轨迹
-> Isaac Sim 执行开夹爪、接近、抓取、回到home position
```

当前 demo 以 Go2-X5 上的 X5 六轴机械臂和双指夹爪为对象。cuRobo 规划器使用仅包含机械臂的模型进行规划 `arm_joint1` 到 `arm_joint6`，Isaac Sim 仿真中则依旧执行完整Go2-X5 articulation。

## 一、当前进度

已经完成：

- 完整 Go2-X5 articulation 中机械臂、夹爪 DOF 映射和 TCP frame 对齐
- 从 Isaac Sim 导出当前 arm state、`T_world_base`、`T_base_tcp` 和局部环境碰撞体
- 从 Stage 当前选中的物体生成基于 bbox 的抓取目标，当前默认优先侧向抓取
- 用 cuRobo 规划 `pregrasp -> grasp` 轨迹，并把 Isaac 导出的附近碰撞体近似成 cuRobo cuboid scene
- 在 Isaac Sim 中执行夹爪开闭、轨迹跟踪、抓取后退回和结果判定
- cuRobo one-shot 子进程规划和可选常驻 planner server 两种运行方式
- 用当前场景验证抓取闭环，夹爪能否真正抓起物体仍依赖 USD 中正确保存的 drive stiffness、damping、max force 和物体碰撞/刚体参数

长期目标：

- 目标物体需要在 Isaac Sim Stage 中手动选中，后期需要自动化
- 当前只规划固定底座上的机械臂，后续需要规划 Go2 行走和四足底盘协同
- 环境避障使用局部 collision AABB 的 cuboid 近似，后期可以考虑替换为完整 mesh-to-collision-world 转换
- 后续自动化后开始批量采集数据，并确定数据集格式

## 二、环境依赖

### 运行分层

项目运行依赖两个进程


| 进程                        | 职责                                                      | 关键依赖                                                   |
| --------------------------- | --------------------------------------------------------- | ---------------------------------------------------------- |
| Isaac Sim GUI Script Editor | 读取当前 USD stage、articulation、物体 bbox，执行关节控制 | Isaac Sim 5.1.x、`omni.usd`、`isaacsim.core`、`pxr`、NumPy |
| 外部 cuRobo Python 进程     | FK、IK、MotionPlanner、环境碰撞规划                       | CUDA、PyTorch CUDA、cuRobo source checkout、NumPy          |

轨迹规划没有放进 Isaac Sim 进程，而是明确把 cuRobo 放到外部 Python 中由外部终端执行。原因是 Isaac Sim 内部已经加载 `omni.warp`，而当前 cuRobo 环境使用另一套Warp/CUDA 组合，把 planner 留在外部进程可以避免进程内依赖冲突

### 当前代码默认路径

主流程脚本目前按以下本机路径配置：

```text
workspace:         /home/light/workspace/arm_vla
cuRobo source:     /home/light/workspace/curobo
external python:   /data/conda_envs/isaacsim51_3dgs_grasp/bin/python
```

对应代码位置：

- `scripts/isaac/05_run_pick_retreat_demo.py` 中的 `WORKSPACE` 和 `PYTHON`
- `scripts/curobo/03_plan_grasp_trajectory.py` 中的 `WORKSPACE` 和 `CUROBO_SOURCE_ROOT`
- `scripts/curobo/grasp_planner_server.py` 中的 `WORKSPACE`
- `source/robot/go2_x5/curobo/go2_x5_arm.yml` 中的 robot asset absolute path

如果仓库路径、conda env 或 cuRobo checkout 位置变化，先改这些路径，再调 demo。

### cuRobo 准备

外部规划环境需要满足：

- 能从 `/home/light/workspace/curobo` 导入 `curobo`
- `torch.cuda.is_available()` 为 `True`
- 能加载 `source/robot/go2_x5/curobo/go2_x5_arm.yml`
- 能读取 arm-only URDF 和 mesh assets
- 能调用 `curobo.motion_planner.MotionPlanner`

可以使用以下脚本验证 cuRobo 机器人模型：

```bash
cd /home/light/workspace/arm_vla

PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/dev_tools/curobo/check_go2_x5_curobo_model.py
```

如果需要检查 Isaac 导出的关节状态和 cuRobo FK 是否一致，先在 Isaac Sim中运行`scripts/isaac/01_export_go2_x5_state.py`，再运行：

```bash
cd /home/light/workspace/arm_vla

PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/dev_tools/curobo/check_isaac_curobo_fk.py
```

## 三、文件结构

### 主流程脚本

最终 demo 只依赖以下顺序链：


| 顺序 | 文件                                         | 运行位置                | 职责                                                                                                                               |
| ---- | -------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 01   | `scripts/isaac/01_export_go2_x5_state.py`    | Isaac Sim Script Editor | 自动解析当前 Go2-X5 articulation root，导出 arm/gripper DOF、TCP pose、附近环境 collision cuboids 到`/tmp/go2_x5_isaac_state.json` |
| 02   | `scripts/isaac/02_generate_grasp_target.py`  | Isaac Sim Script Editor | 读取当前选中物体 bbox 和 step 01 的 base pose，生成 side/top-down grasp target 到`/tmp/go2_x5_target_tcp_pose.json`                |
| 03   | `scripts/curobo/03_plan_grasp_trajectory.py` | 外部 Python             | 读取 state 和 target JSON，加载 cuRobo arm model，规划抓取轨迹到`/tmp/go2_x5_grasp_plan.json`                                      |
| 04   | `scripts/isaac/04_execute_grasp_sequence.py` | Isaac Sim Script Editor | 在完整 articulation 上执行 grasp plan，控制机械臂和夹爪，输出`/tmp/go2_x5_grasp_sequence_result.json`                              |
| 05   | `scripts/isaac/05_run_pick_retreat_demo.py`  | Isaac Sim Script Editor | 一键串联 step 01 到 step 04，优先开启grasp_planner_server，输出`/tmp/go2_x5_task_result.json`，以在长期运行环境下加速规划          |

辅助主流程文件：


| 文件                                     | 职责                                                                                               |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `scripts/curobo/grasp_planner_server.py` | 可选常驻 cuRobo planner service，监听 localhost，减少重复启动 Python 和初始化 MotionPlanner 的开销 |
| `scripts/math/SE3.py`                    | Isaac 脚本和普通 Python 脚本共用的 SE(3)、四元数、pose 变换工具                                    |
|                                          |                                                                                                    |

### 开发与诊断脚本

最终 demo 不直接依赖这些脚本，它们保留在 `scripts/dev_tools/`：


| 文件                                                     | 职责                                                              |
| -------------------------------------------------------- | ----------------------------------------------------------------- |
| `scripts/dev_tools/isaac/inspect_go2_x5_articulation.py` | 检查 articulation root、完整 DOF order、arm/gripper joint mapping |
| `scripts/dev_tools/isaac/inspect_gripper_tcp.py`         | 检查夹爪和 TCP frame，导出 TCP 候选信息                           |
| `scripts/dev_tools/isaac/demo_gripper_control.py`        | 单独测试双指夹爪开闭控制                                          |
| `scripts/dev_tools/curobo/make_go2_x5_arm_urdf.py`       | 从完整 Go2-X5 URDF 派生 cuRobo arm-only URDF                      |
| `scripts/dev_tools/curobo/build_go2_x5_curobo_model.py`  | 包装 cuRobo builder，生成 arm model yml/xrdf                      |
| `scripts/dev_tools/curobo/check_go2_x5_curobo_model.py`  | 检查 cuRobo yml、joint names、tool frame、FK、collision spheres   |
| `scripts/dev_tools/curobo/check_isaac_curobo_fk.py`      | 对比同一 q 下 Isaac TCP pose 和 cuRobo FK                         |
| `scripts/dev_tools/curobo/demo_plan_to_pose.py`          | 单个 TCP target 的 cuRobo planning smoke test                     |
| `scripts/dev_tools/curobo/demo_track_trajectory.py`      | 早期单条轨迹在 Isaac 中的跟踪 demo                                |

### 机器人、场景与历史目录


| 路径                                                                  | 职责                                                              |
| --------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `source/robot/go2_x5/urdf/go2_x5.urdf`                                | 完整 Go2-X5 原始 URDF                                             |
| `source/robot/go2_x5/curobo/go2_x5_arm.urdf`                          | cuRobo 规划使用的 arm-only URDF                                   |
| `source/robot/go2_x5/curobo/go2_x5_arm.yml`                           | 当前 cuRobo MotionPlanner 加载的机器人配置                        |
| `source/robot/go2_x5/curobo/go2_x5_arm.xrdf`                          | 同一 arm model 的 XRDF 描述                                       |
| `source/robot/go2_x5/meshes/`                                         | Go2-X5 和 X5 机械臂 mesh assets                                   |
| `source/robot/go2_x5/urdf/go2_x5/`                                    | Isaac Sim 导入后的 Go2-X5 USD package 和 physics/sensor/base 配置 |
| `source/scene/839920_go2_x5.usd`                                      | 当前主要 Isaac Sim 场景入口之一                                   |
| `source/scene/apple/`、`source/scene/orange/`、`source/scene/bottle/` | 物体 USD、纹理和 annotation assets                                |
|                                                                       |                                                                   |

场景里的纹理、物体 annotation、STL/DAE mesh 属于数据资产，不在 README中逐个列出。下面列出仍然保留在仓库中的代码和说明文件。

## 运行完整 demo

### 1. 准备 Isaac Sim 场景

1. 在 Isaac Sim GUI 中打开目标 USD stage，例如 `source/scene/839920_go2_x5.usd`
2. 确认 stage 中存在 Go2-X5 articulation，当前主流程会扫描
   `UsdPhysics.ArticulationRootAPI` 并自动解析 `/World/go2_x5.../root_joint
3. 固定底座或保证底盘不会在抓取过程中漂移
4. 确认夹爪 drive 参数已经写入 USD。夹爪没有足够 stiffness/max force 时，日志可能显示闭合，但物体不会被真正夹起
5. 确认目标物体有合理的 collider 和 rigid body 物理属性。
6. 在 Stage 面板中选中要抓取的物体 prim

### 2. 可选：启动常驻 planner

常驻 planner 不是必须的。未启动时 step 05 会回退到 one-shot cuRobo
子进程。常驻模式主要减少 planner 初始化开销；单次复杂规划本身仍需要时间。

在普通终端运行：

```bash
cd /home/light/workspace/arm_vla

/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/curobo/grasp_planner_server.py
```

默认监听：

```text
127.0.0.1:8765
```

### 3. 在 Isaac Sim Script Editor 运行一键 demo

推荐直接执行磁盘文件，避免 Script Editor 中残留旧粘贴代码：

```python
exec(open(
    "/home/light/workspace/arm_vla/scripts/isaac/05_run_pick_retreat_demo.py",
    "r",
    encoding="utf-8",
).read())
```

step 05 会依次执行：

```text
01 export state
02 generate grasp target
03 plan grasp trajectory
04 execute grasp sequence
05 write task summary
```

主要输出文件：


| 输出                                     | 含义                                                              |
| ---------------------------------------- | ----------------------------------------------------------------- |
| `/tmp/go2_x5_isaac_state.json`           | 当前 Isaac robot state、frame、local environment collision export |
| `/tmp/go2_x5_target_tcp_pose.json`       | 抓取目标、pregrasp、grasp、retreat/lift 相关 pose                 |
| `/tmp/go2_x5_grasp_plan.json`            | cuRobo 规划出的 segment 和 trajectory                             |
| `/tmp/go2_x5_grasp_sequence_result.json` | Isaac 执行结果、跟踪误差、物体位移                                |
| `/tmp/go2_x5_task_result.json`           | 一键任务汇总结果                                                  |

成功时终端日志应同时看到：

- state dump 成功识别当前 Go2-X5 articulation root
- target JSON 中的 `grasp_mode` 已验证
- cuRobo `all_motion_segments_success: true`
- Isaac 执行 summary 中 task success 为真

### 4. 分步运行

调试时可以按编号拆开运行。

Isaac Sim Script Editor：

```python
exec(open("/home/light/workspace/arm_vla/scripts/isaac/01_export_go2_x5_state.py", "r", encoding="utf-8").read())
exec(open("/home/light/workspace/arm_vla/scripts/isaac/02_generate_grasp_target.py", "r", encoding="utf-8").read())
```

普通终端：

```bash
cd /home/light/workspace/arm_vla

/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/curobo/03_plan_grasp_trajectory.py
```

Isaac Sim Script Editor：

```python
exec(open("/home/light/workspace/arm_vla/scripts/isaac/04_execute_grasp_sequence.py", "r", encoding="utf-8").read())
```

## 运行注意项

- `01_export_go2_x5_state.py` 会把附近 collision prim 的 world AABB 转成局部 cuboid障碍物给 cuRobo。过滤规则在脚本顶部配置；如果场景 obstacle 过大、过远或命名被排除，它不会进入规划 world
- side grasp 当前优先用于桌面较高的场景。side grasp 关闭夹爪后默认沿接近轨迹原路退出，不再额外执行一次竖直 lift
- cuRobo 规划成功不等于 Isaac 执行成功。关节跟踪误差、夹爪 drive、物体 collider、桌面碰撞和仿真帧率都会影响最终抓取

## 推荐排查顺序

1. 先运行 `scripts/dev_tools/isaac/inspect_go2_x5_articulation.py`，确认 robot root、
   articulation root 和 arm/gripper DOF mapping。
2. 单独运行 `scripts/dev_tools/isaac/demo_gripper_control.py`，确认夹爪能稳定开闭且有足够夹持力
3. 运行 step 01 和 `check_isaac_curobo_fk.py`，确认 Isaac 和 cuRobo TCP 对齐
4. 查看 step 02 打印的 grasp target 是否在 `arm_base_link` 可达范围内
5. 查看 step 03 是否加载了期望数量的 world collision cuboids，并确认不是被过粗的AABB 障碍物挡死
6. 查看 step 04 的 joint tracking error、gripper close progress 和物体 bbox 位移
