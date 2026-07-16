# Go2-X5 低层 RL Policy 速度跟踪 Benchmark

本 worktree 用于在 Isaac Sim 平地环境中测试 `pct_multifloor/model_26000.pt` 的速度跟踪能力。测试直接向低层 locomotion policy 下发机体坐标系速度命令，不加载良渚场景，也不经过 PCT、DWA 或 CuRobo，因此可以单独观察：

```text
commanded vx / vy / wz
-> pct_multifloor RL locomotion policy
-> Go2-X5 物理仿真
-> measured vx / vy / wz
```

工作目录和分支：

```text
/mnt/sage_data/workspace/arm_vla_loco_policy_benchmark
loco-policy-benchmark
```

Benchmark 保留 checkpoint 原始接口：

- policy observation：260 维。
- policy action：12 维腿部 action。
- 控制频率：50 Hz，`dt=0.02 s`。
- 地形：Isaac Lab 平面，保留原 height-scan observation 布局。
- 默认关闭外力、质量、摩擦、执行器增益和观测噪声随机化。
- [ ]  1. 环境与资产

Isaac Sim Python：

```text
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python
```

当前 checkpoint：

```text
/home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt
```

ROS2/RViz/实时曲线使用系统 ROS2 Humble：

```bash
source /opt/ros/humble/setup.bash
```

## 2. 最快运行方式

### 2.1 Quick profile

```bash
cd /mnt/sage_data/workspace/arm_vla_loco_policy_benchmark

PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/loco-benchmark-matplotlib \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/locomotion/benchmark_velocity_tracking.py \
  --checkpoint /home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --output-dir /mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/quick \
  --profile quick \
  --headless
```

### 2.2 完整速度矩阵

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/loco-benchmark-matplotlib \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/locomotion/benchmark_velocity_tracking.py \
  --checkpoint /home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --output-dir /mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/full_seed42 \
  --profile full \
  --repeats 1 \
  --seed 42 \
  --headless
```

正式统计建议使用至少 `--repeats 3`，并更换多个 seed。

## 3. 用户指定速度命令

速度命令采用机器人机体坐标系：

```text
--command VX VY WZ
```

- `VX`：前后速度，单位 m/s。
- `VY`：左右速度，单位 m/s。
- `WZ`：yaw 角速度，单位 rad/s。

例如测试 `(vx, vy, wz)=(0.20, 0.05, 0.30)`，持续 10 秒：

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/loco-benchmark-matplotlib \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/locomotion/benchmark_velocity_tracking.py \
  --checkpoint /home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --output-dir /mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/user_command \
  --command 0.20 0.05 0.30 \
  --hold-seconds 10 \
  --stop-seconds 2 \
  --headless
```

`--command` 可以重复，从而顺序执行多组速度：

```bash
--command 0.10 0.00 0.00 \
--command 0.00 0.20 0.00 \
--command 0.25 0.00 0.30
```

每组命令使用相同的 `--hold-seconds`，组间使用 `--stop-seconds` 零命令停稳。

## 4. 使用 JSON 定义命令序列

JSON 模式允许每组命令使用不同持续时间：

```json
{
  "commands": [
    {"name": "forward", "duration_s": 4.0, "vx": 0.20, "vy": 0.0, "wz": 0.0},
    {"name": "arc", "duration_s": 5.0, "vx": 0.25, "vy": 0.0, "wz": 0.30}
  ]
}
```

运行方式：

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/loco-benchmark-matplotlib \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/locomotion/benchmark_velocity_tracking.py \
  --checkpoint /home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --output-dir /mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/custom_json \
  --commands-json configs/locomotion/custom_velocity_commands.example.json \
  --stop-seconds 2 \
  --headless
```

仓库提供两个示例：

- `configs/locomotion/custom_velocity_commands.example.json`
- `configs/locomotion/liangzhu_common_velocity_commands.json`

## 5. 实时三折线图

RViz 适合显示空间轨迹、TF 和速度箭头，不适合显示时间序列。当前使用独立 ROS2/Matplotlib 窗口绘制三个折线图：

- `vx`：command 与 measured。
- `vy`：command 与 measured。
- `wz`：command 与 measured。

实时观察时，benchmark 必须带 `--real-time`，避免仿真以最快速度运行完毕。

### 终端 1：启动 JSONL -> ROS2 bridge

```bash
cd /mnt/sage_data/workspace/arm_vla_loco_policy_benchmark
source /opt/ros/humble/setup.bash

/usr/bin/python3 scripts/locomotion/ros2_velocity_tracking_bridge.py \
  --ros-args \
  -p samples_path:=/mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/user_live/samples.jsonl
```

bridge 默认每 20 ms 发布一帧，与 50 Hz 控制频率一致。需要加速离线回放时可设置：

```bash
-p rows_per_poll:=5
```

### 终端 2：启动 vx/vy/wz 实时曲线

```bash
cd /mnt/sage_data/workspace/arm_vla_loco_policy_benchmark
source /opt/ros/humble/setup.bash

MPLCONFIGDIR=/tmp/loco-ros-plot \
/usr/bin/python3 scripts/locomotion/ros2_velocity_tracking_plot.py \
  --ros-args \
  -p window_seconds:=30.0
```

### 终端 3：运行实时 benchmark

```bash
cd /mnt/sage_data/workspace/arm_vla_loco_policy_benchmark

PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/loco-benchmark-matplotlib \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/locomotion/benchmark_velocity_tracking.py \
  --checkpoint /home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --output-dir /mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/user_live \
  --command 0.20 0.05 0.30 \
  --hold-seconds 20 \
  --stop-seconds 2 \
  --headless \
  --real-time
```

如果还需要空间轨迹，可以额外运行：

```bash
rviz2 -d configs/rviz/loco_velocity_tracking.rviz
```

## 6. 输出文件

每次运行在 `--output-dir` 中生成：


| 文件                    | 内容                                                        |
| ----------------------- | ----------------------------------------------------------- |
| `samples.jsonl`         | 每个 50 Hz 控制帧的命令、实测速度、位姿和状态               |
| `segment_metrics.csv`   | 每组命令的均值、标准差、gain、MAE、RMSE、rise/settling time |
| `summary.json`          | 机器可读汇总和通过/失败结果                                 |
| `metadata.json`         | checkpoint、seed、schedule 来源和完整命令序列               |
| `velocity_tracking.png` | vx、vy、wz 的 command/measured 三折线图                     |
| `report.md`             | 人类可读实验报告                                            |

默认通过标准：

```text
steady-state gain ∈ [0.70, 1.30]
RMSE <= max(0.04, 0.30 * abs(command))
没有跌倒或环境 reset
组合命令的所有非零轴都必须通过
```

## 7. 良渚 pipeline 常用速度专项实验

实验配置：

```text
configs/locomotion/liangzhu_common_velocity_commands.json
```

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/loco-benchmark-matplotlib \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/locomotion/benchmark_velocity_tracking.py \
  --checkpoint /home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --output-dir /mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/liangzhu_common_commands_seed42 \
  --commands-json configs/locomotion/liangzhu_common_velocity_commands.json \
  --settle-seconds 2 \
  --stop-seconds 2 \
  --repeats 1 \
  --seed 42 \
  --headless
```

实验条件：每组命令持续 4 秒，取后 50% 样本作为稳态窗口，组间零命令停稳 2 秒。


| 场景用途            | command vx/vy/wz | measured vx/vy/wz      | 非零轴 gain                    | 结果 |
| ------------------- | ---------------- | ---------------------- | ------------------------------ | ---- |
| 末端 gait 前进      | `0.04/0/0`       | `-0.002/-0.003/-0.000` | vx`-0.05`                      | 失败 |
| 末端恢复前进        | `0.08/0/0`       | `0.001/-0.000/-0.003`  | vx`0.01`                       | 失败 |
| 路径恢复 creep-turn | `0.06/0/0.30`    | `0.029/-0.044/0.329`   | vx`0.49`，wz `1.10`            | 失败 |
| 最终对齐转向        | `0.16/0/0.35`    | `0.127/-0.055/0.363`   | vx`0.79`，wz `1.04`            | 通过 |
| close-goal 前进     | `0.22/0/0`       | `0.191/-0.038/-0.004`  | vx`0.87`                       | 通过 |
| 携物常规弧线        | `0.20/0/0.30`    | `0.167/-0.052/0.303`   | vx`0.84`，wz `1.01`            | 通过 |
| 门口 DWA 弧线       | `0.30/0/0.35`    | `0.258/-0.042/0.340`   | vx`0.86`，wz `0.97`            | 通过 |
| 纯 yaw settle       | `0/0/0.35`       | `-0.035/-0.046/0.387`  | wz`1.11`                       | 通过 |
| 最终位姿混合修正    | `0.16/0.08/0.30` | `0.126/0.020/0.311`    | vx`0.79`，vy `0.25`，wz `1.04` | 失败 |

汇总：

- 9 组中 5 组通过，通过率 `55.6%`。
- 共记录 2800 帧，无跌倒、无环境 reset。
- 最大命令注入误差 `1.19e-8`，可以排除 benchmark 没有正确写入速度命令。
- `vx=0.04` 和 `vx=0.08 m/s` 基本不产生前进运动，处于明显死区。
- `vx=0.06,wz=0.30` 时 yaw 能跟踪，但前进 gain 只有 `0.49`，会导致恢复轨迹转得动但走不够。
- `vx=0.16,vy=0.08,wz=0.30` 时 vy gain 只有 `0.25`，说明最终位姿横移修正不能按理想全向速度模型预测。
- 通过的弧线命令仍普遍出现约 `-0.04~-0.055 m/s` 的非指令横向漂移，DWA 预测模型需要考虑该系统偏差。
- 当前结果只覆盖 flat ground、seed 42、单次重复；它能证明低层速度响应失配，但不能单独证明所有场景导航失败都来自 policy。

完整产物：

```text
/mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/liangzhu_common_commands_seed42/report.md
/mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/liangzhu_common_commands_seed42/summary.json
/mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/liangzhu_common_commands_seed42/segment_metrics.csv
/mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/liangzhu_common_commands_seed42/velocity_tracking.png
```

## 8. 为什么原 pipeline 需要频繁调整 DWA

DWA 根据候选 `vx/vy/wz` 预测未来轨迹，默认低层能近似执行命令。但实测表明当前 policy 存在低速死区、非线性 gain、方向不对称、组合轴耦合、非指令横向漂移和停止滞后。更换场景后，窄门、楼梯入口、桌前操作区和 collision map 离散误差会把这些偏差放大，因此过去必须反复调整局部规划器来补偿。

历史上调整过的参数主要包括：


| 类别            | 参数                                                                                           | 作用与代表变化                                                            |
| --------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 速度/加速度     | `max_linear_velocity`、`max_angular_velocity`、`max_linear_accel`                              | carry 从`0.30/0.35/1.50` 调低到 `0.20/0.30/1.00`，减小楼梯和窄门扰动      |
| policy 激活速度 | `min_active_linear_velocity`、`near_goal_min_active_linear_velocity`、`close_goal_speed_limit` | 避开低速死区，同时避免近目标冲过头                                        |
| 预测参数        | `prediction_horizon`、`lookahead_distance`、waypoint tolerance                                 | 窄通道曾收紧到`0.35 s / 0.12 m / 0.03 m`                                  |
| 采样与评分      | 线/角速度采样数、`speed_bias`、路径/目标/clearance 权重                                        | 调整候选覆盖和速度、贴路径、避障偏好                                      |
| 转向策略        | rotate-in-place angle、yaw align min/max wz、creeping turn                                     | 处理大角度转向、原地漂移和转得动但走不够的问题                            |
| 末端 P 控制     | yaw Kp、激活 vx、最大 vy、lateral deadband、settle/polish 参数                                 | 解决 nav 收尾的位置/yaw 收敛和稳定切换                                    |
| 路径约束        | hard deviation、initial alignment/recovery deviation limit                                     | 防止切角撞墙并避免所有回归候选被硬拒绝；recovery 由`0.20` 放宽到 `0.35 m` |
| 局部地图        | inflate、clearance、route corridor radius                                                      | 适配点云与栅格离散；corridor 由`0.16` 放宽到 `0.24 m`                     |
| 稳定判定        | stable linear/angular tolerance、stable steps                                                  | 控制何时从导航切换到抓取或放置                                            |

所以，频繁调参不是简单说明 DWA 算法本身有问题，而是 DWA 同时承担了低层速度响应补偿和场景局部几何适配。后续应把两者拆开：DWA 保留碰撞和通道相关参数；policy deadzone、gain、侧滑和响应滞后由独立 command-response 模型补偿，或者通过重新训练 policy 解决。

## 9. 测试与常见问题

运行纯逻辑测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONDONTWRITEBYTECODE=1 \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -m pytest -q \
  tests/locomotion/test_velocity_tracking_metrics.py
```

如果 pytest 自动加载 ROS2 Humble 的 Python 3.10 插件并报 `lark` 缺失，请保留 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。Isaac Sim 使用 Python 3.11，而 ROS2 Humble 节点必须使用 `/usr/bin/python3`。

如果实时折线图没有数据，请检查：

1. 三个终端是否都执行了 `source /opt/ros/humble/setup.bash`（Isaac benchmark 终端不需要）。
2. bridge 的 `samples_path` 是否与 benchmark 的 `--output-dir` 完全一致。
3. benchmark 是否带有 `--real-time`。
4. ROS2 topic 是否存在：

```bash
ros2 topic list | grep loco_velocity_tracking
```
