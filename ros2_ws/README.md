# PCT + SCAN ROS 2 工作区运行手册

本文档记录当前开发阶段需要打开的终端、每个终端负责启动的节点，以及节点之间
的 Topic 数据流。以后增加、删除或合并启动步骤时，应同步更新本文档，避免依赖
终端历史记录。

最后核对日期：2026-08-01。

## 先分清 Node、Topic 和终端

- **终端**只是启动和观察 ROS 2 进程的窗口。
- **Node** 是实际运行的 ROS 2 程序。
- **Topic** 是 Node 之间传递消息的通道，本身不是一个需要“启动”的程序。
- 一个 launch 终端可以同时启动多个 Node，因此终端数量不等于 Node 数量。

当前 Odometry 到 TF 的学习链如下：

```text
终端 1：/learning_clock
  └─ 发布 /clock

终端 2：/learning_body_pose_source
  └─ 发布 /isaac/body_pose_raw

终端 3：pct_scan_navigation.launch.py
  ├─ /isaac_navigation_bridge
  │    ├─ 订阅 /isaac/body_pose_raw
  │    └─ 发布 /body_pose
  └─ /odometry_tf_broadcaster
       ├─ 订阅 /body_pose
       └─ 发布 /tf 中的 world -> base_link

终端 4：诊断命令
  └─ 只订阅和检查数据，不负责提供导航输入
```

## 每个终端共同执行的环境初始化

每打开一个新终端，都先执行：

```zsh
cd /mnt/sage_data/workspace/pct_scan/ros2_ws

source /opt/ros/humble/setup.zsh
source install/setup.zsh

export ROS_DOMAIN_ID=189
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

所有终端的 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION` 必须相同。否则即使各个命令
都能正常运行，Node 之间也可能互相发现不了。

必须先 `export`，再启动 `ros2 topic pub` 或 `ros2 launch`。环境变量只会在进程
启动时被读取；修改某个终端的环境变量，不会把已经运行的 Node 搬到新的 ROS
domain。若发现 domain 配置错误，应先用 `Ctrl+C` 停止对应进程，再在正确环境中
重新启动。

可以随时检查当前终端的通信域：

```zsh
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-系统默认}"
```

## 当前学习阶段：必须保持运行的三个发布终端

以下命令用于在没有启动 Isaac Sim 时，人工提供非零仿真时钟和一帧持续重复的
机器人 Odometry。它们只是学习和联调工具，不属于最终导航系统。

### 终端 1：发布临时仿真时钟

先执行上面的环境初始化，然后运行：

```zsh
ros2 topic pub \
  --rate 10 \
  --node-name learning_clock \
  --qos-profile sensor_data \
  /clock \
  rosgraph_msgs/msg/Clock \
  '{clock: {sec: 100, nanosec: 0}}'
```

该终端产生：

| 项目 | 值 |
| --- | --- |
| Node | `/learning_clock` |
| 发布 Topic | `/clock` |
| 消息类型 | `rosgraph_msgs/msg/Clock` |
| 当前用途 | 为所有 `use_sim_time=true` 的节点提供非零测试时间 |

这里使用固定的 `100 s`，只用于当前静态 TF 学习。它不会模拟连续向前推进的真实
物理时间，因此不能用于速度控制、超时行为或完整导航验收。

### 终端 2：发布临时 Isaac 原始 Odometry

先执行环境初始化，然后运行：

```zsh
ros2 topic pub \
  --rate 10 \
  --node-name learning_body_pose_source \
  --qos-profile sensor_data \
  /isaac/body_pose_raw \
  nav_msgs/msg/Odometry \
  '{
    header: {
      stamp: {sec: 100, nanosec: 0},
      frame_id: world
    },
    child_frame_id: base_link,
    pose: {
      pose: {
        position: {x: 0.0, y: 0.0, z: 0.30},
        orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
      }
    },
    twist: {
      twist: {
        linear: {x: 0.0, y: 0.0, z: 0.0},
        angular: {x: 0.0, y: 0.0, z: 0.0}
      }
    }
  }'
```

该终端产生：

| 项目 | 值 |
| --- | --- |
| Node | `/learning_body_pose_source` |
| 发布 Topic | `/isaac/body_pose_raw` |
| 消息类型 | `nav_msgs/msg/Odometry` |
| 当前用途 | 假装机器人静止在 `world` 中的 `(0, 0, 0.30)` |

消息时间戳必须和终端 1 的测试时钟一致，并且四元数 `w` 不能全部为零。

### 终端 3：启动 bridge 和动态 TF 广播器

先执行环境初始化，然后运行：

```zsh
ros2 launch isaac_navigation_bridge \
  pct_scan_navigation.launch.py \
  start_odometry_tf:=true \
  start_scan:=false \
  start_controller:=false \
  start_manual_path:=false \
  start_pct:=false \
  start_supervisor:=false
```

该 launch 启动两个 Node：

| Node | 订阅 | 发布 | 责任 |
| --- | --- | --- | --- |
| `/isaac_navigation_bridge` | `/isaac/body_pose_raw` | `/body_pose` | 校验并规范化 Isaac Odometry |
| `/odometry_tf_broadcaster` | `/body_pose` | `/tf` | 广播动态 `world -> base_link` |

不要再手动发布 `/body_pose`，因为它属于 bridge 的输出。也不要再启动另一个
`world -> base_link` 静态或动态 TF 发布器，否则同一条 TF 会出现多个数据源。

## 终端 4：检查链路，不发布导航输入

先执行环境初始化。下面的命令可以逐个运行：

```zsh
ros2 node list --no-daemon
ros2 topic list --no-daemon -t
ros2 topic info /isaac/body_pose_raw --verbose
ros2 topic info /body_pose --verbose
ros2 topic info /tf --verbose
```

最后持续检查 TF：

```zsh
ros2 run tf2_ros tf2_echo world base_link
```

正常情况下应看到平移约为 `(0, 0, 0.30)`，旋转四元数约为 `(0, 0, 0, 1)`。

一条快速自检命令：

```zsh
ros2 node list --no-daemon | sort
```

当前学习链至少应包含：

```text
/isaac_navigation_bridge
/learning_body_pose_source
/learning_clock
/odometry_tf_broadcaster
```

## 可选终端：显示 PLY 地图和手工 Path

如果还需要同时打开 PLY 地图、RViz 和楼梯手工 Path，可以另开一个终端，执行
共同环境初始化后运行：

```zsh
ros2 launch navigation_visualization \
  ply_map_visualization.launch.py \
  ply_path:=/mnt/sage_data/workspace/pct_scan/source/scene/multifloor/ply/3dgs_collision.ply \
  use_sim_time:=true \
  start_manual_path:=true
```

这个可选 launch 会启动：

| Node | 主要发布或作用 |
| --- | --- |
| `/ply_map_publisher` | 发布静态点云 `/map/ply` |
| `/world_tf_anchor` | 发布静态 `world -> pct_map` |
| `/manual_path_publisher` | 发布 `/initial_path` |
| `/navigation_rviz` | 订阅并显示点云、Path 和 TF |

`world -> pct_map` 和动态 `world -> base_link` 是不同的 TF 分支，可以同时存在。

## 启动和关闭顺序

推荐启动顺序：

1. 终端 1：`/clock`。
2. 终端 2：`/isaac/body_pose_raw`。
3. 终端 3：bridge 和 TF launch。
4. 可选地图/RViz launch。
5. 终端 4：诊断命令。

推荐关闭顺序正好相反：先停止诊断和 RViz，再停止正式 launch，然后停止原始
Odometry，最后停止 `/clock`。每个持续运行的终端使用 `Ctrl+C` 停止。

## 常见现象速查

| 现象 | 优先检查 |
| --- | --- |
| `ros2 node list` 只看到本终端的 Node | 各终端是否使用相同 `ROS_DOMAIN_ID` 和 RMW |
| domain 189 只有 learning Node，domain 0 只有 bridge Node | launch 启动时没有先设置 `ROS_DOMAIN_ID=189`；停止 launch 后在正确环境中重启 |
| `/isaac/body_pose_raw` 的 Publisher count 为 0 | 终端 2 是否仍在运行 |
| `/body_pose` 的 Publisher count 为 0 | 终端 3 的 bridge 是否启动成功 |
| `/tf` 没有 broadcaster 发布 | `start_odometry_tf` 是否为 `true`，TF Node 是否退出 |
| `world` frame 不存在 | 依次检查 `/clock`、原始 Odometry、`/body_pose` 和 `/tf` |
| Node 在运行但 Topic 没有数据 | 检查消息类型、QoS、时间戳和 frame 是否匹配 |
| Path 没有发布 | 检查 `/clock` 是否非零，以及 `manual_path_publisher` 是否仍存活 |

## 接入 Isaac Sim 后会怎样变化

正式接入 Isaac Sim/Isaac Lab 后：

- Isaac 自动发布 `/clock`，所以删除临时终端 1。
- Isaac 自动发布 `/isaac/body_pose_raw` 和 `/isaac/cloud_registered_raw`，所以删除
  临时终端 2。
- 终端 3 的正式组合 launch 继续保留，并逐步启用 PCT、SCAN、controller 和
  supervisor。
- 终端 4 仍作为可选诊断终端。

也就是说，当前三个持续发布终端是为了帮助理解消息链；最终系统不会要求人工
逐个发布仿真时钟和机器人位姿。

## 构建

修改 Python 节点、entry point、launch 或安装文件后执行：

```zsh
cd /mnt/sage_data/workspace/pct_scan/ros2_ws
source /opt/ros/humble/setup.zsh

colcon build \
  --symlink-install \
  --packages-select isaac_navigation_bridge navigation_visualization

source install/setup.zsh
```

`source install/setup.zsh` 只影响当前终端。已经打开的其他终端不会自动获得新的
安装结果，需要在那些终端里重新 source，或者关闭后重新打开。

## 文档维护约定

后续每次改变以下任一项目时，都同步更新本文档：

- 新增、删除或重命名 Node；
- 新增、删除或 remap Topic；
- 一个手动终端被 launch 合并；
- 临时发布器被 Isaac Sim、PCT 或 SCAN 的正式数据源替代；
- `ROS_DOMAIN_ID`、RMW、frame、QoS 或启动顺序发生变化。

包级参数、过滤算法和安全合同的详细说明见
[`isaac_navigation_bridge/README.md`](src/isaac_navigation_bridge/README.md)。
