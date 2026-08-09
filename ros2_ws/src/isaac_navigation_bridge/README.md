# isaac_navigation_bridge

该 ROS 2 Humble 包是 Isaac Sim/Isaac Lab 与 SCAN Planner 之间的传感器边界。
桥节点本身不运行 PCT、SCAN、轨迹控制器或 RL policy，也不产生速度命令；
组合 launch 默认启动 PCT ROS 2 adapter、`scan_planner`、`scan_controller` 与
`navigation_supervisor`。supervisor 只协调类型化状态与 PCT 重规划事务，不发布
`/cmd_vel`，因此闭环控制器仍是唯一速度发布器。

消息链：

```text
/isaac/body_pose_raw          nav_msgs/msg/Odometry
  → /body_pose                nav_msgs/msg/Odometry

/isaac/cloud_registered_raw   sensor_msgs/msg/PointCloud2
  → /cloud_registered         sensor_msgs/msg/PointCloud2（仅 float32 xyz）

/pct/global_path              nav_msgs/msg/Path（组合 launch 默认，地面高度语义）
  → 仅缓存给坡面支撑廊道过滤，不重发或改写 Path

/pct/planning_status          scan_planner_msgs/msg/PCTPlanningStatus
/planning/scan_status         scan_planner_msgs/msg/ScanPlanningStatus
/planning/controller_status   scan_planner_msgs/msg/ControllerStatus
  → navigation_supervisor
  → /navigation/status        scan_planner_msgs/msg/NavigationStatus
  → Isaac OGN NavigationStatus subscriber
  → 唯一 cmd_vel_to_policy writer 的安全门

/pct/planning_command         scan_planner_msgs/srv/PCTPlanningCommand
  ← navigation_supervisor 的有界幂等 REPLAN 请求
```

`NavigationStatus` 由 Isaac OGN subscriber 可靠、transient-local 地送入现有唯一
`cmd_vel_to_policy` writer。writer 会核对状态新鲜度、`goal_id`、
`active_path_stamp`、`state_revision`、`allow_tracking_command` 与
`force_zero_velocity`；状态无效或要求强制停车时，实际写入 policy command buffer
的是零速度。组合 launch 没有新增 raw topic、第二个 Twist gate 或第二个 policy
writer。`stop_confirmed` 仍是 supervisor 发起 PCT 重规划前的控制器停车证据。
supervisor 不另订阅 goal topic；活动目标身份来自 PCT `COMMAND_PLAN` 状态中的
`active_goal`，随后按 typed status、Path 与轨迹身份推进。

订阅和发布统一使用 SensorData QoS（best effort、volatile、keep last），队列深度由
`qos.depth` 配置。参考 Path topic（组合 launch 默认 `/pct/global_path`，
standalone YAML 仍可配置）单独使用 reliable + transient-local，深度由
`qos.path_depth` 配置。`use_sim_time` 在节点无显式覆盖时和默认 launch/config
中均为 `true`。

## 坐标和时间约束

- OmniGraph 必须把原始点坐标直接写成 `frames.cloud` 所指的重力对齐世界系；
  本节点不做 TF 坐标变换，也不会静默改写 frame。原始 Odometry 和点云 frame
  必须分别匹配 `frames.odom`、`frames.base` 和 `frames.cloud`，且当前
  `frames.cloud` 必须等于 `frames.odom`，否则消息或节点启动会被拒绝。
- Odometry 输出 `frames.odom` 与 `frames.base`；默认分别为 `world` 和
  `base_link`。
- 有效的原始时间戳原样保留。原始时间戳为零时使用节点 ROS 时钟；若 `/clock`
  尚未生效且两者都为零，则丢弃该消息，避免发布无效时间。
- 点云过滤依赖最新有效 Odometry。默认在收到首帧 Odometry 前丢弃点云并节流
  告警；Odometry 接收时间超过 `filters.odom_timeout_sec`，或点云与
  Odometry 时间戳差超过 `filters.max_cloud_odom_skew_sec` 时也会丢弃点云。

## 点云过滤

处理顺序为：移除 NaN/Inf、相对底盘的距离和轴向裁剪、机体系地面窄带、
参考 Path 地面廊道、双圆柱自点。世界点使用 Odometry 的完整单位四元数转换
到 `base_link`，距离、裁剪、当前支撑面和双圆柱包络都与真实 roll/pitch/yaw
一致；只提供 yaw 的路径仅作为纯函数兼容接口，不用于在线 bridge。

地面和 Path 支撑面不再无条件丢弃。它们是真实测得的表面，bridge 以
`ray_endpoint_type=0` 的标量 `uint8` 字段将其发布为显式自由射线端点；
保留的障碍端点使用值 1。GridMap 只沿这些有观测证据的射线降低旧占据，
使移动障碍露出地面后可有界恢复；双圆柱自点会遮挡后方空间，仍完全丢弃，
不得伪装成自由射线。不含该可选字段的非空 xyz 点云仍按普通障碍端点处理，
保持对标准 LiDAR 消息的兼容。

过滤前先执行原始输入门禁。默认
`filters.minimum_valid_input_points=64`；raw cloud 去掉 NaN/Inf 后不足 64 点
（包括空帧）会被丢弃，且不会刷新 SCAN/controller 的点云新鲜度。只有至少
64 个有限原始点经过后续分类后，若障碍端点和可用支撑面自由射线都为空，
才发布“本帧有效、无可用端点”的 canonical empty。该消息必须精确满足：
`height=1`、`width=0`，xyz
`float32` fields 的 offsets 为 `0/4/8`、count 均为 1，`point_step=12`、
`row_step=0`、空 `data`、`is_bigendian=false`、`is_dense=true`。GridMap 和
controller 都会拒绝其他畸形空云，raw 空云不能借此伪装成新鲜观测。

当前机器人下方支撑面在机体系按下式计算：

```text
ground_body_z = -filters.body_height_m
remove_body_z = [ground_body_z - filters.ground_band_down_m,
                 ground_body_z + filters.ground_clearance_m]
```

为避免机器人尚在平地时把前方上升坡面误判成障碍，bridge 还可靠缓存
`initial_path_topic` 指定的参考 Path。Path 的 z 按项目合同表示世界系地面高度。缓存接收合同与
SCAN 一致：顶层和逐 Pose 时间戳必须非零，frame 必须匹配，位置和姿态必须
有限且四元数有效；连续点按
`filters.path_min_point_spacing_m=0.05 m` 去重后仍须至少有两个点。较旧
时间戳不能覆盖新缓存，带合法新时间戳的空 Path 会显式清缓存。启用
`filters.drop_cloud_without_ground_path=true` 时，有效 Path 到达前不会向
SCAN 发布点云，防止尚未过滤的坡面支撑点先写入有状态占据地图。

机器人当前 XYZ 会先投影到有序 Path 的三维线段，以区分 XY 重叠的楼层；
后续进度只增不减。每帧只预计算
`[progress - filters.path_ground_backward_arc_m,
progress + filters.path_ground_forward_arc_m]`（默认后退 `1.0 m`、前向
`3.0 m`）内的局部线段，不遍历整条全局 Path。点云分块投影到这些线段，
只在半径 `filters.path_ground_corridor_radius_m=0.70 m` 内删除期望地面
上下 `0.05 m` 的窄带。Path 时间戳晚于点云时禁用该帧 Path 过滤，Path
代际在转换期间变化时丢弃该帧，避免新路径过滤旧点云。廊道外墙体和高于
窄带的动态障碍主体仍保留。`0.70 m` 覆盖双圆柱半径、SCAN `0.20 m`
碰撞代价距离和点云噪声余量，不代表机器人几何半径。

双圆柱中心位于 `base_link` x 轴的正负
`filters.double_cylinder_offset_m`，圆柱半径和相对 base 的竖直上下界全部可配。
默认值不是标准 Go2 包络。它来自当前
`GO2_X5_PCT_DOG_ONLY_CFG` 固定机械臂导航姿态下的 URDF 碰撞体离线测量：

| 项目 | 相对 `base_link` 测量值 |
| --- | --- |
| 全部碰撞体 x 边界 | `[-0.384, 0.367] m` |
| 全部碰撞体 y 边界 | `[-0.155, 0.155] m` |
| 全部碰撞体 z 边界 | `[-0.354, 0.436] m` |
| 导航初始/目标 base 高度 | `0.338 m` |

配置使用半径 `0.27 m`、中心偏移 `0.16 m` 的双圆柱，以及
`[-0.40, 0.50] m` 的竖直边界，为腿部摆动和仿真误差保留余量。后续若修改
收纳姿态或碰撞体，必须重新测量，并通过动态关节扫掠和真实点云复核，不能退回
标准 Go2 默认值。生产 controller/policy 的 `0.65/0.15/0.60` 是速度上限，
不得误写成双圆柱几何参数；各包单独启动时仍使用自身保守默认值。

## 构建与运行

```bash
cd ros2_ws
colcon build --packages-select isaac_navigation_bridge
source install/setup.bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py
```

运行 PCT → SCAN 主线时，PCT、bridge、SCAN、controller 和 supervisor 直接
共享 `/pct/global_path`，不经过 Path relay：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  start_pct:=true \
  start_manual_path:=false \
  start_supervisor:=true
```

只用手工三维地面 Path 验证
`/pct/global_path -> /planning/bspline -> /cmd_vel` 的 ROS 侧隔离链时：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  start_pct:=false \
  start_manual_path:=true \
  start_supervisor:=false
```

生产组合 launch 默认 `start_pct=true`、`start_supervisor=true`。
`start_pct` 与 `start_manual_path` 不能同时为 `true`，否则 launch 会在创建节点前
明确失败；两者都为 `false` 时可接外部可靠 Path 发布器。手工 Path 的隔离链默认
关闭 supervisor，因为该模式没有完整 PCT goal/status/service 生命周期；默认 policy
gate 会保持 policy command buffer 为零，因此该命令不用于验证 RL policy 执行。若专用
测试图完整提供类型化 PCT 生命周期与 command service，可以显式启用 supervisor。
统一参数
`body_height_m` 默认 `0.338`，会同时覆盖 PCT 的目标落地高度、bridge 地面过滤、
SCAN `grid_map.body_height` 与 controller 参考路径高度，避免重复或不一致加高。

当外部探针、LIO 或其他生产者已经直接发布规范化的
`/body_pose` 与 `/cloud_registered` 时，必须关闭本包的原始观测桥，
避免同一 Topic 出现两组位姿/点云 publisher。例如只验收真实 PCT 到
SCAN B-spline、严格不创建 `/cmd_vel` 发布者：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  start_bridge:=false \
  start_scan:=true \
  start_controller:=false \
  start_manual_path:=false \
  start_pct:=true \
  start_supervisor:=false \
  start_odometry_tf:=false \
  body_height_m:=0.338 \
  pct_backend_kind:=upstream

/usr/bin/python3 \
  src/pct_ros2_adapter/test/probe_real_pct_to_scan_chain.py \
  --planning-only
```

`start_bridge` 默认仍是 `true`：Isaac OGN 发布
`/isaac/body_pose_raw` 和 `/isaac/cloud_registered_raw` 的正常生产链不需要改
参数。`start_bridge:=false` 只表示规范化观测由外部负责，不会关闭
PCT 或 SCAN Planner。

PCT、SCAN Planner 和 SCAN Controller 的性能参数统一放在
`config/pct_scan_tuning.yaml`。组合 launch 会按“各包安全默认配置 → 统一调参
文件 → topic/frame/body height 强制合同”的顺序加载。正常调速只修改这一份
文件，不应在 C++/Python 业务代码里改数值。可以用另一个绝对路径做 A/B：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  tuning_config_file:=/abs/path/to/pct_scan_tuning.yaml
```

参数关系、推荐范围、加速顺序和回退方法见
`docs/pct_scan_tuning.md`。其中 `fsm.reference_cruise_speed` 是非最终局部段的
切向巡航速度；最终段仍固定为零终端速度，不会因为提高巡航速度放宽
`finish.distance_xy=0.08` 的真实到达门。

标准 Go2 + MoE-CTS 的纯物理手工楼梯 A/B 入口彼此独立：

```bash
# 0.40 m/s 巡航、0.45 m/s 上限的保守基线
ros2 launch isaac_navigation_bridge \
  go2_moe_cts_stair_navigation.launch.py

# 0.60 m/s 巡航、0.65 m/s 上限的快速实验
ros2 launch isaac_navigation_bridge \
  go2_moe_cts_stair_fast_navigation.launch.py

# 实际 0.30 m 踏面 + 用户高速运动包络
ros2 launch isaac_navigation_bridge \
  go2_moe_cts_stair_wide_fast_navigation.launch.py
```

快速入口复用完全相同的 33 点楼梯 Path、bridge、SCAN、controller、TF、终点门
和急停合同，只把统一覆盖层替换为
`config/go2_moe_cts_stair_fast_tuning.yaml`。宽踏面入口则改用独立的 25 点
`go2_moe_cts_stair_wide_path.yaml` 和低平台高度 `-1.71m`，并使用
用户选定的 planner 巡航/上限 `1.00m/s`、加速度 `1.50m/s²`、controller
`1.50m/s、2.50m/s²` 高速上限，并只在楼梯前后 `0.45m` Path 窗口启用
楼梯航向锁定：机体沿完整 Path 水平切向，SCAN 横向重接由 `vy` 执行。
同一窗口还把明确为正的前向轨迹补到 `1.00m/s`，抵消精确立面 Path 将
三维速度主要分配到 z、而 Go2 policy 只消费平面命令造成的低速。该牵引门
不覆盖零速、反向、横向恢复、终点制动、急停或输入超时。对应
`manager.feasibility_tolerance` 必须收紧为 `0.0045`，否则会宽于
最终 `0.005m/s、0.01m/s²` 动态采样门并在启动时 fail closed。
Isaac 端必须传 `--stair-width 0.31`：在 `0.10m`
高度场网格上它会离散成 3 格，即实际 `0.30m` 踏面。三种入口都要求
`/cmd_vel` 恰好只有
一个发布者；发现重复 `scan_controller` 时必须先关闭旧 launch 和旧 Isaac，不能
用提高速度掩盖交替零速命令。

`world_frame` 与 `base_frame` 默认分别为 `world`、`base_link`。组合 launch 会用
它们同时覆盖 bridge、PCT、SCAN、controller 和 supervisor；bridge 的 Odometry
与点云 frame 都固定使用同一个 `world_frame`，手工 Path 也跟随该世界 frame。
frame 参数必须使用无前导 `/`、无空白且层级完整的规范写法，二者不能相同。例如：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  start_pct:=true \
  world_frame:=map \
  base_frame:=robot/base_link
```

使用自定义 frame 时，Isaac OmniGraph 或 LIO 发布的原始 Odometry/点云 header
也必须使用相同值；组合 launch 不提供 TF 转换，也不会伪造输入 frame。

`start_bridge`、`start_scan`、`start_controller`、`start_pct`、`start_supervisor`、
`start_manual_path`，ROS 侧主链 topic（body pose、cloud、global path、PCT
goal/status、B-spline、SCAN/controller/navigation status、cmd_vel）、PCT command
service、六个节点参数文件及统一 world/base frame 都可由 launch argument 覆盖。
组合 launch 对所有节点显式设置 `use_sim_time=true`；手工 Path 只在仿真时钟生效后
发布一次。

例如把整条状态与重规划链移入命名空间时，生产者和 supervisor 消费者会由同一组
argument 同步改写：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  pct_status_topic:=/robot/pct/planning_status \
  pct_command_service:=/robot/pct/planning_command \
  scan_status_topic:=/robot/planning/scan_status \
  controller_status_topic:=/robot/planning/controller_status \
  navigation_status_topic:=/robot/navigation/status
```

`navigation_status_topic` 在本 launch 中覆盖 supervisor 的 ROS publisher；Isaac OGN
consumer 不属于该 launch。当前完整 Isaac pipeline 使用默认 `/navigation/status`，生产
主线必须保留该默认值。仅在 ROS 侧隔离测试中自定义它，或确保外部 OGN consumer 也
独立配置为完全相同的 topic；否则 policy gate 会因缺失状态而安全写零。

supervisor 的超时、重试次数、状态 QoS 与 `status.heartbeat_sec=0.10` 默认来自
`navigation_supervisor/config/navigation_supervisor.yaml`，可通过
`supervisor_config_file:=/abs/path/to/navigation_supervisor.yaml` 覆盖；上述
topic、service 与 frame launch argument 始终作为最终覆盖层。0.10 秒 heartbeat
小于 policy gate 默认 0.25 秒的新鲜度上限，稳定状态也会持续刷新，避免误停车。

配置文件安装自 `config/pct_scan.yaml`。可用自定义文件覆盖：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  config_file:=/abs/path/to/pct_scan.yaml
```

最终连续斜坡 v24 的 105 个到期点云周期中有 96 帧成功发布；最后有效帧在
`9.62 s` 含 328 点，后续 9 帧因原始有限点不足被拒绝并保持
`point_cloud_timeout` 安全零速，证明空/稀疏 raw 输入没有刷新 freshness。

## 测试

```bash
/usr/bin/python3 -m pytest -q \
  ros2_ws/src/isaac_navigation_bridge/test
```

测试覆盖完整四元数世界系到机体系几何、坡面 Path 的单调进度与局部地面
廊道、Path 到达前点云门禁、XY 重叠楼层、30k 点长路径性能、双圆柱过滤、
Path 代际/时间门控、Odometry/PointCloud2 规范化、空/稀疏/非有限 raw 拒绝、
足量 raw 分类后的 canonical empty、支撑面 free-ray 及畸形 empty 拒绝，以及 SensorData 与
可靠缓存 QoS、PCT/手工 Path 互斥门禁、统一 world/base frame、supervisor
默认启用、参数文件 heartbeat、生产者/消费者绑定和默认 `/pct/global_path`、
状态、重规划 service 合同。
