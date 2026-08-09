# PCT + SCAN 跨楼层导航重构

## 当前发布状态

截至 2026-08-09，默认组合 launch 加载 0.60 生产调参文件。相同代码、参数、
原 Go2-X5 checkpoint 和 146 点 PCT Path 的跨楼层携物导航 seeds 0、1、2 已严格
通过 `3/3`；一次生产完整 pipeline 也已完成
`nav → pick → carry nav → place → LeRobot export → done`。到达验收包含位置、
航向、真实速度、连续驻留以及目标后的持续零速，不是“进入宽松半径即成功”。

稳定结论仅覆盖静态场景，并包含用户确认的楼梯 `chassis_root_lock`：楼梯内冻结
底盘，沿绑定到同一 PCT Path 身份的楼梯段切换楼层，离开后恢复 SCAN 闭环。因此
该链可以作为当前工程发布主线，但不能表述为纯物理爬楼。移动推车绕障和 live PCT
重规划尚未纳入本次验收；DWA 对比已停止，本项目也不声称 SCAN 算法快于 DWA。
0.75 偏航候选没有进入生产默认值。

## 目标链路

本分支只保留以下在线导航链：

```text
PCT 层析地图
-> PCT ROS 2 全局路径
-> SCAN ROS 2 局部三维轨迹
-> 闭环轨迹跟踪
-> cmd_vel
-> Go2-X5 RL locomotion policy
-> Isaac Sim / Isaac Lab
```

旧的 PCT + DWA 与 stair-float 实现只作为历史诊断证据，不是本分支的运行
fallback。当前 `scan_stair_freeze` 是另一条受控工程路径：它只接受 live ROS 2
地面 Path（生产主线为 `/pct/global_path`，手工 smoke 为 `/initial_path`），并用
Path 点列哈希和显式索引绑定楼梯段；触发后冻结导航
root/支撑关节并沿同一三维 Path 推进。该路径必须标记为非物理 root-lock
workaround，不能写成 RL policy 纯物理爬楼成功。抓取、放置和 LeRobot 导出
仍复用原 pipeline，导航模块不能拥有机械臂或数据导出的状态机职责。

## 上游 SCAN 版本

本地 `external/SCAN-Planner` 当前检出 `main`，该分支仍是 ROS 1/catkin。
同一个 clone 已包含远端引用 `origin/ros2-community`，提交为
`d0b921c9b05a6d291d144d60882b2e0e88d2c0e0`。ROS 2 消息、topic 和后续
算法移植以该引用为兼容基准；集成过程不切换或改写用户的 external 工作树。

## ROS 2 与 Isaac Sim 的进程边界

宿主 ROS 2 Humble 使用系统 Python 3.10，Isaac Sim 5.1 使用独立 Python
运行时。不能把 `/opt/ros/humble` 的 Python 3.10 `rclpy` 扩展直接加载到
Isaac Python，也不能用 JSON、文件轮询或 stdin/stdout 绕过 ROS 2。

Isaac 侧使用自带的 `isaacsim.ros2.bridge` OmniGraph 节点发布原始仿真真值：


| Topic                          | 类型                           | 说明                 |
| ------------------------------ | ------------------------------ | -------------------- |
| `/clock`                       | `rosgraph_msgs/msg/Clock`       | 仿真时钟             |
| `/isaac/body_pose_raw`         | `nav_msgs/msg/Odometry`         | root 位姿和速度       |
| `/isaac/cloud_registered_raw`  | `sensor_msgs/msg/PointCloud2`   | 仿真 ground-truth 点云 |
| `/pct/goal`                    | `geometry_msgs/msg/PoseStamped` | pipeline base 目标输出 |
| `/pct/global_path`             | `nav_msgs/msg/Path`             | PCT 地面 Path 输入     |
| `/cmd_vel`                     | `geometry_msgs/msg/Twist`       | SCAN 最终机体系速度输入 |
| `/planning/goal_reached`       | `std_msgs/msg/Bool`             | 最终轨迹到达锁存输入   |
| `/planning/controller_status`  | `scan_planner_msgs/msg/ControllerStatus` | SCAN 轨迹代际与控制状态 |

`source/navigation/isaac_ros2_ogn_bridge.py` 提供 Isaac Sim 5.1 的图构造和发布
边界。默认 direct 模式直接使用 Isaac Lab tensor：
`root_pos_w`、`root_quat_w`、`root_lin_vel_b`、`root_ang_vel_b`；其中
WXYZ 四元数会规范化并转换为 OGN 的 XYZW，机体系速度配合
`publishRawVelocities=true`。状态/时钟与点云分别使用独立
`OnImpulseEvent`，避免把低频点云绑定到 400 Hz physics tick。
`/cmd_vel` 使用专用 Twist subscriber；完成状态使用 generic
`ROS2Subscriber(std_msgs/msg/Bool)`，二者各自拥有 impulse 和接收计数。
`/planning/goal_reached` 订阅采用 reliable + volatile，拒绝把上轮
transient-local 的旧 true 当成本轮到达。`/pct/goal` 使用 generic
`ROS2Publisher(PoseStamped)` 以 reliable + volatile 发布；参考 Path 使用
reliable + transient-local generic subscriber。`/navigation/status` 使用
`scan_planner_msgs/NavigationStatus` generic subscriber，并以 reliable +
transient-local KeepLast(1) 接收 supervisor 的 policy 执行许可。

该模块必须在 `SimulationApp` 建立、`isaacsim.ros2.bridge` extension 启用且
timeline 播放后调用 `setup()`。每个控制周期调用 `update_odometry()`，只有
获得一帧已经变换到 `world` 的有限 N×3 `float32` 点云时才调用
`update_point_cloud(points, timestamp=...)`。direct 模式下两者必须使用同一个
跨 episode 连续仿真时钟，不能传每次 episode 清零的
`SimulationState.timestamp`。frame 只是消息标签，不会执行传感器系到世界系
变换。

Isaac Python 3.11 不能加载 `/opt/ros/humble` 的 Python 3.10 扩展。启动
SimulationApp 前必须先固定 Isaac 解释器，再 source 系统 ROS 和本工作区以
获得自定义消息的 `AMENT_PREFIX_PATH/LD_LIBRARY_PATH`，最后清除
`PYTHONPATH`：

```zsh
export ISAAC_PYTHON="$(command -v python)"
source /opt/ros/humble/setup.zsh
source /mnt/sage_data/workspace/pct_scan/ros2_ws/install/setup.zsh
export ROS_DOMAIN_ID=189
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset PYTHONPATH
"$ISAAC_PYTHON" -B scripts/pipeline/run_full_physics_pipeline.py ...
```

ROS launch 终端使用相同的 `ROS_DOMAIN_ID/RMW_IMPLEMENTATION`，但保留 ROS
Python 3.10 的 `PYTHONPATH`。pipeline 会在 AppLauncher 前检查
`scan_planner_msgs/ControllerStatus` 的 ament interface resource，以及
generator C、typesupport C、introspection C 三个共享库；不能在已启动的
Python 进程内补 source。图驱动器还会拒绝未启用 ROS 2 bridge extension、
未播放 timeline、被新 stage 删除的 graph，以及回退的 direct 时间戳。

若覆盖组合 launch 的 `navigation_status_topic`、`world_frame` 或
`base_frame`，Isaac pipeline 必须分别传入完全相同的
`--ros2-navigation-status-topic`、`--ros2-world-frame` 和
`--ros2-base-frame`。只改 launch 一侧会让执行许可身份或 frame 校验失败，
唯一 policy writer 将持续安全写零。

现有 `IsaacLabNavigationRuntime` 已提供成组启用的
`ros2_ogn_bridge_config`、`depth_point_cloud_config` 与
`cmd_vel_to_policy_config`。启用后，它会：

- 创建前视 D436 的 `distance_to_image_plane` RTX 深度输出；
- 打开 `update_latest_camera_pose`，使用 `quat_w_ros` 把 ROS 光学系深度
  反投影为 `world` 点云；
- 生产配置在 CUDA 张量上按 8 像素抽样，单帧硬上限为 1.2 万点，再转换为连续
  `float32`；
- 原始有限点少于 `minimum_valid_points`（默认 `64`，包括空帧）时不发布、
  也不刷新点云新鲜度，等待 controller 与 policy 安全门按超时停车；
- 每个完整控制步发布一次 Odometry，每 10 个控制步发布一次新鲜点云；
- 用成功执行的 physics-step 总数乘 `physics_dt` 生成跨 episode 连续时钟；
- stage reuse 时保留 OGN graph 与时间门禁，只重绑属性 helper。
- 对每个 pipeline 导航 generation 只发布一次带 base 高度的 `/pct/goal`，并把
  实际发布 stamp/sequence 回报给执行器；手工 Path 模式显式关闭该 publisher；
- 轮询 `/pct/global_path`，把合法空 Path 作为旧代清除事件，把两点及以上的
  非空 Path 作为新代参考路径；一 Pose Path、零时间戳和非法 frame 均失败关闭；
- 轮询 OGN `ROS2SubscribeTwist` 的接收计数，把每条新的 `/cmd_vel` 交给
  policy 安全门；即使连续消息数值相同，也不会误判为缓存旧值。
- 分开记录“OGN 已观察到的 NavigationStatus”和“policy gate 已消费的
  permit”，并在 episode summary 累计 identity-valid 状态与实际放行次数；
  楼梯全程冻结时前者必须存在，后者允许为零且必须由冻结零速原因解释。
- 轮询 `/planning/goal_reached` 的 Bool 接收计数，并为每次 policy 实写和
  goal 样本记录独立递增序号，供 episode 生命周期验收。

CLI 的导航类模式默认并强制启用 ROS 2 bridge。它同时启用传感器发布、
`/cmd_vel` 订阅和 policy 命令所有权，不再只是可选诊断开关；旧 pipeline
动作中的 `base_velocity` 被明确旁路。只有隔离的固定速度楼梯低层探针允许关闭
bridge，主命令会拒绝 A*、PCT→A* fallback 或 `--no-enable-navigation-ros2-bridge`。
policy 安全门负责 `vx/vy/wz` 幅值与
变化率限制、Twist/Odometry/点云超时零速、clock rewind 清空以及进程内独占
写入。Go2-X5 携臂收纳姿态当前保守速度上限为
  `0.65/0.15/0.60 m/s、m/s、rad/s`，变化率上限为
  `1.20/0.40/1.50 m/s²、m/s²、rad/s²`。在线模式禁用 policy 后置的
`0.08` 站立死区，避免限速后的微小命令跨过死区时产生速度阶跃。
主路径直接选择 `ScanRos2NavExecutor`，不会构造或接入 PCT+DWA 运行链；该
执行器只管理完成生命周期，不生成底盘速度，因此在线速度权仍唯一属于
`scan_controller -> cmd_vel -> policy`。DWA 类只保留为隔离历史单测基线。

PCT checkpoint 训练时的六自由度机械臂默认偏置为
`[0.0, 0.3, 0.5, 0.0, 0.0, 0.0]`。导航阶段会显式保持这一收纳姿态；未提供
外部机械臂目标时，policy adapter 保留 command manager 的训练默认值，不再
用全零覆盖它。PCT DogOnly 环境还采用一项仅作用于该配置的 PhysX A/B：
`solver_velocity_iteration_count=1`，并启用
`enable_external_forces_every_iteration=true`；位置迭代、时间步、decimation
和摩擦参数均保持原值，不能把这项试验扩散到其他任务。

运行时必须显式使用 `--navigation-visual-mode collision`，让 RTX 深度能看到
静态碰撞场景。topic、domain、深度量程、抽样步长、最小有效点数和发布间隔
均可配置。episode epoch/reset/ack 协议尚未建立，因此该入口当前强制
`--num-episodes 1`，避免新 episode 接受上一 episode 的旧 B-spline。RGB 与
点云发布周期不整除时，运行时使用两者最大公约数作为渲染节拍。

前视深度能覆盖相机视野内的移动刚体，但不是 360° LiDAR。Liangzhu 平地和
最终坡道 v24 均已用在线点云驱动 SCAN 规划与滚动重规划；坡道已经验收，
楼梯踏面和动态障碍的可见性及支撑面分类仍未验收。full/NuRec 模式会隐藏
碰撞网格，且当前 8 GB GPU 已知不适合同时运行高质量视觉和 IsaacLab RTX
sensor。

ROS 侧 `isaac_navigation_bridge` 只做接口归一化与安全过滤，输出
`/body_pose` 和 `/cloud_registered`。两个输出均使用 SensorData QoS；
topic、frame 和过滤范围全部由参数配置。点云发布给 SCAN 前必须移除机器人
自身双圆柱包络内的点，并滤掉地面薄层，不能把整片地面写成占据障碍。

bridge 还执行独立的原始输入门禁：有限 xyz 少于
`filters.minimum_valid_input_points=64` 的 raw cloud 直接丢弃，不刷新任何
下游新鲜度。至少 64 个有限原始点进入几何分类后，地面和 Path 支撑面会以
`uint8 ray_endpoint_type=0` 保留为显式自由射线，障碍端点值为 1；机器人自点
仍完全丢弃，不能用来清除被机身遮挡的空间。只有障碍端点和可用自由射线
都为空，才发布表示“本帧已观测但没有可用端点”的 canonical empty。其布局必须
精确为 `height=1`、`width=0`、xyz `float32` offsets `0/4/8`、
`point_step=12`、`row_step=0`、空 `data`、`is_bigendian=false`、
`is_dense=true`；GridMap 和 controller 只接受这一布局，其他畸形空云均拒绝。

GridMap 不使用全图时间衰减：移出滑动窗口的环形缓冲会重置，被后续射线
穿过的体素执行 `p_miss` 更新，canonical empty 则只刷新观测生命期。主配置
`p_max=0.98`/`p_miss=0.30`/`p_occ=0.80` 下，饱和旧体素最多被明确
free ray 穿过 3 帧即恢复非占据；未被实际观测的静态区域不会仅因超时被删除。

当前 Go2-X5 DogOnly 固定收纳姿态的 URDF 碰撞体相对 `base_link` 离线边界
为 x=`[-0.384, 0.367] m`、y=`[-0.155, 0.155] m`、
z=`[-0.354, 0.436] m`。桥接默认使用 `body_height=0.30 m`、半径
`0.27 m`、中心偏移 `0.16 m`、竖直范围 `[-0.40, 0.50] m` 的双圆柱；
它包含静态碰撞体并留有余量。坡面支撑过滤廊道半径为 `0.70 m`，覆盖双圆柱、
SCAN `0.20 m` 碰撞代价距离和点云噪声余量。几何包络仍需在 Isaac 动态关节
扫掠和实点云中复核；它与上一段的速度包络是两组不同参数。

GridMap 中 `obstacles_inflation_z_up` 的查询方向对应机器人相对中心的下界，
`obstacles_inflation_z_down` 对应上界；不能按参数英文名直接复制机器人包络。
因此上述 `[-0.40,+0.50] m` 必须配置为 `up=0.40 m、down=0.50 m`。旧值写反
会在非对称边界处产生错误占据判断，现已修正 planner YAML、marker 中心，
并用上、下边界内外四点回归锁定该映射。

## 高度语义

PCT 全局 `Path` 的 `pose.position.z` 始终表示地面高度。后续 SCAN path
adapter 只能在一个位置增加 `body_height`，把地面路径转换为 base 路径。
Isaac bridge 发布的 Odometry 则始终表示真实 `base_link` 高度，两者不能
混用或重复加高。

## 当前阶段边界

阶段 1 建立 ROS 2 workspace、`scan_planner_msgs`、Isaac OmniGraph 发布
边界、ROS 侧过滤节点、统一参数和 launch。当前已完成可构建的消息包、
ROS 侧桥节点，以及前视 RTX 深度到世界系点云的可选 runtime 接入。相关坐标、
连续时间、旧帧拒绝和 stage-reuse 生命周期已有轻量测试。真实时间同步经历了
三次迭代：首次 smoke 暴露 0.24–0.40 秒 cloud/pose 时差告警；只按历史最近
位姿匹配的中间方案复跑后仍有 0.24–0.38 秒告警；最终在 `plan_env` 中让
Odometry 与 PointCloud2 都使用 SensorData QoS，并用容量 100 的
`message_filters` ApproximateTime 双队列配对，同时保留 0.20 秒严格同步
门限。最终 Liangzhu collision 真实复跑没有再出现超限时差告警。

阶段 2 已把 ROS 2 community 的六包 planner-only 依赖链移入当前工作区：
`plan_env`、`path_searching`、`bspline_opt`、`traj_utils`、`scan_planner`
和消息包。`external/SCAN-Planner` 仍保持 `main` 且干净。项目加固包括：

- 配置的参考 Path（生产 `/pct/global_path`，手工 `/initial_path`）使用 reliable
  + transient-local，并严格校验 `world` frame、非零时间戳、有限位姿、单位
  可归一化四元数和最小相邻点距；
- bridge 启用 Path 地面过滤时，在有效参考 Path 到达前不向 SCAN
  发布点云，防止未过滤的坡面支撑点先写入有状态占据地图；
- Path 先于传感器到达时只缓存，直到新鲜 Odometry 与首帧已融合点云地图就绪；
- Odometry 与 PointCloud2 使用 SensorData QoS，并由容量 100 的
  `message_filters` ApproximateTime 双队列配对；frame、时间戳、0.20 秒
  同步差和 0.50 秒新鲜度仍是硬门禁；
- Isaac direct Odometry 的 base-frame 线速度先旋转到 `world`，再交给 SCAN；
- 规划/安全 FSM 改用 ROS time timer；仿真时间暂停时不会继续推进轨迹时间；
- PCT/手工 Path 的 z 保持地面高度，`body_height=0.30 m` 只在 SCAN
  入口增加一次；Liangzhu 直线和转弯配置的 z 均来自 collision PLY 支撑面
  探测，不再用固定 `z=0` 伪造地面。
- 每条正常或急停 B-spline 都携带生成它的 `reference_path_stamp`；controller
  只接受与当前完整参考 Path 同代的轨迹。只有同 stamp、同 payload 的重发保持
  幂等；stamp 增大时即使几何相同也是新代，同 stamp 不同 payload 会使该代
  失败关闭，合法空 Path tombstone 会显式清除当前参考和轨迹。
- 生产参考路径局部前视设为 `1.20 m`，执行约 `0.55 m` 后滚动重规划；接近终点
  时先由当前轨迹继续收敛。若轨迹耗尽时仍未进入 XY `0.08 m` 真到达门，
  reference 模式继续生成短 final B-spline；community 非 reference 模式仍保留
  原有 `<0.20 m` 拒绝行为，短轨迹也不会跳过碰撞、动态可行性或有序走廊检查。
- 当前起点先投影到全局参考轨迹，再按轨迹累计弧长选择局部目标；局部重规划
  只替换 B-spline，不把剩余参考路径重建为当前点到终点的直线，避免在 90°
  转弯、U 形路径和楼梯回转处抄近路。
- reference-path 模式不再用初始局部占据窗口裁剪完整 Path 终点；局部占据只
  影响当前局部目标。规划失败重试按 `0.50 s` 仿真时间节流，避免在同一仿真
  时刻形成毫秒级重规划风暴。
- 全局参考轨迹导出的非最终局部目标以 `0.60 m/s` 巡航，并受 planner
  `max_vel=0.65 m/s` 硬限幅，
  末端制动距离计算同时防止零加速度除法。

新增 `scan_navigation_tools` 发布一次可靠、可缓存的手工三维 Path。使用合成
`/clock`、Odometry 和世界系 PointCloud2 的 ROS 2 探针已经实际收到
`/planning/bspline`（19 个控制点、三阶）。该探针是不依赖 GPU 的确定性
ROS 2 回归，用于证明 `Path -> SCAN -> Bspline` 消息与规划接口可工作。

阶段 2 的执行半段也已落地：

- `scan_controller` 订阅完整有序 `/initial_path`、B-spline、Odometry 与
  点云，以 ROS 仿真时钟闭环输出机体系 `/cmd_vel`；
- 缺 Path、有效同代 B-spline、Odometry 或点云，以及 Odometry/点云超时，
  均严格输出全零并冻结 SCAN 轨迹时间；yaw 偏差过大时只允许原地
  受限转向，轨迹执行时间和软失效时刻同步冻结；冻结最多顺延 12.0 秒，
  超过硬截止仍安全停车，但此时 `execution_frozen=false`，让 planner 对仍有效
  Path 发起重规划而不是继续延长已失效轨迹；
- controller 对当前 Odometry 先减一次 `body_height`，再投影到完整 Path：
  首帧做全路径三维投影，后续只搜索单调进度后退 `1.0 m`、前向 `3.0 m`
  的有序弧长窗口，避免 XY 重叠楼层串层。横向偏差超过 `0.12 m` 时启用严格
  航向门，回到 `0.08 m` 内才释放；局部 B-spline 从当前 Odometry 重规划
  不会重置该全局偏差。
- 生产 `vx/vy/wz` 上限为 `0.65/0.15/0.60`，变化率上限为
  `1.20/0.40/1.50`；
- moving final 首次进入严格 XY `0.055 m` 捕获内门后，controller 连续 `0.06 s`
  发布三轴全零，确保 policy adapter 的独立 slew history 同步清除；随后 terminal
  yaw 只允许 `0.45 rad/s`、`1.00 rad/s²`。控制反馈使用独立 `0.18 rad`
  死区，严格窄于 `0.20 rad` 完成门；因此误差进入验收门后仍会继续收敛到
  更深余量，完成认证阈值本身不变。stationary final hold 不进入该反馈分支，
  仍全程严格零速。物理完成仍使用 `0.08 m`；生产 `0.30 m` 只是 terminal mode
  外圈，不参与完成判定；捕获内外圈之间姿态/速度已稳定时只恢复 SCAN B-spline 的
  XY 闭环，仍锁定完整 Path terminal yaw，避免死区和局部弦航向反复切换；
- moving final 提前捕获后可由严格位置、高度、terminal yaw 与完整运动速度门连续
  `0.50 s` 完成，不再等待被冻结的 B-spline 名义时间；首次捕获的 hard expiry
  固定，同 Path 滚动 final 不能刷新。点云或 Odometry 超时会清零连续驻留，
  但在固定 hard expiry 前保留捕获并持续全零；
- SCAN 轨迹优化速度/加速度上限为 `0.65 m/s`、`1.20 m/s²`，不超过
  controller/policy 实写包络；最终轨迹另有 3.0 秒 RL policy 收敛余量；
- `Bspline.emergency_stop=true` 会锁存严格零速，不能再把六个重合控制点
  当作普通位置轨迹重新发出速度；只有同代正常 B-spline 或合法新 Path 才能
  清除急停锁存；
- `Bspline.is_final` 区分局部分段完成与全局目标到达。正常点云下仍要求名义
  B-spline 时间完成；但真实 Odometry 进入完整 Path 的终点位置门后，controller
  会在进入 `0.055 m` 捕获内门后把平移命令清零并原地收敛 terminal yaw。
  物理完成仍使用 XY `0.08 m`，terminal mode 在 `0.12 m` 外圈释放；捕获首拍
  清除上一段 `wz` 历史，终点转向命令另限为 `0.25 rad/s、0.50 rad/s²`。
  漂出 `0.055 m` 但仍在 `0.12 m` 内且姿态/速度已稳定时仅恢复 B-spline XY，
  仍锁定 terminal yaw；漂出 `0.12 m` 才退出终端
  模式。该捕获不放宽任何到达门。只有最终轨迹、点云和 Odometry 新鲜、
  轨迹仍有效，并同时满足终点 XY `0.08 m`、Z `0.12 m`、yaw `0.20 rad`、
  平面/垂向速度 `0.05 m/s` 和机体系 `|wz| <= 0.10 rad/s` 时，才可由连续
  `0.50 s` 驻留提前锁存 `GOAL_REACHED`。捕获前的运动 final 仍用三轴
  角速度范数，只有已经锁住平移的终点驻留忽略无法由 `cmd_vel` 消除的
  roll/pitch 站立微摆；该分支始终只出零速；
- planner 在新鲜 Odometry、地图和完整 Path 终点门都满足后发布常量
  stationary final hold。controller 对该 hold 使用完整 `/initial_path` 原始末点，
  只增加一次 `body_height`，并从最后一个非退化 XY 段取得终端 yaw；从收到
  hold 起全程输出精确零，并要求约 `3.0 s` 连续稳定驻留。输入超时、位姿、
  yaw、平面/垂向速度或机体系 `|wz| > 0.10 rad/s` 都会清空驻留计时；
  stationary hold 不用三轴角速度范数判断导航转向，避免 roll/pitch 站立微摆
  造成假失败，并继续使用有界硬失效时间，不会在楼梯分阶段释放期间被普通
  软失效提前丢弃。planner 在 hold 发布后保留目标和完整 Path，不重复刷新
  identity；精确匹配该 hold 的 ControllerStatus 为 `GOAL_REACHED` 时才清目标，
  为 `TRAJECTORY_TIMEOUT` 时则沿同代 Path 发布严格更高 `traj_id` 的恢复轨迹；
- 长轨迹只在接收时检查 Header 新鲜度，运行有效期至少覆盖
  `start_time + duration`，不会因 Header 年龄误停车；
- SCAN 激活参考路径时保留全部有序点；默认手工 Path 与机器人同起点由真实
  Odometry 的三维弧长投影处理，再从认证进度向前构造局部 guide；
- 组合 ROS 2 探针已实际打通
  `Path -> SCAN -> Bspline -> scan_controller -> cmd_vel`：默认同起点用例
  生成三阶、19 控制点、5.055328 秒的最终 B-spline，186 条速度消息中
  56 条非零，当时的最大 `|vx|=0.55 m/s`；停发点云后 0.520019 仿真秒归零。
  这些是旧速度上限下的历史观测；当前回归不把 `0.55 m/s` 当作现配置期望；
- `/cmd_vel` 的 ROS 图审计显示发布者数量为 1，唯一发布节点为
  `/scan_controller`。Isaac runtime 内的安全门是 policy command buffer
  的唯一写入者。
- `ScanRos2NavExecutor` 每次 reset 必须按顺序看到本轮 fresh false、有效
  非零 policy 实写、fresh true，以及 true 接收时间之后连续 5 个
  write-sequence、控制 step 和时间戳都连续递增的零速实写。正常零速和唯一
  原因为 `point_cloud_timeout` 的终点安全零速可计数；环境终止、预测碰撞、
  时钟回退、控制租约失效或混合原因均不能冒充成功。同一 observation 重复
  检查保持幂等。
- `scan_stair_freeze` 特例不伪造非零 policy activity：执行证据改为
  `certified_root_lock_progress_seen`，但仍必须看到本轮 fresh false、release
  后的新鲜同 owner 写入、fresh true 上升沿以及其后连续 5 次零速实写。
  summary 必须同时暴露 root-lock/direct-state 使用情况并把纯物理成功置为
  false。
- `--scan-manual-path-goal-xyyaw X Y YAW` 只允许用于
  `navigation-smoke + ROS 2 bridge`，使 pipeline 的 XY/yaw 验证目标与外部
  `/initial_path` 终点一致；它不生成 Path、不切换 planner，原任务的 base z、
  floor 和 slice 均保持不变，来源记录在 `raw_task.runtime_override`。

专用生命周期接入后的首轮真实复跑暴露了确定性缺陷：3.124 秒轨迹仍使用
0.25 秒末端余量，约 90° 的 `ALIGNING_YAW` 只冻结执行时间、没有顺延绝对
失效时刻，最终在仿真 `8.10 s` 精确进入 `TRAJECTORY_TIMEOUT`。修复为软/硬
双截止并降低 SCAN 速度上限后，第二轮 Liangzhu collision 单 episode 已完成
真实平地到达：489 帧、9.76 仿真秒，发布 489 条 Odometry 和 97 条有效点云，
所有 97 个到期点云周期均成功发布。controller 进入 `GOAL_REACHED`；
executor 观察到 goal sequence `474:false -> 475:true`，允许变化率限制消化
2 次残余非零写入后，验证连续 5 次 policy 零速实写。最终 XY 目标误差
`0.0852 m`、base 高度误差 `0.0968 m`、yaw 误差 `0.00354 rad`，线/角速度
分别为 `0.0202 m/s`、`0.0272 rad/s`。pipeline 底盘 action 全程为零，实际
位移 `0.9435 m` 来自 SCAN `/cmd_vel`；没有 teleport 或直接关节状态写入。

随后使用五点 L 形 Path 做真实 90° 平地转弯。诊断 v2 虽然结束在终点附近，
但实际轨迹有 `99.564%` 的长度沿起终点对角线，距折角最近 `0.4076 m`，
末端 yaw 仅 `0.8359 rad`，因此明确判为抄近路而非转弯通过。根因是 1.2 m
路径短于旧局部视距，且参考路径滚动重规划会把剩余路径改写成直达终点。
缩短前视并保留全局参考路径后，v3 首次正确选择中间目标，但手工 Path 的
错误 `z=0` 和未限幅目标速度使加速度连续 5 次达到
`1.546–1.830 m/s²`，超过 `1.500 m/s²` 可行性门限并进入急停。

修正支撑面高度和目标速度后，v4 真实运行通过：SCAN 依次发布 4 条局部
B-spline，其 `is_final` 为 `false、false、true、true`；实际轨迹距折角最近
`0.03690 m`，yaw 从 `-0.00019 rad` 转到 `1.49727 rad`，净左转
`1.49746 rad`。运行共 524 帧，
发布 523 条 Odometry、104 条有效点云并执行 4184 个连续物理步。executor
观察到 `508:false@10.22 s -> 509:true@10.24 s`，允许变化率限制消化
3 次残余非零写入后验证连续 5 次有效 policy 零速实写。过程中一次 A* 搜索
未找到路径，但 rebound optimizer 恢复并生成有效轨迹，没有触发连续失败或
急停；退出时 bridge、planner、manual publisher 和 controller 四个进程均
干净结束。pipeline 的 `stable_physics_success=false` 和
`pure_physics_success=false` 仍是 navigation-smoke 对完整抓放流水线的旧
摘要字段，不表示本次严格 executor 导航验收失败。变化率限制的残余写入使
summary 最后一帧距手工终点 XY 为 `0.09153 m`、yaw 误差为 `0.07353 rad`；
pipeline 展示的 `0.18 m / yaw_alignment_required=false` 仍是旧 verifier
字段，不能把它写成最后一帧通过 controller `0.08 m` 几何门限。真实完成
依据是 controller 在更早样本锁存 `GOAL_REACHED`，以及 executor 随后验证
本轮事件顺序和连续 5 次 policy 零速实写。

连续斜坡历史 build 由 v24 使用同一在线链真实通过。验收场景包含 `0.50 m` 平地
接近、`1.20 m` 水平长度与 `0.18 m` 高差的 `8.53°` 坡面，以及 `0.50 m`
顶平台；Path 的 z 全程保持地面高度。早期资产把封闭实体写成 n-gon，Isaac
Lab RayCaster 却按连续三个索引解释三角形，因而制造了不存在的碰撞孔。
visual/collision 网格现统一为 20 个显式三角形；回归同时验证无退化、闭合
反向配对边、`1.2768 m³` 体积和 `14.449479847951 m²` 表面积。

v24 共记录 527 帧、`10.52 s` 仿真时间、526 条 Odometry、96 条有效点云和
4208 个连续物理步，SCAN 依次生成 7 条局部 B-spline。105 个到期点云周期中
后 9 帧因原始有限点不足被拒绝，最后有效点云在 `9.62 s` 发布；安全门随后
保持唯一原因 `point_cloud_timeout` 的严格零速。当时版本的 controller 在
新鲜 Odometry 同时满足终点位置、航向、平面/垂向/角速度门限后，于
`514:false@10.34 s -> 515:true@10.36 s` 锁存完成。锁存样本的 XY/Z/yaw
误差分别为 `0.03256 m`、`0.03881 m`、`0.03106 rad`，平面、垂向和角速度
分别为 `0.02086 m/s`、`0.01005 m/s`、`0.02453 rad/s`。executor 不计与
true 同时刻的零速，只接受 write `519..523@10.38..10.46 s` 五个连续
step；没有 post-goal 非零写入。summary 为 `success=true`、
`physical_navigation_success=true`、`execution_provenance_verified=true`，
最终 XY/Z/yaw 误差为 `0.03350 m`、`0.03934 m`、`0.03196 rad`，且未使用
teleport、直接 joint-state、base lock 或第二路 pipeline 底盘动作。
该运行早于严格 Path/B-spline identity、任意 yaw 包络、全剩余轨迹碰撞和
有序 corridor 加固，是历史物理证据，不替代最终安全 build 的 live Isaac 复跑。

### 楼梯诊断与低层隔离

单段楼梯的 two-step 过程探针已经走通
`Path -> SCAN -> Bspline -> cmd_vel -> policy -> PhysX`，但未达到终点。
机器人在首阶附近继续收到非零速度请求，却没有形成足够净位移。为避免旧的
100 秒 pipeline timeout 掩盖真实物理停滞，`ScanRos2NavExecutor` 新增固定
窗口进展看门狗：请求并实际写入 `vx>=0.05 m/s`、写入序列连续时，若连续
4.0 秒净 XY 位移不足 0.03 m，即报告 `locomotion_stall`。真实 v4 在仿真
`14.56 s` 以 `0.02648 m` 位移触发；下一状态机动作携带急停原因，runtime
在同一 tick 忽略仍缓存的非零 `/cmd_vel`，将 policy 写入 `[0,0,0]` 并锁存
急停。

two-step 的 6 点 Path 是完整 first-flight 21 点 Path 的严格前缀。短 Path
终点停在机器人中心，前圆柱已经覆盖下一踏面；缺少终点后的 Path 支撑语义会
留下一个真实踏面点并把目标判为 occupied。这不是 Path z 重复增加
`body_height`。two-step 因而只用于首两级过程诊断，正式楼梯到达使用
`tasks/nav_smoke_scan_multifloor_stair_first_flight.json` 及对应完整首段 Path。
本轮没有新增异步 support topic；若未来确需独立支撑路径，必须先定义与
`/initial_path` 的同时间戳配对、严格前缀和失配停车合同。

为隔离低层策略，又加入不创建 PCT/SCAN、不启用 Float 或 base lock 的固定
机体系速度 A/B。三档命令均令名义距离为 0.96 m：

| `vx` | 驱动时长 | 控制 tick | 纵向位移 | 横向位移 | `z` 增量 | yaw 漂移 |
| ---: | -------: | --------: | -------: | -------: | -------: | -------: |
| 0.20 m/s | 4.80 s | 240 | 0.05446 m | 0.00212 m | 0.04547 m | 0.08294 rad |
| 0.25 m/s | 3.84 s | 192 | 0.20880 m | 0.03941 m | 0.08192 m | 0.53797 rad |
| 0.30 m/s | 3.20 s | 160 | 0.21329 m | 0.05736 m | 0.06880 m | 0.60764 rad |

三组命令区间内均严格为 `(vx,0,0)`，截止后的第一条状态转换动作均为零，
但都未完成 0.96 m 或跨过单段楼梯。这三次单次 rollout 强烈缩小了问题范围，
却还不能单独证明 checkpoint 在所有条件下都无法爬楼：旧日志没有逐 tick
回读真实 policy command buffer，也没有同拍 height scan、12 维 action 和
足端接触。下一轮必须先补齐这些证据，再区分 checkpoint 能力、观测覆盖与
接触/训练域问题。

用户确认第一阶段允许“进入楼梯段直接切换为底盘冻结”。按此边界完成的真实
ROS 2 + Isaac v5 位于
`outputs/pct_scan/multifloor_stair_first_flight_freeze_ros2_v5_20260731/`。
完整 21 点 live `/initial_path` 的几何哈希为
`e5e6efa6f2816feb77473973f45019dd6a059b4ecf907f8a57d20c90f067e640`，与冻结
配置一致；冻结段累计推进 `4.34267 m` 并完成 staged release，release 写入
序号为 `1414@28.28 s`；它是 summary 的释放 marker，不是最后一条 inhibit
写入。`1415@28.30 s` 仍被抑制，第一条 fresh、未抑制的恢复写入才是
`1416@28.32 s`。planner 随后在完整 Path 终点发布 stationary final hold，
controller 的 goal 从
`1544:false@30.94 s` 上升到 `1545:true@30.96 s`，executor 验证 true 后连续
5 次零速实写，且 `post_goal_nonzero_write_count=0`。

v5 的 `summary.json` 为 `success=true`、
`success_semantics=scan_stair_root_lock_workaround`、
`navigation_root_lock_workaround_success=true`；同时严格保留
`policy_activity_seen=false`、`physical_navigation_success=false`、
`pure_physics_success=false`、`stable_physics_success=false` 和
`low_level_stair_locomotion_success=false`。运行使用了 navigation base、支撑
关节和全身姿态锁，顶层也记录 `used_base_teleport=true` 与
`used_direct_joint_state=true`，因此它只证明用户接受的工程 workaround 和
SCAN 终点/停车生命周期，不证明 checkpoint 能完成物理楼梯运动，也不具备
训练数据资格。

该次成功还存在必须保留的诊断边界：临时 ROS 日志
`/tmp/pct_scan_ros_68` 中 planner 共记录 36 条 ERROR 和 76 条 WARN，包括 7 次
“连续 5 次重规划失败后 emergency stop”和 7 次恢复；最终 stationary hold 与
GOAL_REACHED 仍然通过，但这些计数没有进入输出目录的 `summary.json`，输出
目录本身也没有持久化 rosout/stdout。startup/legacy 字段还同时出现 `pct`、
PCT disabled 与实际 runtime override `external_ros2_path` 三套表述；本次真实执行
配置以 runtime override 和 executor lifecycle 为准。后续验收必须把 planner
错误/恢复计数和最终生效的 planner source 写入结构化产物，不能只凭顶层
`success=true` 判断过程健康。

楼梯异常路径现已补上锁存急停：只要 root 锁已经在 active、full-lock settle
或 root-release settle 阶段生效，Path 清空/换代冲突、仿真时钟回退或其他执行器
失败都继续下发最后一个 `xyzyaw`、支撑关节锁和该阶段的全身锁，同时精确零速并
请求全局重规划。锁尚未激活或已经完成释放时不会突然重新锁回，以免产生姿态
阶跃。该行为仍属于非物理 workaround，不能计入纯物理成功或训练数据。

`navigation_supervisor` 现已作为 typed ROS 2 节点接入组合图，实现
`IDLE → GLOBAL_PLANNING → LOCAL_PLANNING → TRACKING`、重规划、急停和目标
锁存状态机，以及幂等重规划请求编号。它消费 PCT、SCAN、controller、Path、
Odometry、点云和目标状态，以 reliable + transient-local 发布
`NavigationStatus` 心跳，并通过 `PCTPlanningCommand` 服务请求有界重规划。
Isaac OGN bridge 与唯一 policy 写入者把 goal id、Path stamp、状态 revision 和
sequence 绑定为短时强制许可；缺失、过期、身份不匹配或撤销时立即写零。
真实 ROS 图预检还确认 `/cmd_vel` 只有 `scan_controller` 一个发布者，五个主线
节点可由一次中断干净退出。新许可链尚未在真实 Isaac 跨层 episode 中重跑，
所以这里仍不宣称 live PCT 重规划闭环已经完成。

### PCT ROS 2 adapter 与当前主线边界

`pct_ros2_adapter` 已把固定提交的官方 PCT 原生核心封装为 ROS 2 Humble 节点。
节点订阅 SensorData QoS 的 `/body_pose` 和 reliable `/pct/goal`，发布
reliable + transient-local 的 `/pct/global_path` 与 typed
`/pct/planning_status`。主运行链在同一进程内直接调用 PCT core，不使用旧
JSON、文件轮询或 stdin/stdout 子进程。组合 launch 默认把
`/pct/global_path` 直接作为 SCAN 的 `/initial_path` 输入，不存在 relay；PCT
与手工 Path 发布器互斥。

当前 adapter 已收口以下安全合同：

- Path z 始终是 collision PLY 地面高度，起点和 goal 输入 z 是 base 高度；
- `body_height_m`、world/base frame 和所有 topic 由组合 launch 统一覆盖；
- 新 goal 先发布空 Path、标记旧代际取消；旧 worker 返回后其结果会被丢弃，
不能覆盖新目标；adapter 在仿真时钟首次有效后也发布空 Path + `IDLE`，清理进程重启前
  遗留的 transient-local Path；
- 同 stamp、同 payload 的 goal 重发幂等；同 stamp、不同 payload 会使该代
  失败关闭；更旧 goal 被忽略，更新 stamp 即使几何相同也建立新规划代际；
- future/过期时间戳、frame、四元数、起点漂移、端点支撑面误差、过短 Path、
  snap 距离和错层 snap 均有硬门；compatible A* 在循环内响应取消和单调时钟
  截止；patched upstream native A* 释放 GIL，并由每请求 Python Event 与 sticky
  原子 cancel 协同抢占搜索，过时代结果仍受 plan-id/tombstone 门禁保护；
- 规划成功、无路与异常通过 `PCTPlanningStatus` 明确区分，Path 与状态只使用
  当前 ROS 时钟生成时间戳。

历史 compatible grid core 仍用于隔离回归，但不是生产 backend，也不会在
upstream 失败时接管。`pct_scene/external/PCT` 虽有
`BoZhiStudying233/PCT@b15b68c2cf22e16d3632df71236bc23f2aacde5e`，但该浅克隆
缺少其脚本实际 import 的 `planner_wrapper` 与 `tomography` 源码；直接复制目录
不能满足 `AGENTS.md` 所要求的上游 PCT core 复用。真正官方核心已经定位到
`byangw/PCT_planner@35cd73fd82bcd51bc538429294af7646b2a09815`，其 API 与
`pct_scene` 脚本逐项吻合；固定版本的 ignored 本地副本已通过 GitHub codeload
准备，并保留 GPLv2-or-later 的 `LICENSE`/`NOTICE`。GTSAM 4.1.1、OSQP 与四个
CPython 3.10 原生扩展已经完成 Release 构建，RUNPATH、`ldd`、固定源码哈希和
真实导入均通过。生产 upstream backend 只运行官方 native A*，不调用可能跨过
支撑面的 GPMP 平滑器；局部连续优化明确由 SCAN 承担。

旧 `mutifloor.pickle` 是规则切片兼容资产，没有官方跨层 gateway。新
`mutifloor_upstream.pickle` 从 collision PLY 重建官方五通道语义，并应用
`configs/navigation/pct_multifloor_stair_profile.json` 中来自 `pct-scene` 的 7 个
楼梯 ground anchor。构建阶段用它修补回转平台拓扑；运行阶段仍由原生 PCT A*
决定楼层与楼面路线，adapter 只把已匹配的楼梯区间规范化到这些 anchor，不固定
完整任务路径。当前原生离线探针得到 171 点、`22.831 m`、logical layer
`8→15`、高度跨度 `3.217 m` 的地面 Path；7 个 anchor 的 XY 误差均为零，请求的
精确起终点保持不变。隔离 DDS 生命周期探针收到 `SUCCEEDED` 与 170 点 typed
Path，起终点 base/ground 误差为零，节点 SIGINT 干净退出。受跟踪的 upstream
patch 已增加触达节点状态清理、
原子取消、搜索状态/展开计数和 pybind GIL 释放；同一 A* 实例的 no-path→success、
预取消→reset→success，以及 1024×1024 搜索期间 Python 并发取消均已通过原生探针。

multi-floor 真实 ignored 资产的 upstream 路径点数不是固定接口；状态中的
`path_point_count` 与实际 Path 必须严格一致。历史
compatible 的 169/43 点与 slice `9→15` 只作为隔离基线，不能再用来证明生产
upstream 执行。

首次把 169 点 Path 直接送入组合图时，SCAN 把全部 waypoint 同步送进
minimum-snap 稠密求解，15 秒内没有产生 B-spline，退出时也只能从 SIGINT、
SIGTERM 升级到 SIGKILL。修复后完整 Path 仍原样用于 controller、代际和终点
语义，但 FSM 不再同步拟合整条全局路线。它对原始有序三维折线建立累计弧长，
在局部前向窗口内做有界投影、维护不可回退的认证进度，并构造保留所有转角和
高度锚点的 guide；最多接受 4096 点，机器人距折线超过 `0.50 m` 时不推进进度。
原始 guide 与其初始 B-spline 在膨胀占据图中都全段无碰撞时才跳过 rebound，
只对原控制点做动力学时间拉长；任一有碰撞或未知空间都会进入 A*/rebound
局部绕障。无碰撞 guide 的发布轨迹还要通过全轨迹碰撞检查和 `0.10 m` 有序
双向参考廊道门，防止再次抄直角、漏过高度锚点或跳到重叠路径的错误分支。

最后一轮安全审计又补齐了四个执行合同。第一，Path 与 B-spline 使用 Header
原始纳秒值严格匹配，future tolerance 只允许吸收时钟误差；几何或终点 yaw
变化会立即淘汰旧轨迹；只有同 stamp、同 payload 的重发幂等，stamp 变大即使
几何相同也是新代际。同一 stamp 出现两套语义时整代作废并等待新 stamp。
第二，碰撞查询使用半径 `0.27 m` 与中心偏移 `0.16 m` 形成约 `0.43 m` 的任意
yaw 保守外接包络，覆盖 controller 的起终点原地转向和带航向误差平移；当前
双圆柱 marker 只显示名义姿态，不等于实际安全包络。第三，reference guide
是否无碰撞改为检查原始折线本身；有序 corridor 按单调 guide 进度和单调轨迹
样本匹配，避免 U 形及 XY 重叠楼层段跳支。第四，reference 的 rebound、最终门
和运行期预测碰撞都检查完整剩余轨迹，速度/加速度容差也收紧为硬上界附近。
轨迹硬超时后 controller 仍严格零速，但解除 execution-frozen 请求重规划，
避免末端 yaw 对齐超时与 planner 互相等待。
同一 Path 内的 B-spline 还按 `traj_id + Header 纳秒时间戳` 排序：旧包不能
覆盖新轨迹，精确重复不重置进度，同 identity 不同 payload 会停车并等待更大
`traj_id`。同代 Path 一旦锁存 `GOAL_REACHED`，后到 B-spline 不能解除锁存；
只有严格更新的 Path 或显式空 Path/cancel 才能开启下一目标生命周期。

phase187 曾用当时的安全二进制重跑生产 upstream 首段链。探针最初以约 40 Hz
连续发布点云，高于 GridMap 的 20 Hz wall timer，使每次 Odometry/FSM 检查都
遇到待融合帧；点云改为 10 Hz 后，typed 诊断确认 144/144 个显式 free 端点
被接纳并完成融合。随后又暴露两项真实 SCAN 参数化问题：旧 `lengthenTime()`
只移动中间 knot，短 guide 的端部动态约束没有同比放慢；最少七点的三次样条
还会把 PCT 的合法 45° 网格转角圆滑成超过 2 cm 有序进度门的弧长捷径。当前
实现对全部 knot 等比例缩放，空间曲线保持不变，并按 5 cm 地图尺度细分原折线
边；所有原转角和高度锚点都保留，`0.02 m` 防跳段阈值没有放宽。

隔离 CycloneDDS 复测最终收到 189 点、logical layer `8→15` 的真实 upstream
Path，GridMap observation sequence 21，18 控制点、三阶、非最终正常
B-spline 和非零 `/cmd_vel`，本轮局部规划约 `62.4 ms`。五个主线进程随后由
一次 SIGINT 干净退出。因此
`typed goal -> native PCT Path -> SCAN Bspline -> controller cmd_vel` 首段 ROS 2
消息链已经由最终安全版本验收；点云和 Odometry 仍是合成输入，不能据此宣称
Isaac 物理执行、动态障碍恢复或完整跨层到达已经完成。phase184 的 169 点
compatible/replan 失败记录仍是历史故障证据，不再代表当前生产 upstream 首段。

phase188 关闭 `march=native` 并干净重建 PCT 后，等价 A* tie path 的首段几何略有
变化，首次组合复跑暴露出 SCAN 内部不一致：`checkFeasibility()` 按 XYZ 各轴
分量检查，最终 `checkDynamicFeasibility()` 却按三维向量模长检查，导致斜向
加速度各分量通过而模长 `0.543 > 0.510 m/s²`，连续五次失败后按设计急停。
现在两道门统一使用向量模长并新增斜向速度、斜向加速度回归。隔离 DDS 复跑收到
同一 189 点 Path、144/144 个 free 端点、19 控制点正常 B-spline 和非零
`/cmd_vel`，局部规划约 `67.6 ms`；五个节点一次 SIGINT 全部干净退出。

另一个隔离 ROS 2 90° 探针量化了参考几何门：旧 rebound 轨迹距折角最近
`0.185861 m`、最大折线偏差 `0.131501 m`；修复后分别为 `0.002019 m` 和
`0.027655 m`。可行性修复将轨迹时长从 `2.6 s` 拉长到 `6.781822 s`，属于当前
速度/加速度约束下的安全降速，后续仍需在 live Isaac 上验收实际转角节拍。
完整 Path 末端合法四元数现被保留为最终机体 yaw；最终轨迹到时且进入真实末点
位置门后，controller 输出零平移并主动原地对齐该 yaw，再认证 `GOAL_REACHED`。

该验收使用“PLY 支撑面 + 0.30 m”构造 Odometry 和 goal，只证明当前明确的
消息/高度合同，不代表历史任务 JSON 的 spawn/root z 已与 live Isaac
`/body_pose` 标定。历史 task z 同时受 spawn、锁定和阶段语义影响，不能离线
反推出稳定 root-ground 高度，也不能为迎合端点门禁直接改写。当前继续保留
`0.30 m` 默认值；下一次真实组合 smoke 必须在 F1、pick 后和 F2 分别保持相同
收纳姿态、零速至少 1 秒并采集不少于 50 个 `/body_pose` 样本，再以 root z 到
collision PLY 支撑面的中位数和离散度决定最终 `body_height_m`。未完成该标定前
不采用 `0.55 m` 等离线猜测值。

此外，同一 ROS 图内的 `/clock` 回退/Isaac timeline reset 仍未完成跨 epoch
清理；当前仍要求单 episode，并在时钟回退后重启完整 ROS 2 导航图。PCT
typed 状态、SCAN 连续失败和 replan ACK 已在协议层接入 supervisor，但预测
碰撞触发到新 Path 恢复 TRACKING 的真实 Isaac 闭环仍未验收。

从 `pct_scene` 恢复的 ignored 运行资产共 76 个文件、4,611,993,290 字节；
Liangzhu 与 multi_floor profile 资产检查均 PASS。手工链已完成平地直线、
90° 转弯、连续斜坡，以及用户接受的单段楼梯 root-lock workaround 与完整
到达/零速生命周期验收。phase183 已用真实 Isaac Odometry/点云和 73 点 PCT
Path 跑通用户接受的楼梯冻结跨层 episode；phase184 又接入 typed supervisor
强制许可、结构化 policy gate 生命周期证据和动态推车运行时，并用显式自由
射线补上离开障碍的 GridMap 占据清理。phase183 的成功早于新许可接入，不能
替代 phase184 静态重跑；动态障碍恢复和 supervisor live 重规划闭环也仍未完成。
当前宿主机加载的 NVIDIA 内核模块为 `580.159.03`，用户态 NVML 为
`580.173`，`nvidia-smi` 报 driver/library version mismatch，项目 Torch 报
CUDA 804，因此真实 Isaac 终验必须在宿主机重启并恢复 CUDA 后继续；
因此 `pct_scan_final_chain_verified` 继续保持 false。纯物理 checkpoint 楼梯
能力作为独立改进项保留，不阻塞用户已接受的工程主线。

### phase184 离线安全与动态证据收口

PCT 坐标合同现只通过 `sim_to_pct_xyz()` / `pct_to_sim_xyz()` 实现。固定顺序为
坐标模式、三轴 scale、固定轴 X→Y→Z 欧拉旋转（矩阵为 `Rz @ Ry @ Rx`）、三轴
offset；默认仍严格等于 `(-sim_x,-sim_y,+sim_z)`。scale 必须三轴有限且非零，
非法配置在 CLI、ROS 参数读取、backend 和核心转换层均失败关闭。PCT Path 的 z
继续表示地面高度，非零地面 z 的回归证明 adapter 不会重复增加 body height。

supervisor 已为不同 topic 没有全局 DDS 顺序这一事实加入有界 pending cache。
同一 Path stamp 的 B-spline/SCAN 状态可以先于 Path 到达，Path 到达后按固定顺序
重放；冲突、过期、容量超限、tombstone、旧代和 future 代均失败关闭。真实
CycloneDDS 探针已验证“B-spline + SCAN + controller 先到、Path 后到”可以从零速
恢复到 identity-valid TRACKING，也验证 tombstone 同 stamp 不能复活。仍未解决
的是 `/clock` epoch 回拨，timeline 重启后必须重启完整导航图。

动态 F1 不再用“最后到达”推断绕障过程。SCAN 新增两条 typed 证据流：

- `/planning/grid_map_observation_diagnostics` 逐观测记录 source cloud stamp、
  sensor pose、严格 sequence、过滤后 hit/free 端点和 canonical voxel index；
  free→occupied 阈值穿越会固定保存原始 hit 点、观测代和 Header 时间，随后
  explicit-free `p_miss` 使同一 voxel occupied→free 时直接携带这份来源。
  sliding reset 单独计数，不能冒充显式清除；只有移除 explicit miss 后占据投票
  会改变，才把该次 miss 记为反事实必要原因。
- `/planning/bspline_diagnostics` 与 B-spline 共用完整 Path/header/start/traj identity，
  记录 ordered corridor 三项实测值与门限、0.01 秒全时域轨迹的最多 64 个均匀
  样本和有序 reference 样本，并携带轨迹时长、导数 B-spline 控制点给出的
  连续时间速度上界，以及 planner 进程实际采用的 `0.27 m + 0.16 m` Go2-X5
  双圆柱包络；诊断构造失败时不发布无法审计的 B-spline。

两条 topic 都使用 reliable + transient-local KeepLast(64)。Isaac OGN 对字段集合、
时间戳、sequence、计数和固定 `min(total_count,64)` 数组合同逐项校验；runtime
用 episode ROS 时间 offset 把 Header 时间映射到动态障碍本地时间，并形成五项
可追溯证据：过滤后命中、ordered detour、连续曲线净距、显式 miss 清除旧占据、
清除后不同 identity 的轨迹恢复。轨迹净距不是只看离散点，而是从采样最小值中
减去 `速度上界 × 相邻采样时间 / 2`，所得连续曲线下界仍须不小于 `0.43 m`。

detour 必须晚于同一推车 voxel 的真实 free→occupied 阈值穿越，并在障碍 clear
以前依次出现同 identity 的 controller accepted、有效 TRACKING 状态和绑定该状态
快照的 policy 写入；同 tick 但发生在 clear 之后的写入也不接受。恢复轨迹必须在
clear 后采用不同 B-spline identity、保持同一 PCT Path 代，回到不超过 `0.02 m`
的参考偏差且相对 detour 至少改善 `0.01 m`，再由 controller TRACKING 与 policy
写入共同确认。Twist 本身没有 Header/identity，因此证据只陈述“policy 写入前已
观察到该 controller TRACKING identity”，不声称 Twist payload 自带 B-spline
identity。每个聚合叶子保留 typed topic/header/sequence/identity 引用，validator
会从生命周期重新计算，不能只信 `verified=true`。命中关联容差上限为 `0.05 m`；
同 canonical voxel 的 hit/clear 几何门取半个 voxel 对角线，而非固定距离。

当前全仓 Python 回归为 `1226 passed、5 skipped、24 subtests passed`；ROS 2
workspace XML 累计为 `385 tests、0 errors、0 failures、4 skipped`，其中最新
`pct_ros2_adapter` 87 项、`bspline_opt` 4 项、`scan_planner` 49 项全部通过；另有真实 CycloneDDS supervisor
时序/重规划探针和最终 upstream→SCAN CPU 多进程探针通过。真实静态许可、非冻结
平地许可和动态 F1 仍未运行：截至
2026-08-01 再次检查，内核模块仍为 `580.159.03`，磁盘模块为 `580.173.02`，
`nvidia-smi` 无法连接驱动，Torch `2.7.0+cu128` 返回
`cuda_available=false, device_count=0`。宿主重启恢复 CUDA 后才能执行三项 live
终验；在此之前最终链继续保持 false。

### phase189 楼梯冻结与终验入口收口

生产 PCT Path 不携带手工楼梯索引。本轮在真实 multi-floor PCT ROS 输出上新增
直接接管回归：几何识别得到唯一跨层楼梯组件，机器人到达该组件入口时立即输出
零 `base_velocity`，同时启用 root、support joint、full-body joint-pose 三重锁和
`cmd_vel` inhibit。该动作保持 `pct_scene` 的底盘冻结机制，并继续明确标记为
`non_physical_root_lock_workaround`，不是纯物理爬楼。

冻结期间的传感器合同现有两层。唯一 policy owner 即使处于临时 inhibit，仍会
保留本地 Odometry/点云的 missing、future、timeout 原因；冻结协调器还严格读取
supervisor 实际收到的 `/body_pose`、`/cloud_registered` 新鲜度和当前 Path
identity。任一侧失鲜都会在下一次弧长推进前锁存最后 root/关节目标、持续精确
零速并请求全局重规划；冻结期间单独 `bspline` stale 属于预期状态，不会误报。
完整 pipeline 只要 provenance 出现 navigation root lock，无论运行模式都禁止把
`physical_navigation_success` 标为 true；carry→place 交接也改由本轮实际发生的
SCAN 冻结 provenance 驱动，不再依赖旧 `pct_stair_float` 开关。multi-floor 场景
默认的旧 float 已关闭。

fresh live runner 的预检也已修正：`ros2 node list` 使用独立临时日志目录，避免
rclpy 抢先创建正式 fresh 输出；节点查询不再请求隐藏的 `_ros2cli_*` 自身节点。
隔离真实 ROS domain 预检得到 `nodes=0` 且正式输出目录仍不存在。当前回归为
`1255 passed、5 skipped、24 subtests passed`，受影响 `pct_ros2_adapter` 为
`97/97`；GPU 版本失配仍未变化，因此四类 fresh Isaac 终验尚未运行。

### phase224–225 跨层净空、终点捕获与冻结接管诊断

phase223 的墙边卡点不是 SCAN 修改 PCT 路线，而是 upstream A* 当时只按
tomogram 栅格中心可通选择楼面路径。phase224 在 collision PLY 的机身高度带上
构造连续软净空代价，并让 native A* 直接消费该代价；楼梯 gateway 仍使用原始
traversability，避免唯一跨层通道被误封。当前固定请求得到 188 点、
`24.297071 m` 的地面 Path，较旧路线更长，但已绕开 phase223 的 F1 墙边窄路。
SCAN 仍原样消费 `/pct/global_path`，没有第二个全局路径发布器。

二楼任务精确终点在指定 `-pi/2` yaw 下与静态障碍的双圆柱余量约为负
`0.026 m`。任务目标不能私自移动、圆柱包络也不能缩小，因此 reference FSM
新增末端安全捕获：只有完整终点已进入局部窗口时，才从严格 XY/Z 到达门内部
寻找 terminal-yaw 自由、且向后具有连续 B-spline 支撑区的第一个点。当前离线
候选距精确目标 `0.0509 m`，terminal-yaw 余量约 `0.0244 m`；精确终点继续用于
Path identity、最终误差和 stationary final 认证。

phase225 的一次完整 Isaac carry 运行在楼梯前失败。机器人从
`(-3.480, 6.521)` 运行到 `(0.789, 4.954)`，之后 SCAN 对下一局部目标
`(1.50, 5.59)` 反复报告 A* 无路/机身位于障碍膨胀区；supervisor 与 PCT
重规划协议均工作，但 23 代 live Path 都保持同一楼梯入口拓扑。12000 个导航
tick 后按 `nav_to_place_timeout` 失败，冻结状态始终停在 `approach`，没有发生
root lock。

碰撞模型和 live Path 的联合回放给出了直接原因：该轮重规划的扩展楼梯组件起点
为 `(0.997, 4.864)`，实测停点到组件的距离是 `0.226 m`，而生产接管半径仅
`0.12 m`。`activation_lookahead_m=1.60 m` 只负责接近窗口与超时观察，并不会
触发接管，所以 SCAN 碰撞停车发生在冻结接管之前。生产 profile 已把
`activation_radius_m` 改为 `0.35 m`，契约哈希更新为
`d88694a5765a5a768fa0649bb7598e5be0b049f9a23277e0cbc61d005f5fb329`。
同一 135 点 Path、同一停点的无 Isaac 回放已得到
`scan_stair_freeze_activated` 和 `phase=active`。按照用户要求，本轮结束后没有
自动启动第二次 Isaac，因此修复后的完整跨层效果仍待下一轮单次实测确认。

### phase226 跨层平地整形与 SCAN 首段速度门

phase225 的 B-spline 诊断显示，跨层运行慢的主要原因不是每次约
`0.16–0.18 s` 的局部 A*，而是 PCT 跨层路径在楼梯前后保留了过密的栅格折点。
同一生产速度配置下，平地 phase221 的 8 条轨迹基本达到
`0.50–0.545 m/s`；phase225 跨层前 8 条中有 6 条仅为
`0.196–0.231 m/s`。正常 TRACKING 的 `42.4 s` 中约 `10.8 s` 用于原地 yaw，
平移命令均值仅 `0.202 m/s`。SCAN 没有另选全局路线，它是在严格执行 PCT
提供的每一个有序折点。

upstream adapter 现把跨层路径拆成起点楼面、标定楼梯和目标楼面三段。两端
楼面使用与同层路径相同的净空视线压缩；7 个楼梯 anchor、精确起终点与
layer `8→15` 拓扑完全保留。每条捷径同时检查原 tomogram 阻塞单元、机身高度
带障碍表面和半个栅格对角余量；不能直连的相邻原生 A* 段不改。对 phase225
精确请求，发布点由 188 降至 142，内部控制折点由 79 降至 32，路径长度由
`24.297071 m` 降至 `23.704265 m`。

纯 ROS 2 多进程复核使用 phase225 精确生产起终点与统一
`body_height=0.338 m`，收到同代 142 点 Path、35 控制点正常三阶 B-spline 和
非零 `/cmd_vel`。首条轨迹从 live 旧路径的
`8.524156 s / 0.207441 m/s` 恢复为 `3.002090 s / 0.544554 m/s`；探针新增
`duration<=3.20 s` 和 `velocity>=0.50 m/s` 两个硬门。PCT 规划本身增加约
`0.2 s` 净空检查开销，但执行阶段首段速度提高约 2.6 倍。该复核不启动 Isaac，
因此完整 RL/PhysX 耗时、楼梯冻结释放和是否超过 DWA 仍未宣称完成。

### phase227 全局 Path 的 SCAN 全楼面检查

首段快不等于整条路线都快。新增的全路线探针固定调用当前生产 PCT 后端，并从
同一条 142 点、`23.704265 m` 全局 Path 上按 0.65 m 取样。它跳过
`6.774846–15.515694 m` 的楼梯冻结段和终点最后 1.30 m 的独立捕获段，对其余
22 个 SCAN 平地局部窗口逐一发布 Path 后缀、理想 Odometry 与显式自由点云。

关闭新转角平滑时，这 22 条轨迹平均 `4.133762 s`、最长 `7.659532 s`、最低
速度上界 `0.188562 m/s`；开启后分别为 `2.441531 s`、`3.022893 s` 和
`0.469849 m/s`。所有窗口通过 `0.45 m/s` 最低速度与 `3.20 s` 最长时长门，
最大局部轨迹偏离 PCT 有序折线 `0.032035 m`，仍在 `0.10 m` 安全走廊内。

SCAN 没有重算或替换 PCT 全局路线：`/pct/global_path` 的端点、142 个发布点、
楼层顺序和 7 个楼梯 anchor 均未变化。局部平滑只在原 guide 与初始样条都被
当前地图判定无碰撞时尝试，并在发布前重新检查双圆柱碰撞、路径顺序、动力学
和进度超前；失败即保留旧折线轨迹。这证明当前全局 Path 对 SCAN 的平地执行
是合理且不再包含旧慢窗口，但输入仍是合成自由空间与理想 Odometry。按照用户
要求本阶段没有启动 Isaac，完整跨层成功、实际仿真耗时和相对 DWA 的结论继续
保持未验证。

### fresh live 验收入口

四类最终运行统一由 `scripts/navigation/run_pct_scan_live_acceptance.py` 编排，
模式为 `static_stair`、`flat_policy`、`dynamic_f1` 和
`dynamic_replan_f1`。脚本拒绝已存在的输出目录和非空 ROS domain；等待组合图
五个生产节点后才启动 Isaac，结束时只向本轮 `ros2 launch` 父进程发送 SIGINT，
确认图清空，并要求本轮 `startup_status.json` 为 `completed/exit_code=0`，最后
才校验 `episode_000000/summary.json`。这避免 Isaac 在 recorder 创建前失败时，
旧 summary 被误当成新结果。

```bash
python -B scripts/navigation/run_pct_scan_live_acceptance.py \
  --mode dynamic_replan_f1 \
  --output-dir /mnt/sage_data/outputs/pct_scan/replan_fresh_001 \
  --ros-domain-id 217 \
  --isaac-python /data/conda_envs/isaacsim51_3dgs_grasp/bin/python
```

每次必须换一个原本不存在的目录，并确认指定 domain 没有外部节点。如果再次
出现 GPU 驱动失配，不要把入口启动失败解释成导航失败；应先恢复 `nvidia-smi`
与项目 Torch CUDA probe。phase276 之后的最新实测已经在恢复后的 GPU 上完成。

### phase276–281 最新主线验收

GPU 恢复后，原 Go2-X5 mobile-manipulation checkpoint 已连续形成三份成功证据：
phase276 的 full seed 0、phase277 的 carry seed 1，以及修复末端横向恢复朝向后的
phase279 carry seed 2。它们不是三份“完全相同二进制”的统计样本，因此不能写成
当前精确代码的 3-seed 成功率；但分别覆盖了一楼 SCAN、楼梯底盘冻结/释放、二楼
SCAN、严格 `GOAL_REACHED` 和目标后连续零速。phase279 的严格 validator 为
`0` 错误，最终 XY/Z/yaw 误差约为 `0.0747 m / 0.0513 m / 0.1407 rad`，命令与
policy 实写均保持零，目标后非零写入数为 `0`。

phase280 随后用最新生产配置完成一次完整 seed 0：

```text
nav_to_pick → pick → carry nav_to_place → place → LeRobot export → done
```

该次共执行 `6756` 个控制步，宿主墙钟 `891.79 s`。抓取验证通过，物体到 TCP
距离约 `0.00882 m`、随手臂撤退位移约 `0.23891 m`；放置验证通过，最终 XY/Z
误差约 `0.00834 m / 0.00098 m`，释放峰值线速度 `0.18680 m/s`、水平速度
`0.09392 m/s`、向下速度 `0.18671 m/s`、角速度 `3.18572 rad/s`，均低于任务
质量门。携物导航最终底盘距 `(0.4,-0.02)` 的 XY 误差约 `0.056 m`，yaw 误差约
`0.144 rad`，到点后命令为零。输出位于
`outputs/pct_scan/full_pipeline_phase280_current_code_seed0_v2`。

这次成功仍使用用户确认的 `chassis_root_lock` 楼梯方案，因此只说明 PCT→SCAN
消息链、平地闭环、楼梯交接、抓放交接和最终零速主线已经贯通。它不说明 RL
checkpoint 能纯物理爬楼，也不满足训练数据的纯物理质量门；summary 正确记录
`navigation_root_lock_workaround_success=true`、
`physical_navigation_success=false`、`pure_physics_success=false` 和
`training_quality_gate_passed=false`。

phase280 的运行日志还暴露两个不影响安全和最终成功的收尾问题。第一次到抓取点
后，planner 在 controller 已经锁存 `GOAL_REACHED` 后又发布了两条候选轨迹；
原因是局部优化占用 planner timer 时，ACCEPTED 状态回调可能晚于下一条
B-spline 发布，而旧逻辑只认证“最新发布 identity”。phase281 现在保留与
ControllerStatus QoS 深度相同的 64 条 final 发布身份历史，只接受四元 identity
完全匹配的回执，并在新 Path 激活或到达确认时清空整代证据。这样既能消费排队的
合法 final 回执，也不会让旧 Path 的迟到 GOAL_REACHED 清除新目标。另一个问题是
到点或 tombstone 后，楼梯 publisher 会短暂继续发送旧快照；planner 现在在没有
可执行 Path 绑定时直接忽略，不再反复打印
`reference_path_identity_mismatch`，新 Path 激活后仍恢复全部身份和新鲜度门。

phase281 没有再启动 Isaac。`scan_planner` 构建通过且 `149/149` 测试通过，
`scan_controller` 为 `126/126`，坐标转换、Path 适配、速度限幅、超时停车、状态机、
比较合同和完整 pipeline 的选定 Python 回归为 `390/390`。宿主侧 `nvidia-smi`
可正常读取 RTX 4060，除 Xorg 外没有 GPU 计算进程，本项目 Isaac/ROS 进程均已
退出。

当前结论分为两层：

- 稳定 baseline：在楼梯底盘冻结这个明确前提下，当前精确代码与生产参数的
  cross-floor carry 已完成相同 Path 的 seeds 0、1、2 严格 `3/3`；生产参数的完整
  nav/pick/carry/place/export 也已通过一次。
- 发布选择：保持 `reference_cruise_speed=0.60 m/s`、`max_vx=0.65 m/s` 和
  `max_yaw_rate=0.60 rad/s`。0.75 候选未通过完整放置质量门，不进入默认配置。
- 非本次验收：不再继续 DWA 对比，也不作优于 DWA 的结论；移动推车动态绕障与
  live PCT 全局重规划留待后续单独验收。
