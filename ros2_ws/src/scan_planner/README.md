# scan_planner

该包是从 `external/SCAN-Planner` 的
`origin/ros2-community@d0b921c9b05a6d291d144d60882b2e0e88d2c0e0`
按 planner-only 边界移植的 ROS 2 节点。它只包含
`scan_planner_node`、重规划 FSM、planner manager 和 plan container；
旧控制器、运动学模拟器、Go2 描述、local sensing、map generator 与
mockamap 均不属于此包。

默认接口为：

- `/body_pose`：`nav_msgs/msg/Odometry`
- `/cloud_registered`：世界坐标系 `sensor_msgs/msg/PointCloud2`
- `/initial_path`：地面高度语义的 `nav_msgs/msg/Path`
- `/planning/bspline`：`scan_planner_msgs/msg/Bspline`
- `/planning/controller_status`：`scan_planner_msgs/msg/ControllerStatus`

`/planning/controller_status` 的 publisher 与 planner subscriber 均固定使用
reliable + transient-local + KeepLast(64)。`qos.controller_status_depth` 默认且
只允许为 `64`；其他值会令 planner 拒绝启动，避免 controller 在同一回调中
连续发布旧轨迹终态和新轨迹接管证据时被 KeepLast(1) 覆盖。严格 trajectory
identity gate 仍决定消息能否推进 final hold 或主动感知状态机。

默认配置使用 `use_sim_time=true`、`fsm.navi_mode=3` 和 `world` 坐标系。
`Path` 的 z 表示地面高度，FSM 只在这里加一次 `body_height=0.30`。
双圆柱包络为半径 `0.27 m`、中心偏移 `0.16 m`；planner 速度/加速度上限
分别为 `0.30 m/s` 和 `0.50 m/s²`。

该包只负责局部规划，不发布 `/cmd_vel`。独立的 `scan_controller` 已负责
闭环轨迹跟踪，Isaac runtime 内的 `cmd_vel_to_policy` 安全门负责写入
policy command buffer。

## 当前边界

- 规划与安全 FSM 已改用 ROS time timer；Path 会先校验 frame、非零时间戳、
  有限位姿和近重复点。早到 Path 会缓存，直到新鲜 Odometry 与首帧已融合
  点云地图都就绪。
- Path 代际使用顶层 Header 的原始纳秒 stamp。正常和急停 B-spline 都携带
  `reference_path_stamp`，controller 只接受与活动 Path 严格相等的代际；
  100 ms future tolerance 不再用于代际比较。较旧 Path 被拒绝，只有同 stamp
  且同 payload 的 DDS 重发保持幂等；stamp 变大即为新代际，即使几何相同也会
  淘汰旧局部轨迹。合法空 Path 会清除待激活/活动参考。同一 stamp 若出现不同
  几何或终点 yaw，planner/controller 都会作废整个冲突代际、立即停车并等待
  更大的 stamp，避免迟到旧 B-spline 被误执行。
- 急停发布 `Bspline.emergency_stop=true`；六个重合控制点只保留消息结构
  兼容，不再承担急停语义，controller 会依据显式标志锁存零速。
- 激活缓存 Path 时保留全部有序点；默认同起点由真实 Odometry 在三维弧长上
  投影并从认证进度向前构造局部 guide，不删除 PCT 全局 Path 的前导语义。
- PCT 的完整高密度 Path 原样保留给 controller、终点和代际判断。FSM 不再对
  整条路线做同步 minimum-snap 拟合，而是按有序三维弧长维护单调进度，在局部
  前向窗口内投影机器人、选择目标并构造保留转角/高度锚点的 guide。Path 最多
  接受 4096 点，偏离参考折线超过 `0.50 m` 时不会推进已认证进度。
- 原始 reference guide 在当前膨胀占据图中全段无碰撞、且其初始 B-spline 也
  无碰撞时，才保留折线参数化控制点并只做动力学时间拉长；任一有碰撞或未知
  空间都会进入 A*/rebound 局部绕障。不论 guide 是否有碰撞，最终 B-spline
  都必须通过全轨迹碰撞门和 `0.10 m` 有序双向参考廊道门；因此动态绕障
  可保留最大 `0.10 m` 的合法局部偏移，但不能完全脱离 PCT 顺序。轨迹样本及
  guide 锚点均按不可回退的顺序匹配，guide 弧长进度还不得比轨迹已行距离
  超前 `0.02 m`，防止优化器从直角内侧抄近路、漏过高度锚点，或在 U 形、
  平行近邻及自交/XY 重叠段跳到错误分支。任一门超限都拒绝发布并继续停车。
- 双圆柱碰撞查询不再只使用 B-spline 切线 yaw。占据门以已经按半径
  `0.27 m` 膨胀的地图为基础，再扫掠中心偏移 `0.16 m` 的整个位姿圆盘，等价
  于约 `0.43 m` 的任意 yaw 保守外接包络，并覆盖起点/终点原地旋转及有限
  航向误差平移。现有 marker 仍只画名义朝向的两个圆柱，不能当作真实安全门
  的完整可视化。
- reference 模式下，rebound 前后检查、发布前碰撞门和运行期预测碰撞均覆盖
  完整剩余轨迹，不再跳过末尾约三分之一；速度、加速度硬门的默认容差分别
  收紧为 `0.005 m/s`、`0.01 m/s²`，B-spline 物理可行性比例容差为 `0.01`。
- Isaac direct Odometry 的 `twist.linear` 按 `base_link` 语义旋转到
  `world`；点云/位姿还带有 frame、0.20 秒同步差和 0.50 秒新鲜度门禁。
- 上游 `GridMap` 分别以 SensorData QoS 订阅世界系点云与
  `sensor_pose`。lidar 模式使用容量 100 的 ApproximateTime 同步队列，
  只有时间相近的点云/位姿对到齐后才进入同一个融合回调，并继续执行
  0.20 秒严格门禁，避免跨 topic 回调乱序时误用最后到达的位姿。组合 launch
  使用 `/body_pose` 加配置化 `cloud_sensor_extrinsic_*` 得到真实射线原点；
  Isaac 主配置与 `head_camera` 的 0.28 m 前移、0.07 m 上移及光学姿态一致。
- 空旷 reference guide 受 0.10 m 偏差和 0.035 m 进度超前硬门约束；
  guide 被在线障碍占据时才切换到 0.35 m/0.05 m 的独立有界门限，使 rebound
  能保留 `dist0=0.20 m` 障碍余量。两种情况均必须通过完整双圆柱碰撞检查，
  不会缩小 Go2-X5 包络或放宽空旷路径的防跳段约束。
- GridMap 只接受 bridge 的精确 canonical empty，表示“本帧输入充足，但分类
  后没有障碍端点或可用自由射线”；其他空布局会被拒绝。有限 raw 点不足 64 的帧由 bridge
  在上游丢弃，不会进入 GridMap 或刷新地图新鲜度。
- bridge 可在非空点云中附加标量 `uint8 ray_endpoint_type`：1 是障碍
  命中端点，0 是已认证地面/Path 支撑面的显式自由端点。GridMap 对
  值 0 只执行当前测量射线的 `p_miss` 更新，不做全图 TTL 或无观测清理；
  主配置 `p_max=0.98`/`p_miss=0.30`/`p_occ=0.80` 下，饱和旧体素被明确
  free ray 穿过 3 帧后会降到占据门下。没有该字段的标准非空点云
  仍全部按障碍命中处理；非法字段或非 0/1 值会 fail-closed 拒绝。
- 滑动地图只会重置移出 10×10×5 m 窗口的环形缓冲区；canonical empty
  只刷新观测生命期，不清除历史占据。因此只有后续真实射线穿过的移动
  障碍体素才会恢复，无观测或被遮挡的静态区域仍保守保留。
- 当前节点不直接生成 `/cmd_vel`；Odometry、B-spline 和点云超时零速由
  独立 controller 实现。生产 `navigation_supervisor` 已订阅 typed
  `ScanPlanningStatus`，达到连续失败阈值后通过 `PCTPlanningCommand` 发起有界
  REPLAN，并等待 service ACK、匹配的新 PCT status/Path 与 controller identity
  后才重新许可跟踪。协议层及真实 CycloneDDS 时序探针已经通过；阻断型动态
  推车触发该闭环并在 Isaac 中恢复非零 TRACKING，仍是待 GPU 恢复后的 live
  验收项。
- reference 模式仅在完整 Path 最终点使用零终端速度。非最终局部分段使用
  `fsm.reference_cruise_speed`，沿有序局部 guide 的末切线生成连续巡航边界；
  初始时间参数化同时使用实测起点速度和该末端速度，避免每个短窗口都按
  “起步—停车”曲线拉长。该速度仍被 `manager.max_vel` 硬裁剪，每段继续受
  动力学可行性、反向参考、有序走廊和全轨迹碰撞检查约束。独立 planner
  配置默认巡航为零；生产组合 launch 从统一调参覆盖层显式给值。
- Path 末端合法四元数定义最终机体 yaw，不能由最后一段折线切向替代。最终
  轨迹到时且机器人已进入完整 Path 末点位置门后，controller 会保持零平移并
  主动原地对齐该 yaw；位置、朝向和低速条件全部满足后才发布 `GOAL_REACHED`。
  planner 的 `0.04 m / 0.18 rad` 用于选择末端安全目标，也是发布
  严格零速 stationary hold 的迟滞内门；最终完成仍由 controller
  独立的 `0.08 m / 0.20 rad` 外门与连续 `0.50 s` 稳定驻留共同认证。
  planner 发布 stationary final hold 后保留目标与完整 Path，且不周期刷新
  hold identity；只有 `/planning/controller_status` 与该 B-spline 的 Path stamp、
  Header、start time 和 `traj_id` 全部精确匹配时才消费终态。匹配
  `GOAL_REACHED` 后才清目标；匹配 `TRAJECTORY_TIMEOUT` 则保留同代 Path，
  生成严格更高 `traj_id` 的正常 B-spline，从而避免进入不可恢复的
  `WAIT_TARGET`。
- 合成 `/clock`、Odometry、世界系 PointCloud2 与手工 Path 的组合探针已
  收到有效三阶 B-spline 和非零 `/cmd_vel`，并验证点云超时归零；这不等于
  Isaac Sim、RL policy、平地或楼梯端到端验收。
- 最新真实 PCT multi-floor 资产与合成 ROS 传感器多进程探针使用 phase225
  精确生产起终点，收到 142 点 upstream `/pct/global_path`；typed GridMap 诊断
  确认 144/144 个显式 free 端点完成融合，SCAN 产生 35 控制点、三阶、非最终
  正常 B-spline 和非零 `/cmd_vel`。首条轨迹为 `3.002090 s`、速度上界
  `0.544554 m/s`，已恢复生产 `0.52 m/s` 巡航能力；探针对
  `duration<=3.20 s / velocity>=0.50 m/s` 失败关闭。时间分配和最终发布门统一
  按三维导数模长限速/限加速度；五个主线进程随后可由一次 SIGINT 干净
  退出。该结果打通生产 PCT→SCAN→controller 首段消息链，但仍不替代 live
  Isaac `/body_pose` 高度标定、RL policy、PhysX 或动态障碍验收。
- 全路线平地窗口探针进一步沿同一条 142 点、`23.704265 m` PCT Path 审计楼梯
  接管区外 22 个位置。生产开启空旷转角平滑后，局部轨迹平均时长由
  `4.133762 s` 降至 `2.441531 s`，最长由 `7.659532 s` 降至 `3.022893 s`，
  最低速度上界由 `0.188562 m/s` 提升至 `0.469849 m/s`；最大折线偏差
  `0.032035 m`，全部通过不变的双圆柱、ordered-reference、进度和动力学门。
  该平滑不会修改 `/pct/global_path`；候选失败会保留原始折线轨迹。证据仍是
  合成自由空间和理想 Odometry，不替代完整 Isaac 跨层实测或 DWA 对照。
- `/planning/bspline_diagnostics` 现在为一次性 yaw-only 主动观测发布同一
  B-spline 四元 identity 下的 typed 生命周期：`STARTED → CONTROLLER_ACCEPTED
  → YAW_STABLE → FUSION_PROGRESS → COMPLETED/FAILED`。稳定快照保留实际 settle
  时间、yaw 误差、角速度和连续稳定时长；融合快照保留 settle 时的 sequence
  基线、当前 sequence 与不同采集时间戳计数。完成事件必须有至少 3 个 settle
  后真实非空融合，普通轨迹的全部主动观测字段保持默认值。代码级硬帽固定为
  `|yaw_offset|<=0.22 rad`、`yaw_rate<=0.20 rad/s`、settle yaw 误差
  `<=0.02 rad`、角速度 `<=0.05 rad/s`、连续稳定时间 `>=0.10 s`；typed
  快照无法满足这些边界时不发布不可审计证据并 fail-closed。
- 独立 90° ROS 2 几何探针中，旧 rebound 轨迹距转角最近 `0.185861 m`、最大
  折线偏差 `0.131501 m`；加入稠密 guide 后的历史 build 分别降为 `0.002019 m` 和
  `0.027655 m`。可行性修复把时长从 `2.6 s` 拉长到 `6.781822 s`，这是按当前
  加速度门安全降速；最终组合链已另行复跑，90° 定量几何仍是隔离探针证据。
- 历史连续斜坡 v24 已在真实 Isaac Sim 中依次执行 7 条局部 B-spline，并经
  `/cmd_vel -> RL policy -> PhysX -> GOAL_REACHED -> 5 zero` 完成验收；这不把
  合成 probe 本身扩张为真实证据，也不替代最终安全 build 的 live Isaac 复跑。
- 当前占据图本身仍不区分楼梯踏面与普通障碍。bridge 的有序 Path 支撑廊道
  已支持斜坡，但单段楼梯、动态障碍和实机传感器输入仍须分别验收，不能用
  坡道结果替代。
