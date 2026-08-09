# SCAN 闭环控制器

该包订阅完整有序三维 `/initial_path`、世界系 `/planning/bspline`、
`/body_pose` 与 `/cloud_registered`，只发布机体系 `/cmd_vel`。它不直接写
RL policy command buffer。

控制器要求 `use_sim_time=true`，并执行以下安全门禁：

- Path、B-spline、Odometry、PointCloud2 的 frame、非零时间戳及有限数
  校验；Path 使用 reliable + transient-local + keep-last(1)；
- 非空 PointCloud2 必须是有限 xyz；空观测只接受 bridge 生成的精确
  canonical xyz32 非组织布局，畸形空云不会刷新新鲜度；非空云可附带
  GridMap 使用的 `uint8 ray_endpoint_type` 字段，controller 只校验 xyz 与生命期；
- 缺有效 Path、同代 B-spline、Odometry 或 PointCloud2，以及 Odometry/
  PointCloud2 超时，都立即发布严格全零；B-spline 在接收时检查消息新鲜度，
  接收后至少有效到其 `start_time + duration`；
- `Bspline.reference_path_stamp` 必须与当前 Path 的原始纳秒时间戳严格相等；
  `future_tolerance` 只判断消息是否来自未来，不能充当代际宽限。几何或终点
  yaw 变化会在 Path 回调内立即淘汰旧轨迹并发布零速；同一 stamp 对应不同
  几何或 yaw 时整代作废，等待严格更新的 stamp。较旧 Path 被拒绝，只有同
  stamp、同 payload 的 DDS 重发保持幂等；stamp 变大即使几何相同也是新代际。
  合法空 Path 会留下代际 tombstone 并清除参考和轨迹；Path 最多接收 4096 点，
  node 在分配/遍历前和 tracker 内各执行一次上限校验；
- 同一 Path 内的 B-spline 按 `traj_id` 和原始 Header 纳秒时间戳排序。旧包不能
  覆盖当前轨迹，精确重复包不重置执行时间或驻留计时，同 identity 不同 payload
  会作废该 identity 并停车，只有更大的 `traj_id` 能恢复；
- `/planning/controller_status` 以 reliable + transient-local + keep-last(64)
  发布自包含 typed 快照。主 identity 始终表示当前或最近失效的已接受轨迹，
  拒绝包只写入独立 candidate identity，不能覆盖仍有效的当前轨迹；
  `status_sequence` 对每次发布递增，`acceptance_sequence` 只在严格四元 identity
  首次接受时递增，精确 DDS 重发标记为 `EVENT_DUPLICATE`，不能作为 fresh SCAN
  轨迹。`trajectory_valid` 表示 tracker 是否仍保留结构有效轨迹，是否允许恢复
  运动还必须同时检查 `state` 与 `emergency_stop`；
- 每个首次接受的完整 B-spline identity 都独立重置逐拍命令聚合，并在 typed
  状态中携带 `command_sample_count`、实际首拍、三轴绝对值峰值与上游违规数。
  新 identity 覆盖前会先发布旧 identity 的终态聚合快照；64 深度的持久历史
  防止该快照被紧接着发布的新 `EVENT_ACCEPTED` 覆盖。失效事件继续保留旧
  identity 的最终聚合，但失效后的停车拍不会再污染该轨迹统计；
  `qos.controller_status_depth` 固定要求为 `64`，配置为其他值会拒绝启动，
  不能用运行参数削弱这条跨事件证据合同；
- 含两点 `yaw_pts` 的主动观测轨迹会标记
  `active_sensing_yaw_only=true`。其 identity 接受当拍先同步发布严格全零，
  因而 accepted 快照的 `first_command` 必须三轴全零且样本数至少为一。后续
  `/cmd_vel` 在 node 输出端再次执行不可配置硬门：`vx=vy=0`、
  `|wz|<=0.20 rad/s`。峰值记录实际发布值；tracker 的非有限、非零平移或
  超过硬帽的原始请求会增加 `command_violation_count`，live 验收必须要求其
  为零，不能用输出端限幅掩盖上游回归；
- `Bspline.emergency_stop=true` 会锁存严格全零，不再把六个重合控制点当作
  普通位置轨迹；同代正常 B-spline 或合法新 Path 才能清除锁存；
- yaw 误差超过普通/横向 `0.70/0.20 rad` 时线速度立即归零，发布
  `/planning/go2_execution_frozen=true`，且不推进轨迹执行时间；进入后只有
  分别降到 `0.55/0.18 rad` 内才释放。软失效时刻同步顺延，但累计顺延
  包级默认不超过 6.0 秒硬上限；生产跨层配置使用 12.0 秒，同 Path 的新
  `traj_id` 不能后移仍在进行的对齐 episode 硬截止；
- 正常跟踪时对 `vx`、`vy`、`wz` 同时做幅值与变化率限制；
- 楼梯实验可显式启用 `controller.stair_heading_lock_enabled`。控制器以当前
  Path 进度为中心，在前后 `0.45 m` 弧长窗口内检测坡角不小于 `0.45 rad`
  的陡升段，并用该窗口的完整 Path 水平合成切向控制机体朝向。SCAN 的世界系
  XY 轨迹和避障分量保持不变，因此几厘米横向重接会转换为 `vy`，不会再把
  近 90 度局部短弦转换成饱和 `wz`。它不是全局 `wz=0`：平地或离开楼梯
  窗口后仍按 B-spline 转向，楼梯 Path 本身发生转弯时也会跟随 Path 切向；
- `controller.stair_forward_speed_floor` 默认为 `0`，只允许楼梯实验 overlay
  显式启用。精确踏面 Path 的近竖直段会把三维 B-spline 速度主要分配到 z，
  但四足 policy 只消费平面命令；该参数只在上述陡升窗口、B-spline 切向
  明确为正、未进入横向恢复且未进入终点制动时补足 Path 切向速度。原有
  `vx/vy` 限幅、变化率、B-spline/传感器新鲜度和 emergency stop 继续生效；
  零速与反向意图不会被改写；
- 局部轨迹完成后持续零速；只有 SCAN 明确标记最终轨迹时才发布
  `/planning/goal_reached=true`。运动 final 正常走完名义 B-spline 时间时沿用
  完整物理门；若真实 Odometry 提前进入完整 Path 的终点位置门，则先严格
  清零 `0.06 s`，随后冻结轨迹进度并以有界 XY 位置保持抵消四足策略在纯
  yaw 命令下的真实根位姿漂移；terminal yaw 收敛且所有命令重新严格为零后，
  连续稳定 `0.50 s` 才可完成，避免冻结名义时间形成死锁。moving final 默认
  采用 `0.055 m` 捕获内门、`0.08 m` 物理完成门和 `0.12 m` 释放外圈；生产
  `pct_scan_tuning.yaml` 根据 phase238 的约 0.23m 漂移证据把外圈配置为
  `0.30 m`。外圈只保留 terminal-yaw 状态，不参与完成认证。漂出内门但仍在
  外圈且姿态与速度稳定时仅恢复 B-spline 的 XY 闭环，仍锁定 terminal yaw。
  位置保持包级默认比例增益/合成速度上限为 `0.80 1/s、0.10 m/s`，生产值为
  `2.00 1/s、0.15 m/s`，并继续服从 vx/vy 变化率限制；捕获首拍清除上一段
  三轴命令历史。包级默认终点转向限制
  为 `0.25 rad/s、0.50 rad/s²`，生产调参文件使用 `0.45/1.00`。该机制不放宽
  位置、姿态、速度或传感器认证，且目标认证必须同时拥有新鲜 Odometry 和
  新鲜有效 PointCloud2。点云未来、
  超时或缺失时始终 fail-closed：保持 `CLOUD_TIMEOUT`/等待态与严格零速，
  不得只凭 Odometry 锁存完成。

横向门禁不使用会随滚动重规划重置的局部 B-spline。controller 先把当前
Odometry 的 z 减去一次 `reference_path.body_height_m=0.30 m`，首次在完整
Path 上做三维投影，后续只搜索单调进度后退 `1.0 m`、前向 `3.0 m` 的有序
弧长窗口。这样能区分 XY 重叠的不同楼层。选中投影点的 XY 残差超过
`0.12 m` 时进入严格航向门，回到 `0.08 m` 内才释放，形成滞回并避免阈值
附近反复切换。

初始软截止为
`max(header_stamp + bspline_timeout, start_time + duration + trajectory_expiry_grace)`，
硬截止为初始软截止再加 `max_yaw_alignment_freeze_sec`。新鲜输入下的大航向
误差对齐，以及已经进入最终位置门、正制动并收敛到完整 Path terminal yaw
的末端阶段，都会按本周期 `dt` 顺延软截止并始终受硬截止约束。moving final
首次进入 `0.055 m` 捕获内门时锁存本次捕获的硬截止；同 Path 滚动 B-spline 不得刷新
该截止。stationary final hold 则从 controller 实际接收时刻重新累计驻留，
并使用有界 hard expiry，不沿用已经过去的 planner start time。
超过硬截止后仍严格零速，但会发布
`/planning/go2_execution_frozen=false`，明确要求 planner 对仍有效的参考 Path
重规划；只有硬截止前正在执行的 yaw 对齐才保持 `true` 并顺延轨迹时间。

单独启动：

```bash
ros2 launch scan_controller scan_controller.launch.py
```

默认 `vx/vy/wz` 上限为 `0.30/0.15/0.45 m/s、m/s、rad/s`，变化率上限为
`0.50/0.40/1.00 m/s²、m/s²、rad/s²`。这些值低于当前 `pct_multifloor`
locomotion policy 的训练命令范围，面向 Go2-X5 携臂收纳姿态的第一阶段
保守验收，不代表实机上限；包级终点捕获默认另收紧到 `0.25 rad/s` 和
`0.50 rad/s²`，生产 overlay 为 `0.45 rad/s` 和 `1.00 rad/s²`。moving final
的 terminal yaw 控制死区为 `0.18 rad`，严格
窄于 `0.20 rad` 完成门；误差落在 `0.18~0.20 rad` 时仍继续原地纠偏，避免
RL 稳态停在验收边界。该控制余量不收紧完成认证门，stationary final hold
也不进入反馈分支，仍全程严格零速。首次捕获先连续 `0.06 s` 发布严格三轴全零，使下游
`cmd_vel_to_policy` 的独立变化率历史确定清零；之后只有在 terminal yaw 尚未
收敛或机体漂出严格位置门时才启用有界位置保持。yaw 与位置合格后该补偿必须
重新严格为零。若 terminal yaw 与完成速度门已经稳定但机体漂出捕获内门、仍在
释放外圈内，只恢复 SCAN B-spline 平移闭环，并继续锁定完整 Path terminal yaw。
最终轨迹允许
3.0 秒 policy 末端收敛余量，航向冻结预算最多 6.0 秒；默认到达门限为
XY `0.08 m`、Z `0.12 m`、完整同代 Path 最后 Pose 的 terminal yaw
`0.20 rad`、平面速度
`0.05 m/s`、世界竖直速度 `0.05 m/s`。moving final 尚未进入捕获内门时另要求
三轴角速度范数不超过 `0.10 rad/s`；一旦进入零平移终点捕获，便与 stationary
final hold 一样使用与 `cmd_vel.wz` 同语义的机体系 `|wz| <= 0.10 rad/s`。
这避免四足站立策略的 roll/pitch 微摆反复清零驻留，同时仍由 yaw 误差和
yaw rate 阻止转向中误报完成。
目标收口要求最终轨迹、轨迹未过期，且
Odometry 与 PointCloud2 都新鲜有效；缺点云、未来/超时点云时间戳或任一物理门限
超差都不得锁存完成。moving final 正常走完名义时间时沿用原完成门；若提前进入
位置门并冻结名义时间，则必须让全部严格物理门连续满足 `0.50 s` 才可完成，
避免冻结时间死锁。点云超时会清零 moving capture 与 stationary final hold 的
连续驻留时间，恢复新鲜点云后必须重新完整驻留；超过软/硬截止同样持续发布零速度。

`/planning/goal_reached` 使用 reliable + transient-local 发布。有效的新代际
非空 Path 或合法空 Path 会立即清除上一目标的完成锁存；一旦同代 Path
已经发布 `GOAL_REACHED`，后到的任何同代 B-spline 都只能被拒绝并持续全零，
不能恢复运动。无效 Odometry、点云或 B-spline 触发的安全停车保留已经锁存的
`true`，避免产生瞬时 `false -> true` 毛刺。Path 与 B-spline 跨 topic 乱序
时，只要原始纳秒代际严格相同，后到的首条 Path 不会抹掉该 B-spline 自身的
final 语义。Isaac OGN 订阅端刻意采用 reliable + volatile +
depth 1，并由 executor 检查 fresh false、有效运动、fresh true 和到达后
零速；Bool 本身仍不携带 Header 或 goal id。executor 不计与 true 同一
时间戳的写入，只接受后续 write sequence、仿真 step 和时间戳都连续递增的
正常零速，或唯一原因为 `point_cloud_timeout` 的零速；环境终止、预测碰撞、
时钟回退和混合原因均拒绝。

历史连续斜坡 v24 中，controller 曾在点云超时期间由旧终点门锁存
`514:false@10.34 s -> 515:true@10.36 s`。锁存样本的 XY/Z/yaw 误差为
`0.03256 m / 0.03881 m / 0.03106 rad`，平面、垂向和角速度为
`0.02086 m/s / 0.01005 m/s / 0.02453 rad/s`；executor 随后验证 5 个
连续零速，且没有 post-goal 非零写入。
该运行不仅早于严格 Path/B-spline identity、任意 yaw 包络、全剩余轨迹碰撞和
有序 corridor 加固，也依赖现已删除的“过期点云 + 新鲜 Odometry”收口语义；
它只是历史行为记录，不是当前 fail-closed 安全 build 的验收证据。
