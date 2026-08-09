# navigation_supervisor

该 ROS 2 包负责把 PCT 全局规划、SCAN 局部规划和闭环控制器的类型化状态
汇总为 `/navigation/status`，并在控制器已经证明停车后，通过
`/pct/planning_command` 发起有界、幂等的 PCT 重规划请求。

节点订阅：

- `/body_pose`、`/cloud_registered`
- `/pct/global_path`、`/pct/planning_status`
- `/planning/bspline`、`/planning/scan_status`
- `/planning/controller_status`

节点发布 `/navigation/status`，但**不会发布 `/cmd_vel`**。状态中的
`force_zero_velocity` 和 `allow_tracking_command` 会经 Isaac OGN subscriber
进入现有唯一 `cmd_vel_to_policy` writer，并在身份与新鲜度检查后执行实际
policy 速度门控；状态无效或要求强制停车时 writer 写入零速度。该链没有新增
第二个 `/cmd_vel` 发布器、Twist gate 或 policy writer。

仿真运行必须使用 `use_sim_time=true`。Path、B-spline、SCAN status 和
controller status 会按完整时间戳身份与单调序列对账。任何时钟回拨都会
锁存 fatal epoch reset；由于 PCT、SCAN 与 controller 也保留自己的代际，
此时必须重启完整导航 ROS 图，不能在单个 supervisor 进程内恢复。

全局规划使用两层独立上限：`timeouts.global_planning_sec`（默认 15 秒）
限制初始规划，以及每次故障重规划服务 ACK 后等待匹配 PCT status 与真实
Path 的时间；`limits.max_global_replan_cycles`（默认 3）限制同一故障链中
被 PCT 接受的连续重规划周期。超时或周期用尽都会锁存终止停车，同一 goal
的迟到结果不能恢复执行，必须由新 goal 或显式 cancel 开启新生命周期。
服务调用内部的 `limits.replan_max_attempts` 只是同一固定请求的传输重试，
不会额外消耗全局重规划周期。

`/navigation/status` 在状态变化时立即发布；稳定状态也按
`status.heartbeat_sec`（默认 0.10 秒）刷新 header 时间与单调 sequence，
确保唯一 policy gate 的 freshness 监控不会因 supervisor 去重静默而误停。

```bash
source /opt/ros/humble/setup.zsh
source install/setup.zsh
ros2 launch navigation_supervisor navigation_supervisor.launch.py
```

生产主线通常由 `isaac_navigation_bridge` 包中的
`pct_scan_navigation.launch.py` 一并启动。

## 真实 DDS 协议探针

以下四个 opt-in probe 使用真实 `rclpy` 节点和 DDS endpoints，不会在普通
包级 pytest 中自动占用 UDP discovery。指定的 `ROS_DOMAIN_ID` 会连续使用
四个独立 domain，因此基准值必须位于 `0..229`：

```bash
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=218
export RUN_NAVIGATION_SUPERVISOR_DDS_PROBES=1

python3 -m pytest -q \
  src/navigation_supervisor/test/test_dds_protocol_probe.py \
  -k late_join
python3 -m pytest -q \
  src/navigation_supervisor/test/test_dds_protocol_probe.py \
  -k replan_service
python3 -m pytest -q \
  src/navigation_supervisor/test/test_dds_protocol_probe.py \
  -k 'pending_trajectory or tombstone'
```

`late_join` 用测试内的确定性正 ROS 时间隔离 volatile `/clock` 启动竞态，
并在所有 endpoint 已匹配、单线程 executor 的确定性消费条件下，验证
supervisor 启动前已经写入的 transient-local Path、B-spline、PCT、SCAN
与 controller 快照能够恢复为 `TRACKING`。DDS 不保证不同 topic/DataWriter
之间的全局顺序；节点会按 reference Path stamp 缓存提前到达的有效
B-spline/SCAN 状态，默认最多等待 2 秒、合计最多 64 条，并在匹配活动 Path
到达后重放。相同身份冲突、缓存过期、容量溢出、Path tombstone 或旧代污染
都会清理相应缓存并 fail closed，不能用迟到的同 stamp Path 复活。`/clock`
回拨仍会锁存 fatal epoch reset，必须重启完整导航 ROS 图。

`replan_service` 使用多线程 executor 和可重入的真实 service，故意延迟首
次响应直至 supervisor 发出第二次传输。探针要求两次请求的 CDR payload 与
`request_id` 完全相同、只形成一个已 ACK 的逻辑重规划周期，并在匹配的新
PCT Path/status 到达后回到 `LOCAL_PLANNING`。每个等待阶段都有硬超时。
