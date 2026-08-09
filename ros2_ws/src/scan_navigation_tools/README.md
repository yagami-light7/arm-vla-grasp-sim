# scan_navigation_tools

这个 ROS 2 Humble 包只为 SCAN `navi_mode=3` 验收发布手工
`nav_msgs/msg/Path`，不包含局部规划器、控制器或 PCT fallback。

默认发布端点：

- topic：`/initial_path`
- frame：`world`
- QoS：`reliable + transient_local + keep_last(1)`
- 时钟：强制 `use_sim_time=true`

`points_xyz` 是展平的 `[x0, y0, z0, x1, y1, z1, ...]`。这里的 `z`
始终表示地面高度；发布器不会增加 `body_height`，该转换只能由 SCAN
入口执行一次。每个 `PoseStamped` 与外层 `Path` 使用相同的非空 frame
和有效仿真时间戳，姿态是沿路径平面切线的单位 yaw 四元数。

## 启动

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch scan_navigation_tools manual_path.launch.py
```

如需换路径，复制
`share/scan_navigation_tools/config/manual_path.yaml`，修改后运行：

```bash
ros2 launch scan_navigation_tools manual_path.launch.py \
  config_file:=/absolute/path/to/manual_path.yaml
```

斜坡地面高度点列示例：

```yaml
points_xyz:
  - 0.0
  - 0.0
  - 0.0
  - 1.0
  - 0.0
  - 0.15
  - 2.0
  - 0.0
  - 0.30
```

可配置参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `topic` | `/initial_path` | Path 发布 topic |
| `frame_id` | `world` | Path 与所有 PoseStamped 的 frame |
| `points_xyz` | 三个平地点 | 展平的三维地面路径点 |
| `min_point_distance_m` | `0.02` | 移除相邻近重复点的三维距离阈值 |
| `startup_delay_sec` | `1.0` | 发布前等待的仿真时间 |

节点会在启动时校验 topic、frame、点数、数值有限性和距离阈值。去重后少于
两个点或任何参数非法都会直接终止。仿真时钟仍为零时不会发布，以免 SCAN
收到无效时间戳；路径成功发布一次后由 transient-local publisher 缓存。

Path 顶层 Header stamp 同时是本轮参考路径代际。SCAN planner 会把它复制到
`Bspline.reference_path_stamp`，controller 仅执行与当前 Path 同代的轨迹。
因此验收时不能在同一个仿真时刻伪造多代路径，也不能用旧 Path 的 B-spline
驱动新路径。相同几何 Path 的重复发布保持幂等；需要撤销路径时应发布具有
合法新时间戳和相同 frame 的空 Path，而不是停掉 publisher 后依赖缓存过期。

仓库还提供 `validation_ramp_path.yaml`、
`multifloor_stair_first_flight_path.yaml` 和
`multifloor_stair_two_step_path.yaml`。三者都只描述地面高度参考；其中
`validation_ramp_path.yaml` 已由真实坡道 v24 完成同链端到端验收，两份 stair
配置仍只是待验输入，单段楼梯尚未通过。

标准 Go2 + MoE-CTS 的独立确定性高度场使用
`go2_moe_cts_stair_path.yaml`。它按 live 点云确认的 `z=-2.47m` 低平台、
`0.19m` 级高和当前实际 `0.20m` 踏面编写，以水平踏面首尾点显式表示
支撑面，并以 `[-1,-1]` 禁用 Go2-X5 root-lock 冻结索引。该配置必须与
`go2_moe_cts_stair_navigation.launch.py` 成套使用；真实单跑验收完成前仍视为
待验输入。

实际 `0.30m` 踏面的单变量 A/B 使用
`go2_moe_cts_stair_wide_path.yaml`。Isaac Lab 高度场的水平分辨率为
`0.10m`，所以 Isaac 端要请求 `--stair-width 0.31`，经整数格离散后
得到 3 格、实际 `0.30m` 踏面。该路径共 25 点，匹配 9 次
`0.19m` 抬升和 `z=-1.71m` 低平台，必须与
`go2_moe_cts_stair_wide_fast_navigation.launch.py` 成套使用，不能与 33 点的
`go2_moe_cts_stair_path.yaml` 交叉组合。

## 测试

```bash
/usr/bin/python3 -m pytest -q -p no:cacheprovider \
  ros2_ws/src/scan_navigation_tools/test
```

纯函数测试不导入 ROS，覆盖点列校验、有限性、近重复点清理、地面高度保持、
切线朝向及 topic/frame 校验。
