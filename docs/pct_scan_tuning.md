# PCT + SCAN 导航调参说明

主线性能参数统一位于：

```text
ros2_ws/src/isaac_navigation_bridge/config/pct_scan_tuning.yaml
```

组合 launch 会让 `pct_ros2_adapter`、`scan_planner_node` 和
`scan_controller` 共同读取这一个文件。各包原配置保留独立启动时的安全默认值；
主线有效值以本文件为准。资产路径、topic、frame、机器人碰撞包络、传感器超时
仍由各自安全配置和 launch 合同管理。`navigation_contract.body_height_m` 虽然不是
性能旋钮，也集中放在本文件中：live runner 会读取它，并把同一个值传给 PCT、
SCAN、点云过滤、controller 与 Isaac 楼梯执行，避免高度合同漂移。

## 当前 fast-stable 起点

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `navigation_contract.body_height_m` | `0.338 m` | 实测 base_link 到碰撞支撑面的高度；不能凭感觉调 |
| `fsm.planning_horizon` | `1.20 m` | SCAN 每次保留的前向局部轨迹长度 |
| `fsm.thresh_replan` | `0.55 m` | 从本段真实起点前进多少后滚动重规划 |
| `fsm.reference_cruise_speed` | `0.52 m/s` | 非最终段的路径切向巡航速度 |
| `fsm.reference_velocity_filter_time_constant_sec` | `0.30 s` | 平滑四足步态造成的单帧 Odometry 速度振荡；不使用命令补速度 |
| `manager.max_vel` / `optimization.max_vel` | `0.55 m/s` | B-spline 速度硬上限 |
| `manager.max_acc` / `optimization.max_acc` | `0.80 m/s²` | B-spline 加速度硬上限 |
| `manager.feasibility_tolerance` | `0.0085` | 连续可行性检查的数值比例；与 0.55m/s 上限和 0.005 优化容差保持合同一致 |
| `manager.reference_profile_acceleration_scale` | `0.50` | reference 初始速度剖面比例；修复路径和初切线后已重新闭环验证 |
| `limits.max_vx` | `0.55 m/s` | controller 与 Isaac policy 的纵向速度硬上限 |
| `limits.max_ax` | `0.80 m/s²` | controller 与 Isaac policy 的纵向变化率硬上限 |
| `limits.max_yaw_rate` | `0.60 rad/s` | controller 与 Isaac policy 的偏航角速度硬上限 |
| `controller.time_forward` | `0.60 s` | 闭环跟踪前视时间 |
| `controller.kp_position` | `0.80` | 位置误差反馈强度 |

2026-08-02 至 2026-08-03 的平地物理对照均使用原 Go2-X5 locomotion
checkpoint，并完成严格 `GOAL_REACHED`：

| 版本 | 仿真导航时间 | 终点 XY 误差 | B-spline 数 | 平均速度上界 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 旧高度 `0.300 m` | `47.82 s` | `0.0773 m` | 11 | `0.213 m/s` | 稳定但慢 |
| 高度修正 `0.338 m` | `34.60 s` | `0.0583 m` | 10 | `0.286 m/s` | 高度修正基线 |
| 高度修正 + profile 比例 `0.50` | `39.98 s` | `0.0752 m` | 11 | `0.342 m/s` | 数学轨迹更快，闭环反而退化 |
| PCT 同层安全直线后处理 | `27.56 s` | `0.0624 m` | 10 | `0.343 m/s` | 去掉 A* 栅格锯齿 |
| 同层直线 + `0.30 s` 实测速度滤波 | `26.90 s` | `0.0612 m` | 9 | `0.347 m/s` | phase206 |
| 直接注入 semantic 首切线速度（已撤回） | `32.92 s` | `0.0712 m` | 10 | `0.274 m/s` | 时间重分配反向拉长轨迹 |
| pristine GridMap + PCT 预加载，`0.45 m/s` 上限 | `17.10 s` | `0.0663 m` | 8 | `0.426 m/s` 实测峰值 | phase216 |
| `0.47 m/s` 巡航 + `0.50 m/s` 上限，seed 0 | **`15.82 s`** | `0.0672 m` | 8 | `0.495 m/s` 轨迹上界 | phase219 |
| 同配置，seeds 1/2 | `15.86 / 15.86 s` | `0.0620 / 0.0674 m` | `8 / 8` | 同一路径 | phase219 多 seed |
| `0.52 m/s` 巡航 + `0.55 m/s` 上限，seed 0 | **`14.98 s`** | `0.0635 m` | 8 | `0.545 m/s` 轨迹上界 | phase220 |
| 同配置，seeds 1/2 | `14.94 / 14.92 s` | `0.0640 / 0.0671 m` | `8 / 8` | 同一路径 | phase220 多 seed |
| 同配置 + 严格快速预检，seed 0 | **`14.56 s`** | `0.0700 m` | 8 | 路径不变 | phase221 |
| 同配置 + 严格快速预检，seeds 1/2 | `14.34 / 14.48 s` | `0.0686 / 0.0673 m` | `8 / 8` | 同一路径 | phase221 多 seed |

`body_height=0.338 m` 已获得真实 Isaac 验证；随后同层安全直线、实测速度滤波、
pristine GridMap 重心化和 PCT 后端预加载共同把 seed 0 从 `47.82 s` 降到
`17.10 s`。phase219 再把巡航/硬上限成套提高到 `0.47/0.50 m/s`，seed 0 降至
`15.82 s`，相对 phase216 缩短 `7.5%`。seeds 0/1/2 平均 `15.847 s`，极差仅
`0.04 s`，平均终点误差 `0.0655 m`；三轮均为 8 条 B-spline、0 急停、0 轨迹
拒绝、到达后 0 次非零 policy 写入。三轮全局 Path 哈希一致，说明提速来自局部
轨迹与执行参数，不是 PCT 换了更短路线。

phase220 将巡航/硬上限继续成套提高到 `0.52/0.55 m/s`，并把连续可行性容差
收紧到 `0.0085`。seeds 0/1/2 分别为 `14.98/14.94/14.92 s`，平均
`14.947 s`、极差 `0.06 s`，比 phase219 平均缩短 `5.68%`；平均终点误差
`0.0649 m`，仍为每轮 8 条 B-spline、0 急停、0 候选拒绝、目标后 0 非零
policy 写入。27 点全局 Path 哈希继续保持
`6a5536a6c723cf1110f59157257083eb301cb527286012a04e2c7457623895c4`。

phase221 没有再提高运动速度，而是把原 51 样本、`1.00 s` 的静稳预检改成
“26 样本、`0.50 s` 严格快速窗口 + 原完整窗口回退”。快速窗口还额外要求
高度 MAD 不超过 `0.0025 m`、P95-P05 不超过 `0.0075 m`，且实测高度与配置
差不超过 `0.020 m`；第一次快速认证失败后不会继续反复尝试，而是只等待完整
窗口。seeds 0/1/2 均一次通过，时间为 `14.56/14.34/14.48 s`，平均
`14.460 s`，比 phase220 平均缩短 `0.487 s`（`3.26%`）。三轮均为 8 条
B-spline、0 急停、0 候选拒绝、目标后 0 非零写入，终点误差仍小于 `0.08 m`。

`reference_profile_acceleration_scale=0.50` 在 phase204 的旧路径/旧初切线实现上
曾把闭环从 `34.60 s` 退化到 `39.98 s`，所以当时撤回。phase211 修复同层路径、
亚栅格初切线和时间参数化后重新实测为正收益，phase219 又完成 3-seed 严格
闭环，因此当前生产值为 `0.50`。这也说明参数结论必须绑定代码和路径版本，不能
只看离线 B-spline 时长，也不能把旧实验结论永久套到新实现。

旧 DWA seed 0 隔离记录为 `13.94–13.96 s`，但口径更宽松：`0.18 m` 内立即完成，
不要求停稳，也没有 SCAN 的 `2.02 s` body-height/机械臂静稳预检和 `0.50 s`
终点驻留。DWA 完成时距目标 `0.178 m`，机体仍以约 `0.231 m/s` 平移、
`0.253 rad/s` 转动。SCAN 的 phase220 原始总时间均值仍比该数字多约 `1.01 s`，所以项目
状态继续保守记为“尚未在同口径实测中超过 DWA”；若只扣除 SCAN 独有的
`2.02 s` 预检，phase220 均值为约 `12.93 s`，且它还完成了 8cm 内停稳和驻留。
这项归一化只能解释差异，不能替代真正同终止合同的 DWA 重跑。

另外，`pct_scene` 历史任务的 pick 底盘目标为 `y=6.7491`，当前生产任务为
`y=6.6691`，相差 `0.08 m`。这两个结果也不能直接作为局部规划器公平对照。
正式比较 SCAN 与 DWA 时，应固定同一任务端点、地图、PCT 后端，并记录一份完全
相同的 PCT `Path` 分别回放给两个局部规划器；生产任务无需为了迁就旧基线而改回
历史站位，隔离基准应复用当前生产输入。

不要把滤波后的完整前向速度直接设为 B-spline 起始导数。phase207 实测中，
起始导数虽由近零恢复到 `0.23–0.34 m/s`，现有时间重分配器却把单段时长从约
`3.8–4.2 s` 拉到 `4.8–5.9 s`，轨迹速度上界降到约 `0.24–0.29 m/s`，总时间
退化到 `32.92 s`。生产代码已撤回该注入；后续应修改轨迹参数化或重规划距离，
不能用更大的起始速度硬顶。

## 必须保持的参数关系

```text
0 < fsm.thresh_no_replan
  < fsm.thresh_replan
  < fsm.planning_horizon

fsm.reference_cruise_speed
  <= manager.max_vel
   = optimization.max_vel
  <= scan_controller limits.max_vx

manager.max_acc
  = optimization.max_acc
  <= scan_controller limits.max_ax（建议保持相等）
```

`manager.planning_horizon` 是异常轨迹长度检查，不是 SCAN 的真实前视距离；建议
至少为 `2 × fsm.planning_horizon`，避免把 5 cm 密采样的合法局部 guide 误判为
异常长轨迹。

## 怎样加速

先确认 `body_height_m` 来自静止标定，轨迹开头不再反复出现假升降；随后再改
`fsm.reference_cruise_speed`，每次最多增加 `0.02 m/s`：

```yaml
scan_planner_node:
  ros__parameters:
    fsm.reference_cruise_speed: 0.52
```

当前 Go2-X5 携臂收纳姿态已经用 seeds 0/1/2 跑通 `0.52 m/s` 巡航请求与
`0.55 m/s` 硬上限。三项硬上限必须一起修改，并同步检查
`manager.feasibility_tolerance` 与优化器速度容差的数值合同：

```yaml
scan_planner_node:
  ros__parameters:
    manager.max_vel: 0.55
    optimization.max_vel: 0.55

scan_controller:
  ros__parameters:
    limits.max_vx: 0.55
```

当前巡航比硬上限低 `0.03 m/s`，给转弯、落脚扰动和闭环反馈保留余量。若继续
加速，先完成转弯与动态障碍回归；当前 `0.55 m/s` 已到该 checkpoint 的训练
纵向命令边界，不应再靠提高硬上限追数字。下一步优先优化可回退的预检等待，
同时保留 51 样本完整校验作为失败回退。楼梯段仍由既定底盘冻结逻辑处理，
不能用平地提速结果宣称纯物理上楼稳定。

`manager.reference_profile_acceleration_scale` 只控制 B-spline 初始拟合速度剖面，
不改变最终物理限幅。当前生产值为 `0.50`，它已绑定当前路径压缩、初切线和时间
参数化实现完成 3-seed 验收；换回旧实现或修改加速度上限后必须重新 A/B。

长直路仍频繁生成短轨迹时，可以把 `fsm.planning_horizon` 调到
`1.2–1.5 m`；动态障碍密集或急转弯多时使用 `0.8–1.2 m`。重规划距离通常取
前视距离的 `30%–45%`：值小响应更快、计算更频繁；值大轨迹更连贯，但对新障碍
反应更迟。不要只增大前视距离而不检查有序走廊、转角和动态障碍恢复。

## 怎样提高稳定性

- 直线左右摆动：先将 `fsm.reference_cruise_speed` 下调 `0.02 m/s`；若命令仍
  高频摆动，再把 `controller.kp_position` 从 `0.80` 降到 `0.70–0.75`。
- 转角冲出 PCT Path：把 `fsm.planning_horizon` 降到 `0.8–1.0 m`，并把
  `fsm.thresh_replan` 保持在其 `30%–40%`；不要放宽有序走廊偏差门。
- 经常进入原地 yaw 对齐：先检查 PCT Path 是否锯齿、B-spline 切向是否连续；
  不要首先提高 yaw 阈值掩盖路径问题。
- 跟踪落后但轨迹平滑：小幅增加 `controller.time_forward`，每次不超过
  `0.05 s`；若出现过冲立即回退。`kp_position` 和前视时间不要同时大幅增加。
- 动态障碍恢复慢：减小 `fsm.thresh_replan`，不要通过缩小双圆柱包络或障碍
  膨胀距离换取速度。

PCT 的常用参数是：

- `planner.path_sample_spacing_m`：建议 `0.15–0.25 m`。太小会把地图锯齿完整
  传给 SCAN，太大会丢失楼梯平台和急转角。
- `planner.grid_compress_max_segment_m`：建议 `0.60–1.00 m`。只简化可直连长段；
  出现抄近路时应降低，而不是放宽 SCAN 走廊。
- `planner.obstacle_clearance_radius_m`：只作用于兼容后端；当前生产 upstream
  后端使用下面的机身高度带净空 overlay，不要误以为修改此项会改变生产路线。
- `planner.upstream_astar_step_cost_weight`：当前 `0.20`。受跟踪的 native A*
  补丁已经移除旧 `cost<5` 代价死区，所以此值现在会连续权衡路长与净空代价；
  增大后更偏向高净空但通常更绕、更慢，减小后更接近几何最短路。
- `planner.upstream_body_clearance_radius_m`：当前 `0.80 m`，表示静态障碍开始
  产生软代价的影响半径，不是 SCAN 的 `0.27 m` 圆柱半径，也不是硬通行保证。
- `planner.upstream_body_clearance_maximum_cost` / `power`：当前 `20.0 / 2.0`。
  前者决定贴近机身高度障碍时的最大代价，后者决定代价随距离衰减的形状；楼梯
  标定走廊会受单独保护，避免净空 overlay 破坏唯一跨层 gateway。
- `planner.upstream_same_layer_shortcut_clearance_m`：当前 `0.27 m`，表示官方
  upstream A* 的平地折线直线净空足够时，用 Go2-X5 单个圆柱半径做保守视线
  简化；检查还会加入半个栅格对角线，并同时避开机身高度带障碍表面。同层路径
  可整体处理；跨层路径只处理楼梯 profile 前后的两段平地，7 个楼梯 anchor
  原样保留。SCAN 的 `0.27+0.16=0.43 m` 任意航向碰撞门仍是最终安全门。
- `planner.upstream_same_layer_shortcut_max_segment_m`：限制每条平地捷径的最大
  长度，防止在超长地图上用一次稀疏检查替代正常全局搜索。减小会保留更多
  A* 折点、通常更慢；增大仍不能绕过净空检查。
- `planner.upstream_stair_profile_path`：跨层路径使用的场景级楼梯中心线合同。
  它不是提速参数，也不替代 PCT 的楼层选择和楼面选路；PCT 原生 A* 完成后，
  adapter 只把已经匹配到的楼梯区间对齐到 7 个实测 ground anchor。profile、
  tomogram、collision PLY 和坐标变换的 hash 任一不一致都会失败关闭。
- `planner.upstream_stair_profile_match_tolerance_m`：原生路径与楼梯首尾 anchor
  的最大匹配误差，当前为 `0.60 m`。这是防止把中心线套到错误路段的安全门，
  不是障碍净空或允许偏离楼梯的距离；不要把它作为速度旋钮放大。

当前 seed=0 路线的原生 A* 折线曾有 51 个发布点、长 `5.332 m`，比起终点直线
多 `4.28%`，并包含 45° 栅格折角。地图复核显示直线到最近障碍中心为
`0.463 m`；Go2-X5 半径 `0.270 m` 加 0.20m 栅格半对角 `0.141 m` 的保守要求为
`0.411 m`，因此同层直线安全。修正后原始 27 个 A* anchor 被安全压成 2 个，
再按 `0.20 m` 间距发布成 27 点、`5.114 m` 的直线 Path。phase225 以前跨层
路径不做平地 LOS 压缩；该历史结果仍用于解释旧 Path 点数。PCT 始终决定
layer `8→15` 的楼层拓扑，adapter 只规范化已匹配楼梯区间，并从 phase226 起
安全压缩其前后平地折线。phase220 的真实 ROS 组合复核中，SCAN 原数接收当时
的 170 点 Path，
沿原始三维折线生成 35 控制点的首条局部 B-spline 和非零 `/cmd_vel`；它没有
改写全局路线。静态楼梯 Isaac 回归仍收到与 phase218 完全相同哈希的 47 点、
`8.784582 m` Path，并达到 100% 冻结进度。该中心线合同影响稳定性与可比性，
不代表 SCAN 计算本身更快。

phase221 用当前任务精确端点重新计算 pick-to-place：PCT 返回 175 点、
`23.236705 m`，逻辑层为 `8→15`，高差 `3.217233 m`；请求起终点 XY 误差和
7 个楼梯 anchor 最大 XY 误差均为 `0`。这证明当前端点的全局几何合理，但该
175 点路线目前仍是离线官方后端证据；还需在完整 pick/place 实跑中验证 SCAN
在线接收和执行。平地快速预检的 Path 哈希由 phase220 的 `6a5536…` 变为
`b1fc86…`，原因是 PCT 规划时刻提前 `0.50 s`，实测起点相差约 `0.00011 m`；
两者仍同为 27 点、`5.103 m` 的安全直线，phase221 三个 seed 的新哈希完全一致，
不能把这种浮点级起点变化误判为 PCT 改走另一条路线。

phase224 在生产 upstream 地图上加入机身高度带净空 overlay 后，当前跨层请求变为
188 点、`24.297071 m`。这比 phase221 长约 `1.06 m`，是 PCT 主动避开 phase223
墙边窄路的结果，不是 SCAN 改写全局路径。离线双圆柱审计确认楼梯接管前的发布
中心线具有正余量；二楼精确任务终点在指定 terminal yaw 下仍贴近静态障碍，因此
SCAN 末段新增“严格到达门内安全捕获点”：保持任务终点和 yaw 身份不变，只允许
局部 B-spline 在 `0.04 m` 末端目标门内选择有连续后向支撑区的自由点停车。

phase225 完整跨层 carry 实测没有形成有效耗时结果。机器人约在仿真 `44.5 s`
到达 `(0.789, 4.954)` 后，SCAN 对下一局部目标 `(1.50, 5.59)` 的 A* 每次约
`0.16–0.18 s`，但实际机身已经触及楼梯入口障碍膨胀区，随后进入安全停车与
PCT 重规划循环。运行在 12000 个导航 tick（约 `240 s` 仿真时间）后以
`nav_to_place_timeout` 结束，共观察到 23 代 live Path。这个 `240 s` 是失败等待
上限，不是 SCAN 规划计算时间，也不能与 DWA 完成时间比较。

本轮真正的门限冲突位于楼梯冻结配置：生产路径提取出的扩展楼梯组件起点为
`(0.997, 4.864)`，实测停点距它 `0.226 m`；旧
`activation_radius_m=0.12 m` 要求机器人继续靠近后才接管，但 SCAN 会更早按
双圆柱碰撞停车。生产 profile 已把接管半径改为 `0.35 m`，保持
`approach_distance_m=1.50 m` 和 `activation_lookahead_m=1.60 m` 不变。用同一
135 点重规划 Path 和真实停点离线回放后，动作从普通 SCAN 零速变为
`scan_stair_freeze_activated`。该参数位于
`configs/navigation/scan_stair_freeze_go2_x5_multifloor_v1.json`：

- `approach_distance_m` 决定从检测到的楼梯几何向入口前延长多少路径；
- `activation_radius_m` 才是机器人距离扩展组件多近时真正切换到底盘冻结；
- `activation_lookahead_m` 只定义接近窗口和超时观察范围，不会单独触发冻结。

上述修复尚未启动第二次 Isaac；完整跨层到达、实际完成时间和是否超过 DWA 都
继续保持未验证。

### phase226 跨层平地折点压缩与首段速度恢复

phase225 日志进一步排除了“SCAN A* 计算太慢”这个解释。失败前正常 TRACKING
约 `42.4 s`，其中约 `10.8 s` 是原地 yaw，对外平移约 `31.2 s`；平移命令均值
只有 `0.202 m/s`。同一生产参数下，phase221 平地 8 条 B-spline 的速度上界
基本为 `0.50–0.545 m/s`，而 phase225 跨层前 8 条中有 6 条只有
`0.196–0.231 m/s`，时长达到 `6.39–8.52 s`。原因是跨层 PCT 输出保留了楼梯
前后每一个 A* 栅格折点，SCAN 又按安全合同严格保持这些有序折点，导致局部
轨迹反复为转角降速。

upstream adapter 现在对跨层路径分成“起点楼面、标定楼梯、目标楼面”三段：
只对两端楼面调用与同层路径相同的视线压缩，楼梯 7 个标定 anchor、请求精确
起终点和 layer `8→15` 拓扑完全不变。视线段必须同时避开原 tomogram 阻塞单元
和 body-height overlay 的障碍表面，并继续加入 `0.141 m` 半栅格对角余量；
检查失败的相邻原生 A* 段原样保留，不会为了减少点数硬拉直。

对 phase225 精确起终点的离线复算得到：发布 Path 从 `188` 点降为 `142` 点，
内部折线控制点从 `79` 降为 `32`，长度从 `24.297071 m` 降为
`23.704265 m`；累计折线转角从 `16.543 rad` 降为 `9.799 rad`，大于 20° 的
折角从 16 个降为 10 个。PCT 自身规划耗时由约 `0.16 s` 增至约 `0.35 s`，这是
一次性的约 `0.2 s` 净空检查开销，不是导航执行耗时。

无 Isaac 的真实 ROS 2 多进程探针使用 phase225 精确生产起终点和统一
`body_height=0.338 m`，收到 142 点 `/pct/global_path`、35 控制点三阶
B-spline 和非零 `/cmd_vel`。首条局部轨迹从 phase225 live 的
`8.524156 s / 0.207441 m/s` 恢复为 `3.002090 s / 0.544554 m/s`，并新增
`duration<=3.20 s`、`velocity>=0.50 m/s` 的自动回归门。这个结果证明路径折点
造成的首段降速已经消除；它仍是合成自由空间传感器探针，不替代 RL policy、
PhysX、楼梯冻结释放和完整到达耗时。遵照用户要求，本阶段没有再启动 Isaac；
下一次单次 live 应在用户允许后验证 `activation_radius=0.35 m` 与整条新 Path。

### phase227 全路线平地窗口验收与受限转角平滑

phase226 只验了第一条局部轨迹，不能代表 23.704 m 路线上后续每一个转角。
新的 `probe_crossfloor_scan_floor_windows.py` 直接调用同一个生产 upstream 后端，
确认本轮仍是 142 点、`23.704265 m`、layer `8→15` 且包含全部 7 个楼梯 anchor，
再按 0.65 m 间距把楼梯接管区外的 22 个平地位置逐一送入真实 SCAN/controller
ROS 2 节点。输入使用显式自由射线和理想 Odometry，因此它是确定性的几何与
动力学验收，不是 Isaac、RL policy 或真实障碍证据。

同一组 22 个位置只切换
`manager.reference_free_guide_refine_enabled` 得到严格 A/B：关闭时局部轨迹平均
`4.133762 s`、最长 `7.659532 s`、最低速度上界 `0.188562 m/s`；开启时分别为
`2.441531 s`、`3.022893 s` 和 `0.469849 m/s`。平均时长下降 `40.94%`，最长
时长下降 `60.53%`，最差位置的速度上界提高约 `2.49` 倍。22 条轨迹全部满足
`velocity>=0.45 m/s`、`duration<=3.20 s`，最大参考折线偏差为 `0.032035 m`，
明显小于不变的 `0.10 m` ordered-reference 走廊。

这里的“平滑”只处理空旷 guide 中 45°–90° 转角导致的整段动力学拉长。候选
必须重新通过当前 GridMap 双圆柱碰撞、参考顺序、最大 `0.035 m` 进度超前、
完整速度/加速度和最小时长收益门；任一失败都继续发布原始安全折线。它不会
修改 `/pct/global_path`，也不会提高 `0.55 m/s`、`0.80 m/s²` 上限。L-BFGS
在近直线、已接近极小值时可能以舍入限制提前结束；只有当前点与代价均为有限值
时才保留为候选并记警告，随后仍需通过上述全部独立安全门，其他求解错误直接
拒绝。

这组结果说明当前 PCT 全局路径在“SCAN 负责的平地区段”已经不存在旧版
`0.19–0.23 m/s` 的隐藏慢窗口。楼梯段仍按用户确认的 0.35 m 接管线进入底盘
冻结，终点最后 1.30 m 由独立到达/驻留合同验收，二者没有被混入平地速度门。
本阶段没有启动 Isaac，完整跨层完成时间和是否超过 multi-floor DWA 仍未验证。

### phase265 真实耗时拆分与偏航单变量 A/B

`analyze_pct_scan_live_timing.py` 直接读取 fresh run 的 `frames.jsonl`、
`summary.json` 和楼梯冻结前的 `ros2_launch.log`，把一楼导航拆成启动握手、平移、
原地转向和零命令四部分。它不会把楼梯冻结、二楼、终点驻留或 place 混进一楼
数字，也不会把整个 pipeline 时间误叫成 SCAN 规划时间。

对 phase244 与 phase264 的原 Go2-X5 checkpoint 日志重新计算后：一楼 SCAN
控制段由 `30.20 s` 降到 `27.20 s`，缩短 `3.00 s`（`9.93%`）；原地转向由
`15.00 s` 降到 `12.40 s`，缩短 `2.60 s`（`17.33%`），平移时间仅从
`15.00 s` 变为 `14.80 s`。同一阶段所有成功 SCAN 规划的累计墙钟时间分别只有
`0.5870 s` 和 `0.5981 s`。因此当前可见瓶颈是频繁/受限的机体转向，不是
B-spline 优化器每次算得慢。

复算旧日志时必须显式给出当时的偏航上限；fresh runner 新生成的运行会自动携带
不可变配置快照，分析器会优先读取快照：

```bash
python3 scripts/navigation/analyze_pct_scan_live_timing.py \
  --configured-max-yaw-rate 0.60 \
  outputs/pct_scan/live_crossfloor_carry_phase244_terminal_min_gait_seed0_v1 \
  outputs/pct_scan/live_crossfloor_carry_phase264_mainline_terminal_seed0_v1
```

生产 YAML 目前仍保持 `limits.max_yaw_rate=0.60 rad/s`。`0.75 rad/s` 只是一份
待 Isaac 验证的单变量实验，配方位于
`configs/navigation/pct_scan_yaw_rate_075_experiment.yaml`。配方生成器只允许覆盖
基础 YAML 中已经存在的叶子、拒绝拼错键和覆盖已有输出，并报告完整配置哈希及
唯一语义差异：

```bash
python3 scripts/navigation/materialize_pct_scan_tuning_variant.py \
  --recipe configs/navigation/pct_scan_yaw_rate_075_experiment.yaml \
  --output /tmp/pct_scan_tuning_yaw_rate_075_fresh.yaml
```

GPU 恢复后的顺序固定为：先用未修改的生产 YAML 重跑 seed 0，确认二楼末段严格
`GOAL_REACHED` 和 place；然后才用上面生成的完整 YAML 做同 seed A/B：

```bash
python3 scripts/navigation/run_pct_scan_live_acceptance.py \
  --mode crossfloor_carry \
  --seed 0 \
  --ros-domain-id 225 \
  --isaac-python /data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  --tuning-config-file /tmp/pct_scan_tuning_yaw_rate_075_fresh.yaml \
  --output-dir outputs/pct_scan/live_crossfloor_carry_yaw075_seed0_fresh
```

每个 fresh 输出都会逐字节复制一份 `pct_scan_tuning_snapshot.yaml`，记录 SHA256
和关键参数，并在 pipeline 结束时确认源文件与快照均未被中途修改。只有全局 Path
身份、终止门、楼梯冻结 profile 和 checkpoint 均相同，且 `0.75 rad/s` 在末端
误差、急停、横滚/俯仰、碰撞与 place 上不回归时，才允许考虑把实验值提升为
生产值；单看一楼快几秒不能改主线。

### phase267 完整 Path 快照与公平 DWA 输入合同

旧 `frames.jsonl` 为了避免一条百点 Path 在数千控制 tick 中反复写入，会在普通
诊断帧中删除 `points_ground_xyz`，只保留 Path 哈希。phase264 因此能证明 SCAN
实际消费的路线身份，却不能把完整点列原样交给隔离 DWA。不能根据同一个哈希
重新调用 PCT 后端补造路线：机器人实时起姿、发布时间戳或后端代码只要变化，
补造结果就不再是那次运行的精确输入。

记录器现在额外生成 `episode_000000/navigation_path_snapshots.jsonl`。每个 ROS Path
代际只写一次完整 ground-height 点列、terminal yaw、topic、frame、stamp、sequence
和 Path SHA256；逐帧 `frames.jsonl` 仍保持紧凑。记录器先用小型代际身份去重，
不会在每个控制 tick 重新序列化整条路线。episode summary 同时记录快照数量和
绝对路径。

一次生产 `crossfloor_carry` 完整通过后，用下列命令生成不可覆盖的公平对照合同：

```bash
python3 scripts/navigation/export_planner_comparison_contract.py \
  --run-dir outputs/pct_scan/<fresh_scan_seed0> \
  --output outputs/pct_scan/<fresh_scan_seed0>/planner_comparison_contract.json
```

导出器会重新计算完整点列哈希、快照文件哈希和合同自身哈希，并强制检查：

- 原 Go2-X5 mobile-manipulation task 与 checkpoint；
- 同一 seed、任务、场景、collision PLY、`body_height=0.338 m`；
- 同一 PCT Path 完整点列与 terminal yaw；
- 两边统一 `0.08 m` 终点、yaw、停稳速度和 `0.50 s` 驻留；
- 两边统一用户确认的 `chassis_root_lock` 楼梯段；
- 到达后连续零速，且完整 pipeline 仍必须进入 place；
- 至少 3 个 seed、成功率不回退，SCAN 平均受控导航时间低于 DWA 才能称为超过。

主指标只比较局部规划器实际控制 policy 的一楼和二楼区间；两边共同的楼梯冻结
时间、启动建场和 manipulation 不混入局部规划器胜负。同时必须另报完整导航阶段、
楼梯冻结、规划计算、原地 yaw 与末端捕获时间，不能把排除项藏起来。DWA 只在旧
分支/worktree 读取该合同运行，本 `pct-scan` 主线不增加 DWA 开关或 fallback。

phase264 历史失败运行只有哈希摘要，导出器会明确拒绝并要求 fresh run；它不会
伪造完整点列。GPU 恢复后的生产 carry seed 0 将自动产生新快照，严格到达 place
底盘交接点并保持零速后才有资格成为 DWA 回放源；实际释放仍由随后完整
pick→nav→place pipeline 单独验收。

### phase269 生产参数实测结论

GPU 恢复后，生产 `max_yaw_rate=0.60 rad/s` 的 cross-floor seeds 0、1 已严格
通过。两次完整导航仿真时间为 `107.80 s`、`109.24 s`，但都包含共同的楼梯
底盘冻结；不能把这个数字称为 SCAN 规划耗时。seed 0 的分段分析中，局部规划器
实际控制的一楼加二楼约 `43.88 s`，22 次成功 SCAN 优化累计墙钟仅
`1.5524 s`。因此当前主要耗时仍是机器人沿轨迹平移、转向和末端稳定，不是
B-spline 求解器一直在计算。

`0.75 rad/s` 单变量实验没有通过：机器人在楼梯入口附近更早到达
`(0.80, 4.87, 0.21)` 一带，局部候选连续被支撑占据门拒绝；规划失败达到阈值与
楼梯冻结接管发生在相邻控制拍，supervisor 启动 PCT replan 并清除旧 Path，严格
失败原因为 `scan_reference_path_cleared_during_stair_freeze`。这不是可以忽略的
“慢一点就好”告警，因此生产值继续保持 `0.60 rad/s`。不要为了让实验通过而
放宽碰撞、连续失败或 Path 代际安全门。

完整 `nav → pick → carry nav → place` seed 0 也已通过。圆形苹果使用任务级
`place_release_clearance_min_m=0.004`，把释放峰值角速度从 `12.3421` 降到
`3.9462 rad/s`，而 `12.0 rad/s` 质量阈值没有改变。这个参数只决定 CuRobo
计划释放中心比最终支撑目标高多少：值过大会让物体自由落下并旋转，值过小则
可能在开爪前压入支撑面。它不是 SCAN/PCT 导航参数，也不应复制到其他形状物体
而不做接触实测。

### phase270–273 末端余量与冻结恢复时序

生产末端现在使用以下三层距离合同：

- `fsm.reference_goal_hold_distance_xy=0.04`：SCAN 尽量把移动轨迹送到距精确目标
  4 cm 内；
- `terminal_capture.entry_distance_xy=0.055`：controller 只在 5.5 cm 内进入捕获；
- `finish.distance_xy=0.08`：pipeline 最终严格验收仍为 8 cm。

controller 的移动 final 连续驻留为 `0.50 s`，planner 的静止 hold 连续驻留为
`0.75 s`。planner 的驻留必须更长，避免它先用静止轨迹覆盖仍在做物理收口的移动
轨迹。seed 2 在这套参数下终点 XY 约 `0.0657 m`，严格通过；不要通过提高
`finish.distance_xy` 或删除驻留门来换取表面成功率。

终点捕获后的旋转稳定门使用 `finish.max_yaw_rate=0.10 rad/s`；planner 的兜底
驻留使用同语义的 `fsm.reference_goal_hold_yaw_rate=0.10 rad/s`。两者都读取
机体系 `|wz|`，不再把 Go2-X5 零命令站立时的 roll/pitch 微摆当作导航仍在转向。
moving final 尚未进入捕获内门时仍保留 `finish.max_angular_speed=0.10 rad/s`
三轴范数门，因此该修复不会让运动中的轨迹提前报到达。

最新 full seed 0 在楼梯入口发现的 `stair_sensor_freshness_fault` 不是传感器真的
失鲜：失败前最后状态中 `stale_inputs=[]`，点云和 Odometry 都持续更新。问题是
同一条旧 B-spline 的 `ALIGNING_YAW` 状态变化曾满足 supervisor 的恢复序号条件。
恢复安全门现在额外要求 trajectory identity 不得等于故障前被冻结的 identity；
因此旧轨迹不能撤销楼梯停车，释放后新生成的 B-spline 才能恢复。该修改已经通过
包级和主线回归，但尚未做修复后的第二次 full Isaac，调参结论暂不包含完整
pick/place 成功声明。运行退出后 GPU 计算接口曾短暂不可用；`23:24` 复检时
`nvidia-smi` 和正式 Isaac Python 的 CUDA 张量分配/同步均已恢复，设备数为 1。
下一次 live 复验开始前仍须即时执行同样的双重检查；本轮按约定没有自动开启
第二个 Isaac。

### phase274–281 最终恢复、完整 pipeline 与速度结论

phase278 seed 2 的失败不是 PCT 改走了另一条全局路线，也不是 SCAN 优化器算了
很久。安全 final B-spline 明确朝目标 `+X` 推进，但 controller 在终点外横向恢复
时仍沿完整 PCT Path 的末端 `-Y` 切线对齐，于是给原 Go2-X5 policy 发出以侧移为主
的命令；该 checkpoint 的横向响应较弱，机器人在目标外缓慢徘徊并触发
`nav_to_place_timeout`。修复后，仅在“final、仍处于终点捕获外、横向恢复已触发”
这三个条件同时满足时，controller 优先朝已通过碰撞检查的当前 SCAN 局部曲线航向；
局部弦过短才退回真实 Odometry 指向完整目标的方向。进入终点捕获后仍严格使用
PCT terminal yaw，非 final 行为没有改变。

phase279 seed 2 随后严格通过。分段时间为：完整导航阶段 `161.24 s`，共同楼梯
冻结 `62.20 s`，排除楼梯冻结后的 planner 控制时间 `98.46 s`，其中 F1
`18.20 s`、F2 `80.26 s`、terminal `48.66 s`。30 次成功 SCAN 优化累计墙钟仅
`2.43059 s`。这次慢样本的主要增量是终点附近 RL 回弹、重新接近和稳定驻留，
不是 161 秒都在计算 B-spline。作为对照，phase277 seed 1 的完整导航阶段为
`109.86 s`，楼梯冻结 `61.40 s`，planner 控制 `47.88 s`，terminal 仅
`5.28 s`，23 次成功优化累计 `2.05 s`。两次成功但离散度很大，所以不能用某一次
最快值或两次均值宣称超过 DWA。

phase280 的最新配置 full seed 0 已跑通导航到抓取点、抓取、携物跨层导航、放置
和导出。它说明下列生产组合已经达到“先稳定完成任务”的目标：

- `manager.reference_cruise_speed=0.60 m/s`；
- planner/controller/policy 的前进上限统一为 `0.65 m/s`；
- planner/controller/policy 的前进加速度/变化率统一为 `1.20 m/s²`；
- `limits.max_yaw_rate=0.60 rad/s`，不要切到已失败的 `0.75 rad/s` 实验；
- 横向恢复前进上限 `0.30 m/s`；
- planner 末端安全目标 `0.04 m`、controller 捕获入口 `0.055 m`、最终验收
  `0.08 m`；
- `body_height=0.338 m`，Go2-X5 双圆柱半径/偏移和点云上下膨胀保持生产安全值；
- 楼梯 `activation_radius_m=0.35 m`，进入认证楼梯段后按用户决定冻结底盘。

要让路径“又快又稳”，建议只从上述生产 YAML 复制完整配置做单变量 A/B：首先
观察 `yaw_only_command_s` 和 `terminal_capture_sim_time_s`，再小幅调整巡航速度或
横摆上限；每次都要复查碰撞、急停、终点误差、目标后零速和 place。不要通过增大
`finish.distance_xy`、缩小双圆柱、吞掉低障碍、延长传感器超时或降低驻留时间来
换表面速度。phase279 已证明末端慢样本可能来自 policy 物理响应，因此继续提高
B-spline 速度上限不一定缩短任务时间，反而可能增加回弹。

phase280 到抓取点后出现过两次无效的 final 重规划。根因是 planner 忙时
ControllerStatus 排队，回执到达时已不再是“最新发布 B-spline”。phase281 以
64 条有界、精确 identity 发布历史认证迟到回执，并在 Path 换代时清空；同时忽略
无 active Path 时迟到的旧楼梯快照。该收尾已通过 `scan_planner 149/149`、
`scan_controller 126/126` 和选定 Python 主线 `390/390`，但遵照约定没有再开
Isaac。它不改变 phase280 已成功的物理轨迹，只减少到点后的多余计算和日志。

相对 DWA 的最终结论仍是“未验证超过”。phase279 的 146 点完整 PCT Path 和统一
`0.08 m + yaw + 停稳 + 0.50 s 驻留 + 目标后零速` 合同已经导出；下一步必须在旧
DWA worktree 按相同 seed、checkpoint、楼梯冻结和时间口径逐 seed 回放。至少
3 个 seed、成功率不回退且排除共同楼梯冻结后的平均 planner 控制时间更低，才可
写成 SCAN 超过 DWA。历史 DWA `13.94 s` 是同层、`0.18 m` 宽松且未停稳的记录，
不能与当前跨层数字直接比较。

### phase303–306 偏航提速复验与 DWA 交接口径纠正

后续代码已经修复 phase269 中楼梯接管与 supervisor replan 同拍竞争，因此重新对
`limits.max_yaw_rate=0.75 rad/s` 做了严格单变量复验。seed 0、1、2 均使用同一
146 点 PCT Path、原 Go2-X5 checkpoint、同一碰撞包络和严格终点合同，三次
`crossfloor_carry` 全部 `valid=true`、`error_count=0`。排除楼梯冻结后的 SCAN
控制时间分别为 `50.30 s`、`51.92 s`、`47.30 s`，平均 `49.84 s`；对应
`0.60 rad/s` 三次基准平均为 `59.11 s`，缩短 `9.27 s`（`15.7%`）。完整导航
阶段平均由 `125.16 s` 降到 `112.62 s`（`10.0%`）。成功 SCAN 优化累计墙钟仍
只有每轮约 `1.58–1.61 s`，收益来自减少停下转向和重复起步，不是削减碰撞检查。

这三次只证明 carry 导航候选参数稳定；在同一参数完成完整 pick→carry→place
复验前，生产 YAML 继续保留 `0.60 rad/s`，实验仍通过
`configs/navigation/pct_scan_yaw_rate_075_experiment.yaml` 单独生成，不能把候选
结果提前写成生产默认。

第一次隔离 DWA 回放还暴露了比较合同本身的问题。原 phase104/105 DWA 稳定
pipeline 在楼梯后继续使用约 `6.0 m` 的 stair-float，进入二楼安全区域后才交回
DWA；隔离回放为了匹配当前 SCAN 冻结区间，把该后延伸强制改成 `0.4 m`。旧 DWA
因此在膨胀地图仍占用的交接点接管，release bridge 不可通，三次均立即
`nav_collision`。这三次不是原 DWA baseline 的失败，不能计为 DWA `0/3`，此前
生成的 phase302 胜负报告已作废。

比较分析器现在要求楼梯后位姿误差在门内，并且 release bridge 必须是
`collision_checked_direct` 且明确无碰撞；否则标记为“交接合同不兼容”，拒绝进入
成功率和速度汇总。后续只能保留 DWA 原来的交接配置做完整 pipeline baseline，或
另选双方都从共同自由空间开始的同层路段比较局部规划器，不能通过改变 DWA 的
既有 6 m 交接来制造失败。

### phase307 完整 pipeline 候选结果

`0.75 rad/s` 候选在完整 seed 0 pipeline 中完成了导航到抓取点、抓取、携物跨层
导航以及严格 `GOAL_REACHED`。`nav_to_place` 从生产成功样本的 `6520` 个控制步
降到 `5431` 个，即 `130.40 s → 108.62 s`，缩短 `21.78 s`（`16.7%`）。终点
XY/Z/yaw 误差为 `0.07158 m / 0.05319 m / 0.16077 rad`，线速度和角速度分别
只有 `0.00884 m/s / 0.00484 rad/s`，均满足原严格门；到点后底盘保持零速。
两次 full run 都是 147 点路线，但抓取后的真实起姿不同，Path SHA256 分别为
`8837c9cc...` 与 `d2def4ff...`，因此这项 `16.7%` 只能作为同任务观察值，不能
替代 phase303–305 同一路径三 seed 的 `15.7%` 严格 A/B 结论。

但完整 pipeline 最终没有通过。苹果释放后的峰值水平速度为 `0.17038 m/s`，超过
`0.15 m/s` 质量门；峰值角速度为 `18.2288 rad/s`，超过 `12.0 rad/s` 质量门，
因此严格判为 `place_release_ejected`。物体最后虽稳定在目标 XY/Z 约
`0.00649 m / 0.00097 m` 误差处，也不能改写成放置成功。生产 YAML 因而继续保持
`limits.max_yaw_rate=0.60 rad/s`，phase303–305 的 3/3 carry 结果不能替代完整
pick/place 验收。

该次抓取点终端 yaw 误差为 `0.13780 rad`，生产成功样本为 `0.09981 rad`；当前
`finish.yaw_control_deadband=0.18 rad` 允许二者都结束。两次抓取后的 object→TCP
偏置也确实不同，但单个接触样本不足以证明更大终端 yaw 导致释放旋转。下一轮只把
这个关系当作待证假设：配方
`configs/navigation/pct_scan_yaw075_terminal_yaw012_experiment.yaml` 保留快速
`0.75 rad/s`，同时把末端 yaw 控制死区收紧到 `0.12 rad`。它是两变量候选，必须
重新做 carry 三 seed 和完整 pick/place；不能继承旧单变量实验的通过结论。本阶段
结束后没有再次启动 Isaac。

### GitHub 发布参数决定

本次发布停止 DWA 对比，已有对照数据只作为历史审计记录，不能用于宣传 SCAN
算法优于 DWA。默认组合 launch 继续加载仓库内的生产
`pct_scan_tuning.yaml`，保留 `reference_cruise_speed=0.60 m/s`、
`limits.max_vx=0.65 m/s` 和 `limits.max_yaw_rate=0.60 rad/s`。所有 0.75 配方继续
留在 `configs/navigation/*experiment.yaml`，必须显式生成并传入才会生效，不能
覆盖生产文件。

稳定发布依据是相同代码、参数、原 Go2-X5 checkpoint 与同一 146 点 PCT Path 的
cross-floor carry seeds 0、1、2 严格 `3/3`，以及生产 0.60 参数的一次完整
nav/pick/carry/place/export 成功。结论包含楼梯底盘冻结，不包含纯物理爬楼、移动
推车绕障或 live PCT 重规划。

## 每次修改后的检查顺序

1. 运行 `colcon build --packages-up-to scan_planner isaac_navigation_bridge --symlink-install`。
2. 运行 SCAN 和 bridge 包级测试，确认参数关系与安全门未破坏。
3. 用无 Isaac 的 PCT→SCAN CPU 探针检查 Path 代际、B-spline、非零 `cmd_vel`。
4. 再在 Isaac 平地直线、转弯、斜坡分别测试至少 3 个 seed。
5. 记录任务完成时间、平均/最大命令速度、轨迹数量、yaw-only 时间、终点误差、
   急停和重规划次数；不能只看某一次最快结果。
6. 静态 3-seed 不回归后，再测试移动推车的绕障和原 Path 恢复。

使用自定义文件进行 A/B 时：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  tuning_config_file:=/absolute/path/pct_scan_tuning.yaml \
  body_height_m:=0.338
```

通过 `run_pct_scan_live_acceptance.py` 启动时无需重复填写高度；脚本会自动读取
`navigation_contract.body_height_m`。若直接执行 `ros2 launch`，launch 无法自行
解析这个共享伪节点，因此应像上面一样显式传入相同值。

`finish.distance_xy=0.08`、自碰撞过滤、双圆柱半径/偏移、上下膨胀和所有传感器
超时均是安全合同，不应作为“提速参数”放宽。
