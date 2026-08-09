# pct_ros2_adapter

该 ROS 2 Humble 包把固定提交的官方 PCT TomogramPlanner 封装成在线全局规划
节点。节点接收当前 `base_link` Odometry 和目标 Pose，在单个后台 worker 中
进程内调用官方核心，并发布可供 SCAN 使用的完整三维
`nav_msgs/msg/Path`。主消息链不使用 JSON、文件轮询或 stdin/stdout 子进程。

生产 YAML 与 PCT→SCAN 组合 launch 默认显式选择 `planner.backend_kind=upstream`。
该 backend 固定到
`byangw/PCT_planner@35cd73fd82bcd51bc538429294af7646b2a09815`，启动时核对
核心源码哈希和当前 Python ABI 的四个扩展；源码、扩展、GTSAM/OSQP 共享库或
tomogram 任一项不可用都会失败关闭，不会静默切换规划算法。

`planner.backend_kind=compatible` 仍保留用于隔离历史资产回归，内部调用本仓
自含 3D A*；它不是生产默认，也不是 upstream 失败时的 fallback。两种 backend
都复用唯一的 `sim_to_pct_xyz()` / `pct_to_sim_xyz()` 坐标边界和 collision PLY
地面投影。来源 pin 与许可证说明安装在 `share/pct_ros2_adapter/upstream/`；官方
源码按 GPLv2-or-later 保持为独立 external 依赖，不得重新标记为本包的
Apache-2.0 源码。

## 在线接口

| Topic | 类型 | QoS | 语义 |
| --- | --- | --- | --- |
| `/body_pose` | `nav_msgs/msg/Odometry` | SensorData | `world -> base_link` 当前位姿 |
| `/pct/goal` | `geometry_msgs/msg/PoseStamped` | reliable、volatile | 带非零 ROS 时间戳的 base 目标位姿 |
| `/pct/planning_command` | `scan_planner_msgs/srv/PCTPlanningCommand` | service | 带 goal/request/path 代际的 PLAN、REPLAN 与 CANCEL 快速 ACK |
| `/pct/global_path` | `nav_msgs/msg/Path` | reliable、transient local | PCT 原始全局地面路径，用于审计与 RViz |
| `/initial_path` | `nav_msgs/msg/Path` | reliable、transient local | 与 PCT Path 完全同 stamp/payload 的 SCAN 输入 |
| `/pct/planning_status` | `scan_planner_msgs/msg/PCTPlanningStatus` | reliable、transient local | 等待、规划、成功、无路径与异常状态 |

目标和 Odometry 必须使用同一 ROS 仿真时钟，frame 分别匹配配置的 `world` 与
`base_link`。新目标先发布一个带新代际时间戳的空 Path，使下游清除旧参考路径；
成功状态与非空 Path 使用相同时间戳。较慢的旧规划结果不会覆盖更新目标。
同 stamp、同 payload 的目标重发幂等，更旧 stamp 被忽略；同 stamp、不同
payload 会撤销该代，更新 stamp 即使目标几何相同也开始新规划。成功非空 Path
必须严格晚于 goal 与本代空 Path tombstone。

PCT 源 Path 与 SCAN 输入 Path 由同一个 adapter、同一个回调和同一个消息对象
发布。节点不会重新采样、重写 z 或生成第二个时间戳；成功代际和空 tombstone
都会同时发布。两个 topic 被配置成相同名称时节点拒绝启动，避免同一节点在一个
topic 上建立两个身份含混的 publisher。

typed service 是 supervisor 的生产控制边界。每个被接受的 PLAN、REPLAN 或
CANCEL 都会先发布严格更新的空 Path；精确重复 request 只返回 `DUPLICATE`，
不会再次触发 tombstone 或 worker。同 request 不同 payload 返回 `CONFLICT`，
旧 request 或不匹配当前 Path 的 `expected_path_stamp` 返回 `STALE`。PCT 在成功、
无路径或异常后仍保留活动目标快照，只有 CANCEL 或新 PLAN 才清除/替换；因此
REPLAN 可以从最新新鲜 Odometry 对同一目标重新计算。

## 高度与坐标合同

节点输入的起点和目标 z 都是机器人 base 高度，输出 Path 中每个 z 都是地面
高度。独立配置中的 `planner.slice_query_root_to_floor_m` 与
`planner.goal_base_to_ground_m` 默认均为 `0.30 m`；组合 launch 的
`body_height_m` 会同步覆盖 PCT、bridge、SCAN 和 controller 的对应参数，防止
同一条链重复加高或使用不同高度。PCT 粗高度最终会投影到 collision PLY 的
真实三角支撑面，SCAN 只在自己的入口增加一次 `body_height`。

端点还有独立的 `0.08 m` 支撑面误差硬门。当前真实 DDS 验收使用
“collision PLY 支撑面 + 0.30 m”构造 Odometry 与 goal，证明消息和规划合同，
但还没有证明历史任务 JSON 的 spawn/root z 与实时 `/body_pose` 使用完全相同的
高度基准。完整 Isaac 联调前必须用稳定后的 `/body_pose` 对 collision PLY 实测
`body_height_m`；不要直接把任务 JSON 的 z 发布为 `/pct/goal`，也不要为通过
门禁而改动物理 spawn z。

默认坐标关系为：

```text
pct_x = -sim_x
pct_y = -sim_y
pct_z =  sim_z
```

完整坐标变换固定为 `coord_mode → XYZ scale → 固定轴 X/Y/Z 欧拉旋转
(Rz @ Ry @ Rx) → XYZ offset`，旋转参数单位为弧度。正反变换只由
`sim_to_pct_xyz()` 和 `pct_to_sim_xyz()` 实现；参数必须有限且三个 scale 均不得
为零，否则节点按配置错误停止，不发布路径。offset、scale、rotation、网格搜索、
楼梯 gateway、支撑面投影误差与所有资产路径都在
`config/pct_ros2_adapter.yaml` 中配置。大型 tomogram、NPY 与 PLY 是本地运行资产，
不应加入 Git。

## 构建与独立运行

### 可复现 upstream 源码准备

`scripts/navigation/manage_pct_upstream.py` 只接受
`schema_version=2` 的来源 manifest 和显式提供的本地归档，不自行联网，也不会
覆盖不匹配的已有源码目录。manifest 必须固定归档 SHA256、唯一归档根目录、
pristine/patched 源码树身份，以及每个补丁的 SHA256、preimage 和 postimage。
源码树身份按排序后的 `相对路径 + 文件大小 + 文件 SHA256` 聚合，因此未声明的
源码漂移或额外文件也会失败关闭。构建后的验证可以显式增加
`--allow-generated`；它只忽略 manifest `source.generated_path_patterns` 中逐项固定的
根相对 glob，并在 JSON 结果中完整报告匹配文件。非白名单额外源码和任何受管
源码变化仍会使整树身份失败，禁止使用 `**` 之类覆盖整个源码树的宽泛规则。
GTSAM install 中的 SONAME 链接只在链接自身和最终普通文件命中同一 generated
pattern 时允许；目标必须是无 `..` 的相对路径，悬空、绝对目标和源码根逃逸都会
失败，链接本身只进入 generated 报告而不进入源码 tree digest。

首次准备使用：

```bash
/usr/bin/python3 scripts/navigation/manage_pct_upstream.py prepare \
  --manifest ros2_ws/src/pct_ros2_adapter/upstream/PCT_PLANNER_SOURCE.json \
  --archive /abs/path/PCT_planner-35cd73fd82bcd51bc538429294af7646b2a09815.tar.gz \
  --source-root external/PCT_planner
```

`prepare` 会先在同一文件系统的临时目录中验证和应用全部补丁，最终状态完全匹配
后才原子放置到目标目录；目标已是正确 patched 状态时幂等返回。已有 pristine
源码也可以显式执行 `apply`，日常预检则使用 `verify-source`：

```bash
/usr/bin/python3 scripts/navigation/manage_pct_upstream.py apply \
  --manifest ros2_ws/src/pct_ros2_adapter/upstream/PCT_PLANNER_SOURCE.json \
  --source-root external/PCT_planner

/usr/bin/python3 scripts/navigation/manage_pct_upstream.py verify-source \
  --manifest ros2_ws/src/pct_ros2_adapter/upstream/PCT_PLANNER_SOURCE.json \
  --source-root external/PCT_planner \
  --state patched \
  --allow-generated
```

`build-plan` 只输出结构化 argv，不执行其中任何命令。它固定系统
`/usr/bin/python3`、CPython 3.10 SOABI、Release、CMake policy 3.5，并明确关闭
GTSAM `march=native`；build root 必须位于 upstream 源码树之外。审核 JSON 后由
操作者逐条执行命令，再用 `verify-binaries` 检查四个 Python 扩展、五个内部共享
库、ELF64 x86-64、相对 RUNPATH 和 `ldd` 闭包：

```bash
/usr/bin/python3 scripts/navigation/manage_pct_upstream.py build-plan \
  --manifest ros2_ws/src/pct_ros2_adapter/upstream/PCT_PLANNER_SOURCE.json \
  --source-root external/PCT_planner \
  --build-root /tmp/pct-planner-build \
  --jobs 4

/usr/bin/python3 scripts/navigation/manage_pct_upstream.py verify-binaries \
  --manifest ros2_ws/src/pct_ros2_adapter/upstream/PCT_PLANNER_SOURCE.json \
  --source-root external/PCT_planner
```

二进制 RUNPATH 只允许 `$ORIGIN` 以及 manifest 中固定的两个 `$ORIGIN/3rdparty/...`
目录；任何 `/home`、`/mnt`、`/tmp` 或其他绝对路径都会失败。`build-plan` 不会自动
编译，`verify-binaries` 也不会修改产物；随机目录迁移导入仍应作为发布前独立 smoke。

官方源码位于 ignored 的 `external/PCT_planner`。当前主机的 CMake 4.1.3 需要在
配置其旧版 GTSAM/OSQP 时显式增加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`，并且
planner 扩展必须明确使用 ROS 2 的 `/usr/bin/python3`（Python 3.10）ABI。本机
已经完成 Release 版 GTSAM 4.1.1、OSQP、`a_star`、`traj_opt`、`ele_planner` 和
`py_map_manager` 构建；扩展 RUNPATH/`ldd` 闭包、真实导入与固定源码哈希均已
通过。当前运行身份是官方 commit 加 manifest 固定的
`pct-scan-native-astar-cancel-cost-aware-stable-queue-relocatable-v2` 补丁；补丁为
A* 增加按触达节点清理、原子取消、搜索状态/计数、稳定优先队列和 pybind GIL
释放，并把原本硬编码为 `0.2` 的地形代价权重接到 ROS 参数，同时移除
`march=native` 传播及绝对 RUNPATH。生产 backend 只调用官方
`OfflineElePlanner` 的 native A*；
连续局部平滑与避障由 SCAN 负责，不再调用上游 GPMP 优化器。

正式 multi-floor tomogram 由 collision PLY 的官方五通道语义加
`configs/navigation/pct_multifloor_stair_profile.json` 生成。该 profile 来自
`pct-scene`，构建时只给楼梯区域注入 7 个跨层 gateway，不硬编码起点到终点
全程路线。运行时 upstream backend 仍让原生 PCT A* 决定楼层与楼面路线，只把
已匹配到的楼梯区间规范化到同一组 ground anchor，并保留请求的精确起终点：

```bash
/usr/bin/python3 scripts/navigation/build_pct_multifloor_assets.py \
  --tomogram-kind upstream \
  --output-tomogram source/scene/multifloor/mutifloor_upstream.pickle \
  --output-walkable /tmp/mutifloor_upstream_walkable.npy \
  --report-output outputs/pct_multifloor_upstream_asset_build_report.json

/usr/bin/python3 ros2_ws/src/pct_ros2_adapter/test/probe_upstream_pct_multifloor.py
```

tomogram 与 PLY 仍是 ignored 本地资产；JSON profile 和构建算法才是可提交的
可复现来源。profile 会核对 collision PLY、基础 tomogram hash、shape、center
和简化层索引；运行时还会核对 profile、生产 tomogram、PLY 与坐标变换合同。
参数或资产漂移时直接失败，不会把中心线先验套到另一张地图或错误路段。

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to pct_ros2_adapter
source install/setup.bash
ros2 launch pct_ros2_adapter pct_ros2_adapter.launch.py
```

如果当前目录不在仓库内，应显式传入自定义 YAML，在其中设置
`planner.project_root` 与资产路径：

```bash
ros2 launch pct_ros2_adapter pct_ros2_adapter.launch.py \
  config_file:=/abs/path/to/pct_ros2_adapter.yaml
```

## 接入 PCT → SCAN 主线

组合 launch 保留 `/pct/global_path` 作为 PCT 源结果，同时让 bridge、SCAN、
controller 和 supervisor 统一消费 `/initial_path`。这不是第二次规划：两个
topic 的消息由同一个 PCT adapter 原样双发，header stamp、frame、pose 和地面
高度完全一致：

```bash
ros2 launch isaac_navigation_bridge pct_scan_navigation.launch.py \
  start_pct:=true \
  start_manual_path:=false
```

PCT 与手工 Path 发布器互斥；生产组合 launch 默认 `start_pct=true`。手工 Path smoke 必须
显式传 `start_pct:=false start_manual_path:=true`；同时为 `true` 会在 launch
展开阶段明确失败。两者都关闭时，可以由外部可靠 Path 发布器写入
`initial_path_topic`。pipeline 存在手工 Path 时不会生成 PCT goal；没有手工
Path 时 `NavGoal` 必须显式提供 base z，并由 Isaac OGN 发布 `/pct/goal`。

组合 launch 还统一暴露 `world_frame` 与 `base_frame`，并在创建节点前拒绝
前导斜杠、空白、空层级和相同的 world/base frame。Isaac 原始 Odometry 与
点云 header 必须与所选 world frame 一致；launch 只统一参数，不执行坐标变换。

新 goal 会先发布空 Path 清除旧代际。compatible grid A* 使用单调时钟截止时间，
取消和超时会在搜索循环内生效。upstream native A* 现通过受跟踪补丁提供 sticky
原子取消，`init/search/init_map/plan` 在 pybind 调用期间释放 GIL；ROS worker
同时设置每请求 Python Event 和 native cancel，旧结果仍由 plan-id/tombstone
门禁丢弃。1024×1024 不可达网格的并发探针已在搜索展开 32 个节点后从 Python
线程发出取消并于 2 秒门限内退出。A* 每次搜索只清理上一次触达的节点，no-path
或正常取消后可安全复用同一 `TomogramPlanner`；只有 native 异常、损坏输出或
重置失败才重建并重新加载 tomogram。起终点必须映射到可通行的真实逻辑层，且
collision PLY 投影误差通过独立硬门。未来时间戳、过期 Odometry、异常 frame、
不可投影端点和过短结果都不会发布成功 Path。adapter 在仿真时钟首次有效时也会发布空 Path 与 `IDLE`，
清理自身重启前遗留的 transient-local 数据。

目标代际现已跨越 executor、Isaac runtime、OGN、PCT adapter 和 SCAN Path
消费端：执行器要求每代 goal 只有一次发布回执，并只接受严格晚于 goal stamp
的非空 Path；同 stamp 冲突会失败关闭。PCT worker 取消仍是 adapter 进程内
发起并由 patched native A* 协同执行的机制。supervisor 的 PLAN/REPLAN
service ACK、固定 request 重试、匹配的新 status/Path 及恢复许可已接通，并通过真实 CycloneDDS 时序探针；动态阻断场景
的 Isaac live 恢复仍待最终验收。若 Isaac timeline 或 `/clock` 回退，必须
重启完整 ROS 2 导航图；尚未验收同一图内跨 epoch 复用。

## 静态检查

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m py_compile \
  pct_ros2_adapter/*.py launch/*.launch.py setup.py
```

ROS 2 构建后还应运行 `colcon test --packages-select pct_ros2_adapter`，并在真实
资产与 `/clock` 存在时验证 goal、status 和 Path 的完整 topic 生命周期。

本机 multi-floor ignored 资产的 upstream 离线探针实际调用固定官方原生扩展，
当前标准请求得到 141 点、`23.543 m` 的 F1→F2 Path、logical layer `8→15`、
地面高度跨度 `3.209 m`；7 个楼梯 anchor 的 XY 误差均为零，精确请求起终点
保持不变。跨层路径只压缩楼梯前后经过净空验证的平地折线，楼梯 profile 不变。
点数不是固定接口；状态中的
`path_point_count` 必须与实际 Path 完全一致。历史 compatible 的 169/43 点结果
只保留隔离回归。真实 topic 生命周期回归位于
`test/test_real_asset_lifecycle.py`，upstream 探针位于
`test/probe_upstream_pct_ros_lifecycle.py`。

最新真实多进程 ROS 2 探针使用 phase225 精确生产起终点，把 142 点 upstream
Path 直接送入 SCAN。SCAN 不再同步
拟合整条全局路线，而是沿原始有序三维折线选择局部弧长窗口；无碰撞 guide
保留转角和高度锚点，按 5 cm 地图尺度加密后只做等比例动力学延时。最终收到
35 控制点、三阶、`is_final=false` 的正常 B-spline 和非零 `/cmd_vel`；typed
GridMap 证据确认 144/144 个显式 free 端点已融合。首条轨迹时长
`3.002090 s`、速度上界 `0.544554 m/s`，探针会对
`duration<=3.20 s / velocity>=0.50 m/s` 失败关闭；五个主线节点可由一次
SIGINT 干净退出。独立 90° 探针还把最大折线偏差从旧版
`0.131501 m` 降到 `0.027655 m`。这些结果证明 PCT→SCAN→controller 首段消息链
和局部几何保持。phase218 又完成同一 Isaac episode 的静态楼梯 root-lock 验收；
该结果是用户接受的底盘冻结通路，不是纯物理爬楼，也不替代动态恢复和全局
重规划 smoke。

phase227 又对同一固定生产请求的整条 Path 做了逐段检查：PCT 输出仍为 142 点、
`23.704265 m`，layer `8→15`、精确起终点和 7 个楼梯 anchor 不变。探针只在
楼梯冻结区外取 22 个 Path 后缀交给 SCAN；局部平滑前后的 A/B 不会回写或重排
`/pct/global_path`。开启受限平滑后，22 条局部轨迹最长 `3.022893 s`、最低速度
上界 `0.469849 m/s`，最大有序折线偏差 `0.032035 m`。因此这里“路线合理”指
PCT 的跨层拓扑和净空合同保持不变，并且 SCAN 能在所有平地区段安全执行；它仍
是合成自由空间/理想 Odometry 证据，不能替代 Isaac 中对真实点云、楼梯释放和
完整到达的验收。

复现该首段链路时先启动组合图，再在相同 `ROS_DOMAIN_ID` 下运行真实资产探针。
探针自己直接提供已规范化的位姿和点云，因此必须显式使用
`start_bridge:=false`。当前默认验收是只规划模式：

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
  ros2_ws/src/pct_ros2_adapter/test/probe_real_pct_to_scan_chain.py \
  --planning-only
```

探针直接发布已经满足 bridge 输出合同的 `/body_pose`、世界系合成点云和
`/pct/goal`，因此不要同时让 Isaac bridge 向这两个规范化 topic 发布。它只用于
确定性 ROS 2 接口验收，不是仿真输入替代方案。`--planning-only` 要求正常
B-spline，并严格要求 `/cmd_vel` 没有 publisher；不带该参数的历史模式则会
继续等待 controller 的非零速度。最新宿主 DDS 只规划验收得到 142 点 PCT
Path、同样 142 点 `/initial_path`、35 个控制点的三阶 B-spline、
`3.002090 s` 轨迹时长与 `0.544554 m/s` 速度上界；`cmd_nonzero=0`、
`cmd_publishers=0`。
