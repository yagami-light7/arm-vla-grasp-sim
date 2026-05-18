# cuRobo 轨迹生成与轨迹追踪开发流程

本文记录 Go2-X5 导航 + 抓取任务中，机械臂轨迹生成与 Isaac Sim 轨迹追踪的开发路线。

当前阶段只做第一版固定底盘机械臂规划：

```text
完整 Go2-X5 模型用于 Isaac Sim 仿真与后续导航
arm-only 派生模型用于 cuRobo 机械臂规划
```

## 0. 当前目录约定

`source/` 只存放机器人模型、场景模型和 cuRobo 生成配置，不存放 Python 脚本。

```text
source/
  scene/
    839920_mecarm.usda

  robot/
    go2_x5/
      urdf/
        go2_x5.urdf
      meshes/
        ...
      curobo/
        go2_x5_arm.urdf
        go2_x5_arm.yml
        go2_x5_arm.xrdf
```

脚本放在根目录 `scripts/`：

```text
scripts/
  curobo/
    0_make_go2_x5_arm_urdf.py
    1_build_go2_x5_curobo_model.py
    2_check_go2_x5_curobo_model.py
    3_check_isaac_curobo_fk.py
    4_demo_plan_to_pose.py
    5_demo_track_trajectory.py
    9_task_nav_grasp_demo.py

  isaac/
    0_inspect_go2_x5_articulation.py
    1_dump_go2_x5_state.py
```

## 1. 原始整机 URDF

原始整机 URDF 来自 `Automatonzy/DWA`：

```text
source/robot/go2_x5/urdf/go2_x5.urdf
```

该文件保持未修改，用于描述完整系统：

```text
Go2 四足底盘
X5 六轴机械臂
双指夹爪
arm_eef_link 末端坐标系
```

当前已经确认项目内 URDF 与 DWA 仓库原始 URDF 的 sha256 一致：

```text
012e325fcc0ddb9e4a160a999d61237f31315e1f9fb687befe7e5a239446b468
```

## 2. 生成 arm-only URDF

cuRobo 第一版只规划机械臂，不规划狗腿和底盘。因此从完整 `go2_x5.urdf` 派生：

```text
source/robot/go2_x5/curobo/go2_x5_arm.urdf
```

生成脚本：

```bash
python scripts/curobo/0_make_go2_x5_arm_urdf.py
```

派生模型约定：

```text
base link: arm_base_link
active joints: arm_joint1 ~ arm_joint6
tool frame: arm_eef_link
gripper joints: arm_joint7 / arm_joint8 固定为 fixed
```

检查命令：

```bash
rg -n "<robot|<link name=\"arm_|<joint name=\"arm_joint[1-8]|arm_gripper_fixed_joint|type=\"fixed\"|type=\"revolute\"" \
  source/robot/go2_x5/curobo/go2_x5_arm.urdf
```

当前检查结果：

```text
arm_joint1 ~ arm_joint6: revolute
arm_joint7 / arm_joint8: fixed
arm_gripper_fixed_joint: fixed
arm_eef_link: exists
```

## 3. 调用官方 CLI 生成 yml/xrdf

为了和 cuRobo 官方教程保持一致，项目脚本只包装官方 CLI：

```text
python -m curobo.examples.getting_started.build_robot_model
```

项目脚本：

```bash
PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/curobo/1_build_go2_x5_curobo_model.py
```

等价官方命令：

```bash
PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  -m curobo.examples.getting_started.build_robot_model \
  --urdf /home/light/workspace/arm_vla/source/robot/go2_x5/curobo/go2_x5_arm.urdf \
  --asset-path /home/light/workspace/arm_vla/source/robot/go2_x5 \
  --output /home/light/workspace/arm_vla/source/robot/go2_x5/curobo/go2_x5_arm.yml \
  --export-xrdf \
  --tool-frames arm_eef_link \
  --sphere-density 1.0 \
  --num-collision-samples 1000 \
  --compute-metrics \
  --seed 42
```

生成文件：

```text
source/robot/go2_x5/curobo/go2_x5_arm.yml
source/robot/go2_x5/curobo/go2_x5_arm.xrdf
```

## 4. yml 关键字段检查

检查命令：

```bash
rg -n "base_link|tool_frames|joint_names|collision_spheres|self_collision|arm_eef_link|arm_joint" \
  source/robot/go2_x5/curobo/go2_x5_arm.yml
```

当前关键结果：

```text
base_link: arm_base_link
joint_names:
  - arm_joint1
  - arm_joint2
  - arm_joint3
  - arm_joint4
  - arm_joint5
  - arm_joint6
tool_frames:
  - arm_eef_link
```

## 5. arm_link1 碰撞球修正

第一次构建时，`arm_link1` 的 sphere fitting 指标异常：

```text
arm_link1 cover% = 0.4%
```

这个结果通常表示碰撞球没有有效覆盖 link mesh。后续用官方 edit/refit 模式只重拟合 `arm_link1`：

```bash
PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  -m curobo.examples.getting_started.build_robot_model \
  --edit-config /home/light/workspace/arm_vla/source/robot/go2_x5/curobo/go2_x5_arm.yml \
  --output /home/light/workspace/arm_vla/source/robot/go2_x5/curobo/go2_x5_arm_refit_link1.yml \
  --export-xrdf \
  --refit-link arm_link1 \
  --sphere-density 3.0 \
  --recompute-collisions \
  --num-collision-samples 1000 \
  --visualize \
  --viz-port 8081
```

这一步可以作为后续优化碰撞球的工具链参考。

注意：当前正在使用、且已经让 `planner_success=True` 的 `go2_x5_arm.yml`
是官方 CLI 生成版本再手动修正 `self_collision_ignore` 后的版本，不是
`arm_link1` refit 临时版本。

当前实际使用版本中：

```text
arm_link1 sphere_count: 12
arm_link1 radius_min: 0.002000 m
arm_link1 radius_max: 0.002000 m
arm_link1 radius_avg: 0.002000 m
total robot spheres: 62
```

最终只保留：

```text
source/robot/go2_x5/curobo/go2_x5_arm.urdf
source/robot/go2_x5/curobo/go2_x5_arm.yml
source/robot/go2_x5/curobo/go2_x5_arm.xrdf
```

## 6. MotionPlanner 加载检查

最终 `go2_x5_arm.yml` 已通过 cuRobo MotionPlanner 最小 FK 检查。

检查脚本：

```bash
PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/curobo/2_check_go2_x5_curobo_model.py
```

当前验证结果：

```text
torch.cuda.is_available: True
joint_names: ['arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6']
tool_frames: ['arm_eef_link']
default_q: [[0.26100003719329834, 1.5700000524520874, 1.5700000524520874, 0.0, 0.0, 0.0]]
arm_eef_link_pos: [0.43301493 0.11513431 0.42049992]
robot_spheres_shape: (1, 1, 54, 4)
```

说明：

```text
go2_x5_arm.yml 可以被 cuRobo 加载
FK 可以计算 arm_eef_link
collision spheres 已生效
```

## 7. 下一步：Isaac Sim 状态导出

已经完成第一版 Go2-X5 articulation 检查脚本：

```text
scripts/isaac/0_inspect_go2_x5_articulation.py
```

该脚本在 Isaac Sim Script Editor 中运行，只读检查，不控制机器人。

当前 Isaac Sim 检查结果：

```text
full DOF count: 20

arm_joint1 -> Isaac DOF index 8
arm_joint2 -> Isaac DOF index 13
arm_joint3 -> Isaac DOF index 14
arm_joint4 -> Isaac DOF index 15
arm_joint5 -> Isaac DOF index 16
arm_joint6 -> Isaac DOF index 17

gripper:
arm_joint7 -> Isaac DOF index 18
arm_joint8 -> Isaac DOF index 19

q_arm:
[-9.720044e-08, -6.754778e-06, 3.272774e-04,
  5.515186e-05, 1.818794e-08, -4.312500e-10]

dq_arm:
[7.079308e-06, 5.403319e-04, -5.967533e-04,
 -1.689788e-02, 9.961595e-06, -4.473274e-07]

base frame:
/World/go2_x5/arm_base_link

tcp frame:
/World/go2_x5/arm_link6/arm_eef_link

T_base_tcp:
position_xyz=(0.184251, -0.000501, 0.156651)
quat_wxyz=(1.000000, 0.000000, -0.000195, -0.000000)
```

关键结论：

```text
Go2-X5 在 Isaac Sim 中是 20 DOF articulation
arm_joint1~6 在完整 DOF order 中不是连续的一整段
后续执行 cuRobo trajectory 时必须按 joint name 映射写回完整 articulation
arm_eef_link 在 stage 中直接存在，不需要 fallback 到 arm_link6 + offset
```

注意：

```text
本次输出中腿部关节速度较大，后续做轨迹追踪前需要让 Go2 底盘固定或进入稳定站立状态。
这不影响当前 arm joint 映射和 TCP frame 检查。
```

状态导出脚本：

```text
scripts/isaac/1_dump_go2_x5_state.py
```

该脚本在 Isaac Sim Script Editor 中运行，导出：

```text
full DOF order
q_full / dq_full
arm_joint1 ~ arm_joint6 的 q_arm / dq_arm
T_world_arm_base_link
T_world_arm_eef_link
T_arm_base_link_arm_eef_link
```

当前导出结果：

```text
JSON: /tmp/go2_x5_isaac_state.json

q_arm:
[-1.027045e-07, -6.764444e-06, 3.272372e-04,
  5.525928e-05, 2.924621e-08, -6.144016e-10]

dq_arm:
[0, 0, 0, 0, 0, 0]

T_base_tcp position:
[0.1842512795, -0.0005006811, 0.1566514643]

T_base_tcp quat_wxyz:
[0.9999999811, 3.6427732531e-08, -0.0001946613, -1.2153377168e-08]
```

这里最重要的是确认 Isaac 的 DOF order 中能找到：

```text
arm_joint1
arm_joint2
arm_joint3
arm_joint4
arm_joint5
arm_joint6
```

cuRobo 只吃 `q_arm`，不能直接吃完整 Go2-X5 的 `q_full`。

## 8. Isaac FK 与 cuRobo FK 对齐

普通 Python 检查脚本：

```text
scripts/curobo/3_check_isaac_curobo_fk.py
```

目标：

```text
输入：Isaac 导出的 q_arm
cuRobo FK：arm_eef_link in arm_base_link
Isaac FK：arm_eef_link in arm_base_link
输出：位置误差和姿态误差
```

通过标准建议：

```text
position error < 0.02 m
orientation error < 5 deg
```

当前运行结果：

```text
Isaac position:
[ 0.18425128 -0.00050068  0.15665146]

cuRobo position:
[ 0.18425128 -0.00050043  0.15665136]

position error: 2.663979200521848e-07 m
orientation error: 8.189109132934075e-06 deg
result: FK 对齐通过
```

结论：

```text
Isaac Sim 中的 arm_base_link -> arm_eef_link
与 cuRobo 中的 arm_base_link -> arm_eef_link
在当前 q_arm 下已经对齐。
```

## 9. 下一步：轨迹生成

FK 对齐后，再进入轨迹生成：

```text
scripts/curobo/4_demo_plan_to_pose.py
```

第一版输入：

```text
q_arm_current
target_tcp_pose in arm_base_link frame
简单 cuboid obstacle
```

输出：

```text
q_arm trajectory
TCP path
final TCP error
joint limit check
singularity warning
```

注意：

```text
cuRobo 规划输出的是 arm_joint1~6 的轨迹
执行到 Isaac 时，需要写回完整 articulation 中对应的 arm joint index
```

当前单 TCP pose 规划已经跑通：

```bash
PYTHONPATH=/home/light/workspace/curobo:${PYTHONPATH:-} \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/curobo/4_demo_plan_to_pose.py
```

输出文件：

```text
/tmp/go2_x5_arm_plan_to_pose.json
```

当前结果：

```text
planner_success: True
trajectory shape: (41, 6)
final_position_error_m: 3.332412745749025e-08
final_orientation_error_deg: 2.9575586669421963e-06
```

调试过程记录：

```text
第一次运行时，末端 pose 已经收敛，但 planner_success=False。
RobotDebugger 显示起点附近存在 arm_link2 <-> arm_link4 自碰撞。
最大 penetration 约 0.002293 m。
```

修复方式是在 `source/robot/go2_x5/curobo/go2_x5_arm.yml` 中加入：

```yaml
self_collision_ignore:
  arm_link2:
  - arm_link1
  - arm_link3
  - arm_link4
  arm_link4:
  - arm_link2
  - arm_link3
  - arm_link5
  - arm_link7
  - arm_link8
```

修复后，RobotDebugger 检查保存的 41 个 waypoint：

```text
self_collision_check: passed for saved trajectory
```

## 10. 下一步：轨迹追踪

轨迹追踪脚本建议：

```text
scripts/curobo/5_demo_track_trajectory.py
```

第一版只控制机械臂关节：

```text
arm_joint1 ~ arm_joint6
```

非机械臂关节：

```text
Go2 腿部关节保持站立状态或当前状态
夹爪 arm_joint7/8 暂时单独控制 open/close
```

通过标准建议：

```text
最终 TCP position error < 0.03 m
非 arm joints 不被错误覆盖
轨迹执行过程中无明显抖动或关节跳变
```

## 11. 后续导航 + 抓取整合

完整任务会分成两层：

```text
导航层：
  DWA / A* 输出 base x, y, yaw 路径

操作层：
  cuRobo 在某个 base pose 下规划 arm_joint1~6
```

第一版组合方式：

```text
1. Go2 导航到抓取位姿附近
2. 固定或稳定底盘
3. 导出当前 arm_base_link 和 arm_eef_link
4. cuRobo 规划机械臂轨迹
5. Isaac 执行 arm trajectory
6. 后续再加入夹爪 close/open 和 lift/place
```
