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

  isaac/
    后续放 Isaac Sim Script Editor 脚本
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

比较后选择 refit 版本作为最终版本。

当前最终版本中：

```text
arm_link1 sphere_count: 4
arm_link1 radius_min: 0.015955 m
arm_link1 radius_max: 0.028712 m
arm_link1 radius_avg: 0.022301 m
total robot spheres: 54
```

最终只保留：

```text
source/robot/go2_x5/curobo/go2_x5_arm.urdf
source/robot/go2_x5/curobo/go2_x5_arm.yml
source/robot/go2_x5/curobo/go2_x5_arm.xrdf
```

## 6. MotionPlanner 加载检查

最终 `go2_x5_arm.yml` 已通过 cuRobo MotionPlanner 最小 FK 检查。

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

下一步要在 Isaac Sim 中加载完整 Go2-X5 articulation，并导出：

```text
full DOF order
q_full / dq_full
arm_joint1 ~ arm_joint6 的 q_arm / dq_arm
T_world_arm_base_link
T_world_arm_eef_link
T_arm_base_link_arm_eef_link
```

建议脚本：

```text
scripts/isaac/0_inspect_go2_x5_articulation.py
scripts/isaac/1_dump_go2_x5_state.py
```

其中最重要的是确认 Isaac 的 DOF order 中能找到：

```text
arm_joint1
arm_joint2
arm_joint3
arm_joint4
arm_joint5
arm_joint6
```

cuRobo 只吃 `q_arm`，不能直接吃完整 Go2-X5 的 `q_full`。

## 8. 下一步：Isaac FK 与 cuRobo FK 对齐

在拿到 Isaac 导出的状态后，写普通 Python 脚本：

```text
scripts/curobo/2_check_go2_x5_fk_align.py
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

## 9. 下一步：轨迹生成

FK 对齐后，再进入轨迹生成：

```text
scripts/curobo/3_demo_plan_to_pose.py
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

## 10. 下一步：轨迹追踪

轨迹追踪脚本建议：

```text
scripts/isaac/2_track_go2_x5_arm_trajectory.py
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

