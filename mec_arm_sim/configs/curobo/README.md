# mec_arm cuRobo 配置说明

## 当前文件

- `mec_arm.yml`：cuRobo MotionPlanner 使用的机器人配置。
- `assets/mec_arm_curobo.urdf`：cuRobo 专用 URDF 副本，mesh 路径已从 `package://mec_arm_model/...` 改成绝对路径。
- `worlds/smoke_cuboid.yml`：第一版 cuboid 障碍物配置，用于 smoke test。

## 关键约定

- Isaac Sim robot root：`/World/mec_arm`
- Isaac Sim articulation root：`/World/mec_arm/root_joint`
- cuRobo base link：`base_link`
- cuRobo TCP/EE link：`TCP_link`
- Isaac Sim DOF order：`Joint1, Joint2, Joint3, Joint4, Joint5, Joint6`

`mec_arm.yml` 里的 `cspace.joint_names` 必须和 Isaac Sim DOF order 完全一致。
如果 cuRobo 初始化后返回的 joint order 不一致，必须做重排，不能直接执行 trajectory。

## 第一版碰撞模型

`collision_spheres` 是根据 STL 包围盒估算的粗略球模型，只用于 MotionPlanner 第一次跑通。
后续需要精修：

1. 在 Isaac Sim 中用 Lula Robot Description Editor 生成更合理的 collision spheres。
2. 或者手工调整每个 link 的 sphere 数量、中心和半径。
3. 再逐步接入完整 3DGS collision mesh。

## 安装 cuRobo 前的提醒

当前 `/data/conda_envs/isaacsim51_3dgs_grasp` 里尚未检测到 `curobo`。
安装 cuRobo 需要联网和编译/安装依赖，执行前需要确认命令。

示例命令只供审阅，不要盲目执行：

```bash
cd /home/light/workspace
git clone https://github.com/NVlabs/curobo.git
cd curobo
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -m pip install -e . --no-build-isolation
```

如果网络不稳定，建议先手动下载或指定代理，再安装。
