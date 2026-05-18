# 1-从 Isaac Sim 导出机械臂状态

本节对应脚本：

```text
/home/light/workspace/arm_vla/mec_arm_sim/scripts/curobo/dump_isaac_state.py
```

本节目标是学习如何在 Isaac Sim 5.1.0 中读取当前机械臂状态，并保存成后续 cuRobo FK 对齐检查可以使用的 JSON。

这个脚本只做一件事：

```text
Isaac Sim 当前 stage
-> 读取机械臂 articulation state
-> 读取 base_link / TCP_link 的 world pose
-> 计算 TCP_link in base_link
-> 保存 /tmp/mec_arm_isaac_state.json
```

它不导入 cuRobo，不做 IK，不做 MotionPlanner，也不驱动机械臂运动。

## 为什么需要这个脚本

cuRobo 后续规划使用的是 robot base frame 下的 TCP 目标位姿，例如：

```text
T_base_tcp
```

但 Isaac Sim 里直接读到的 prim 位姿通常是 world frame 下的：

```text
T_world_base
T_world_tcp
```

所以我们必须在 Isaac Sim 中导出这些信息，并计算：

```text
T_base_tcp = inverse(T_world_base) @ T_world_tcp
```

这个 JSON 会作为下一步 `check_fk_align.py` 的输入，用来验证：

```text
同一个 q_current
Isaac Sim 看到的 TCP pose
cuRobo FK 算出来的 TCP pose
是否一致
```

如果这一步不对齐，后面的 IK、MotionPlanner、trajectory 都没有可靠基础。

## 运行位置

这个脚本必须在 Isaac Sim GUI 的 Script Editor 中运行。

推荐运行方式：

```python
exec(open("/home/light/workspace/arm_vla/mec_arm_sim/scripts/curobo/dump_isaac_state.py", "r", encoding="utf-8").read())
```

不要在普通终端直接运行：

```bash
python mec_arm_sim/scripts/curobo/dump_isaac_state.py
```

原因是普通 Python 没有当前打开的 USD stage，也没有 Isaac Kit 的运行时上下文。`omni.usd.get_context().get_stage()` 只有在 Isaac Sim GUI / Script Editor 里才有意义。

## 配置项

脚本顶部有几个关键路径：

```python
ARTICULATION_ROOT_PATH = "/World/mec_arm/root_joint"
ROBOT_BASE_FRAME_PATH = "/World/mec_arm/base_link"
TCP_FRAME_PATH = "/World/mec_arm/Empty_Link6/TCP_link"
EXPECTED_JOINT_ORDER = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
OUTPUT_JSON_PATH = Path("/tmp/mec_arm_isaac_state.json")
```

含义如下：

- `ARTICULATION_ROOT_PATH`：Isaac Sim 中机械臂 ArticulationRoot 的 prim path，用于创建 `SingleArticulation` 并读取关节状态。
- `ROBOT_BASE_FRAME_PATH`：cuRobo robot model 使用的 base frame，这里是 `/World/mec_arm/base_link`。
- `TCP_FRAME_PATH`：Isaac stage 中 TCP frame 的 prim path，这里是 `/World/mec_arm/Empty_Link6/TCP_link`。
- `EXPECTED_JOINT_ORDER`：期望的 DOF 顺序，必须和 cuRobo yml 中的 `cspace.joint_names` 一致。
- `OUTPUT_JSON_PATH`：导出的状态文件，后续普通 Python 脚本会读取它。

如果未来机器人 prim path 改了，优先改这几个常量。

## Isaac Sim API：获取当前 stage

脚本中使用：

```python
def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD stage。请先在 Isaac Sim GUI 中打开场景。")
    return stage
```

这里的关键 API 是：

```python
omni.usd.get_context().get_stage()
```

它返回当前 Isaac Sim GUI 中已经打开的 USD stage。

注意不要写成：

```python
Usd.Stage.Open(omni.usd.get_context().get_stage_file_path())
```

Isaac Sim 5.1.0 的 `UsdContext` 没有 `get_stage_file_path()`，而且 Script Editor 中也不应该重新打开 stage。重新打开文件会绕过当前 GUI stage 的运行时状态，不适合读取 articulation 的当前状态。

## Isaac Sim API：创建 SingleArticulation

脚本中使用：

```python
async def create_robot_handle():
    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    await world.play_async()
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(
        prim_path=ARTICULATION_ROOT_PATH,
        name="mec_arm_dump_state",
    )
    robot.initialize()

    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 无效: {ARTICULATION_ROOT_PATH}")

    return robot
```

这里涉及两个重要对象：

```python
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
```

`World` 是 Isaac Sim 的仿真上下文。读取 Articulation 状态前，通常要保证 physics context 已初始化，并进入 play 状态。

`SingleArticulation` 是 Isaac Sim 对一个已有 articulation prim 的 Python 封装。它不会创建新机器人，只是绑定当前 stage 中已经存在的机器人。

关键调用：

```python
await world.play_async()
await omni.kit.app.get_app().next_update_async()
```

这两行的作用是让 Isaac Sim 进入物理仿真状态，并等待一帧。否则 articulation view 可能还没初始化，读取关节状态容易失败或返回空值。

## 读取 DOF 顺序

脚本中使用：

```python
def get_dof_names(robot):
    try:
        return list(robot.dof_names)
    except Exception:
        pass

    view = getattr(robot, "_articulation_view", None)
    if view is not None:
        try:
            return list(view.dof_names)
        except Exception:
            pass

    return []
```

DOF 顺序非常关键。

后续我们会把 Isaac 读出来的 `q_current` 直接传给 cuRobo：

```text
q_current = [q1, q2, q3, q4, q5, q6]
```

如果 Isaac 的 DOF 顺序和 cuRobo 的 `cspace.joint_names` 不一致，同一个数组就会被解释成错误的关节角。

当前期望顺序是：

```text
['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']
```

脚本会强制检查：

```python
if dof_names != EXPECTED_JOINT_ORDER:
    raise RuntimeError(...)
```

这一步宁可提前报错，也不要继续用错误的关节顺序做 FK/IK。

## 读取关节位置和速度

主流程中使用：

```python
q_current = safe_numpy(robot.get_joint_positions())

try:
    dq_current = safe_numpy(robot.get_joint_velocities())
except Exception:
    dq_current = np.zeros_like(q_current)
```

`robot.get_joint_positions()` 返回当前 articulation 的 DOF 位置，顺序就是 `dof_names` 的顺序。

`robot.get_joint_velocities()` 返回当前 DOF 速度。当前你是在静止状态导出，所以输出是：

```text
dq_current: [0. 0. 0. 0. 0. 0.]
```

`safe_numpy()` 的作用是把 Isaac 返回的数据统一整理成一维 numpy array：

```python
def safe_numpy(value):
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(float, copy=False)
```

这是因为 Isaac 的 API 有时会返回形如 `[1, dof]` 的 batch 数据，有时返回 `[dof]`。后续保存 JSON 和传给 cuRobo 时，我们希望统一成 `[dof]`。

## 读取 base_link 和 TCP_link 的 world pose

脚本中使用：

```python
def usd_pose_to_matrix(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim path 不存在: {prim_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    usd_matrix = cache.GetLocalToWorldTransform(prim)

    translation = usd_matrix.ExtractTranslation()
    rotation = usd_matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()

    position = np.array([translation[0], translation[1], translation[2]], dtype=float)
    quaternion_wxyz = normalize_quat_wxyz(
        [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]]
    )

    return pose_to_matrix(position, quaternion_wxyz)
```

关键 API：

```python
stage.GetPrimAtPath(prim_path)
```

根据 prim path 获取 USD prim。

```python
UsdGeom.XformCache(Usd.TimeCode.Default())
```

创建一个 transform cache，用于高效计算 prim 的 world transform。

```python
cache.GetLocalToWorldTransform(prim)
```

获取该 prim 相对于 world frame 的 4x4 USD transform。

```python
ExtractTranslation()
ExtractRotationQuat()
```

分别提取平移和旋转四元数。

这里统一使用四元数顺序：

```text
wxyz
```

这是 cuRobo `Pose.quaternion` 使用的顺序。

## 坐标系关系

脚本中核心坐标变换是：

```python
world_from_base = usd_pose_to_matrix(stage, ROBOT_BASE_FRAME_PATH)
world_from_tcp = usd_pose_to_matrix(stage, TCP_FRAME_PATH)

base_from_world = np.linalg.inv(world_from_base)
base_from_tcp = base_from_world @ world_from_tcp
```

含义是：

```text
T_world_base = base_link 在 world frame 下的位姿
T_world_tcp  = TCP_link 在 world frame 下的位姿
T_base_world = inverse(T_world_base)
T_base_tcp   = T_base_world @ T_world_tcp
```

后续 cuRobo FK 的输出也是：

```text
TCP_link in base_link frame
```

所以我们会用这里导出的 `base_from_tcp` 和 cuRobo FK 结果做对比。

## JSON 输出结构

脚本最终保存：

```text
/tmp/mec_arm_isaac_state.json
```

JSON 结构如下：

```json
{
  "schema": "mec_arm_isaac_state_v1",
  "source": "Isaac Sim Script Editor",
  "robot": {
    "articulation_root_path": "/World/mec_arm/root_joint",
    "base_frame_path": "/World/mec_arm/base_link",
    "tcp_frame_path": "/World/mec_arm/Empty_Link6/TCP_link"
  },
  "joint_state": {
    "joint_names": ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"],
    "q_current": [],
    "dq_current": []
  },
  "transforms": {
    "world_from_base": [],
    "world_from_tcp": [],
    "base_from_tcp": []
  },
  "poses": {
    "tcp_world": {
      "position_xyz": [],
      "quaternion_wxyz": []
    },
    "tcp_base": {
      "position_xyz": [],
      "quaternion_wxyz": []
    }
  }
}
```

其中最重要的是：

```text
joint_state.q_current
poses.tcp_base
transforms.base_from_tcp
```

下一步 `check_fk_align.py` 会读取它们。

## 本次运行结果

你本次运行输出：

```text
========== Dump Isaac Robot State ==========
[Isaac] articulation root: /World/mec_arm/root_joint
[Isaac] base frame: /World/mec_arm/base_link
[Isaac] tcp frame: /World/mec_arm/Empty_Link6/TCP_link
[Isaac] DOF order: ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']
[Isaac] q_current: [-4.440384e-06 -8.583847e-04  6.503440e-08  5.961021e-06  1.631195e-04 -5.937852e-07]
[Isaac] dq_current: [0. 0. 0. 0. 0. 0.]
[Isaac] TCP world position: [-1.051486  1.400999  0.973004]
[Isaac] TCP world quat_wxyz: [-5.070024e-06  9.999365e-01  4.125540e-05 -1.126546e-02]
[Isaac] TCP base position: [0.148514 0.000999 0.104366]
[Isaac] TCP base quat_wxyz: [-5.070024e-06  9.999365e-01  4.125540e-05 -1.126546e-02]
[输出] 已保存 Isaac 状态 JSON: /tmp/mec_arm_isaac_state.json
========== Dump complete ==========
```

这说明：

- Isaac Sim 能正确读取 articulation。
- DOF 顺序和 cuRobo yml 一致。
- 当前机械臂几乎在零位附近。
- TCP 在 base_link 下的位置约为：

```text
[0.148514, 0.000999, 0.104366]
```

这个值会成为下一步 cuRobo FK 对齐检查的参考值。

## 常见错误

### get_stage_file_path 报错

错误：

```text
AttributeError: 'omni.usd._usd.UsdContext' object has no attribute 'get_stage_file_path'
```

原因是 Isaac Sim 5.1.0 的 `UsdContext` 没有这个 API，而且 Script Editor 中不应该重新打开 stage。

正确写法：

```python
stage = omni.usd.get_context().get_stage()
```

### SingleArticulation 无效

如果出现：

```text
SingleArticulation 无效: /World/mec_arm/root_joint
```

优先检查：

- 机器人 prim path 是否变化。
- 当前 stage 是否已经打开机器人。
- articulation root 是否仍然是 `/World/mec_arm/root_joint`。
- 是否已经进入 play 状态并等待一帧。

### prim path 不存在

如果出现：

```text
prim path 不存在: /World/mec_arm/Empty_Link6/TCP_link
```

说明 TCP prim path 和当前 stage 不一致。需要重新扫描 stage，确认 `TCP_link` 是否还存在，或者是否被导入器合并/省略。

### DOF 顺序不一致

如果 DOF 顺序不等于：

```text
['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']
```

不要继续规划。先确认 Isaac Sim 导入模型、URDF、cuRobo yml 是否来自同一个版本。

## 后续代码文件规划

当前已经完成：

```text
check_generated_robot_model.py
  验证 cuRobo 能加载 mec_arm_from_urdf.yml，并能计算 default q 的 FK。

dump_isaac_state.py
  从 Isaac Sim 当前 stage 导出 q_current、DOF order、base/TCP 位姿。
```

接下来按这个顺序写代码。

### 1. check_fk_align.py

路径：

```text
mec_arm_sim/scripts/curobo/check_fk_align.py
```

运行位置：

```text
普通终端，使用 isaacsim51_3dgs_grasp conda Python。
```

输入：

```text
/tmp/mec_arm_isaac_state.json
mec_arm_sim/configs/curobo/mec_arm_from_urdf.yml
```

作用：

```text
读取 Isaac 导出的 q_current
-> 用 cuRobo compute_kinematics 做 FK
-> 得到 TCP_link in base_link
-> 和 JSON 中的 tcp_base 对比
```

通过标准：

```text
position error < 1e-3 m
orientation error < 0.5 deg
joint_names 完全一致
```

如果这一步失败，先不做 IK 和 MotionPlanner。

### 2. check_ik_current_tcp.py

路径：

```text
mec_arm_sim/scripts/curobo/check_ik_current_tcp.py
```

作用：

```text
把 Isaac 当前 tcp_base pose 作为 IK target
-> cuRobo 求 IK
-> 对 IK 解再做 FK
-> 检查是否回到同一个 TCP pose
```

通过标准：

```text
position error < 0.01 m
orientation error < 3 deg
q_solution 在 joint limits 内
```

### 3. check_plan_empty.py

路径：

```text
mec_arm_sim/scripts/curobo/check_plan_empty.py
```

作用：

```text
q_current + 小幅 TCP target offset
-> 不加载外部障碍物
-> MotionPlanner 生成 joint trajectory
```

这一步只验证 planner 本体，不碰 cuboid 和 3DGS。

### 4. check_plan_cuboid.py

路径：

```text
mec_arm_sim/scripts/curobo/check_plan_cuboid.py
```

作用：

```text
q_current + target_tcp_pose + smoke_cuboid
-> MotionPlanner 生成避障 trajectory
```

这一步再验证外部障碍物。

### 5. visualize_traj_isaac.py

路径：

```text
mec_arm_sim/scripts/curobo/visualize_traj_isaac.py
```

运行位置：

```text
Isaac Sim Script Editor
```

作用：

```text
读取 cuRobo 输出 trajectory JSON
-> 在 Isaac Sim 里画 TCP path、waypoints、start、goal、obstacle
```

这一步只可视化，不驱动机械臂。

### 6. exec_traj_isaac.py

路径：

```text
mec_arm_sim/scripts/curobo/exec_traj_isaac.py
```

运行位置：

```text
Isaac Sim Script Editor
```

作用：

```text
读取 joint trajectory
-> 使用 ArticulationController / position target 逐帧执行
-> 记录 q_actual
-> 对比 q_desired 和 q_actual
```

这一步才进入执行，不是当前 FK 对齐阶段的任务。
