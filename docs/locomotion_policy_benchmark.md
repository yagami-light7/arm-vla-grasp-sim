# Go2-X5 low-level velocity tracking benchmark

This worktree tests `pct_multifloor/model_26000.pt` without PCT, DWA, CuRobo, or the Liangzhu scene. It retains the checkpoint's original DogOnly observation/action interface and replaces only the generated rough terrain with an Isaac Lab plane.

The benchmark applies body-frame `vx`, `vy`, and `wz` steps and exports `samples.jsonl`, `segment_metrics.csv`, `velocity_tracking.png`, `summary.json`, and `report.md`.

```bash
cd /mnt/sage_data/workspace/arm_vla_loco_policy_benchmark

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/loco-benchmark-matplotlib \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/locomotion/benchmark_velocity_tracking.py \
  --checkpoint /home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --output-dir /mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/quick \
  --profile quick \
  --headless
```

Use `--profile full --repeats 3` for the formal matrix. Add `--no-headless --real-time` to inspect the robot in Isaac Sim.

For live RViz2 visualization, start this bridge while the benchmark is running:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/locomotion/ros2_velocity_tracking_bridge.py --ros-args \
  -p samples_path:=/mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/quick/samples.jsonl
```

Open the included RViz configuration in another terminal:

```bash
source /opt/ros/humble/setup.bash
rviz2 -d configs/rviz/loco_velocity_tracking.rviz
```

It sets Fixed Frame to `map` and includes the actual path, integrated-command path, velocity MarkerArray, and Odometry topics under `/loco_velocity_tracking_bridge/*`.

A failed segment demonstrates that the low-level closed-loop plant does not match DWA's commanded-velocity assumption in that regime. It does not alone prove every DWA failure is policy-caused; replaying exact DWA command traces is the next causal test.
