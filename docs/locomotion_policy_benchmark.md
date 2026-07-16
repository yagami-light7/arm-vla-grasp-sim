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

Use `--profile full --repeats 3` for the formal matrix. Omit `--headless` and add `--real-time` to inspect the robot in Isaac Sim.

Run one user-defined command, where values are body-frame `VX VY WZ`:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/loco-benchmark-matplotlib \
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/locomotion/benchmark_velocity_tracking.py \
  --checkpoint /home/light/workspace/arm_vla_liangzhu/checkpoints/go2_x5/pct_multifloor/model_26000.pt \
  --output-dir /mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/user_command \
  --command 0.20 0.05 0.30 \
  --hold-seconds 10 \
  --stop-seconds 2 \
  --real-time
```

Repeat `--command` to run several commands, or use a JSON sequence with per-command durations:

```bash
--commands-json configs/locomotion/custom_velocity_commands.example.json
```

For live RViz2 visualization, start this bridge while the benchmark is running:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/locomotion/ros2_velocity_tracking_bridge.py --ros-args \
  -p samples_path:=/mnt/sage_data/outputs/arm_vla_loco_policy_benchmark/quick/samples.jsonl
```

The bridge replays one sample every 20 ms by default, matching the benchmark's 50 Hz control rate. Set `-p rows_per_poll:=N` only when faster replay is desired.

Open the included RViz configuration in another terminal:

```bash
source /opt/ros/humble/setup.bash
rviz2 -d configs/rviz/loco_velocity_tracking.rviz
```

It sets Fixed Frame to `map` and includes the actual path, integrated-command path, velocity MarkerArray, and Odometry topics under `/loco_velocity_tracking_bridge/*`.

RViz is retained for spatial trajectories. For the requested time-series view, run the dedicated ROS2 plot window in a third terminal:

```bash
cd /mnt/sage_data/workspace/arm_vla_loco_policy_benchmark
source /opt/ros/humble/setup.bash
MPLCONFIGDIR=/tmp/loco-ros-plot \
/usr/bin/python3 scripts/locomotion/ros2_velocity_tracking_plot.py --ros-args \
  -p window_seconds:=30.0
```

The window contains three independent line charts for `vx`, `vy`, and `wz`. Each chart overlays the commanded velocity with the measured velocity. Run the Isaac benchmark with `--real-time` when observing it live.

A failed segment demonstrates that the low-level closed-loop plant does not match DWA's commanded-velocity assumption in that regime. It does not alone prove every DWA failure is policy-caused; replaying exact DWA command traces is the next causal test.
