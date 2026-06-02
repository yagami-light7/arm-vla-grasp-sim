# Go2-X5 Navigation and Manipulation Integration

## Verified Starting Point

The existing Isaac Sim GUI demo exports the full Go2-X5 articulation state,
`T_world_base`, `T_base_tcp`, and local cuboid collisions. It then generates a
bbox grasp target, plans `arm_joint1` through `arm_joint6` in an external
cuRobo process, and executes the arm trajectory on the full articulation.

The DWA reference repository provides a separate Isaac Lab navigation runtime:

```text
USD /World/scene_collision
  -> occupancy.pgm + map.json
  -> OccupancyGridMap.inflate()
  -> AStarPlanner.plan()
  -> DWAController.compute_command()
  -> body command [vx, 0.0, wz]
  -> Isaac Lab base_velocity command term
  -> RSL-RL policy
```

The reference DWA controller is non-holonomic: v1 intentionally keeps
`vy = 0.0`. The imported USD robot asset matches this repository, while this
repository's URDF additionally defines `grasp_tcp_link`; the main URDF remains
authoritative.

## Runtime Architecture

The first integrated version uses a deliberate two-stage boundary:

```text
Isaac Lab navigation runtime
  -> A* + DWA + Go2 locomotion policy
  -> episodes/<task>/<episode>/nav and yaw_align
  -> /tmp/go2_x5_nav_result.json

Isaac Sim GUI grasp runtime
  -> restore final_base_pose_world
  -> settle and velocity check
  -> GraspPipeline
  -> external cuRobo server or one-shot process
  -> episodes/<task>/<episode>/grasp
```

This avoids loading Isaac Sim `omni.warp` and cuRobo's Warp/CUDA stack into the
same process. A future standalone Isaac Sim grasp runner can automate the
handoff without changing the JSON contract.

## Implemented Modules

- `source/navigation/navlib`: selected occupancy-grid, A*, DWA, rendering, and
  path serialization modules migrated from `Automatonzy/DWA`.
- `source/navigation/adapters/frame_utils.py`: rotated map-origin transforms,
  yaw wrapping, body/world velocity transforms, and planar base transforms.
- `source/navigation/adapters/dwa_nav_adapter.py`: stable `NavPlanner` API.
- `source/navigation/adapters/isaaclab_go2_adapter.py`: command-conditioned
  Isaac Lab locomotion adapter with settle and recorder snapshots.
- `source/navigation/adapters/terrain_utils.py`: collision-only USDA wrapper
  generation so indoor navigation imports `/World/scene_collision` without
  importing the full scene or its original Go2-X5 prim as terrain.
- `source/data`: JSON task parser, fixed-schema phase recorder, and
  validation-first LeRobot converter interface.
- `source/manipulation/grasp_pipeline.py`: callable wrapper around the existing
  export, target, external plan, and execution scripts.
- `source/robot_lab`: minimal Go2-X5 locomotion task package migrated from the
  reference repository and pointed at this repository's URDF.

## Validation Sequence

Run pure Python checks first:

```bash
python -m unittest discover -s tests -v

python scripts/navigation/visualize_astar_dwa.py \
  --map source/scene/nav_maps/839920/map.json \
  --start X Y \
  --start-yaw YAW \
  --goal X Y \
  --inflate-radius 0.40 \
  --local-clearance-radius 0.35
```

Export a map from Isaac Lab:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/navigation/export_nav_map.py \
  --map source/scene/839920_go2_x5.usd \
  --output-dir source/scene/nav_maps/839920
```

Run navigation on a GPU-enabled Isaac Lab host:

```bash
export GO2_X5_CHECKPOINT=/absolute/path/to/flat/model_8500.pt
test -f "$GO2_X5_CHECKPOINT"

python scripts/pipeline/run_nav_then_pick.py \
  --task-json tasks/nav_pick_example.json \
  --task RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint "$GO2_X5_CHECKPOINT" \
  --isaaclab-launcher /path/to/IsaacLab/isaaclab.sh \
  --nav-only \
  --head-camera
```

The checkpoint must be the Go2-X5 RSL-RL locomotion checkpoint. Do not pass a
documentation placeholder or a `.pt` file trained for another task.

After navigation succeeds, run the printed command in Isaac Sim Script Editor.

Indoor navigation uses a collision-only terrain wrapper. It does not add a
second ground plane by default, matching the reference DWA `play_nav_cs.py`
runtime. If a collision subtree genuinely lacks a walkable floor, use
`--add-nav-ground --ground-height Z`. Use
`--flat-terrain --debug-command 0.3 0 0` to isolate locomotion-policy
validation from indoor collision loading.

The occupancy-grid inflation treats unknown space beyond the raster as occupied.
Map export conservatively rasterizes triangle edges before filling triangle
interiors. This preserves vertical collision walls whose top-down XY projection
has zero area. Regenerate `occupancy.pgm` and `map.json` after changing exporter
logic or the collision USD.
The GUI handoff rejects historical navigation results with a stale task goal or
without obstacle and map-boundary clearance before teleporting the robot. It restores planar `x/y/yaw`
while keeping the GUI stage's stable root height, then settles before exporting
the grasp state. `GraspPipeline` rejects targets clearly outside the arm
workspace before invoking external cuRobo.

The Isaac Lab navigation runner also has a stall watchdog. If at least
`--stall-min-forward-ratio` of a `--stall-window-steps` window requests forward
motion but displacement remains below `--stall-min-progress`, the run ends with
`nav_collision`. Occasional slow DWA commands no longer reset the full window.
The local planner truncates candidate rollouts once they enter the goal
tolerance. This matters for grasp handoff poses: a valid approach command must
not be rejected because the remainder of its fixed prediction horizon passes
through the table or wall behind the requested stopping pose. Runtime logs
include DWA candidate counts, clearance, base roll/pitch, and foot/non-foot
contact-force maxima for follow-up GPU diagnosis.
Candidate collision sampling is limited to the locomotion control period so a
rollout cannot skip an intermediate occupancy cell. The apple-pick route has a
narrow turn and should be launched with `--inflate-radius 0.25`,
`--local-clearance-radius 0.25`, and `--prediction-horizon 0.90`. The larger
`0.40/0.35 m` footprint remains appropriate for the open-corridor smoke test.

Use `tasks/nav_smoke_example.json` for a safe indoor locomotion-only route. Use
`tasks/nav_pick_example.json` for the first apple-pick candidate route. The
candidate base pose remains subject to GPU playback and grasp verification.

## Known Risks

- GPU playback has validated Foundation checkpoint loading, the `259 -> 18`
  policy interface, command injection, and real forward movement on flat
  terrain. Conservative wall rasterization increased occupied cells from
  `18196` to `21611`. Two indoor replays stopped about `0.35 m` before their
  respective goals because fixed-horizon DWA collision checks continued past
  the goal toward nearby geometry. Goal-aware rollout truncation is now covered
  by a regression test and a real indoor smoke replay. The apple-pick route
  passes pure-Python A* and DWA checks with a `0.25/0.25 m` footprint and
  `0.90 s` prediction horizon, but still requires a monitored GPU replay before
  using its result as a grasp handoff.
- The current scene reports unresolved visual USD references during headless
  loading. Collision-map export works, but camera output must be checked on the
  target host.
- The first LeRobot converter validates the stable schema only. Tensor and
  video materialization should be added after a real multi-phase episode is
  captured.
- Carry navigation and placement remain schema-compatible follow-up work.
