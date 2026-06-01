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
  --goal X Y \
  --inflate-radius 0.30
```

Export a map from Isaac Lab:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/navigation/export_nav_map.py \
  --map source/scene/839920_go2_x5.usd \
  --output-dir source/scene/nav_maps/839920
```

Run navigation on a GPU-enabled Isaac Lab host:

```bash
python scripts/pipeline/run_nav_then_pick.py \
  --task-json tasks/nav_pick_example.json \
  --checkpoint /path/to/model_8500.pt \
  --isaaclab-launcher /path/to/IsaacLab/isaaclab.sh \
  --nav-only \
  --head-camera
```

After navigation succeeds, run the printed command in Isaac Sim Script Editor.

## Known Risks

- The current sandbox cannot validate CUDA policy playback. Pure Python map,
  planner, transform, recorder, and coordinator checks remain runnable.
- The current scene reports unresolved visual USD references during headless
  loading. Collision-map export works, but camera output must be checked on the
  target host.
- The first LeRobot converter validates the stable schema only. Tensor and
  video materialization should be added after a real multi-phase episode is
  captured.
- Carry navigation and placement remain schema-compatible follow-up work.
