# PCT Multi-Floor Navigation

## Architecture

```text
task JSON
-> EpisodeSpec
-> PCTNavPlanner
-> PCT server
-> NavPlan
-> DwaNavExecutor / RL policy
-> Isaac Sim
```

`PCTNavPlanner` is a global planner adapter. It does not replace the pipeline
state machine. The adapter calls the external PCT server over stdin/stdout JSON,
converts the returned 3D trajectory back to Isaac Sim coordinates, stores the
full path in `NavPlan.metadata["path_3d"]`, and exposes XY waypoints for the
current `DwaNavExecutor`.

## Why PCT

The existing A* planner uses a 2D occupancy grid, which is appropriate for flat
single-floor scenes. PCT plans over tomogram slices, so it can represent
multi-floor routes, stairs, ramps, and suspended or overlapping structures.

DWA or the locomotion RL policy still owns local execution. PCT only provides
the global route. A trained multi-floor policy is required for reliable stair or
ramp traversal.

## Asset Preparation

This repository does not commit large multi-floor assets, tomograms, walkable
maps, or policy checkpoints. Prepare these locally:

1. Add or mount the mutifloor USD/PLY scene.
2. Build the PCT tomogram with the external PCT script, for example
   `scripts/navigation/build_tomogram.py`.
3. Precompute walkable cells with `scripts/navigation/precompute_ply_walkable.py`.
4. Keep the trained locomotion checkpoint under:
   `checkpoints/go2_x5/pct_multifloor/model_*.pt`.

The example task `tasks/nav_pick_place_apple_multifloor_pct.json` is a template.
Update `scene_usd`, target coordinates, floor ids, slice ids, and PCT asset
paths for the actual scene.

## Commands

Full-physics run with PCT:

```bash
python scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --navigation-smoke \
  --global-planner pct \
  --pct-planner-root /path/to/PCT_planner \
  --pct-server-python /path/to/conda/env/bin/python \
  --pct-tomogram-path /path/to/mutifloor.pickle \
  --pct-walkable-path /path/to/mutifloor_ply_walkable.npy \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_XXXXX.pt \
  --policy-profile pct_multifloor
```

Disable A* fallback while debugging PCT:

```bash
python scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_multifloor_pct.json \
  --navigation-smoke \
  --global-planner pct \
  --pct-no-fallback \
  --pct-planner-root /path/to/PCT_planner \
  --pct-tomogram-path /path/to/mutifloor.pickle \
  --pct-walkable-path /path/to/mutifloor_ply_walkable.npy \
  --locomotion-checkpoint checkpoints/go2_x5/pct_multifloor/model_XXXXX.pt \
  --policy-profile pct_multifloor
```

Training scaffold dry run:

```bash
python scripts/navigation/train_pct_multifloor_policy.py --dry-run
```

## Coordinate Frame

The default adapter mode is `sim_to_pct_180deg`:

```text
PCT_x = -sim_x
PCT_y = -sim_y
PCT_z =  sim_z
```

Use `--pct-offset-x`, `--pct-offset-y`, `--pct-scale-x`, and `--pct-scale-y`
when the tomogram origin or units differ from Isaac Sim. If paths appear
mirrored, first check whether x/y signs are reversed.

## Troubleshooting

- Current directory is not `arm_vla_pct`: run from `/home/light/workspace/arm_vla_pct`.
- Current branch is not `pct`: stop and switch outside the task flow only if intended.
- PCT server does not print `READY`: check `--pct-server-script`, `--pct-server-python`, and the PCT environment.
- Tomogram path is wrong: verify `--pct-tomogram-path` and `PCT_TOMOGRAM_PATH`.
- Walkable map path is wrong: verify `--pct-walkable-path` and `PCT_WALKABLE_PATH`.
- Coordinates are mirrored: adjust `--pct-offset-*`, `--pct-scale-*`, or the adapter coord mode.
- Goal snaps to the wrong floor: inspect `metadata["slice_start"]`, `metadata["slice_end"]`, and snap distances.
- Planning works but the robot cannot climb stairs: train or pass a valid `pct_multifloor` locomotion checkpoint.
- Isaac Sim and RViz Vulkan conflict: run only one Vulkan consumer or move one process to a different display/GPU setup.
