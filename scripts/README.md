# Go2-X5 Script Layout

This repository now keeps only the full-physics nav-pick-place pipeline and its
supporting tools.  The old Script Editor/video-baseline handoff scripts have
been removed from this branch.

Current maintained entrypoints:

1. `pipeline/run_full_physics_pipeline.py` for one episode.
2. `pipeline/run_full_physics_batch.py` for automation.
3. `pipeline/validate_lerobot_episode.py` for exported data checks.
4. `curobo/03_plan_grasp_trajectory.py` for cuRobo one-shot planning fallback.
5. `curobo/grasp_planner_server.py` for persistent online planning.

Setup, inspection, FK checks, and single-purpose diagnostics live under
`dev_tools/` so they do not look like required demo steps.

## Full-physics refactor

The experimental single-process pipeline starts at:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_dry_run \
  --num-episodes 1 \
  --seed 0 \
  --dry-run
```

The dependency-free control-flow dry run always sets
`pure_physics_success=false`; legacy video-baseline paths are not maintained in this branch.

Episode-level pick/place XY randomization is enabled by default. Use
`--no-randomize-task` when reproducing the validated fixed task:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_randomized_dry_run \
  --num-episodes 3 \
  --seed 100 \
  --randomize-task \
  --show-randomization-debug \
  --dry-run
```

`--show-randomization-debug` defaults to off. In a real GUI run it creates
green pick and blue place region guides plus sampled-position markers; these
display prims use the viewport-visible default USD purpose and have no
collision or rigid-body API. Seeds advance as `seed + episode_index`.

Real Isaac modes are intentionally one episode per process in
`run_full_physics_pipeline.py`. Use the batch launcher below for automation; it
starts one child process per episode so each run still owns exactly one Isaac
World lifecycle:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_batch.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir /tmp/full_physics_random_batch \
  --num-episodes 3 \
  --seed 100
```

The batch launcher writes per-run summaries under
`episode_000000/episode_000000/summary.json` and a top-level
`batch_summary.jsonl`.

`--simulation-smoke` intentionally exits immediately after stage build and
episode reset. Add `--keep-window-open --no-headless` when the purpose is to
inspect the generated stage or randomization guides. The runtime pauses before
the GUI hold loop, so this option does not introduce a second physics loop.

The second-stage real scene/reset smoke check is:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir outputs/full_physics_simulation_smoke \
  --simulation-smoke
```

`--simulation-smoke` launches Isaac Sim, opens one Stage, creates one World,
checks the robot, collision scene, task object, and camera, then performs one
episode reset. It does not invoke navigation, cuRobo, or arm control, and it
never reports `pure_physics_success=true`.

The third-stage physical navigation smoke check is:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_smoke_example.json \
  --output-dir outputs/full_physics_navigation_smoke \
  --seed 31 \
  --navigation-smoke
```

`--navigation-smoke` launches Isaac Lab, loads the locomotion policy, plans an
A* route, and lets the pipeline drive DWA velocity commands one tick at a time.
It exits after `nav_to_pick_success`, so it validates physical navigation
without claiming full pick/place or `pure_physics_success`. Navigation handoff
matches the latest video baseline: success requires base XY to enter the
position tolerance; yaw and base velocity are recorded for diagnostics but do
not reject the episode because the arm planner can absorb remaining yaw error.

The full online nav-pick-place physics check is:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/conda_envs/isaacsim51_3dgs_grasp/bin/python -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --task-json tasks/nav_pick_place_apple_contact.json \
  --output-dir /tmp/full_physics_online_place_gui \
  --seed 100 \
  --show-randomization-debug \
  --no-headless \
  --keep-window-open
```

Full physics is the default mode. It uses one IsaacLab stage/runtime for nav-to-pick, online
current-state cuRobo pick planning, physical pick execution, closed-gripper
carry navigation, online current-state cuRobo place planning, physical place
execution, and LeRobot export. It does not accept `--pick-plan-json` or
`--place-plan-json`; those offline plan files are only for
`--manipulation-apply-smoke`.

Mechanical-arm execution locks the floating base root pose and support joints
by default because the current locomotion policy was not trained for large
arm-induced center of mass changes. These stable defaults are fixed in
`FullPhysicsConfig` rather than exposed as production CLI switches. The lock
applies only in manipulation and terminal hold phases; navigation remains physically driven.
Any lock use is recorded as `used_manipulation_base_lock=true` and
`used_manipulation_support_joint_lock=true`, so successful default runs report
`stable_physics_success=true` and `pure_physics_success=false`.

Full-physics data recording follows `/home/light/workspace/DWA`:

- physics `dt=0.0025` (400 Hz in the current DWA source), control 50 Hz;
- `RecordingSettings.dataset_fps` defaults to 5 Hz and samples on a fixed
  dataset-time grid; it can be changed to 10/15 Hz without changing physics;
- `480x640` RGB JPEG at quality 90, named `camera0_00000.jpg`;
- raw files: `data.csv`, `samples.jsonl`, and optionally `images/<camera>/`;
- LeRobot v2.1 files: `data/chunk-*/*.parquet`,
  `videos/chunk-*/observation.images.<camera>/*.mp4`, and `meta/*`.

The raw `data.csv` keeps DWA's 17-dimensional robot state and measured base
velocity columns. `samples.jsonl` adds synchronized object state, TCP
quaternion, pipeline phase, and the trainable 10-dimensional command vector:
base command 3 + arm joint targets 6 + two gripper joint targets. The LeRobot
`action` feature uses this 11-dimensional high-level command vector. Parquet
also stores body-frame `observation.base_velocity=[vx,vy,wz]` and
`pipeline_state`; image features remain video-backed and are declared in
`meta/info.json`.

Batch episode files are written directly under
`<output-dir>/episode_000000/`, `<output-dir>/episode_000001/`, and so on.
There is no second nested `episode_000000` directory. Batch runs merge all
successful episodes into `<output-dir>/lerobot_dataset`.
The merger only uses successful episodes from the current batch invocation, so
an existing output directory cannot silently inject older episodes.
Existing raw episodes can be converted again without rerunning simulation:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python -m source.data.lerobot_converter \
  --episodes-root outputs/full_physics_batch
```

Validate either a single episode or the unified dataset:

```bash
/data/conda_envs/isaacsim51_3dgs_grasp/bin/python \
  scripts/pipeline/validate_lerobot_episode.py \
  --dataset-root outputs/full_physics_batch/lerobot_dataset
```
