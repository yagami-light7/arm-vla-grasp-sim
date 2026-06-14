# Go2-X5 Script Layout

The final pick and retreat demo uses one ordered script chain:

1. `isaac/01_export_go2_x5_state.py`
2. `isaac/02_generate_grasp_target.py`
3. `curobo/03_plan_grasp_trajectory.py`
4. `isaac/04_execute_grasp_sequence.py`
5. `isaac/05_run_pick_retreat_demo.py`

Run step 05 from Isaac Sim Script Editor for the one-click demo. It calls steps
01, 02, 03, and 04 in order.

`curobo/grasp_planner_server.py` is the optional persistent planner service for
step 03.

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
`pure_physics_success=false`; the existing video baseline remains unchanged.

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
  --seed 100 \
  --full-physics
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
  --full-physics \
  --randomize-task \
  --show-randomization-debug \
  --no-headless \
  --keep-window-open
```

`--full-physics` uses one IsaacLab stage/runtime for nav-to-pick, online
current-state cuRobo pick planning, physical pick execution, closed-gripper
carry navigation, online current-state cuRobo place planning, physical place
execution, and LeRobot export. It does not accept `--pick-plan-json` or
`--place-plan-json`; those offline plan files are only for
`--manipulation-apply-smoke`. The old `--integrated-apply-smoke` flag is kept
only as an error message that points to `--full-physics`.

Mechanical-arm execution locks the floating base root pose and support joints
by default because the current locomotion policy was not trained for large
arm-induced center of mass changes. Use
`--no-lock-base-during-manipulation --no-lock-support-joints-during-manipulation`
only when validating a new locomotion policy. The default lock applies only in
manipulation and terminal hold phases; navigation remains physically driven.
Any lock use is recorded as `used_manipulation_base_lock=true` and
`used_manipulation_support_joint_lock=true`, so successful default runs report
`stable_physics_success=true` and `pure_physics_success=false`.
