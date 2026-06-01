# Navigation-Pick Episode Format

Each task uses one JSON file and produces one multi-phase episode:

```text
episodes/<task-id>/<episode-id>/
  task.json
  summary.json
  nav/data.csv
  nav/images/front/
  yaw_align/data.csv
  yaw_align/images/front/
  grasp/data.csv
  grasp/images/front/
  grasp/images/wrist/
  place/data.csv
  place/images/front/
```

Only phases that execute need a `data.csv`. All phase CSV files use the same
columns so conversion does not depend on the active controller.

Navigation rows record body-frame `[cmd_vx, cmd_vy, cmd_wz]`. The v1 DWA
controller always writes `cmd_vy = 0.0`. Grasp rows record arm joint targets
and gripper targets. Fields that are unavailable in a phase remain empty.

`summary.json` records `success`, `failure_reason`, the navigation handoff, and
the grasp execution summary. Stable failure reasons are:

```text
nav_timeout
nav_collision
yaw_align_failed
base_not_stable
grasp_target_unreachable
curobo_plan_failed
arm_tracking_failed
gripper_failed
object_not_lifted
```

Validate an episode before implementing or running LeRobot materialization:

```bash
python -m source.data.lerobot_converter \
  --episode-dir episodes/<task-id>/<episode-id>
```
