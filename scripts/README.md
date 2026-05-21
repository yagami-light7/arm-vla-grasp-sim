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
