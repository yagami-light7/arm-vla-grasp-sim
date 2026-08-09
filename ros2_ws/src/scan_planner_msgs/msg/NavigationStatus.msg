# PCT + SCAN 导航 supervisor 的类型化状态快照。
uint8 STATE_IDLE=0
uint8 STATE_GLOBAL_PLANNING=1
uint8 STATE_LOCAL_PLANNING=2
uint8 STATE_TRACKING=3
uint8 STATE_GLOBAL_REPLAN=4
uint8 STATE_EMERGENCY_STOP=5
uint8 STATE_GOAL_REACHED=6
uint8 STATE_UNKNOWN=255

std_msgs/Header header
uint64 status_sequence
uint64 state_revision
uint64 goal_id
uint8 state

bool allow_tracking_command
bool force_zero_velocity
bool stop_confirmed

bool global_replan_requested
bool global_replan_in_flight
uint64 global_replan_request_id
uint64 pct_plan_id
builtin_interfaces/Time active_path_stamp

uint32 consecutive_scan_failures
string[] stale_inputs
string reason
