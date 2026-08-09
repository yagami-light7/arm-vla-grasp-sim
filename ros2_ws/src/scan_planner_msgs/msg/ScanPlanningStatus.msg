# SCAN 局部规划器的类型化事件与当前快照。
uint8 EVENT_INITIAL=0
uint8 EVENT_REFERENCE_ACCEPTED=1
uint8 EVENT_REFERENCE_CLEARED=2
uint8 EVENT_TRAJECTORY_PUBLISHED=3
uint8 EVENT_PLANNING_FAILED=4
uint8 EVENT_PREDICTED_COLLISION=5
uint8 EVENT_EMERGENCY_STOP=6
uint8 EVENT_RECOVERED=7
uint8 EVENT_GOAL_HOLD=8
uint8 EVENT_STAIR_INHIBITED=9
uint8 EVENT_STAIR_RESUME_WAITING=10

uint8 STATE_WAITING_FOR_REFERENCE=0
uint8 STATE_PLANNING=1
uint8 STATE_TRACKING=2
uint8 STATE_EMERGENCY_STOP=3
uint8 STATE_GOAL_HOLD=4
uint8 STATE_STAIR_INHIBITED=5
uint8 STATE_UNKNOWN=255

# header.stamp 是事件发布时间，frame_id 是 SCAN 使用的世界坐标系。
std_msgs/Header header
uint64 status_sequence
uint8 event
uint8 state

# 当前事件绑定的全局 Path 代际；没有有效 Path 时为零时间。
builtin_interfaces/Time reference_path_stamp

# 若本事件同时发布了一条 B-spline，以下身份必须与该消息逐字段一致。
bool trajectory_present
builtin_interfaces/Time bspline_header_stamp
builtin_interfaces/Time trajectory_start_time
int64 trajectory_id
bool trajectory_is_final
bool trajectory_emergency_stop

uint32 consecutive_planning_failures
bool stop_required
bool global_replan_recommended
string reason
