uint8 IDLE=0
uint8 WAITING_FOR_ODOMETRY=1
uint8 PLANNING=2
uint8 SUCCEEDED=3
uint8 NO_PATH=4
uint8 ERROR=5

uint8 COMMAND_NONE=0
uint8 COMMAND_PLAN=1
uint8 COMMAND_REPLAN=2
uint8 COMMAND_CANCEL=3

std_msgs/Header header
uint64 plan_id
uint64 goal_id
uint64 request_id
uint8 command
uint8 state
builtin_interfaces/Time goal_stamp
builtin_interfaces/Time path_stamp
bool has_active_goal
geometry_msgs/PoseStamped active_goal
builtin_interfaces/Time active_path_stamp
string message
uint32 path_point_count
