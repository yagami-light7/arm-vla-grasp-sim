# SCAN 闭环控制器的可审计轨迹生命周期状态。
uint8 STATE_WAITING_FOR_TRAJECTORY=0
uint8 STATE_WAITING_FOR_REFERENCE_PATH=1
uint8 STATE_WAITING_FOR_ODOMETRY=2
uint8 STATE_WAITING_FOR_CLOUD=3
uint8 STATE_TRAJECTORY_TIMEOUT=4
uint8 STATE_ODOMETRY_TIMEOUT=5
uint8 STATE_CLOUD_TIMEOUT=6
uint8 STATE_INVALID_CLOCK=7
uint8 STATE_EMERGENCY_STOP=8
uint8 STATE_ALIGNING_YAW=9
uint8 STATE_TRACKING=10
uint8 STATE_TRAJECTORY_FINISHED=11
uint8 STATE_GOAL_REACHED=12
uint8 STATE_UNKNOWN=255

uint8 EVENT_INITIAL=0
uint8 EVENT_ACCEPTED=1
uint8 EVENT_REJECTED=2
uint8 EVENT_INVALIDATED=3
uint8 EVENT_STATE_CHANGED=4
uint8 EVENT_DUPLICATE=5

# header.stamp 是本条状态的发布时间，frame_id 是控制器世界坐标系。
std_msgs/Header header
uint64 status_sequence
uint64 acceptance_sequence
uint8 event

# 以下四项始终标识当前或最近失效的已接受 B-spline，不会被拒绝候选覆盖。
builtin_interfaces/Time reference_path_stamp
builtin_interfaces/Time bspline_header_stamp
builtin_interfaces/Time start_time
int64 traj_id

# accepted 表示上述身份曾被接受；trajectory_valid 表示 tracker 仍保留该结构化轨迹。
bool accepted
bool trajectory_valid
bool is_final
bool emergency_stop
uint8 state
string reason

# REJECTED 事件把被拒消息身份放在独立候选字段，避免覆盖仍有效的当前快照。
bool candidate_present
builtin_interfaces/Time candidate_reference_path_stamp
builtin_interfaces/Time candidate_bspline_header_stamp
builtin_interfaces/Time candidate_start_time
int64 candidate_traj_id

# 当前已接受 identity 的逐拍输出聚合。active_sensing_yaw_only 由 controller
# 对 B-spline 位置与 yaw payload 做严格结构判定，不依赖日志或 reason 文本。
bool active_sensing_yaw_only
uint64 command_sample_count
geometry_msgs/Twist first_command
float64 max_abs_vx
float64 max_abs_vy
float64 max_abs_wz
uint64 command_violation_count
