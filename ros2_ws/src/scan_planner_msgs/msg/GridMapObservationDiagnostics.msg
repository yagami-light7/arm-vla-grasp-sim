# 与一次已接纳 /cloud_registered 观测严格对应的有界证据。
std_msgs/Header header
uint64 observation_sequence
builtin_interfaces/Time sensor_pose_stamp
geometry_msgs/Point sensor_origin

bool canonical_empty
bool map_fusion_performed
float64 map_resolution
uint32 input_point_count
uint32 accepted_endpoint_count
uint32 hit_endpoint_count
uint32 explicit_free_endpoint_count

# 命中端点按世界坐标发布；总数大于数组长度表示发生了有界抽样。
bool hit_endpoint_samples_truncated
geometry_msgs/Point[<=64] hit_endpoint_samples
# 与 hit_endpoint_samples 逐点对齐的 C++ canonical global voxel index，
# 每个样本依次存放 ix、iy、iz，禁止在审计端用浮点坐标反推边界体素。
int64[<=192] hit_endpoint_sample_voxel_indices_xyz

# 本观测中由 hit 正更新使 raw occupancy 首次穿越为 occupied 的 proof
# 样本。它不参与通用均匀抽样，专门 pin 住连续 occupied epoch 的来源。
uint32 free_to_occupied_transition_count
bool free_to_occupied_transition_samples_truncated
geometry_msgs/Point[<=64] free_to_occupied_transition_hit_samples
int64[<=192] free_to_occupied_transition_voxel_indices_xyz

# 仅统计显式 free ray 对体素执行的 p_miss 更新。
uint32 explicit_free_miss_voxel_count
# 仅统计上述 p_miss 更新导致 raw occupancy 穿越占据门限的体素。
uint32 occupied_to_free_by_explicit_miss_count
bool occupied_to_free_samples_truncated
geometry_msgs/Point[<=64] occupied_to_free_by_explicit_miss_samples
# 与 clear samples 逐点对齐的 canonical index 和本轮占据 epoch 的
# free→occupied 阈值穿越来源 observation sequence。0 表示无可认证来源。
int64[<=192] occupied_to_free_sample_voxel_indices_xyz
uint64[<=64] occupied_to_free_transition_hit_observation_sequences
# 与 clear 样本逐点对齐的原始阈值穿越 hit endpoint 与其 ROS Header 纳秒值；
# 由 GridMap 在整个 occupied epoch 内保存，避免旧 report 的 64 点抽样漏证。
geometry_msgs/Point[<=64] occupied_to_free_transition_hit_samples
int64[<=64] occupied_to_free_transition_hit_header_stamp_ns

# 滑窗移位清理与 p_miss 清理分开记账，禁止把前者当作 ghost clear。
uint32 occupied_removed_by_sliding_reset_count
