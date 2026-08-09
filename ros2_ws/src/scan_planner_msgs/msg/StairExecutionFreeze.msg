# Isaac 楼梯 root-lock 对当前 PCT Path 的规划抑制快照。
# Header 时间戳使用连续 ROS/仿真时钟，frame_id 与 Path 的 world frame 一致。
std_msgs/Header header

# 当前冻结状态唯一绑定的 nav_msgs/Path.header.stamp。
builtin_interfaces/Time reference_path_stamp

# writer_id 标识逻辑发布器；writer_epoch 标识本次发布器生命周期。
# 两者均不能为空，订阅端在进程生命周期内只接受首个固定组合。
string writer_id
string writer_epoch

# 同一 writer epoch 内从 1 开始严格递增。
uint64 sequence

# true 抑制 SCAN 规划；false 仅在身份、序列和新鲜度全部通过后解除。
bool frozen
