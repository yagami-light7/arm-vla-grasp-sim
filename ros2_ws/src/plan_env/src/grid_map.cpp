#include "plan_env/grid_map.h"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace
{
template <typename T>
void load_parameter(rclcpp::Node *node, const std::string &name, T &value, const T &default_value)
{
  if (!node->has_parameter(name))
    node->declare_parameter<T>(name, default_value);
  node->get_parameter(name, value);
}

int64_t message_stamp_ns(const builtin_interfaces::msg::Time &stamp)
{
  return rclcpp::Time(stamp).nanoseconds();
}

bool finite_pose(const geometry_msgs::msg::Pose &pose)
{
  const double values[] = {
      pose.position.x, pose.position.y, pose.position.z,
      pose.orientation.x, pose.orientation.y,
      pose.orientation.z, pose.orientation.w};
  for (const double value : values)
    if (!std::isfinite(value))
      return false;

  const double quaternion_norm =
      std::sqrt(
          pose.orientation.x * pose.orientation.x +
          pose.orientation.y * pose.orientation.y +
          pose.orientation.z * pose.orientation.z +
          pose.orientation.w * pose.orientation.w);
  return quaternion_norm > 1.0e-6;
}

bool canonical_empty_xyz32_cloud(
    const sensor_msgs::msg::PointCloud2 &cloud)
{
  if (cloud.width != 0U || cloud.height != 1U ||
      cloud.point_step != 3U * sizeof(float) || cloud.row_step != 0U ||
      !cloud.data.empty() || cloud.is_bigendian || !cloud.is_dense ||
      cloud.fields.size() != 3U)
    return false;

  const char *expected_names[] = {"x", "y", "z"};
  const std::uint32_t expected_offsets[] = {
      0U, static_cast<std::uint32_t>(sizeof(float)),
      static_cast<std::uint32_t>(2U * sizeof(float))};
  for (std::size_t index = 0; index < cloud.fields.size(); ++index)
  {
    const auto &field = cloud.fields[index];
    if (field.name != expected_names[index] ||
        field.offset != expected_offsets[index] ||
        field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        field.count != 1U)
      return false;
  }
  return true;
}

bool extract_ray_endpoint_hits(
    const sensor_msgs::msg::PointCloud2 &cloud,
    std::vector<char> &endpoint_hits,
    std::string &error)
{
  constexpr const char *field_name = "ray_endpoint_type";
  const sensor_msgs::msg::PointField *endpoint_field = nullptr;
  for (const auto &field : cloud.fields)
  {
    if (field.name != field_name)
      continue;
    if (endpoint_field != nullptr)
    {
      error = "PointCloud2 含重复 ray_endpoint_type 字段";
      return false;
    }
    endpoint_field = &field;
  }

  const std::uint64_t point_count_u64 =
      static_cast<std::uint64_t>(cloud.width) * cloud.height;
  if (point_count_u64 > std::numeric_limits<std::size_t>::max())
  {
    error = "PointCloud2 点数溢出";
    return false;
  }
  endpoint_hits.assign(static_cast<std::size_t>(point_count_u64), 1);
  if (endpoint_field == nullptr)
    return true;

  if (endpoint_field->datatype != sensor_msgs::msg::PointField::UINT8 ||
      endpoint_field->count != 1U || cloud.height == 0U ||
      cloud.point_step == 0U || endpoint_field->offset >= cloud.point_step)
  {
    error = "ray_endpoint_type 必须是 point_step 内的标量 uint8";
    return false;
  }

  const std::uint64_t minimum_row_step =
      static_cast<std::uint64_t>(cloud.width) * cloud.point_step;
  const std::uint64_t required_bytes =
      static_cast<std::uint64_t>(cloud.height) * cloud.row_step;
  if (cloud.row_step < minimum_row_step ||
      required_bytes != cloud.data.size())
  {
    error = "PointCloud2 row_step 或 data 长度与端点字段不匹配";
    return false;
  }

  std::size_t index = 0;
  for (std::uint32_t row = 0; row < cloud.height; ++row)
  {
    for (std::uint32_t column = 0; column < cloud.width; ++column, ++index)
    {
      const std::size_t offset =
          static_cast<std::size_t>(row) * cloud.row_step +
          static_cast<std::size_t>(column) * cloud.point_step +
          endpoint_field->offset;
      const std::uint8_t endpoint_type = cloud.data[offset];
      if (endpoint_type > 1U)
      {
        error = "ray_endpoint_type 只允许 0(自由) 或 1(占据)";
        return false;
      }
      endpoint_hits[index] = endpoint_type == 1U ? 1 : 0;
    }
  }
  return true;
}

std::uint32_t bounded_uint32(const std::size_t value)
{
  return static_cast<std::uint32_t>(std::min<std::size_t>(
      value, std::numeric_limits<std::uint32_t>::max()));
}

void increment_saturated(std::uint32_t &value)
{
  if (value < std::numeric_limits<std::uint32_t>::max())
    ++value;
}

geometry_msgs::msg::Point geometry_point(const Eigen::Vector3d &point)
{
  geometry_msgs::msg::Point result;
  result.x = point.x();
  result.y = point.y();
  result.z = point.z();
  return result;
}

struct OccupiedTransitionDiagnosticCandidate
{
  geometry_msgs::msg::Point hit_endpoint;
  std::array<std::int64_t, 3> voxel_index_xyz;
};

struct ExplicitMissClearDiagnosticCandidate
{
  geometry_msgs::msg::Point clear_point;
  std::array<std::int64_t, 3> voxel_index_xyz;
  std::uint64_t hit_observation_sequence;
  geometry_msgs::msg::Point hit_endpoint;
  std::int64_t hit_header_stamp_ns;
};

std::vector<std::size_t> deterministic_uniform_sample_indices(
    const std::size_t total_count,
    const std::size_t maximum_sample_count)
{
  const std::size_t sample_count =
      std::min(total_count, maximum_sample_count);
  std::vector<std::size_t> indices;
  indices.reserve(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    const double ratio = sample_count == 1U
                             ? 0.0
                             : static_cast<double>(sample) /
                                   static_cast<double>(sample_count - 1U);
    indices.push_back(static_cast<std::size_t>(std::llround(
        ratio * static_cast<double>(total_count - 1U))));
  }
  return indices;
}
}  // namespace

void GridMap::initMap(rclcpp::Node *node)
{
  node_ = node;

  /* get parameter */
  double x_size, y_size, z_size;
  load_parameter(node_, "grid_map.resolution", mp_.resolution_, -1.0);
  load_parameter(node_, "grid_map.sliding_map_size_x", x_size, -1.0);
  load_parameter(node_, "grid_map.sliding_map_size_y", y_size, -1.0);
  load_parameter(node_, "grid_map.sliding_map_size_z", z_size, -1.0);
  load_parameter(node_, "grid_map.local_update_range_x", mp_.local_update_range_(0), x_size / 2.0);
  load_parameter(node_, "grid_map.local_update_range_y", mp_.local_update_range_(1), y_size / 2.0);
  load_parameter(node_, "grid_map.local_update_range_z", mp_.local_update_range_(2), z_size / 2.0);

  load_parameter(node_, "grid_map.obstacles_inflation_z_up", mp_.obstacles_inflation_z_up, -1.0);
  load_parameter(node_, "grid_map.obstacles_inflation_z_down", mp_.obstacles_inflation_z_down, -1.0);
  load_parameter(node_, "grid_map.double_cylinder_radius", mp_.double_cylinder_radius_, -1.0);
  load_parameter(node_, "grid_map.double_cylinder_offset", mp_.double_cylinder_offset_, 0.0);
  load_parameter(node_, "grid_map.map_sliding_en", mp_.map_sliding_en_, true);
  load_parameter(node_, "grid_map.map_sliding_thresh", mp_.map_sliding_thresh_, mp_.resolution_);

  load_parameter(node_, "grid_map.fx", mp_.fx_, -1.0);
  load_parameter(node_, "grid_map.fy", mp_.fy_, -1.0);
  load_parameter(node_, "grid_map.cx", mp_.cx_, -1.0);
  load_parameter(node_, "grid_map.cy", mp_.cy_, -1.0);

  load_parameter(node_, "grid_map.depth_filter_maxdist", mp_.depth_filter_maxdist_, -1.0);
  load_parameter(node_, "grid_map.depth_filter_mindist", mp_.depth_filter_mindist_, -1.0);
  load_parameter(node_, "grid_map.depth_filter_margin", mp_.depth_filter_margin_, -1);
  load_parameter(node_, "grid_map.k_depth_scaling_factor", mp_.k_depth_scaling_factor_, -1.0);
  load_parameter(node_, "grid_map.skip_pixel", mp_.skip_pixel_, -1);

  load_parameter(node_, "grid_map.p_hit", mp_.p_hit_, -1.0);
  load_parameter(node_, "grid_map.p_miss", mp_.p_miss_, -1.0);
  load_parameter(node_, "grid_map.p_min", mp_.p_min_, -1.0);
  load_parameter(node_, "grid_map.p_max", mp_.p_max_, -1.0);
  load_parameter(node_, "grid_map.p_occ", mp_.p_occ_, -1.0);
  load_parameter(node_, "grid_map.max_ray_length", mp_.max_ray_length_, -0.1);

  load_parameter(node_, "grid_map.vis_height", mp_.vis_height_, 0.3);
  load_parameter(node_, "grid_map.show_occ_time", mp_.show_occ_time_, false);

  load_parameter(node_, "grid_map.frame_id", mp_.frame_id_, string("world"));
  load_parameter(node_, "grid_map.sliding_map_frame_id", mp_.sliding_map_frame_id_, string("sliding_map"));
  load_parameter(node_, "grid_map.ground_height", mp_.ground_height_, 0.0);

  load_parameter(node_, "grid_map.sensor_type", mp_.sensor_type_, string("lidar"));
  load_parameter(node_, "grid_map.base_frame_id", mp_.base_frame_id_, string("base_link"));
  load_parameter(node_, "grid_map.cloud_is_world", mp_.cloud_is_world_, true);
  load_parameter(node_, "grid_map.need_extrinsic", mp_.need_extrinsic_, true);
  double cloud_sensor_extrinsic_x;
  double cloud_sensor_extrinsic_y;
  double cloud_sensor_extrinsic_z;
  double cloud_sensor_extrinsic_qw;
  double cloud_sensor_extrinsic_qx;
  double cloud_sensor_extrinsic_qy;
  double cloud_sensor_extrinsic_qz;
  load_parameter(
      node_, "grid_map.cloud_sensor_extrinsic_x",
      cloud_sensor_extrinsic_x, -0.01100);
  load_parameter(
      node_, "grid_map.cloud_sensor_extrinsic_y",
      cloud_sensor_extrinsic_y, -0.02329);
  load_parameter(
      node_, "grid_map.cloud_sensor_extrinsic_z",
      cloud_sensor_extrinsic_z, 0.04412);
  load_parameter(
      node_, "grid_map.cloud_sensor_extrinsic_qw",
      cloud_sensor_extrinsic_qw, 1.0);
  load_parameter(
      node_, "grid_map.cloud_sensor_extrinsic_qx",
      cloud_sensor_extrinsic_qx, 0.0);
  load_parameter(
      node_, "grid_map.cloud_sensor_extrinsic_qy",
      cloud_sensor_extrinsic_qy, 0.0);
  load_parameter(
      node_, "grid_map.cloud_sensor_extrinsic_qz",
      cloud_sensor_extrinsic_qz, 0.0);
  load_parameter(
      node_, "grid_map.observation_timeout_sec",
      mp_.observation_timeout_sec_, 0.5);
  load_parameter(
      node_, "grid_map.max_cloud_pose_skew_sec",
      mp_.max_cloud_pose_skew_sec_, 0.2);
  load_parameter(
      node_, "grid_map.sensor_sync_queue_size",
      mp_.sensor_sync_queue_size_, 100);
  load_parameter(
      node_, "grid_map.diagnostic_max_samples",
      mp_.diagnostic_max_samples_, 64);
  load_parameter(
      node_, "grid_map.diagnostic_history_depth",
      mp_.diagnostic_history_depth_, 64);
  load_parameter(
      node_, "topics.grid_map_observation_diagnostics",
      mp_.observation_diagnostics_topic_,
      string("/planning/grid_map_observation_diagnostics"));
  if (!std::isfinite(mp_.resolution_) || mp_.resolution_ <= 0.0 ||
      !std::isfinite(mp_.double_cylinder_radius_) ||
      mp_.double_cylinder_radius_ <= 0.0 ||
      !std::isfinite(mp_.double_cylinder_offset_) ||
      mp_.double_cylinder_offset_ < 0.0 ||
      !std::isfinite(mp_.obstacles_inflation_z_up) ||
      mp_.obstacles_inflation_z_up < 0.0 ||
      !std::isfinite(mp_.obstacles_inflation_z_down) ||
      mp_.obstacles_inflation_z_down < 0.0)
  {
    throw std::runtime_error(
        "grid_map 分辨率、双圆柱包络和垂向膨胀参数必须为有限合法值");
  }
  if (mp_.frame_id_.empty() || mp_.base_frame_id_.empty())
    throw std::runtime_error("grid_map frame_id 与 base_frame_id 不能为空");
  if (mp_.observation_timeout_sec_ <= 0.0 ||
      mp_.max_cloud_pose_skew_sec_ < 0.0)
    throw std::runtime_error("grid_map 输入超时必须为正，时间差门限不能为负");
  if (mp_.sensor_sync_queue_size_ <= 0)
    throw std::runtime_error("grid_map.sensor_sync_queue_size 必须为正数");
  const double cloud_sensor_extrinsic_values[] = {
      cloud_sensor_extrinsic_x,
      cloud_sensor_extrinsic_y,
      cloud_sensor_extrinsic_z,
      cloud_sensor_extrinsic_qw,
      cloud_sensor_extrinsic_qx,
      cloud_sensor_extrinsic_qy,
      cloud_sensor_extrinsic_qz};
  for (const double value : cloud_sensor_extrinsic_values)
    if (!std::isfinite(value))
      throw std::runtime_error("grid_map 点云传感器外参必须为有限数值");
  Eigen::Quaterniond cloud_sensor_extrinsic_q(
      cloud_sensor_extrinsic_qw,
      cloud_sensor_extrinsic_qx,
      cloud_sensor_extrinsic_qy,
      cloud_sensor_extrinsic_qz);
  if (cloud_sensor_extrinsic_q.norm() <= 1.0e-6)
    throw std::runtime_error("grid_map 点云传感器外参四元数不能为零");
  cloud_sensor_extrinsic_q.normalize();
  // Isaac OGN generic subscriber 与最终 validator 使用固定 KeepLast(64)
  // 合同；若这里允许更小数组或其他 history depth，会让合法 C++ 配置在
  // 跨 DDS 审计端被误判。第一版因此固定为 64，不暴露伪可配置范围。
  if (mp_.diagnostic_max_samples_ != 64 ||
      mp_.diagnostic_history_depth_ != 64 ||
      mp_.observation_diagnostics_topic_.empty())
    throw std::runtime_error(
        "GridMap 诊断样本上限与历史深度必须固定为 64，topic 不能为空");

  mp_.lidar_extrinsic_.setIdentity();
  mp_.lidar_extrinsic_.block<3, 3>(0, 0) =
      cloud_sensor_extrinsic_q.toRotationMatrix();
  mp_.lidar_extrinsic_.block<3, 1>(0, 3) = Eigen::Vector3d(
      cloud_sensor_extrinsic_x,
      cloud_sensor_extrinsic_y,
      cloud_sensor_extrinsic_z);

  mp_.depth_extrinsic_ <<
      0.0,  0.707107, 0.707107, -0.15170,
     -1.0,  0.000000, 0.000000,  0.00000,
      0.0, -0.707107, 0.707107,  0.07510,
      0.0,  0.000000, 0.000000,  1.00000;

  if (mp_.sensor_type_ != "lidar" && mp_.sensor_type_ != "depth")
  {
    RCLCPP_ERROR(node_->get_logger(), "[GridMap] invalid grid_map.sensor_type: %s; falling back to lidar",
                 mp_.sensor_type_.c_str());
    mp_.sensor_type_ = "lidar";
  }

  mp_.resolution_inv_ = 1 / mp_.resolution_;
  mp_.map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_.ground_height_);
  mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

  mp_.prob_hit_log_ = logit(mp_.p_hit_);
  mp_.prob_miss_log_ = logit(mp_.p_miss_);
  mp_.clamp_min_log_ = logit(mp_.p_min_);
  mp_.clamp_max_log_ = logit(mp_.p_max_);
  mp_.min_occupancy_log_ = logit(mp_.p_occ_);
  mp_.unknown_flag_ = 0.01;
  mp_.map_sliding_thresh_vox_ = std::max(1, static_cast<int>(std::ceil(mp_.map_sliding_thresh_ * mp_.resolution_inv_)));

  cout << "hit: " << mp_.prob_hit_log_ << endl;
  cout << "miss: " << mp_.prob_miss_log_ << endl;
  cout << "min log: " << mp_.clamp_min_log_ << endl;
  cout << "max: " << mp_.clamp_max_log_ << endl;
  cout << "thresh log: " << mp_.min_occupancy_log_ << endl;

  for (int i = 0; i < 3; ++i)
    mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

  mp_.map_min_boundary_ = mp_.map_origin_;
  mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;
  posToIndex(mp_.map_origin_, mp_.map_bound_min_idx_);
  mp_.map_bound_max_idx_ = mp_.map_bound_min_idx_ + mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
  mp_.map_origin_idx_ = mp_.map_bound_min_idx_ + mp_.map_voxel_num_ / 2;
  updateMapBoundaryFromIndex();

  // initialize data buffers

  int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);

  md_.occupancy_buffer_ = vector<double>(buffer_size, mp_.clamp_min_log_ - mp_.unknown_flag_);
  md_.occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);
  md_.occupancy_buffer_inflate_cnt_ = vector<int>(buffer_size, 0);
  rebuildInflationOffsets();

  md_.count_hit_and_miss_ = vector<short>(buffer_size, 0);
  md_.count_hit_ = vector<short>(buffer_size, 0);
  md_.count_explicit_free_miss_ = vector<short>(buffer_size, 0);
  md_.occupancy_transition_hit_observation_sequence_ =
      vector<std::uint64_t>(buffer_size, 0U);
  md_.occupancy_transition_provenance_by_address_.clear();
  md_.flag_rayend_ = vector<char>(buffer_size, -1);
  md_.flag_traverse_ = vector<char>(buffer_size, -1);

  md_.raycast_num_ = 0;

  md_.proj_points_.resize(640 * 480 / mp_.skip_pixel_ / mp_.skip_pixel_);
  md_.proj_points_endpoint_hit_.resize(md_.proj_points_.size(), 1);
  md_.proj_points_cnt = 0;

  /* init callback */
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*node_);

  if (mp_.sensor_type_ == "depth")
  {
    depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>();
    depth_pose_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>();
    depth_sub_->subscribe(node_, "depth", rmw_qos_profile_sensor_data);
    depth_pose_sub_->subscribe(node_, "sensor_pose", rmw_qos_profile_sensor_data);

    sync_image_pose_.reset(new message_filters::Synchronizer<SyncPolicyImagePose>(
        SyncPolicyImagePose(100), *depth_sub_, *depth_pose_sub_));
    sync_image_pose_->registerCallback(
        std::bind(&GridMap::depthPoseCallback, this, std::placeholders::_1, std::placeholders::_2));
  }
  else if (mp_.sensor_type_ == "lidar")
  {
    // 点云和位姿跨 topic 不保证回调顺序。同步器必须同时缓存两侧消息，
    // 等到时间相近的一对到齐后再融合，不能在点云回调里读取“最新位姿”。
    rmw_qos_profile_t sync_qos = rmw_qos_profile_sensor_data;
    sync_qos.depth =
        static_cast<std::size_t>(mp_.sensor_sync_queue_size_);
    lidar_pose_sub_ =
        std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>();
    cloud_sub_ =
        std::make_shared<
            message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>();
    lidar_pose_sub_->subscribe(node_, "sensor_pose", sync_qos);
    cloud_sub_->subscribe(node_, "cloud", sync_qos);

    SyncPolicyCloudPose sync_policy(mp_.sensor_sync_queue_size_);
    sync_policy.setMaxIntervalDuration(
        rclcpp::Duration::from_seconds(mp_.max_cloud_pose_skew_sec_));
    sync_cloud_pose_ =
        std::make_shared<
            message_filters::Synchronizer<SyncPolicyCloudPose>>(
            SyncPolicyCloudPose(sync_policy), *cloud_sub_, *lidar_pose_sub_);
    sync_cloud_pose_->registerCallback(
        std::bind(
            &GridMap::cloudPoseCallback, this,
            std::placeholders::_1, std::placeholders::_2));
  }

  sliding_map_frame_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
      "body_pose", rclcpp::SensorDataQoS(),
      std::bind(&GridMap::slidingMapFrameCallback, this, std::placeholders::_1));

  occ_timer_ = node_->create_wall_timer(std::chrono::milliseconds(50),
                                        std::bind(&GridMap::updateOccupancyCallback, this));
  vis_timer_ = node_->create_wall_timer(std::chrono::milliseconds(50),
                                        std::bind(&GridMap::visCallback, this));

  map_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("grid_map/occupancy", rclcpp::SensorDataQoS());
  map_inf_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("grid_map/occupancy_inflate", rclcpp::SensorDataQoS());
  sliding_map_bbox_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("grid_map/sliding_map_bbox", 10);

  unknown_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("grid_map/unknown", rclcpp::SensorDataQoS());
  depth_cloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("grid_map/depth_cloud", rclcpp::SensorDataQoS());
  extrinsic_pose_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("grid_map/sensor_pose_extrinsic", 10);
  observation_diagnostics_pub_ =
      node_->create_publisher<
          scan_planner_msgs::msg::GridMapObservationDiagnostics>(
          mp_.observation_diagnostics_topic_,
          rclcpp::QoS(rclcpp::KeepLast(
              static_cast<std::size_t>(mp_.diagnostic_history_depth_)))
              .reliable()
              .transient_local());

  md_.occ_need_update_ = false;
  md_.use_cloud_update_ = false;
  md_.has_first_depth_ = false;
  md_.has_ray_pose_ = false;
  md_.has_cloud_ = false;
  md_.has_fused_observation_ = false;
  md_.sliding_center_initialized_ = false;
  md_.last_cloud_was_canonical_empty_ = false;
  md_.last_ray_pose_stamp_ns_ = 0;
  md_.last_cloud_stamp_ns_ = 0;
  md_.pending_cloud_stamp_ns_ = 0;
  md_.last_fused_stamp_ns_ = 0;
  md_.image_cnt_ = 0;
  md_.ray_pos_.setZero();
  md_.sliding_map_frame_pos_.setZero();
  md_.ray_q_ = Eigen::Quaterniond::Identity();

  md_.fuse_time_ = 0.0;
  md_.update_num_ = 0;
  md_.max_fuse_time_ = 0.0;
  md_.local_bound_min_ = mp_.map_bound_min_idx_;
  md_.local_bound_max_ = mp_.map_bound_max_idx_;
  pending_observation_diagnostics_valid_ = false;
  observation_diagnostics_sequence_ = 0;
  {
    std::lock_guard<std::mutex> lock(fused_observation_history_mutex_);
    fused_observation_history_.clear();
    fused_observation_sequence_.store(0U, std::memory_order_release);
  }

  // rand_noise_ = uniform_real_distribution<double>(-0.2, 0.2);
  // rand_noise2_ = normal_distribution<double>(0, 0.2);
  // random_device rd;
  // eng_ = default_random_engine(rd());
}

void GridMap::updateMapBoundaryFromIndex()
{
  mp_.map_bound_min_idx_ = mp_.map_origin_idx_ - mp_.map_voxel_num_ / 2;
  mp_.map_bound_max_idx_ = mp_.map_bound_min_idx_ + mp_.map_voxel_num_ - Eigen::Vector3i::Ones();

  mp_.map_min_boundary_ = mp_.map_bound_min_idx_.cast<double>() * mp_.resolution_;
  mp_.map_max_boundary_ = (mp_.map_bound_max_idx_.cast<double>() + Eigen::Vector3d::Ones()) * mp_.resolution_;
  mp_.map_origin_ = mp_.map_min_boundary_;
}

void GridMap::rebuildInflationOffsets()
{
  const double double_radius = std::max(0.0, mp_.double_cylinder_radius_);
  const int inf_step_xy = ceil(double_radius / mp_.resolution_);
  const int inf_step_z_up = ceil(mp_.obstacles_inflation_z_up / mp_.resolution_);
  const int inf_step_z_down = ceil(mp_.obstacles_inflation_z_down / mp_.resolution_);

  md_.inflate_offsets_.clear();
  for (int x = -inf_step_xy; x <= inf_step_xy; ++x)
    for (int y = -inf_step_xy; y <= inf_step_xy; ++y)
    {
      Eigen::Vector2d offset_xy(x * mp_.resolution_, y * mp_.resolution_);
      if (offset_xy.norm() >= double_radius)
        continue;

      for (int z = -inf_step_z_down; z <= inf_step_z_up; ++z)
        md_.inflate_offsets_.push_back(Eigen::Vector3i(x, y, z));
    }
}

void GridMap::resetAllMapData()
{
  std::fill(md_.occupancy_buffer_.begin(), md_.occupancy_buffer_.end(), mp_.clamp_min_log_ - mp_.unknown_flag_);
  std::fill(md_.occupancy_buffer_inflate_.begin(), md_.occupancy_buffer_inflate_.end(), 0);
  std::fill(md_.occupancy_buffer_inflate_cnt_.begin(), md_.occupancy_buffer_inflate_cnt_.end(), 0);
  std::fill(md_.count_hit_and_miss_.begin(), md_.count_hit_and_miss_.end(), 0);
  std::fill(md_.count_hit_.begin(), md_.count_hit_.end(), 0);
  std::fill(
      md_.count_explicit_free_miss_.begin(),
      md_.count_explicit_free_miss_.end(), 0);
  std::fill(
      md_.occupancy_transition_hit_observation_sequence_.begin(),
      md_.occupancy_transition_hit_observation_sequence_.end(), 0U);
  md_.occupancy_transition_provenance_by_address_.clear();
  std::fill(md_.flag_rayend_.begin(), md_.flag_rayend_.end(), -1);
  std::fill(md_.flag_traverse_.begin(), md_.flag_traverse_.end(), -1);
  std::queue<Eigen::Vector3i> empty;
  std::swap(md_.cache_voxel_, empty);
}

void GridMap::hashIdToGlobalIndex(int addr, Eigen::Vector3i& id_g) const
{
  Eigen::Vector3i id_l;
  id_l(0) = addr / (mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2));
  id_l(1) = (addr - id_l(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2)) / mp_.map_voxel_num_(2);
  id_l(2) = addr - id_l(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) -
            id_l(1) * mp_.map_voxel_num_(2);

  for (int i = 0; i < 3; ++i)
  {
    const int min_l = getLocalIndex(mp_.map_bound_min_idx_(i), i);
    int dist = id_l(i) - min_l;
    if (dist < 0)
      dist += mp_.map_voxel_num_(i);
    id_g(i) = mp_.map_bound_min_idx_(i) + dist;
  }
}

void GridMap::updateInflationLayer(const Eigen::Vector3i& id, int delta,
                                   const vector<Eigen::Vector3i>& offsets,
                                   std::vector<int>& cnt_buffer,
                                   std::vector<char>& flag_buffer,
                                   const std::vector<char>* ignore_mask)
{
  for (const auto& offset : offsets)
  {
    const Eigen::Vector3i inf_id = id + offset;
    if (!isInMap(inf_id))
      continue;

    const int addr = toAddress(inf_id);
    if (ignore_mask && (*ignore_mask)[addr])
      continue;

    cnt_buffer[addr] += delta;
    if (cnt_buffer[addr] < 0)
      cnt_buffer[addr] = 0;
    flag_buffer[addr] = cnt_buffer[addr] > 0 ? 1 : 0;
  }
}

void GridMap::updateInflation(const Eigen::Vector3i& id, int delta, const std::vector<char>* ignore_mask)
{
  updateInflationLayer(id, delta, md_.inflate_offsets_, md_.occupancy_buffer_inflate_cnt_,
                       md_.occupancy_buffer_inflate_, ignore_mask);
}

void GridMap::applyOccupancyUpdate(const Eigen::Vector3i& id, double new_log_odds)
{
  if (!isInMap(id))
    return;

  // setOccupied()/setOccupancy() 也会经过这里。只要调用方已经把数据写入
  // 当前循环栅格，首次滑窗就不得再走“空地图直接重定位”的快速路径。
  md_.sliding_center_initialized_ = true;
  const int addr = toAddress(id);
  const bool was_occ = md_.occupancy_buffer_[addr] > mp_.min_occupancy_log_;
  const bool now_occ = new_log_odds > mp_.min_occupancy_log_;

  md_.occupancy_buffer_[addr] = new_log_odds;
  if (was_occ != now_occ)
    updateInflation(id, now_occ ? 1 : -1);
}

void GridMap::resetCellByAddress(int addr)
{
  Eigen::Vector3i id_g;
  hashIdToGlobalIndex(addr, id_g);
  if (md_.occupancy_buffer_[addr] > mp_.min_occupancy_log_)
    updateInflation(id_g, -1);

  md_.occupancy_buffer_[addr] = mp_.clamp_min_log_ - mp_.unknown_flag_;
  md_.count_hit_[addr] = 0;
  md_.count_hit_and_miss_[addr] = 0;
  md_.count_explicit_free_miss_[addr] = 0;
  md_.occupancy_transition_hit_observation_sequence_[addr] = 0U;
  md_.occupancy_transition_provenance_by_address_.erase(addr);
  md_.flag_rayend_[addr] = -1;
  md_.flag_traverse_[addr] = -1;
}

void GridMap::resetCellByAddressForSliding(int addr, const std::vector<char>& clear_mask)
{
  Eigen::Vector3i id_g;
  hashIdToGlobalIndex(addr, id_g);
  if (md_.occupancy_buffer_[addr] > mp_.min_occupancy_log_)
    updateInflation(id_g, -1, &clear_mask);
}

std::uint32_t GridMap::updateSlidingMap(const Eigen::Vector3d& center)
{
  if (!mp_.map_sliding_en_)
    return 0U;

  Eigen::Vector3i new_origin_idx;
  posToIndex(center, new_origin_idx);
  const Eigen::Vector3i shift_num = new_origin_idx - mp_.map_origin_idx_;
  if (!md_.sliding_center_initialized_)
  {
    // 初始化缓冲区全部是相同的 unknown/free 初值，还没有任何全局坐标
    // 语义。逐列构造 clear_mask 并清除从世界原点到机器人起点之间的百万级
    // 空体素既不改变地图，又会阻塞首条 Path。首帧只更新循环窗口坐标即可。
    md_.sliding_center_initialized_ = true;
    if (shift_num.cwiseAbs().maxCoeff() >= mp_.map_sliding_thresh_vox_)
    {
      mp_.map_origin_idx_ = new_origin_idx;
      updateMapBoundaryFromIndex();
      md_.local_bound_min_ = mp_.map_bound_min_idx_;
      md_.local_bound_max_ = mp_.map_bound_max_idx_;
    }
    return 0U;
  }
  if (shift_num.cwiseAbs().maxCoeff() < mp_.map_sliding_thresh_vox_)
    return 0U;

  std::uint32_t occupied_removed = 0U;

  if ((shift_num.cwiseAbs().array() >= mp_.map_voxel_num_.array()).any())
  {
    for (const double occupancy : md_.occupancy_buffer_)
      if (occupancy > mp_.min_occupancy_log_)
        increment_saturated(occupied_removed);
    resetAllMapData();
    mp_.map_origin_idx_ = new_origin_idx;
    updateMapBoundaryFromIndex();
    md_.local_bound_min_ = mp_.map_bound_min_idx_;
    md_.local_bound_max_ = mp_.map_bound_max_idx_;
    return occupied_removed;
  }

  const int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);
  std::vector<char> clear_mask(buffer_size, 0);
  std::vector<int> clear_addrs;
  clear_addrs.reserve(buffer_size / 8);

  auto add_clear_addr = [&](const Eigen::Vector3i& id_l) {
    const int addr = toAddressLocal(id_l);
    if (!clear_mask[addr])
    {
      clear_mask[addr] = 1;
      clear_addrs.push_back(addr);
    }
  };

  for (int dim = 0; dim < 3; ++dim)
  {
    const int shift = shift_num(dim);
    if (shift == 0)
      continue;

    if (shift > 0)
    {
      for (int k = 0; k < shift; ++k)
      {
        const int clear_g = mp_.map_bound_min_idx_(dim) + k;
        const int clear_l = getLocalIndex(clear_g, dim);

        for (int a = 0; a < mp_.map_voxel_num_((dim + 1) % 3); ++a)
          for (int b = 0; b < mp_.map_voxel_num_((dim + 2) % 3); ++b)
          {
            Eigen::Vector3i id_l;
            id_l(dim) = clear_l;
            id_l((dim + 1) % 3) = a;
            id_l((dim + 2) % 3) = b;
            add_clear_addr(id_l);
          }
      }
    }
    else
    {
      for (int k = 0; k < -shift; ++k)
      {
        const int clear_g = mp_.map_bound_max_idx_(dim) - k;
        const int clear_l = getLocalIndex(clear_g, dim);

        for (int a = 0; a < mp_.map_voxel_num_((dim + 1) % 3); ++a)
          for (int b = 0; b < mp_.map_voxel_num_((dim + 2) % 3); ++b)
          {
            Eigen::Vector3i id_l;
            id_l(dim) = clear_l;
            id_l((dim + 1) % 3) = a;
            id_l((dim + 2) % 3) = b;
            add_clear_addr(id_l);
          }
      }
    }
  }

  for (int addr : clear_addrs)
  {
    if (md_.occupancy_buffer_[addr] > mp_.min_occupancy_log_)
      increment_saturated(occupied_removed);
    resetCellByAddressForSliding(addr, clear_mask);
  }

  for (int addr : clear_addrs)
  {
    md_.occupancy_buffer_[addr] = mp_.clamp_min_log_ - mp_.unknown_flag_;
    md_.occupancy_buffer_inflate_cnt_[addr] = 0;
    md_.occupancy_buffer_inflate_[addr] = 0;
    md_.count_hit_[addr] = 0;
    md_.count_hit_and_miss_[addr] = 0;
    md_.count_explicit_free_miss_[addr] = 0;
    md_.occupancy_transition_hit_observation_sequence_[addr] = 0U;
    md_.occupancy_transition_provenance_by_address_.erase(addr);
    md_.flag_rayend_[addr] = -1;
    md_.flag_traverse_[addr] = -1;
  }

  mp_.map_origin_idx_ = new_origin_idx;
  updateMapBoundaryFromIndex();
  boundIndex(md_.local_bound_min_);
  boundIndex(md_.local_bound_max_);
  return occupied_removed;
}

void GridMap::resetBuffer()
{
  resetAllMapData();
  md_.local_bound_min_ = mp_.map_bound_min_idx_;
  md_.local_bound_max_ = mp_.map_bound_max_idx_;
}

void GridMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos)
{
  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);

  boundIndex(min_id);
  boundIndex(max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
    {
      for (int z = min_id(2); z <= max_id(2); ++z)
      {
        resetCellByAddress(toAddress(x, y, z));
      }
    }
}

int GridMap::setCacheOccupancy(
    Eigen::Vector3d pos, int occ, const bool explicit_free_ray)
{
  if (occ != 1 && occ != 0)
    return INVALID_IDX;

  Eigen::Vector3i id;
  posToIndex(pos, id);
  if (!isInMap(id))
    return INVALID_IDX;

  int idx_ctns = toAddress(id);

  md_.count_hit_and_miss_[idx_ctns] += 1;

  if (md_.count_hit_and_miss_[idx_ctns] == 1)
  {
    md_.cache_voxel_.push(id);
  }

  if (occ == 1)
    md_.count_hit_[idx_ctns] += 1;
  else if (explicit_free_ray &&
           md_.count_explicit_free_miss_[idx_ctns] <
               std::numeric_limits<short>::max())
    md_.count_explicit_free_miss_[idx_ctns] += 1;

  return idx_ctns;
}

void GridMap::projectDepthImage()
{
  // md_.proj_points_.clear();
  md_.proj_points_cnt = 0;

  uint16_t *row_ptr;
  // int cols = current_img_.cols, rows = current_img_.rows;
  int cols = md_.depth_image_.cols;
  int rows = md_.depth_image_.rows;

  double depth;

  Eigen::Matrix3d sensor_r = md_.ray_q_.toRotationMatrix();

  // cout << "rotate: " << md_.ray_q_.toRotationMatrix() << endl;
  // std::cout << "pos in proj: " << md_.ray_pos_ << std::endl;

  if (!md_.has_first_depth_)
  {
    md_.has_first_depth_ = true;
    return;
  }

  Eigen::Vector3d pt_cur, pt_world;
  const double inv_factor = 1.0 / mp_.k_depth_scaling_factor_;

  for (int v = mp_.depth_filter_margin_; v < rows - mp_.depth_filter_margin_; v += mp_.skip_pixel_)
  {
    row_ptr = md_.depth_image_.ptr<uint16_t>(v) + mp_.depth_filter_margin_;

    for (int u = mp_.depth_filter_margin_; u < cols - mp_.depth_filter_margin_; u += mp_.skip_pixel_)
    {
      const uint16_t raw_depth = *row_ptr;
      depth = raw_depth * inv_factor;
      row_ptr = row_ptr + mp_.skip_pixel_;

      // filter depth
      // depth += rand_noise_(eng_);
      // if (depth > 0.01) depth += rand_noise2_(eng_);

      if (raw_depth == 0)
      {
        depth = mp_.max_ray_length_ + 0.1;
      }
      else if (depth < mp_.depth_filter_mindist_)
      {
        continue;
      }
      else if (depth > mp_.depth_filter_maxdist_)
      {
        depth = mp_.max_ray_length_ + 0.1;
      }

      // project to world frame
      pt_cur(0) = (u - mp_.cx_) * depth / mp_.fx_;
      pt_cur(1) = (v - mp_.cy_) * depth / mp_.fy_;
      pt_cur(2) = depth;

      pt_world = sensor_r * pt_cur + md_.ray_pos_;
      // if (!isInMap(pt_world)) {
      //   pt_world = closetPointInMap(pt_world, md_.ray_pos_);
      // }

      if (md_.proj_points_cnt >= static_cast<int>(md_.proj_points_.size()))
      {
        md_.proj_points_.push_back(pt_world);
        md_.proj_points_endpoint_hit_.push_back(1);
      }
      else
      {
        md_.proj_points_[md_.proj_points_cnt] = pt_world;
        md_.proj_points_endpoint_hit_[md_.proj_points_cnt] = 1;
      }
      ++md_.proj_points_cnt;
    }
  }
}

void GridMap::raycastProcess()
{
  // if (md_.proj_points_.size() == 0)
  if (md_.proj_points_cnt == 0)
    return;

  const std::uint32_t sliding_removed = updateSlidingMap(md_.ray_pos_);
  if (pending_observation_diagnostics_valid_)
  {
    const std::uint64_t total =
        static_cast<std::uint64_t>(
            pending_observation_diagnostics_
                .occupied_removed_by_sliding_reset_count) +
        sliding_removed;
    pending_observation_diagnostics_
        .occupied_removed_by_sliding_reset_count =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            total, std::numeric_limits<std::uint32_t>::max()));
  }

  md_.raycast_num_ += 1;

  int vox_idx;
  double length;

  // bounding box of updated region
  double min_x = mp_.map_max_boundary_(0);
  double min_y = mp_.map_max_boundary_(1);
  double min_z = mp_.map_max_boundary_(2);

  double max_x = mp_.map_min_boundary_(0);
  double max_y = mp_.map_min_boundary_(1);
  double max_z = mp_.map_min_boundary_(2);

  RayCaster raycaster;
  Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
  Eigen::Vector3d ray_pt, pt_w;
  std::unordered_map<int, Eigen::Vector3d> hit_endpoint_by_address;
  std::vector<OccupiedTransitionDiagnosticCandidate>
      occupied_transition_diagnostic_candidates;
  std::vector<ExplicitMissClearDiagnosticCandidate>
      explicit_miss_clear_diagnostic_candidates;

  for (int i = 0; i < md_.proj_points_cnt; ++i)
  {
    pt_w = md_.proj_points_[i];
    const bool endpoint_hit = md_.proj_points_endpoint_hit_[i] != 0;
    const bool explicit_free_ray = !endpoint_hit;
    bool endpoint_hit_applied = false;

    // set flag for projected point

    if (!isInMap(pt_w))
    {
      pt_w = closetPointInMap(pt_w, md_.ray_pos_);

      length = (pt_w - md_.ray_pos_).norm();
      if (length > mp_.max_ray_length_)
      {
        pt_w = (pt_w - md_.ray_pos_) / length * mp_.max_ray_length_ + md_.ray_pos_;
      }
      vox_idx = setCacheOccupancy(pt_w, 0, explicit_free_ray);
    }
    else
    {
      length = (pt_w - md_.ray_pos_).norm();

      if (length > mp_.max_ray_length_)
      {
        pt_w = (pt_w - md_.ray_pos_) / length * mp_.max_ray_length_ + md_.ray_pos_;
        vox_idx = setCacheOccupancy(pt_w, 0, explicit_free_ray);
      }
      else
      {
        vox_idx = setCacheOccupancy(
            pt_w, endpoint_hit ? 1 : 0, explicit_free_ray);
        endpoint_hit_applied = endpoint_hit;
      }
    }
    if (endpoint_hit_applied && vox_idx != INVALID_IDX)
      hit_endpoint_by_address[vox_idx] = pt_w;

    max_x = max(max_x, pt_w(0));
    max_y = max(max_y, pt_w(1));
    max_z = max(max_z, pt_w(2));

    min_x = min(min_x, pt_w(0));
    min_y = min(min_y, pt_w(1));
    min_z = min(min_z, pt_w(2));

    // raycasting between ray origin and point

    if (vox_idx != INVALID_IDX)
    {
      if (md_.flag_rayend_[vox_idx] == md_.raycast_num_)
      {
        continue;
      }
      else
      {
        md_.flag_rayend_[vox_idx] = md_.raycast_num_;
      }
    }

    raycaster.setInput(pt_w / mp_.resolution_, md_.ray_pos_ / mp_.resolution_);

    while (raycaster.step(ray_pt))
    {
      Eigen::Vector3d tmp = (ray_pt + half) * mp_.resolution_;
      length = (tmp - md_.ray_pos_).norm();

      vox_idx = setCacheOccupancy(tmp, 0, explicit_free_ray);

      if (vox_idx != INVALID_IDX)
      {
        if (md_.flag_traverse_[vox_idx] == md_.raycast_num_)
        {
          break;
        }
        else
        {
          md_.flag_traverse_[vox_idx] = md_.raycast_num_;
        }
      }
    }
  }

  min_x = min(min_x, md_.ray_pos_(0));
  min_y = min(min_y, md_.ray_pos_(1));
  min_z = min(min_z, md_.ray_pos_(2));

  max_x = max(max_x, md_.ray_pos_(0));
  max_y = max(max_y, md_.ray_pos_(1));
  max_z = max(max_z, md_.ray_pos_(2));
  max_z = max(max_z, mp_.ground_height_);

  posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_.local_bound_max_);
  posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_.local_bound_min_);
  boundIndex(md_.local_bound_min_);
  boundIndex(md_.local_bound_max_);

  // update occupancy cached in queue
  Eigen::Vector3d local_range_min = md_.ray_pos_ - mp_.local_update_range_;
  Eigen::Vector3d local_range_max = md_.ray_pos_ + mp_.local_update_range_;

  Eigen::Vector3i min_id, max_id;
  posToIndex(local_range_min, min_id);
  posToIndex(local_range_max, max_id);
  boundIndex(min_id);
  boundIndex(max_id);

  // std::cout << "cache all: " << md_.cache_voxel_.size() << std::endl;

  while (!md_.cache_voxel_.empty())
  {

    Eigen::Vector3i idx = md_.cache_voxel_.front();
    int idx_ctns = toAddress(idx);
    md_.cache_voxel_.pop();

    const int hit_count = md_.count_hit_[idx_ctns];
    const int total_count = md_.count_hit_and_miss_[idx_ctns];
    const int explicit_miss_count =
        md_.count_explicit_free_miss_[idx_ctns];
    const int total_miss_count = total_count - hit_count;
    const int ordinary_miss_count =
        std::max(0, total_miss_count - explicit_miss_count);
    const double log_odds_update =
        hit_count >= total_miss_count
            ? mp_.prob_hit_log_
            : mp_.prob_miss_log_;

    // 只有移除本观测的 explicit miss 后多数投票会从 miss 变为 hit，
    // explicit ray 才是本次 p_miss 的反事实必要原因。普通 miss 已足够时
    // 禁止搭便车记成 explicit clear。
    const bool explicit_miss_is_causal =
        explicit_miss_count > 0 &&
        hit_count < total_miss_count &&
        hit_count >= ordinary_miss_count;
    md_.count_hit_[idx_ctns] = md_.count_hit_and_miss_[idx_ctns] = 0;
    md_.count_explicit_free_miss_[idx_ctns] = 0;

    const bool in_local =
        idx(0) >= min_id(0) && idx(0) <= max_id(0) &&
        idx(1) >= min_id(1) && idx(1) <= max_id(1) &&
        idx(2) >= min_id(2) && idx(2) <= max_id(2);
    const bool explicit_miss_update =
        in_local && log_odds_update < 0.0 && explicit_miss_is_causal;
    if (explicit_miss_update && pending_observation_diagnostics_valid_)
      increment_saturated(
          pending_observation_diagnostics_.explicit_free_miss_voxel_count);

    if (log_odds_update >= 0 && md_.occupancy_buffer_[idx_ctns] >= mp_.clamp_max_log_)
    {
      continue;
    }
    else if (log_odds_update <= 0 && md_.occupancy_buffer_[idx_ctns] <= mp_.clamp_min_log_)
    {
      applyOccupancyUpdate(idx, mp_.clamp_min_log_);
      md_.occupancy_transition_hit_observation_sequence_[idx_ctns] = 0U;
      continue;
    }

    const bool was_occupied =
        md_.occupancy_buffer_[idx_ctns] > mp_.min_occupancy_log_;
    if (!in_local)
    {
      applyOccupancyUpdate(idx, mp_.clamp_min_log_);
    }

    const double new_log_odds =
        std::min(std::max(md_.occupancy_buffer_[idx_ctns] + log_odds_update, mp_.clamp_min_log_),
                 mp_.clamp_max_log_);
    applyOccupancyUpdate(idx, new_log_odds);
    const bool now_occupied = new_log_odds > mp_.min_occupancy_log_;
    if (log_odds_update > 0.0 && !was_occupied && now_occupied)
    {
      if (!pending_observation_diagnostics_valid_ ||
          observation_diagnostics_sequence_ ==
              std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error(
            "GridMap 占据阈值穿越缺少可发布 observation sequence");
      md_.occupancy_transition_hit_observation_sequence_[idx_ctns] =
          observation_diagnostics_sequence_ + 1U;
      const auto endpoint = hit_endpoint_by_address.find(idx_ctns);
      if (endpoint == hit_endpoint_by_address.end())
        throw std::runtime_error(
            "GridMap free→occupied 阈值穿越缺少本观测 hit endpoint");
      OccupancyTransitionProvenance provenance;
      provenance.global_voxel_index = idx;
      provenance.hit_endpoint_world = endpoint->second;
      provenance.observation_sequence =
          observation_diagnostics_sequence_ + 1U;
      provenance.header_stamp_ns = message_stamp_ns(
          pending_observation_diagnostics_.header.stamp);
      md_.occupancy_transition_provenance_by_address_[idx_ctns] =
          provenance;
      increment_saturated(
          pending_observation_diagnostics_
              .free_to_occupied_transition_count);
      occupied_transition_diagnostic_candidates.push_back(
          OccupiedTransitionDiagnosticCandidate{
              geometry_point(endpoint->second),
              {static_cast<std::int64_t>(idx(0)),
               static_cast<std::int64_t>(idx(1)),
               static_cast<std::int64_t>(idx(2))}});
    }

    // 只有明确 free ray 的 p_miss 更新在局部有效范围内使 raw occupancy
    // 穿过阈值，才记为 ghost clear。滑窗 reset 与普通 hit ray 的内部
    // miss 都不经过此分支，因而无法冒充该证据。
    if (explicit_miss_update && was_occupied && !now_occupied &&
        pending_observation_diagnostics_valid_)
    {
      increment_saturated(
          pending_observation_diagnostics_
              .occupied_to_free_by_explicit_miss_count);
      Eigen::Vector3d position;
      indexToPos(idx, position);
      const auto provenance =
          md_.occupancy_transition_provenance_by_address_.find(idx_ctns);
      ExplicitMissClearDiagnosticCandidate candidate{
          geometry_point(position),
          {static_cast<std::int64_t>(idx(0)),
           static_cast<std::int64_t>(idx(1)),
           static_cast<std::int64_t>(idx(2))},
          md_.occupancy_transition_hit_observation_sequence_[idx_ctns],
          geometry_point(position),
          0};
      if (
          provenance !=
              md_.occupancy_transition_provenance_by_address_.end() &&
          (provenance->second.global_voxel_index.array() == idx.array())
              .all())
      {
        candidate.hit_endpoint =
            geometry_point(provenance->second.hit_endpoint_world);
        candidate.hit_header_stamp_ns = provenance->second.header_stamp_ns;
      }
      // setOccupied() 等非传感器预置没有可认证 hit 来源；仍保持成组字段
      // 对齐，但 sequence/stamp=0 会让 runtime/validator 失败关闭。
      explicit_miss_clear_diagnostic_candidates.push_back(candidate);
    }
    if (was_occupied && !now_occupied)
    {
      md_.occupancy_transition_hit_observation_sequence_[idx_ctns] = 0U;
      md_.occupancy_transition_provenance_by_address_.erase(idx_ctns);
    }
  }

  // cache_voxel_ 的 FIFO 顺序受点云扫描顺序影响。只保留前 64 个会稳定
  // 漏掉画面后部的动态体素；先收集全部合格 transition，再固定包含首尾地
  // 等距抽样。每个 candidate 内封装完整 provenance，避免并行数组错位。
  const auto occupied_transition_sample_indices =
      deterministic_uniform_sample_indices(
          occupied_transition_diagnostic_candidates.size(),
          static_cast<std::size_t>(mp_.diagnostic_max_samples_));
  for (const std::size_t sample_index : occupied_transition_sample_indices)
  {
    const auto &candidate =
        occupied_transition_diagnostic_candidates[sample_index];
    pending_observation_diagnostics_
        .free_to_occupied_transition_hit_samples.push_back(
            candidate.hit_endpoint);
    auto &indices = pending_observation_diagnostics_
                        .free_to_occupied_transition_voxel_indices_xyz;
    indices.insert(
        indices.end(),
        candidate.voxel_index_xyz.begin(),
        candidate.voxel_index_xyz.end());
  }

  const auto explicit_clear_sample_indices =
      deterministic_uniform_sample_indices(
          explicit_miss_clear_diagnostic_candidates.size(),
          static_cast<std::size_t>(mp_.diagnostic_max_samples_));
  for (const std::size_t sample_index : explicit_clear_sample_indices)
  {
    const auto &candidate =
        explicit_miss_clear_diagnostic_candidates[sample_index];
    pending_observation_diagnostics_
        .occupied_to_free_by_explicit_miss_samples.push_back(
            candidate.clear_point);
    auto &indices = pending_observation_diagnostics_
                        .occupied_to_free_sample_voxel_indices_xyz;
    indices.insert(
        indices.end(),
        candidate.voxel_index_xyz.begin(),
        candidate.voxel_index_xyz.end());
    pending_observation_diagnostics_
        .occupied_to_free_transition_hit_observation_sequences.push_back(
            candidate.hit_observation_sequence);
    pending_observation_diagnostics_
        .occupied_to_free_transition_hit_samples.push_back(
            candidate.hit_endpoint);
    pending_observation_diagnostics_
        .occupied_to_free_transition_hit_header_stamp_ns.push_back(
            candidate.hit_header_stamp_ns);
  }
}

Eigen::Vector3d GridMap::closetPointInMap(const Eigen::Vector3d &pt, const Eigen::Vector3d &ray_pos)
{
  Eigen::Vector3d diff = pt - ray_pos;
  Eigen::Vector3d max_tc = mp_.map_max_boundary_ - ray_pos;
  Eigen::Vector3d min_tc = mp_.map_min_boundary_ - ray_pos;

  double min_t = 1000000;

  for (int i = 0; i < 3; ++i)
  {
    if (fabs(diff[i]) > 0)
    {

      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t)
        min_t = t1;

      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t)
        min_t = t2;
    }
  }

  return ray_pos + (min_t - 1e-3) * diff;
}

void GridMap::visCallback()
{

  publishMap();
  publishMapInflate(true);
  publishSlidingMapFrame();
  publishSlidingMapBBox();
  publishDepthCloud();
}

void GridMap::publishObservationDiagnostics()
{
  if (!pending_observation_diagnostics_valid_ ||
      !observation_diagnostics_pub_)
    return;
  if (observation_diagnostics_sequence_ ==
      std::numeric_limits<std::uint64_t>::max())
    throw std::runtime_error("GridMap observation_sequence 已耗尽");

  pending_observation_diagnostics_.observation_sequence =
      ++observation_diagnostics_sequence_;
  pending_observation_diagnostics_.hit_endpoint_samples_truncated =
      pending_observation_diagnostics_.hit_endpoint_count >
      pending_observation_diagnostics_.hit_endpoint_samples.size();
  pending_observation_diagnostics_
      .free_to_occupied_transition_samples_truncated =
      pending_observation_diagnostics_
          .free_to_occupied_transition_count >
      pending_observation_diagnostics_
          .free_to_occupied_transition_hit_samples.size();
  pending_observation_diagnostics_.occupied_to_free_samples_truncated =
      pending_observation_diagnostics_
          .occupied_to_free_by_explicit_miss_count >
      pending_observation_diagnostics_
          .occupied_to_free_by_explicit_miss_samples.size();
  if (pending_observation_diagnostics_
          .hit_endpoint_sample_voxel_indices_xyz.size() !=
      3U * pending_observation_diagnostics_.hit_endpoint_samples.size())
    throw std::runtime_error(
        "GridMap hit 样本与 canonical voxel index 数量不一致");
  if (pending_observation_diagnostics_
          .free_to_occupied_transition_voxel_indices_xyz.size() !=
      3U * pending_observation_diagnostics_
               .free_to_occupied_transition_hit_samples.size())
    throw std::runtime_error(
        "GridMap transition hit 样本与 canonical voxel index 数量不一致");
  if (pending_observation_diagnostics_
          .occupied_to_free_sample_voxel_indices_xyz.size() !=
          3U * pending_observation_diagnostics_
                   .occupied_to_free_by_explicit_miss_samples.size() ||
      pending_observation_diagnostics_
          .occupied_to_free_transition_hit_observation_sequences.size() !=
          pending_observation_diagnostics_
              .occupied_to_free_by_explicit_miss_samples.size() ||
      pending_observation_diagnostics_
          .occupied_to_free_transition_hit_samples.size() !=
          pending_observation_diagnostics_
              .occupied_to_free_by_explicit_miss_samples.size() ||
      pending_observation_diagnostics_
          .occupied_to_free_transition_hit_header_stamp_ns.size() !=
          pending_observation_diagnostics_
              .occupied_to_free_by_explicit_miss_samples.size())
    throw std::runtime_error(
        "GridMap clear 样本与阈值穿越 provenance 数量不一致");
  observation_diagnostics_pub_->publish(
      pending_observation_diagnostics_);
  pending_observation_diagnostics_valid_ = false;
}

void GridMap::updateOccupancyCallback()
{
  if (!md_.occ_need_update_)
    return;

  /* update occupancy */
  // ros::Time t1, t2, t3, t4;
  // t1 = ros::Time::now();

  if (!md_.use_cloud_update_)
    projectDepthImage();
  // t2 = ros::Time::now();
  // 默认 callback group 串行执行本回调。先在任何地图写入前检查代数空间，
  // 避免计数耗尽时仍融合一帧却无法给它分配唯一代数。
  const bool has_projected_points = md_.proj_points_cnt > 0;
  const std::int64_t acquisition_stamp_ns =
      md_.use_cloud_update_
          ? md_.pending_cloud_stamp_ns_
          : md_.last_ray_pose_stamp_ns_;
  std::unique_lock<std::mutex> history_lock(
      fused_observation_history_mutex_, std::defer_lock);
  if (has_projected_points)
  {
    history_lock.lock();
    if (acquisition_stamp_ns <= 0)
      throw std::runtime_error("GridMap 融合缺少有效采集时间戳");
    if (fused_observation_sequence_.load(std::memory_order_relaxed) ==
        std::numeric_limits<std::uint64_t>::max())
      throw std::runtime_error("GridMap fused observation sequence 已耗尽");
  }
  raycastProcess();
  if (has_projected_points)
  {
    const std::uint64_t sequence =
        fused_observation_sequence_.load(std::memory_order_relaxed) + 1U;
    fused_observation_history_.push_back(
        FusedObservationRecord{sequence, acquisition_stamp_ns});
    if (fused_observation_history_.size() >
        kFusedObservationHistoryCapacity)
      fused_observation_history_.pop_front();
    // 先写完整记录，再用 release 发布 current sequence；查询不会看到只有
    // sequence、没有对应 acquisition stamp 的半个融合事务。
    fused_observation_sequence_.store(sequence, std::memory_order_release);
    history_lock.unlock();
    md_.has_fused_observation_ = true;
    md_.last_fused_stamp_ns_ = acquisition_stamp_ns;
    if (md_.use_cloud_update_ &&
        pending_observation_diagnostics_valid_)
    {
      pending_observation_diagnostics_.map_fusion_performed = true;
      publishObservationDiagnostics();
    }
  }
  // t3 = ros::Time::now();

  // t4 = ros::Time::now();

  // cout << setprecision(7);
  // cout << "t2=" << (t2-t1).toSec() << " t3=" << (t3-t2).toSec() << " t4=" << (t4-t3).toSec() << endl;;

  // md_.fuse_time_ += (t2 - t1).toSec();
  // md_.max_fuse_time_ = max(md_.max_fuse_time_, (t2 - t1).toSec());

  // if (mp_.show_occ_time_)
  //   ROS_WARN("Fusion: cur t = %lf, avg t = %lf, max t = %lf", (t2 - t1).toSec(),
  //            md_.fuse_time_ / md_.update_num_, md_.max_fuse_time_);

  md_.occ_need_update_ = false;
  md_.use_cloud_update_ = false;
}

void GridMap::depthPoseCallback(const sensor_msgs::msg::Image::ConstSharedPtr &img,
                                const nav_msgs::msg::Odometry::ConstSharedPtr &pose)
{
  if (mp_.sensor_type_ != "depth")
    return;
  const int64_t image_stamp_ns = message_stamp_ns(img->header.stamp);
  const int64_t pose_stamp_ns = message_stamp_ns(pose->header.stamp);
  if (image_stamp_ns <= 0 || pose_stamp_ns <= 0 ||
      pose->header.frame_id != mp_.frame_id_ ||
      pose->child_frame_id != mp_.base_frame_id_ ||
      !finite_pose(pose->pose.pose))
  {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "[GridMap] 拒绝 frame、时间戳或位姿无效的深度输入");
    return;
  }
  const double skew_sec =
      std::abs(static_cast<double>(image_stamp_ns - pose_stamp_ns)) * 1.0e-9;
  if (skew_sec > mp_.max_cloud_pose_skew_sec_)
  {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "[GridMap] 深度与位姿时间差 %.3fs 超过门限", skew_sec);
    return;
  }

  /* get depth image */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, mp_.k_depth_scaling_factor_);
  }
  cv_ptr->image.copyTo(md_.depth_image_);

  // std::cout << "depth: " << md_.depth_image_.cols << ", " << md_.depth_image_.rows << std::endl;

  /* get pose */
  const geometry_msgs::msg::Pose &sensor_pose = pose->pose.pose;
  Eigen::Quaterniond ray_q(sensor_pose.orientation.w, sensor_pose.orientation.x,
                           sensor_pose.orientation.y, sensor_pose.orientation.z);
  if (ray_q.norm() < 1e-6)
    return;
  ray_q.normalize();

  Eigen::Vector3d ray_pos(sensor_pose.position.x, sensor_pose.position.y, sensor_pose.position.z);
  if (mp_.need_extrinsic_)
  {
    const Eigen::Matrix3d pose_r = ray_q.toRotationMatrix();
    ray_pos += pose_r * mp_.depth_extrinsic_.block<3, 1>(0, 3);
    ray_q = Eigen::Quaterniond(pose_r * mp_.depth_extrinsic_.block<3, 3>(0, 0));
    ray_q.normalize();
  }

  nav_msgs::msg::Odometry extrinsic_pose = *pose;
  extrinsic_pose.pose.pose.position.x = ray_pos.x();
  extrinsic_pose.pose.pose.position.y = ray_pos.y();
  extrinsic_pose.pose.pose.position.z = ray_pos.z();
  extrinsic_pose.pose.pose.orientation.x = ray_q.x();
  extrinsic_pose.pose.pose.orientation.y = ray_q.y();
  extrinsic_pose.pose.pose.orientation.z = ray_q.z();
  extrinsic_pose.pose.pose.orientation.w = ray_q.w();
  extrinsic_pose.child_frame_id =
      pose->child_frame_id.empty() ? "sensor_extrinsic" : pose->child_frame_id + "_extrinsic";
  extrinsic_pose_pub_->publish(extrinsic_pose);

  md_.ray_pos_ = ray_pos;
  md_.ray_q_ = ray_q;
  md_.last_ray_pose_stamp_ns_ = pose_stamp_ns;
  md_.last_cloud_stamp_ns_ = image_stamp_ns;
  md_.use_cloud_update_ = false;
  updateSlidingMap(md_.ray_pos_);
  if (isInMap(md_.ray_pos_))
  {
    md_.has_ray_pose_ = true;
    md_.update_num_ += 1;
    md_.occ_need_update_ = true;
  }
  else
  {
    md_.occ_need_update_ = false;
  }
}

void GridMap::slidingMapFrameCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &pose)
{
  if (message_stamp_ns(pose->header.stamp) <= 0 ||
      pose->header.frame_id != mp_.frame_id_ ||
      pose->child_frame_id != mp_.base_frame_id_ ||
      !finite_pose(pose->pose.pose))
    return;
  const geometry_msgs::msg::Point &pos = pose->pose.pose.position;
  md_.sliding_map_frame_pos_ = Eigen::Vector3d(pos.x, pos.y, pos.z);
}

void GridMap::cloudPoseCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &img,
    const nav_msgs::msg::Odometry::ConstSharedPtr &pose_msg)
{
  if (mp_.sensor_type_ != "lidar")
    return;
  const int64_t cloud_stamp_ns = message_stamp_ns(img->header.stamp);
  const int64_t pose_stamp_ns = message_stamp_ns(pose_msg->header.stamp);
  if (cloud_stamp_ns <= 0 || img->header.frame_id.empty() ||
      (mp_.cloud_is_world_ && img->header.frame_id != mp_.frame_id_) ||
      pose_stamp_ns <= 0 ||
      pose_msg->header.frame_id != mp_.frame_id_ ||
      pose_msg->child_frame_id != mp_.base_frame_id_ ||
      !finite_pose(pose_msg->pose.pose))
  {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "[GridMap] 拒绝 frame、时间戳或位姿无效的点云/传感器位姿对");
    return;
  }

  const double pose_skew_sec =
      std::abs(
          static_cast<double>(
              cloud_stamp_ns - pose_stamp_ns)) *
      1.0e-9;
  if (pose_skew_sec > mp_.max_cloud_pose_skew_sec_)
  {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "[GridMap] 点云与 sensor_pose 时间差 %.3fs 超过门限",
        pose_skew_sec);
    return;
  }

  const geometry_msgs::msg::Pose &sensor_pose = pose_msg->pose.pose;
  Eigen::Quaterniond ray_q(
      sensor_pose.orientation.w,
      sensor_pose.orientation.x,
      sensor_pose.orientation.y,
      sensor_pose.orientation.z);
  if (ray_q.norm() < 1e-6)
    return;
  ray_q.normalize();
  Eigen::Vector3d ray_pos(
      sensor_pose.position.x,
      sensor_pose.position.y,
      sensor_pose.position.z);
  if (mp_.need_extrinsic_)
  {
    const Eigen::Matrix3d pose_r = ray_q.toRotationMatrix();
    ray_pos += pose_r * mp_.lidar_extrinsic_.block<3, 1>(0, 3);
    ray_q = Eigen::Quaterniond(
        pose_r * mp_.lidar_extrinsic_.block<3, 3>(0, 0));
    ray_q.normalize();
  }
  if (!std::isfinite(ray_pos.x()) ||
      !std::isfinite(ray_pos.y()) ||
      !std::isfinite(ray_pos.z()))
    return;

  const bool canonical_empty = canonical_empty_xyz32_cloud(*img);
  if (img->width == 0U && !canonical_empty)
  {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "[GridMap] 拒绝非规范空 PointCloud2");
    return;
  }

  // 诊断与融合结果必须逐观测对应。若 50 ms timer 尚未处理上一帧，先用
  // 上一帧的 ray pose 与端点完成融合，避免新消息覆盖待审计证据。
  if (md_.occ_need_update_)
    updateOccupancyCallback();

  md_.ray_pos_ = ray_pos;
  md_.ray_q_ = ray_q;
  md_.has_ray_pose_ = true;
  md_.last_ray_pose_stamp_ns_ = pose_stamp_ns;

  if (canonical_empty)
  {
    // bridge 已完成地面、Path 支撑面和机器人自点过滤；零点表示本帧没有
    // 障碍端点，而不是传感器失活。只刷新观测生命期，不执行 raycast，
    // 不移动地图窗口，也不清除历史占据，保持对旧障碍的安全保守记忆。
    md_.has_cloud_ = true;
    md_.last_cloud_stamp_ns_ = cloud_stamp_ns;
    md_.last_cloud_was_canonical_empty_ = true;
    pending_observation_diagnostics_ =
        scan_planner_msgs::msg::GridMapObservationDiagnostics();
    pending_observation_diagnostics_.header.stamp = img->header.stamp;
    pending_observation_diagnostics_.header.frame_id = mp_.frame_id_;
    pending_observation_diagnostics_.sensor_pose_stamp =
        pose_msg->header.stamp;
    pending_observation_diagnostics_.sensor_origin =
        geometry_point(ray_pos);
    pending_observation_diagnostics_.canonical_empty = true;
    pending_observation_diagnostics_.map_fusion_performed = false;
    pending_observation_diagnostics_.map_resolution = mp_.resolution_;
    pending_observation_diagnostics_valid_ = true;
    publishObservationDiagnostics();
    return;
  }

  std::vector<char> latest_endpoint_hits;
  std::string endpoint_error;
  if (!extract_ray_endpoint_hits(
          *img, latest_endpoint_hits, endpoint_error))
  {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "[GridMap] 拒绝非法射线端点字段：%s",
        endpoint_error.c_str());
    return;
  }

  pcl::PointCloud<pcl::PointXYZ> latest_cloud;
  pcl::fromROSMsg(*img, latest_cloud);

  if (latest_cloud.points.size() == 0 ||
      latest_cloud.points.size() != latest_endpoint_hits.size())
    return;

  const Eigen::Matrix3d sensor_r = ray_q.toRotationMatrix();

  md_.proj_points_cnt = 0;
  scan_planner_msgs::msg::GridMapObservationDiagnostics diagnostics;
  diagnostics.header.stamp = img->header.stamp;
  diagnostics.header.frame_id = mp_.frame_id_;
  diagnostics.sensor_pose_stamp = pose_msg->header.stamp;
  diagnostics.sensor_origin = geometry_point(ray_pos);
  diagnostics.canonical_empty = false;
  diagnostics.map_fusion_performed = false;
  diagnostics.map_resolution = mp_.resolution_;
  diagnostics.input_point_count =
      bounded_uint32(latest_cloud.points.size());

  for (size_t i = 0; i < latest_cloud.points.size(); ++i)
  {
    const pcl::PointXYZ &pt = latest_cloud.points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;

    Eigen::Vector3d pt_world;
    if (mp_.cloud_is_world_)
    {
      pt_world = Eigen::Vector3d(pt.x, pt.y, pt.z);
    }
    else
    {
      const Eigen::Vector3d pt_sensor(pt.x, pt.y, pt.z);
      pt_world = sensor_r * pt_sensor + ray_pos;
    }
    const Eigen::Vector3d devi = pt_world - ray_pos;
    const double ray_length = devi.norm();
    const bool in_local_range =
        fabs(devi(0)) <= mp_.local_update_range_(0) && fabs(devi(1)) <= mp_.local_update_range_(1) &&
        fabs(devi(2)) <= mp_.local_update_range_(2);
    if (!in_local_range && ray_length <= mp_.max_ray_length_)
      continue;

    if (md_.proj_points_cnt >= static_cast<int>(md_.proj_points_.size()))
    {
      md_.proj_points_.push_back(pt_world);
      md_.proj_points_endpoint_hit_.push_back(latest_endpoint_hits[i]);
    }
    else
    {
      md_.proj_points_[md_.proj_points_cnt] = pt_world;
      md_.proj_points_endpoint_hit_[md_.proj_points_cnt] =
          latest_endpoint_hits[i];
    }

    md_.proj_points_cnt++;
    increment_saturated(diagnostics.accepted_endpoint_count);
    if (latest_endpoint_hits[i] != 0)
    {
      increment_saturated(diagnostics.hit_endpoint_count);
    }
    else
    {
      increment_saturated(diagnostics.explicit_free_endpoint_count);
    }
  }

  if (md_.proj_points_cnt == 0)
    return;

  // 输入点的排列通常来自深度图扫描顺序；只取前 N 点会稳定漏掉画面后部
  // 的动态障碍。这里在全部已接纳 hit 端点上均匀取样，始终覆盖首尾，
  // 并由 total count + truncated 明确说明是否完整。
  const std::size_t hit_count = diagnostics.hit_endpoint_count;
  const std::size_t hit_sample_count = std::min<std::size_t>(
      hit_count, static_cast<std::size_t>(mp_.diagnostic_max_samples_));
  std::vector<std::size_t> hit_sample_ordinals;
  hit_sample_ordinals.reserve(hit_sample_count);
  for (std::size_t sample = 0; sample < hit_sample_count; ++sample)
  {
    const double ratio = hit_sample_count == 1U
                             ? 0.0
                             : static_cast<double>(sample) /
                                   static_cast<double>(hit_sample_count - 1U);
    hit_sample_ordinals.push_back(static_cast<std::size_t>(std::llround(
        ratio * static_cast<double>(hit_count - 1U))));
  }
  std::size_t hit_ordinal = 0U;
  std::size_t next_sample = 0U;
  for (int index = 0;
       index < md_.proj_points_cnt && next_sample < hit_sample_ordinals.size();
       ++index)
  {
    if (md_.proj_points_endpoint_hit_[index] == 0)
      continue;
    if (hit_ordinal == hit_sample_ordinals[next_sample])
    {
      diagnostics.hit_endpoint_samples.push_back(
          geometry_point(md_.proj_points_[index]));
      Eigen::Vector3i voxel_index;
      posToIndex(md_.proj_points_[index], voxel_index);
      diagnostics.hit_endpoint_sample_voxel_indices_xyz.push_back(
          static_cast<std::int64_t>(voxel_index(0)));
      diagnostics.hit_endpoint_sample_voxel_indices_xyz.push_back(
          static_cast<std::int64_t>(voxel_index(1)));
      diagnostics.hit_endpoint_sample_voxel_indices_xyz.push_back(
          static_cast<std::int64_t>(voxel_index(2)));
      ++next_sample;
    }
    ++hit_ordinal;
  }

  pending_observation_diagnostics_ = std::move(diagnostics);
  pending_observation_diagnostics_valid_ = true;
  pending_observation_diagnostics_
      .occupied_removed_by_sliding_reset_count =
      updateSlidingMap(ray_pos);

  md_.has_cloud_ = true;
  md_.last_cloud_stamp_ns_ = cloud_stamp_ns;
  md_.pending_cloud_stamp_ns_ = cloud_stamp_ns;
  md_.last_cloud_was_canonical_empty_ = false;
  md_.use_cloud_update_ = true;
  md_.occ_need_update_ = true;
}

void GridMap::publishMap()
{

  if (map_pub_->get_subscription_count() == 0)
    return;

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = mp_.map_bound_min_idx_;
  Eigen::Vector3i max_cut = mp_.map_bound_max_idx_;

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_.occupancy_buffer_[toAddress(x, y, z)] < mp_.min_occupancy_log_)
          continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (md_.has_ray_pose_ && pos(2) > md_.ray_pos_(2) + mp_.vis_height_)
          continue;
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::msg::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  cloud_msg.header.stamp = node_->now();
  map_pub_->publish(cloud_msg);
}

void GridMap::publishMapInflate(bool all_info)
{

  if (map_inf_pub_->get_subscription_count() == 0)
    return;

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = mp_.map_bound_min_idx_;
  Eigen::Vector3i max_cut = mp_.map_bound_max_idx_;

  const std::vector<char> &inflate_buffer = md_.occupancy_buffer_inflate_;
  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (inflate_buffer[toAddress(x, y, z)] == 0)
          continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (md_.has_ray_pose_ && pos(2) > md_.ray_pos_(2) + mp_.vis_height_)
          continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::msg::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  cloud_msg.header.stamp = node_->now();
  map_inf_pub_->publish(cloud_msg);

  // ROS_INFO("pub map");
}

void GridMap::publishSlidingMapFrame()
{
  if (mp_.sliding_map_frame_id_.empty())
    return;

  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = node_->now();
  transform.header.frame_id = mp_.frame_id_;
  transform.child_frame_id = mp_.sliding_map_frame_id_;
  transform.transform.translation.x = md_.sliding_map_frame_pos_.x();
  transform.transform.translation.y = md_.sliding_map_frame_pos_.y();
  transform.transform.translation.z = md_.sliding_map_frame_pos_.z();
  transform.transform.rotation.w = 1.0;
  tf_broadcaster_->sendTransform(transform);
}

void GridMap::publishSlidingMapBBox()
{
  if (sliding_map_bbox_pub_->get_subscription_count() == 0)
    return;

  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = mp_.frame_id_;
  marker.header.stamp = node_->now();
  marker.ns = "sliding_map";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.04;
  marker.color.r = 0.0;
  marker.color.g = 0.8;
  marker.color.b = 1.0;
  marker.color.a = 1.0;

  const Eigen::Vector3d& min_pt = mp_.map_min_boundary_;
  const Eigen::Vector3d& max_pt = mp_.map_max_boundary_;
  Eigen::Vector3d corners[8] = {
      {min_pt.x(), min_pt.y(), min_pt.z()},
      {max_pt.x(), min_pt.y(), min_pt.z()},
      {max_pt.x(), max_pt.y(), min_pt.z()},
      {min_pt.x(), max_pt.y(), min_pt.z()},
      {min_pt.x(), min_pt.y(), max_pt.z()},
      {max_pt.x(), min_pt.y(), max_pt.z()},
      {max_pt.x(), max_pt.y(), max_pt.z()},
      {min_pt.x(), max_pt.y(), max_pt.z()},
  };

  auto pushPoint = [&](const Eigen::Vector3d& p) {
    geometry_msgs::msg::Point point;
    point.x = p.x();
    point.y = p.y();
    point.z = p.z();
    marker.points.push_back(point);
  };

  const int edges[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},
      {4, 5}, {5, 6}, {6, 7}, {7, 4},
      {0, 4}, {1, 5}, {2, 6}, {3, 7},
  };

  for (const auto& edge : edges)
  {
    pushPoint(corners[edge[0]]);
    pushPoint(corners[edge[1]]);
  }

  sliding_map_bbox_pub_->publish(marker);
}

void GridMap::publishUnknown()
{
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;

  boundIndex(max_cut);
  boundIndex(min_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {

        if (md_.occupancy_buffer_[toAddress(x, y, z)] < mp_.clamp_min_log_ - 1e-3)
        {
          Eigen::Vector3d pos;
          indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (md_.has_ray_pose_ && pos(2) > md_.ray_pos_(2) + mp_.vis_height_)
            continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  cloud_msg.header.stamp = node_->now();
  unknown_pub_->publish(cloud_msg);
}

bool GridMap::odomValid() { return md_.has_ray_pose_; }

bool GridMap::hasDepthObservation() { return md_.has_first_depth_; }

int64_t GridMap::observationStampNs() const
{
  return md_.last_cloud_stamp_ns_ > 0 ? md_.last_cloud_stamp_ns_ : 0;
}

std::uint64_t GridMap::fusedObservationSequence() const noexcept
{
  return fused_observation_sequence_.load(std::memory_order_acquire);
}

FusedObservationEvidence GridMap::fusedObservationEvidenceAfter(
    const std::int64_t settle_stamp_ns,
    const std::uint64_t baseline_sequence,
    const std::int64_t now_ns,
    const std::uint64_t required_observations) const noexcept
{
  std::lock_guard<std::mutex> lock(fused_observation_history_mutex_);
  return evaluateFusedObservationEvidence(
      fused_observation_history_,
      fused_observation_sequence_.load(std::memory_order_relaxed),
      settle_stamp_ns, baseline_sequence, now_ns,
      required_observations);
}

bool GridMap::observationReady()
{
  if (!node_ || !md_.has_ray_pose_)
    return false;

  int64_t observation_stamp_ns = 0;
  if (mp_.sensor_type_ == "lidar")
  {
    if (!md_.has_cloud_ || md_.occ_need_update_)
      return false;
    if (md_.last_cloud_was_canonical_empty_)
    {
      observation_stamp_ns = md_.last_cloud_stamp_ns_;
    }
    else
    {
      if (!md_.has_fused_observation_ ||
          md_.last_fused_stamp_ns_ < md_.last_cloud_stamp_ns_)
        return false;
      observation_stamp_ns = md_.last_fused_stamp_ns_;
    }
  }
  else
  {
    if (!md_.has_fused_observation_)
      return false;
    observation_stamp_ns = md_.last_fused_stamp_ns_;
  }

  const int64_t now_ns = node_->now().nanoseconds();
  if (now_ns <= 0 || md_.last_ray_pose_stamp_ns_ <= 0 ||
      observation_stamp_ns <= 0)
    return false;
  const int64_t timeout_ns =
      static_cast<int64_t>(mp_.observation_timeout_sec_ * 1.0e9);
  const int64_t pose_age_ns = now_ns - md_.last_ray_pose_stamp_ns_;
  const int64_t observation_age_ns = now_ns - observation_stamp_ns;
  return pose_age_ns >= 0 && pose_age_ns <= timeout_ns &&
         observation_age_ns >= 0 && observation_age_ns <= timeout_ns;
}

Eigen::Vector3d GridMap::getOrigin() { return mp_.map_origin_; }

// int GridMap::getVoxelNum() {
//   return mp_.map_voxel_num_[0] * mp_.map_voxel_num_[1] * mp_.map_voxel_num_[2];
// }

void GridMap::getRegion(Eigen::Vector3d &ori, Eigen::Vector3d &size)
{
  ori = mp_.map_origin_, size = mp_.map_size_;
}

// GridMap

void GridMap::publishDepthCloud()
{
  if (depth_cloud_pub_->get_subscription_count() == 0)
    return;

  if (md_.proj_points_cnt == 0)
    return;

  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::PointXYZ pt;

  for (int i = 0; i < md_.proj_points_cnt; ++i)
  {
    pt.x = md_.proj_points_[i](0);
    pt.y = md_.proj_points_[i](1);
    pt.z = md_.proj_points_[i](2);
    cloud.push_back(pt);
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  cloud_msg.header.stamp = node_->now();
  depth_cloud_pub_->publish(cloud_msg);
}
