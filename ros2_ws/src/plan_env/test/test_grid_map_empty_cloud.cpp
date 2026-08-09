#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <scan_planner_msgs/msg/grid_map_observation_diagnostics.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "plan_env/grid_map.h"

namespace
{
using namespace std::chrono_literals;

rclcpp::NodeOptions gridMapNodeOptions(const std::string &node_name)
{
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=" + node_name});
  options.append_parameter_override("grid_map.resolution", 0.25);
  options.append_parameter_override("grid_map.sliding_map_size_x", 2.0);
  options.append_parameter_override("grid_map.sliding_map_size_y", 2.0);
  options.append_parameter_override("grid_map.sliding_map_size_z", 1.0);
  options.append_parameter_override("grid_map.local_update_range_x", 1.0);
  options.append_parameter_override("grid_map.local_update_range_y", 1.0);
  options.append_parameter_override("grid_map.local_update_range_z", 0.5);
  options.append_parameter_override(
      "grid_map.obstacles_inflation_z_up", 0.25);
  options.append_parameter_override(
      "grid_map.obstacles_inflation_z_down", 0.25);
  options.append_parameter_override(
      "grid_map.double_cylinder_radius", 0.27);
  options.append_parameter_override(
      "grid_map.double_cylinder_offset", 0.16);
  options.append_parameter_override("grid_map.map_sliding_en", true);
  options.append_parameter_override("grid_map.map_sliding_thresh", 0.25);
  options.append_parameter_override("grid_map.fx", 100.0);
  options.append_parameter_override("grid_map.fy", 100.0);
  options.append_parameter_override("grid_map.cx", 2.0);
  options.append_parameter_override("grid_map.cy", 2.0);
  options.append_parameter_override("grid_map.depth_filter_maxdist", 5.0);
  options.append_parameter_override("grid_map.depth_filter_mindist", 0.1);
  options.append_parameter_override("grid_map.depth_filter_margin", 1);
  options.append_parameter_override(
      "grid_map.k_depth_scaling_factor", 1000.0);
  options.append_parameter_override("grid_map.skip_pixel", 2);
  options.append_parameter_override("grid_map.p_hit", 0.85);
  options.append_parameter_override("grid_map.p_miss", 0.30);
  options.append_parameter_override("grid_map.p_min", 0.12);
  options.append_parameter_override("grid_map.p_max", 0.98);
  options.append_parameter_override("grid_map.p_occ", 0.80);
  options.append_parameter_override("grid_map.max_ray_length", 5.0);
  options.append_parameter_override("grid_map.frame_id", "world");
  options.append_parameter_override("grid_map.base_frame_id", "base_link");
  options.append_parameter_override("grid_map.sensor_type", "lidar");
  options.append_parameter_override("grid_map.cloud_is_world", true);
  options.append_parameter_override("grid_map.need_extrinsic", false);
  options.append_parameter_override(
      "grid_map.observation_timeout_sec", 0.50);
  options.append_parameter_override(
      "grid_map.max_cloud_pose_skew_sec", 0.20);
  options.append_parameter_override("grid_map.sensor_sync_queue_size", 10);
  return options;
}

rclcpp::NodeOptions asymmetricInflationNodeOptions(
    const std::string &node_name)
{
  auto options = gridMapNodeOptions(node_name);
  options.append_parameter_override("grid_map.resolution", 0.05);
  options.append_parameter_override("grid_map.sliding_map_size_z", 2.0);
  options.append_parameter_override("grid_map.ground_height", -1.0);
  options.append_parameter_override(
      "grid_map.obstacles_inflation_z_up", 0.40);
  options.append_parameter_override(
      "grid_map.obstacles_inflation_z_down", 0.50);
  options.append_parameter_override(
      "grid_map.double_cylinder_radius", 0.05);
  options.append_parameter_override(
      "grid_map.double_cylinder_offset", 0.0);
  return options;
}

rclcpp::NodeOptions rotationSweepNodeOptions(const std::string &node_name)
{
  auto options = asymmetricInflationNodeOptions(node_name);
  options.append_parameter_override(
      "grid_map.double_cylinder_offset", 0.16);
  return options;
}

rclcpp::NodeOptions productionCorridorNodeOptions(
    const std::string &node_name)
{
  auto options = rotationSweepNodeOptions(node_name);
  options.append_parameter_override("grid_map.sliding_map_size_x", 3.0);
  options.append_parameter_override("grid_map.sliding_map_size_y", 3.0);
  options.append_parameter_override("grid_map.local_update_range_x", 1.5);
  options.append_parameter_override("grid_map.local_update_range_y", 1.5);
  options.append_parameter_override(
      "grid_map.double_cylinder_radius", 0.27);
  return options;
}

rclcpp::NodeOptions cameraExtrinsicNodeOptions(
    const std::string &node_name)
{
  auto options = gridMapNodeOptions(node_name);
  options.append_parameter_override("grid_map.need_extrinsic", true);
  options.append_parameter_override(
      "grid_map.cloud_sensor_extrinsic_x", 0.28);
  options.append_parameter_override(
      "grid_map.cloud_sensor_extrinsic_y", 0.0);
  options.append_parameter_override(
      "grid_map.cloud_sensor_extrinsic_z", 0.07);
  options.append_parameter_override(
      "grid_map.cloud_sensor_extrinsic_qw", 1.0);
  options.append_parameter_override(
      "grid_map.cloud_sensor_extrinsic_qx", 0.0);
  options.append_parameter_override(
      "grid_map.cloud_sensor_extrinsic_qy", 0.0);
  options.append_parameter_override(
      "grid_map.cloud_sensor_extrinsic_qz", 0.0);
  return options;
}

rclcpp::NodeOptions diagnosticUniformSamplingNodeOptions(
    const std::string &node_name)
{
  auto options = gridMapNodeOptions(node_name);
  options.append_parameter_override("grid_map.resolution", 0.05);
  options.append_parameter_override("grid_map.sliding_map_size_x", 4.0);
  options.append_parameter_override("grid_map.sliding_map_size_y", 4.0);
  options.append_parameter_override("grid_map.sliding_map_size_z", 1.0);
  options.append_parameter_override("grid_map.local_update_range_x", 2.0);
  options.append_parameter_override("grid_map.local_update_range_y", 2.0);
  options.append_parameter_override("grid_map.local_update_range_z", 0.5);
  options.append_parameter_override(
      "grid_map.obstacles_inflation_z_up", 0.0);
  options.append_parameter_override(
      "grid_map.obstacles_inflation_z_down", 0.0);
  options.append_parameter_override(
      "grid_map.double_cylinder_radius", 0.05);
  options.append_parameter_override(
      "grid_map.double_cylinder_offset", 0.0);
  options.append_parameter_override("grid_map.map_sliding_en", false);
  return options;
}

sensor_msgs::msg::PointCloud2 canonicalEmptyCloud(
    const builtin_interfaces::msg::Time &stamp)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp = stamp;
  cloud.header.frame_id = "world";
  cloud.height = 1U;
  cloud.width = 0U;
  cloud.fields = {
      sensor_msgs::msg::PointField()
        .set__name("x")
        .set__offset(0U)
        .set__datatype(sensor_msgs::msg::PointField::FLOAT32)
        .set__count(1U),
      sensor_msgs::msg::PointField()
        .set__name("y")
        .set__offset(4U)
        .set__datatype(sensor_msgs::msg::PointField::FLOAT32)
        .set__count(1U),
      sensor_msgs::msg::PointField()
        .set__name("z")
        .set__offset(8U)
        .set__datatype(sensor_msgs::msg::PointField::FLOAT32)
        .set__count(1U)};
  cloud.is_bigendian = false;
  cloud.point_step = 12U;
  cloud.row_step = 0U;
  cloud.data.clear();
  cloud.is_dense = true;
  return cloud;
}

sensor_msgs::msg::PointCloud2 taggedRayEndpointCloud(
    const builtin_interfaces::msg::Time &stamp,
    const std::array<float, 3> &point,
    const std::uint8_t endpoint_type)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp = stamp;
  cloud.header.frame_id = "world";
  cloud.height = 1U;
  cloud.width = 1U;
  cloud.fields = {
      sensor_msgs::msg::PointField()
        .set__name("x")
        .set__offset(0U)
        .set__datatype(sensor_msgs::msg::PointField::FLOAT32)
        .set__count(1U),
      sensor_msgs::msg::PointField()
        .set__name("y")
        .set__offset(4U)
        .set__datatype(sensor_msgs::msg::PointField::FLOAT32)
        .set__count(1U),
      sensor_msgs::msg::PointField()
        .set__name("z")
        .set__offset(8U)
        .set__datatype(sensor_msgs::msg::PointField::FLOAT32)
        .set__count(1U),
      sensor_msgs::msg::PointField()
        .set__name("ray_endpoint_type")
        .set__offset(12U)
        .set__datatype(sensor_msgs::msg::PointField::UINT8)
        .set__count(1U)};
  cloud.is_bigendian = false;
  cloud.point_step = 13U;
  cloud.row_step = cloud.point_step;
  cloud.data.resize(cloud.point_step);
  for (std::size_t axis = 0; axis < point.size(); ++axis)
  {
    std::memcpy(
        cloud.data.data() + axis * sizeof(float),
        &point[axis],
        sizeof(float));
  }
  cloud.data[12] = endpoint_type;
  cloud.is_dense = true;
  return cloud;
}

sensor_msgs::msg::PointCloud2 manyTaggedHitEndpointsCloud(
    const builtin_interfaces::msg::Time &stamp,
    const std::size_t point_count)
{
  auto cloud = taggedRayEndpointCloud(
      stamp, {0.1F, 0.0F, 0.5F}, 1U);
  cloud.width = static_cast<std::uint32_t>(point_count);
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.assign(cloud.row_step, 0U);
  for (std::size_t index = 0; index < point_count; ++index)
  {
    const std::array<float, 3> point{
        static_cast<float>(
            0.1 + 0.8 * static_cast<double>(index) /
                      static_cast<double>(point_count - 1U)),
        0.0F, 0.5F};
    const std::size_t base = index * cloud.point_step;
    for (std::size_t axis = 0; axis < point.size(); ++axis)
      std::memcpy(
          cloud.data.data() + base + axis * sizeof(float),
          &point[axis], sizeof(float));
    cloud.data[base + 12U] = 1U;
  }
  return cloud;
}

sensor_msgs::msg::PointCloud2 taggedRayEndpointsCloud(
    const builtin_interfaces::msg::Time &stamp,
    const std::vector<std::array<float, 3>> &points,
    const std::uint8_t endpoint_type)
{
  auto cloud = taggedRayEndpointCloud(
      stamp, points.empty() ? std::array<float, 3>{0.0F, 0.0F, 0.0F}
                            : points.front(),
      endpoint_type);
  cloud.width = static_cast<std::uint32_t>(points.size());
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.assign(cloud.row_step, 0U);
  for (std::size_t index = 0; index < points.size(); ++index)
  {
    const std::size_t base = index * cloud.point_step;
    for (std::size_t axis = 0; axis < points[index].size(); ++axis)
      std::memcpy(
          cloud.data.data() + base + axis * sizeof(float),
          &points[index][axis], sizeof(float));
    cloud.data[base + 12U] = endpoint_type;
  }
  return cloud;
}

std::vector<std::array<float, 3>> circularRayEndpoints(
    const std::size_t point_count)
{
  std::vector<std::array<float, 3>> points;
  points.reserve(point_count);
  constexpr double radius_m = 1.5;
  constexpr double sensor_height_m = 0.5;
  const double full_turn = 2.0 * std::acos(-1.0);
  for (std::size_t index = 0; index < point_count; ++index)
  {
    const double angle =
        full_turn * static_cast<double>(index) /
        static_cast<double>(point_count);
    points.push_back(
        {static_cast<float>(radius_m * std::cos(angle)),
         static_cast<float>(radius_m * std::sin(angle)),
         static_cast<float>(sensor_height_m)});
  }
  return points;
}

sensor_msgs::msg::PointCloud2 pairedHitAndExplicitFreeCloud(
    const builtin_interfaces::msg::Time &stamp,
    const std::array<float, 3> &point)
{
  auto cloud = taggedRayEndpointCloud(stamp, point, 1U);
  cloud.width = 2U;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(cloud.row_step);
  for (std::size_t sample = 0; sample < 2U; ++sample)
  {
    const std::size_t base = sample * cloud.point_step;
    for (std::size_t axis = 0; axis < point.size(); ++axis)
      std::memcpy(
          cloud.data.data() + base + axis * sizeof(float),
          &point[axis], sizeof(float));
    cloud.data[base + 12U] = sample == 0U ? 1U : 0U;
  }
  return cloud;
}

nav_msgs::msg::Odometry sensorPose(
    const builtin_interfaces::msg::Time &stamp)
{
  nav_msgs::msg::Odometry pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = "world";
  pose.child_frame_id = "base_link";
  pose.pose.pose.position.z = 0.5;
  pose.pose.pose.orientation.w = 1.0;
  return pose;
}

nav_msgs::msg::Odometry sensorPoseAt(
    const builtin_interfaces::msg::Time &stamp,
    const double x,
    const double y,
    const double z)
{
  auto pose = sensorPose(stamp);
  pose.pose.pose.position.x = x;
  pose.pose.pose.position.y = y;
  pose.pose.pose.position.z = z;
  return pose;
}

bool waitForSubscriptions(
    const rclcpp::PublisherBase::SharedPtr &publisher,
    rclcpp::executors::SingleThreadedExecutor &executor)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    if (publisher->get_subscription_count() > 0U)
      return true;
    std::this_thread::sleep_for(10ms);
  }
  return false;
}

bool waitForObservation(
    GridMap &grid_map,
    rclcpp::executors::SingleThreadedExecutor &executor)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    if (grid_map.observationReady())
      return true;
    std::this_thread::sleep_for(5ms);
  }
  return false;
}

bool waitForObservationStamp(
    GridMap &grid_map,
    rclcpp::executors::SingleThreadedExecutor &executor,
    const std::int64_t expected_stamp_ns)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    if (grid_map.observationStampNs() == expected_stamp_ns &&
        grid_map.observationReady())
      return true;
    std::this_thread::sleep_for(5ms);
  }
  return false;
}

bool waitForFusedObservationSequence(
    GridMap &grid_map,
    rclcpp::executors::SingleThreadedExecutor &executor,
    const std::uint64_t expected_sequence)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    if (grid_map.fusedObservationSequence() == expected_sequence)
      return true;
    std::this_thread::sleep_for(5ms);
  }
  return false;
}

bool waitForDiagnosticCount(
    const std::vector<
        scan_planner_msgs::msg::GridMapObservationDiagnostics> &diagnostics,
    rclcpp::executors::SingleThreadedExecutor &executor,
    const std::size_t expected_count)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    if (diagnostics.size() >= expected_count)
      return true;
    std::this_thread::sleep_for(5ms);
  }
  return false;
}

class GridMapEmptyCloudTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok())
    {
      int argc = 0;
      char **argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  static void TearDownTestSuite()
  {
    if (rclcpp::ok())
      rclcpp::shutdown();
  }
};

TEST(FusedObservationEvidence, CountsOnlyDistinctPostSettleStamps)
{
  std::deque<FusedObservationRecord> history{
      {1U, 90}, {2U, 95}, {3U, 99}};
  auto evidence = evaluateFusedObservationEvidence(
      history, 3U, 100, 0U, 200, 3U);
  ASSERT_TRUE(evidence.valid);
  EXPECT_FALSE(evidence.ready);
  EXPECT_EQ(evidence.distinct_stamp_count, 0U);

  history.push_back({4U, 101});
  history.push_back({5U, 101});
  history.push_back({6U, 102});
  evidence = evaluateFusedObservationEvidence(
      history, 6U, 100, 0U, 200, 3U);
  ASSERT_TRUE(evidence.valid);
  EXPECT_FALSE(evidence.ready);
  EXPECT_EQ(evidence.distinct_stamp_count, 2U);

  history.push_back({7U, 103});
  evidence = evaluateFusedObservationEvidence(
      history, 7U, 100, 0U, 200, 3U);
  ASSERT_TRUE(evidence.valid);
  EXPECT_TRUE(evidence.ready);
  EXPECT_EQ(evidence.distinct_stamp_count, 3U);
}

TEST(FusedObservationEvidence, RejectsFutureMissingTruncatedAndOverflowEvidence)
{
  EXPECT_FALSE(evaluateFusedObservationEvidence(
      {{1U, 101}}, 1U, 100, 0U, 100, 1U).valid);
  EXPECT_FALSE(evaluateFusedObservationEvidence(
      {{1U, 101}, {3U, 103}}, 3U, 100, 0U, 200, 2U).valid);
  EXPECT_FALSE(evaluateFusedObservationEvidence(
      {{2U, 102}, {3U, 103}, {4U, 104}},
      4U, 100, 0U, 200, 3U).valid);
  EXPECT_FALSE(evaluateFusedObservationEvidence(
      {{1U, 101}}, 1U, 100, 2U, 200, 1U).valid);
  EXPECT_FALSE(evaluateFusedObservationEvidence(
      {}, std::numeric_limits<std::uint64_t>::max(),
      100, std::numeric_limits<std::uint64_t>::max(), 200, 1U).valid);
  EXPECT_FALSE(evaluateFusedObservationEvidence(
      {}, 0U, 100, 0U, 200,
      kFusedObservationHistoryCapacity + 1U).valid);

  // baseline 后的完整区间即使历史从 sequence=2 开始也可直接证明；
  // baseline 之前的记录是否仍在 ring 中不影响本次证据。
  const auto retained = evaluateFusedObservationEvidence(
      {{2U, 102}, {3U, 103}, {4U, 104}},
      4U, 100, 1U, 200, 3U);
  EXPECT_TRUE(retained.valid);
  EXPECT_TRUE(retained.ready);

  // baseline 以前的旧 epoch 大时间戳不能进入本窗口，也不能让本窗口失效。
  const auto old_epoch_ignored = evaluateFusedObservationEvidence(
      {{1U, 1000}, {2U, 102}, {3U, 103}, {4U, 104}},
      4U, 100, 1U, 200, 3U);
  EXPECT_TRUE(old_epoch_ignored.valid);
  EXPECT_TRUE(old_epoch_ignored.ready);
}

TEST(FusedObservationEvidence, SixtyFourRecordBoundaryIsExplicit)
{
  std::deque<FusedObservationRecord> history;
  for (std::uint64_t sequence = 1U;
       sequence <= kFusedObservationHistoryCapacity; ++sequence)
    history.push_back({sequence, 100 + static_cast<std::int64_t>(sequence)});

  auto evidence = evaluateFusedObservationEvidence(
      history, kFusedObservationHistoryCapacity,
      100, 0U, 1000, 3U);
  ASSERT_TRUE(evidence.valid);
  EXPECT_TRUE(evidence.ready);

  history.pop_front();
  history.push_back({65U, 165});
  evidence = evaluateFusedObservationEvidence(
      history, 65U, 100, 0U, 1000, 3U);
  EXPECT_FALSE(evidence.valid);
  evidence = evaluateFusedObservationEvidence(
      history, 65U, 100, 64U, 1000, 1U);
  EXPECT_TRUE(evidence.valid);
  EXPECT_TRUE(evidence.ready);
}

TEST_F(
    GridMapEmptyCloudTest,
    DelayedPreSettleCloudsCannotUnlockPostSettleEvidence)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_fused_evidence_test",
      gridMapNodeOptions("grid_map_fused_evidence_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const std::int64_t settle_stamp_ns = node->now().nanoseconds();
  ASSERT_GT(settle_stamp_ns, 30000000LL);
  constexpr std::array<std::int64_t, 3> old_offsets_ns{
      30000000LL, 20000000LL, 10000000LL};
  for (std::size_t index = 0; index < old_offsets_ns.size(); ++index)
  {
    const builtin_interfaces::msg::Time acquisition_stamp = rclcpp::Time(
        settle_stamp_ns - old_offsets_ns[index]);
    cloud_publisher->publish(taggedRayEndpointCloud(
        acquisition_stamp, {0.875F, 0.0F, 0.5F}, 0U));
    pose_publisher->publish(sensorPose(acquisition_stamp));
    ASSERT_TRUE(waitForFusedObservationSequence(
        grid_map, executor, index + 1U));
  }

  auto evidence = grid_map.fusedObservationEvidenceAfter(
      settle_stamp_ns, 0U, node->now().nanoseconds(), 3U);
  ASSERT_TRUE(evidence.valid);
  EXPECT_FALSE(evidence.ready);
  EXPECT_EQ(evidence.distinct_stamp_count, 0U);

  for (std::uint64_t index = 0U; index < 3U; ++index)
  {
    const builtin_interfaces::msg::Time acquisition_stamp = node->now();
    ASSERT_GT(
        rclcpp::Time(acquisition_stamp).nanoseconds(),
        settle_stamp_ns);
    cloud_publisher->publish(taggedRayEndpointCloud(
        acquisition_stamp, {0.875F, 0.0F, 0.5F}, 0U));
    pose_publisher->publish(sensorPose(acquisition_stamp));
    ASSERT_TRUE(waitForFusedObservationSequence(
        grid_map, executor, 4U + index));
  }

  evidence = grid_map.fusedObservationEvidenceAfter(
      settle_stamp_ns, 0U, node->now().nanoseconds(), 3U);
  ASSERT_TRUE(evidence.valid);
  EXPECT_TRUE(evidence.ready);
  EXPECT_EQ(evidence.distinct_stamp_count, 3U);
  EXPECT_EQ(evidence.current_sequence, 6U);
}

TEST_F(
    GridMapEmptyCloudTest,
    CanonicalEmptyRefreshesObservationWithoutClearingOccupancy)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_empty_cloud_test",
      gridMapNodeOptions("grid_map_empty_cloud_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  const Eigen::Vector3d occupied_point(0.25, 0.25, 0.25);
  grid_map.setOccupied(occupied_point);
  ASSERT_EQ(grid_map.getOccupancy(occupied_point), 1);
  ASSERT_FALSE(grid_map.observationReady());
  ASSERT_EQ(grid_map.observationStampNs(), 0);
  ASSERT_EQ(grid_map.fusedObservationSequence(), 0U);

  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  cloud_publisher->publish(canonicalEmptyCloud(stamp));
  pose_publisher->publish(sensorPose(stamp));

  ASSERT_TRUE(waitForObservation(grid_map, executor));
  EXPECT_TRUE(grid_map.odomValid());
  EXPECT_EQ(
      grid_map.observationStampNs(),
      rclcpp::Time(stamp).nanoseconds());
  EXPECT_EQ(grid_map.getOccupancy(occupied_point), 1);
  EXPECT_EQ(grid_map.fusedObservationSequence(), 0U);

  // 让 50 ms 占据更新定时器实际运行，确认空云没有排入 raycast。
  const auto timer_deadline = std::chrono::steady_clock::now() + 100ms;
  while (std::chrono::steady_clock::now() < timer_deadline)
  {
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_EQ(grid_map.getOccupancy(occupied_point), 1);
  EXPECT_EQ(grid_map.fusedObservationSequence(), 0U);
  ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, 1U));
  EXPECT_EQ(diagnostics.back().observation_sequence, 1U);
  EXPECT_EQ(
      rclcpp::Time(diagnostics.back().header.stamp).nanoseconds(),
      rclcpp::Time(stamp).nanoseconds());
  EXPECT_TRUE(diagnostics.back().canonical_empty);
  EXPECT_FALSE(diagnostics.back().map_fusion_performed);
  EXPECT_EQ(diagnostics.back().input_point_count, 0U);
  EXPECT_EQ(
      diagnostics.back().occupied_to_free_by_explicit_miss_count, 0U);
  (void)diagnostics_subscription;
}

TEST_F(
    GridMapEmptyCloudTest,
    CloudRayOriginUsesConfiguredCameraExtrinsic)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_camera_extrinsic_test",
      cameraExtrinsicNodeOptions("grid_map_camera_extrinsic_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());
  ASSERT_EQ(grid_map.fusedObservationSequence(), 0U);

  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  auto pose = sensorPoseAt(stamp, 1.0, 2.0, 0.5);
  const double half_sqrt_two = std::sqrt(0.5);
  pose.pose.pose.orientation.w = half_sqrt_two;
  pose.pose.pose.orientation.z = half_sqrt_two;
  cloud_publisher->publish(canonicalEmptyCloud(stamp));
  pose_publisher->publish(pose);

  ASSERT_TRUE(waitForObservation(grid_map, executor));
  ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, 1U));
  EXPECT_NEAR(diagnostics.back().sensor_origin.x, 1.0, 1.0e-9);
  EXPECT_NEAR(diagnostics.back().sensor_origin.y, 2.28, 1.0e-9);
  EXPECT_NEAR(diagnostics.back().sensor_origin.z, 0.57, 1.0e-9);
  (void)diagnostics_subscription;
}

TEST_F(GridMapEmptyCloudTest, NonCanonicalEmptyDoesNotRefreshObservation)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_invalid_empty_cloud_test",
      gridMapNodeOptions("grid_map_invalid_empty_cloud_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  auto malformed = canonicalEmptyCloud(stamp);
  malformed.point_step = 0U;
  cloud_publisher->publish(malformed);
  pose_publisher->publish(sensorPose(stamp));

  const auto deadline = std::chrono::steady_clock::now() + 150ms;
  while (std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_FALSE(grid_map.observationReady());
  EXPECT_EQ(grid_map.observationStampNs(), 0);

  malformed = canonicalEmptyCloud(stamp);
  malformed.is_dense = false;
  cloud_publisher->publish(malformed);
  pose_publisher->publish(sensorPose(stamp));

  const auto dense_deadline = std::chrono::steady_clock::now() + 150ms;
  while (std::chrono::steady_clock::now() < dense_deadline)
  {
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_FALSE(grid_map.observationReady());
  EXPECT_EQ(grid_map.observationStampNs(), 0);
  EXPECT_TRUE(diagnostics.empty());
  (void)diagnostics_subscription;
}

TEST_F(
    GridMapEmptyCloudTest,
    ExplicitFreeRayClearsSaturatedVoxelWithinThreeFusedObservations)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_explicit_free_ray_test",
      gridMapNodeOptions("grid_map_explicit_free_ray_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());
  ASSERT_EQ(grid_map.fusedObservationSequence(), 0U);

  const Eigen::Vector3d old_obstacle(0.375, 0.0, 0.5);
  grid_map.setOccupied(old_obstacle);
  ASSERT_EQ(grid_map.getOccupancy(old_obstacle), 1);

  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  // p_max=0.98、p_miss=0.30、p_occ=0.80 时，饱和体素最多经过
  // 3 次明确 free ray 即低于占据门限。清除仅发生在真实
  // 射线穿过的体素，不会按时间删除未观测的静态障碍。
  for (int miss_count = 1; miss_count <= 3; ++miss_count)
  {
    const builtin_interfaces::msg::Time stamp = node->now();
    cloud_publisher->publish(taggedRayEndpointCloud(
        stamp, {0.875F, 0.0F, 0.5F}, 0U));
    pose_publisher->publish(sensorPose(stamp));
    ASSERT_TRUE(waitForObservationStamp(
        grid_map, executor, rclcpp::Time(stamp).nanoseconds()));
    ASSERT_TRUE(waitForDiagnosticCount(
        diagnostics, executor, static_cast<std::size_t>(miss_count)));
    EXPECT_EQ(
        grid_map.fusedObservationSequence(),
        static_cast<std::uint64_t>(miss_count));
    EXPECT_EQ(
        grid_map.getOccupancy(old_obstacle),
        miss_count < 3 ? 1 : 0);
  }
  EXPECT_EQ(grid_map.getInflateOccupancy(old_obstacle, 0.0), 0);
  EXPECT_EQ(grid_map.fusedObservationSequence(), 3U);
  ASSERT_EQ(diagnostics.size(), 3U);
  for (std::size_t index = 0; index < diagnostics.size(); ++index)
  {
    EXPECT_EQ(diagnostics[index].observation_sequence, index + 1U);
    EXPECT_FALSE(diagnostics[index].canonical_empty);
    EXPECT_TRUE(diagnostics[index].map_fusion_performed);
    EXPECT_EQ(diagnostics[index].explicit_free_endpoint_count, 1U);
    EXPECT_GT(diagnostics[index].explicit_free_miss_voxel_count, 0U);
    EXPECT_EQ(
        diagnostics[index].occupied_removed_by_sliding_reset_count, 0U);
  }
  const auto &clear = diagnostics.back();
  EXPECT_EQ(clear.occupied_to_free_by_explicit_miss_count, 1U);
  EXPECT_FALSE(clear.occupied_to_free_samples_truncated);
  ASSERT_EQ(clear.occupied_to_free_by_explicit_miss_samples.size(), 1U);
  EXPECT_NEAR(
      clear.occupied_to_free_by_explicit_miss_samples.front().x,
      0.375, 1.0e-12);
  EXPECT_NEAR(
      clear.occupied_to_free_by_explicit_miss_samples.front().y,
      0.125, 1.0e-12);
  EXPECT_NEAR(
      clear.occupied_to_free_by_explicit_miss_samples.front().z,
      0.625, 1.0e-12);
  ASSERT_EQ(
      clear.occupied_to_free_transition_hit_observation_sequences.size(),
      1U);
  EXPECT_EQ(
      clear.occupied_to_free_transition_hit_observation_sequences.front(),
      0U);
  (void)diagnostics_subscription;
}

TEST_F(
    GridMapEmptyCloudTest,
    RealHitTransitionIsPinnedAndExplicitClearReferencesSameVoxel)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_transition_provenance_test",
      gridMapNodeOptions("grid_map_transition_provenance_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const std::array<float, 3> endpoint{0.875F, 0.0F, 0.5F};
  for (std::size_t observation = 1U; observation <= 2U; ++observation)
  {
    const builtin_interfaces::msg::Time stamp = node->now();
    cloud_publisher->publish(taggedRayEndpointCloud(stamp, endpoint, 1U));
    pose_publisher->publish(sensorPose(stamp));
    ASSERT_TRUE(waitForObservationStamp(
        grid_map, executor, rclcpp::Time(stamp).nanoseconds()));
    ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, observation));
  }

  ASSERT_EQ(diagnostics.size(), 2U);
  const auto &transition = diagnostics.back();
  EXPECT_DOUBLE_EQ(transition.map_resolution, 0.25);
  EXPECT_EQ(transition.observation_sequence, 2U);
  EXPECT_EQ(transition.free_to_occupied_transition_count, 1U);
  EXPECT_FALSE(transition.free_to_occupied_transition_samples_truncated);
  ASSERT_EQ(transition.free_to_occupied_transition_hit_samples.size(), 1U);
  ASSERT_EQ(
      transition.free_to_occupied_transition_voxel_indices_xyz.size(), 3U);
  EXPECT_EQ(
      std::vector<std::int64_t>(
          transition.free_to_occupied_transition_voxel_indices_xyz.begin(),
          transition.free_to_occupied_transition_voxel_indices_xyz.end()),
      (std::vector<std::int64_t>{3, 0, 2}));

  const builtin_interfaces::msg::Time clear_stamp = node->now();
  cloud_publisher->publish(
      taggedRayEndpointCloud(clear_stamp, endpoint, 0U));
  pose_publisher->publish(sensorPose(clear_stamp));
  ASSERT_TRUE(waitForObservationStamp(
      grid_map, executor, rclcpp::Time(clear_stamp).nanoseconds()));
  ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, 3U));

  const auto &clear = diagnostics.back();
  EXPECT_EQ(clear.observation_sequence, 3U);
  EXPECT_EQ(clear.occupied_to_free_by_explicit_miss_count, 1U);
  ASSERT_EQ(clear.occupied_to_free_by_explicit_miss_samples.size(), 1U);
  EXPECT_EQ(
      std::vector<std::int64_t>(
          clear.occupied_to_free_sample_voxel_indices_xyz.begin(),
          clear.occupied_to_free_sample_voxel_indices_xyz.end()),
      (std::vector<std::int64_t>{3, 0, 2}));
  ASSERT_EQ(
      clear.occupied_to_free_transition_hit_observation_sequences.size(),
      1U);
  EXPECT_EQ(
      clear.occupied_to_free_transition_hit_observation_sequences.front(),
      transition.observation_sequence);
  ASSERT_EQ(clear.occupied_to_free_transition_hit_samples.size(), 1U);
  EXPECT_NEAR(
      clear.occupied_to_free_transition_hit_samples.front().x,
      endpoint[0], 1.0e-6);
  ASSERT_EQ(
      clear.occupied_to_free_transition_hit_header_stamp_ns.size(), 1U);
  EXPECT_EQ(
      clear.occupied_to_free_transition_hit_header_stamp_ns.front(),
      rclcpp::Time(transition.header.stamp).nanoseconds());
  (void)diagnostics_subscription;
}

TEST_F(
    GridMapEmptyCloudTest,
    TransitionAndClearDiagnosticsUniformlyCoverCandidateAfterFirst64)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_uniform_transition_diagnostic_test",
      diagnosticUniformSamplingNodeOptions(
          "grid_map_uniform_transition_diagnostic_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  // 第 65 个端点在旧 FIFO 前 64 抽样中必然缺失。圆周上的射线端点落在
  // 互异体素且互不遮挡，第二次 hit 会产生 65 个真实 occupancy transition。
  constexpr std::size_t transition_count = 65U;
  const auto endpoints = circularRayEndpoints(transition_count);
  for (std::size_t observation = 1U; observation <= 2U; ++observation)
  {
    const builtin_interfaces::msg::Time stamp = node->now();
    cloud_publisher->publish(
        taggedRayEndpointsCloud(stamp, endpoints, 1U));
    pose_publisher->publish(sensorPose(stamp));
    ASSERT_TRUE(waitForObservationStamp(
        grid_map, executor, rclcpp::Time(stamp).nanoseconds()));
    ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, observation));
  }

  ASSERT_EQ(diagnostics.size(), 2U);
  const auto &transition = diagnostics.back();
  ASSERT_EQ(
      transition.free_to_occupied_transition_count,
      transition_count);
  EXPECT_TRUE(transition.free_to_occupied_transition_samples_truncated);
  ASSERT_EQ(
      transition.free_to_occupied_transition_hit_samples.size(), 64U);
  ASSERT_EQ(
      transition.free_to_occupied_transition_voxel_indices_xyz.size(),
      3U * 64U);
  const auto &target_endpoint = endpoints.back();
  EXPECT_NEAR(
      transition.free_to_occupied_transition_hit_samples.back().x,
      target_endpoint[0], 1.0e-6);
  EXPECT_NEAR(
      transition.free_to_occupied_transition_hit_samples.back().y,
      target_endpoint[1], 1.0e-6);
  EXPECT_NEAR(
      transition.free_to_occupied_transition_hit_samples.back().z,
      target_endpoint[2], 1.0e-6);
  const std::array<std::int64_t, 3> target_voxel_index{
      transition.free_to_occupied_transition_voxel_indices_xyz[
          3U * 63U],
      transition.free_to_occupied_transition_voxel_indices_xyz[
          3U * 63U + 1U],
      transition.free_to_occupied_transition_voxel_indices_xyz[
          3U * 63U + 2U]};

  const builtin_interfaces::msg::Time clear_stamp = node->now();
  cloud_publisher->publish(
      taggedRayEndpointsCloud(clear_stamp, endpoints, 0U));
  pose_publisher->publish(sensorPose(clear_stamp));
  ASSERT_TRUE(waitForObservationStamp(
      grid_map, executor, rclcpp::Time(clear_stamp).nanoseconds()));
  ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, 3U));

  const auto &clear = diagnostics.back();
  ASSERT_EQ(
      clear.occupied_to_free_by_explicit_miss_count,
      transition_count);
  EXPECT_TRUE(clear.occupied_to_free_samples_truncated);
  ASSERT_EQ(clear.occupied_to_free_by_explicit_miss_samples.size(), 64U);
  ASSERT_EQ(
      clear.occupied_to_free_sample_voxel_indices_xyz.size(), 3U * 64U);
  ASSERT_EQ(
      clear.occupied_to_free_transition_hit_observation_sequences.size(),
      64U);
  ASSERT_EQ(clear.occupied_to_free_transition_hit_samples.size(), 64U);
  ASSERT_EQ(
      clear.occupied_to_free_transition_hit_header_stamp_ns.size(), 64U);
  EXPECT_EQ(
      (std::array<std::int64_t, 3>{
          clear.occupied_to_free_sample_voxel_indices_xyz[3U * 63U],
          clear.occupied_to_free_sample_voxel_indices_xyz[3U * 63U + 1U],
          clear.occupied_to_free_sample_voxel_indices_xyz[3U * 63U + 2U]}),
      target_voxel_index);
  EXPECT_EQ(
      clear.occupied_to_free_transition_hit_observation_sequences.back(),
      transition.observation_sequence);
  EXPECT_NEAR(
      clear.occupied_to_free_transition_hit_samples.back().x,
      target_endpoint[0], 1.0e-6);
  EXPECT_NEAR(
      clear.occupied_to_free_transition_hit_samples.back().y,
      target_endpoint[1], 1.0e-6);
  EXPECT_NEAR(
      clear.occupied_to_free_transition_hit_samples.back().z,
      target_endpoint[2], 1.0e-6);
  EXPECT_EQ(
      clear.occupied_to_free_transition_hit_header_stamp_ns.back(),
      rclcpp::Time(transition.header.stamp).nanoseconds());
  (void)diagnostics_subscription;
}

TEST_F(
    GridMapEmptyCloudTest,
    ExplicitMissCannotHitchhikeOnSufficientOrdinaryMisses)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_explicit_counterfactual_test",
      gridMapNodeOptions("grid_map_explicit_counterfactual_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());
  const Eigen::Vector3d preoccupied_voxel(0.375, 0.0, 0.5);
  grid_map.setOccupied(preoccupied_voxel);

  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  cloud_publisher->publish(pairedHitAndExplicitFreeCloud(
      stamp, {0.875F, 0.0F, 0.5F}));
  pose_publisher->publish(sensorPose(stamp));
  ASSERT_TRUE(waitForObservationStamp(
      grid_map, executor, rclcpp::Time(stamp).nanoseconds()));
  ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, 1U));

  const auto &diagnostic = diagnostics.back();
  EXPECT_EQ(diagnostic.hit_endpoint_count, 1U);
  EXPECT_EQ(diagnostic.explicit_free_endpoint_count, 1U);
  // endpoint voxel 的一条 explicit miss 仍可能是必要原因；关键是已有
  // ordinary miss 足够决定的预占据穿越 voxel 不能搭便车记成 clear。
  EXPECT_EQ(diagnostic.explicit_free_miss_voxel_count, 1U);
  EXPECT_EQ(diagnostic.occupied_to_free_by_explicit_miss_count, 0U);
  EXPECT_TRUE(
      diagnostic.occupied_to_free_by_explicit_miss_samples.empty());
  EXPECT_EQ(grid_map.getOccupancy(preoccupied_voxel), 1);
  (void)diagnostics_subscription;
}

TEST_F(GridMapEmptyCloudTest, HitEndpointDiagnosticUsesSourceStampAndWorldPoint)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_hit_diagnostic_test",
      gridMapNodeOptions("grid_map_hit_diagnostic_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  cloud_publisher->publish(taggedRayEndpointCloud(
      stamp, {0.875F, 0.0F, 0.5F}, 1U));
  pose_publisher->publish(sensorPose(stamp));
  ASSERT_TRUE(waitForObservationStamp(
      grid_map, executor, rclcpp::Time(stamp).nanoseconds()));
  ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, 1U));

  const auto &diagnostic = diagnostics.back();
  EXPECT_EQ(
      rclcpp::Time(diagnostic.header.stamp).nanoseconds(),
      rclcpp::Time(stamp).nanoseconds());
  EXPECT_EQ(diagnostic.header.frame_id, "world");
  EXPECT_EQ(diagnostic.input_point_count, 1U);
  EXPECT_EQ(diagnostic.accepted_endpoint_count, 1U);
  EXPECT_EQ(diagnostic.hit_endpoint_count, 1U);
  EXPECT_EQ(diagnostic.explicit_free_endpoint_count, 0U);
  EXPECT_FALSE(diagnostic.hit_endpoint_samples_truncated);
  ASSERT_EQ(diagnostic.hit_endpoint_samples.size(), 1U);
  EXPECT_NEAR(diagnostic.hit_endpoint_samples.front().x, 0.875, 1.0e-6);
  EXPECT_NEAR(diagnostic.hit_endpoint_samples.front().y, 0.0, 1.0e-6);
  EXPECT_NEAR(diagnostic.hit_endpoint_samples.front().z, 0.5, 1.0e-6);
  EXPECT_EQ(diagnostic.occupied_to_free_by_explicit_miss_count, 0U);
  (void)diagnostics_subscription;
}

TEST_F(GridMapEmptyCloudTest, SlidingResetCannotMasqueradeAsExplicitMissClear)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_sliding_diagnostic_test",
      gridMapNodeOptions("grid_map_sliding_diagnostic_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());
  grid_map.setOccupied(Eigen::Vector3d(-0.875, 0.0, 0.5));

  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  cloud_publisher->publish(taggedRayEndpointCloud(
      stamp, {1.0F, 0.0F, 0.5F}, 0U));
  pose_publisher->publish(sensorPoseAt(stamp, 0.5, 0.0, 0.5));
  ASSERT_TRUE(waitForObservationStamp(
      grid_map, executor, rclcpp::Time(stamp).nanoseconds()));
  ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, 1U));

  const auto &diagnostic = diagnostics.back();
  EXPECT_EQ(diagnostic.occupied_removed_by_sliding_reset_count, 1U);
  EXPECT_GT(diagnostic.explicit_free_miss_voxel_count, 0U);
  EXPECT_EQ(diagnostic.occupied_to_free_by_explicit_miss_count, 0U);
  EXPECT_TRUE(
      diagnostic.occupied_to_free_by_explicit_miss_samples.empty());
  EXPECT_FALSE(diagnostic.occupied_to_free_samples_truncated);
  (void)diagnostics_subscription;
}

TEST_F(GridMapEmptyCloudTest, HitEndpointSamplesAreBoundedAcrossWholeObservation)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_bounded_hit_diagnostic_test",
      gridMapNodeOptions("grid_map_bounded_hit_diagnostic_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  std::vector<scan_planner_msgs::msg::GridMapObservationDiagnostics>
      diagnostics;
  auto diagnostics_subscription = node->create_subscription<
      scan_planner_msgs::msg::GridMapObservationDiagnostics>(
      "/planning/grid_map_observation_diagnostics",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [&diagnostics](
          const scan_planner_msgs::msg::GridMapObservationDiagnostics::
              ConstSharedPtr &message) { diagnostics.push_back(*message); });
  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  cloud_publisher->publish(manyTaggedHitEndpointsCloud(stamp, 65U));
  pose_publisher->publish(sensorPose(stamp));
  ASSERT_TRUE(waitForObservationStamp(
      grid_map, executor, rclcpp::Time(stamp).nanoseconds()));
  ASSERT_TRUE(waitForDiagnosticCount(diagnostics, executor, 1U));

  const auto &diagnostic = diagnostics.back();
  EXPECT_EQ(diagnostic.hit_endpoint_count, 65U);
  EXPECT_TRUE(diagnostic.hit_endpoint_samples_truncated);
  ASSERT_EQ(diagnostic.hit_endpoint_samples.size(), 64U);
  EXPECT_NEAR(diagnostic.hit_endpoint_samples.front().x, 0.1, 1.0e-6);
  EXPECT_NEAR(diagnostic.hit_endpoint_samples.back().x, 0.9, 1.0e-6);
  (void)diagnostics_subscription;
}

TEST_F(GridMapEmptyCloudTest, InvalidRayEndpointTypeIsRejectedFailClosed)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_invalid_ray_endpoint_test",
      gridMapNodeOptions("grid_map_invalid_ray_endpoint_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  const Eigen::Vector3d old_obstacle(0.375, 0.0, 0.5);
  grid_map.setOccupied(old_obstacle);
  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  cloud_publisher->publish(taggedRayEndpointCloud(
      stamp, {0.875F, 0.0F, 0.5F}, 2U));
  pose_publisher->publish(sensorPose(stamp));
  const auto deadline = std::chrono::steady_clock::now() + 150ms;
  while (std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_FALSE(grid_map.observationReady());
  EXPECT_EQ(grid_map.observationStampNs(), 0);
  EXPECT_EQ(grid_map.getOccupancy(old_obstacle), 1);
}

TEST_F(GridMapEmptyCloudTest, TaggedCloudWithTrailingDataIsRejectedFailClosed)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_trailing_cloud_data_test",
      gridMapNodeOptions("grid_map_trailing_cloud_data_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  const Eigen::Vector3d old_obstacle(0.375, 0.0, 0.5);
  grid_map.setOccupied(old_obstacle);
  auto cloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS());
  auto pose_publisher = node->create_publisher<nav_msgs::msg::Odometry>(
      "sensor_pose", rclcpp::SensorDataQoS());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  ASSERT_TRUE(waitForSubscriptions(cloud_publisher, executor));
  ASSERT_TRUE(waitForSubscriptions(pose_publisher, executor));

  const builtin_interfaces::msg::Time stamp = node->now();
  auto malformed_cloud = taggedRayEndpointCloud(
      stamp, {0.875F, 0.0F, 0.5F}, 0U);
  malformed_cloud.data.push_back(0U);
  cloud_publisher->publish(malformed_cloud);
  pose_publisher->publish(sensorPose(stamp));
  const auto deadline = std::chrono::steady_clock::now() + 150ms;
  while (std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_FALSE(grid_map.observationReady());
  EXPECT_EQ(grid_map.observationStampNs(), 0);
  EXPECT_EQ(grid_map.getOccupancy(old_obstacle), 1);
}

TEST_F(GridMapEmptyCloudTest, VerticalInflationDirectionsMatchRobotEnvelope)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_asymmetric_inflation_test",
      asymmetricInflationNodeOptions("grid_map_asymmetric_inflation_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  // 障碍位于原点。障碍向上膨胀 0.40 m，等价于机器人中心以下
  // 0.40 m 的包络；向下膨胀 0.50 m，等价于中心以上 0.50 m。
  grid_map.setOccupied(Eigen::Vector3d(0.0, 0.0, 0.0));
  EXPECT_EQ(
      grid_map.getInflateOccupancy(Eigen::Vector3d(0.0, 0.0, 0.40), 0.0),
      1);
  EXPECT_EQ(
      grid_map.getInflateOccupancy(Eigen::Vector3d(0.0, 0.0, 0.46), 0.0),
      0);
  EXPECT_EQ(
      grid_map.getInflateOccupancy(Eigen::Vector3d(0.0, 0.0, -0.50), 0.0),
      1);
  EXPECT_EQ(
      grid_map.getInflateOccupancy(Eigen::Vector3d(0.0, 0.0, -0.56), 0.0),
      0);
}

TEST_F(GridMapEmptyCloudTest, DoubleCylinderSeparatesTravelYawFromRotationSweep)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_rotation_sweep_test",
      rotationSweepNodeOptions("grid_map_rotation_sweep_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  // 障碍位于机器人侧方。yaw=0 直行时前后圆柱均不覆盖该点；
  // 但机器人原地转到 yaw=pi/2 时前圆柱会扫过该点。两种操作必须使用
  // 不同查询，不得用旋转外接圆否定合法的窄走廊直行轨迹。
  grid_map.setOccupied(Eigen::Vector3d(0.0, 0.15, 0.0));
  EXPECT_EQ(
      grid_map.getInflateOccupancy(Eigen::Vector3d::Zero(), 0.0),
      0);
  EXPECT_NE(
      grid_map.getRotationSweepOccupancy(Eigen::Vector3d::Zero()),
      0);
  EXPECT_EQ(
      grid_map.getInflateOccupancy(Eigen::Vector3d(0.45, 0.0, 0.0), 0.0),
      0);
  EXPECT_NE(
      grid_map.getInflateOccupancy(
          Eigen::Vector3d(
              std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0),
          0.0),
      0);
  EXPECT_NE(
      grid_map.getInflateOccupancy(
          Eigen::Vector3d::Zero(),
          std::numeric_limits<double>::quiet_NaN()),
      0);
  EXPECT_NE(
      grid_map.getRotationSweepOccupancy(
          Eigen::Vector3d(
              std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0)),
      0);
}

TEST_F(GridMapEmptyCloudTest, ProductionEnvelopeTraversesF2WidthWhenAligned)
{
  auto node = std::make_shared<rclcpp::Node>(
      "grid_map_f2_corridor_test",
      productionCorridorNodeOptions("grid_map_f2_corridor_test"));
  GridMap grid_map;
  grid_map.initMap(node.get());

  // phase231 二楼卡点的 PCT 段沿 -Y 方向，实测最小净空约 0.447 m。
  // 用 0.45 m 两侧墙复现同一量级：半径 0.27 m、前后偏置 0.16 m
  // 的双圆柱沿走廊方向能够通过，但原地任意旋转的扫掠包络不能通过。
  for (int sample = -14; sample <= 14; ++sample)
  {
    const double y = 0.05 * static_cast<double>(sample);
    grid_map.setOccupied(Eigen::Vector3d(-0.45, y, 0.0));
    grid_map.setOccupied(Eigen::Vector3d(0.45, y, 0.0));
  }

  const double travel_yaw = -0.5 * std::acos(-1.0);
  for (int sample = -10; sample <= 10; ++sample)
  {
    const Eigen::Vector3d position(
        0.0, 0.05 * static_cast<double>(sample), 0.0);
    EXPECT_EQ(grid_map.getInflateOccupancy(position, travel_yaw), 0)
        << "aligned travel sample=" << sample;
  }
  EXPECT_NE(
      grid_map.getRotationSweepOccupancy(Eigen::Vector3d::Zero()),
      0);
}
}  // namespace
