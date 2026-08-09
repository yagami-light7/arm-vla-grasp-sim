#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <scan_planner_msgs/msg/bspline.hpp>
#include <scan_planner_msgs/msg/controller_status.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "scan_controller/scan_controller_node.hpp"

namespace
{

using namespace std::chrono_literals;
using Status = scan_planner_msgs::msg::ControllerStatus;

builtin_interfaces::msg::Time makeStamp(
  std::int32_t seconds, std::uint32_t nanoseconds = 0U)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = seconds;
  stamp.nanosec = nanoseconds;
  return stamp;
}

nav_msgs::msg::Path makePath(
  const builtin_interfaces::msg::Time &stamp, bool turn = false)
{
  nav_msgs::msg::Path path;
  path.header.frame_id = "world";
  path.header.stamp = stamp;
  path.poses.resize(2U);
  for (auto &pose : path.poses) {
    pose.header = path.header;
    pose.pose.orientation.w = 1.0;
  }
  path.poses[1].pose.position.x = turn ? 0.0 : 1.0;
  path.poses[1].pose.position.y = turn ? 1.0 : 0.0;
  return path;
}

scan_planner_msgs::msg::Bspline makeBspline(
  std::int64_t trajectory_id,
  const builtin_interfaces::msg::Time &path_stamp,
  const builtin_interfaces::msg::Time &trajectory_stamp)
{
  scan_planner_msgs::msg::Bspline message;
  message.header.frame_id = "world";
  message.header.stamp = trajectory_stamp;
  message.order = 3;
  message.traj_id = trajectory_id;
  message.start_time = trajectory_stamp;
  message.reference_path_stamp = path_stamp;
  message.is_final = false;
  message.emergency_stop = false;
  message.knots = {0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0};
  message.pos_pts.resize(6U);
  for (std::size_t index = 0; index < message.pos_pts.size(); ++index) {
    message.pos_pts[index].x = static_cast<double>(index) / 5.0;
  }
  return message;
}

nav_msgs::msg::Odometry makeOdometry(
  const builtin_interfaces::msg::Time &stamp, double yaw = 0.0)
{
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "world";
  odometry.header.stamp = stamp;
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.position.z = 0.30;
  odometry.pose.pose.orientation.z = std::sin(0.5 * yaw);
  odometry.pose.pose.orientation.w = std::cos(0.5 * yaw);
  return odometry;
}

sensor_msgs::msg::PointCloud2 makeEmptyCloud(
  const builtin_interfaces::msg::Time &stamp)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = "world";
  cloud.header.stamp = stamp;
  cloud.height = 1U;
  cloud.width = 0U;
  cloud.is_bigendian = false;
  cloud.point_step = 12U;
  cloud.row_step = 0U;
  cloud.is_dense = true;
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
  };
  return cloud;
}

class ControllerStatusTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      const std::filesystem::path log_directory =
        "/tmp/scan_controller_status_test_logs";
      std::filesystem::create_directories(log_directory);
      setenv("ROS_LOG_DIR", log_directory.c_str(), 1);
      int argc = 0;
      char **argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  static void TearDownTestSuite()
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
      rclcpp::Parameter("use_sim_time", true),
      rclcpp::Parameter("controller.publish_rate_hz", 50.0),
      rclcpp::Parameter("topics.bspline", "/test/controller_status/bspline"),
      rclcpp::Parameter("topics.initial_path", "/test/controller_status/path"),
      rclcpp::Parameter("topics.body_pose", "/test/controller_status/odom"),
      rclcpp::Parameter("topics.cloud", "/test/controller_status/cloud"),
      rclcpp::Parameter("topics.cmd_vel", "/test/controller_status/cmd_vel"),
      rclcpp::Parameter(
        "topics.execution_frozen", "/test/controller_status/frozen"),
      rclcpp::Parameter(
        "topics.goal_reached", "/test/controller_status/goal_reached"),
      rclcpp::Parameter(
        "topics.trajectory_finished", "/test/controller_status/finished"),
      rclcpp::Parameter(
        "topics.controller_status", "/test/controller_status/status"),
    });
    controller_ = scan_controller::makeScanControllerNode(options);
    peer_ = std::make_shared<rclcpp::Node>("controller_status_test_peer");

    status_sub_ = peer_->create_subscription<Status>(
      "/test/controller_status/status",
      rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
      [this](const Status::ConstSharedPtr message) {
        statuses_.push_back(*message);
      });
    cmd_vel_sub_ = peer_->create_subscription<geometry_msgs::msg::Twist>(
      "/test/controller_status/cmd_vel", rclcpp::QoS(10).reliable(),
      [this](const geometry_msgs::msg::Twist::ConstSharedPtr message) {
        commands_.push_back(*message);
      });
    clock_pub_ = peer_->create_publisher<rosgraph_msgs::msg::Clock>(
      "/clock", rclcpp::ClockQoS());
    path_pub_ = peer_->create_publisher<nav_msgs::msg::Path>(
      "/test/controller_status/path",
      rclcpp::QoS(1).reliable().transient_local());
    bspline_pub_ = peer_->create_publisher<scan_planner_msgs::msg::Bspline>(
      "/test/controller_status/bspline",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    odom_pub_ = peer_->create_publisher<nav_msgs::msg::Odometry>(
      "/test/controller_status/odom", rclcpp::SensorDataQoS());
    cloud_pub_ = peer_->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/test/controller_status/cloud", rclcpp::SensorDataQoS());

    executor_.add_node(controller_);
    executor_.add_node(peer_);
    spinFor(100ms);
  }

  void TearDown() override
  {
    executor_.remove_node(peer_);
    executor_.remove_node(controller_);
    status_sub_.reset();
    cmd_vel_sub_.reset();
    clock_pub_.reset();
    path_pub_.reset();
    bspline_pub_.reset();
    odom_pub_.reset();
    cloud_pub_.reset();
    peer_.reset();
    controller_.reset();
  }

  void spinFor(std::chrono::milliseconds duration)
  {
    const auto deadline = std::chrono::steady_clock::now() + duration;
    while (std::chrono::steady_clock::now() < deadline) {
      executor_.spin_some();
      std::this_thread::sleep_for(2ms);
    }
    executor_.spin_some();
  }

  bool spinUntil(
    const std::function<bool()> &predicate,
    std::chrono::milliseconds timeout = 2000ms)
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      executor_.spin_some();
      if (predicate()) {
        return true;
      }
      std::this_thread::sleep_for(2ms);
    }
    executor_.spin_some();
    return predicate();
  }

  bool setClock(const builtin_interfaces::msg::Time &stamp)
  {
    rosgraph_msgs::msg::Clock message;
    message.clock = stamp;
    const std::int64_t expected =
      static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
      static_cast<std::int64_t>(stamp.nanosec);
    for (int attempt = 0; attempt < 5; ++attempt) {
      clock_pub_->publish(message);
      if (spinUntil(
          [this, expected]() {
            return controller_->get_clock()->now().nanoseconds() == expected;
          }, 300ms))
      {
        return true;
      }
    }
    return false;
  }

  const Status *findStatus(const std::function<bool(const Status &)> &predicate)
  {
    for (
      auto iterator = statuses_.rbegin();
      iterator != statuses_.rend(); ++iterator)
    {
      if (predicate(*iterator)) {
        return &*iterator;
      }
    }
    return nullptr;
  }

  rclcpp::executors::SingleThreadedExecutor executor_;
  std::shared_ptr<rclcpp::Node> controller_;
  rclcpp::Node::SharedPtr peer_;
  std::vector<Status> statuses_;
  std::vector<geometry_msgs::msg::Twist> commands_;
  rclcpp::Subscription<Status>::SharedPtr status_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<scan_planner_msgs::msg::Bspline>::SharedPtr bspline_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

TEST_F(ControllerStatusTest, PublishesAcceptedStateInvalidatedAndRejectedEvents)
{
  const auto path_stamp = makeStamp(10, 100000001U);
  ASSERT_TRUE(setClock(path_stamp));

  const auto path = makePath(path_stamp);
  path_pub_->publish(path);
  spinFor(100ms);

  const auto trajectory_stamp = makeStamp(10, 100000002U);
  ASSERT_TRUE(setClock(trajectory_stamp));
  const auto trajectory = makeBspline(7, path_stamp, trajectory_stamp);
  bspline_pub_->publish(trajectory);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 7 && status.accepted &&
             status.trajectory_valid &&
             status.event == Status::EVENT_ACCEPTED;
    }) != nullptr;
  }));
  const Status *accepted = findStatus([](const Status &status) {
    return status.traj_id == 7 && status.accepted &&
           status.trajectory_valid && status.event == Status::EVENT_ACCEPTED;
  });
  ASSERT_NE(accepted, nullptr);
  EXPECT_EQ(accepted->header.frame_id, "world");
  EXPECT_GT(accepted->header.stamp.sec, 0);
  EXPECT_EQ(accepted->reference_path_stamp, path_stamp);
  EXPECT_EQ(accepted->bspline_header_stamp, trajectory_stamp);
  EXPECT_EQ(accepted->start_time, trajectory_stamp);
  EXPECT_EQ(accepted->acceptance_sequence, 1U);
  EXPECT_FALSE(accepted->candidate_present);
  EXPECT_FALSE(accepted->is_final);
  EXPECT_FALSE(accepted->emergency_stop);
  const std::uint64_t accepted_sequence = accepted->status_sequence;
  const std::uint64_t acceptance_sequence = accepted->acceptance_sequence;

  bspline_pub_->publish(trajectory);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 7 && status.accepted &&
             status.trajectory_valid &&
             status.event == Status::EVENT_DUPLICATE;
    }) != nullptr;
  }));
  const Status *duplicate = findStatus([](const Status &status) {
    return status.traj_id == 7 && status.event == Status::EVENT_DUPLICATE;
  });
  ASSERT_NE(duplicate, nullptr);
  EXPECT_EQ(duplicate->acceptance_sequence, acceptance_sequence);
  EXPECT_GT(duplicate->status_sequence, accepted_sequence);

  const auto older_trajectory = makeBspline(6, path_stamp, trajectory_stamp);
  bspline_pub_->publish(older_trajectory);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 7 && status.accepted &&
             status.trajectory_valid && status.event == Status::EVENT_REJECTED &&
             status.candidate_present && status.candidate_traj_id == 6;
    }) != nullptr;
  }));
  const Status *preserved_after_reject = findStatus([](const Status &status) {
    return status.traj_id == 7 && status.event == Status::EVENT_REJECTED &&
           status.candidate_traj_id == 6;
  });
  ASSERT_NE(preserved_after_reject, nullptr);
  EXPECT_EQ(preserved_after_reject->acceptance_sequence, acceptance_sequence);
  EXPECT_EQ(preserved_after_reject->reference_path_stamp, path_stamp);

  ASSERT_TRUE(setClock(makeStamp(10, 150000000U)));
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 7 && status.accepted &&
             status.trajectory_valid &&
             status.state == Status::STATE_WAITING_FOR_ODOMETRY;
    }) != nullptr;
  }));

  const auto sensor_stamp = makeStamp(10, 160000000U);
  ASSERT_TRUE(setClock(sensor_stamp));
  odom_pub_->publish(makeOdometry(sensor_stamp));
  cloud_pub_->publish(makeEmptyCloud(sensor_stamp));
  spinFor(100ms);
  ASSERT_TRUE(setClock(makeStamp(10, 180000000U)));
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 7 && status.accepted &&
             status.trajectory_valid && status.state == Status::STATE_TRACKING;
    }) != nullptr;
  }));
  const Status *tracking = findStatus([](const Status &status) {
    return status.traj_id == 7 && status.accepted &&
           status.trajectory_valid && status.state == Status::STATE_TRACKING;
  });
  ASSERT_NE(tracking, nullptr);
  EXPECT_GT(tracking->status_sequence, accepted_sequence);
  const std::uint64_t tracking_sequence = tracking->status_sequence;

  const auto next_path_stamp = makeStamp(10, 190000000U);
  ASSERT_TRUE(setClock(next_path_stamp));
  path_pub_->publish(makePath(next_path_stamp, true));
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 7 && status.accepted &&
             !status.trajectory_valid &&
             status.event == Status::EVENT_INVALIDATED &&
             status.reason.find("参考 Path") != std::string::npos;
    }) != nullptr;
  }));
  const Status *invalidated = findStatus([](const Status &status) {
    return status.traj_id == 7 && status.accepted &&
           !status.trajectory_valid &&
           status.event == Status::EVENT_INVALIDATED &&
           status.reason.find("参考 Path") != std::string::npos;
  });
  ASSERT_NE(invalidated, nullptr);
  EXPECT_GT(invalidated->status_sequence, tracking_sequence);
  EXPECT_EQ(invalidated->reference_path_stamp, path_stamp);
  const std::uint64_t invalidated_sequence = invalidated->status_sequence;

  const auto rejected_stamp = makeStamp(10, 200000000U);
  ASSERT_TRUE(setClock(rejected_stamp));
  auto rejected_trajectory = makeBspline(8, next_path_stamp, rejected_stamp);
  rejected_trajectory.header.frame_id = "map";
  bspline_pub_->publish(rejected_trajectory);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 7 && status.accepted &&
             !status.trajectory_valid &&
             status.event == Status::EVENT_REJECTED &&
             status.candidate_present && status.candidate_traj_id == 8 &&
             status.reason.find("frame_id") != std::string::npos;
    }) != nullptr;
  }));
  const Status *rejected = findStatus([](const Status &status) {
    return status.traj_id == 7 && status.accepted &&
           !status.trajectory_valid &&
           status.event == Status::EVENT_REJECTED &&
           status.candidate_present && status.candidate_traj_id == 8 &&
           status.reason.find("frame_id") != std::string::npos;
  });
  ASSERT_NE(rejected, nullptr);
  EXPECT_GT(rejected->status_sequence, invalidated_sequence);
  EXPECT_EQ(rejected->reference_path_stamp, path_stamp);
  EXPECT_EQ(rejected->candidate_reference_path_stamp, next_path_stamp);
  EXPECT_EQ(rejected->candidate_bspline_header_stamp, rejected_stamp);
  EXPECT_EQ(rejected->candidate_start_time, rejected_stamp);
  EXPECT_EQ(rejected->acceptance_sequence, acceptance_sequence);
  const std::uint64_t rejected_sequence = rejected->status_sequence;

  std::vector<Status> late_statuses;
  auto late_subscription = peer_->create_subscription<Status>(
    "/test/controller_status/status",
    rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
    [&late_statuses](const Status::ConstSharedPtr message) {
      late_statuses.push_back(*message);
    });
  ASSERT_TRUE(spinUntil([&late_statuses, rejected_sequence]() {
    return std::any_of(
      late_statuses.begin(), late_statuses.end(),
      [rejected_sequence](const Status &status) {
        return status.status_sequence == rejected_sequence;
      });
  }));
  const auto late_rejected = std::find_if(
    late_statuses.begin(), late_statuses.end(),
    [rejected_sequence](const Status &status) {
      return status.status_sequence == rejected_sequence;
    });
  ASSERT_NE(late_rejected, late_statuses.end());
  EXPECT_EQ(late_rejected->traj_id, 7);
  EXPECT_EQ(late_rejected->candidate_traj_id, 8);
}

TEST_F(ControllerStatusTest, RejectsControllerStatusHistoryDepthBelowEvidenceContract)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("use_sim_time", true),
    rclcpp::Parameter("qos.controller_status_depth", 1),
  });
  EXPECT_THROW(
    scan_controller::makeScanControllerNode(options),
    std::runtime_error);
}

TEST_F(ControllerStatusTest, StationaryFinalUsesBodyYawRateFromOdometry)
{
  const auto path_stamp = makeStamp(20, 100000000U);
  ASSERT_TRUE(setClock(path_stamp));
  path_pub_->publish(makePath(path_stamp));
  spinFor(100ms);

  const auto trajectory_stamp = makeStamp(20, 100000001U);
  ASSERT_TRUE(setClock(trajectory_stamp));
  auto hold = makeBspline(17, path_stamp, trajectory_stamp);
  hold.is_final = true;
  for (auto &point : hold.pos_pts) {
    point.x = 1.0;
    point.y = 0.0;
    point.z = 0.30;
  }
  bspline_pub_->publish(hold);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 17 && status.accepted &&
             status.event == Status::EVENT_ACCEPTED;
    }) != nullptr;
  }));

  // bridge 发布的是 base_link 机体系 twist。wx/wy 模拟四足静止支撑微摆，
  // wz 低于导航完成门；旧实现取三轴范数会永远无法累计 3 秒驻留。
  for (int step = 1; step <= 35; ++step) {
    const std::int64_t nanoseconds =
      20000000000LL + 100000000LL +
      static_cast<std::int64_t>(step) * 100000000LL;
    const auto stamp = makeStamp(
      static_cast<std::int32_t>(nanoseconds / 1000000000LL),
      static_cast<std::uint32_t>(nanoseconds % 1000000000LL));
    ASSERT_TRUE(setClock(stamp));
    auto odometry = makeOdometry(stamp);
    odometry.pose.pose.position.x = 1.0;
    odometry.twist.twist.angular.x = 0.40;
    odometry.twist.twist.angular.y = -0.30;
    odometry.twist.twist.angular.z = 0.05;
    odom_pub_->publish(odometry);
    cloud_pub_->publish(makeEmptyCloud(stamp));
    spinFor(20ms);
  }

  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 17 &&
             status.state == Status::STATE_GOAL_REACHED;
    }) != nullptr;
  }));
}

TEST_F(ControllerStatusTest, CopiesYawPayloadAndRejectsSameIdentityChange)
{
  const auto path_stamp = makeStamp(30, 100000000U);
  ASSERT_TRUE(setClock(path_stamp));
  path_pub_->publish(makePath(path_stamp));
  spinFor(100ms);

  const auto trajectory_stamp = makeStamp(30, 100000001U);
  ASSERT_TRUE(setClock(trajectory_stamp));
  odom_pub_->publish(makeOdometry(trajectory_stamp));
  cloud_pub_->publish(makeEmptyCloud(trajectory_stamp));
  spinFor(100ms);
  auto active_observation = makeBspline(
    27, path_stamp, trajectory_stamp);
  for (auto &point : active_observation.pos_pts) {
    point.x = 0.0;
    point.y = 0.0;
    point.z = 0.30;
  }
  active_observation.yaw_pts = {0.0, 0.20};
  active_observation.yaw_dt = 1.0;
  bspline_pub_->publish(active_observation);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 27 && status.accepted &&
             status.trajectory_valid &&
             status.event == Status::EVENT_ACCEPTED;
    }) != nullptr;
  }));
  const Status *accepted = findStatus([](const Status &status) {
    return status.traj_id == 27 && status.accepted &&
           status.trajectory_valid && status.event == Status::EVENT_ACCEPTED;
  });
  ASSERT_NE(accepted, nullptr);
  EXPECT_EQ(accepted->state, Status::STATE_ALIGNING_YAW);
  EXPECT_TRUE(accepted->active_sensing_yaw_only);
  EXPECT_EQ(accepted->command_sample_count, 1U);
  EXPECT_DOUBLE_EQ(accepted->first_command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(accepted->first_command.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(accepted->first_command.angular.z, 0.0);
  EXPECT_DOUBLE_EQ(accepted->max_abs_vx, 0.0);
  EXPECT_DOUBLE_EQ(accepted->max_abs_vy, 0.0);
  EXPECT_DOUBLE_EQ(accepted->max_abs_wz, 0.0);
  EXPECT_EQ(accepted->command_violation_count, 0U);
  ASSERT_FALSE(commands_.empty());
  const auto &first_active_command = commands_.back();
  EXPECT_DOUBLE_EQ(first_active_command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(first_active_command.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(first_active_command.angular.z, 0.0);

  active_observation.yaw_pts[1] = 0.19;
  bspline_pub_->publish(active_observation);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 27 && status.accepted &&
             !status.trajectory_valid &&
             status.event == Status::EVENT_REJECTED &&
             status.reason.find("同一 B-spline identity") !=
             std::string::npos;
    }) != nullptr;
  }));
  const Status *invalidated = findStatus([](const Status &status) {
    return status.traj_id == 27 && status.accepted &&
           !status.trajectory_valid &&
           status.event == Status::EVENT_INVALIDATED;
  });
  ASSERT_NE(invalidated, nullptr);
  EXPECT_TRUE(invalidated->active_sensing_yaw_only);
  EXPECT_GE(invalidated->command_sample_count, 2U);
  EXPECT_DOUBLE_EQ(invalidated->max_abs_vx, 0.0);
  EXPECT_DOUBLE_EQ(invalidated->max_abs_vy, 0.0);
  EXPECT_LE(invalidated->max_abs_wz, 0.20);
  EXPECT_EQ(invalidated->command_violation_count, 0U);
}

TEST_F(
  ControllerStatusTest,
  ActiveSensingAggregateSurvivesReplacementAndTransientHistory)
{
  const auto path_stamp = makeStamp(40, 100000000U);
  ASSERT_TRUE(setClock(path_stamp));
  path_pub_->publish(makePath(path_stamp));
  spinFor(100ms);

  const auto active_stamp = makeStamp(40, 100000001U);
  ASSERT_TRUE(setClock(active_stamp));
  odom_pub_->publish(makeOdometry(active_stamp));
  cloud_pub_->publish(makeEmptyCloud(active_stamp));
  spinFor(100ms);

  auto active_observation = makeBspline(37, path_stamp, active_stamp);
  for (auto &point : active_observation.pos_pts) {
    point.x = 0.0;
    point.y = 0.0;
    point.z = 0.30;
  }
  active_observation.yaw_pts = {0.0, 0.20};
  active_observation.yaw_dt = 1.0;
  const std::size_t active_command_begin = commands_.size();
  bspline_pub_->publish(active_observation);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 37 && status.accepted &&
             status.trajectory_valid &&
             status.event == Status::EVENT_ACCEPTED;
    }) != nullptr;
  }));
  const Status *active_accepted = findStatus([](const Status &status) {
    return status.traj_id == 37 && status.accepted &&
           status.trajectory_valid && status.event == Status::EVENT_ACCEPTED;
  });
  ASSERT_NE(active_accepted, nullptr);
  EXPECT_TRUE(active_accepted->active_sensing_yaw_only);
  EXPECT_EQ(active_accepted->command_sample_count, 1U);
  EXPECT_DOUBLE_EQ(active_accepted->first_command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(active_accepted->first_command.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(active_accepted->first_command.angular.z, 0.0);
  EXPECT_EQ(active_accepted->command_violation_count, 0U);

  for (std::uint32_t nanoseconds : {
      150000000U, 200000000U, 250000000U, 300000000U})
  {
    const auto sensor_stamp = makeStamp(40, nanoseconds);
    ASSERT_TRUE(setClock(sensor_stamp));
    odom_pub_->publish(makeOdometry(sensor_stamp));
    cloud_pub_->publish(makeEmptyCloud(sensor_stamp));
    spinFor(40ms);
  }
  const auto replacement_stamp = makeStamp(40, 300000001U);
  ASSERT_TRUE(setClock(replacement_stamp));
  const std::size_t active_command_end = commands_.size();
  ASSERT_GT(active_command_end, active_command_begin + 1U);
  for (
    std::size_t index = active_command_begin;
    index < active_command_end; ++index)
  {
    EXPECT_DOUBLE_EQ(commands_[index].linear.x, 0.0);
    EXPECT_DOUBLE_EQ(commands_[index].linear.y, 0.0);
    EXPECT_LE(std::abs(commands_[index].angular.z), 0.20 + 1.0e-12);
  }
  EXPECT_DOUBLE_EQ(commands_[active_command_begin].angular.z, 0.0);

  const auto replacement = makeBspline(38, path_stamp, replacement_stamp);
  bspline_pub_->publish(replacement);
  ASSERT_TRUE(spinUntil([this]() {
    return findStatus([](const Status &status) {
      return status.traj_id == 38 && status.accepted &&
             status.trajectory_valid &&
             status.event == Status::EVENT_ACCEPTED;
    }) != nullptr;
  }));

  const Status *active_final = findStatus([](const Status &status) {
    return status.traj_id == 37 && status.accepted &&
           status.event == Status::EVENT_STATE_CHANGED &&
           status.reason.find("替换前") != std::string::npos;
  });
  const Status *replacement_accepted = findStatus([](const Status &status) {
    return status.traj_id == 38 && status.accepted &&
           status.event == Status::EVENT_ACCEPTED;
  });
  ASSERT_NE(active_final, nullptr);
  ASSERT_NE(replacement_accepted, nullptr);
  EXPECT_LT(
    active_final->status_sequence,
    replacement_accepted->status_sequence);
  EXPECT_TRUE(active_final->active_sensing_yaw_only);
  EXPECT_EQ(
    active_final->command_sample_count,
    active_command_end - active_command_begin);
  EXPECT_DOUBLE_EQ(active_final->first_command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(active_final->first_command.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(active_final->first_command.angular.z, 0.0);
  EXPECT_DOUBLE_EQ(active_final->max_abs_vx, 0.0);
  EXPECT_DOUBLE_EQ(active_final->max_abs_vy, 0.0);
  EXPECT_GT(active_final->max_abs_wz, 0.0);
  EXPECT_LE(active_final->max_abs_wz, 0.20 + 1.0e-12);
  EXPECT_EQ(active_final->command_violation_count, 0U);
  EXPECT_FALSE(replacement_accepted->active_sensing_yaw_only);
  EXPECT_EQ(replacement_accepted->command_sample_count, 0U);
  EXPECT_EQ(replacement_accepted->command_violation_count, 0U);

  // 两个事件在同一 B-spline 回调内连续发布。late joiner 仍必须从生产
  // KeepLast(64) transient history 同时取回旧 active 终态与新 accepted，
  // 否则 live 验收无法证明替换边界。
  std::vector<Status> late_statuses;
  auto late_subscription = peer_->create_subscription<Status>(
    "/test/controller_status/status",
    rclcpp::QoS(rclcpp::KeepLast(64)).reliable().transient_local(),
    [&late_statuses](const Status::ConstSharedPtr message) {
      late_statuses.push_back(*message);
    });
  ASSERT_TRUE(spinUntil([&late_statuses, active_final, replacement_accepted]() {
    bool found_active_final = false;
    bool found_replacement = false;
    for (const auto &status : late_statuses) {
      found_active_final = found_active_final ||
        status.status_sequence == active_final->status_sequence;
      found_replacement = found_replacement ||
        status.status_sequence == replacement_accepted->status_sequence;
    }
    return found_active_final && found_replacement;
  }));
}

}  // 匿名命名空间
