#include <cmath>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "scan_controller/trajectory_tracker.hpp"

namespace
{

scan_controller::TrackerConfig makeConfig()
{
  scan_controller::TrackerConfig config;
  config.bspline_timeout_sec = 10.0;
  config.odom_timeout_sec = 10.0;
  config.cloud_timeout_sec = 10.0;
  return config;
}

scan_controller::TrajectoryInput makeStraightTrajectory(
  double stamp_sec, bool is_final = false)
{
  scan_controller::TrajectoryInput input;
  input.order = 3;
  input.trajectory_id = 7;
  input.header_stamp_sec = stamp_sec;
  input.start_stamp_sec = stamp_sec;
  input.reference_path_stamp_sec = stamp_sec;
  input.is_final = is_final;
  input.control_points.resize(3, 6);
  input.control_points <<
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3;
  input.knots.resize(10);
  input.knots <<
    -3.0, -2.0, -1.0, 0.0, 1.0,
    2.0, 3.0, 4.0, 5.0, 6.0;
  return input;
}

scan_controller::TrajectoryInput makeStationaryTrajectory(
  double header_stamp_sec, double start_stamp_sec, bool is_final)
{
  auto input = makeStraightTrajectory(header_stamp_sec, is_final);
  input.start_stamp_sec = start_stamp_sec;
  input.control_points.setZero();
  input.control_points.row(2).setConstant(0.3);
  return input;
}

scan_controller::TrajectoryInput makeYawOnlyTrajectory(
  double stamp_sec, double start_yaw, double target_yaw,
  double yaw_dt = 1.0)
{
  auto input = makeStationaryTrajectory(stamp_sec, stamp_sec, false);
  input.yaw_points = {start_yaw, target_yaw};
  input.yaw_dt = yaw_dt;
  return input;
}

scan_controller::TrajectoryInput makeShortReverseChordTrajectory(
  double stamp_sec)
{
  auto input = makeStraightTrajectory(stamp_sec);
  input.control_points <<
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.01, 0.0, -0.01, 0.10, 0.30, 0.50,
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3;
  return input;
}

scan_controller::TrajectoryInput makeSteepSlowTrajectory(
  double stamp_sec, bool reverse = false)
{
  auto input = makeStraightTrajectory(stamp_sec);
  if (reverse) {
    input.control_points.row(0) <<
      0.05, 0.04, 0.03, 0.02, 0.01, 0.00;
  } else {
    input.control_points.row(0) <<
      0.00, 0.01, 0.02, 0.03, 0.04, 0.05;
  }
  input.control_points.row(1).setZero();
  input.control_points.row(2) <<
    0.300, 0.395, 0.490, 0.585, 0.680, 0.775;
  return input;
}

void setStairReferencePath(
  scan_controller::TrajectoryTracker & tracker, double stamp_sec)
{
  const std::vector<Eigen::Vector3d> points{
    Eigen::Vector3d(-0.30, 0.0, 0.00),
    Eigen::Vector3d(0.00, 0.0, 0.00),
    Eigen::Vector3d(0.02, 0.0, 0.19),
    Eigen::Vector3d(0.32, 0.0, 0.19),
    Eigen::Vector3d(0.34, 0.0, 0.38),
    Eigen::Vector3d(0.64, 0.0, 0.38),
  };
  std::string error;
  ASSERT_TRUE(
    tracker.setReferencePath(points, stamp_sec, stamp_sec, error)) << error;
}

Eigen::Vector3d trajectoryFinalPosition(
  const scan_controller::TrajectoryInput & trajectory)
{
  scan_planner::UniformBspline spline(
    trajectory.control_points, trajectory.order, 1.0);
  spline.setKnot(trajectory.knots);
  const double duration =
    trajectory.knots(trajectory.control_points.cols()) -
    trajectory.knots(trajectory.order);
  return spline.evaluateDeBoorT(duration);
}

Eigen::Vector3d trajectoryPositionAt(
  const scan_controller::TrajectoryInput & trajectory, double time_sec)
{
  scan_planner::UniformBspline spline(
    trajectory.control_points, trajectory.order, 1.0);
  spline.setKnot(trajectory.knots);
  return spline.evaluateDeBoorT(time_sec);
}

void setReferencePath(
  scan_controller::TrajectoryTracker & tracker, double stamp_sec,
  double y = 0.0)
{
  const std::vector<Eigen::Vector3d> points{
    Eigen::Vector3d(-0.5, y, 0.0),
    Eigen::Vector3d(0.5, y, 0.0),
    Eigen::Vector3d(1.5, y, 0.2),
    Eigen::Vector3d(2.5, y, 0.2),
  };
  std::string error;
  ASSERT_TRUE(
    tracker.setReferencePath(points, stamp_sec, stamp_sec, error)) << error;
}

void setFinalReferencePath(
  scan_controller::TrajectoryTracker & tracker,
  double stamp_sec,
  const Eigen::Vector3d & base_goal,
  double terminal_yaw = 0.0,
  double body_height = 0.30)
{
  Eigen::Vector3d ground_goal = base_goal;
  ground_goal.z() -= body_height;
  const Eigen::Vector3d tangent(
    std::cos(terminal_yaw), std::sin(terminal_yaw), 0.0);
  const std::vector<Eigen::Vector3d> points{
    ground_goal - tangent,
    ground_goal - 0.5 * tangent,
    ground_goal,
  };
  std::string error;
  ASSERT_TRUE(
    tracker.setReferencePath(
      points, stamp_sec, stamp_sec, error, terminal_yaw)) << error;
}

void setFreshInputs(
  scan_controller::TrajectoryTracker & tracker, double stamp_sec,
  double yaw = 0.0, const Eigen::Vector3d & position =
  Eigen::Vector3d(0.0, 0.0, 0.3), double planar_speed = 0.0)
{
  if (!tracker.hasReferencePath()) {
    setReferencePath(tracker, stamp_sec);
  }
  scan_controller::OdometryInput odometry;
  odometry.position = position;
  odometry.yaw = yaw;
  odometry.planar_speed = planar_speed;
  odometry.stamp_sec = stamp_sec;
  std::string error;
  ASSERT_TRUE(tracker.setOdometry(odometry, stamp_sec, error)) << error;
  ASSERT_TRUE(
    tracker.setCloudObservation(stamp_sec, stamp_sec, error)) << error;
}

scan_controller::ControlOutput waitForMovingFinalStableCompletion(
  scan_controller::TrajectoryTracker & tracker, double start_stamp_sec,
  double yaw, const Eigen::Vector3d & position)
{
  scan_controller::ControlOutput output;
  // 默认配置要求 0.06s 严格制动和 0.50s 连续物理稳定；以 20Hz 更新最多
  // 1 秒，既覆盖完整握手，也让调用测试保留明确的超时上界。
  for (int step = 0; step <= 20; ++step) {
    const double now =
      start_stamp_sec + 0.05 * static_cast<double>(step);
    setFreshInputs(tracker, now, yaw, position);
    output = tracker.update(now);
    if (output.goal_reached) {
      break;
    }
  }
  return output;
}

TEST(TrajectoryTracker, RejectsInvalidConfiguration)
{
  auto config = makeConfig();
  config.max_vx = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.cross_track_heading_error_threshold =
    config.heading_error_threshold + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.heading_error_release_threshold =
    config.heading_error_threshold;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.cross_track_heading_error_release_threshold =
    config.cross_track_heading_error_threshold + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.cross_track_recovery_forward_speed = config.max_vx + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.cross_track_heading_assist_max =
    config.cross_track_heading_error_threshold;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.turning_yaw_rate_threshold = config.max_yaw_rate + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.turning_max_planar_speed =
    std::hypot(config.max_vx, config.max_vy) + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.cross_track_recovery_lateral_speed = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.reference_path_max_points = 1;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.stair_heading_lock_half_window_arc = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.stair_heading_lock_min_pitch_rad = M_PI_2;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.stair_forward_speed_floor = -0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.stair_heading_lock_enabled = true;
  config.stair_forward_speed_floor = config.max_vx + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.stair_forward_speed_floor = 0.20;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.yaw_alignment_min_chord_distance = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_capture_entry_distance_xy = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_capture_entry_distance_xy = config.finish_distance_xy;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_capture_zero_hold_distance_xy =
    config.terminal_capture_entry_distance_xy;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_capture_zero_hold_distance_xy = config.finish_distance_xy;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_capture_release_distance_xy = config.finish_distance_xy;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_max_yaw_rate = config.max_yaw_rate + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_max_yaw_acc = config.max_yaw_acc + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_capture_brake_hold_sec = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_capture_stable_dwell_sec = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_position_hold_gain = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_position_hold_max_speed = config.max_vy + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_approach_min_speed =
    std::hypot(config.max_vx, config.max_vy) + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.finish_yaw_rate = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.terminal_yaw_control_deadband = config.finish_yaw_error;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.active_sensing_max_yaw_excursion = 0.0;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.active_sensing_max_yaw_rate = config.max_yaw_rate + 0.01;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.active_sensing_max_yaw_excursion = 0.221;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.active_sensing_max_yaw_rate = 0.201;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);

  config = makeConfig();
  config.active_sensing_yaw_tolerance =
    config.active_sensing_max_yaw_excursion;
  EXPECT_THROW(
    scan_controller::TrajectoryTracker tracker(config),
    std::invalid_argument);
}

TEST(TrajectoryTracker, YawOnlyContractFailsClosedForInvalidPayloads)
{
  const auto expect_rejected = [](scan_controller::TrajectoryInput input) {
      scan_controller::TrajectoryTracker tracker(makeConfig());
      setReferencePath(tracker, 10.0);
      setFreshInputs(tracker, 10.0, 0.0);
      std::string error;
      EXPECT_FALSE(tracker.setTrajectory(input, 10.0, error));
      EXPECT_FALSE(tracker.hasTrajectory());
      EXPECT_FALSE(error.empty());
    };

  auto wrong_count = makeYawOnlyTrajectory(10.0, 0.0, 0.20);
  wrong_count.yaw_points = {0.0};
  expect_rejected(wrong_count);

  auto nonfinite = makeYawOnlyTrajectory(10.0, 0.0, 0.20);
  nonfinite.yaw_points[1] = std::numeric_limits<double>::infinity();
  expect_rejected(nonfinite);

  auto invalid_dt = makeYawOnlyTrajectory(10.0, 0.0, 0.20);
  invalid_dt.yaw_dt = 0.0;
  expect_rejected(invalid_dt);

  auto moving_position = makeYawOnlyTrajectory(10.0, 0.0, 0.20);
  moving_position.control_points(0, 5) = 1.0e-12;
  expect_rejected(moving_position);

  auto final_request = makeYawOnlyTrajectory(10.0, 0.0, 0.20);
  final_request.is_final = true;
  expect_rejected(final_request);

  auto emergency_request = makeYawOnlyTrajectory(10.0, 0.0, 0.20);
  emergency_request.emergency_stop = true;
  expect_rejected(emergency_request);

  expect_rejected(makeYawOnlyTrajectory(10.0, 0.0, 0.221, 2.0));
  expect_rejected(makeYawOnlyTrajectory(10.0, 0.0, 0.20, 0.99));
}

TEST(TrajectoryTracker, EmptyYawPointsRetainLegacyTrajectoryBehavior)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  auto trajectory = makeStraightTrajectory(10.0);
  trajectory.yaw_dt = 0.0;
  std::string error;

  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(tracker, 10.0);
  const auto output = tracker.update(10.1);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_GT(output.vx, 0.0);
}

TEST(TrajectoryTracker, YawOnlyStartMustMatchFreshWrappedOdometry)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  setFreshInputs(tracker, 10.0, 0.0);
  std::string error;

  EXPECT_FALSE(
    tracker.setTrajectory(
      makeYawOnlyTrajectory(10.0, 3.0, 3.2), 10.0, error));
  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_NE(error.find("新鲜 Odometry"), std::string::npos);

  scan_controller::TrajectoryTracker wrapped_tracker(makeConfig());
  setReferencePath(wrapped_tracker, 20.0);
  setFreshInputs(wrapped_tracker, 20.0, -M_PI + 0.01);
  ASSERT_TRUE(
    wrapped_tracker.setTrajectory(
      makeYawOnlyTrajectory(
        20.0, M_PI + 0.01, M_PI + 0.20, 1.0),
      20.0, error)) << error;
  EXPECT_TRUE(wrapped_tracker.hasTrajectory());
}

TEST(TrajectoryTracker, YawOnlyTargetMustStayWithinActualOdometryExcursion)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  setFreshInputs(tracker, 10.0, -0.019);
  std::string error;

  EXPECT_FALSE(
    tracker.setTrajectory(
      makeYawOnlyTrajectory(10.0, 0.0, 0.22, 1.1), 10.0, error));
  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_NE(error.find("安全转角"), std::string::npos);
}

TEST(TrajectoryTracker, RequiredReferencePathAbsenceForcesZeroVelocity)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  scan_controller::OdometryInput odometry;
  odometry.position = Eigen::Vector3d(0.0, 0.0, 0.3);
  odometry.stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.0, error)) << error;
  ASSERT_TRUE(
    tracker.setCloudObservation(10.0, 10.0, error)) << error;

  const auto output = tracker.update(10.1);

  EXPECT_EQ(
    output.state,
    scan_controller::ControllerState::kWaitingForReferencePath);
  EXPECT_TRUE(output.execution_frozen);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
}

TEST(TrajectoryTracker, RejectsDegenerateReferencePath)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  const std::vector<Eigen::Vector3d> points{
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 1.0),
  };
  std::string error;

  EXPECT_FALSE(
    tracker.setReferencePath(points, 10.0, 10.0, error));
  EXPECT_FALSE(error.empty());
}

TEST(TrajectoryTracker, RejectsReferencePathAboveConfiguredPointLimit)
{
  auto config = makeConfig();
  config.reference_path_max_points = 3;
  scan_controller::TrajectoryTracker tracker(config);
  const std::vector<Eigen::Vector3d> points{
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(0.5, 0.0, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0),
    Eigen::Vector3d(1.5, 0.0, 0.0),
  };
  std::string error;

  EXPECT_FALSE(
    tracker.setReferencePath(points, 10.0, 10.0, error));
  EXPECT_NE(error.find("点数上限"), std::string::npos);
  EXPECT_FALSE(tracker.hasReferencePath());
}

TEST(TrajectoryTracker, RejectsNonFiniteControlPoint)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  auto trajectory = makeStraightTrajectory(10.0);
  trajectory.control_points(0, 2) =
    std::numeric_limits<double>::quiet_NaN();
  std::string error;
  EXPECT_FALSE(tracker.setTrajectory(trajectory, 10.0, error));
  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_FALSE(error.empty());
}

TEST(TrajectoryTracker, RejectsInvalidKnotCount)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  auto trajectory = makeStraightTrajectory(10.0);
  trajectory.knots.conservativeResize(9);
  std::string error;
  EXPECT_FALSE(tracker.setTrajectory(trajectory, 10.0, error));
  EXPECT_FALSE(error.empty());
}

TEST(TrajectoryTracker, MissingCloudForcesZeroVelocity)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  scan_controller::OdometryInput odometry;
  odometry.position = Eigen::Vector3d(0.0, 0.0, 0.3);
  odometry.stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.0, error)) << error;
  setReferencePath(tracker, 10.0);

  const auto output = tracker.update(10.1);
  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kWaitingForCloud);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
  EXPECT_TRUE(output.execution_frozen);
}

TEST(TrajectoryTracker, YawAlignmentFreezesTrajectoryTime)
{
  auto config = makeConfig();
  config.max_yaw_acc = 1.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setFreshInputs(tracker, 10.0, M_PI);
  const double before = tracker.executionTimeSec();

  const auto output = tracker.update(10.1);
  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(output.execution_frozen);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_NE(output.wz, 0.0);
  EXPECT_LE(std::abs(output.wz), 0.100001);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), before);
}

TEST(TrajectoryTracker, YawAlignmentUsesEnterReleaseHysteresis)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.heading_error_release_threshold = 0.55;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;

  setFreshInputs(tracker, 10.0, -0.71);
  const auto entered = tracker.update(10.1);
  EXPECT_EQ(
    entered.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(entered.execution_frozen);
  EXPECT_DOUBLE_EQ(entered.vx, 0.0);
  EXPECT_DOUBLE_EQ(entered.vy, 0.0);

  setFreshInputs(tracker, 10.2, -0.69);
  const auto below_enter = tracker.update(10.2);
  EXPECT_EQ(
    below_enter.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(below_enter.execution_frozen);

  setFreshInputs(tracker, 10.3, -0.56);
  const auto above_release = tracker.update(10.3);
  EXPECT_EQ(
    above_release.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(above_release.execution_frozen);

  const double execution_time_before_release = tracker.executionTimeSec();
  setFreshInputs(tracker, 10.4, -0.54);
  const auto released = tracker.update(10.4);
  EXPECT_EQ(released.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(released.execution_frozen);
  EXPECT_GT(tracker.executionTimeSec(), execution_time_before_release);
}

TEST(TrajectoryTracker, FrozenYawTargetUsesSplineChordDespiteOdometryDrift)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.heading_error_release_threshold = 0.55;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;

  setFreshInputs(tracker, 10.0, M_PI);
  const auto entered = tracker.update(10.1);
  ASSERT_EQ(
    entered.state, scan_controller::ControllerState::kAligningYaw);
  ASSERT_LT(entered.wz, 0.0);
  const double frozen_time = tracker.executionTimeSec();

  // 模拟 v3 中 root 在原地转向时漂过近距离前视点。实际位置已经越过
  // 前视点，但 B-spline 的有向弦仍指向 +X，航向目标不能翻转到 -X。
  setFreshInputs(
    tracker, 10.2, M_PI - 0.1, Eigen::Vector3d(2.0, 0.0, 0.3));
  const auto drifted = tracker.update(10.2);

  EXPECT_EQ(
    drifted.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(drifted.execution_frozen);
  EXPECT_LT(drifted.wz, 0.0);
  EXPECT_DOUBLE_EQ(drifted.vx, 0.0);
  EXPECT_DOUBLE_EQ(drifted.vy, 0.0);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), frozen_time);
}

TEST(TrajectoryTracker, AlignedSplineChordDoesNotFreezeOnDynamicLookaheadBearing)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.heading_error_release_threshold = 0.55;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  const auto trajectory = makeStraightTrajectory(10.0);
  const Eigen::Vector3d spline_start = trajectoryPositionAt(trajectory, 0.0);
  const Eigen::Vector3d lookahead =
    trajectoryPositionAt(trajectory, config.time_forward);
  const Eigen::Vector3d chord = lookahead - spline_start;
  const Eigen::Vector3d odom_position =
    lookahead + Eigen::Vector3d(0.0, -0.02, 0.0);
  const Eigen::Vector3d dynamic_los = lookahead - odom_position;
  ASSERT_NEAR(std::atan2(chord.y(), chord.x()), 0.0, 1.0e-12);
  ASSERT_GT(
    std::abs(std::atan2(dynamic_los.y(), dynamic_los.x())),
    config.heading_error_threshold);

  std::string error;
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(tracker, 10.0, 0.0, odom_position);
  const double execution_time_before = tracker.executionTimeSec();

  const auto output = tracker.update(10.1);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.execution_frozen);
  EXPECT_GT(tracker.executionTimeSec(), execution_time_before);
}

TEST(TrajectoryTracker, OdometryBearingCannotReleaseActiveChordAlignment)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.heading_error_release_threshold = 0.55;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  const auto trajectory = makeStraightTrajectory(10.0);
  const Eigen::Vector3d lookahead =
    trajectoryPositionAt(trajectory, config.time_forward);
  constexpr double odom_yaw = 1.0;

  std::string error;
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, odom_yaw,
    lookahead + Eigen::Vector3d(-0.10, 0.0, 0.0));
  const auto entered = tracker.update(10.1);
  ASSERT_EQ(
    entered.state, scan_controller::ControllerState::kAligningYaw);
  ASSERT_TRUE(entered.execution_frozen);
  ASSERT_LT(entered.wz, 0.0);
  const double frozen_time = tracker.executionTimeSec();

  // 让“前视点 - odometry”恰好等于当前机体航向；旧的动态 LOS 算法会
  // 因瞬时误差为零而错误释放，但冻结期的 B-spline 有向弦仍指向 +X。
  const Eigen::Vector3d drifted_position = lookahead - 0.02 * Eigen::Vector3d(
    std::cos(odom_yaw), std::sin(odom_yaw), 0.0);
  const Eigen::Vector3d dynamic_los = lookahead - drifted_position;
  ASSERT_NEAR(
    std::atan2(dynamic_los.y(), dynamic_los.x()), odom_yaw, 1.0e-12);
  setFreshInputs(tracker, 10.2, odom_yaw, drifted_position);

  const auto drifted = tracker.update(10.2);

  EXPECT_EQ(
    drifted.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(drifted.execution_frozen);
  EXPECT_LT(drifted.wz, 0.0);
  EXPECT_DOUBLE_EQ(drifted.vx, 0.0);
  EXPECT_DOUBLE_EQ(drifted.vy, 0.0);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), frozen_time);
}

TEST(TrajectoryTracker, ShortReverseChordCannotStartYawAlignment)
{
  auto config = makeConfig();
  config.yaw_alignment_min_chord_distance = 0.03;
  config.max_yaw_acc = 1.0;
  scan_controller::TrajectoryTracker tracker(config);
  const auto trajectory = makeShortReverseChordTrajectory(10.0);
  const Eigen::Vector3d spline_start = trajectoryPositionAt(trajectory, 0.0);
  const Eigen::Vector3d short_lookahead =
    trajectoryPositionAt(trajectory, config.time_forward);
  const Eigen::Vector3d short_chord = short_lookahead - spline_start;
  ASSERT_LT(short_chord.y(), 0.0);
  ASSERT_LT(
    short_chord.head<2>().norm(),
    config.yaw_alignment_min_chord_distance);

  const std::vector<Eigen::Vector3d> reference_path{
    Eigen::Vector3d(0.0, -0.5, 0.0),
    Eigen::Vector3d(0.0, 0.5, 0.0),
    Eigen::Vector3d(0.0, 1.5, 0.0),
  };
  std::string error;
  ASSERT_TRUE(
    tracker.setReferencePath(reference_path, 10.0, 10.0, error)) << error;
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(tracker, 10.0, M_PI_2, spline_start);
  const double execution_time_before = tracker.executionTimeSec();

  // v4 的滚动轨迹在起步时出现毫米级反向短弦。它可以产生瞬时跟踪修正，
  // 但不能被锁存为世界系航向并冻结整条轨迹。
  const auto output = tracker.update(10.1);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.execution_frozen);
  EXPECT_GT(tracker.executionTimeSec(), execution_time_before);
  EXPECT_LE(std::abs(output.wz), 0.100001);

  for (int index = 2; index <= 15; ++index) {
    const double now_sec = 10.0 + 0.1 * static_cast<double>(index);
    setFreshInputs(tracker, now_sec, M_PI_2, spline_start);
    const auto progressed = tracker.update(now_sec);
    EXPECT_EQ(
      progressed.state, scan_controller::ControllerState::kTracking)
      << "index=" << index;
    EXPECT_FALSE(progressed.execution_frozen) << "index=" << index;
  }
}

TEST(TrajectoryTracker, CrossTrackRecoveryKeepsPathTangentAndUsesLateralVelocity)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.cross_track_alignment_distance = 0.10;
  config.cross_track_heading_error_threshold = 0.18;
  config.cross_track_heading_error_release_threshold = 0.16;
  config.max_yaw_acc = 1.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setReferencePath(tracker, 10.0);
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.0, -0.50, 0.3));
  const double execution_time_before = tracker.executionTimeSec();
  const auto output = tracker.update(10.1);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.execution_frozen);
  EXPECT_GT(output.vx, 0.0);
  EXPECT_GT(output.vy, 0.0);
  EXPECT_GT(output.wz, 0.0);
  EXPECT_GT(tracker.executionTimeSec(), execution_time_before);
}

TEST(TrajectoryTracker, FinalCrossTrackRecoveryFacesSafeLocalSplineBeforeTerminalYaw)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.cross_track_alignment_distance = 0.10;
  config.cross_track_heading_error_threshold = 0.18;
  config.cross_track_heading_error_release_threshold = 0.16;
  // 让旧的 Path 切向逻辑不会仅因微小回正辅助进入原地转向，从而精确复现
  // seed=2：完整 Path 末段朝 -Y，安全 final B-spline 从终点西侧朝 +X。
  config.cross_track_heading_assist_gain = 1.0e-6;
  config.cross_track_heading_assist_max = 1.0e-6;
  config.max_yaw_acc = 1.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  const Eigen::Vector3d spline_start = trajectoryPositionAt(trajectory, 0.0);
  ASSERT_GT(
    (goal.head<2>() - spline_start.head<2>()).norm(),
    config.terminal_capture_release_distance_xy);
  setFinalReferencePath(tracker, 10.0, goal, -M_PI_2);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(tracker, 10.0, -M_PI_2, spline_start);
  const double execution_time_before = tracker.executionTimeSec();

  const auto aligning = tracker.update(10.1);

  EXPECT_EQ(aligning.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(aligning.execution_frozen);
  EXPECT_DOUBLE_EQ(aligning.vx, 0.0);
  EXPECT_DOUBLE_EQ(aligning.vy, 0.0);
  EXPECT_GT(aligning.wz, 0.0);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), execution_time_before);

  // 朝 +X 对齐后，原世界系回收速度成为策略擅长的正向 vx，而不是持续 vy。
  setFreshInputs(tracker, 10.2, 0.0, spline_start);
  const auto recovering = tracker.update(10.2);
  EXPECT_EQ(recovering.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(recovering.execution_frozen);
  EXPECT_GT(recovering.vx, 0.0);
  EXPECT_NEAR(recovering.vy, 0.0, 1.0e-9);
  EXPECT_GT(tracker.executionTimeSec(), execution_time_before);
}

TEST(TrajectoryTracker, PositiveCrossTrackErrorUsesNegativeLateralVelocity)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.cross_track_alignment_distance = 0.10;
  config.cross_track_heading_error_threshold = 0.18;
  config.cross_track_heading_error_release_threshold = 0.16;
  config.max_yaw_acc = 1.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setReferencePath(tracker, 10.0);
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.0, 0.50, 0.3));

  const auto output = tracker.update(10.1);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.execution_frozen);
  EXPECT_GT(output.vx, 0.0);
  EXPECT_LT(output.vy, 0.0);
  EXPECT_LT(output.wz, 0.0);
}

TEST(TrajectoryTracker, SmallCrossTrackErrorKeepsSmoothTracking)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setReferencePath(tracker, 10.0);
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.0, -0.05, 0.3));

  const auto output = tracker.update(10.1);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.execution_frozen);
  EXPECT_GT(output.vx, 0.0);
  EXPECT_GT(output.vy, 0.0);
  EXPECT_GT(output.wz, 0.0);
}

TEST(TrajectoryTracker, StairHeadingLockTurnsLateralRejoinIntoVy)
{
  auto config = makeConfig();
  config.stair_heading_lock_enabled = true;
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto lateral_rejoin = makeStraightTrajectory(10.0);
  lateral_rejoin.control_points.row(0) <<
    0.16, 0.16, 0.16, 0.16, 0.50, 0.80;
  lateral_rejoin.control_points.row(1).setZero();
  const std::vector<Eigen::Vector3d> stair_path{
    Eigen::Vector3d(-0.50, 0.0, 0.00),
    Eigen::Vector3d(0.15, 0.0, 0.00),
    Eigen::Vector3d(0.17, 0.0, 0.19),
    Eigen::Vector3d(0.47, 0.0, 0.19),
    Eigen::Vector3d(0.49, 0.0, 0.38),
    Eigen::Vector3d(0.79, 0.0, 0.38),
  };
  std::string error;
  ASSERT_TRUE(
    tracker.setReferencePath(stair_path, 10.0, 10.0, error)) << error;
  ASSERT_TRUE(tracker.setTrajectory(lateral_rejoin, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.16, -0.047, 0.30));

  const auto output = tracker.update(10.1);

  // 局部前视点几乎位于机体正左侧，但完整 Path 的上楼方向是 +X。
  // 修复后机体保持 +X，侧向 4.7cm 重接由 vy 承担，不再生成饱和 wz。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.execution_frozen);
  EXPECT_GT(output.vy, 0.0);
  EXPECT_NEAR(output.wz, 0.0, 1.0e-12);
}

TEST(TrajectoryTracker, StairHeadingLockDoesNotSuppressFlatSplineTurn)
{
  auto config = makeConfig();
  config.stair_heading_lock_enabled = true;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto lateral_turn = makeStraightTrajectory(10.0);
  lateral_turn.control_points.row(0) <<
    0.16, 0.16, 0.16, 0.16, 0.50, 0.80;
  lateral_turn.control_points.row(1).setZero();
  const std::vector<Eigen::Vector3d> flat_path{
    Eigen::Vector3d(-0.50, 0.0, 0.0),
    Eigen::Vector3d(0.50, 0.0, 0.0),
    Eigen::Vector3d(1.50, 0.0, 0.0),
  };
  std::string error;
  ASSERT_TRUE(
    tracker.setReferencePath(flat_path, 10.0, 10.0, error)) << error;
  ASSERT_TRUE(tracker.setTrajectory(lateral_turn, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.16, -0.047, 0.30));

  const auto output = tracker.update(10.1);

  // 没有陡升段时不启用楼梯语义，普通 B-spline 转向反馈保持原行为。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_GT(output.wz, 0.0);
}

TEST(TrajectoryTracker, StairForwardFloorRecoversPlanarTraction)
{
  auto config = makeConfig();
  config.stair_heading_lock_enabled = true;
  config.stair_forward_speed_floor = 1.0;
  config.max_vx = 1.5;
  config.max_ax = 20.0;
  config.max_ay = 20.0;
  config.max_yaw_acc = 20.0;
  scan_controller::TrajectoryTracker tracker(config);
  const auto trajectory = makeSteepSlowTrajectory(10.0);
  std::string error;
  setStairReferencePath(tracker, 10.0);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, 0.0, trajectoryPositionAt(trajectory, 0.1));

  const auto output = tracker.update(10.1);

  // 近竖直 B-spline 的原始水平导数约为 0.01m/s；楼梯牵引只补 +X
  // 切向，使 policy 获得与固定速度演示一致的有效步态命令。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.execution_frozen);
  EXPECT_NEAR(output.vx, 1.0, 1.0e-9);
  EXPECT_NEAR(output.vy, 0.0, 1.0e-9);
  EXPECT_NEAR(output.wz, 0.0, 1.0e-9);
}

TEST(TrajectoryTracker, StairForwardFloorDoesNotOverrideReverseIntent)
{
  auto config = makeConfig();
  config.stair_heading_lock_enabled = true;
  config.stair_forward_speed_floor = 1.0;
  config.max_vx = 1.5;
  config.max_ax = 20.0;
  config.max_ay = 20.0;
  config.max_yaw_acc = 20.0;
  scan_controller::TrajectoryTracker tracker(config);
  const auto trajectory = makeSteepSlowTrajectory(10.0, true);
  std::string error;
  setStairReferencePath(tracker, 10.0);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, 0.0, trajectoryPositionAt(trajectory, 0.1));

  const auto output = tracker.update(10.1);

  // SCAN 若明确要求回退，楼梯牵引不得把负向命令改写为固定前进。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_LT(output.vx, 0.0);
  EXPECT_GT(output.vx, -0.10);
}

TEST(TrajectoryTracker, StairForwardFloorYieldsToCrossTrackRecovery)
{
  auto config = makeConfig();
  config.stair_heading_lock_enabled = true;
  config.stair_forward_speed_floor = 1.0;
  config.max_vx = 1.5;
  config.max_ax = 20.0;
  config.max_ay = 20.0;
  config.max_yaw_acc = 20.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto trajectory = makeSteepSlowTrajectory(10.0);
  trajectory.control_points.row(1).setConstant(0.20);
  std::string error;
  setStairReferencePath(tracker, 10.0);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, 0.0, trajectoryPositionAt(trajectory, 0.1));

  const auto output = tracker.update(10.1);

  // 横向偏离超过恢复门后保留既有 0.18m/s 切向上限，不用楼梯牵引绕过它。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_GT(output.vx, 0.0);
  EXPECT_LE(output.vx, config.cross_track_recovery_forward_speed + 1.0e-9);
  EXPECT_LT(output.vy, 0.0);
}

TEST(TrajectoryTracker, StairForwardFloorDoesNotAffectFlatPath)
{
  auto config = makeConfig();
  config.stair_heading_lock_enabled = true;
  config.stair_forward_speed_floor = 1.0;
  config.max_vx = 1.5;
  config.max_ax = 20.0;
  config.max_ay = 20.0;
  config.max_yaw_acc = 20.0;
  scan_controller::TrajectoryTracker tracker(config);
  const auto trajectory = makeSteepSlowTrajectory(10.0);
  std::string error;
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setReferencePath(tracker, 10.0);
  setFreshInputs(
    tracker, 10.0, 0.0, trajectoryPositionAt(trajectory, 0.1));

  const auto output = tracker.update(10.1);

  // 完整 Path 没有陡升段时保持 B-spline 原始小水平速度。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_GT(output.vx, 0.0);
  EXPECT_LT(output.vx, 0.10);
}

TEST(TrajectoryTracker, GlobalPathGateSurvivesLocalSplineResetAtOdometry)
{
  auto config = makeConfig();
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto local_trajectory = makeStraightTrajectory(10.0);
  local_trajectory.control_points.row(1).setConstant(-0.13);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(local_trajectory, 10.0, error)) << error;
  setReferencePath(tracker, 10.0);
  setFreshInputs(
    tracker, 10.0, -0.30, Eigen::Vector3d(0.0, -0.13, 0.3));

  const auto output = tracker.update(10.1);

  // 局部轨迹恰好从当前侧偏位置起步；只有完整 Path 基准才能触发严格门。
  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(output.execution_frozen);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_GT(output.wz, 0.0);
}

TEST(TrajectoryTracker, GlobalPathRecoverySurvivesLocalSplineLateralReset)
{
  auto config = makeConfig();
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto local_trajectory = makeStraightTrajectory(10.0);
  local_trajectory.control_points.row(1).setConstant(-0.13);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(local_trajectory, 10.0, error)) << error;
  setReferencePath(tracker, 10.0);
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.0, -0.13, 0.3));

  const auto output = tracker.update(10.1);

  // 局部轨迹与 Odometry 都位于 y=-0.13；恢复门仍必须依据完整 Path
  // 输出受限前向牵引、朝中心的法向速度与同向小幅航向辅助。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.execution_frozen);
  EXPECT_NEAR(
    output.vx, config.cross_track_recovery_forward_speed, 1.0e-12);
  EXPECT_GT(output.vy, 0.0);
  EXPECT_GT(output.wz, 0.0);
  EXPECT_LE(
    output.wz,
    config.kp_yaw * config.cross_track_heading_assist_max + 1.0e-12);
  EXPECT_LE(
    output.vy,
    config.cross_track_recovery_lateral_speed + 1.0e-12);
}

TEST(TrajectoryTracker, CrossTrackRecoveryPreservesLocalStopAndReverse)
{
  auto config = makeConfig();
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  config.max_yaw_acc = 10.0;
  std::string error;

  scan_controller::TrajectoryTracker stopped_tracker(config);
  auto stopped_trajectory = makeStraightTrajectory(10.0);
  stopped_trajectory.control_points.row(0).setZero();
  stopped_trajectory.control_points.row(1).setConstant(-0.13);
  ASSERT_TRUE(
    stopped_tracker.setTrajectory(stopped_trajectory, 10.0, error)) << error;
  setReferencePath(stopped_tracker, 10.0);
  setFreshInputs(
    stopped_tracker, 10.0, 0.0, Eigen::Vector3d(0.0, -0.13, 0.3));

  const auto stopped = stopped_tracker.update(10.1);
  EXPECT_EQ(stopped.state, scan_controller::ControllerState::kTracking);
  EXPECT_NEAR(stopped.vx, 0.0, 1.0e-12);
  EXPECT_GT(stopped.vy, 0.0);

  scan_controller::TrajectoryTracker reverse_tracker(config);
  auto reverse_trajectory = makeStraightTrajectory(10.0);
  reverse_trajectory.control_points.row(0) *= -1.0;
  reverse_trajectory.control_points.row(1).setConstant(-0.13);
  ASSERT_TRUE(
    reverse_tracker.setTrajectory(reverse_trajectory, 10.0, error)) << error;
  setReferencePath(reverse_tracker, 10.0);
  setFreshInputs(
    reverse_tracker, 10.0, 0.0, Eigen::Vector3d(0.0, -0.13, 0.3));

  const auto reverse = reverse_tracker.update(10.1);
  EXPECT_EQ(reverse.state, scan_controller::ControllerState::kTracking);
  EXPECT_LT(reverse.vx, 0.0);
  EXPECT_GE(
    reverse.vx,
    -config.cross_track_recovery_forward_speed - 1.0e-12);
}

TEST(TrajectoryTracker, CrossTrackRecoveryPreservesScanDetourDirection)
{
  auto config = makeConfig();
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto detour = makeStraightTrajectory(10.0);
  detour.control_points.row(1) <<
    -0.13, -0.33, -0.53, -0.73, -0.93, -1.13;
  std::string error;
  ASSERT_TRUE(tracker.setTrajectory(detour, 10.0, error)) << error;
  setReferencePath(tracker, 10.0);
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.0, -0.13, 0.3));

  const auto output = tracker.update(10.1);

  // 朝路径外侧的 SCAN 绕障分量与全局回正方向相反时，必须完整优先。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_LT(output.vy, 0.0);
}

TEST(TrajectoryTracker, CrossTrackRecoveryTapersForwardSpeedAtPathEnd)
{
  auto config = makeConfig();
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto local_trajectory = makeStraightTrajectory(10.0);
  local_trajectory.control_points.row(0).array() += 2.5;
  local_trajectory.control_points.row(1).setConstant(-0.13);
  local_trajectory.control_points.row(2).setConstant(0.5);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(local_trajectory, 10.0, error)) << error;
  setReferencePath(tracker, 10.0);
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(2.5, -0.13, 0.5));

  const auto output = tracker.update(10.1);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_NEAR(output.vx, 0.0, 1.0e-12);
  EXPECT_GT(output.vy, 0.0);
}

TEST(TrajectoryTracker, CrossTrackHeadingThresholdSeparatesMoveAndFreeze)
{
  auto config = makeConfig();
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  config.max_yaw_acc = 10.0;
  std::string error;

  scan_controller::TrajectoryTracker moving_tracker(config);
  ASSERT_TRUE(
    moving_tracker.setTrajectory(
      makeStraightTrajectory(10.0), 10.0, error)) << error;
  setReferencePath(moving_tracker, 10.0);
  setFreshInputs(
    moving_tracker, 10.0, -0.03, Eigen::Vector3d(0.0, -0.50, 0.3));
  const auto moving = moving_tracker.update(10.1);
  EXPECT_EQ(moving.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(moving.execution_frozen);
  EXPECT_GT(moving.vx, 0.0);

  scan_controller::TrajectoryTracker frozen_tracker(config);
  ASSERT_TRUE(
    frozen_tracker.setTrajectory(
      makeStraightTrajectory(10.0), 10.0, error)) << error;
  setReferencePath(frozen_tracker, 10.0);
  setFreshInputs(
    frozen_tracker, 10.0, -0.05, Eigen::Vector3d(0.0, -0.50, 0.3));
  const auto frozen = frozen_tracker.update(10.1);
  EXPECT_EQ(
    frozen.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(frozen.execution_frozen);
  EXPECT_DOUBLE_EQ(frozen.vx, 0.0);
  EXPECT_DOUBLE_EQ(frozen.vy, 0.0);
}

TEST(TrajectoryTracker, CrossTrackAlignmentUsesEnterReleaseHysteresis)
{
  auto config = makeConfig();
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto local_trajectory = makeStraightTrajectory(10.0);
  local_trajectory.control_points.row(1).setConstant(-0.13);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(local_trajectory, 10.0, error)) << error;
  setReferencePath(tracker, 10.0);

  setFreshInputs(
    tracker, 10.0, -0.30, Eigen::Vector3d(0.0, -0.13, 0.3));
  EXPECT_EQ(
    tracker.update(10.1).state,
    scan_controller::ControllerState::kAligningYaw);

  setFreshInputs(
    tracker, 10.2, -0.30, Eigen::Vector3d(0.0, -0.10, 0.3));
  EXPECT_EQ(
    tracker.update(10.2).state,
    scan_controller::ControllerState::kAligningYaw);

  setFreshInputs(
    tracker, 10.3, -0.30, Eigen::Vector3d(0.0, -0.07, 0.3));
  EXPECT_EQ(
    tracker.update(10.3).state,
    scan_controller::ControllerState::kTracking);
}

TEST(TrajectoryTracker, ThreeDimensionalProjectionSelectsOverlappingUpperFloor)
{
  auto config = makeConfig();
  config.max_yaw_acc = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  auto local_trajectory = makeStraightTrajectory(10.0);
  local_trajectory.control_points.row(1).setConstant(0.13);
  local_trajectory.control_points.row(2).setConstant(3.3);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(local_trajectory, 10.0, error)) << error;
  const std::vector<Eigen::Vector3d> path{
    Eigen::Vector3d(0.0, 0.13, 0.0),
    Eigen::Vector3d(2.0, 0.13, 0.0),
    Eigen::Vector3d(2.0, 0.0, 3.0),
    Eigen::Vector3d(0.0, 0.0, 3.0),
  };
  ASSERT_TRUE(
    tracker.setReferencePath(path, 10.0, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, -0.30, Eigen::Vector3d(1.0, 0.13, 3.3));

  const auto output = tracker.update(10.1);

  // 下层线段在 XY 上恰好重合机器人；正确的 3D 投影应选择上层 y=0，
  // 保留 0.13m 横向误差并进入严格回正。
  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kAligningYaw);
}

TEST(TrajectoryTracker, ExpiredTrajectoryResumeReacquiresForwardUpperFloor)
{
  auto config = makeConfig();
  config.max_yaw_acc = 10.0;
  config.reference_path_backward_arc = 1.0;
  config.reference_path_forward_arc = 3.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;

  const std::vector<Eigen::Vector3d> cross_floor_path{
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 2.0, 0.0),
    Eigen::Vector3d(0.0, 6.0, 3.0),
    Eigen::Vector3d(4.0, 6.0, 3.0),
    Eigen::Vector3d(4.0, 3.0, 3.0),
  };
  ASSERT_TRUE(
    tracker.setReferencePath(
      cross_floor_path, 10.0, 10.0, error, -M_PI_2)) << error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.0, 1.0, 0.3));
  tracker.update(10.1);

  // 旧轨迹已过期后，模拟底盘冻结沿楼梯向前推进到上层。上层局部轨迹
  // 指向 -Y，而旧进度窗只能看到下层/楼梯的 +Y 段。
  auto upper_floor_trajectory = makeStraightTrajectory(21.0);
  upper_floor_trajectory.trajectory_id = 8;
  upper_floor_trajectory.reference_path_stamp_sec = 10.0;
  upper_floor_trajectory.control_points <<
    4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    5.5, 5.0, 4.5, 4.0, 3.5, 3.0,
    3.3, 3.3, 3.3, 3.3, 3.3, 3.3;
  ASSERT_TRUE(
    tracker.setTrajectory(upper_floor_trajectory, 21.0, error)) << error;
  setFreshInputs(
    tracker, 21.0, -M_PI_2, Eigen::Vector3d(4.0, 5.5, 3.3));

  const auto resumed = tracker.update(21.1);

  EXPECT_EQ(resumed.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(resumed.execution_frozen);
  EXPECT_NEAR(resumed.wz, 0.0, 1.0e-9);
}

TEST(TrajectoryTracker, RampHeightResidualDoesNotBecomeCrossTrackError)
{
  auto config = makeConfig();
  config.cross_track_alignment_distance = 0.05;
  config.cross_track_alignment_release_distance = 0.03;
  scan_controller::TrajectoryTracker tracker(config);
  auto local_trajectory = makeStraightTrajectory(10.0);
  // 让局部轨迹从当前 x 附近向前延伸，避免本用例被无关的普通航向门触发。
  local_trajectory.control_points.row(0).array() += 1.0;
  local_trajectory.control_points.row(2).setConstant(0.8);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(local_trajectory, 10.0, error)) << error;
  const std::vector<Eigen::Vector3d> ramp_path{
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(2.0, 0.0, 0.4),
  };
  ASSERT_TRUE(
    tracker.setReferencePath(ramp_path, 10.0, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(1.0, 0.0, 0.8));

  const auto output = tracker.update(10.1);

  // 3D 投影会因故意加入的高度误差沿坡面方向移动，但法向误差仍为零。
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_NEAR(output.wz, 0.0, 1.0e-12);
}

TEST(TrajectoryTracker, SameStampIdenticalPathRepublishIsIdempotent)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  const std::vector<Eigen::Vector3d> same_path{
    Eigen::Vector3d(-0.5, 0.0, 0.0),
    Eigen::Vector3d(0.5, 0.0, 0.0),
    Eigen::Vector3d(1.5, 0.0, 0.2),
    Eigen::Vector3d(2.5, 0.0, 0.2),
  };

  ASSERT_TRUE(
    tracker.setReferencePath(same_path, 10.0, 10.01, error)) << error;

  EXPECT_TRUE(tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(tracker.referencePathStampSec(), 10.0);
}

TEST(TrajectoryTracker, NewStampIdenticalGeometryCreatesNewPathGeneration)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  const std::vector<Eigen::Vector3d> same_path{
    Eigen::Vector3d(-0.5, 0.0, 0.0),
    Eigen::Vector3d(0.5, 0.0, 0.0),
    Eigen::Vector3d(1.5, 0.0, 0.2),
    Eigen::Vector3d(2.5, 0.0, 0.2),
  };

  ASSERT_TRUE(
    tracker.setReferencePath(same_path, 11.0, 11.0, error)) << error;

  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(tracker.referencePathStampSec(), 11.0);
}

TEST(TrajectoryTracker, TerminalYawChangeCreatesNewPathGeneration)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  const std::vector<Eigen::Vector3d> same_geometry{
    Eigen::Vector3d(-0.5, 0.0, 0.0),
    Eigen::Vector3d(0.5, 0.0, 0.0),
    Eigen::Vector3d(1.5, 0.0, 0.2),
    Eigen::Vector3d(2.5, 0.0, 0.2),
  };

  ASSERT_TRUE(
    tracker.setReferencePath(
      same_geometry, 11.0, 11.0, error, M_PI_2)) << error;

  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(tracker.referencePathStampSec(), 11.0);
}

TEST(TrajectoryTracker, ChangedGeometryWithSameStampPoisonsAmbiguousGeneration)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  const std::vector<Eigen::Vector3d> changed_path{
    Eigen::Vector3d(-0.5, 0.1, 0.0),
    Eigen::Vector3d(0.5, 0.1, 0.0),
    Eigen::Vector3d(1.5, 0.1, 0.2),
  };

  EXPECT_FALSE(
    tracker.setReferencePath(changed_path, 10.0, 10.0, error));

  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_FALSE(tracker.hasReferencePath());
  EXPECT_NE(error.find("同一参考 Path 代际"), std::string::npos);

  // 冲突代际必须持续作废，不能靠第三条同 stamp Path 或 B-spline 复活。
  EXPECT_FALSE(
    tracker.setReferencePath(changed_path, 10.0, 10.0, error));
  EXPECT_NE(error.find("作废"), std::string::npos);
  EXPECT_FALSE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error));
  EXPECT_NE(error.find("作废"), std::string::npos);

  EXPECT_TRUE(
    tracker.setReferencePath(changed_path, 10.01, 10.01, error)) << error;
  EXPECT_TRUE(tracker.hasReferencePath());
}

TEST(TrajectoryTracker, TerminalYawChangeWithinClockToleranceInvalidatesSpline)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  const std::vector<Eigen::Vector3d> same_geometry{
    Eigen::Vector3d(-0.5, 0.0, 0.0),
    Eigen::Vector3d(0.5, 0.0, 0.0),
    Eigen::Vector3d(1.5, 0.0, 0.2),
    Eigen::Vector3d(2.5, 0.0, 0.2),
  };

  ASSERT_TRUE(
    tracker.setReferencePath(
      same_geometry, 10.05, 10.05, error, M_PI_2)) << error;

  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(tracker.referencePathStampSec(), 10.05);
}

TEST(TrajectoryTracker, RejectsSplineFromNearbyButDifferentPathGeneration)
{
  for (const double spline_path_stamp : {9.95, 10.05}) {
    scan_controller::TrajectoryTracker tracker(makeConfig());
    setReferencePath(tracker, 10.0);
    auto trajectory = makeStraightTrajectory(10.05);
    trajectory.reference_path_stamp_sec = spline_path_stamp;
    std::string error;

    EXPECT_FALSE(tracker.setTrajectory(trajectory, 10.05, error));
    EXPECT_FALSE(tracker.hasTrajectory());
    EXPECT_NE(error.find("代际不匹配"), std::string::npos);
  }
}

TEST(TrajectoryTracker, NewPathGenerationInvalidatesOldSpline)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  const std::vector<Eigen::Vector3d> new_path{
    Eigen::Vector3d(-0.5, 0.1, 0.0),
    Eigen::Vector3d(0.5, 0.1, 0.0),
    Eigen::Vector3d(1.5, 0.1, 0.2),
  };

  ASSERT_TRUE(
    tracker.setReferencePath(new_path, 11.0, 11.0, error)) << error;

  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(tracker.referencePathStampSec(), 11.0);
}

TEST(TrajectoryTracker, ExactBsplineDuplicateIsIdempotent)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  const auto trajectory = makeStraightTrajectory(10.0);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(tracker, 10.0);
  tracker.update(10.10);
  const double execution_before_duplicate = tracker.executionTimeSec();

  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.10, error)) << error;

  EXPECT_TRUE(tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(
    tracker.executionTimeSec(), execution_before_duplicate);

  // 即使 exact duplicate 自身按接收时刻已经 stale，也只能幂等忽略，不能
  // 通过 DDS 延迟包清空当前状态。
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 30.0, error)) << error;
  EXPECT_TRUE(tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(
    tracker.executionTimeSec(), execution_before_duplicate);
}

TEST(TrajectoryTracker, SameIdentityYawChangePoisonsPayload)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  setFreshInputs(tracker, 10.0, 0.0);
  std::string error;
  const auto original = makeYawOnlyTrajectory(10.0, 0.0, 0.20);
  ASSERT_TRUE(tracker.setTrajectory(original, 10.0, error)) << error;

  auto changed_yaw = original;
  changed_yaw.yaw_points[1] = 0.19;
  EXPECT_FALSE(tracker.setTrajectory(changed_yaw, 10.0, error));
  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_NE(error.find("同一 B-spline identity"), std::string::npos);

  EXPECT_FALSE(tracker.setTrajectory(original, 10.0, error));
  EXPECT_NE(error.find("identity 已"), std::string::npos);
}

TEST(TrajectoryTracker, YawOnlyCommandsStrictZeroTranslationWithRateLimits)
{
  auto config = makeConfig();
  config.max_yaw_acc = 0.10;
  config.terminal_max_yaw_acc = 0.10;
  scan_controller::TrajectoryTracker tracker(config);
  setReferencePath(tracker, 10.0);
  setFreshInputs(tracker, 10.0, 0.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(
      makeYawOnlyTrajectory(10.0, 0.0, 0.20), 10.0, error)) << error;

  const auto first = tracker.update(10.1);
  EXPECT_EQ(first.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(first.execution_frozen);
  EXPECT_DOUBLE_EQ(first.vx, 0.0);
  EXPECT_DOUBLE_EQ(first.vy, 0.0);
  EXPECT_NEAR(first.wz, 0.01, 1.0e-12);
  EXPECT_LE(std::abs(first.wz), config.active_sensing_max_yaw_rate);

  setFreshInputs(tracker, 10.2, 0.0);
  const auto second = tracker.update(10.2);
  EXPECT_DOUBLE_EQ(second.vx, 0.0);
  EXPECT_DOUBLE_EQ(second.vy, 0.0);
  EXPECT_NEAR(second.wz, 0.02, 1.0e-12);
  EXPECT_LE(second.wz - first.wz, config.max_yaw_acc * 0.1 + 1.0e-12);
}

TEST(TrajectoryTracker, YawOnlyPreservesSignedTurnDirection)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  setFreshInputs(tracker, 10.0, 0.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(
      makeYawOnlyTrajectory(10.0, 0.0, -0.20), 10.0, error)) << error;
  const auto output = tracker.update(10.1);

  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_LT(output.wz, 0.0);
  EXPECT_GE(output.wz, -makeConfig().active_sensing_max_yaw_rate);
}

TEST(TrajectoryTracker, YawOnlyPreservesTurnSignAcrossPiBoundary)
{
  std::string error;
  scan_controller::TrajectoryTracker positive_tracker(makeConfig());
  setReferencePath(positive_tracker, 10.0);
  setFreshInputs(positive_tracker, 10.0, M_PI - 0.10);
  ASSERT_TRUE(
    positive_tracker.setTrajectory(
      makeYawOnlyTrajectory(
        10.0, M_PI - 0.10, M_PI + 0.05, 0.75),
      10.0, error)) << error;
  const auto positive = positive_tracker.update(10.1);
  EXPECT_EQ(
    positive.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_GT(positive.wz, 0.0);

  scan_controller::TrajectoryTracker negative_tracker(makeConfig());
  setReferencePath(negative_tracker, 20.0);
  setFreshInputs(negative_tracker, 20.0, -M_PI + 0.10);
  ASSERT_TRUE(
    negative_tracker.setTrajectory(
      makeYawOnlyTrajectory(
        20.0, -M_PI + 0.10, -M_PI - 0.05, 0.75),
      20.0, error)) << error;
  const auto negative = negative_tracker.update(20.1);
  EXPECT_EQ(
    negative.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_LT(negative.wz, 0.0);
}

TEST(TrajectoryTracker, YawOnlyHoldsValidAtTargetWithoutFinishing)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  setFreshInputs(tracker, 10.0, 0.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(
      makeYawOnlyTrajectory(10.0, 0.0, 0.20), 10.0, error)) << error;
  ASSERT_GT(tracker.update(10.1).wz, 0.0);

  setFreshInputs(tracker, 10.2, 0.185);
  const auto reached = tracker.update(10.2);
  EXPECT_EQ(reached.state, scan_controller::ControllerState::kTracking);
  EXPECT_TRUE(reached.execution_frozen);
  EXPECT_FALSE(reached.trajectory_finished);
  EXPECT_FALSE(reached.goal_reached);
  EXPECT_DOUBLE_EQ(reached.vx, 0.0);
  EXPECT_DOUBLE_EQ(reached.vy, 0.0);
  EXPECT_DOUBLE_EQ(reached.wz, 0.0);

  setFreshInputs(tracker, 10.3, 0.20);
  const auto held = tracker.update(10.3);
  EXPECT_EQ(held.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(held.trajectory_finished);
  EXPECT_DOUBLE_EQ(held.wz, 0.0);
}

TEST(TrajectoryTracker, YawOnlyStopsOnStalePathChangeAndEmergency)
{
  auto config = makeConfig();
  config.odom_timeout_sec = 0.15;
  scan_controller::TrajectoryTracker stale_tracker(config);
  setReferencePath(stale_tracker, 10.0);
  setFreshInputs(stale_tracker, 10.0, 0.0);
  std::string error;
  ASSERT_TRUE(
    stale_tracker.setTrajectory(
      makeYawOnlyTrajectory(10.0, 0.0, 0.20), 10.0, error)) << error;
  ASSERT_GT(stale_tracker.update(10.1).wz, 0.0);
  const auto stale = stale_tracker.update(10.2);
  EXPECT_EQ(stale.state, scan_controller::ControllerState::kOdometryTimeout);
  EXPECT_DOUBLE_EQ(stale.vx, 0.0);
  EXPECT_DOUBLE_EQ(stale.vy, 0.0);
  EXPECT_DOUBLE_EQ(stale.wz, 0.0);

  scan_controller::TrajectoryTracker path_tracker(makeConfig());
  setReferencePath(path_tracker, 20.0);
  setFreshInputs(path_tracker, 20.0, 0.0);
  auto path_yaw = makeYawOnlyTrajectory(20.0, 0.0, 0.20);
  ASSERT_TRUE(path_tracker.setTrajectory(path_yaw, 20.0, error)) << error;
  ASSERT_GT(path_tracker.update(20.1).wz, 0.0);
  const std::vector<Eigen::Vector3d> new_path{
    Eigen::Vector3d(0.0, 0.1, 0.0),
    Eigen::Vector3d(1.0, 0.1, 0.0)};
  ASSERT_TRUE(
    path_tracker.setReferencePath(new_path, 20.2, 20.2, error)) << error;
  const auto path_changed = path_tracker.immediateStop(20.2);
  EXPECT_FALSE(path_tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(path_changed.vx, 0.0);
  EXPECT_DOUBLE_EQ(path_changed.vy, 0.0);
  EXPECT_DOUBLE_EQ(path_changed.wz, 0.0);

  scan_controller::TrajectoryTracker emergency_tracker(makeConfig());
  setReferencePath(emergency_tracker, 30.0);
  setFreshInputs(emergency_tracker, 30.0, 0.0);
  auto emergency_yaw = makeYawOnlyTrajectory(30.0, 0.0, 0.20);
  ASSERT_TRUE(
    emergency_tracker.setTrajectory(emergency_yaw, 30.0, error)) << error;
  ASSERT_GT(emergency_tracker.update(30.1).wz, 0.0);
  auto emergency = makeStraightTrajectory(30.2);
  emergency.reference_path_stamp_sec = 30.0;
  emergency.trajectory_id = 8;
  emergency.emergency_stop = true;
  ASSERT_TRUE(
    emergency_tracker.setTrajectory(emergency, 30.2, error)) << error;
  const auto stopped = emergency_tracker.update(30.2);
  EXPECT_EQ(stopped.state, scan_controller::ControllerState::kEmergencyStop);
  EXPECT_DOUBLE_EQ(stopped.vx, 0.0);
  EXPECT_DOUBLE_EQ(stopped.vy, 0.0);
  EXPECT_DOUBLE_EQ(stopped.wz, 0.0);
}

TEST(TrajectoryTracker, YawOnlyHardTimeoutIsStrictZero)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.10;
  config.trajectory_expiry_grace_sec = 0.05;
  scan_controller::TrajectoryTracker tracker(config);
  setReferencePath(tracker, 10.0);
  setFreshInputs(tracker, 10.0, 0.0);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(
      makeYawOnlyTrajectory(10.0, 0.0, 0.20), 10.0, error)) << error;
  ASSERT_GT(tracker.update(10.1).wz, 0.0);
  setFreshInputs(tracker, 13.1, 0.0);

  const auto expired = tracker.update(13.1);

  EXPECT_EQ(
    expired.state, scan_controller::ControllerState::kTrajectoryTimeout);
  EXPECT_FALSE(expired.execution_frozen);
  EXPECT_DOUBLE_EQ(expired.vx, 0.0);
  EXPECT_DOUBLE_EQ(expired.vy, 0.0);
  EXPECT_DOUBLE_EQ(expired.wz, 0.0);
}

TEST(TrajectoryTracker, OlderBsplineCannotReplaceCurrentTrajectory)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  auto current = makeStraightTrajectory(10.10);
  current.reference_path_stamp_sec = 10.0;
  current.trajectory_id = 8;
  ASSERT_TRUE(tracker.setTrajectory(current, 10.10, error)) << error;

  auto old = makeStraightTrajectory(10.05);
  old.reference_path_stamp_sec = 10.0;
  old.control_points.row(1).setConstant(0.1);
  EXPECT_FALSE(tracker.setTrajectory(old, 10.10, error));

  EXPECT_TRUE(tracker.hasTrajectory());
  EXPECT_NE(error.find("旧 B-spline"), std::string::npos);
}

TEST(TrajectoryTracker, ConflictingBsplineIdentityStaysPoisonedUntilNewerId)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  std::string error;
  auto original = makeStraightTrajectory(10.10);
  original.reference_path_stamp_sec = 10.0;
  original.trajectory_id = 8;
  ASSERT_TRUE(tracker.setTrajectory(original, 10.10, error)) << error;

  auto conflict = original;
  conflict.control_points(1, 2) = 0.1;
  EXPECT_FALSE(tracker.setTrajectory(conflict, 10.10, error));
  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_NE(error.find("同一 B-spline identity"), std::string::npos);

  EXPECT_FALSE(tracker.setTrajectory(original, 10.10, error));
  EXPECT_NE(error.find("identity 已"), std::string::npos);

  auto newer = makeStraightTrajectory(10.20);
  newer.reference_path_stamp_sec = 10.0;
  newer.trajectory_id = 9;
  EXPECT_TRUE(tracker.setTrajectory(newer, 10.20, error)) << error;
  EXPECT_TRUE(tracker.hasTrajectory());
}

TEST(TrajectoryTracker, EmergencyStopSplineLatchesExactZero)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  setReferencePath(tracker, 10.0);
  auto emergency = makeStraightTrajectory(10.0);
  emergency.emergency_stop = true;
  std::string error;
  ASSERT_TRUE(tracker.setTrajectory(emergency, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.0, 0.0, Eigen::Vector3d(0.8, -0.4, 0.3));

  const auto stopped = tracker.update(10.1);

  EXPECT_EQ(
    stopped.state, scan_controller::ControllerState::kEmergencyStop);
  EXPECT_TRUE(stopped.execution_frozen);
  EXPECT_DOUBLE_EQ(stopped.vx, 0.0);
  EXPECT_DOUBLE_EQ(stopped.vy, 0.0);
  EXPECT_DOUBLE_EQ(stopped.wz, 0.0);

  auto recovery = makeStraightTrajectory(10.2);
  recovery.reference_path_stamp_sec = 10.0;
  recovery.trajectory_id = 8;
  ASSERT_TRUE(tracker.setTrajectory(recovery, 10.2, error)) << error;
  setFreshInputs(tracker, 10.2);
  EXPECT_EQ(
    tracker.update(10.3).state,
    scan_controller::ControllerState::kTracking);
}

TEST(TrajectoryTracker, YawAlignmentAlsoPreservesTrajectoryExpiryBudget)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.30;
  config.trajectory_expiry_grace_sec = 0.25;
  config.max_yaw_alignment_freeze_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  setReferencePath(tracker, 10.0);
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;

  // 原始软失效时刻为 13.25；连续航向冻结跨过该时刻后恢复正确航向，
  // 仍应拥有完整的平移执行预算。
  for (int tick = 1; tick <= 34; ++tick) {
    const double stamp = 10.0 + 0.1 * tick;
    setFreshInputs(tracker, stamp, M_PI);
    ASSERT_TRUE(tracker.hasTrajectory()) << "tick=" << tick;
    const auto aligning = tracker.update(stamp);
    ASSERT_EQ(
      aligning.state, scan_controller::ControllerState::kAligningYaw);
    ASSERT_TRUE(aligning.execution_frozen);
  }
  setFreshInputs(tracker, 13.5, 0.0);
  const auto resumed = tracker.update(13.5);

  EXPECT_EQ(resumed.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(resumed.execution_frozen);
}

TEST(TrajectoryTracker, YawAlignmentCannotExtendPastHardExpiry)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.30;
  config.trajectory_expiry_grace_sec = 0.25;
  config.max_yaw_alignment_freeze_sec = 0.30;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  setReferencePath(tracker, 10.0);
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;

  for (int tick = 1; tick <= 35; ++tick) {
    const double stamp = 10.0 + 0.1 * tick;
    setFreshInputs(tracker, stamp, M_PI);
    ASSERT_TRUE(tracker.hasTrajectory()) << "tick=" << tick;
    ASSERT_EQ(
      tracker.update(stamp).state,
      scan_controller::ControllerState::kAligningYaw);
  }
  setFreshInputs(tracker, 13.6, M_PI);
  const auto timed_out = tracker.update(13.6);

  EXPECT_EQ(
    timed_out.state,
    scan_controller::ControllerState::kTrajectoryTimeout);
  EXPECT_FALSE(timed_out.execution_frozen);
  EXPECT_DOUBLE_EQ(timed_out.vx, 0.0);
  EXPECT_DOUBLE_EQ(timed_out.vy, 0.0);
  EXPECT_DOUBLE_EQ(timed_out.wz, 0.0);
}

TEST(TrajectoryTracker, YawChatterReplacementCannotMoveEpisodeHardExpiry)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.heading_error_release_threshold = 0.55;
  config.bspline_timeout_sec = 0.30;
  config.trajectory_expiry_grace_sec = 0.25;
  config.max_yaw_alignment_freeze_sec = 0.30;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  setReferencePath(tracker, 10.0);
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;

  // traj1 的软截止为 13.25，hard expiry 为 13.55。先持续对齐，再把
  // 误差放到 enter 与 release 之间，模拟单阈值实现中的 TRACKING 毛刺。
  for (int tick = 1; tick <= 33; ++tick) {
    const double stamp = 10.0 + 0.1 * static_cast<double>(tick);
    setFreshInputs(tracker, stamp, -0.71);
    ASSERT_EQ(
      tracker.update(stamp).state,
      scan_controller::ControllerState::kAligningYaw) << "tick=" << tick;
  }
  setFreshInputs(tracker, 13.4, -0.69);
  const auto chatter = tracker.update(13.4);
  ASSERT_EQ(
    chatter.state, scan_controller::ControllerState::kAligningYaw);
  ASSERT_TRUE(chatter.execution_frozen);

  auto replacement = makeStraightTrajectory(13.4);
  replacement.reference_path_stamp_sec = 10.0;
  replacement.trajectory_id = 8;
  ASSERT_TRUE(tracker.setTrajectory(replacement, 13.4, error)) << error;

  setFreshInputs(tracker, 13.56, -0.69);
  const auto timed_out = tracker.update(13.56);
  EXPECT_EQ(
    timed_out.state,
    scan_controller::ControllerState::kTrajectoryTimeout);
  EXPECT_FALSE(timed_out.execution_frozen);
  EXPECT_DOUBLE_EQ(timed_out.vx, 0.0);
  EXPECT_DOUBLE_EQ(timed_out.vy, 0.0);
  EXPECT_DOUBLE_EQ(timed_out.wz, 0.0);
}

TEST(TrajectoryTracker, YawReleaseAllowsReplacementToStartNewEpisode)
{
  auto config = makeConfig();
  config.heading_error_threshold = 0.70;
  config.heading_error_release_threshold = 0.55;
  config.bspline_timeout_sec = 0.30;
  config.trajectory_expiry_grace_sec = 0.25;
  config.max_yaw_alignment_freeze_sec = 0.30;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  setReferencePath(tracker, 10.0);
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;

  for (int tick = 1; tick <= 33; ++tick) {
    const double stamp = 10.0 + 0.1 * static_cast<double>(tick);
    setFreshInputs(tracker, stamp, -0.71);
    ASSERT_EQ(
      tracker.update(stamp).state,
      scan_controller::ControllerState::kAligningYaw) << "tick=" << tick;
  }

  // 真正降到 release 内会结束旧 episode；此后新轨迹可以拥有自己的
  // 有界截止，不能被已经完成的旧对齐误杀。
  setFreshInputs(tracker, 13.4, -0.54);
  const auto released = tracker.update(13.4);
  ASSERT_EQ(released.state, scan_controller::ControllerState::kTracking);
  ASSERT_FALSE(released.execution_frozen);

  auto replacement = makeStraightTrajectory(13.4);
  replacement.reference_path_stamp_sec = 10.0;
  replacement.trajectory_id = 8;
  ASSERT_TRUE(tracker.setTrajectory(replacement, 13.4, error)) << error;

  setFreshInputs(tracker, 13.56, -0.71);
  const auto new_episode = tracker.update(13.56);
  EXPECT_EQ(
    new_episode.state,
    scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(new_episode.execution_frozen);
  EXPECT_DOUBLE_EQ(new_episode.vx, 0.0);
  EXPECT_DOUBLE_EQ(new_episode.vy, 0.0);
  EXPECT_NE(new_episode.wz, 0.0);
}

TEST(TrajectoryTracker, TrackingCommandObeysMagnitudeAndRateLimits)
{
  auto config = makeConfig();
  config.max_vx = 0.60;
  config.max_vy = 0.30;
  config.max_yaw_rate = 0.80;
  config.max_ax = 0.50;
  config.max_ay = 0.40;
  config.max_yaw_acc = 0.60;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setFreshInputs(tracker, 10.0);

  const auto output = tracker.update(10.1);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_LE(std::abs(output.vx), 0.050001);
  EXPECT_LE(std::abs(output.vy), 0.040001);
  EXPECT_LE(std::abs(output.wz), 0.060001);
  EXPECT_LE(std::abs(output.vx), config.max_vx);
  EXPECT_LE(std::abs(output.vy), config.max_vy);
  EXPECT_LE(std::abs(output.wz), config.max_yaw_rate);
}

TEST(TrajectoryTracker, WorldVelocityIsRotatedIntoBodyFrame)
{
  auto config = makeConfig();
  config.heading_error_threshold = M_PI;
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setFreshInputs(tracker, 10.0, M_PI_2);

  const auto output = tracker.update(10.1);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_LT(output.vy, 0.0);
}

TEST(TrajectoryTracker, TurningSpeedLimitCapsOnlyHighYawRateTracking)
{
  auto config = makeConfig();
  config.turning_speed_limit_enabled = true;
  config.turning_yaw_rate_threshold = 0.35;
  config.turning_max_planar_speed = 0.42;
  config.heading_error_threshold = 1.00;
  config.heading_error_release_threshold = 0.80;
  config.max_vx = 0.65;
  config.max_vy = 0.15;
  config.max_yaw_rate = 0.60;
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  config.max_yaw_acc = 10.0;

  std::string error;
  scan_controller::TrajectoryTracker turning_tracker(config);
  ASSERT_TRUE(
    turning_tracker.setTrajectory(
      makeStraightTrajectory(10.0), 10.0, error)) << error;
  setFreshInputs(turning_tracker, 10.0, -0.30);
  const auto turning = turning_tracker.update(10.1);
  ASSERT_EQ(turning.state, scan_controller::ControllerState::kTracking);
  ASSERT_GE(
    std::abs(turning.wz), config.turning_yaw_rate_threshold - 1.0e-12);
  EXPECT_LE(
    std::hypot(turning.vx, turning.vy),
    config.turning_max_planar_speed + 1.0e-12);

  scan_controller::TrajectoryTracker straight_tracker(config);
  ASSERT_TRUE(
    straight_tracker.setTrajectory(
      makeStraightTrajectory(10.0), 10.0, error)) << error;
  setFreshInputs(straight_tracker, 10.0, 0.0);
  const auto straight = straight_tracker.update(10.1);
  ASSERT_EQ(straight.state, scan_controller::ControllerState::kTracking);
  EXPECT_LT(
    std::abs(straight.wz), config.turning_yaw_rate_threshold);
  EXPECT_GT(
    std::hypot(straight.vx, straight.vy),
    config.turning_max_planar_speed);
}

TEST(TrajectoryTracker, EachInputTimeoutForcesImmediateStop)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.30;
  config.trajectory_expiry_grace_sec = 0.25;
  config.max_control_dt_sec = 1.0;
  config.odom_timeout_sec = 0.20;
  config.cloud_timeout_sec = 0.25;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setFreshInputs(tracker, 10.0);

  scan_controller::OdometryInput late_odometry;
  late_odometry.position = Eigen::Vector3d(0.0, 0.0, 0.3);
  late_odometry.stamp_sec = 13.26;
  ASSERT_TRUE(
    tracker.setOdometry(late_odometry, 13.26, error)) << error;
  ASSERT_TRUE(
    tracker.setCloudObservation(13.26, 13.26, error)) << error;
  auto output = tracker.update(13.26);
  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kTrajectoryTimeout);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_FALSE(output.execution_frozen);

  config.bspline_timeout_sec = 10.0;
  scan_controller::TrajectoryTracker odom_tracker(config);
  ASSERT_TRUE(
    odom_tracker.setTrajectory(
      makeStraightTrajectory(10.0), 10.0, error)) << error;
  setFreshInputs(odom_tracker, 10.0);
  output = odom_tracker.update(10.21);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kOdometryTimeout);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_TRUE(output.execution_frozen);

  config.odom_timeout_sec = 10.0;
  scan_controller::TrajectoryTracker cloud_tracker(config);
  ASSERT_TRUE(
    cloud_tracker.setTrajectory(
      makeStraightTrajectory(10.0), 10.0, error)) << error;
  setFreshInputs(cloud_tracker, 10.0);
  output = cloud_tracker.update(10.26);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kCloudTimeout);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_TRUE(output.execution_frozen);
}

TEST(TrajectoryTracker, LongTrajectoryDoesNotExpireFromHeaderAge)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.30;
  config.max_control_dt_sec = 1.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setFreshInputs(tracker, 10.0);

  scan_controller::OdometryInput odometry;
  odometry.position = Eigen::Vector3d(0.0, 0.0, 0.3);
  odometry.stamp_sec = 10.50;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.50, error)) << error;
  ASSERT_TRUE(
    tracker.setCloudObservation(10.50, 10.50, error)) << error;
  const auto output = tracker.update(10.50);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
}

TEST(TrajectoryTracker, StationaryFinalHoldIsExactZeroForPositiveDwell)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);

  // 先制造一拍真实非零历史，确认 final hold 不会经由变化率限制残留速度。
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setFreshInputs(tracker, 10.1, 0.0, goal);
  ASSERT_GT(tracker.update(10.1).vx, 0.0);

  auto hold = makeStationaryTrajectory(10.1, 7.1, true);
  hold.reference_path_stamp_sec = 10.0;
  hold.trajectory_id = 8;
  ASSERT_TRUE(
    tracker.setTrajectory(hold, 10.1, error)) << error;
  const double duration = tracker.trajectoryDurationSec();
  ASSERT_GT(duration, 0.0);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), 0.0);

  auto output = tracker.update(10.1);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_TRUE(output.execution_frozen);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);

  for (int step = 1; step <= 30; ++step) {
    const double now = 10.1 + 0.1 * static_cast<double>(step);
    setFreshInputs(tracker, now, 0.0, goal);
    output = tracker.update(now);
    EXPECT_DOUBLE_EQ(output.vx, 0.0);
    EXPECT_DOUBLE_EQ(output.vy, 0.0);
    EXPECT_DOUBLE_EQ(output.wz, 0.0);
    if (0.1 * static_cast<double>(step) < duration - 1.0e-9) {
      EXPECT_FALSE(output.goal_reached);
    }
  }

  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.trajectory_finished);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
  output = tracker.update(13.11);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, StationaryFinalHoldCannotSkipDwellOnCloudTimeout)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  config.cloud_timeout_sec = 0.25;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  auto hold = makeStationaryTrajectory(10.0, 7.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(
    tracker.setTrajectory(hold, 10.0, error)) << error;
  setFreshInputs(tracker, 10.0, 0.0, goal);

  scan_controller::OdometryInput odometry;
  odometry.position = goal;
  odometry.planar_speed = 0.0;
  odometry.stamp_sec = 10.26;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.26, error)) << error;

  auto output = tracker.update(10.26);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kCloudTimeout);
  EXPECT_FALSE(output.trajectory_finished);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), 0.0);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);

  // 超时打断连续驻留后，楼梯冻结终点在点云恢复时仍可收口，
  // 但必须在新鲜点云下重新完整累计驻留时长。
  for (int step = 1; step <= 31; ++step) {
    const double now = 10.26 + 0.1 * static_cast<double>(step);
    setFreshInputs(tracker, now, 0.0, goal);
    output = tracker.update(now);
    EXPECT_DOUBLE_EQ(output.vx, 0.0);
    EXPECT_DOUBLE_EQ(output.vy, 0.0);
    EXPECT_DOUBLE_EQ(output.wz, 0.0);
    if (step < 30) {
      EXPECT_FALSE(output.goal_reached);
    }
  }
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.trajectory_finished);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, StationaryFinalHoldCompletesAfterSubGateDrift)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  auto hold = makeStationaryTrajectory(10.0, 10.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(hold, 10.0, error)) << error;

  scan_controller::ControlOutput output;
  for (int step = 1; step <= 30; ++step) {
    const double now = 10.0 + 0.1 * static_cast<double>(step);
    // planner 只会在 0.06m 内发布保持；严格零速后即使站立策略漂到
    // 0.064m，仍应处于独立的 0.08m 完成门内并连续累计三秒。
    const double distance_xy = step == 1 ? 0.060 : 0.064;
    setFreshInputs(
      tracker, now, 0.0,
      goal + Eigen::Vector3d(distance_xy, 0.0, 0.0));
    output = tracker.update(now);
    EXPECT_DOUBLE_EQ(output.vx, 0.0);
    EXPECT_DOUBLE_EQ(output.vy, 0.0);
    EXPECT_DOUBLE_EQ(output.wz, 0.0);
    if (step < 30) {
      EXPECT_FALSE(output.goal_reached);
    }
  }

  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, StationaryFinalHoldOutsideCompletionGateTimesOut)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.10;
  config.trajectory_expiry_grace_sec = 0.10;
  config.max_yaw_alignment_freeze_sec = 0.30;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  auto hold = makeStationaryTrajectory(10.0, 10.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(hold, 10.0, error)) << error;

  scan_controller::ControlOutput output;
  for (int step = 1; step <= 35; ++step) {
    const double now = 10.0 + 0.1 * static_cast<double>(step);
    setFreshInputs(
      tracker, now, 0.0,
      goal + Eigen::Vector3d(
        config.finish_distance_xy + 0.001, 0.0, 0.0));
    output = tracker.update(now);
    EXPECT_DOUBLE_EQ(output.vx, 0.0);
    EXPECT_DOUBLE_EQ(output.vy, 0.0);
    EXPECT_DOUBLE_EQ(output.wz, 0.0);
    EXPECT_FALSE(output.goal_reached);
  }

  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kTrajectoryTimeout);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), 0.0);
}

TEST(TrajectoryTracker, StationaryFinalHoldUsesYawRateInsteadOfStanceAngularNorm)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  auto hold = makeStationaryTrajectory(10.0, 10.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(hold, 10.0, error)) << error;
  const double stationary_yaw_inside_finish =
    0.5 * (config.terminal_yaw_control_deadband + config.finish_yaw_error);
  ASSERT_LT(stationary_yaw_inside_finish, config.finish_yaw_error);

  auto update_motion = [&](double now, double yaw_rate) {
      scan_controller::OdometryInput odometry;
      odometry.position = goal;
      // stationary final 不使用 moving final 的控制死区主动转向；只要仍在
      // 完成 yaw 门内，就保持严格零速并按 |wz| 认证连续驻留。
      odometry.yaw = stationary_yaw_inside_finish;
      // 模拟四足站立策略的 roll/pitch 微摆；stationary final 只应由
      // cmd_vel 对应的机体系 yaw rate 判断是否仍在转向。
      odometry.angular_speed = 0.25;
      odometry.yaw_rate = yaw_rate;
      odometry.stamp_sec = now;
      EXPECT_TRUE(tracker.setOdometry(odometry, now, error)) << error;
      EXPECT_TRUE(tracker.setCloudObservation(now, now, error)) << error;
      return tracker.update(now);
    };

  scan_controller::ControlOutput output;
  for (int step = 1; step <= 15; ++step) {
    output = update_motion(10.0 + 0.1 * static_cast<double>(step), 0.05);
    EXPECT_FALSE(output.goal_reached);
    EXPECT_DOUBLE_EQ(output.vx, 0.0);
    EXPECT_DOUBLE_EQ(output.vy, 0.0);
    EXPECT_DOUBLE_EQ(output.wz, 0.0);
  }
  EXPECT_GT(tracker.executionTimeSec(), 1.4);

  // 真实 yaw 仍在转动时必须清零已经累计的驻留时间。
  output = update_motion(11.6, config.finish_yaw_rate + 0.01);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), 0.0);

  for (int step = 1; step <= 30; ++step) {
    output = update_motion(
      11.6 + 0.1 * static_cast<double>(step), 0.05);
    EXPECT_DOUBLE_EQ(output.vx, 0.0);
    EXPECT_DOUBLE_EQ(output.vy, 0.0);
    EXPECT_DOUBLE_EQ(output.wz, 0.0);
    if (step < 30) {
      EXPECT_FALSE(output.goal_reached);
    }
  }
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, StationaryFinalHoldOutsideYawGateTimesOut)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.10;
  config.trajectory_expiry_grace_sec = 0.10;
  config.max_yaw_alignment_freeze_sec = 0.30;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  auto hold = makeStationaryTrajectory(10.0, 10.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(hold, 10.0, error)) << error;

  setFreshInputs(tracker, 10.10, 0.0, goal);
  auto output = tracker.update(10.10);
  ASSERT_GT(tracker.executionTimeSec(), 0.0);
  ASSERT_FALSE(output.goal_reached);

  for (int step = 2; step <= 35; ++step) {
    const double now = 10.0 + 0.1 * static_cast<double>(step);
    setFreshInputs(
      tracker, now, config.finish_yaw_error + 0.001, goal);
    output = tracker.update(now);
    EXPECT_DOUBLE_EQ(output.vx, 0.0);
    EXPECT_DOUBLE_EQ(output.vy, 0.0);
    EXPECT_DOUBLE_EQ(output.wz, 0.0);
    EXPECT_FALSE(output.goal_reached);
    if (step == 2) {
      EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), 0.0);
    }
  }

  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kTrajectoryTimeout);
}

TEST(TrajectoryTracker, StationaryFinalYawMotionTimesOutAtHardExpiry)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.10;
  config.trajectory_expiry_grace_sec = 0.10;
  config.max_yaw_alignment_freeze_sec = 0.30;
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  auto hold = makeStationaryTrajectory(10.0, 10.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(hold, 10.0, error)) << error;

  scan_controller::ControlOutput output;
  for (int step = 1; step <= 35; ++step) {
    const double now = 10.0 + 0.1 * static_cast<double>(step);
    scan_controller::OdometryInput odometry;
    odometry.position = goal;
    odometry.angular_speed = config.finish_yaw_rate + 0.01;
    odometry.yaw_rate = config.finish_yaw_rate + 0.01;
    odometry.stamp_sec = now;
    ASSERT_TRUE(tracker.setOdometry(odometry, now, error)) << error;
    ASSERT_TRUE(tracker.setCloudObservation(now, now, error)) << error;
    output = tracker.update(now);
    EXPECT_DOUBLE_EQ(output.vx, 0.0);
    EXPECT_DOUBLE_EQ(output.vy, 0.0);
    EXPECT_DOUBLE_EQ(output.wz, 0.0);
  }

  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kTrajectoryTimeout);
  EXPECT_FALSE(output.goal_reached);
}

TEST(TrajectoryTracker, StationaryFinalHoldUsesBoundedHardExpiryAfterUnstableRelease)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.10;
  config.trajectory_expiry_grace_sec = 0.10;
  config.max_yaw_alignment_freeze_sec = 5.0;
  config.max_control_dt_sec = 0.20;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  auto hold = makeStationaryTrajectory(10.0, 10.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(hold, 10.0, error)) << error;

  // 先模拟分阶段解锁：位置门连续失配到 soft-expiry 之后，驻留计时应为零，
  // 但仍处在有界 hard-expiry 内，不能提前退化成 TRAJECTORY_TIMEOUT。
  scan_controller::ControlOutput output;
  for (int step = 0; step <= 32; ++step) {
    const double now = 10.0 + 0.1 * static_cast<double>(step);
    setFreshInputs(
      tracker, now, 0.0,
      goal + Eigen::Vector3d(config.finish_distance_xy + 0.01, 0.0, 0.0));
    output = tracker.update(now);
  }
  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), 0.0);

  // 解锁稳定后仍必须重新连续满足完整正时长，且可在 hard-expiry 前收口。
  for (int step = 1; step <= 30; ++step) {
    const double now = 13.2 + 0.1 * static_cast<double>(step);
    setFreshInputs(tracker, now, 0.0, goal);
    output = tracker.update(now);
  }
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
}

TEST(TrajectoryTracker, MovingFinalRequiresFreshCloudAtNominalEnd)
{
  auto config = makeConfig();
  config.cloud_timeout_sec = 0.25;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  setFreshInputs(tracker, 13.0, 0.0, goal);
  ASSERT_GE(tracker.executionTimeSec(), tracker.trajectoryDurationSec());

  scan_controller::OdometryInput odometry;
  odometry.position = goal;
  odometry.planar_speed = 0.0;
  odometry.stamp_sec = 13.26;
  ASSERT_TRUE(tracker.setOdometry(odometry, 13.26, error)) << error;

  auto output = tracker.update(13.26);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kCloudTimeout);
  EXPECT_FALSE(output.trajectory_finished);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
  EXPECT_GE(tracker.executionTimeSec(), tracker.trajectoryDurationSec());

  // 名义时间和全部物理门都已满足也不能用过期点云收口；恢复新鲜点云后
  // 仍要完成严格制动与连续稳定驻留，不能依赖一个恢复瞬间的样本。
  setFreshInputs(tracker, 13.27, 0.0, goal);
  output = tracker.update(13.27);
  EXPECT_FALSE(output.goal_reached);
  output = waitForMovingFinalStableCompletion(
    tracker, 13.32, 0.0, goal);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_TRUE(output.trajectory_finished);
}

TEST(TrajectoryTracker, FinalCompletionUsesFullPathEndpointAndBodyHeight)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d local_spline_goal =
    trajectoryFinalPosition(trajectory);
  const Eigen::Vector3d path_ground_goal(4.0, 1.0, 0.5);
  const Eigen::Vector3d path_base_goal =
    path_ground_goal + Eigen::Vector3d(0.0, 0.0, 0.30);
  setFinalReferencePath(tracker, 10.0, path_base_goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  setFreshInputs(tracker, 13.0, 0.0, local_spline_goal);

  auto output = tracker.update(13.01);
  EXPECT_FALSE(output.goal_reached);

  // Path 的 z 是地面高度，停在原始 z 而没有增加一次 body_height 不能完成。
  setFreshInputs(tracker, 13.02, 0.0, path_ground_goal);
  output = tracker.update(13.02);
  EXPECT_FALSE(output.goal_reached);

  setFreshInputs(tracker, 13.03, 0.0, path_base_goal);
  output = tracker.update(13.03);
  EXPECT_FALSE(output.goal_reached);
  output = waitForMovingFinalStableCompletion(
    tracker, 13.08, 0.0, path_base_goal);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, FinalCompletionUsesLastNondegeneratePathXYYaw)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const std::vector<Eigen::Vector3d> path{
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.2),
    Eigen::Vector3d(1.0, 1.0, 0.4),
    // 末段只有高度变化，终端 yaw 必须继续使用前一条 +Y 线段。
    Eigen::Vector3d(1.0, 1.0, 0.7),
  };
  ASSERT_TRUE(tracker.setReferencePath(path, 10.0, 10.0, error)) << error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  const Eigen::Vector3d base_goal(1.0, 1.0, 1.0);
  setFreshInputs(tracker, 13.0, 0.0, base_goal);

  auto output = tracker.update(13.01);
  EXPECT_FALSE(output.goal_reached);

  setFreshInputs(tracker, 13.02, M_PI_2, base_goal);
  EXPECT_FALSE(tracker.update(13.02).goal_reached);
  setFreshInputs(tracker, 13.08, M_PI_2, base_goal);
  output = tracker.update(13.08);
  EXPECT_FALSE(output.goal_reached);
  output = waitForMovingFinalStableCompletion(
    tracker, 13.13, M_PI_2, base_goal);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, ExplicitTerminalYawOverridesLastPathSegment)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const std::vector<Eigen::Vector3d> path{
    Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 0.0),
  };
  ASSERT_TRUE(
    tracker.setReferencePath(
      path, 10.0, 10.0, error, -M_PI_2)) << error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  const Eigen::Vector3d base_goal(0.0, 0.0, 0.3);
  setFreshInputs(tracker, 13.0, 0.0, base_goal);
  EXPECT_FALSE(tracker.update(13.01).goal_reached);

  setFreshInputs(tracker, 13.02, -M_PI_2, base_goal);
  EXPECT_FALSE(tracker.update(13.02).goal_reached);
  setFreshInputs(tracker, 13.08, -M_PI_2, base_goal);
  auto output = tracker.update(13.08);
  EXPECT_FALSE(output.goal_reached);
  output = waitForMovingFinalStableCompletion(
    tracker, 13.13, -M_PI_2, base_goal);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, FinalTrajectoryActivelyAlignsExplicitTerminalYaw)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  std::string error;
  const std::vector<Eigen::Vector3d> path{
    Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 0.0),
  };
  ASSERT_TRUE(
    tracker.setReferencePath(
      path, 10.0, 10.0, error, -M_PI_2)) << error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  const Eigen::Vector3d base_goal(0.0, 0.0, 0.3);
  setFreshInputs(tracker, 13.0, 0.0, base_goal);

  const auto aligning = tracker.update(13.01);
  EXPECT_EQ(
    aligning.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(aligning.execution_frozen);
  EXPECT_DOUBLE_EQ(aligning.vx, 0.0);
  EXPECT_DOUBLE_EQ(aligning.vy, 0.0);
  // 首拍必须给下游 policy writer 一个可观测的严格全零，清除旧限速历史。
  EXPECT_DOUBLE_EQ(aligning.wz, 0.0);
  EXPECT_FALSE(aligning.goal_reached);

  setFreshInputs(tracker, 13.08, 0.0, base_goal);
  const auto rotating = tracker.update(13.08);
  EXPECT_EQ(
    rotating.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_LT(rotating.wz, 0.0);
  EXPECT_GE(
    rotating.wz,
    -makeConfig().terminal_max_yaw_acc * 0.07 - 1.0e-9);

  scan_controller::OdometryInput odometry;
  odometry.position = base_goal;
  odometry.yaw = -M_PI_2;
  odometry.stamp_sec = 13.09;
  ASSERT_TRUE(tracker.setOdometry(odometry, 13.09, error)) << error;
  ASSERT_TRUE(tracker.setCloudObservation(13.09, 13.09, error)) << error;
  auto completed = tracker.update(13.09);
  EXPECT_FALSE(completed.goal_reached);
  completed = waitForMovingFinalStableCompletion(
    tracker, 13.14, -M_PI_2, base_goal);
  EXPECT_EQ(
    completed.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(completed.goal_reached);
  EXPECT_DOUBLE_EQ(completed.vx, 0.0);
  EXPECT_DOUBLE_EQ(completed.vy, 0.0);
  EXPECT_DOUBLE_EQ(completed.wz, 0.0);
}

TEST(TrajectoryTracker, TerminalYawPositionHoldCountersPolicyDrift)
{
  auto config = makeConfig();
  // phase238 在 0.45rad/s 终点转向下实测漂移约 0.23m；0.30m 外圈只保留
  // terminal yaw 状态，不参与仍为 0.08m 的严格完成认证。提高增益和上限
  // 是为了越过原 Go2-X5 policy 的低速弱响应区，不是放宽成功门。
  config.terminal_capture_release_distance_xy = 0.30;
  config.terminal_position_hold_gain = 2.00;
  config.terminal_position_hold_max_speed = 0.15;
  config.max_yaw_alignment_freeze_sec = 12.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal, -M_PI_2);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  const Eigen::Vector3d capture_position =
    goal - Eigen::Vector3d(
    config.terminal_capture_entry_distance_xy - 0.01, 0.0, 0.0);
  setFreshInputs(tracker, 10.10, 0.0, capture_position);
  const auto braking = tracker.update(10.10);
  ASSERT_TRUE(braking.execution_frozen);
  EXPECT_DOUBLE_EQ(braking.vx, 0.0);
  EXPECT_DOUBLE_EQ(braking.vy, 0.0);
  EXPECT_DOUBLE_EQ(braking.wz, 0.0);

  // 模拟下游四足策略在纯 yaw 命令下向终点反方向漂移。全零制动窗口结束
  // 后，控制器继续冻结 B-spline 时间，但应给出朝终点的有界位置保持命令。
  const Eigen::Vector3d drifted_position =
    goal - Eigen::Vector3d(0.23, 0.0, 0.0);
  setFreshInputs(tracker, 10.18, 0.0, drifted_position);
  const auto compensating = tracker.update(10.18);
  EXPECT_EQ(
    compensating.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(compensating.execution_frozen);
  EXPECT_GT(compensating.vx, 0.0);
  EXPECT_DOUBLE_EQ(compensating.vy, 0.0);
  EXPECT_LT(compensating.wz, 0.0);
  EXPECT_LE(
    std::hypot(compensating.vx, compensating.vy),
    config.terminal_position_hold_max_speed + 1.0e-9);
  EXPECT_FALSE(compensating.goal_reached);

  // 补偿把机体留在严格完成门内、terminal yaw 收敛后，命令必须重新变为
  // 三轴严格零，并仍需完整的连续驻留，不能因位置保持而提前报到达。
  scan_controller::ControlOutput completed;
  for (int step = 1; step <= 12; ++step) {
    const double now = 10.18 + 0.10 * static_cast<double>(step);
    setFreshInputs(tracker, now, -M_PI_2, goal);
    completed = tracker.update(now);
    if (completed.goal_reached) {
      break;
    }
  }
  EXPECT_TRUE(completed.goal_reached);
  EXPECT_EQ(completed.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_DOUBLE_EQ(completed.vx, 0.0);
  EXPECT_DOUBLE_EQ(completed.vy, 0.0);
  EXPECT_DOUBLE_EQ(completed.wz, 0.0);
}

TEST(TrajectoryTracker, FinalApproachKeepsMinimumEffectiveGaitSpeed)
{
  auto config = makeConfig();
  config.terminal_capture_release_distance_xy = 0.30;
  config.terminal_approach_min_speed = 0.22;
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  trajectory.control_points(0, 4) = 1.5;
  trajectory.control_points(0, 5) = 1.5;
  const Eigen::Vector3d local_safe_endpoint =
    trajectoryFinalPosition(trajectory);
  const Eigen::Vector3d task_goal =
    local_safe_endpoint + Eigen::Vector3d(0.058, 0.0, 0.0);
  const Eigen::Vector3d robot_position =
    local_safe_endpoint - Eigen::Vector3d(0.037, 0.0, 0.0);
  setFinalReferencePath(tracker, 10.0, task_goal);
  setFreshInputs(tracker, 12.99, 0.0, robot_position);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 12.99, error)) << error;

  // 首拍建立控制时钟；第二拍已到局部安全端点附近，但相对任务终点仍有
  // 9.5cm，必须持续给出策略可执行的最小步态，不能在 7.5cm 捕获门外渐停。
  const auto initial = tracker.update(12.99);
  EXPECT_EQ(initial.state, scan_controller::ControllerState::kTracking);
  setFreshInputs(tracker, 13.09, 0.0, robot_position);
  const auto approaching = tracker.update(13.09);
  EXPECT_EQ(approaching.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(approaching.execution_frozen);
  EXPECT_NEAR(
    std::hypot(approaching.vx, approaching.vy),
    config.terminal_approach_min_speed, 1.0e-9);
  EXPECT_GT(approaching.vx, 0.0);
  EXPECT_DOUBLE_EQ(approaching.vy, 0.0);
  EXPECT_FALSE(approaching.goal_reached);
}

TEST(TrajectoryTracker, CompositeTerminalGaitRespectsSmallerLateralLimit)
{
  auto config = makeConfig();
  config.max_vx = 0.55;
  config.max_vy = 0.15;
  config.terminal_capture_release_distance_xy = 0.30;
  config.terminal_approach_min_speed = 0.22;
  config.heading_error_threshold = 1.00;
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  trajectory.control_points(0, 4) = 1.5;
  trajectory.control_points(0, 5) = 1.5;
  const Eigen::Vector3d local_safe_endpoint =
    trajectoryFinalPosition(trajectory);
  const Eigen::Vector3d task_goal =
    local_safe_endpoint + Eigen::Vector3d(0.058, 0.0, 0.0);
  const Eigen::Vector3d robot_position =
    local_safe_endpoint - Eigen::Vector3d(0.037, 0.0, 0.0);
  setFinalReferencePath(tracker, 10.0, task_goal, M_PI_4);
  setFreshInputs(tracker, 12.99, M_PI_4, robot_position);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 12.99, error)) << error;

  // 0.22m/s 是二维合成最低步态，可以高于携臂 vy=0.15m/s 单轴上限。
  // 世界系前向轨迹旋转到 45° 机体系后，vy 必须继续被硬裁剪，而 vx
  // 保留足够的有效步态分量，不能因为较小的横移上限拒绝整份配置。
  EXPECT_EQ(
    tracker.update(12.99).state,
    scan_controller::ControllerState::kTracking);
  setFreshInputs(tracker, 13.09, M_PI_4, robot_position);
  const auto approaching = tracker.update(13.09);
  EXPECT_EQ(approaching.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(approaching.execution_frozen);
  EXPECT_NEAR(
    approaching.vx,
    config.terminal_approach_min_speed / std::sqrt(2.0), 1.0e-9);
  EXPECT_DOUBLE_EQ(approaching.vy, -config.max_vy);
  EXPECT_LE(std::abs(approaching.vx), config.max_vx);
  EXPECT_LE(std::abs(approaching.vy), config.max_vy);
  EXPECT_FALSE(approaching.goal_reached);
}

TEST(TrajectoryTracker, TerminalRecoveryKeepsMinimumEffectiveGaitSpeed)
{
  auto config = makeConfig();
  config.terminal_capture_release_distance_xy = 0.30;
  config.terminal_approach_min_speed = 0.22;
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  trajectory.control_points(0, 4) = 1.5;
  trajectory.control_points(0, 5) = 1.5;
  const Eigen::Vector3d local_safe_endpoint =
    trajectoryFinalPosition(trajectory);
  const Eigen::Vector3d task_goal =
    local_safe_endpoint + Eigen::Vector3d(0.058, 0.0, 0.0);
  setFinalReferencePath(tracker, 10.0, task_goal);

  // 先进入 7.5cm 捕获内门，锁存终点姿态模式并执行严格零速制动。
  const Eigen::Vector3d captured_position = task_goal - Eigen::Vector3d(
    config.terminal_capture_entry_distance_xy - 0.009, 0.0, 0.0);
  setFreshInputs(tracker, 12.99, 0.0, captured_position);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 12.99, error)) << error;
  const auto captured = tracker.update(12.99);
  ASSERT_TRUE(captured.execution_frozen);
  EXPECT_DOUBLE_EQ(captured.vx, 0.0);
  EXPECT_DOUBLE_EQ(captured.vy, 0.0);
  EXPECT_FALSE(captured.goal_reached);

  // 模拟 phase242：转向耦合使机体漂到约 9.5cm，姿态和速度已稳定，控制器
  // 因而解除平移制动并恢复同一条 final B-spline。此时虽已锁存捕获状态，
  // 回收命令仍必须越过原策略低速弱响应区，不能再次降到约 0.04m/s。
  const Eigen::Vector3d drifted_position =
    task_goal - Eigen::Vector3d(0.095, 0.0, 0.0);
  setFreshInputs(tracker, 13.09, 0.0, drifted_position);
  const auto recovering = tracker.update(13.09);
  EXPECT_EQ(recovering.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(recovering.execution_frozen);
  EXPECT_NEAR(
    std::hypot(recovering.vx, recovering.vy),
    config.terminal_approach_min_speed, 1.0e-9);
  EXPECT_GT(recovering.vx, 0.0);
  EXPECT_DOUBLE_EQ(recovering.vy, 0.0);
  EXPECT_FALSE(recovering.goal_reached);
}

TEST(TrajectoryTracker, MovingFinalYawFeedbackUsesNarrowerControlDeadband)
{
  auto config = makeConfig();
  config.terminal_capture_brake_hold_sec = 0.05;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  const double yaw_inside_finish = 0.181;
  ASSERT_GT(yaw_inside_finish, config.terminal_yaw_control_deadband);
  ASSERT_LT(yaw_inside_finish, config.finish_yaw_error);
  setFreshInputs(tracker, 10.10, yaw_inside_finish, goal);

  // 首拍仍执行终点捕获的严格零速制动，但控制状态已经表明仍需收敛 yaw。
  const auto captured = tracker.update(10.10);
  EXPECT_EQ(captured.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(captured.execution_frozen);
  EXPECT_DOUBLE_EQ(captured.vx, 0.0);
  EXPECT_DOUBLE_EQ(captured.vy, 0.0);
  EXPECT_DOUBLE_EQ(captured.wz, 0.0);
  EXPECT_FALSE(captured.goal_reached);

  // 误差虽已进入 0.20rad 完成门，但尚未进入 0.18rad 控制死区；moving
  // final 必须继续原地纠偏，不能把 RL 稳态停在验收边界。
  setFreshInputs(tracker, 10.16, yaw_inside_finish, goal);
  const auto correcting = tracker.update(10.16);
  EXPECT_EQ(correcting.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_TRUE(correcting.execution_frozen);
  EXPECT_DOUBLE_EQ(correcting.vx, 0.0);
  EXPECT_DOUBLE_EQ(correcting.vy, 0.0);
  EXPECT_LT(correcting.wz, 0.0);
  EXPECT_FALSE(correcting.goal_reached);
  EXPECT_LT(tracker.executionTimeSec(), tracker.trajectoryDurationSec());
}

TEST(TrajectoryTracker, MovingFinalInsideYawDeadbandStopsWithoutInstantGoal)
{
  auto config = makeConfig();
  config.terminal_capture_brake_hold_sec = 0.05;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  constexpr double yaw_inside_control_deadband = 0.179;
  ASSERT_LT(
    yaw_inside_control_deadband, config.terminal_yaw_control_deadband);
  setFreshInputs(tracker, 10.10, yaw_inside_control_deadband, goal);
  const auto captured = tracker.update(10.10);
  EXPECT_TRUE(captured.execution_frozen);
  EXPECT_FALSE(captured.goal_reached);
  EXPECT_DOUBLE_EQ(captured.vx, 0.0);
  EXPECT_DOUBLE_EQ(captured.vy, 0.0);
  EXPECT_DOUBLE_EQ(captured.wz, 0.0);

  setFreshInputs(tracker, 10.16, yaw_inside_control_deadband, goal);
  const auto holding = tracker.update(10.16);
  EXPECT_EQ(holding.state, scan_controller::ControllerState::kTracking);
  EXPECT_TRUE(holding.execution_frozen);
  EXPECT_FALSE(holding.goal_reached);
  EXPECT_DOUBLE_EQ(holding.vx, 0.0);
  EXPECT_DOUBLE_EQ(holding.vy, 0.0);
  EXPECT_DOUBLE_EQ(holding.wz, 0.0);
}

TEST(TrajectoryTracker, MovingFinalCaptureEntryIsNarrowerThanCompletionGate)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  const double inside_completion_outside_capture = 0.5 * (
    config.terminal_capture_entry_distance_xy + config.finish_distance_xy);
  setFreshInputs(
    tracker, 10.10, 0.0,
    goal - Eigen::Vector3d(inside_completion_outside_capture, 0.0, 0.0));
  const auto not_captured = tracker.update(10.10);
  EXPECT_FALSE(not_captured.execution_frozen);
  EXPECT_NE(not_captured.vx, 0.0);
  EXPECT_FALSE(not_captured.goal_reached);

  setFreshInputs(
    tracker, 10.20, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_entry_distance_xy - 0.001, 0.0, 0.0));
  const auto captured = tracker.update(10.20);
  EXPECT_TRUE(captured.execution_frozen);
  EXPECT_DOUBLE_EQ(captured.vx, 0.0);
  EXPECT_DOUBLE_EQ(captured.vy, 0.0);
  EXPECT_DOUBLE_EQ(captured.wz, 0.0);
  EXPECT_FALSE(captured.goal_reached);
}

TEST(TrajectoryTracker, NominalFinalInsideOuterGateRecentersBeforeDwell)
{
  auto config = makeConfig();
  config.terminal_capture_brake_hold_sec = 0.05;
  config.max_ax = 10.0;
  config.max_ay = 10.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);

  // 让 B-spline 名义时间先结束，并把机体放在 8cm 完成外门以内、7.5cm
  // 零速保持门以外。即使真实速度尚未稳定，也必须先进入严格零速制动，
  // 不能继续用 final 最低步态速度穿过完成圈。
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  ASSERT_GE(tracker.executionTimeSec(), tracker.trajectoryDurationSec());
  const double completion_ring_distance = 0.5 * (
    config.terminal_capture_zero_hold_distance_xy +
    config.finish_distance_xy);
  const Eigen::Vector3d completion_ring_position =
    goal - Eigen::Vector3d(completion_ring_distance, 0.0, 0.0);
  setFreshInputs(
    tracker, 13.01, 0.0, completion_ring_position,
    config.finish_speed + 0.05);
  const auto braking = tracker.update(13.01);
  EXPECT_TRUE(braking.execution_frozen);
  EXPECT_DOUBLE_EQ(braking.vx, 0.0);
  EXPECT_DOUBLE_EQ(braking.vy, 0.0);
  EXPECT_DOUBLE_EQ(braking.wz, 0.0);
  EXPECT_FALSE(braking.goal_reached);

  // 制动首窗结束且姿态/速度稳定后，机体仍在 7.5~8cm 环带，因此应解除
  // 平移冻结，恢复碰撞安全 B-spline 的最低有效步态；0.15m/s 位置保持对
  // 原 Go2-X5 policy 太弱，会被卡死 watchdog 如实判失败。
  setFreshInputs(tracker, 13.07, 0.0, completion_ring_position);
  const auto recentering = tracker.update(13.07);
  EXPECT_FALSE(recentering.execution_frozen);
  EXPECT_GT(recentering.vx, 0.0);
  EXPECT_DOUBLE_EQ(recentering.vy, 0.0);
  EXPECT_DOUBLE_EQ(recentering.wz, 0.0);
  EXPECT_GE(
    std::hypot(recentering.vx, recentering.vy),
    config.terminal_approach_min_speed);
  EXPECT_LE(
    std::hypot(recentering.vx, recentering.vy),
    config.max_vx);
  EXPECT_FALSE(recentering.goal_reached);

  // 一旦真正进入 5.5cm 内门，必须再次清空历史并严格刹停，不能用最低
  // 步态穿过目标点。
  const Eigen::Vector3d inner_position =
    goal - Eigen::Vector3d(
    config.terminal_capture_entry_distance_xy - 0.001, 0.0, 0.0);
  setFreshInputs(tracker, 13.08, 0.0, inner_position);
  const auto inner_braking = tracker.update(13.08);
  EXPECT_TRUE(inner_braking.execution_frozen);
  EXPECT_DOUBLE_EQ(inner_braking.vx, 0.0);
  EXPECT_DOUBLE_EQ(inner_braking.vy, 0.0);
  EXPECT_DOUBLE_EQ(inner_braking.wz, 0.0);
  EXPECT_FALSE(inner_braking.goal_reached);
}

TEST(TrajectoryTracker, FinalTrajectoryBrakesInsideGoalBeforeNominalEnd)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  // 先在终点门外产生一拍非零跟踪命令，模拟 v14 接近终点时的速度历史。
  setFreshInputs(
    tracker, 10.10, 0.0,
    goal - Eigen::Vector3d(config.finish_distance_xy + 0.02, 0.0, 0.0));
  const auto approaching = tracker.update(10.10);
  EXPECT_FALSE(approaching.execution_frozen);
  EXPECT_NE(approaching.vx, 0.0);

  scan_controller::OdometryInput odometry;
  odometry.position =
    goal - Eigen::Vector3d(
    config.terminal_capture_entry_distance_xy - 0.01, 0.0, 0.0);
  odometry.yaw = 0.096;
  odometry.planar_speed = config.finish_speed + 0.10;
  odometry.angular_speed = config.finish_angular_speed + 0.05;
  odometry.stamp_sec = 10.20;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.20, error)) << error;
  ASSERT_TRUE(tracker.setCloudObservation(10.20, 10.20, error)) << error;

  const auto captured = tracker.update(10.20);

  ASSERT_LT(tracker.executionTimeSec(), tracker.trajectoryDurationSec());
  EXPECT_EQ(captured.state, scan_controller::ControllerState::kTracking);
  EXPECT_TRUE(captured.execution_frozen);
  EXPECT_FALSE(captured.goal_reached);
  EXPECT_DOUBLE_EQ(captured.vx, 0.0);
  EXPECT_DOUBLE_EQ(captured.vy, 0.0);
  EXPECT_LE(std::abs(captured.wz), config.terminal_max_yaw_rate);
}

TEST(TrajectoryTracker, TerminalCaptureRecoversTranslationInsideReleaseBand)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  setFreshInputs(
    tracker, 10.10, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_entry_distance_xy - 0.01, 0.0, 0.0),
    config.finish_speed + 0.05);
  const auto entered = tracker.update(10.10);
  EXPECT_TRUE(entered.execution_frozen);
  EXPECT_DOUBLE_EQ(entered.vx, 0.0);
  EXPECT_FALSE(entered.goal_reached);

  // 轻微漂出完成门但仍在 0.12m 外圈内，且姿态与三类速度均已稳定时，
  // 只恢复 SCAN B-spline 平移闭环，避免永久停在活性死区。
  setFreshInputs(
    tracker, 10.20, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_release_distance_xy - 0.01, 0.0, 0.0));
  const auto recovering = tracker.update(10.20);
  EXPECT_FALSE(recovering.execution_frozen);
  EXPECT_NE(recovering.vx, 0.0);
  EXPECT_FALSE(recovering.goal_reached);

  setFreshInputs(
    tracker, 10.30, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_release_distance_xy + 0.001, 0.0, 0.0));
  const auto released = tracker.update(10.30);
  EXPECT_FALSE(released.execution_frozen);
  EXPECT_NE(released.vx, 0.0);
  EXPECT_FALSE(released.goal_reached);
}

TEST(TrajectoryTracker, TerminalCaptureZeroHoldHysteresisAbsorbsSmallDrift)
{
  auto config = makeConfig();
  config.terminal_capture_brake_hold_sec = 0.05;
  config.terminal_capture_stable_dwell_sec = 0.20;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  const Eigen::Vector3d captured_position =
    goal - Eigen::Vector3d(
    config.terminal_capture_entry_distance_xy - 0.005, 0.0, 0.0);
  setFreshInputs(
    tracker, 10.10, 0.0, captured_position,
    config.finish_speed + 0.05);
  const auto captured = tracker.update(10.10);
  ASSERT_TRUE(captured.execution_frozen);
  EXPECT_DOUBLE_EQ(captured.vx, 0.0);
  EXPECT_DOUBLE_EQ(captured.vy, 0.0);

  const double drift_distance =
    0.5 * (
    config.terminal_capture_entry_distance_xy +
    config.terminal_capture_zero_hold_distance_xy);
  const Eigen::Vector3d drifted_position =
    goal - Eigen::Vector3d(drift_distance, 0.0, 0.0);
  ASSERT_GT(
    drift_distance, config.terminal_capture_entry_distance_xy);
  ASSERT_LT(
    drift_distance, config.terminal_capture_zero_hold_distance_xy);

  setFreshInputs(tracker, 10.16, 0.0, drifted_position);
  const auto first_hold = tracker.update(10.16);
  EXPECT_TRUE(first_hold.execution_frozen);
  EXPECT_DOUBLE_EQ(first_hold.vx, 0.0);
  EXPECT_DOUBLE_EQ(first_hold.vy, 0.0);
  EXPECT_FALSE(first_hold.goal_reached);

  setFreshInputs(tracker, 10.26, 0.0, drifted_position);
  EXPECT_FALSE(tracker.update(10.26).goal_reached);
  setFreshInputs(tracker, 10.36, 0.0, drifted_position);
  const auto completed = tracker.update(10.36);
  EXPECT_TRUE(completed.goal_reached);
  EXPECT_EQ(
    completed.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_DOUBLE_EQ(completed.vx, 0.0);
  EXPECT_DOUBLE_EQ(completed.vy, 0.0);
  EXPECT_DOUBLE_EQ(completed.wz, 0.0);
}

TEST(TrajectoryTracker, TerminalCaptureKeepsFrozenAndCorrectsDriftWhileMoving)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  setFreshInputs(
    tracker, 10.10, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_entry_distance_xy - 0.01, 0.0, 0.0),
    config.finish_speed + 0.05);
  EXPECT_TRUE(tracker.update(10.10).execution_frozen);

  setFreshInputs(
    tracker, 10.20, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_release_distance_xy - 0.01, 0.0, 0.0),
    config.finish_speed + 0.01);
  const auto retained = tracker.update(10.20);
  EXPECT_TRUE(retained.execution_frozen);
  // 轨迹名义时间继续冻结，但严格制动首窗结束后应以受限闭环把真实机体
  // 拉回终点；继续要求 vx=vy=0 会复现 phase237 的捕获圈往返死循环。
  EXPECT_GT(retained.vx, 0.0);
  EXPECT_DOUBLE_EQ(retained.vy, 0.0);
  EXPECT_LE(retained.vx, config.terminal_position_hold_max_speed);
  EXPECT_FALSE(retained.goal_reached);
}

TEST(TrajectoryTracker, TerminalCaptureClearsOppositeYawCommandHistory)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  scan_controller::ControlOutput approaching;
  for (int step = 1; step <= 5; ++step) {
    const double now = 10.0 + 0.1 * static_cast<double>(step);
    setFreshInputs(
      tracker, now, -1.0,
      goal - Eigen::Vector3d(
        config.terminal_capture_release_distance_xy + 0.08, 0.0, 0.0));
    approaching = tracker.update(now);
  }
  ASSERT_GT(approaching.wz, config.terminal_max_yaw_rate);

  // 进入终点门时 terminal yaw 已在当前 yaw 的反方向。捕获必须丢弃上一拍
  // 的正向角速度命令历史，从零开始生成受限的负向制动命令。
  scan_controller::OdometryInput odometry;
  odometry.position =
    goal - Eigen::Vector3d(
    config.terminal_capture_entry_distance_xy - 0.01, 0.0, 0.0);
  odometry.yaw = 0.80;
  odometry.angular_speed = 0.40;
  odometry.stamp_sec = 10.60;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.60, error)) << error;
  ASSERT_TRUE(tracker.setCloudObservation(10.60, 10.60, error)) << error;

  const auto captured = tracker.update(10.60);
  EXPECT_TRUE(captured.execution_frozen);
  EXPECT_EQ(captured.state, scan_controller::ControllerState::kAligningYaw);
  EXPECT_DOUBLE_EQ(captured.wz, 0.0);
  EXPECT_DOUBLE_EQ(captured.vx, 0.0);
  EXPECT_DOUBLE_EQ(captured.vy, 0.0);
  EXPECT_FALSE(captured.goal_reached);

  setFreshInputs(tracker, 10.64, 0.80, odometry.position);
  const auto holding_zero = tracker.update(10.64);
  EXPECT_DOUBLE_EQ(holding_zero.vx, 0.0);
  EXPECT_DOUBLE_EQ(holding_zero.vy, 0.0);
  EXPECT_DOUBLE_EQ(holding_zero.wz, 0.0);

  setFreshInputs(tracker, 10.67, 0.80, odometry.position);
  const auto reversing = tracker.update(10.67);
  EXPECT_LT(reversing.wz, 0.0);
  EXPECT_GE(
    reversing.wz,
    -config.terminal_max_yaw_acc * 0.03 - 1.0e-9);
  EXPECT_LE(std::abs(reversing.wz), config.terminal_max_yaw_rate);
}

TEST(TrajectoryTracker, TerminalRecoveryKeepsFullPathYawWhileResumingSplineXY)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal, M_PI);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  setFreshInputs(
    tracker, 10.10, M_PI - 0.10,
    goal - Eigen::Vector3d(
      config.terminal_capture_entry_distance_xy - 0.01, 0.0, 0.0),
    config.finish_speed + 0.05);
  const auto captured = tracker.update(10.10);
  ASSERT_TRUE(captured.execution_frozen);
  ASSERT_DOUBLE_EQ(captured.wz, 0.0);

  setFreshInputs(
    tracker, 10.20, M_PI - 0.10,
    goal - Eigen::Vector3d(config.finish_distance_xy + 0.01, 0.0, 0.0));
  const auto recovering = tracker.update(10.20);

  EXPECT_FALSE(recovering.execution_frozen);
  EXPECT_NE(recovering.vx, 0.0);
  // 局部 B-spline 弦朝 +X；若 terminal mode 被误清，wz 会转向相反符号。
  // 当前 terminal yaw 已在控制死区内，因此允许严格零，但绝不能变为负向。
  EXPECT_GE(recovering.wz, 0.0);
  EXPECT_LE(recovering.wz, config.terminal_max_yaw_rate);
}

TEST(TrajectoryTracker, EarlyTerminalCaptureCompletesAfterStablePhysicalDwell)
{
  auto config = makeConfig();
  config.terminal_capture_brake_hold_sec = 0.05;
  config.terminal_capture_stable_dwell_sec = 0.20;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  setFreshInputs(
    tracker, 10.10, 0.0, goal, config.finish_speed + 0.05);
  EXPECT_FALSE(tracker.update(10.10).goal_reached);
  ASSERT_LT(tracker.executionTimeSec(), tracker.trajectoryDurationSec());

  setFreshInputs(tracker, 10.16, 0.0, goal);
  EXPECT_FALSE(tracker.update(10.16).goal_reached);
  setFreshInputs(tracker, 10.26, 0.0, goal);
  EXPECT_FALSE(tracker.update(10.26).goal_reached);
  setFreshInputs(tracker, 10.31, 0.0, goal);
  const auto completed = tracker.update(10.31);

  EXPECT_EQ(
    completed.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(completed.goal_reached);
  EXPECT_LT(tracker.executionTimeSec(), tracker.trajectoryDurationSec());
}

TEST(TrajectoryTracker, MovingFinalCaptureUsesYawRateDespiteStanceTiltMotion)
{
  auto config = makeConfig();
  config.terminal_capture_brake_hold_sec = 0.05;
  config.terminal_capture_stable_dwell_sec = 0.20;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  const auto set_stance_sample = [&](double now, double yaw_rate) {
      scan_controller::OdometryInput odometry;
      odometry.position = goal;
      odometry.yaw = 0.0;
      odometry.planar_speed = 0.0;
      odometry.vertical_speed = 0.0;
      // 复现 Go2-X5 零命令站立时的 roll/pitch 微摆：三轴范数超过旧门，
      // 但与导航控制轴同语义的机体系 |wz| 已经稳定。
      odometry.angular_speed = config.finish_angular_speed + 0.15;
      odometry.yaw_rate = yaw_rate;
      odometry.stamp_sec = now;
      EXPECT_TRUE(tracker.setOdometry(odometry, now, error)) << error;
      EXPECT_TRUE(tracker.setCloudObservation(now, now, error)) << error;
      return tracker.update(now);
    };

  auto output = set_stance_sample(10.10, 0.05);
  EXPECT_TRUE(output.execution_frozen);
  EXPECT_FALSE(output.goal_reached);

  // 真实 yaw 仍在转动时必须清零 moving capture 的连续驻留。
  output = set_stance_sample(10.16, config.finish_yaw_rate + 0.01);
  EXPECT_FALSE(output.goal_reached);

  output = set_stance_sample(10.22, 0.05);
  EXPECT_FALSE(output.goal_reached);
  output = set_stance_sample(10.33, 0.05);
  EXPECT_FALSE(output.goal_reached);
  output = set_stance_sample(10.43, 0.05);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
}

TEST(TrajectoryTracker, MovingFinalRejectsSingleStableSampleAtNominalEnd)
{
  auto config = makeConfig();
  config.max_control_dt_sec = 1.0;
  config.terminal_capture_brake_hold_sec = 0.05;
  config.terminal_capture_stable_dwell_sec = 0.20;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  ASSERT_GE(tracker.executionTimeSec(), tracker.trajectoryDurationSec());

  // 名义时间已经结束且首个 Odometry 样本完全合格，也只能先严格制动，
  // 不能在四足底盘仍可能回弹时立即锁存 GOAL_REACHED。
  setFreshInputs(tracker, 13.01, 0.0, goal);
  auto output = tracker.update(13.01);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_TRUE(output.execution_frozen);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);

  setFreshInputs(tracker, 13.07, 0.0, goal);
  EXPECT_FALSE(tracker.update(13.07).goal_reached);

  // 制动后出现一次速度回弹，必须清空已经累计的稳定时间。
  setFreshInputs(tracker, 13.12, 0.0, goal, config.finish_speed + 0.01);
  output = tracker.update(13.12);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);

  setFreshInputs(tracker, 13.18, 0.0, goal);
  EXPECT_FALSE(tracker.update(13.18).goal_reached);
  setFreshInputs(tracker, 13.29, 0.0, goal);
  EXPECT_FALSE(tracker.update(13.29).goal_reached);
  setFreshInputs(tracker, 13.40, 0.0, goal);
  output = tracker.update(13.40);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
}

TEST(TrajectoryTracker, TerminalStableDwellRestartsAfterCloudTimeout)
{
  auto config = makeConfig();
  config.cloud_timeout_sec = 0.05;
  config.terminal_capture_brake_hold_sec = 0.05;
  config.terminal_capture_stable_dwell_sec = 0.20;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  setFreshInputs(
    tracker, 10.10, 0.0, goal, config.finish_speed + 0.05);
  EXPECT_FALSE(tracker.update(10.10).goal_reached);
  setFreshInputs(tracker, 10.16, 0.0, goal);
  EXPECT_FALSE(tracker.update(10.16).goal_reached);
  setFreshInputs(tracker, 10.20, 0.0, goal);
  EXPECT_FALSE(tracker.update(10.20).goal_reached);

  scan_controller::OdometryInput odometry;
  odometry.position = goal;
  odometry.stamp_sec = 10.27;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.27, error)) << error;
  EXPECT_EQ(
    tracker.update(10.27).state,
    scan_controller::ControllerState::kCloudTimeout);

  setFreshInputs(tracker, 10.28, 0.0, goal);
  EXPECT_FALSE(tracker.update(10.28).goal_reached);
  setFreshInputs(tracker, 10.39, 0.0, goal);
  EXPECT_FALSE(tracker.update(10.39).goal_reached);
  setFreshInputs(tracker, 10.49, 0.0, goal);
  EXPECT_TRUE(tracker.update(10.49).goal_reached);
}

TEST(TrajectoryTracker, SamePathFinalReplacementCannotRefreshCaptureHardExpiry)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.10;
  config.trajectory_expiry_grace_sec = 0.10;
  config.max_yaw_alignment_freeze_sec = 0.20;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.10, 0.0, goal, config.finish_speed + 0.05);
  ASSERT_TRUE(tracker.update(10.10).execution_frozen);

  auto replacement = makeStraightTrajectory(13.20, true);
  replacement.trajectory_id = 8;
  replacement.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(replacement, 13.20, error)) << error;
  setFreshInputs(tracker, 13.31, 0.0, goal);
  const auto expired = tracker.update(13.31);

  EXPECT_EQ(
    expired.state, scan_controller::ControllerState::kTrajectoryTimeout);
  EXPECT_FALSE(expired.execution_frozen);
  EXPECT_FALSE(expired.goal_reached);
  EXPECT_DOUBLE_EQ(expired.vx, 0.0);
  EXPECT_DOUBLE_EQ(expired.vy, 0.0);
  EXPECT_DOUBLE_EQ(expired.wz, 0.0);
}

TEST(TrajectoryTracker, SamePathFinalReplacementRefreshesExpiryAfterTranslationRecovery)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.10;
  config.trajectory_expiry_grace_sec = 0.10;
  config.max_yaw_alignment_freeze_sec = 0.20;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  setFreshInputs(
    tracker, 10.10, 0.0, goal, config.finish_speed + 0.05);
  ASSERT_TRUE(tracker.update(10.10).execution_frozen);

  // 姿态和速度稳定后漂到完成门外、捕获外圈内，控制器恢复执行 SCAN
  // 平移轨迹。此时上一轮冻结已经结束，但 terminal yaw 模式仍应保留。
  setFreshInputs(
    tracker, 10.20, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_release_distance_xy - 0.01, 0.0, 0.0));
  const auto recovering = tracker.update(10.20);
  ASSERT_FALSE(recovering.execution_frozen);
  ASSERT_NE(recovering.vx, 0.0);

  auto replacement = makeStraightTrajectory(13.20, true);
  replacement.trajectory_id = 8;
  replacement.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(replacement, 13.20, error)) << error;
  setFreshInputs(
    tracker, 13.31, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_release_distance_xy - 0.01, 0.0, 0.0));
  const auto tracking = tracker.update(13.31);

  // 新轨迹到达时没有任何执行冻结，不得被旧冻结 episode 的 hard-expiry
  // 立即判超时；它仍在有限的新轨迹有效期内继续收回终点位置。
  EXPECT_EQ(tracking.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(tracking.execution_frozen);
  EXPECT_NE(tracking.vx, 0.0);
  EXPECT_FALSE(tracking.goal_reached);
}

TEST(TrajectoryTracker, TerminalRecoveryCannotUseExpiredSplineForOneTick)
{
  auto config = makeConfig();
  config.bspline_timeout_sec = 0.10;
  config.trajectory_expiry_grace_sec = 0.10;
  config.max_yaw_alignment_freeze_sec = 1.00;
  config.max_control_dt_sec = 5.00;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;

  setFreshInputs(
    tracker, 10.10, 0.0, goal, config.finish_speed + 0.05);
  ASSERT_TRUE(tracker.update(10.10).execution_frozen);

  // 捕获首拍顺延后的 soft expiry 为 13.20s，固定 hard expiry 为 14.10s。
  // 外圈释放发生在两者之间时必须当拍超时全零，不能先执行旧 B-spline。
  setFreshInputs(
    tracker, 13.21, 0.0,
    goal - Eigen::Vector3d(
      config.terminal_capture_release_distance_xy + 0.01, 0.0, 0.0));
  const auto expired = tracker.update(13.21);

  EXPECT_EQ(
    expired.state, scan_controller::ControllerState::kTrajectoryTimeout);
  EXPECT_FALSE(expired.execution_frozen);
  EXPECT_DOUBLE_EQ(expired.vx, 0.0);
  EXPECT_DOUBLE_EQ(expired.vy, 0.0);
  EXPECT_DOUBLE_EQ(expired.wz, 0.0);
}

TEST(TrajectoryTracker, FinalToNonFinalClearsTerminalYawMode)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto final_trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(final_trajectory);
  const std::vector<Eigen::Vector3d> path{
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0),
    Eigen::Vector3d(goal.x(), 0.0, 0.0),
  };
  ASSERT_TRUE(
    tracker.setReferencePath(
      path, 10.0, 10.0, error, M_PI_2)) << error;
  ASSERT_TRUE(tracker.setTrajectory(final_trajectory, 10.0, error)) << error;
  setFreshInputs(
    tracker, 10.10, 0.0, goal, config.finish_speed + 0.05);
  ASSERT_TRUE(tracker.update(10.10).execution_frozen);

  auto non_final = makeStraightTrajectory(10.20, false);
  non_final.trajectory_id = 8;
  non_final.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(non_final, 10.20, error)) << error;
  setFreshInputs(
    tracker, 10.30, 0.0, trajectoryPositionAt(non_final, 0.10));
  const auto tracking = tracker.update(10.30);

  EXPECT_EQ(tracking.state, scan_controller::ControllerState::kTracking);
  EXPECT_FALSE(tracking.execution_frozen);
  EXPECT_NE(tracking.vx, 0.0);
  EXPECT_NEAR(tracking.wz, 0.0, 1.0e-9);
}

TEST(TrajectoryTracker, FinalCompletionPreservesRawPathLastPointAfterDedup)
{
  auto config = makeConfig();
  config.finish_distance_xy = 0.005;
  config.terminal_capture_entry_distance_xy = 0.0025;
  config.terminal_capture_zero_hold_distance_xy = 0.0035;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const std::vector<Eigen::Vector3d> path{
    Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 0.0),
    // 小于局部投影去重间距，但仍是完整 Path 的真实终点。
    Eigen::Vector3d(0.01, 0.0, 0.0),
  };
  ASSERT_TRUE(tracker.setReferencePath(path, 10.0, 10.0, error)) << error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  setFreshInputs(tracker, 13.0, 0.0, Eigen::Vector3d(0.0, 0.0, 0.3));
  EXPECT_FALSE(tracker.update(13.01).goal_reached);

  setFreshInputs(
    tracker, 13.02, 0.0, Eigen::Vector3d(0.01, 0.0, 0.3));
  EXPECT_FALSE(tracker.update(13.02).goal_reached);
  const auto completed = waitForMovingFinalStableCompletion(
    tracker, 13.07, 0.0, Eigen::Vector3d(0.01, 0.0, 0.3));
  EXPECT_TRUE(completed.goal_reached);
}

TEST(TrajectoryTracker, StationaryFinalHoldWrongYawRemainsExactZero)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal, M_PI_2);
  auto hold = makeStationaryTrajectory(10.0, 10.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setTrajectory(hold, 10.0, error)) << error;
  setFreshInputs(tracker, 10.1, 0.0, goal);

  const auto output = tracker.update(10.1);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_TRUE(output.execution_frozen);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(tracker.executionTimeSec(), 0.0);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
}

TEST(TrajectoryTracker, FreshCloudFinalGoalStillWaitsForNominalTime)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(
    tracker.setTrajectory(
      makeStationaryTrajectory(10.0, 10.0, true), 10.0, error)) << error;
  setFreshInputs(tracker, 10.0, 0.0, goal);
  ASSERT_LT(tracker.executionTimeSec(), tracker.trajectoryDurationSec());

  const auto output = tracker.update(10.01);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kTracking);
  EXPECT_TRUE(output.execution_frozen);
  EXPECT_FALSE(output.trajectory_finished);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
}

TEST(TrajectoryTracker, FreshCloudGoalRequiresStableVerticalAndAngularSpeed)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;

  scan_controller::OdometryInput odometry;
  odometry.position = goal;
  odometry.vertical_speed = config.finish_vertical_speed + 0.01;
  odometry.stamp_sec = 13.0;
  ASSERT_TRUE(tracker.setOdometry(odometry, 13.0, error)) << error;
  ASSERT_TRUE(tracker.setCloudObservation(13.0, 13.0, error)) << error;
  auto output = tracker.update(13.01);
  EXPECT_FALSE(output.goal_reached);

  odometry.vertical_speed = 0.0;
  odometry.angular_speed = config.finish_angular_speed + 0.01;
  odometry.stamp_sec = 13.02;
  ASSERT_TRUE(tracker.setOdometry(odometry, 13.02, error)) << error;
  ASSERT_TRUE(tracker.setCloudObservation(13.02, 13.02, error)) << error;
  output = tracker.update(13.02);
  EXPECT_FALSE(output.goal_reached);

  odometry.angular_speed = 0.0;
  odometry.stamp_sec = 13.03;
  ASSERT_TRUE(tracker.setOdometry(odometry, 13.03, error)) << error;
  ASSERT_TRUE(tracker.setCloudObservation(13.03, 13.03, error)) << error;
  EXPECT_FALSE(tracker.update(13.03).goal_reached);
  odometry.stamp_sec = 13.08;
  ASSERT_TRUE(tracker.setOdometry(odometry, 13.08, error)) << error;
  ASSERT_TRUE(tracker.setCloudObservation(13.08, 13.08, error)) << error;
  output = tracker.update(13.08);
  EXPECT_FALSE(output.goal_reached);
  output = waitForMovingFinalStableCompletion(
    tracker, 13.13, 0.0, goal);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, FreshCloudGoalRequiresTerminalYaw)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  scan_planner::UniformBspline spline(
    trajectory.control_points, trajectory.order, 1.0);
  spline.setKnot(trajectory.knots);
  double start_time = 0.0;
  double end_time = 0.0;
  ASSERT_TRUE(spline.getTimeSpan(start_time, end_time));
  const double duration = end_time - start_time;
  const Eigen::Vector3d final_position = spline.evaluateDeBoorT(duration);
  setFinalReferencePath(tracker, 10.0, final_position);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  setFreshInputs(
    tracker, 13.0, config.finish_yaw_error + 0.01, final_position);
  auto output = tracker.update(13.01);
  EXPECT_FALSE(output.goal_reached);

  setFreshInputs(tracker, 13.02, 0.0, final_position);
  EXPECT_FALSE(tracker.update(13.02).goal_reached);
  setFreshInputs(tracker, 13.08, 0.0, final_position);
  output = tracker.update(13.08);
  EXPECT_FALSE(output.goal_reached);
  output = waitForMovingFinalStableCompletion(
    tracker, 13.13, 0.0, final_position);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
}

TEST(TrajectoryTracker, NonFinalTrajectoryStillWaitsForNominalTime)
{
  auto config = makeConfig();
  config.cloud_timeout_sec = 0.25;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(
      makeStationaryTrajectory(10.0, 10.0, false), 10.0, error)) << error;
  setFreshInputs(tracker, 10.0);

  scan_controller::OdometryInput odometry;
  odometry.position = Eigen::Vector3d(0.0, 0.0, 0.3);
  odometry.planar_speed = 0.0;
  odometry.stamp_sec = 10.26;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.26, error)) << error;

  const auto output = tracker.update(10.26);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kCloudTimeout);
  EXPECT_FALSE(output.trajectory_finished);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
}

TEST(TrajectoryTracker, FreshCloudGoalRequiresLowSpeedOdometry)
{
  auto config = makeConfig();
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 13.0, error)) << error;
  setFreshInputs(tracker, 13.0, 0.0, goal, 0.06);
  auto output = tracker.update(13.01);
  EXPECT_FALSE(output.goal_reached);

  setFreshInputs(tracker, 13.02, 0.0, goal, 0.0);
  EXPECT_FALSE(tracker.update(13.02).goal_reached);
  setFreshInputs(tracker, 13.08, 0.0, goal, 0.0);
  output = tracker.update(13.08);
  EXPECT_FALSE(output.goal_reached);
  output = waitForMovingFinalStableCompletion(
    tracker, 13.13, 0.0, goal);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
}

TEST(TrajectoryTracker, StaleCloudCannotCertifyGoalOutsideTolerance)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  config.cloud_timeout_sec = 0.25;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto trajectory = makeStraightTrajectory(10.0, true);
  const Eigen::Vector3d goal = trajectoryFinalPosition(trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(tracker.setTrajectory(trajectory, 10.0, error)) << error;
  setFreshInputs(tracker, 10.0, 0.0, goal);

  scan_controller::OdometryInput odometry;
  odometry.position = goal + Eigen::Vector3d(0.09, 0.0, 0.0);
  odometry.stamp_sec = 10.26;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.26, error)) << error;
  const auto output = tracker.update(10.26);

  EXPECT_EQ(output.state, scan_controller::ControllerState::kCloudTimeout);
  EXPECT_FALSE(output.trajectory_finished);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
}

TEST(TrajectoryTracker, MissingCloudCannotCertifyFinalGoal)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  const Eigen::Vector3d goal(0.0, 0.0, 0.3);
  setFinalReferencePath(tracker, 10.0, goal);
  auto hold = makeStationaryTrajectory(10.0, 7.0, true);
  hold.reference_path_stamp_sec = 10.0;
  ASSERT_TRUE(
    tracker.setTrajectory(hold, 10.0, error)) << error;
  scan_controller::OdometryInput odometry;
  odometry.position = goal;
  odometry.stamp_sec = 10.0;
  ASSERT_TRUE(tracker.setOdometry(odometry, 10.0, error)) << error;

  const auto output = tracker.update(10.01);

  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kWaitingForCloud);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
}

TEST(TrajectoryTracker, SamePathTrajectoryCannotClearLatchedCompletion)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto completed_trajectory = makeStraightTrajectory(10.0, true);
  completed_trajectory.start_stamp_sec = 7.0;
  const Eigen::Vector3d goal = trajectoryFinalPosition(completed_trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(
    tracker.setTrajectory(completed_trajectory, 10.0, error)) << error;
  ASSERT_TRUE(
    waitForMovingFinalStableCompletion(tracker, 10.01, 0.0, goal).
    goal_reached);

  auto next_trajectory = makeStraightTrajectory(11.02, true);
  next_trajectory.trajectory_id = 8;
  next_trajectory.reference_path_stamp_sec = 10.0;
  EXPECT_FALSE(
    tracker.setTrajectory(next_trajectory, 11.02, error));

  const auto output = tracker.immediateStop(11.02);
  EXPECT_TRUE(tracker.hasTrajectory());
  EXPECT_TRUE(output.trajectory_finished);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_NE(error.find("不能解除完成锁存"), std::string::npos);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);
}

TEST(TrajectoryTracker, InvalidInputDoesNotClearLatchedGoal)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto completed_trajectory = makeStraightTrajectory(10.0, true);
  completed_trajectory.start_stamp_sec = 7.0;
  const Eigen::Vector3d goal = trajectoryFinalPosition(completed_trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(
    tracker.setTrajectory(completed_trajectory, 10.0, error)) << error;
  ASSERT_TRUE(
    waitForMovingFinalStableCompletion(tracker, 10.01, 0.0, goal).
    goal_reached);

  auto invalid_trajectory = makeStraightTrajectory(11.02, true);
  invalid_trajectory.control_points(0, 1) =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
    tracker.setTrajectory(invalid_trajectory, 11.02, error));
  // 不同 Path 代际的畸形迟到包不得打掉已经完成的有效同代轨迹。
  EXPECT_TRUE(tracker.hasTrajectory());

  auto output = tracker.immediateStop(11.02);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kGoalReached);
  EXPECT_TRUE(output.trajectory_finished);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
  EXPECT_DOUBLE_EQ(output.vy, 0.0);
  EXPECT_DOUBLE_EQ(output.wz, 0.0);

  scan_controller::OdometryInput invalid_odometry;
  invalid_odometry.stamp_sec =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
    tracker.setOdometry(invalid_odometry, 11.03, error));
  output = tracker.immediateStop(11.03);
  EXPECT_TRUE(output.trajectory_finished);
  EXPECT_TRUE(output.goal_reached);
  EXPECT_TRUE(tracker.update(11.04).goal_reached);
}

TEST(TrajectoryTracker, ValidNewReferencePathClearsLatchedGoalBeforeSpline)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto completed_trajectory = makeStraightTrajectory(10.0, true);
  completed_trajectory.start_stamp_sec = 7.0;
  const Eigen::Vector3d goal = trajectoryFinalPosition(completed_trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(
    tracker.setTrajectory(completed_trajectory, 10.0, error)) << error;
  ASSERT_TRUE(
    waitForMovingFinalStableCompletion(tracker, 10.01, 0.0, goal).
    goal_reached);

  const std::vector<Eigen::Vector3d> next_path{
    Eigen::Vector3d(2.0, 0.0, 0.0),
    Eigen::Vector3d(2.5, 0.5, 0.0),
    Eigen::Vector3d(3.0, 0.5, 0.0),
  };
  ASSERT_TRUE(
    tracker.setReferencePath(next_path, 11.0, 11.0, error)) << error;

  const auto output = tracker.immediateStop(11.0);
  EXPECT_FALSE(output.trajectory_finished);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_FALSE(tracker.hasTrajectory());
}

TEST(TrajectoryTracker, ExplicitReferenceClearClearsLatchedGoal)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto completed_trajectory = makeStraightTrajectory(10.0, true);
  completed_trajectory.start_stamp_sec = 7.0;
  const Eigen::Vector3d goal = trajectoryFinalPosition(completed_trajectory);
  setFinalReferencePath(tracker, 10.0, goal);
  ASSERT_TRUE(
    tracker.setTrajectory(completed_trajectory, 10.0, error)) << error;
  ASSERT_TRUE(
    waitForMovingFinalStableCompletion(tracker, 10.01, 0.0, goal).
    goal_reached);

  tracker.invalidateReferencePath(10000000000LL);
  tracker.invalidateTrajectory();
  tracker.clearCompletionLatch();

  const auto output = tracker.immediateStop(10.70);
  EXPECT_FALSE(output.trajectory_finished);
  EXPECT_FALSE(output.goal_reached);

  const std::vector<Eigen::Vector3d> next_path{
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0),
  };
  EXPECT_FALSE(
    tracker.setReferencePath(next_path, 10.0, 10.02, error));
  EXPECT_NE(error.find("已被清除"), std::string::npos);
  EXPECT_TRUE(
    tracker.setReferencePath(next_path, 11.0, 11.0, error)) << error;
}

TEST(TrajectoryTracker, LateMatchingPathPreservesPreReceivedFinalSplineSemantics)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  auto final_trajectory = makeStraightTrajectory(10.0, true);
  final_trajectory.start_stamp_sec = 7.0;
  const Eigen::Vector3d goal = trajectoryFinalPosition(final_trajectory);
  ASSERT_TRUE(
    tracker.setTrajectory(final_trajectory, 10.0, error)) << error;

  setFinalReferencePath(tracker, 10.0, goal);

  EXPECT_TRUE(tracker.hasTrajectory());
  EXPECT_TRUE(
    waitForMovingFinalStableCompletion(tracker, 10.01, 0.0, goal).
    goal_reached);
}

TEST(TrajectoryTracker, NonFinalTrajectoryNeverReportsGlobalGoal)
{
  auto config = makeConfig();
  config.max_start_stamp_skew_sec = 5.0;
  scan_controller::TrajectoryTracker tracker(config);
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(
      makeStationaryTrajectory(10.0, 7.0, false), 10.0, error)) << error;
  setFreshInputs(tracker, 10.0);

  const auto output = tracker.update(10.01);
  EXPECT_EQ(
    output.state, scan_controller::ControllerState::kTrajectoryFinished);
  EXPECT_TRUE(output.trajectory_finished);
  EXPECT_FALSE(output.goal_reached);
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
}

TEST(TrajectoryTracker, ClockRewindInvalidatesAllInputs)
{
  scan_controller::TrajectoryTracker tracker(makeConfig());
  std::string error;
  ASSERT_TRUE(
    tracker.setTrajectory(makeStraightTrajectory(10.0), 10.0, error))
    << error;
  setFreshInputs(tracker, 10.0);
  EXPECT_EQ(
    tracker.update(10.1).state,
    scan_controller::ControllerState::kTracking);

  const auto output = tracker.update(9.0);
  EXPECT_EQ(output.state, scan_controller::ControllerState::kInvalidClock);
  EXPECT_FALSE(tracker.hasTrajectory());
  EXPECT_DOUBLE_EQ(output.vx, 0.0);
}

}  // 匿名命名空间
