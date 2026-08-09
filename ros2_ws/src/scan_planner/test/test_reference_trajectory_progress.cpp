#include <gtest/gtest.h>

#include <Eigen/Core>
#include <limits>
#include <bspline_opt/uniform_bspline.h>
#include <plan_manage/reference_spline_boundary.h>
#include <plan_manage/reference_trajectory_initialization.h>
#include <plan_manage/trajectory_progress.h>
#include <plan_manage/trajectory_diagnostics.h>
#include <plan_manage/trajectory_timing.h>

namespace
{

scan_planner::UniformBspline makePolylineTrajectory(
    const std::vector<Eigen::Vector3d> &points)
{
  Eigen::MatrixXd control_points(3, static_cast<Eigen::Index>(points.size()));
  for (std::size_t index = 0; index < points.size(); ++index)
    control_points.col(static_cast<Eigen::Index>(index)) = points[index];
  return scan_planner::UniformBspline(control_points, 1, 1.0);
}

}  // namespace

TEST(TrajectoryProgress, AcceptsMonotonicPlanarSpline)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 8);
  for (int index = 0; index < control_points.cols(); ++index)
  {
    control_points(0, index) = 0.1 * static_cast<double>(index);
    control_points(2, index) = 0.3;
  }

  scan_planner::UniformBspline trajectory(control_points, 3, 0.2);
  const double duration = trajectory.getTimeSum();
  const auto result = scan_planner::checkTrajectoryForwardProgress(
      trajectory,
      trajectory.evaluateDeBoorT(0.0),
      trajectory.evaluateDeBoorT(duration),
      0.02,
      0.02);

  EXPECT_TRUE(result.safe);
  EXPECT_NEAR(result.maximum_reverse_distance, 0.0, 1.0e-9);
  EXPECT_GT(result.minimum_projected_velocity, 0.0);
}

TEST(TrajectoryProgress, RejectsObservedRampBacktrackingSpline)
{
  // 该控制点序列来自 ramp_validation_ros2_v5 的第三条真实 B-spline。
  const double observed_x[] = {
      -0.1199176,
      0.0493591,
      0.1137275,
      0.1118082,
      0.08594545,
      0.07667357,
      0.1172055,
      0.2280914,
      0.4115533,
      0.6460519,
      0.8807761};
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 11);
  for (int index = 0; index < control_points.cols(); ++index)
  {
    control_points(0, index) = observed_x[index];
    control_points(2, index) = 0.3;
  }

  scan_planner::UniformBspline trajectory(control_points, 3, 0.48);
  const double duration = trajectory.getTimeSum();
  const auto result = scan_planner::checkTrajectoryForwardProgress(
      trajectory,
      trajectory.evaluateDeBoorT(0.0),
      trajectory.evaluateDeBoorT(duration),
      0.02,
      0.02);

  EXPECT_FALSE(result.safe);
  EXPECT_GT(result.maximum_reverse_distance, 0.02);
  EXPECT_LT(result.minimum_projected_velocity, -0.03);
}

TEST(TrajectoryProgress, PlanarCheckCannotBeMaskedByIncreasingRampHeight)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 8);
  const double x[] = {0.0, 0.1, 0.2, 0.17, 0.08, 0.3, 0.4, 0.5};
  for (int index = 0; index < control_points.cols(); ++index)
  {
    control_points(0, index) = x[index];
    control_points(2, index) = 0.3 + 0.1 * static_cast<double>(index);
  }

  scan_planner::UniformBspline trajectory(control_points, 3, 0.2);
  const double duration = trajectory.getTimeSum();
  const auto result = scan_planner::checkTrajectoryForwardProgress(
      trajectory,
      trajectory.evaluateDeBoorT(0.0),
      trajectory.evaluateDeBoorT(duration),
      0.02,
      0.02);

  EXPECT_FALSE(result.safe);
  EXPECT_GT(result.maximum_reverse_distance, 0.02);
}

TEST(TrajectoryProgress, ReferenceCorridorAcceptsPolylineThatVisitsTurnAnchor)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 3);
  control_points.col(0) = Eigen::Vector3d(0.0, 0.0, 0.3);
  control_points.col(1) = Eigen::Vector3d(1.0, 0.0, 0.3);
  control_points.col(2) = Eigen::Vector3d(1.0, 1.0, 0.3);
  const std::vector<Eigen::Vector3d> guide{
      control_points.col(0), control_points.col(1), control_points.col(2)};
  scan_planner::UniformBspline trajectory(control_points, 1, 1.0);

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.01, 0.02);

  EXPECT_TRUE(result.safe);
  EXPECT_NEAR(result.maximum_trajectory_deviation, 0.0, 1.0e-9);
  EXPECT_NEAR(result.maximum_guide_anchor_deviation, 0.0, 1.0e-9);
}

TEST(TrajectoryProgress, ReferenceCorridorRejectsDiagonalTurnShortcut)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 2);
  control_points.col(0) = Eigen::Vector3d(0.0, 0.0, 0.3);
  control_points.col(1) = Eigen::Vector3d(1.0, 1.0, 0.3);
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 1.0, 0.3)};
  scan_planner::UniformBspline trajectory(control_points, 1, 1.0);

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.035, 0.01, 0.0, 0.001);

  EXPECT_FALSE(result.safe);
  EXPECT_GT(result.maximum_trajectory_deviation, 0.49);
  EXPECT_GT(result.maximum_guide_anchor_deviation, 0.70);
}

TEST(TrajectoryProgress, ReferenceCorridorRejectsMissingOrderedReturnSegment)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 2);
  control_points.col(0) = Eigen::Vector3d(0.0, 0.0, 0.3);
  control_points.col(1) = Eigen::Vector3d(1.0, 0.0, 0.3);
  // guide 在同一几何线段上先前进再返回。无序 Hausdorff 距离为零，但只前进
  // 的轨迹没有按顺序访问最后一个返回锚点，必须拒绝。
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(0.0, 0.0, 0.3)};
  scan_planner::UniformBspline trajectory(control_points, 1, 1.0);

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.05, 0.02);

  EXPECT_FALSE(result.safe);
  EXPECT_GT(result.maximum_guide_anchor_deviation, 0.9);
}

TEST(TrajectoryProgress, ReferenceCorridorRejectsUShapeChordShortcut)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 1.0, 0.3),
      Eigen::Vector3d(0.0, 1.0, 0.3)};
  const auto trajectory = makePolylineTrajectory(
      {guide.front(), guide.back()});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02);

  EXPECT_FALSE(result.safe);
  EXPECT_GT(result.maximum_trajectory_deviation, 0.49);
  EXPECT_GT(result.maximum_guide_anchor_deviation, 0.99);
  EXPECT_GT(result.maximum_guide_progress_lead, 1.9);
}

TEST(TrajectoryProgress, ReferenceCorridorRejectsParallelNeighborShortcut)
{
  // 两条平行分支只相距 0.08 m，纯空间 Hausdorff 门会把
  // B->D 的对角线误当成合法 0.10 m 偏移。该轨迹实际跳过了 C 处
  // 的有序折返，guide 进度超前门必须独立拒绝。
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.08, 0.3),
      Eigen::Vector3d(0.0, 0.08, 0.3)};
  const auto trajectory = makePolylineTrajectory(
      {guide[0], guide[1], guide[3]});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.035, 0.01, 0.0, 0.001);

  EXPECT_FALSE(result.safe);
  EXPECT_LE(result.maximum_trajectory_deviation, 0.10 + 1.0e-9);
  EXPECT_LE(result.maximum_guide_anchor_deviation, 0.10 + 1.0e-9);
  EXPECT_GT(result.maximum_guide_progress_lead, 0.05);
}

TEST(TrajectoryProgress, ReferenceCorridorRejectsSelfIntersectionJump)
{
  const Eigen::Vector3d crossing(0.5, 0.5, 0.3);
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 1.0, 0.3),
      Eigen::Vector3d(0.0, 1.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(2.0, 0.0, 0.3)};
  // 在自交点从首段直接切到第三段，然后驶向终点。
  const auto trajectory = makePolylineTrajectory(
      {guide.front(), crossing, guide[3], guide.back()});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02);

  EXPECT_FALSE(result.safe);
  EXPECT_GT(result.maximum_guide_progress_lead, 1.0);
}

TEST(TrajectoryProgress, ReferenceCorridorAcceptsOrderedDynamicDetour)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(2.0, 0.0, 0.3)};
  // 模拟在 x=1.0 处绕开短时动态障碍：弧长顺序不变，偏移恰好用满
  // 可配置的 0.10 m 空间余量，不应被 fail-closed 门误拒绝。
  const auto trajectory = makePolylineTrajectory({
      guide.front(),
      Eigen::Vector3d(0.75, 0.0, 0.3),
      Eigen::Vector3d(0.85, 0.10, 0.3),
      Eigen::Vector3d(1.15, 0.10, 0.3),
      Eigen::Vector3d(1.25, 0.0, 0.3),
      guide.back()});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02);

  EXPECT_TRUE(result.safe);
  EXPECT_NEAR(result.maximum_trajectory_deviation, 0.10, 1.0e-9);
  EXPECT_NEAR(result.maximum_guide_anchor_deviation, 0.10, 1.0e-9);
  EXPECT_LE(result.maximum_guide_progress_lead, 1.0e-9);
}

TEST(TrajectoryProgress, SemanticCorridorAcceptsBoundedReacquisitionShortcut)
{
  const Eigen::Vector3d actual_start(0.25, 0.0, 0.3);
  const std::vector<Eigen::Vector3d> pct_semantic_guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(0.0, 1.0, 0.3)};
  const std::vector<Eigen::Vector3d> guide_with_synthetic_connector{
      actual_start,
      pct_semantic_guide.front(),
      pct_semantic_guide.back()};
  const auto trajectory = makePolylineTrajectory({
      actual_start,
      pct_semantic_guide.back()});

  const auto synthetic_result =
      scan_planner::checkTrajectoryReferenceCorridor(
          trajectory, guide_with_synthetic_connector, 0.35, 0.05);
  const auto semantic_result =
      scan_planner::checkTrajectoryReferenceCorridor(
          trajectory, guide_with_synthetic_connector,
          pct_semantic_guide, 0.35, 0.05);

  // v6 的安全回归同样把约 0.25 m 横向连接平滑成斜线；人工折线会
  // 虚增约 0.22 m guide 进度，而 PCT semantic guide 不会制造额度。
  EXPECT_FALSE(synthetic_result.safe);
  EXPECT_GT(synthetic_result.maximum_guide_progress_lead, 0.20);
  EXPECT_TRUE(semantic_result.safe);
  EXPECT_LE(semantic_result.maximum_guide_progress_lead, 1.0e-9);
  EXPECT_LT(semantic_result.maximum_guide_anchor_deviation, 0.25);
}

TEST(TrajectoryProgress, SemanticCorridorStillRejectsParallelBranchShortcut)
{
  const std::vector<Eigen::Vector3d> pct_semantic_guide{
      Eigen::Vector3d(0.25, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.08, 0.3),
      Eigen::Vector3d(0.0, 0.08, 0.3)};
  const Eigen::Vector3d actual_start(0.25, -0.10, 0.3);
  const std::vector<Eigen::Vector3d> spatial_guide{
      actual_start,
      pct_semantic_guide[0],
      pct_semantic_guide[1],
      pct_semantic_guide[2],
      pct_semantic_guide[3]};
  const auto trajectory = makePolylineTrajectory({
      actual_start,
      pct_semantic_guide[0],
      pct_semantic_guide[1],
      pct_semantic_guide.back()});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, spatial_guide, pct_semantic_guide, 0.35, 0.05);

  // 初始横向偏离可以在空间包络内回归，但不能借此跳过后续真实折返。
  EXPECT_FALSE(result.safe);
  EXPECT_LE(result.maximum_trajectory_deviation, 0.35 + 1.0e-9);
  EXPECT_LE(result.maximum_guide_anchor_deviation, 0.35 + 1.0e-9);
  EXPECT_GT(result.maximum_guide_progress_lead, 0.05);
}

TEST(TrajectoryProgress, ReferenceCorridorNormalizesBoundedSplineStartBias)
{
  std::vector<Eigen::Vector3d> guide;
  for (int index = 0; index <= 12; ++index)
    guide.emplace_back(0.05 * static_cast<double>(index), 0.0, 0.3);

  const double max_vel = 0.30;
  const double max_acc = 0.50;
  const double total_length = 0.60;
  const double acceleration_distance = max_vel * max_vel / max_acc;
  const double duration =
      (total_length - acceleration_distance) / max_vel +
      2.0 * max_vel / max_acc;
  const double time_step =
      duration / static_cast<double>(guide.size() - 1);
  const std::vector<Eigen::Vector3d> derivatives{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::MatrixXd control_points;
  scan_planner::UniformBspline::parameterizeToBspline(
      time_step, guide, derivatives, control_points);
  scan_planner::UniformBspline trajectory(control_points, 3, time_step);

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02);

  // 该偏置来自 SCAN 当前的过定约束参数化，不是路径跳段；它本身仍须
  // 小于 0.02 m，扣除一次后不能再令有序进度门误报。
  EXPECT_TRUE(result.safe);
  EXPECT_GT(result.initial_guide_progress, 0.01);
  EXPECT_LT(result.initial_guide_progress, 0.02);
  EXPECT_LE(result.maximum_guide_progress_lead, 0.02 + 1.0e-9);
}

TEST(ReferenceTrajectoryInitialization, UsesAccelerationAwareUniformTimeSamples)
{
  const std::vector<Eigen::Vector3d> raw_guide{
      Eigen::Vector3d(0.0, 0.0, 0.30),
      Eigen::Vector3d(0.30, 0.0, 0.30),
      Eigen::Vector3d(0.60, 0.0, 0.30),
  };
  scan_planner::ReferenceTrajectoryInitialization initialization;
  ASSERT_TRUE(scan_planner::initializeReferenceTrajectory(
      raw_guide, raw_guide.front(), raw_guide.back(),
      0.30, 0.50, 0.05, initialization));

  ASSERT_GE(initialization.parameterization_points.size(), 7U);
  EXPECT_TRUE(initialization.parameterization_points.front().isApprox(
      raw_guide.front(), 1.0e-12));
  EXPECT_TRUE(initialization.parameterization_points.back().isApprox(
      raw_guide.back(), 1.0e-12));
  EXPECT_NEAR(initialization.duration, 2.60, 1.0e-12);
  EXPECT_NEAR(
      initialization.time_step *
      static_cast<double>(initialization.parameterization_points.size() - 1),
      initialization.duration, 1.0e-12);

  const double first_displacement =
      (initialization.parameterization_points[1] -
      initialization.parameterization_points[0]).norm();
  const std::size_t middle =
      initialization.parameterization_points.size() / 2;
  const double middle_displacement =
      (initialization.parameterization_points[middle] -
      initialization.parameterization_points[middle - 1]).norm();
  EXPECT_GT(middle_displacement, 2.0 * first_displacement);

  const std::vector<Eigen::Vector3d> derivatives{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::MatrixXd control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      initialization.time_step,
      initialization.parameterization_points,
      derivatives, control_points));
  scan_planner::UniformBspline trajectory(
      control_points, 3, initialization.time_step);
  ASSERT_TRUE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      trajectory, 0.30, 0.50, 0.01));

  // 与旧实现做同条件配对：旧实现把等空间点直接当成
  // 等时间点，起终点零速/零加速与中间恒速位移冲突，导致
  // 后续只能整体拉长时间。
  const double old_time_step = initialization.duration /
      static_cast<double>(initialization.spatial_guide.size() - 1);
  Eigen::MatrixXd old_control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      old_time_step, initialization.spatial_guide,
      derivatives, old_control_points));
  scan_planner::UniformBspline old_trajectory(
      old_control_points, 3, old_time_step);
  ASSERT_TRUE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      old_trajectory, 0.30, 0.50, 0.01));

  EXPECT_LT(trajectory.getTimeSum(), old_trajectory.getTimeSum());
  EXPECT_LE(trajectory.getTimeSum(), 4.0);
}

TEST(ReferenceTrajectoryInitialization, KeepsOneVoxelTerminalCaptureSegment)
{
  const std::vector<Eigen::Vector3d> raw_guide{
      Eigen::Vector3d(0.0, 0.0, 0.30),
      Eigen::Vector3d(0.05, 0.0, 0.30),
  };
  scan_planner::ReferenceTrajectoryInitialization initialization;

  ASSERT_TRUE(scan_planner::initializeReferenceTrajectory(
      raw_guide, raw_guide.front(), raw_guide.back(),
      0.55, 0.40, 0.05, initialization));
  ASSERT_GE(initialization.parameterization_points.size(), 7U);
  EXPECT_TRUE(initialization.parameterization_points.front().isApprox(
      raw_guide.front(), 1.0e-12));
  EXPECT_TRUE(initialization.parameterization_points.back().isApprox(
      raw_guide.back(), 1.0e-12));
  EXPECT_TRUE(initialization.spatial_guide.front().isApprox(
      raw_guide.front(), 1.0e-12));
  EXPECT_TRUE(initialization.spatial_guide.back().isApprox(
      raw_guide.back(), 1.0e-12));
  EXPECT_GT(initialization.duration, 0.0);
}

TEST(ReferenceTrajectoryInitialization, KeepsSpatialTurnAnchorsForSafetyChecks)
{
  const std::vector<Eigen::Vector3d> raw_guide{
      Eigen::Vector3d(0.0, 0.0, 0.30),
      Eigen::Vector3d(0.40, 0.0, 0.30),
      Eigen::Vector3d(0.40, 0.30, 0.35),
  };
  scan_planner::ReferenceTrajectoryInitialization initialization;
  ASSERT_TRUE(scan_planner::initializeReferenceTrajectory(
      raw_guide, raw_guide.front(), raw_guide.back(),
      0.30, 0.50, 0.05, initialization));

  bool found_turn_anchor = false;
  for (const Eigen::Vector3d &point : initialization.spatial_guide)
  {
    if (point.isApprox(raw_guide[1], 1.0e-12))
      found_turn_anchor = true;
  }
  EXPECT_TRUE(found_turn_anchor);
  for (std::size_t index = 1;
       index < initialization.spatial_guide.size(); ++index)
  {
    EXPECT_LE(
        (initialization.spatial_guide[index] -
        initialization.spatial_guide[index - 1]).norm(),
        0.05 + 1.0e-12);
  }
}

TEST(ReferenceTrajectoryInitialization, CruiseBoundaryAvoidsArtificialTimeStretch)
{
  const std::vector<Eigen::Vector3d> raw_guide{
      Eigen::Vector3d(0.0, 0.0, 0.30),
      Eigen::Vector3d(0.60, 0.0, 0.30),
      Eigen::Vector3d(1.20, 0.0, 0.30),
  };
  const Eigen::Vector3d start_velocity = Eigen::Vector3d::Zero();
  const Eigen::Vector3d terminal_velocity(0.24, 0.0, 0.0);
  scan_planner::ReferenceTrajectoryInitialization cruising;
  ASSERT_TRUE(scan_planner::initializeReferenceTrajectory(
      raw_guide, raw_guide.front(), raw_guide.back(),
      0.30, 0.50, 0.05, start_velocity, terminal_velocity, cruising));

  EXPECT_NEAR(cruising.start_speed, 0.0, 1.0e-12);
  EXPECT_NEAR(cruising.terminal_speed, 0.24, 1.0e-12);
  EXPECT_NEAR(cruising.duration, 4.312, 1.0e-12);

  const std::vector<Eigen::Vector3d> derivatives{
      start_velocity, terminal_velocity,
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::MatrixXd cruising_control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      cruising.time_step, cruising.parameterization_points,
      derivatives, cruising_control_points));
  scan_planner::UniformBspline cruising_trajectory(
      cruising_control_points, 3, cruising.time_step);
  ASSERT_TRUE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      cruising_trajectory, 0.30, 0.50, 0.01));

  scan_planner::ReferenceTrajectoryInitialization stopping_profile;
  ASSERT_TRUE(scan_planner::initializeReferenceTrajectory(
      raw_guide, raw_guide.front(), raw_guide.back(),
      0.30, 0.50, 0.05, stopping_profile));
  Eigen::MatrixXd mismatched_control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      stopping_profile.time_step,
      stopping_profile.parameterization_points,
      derivatives, mismatched_control_points));
  scan_planner::UniformBspline mismatched_trajectory(
      mismatched_control_points, 3, stopping_profile.time_step);
  ASSERT_TRUE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      mismatched_trajectory, 0.30, 0.50, 0.01));

  EXPECT_LT(
      cruising_trajectory.getTimeSum(),
      mismatched_trajectory.getTimeSum());
  // 精确三次样条边界仍会为 0.50m/s² 加速度上限留出拟合余量，
  // 但 1.2m 首段必须保持在 6.5s 内，不能回到旧错配配置的约 10s。
  EXPECT_LE(cruising_trajectory.getTimeSum(), 6.5);
}

TEST(ReferenceTrajectoryInitialization, SubgridConnectorCannotStretchMovingStart)
{
  const std::vector<Eigen::Vector3d> raw_guide{
      Eigen::Vector3d(0.0, 0.0, 0.30),
      Eigen::Vector3d(0.0006, -0.0001, 0.3005),
      Eigen::Vector3d(0.0101, 0.0143, 0.2950),
      Eigen::Vector3d(0.002, 0.050, 0.30),
      Eigen::Vector3d(0.0, 1.20, 0.30)};
  constexpr double max_velocity = 0.45;
  constexpr double max_acceleration = 0.80;
  const Eigen::Vector3d start_velocity(0.014, 0.349, 0.0);
  const Eigen::Vector3d terminal_velocity(0.0, 0.42, 0.0);

  scan_planner::ReferenceTrajectoryInitialization moving;
  ASSERT_TRUE(scan_planner::initializeReferenceTrajectory(
      raw_guide, raw_guide.front(), raw_guide.back(), max_velocity,
      max_acceleration, 0.05, start_velocity, terminal_velocity, moving));
  ASSERT_GT(moving.spatial_guide.size(), 2U);
  const Eigen::Vector3d initial_tangent =
      (moving.spatial_guide[1] - moving.spatial_guide[0]).normalized();
  const Eigen::Vector3d expected_tangent =
      (raw_guide[3] - raw_guide[0]).normalized();
  EXPECT_NEAR(initial_tangent.cross(expected_tangent).norm(), 0.0, 1.0e-12);
  EXPECT_GT(moving.start_speed, 0.34);

  const std::vector<Eigen::Vector3d> moving_derivatives{
      start_velocity, terminal_velocity,
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::MatrixXd moving_control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      moving.time_step, moving.parameterization_points,
      moving_derivatives, moving_control_points));
  scan_planner::UniformBspline moving_trajectory(
      moving_control_points, 3, moving.time_step);
  ASSERT_TRUE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      moving_trajectory, max_velocity, max_acceleration, 0.01));

  scan_planner::ReferenceTrajectoryInitialization stationary;
  ASSERT_TRUE(scan_planner::initializeReferenceTrajectory(
      raw_guide, raw_guide.front(), raw_guide.back(), max_velocity,
      max_acceleration, 0.05, Eigen::Vector3d::Zero(), terminal_velocity,
      stationary));
  const std::vector<Eigen::Vector3d> stationary_derivatives{
      Eigen::Vector3d::Zero(), terminal_velocity,
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::MatrixXd stationary_control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      stationary.time_step, stationary.parameterization_points,
      stationary_derivatives, stationary_control_points));
  scan_planner::UniformBspline stationary_trajectory(
      stationary_control_points, 3, stationary.time_step);
  ASSERT_TRUE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      stationary_trajectory, max_velocity, max_acceleration, 0.01));

  EXPECT_LT(
      moving_trajectory.getTimeSum(), stationary_trajectory.getTimeSum());
  EXPECT_LT(moving_trajectory.getTimeSum(), 4.20);
}

TEST(ReferenceTrajectoryInitialization, ReservedProfileAccelerationAvoidsWholeSplineSlowdown)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.30),
      Eigen::Vector3d(0.60, 0.0, 0.30),
      Eigen::Vector3d(1.20, 0.0, 0.30),
  };
  constexpr double max_velocity = 0.45;
  constexpr double max_acceleration = 0.80;
  const Eigen::Vector3d start_velocity = Eigen::Vector3d::Zero();
  const Eigen::Vector3d terminal_velocity(0.42, 0.0, 0.0);
  const std::vector<Eigen::Vector3d> derivatives{
      start_velocity, terminal_velocity,
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};

  const auto build_trajectory = [&](const double profile_scale) {
      scan_planner::ReferenceTrajectoryInitialization initialization;
      EXPECT_TRUE(scan_planner::initializeReferenceTrajectory(
          guide, guide.front(), guide.back(), max_velocity,
          max_acceleration * profile_scale, 0.05,
          start_velocity, terminal_velocity, initialization));
      Eigen::MatrixXd control_points;
      EXPECT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
          initialization.time_step,
          initialization.parameterization_points,
          derivatives, control_points));
      scan_planner::UniformBspline trajectory(
          control_points, 3, initialization.time_step);
      EXPECT_TRUE(scan_planner::rescaleTrajectoryToPhysicalLimits(
          trajectory, max_velocity, max_acceleration, 0.01));
      return trajectory;
    };

  auto full_limit_profile = build_trajectory(1.0);
  auto reserved_profile = build_trajectory(0.5);
  const double full_limit_duration = full_limit_profile.getTimeSum();
  const double reserved_duration = reserved_profile.getTimeSum();
  const double reserved_velocity_upper_bound =
      scan_planner::trajectoryMaximumVelocityUpperBound(reserved_profile);
  double remaining_ratio = 1.0;

  EXPECT_LT(reserved_duration, 0.85 * full_limit_duration);
  EXPECT_LT(reserved_duration, 3.50);
  EXPECT_GT(reserved_velocity_upper_bound, 0.44);
  EXPECT_LE(reserved_velocity_upper_bound, max_velocity * 1.01 + 1.0e-4);
  reserved_profile.setPhysicalLimits(
      max_velocity, max_acceleration, 0.01);
  EXPECT_TRUE(reserved_profile.checkFeasibility(remaining_ratio, false));
}

TEST(ReferenceTrajectoryInitialization, InvalidInputsFailClosed)
{
  scan_planner::ReferenceTrajectoryInitialization initialization;
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX()};
  EXPECT_FALSE(scan_planner::initializeReferenceTrajectory(
      guide, guide.front(), guide.back(), 0.0, 0.50, 0.05,
      initialization));
  EXPECT_FALSE(scan_planner::initializeReferenceTrajectory(
      guide, guide.front(), guide.back(), 0.30, 0.0, 0.05,
      initialization));
  EXPECT_FALSE(scan_planner::initializeReferenceTrajectory(
      guide, guide.front(), guide.back(), 0.30, 0.50, 0.0,
      initialization));
}

TEST(TrajectoryProgress, FreeRunwayMovesTerminalSplineSupportPastObstacle)
{
  const auto parameterize_straight_reference =
      [](const double target_y, const int sample_count) {
        const double start_y = 3.65;
        const double max_vel = 0.30;
        const double max_acc = 0.50;
        const double total_length = target_y - start_y;
        const double acceleration_distance = max_vel * max_vel / max_acc;
        const double duration =
            (total_length - acceleration_distance) / max_vel +
            2.0 * max_vel / max_acc;
        const double time_step =
            duration / static_cast<double>(sample_count - 1);

        std::vector<Eigen::Vector3d> guide;
        guide.reserve(static_cast<std::size_t>(sample_count));
        for (int index = 0; index < sample_count; ++index)
        {
          const double ratio =
              static_cast<double>(index) /
              static_cast<double>(sample_count - 1);
          guide.emplace_back(
              -3.50, start_y + ratio * total_length, 0.30);
        }
        const std::vector<Eigen::Vector3d> derivatives{
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
        Eigen::MatrixXd control_points;
        scan_planner::UniformBspline::parameterizeToBspline(
            time_step, guide, derivatives, control_points);
        return control_points;
      };

  // v4 的 4.65 m 单点目标本身自由，但零末速参数化把最后三个支撑控制点
  // 全部拉回障碍区；这正是 optimizer 报 terminal occupied 的数值根因。
  const Eigen::MatrixXd old_control_points =
      parameterize_straight_reference(4.65, 21);
  ASSERT_EQ(old_control_points.cols(), 23);
  EXPECT_LT(old_control_points.rightCols(3).row(1).maxCoeff(), 4.65);
  EXPECT_NEAR(old_control_points(1, 20), 4.63525458, 1.0e-7);

  // 将局部目标前移一个 0.10 m 连续自由支撑区后，最后三个控制点均越过
  // 4.65 m 障碍边界，优化器可以找到离开障碍段的 out_id。
  const Eigen::MatrixXd runway_control_points =
      parameterize_straight_reference(4.75, 23);
  ASSERT_EQ(runway_control_points.cols(), 25);
  EXPECT_GT(runway_control_points.rightCols(3).row(1).minCoeff(), 4.65);
  EXPECT_NEAR(runway_control_points(1, 22), 4.73526312, 1.0e-7);
}

TEST(TrajectoryTiming, RescalesFinalRefinedCurveWithoutChangingGeometry)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 8);
  for (int index = 0; index < control_points.cols(); ++index)
  {
    const double coordinate = 0.08 * static_cast<double>(index * index);
    control_points(0, index) = coordinate;
    control_points(1, index) = coordinate;
    control_points(2, index) = 0.30;
  }
  scan_planner::UniformBspline trajectory(control_points, 3, 0.20);
  const double original_duration = trajectory.getTimeSum();
  scan_planner::UniformBspline original_velocity = trajectory.getDerivative();
  scan_planner::UniformBspline original_acceleration =
      original_velocity.getDerivative();
  std::vector<Eigen::Vector3d> original_samples;
  std::vector<Eigen::Vector3d> original_velocity_samples;
  std::vector<Eigen::Vector3d> original_acceleration_samples;
  for (int index = 0; index <= 10; ++index)
  {
    const double original_time =
        original_duration * static_cast<double>(index) / 10.0;
    original_samples.push_back(
        trajectory.evaluateDeBoorT(original_time));
    original_velocity_samples.push_back(
        original_velocity.evaluateDeBoorT(original_time));
    original_acceleration_samples.push_back(
        original_acceleration.evaluateDeBoorT(original_time));
  }

  EXPECT_TRUE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      trajectory, 10.0, 0.50, 0.01));
  EXPECT_GT(trajectory.getTimeSum(), original_duration);
  const double applied_ratio =
      trajectory.getTimeSum() / original_duration;
  scan_planner::UniformBspline scaled_velocity = trajectory.getDerivative();
  scan_planner::UniformBspline scaled_acceleration =
      scaled_velocity.getDerivative();
  double remaining_ratio = 1.0;
  EXPECT_TRUE(trajectory.checkFeasibility(remaining_ratio, false));
  for (int index = 0; index <= 10; ++index)
  {
    const double scaled_time =
        trajectory.getTimeSum() * static_cast<double>(index) / 10.0;
    const Eigen::Vector3d scaled_sample =
        trajectory.evaluateDeBoorT(scaled_time);
    EXPECT_TRUE(scaled_sample.isApprox(original_samples[index], 1.0e-10));
    EXPECT_TRUE(scaled_velocity.evaluateDeBoorT(scaled_time).isApprox(
        original_velocity_samples[index] / applied_ratio, 1.0e-10));
    EXPECT_TRUE(scaled_acceleration.evaluateDeBoorT(scaled_time).isApprox(
        original_acceleration_samples[index] /
            (applied_ratio * applied_ratio),
        1.0e-10));
  }
}

TEST(TrajectoryTiming, RejectsNonFiniteCurveBeforeFeasibilityCheck)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 6);
  for (int index = 0; index < control_points.cols(); ++index)
    control_points(0, index) = 0.1 * static_cast<double>(index);

  Eigen::MatrixXd nan_points = control_points;
  nan_points(1, 3) = std::numeric_limits<double>::quiet_NaN();
  scan_planner::UniformBspline nan_trajectory(nan_points, 3, 0.20);
  EXPECT_FALSE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      nan_trajectory, 0.30, 0.50, 0.01));

  scan_planner::UniformBspline infinite_knot_trajectory(
      control_points, 3, 0.20);
  Eigen::VectorXd knots = infinite_knot_trajectory.getKnot();
  knots(4) = std::numeric_limits<double>::infinity();
  infinite_knot_trajectory.setKnot(knots);
  EXPECT_FALSE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      infinite_knot_trajectory, 0.30, 0.50, 0.01));
}

TEST(TrajectoryTiming, RejectsNonIncreasingKnotBeforeFeasibilityCheck)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 6);
  scan_planner::UniformBspline trajectory(control_points, 3, 0.20);
  Eigen::VectorXd knots = trajectory.getKnot();
  knots(5) = knots(4);
  trajectory.setKnot(knots);

  EXPECT_FALSE(scan_planner::rescaleTrajectoryToPhysicalLimits(
      trajectory, 0.30, 0.50, 0.01));
}

TEST(TrajectoryTiming, RatioMultiplicationOverflowFailsWithoutThrowing)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 6);
  for (int index = 0; index < control_points.cols(); ++index)
    control_points(0, index) = static_cast<double>(index);
  scan_planner::UniformBspline trajectory(control_points, 3, 1.0);
  const double tiny_velocity_limit =
      1.0 / (0.995 * std::numeric_limits<double>::max());
  ASSERT_TRUE(std::isfinite(tiny_velocity_limit));
  ASSERT_GT(tiny_velocity_limit, 0.0);

  bool result = true;
  EXPECT_NO_THROW(
      result = scan_planner::rescaleTrajectoryToPhysicalLimits(
          trajectory, tiny_velocity_limit, 100.0, 0.0));
  EXPECT_FALSE(result);
  EXPECT_TRUE(trajectory.getKnot().allFinite());
}

TEST(TrajectoryTiming, ScaledKnotOverflowFailsWithoutMutation)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 6);
  for (int index = 0; index < control_points.cols(); ++index)
    control_points(0, index) = 1.0e307 * static_cast<double>(index);
  const double interval = std::numeric_limits<double>::max() / 7.0;
  scan_planner::UniformBspline trajectory(control_points, 3, interval);
  const Eigen::VectorXd original_knots = trajectory.getKnot();
  trajectory.setPhysicalLimits(0.15, 100.0, 0.0);
  double required_ratio = 1.0;
  ASSERT_FALSE(trajectory.checkFeasibility(required_ratio, false));
  ASSERT_GT(required_ratio, 1.0);

  bool result = true;
  EXPECT_NO_THROW(
      result = scan_planner::rescaleTrajectoryToPhysicalLimits(
          trajectory, 0.15, 100.0, 0.0));
  EXPECT_FALSE(result);
  EXPECT_TRUE(
      (trajectory.getKnot().array() == original_knots.array()).all());
  EXPECT_DOUBLE_EQ(trajectory.getInterval(), interval);
}

TEST(TrajectoryTiming, DynamicToleranceCompatibilityUsesContinuousMargins)
{
  EXPECT_TRUE(scan_planner::dynamicFeasibilityTolerancesCompatible(
      0.30, 0.50, 0.01, 0.005, 0.01));
  EXPECT_FALSE(scan_planner::dynamicFeasibilityTolerancesCompatible(
      0.30, 0.50, 0.02, 0.005, 0.01));
  EXPECT_FALSE(scan_planner::dynamicFeasibilityTolerancesCompatible(
      0.30, 0.50, 0.01, 0.005,
      std::numeric_limits<double>::quiet_NaN()));
}

TEST(TrajectoryProgress, ReferenceCorridorDoesNotAddStartBiasTwice)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.015, 0.3),
      Eigen::Vector3d(0.0, 0.015, 0.3)};
  const auto trajectory = makePolylineTrajectory({
      Eigen::Vector3d(0.015, 0.0, 0.3),
      guide[1],
      guide.back()});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02);

  // 起点最小二乘偏置与后续折角收缩分别小于 0.02 m；旧算法把两者
  // 相加成约 0.03 m 并误拒，归一化后仍分别受原硬门约束。
  EXPECT_TRUE(result.safe);
  EXPECT_NEAR(result.initial_guide_progress, 0.015, 1.0e-9);
  EXPECT_NEAR(result.maximum_guide_progress_lead, 0.015, 1.0e-4);
}

TEST(TrajectoryProgress, ReferenceCorridorRejectsExcessiveInitialGuideJump)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(0.03, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3)};
  const auto trajectory = makePolylineTrajectory({guide[1], guide.back()});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.05, 0.02);

  // 基线归一化不能掩盖真正从 guide 后段起步的捷径。
  EXPECT_FALSE(result.safe);
  EXPECT_NEAR(result.initial_guide_progress, 0.03, 1.0e-9);
  EXPECT_GT(result.maximum_guide_progress_lead, 0.02);
}

TEST(TrajectoryProgress, ReferenceCorridorAllowsOnlyBoundedInitialFitTolerance)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3)};
  const auto trajectory = makePolylineTrajectory({
      Eigen::Vector3d(0.0201, 0.0, 0.3), guide.back()});

  const auto strict_result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02);
  const auto fitted_result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02, 0.01, 0.001);

  EXPECT_FALSE(strict_result.safe);
  EXPECT_TRUE(fitted_result.safe);
  EXPECT_NEAR(fitted_result.initial_guide_progress, 0.0201, 1.0e-9);
  EXPECT_LE(
      fitted_result.maximum_relative_guide_progress_lead,
      0.02 + 1.0e-9);
}

TEST(TrajectoryProgress, InitialFitToleranceCannotHideLaterShortcut)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.03, 0.3),
      Eigen::Vector3d(0.0, 0.03, 0.3)};
  const auto trajectory = makePolylineTrajectory({
      guide.front(), guide[1], guide.back()});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02, 0.01, 0.001);

  EXPECT_FALSE(result.safe);
  EXPECT_NEAR(result.initial_guide_progress, 0.0, 1.0e-9);
  EXPECT_GT(result.maximum_relative_guide_progress_lead, 0.02);
}

TEST(TrajectoryProgress, RelativeProgressUsesOnlyBoundedMeasurementTolerance)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0202, 0.3),
      Eigen::Vector3d(0.0, 0.0202, 0.3)};
  const auto trajectory = makePolylineTrajectory({
      guide.front(), guide[1], guide.back()});

  const auto strict_result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02);
  const auto measured_result =
      scan_planner::checkTrajectoryReferenceCorridor(
          trajectory, guide, 0.10, 0.02, 0.01, 0.0, 0.001);

  EXPECT_FALSE(strict_result.safe);
  EXPECT_TRUE(measured_result.safe);
  EXPECT_GT(measured_result.maximum_relative_guide_progress_lead, 0.02);
  EXPECT_LT(measured_result.maximum_relative_guide_progress_lead, 0.021);
}

TEST(TrajectoryProgress, MeasurementToleranceCannotHideLargerShortcut)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.022, 0.3),
      Eigen::Vector3d(0.0, 0.022, 0.3)};
  const auto trajectory = makePolylineTrajectory({
      guide.front(), guide[1], guide.back()});

  const auto result = scan_planner::checkTrajectoryReferenceCorridor(
      trajectory, guide, 0.10, 0.02, 0.01, 0.0, 0.001);

  EXPECT_FALSE(result.safe);
  EXPECT_GT(result.maximum_relative_guide_progress_lead, 0.021);
}

TEST(TrajectoryProgress, ExactStartBoundaryRemovesLeastSquaresDerivativeDrift)
{
  std::vector<Eigen::Vector3d> guide;
  for (int index = 0; index < 8; ++index)
  {
    const double ratio = static_cast<double>(index) / 7.0;
    guide.emplace_back(0.36 * ratio, 0.43 * ratio, 0.3);
  }
  const double interval = 0.25;
  const Eigen::Vector3d position = guide.front();
  const Eigen::Vector3d velocity(0.01, 0.012, 0.0);
  const Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  const std::vector<Eigen::Vector3d> derivatives{
      velocity, Eigen::Vector3d::Zero(), acceleration,
      Eigen::Vector3d::Zero()};
  Eigen::MatrixXd control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      interval, guide, derivatives, control_points));
  scan_planner::UniformBspline trajectory(control_points, 3, interval);
  auto trajectory_velocity = trajectory.getDerivative();
  auto trajectory_acceleration = trajectory_velocity.getDerivative();

  EXPECT_TRUE(trajectory.evaluateDeBoorT(0.0).isApprox(position, 1.0e-12));
  EXPECT_TRUE(
      trajectory_velocity.evaluateDeBoorT(0.0).isApprox(velocity, 1.0e-12));
  EXPECT_TRUE(
      trajectory_acceleration.evaluateDeBoorT(0.0).isApprox(
          acceleration, 1.0e-12));
}

TEST(TrajectoryProgress, ExactStartFitKeepsMonotoneForwardControlPolygon)
{
  // 参考点沿 start->target 的投影严格递增，但非均匀采样和横向折弯会让
  // 无约束最小二乘的第 4、5 个控制点投影从 0.295 m 回退到 0.257 m。
  const std::vector<Eigen::Vector3d> guide{
      {0.0, 0.0, 0.0},
      {0.0772545886446173, 0.08697147695760028, 0.0},
      {0.1274947524335193, 0.18176119872073498, 0.0},
      {0.10584251908672361, 0.27899847620002466, 0.0},
      {0.20542267947793802, 0.24707242844029234, 0.0},
      {0.2851470311466255, 0.2749596869495411, 0.0},
      {0.2636192265458779, 0.3694165191994064, 0.0},
      {0.36, 0.43, 0.0}};
  const std::vector<Eigen::Vector3d> derivatives{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::MatrixXd control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      0.25, guide, derivatives, control_points));

  Eigen::Vector3d direction = guide.back() - guide.front();
  direction.z() = 0.0;
  direction.normalize();
  for (Eigen::Index index = 1; index < control_points.cols(); ++index)
  {
    EXPECT_GE(
        (control_points.col(index) - control_points.col(index - 1)).dot(
            direction),
        -1.0e-12);
  }

  scan_planner::UniformBspline trajectory(control_points, 3, 0.25);
  const auto velocity = trajectory.getDerivative();
  for (int index = 0; index <= 200; ++index)
  {
    const double time = trajectory.getTimeSum() *
        static_cast<double>(index) / 200.0;
    EXPECT_GE(velocity.evaluateDeBoorT(time).dot(direction), -1.0e-12);
  }
}

TEST(TrajectoryProgress, ForwardFitPreservesBoundedMeasuredStartDrift)
{
  std::vector<Eigen::Vector3d> guide;
  for (int index = 0; index < 8; ++index)
  {
    const double ratio = static_cast<double>(index) / 7.0;
    guide.emplace_back(0.42 * ratio, 0.42 * ratio, -0.05 * ratio);
  }
  const Eigen::Vector3d measured_velocity(0.000188, -0.000283, -0.00209);
  const std::vector<Eigen::Vector3d> derivatives{
      measured_velocity, Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::MatrixXd control_points;
  ASSERT_TRUE(scan_planner::parameterizeCubicBsplineWithExactStartBoundary(
      0.25, guide, derivatives, control_points));

  scan_planner::UniformBspline trajectory(control_points, 3, 0.25);
  const auto velocity = trajectory.getDerivative();
  EXPECT_TRUE(
      velocity.evaluateDeBoorT(0.0).isApprox(measured_velocity, 1.0e-12));
  const auto progress = scan_planner::checkTrajectoryForwardProgress(
      trajectory, guide.front(), guide.back(), 0.02, 0.02);
  EXPECT_TRUE(progress.safe);
}

TEST(TrajectoryProgress, ExactStartBoundaryRejectsInvalidInputs)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 8);
  EXPECT_FALSE(scan_planner::enforceCubicBsplineStartBoundary(
      control_points, 0.0, Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));

  Eigen::MatrixXd too_short = Eigen::MatrixXd::Zero(3, 2);
  EXPECT_FALSE(scan_planner::enforceCubicBsplineStartBoundary(
      too_short, 0.1, Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
}

TEST(TrajectoryDiagnostics, BoundedSamplesCoverWholeTrajectory)
{
  Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(3, 8);
  for (int index = 0; index < control_points.cols(); ++index)
    control_points(0, index) = static_cast<double>(index);
  scan_planner::UniformBspline trajectory(control_points, 3, 0.5);

  const auto samples = scan_planner::sampleTrajectoryGeometry(
      trajectory, 64, 0.01);

  ASSERT_TRUE(samples.valid);
  EXPECT_GT(samples.total_count, 64U);
  EXPECT_TRUE(samples.truncated);
  ASSERT_EQ(samples.points.size(), 64U);
  EXPECT_TRUE(samples.points.front().isApprox(
      trajectory.evaluateDeBoorT(0.0), 1.0e-12));
  EXPECT_TRUE(samples.points.back().isApprox(
      trajectory.evaluateDeBoorT(trajectory.getTimeSum()), 1.0e-12));

  const double velocity_upper_bound =
      scan_planner::trajectoryMaximumVelocityUpperBound(trajectory);
  ASSERT_TRUE(std::isfinite(velocity_upper_bound));
  EXPECT_GT(velocity_upper_bound, 0.0);
  const auto velocity = trajectory.getDerivative();
  for (int index = 0; index <= 100; ++index)
  {
    const double time = trajectory.getTimeSum() *
        static_cast<double>(index) / 100.0;
    EXPECT_LE(
        velocity.evaluateDeBoorT(time).norm(),
        velocity_upper_bound + 1.0e-12);
  }
}

TEST(TrajectoryDiagnostics, BoundedReferenceSamplesPreserveOrderAndEndpoints)
{
  std::vector<Eigen::Vector3d> guide;
  for (int index = 0; index < 70; ++index)
    guide.emplace_back(
        0.01 * static_cast<double>(index),
        0.001 * static_cast<double>(index * index), 0.3);

  const auto samples = scan_planner::sampleOrderedReferenceGeometry(
      guide, 64);

  ASSERT_TRUE(samples.valid);
  EXPECT_EQ(samples.total_count, 70U);
  EXPECT_TRUE(samples.truncated);
  ASSERT_EQ(samples.points.size(), 64U);
  EXPECT_TRUE(samples.points.front().isApprox(guide.front(), 1.0e-12));
  EXPECT_TRUE(samples.points.back().isApprox(guide.back(), 1.0e-12));
  for (std::size_t index = 1; index < samples.points.size(); ++index)
    EXPECT_GT(samples.points[index].x(), samples.points[index - 1].x());
}

TEST(TrajectoryDiagnostics, StationaryClassificationUsesAllControlPoints)
{
  Eigen::MatrixXd stationary_points = Eigen::MatrixXd::Zero(3, 6);
  stationary_points.row(2).setConstant(0.3);
  scan_planner::UniformBspline stationary(stationary_points, 3, 1.0);
  EXPECT_TRUE(scan_planner::trajectoryIsStationary(stationary));

  stationary_points(0, 4) = 1.0e-3;
  scan_planner::UniformBspline moving(stationary_points, 3, 1.0);
  EXPECT_FALSE(scan_planner::trajectoryIsStationary(moving));
}
