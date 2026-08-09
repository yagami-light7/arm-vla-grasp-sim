#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <plan_manage/reference_velocity.h>

TEST(ReferenceVelocity, LowPassRejectsQuadrupedGaitOscillation)
{
  scan_planner::ReferenceVelocityLowPassFilter filter;
  Eigen::Vector3d filtered = Eigen::Vector3d::Zero();
  constexpr double mean_speed = 0.30;
  constexpr double pi = 3.14159265358979323846;
  for (int sample = 0; sample <= 250; ++sample)
  {
    const double time_sec = 0.02 * sample;
    const double oscillation = 0.24 * std::sin(2.0 * pi * 4.0 * time_sec);
    filtered = filter.update(
        Eigen::Vector3d(mean_speed + oscillation, 0.0, 0.0),
        1'000'000'000LL + static_cast<std::int64_t>(time_sec * 1.0e9),
        0.30);
  }

  EXPECT_NEAR(filtered.x(), mean_speed, 0.04);
  EXPECT_NEAR(filtered.y(), 0.0, 1.0e-12);
  EXPECT_NEAR(filtered.z(), 0.0, 1.0e-12);
}

TEST(ReferenceVelocity, ZeroTimeConstantPreservesMeasuredVelocity)
{
  scan_planner::ReferenceVelocityLowPassFilter filter;
  EXPECT_TRUE(filter.update(
      Eigen::Vector3d(0.1, 0.2, 0.3), 1, 0.0).isApprox(
          Eigen::Vector3d(0.1, 0.2, 0.3)));
  EXPECT_TRUE(filter.update(
      Eigen::Vector3d(0.4, 0.5, 0.6), 2, 0.0).isApprox(
          Eigen::Vector3d(0.4, 0.5, 0.6)));
}

TEST(ReferenceVelocity, LateSampleCannotRewindFilter)
{
  scan_planner::ReferenceVelocityLowPassFilter filter;
  const Eigen::Vector3d first = filter.update(
      Eigen::Vector3d(0.2, 0.0, 0.0), 200, 0.30);
  const Eigen::Vector3d late = filter.update(
      Eigen::Vector3d(-1.0, 0.0, 0.0), 100, 0.30);

  EXPECT_TRUE(late.isApprox(first));
}

TEST(ReferenceVelocity, ResetDropsPreviousMotionHistory)
{
  scan_planner::ReferenceVelocityLowPassFilter filter;
  filter.update(Eigen::Vector3d(0.4, 0.0, 0.0), 100, 0.30);
  filter.reset();
  EXPECT_FALSE(filter.initialized());
  EXPECT_TRUE(filter.update(
      Eigen::Vector3d::Zero(), 200, 0.30).isZero(1.0e-12));
}

TEST(ReferenceVelocity, RedirectsObservedRampSideSlipTowardLocalTarget)
{
  const Eigen::Vector3d measured(0.29, -0.327, 0.102);
  const Eigen::Vector3d start(0.593, -0.354, 0.442);
  const Eigen::Vector3d target(1.19, 0.0, 0.479);

  const Eigen::Vector3d projected =
      scan_planner::projectVelocityOntoReference(
          measured, start, target, 0.50);
  const Eigen::Vector3d direction = (target - start).normalized();

  EXPECT_GT(projected.x(), 0.0);
  EXPECT_GT(projected.y(), 0.0);
  EXPECT_NEAR(projected.cross(direction).norm(), 0.0, 1.0e-9);
  EXPECT_LE(projected.norm(), 0.50);
}

TEST(ReferenceVelocity, RemovesBackwardVelocity)
{
  const Eigen::Vector3d projected =
      scan_planner::projectVelocityOntoReference(
          Eigen::Vector3d(-0.2, 0.0, 0.0),
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d(1.0, 0.0, 0.0),
          0.50);

  EXPECT_TRUE(projected.isZero(1.0e-12));
}

TEST(ReferenceVelocity, ClampsForwardVelocity)
{
  const Eigen::Vector3d projected =
      scan_planner::projectVelocityOntoReference(
          Eigen::Vector3d(0.8, 0.0, 0.0),
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d(1.0, 0.0, 0.0),
          0.50);

  EXPECT_NEAR(projected.x(), 0.50, 1.0e-12);
  EXPECT_NEAR(projected.y(), 0.0, 1.0e-12);
  EXPECT_NEAR(projected.z(), 0.0, 1.0e-12);
}

TEST(ReferenceVelocity, IgnoresSubgridProjectionConnector)
{
  const Eigen::Vector3d start = Eigen::Vector3d::Zero();
  const std::vector<Eigen::Vector3d> guide{
      start,
      Eigen::Vector3d(0.0006, -0.0001, 0.0005),
      Eigen::Vector3d(0.0101, 0.0143, -0.0050),
      Eigen::Vector3d(0.002, 0.050, 0.0),
      Eigen::Vector3d(0.0, 1.20, 0.0)};
  const Eigen::Vector3d measured_velocity(0.04, 0.35, -0.01);

  const Eigen::Vector3d projected =
      scan_planner::projectVelocityOntoReferenceGuide(
          measured_velocity, start, guide, 0.05, 0.45);
  const Eigen::Vector3d expected_direction = guide[3].normalized();

  EXPECT_GT(projected.norm(), 0.34);
  EXPECT_NEAR(projected.cross(expected_direction).norm(), 0.0, 1.0e-12);
  EXPECT_LE(projected.norm(), 0.45);
}

TEST(ReferenceVelocity, MissingResolvableLookaheadFailsClosed)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0005, 0.0, 0.0)};

  EXPECT_TRUE(scan_planner::projectVelocityOntoReferenceGuide(
      Eigen::Vector3d(0.3, 0.0, 0.0), Eigen::Vector3d::Zero(), guide,
      0.05, 0.45).isZero(1.0e-12));
}

TEST(ReferenceVelocity, CruiseVelocityUsesOrderedGuideFinalTangent)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.5, 0.3)};

  const Eigen::Vector3d velocity =
      scan_planner::referenceCruiseVelocityAlongGuide(guide, 0.24, 0.30);

  EXPECT_NEAR(velocity.x(), 0.0, 1.0e-12);
  EXPECT_NEAR(velocity.y(), 0.24, 1.0e-12);
  EXPECT_NEAR(velocity.z(), 0.0, 1.0e-12);
}

TEST(ReferenceVelocity, CruiseVelocityIsClampedByPlannerLimit)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3)};

  const Eigen::Vector3d velocity =
      scan_planner::referenceCruiseVelocityAlongGuide(guide, 0.45, 0.30);

  EXPECT_NEAR(velocity.norm(), 0.30, 1.0e-12);
}

TEST(ReferenceVelocity, ZeroCruiseSpeedRetainsSafeStopBehavior)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(0.0, 0.0, 0.3),
      Eigen::Vector3d(1.0, 0.0, 0.3)};

  const Eigen::Vector3d velocity =
      scan_planner::referenceCruiseVelocityAlongGuide(guide, 0.0, 0.30);

  EXPECT_TRUE(velocity.isZero(1.0e-12));
}

TEST(ReferenceVelocity, DegenerateGuideFailsClosed)
{
  const std::vector<Eigen::Vector3d> guide{
      Eigen::Vector3d(1.0, 2.0, 0.3),
      Eigen::Vector3d(1.0, 2.0, 0.3)};

  const Eigen::Vector3d velocity =
      scan_planner::referenceCruiseVelocityAlongGuide(guide, 0.24, 0.30);

  EXPECT_TRUE(velocity.isZero(1.0e-12));
}
