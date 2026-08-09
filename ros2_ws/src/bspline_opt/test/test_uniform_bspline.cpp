#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <bspline_opt/uniform_bspline.h>

TEST(UniformBspline, EvaluatesLinearControlPoints)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  for (int i = 0; i < points.cols(); ++i) points(0, i) = static_cast<double>(i);

  scan_planner::UniformBspline spline(points, 3, 1.0);
  EXPECT_NEAR(spline.evaluateDeBoorT(0.0).x(), 1.0, 1e-9);
  EXPECT_NEAR(spline.evaluateDeBoorT(2.0).x(), 3.0, 1e-9);
  EXPECT_NEAR(spline.evaluateDeBoorT(2.0).y(), 0.0, 1e-9);
}

TEST(UniformBspline, DerivativeMatchesLinearSlope)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  for (int i = 0; i < points.cols(); ++i) points(0, i) = static_cast<double>(i);

  auto derivative = scan_planner::UniformBspline(points, 3, 0.5).getDerivative();
  const Eigen::Vector3d velocity = derivative.evaluateDeBoorT(0.75);
  EXPECT_NEAR(velocity.x(), 2.0, 1e-9);
  EXPECT_NEAR(velocity.y(), 0.0, 1e-9);
  EXPECT_NEAR(velocity.z(), 0.0, 1e-9);
}

TEST(UniformBspline, UniformTimeScalingPreservesCurveAndRestoresFeasibility)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  for (int i = 0; i < points.cols(); ++i)
  {
    points(0, i) = static_cast<double>(i);
    points(1, i) = 0.1 * static_cast<double>(i * i);
  }

  scan_planner::UniformBspline spline(points, 3, 0.2);
  const double original_duration = spline.getTimeSum();
  std::vector<Eigen::Vector3d> original_samples;
  for (int i = 0; i <= 8; ++i)
  {
    original_samples.push_back(
        spline.evaluateDeBoorT(original_duration * static_cast<double>(i) / 8.0));
  }

  spline.setPhysicalLimits(1.0, 100.0, 0.0);
  double required_ratio = 1.0;
  ASSERT_FALSE(spline.checkFeasibility(required_ratio, false));
  ASSERT_GT(required_ratio, 1.0);

  const double applied_ratio = 1.01 * required_ratio;
  spline.scaleTimeUniformly(applied_ratio);
  EXPECT_NEAR(
      spline.getTimeSum(), original_duration * applied_ratio, 1e-12);
  for (int i = 0; i <= 8; ++i)
  {
    const Eigen::Vector3d scaled_sample = spline.evaluateDeBoorT(
        spline.getTimeSum() * static_cast<double>(i) / 8.0);
    EXPECT_TRUE(scaled_sample.isApprox(original_samples[i], 1e-10));
  }

  double remaining_ratio = 1.0;
  EXPECT_TRUE(spline.checkFeasibility(remaining_ratio, false));
}

TEST(UniformBspline, RejectsDiagonalVelocityByVectorNorm)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  for (int i = 0; i < points.cols(); ++i)
  {
    points(0, i) = 0.8 * static_cast<double>(i);
    points(1, i) = 0.8 * static_cast<double>(i);
  }

  scan_planner::UniformBspline spline(points, 3, 1.0);
  spline.setPhysicalLimits(1.0, 100.0, 0.0);
  double required_ratio = 1.0;
  EXPECT_FALSE(spline.checkFeasibility(required_ratio, false));
  EXPECT_GT(required_ratio, 1.0);

  spline.scaleTimeUniformly(1.01 * required_ratio);
  EXPECT_TRUE(spline.checkFeasibility(required_ratio, false));
}

TEST(UniformBspline, RejectsDiagonalAccelerationByVectorNorm)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  for (int i = 0; i < points.cols(); ++i)
  {
    const double index = static_cast<double>(i);
    points(0, i) = index * index;
    points(1, i) = index * index;
  }

  scan_planner::UniformBspline spline(points, 3, 1.0);
  spline.setPhysicalLimits(100.0, 2.1, 0.0);
  double required_ratio = 1.0;
  EXPECT_FALSE(spline.checkFeasibility(required_ratio, false));
  EXPECT_GT(required_ratio, 1.0);

  spline.scaleTimeUniformly(1.01 * required_ratio);
  EXPECT_TRUE(spline.checkFeasibility(required_ratio, false));
}

TEST(UniformBspline, FeasibilityRejectsNonFiniteControlPoint)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  points(0, 2) = std::numeric_limits<double>::quiet_NaN();
  scan_planner::UniformBspline spline(points, 3, 1.0);
  spline.setPhysicalLimits(1.0, 1.0, 0.01);

  double required_ratio = 1.0;
  EXPECT_FALSE(spline.checkFeasibility(required_ratio, false));
  EXPECT_TRUE(std::isinf(required_ratio));
}

TEST(UniformBspline, FeasibilityRejectsNonIncreasingKnot)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  for (int index = 0; index < points.cols(); ++index)
    points(0, index) = static_cast<double>(index);
  scan_planner::UniformBspline spline(points, 3, 1.0);
  Eigen::VectorXd knots = spline.getKnot();
  knots(5) = knots(4);
  spline.setKnot(knots);
  spline.setPhysicalLimits(1.0, 1.0, 0.01);

  double required_ratio = 1.0;
  EXPECT_FALSE(spline.checkFeasibility(required_ratio, false));
  EXPECT_TRUE(std::isinf(required_ratio));
}

TEST(UniformBspline, FeasibilityRejectsOverflowingDerivative)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  for (int index = 0; index < points.cols(); ++index)
    points(0, index) = index % 2 == 0
                           ? -0.5 * std::numeric_limits<double>::max()
                           : 0.5 * std::numeric_limits<double>::max();
  scan_planner::UniformBspline spline(points, 3, 1.0);
  spline.setPhysicalLimits(1.0, 1.0, 0.01);

  double required_ratio = 1.0;
  EXPECT_FALSE(spline.checkFeasibility(required_ratio, false));
  EXPECT_TRUE(std::isinf(required_ratio));
}

TEST(UniformBspline, OverflowingUniformScaleKeepsOriginalTiming)
{
  Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 6);
  const double interval = std::numeric_limits<double>::max() / 7.0;
  scan_planner::UniformBspline spline(points, 3, interval);
  const Eigen::VectorXd original_knots = spline.getKnot();

  EXPECT_THROW(spline.scaleTimeUniformly(2.0), std::overflow_error);
  EXPECT_TRUE(
      (spline.getKnot().array() == original_knots.array()).all());
  EXPECT_DOUBLE_EQ(spline.getInterval(), interval);
}
