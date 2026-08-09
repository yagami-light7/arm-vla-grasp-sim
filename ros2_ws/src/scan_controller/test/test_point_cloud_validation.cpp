#include <cmath>
#include <cstring>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "scan_controller/point_cloud_validation.hpp"

namespace
{

sensor_msgs::msg::PointCloud2 makeCanonicalEmptyCloud()
{
  sensor_msgs::msg::PointCloud2 cloud;
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

sensor_msgs::msg::PointCloud2 makeSinglePointCloud(float z)
{
  auto cloud = makeCanonicalEmptyCloud();
  cloud.width = 1U;
  cloud.row_step = cloud.point_step;
  cloud.data.resize(cloud.point_step);
  const float coordinates[] = {1.0F, 2.0F, z};
  std::memcpy(cloud.data.data(), coordinates, sizeof(coordinates));
  return cloud;
}

TEST(PointCloudValidation, AcceptsCanonicalEmptyObservation)
{
  std::string error;
  EXPECT_TRUE(
    scan_controller::validPointCloudLayout(
      makeCanonicalEmptyCloud(), error)) << error;
  EXPECT_TRUE(error.empty());
}

TEST(PointCloudValidation, RejectsMalformedEmptyObservation)
{
  auto cloud = makeCanonicalEmptyCloud();
  cloud.height = 0U;
  std::string error;
  EXPECT_FALSE(scan_controller::validPointCloudLayout(cloud, error));
  EXPECT_FALSE(error.empty());

  cloud = makeCanonicalEmptyCloud();
  cloud.data.push_back(0U);
  EXPECT_FALSE(scan_controller::validPointCloudLayout(cloud, error));
  EXPECT_FALSE(error.empty());

  cloud = makeCanonicalEmptyCloud();
  std::swap(cloud.fields[0], cloud.fields[1]);
  EXPECT_FALSE(scan_controller::validPointCloudLayout(cloud, error));
  EXPECT_FALSE(error.empty());

  cloud = makeCanonicalEmptyCloud();
  cloud.is_dense = false;
  EXPECT_FALSE(scan_controller::validPointCloudLayout(cloud, error));
  EXPECT_FALSE(error.empty());
}

TEST(PointCloudValidation, StillRejectsMissingFieldsAndNonFinitePoints)
{
  auto cloud = makeCanonicalEmptyCloud();
  cloud.fields.pop_back();
  std::string error;
  EXPECT_FALSE(scan_controller::validPointCloudLayout(cloud, error));
  EXPECT_FALSE(error.empty());

  cloud = makeSinglePointCloud(std::nanf(""));
  EXPECT_FALSE(scan_controller::validPointCloudLayout(cloud, error));
  EXPECT_FALSE(error.empty());
}

TEST(PointCloudValidation, AcceptsFiniteNonEmptyObservation)
{
  std::string error;
  EXPECT_TRUE(
    scan_controller::validPointCloudLayout(
      makeSinglePointCloud(3.0F), error)) << error;
  EXPECT_TRUE(error.empty());
}

TEST(PointCloudValidation, AcceptsExplicitFreeRayEndpointField)
{
  auto cloud = makeSinglePointCloud(3.0F);
  cloud.fields.push_back(
    sensor_msgs::msg::PointField()
    .set__name("ray_endpoint_type")
    .set__offset(12U)
    .set__datatype(sensor_msgs::msg::PointField::UINT8)
    .set__count(1U));
  cloud.point_step = 13U;
  cloud.row_step = cloud.point_step;
  cloud.data.resize(cloud.point_step);
  cloud.data[12] = 0U;

  std::string error;
  EXPECT_TRUE(scan_controller::validPointCloudLayout(cloud, error)) << error;
  EXPECT_TRUE(error.empty());
}

}  // 匿名命名空间
