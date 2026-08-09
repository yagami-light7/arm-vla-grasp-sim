#ifndef SCAN_CONTROLLER__POINT_CLOUD_VALIDATION_HPP_
#define SCAN_CONTROLLER__POINT_CLOUD_VALIDATION_HPP_

#include <string>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace scan_controller
{

/// 校验控制器用于新鲜度门禁的 PointCloud2 布局与有限坐标。
bool validPointCloudLayout(
  const sensor_msgs::msg::PointCloud2 &cloud, std::string &error);

}  // 命名空间 scan_controller

#endif  // SCAN_CONTROLLER__POINT_CLOUD_VALIDATION_HPP_
