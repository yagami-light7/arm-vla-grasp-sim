#include "scan_controller/point_cloud_validation.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <sensor_msgs/msg/point_field.hpp>

namespace scan_controller
{
namespace
{

const sensor_msgs::msg::PointField * findField(
  const sensor_msgs::msg::PointCloud2 &cloud, const std::string &name)
{
  for (const auto &field : cloud.fields) {
    if (field.name == name) {
      return &field;
    }
  }
  return nullptr;
}

bool readFiniteCoordinate(
  const std::vector<std::uint8_t> &data, std::size_t offset,
  std::uint8_t datatype)
{
  if (datatype == sensor_msgs::msg::PointField::FLOAT32) {
    if (offset + sizeof(float) > data.size()) {
      return false;
    }
    float value = 0.0F;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return std::isfinite(value);
  }
  if (datatype == sensor_msgs::msg::PointField::FLOAT64) {
    if (offset + sizeof(double) > data.size()) {
      return false;
    }
    double value = 0.0;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return std::isfinite(value);
  }
  return false;
}

}  // 匿名命名空间

bool validPointCloudLayout(
  const sensor_msgs::msg::PointCloud2 &cloud, std::string &error)
{
  error.clear();
  if (
    cloud.height == 0U || cloud.point_step == 0U ||
    cloud.is_bigendian)
  {
    error = "PointCloud2 高度、point_step 或字节序非法";
    return false;
  }

  // bridge 在有效原始深度点分类后没有障碍端点或可用
  // 自由射线时发布 canonical empty；它仍是新鲜传感器观测。
  if (cloud.width == 0U) {
    const char *expected_names[] = {"x", "y", "z"};
    const std::uint32_t expected_offsets[] = {0U, 4U, 8U};
    bool canonical =
      cloud.height == 1U && cloud.point_step == 12U &&
      cloud.row_step == 0U && cloud.data.empty() &&
      cloud.is_dense && cloud.fields.size() == 3U;
    for (
      std::size_t index = 0U;
      canonical && index < cloud.fields.size(); ++index)
    {
      const auto &field = cloud.fields[index];
      canonical =
        field.name == expected_names[index] &&
        field.offset == expected_offsets[index] &&
        field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
        field.count == 1U;
    }
    if (!canonical) {
      error = "空 PointCloud2 必须使用 canonical xyz32 非组织布局";
      return false;
    }
  }

  const std::uint64_t minimum_row_step =
    static_cast<std::uint64_t>(cloud.width) * cloud.point_step;
  const std::uint64_t required_size =
    static_cast<std::uint64_t>(cloud.row_step) * cloud.height;
  if (
    cloud.row_step < minimum_row_step ||
    required_size != cloud.data.size())
  {
    error = "PointCloud2 row_step、point_step 与 data 长度不一致";
    return false;
  }

  const auto *x_field = findField(cloud, "x");
  const auto *y_field = findField(cloud, "y");
  const auto *z_field = findField(cloud, "z");
  const sensor_msgs::msg::PointField *fields[] = {
    x_field, y_field, z_field};
  for (const auto *field : fields) {
    if (
      field == nullptr || field->count != 1U ||
      (field->datatype != sensor_msgs::msg::PointField::FLOAT32 &&
      field->datatype != sensor_msgs::msg::PointField::FLOAT64))
    {
      error = "PointCloud2 必须包含标量 float32/float64 x、y、z 字段";
      return false;
    }
    const std::uint32_t field_size =
      field->datatype == sensor_msgs::msg::PointField::FLOAT32 ?
      sizeof(float) : sizeof(double);
    if (field->offset + field_size > cloud.point_step) {
      error = "PointCloud2 坐标字段超出 point_step";
      return false;
    }
  }

  for (std::uint32_t row = 0; row < cloud.height; ++row) {
    const std::size_t row_offset =
      static_cast<std::size_t>(row) * cloud.row_step;
    for (std::uint32_t column = 0; column < cloud.width; ++column) {
      const std::size_t point_offset =
        row_offset + static_cast<std::size_t>(column) * cloud.point_step;
      for (const auto *field : fields) {
        if (!readFiniteCoordinate(
            cloud.data, point_offset + field->offset, field->datatype))
        {
          error = "PointCloud2 含非有限坐标或越界字段";
          return false;
        }
      }
    }
  }
  return true;
}

}  // 命名空间 scan_controller
