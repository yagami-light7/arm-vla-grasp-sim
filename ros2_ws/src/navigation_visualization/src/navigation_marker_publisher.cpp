#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace navigation_visualization
{

class NavigationMarkerPublisher final : public rclcpp::Node
{
public:
  /**
   * @brief 创建 PCT 路径标记与机器人实际轨迹发布器
   */
  NavigationMarkerPublisher()
  : Node("navigation_marker_publisher")
  {
    const auto global_path_topic = declare_parameter<std::string>(
      "topics.global_path", "/pct/global_path");
    const auto odometry_topic = declare_parameter<std::string>(
      "topics.odometry", "/body_pose");
    const auto robot_path_topic = declare_parameter<std::string>(
      "topics.robot_path", "/visualization/robot_path");
    const auto path_cylinders_topic = declare_parameter<std::string>(
      "topics.path_cylinders", "/visualization/pct_path_cylinders");
    const auto start_goal_topic = declare_parameter<std::string>(
      "topics.start_goal", "/visualization/pct_start_goal");

    cylinder_radius_m_ = declare_parameter<double>(
      "path_cylinder_radius_m", 0.035);
    robot_path_min_spacing_m_ = declare_parameter<double>(
      "robot_path_min_spacing_m", 0.05);
    robot_path_max_points_ = declare_parameter<int>(
      "robot_path_max_points", 2000);
    validateParameters();

    const auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1))
      .reliable()
      .transient_local();
    path_cylinders_publisher_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
      path_cylinders_topic, latched_qos);
    start_goal_publisher_ =
      create_publisher<visualization_msgs::msg::MarkerArray>(
      start_goal_topic, latched_qos);
    robot_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
      robot_path_topic, latched_qos);

    global_path_subscription_ = create_subscription<nav_msgs::msg::Path>(
      global_path_topic,
      latched_qos,
      std::bind(
        &NavigationMarkerPublisher::handleGlobalPath,
        this,
        std::placeholders::_1));
    odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      odometry_topic,
      rclcpp::SensorDataQoS(),
      std::bind(
        &NavigationMarkerPublisher::handleOdometry,
        this,
        std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "导航可视化标记已启用：global_path=%s, odometry=%s",
      global_path_topic.c_str(),
      odometry_topic.c_str());
  }

private:
  /**
   * @brief 验证显示参数，防止无界轨迹或非法 Marker 尺寸
   * @return 无；参数非法时抛出 std::invalid_argument
   */
  void validateParameters() const
  {
    if (!std::isfinite(cylinder_radius_m_) || cylinder_radius_m_ <= 0.0) {
      throw std::invalid_argument("path_cylinder_radius_m 必须是有限正数");
    }
    if (
      !std::isfinite(robot_path_min_spacing_m_) ||
      robot_path_min_spacing_m_ <= 0.0)
    {
      throw std::invalid_argument("robot_path_min_spacing_m 必须是有限正数");
    }
    if (robot_path_max_points_ < 2) {
      throw std::invalid_argument("robot_path_max_points 至少为 2");
    }
  }

  /**
   * @brief 判断 Pose 的位置与四元数是否全为有限数
   * @param pose 待检查的 Pose
   * @return 全部字段有限时返回 true
   */
  static bool poseIsFinite(const geometry_msgs::msg::Pose & pose)
  {
    return
      std::isfinite(pose.position.x) &&
      std::isfinite(pose.position.y) &&
      std::isfinite(pose.position.z) &&
      std::isfinite(pose.orientation.x) &&
      std::isfinite(pose.orientation.y) &&
      std::isfinite(pose.orientation.z) &&
      std::isfinite(pose.orientation.w);
  }

  /**
   * @brief 计算两个位置之间的三维距离
   * @param lhs 第一个位置
   * @param rhs 第二个位置
   * @return 两点欧氏距离，单位 m
   */
  static double positionDistance(
    const geometry_msgs::msg::Point & lhs,
    const geometry_msgs::msg::Point & rhs)
  {
    return std::hypot(
      std::hypot(lhs.x - rhs.x, lhs.y - rhs.y),
      lhs.z - rhs.z);
  }

  /**
   * @brief 生成把 Marker 局部 Z 轴旋转到指定三维方向的四元数
   * @param dx 方向 X 分量
   * @param dy 方向 Y 分量
   * @param dz 方向 Z 分量
   * @param length 三维方向模长
   * @return ROS xyzw 四元数
   */
  static geometry_msgs::msg::Quaternion alignZAxisQuaternion(
    const double dx,
    const double dy,
    const double dz,
    const double length)
  {
    geometry_msgs::msg::Quaternion quaternion;
    const double unit_z = dz / length;
    if (unit_z >= 1.0 - 1.0e-12) {
      quaternion.w = 1.0;
      return quaternion;
    }
    if (unit_z <= -1.0 + 1.0e-12) {
      quaternion.x = 1.0;
      return quaternion;
    }

    const double scale = std::sqrt(2.0 * (1.0 + unit_z));
    quaternion.x = -dy / length / scale;
    quaternion.y = dx / length / scale;
    quaternion.z = 0.0;
    quaternion.w = 0.5 * scale;
    return quaternion;
  }

  /**
   * @brief 接收 PCT 全局路径并发布圆柱路径、起点和终点
   * @param message PCT 发布的世界系地面 Path
   * @return 无；非法或不足两点的 Path 会被拒绝
   */
  void handleGlobalPath(const nav_msgs::msg::Path::SharedPtr message)
  {
    if (message->header.frame_id.empty() || message->poses.size() < 2) {
      RCLCPP_WARN(get_logger(), "忽略缺少 frame 或不足两点的 PCT Path");
      return;
    }
    for (const auto & pose : message->poses) {
      if (!poseIsFinite(pose.pose)) {
        RCLCPP_WARN(get_logger(), "忽略包含 NaN/Inf 的 PCT Path");
        return;
      }
    }

    visualization_msgs::msg::MarkerArray cylinders;
    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    cylinders.markers.push_back(delete_all);

    int marker_id = 0;
    for (std::size_t index = 1; index < message->poses.size(); ++index) {
      const auto & start = message->poses[index - 1].pose.position;
      const auto & end = message->poses[index].pose.position;
      const double dx = end.x - start.x;
      const double dy = end.y - start.y;
      const double dz = end.z - start.z;
      const double length = std::hypot(std::hypot(dx, dy), dz);
      if (length <= 1.0e-6) {
        continue;
      }

      visualization_msgs::msg::Marker cylinder;
      cylinder.header = message->header;
      cylinder.ns = "pct_path_cylinders";
      cylinder.id = marker_id++;
      cylinder.type = visualization_msgs::msg::Marker::CYLINDER;
      cylinder.action = visualization_msgs::msg::Marker::ADD;
      cylinder.pose.position.x = 0.5 * (start.x + end.x);
      cylinder.pose.position.y = 0.5 * (start.y + end.y);
      cylinder.pose.position.z = 0.5 * (start.z + end.z) + 0.025;
      cylinder.pose.orientation = alignZAxisQuaternion(dx, dy, dz, length);
      cylinder.scale.x = 2.0 * cylinder_radius_m_;
      cylinder.scale.y = 2.0 * cylinder_radius_m_;
      cylinder.scale.z = length;
      cylinder.color.r = 0.04F;
      cylinder.color.g = 0.90F;
      cylinder.color.b = 1.00F;
      cylinder.color.a = 0.80F;
      cylinders.markers.push_back(cylinder);
    }
    path_cylinders_publisher_->publish(cylinders);
    publishStartGoal(*message);
  }

  /**
   * @brief 发布 PCT Path 的起点与终点球体
   * @param path 已完成有限性验证的 PCT Path
   * @return 无
   */
  void publishStartGoal(const nav_msgs::msg::Path & path)
  {
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_all);

    const auto add_sphere = [&](const int id, const geometry_msgs::msg::Point & point,
        const float red, const float green, const float blue)
      {
        visualization_msgs::msg::Marker marker;
        marker.header = path.header;
        marker.ns = "pct_start_goal";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position = point;
        marker.pose.position.z += 0.12;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.24;
        marker.color.r = red;
        marker.color.g = green;
        marker.color.b = blue;
        marker.color.a = 1.0F;
        markers.markers.push_back(marker);
      };
    add_sphere(0, path.poses.front().pose.position, 0.10F, 1.00F, 0.20F);
    add_sphere(1, path.poses.back().pose.position, 1.00F, 0.20F, 0.10F);
    start_goal_publisher_->publish(markers);
  }

  /**
   * @brief 接收机器人里程计并维护有界的实际运动轨迹
   * @param message 世界系 base_link Odometry
   * @return 无；位移不足采样间距时不追加点
   */
  void handleOdometry(const nav_msgs::msg::Odometry::SharedPtr message)
  {
    if (message->header.frame_id.empty() || !poseIsFinite(message->pose.pose)) {
      return;
    }
    if (
      !robot_path_frame_.empty() &&
      robot_path_frame_ != message->header.frame_id)
    {
      robot_path_.clear();
    }
    robot_path_frame_ = message->header.frame_id;

    if (
      !robot_path_.empty() &&
      positionDistance(
        robot_path_.back().pose.position,
        message->pose.pose.position) < robot_path_min_spacing_m_)
    {
      return;
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header = message->header;
    pose.pose = message->pose.pose;
    robot_path_.push_back(pose);
    while (robot_path_.size() > static_cast<std::size_t>(robot_path_max_points_)) {
      robot_path_.pop_front();
    }

    nav_msgs::msg::Path path;
    path.header = message->header;
    path.poses.assign(robot_path_.begin(), robot_path_.end());
    robot_path_publisher_->publish(path);
  }

  double cylinder_radius_m_{0.035};
  double robot_path_min_spacing_m_{0.05};
  int robot_path_max_points_{2000};
  std::string robot_path_frame_;
  std::deque<geometry_msgs::msg::PoseStamped> robot_path_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    path_cylinders_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    start_goal_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr robot_path_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;
};

}  // namespace navigation_visualization

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<navigation_visualization::NavigationMarkerPublisher>());
  rclcpp::shutdown();
  return 0;
}
