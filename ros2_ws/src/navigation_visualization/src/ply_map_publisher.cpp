#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

using namespace std::chrono_literals;

class PlyMapPublisher final : public rclcpp::Node
{
public:
    PlyMapPublisher()
    : Node("ply_map_publisher")
    {
        const std::string ply_path = 
            declare_parameter<std::string>("ply_path", "");
        const std::string topic = 
            declare_parameter<std::string>("topic", "/map/ply");
        const std::string frame_id = 
            declare_parameter<std::string>("frame_id", "world");
        const double voxel_leaf_size =
            declare_parameter<double>("voxel_leaf_size_m", 10.0);

        if (ply_path.empty()){
            throw std::runtime_error("参数ply_path不能为空");
        }
        if (voxel_leaf_size < 0.0){
            throw std::runtime_error("参数voxel_leaf_size_m不能为负数");
        }

        // 体素降采样 ply 包含三角面，因此先按PolygonMesh读取，再提取顶点
        pcl::PolygonMesh mesh;
        if (pcl::io::loadPLYFile(ply_path, mesh) < 0){ // 读取ply文件
            throw std::runtime_error("无法读取ply文件: " + ply_path);
        }
        
        auto raw_cloud =
            std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();// 转换为xyz点云
        pcl::fromPCLPointCloud2(mesh.cloud, *raw_cloud);

        auto output_cloud = 
            std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        if (voxel_leaf_size > 0.0){
            // Rviz只负责显示，不需要保留碰撞网格的全部顶点
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setInputCloud(raw_cloud);
            voxel_filter.setLeafSize(
                static_cast<float>(voxel_leaf_size),
                static_cast<float>(voxel_leaf_size),
                static_cast<float>(voxel_leaf_size));
            voxel_filter.filter(*output_cloud);
        }
        else
        {
            *output_cloud = *raw_cloud;
        }

        // pcl 点云转换为ROS2消息
        pcl::toROSMsg(*output_cloud, cloud_message_);
        cloud_message_.header.frame_id = frame_id;

        // 静态地图使用可靠、可缓存的QoS，后启动的Rviz也可以收到
        const auto qos = 
            rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

        // 创建publihser    
        publisher_ =
            create_publisher<sensor_msgs::msg::PointCloud2>(topic, qos);

        // use_sim_time=true 时等待/clock非零，避免发布无效时间戳
        publish_timer_ = create_wall_timer(
            100ms,
            [this]()
            {
                const auto stamp = now();
                if (stamp.nanoseconds() <= 0)
                {
                    return;
                }
            
                cloud_message_.header.stamp = stamp;
                publisher_->publish(cloud_message_);
                publish_timer_->cancel();

                const auto point_count = static_cast<std::size_t>(cloud_message_.width) * 
                                        static_cast<std::size_t>(cloud_message_.height);
                RCLCPP_INFO(
                    get_logger(),
                    "已发布ply地图，点云数量: %zu, 时间戳: %u.%09u",
                    point_count,
                    cloud_message_.header.stamp.sec,
                    cloud_message_.header.stamp.nanosec);
            });
    }
private:
    sensor_msgs::msg::PointCloud2 cloud_message_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    try{
        rclcpp::spin(std::make_shared<PlyMapPublisher>());
    }
    catch (const std::exception & error){
        RCLCPP_FATAL(rclcpp::get_logger("ply_map_publisher"), "Error: %s", error.what());
        rclcpp::shutdown();
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}