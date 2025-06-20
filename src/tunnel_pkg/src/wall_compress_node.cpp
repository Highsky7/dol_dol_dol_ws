#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

ros::Publisher pub;

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
    // ROS PointCloud2 메시지를 PCL PointCloud로 변환
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*input, cloud);

    // 모든 포인트의 z 값을 0으로 설정 (평면으로 압축)
    for (auto& pt : cloud.points) {
        pt.z = 0.0f;
    }

    // VoxelGrid 필터로 다운샘플링 (~m)
    pcl::PointCloud<pcl::PointXYZ> cloud_filtered;
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud.makeShared());
    sor.setLeafSize(0.25f, 0.25f, 0.25f);
    sor.filter(cloud_filtered);

    // 필터링된 PCL 점군을 다시 ROS 메시지로 변환
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(cloud_filtered, output);

    // 입력 메시지의 헤더(프레임, 타임스탬프 등)를 복사
    output.header = input->header;

    // 결과 발행
    pub.publish(output);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "wall_compress_node");
    ros::NodeHandle nh;

    // 입력 토픽 /detected_wall 구독
    ros::Subscriber sub = nh.subscribe("/detected_wall", 1, cloudCallback);
    // 출력 토픽 /compressed_wall 발행
    pub = nh.advertise<sensor_msgs::PointCloud2>("/compressed_wall", 1);

    ros::spin();
    return 0;
}
