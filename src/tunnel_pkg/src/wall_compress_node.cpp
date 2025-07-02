#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <sensor_msgs/point_cloud2_iterator.h>

ros::Publisher pub;

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{


    // 1) PointCloud2 메시지 복사
    sensor_msgs::PointCloud2 projected = *input;

    // 2) 인플레이스 Z=0 투영
    for (sensor_msgs::PointCloud2Iterator<float> it(projected, "z"); it != it.end(); ++it) {
        *it = 0.0f;
    }

    // 3) PCL 변환 (이제 모든 z가 0인 상태)
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(projected, cloud);


    // 4) ApproximateVoxelGrid 다운샘플링
    pcl::PointCloud<pcl::PointXYZ> cloud_filtered;
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud.makeShared());
    sor.setLeafSize(0.4f, 0.4f, 0.4f);
    sor.filter(cloud_filtered);

    // 5) 다시 PointCloud2로 변환 후 퍼블리시
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(cloud_filtered, output);
    output.header = projected.header;  // 복사한 메시지의 헤더(프레임/타임스탬프) 유지
    pub.publish(output);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "wall_compress_node");
    ros::NodeHandle nh;

    // 입력 토픽 /detected_wall 구독
      ros::Subscriber sub = nh.subscribe(
          "/detected_wall", 1, cloudCallback,
          ros::TransportHints().tcpNoDelay(true)
      );

    pub = nh.advertise<sensor_msgs::PointCloud2>("/compressed_wall", 1);

    ros::AsyncSpinner spinner(4);  // CPU 코어 수나 테스트 결과에 맞게 설정
    spinner.start();
    ros::waitForShutdown();
    return 0;
}
