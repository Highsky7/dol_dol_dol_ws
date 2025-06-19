#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>  // for pcl::fromROSMsg, toROSMsg
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

ros::Publisher wall_pub;

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_msg) {
    // 1) ROS 메시지를 PCL 포맷으로 변환
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*input_msg, *cloud);

    // 2) 선택적 다운샘플링 (VoxelGrid)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(0.02f, 0.02f, 0.02f);  // 2cm resolution
    voxel.filter(*cloud_down);

    // 3) RANSAC 평면 분할 설정
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(200);
    seg.setDistanceThreshold(0.05);  // 점이 평면으로부터 0.1 m(10 cm) 이내에 있으면 이 점을 이 평면의 부분이라고 보겠다는 의미.

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    seg.setInputCloud(cloud_down);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        ROS_WARN("평면을 찾지 못했습니다.");
        return;
    }

    // 4) 평면 법선 확인 (수평면 제거)
    Eigen::Vector3f normal(coefficients->values[0],
                           coefficients->values[1],
                           coefficients->values[2]);
    normal.normalize();
    double cos_angle = std::fabs(normal.dot(Eigen::Vector3f::UnitZ()));
    double cos15 = std::cos(15 * M_PI / 180.0);  // ≈0.9659
    if (cos_angle >= cos15) {
        // 수평면(바닥/천장)일 경우 제거하고 재시도
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rem(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_down);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_rem);
        if (!cloud_rem->empty()) {
            seg.setInputCloud(cloud_rem);
            seg.segment(*inliers, *coefficients);
            if (inliers->indices.empty()) {
                ROS_WARN("수평면 제거 후에도 평면을 찾지 못했습니다.");
                return;
            }
            normal = Eigen::Vector3f(coefficients->values[0],
                                     coefficients->values[1],
                                     coefficients->values[2]);
            normal.normalize();
            cos_angle = std::fabs(normal.dot(Eigen::Vector3f::UnitZ()));
        }
    }

    // 5) 수직 평면(벽) 판정
    double cos75 = std::cos(75 * M_PI / 180.0);  // ≈0.2588
    if (cos_angle < cos75) {
        // 벽 평면으로 간주 (15° 이내 수직)
    } else {
        ROS_WARN("감지된 평면이 수직 벽이 아닙니다.");
    }

    // 6) 원본 클라우드에서 평면 인라이어 점 추출
    pcl::PointCloud<pcl::PointXYZ>::Ptr wall_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    // extract.setInputCloud(cloud);      // ❌ original cloud
    extract.setInputCloud(cloud_down); // ✅ downsampled cloud
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*wall_cloud);

    // 7) 높이 필터링 (0.5m~2.0m)
    pcl::PointCloud<pcl::PointXYZ>::Ptr wall_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(wall_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.5, 2.0); // PassThrough 필터로 Z값(높이) 0.5m 이상, 2.0m 이하인 점만 통과
    pass.filter(*wall_filtered);

    // 8) 통계적 이상치 제거(Optional)
    if (!wall_filtered->empty()) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(wall_filtered);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*wall_filtered);
    }

    // 9) 결과 퍼블리시
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*wall_filtered, output_msg);
    output_msg.header = input_msg->header;
    wall_pub.publish(output_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "wall_filter_node");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("/velodyne_points", 1, pointCloudCallback);
    wall_pub = nh.advertise<sensor_msgs::PointCloud2>("/wall_points_filtered", 1);

    ROS_INFO("Wall filter node 시작: 포인트 클라우드를 대기 중...");
    ros::spin();
    return 0;
}


// 3) setDistanceThreshold 클수록 울퉁불퉁한 벽도 벽으로 인식
// 5) cos 각도를 90에 가깝게 할수록 수직인 것만 벽으로 봄
// 6) setInputCloud는 출력결과를 낼 때에 매개변수로 원본 클라우드나 다운샘플링된 클라우드를 입력 가능하다.