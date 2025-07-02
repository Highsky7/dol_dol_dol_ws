#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>         // ROS ↔ PCL 메시지 변환
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/approximate_voxel_grid.h>

class WallDetector {
public:
    WallDetector() {
        // 구독자 설정: Velodyne 포인트 클라우드 구독
        sub_ = nh_.subscribe("/velodyne_points", 1,
            &WallDetector::pointCloudCallback, this,
            ros::TransportHints().tcpNoDelay(true)
        );
        // 퍼블리셔 설정: 벽 포인트 클라우드 퍼블리시
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/detected_wall", 1);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        // 1) ROS → PCL 변환
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud);

        // 2) ROI 필터1: 0 < x < 6
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(0.0f, 6.0f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi_x(new pcl::PointCloud<pcl::PointXYZ>);
        pass_x.filter(*cloud_roi_x);

        // 3) ROI 필터2: -3 < y < 3
        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(cloud_roi_x);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(-3.0f, 3.0f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi_y(new pcl::PointCloud<pcl::PointXYZ>);
        pass_y.filter(*cloud_roi_y);

        // 4) ROI 필터3: -1 < z < 2
        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(cloud_roi_y);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(-1.0f, 2.0f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>);
        pass_z.filter(*cloud_roi);

        // 5) ApproximateVoxelGrid 다운샘플링 (leaf = 0.05m)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud_roi);
        voxel.setLeafSize(0.05f, 0.05f, 0.05f);
        voxel.filter(*cloud_down);

        // 6) z 필터 (바닥/천장 제거)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_down);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.0, 1.0);
        pass.filter(*cloud_filtered);

        // 7) RANSAC 평면 세그멘테이션 설정
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(60);
        seg.setDistanceThreshold(0.02);

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointCloud<pcl::PointXYZ>::Ptr wall_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        // 최소 포인트 개수 기준
        constexpr size_t MIN_WALL_POINTS = 50;

        // 최대 10개의 평면까지 추출 시도
        for (int i = 0; i < 10; ++i) {
            if (cloud_filtered->points.empty()) break;

            // 7.1) 평면 모델 추출
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
            seg.setInputCloud(cloud_filtered);
            seg.segment(*inliers, *coeff);
            if (inliers->indices.empty()) break;

            // 평면 법선의 Z성분을 확인하여 수직에 가까운지 판단
            Eigen::Vector3f plane_normal(coeff->values[0],
                                         coeff->values[1],
                                         coeff->values[2]);
            plane_normal.normalize();
            float nz = plane_normal[2];
            const float cos_60 = 0.5f;
            const float cos_70 = 0.3420f;
            const float cos_75 = 0.2588f;
            const float cos_80 = 0.1736f;
            const float cos_85 = 0.0871f;
            const float cos_90 = 0.0f;
            bool isWall = (std::fabs(nz) < cos_80);

            if (isWall) {
                // 7.2) 인라이어 점 추출
                extract.setInputCloud(cloud_filtered);
                extract.setIndices(inliers);
                extract.setNegative(false);
                pcl::PointCloud<pcl::PointXYZ>::Ptr plane_points(new pcl::PointCloud<pcl::PointXYZ>);
                extract.filter(*plane_points);

                // ─── 포인트 개수 체크 추가 ───
                if (plane_points->points.size() < MIN_WALL_POINTS) {
                    // 작으면 이 평면만 제거하고 다음 반복
                    extract.setNegative(true);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
                    extract.filter(*tmp);
                    cloud_filtered.swap(tmp);
                    continue;
                }
                // ─────────────────────────────

                // 7.3) wall_cloud에 누적
                wall_cloud->points.insert(
                    wall_cloud->points.end(),
                    plane_points->points.begin(),
                    plane_points->points.end()
                );
            }

            // 7.4) 현재 평면 제거
            extract.setInputCloud(cloud_filtered);
            extract.setIndices(inliers);
            extract.setNegative(true);
            pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
            extract.filter(*remaining);
            cloud_filtered.swap(remaining);
        }

        // 8) 결과 퍼블리시
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*wall_cloud, output_msg);
        output_msg.header = cloud_msg->header;
        pub_.publish(output_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "wall_detect_node");
    WallDetector wd;
    ROS_INFO("wall_detect_node started.");
    ros::spin();
    return 0;
}
