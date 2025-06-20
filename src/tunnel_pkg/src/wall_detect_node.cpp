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

class WallDetector {
public:
    WallDetector() {
        // 구독자 설정: Velodyne 포인트 클라우드 구독
        sub_ = nh_.subscribe("/velodyne_points", 1, &WallDetector::pointCloudCallback, this);
        // 퍼블리셔 설정: 벽 포인트 클라우드 퍼블리시
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/detected_wall", 1);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        // ROS PointCloud2 메시지를 PCL PointCloud로 변환 (xyz 만 사용)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud);

        // VoxelGrid 다운샘플링 필터 적용
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(0.1f, 0.1f, 0.1f);            // [파라미터] 10cm voxel 크기 (XYZ 각각) - 필요시 조절
        voxel.filter(*cloud_down);
        // [변경] 입력 데이터를 다운샘플링하여 처리 속도 향상 및 일관된 결과 사용

        // PassThrough 필터 적용 (예: z축 방향 바닥/천장 제거)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_down);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.3, 1.5);                // [파라미터] z 범위 제한값 (환경에 따라 바닥/천장 높이 조정)
        pass.filter(*cloud_filtered);
        // [변경] 센서 기준 너무 낮거나 높은 위치의 점 제거 (불필요한 바닥/천장 평면 배제)

        // 평면 세그멘테이션 객체와 보조 객체 생성
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);              // (옵션) 계수 최적화 활성화
        seg.setModelType(pcl::SACMODEL_PLANE);          // 평면 모델 유형
        seg.setMethodType(pcl::SAC_RANSAC);             // RANSAC 방법 사용
        seg.setMaxIterations(1000);                     // (옵션) RANSAC 최대 반복 횟수
        seg.setDistanceThreshold(0.1);                  // [파라미터] 평면으로 간주할 거리 임계값 (0.1m 내의 점들을 인라이어로)
        // ※ DistanceThreshold 값이 작으면 평면 적합 정확도 ↑, 크면 노이즈에 민감 ↓:contentReference[oaicite:7]{index=7}

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointCloud<pcl::PointXYZ>::Ptr wall_cloud(new pcl::PointCloud<pcl::PointXYZ>);  
        // 최종 벽 포인트들을 담을 클라우드 (초기 비어있음)

        // 최대 5개의 평면까지 추출 시도
        for (int i = 0; i < 5; ++i) {
            if (cloud_filtered->points.empty()) break;  // 남은 점이 없으면 종료

            // 평면 모델 계수와 인라이어 인덱스 객체
            pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

            // 현재 남은 점군에서 가장 큰 평면 추출 (세그멘테이션)
            seg.setInputCloud(cloud_filtered);
            seg.segment(*inliers, *coeff);
            if (inliers->indices.empty()) {
                // 더 이상 평면을 찾지 못함
                break;
            }

            // 추출한 평면의 법선 계산 (계수: ax+by+cz+d=0에서 법선 = [a,b,c])
            Eigen::Vector3f plane_normal(coeff->values[0], coeff->values[1], coeff->values[2]);
            plane_normal.normalize();
            float nz = plane_normal[2];

            // 평면 법선의 Z성분을 확인하여 수직에 가까운지 판단
            bool isWall = (std::fabs(nz) < 0.1736f);     // 0.1736 ≈ cos(80°), |nz| < 0.1736이면 법선이 거의 수평 -> 평면은 거의 수직
            // [변경] ±10° 이내로 수직인 평면만 벽으로 간주 (벽은 Z축 대비 수직, 법선의 Z성분이 작음):contentReference[oaicite:8]{index=8}

            if (isWall) {
                // 벽 평면으로 확인된 경우: 인라이어 점들을 추출하여 wall_cloud에 누적
                pcl::PointCloud<pcl::PointXYZ>::Ptr plane_points(new pcl::PointCloud<pcl::PointXYZ>);
                extract.setInputCloud(cloud_filtered);
                extract.setIndices(inliers);
                extract.setNegative(false);             // 인라이어 추출
                extract.filter(*plane_points);
                // 추출된 평면 점들을 최종 벽 포인트 클라우드에 추가
                wall_cloud->points.insert(wall_cloud->points.end(), 
                                          plane_points->points.begin(), plane_points->points.end());
            }
            // 인라이어 점들을 입력 클라우드에서 제거하여 다음 평면 검색 준비
            extract.setInputCloud(cloud_filtered);
            extract.setIndices(inliers);
            extract.setNegative(true);                  // 인라이어 제거하고 나머지 유지
            pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
            extract.filter(*remaining);
            cloud_filtered.swap(remaining);
        }

        // 벽 포인트들의 PCL 클라우드가 준비됨
        if (!wall_cloud->points.empty()) {
            // (옵션) 이상치 제거 필터 적용하여 노이즈 감소
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(wall_cloud);
            sor.setMeanK(50);                           // [파라미터] 이웃 점 50개 사용
            sor.setStddevMulThresh(1.0);                // [파라미터] 허용 표준편차 거리: 1.0 -> 평균거리 + 1σ 이상은 제거
            sor.filter(*wall_cloud);
            // [옵션 변경] 벽 클라우드에서 주변과 동떨어진 노이즈 점 제거 (MeanK, StddevMulThresh 조절 가능):contentReference[oaicite:9]{index=9}

            // wall_cloud의 width/height 갱신
            wall_cloud->width = wall_cloud->points.size();
            wall_cloud->height = 1;
            wall_cloud->is_dense = true;
        }

        // ROS 메시지로 변환 후 퍼블리시
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*wall_cloud, output_msg);
        output_msg.header = cloud_msg->header;           // 입력과 동일한 헤더 사용 (frame_id, timestamp 유지)
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
