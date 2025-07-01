#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/Float32.h>
#include <cmath>

// NavPVT 사용하여 heading 사용한 global yaw estimator 노드

// -PI ~ PI 범위로 각도를 정규화하는 함수
double wrap_angle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

class GlobalYawEstimator {
public:
    GlobalYawEstimator() : 
        has_prev_fix_(false), 
        filtered_yaw_(0.0), 
        filtered_init_(false) 
    {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");

        // 파라미터 로드
        pnh.param("alpha", alpha_, 0.1); // 필터 강도, 조금 더 부드럽게 기본값 조정
        pnh.param("min_distance_threshold", min_distance_threshold_, 0.2); // 최소 이동 거리 (미터)

        // 퍼블리셔 설정 (토픽명 유지)
        raw_yaw_pub_ = nh.advertise<std_msgs::Float32>("/raw_global_yaw", 10);
        filtered_yaw_pub_ = nh.advertise<std_msgs::Float32>("/global_yaw", 10);

        // 서브스크라이버 설정 (입력을 NavSatFix로 변경)
        fix_sub_ = nh.subscribe("/ublox_gps/fix", 10, &GlobalYawEstimator::fixCallback, this);

        ROS_INFO("Upgraded GlobalYawEstimator node started.");
        ROS_INFO(" -> Subscribing to: %s", fix_sub_.getTopic().c_str());
        ROS_INFO(" -> Publishing filtered yaw to: %s", filtered_yaw_pub_.getTopic().c_str());
        ROS_INFO(" -> Filter alpha: %.2f", alpha_);
        ROS_INFO(" -> Min distance threshold: %.2f m", min_distance_threshold_);
    }

private:
    // 두 지점(위도/경도) 사이의 거리를 Haversine 공식으로 계산 (미터 단위)
    double calculate_distance(double lat1, double lon1, double lat2, double lon2) const {
        const double R = 6371000.0; // 지구 반지름 (미터)
        double phi1 = lat1 * M_PI / 180.0;
        double phi2 = lat2 * M_PI / 180.0;
        double delta_phi = (lat2 - lat1) * M_PI / 180.0;
        double delta_lambda = (lon2 - lon1) * M_PI / 180.0;

        double a = std::sin(delta_phi / 2.0) * std::sin(delta_phi / 2.0) +
                   std::cos(phi1) * std::cos(phi2) *
                   std::sin(delta_lambda / 2.0) * std::sin(delta_lambda / 2.0);
        double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));

        return R * c;
    }

    // 두 지점(위도/경도) 사이의 방위각(Bearing)을 계산 (진북 기준, 라디안)
    double calculate_bearing(double lat1, double lon1, double lat2, double lon2) const {
        double phi1 = lat1 * M_PI / 180.0;
        double phi2 = lat2 * M_PI / 180.0;
        double delta_lambda = (lon2 - lon1) * M_PI / 180.0;

        double y = std::sin(delta_lambda) * std::cos(phi2);
        double x = std::cos(phi1) * std::sin(phi2) -
                   std::sin(phi1) * std::cos(phi2) * std::cos(delta_lambda);
        
        // atan2(y, x) 순서가 Bearing 공식에 맞음
        return std::atan2(y, x);
    }

    void fixCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
        // GPS 상태가 좋지 않으면 무시
        if (msg->status.status < sensor_msgs::NavSatStatus::STATUS_FIX) {
            ROS_WARN_THROTTLE(5.0, "Waiting for valid GPS fix...");
            return;
        }

        double current_lat = msg->latitude;
        double current_lon = msg->longitude;

        if (has_prev_fix_) {
            // 1단계: 유효성 검사 (최소 이동 거리 체크)
            double distance = calculate_distance(prev_lat_, prev_lon_, current_lat, current_lon);
            
            if (distance > min_distance_threshold_) {
                // 2단계: 원시 방위각 계산 (진북 기준 Bearing)
                double bearing_rad = calculate_bearing(prev_lat_, prev_lon_, current_lat, current_lon);

                // 4단계 (변환 먼저): Bearing을 ENU 좌표계의 Yaw로 변환
                // Bearing: 북쪽=0, 동쪽=PI/2. ENU Yaw: 동쪽=0, 북쪽=PI/2
                // ENU Yaw = -Bearing + PI/2
                double raw_yaw_enu = wrap_angle(-bearing_rad + (M_PI / 2.0));

                // 원시 Yaw 발행
                std_msgs::Float32 raw_msg;
                raw_msg.data = raw_yaw_enu;
                raw_yaw_pub_.publish(raw_msg);

                // 3단계: 출력 안정화 (지수 이동 평균 필터)
                if (!filtered_init_) {
                    filtered_yaw_ = raw_yaw_enu;
                    filtered_init_ = true;
                } else {
                    double diff = wrap_angle(raw_yaw_enu - filtered_yaw_);
                    filtered_yaw_ = wrap_angle(filtered_yaw_ + alpha_ * diff);
                }

                // 필터링된 최종 Yaw 발행
                std_msgs::Float32 filt_msg;
                filt_msg.data = filtered_yaw_;
                filtered_yaw_pub_.publish(filt_msg);

                ROS_INFO("Dist: %.2fm, Raw Yaw: %.2f deg, Filtered Yaw: %.2f deg",
                         distance,
                         raw_yaw_enu * 180.0 / M_PI, 
                         filtered_yaw_ * 180.0 / M_PI);

                // 현재 위치를 다음 계산을 위해 저장
                prev_lat_ = current_lat;
                prev_lon_ = current_lon;
            }
        } else {
            // 첫 번째 메시지는 위치만 저장
            prev_lat_ = current_lat;
            prev_lon_ = current_lon;
            has_prev_fix_ = true;
        }
    }

    ros::Subscriber fix_sub_;
    ros::Publisher raw_yaw_pub_, filtered_yaw_pub_;
    
    // 이전 GPS 위치 정보
    double prev_lat_, prev_lon_;
    bool has_prev_fix_;

    // 필터 관련 변수
    double filtered_yaw_;
    bool filtered_init_;
    
    // 파라미터
    double alpha_;
    double min_distance_threshold_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "global_yaw_estimator");
    GlobalYawEstimator gye;
    ros::spin();
    return 0;
}