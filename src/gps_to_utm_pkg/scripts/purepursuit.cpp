#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Point.h>
#include <vector>
#include <algorithm>
#include <cmath>

class PurePursuit {
public:
    PurePursuit() : wheelbase_(0.75), has_latest_roi_(false) {
        // --- 파라미터 로드 ---
        // 기존 휠베이스 파라미터
        ros::param::param<double>("~wheelbase", wheelbase_, 0.75);

        // ======================= [핵심 수정 1: 동적 전방주시거리 파라미터 로드] =======================
        ros::param::param<double>("~throttle_min", THROTTLE_MIN_, 0.4);
        ros::param::param<double>("~throttle_max", THROTTLE_MAX_, 0.6);
        ros::param::param<double>("~min_lookahead_distance", MIN_LOOKAHEAD_DISTANCE_, 3.5);
        ros::param::param<double>("~max_lookahead_distance", MAX_LOOKAHEAD_DISTANCE_, 3.5);

        // 현재 스로틀 값을 안전한 최소값으로 초기화
        current_throttle_ = THROTTLE_MIN_;
        // ========================================================================================

        // --- 구독자 설정 ---
        roi_sub_ = nh_.subscribe("roi_path_marker", 10, &PurePursuit::roiCallback, this);

        // ======================= [핵심 수정 2: 스로틀 토픽 구독자 추가] =======================
        throttle_sub_ = nh_.subscribe("/auto_throttle", 10, &PurePursuit::throttleCallback, this);
        // ========================================================================================

        // --- 퍼블리셔 설정 ---
        steer_pub_ = nh_.advertise<std_msgs::Float32>("/auto_steer_angle_gps", 10);
        lookahead_gps_pub_ = nh_.advertise<visualization_msgs::Marker>("lookahead_gps_marker", 10);
        
        // 타이머 콜백: 10Hz (0.1초 간격)
        timer_ = nh_.createTimer(ros::Duration(0.1), &PurePursuit::timerCallback, this);
        
        ROS_INFO("PurePursuit (Dynamic Ld) 노드 시작 (wheelbase: %.2f m)", wheelbase_);
        ROS_INFO("Dynamic Ld Params: MinLd=%.2f, MaxLd=%.2f, ThrMin=%.2f, ThrMax=%.2f",
                 MIN_LOOKAHEAD_DISTANCE_, MAX_LOOKAHEAD_DISTANCE_, THROTTLE_MIN_, THROTTLE_MAX_);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber roi_sub_;
    ros::Subscriber throttle_sub_; // 스로틀 구독자 멤버 변수
    ros::Publisher steer_pub_;
    ros::Publisher lookahead_gps_pub_;
    ros::Timer timer_;
    
    // --- 멤버 변수 ---
    double wheelbase_;
    bool has_latest_roi_;
    visualization_msgs::Marker latest_roi_marker_;

    // ======================= [핵심 수정 3: 동적 Ld 관련 멤버 변수 추가] =======================
    double THROTTLE_MIN_;
    double THROTTLE_MAX_;
    double MIN_LOOKAHEAD_DISTANCE_;
    double MAX_LOOKAHEAD_DISTANCE_;
    double current_throttle_;
    // ======================================================================================

    void roiCallback(const visualization_msgs::Marker::ConstPtr& msg) {
        latest_roi_marker_ = *msg;
        has_latest_roi_ = true;
    }

    // ======================= [핵심 수정 4: 스로틀 콜백 함수 추가] =======================
    void throttleCallback(const std_msgs::Float32::ConstPtr& msg) {
        // 수신된 스로틀 값을 지정된 범위(min~max) 내로 제한하여 저장
        current_throttle_ = std::max(THROTTLE_MIN_, std::min(THROTTLE_MAX_, (double)msg->data));
    }
    // =================================================================================

    void timerCallback(const ros::TimerEvent&) {
        if (!has_latest_roi_ || latest_roi_marker_.points.empty()) {
            ROS_WARN_THROTTLE(5, "No ROI Marker.");
            return;
        }

        // ======================= [핵심 수정 5: 동적 전방주시거리 계산] =======================
        double throttle_range = THROTTLE_MAX_ - THROTTLE_MIN_;
        double normalized_throttle = 0.0;
        if (throttle_range > 0) {
            normalized_throttle = (current_throttle_ - THROTTLE_MIN_) / throttle_range;
        }
        double dynamic_lookahead_distance = MIN_LOOKAHEAD_DISTANCE_ + 
                                            (MAX_LOOKAHEAD_DISTANCE_ - MIN_LOOKAHEAD_DISTANCE_) * normalized_throttle;
        // =================================================================================

        std::vector<geometry_msgs::Point> valid_points;
        for (const auto& pt : latest_roi_marker_.points) {
            if (pt.x > 0) valid_points.push_back(pt);
        }
        if (valid_points.empty()) {
            ROS_WARN_THROTTLE(5, "No points in ROI.");
            return;
        }

        std::sort(valid_points.begin(), valid_points.end(),
                  [](const geometry_msgs::Point& a, const geometry_msgs::Point& b) {
                      return std::hypot(a.x, a.y) < std::hypot(b.x, b.y);
                  });
                  
        geometry_msgs::Point lookahead_pt = valid_points.back();
        for (const auto& pt : valid_points) {
            // 고정된 Ld 대신 동적으로 계산된 Ld 사용
            if (std::hypot(pt.x, pt.y) >= dynamic_lookahead_distance) {
                lookahead_pt = pt;
                break;
            }
        }
        double dist_to_lookahead = std::hypot(lookahead_pt.x, lookahead_pt.y);

        double alpha = std::atan2(lookahead_pt.y, lookahead_pt.x);
        double steer_angle = std::atan2(2 * wheelbase_ * std::sin(alpha), dist_to_lookahead);
        double steer_angle_deg = -steer_angle * 180.0 / M_PI;

        std_msgs::Float32 steer_msg;
        steer_msg.data = steer_angle_deg;
        steer_pub_.publish(steer_msg);
        
        // 로그 메시지에 동적 Ld 값 추가
        ROS_INFO_THROTTLE(1, "Ld: %.2f m, Lookahead: (%.2f, %.2f), dist: %.2f m, steer: %.2f deg",
                          dynamic_lookahead_distance, lookahead_pt.x, lookahead_pt.y, 
                          dist_to_lookahead, steer_angle_deg);

        visualization_msgs::Marker lookahead_marker;
        lookahead_marker.header.stamp = ros::Time::now();
        lookahead_marker.header.frame_id = "velodyne"; 
        lookahead_marker.ns = "only_gps_purepursuit";
        lookahead_marker.id = 0;
        lookahead_marker.type = visualization_msgs::Marker::CUBE;
        lookahead_marker.action = visualization_msgs::Marker::ADD;
        lookahead_marker.scale.x = 0.5;
        lookahead_marker.scale.y = 0.5;
        lookahead_marker.scale.z = 0.5;
        lookahead_marker.color.r = 1.0;
        lookahead_marker.color.g = 0.0;
        lookahead_marker.color.b = 1.0;
        lookahead_marker.color.a = 1.0;
        lookahead_marker.lifetime = ros::Duration(0.2);
        lookahead_marker.pose.position = lookahead_pt;
        lookahead_marker.pose.orientation.w = 1.0;
        lookahead_gps_pub_.publish(lookahead_marker);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "purepursuit_dynamic_ld");
    PurePursuit pp;
    ros::spin();
    return 0;
}