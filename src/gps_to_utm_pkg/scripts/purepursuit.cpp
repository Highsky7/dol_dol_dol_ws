#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Point.h>
#include <vector>
#include <algorithm>
#include <cmath>

class PurePursuit {
public:
    PurePursuit() : lookahead_distance_(3.5), wheelbase_(0.75), has_latest_roi_(false) {
        // 파라미터: lookahead distance와 차량 휠베이스 (단위: 미터)
        // ROI Marker를 구독하여 최신 데이터를 저장
        roi_sub_ = nh_.subscribe("roi_path_marker", 10, &PurePursuit::roiCallback, this);
        // 조향각을 publish할 토픽 (Float32, 단위: degree)
        steer_pub_ = nh_.advertise<std_msgs::Float32>("/auto_steer_angle_gps", 10);
        // Lookahead point Marker를 publish할 토픽
        lookahead_gps_pub_ = nh_.advertise<visualization_msgs::Marker>("lookahead_gps_marker", 10);
        // 타이머 콜백: 10Hz (0.1초 간격)
        timer_ = nh_.createTimer(ros::Duration(0.1), &PurePursuit::timerCallback, this);
        ROS_INFO("PurePursuit 노드 시작 (lookahead_distance: %.2f m, wheelbase: %.2f m)", 
                 lookahead_distance_, wheelbase_);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber roi_sub_;
    ros::Publisher steer_pub_;
    ros::Publisher lookahead_gps_pub_;
    ros::Timer timer_;
    double lookahead_distance_;
    double wheelbase_;
    bool has_latest_roi_;
    visualization_msgs::Marker latest_roi_marker_;

    void roiCallback(const visualization_msgs::Marker::ConstPtr& msg) {
        // 최신 ROI Marker 데이터 저장
        latest_roi_marker_ = *msg;
        has_latest_roi_ = true;
    }

    void timerCallback(const ros::TimerEvent&) {
        if (!has_latest_roi_ || latest_roi_marker_.points.empty()) {
            ROS_WARN_THROTTLE(5, "No ROI Marker.");
            return;
        }

        // 차량 좌표계에서, x > 0인 점들만 필터링 (전방)
        std::vector<geometry_msgs::Point> valid_points;
        for (const auto& pt : latest_roi_marker_.points) {
            if (pt.x > 0) valid_points.push_back(pt);
        }
        if (valid_points.empty()) {
            ROS_WARN_THROTTLE(5, "No points in ROI.");
            return;
        }

        // lookahead_distance 이상인 첫 번째 점을 선택 (거리 기준 오름차순 정렬)
        std::sort(valid_points.begin(), valid_points.end(),
                  [](const geometry_msgs::Point& a, const geometry_msgs::Point& b) {
                      return std::hypot(a.x, a.y) < std::hypot(b.x, b.y);
                  });
        geometry_msgs::Point lookahead_pt = valid_points.back();
        for (const auto& pt : valid_points) {
            if (std::hypot(pt.x, pt.y) >= lookahead_distance_) {
                lookahead_pt = pt;
                break;
            }
        }
        double dist = std::hypot(lookahead_pt.x, lookahead_pt.y);

        // 차량 좌표계에서는 전방이 x축이므로, alpha = arctan2(y, x)
        double alpha = std::atan2(lookahead_pt.y, lookahead_pt.x);
        // Pure pursuit 조향각 공식: δ = arctan( 2L sin(α) / d )
        double steer_angle = std::atan2(2 * wheelbase_ * std::sin(alpha), dist);

        // 라디안으로 계산된 조향각을 degree로 변환
        double steer_angle_deg = - steer_angle * 180.0 / M_PI;

        std_msgs::Float32 steer_msg;
        steer_msg.data = steer_angle_deg;
        steer_pub_.publish(steer_msg);
        ROS_INFO_THROTTLE(1, "Lookahead: (%.2f, %.2f), dist: %.2f m, α: %.2f deg, steer: %.2f deg",
                          lookahead_pt.x, lookahead_pt.y, dist,
                          alpha * 180.0 / M_PI, steer_angle_deg);

        // Lookahead point Marker (CUBE)
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
    ros::init(argc, argv, "purepursuit");
    PurePursuit pp;
    ros::spin();
    return 0;
}
