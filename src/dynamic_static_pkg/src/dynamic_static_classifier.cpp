#include <ros/ros.h>
#include <dynamic_static_pkg/TrackedObjects.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cmath>

// 퍼블리셔 선언
ros::Publisher dynamic_pub;
ros::Publisher static_pub;
ros::Publisher bool_pub;

// 분류용 임계치 (기본값 ~)
float vy_threshold = 0.1f;

// 콜백 함수: 트래킹된 객체 메시지 처리
void trackedObjectsCallback(const dynamic_static_pkg::TrackedObjects::ConstPtr& msg) {
    visualization_msgs::MarkerArray dynamic_markers;
    visualization_msgs::MarkerArray static_markers;
    bool obstacle_in_roi = false;

    for (size_t i = 0; i < msg->id.size(); ++i) {
        const auto& center = msg->center[i];
        float vy = msg->vy[i];

        // 1) ROI 체크: 범위 밖이면 건너뜀
        if (!(center.x > 0.0 && center.x < 5.0 &&
              center.y > -2.0 && center.y < 2.0)) {
            continue;
        }

        // ROI 안에 들어온 객체만 여기서 동적/정적 분류
        visualization_msgs::Marker marker;
        marker.header   = msg->header;
        // 네임스페이스도 임계치 변수 사용
        bool is_dynamic = std::fabs(vy) >= vy_threshold;
        marker.ns       = is_dynamic ? "dynamic_objects" : "static_objects";
        marker.id       = msg->id[i];
        marker.type     = visualization_msgs::Marker::CUBE;
        marker.action   = visualization_msgs::Marker::ADD;
        marker.pose.position = center;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0;
        marker.color.a = 0.7;
        marker.lifetime = ros::Duration(0.5);

        if (is_dynamic) {
            // 동적 객체 → 빨간색
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0;
            dynamic_markers.markers.push_back(marker);
            obstacle_in_roi = true;
        } else {
            // 정적 객체 → 초록색
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;
            static_markers.markers.push_back(marker);
        }
    }

    // 퍼블리시
    dynamic_pub.publish(dynamic_markers);
    static_pub.publish(static_markers);

    std_msgs::Bool bool_msg;
    bool_msg.data = obstacle_in_roi;
    bool_pub.publish(bool_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "dynamic_static_classifier");
    ros::NodeHandle nh("~");  // private namespace for params

    // ROS 파라미터로도 Threshold 설정 가능 (없으면 기본 0.2)
    nh.param("vy_threshold", vy_threshold, vy_threshold);

    // 퍼블리셔 초기화
    dynamic_pub = nh.advertise<visualization_msgs::MarkerArray>("/dynamic_objects", 1);
    static_pub  = nh.advertise<visualization_msgs::MarkerArray>("/static_objects", 1);
    bool_pub    = nh.advertise<std_msgs::Bool>("/dynamic_obstacle", 1);

    // 구독자 설정
    ros::Subscriber sub = nh.subscribe("/tracked_objects", 1, trackedObjectsCallback);

    ros::spin();
    return 0;
}
