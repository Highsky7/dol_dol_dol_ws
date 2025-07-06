#include <ros/ros.h>
#include <dynamic_static_pkg/TrackedObjects.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <cmath>

// 퍼블리셔 선언
ros::Publisher dynamic_pub;
ros::Publisher static_pub;
ros::Publisher bool_pub;

// 분류용 임계치
float vy_threshold = 0.06f;
float steering_angle_threshold = 10.0f;  // 스티어링 각 임계값 (기본값 10도)

// ROI 파라미터 (기본값)
float roi_x_min = 0.0f;
float roi_x_max = 5.0f;
float roi_y_min = -2.0f;
float roi_y_max = 2.0f;

// 전역 변수
float steering_angle = 0.0f;  // 현재 스티어링 각도

// 콜백 함수: 스티어링 각도 업데이트 및 ROS_INFO 출력
void steeringAngleCallback(const std_msgs::Float32::ConstPtr& msg) {
    steering_angle = msg->data;
    ROS_INFO("Received /steering_angle: %.2f", steering_angle);
}

// 콜백 함수: 트래킹된 객체 메시지 처리
void trackedObjectsCallback(const dynamic_static_pkg::TrackedObjects::ConstPtr& msg) {
    // 스티어링 각도의 절댓값이 임계값 이상이면 분류를 진행하지 않음
    if (std::fabs(steering_angle) >= steering_angle_threshold) {
        return;
    }

    visualization_msgs::MarkerArray dynamic_markers;
    visualization_msgs::MarkerArray static_markers;
    bool obstacle_in_roi = false;

    for (size_t i = 0; i < msg->id.size(); ++i) {
        const auto& center = msg->center[i];
        float vy = msg->vy[i];

        // ROI 체크: 동적으로 설정된 ROI 범위 사용
        if (!(center.x > roi_x_min && center.x < roi_x_max &&
              center.y > roi_y_min && center.y < roi_y_max)) {
            continue;
        }

        // ROI 안에 들어온 객체만 동적/정적 분류
        visualization_msgs::Marker marker;
        marker.header = msg->header;
        bool is_dynamic = std::fabs(vy) >= vy_threshold;
        marker.ns = is_dynamic ? "dynamic_objects" : "static_objects";
        marker.id = msg->id[i];
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position = center;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = marker.scale.y = 1.0;
        marker.scale.z = 0.0;
        if (is_dynamic) {
            // 동적 객체: 빨간색
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 1.0; marker.lifetime = ros::Duration(0.7);
            dynamic_markers.markers.push_back(marker);
            obstacle_in_roi = true;
        } else {
            // 정적 객체: 초록색
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.7; marker.lifetime = ros::Duration(0.3);
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

    // ROS 파라미터로 임계값 및 ROI 설정
    nh.param("vy_threshold", vy_threshold, vy_threshold);
    nh.param("steering_angle_threshold", steering_angle_threshold, steering_angle_threshold);
    nh.param("roi_x_min", roi_x_min, roi_x_min);
    nh.param("roi_x_max", roi_x_max, roi_x_max);
    nh.param("roi_y_min", roi_y_min, roi_y_min);
    nh.param("roi_y_max", roi_y_max, roi_y_max);

    // 퍼블리셔 초기화
    dynamic_pub = nh.advertise<visualization_msgs::MarkerArray>("/dynamic_objects", 1);
    static_pub = nh.advertise<visualization_msgs::MarkerArray>("/static_objects", 1);
    bool_pub = nh.advertise<std_msgs::Bool>("/dynamic_obstacle", 1);

    // 구독자 설정
    ros::Subscriber sub = nh.subscribe("/tracked_objects", 1, trackedObjectsCallback);
    ros::Subscriber steering_sub = nh.subscribe("/steering_angle", 1, steeringAngleCallback);

    ros::spin();
    return 0;
}