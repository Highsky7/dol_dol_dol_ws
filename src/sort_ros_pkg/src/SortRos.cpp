#include "SortRos.h"
#include <std_msgs/Float32.h>   // speed 값 전용 메시지 타입
#include <std_msgs/Bool.h>      // dynamic_obstacle 메시지 타입

#ifndef NULL
#define NULL 0
#endif

SortRos* SortRos::instance = nullptr;

ros::Publisher SortRos::pub;
ros::Subscriber SortRos::sub;
ros::Publisher SortRos::speed_pub;          // speed 값을 발행할 publisher
ros::Publisher SortRos::dynamic_obstacle_pub; // dynamic_obstacle 메시지 publisher

Sort *SortRos::s;

void SortRos::setup(void) {
    double maxAge = 2;
    double minHits = 3;
    double iouThreshold = 0.3;

    ros::param::get("/sort_ros/max_age", maxAge);
    ros::param::get("/sort_ros/min_hits", minHits);
    ros::param::get("/sort_ros/iou_threshold", iouThreshold);

    SortRos::s = new Sort(maxAge, minHits, iouThreshold);

    // /markers_detected 토픽 구독, /markers_tracked 토픽에 결과 발행
    SortRos::sub = nh.subscribe<visualization_msgs::MarkerArray> ("/markers_detected", 1, SortRos::rectArrayCallback);
    SortRos::pub = nh.advertise<visualization_msgs::MarkerArray> ("/markers_tracked", 1);

    // 새 토픽: speed 값을 발행 (/speed)
    SortRos::speed_pub = nh.advertise<std_msgs::Float32>("/speed", 1);
    
    // 새 토픽: dynamic_obstacle (Bool 타입)
    SortRos::dynamic_obstacle_pub = nh.advertise<std_msgs::Bool>("dynamic_obstacle", 1);
}

void SortRos::rectArrayCallback (const visualization_msgs::MarkerArray::ConstPtr& markerArray) {
    std::string frame_id;
    std::vector<SortRect> rects;

    for(auto marker : markerArray->markers) {
        frame_id = marker.header.frame_id;

        SortRect rect;
        rect.id = 0;
        rect.centerX = marker.pose.position.x;
        rect.centerY = marker.pose.position.y;
        rect.width = marker.scale.x;
        rect.height = marker.scale.y;

        rects.push_back(rect);
    }

    std::vector<SortRect> output = SortRos::s->update(rects);
    visualization_msgs::MarkerArray markerArrayOutput;


    // 최대값을 추적하기 위한 변수 (속도는 음수가 아니므로 0으로 초기화)
    float max_speed = 0.0;

    // dynamic_obstacle 여부를 판단할 플래그
    bool dynamic_found = false;

    // 기존 동적/정적 분류 관련 코드 (이 부분은 그대로 유지)
    float speed_threshold = 0.020;

    for(auto rect : output) {
        // SortRect의 toTrackerState() 함수를 호출하여 TrackerState를 출력
        TrackerState state = rect.toTrackerState();

        // 차량 기준 전방 반원 영역 조건 추가: 
        // (centerX)^2 + (centerY)^2 <= rm^2 (여기서는 20.0, 즉 반지름 약 4.47m) 그리고 centerX > 0 (전방)
        if ((state.centerX * state.centerX + state.centerY * state.centerY) > 20.0f || state.centerX <= 0.0f) {  
            continue;  // 해당 영역에 포함되지 않으면 넘어감
        }

        // 계산된 speed (vx^2 + 10*vy^2)
        float speed = state.vx * state.vx + 10 * state.vy * state.vy;
        // 최대값 갱신
        if (speed > max_speed) {
            max_speed = speed;
        }
        // ROS_INFO("ID: %d, vx: %f, vy: %f, speed: %f", rect.id, state.vx, state.vy, speed);

        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = frame_id;
        marker.frame_locked = true;
        marker.lifetime = ros::Duration(0.5);
        marker.ns = "markers_tracked";
        marker.id = rect.id;
        marker.action = visualization_msgs::Marker::ADD;

        // marker.type = visualization_msgs::Marker::CUBE;
        marker.type = visualization_msgs::Marker::CYLINDER;

        marker.pose.position.x = rect.centerX;
        marker.pose.position.y = rect.centerY;
        marker.pose.position.z = 0.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.4;
        marker.scale.y = 0.4;
        marker.scale.z = 0.8;

        // 기존: 속도가 임계값 미만이면 파란색, 이상이면 pink
        if (speed < speed_threshold) {
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
        } else {
            marker.color.r = 1.0;
            marker.color.g = 0.0784;
            marker.color.b = 0.5765;                                 
            // dynamic (else) 분기: 동적 장애물이 존재함을 표시
            dynamic_found = true;
        }
        marker.color.a = 0.6;

        markerArrayOutput.markers.push_back(marker);
    }

    SortRos::pub.publish(markerArrayOutput);

    // 해당 영역 내에 속도 데이터가 있다면 최대 speed 값을 발행
    if(max_speed > 0.0) {
        std_msgs::Float32 speed_msg;
        speed_msg.data = max_speed;
        SortRos::speed_pub.publish(speed_msg);
    }
    
    // dynamic_found 값 출력 (ROS_INFO) 및 dynamic_obstacle 토픽 발행
    ROS_INFO("dynamic_obstacle: %s", dynamic_found ? "true" : "false");
    std_msgs::Bool dynamic_msg;
    dynamic_msg.data = dynamic_found;
    SortRos::dynamic_obstacle_pub.publish(dynamic_msg);
}