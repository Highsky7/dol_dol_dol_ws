#include "SortRos.h"
#include <map>         // 추가: 각 객체 ID별 색상 매핑을 위한 map
#include <tuple>       // 추가: tuple을 사용하여 (r, g, b) 값을 저장
#include <cstdlib>     // 추가: rand(), srand() 사용
#include <ctime>       // 추가: time() 사용
#include <sstream>     // 추가: 문자열 변환을 위한 sstream



SortRos* SortRos::instance = nullptr;

ros::Publisher SortRos::pub;
ros::Subscriber SortRos::sub;

Sort *SortRos::s;

void SortRos::setup(void) {
    double maxAge = 30; //up   객체가 30프레임 동안 사라졌더라도 같은 객체로 간주하여 추적을 계속함
    double minHits = 2;  // down    객체가 최소 2프레임 동안 추적되어야만 유효한 객체로 간주됨
    double iouThreshold = 0.1;  //down    두 객체가 10% 이상 겹치면 같은 객체로 간주됨
    ros::param::get("/sort_ros/max_age", maxAge);
    ros::param::get("/sort_ros/min_hits", minHits);
    ros::param::get("/sort_ros/iou_threshold", iouThreshold);
    SortRos::s = new Sort(maxAge, minHits, iouThreshold);


    // /markers_detected 토픽을 구독, /markers_tracked 토픽에 결과 발행
    SortRos::sub = nh.subscribe<visualization_msgs::MarkerArray>("/markers_detected", 1, SortRos::rectArrayCallback);
    SortRos::pub = nh.advertise<visualization_msgs::MarkerArray>("/markers_tracked", 1);
}

void SortRos::rectArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& markerArray) {
    std::string frame_id;
    std::vector<SortRect> rects;

    static std::map<int, std::tuple<float, float, float>> id2color; // 객체 id -> (r,g,b)
    static bool seeded = false;
    if (!seeded) {
        std::srand(std::time(nullptr)); // std::time 사용
        seeded = true;
    }

    for (auto marker : markerArray->markers) {
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

    for (auto rect : output) {
        
        // SortRect의 toTrackerState() 함수를 호출하여 TrackerState를 출력
        TrackerState state = rect.toTrackerState();
        // 속도 성분을 추출하여 그대로 사용 (vx, vy)
        float vx = state.vx;
        float vy = state.vy;

        // 바운딩 박스를 위한 Marker 생성
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = frame_id;
        marker.frame_locked = true;
        marker.lifetime = ros::Duration(0.5);
        marker.ns = "bounding_box";
        marker.id = rect.id;
        marker.action = visualization_msgs::Marker::ADD;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.pose.position.x = rect.centerX;
        marker.pose.position.y = rect.centerY;
        marker.pose.position.z = 0.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = rect.width;
        marker.scale.y = rect.height;
        marker.scale.z = 1.0;
        // 랜덤 색상 할당
        float r, g, b;
        if (id2color.find(rect.id) == id2color.end()) {
            r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            g = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            b = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            id2color[rect.id] = std::make_tuple(r, g, b);
        } else {
            std::tie(r, g, b) = id2color[rect.id];
        }
        marker.color.a = 0.7;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        markerArrayOutput.markers.push_back(marker);

        // 속도 텍스트를 위한 Marker 생성
        visualization_msgs::Marker textMarker;
        textMarker.header.stamp = ros::Time::now();
        textMarker.header.frame_id = frame_id;
        textMarker.frame_locked = true;
        textMarker.ns = "text";
        textMarker.id = rect.id + 1000; // 고유 ID 부여
        textMarker.action = visualization_msgs::Marker::ADD;
        textMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        textMarker.pose.position.x = rect.centerX;
        textMarker.pose.position.y = rect.centerY;
        textMarker.pose.position.z = rect.height + 0.3; // 바운딩 박스 위에 텍스트 배치
        textMarker.pose.orientation.x = 0.0;
        textMarker.pose.orientation.y = 0.0;
        textMarker.pose.orientation.z = 0.0;
        textMarker.pose.orientation.w = 1.0;
        // 속도 값을 텍스트로 표시 (vx와 vy를 두 줄로 출력)
        std::stringstream ss;
        ss << "vx: " << vx << "\nvy: " << vy;  // 줄바꿈을 추가하여 두 줄로 표시
        textMarker.text = ss.str();
        // 텍스트 마커 색상 설정 (흰색)
        textMarker.color.a = 1.0;
        textMarker.color.r = 1.0;
        textMarker.color.g = 1.0;
        textMarker.color.b = 1.0;
        // 텍스트 크기를 키움 (기본값보다 더 크게)
        textMarker.scale.z = 0.3; // 글자 크기 조정 (더 크게 설정)
        // 텍스트 마커의 lifetime을 바운딩 박스와 동일하게 설정
        textMarker.lifetime = ros::Duration(0.5); // lifetime 설정
        // 텍스트 마커 추가
        markerArrayOutput.markers.push_back(textMarker);
    }

    // 마커 배열을 발행
    pub.publish(markerArrayOutput);
}