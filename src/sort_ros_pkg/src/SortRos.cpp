#include "SortRos.h"
#include "PredictTrajectory.h"  // 예측 모듈 header 포함
#include <map>
#include <tuple>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <opencv2/core.hpp>
#include "visualization_msgs/Marker.h"

// 정적 멤버 정의
SortRos* SortRos::instance = nullptr;
ros::Publisher SortRos::pub;
ros::Publisher SortRos::pub_text;
ros::Publisher SortRos::pub_pred;  // predicted trajectory publisher
ros::Subscriber SortRos::sub;
ros::Publisher SortRos::pub_pred_endpoint;
Sort *SortRos::s;

void SortRos::setup(void) {
    double maxAge = 30;
    double minHits = 2;
    double iouThreshold = 0.1;
    ros::param::get("/sort_ros/max_age", maxAge);
    ros::param::get("/sort_ros/min_hits", minHits);
    ros::param::get("/sort_ros/iou_threshold", iouThreshold);
    SortRos::s = new Sort(maxAge, minHits, iouThreshold);

    // 구독 및 퍼블리셔 초기화
    SortRos::sub = nh.subscribe<visualization_msgs::MarkerArray>("/detected_2D_Box", 1, SortRos::rectArrayCallback);
    SortRos::pub = nh.advertise<visualization_msgs::MarkerArray>("/tracked_3D_Box", 1);
    SortRos::pub_text = nh.advertise<visualization_msgs::MarkerArray>("/tracked_vx_vy", 1);
    SortRos::pub_pred = nh.advertise<visualization_msgs::MarkerArray>("/predicted_trajectory", 1);
    SortRos::pub_pred_endpoint = nh.advertise<visualization_msgs::MarkerArray>("/predicted_trajectory_endpoint", 1);
}

void SortRos::rectArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& markerArray) {
    std::string frame_id;
    std::vector<SortRect> rects;

    // 객체별 색상 매핑 (bounding box, 텍스트 등에 사용)
    static std::map<int, std::tuple<float, float, float>> id2color;
    static bool seeded = false;
    if (!seeded) {
        std::srand(std::time(nullptr));
        seeded = true;
    }

    // 2D detection 마커로부터 SortRect 채우기
    for (auto marker : markerArray->markers) {
        frame_id = marker.header.frame_id;
        SortRect rect;
        rect.id = marker.id;
        rect.centerX = marker.pose.position.x;
        rect.centerY = marker.pose.position.y;
        rect.width = marker.scale.x;
        rect.height = marker.scale.y;
        rects.push_back(rect);
    }

    // SORT update 수행 (기존 트랙 업데이트)
    std::vector<SortRect> output = SortRos::s->update(rects);
    visualization_msgs::MarkerArray bboxArrayOutput;
    visualization_msgs::MarkerArray textArrayOutput;
    visualization_msgs::MarkerArray trajArrayOutput;  // predicted trajectory markers
    visualization_msgs::MarkerArray endpointArrayOutput; // 추가: 예측 endpoint 마커 (sphere)

    for (auto rect : output) {
        // 1. Bounding Box Marker 생성 (Cube)
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = frame_id;
        marker.frame_locked = true;
        marker.lifetime = ros::Duration(0.2);
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
        bboxArrayOutput.markers.push_back(marker);

        // 2. 텍스트 마커 생성 (속도 정보)
        TrackerState state = rect.toTrackerState();
        float vx = state.vx;
        float vy = state.vy;
        visualization_msgs::Marker textMarker;
        textMarker.header.stamp = ros::Time::now();
        textMarker.header.frame_id = frame_id;
        textMarker.frame_locked = true;
        textMarker.ns = "tracked_vx_vy";
        textMarker.id = rect.id;
        textMarker.action = visualization_msgs::Marker::ADD;
        textMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        textMarker.pose.position.x = rect.centerX;
        textMarker.pose.position.y = rect.centerY;
        textMarker.pose.position.z = rect.height + 0.3;
        textMarker.pose.orientation.x = 0.0;
        textMarker.pose.orientation.y = 0.0;
        textMarker.pose.orientation.z = 0.0;
        textMarker.pose.orientation.w = 1.0;
        std::stringstream ss;
        ss << "vx: " << vx << "\nvy: " << vy;
        textMarker.text = ss.str();
        textMarker.color.a = 1.0;
        textMarker.color.r = 1.0;
        textMarker.color.g = 1.0;
        textMarker.color.b = 1.0;
        textMarker.scale.z = 0.3;
        textMarker.lifetime = ros::Duration(0.2);
        textArrayOutput.markers.push_back(textMarker);


        
        // // 3. 예측 경로(trajectory) Marker 생성
        // // 4. 예측 경로의 끝 점을 sphere 마커로 생성
        // --- 변경된 코드: 예측 마커 퍼블리싱은 PredictTrajectory 모듈에 위임 ---
        TrackerState currentState = rect.toTrackerState();
        int predictionSteps = 10;

        // Transition matrix is defined locally or obtained from tracker
        cv::Mat transitionMatrix = (cv::Mat_<float>(7, 7) <<
            1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1);

        // Instead of creating and publishing predicted markers here,
        // call the PredictTrajectory module function to handle it.
        TrajectoryPredictor::publishPredictedTrajectory(currentState, transitionMatrix, predictionSteps, rect.id, frame_id);

    }

    // 발행: bounding box, 텍스트, 예측 경로, 예측 endpoint
    pub.publish(bboxArrayOutput);
    pub_text.publish(textArrayOutput);
    pub_pred.publish(trajArrayOutput);
    pub_pred_endpoint.publish(endpointArrayOutput);
}