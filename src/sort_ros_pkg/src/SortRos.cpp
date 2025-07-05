#include "SortRos.h"
#include "PredictTrajectory.h"  // 예측 모듈 header 포함
#include <map>
#include <tuple>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <opencv2/core.hpp>
#include "visualization_msgs/Marker.h"
// #include <dynamic_static_pkg/TrackedObjects.h>  // MODIFIED: 올바른 include

// 정적 멤버 정의
SortRos* SortRos::instance = nullptr;
ros::Publisher SortRos::pub;
ros::Publisher SortRos::pub_text;
ros::Publisher SortRos::pub_pred;  // predicted trajectory publisher
ros::Publisher SortRos::pub_pred_endpoint;
ros::Subscriber SortRos::sub;
Sort *SortRos::s;
ros::Publisher SortRos::pub_tracks;
dynamic_static_pkg::TrackedObjects tracks_msg;

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
    SortRos::pub_text = nh.advertise<visualization_msgs::MarkerArray>("/relative_velodyne_vx_vy", 1);
    SortRos::pub_pred = nh.advertise<visualization_msgs::MarkerArray>("/predicted_trajectory", 1);
    SortRos::pub_pred_endpoint = nh.advertise<visualization_msgs::MarkerArray>("/predicted_trajectory_endpoint", 1);
    SortRos::pub_tracks = nh.advertise<dynamic_static_pkg::TrackedObjects>("/tracked_objects", 1);
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

    // <--- ① TrackedObjects 메시지 생성
    dynamic_static_pkg::TrackedObjects tracks_msg;
    tracks_msg.header.stamp = ros::Time::now();
    tracks_msg.header.frame_id = frame_id;

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

        // ② TrackedObjects 메시지에 정보 채우기
        tracks_msg.id.push_back(rect.id);
        geometry_msgs::Point pt;
        pt.x = rect.centerX;
        pt.y = rect.centerY;
        pt.z = 0.0;
        tracks_msg.center.push_back(pt);

        // 여기서 한 번만 선언
        TrackerState state = rect.toTrackerState();
        tracks_msg.vx.push_back(state.vx);
        tracks_msg.vy.push_back(state.vy);


        // 2. 텍스트 마커 생성 (속도 정보)
        float vx = state.vx;
        float vy = state.vy;
        visualization_msgs::Marker textMarker;
        textMarker.header.stamp = ros::Time::now();
        textMarker.header.frame_id = frame_id;
        textMarker.frame_locked = true;
        textMarker.ns = "sort_relative_velodyne_vx_vy";
        textMarker.id = rect.id;
        textMarker.action = visualization_msgs::Marker::ADD;
        textMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        textMarker.pose.position.x = rect.centerX + 0.35 ;
        textMarker.pose.position.y = rect.centerY + 1.2;
        textMarker.pose.position.z = rect.height + 1.5;
        textMarker.pose.orientation.x = 0.0;
        textMarker.pose.orientation.y = 0.0;
        textMarker.pose.orientation.z = 0.0;
        textMarker.pose.orientation.w = 1.0;
        std::stringstream ss;
        ss << "rel_vx " << std::fixed << std::setprecision(3) << vx << "\nrel_vy " << std::fixed << std::setprecision(3) << vy;
        textMarker.text = ss.str();
        textMarker.color.a = 1.0;
        textMarker.color.r = 1.0;
        textMarker.color.g = 1.0;
        textMarker.color.b = 1.0;
        textMarker.scale.z = 0.3;
        textMarker.lifetime = ros::Duration(0.5);
        textArrayOutput.markers.push_back(textMarker);



        // 3. 예측 경로 퍼블리싱 (PredictTrajectory 모듈 사용)
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

    // MODIFIED: Classifier가 구독할 메시지 퍼블리시
    SortRos::pub_tracks.publish(tracks_msg);

    // 발행: bounding box, 텍스트, 예측 경로, 예측 endpoint
    SortRos::pub.publish(bboxArrayOutput);
    SortRos::pub_text.publish(textArrayOutput);
    SortRos::pub_pred.publish(trajArrayOutput);
    SortRos::pub_pred_endpoint.publish(endpointArrayOutput);
}