#include "SortRos.h"
#include "KalmanFilter.h"   // 추가: 칼만필터 헤더 포함
#include <map>         // 기존 및 객체 ID별 색상 매핑
#include <tuple>       // (r, g, b) 값 저장
#include <cstdlib>     // rand(), srand() 사용
#include <ctime>       // time() 사용
#include <sstream>     // 문자열 변환용 sstream

// 추가: trajectory marker를 위한 헤더
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

SortRos* SortRos::instance = nullptr;

ros::Publisher SortRos::pub;
ros::Subscriber SortRos::sub;

// 추가: 실제 이동궤적과 예측 이동궤적을 발행할 퍼블리셔
ros::Publisher SortRos::trajectoryPredictedPub;

ros::Publisher SortRos::trajectoryEndpointsPub;

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

    // 추가: 실제 궤적과 예측 궤적을 위한 퍼블리셔 생성 (토픽 이름: trajectory_actual, trajectory_predicted)
    SortRos::trajectoryPredictedPub = nh.advertise<visualization_msgs::MarkerArray>("trajectory_predicted", 1);

    SortRos::trajectoryEndpointsPub = nh.advertise<visualization_msgs::MarkerArray>("trajectory_endpoints", 1);

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

    // 추가: 각 객체의 실제 궤적과 칼만필터 관리를 위한 정적 변수들
    static std::map<int, std::vector<geometry_msgs::Point>> actualTrajMap;
    static std::map<int, std::vector<geometry_msgs::Point>> predictedTrajMap;
    static std::map<int, KalmanFilter> kfMap;
   
    // 현재 프레임에 검출된 객체들의 ID를 저장할 컨테이너 (active IDs)
    std::set<int> activeIDs;
    
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

    std::vector<SortRect> output = SortRos::s->update(rects);
    visualization_msgs::MarkerArray markerArrayOutput;

    // 각 객체에 대해 bounding box와 텍스트 마커 생성 + 궤적 업데이트
    for (auto rect : output) {
        activeIDs.insert(rect.id);  // 현재 추적 중인 객체 ID 추가
        // SortRect의 toTrackerState() 함수를 호출하여 TrackerState를 출력
        TrackerState state = rect.toTrackerState();
        // 속도 성분 추출 (vx, vy)
        float vx = state.vx;
        float vy = state.vy;

        // 바운딩 박스를 위한 Marker 생성
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
        markerArrayOutput.markers.push_back(marker);

        // 속도 텍스트를 위한 Marker 생성
        visualization_msgs::Marker textMarker;
        textMarker.header.stamp = ros::Time::now();
        textMarker.header.frame_id = frame_id;
        textMarker.frame_locked = true;
        textMarker.ns = "text";
        textMarker.id = rect.id; // 고유 ID 부여
        textMarker.action = visualization_msgs::Marker::ADD;
        textMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        textMarker.pose.position.x = rect.centerX;
        textMarker.pose.position.y = rect.centerY;
        textMarker.pose.position.z = rect.height + 0.3; // 바운딩 박스 위에 텍스트 배치
        textMarker.pose.orientation.x = 0.0;
        textMarker.pose.orientation.y = 0.0;
        textMarker.pose.orientation.z = 0.0;
        textMarker.pose.orientation.w = 1.0;
        // 속도 값을 두 줄 텍스트로 표시
        std::stringstream ss;
        ss << "vx: " << vx << "\nvy: " << vy;
        textMarker.text = ss.str();
        textMarker.color.a = 1.0;
        textMarker.color.r = 1.0;
        textMarker.color.g = 1.0;
        textMarker.color.b = 1.0;
        textMarker.scale.z = 0.3;
        textMarker.lifetime = ros::Duration(0.2);
        markerArrayOutput.markers.push_back(textMarker);



        // ----------------------------------------------------------
        // 칼만 필터 업데이트: sort가 계산한 vx, vy를 사용
        if (kfMap.find(rect.id) == kfMap.end()) {
            KalmanFilter kf;
            // 필요하면 여기서 kf.setPredictionSteps(원하는단계수)로 설정 가능
            kf.init(rect.centerX, rect.centerY, vx, vy);
            kfMap[rect.id] = kf;
        } else {
            kfMap[rect.id].update(rect.centerX, rect.centerY, vx, vy);
        }
        // 예측 궤적은 KalmanFilter의 predictTrajectory() 함수를 통해 내부 파라미터(predictionSteps_)를 사용하여 계산
        std::vector<geometry_msgs::Point> predictedPoints = kfMap[rect.id].predictTrajectory();
        predictedTrajMap[rect.id] = predictedPoints;
        // ----------------------------------------------------------
    
    }
    // bounding box 및 텍스트 마커 발행
    pub.publish(markerArrayOutput);


    // 활성 객체(activeIDs)에 없는(추적되지 않는) 객체의 예측 궤적 삭제
    std::vector<int> keysToRemove;
    for (auto const& pair : predictedTrajMap) {
        if (activeIDs.find(pair.first) == activeIDs.end()) {
            keysToRemove.push_back(pair.first);
        }
    }
    for (auto id : keysToRemove) {
        predictedTrajMap.erase(id);
        kfMap.erase(id);  // 여기를 추가하여 칼만 필터도 삭제
    }


    // 예측(빨간) 궤적 MarkerArray 생성 및 발행
    visualization_msgs::MarkerArray predictedTrajMarkers;
    for (auto const& pair : predictedTrajMap) {
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = frame_id;
        marker.ns = "trajectory_predicted";
        marker.id = pair.first;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.1;
        // 예측 궤적은 빨간색 선
        marker.color.a = 1.0;
        marker.color.r = 255.0/255.0;    // 1.0
        marker.color.g = 20.0/255.0;     // 약 0.078
        marker.color.b = 147.0/255.0;    // 약 0.576
        marker.lifetime = ros::Duration(0.1);
        marker.points = pair.second;
        predictedTrajMarkers.markers.push_back(marker);
    }
    // 궤적 토픽 발행 (실제 궤적은 발행하지 않음)
    trajectoryPredictedPub.publish(predictedTrajMarkers);





    
    // ----- [추가] 궤적 끝점 MarkerArray 생성 및 발행 -----
    visualization_msgs::MarkerArray endpointsMarkerArray;
    // Marker 타입을 SPHERE_LIST를 사용하여, 각 Marker에 여러 끝점을 담거나,
    // 각 객체마다 하나의 SPHERE 마커를 생성할 수도 있습니다.
    // 여기서는 각 객체마다 하나의 SPHERE 마커를 생성하는 방식입니다.
    int markerId = 0;
    for (auto const& pair : predictedTrajMap) {
        const std::vector<geometry_msgs::Point>& pts = pair.second;
        if (pts.empty()) continue;
        // 궤적의 마지막 점이 끝점입니다.
        geometry_msgs::Point endpoint = pts.back();
        visualization_msgs::Marker endpointMarker;
        endpointMarker.header.stamp = ros::Time::now();
        endpointMarker.header.frame_id = frame_id;
        endpointMarker.ns = "trajectory_endpoints";
        endpointMarker.id = markerId++;
        endpointMarker.type = visualization_msgs::Marker::SPHERE;
        endpointMarker.action = visualization_msgs::Marker::ADD;
        endpointMarker.pose.position = endpoint;
        endpointMarker.pose.orientation.w = 1.0;
        endpointMarker.scale.x = 0.3;  // 원하는 크기로 조정
        endpointMarker.scale.y = 0.3;
        endpointMarker.scale.z = 0.3;
        endpointMarker.color.a = 1.0;
        endpointMarker.color.r = 1.0;
        endpointMarker.color.g = 1.0;
        endpointMarker.color.b = 0.0;  // 예를 들어 노란색
        endpointMarker.lifetime = ros::Duration(0.1);  // 짧게 설정
        endpointsMarkerArray.markers.push_back(endpointMarker);
    }
    trajectoryEndpointsPub.publish(endpointsMarkerArray);
    // -----------------------------------------------------
    
    // ... 나머지 기존 코드 (활성 객체 처리, predictedTrajMap 정리 등) ...
}