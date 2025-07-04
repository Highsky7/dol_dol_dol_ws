#include "SortRos.h"
#include "PredictTrajectory.h"
#include <map>
#include <tuple>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <opencv2/core.hpp>
#include <visualization_msgs/Marker.h>
#include <dynamic_static_pkg/TrackedObjects.h>

SortRos* SortRos::instance = nullptr;
ros::Publisher SortRos::pub;
ros::Publisher SortRos::pub_text;
ros::Publisher SortRos::pub_pred;
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

    SortRos::sub = nh.subscribe<visualization_msgs::MarkerArray>("/detected_2D_Box", 1, SortRos::rectArrayCallback);
    SortRos::pub = nh.advertise<visualization_msgs::MarkerArray>("/tracked_3D_Box", 1);
    SortRos::pub_text = nh.advertise<visualization_msgs::MarkerArray>("/tracked_vx_vy", 1);
    SortRos::pub_pred = nh.advertise<visualization_msgs::MarkerArray>("/predicted_trajectory", 1);
    SortRos::pub_pred_endpoint = nh.advertise<visualization_msgs::MarkerArray>("/predicted_trajectory_endpoint", 1);
    SortRos::pub_tracks = nh.advertise<dynamic_static_pkg::TrackedObjects>("/tracked_objects", 1);
}

void SortRos::rectArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& markerArray) {
    std::string frame_id;
    std::vector<SortRect> rects;

    static std::map<int, std::tuple<float, float, float>> id2color;
    static bool seeded = false;
    if (!seeded) {
        std::srand(std::time(nullptr));
        seeded = true;
    }

    for (auto marker : markerArray->markers) {
        frame_id = marker.header.frame_id;
        SortRect rect;
        rect.id = marker.id;
        rect.centerX = marker.pose.position.x;
        rect.centerY = marker.pose.position.y;
        rect.width = marker.scale.x;
        rect.height = marker.scale.y;
        rect.d = std::sqrt(rect.centerX * rect.centerX + rect.centerY * rect.centerY); // 거리 계산
        rect.vd = 0.0f; // 초기 거리 속도
        rects.push_back(rect);
    }

    std::vector<SortRect> output = SortRos::s->update(rects);
    visualization_msgs::MarkerArray bboxArrayOutput;
    visualization_msgs::MarkerArray textArrayOutput;
    visualization_msgs::MarkerArray trajArrayOutput;
    visualization_msgs::MarkerArray endpointArrayOutput;

    dynamic_static_pkg::TrackedObjects tracks_msg;
    tracks_msg.header.stamp = ros::Time::now();
    tracks_msg.header.frame_id = frame_id;

    for (auto rect : output) {
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

        tracks_msg.id.push_back(rect.id);
        geometry_msgs::Point pt;
        pt.x = rect.centerX;
        pt.y = rect.centerY;
        pt.z = 0.0;
        tracks_msg.center.push_back(pt);

        TrackerState state = rect.toTrackerState();
        tracks_msg.vx.push_back(state.vx);
        tracks_msg.vy.push_back(state.vy);

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
        ss << "vx: " << state.vx << "\nvy: " << state.vy;
        textMarker.text = ss.str();
        textMarker.color.a = 1.0;
        textMarker.color.r = 1.0;
        textMarker.color.g = 1.0;
        textMarker.color.b = 1.0;
        textMarker.scale.z = 0.3;
        textMarker.lifetime = ros::Duration(0.2);
        textArrayOutput.markers.push_back(textMarker);

        cv::Mat transitionMatrix = (cv::Mat_<float>(8, 8) <<
            1, 0, 0, 0, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 1);
        TrajectoryPredictor::publishPredictedTrajectory(state, transitionMatrix, 10, rect.id, frame_id);
    }

    SortRos::pub_tracks.publish(tracks_msg);
    SortRos::pub.publish(bboxArrayOutput);
    SortRos::pub_text.publish(textArrayOutput);
    SortRos::pub_pred.publish(trajArrayOutput);
    SortRos::pub_pred_endpoint.publish(endpointArrayOutput);
}