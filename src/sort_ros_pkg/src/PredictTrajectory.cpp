#include "PredictTrajectory.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <cmath>

ros::NodeHandle* TrajectoryPredictor::nh_pred = nullptr;
ros::Publisher TrajectoryPredictor::pub_pred_traj;
ros::Publisher TrajectoryPredictor::pub_pred_endpoint;
bool TrajectoryPredictor::initialized = false;

void TrajectoryPredictor::initPublishers() {
    if (!initialized) {
        nh_pred = new ros::NodeHandle();
        pub_pred_traj = nh_pred->advertise<visualization_msgs::MarkerArray>("/predicted_trajectory", 1);
        pub_pred_endpoint = nh_pred->advertise<visualization_msgs::MarkerArray>("/predicted_trajectory_endpoint", 1);
        initialized = true;
    }
}

void TrajectoryPredictor::publishPredictedTrajectory(const TrackerState& currentState,
                                                     const cv::Mat& transitionMatrix,
                                                     int steps,
                                                     int track_id,
                                                     const std::string& frame_id) {
    if (!initialized) {
        initPublishers();
    }

    std::vector<TrackerState> predictedStates = predictTrajectory(currentState, transitionMatrix, steps);

    visualization_msgs::Marker trajMarker;
    trajMarker.header.stamp = ros::Time::now();
    trajMarker.header.frame_id = frame_id;
    trajMarker.ns = "predicted_trajectory";
    trajMarker.id = track_id;
    trajMarker.type = visualization_msgs::Marker::LINE_STRIP;
    trajMarker.action = visualization_msgs::Marker::ADD;
    trajMarker.lifetime = ros::Duration(0.2);
    trajMarker.frame_locked = true;
    trajMarker.scale.x = 0.05;
    trajMarker.color.a = 1.0;
    trajMarker.color.r = 1.0;
    trajMarker.color.g = 1.0;
    trajMarker.color.b = 1.0;
    trajMarker.pose.orientation.w = 1.0;
    trajMarker.pose.orientation.x = 0.0;
    trajMarker.pose.orientation.y = 0.0;
    trajMarker.pose.orientation.z = 0.0;
    trajMarker.pose.position.x = 0.0;
    trajMarker.pose.position.y = 0.0;
    trajMarker.pose.position.z = 0.0;

    geometry_msgs::Point pt;
    pt.x = currentState.centerX;
    pt.y = currentState.centerY;
    pt.z = 0.0;
    trajMarker.points.push_back(pt);

    for (const TrackerState& predState : predictedStates) {
        pt.x = predState.centerX;
        pt.y = predState.centerY;
        pt.z = 0.0;
        trajMarker.points.push_back(pt);
    }

    visualization_msgs::Marker endpointMarker;
    endpointMarker.header.stamp = ros::Time::now();
    endpointMarker.header.frame_id = frame_id;
    endpointMarker.ns = "predicted_trajectory_endpoint";
    endpointMarker.id = track_id;
    endpointMarker.type = visualization_msgs::Marker::SPHERE;
    endpointMarker.action = visualization_msgs::Marker::ADD;
    endpointMarker.lifetime = ros::Duration(0.2);
    endpointMarker.frame_locked = true;
    if (!predictedStates.empty()) {
        const TrackerState& finalState = predictedStates.back();
        endpointMarker.pose.position.x = finalState.centerX;
        endpointMarker.pose.position.y = finalState.centerY;
    } else {
        endpointMarker.pose.position.x = currentState.centerX;
        endpointMarker.pose.position.y = currentState.centerY;
    }
    endpointMarker.pose.position.z = 0.0;
    endpointMarker.pose.orientation.x = 0.0;
    endpointMarker.pose.orientation.y = 0.0;
    endpointMarker.pose.orientation.z = 0.0;
    endpointMarker.pose.orientation.w = 1.0;
    endpointMarker.scale.x = 0.38;
    endpointMarker.scale.y = 0.38;
    endpointMarker.scale.z = 0.38;
    endpointMarker.color.a = 1.0;
    endpointMarker.color.r = 1.0;
    endpointMarker.color.g = 1.0;
    endpointMarker.color.b = 0.0;

    visualization_msgs::MarkerArray trajArray;
    trajArray.markers.push_back(trajMarker);
    visualization_msgs::MarkerArray endpointArray;
    endpointArray.markers.push_back(endpointMarker);
    pub_pred_traj.publish(trajArray);
    pub_pred_endpoint.publish(endpointArray);
}

std::vector<TrackerState> TrajectoryPredictor::predictTrajectory(const TrackerState& currentState,
                                                                 const cv::Mat& transitionMatrix,
                                                                 int steps) {
    std::vector<TrackerState> trajectory;
    cv::Mat state = cv::Mat::zeros(8, 1, CV_32F);
    state.at<float>(0, 0) = currentState.centerX;
    state.at<float>(1, 0) = currentState.centerY;
    state.at<float>(2, 0) = currentState.area;
    state.at<float>(3, 0) = currentState.aspectRatio;
    state.at<float>(4, 0) = currentState.vx;
    state.at<float>(5, 0) = currentState.vy;
    state.at<float>(6, 0) = currentState.d;
    state.at<float>(7, 0) = currentState.vd;

    float prev_vx = currentState.vx;
    float prev_vy = currentState.vy;

    for (int i = 0; i < steps; ++i) {
        state = transitionMatrix * state;
        TrackerState predicted;
        predicted.centerX     = state.at<float>(0, 0);
        predicted.centerY     = state.at<float>(1, 0);
        predicted.area        = state.at<float>(2, 0);
        predicted.aspectRatio = state.at<float>(3, 0);
        predicted.vx          = state.at<float>(4, 0);
        predicted.vy          = state.at<float>(5, 0);
        predicted.d           = state.at<float>(6, 0);
        predicted.vd          = state.at<float>(7, 0);

        // 회전 여부 판단
        float dot_product = prev_vx * predicted.vx + prev_vy * predicted.vy;
        float mag_prev = std::sqrt(prev_vx * prev_vx + prev_vy * prev_vy);
        float mag_curr = std::sqrt(predicted.vx * predicted.vx + predicted.vy * predicted.vy);
        float cos_theta = (mag_prev > 0 && mag_curr > 0) ? dot_product / (mag_prev * mag_curr) : 1.0f;
        float angular_change = std::acos(std::max(-1.0f, std::min(1.0f, cos_theta)));

        // 정규화 비율 조정
        float normalization_factor;
        if (angular_change > 0.1f && predicted.d > 0) { // 회전 운동
            normalization_factor = 1.0f / predicted.d; // 강한 정규화
        } else { // 직선 운동
            normalization_factor = 1.0f / std::sqrt(predicted.d + 1.0f); // 약한 정규화
        }

        if (predicted.d > 0) {
            predicted.vx *= normalization_factor;
            predicted.vy *= normalization_factor;
        }

        prev_vx = predicted.vx;
        prev_vy = predicted.vy;

        trajectory.push_back(predicted);
    }

    return trajectory;
}