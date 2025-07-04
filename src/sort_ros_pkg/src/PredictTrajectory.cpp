#include "PredictTrajectory.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

// Define static members
ros::NodeHandle* TrajectoryPredictor::nh_pred = nullptr;
ros::Publisher TrajectoryPredictor::pub_pred_traj;
ros::Publisher TrajectoryPredictor::pub_pred_endpoint;
bool TrajectoryPredictor::initialized = false;

void TrajectoryPredictor::initPublishers() {
    if (!initialized) {
        // Create a NodeHandle for this predictor (will use the existing ROS node context)
        nh_pred = new ros::NodeHandle();
        // Advertise the predicted trajectory and endpoint topics
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
    // Ensure publishers are initialized
    if (!initialized) {
        initPublishers();
    }

    // Predict future states of the object
    std::vector<TrackerState> predictedStates = TrajectoryPredictor::predictTrajectory(currentState, transitionMatrix, steps);

    // Prepare LINE_STRIP marker for the predicted trajectory
    visualization_msgs::Marker trajMarker;
    trajMarker.header.stamp = ros::Time::now();
    trajMarker.header.frame_id = frame_id;
    trajMarker.ns = "predicted_trajectory";
    trajMarker.id = track_id;
    trajMarker.type = visualization_msgs::Marker::LINE_STRIP;
    trajMarker.action = visualization_msgs::Marker::ADD;
    trajMarker.lifetime = ros::Duration(0.2);
    trajMarker.frame_locked = true;
    // Line width
    trajMarker.scale.x = 0.05;
    // Color: pink
    trajMarker.color.a = 1.0;
    trajMarker.color.r = 1;
    trajMarker.color.g = 1;
    trajMarker.color.b = 1;
    // trajMarker.color.r = 240.0 / 255.0;
    // trajMarker.color.g = 15.0 / 255.0;
    // trajMarker.color.b = 135.0 / 255.0;
    // Identity pose (points are in the specified frame coordinates)
    trajMarker.pose.orientation.w = 1.0;
    trajMarker.pose.orientation.x = 0.0;
    trajMarker.pose.orientation.y = 0.0;
    trajMarker.pose.orientation.z = 0.0;
    trajMarker.pose.position.x = 0.0;
    trajMarker.pose.position.y = 0.0;
    trajMarker.pose.position.z = 0.0;
    // Start the trajectory at the current object position
    geometry_msgs::Point pt;
    pt.x = currentState.centerX;
    pt.y = currentState.centerY;
    pt.z = 0.0;
    trajMarker.points.push_back(pt);
    // Append predicted future positions to the trajectory
    for (const TrackerState& predState : predictedStates) {
        pt.x = predState.centerX;
        pt.y = predState.centerY;
        pt.z = 0.0;
        trajMarker.points.push_back(pt);
    }



    
    // Prepare SPHERE marker for the endpoint of the predicted trajectory
    visualization_msgs::Marker endpointMarker;
    endpointMarker.header.stamp = ros::Time::now();
    endpointMarker.header.frame_id = frame_id;
    endpointMarker.ns = "predicted_trajectory_endpoint";
    endpointMarker.id = track_id;
    endpointMarker.type = visualization_msgs::Marker::SPHERE;
    endpointMarker.action = visualization_msgs::Marker::ADD;
    endpointMarker.lifetime = ros::Duration(0.2);
    endpointMarker.frame_locked = true;
    // Position the sphere at the last predicted point (or current position if no prediction)
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
    // Sphere size
    endpointMarker.scale.x = 0.38;
    endpointMarker.scale.y = 0.38;
    endpointMarker.scale.z = 0.38;
    // Color: yellow
    endpointMarker.color.a = 1.0;
    endpointMarker.color.r = 1.0;
    endpointMarker.color.g = 1.0;
    endpointMarker.color.b = 0.0;

    // Publish the markers using MarkerArray messages (one for each topic)
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
    // Initialize the state vector (7x1) from current state: 
    // [0]: centerX, [1]: centerY, [2]: area, [3]: aspectRatio, [4]: vx, [5]: vy, [6]: area_change_rate
    cv::Mat state = cv::Mat::zeros(7, 1, CV_32F);
    state.at<float>(0, 0) = currentState.centerX;
    state.at<float>(1, 0) = currentState.centerY;
    state.at<float>(2, 0) = currentState.area;
    state.at<float>(3, 0) = currentState.aspectRatio;
    state.at<float>(4, 0) = currentState.vx;
    state.at<float>(5, 0) = currentState.vy;
    state.at<float>(6, 0) = 0.0f;  // assume no immediate area change unless provided

    // Apply the transition matrix repeatedly for 'steps' future frames
    for (int i = 0; i < steps; ++i) {
        state = transitionMatrix * state;
        // Convert the predicted state vector back to TrackerState structure
        TrackerState predicted;
        predicted.centerX     = state.at<float>(0, 0);
        predicted.centerY     = state.at<float>(1, 0);
        predicted.area        = state.at<float>(2, 0);
        predicted.aspectRatio = state.at<float>(3, 0);
        predicted.vx          = state.at<float>(4, 0);
        predicted.vy          = state.at<float>(5, 0);
        // (Note: area_change_rate at index 6 is not stored in TrackerState in this implementation)
        trajectory.push_back(predicted);
    }

    return trajectory;
}
