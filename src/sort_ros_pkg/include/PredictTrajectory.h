#ifndef PREDICT_TRAJECTORY_H
#define PREDICT_TRAJECTORY_H

#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include "TrackerState.h"

class TrajectoryPredictor {
public:
    // Initialize ROS publishers for predicted trajectory markers (line strip and sphere).
    static void initPublishers();

    // Publish predicted trajectory (LINE_STRIP) and endpoint (SPHERE) markers for the given object state.
    // `currentState` is the current TrackerState of the object,
    // `transitionMatrix` is the Kalman transition matrix,
    // `steps` is how many future steps to predict,
    // `track_id` is a unique identifier for the object (used in marker IDs),
    // `frame_id` is the coordinate frame in which to publish the markers.
    static void publishPredictedTrajectory(const TrackerState& currentState,
                                           const cv::Mat& transitionMatrix,
                                           int steps,
                                           int track_id,
                                           const std::string& frame_id);

    // Returns the predicted trajectory states based on the current TrackerState, 
    // transition matrix, and number of prediction steps.
    static std::vector<TrackerState> predictTrajectory(const TrackerState& currentState,
                                                       const cv::Mat& transitionMatrix,
                                                       int steps);

private:
    // Node handle and publishers for predicted trajectory and endpoint markers.
    static ros::NodeHandle* nh_pred;
    static ros::Publisher pub_pred_traj;
    static ros::Publisher pub_pred_endpoint;
    static bool initialized;
};

#endif  // PREDICT_TRAJECTORY_H
