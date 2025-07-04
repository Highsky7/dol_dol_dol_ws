#ifndef TRACKER_H
#define TRACKER_H

#include "cv_bridge/cv_bridge.h"
#include "opencv2/video/tracking.hpp"
#include "TrackerState.h"

#define STATE_NUM 8

class Tracker {
public:
    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_id;

    Tracker(TrackerState state);
    TrackerState predict(void);
    void update(TrackerState state);
    TrackerState getState();

private:
    static int kf_count;
    cv::KalmanFilter kf;
    float prev_vx; // 회전 판단용 이전 속도
    float prev_vy;
};

#endif