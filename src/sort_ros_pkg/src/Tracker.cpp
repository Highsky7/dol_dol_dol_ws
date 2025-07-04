#include "Tracker.h"
#include <cmath>

int Tracker::kf_count = 0;

Tracker::Tracker(TrackerState state) {
    kf_count++;

    m_time_since_update = 0;
    m_hits = 0;
    m_hit_streak = 0;
    m_id = kf_count;

    kf = cv::KalmanFilter(STATE_NUM, MEASURE_NUM, 0);

    kf.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, 1, 0, 0, 0, // centerX <- centerX + vx
        0, 1, 0, 0, 0, 1, 0, 0, // centerY <- centerY + vy
        0, 0, 1, 0, 0, 0, 0, 1, // area <- area + area_change_rate
        0, 0, 0, 1, 0, 0, 0, 0, // aspectRatio (maintain)
        0, 0, 0, 0, 1, 0, 0, 0, // vx (maintain)
        0, 0, 0, 0, 0, 1, 0, 0, // vy (maintain)
        0, 0, 0, 0, 0, 0, 1, 1, // d <- d + vd
        0, 0, 0, 0, 0, 0, 0, 1  // vd (maintain)
    );

    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    kf.statePost.at<float>(0, 0) = state.centerX;
    kf.statePost.at<float>(1, 0) = state.centerY;
    kf.statePost.at<float>(2, 0) = state.area;
    kf.statePost.at<float>(3, 0) = state.aspectRatio;
    kf.statePost.at<float>(4, 0) = state.vx;
    kf.statePost.at<float>(5, 0) = state.vy;
    kf.statePost.at<float>(6, 0) = state.d;
    kf.statePost.at<float>(7, 0) = state.vd;

    prev_vx = state.vx;
    prev_vy = state.vy;
}

TrackerState Tracker::predict(void) {
    if (m_time_since_update > 0)
        m_hit_streak = 0;
    m_time_since_update += 1;

    TrackerState state;
    state.fromMat(kf.predict());

    if (state.area < 0) {
        state.area = 0;
    }

    // 회전 여부 판단: vx, vy의 방향 변화량 계산
    float dot_product = prev_vx * state.vx + prev_vy * state.vy;
    float mag_prev = std::sqrt(prev_vx * prev_vx + prev_vy * prev_vy);
    float mag_curr = std::sqrt(state.vx * state.vx + state.vy * state.vy);
    float cos_theta = (mag_prev > 0 && mag_curr > 0) ? dot_product / (mag_prev * mag_curr) : 1.0f;
    float angular_change = std::acos(std::max(-1.0f, std::min(1.0f, cos_theta)));

    // 정규화 비율 조정
    float normalization_factor;
    if (angular_change > 0.1f && state.d > 0) { // 회전 운동
        normalization_factor = 1.0f / state.d; // 강한 정규화
    } else { // 직선 운동
        normalization_factor = 1.0f / std::sqrt(state.d + 1.0f); // 약한 정규화
    }

    // 속도 정규화
    if (state.d > 0) {
        state.vx *= normalization_factor;
        state.vy *= normalization_factor;
    }

    prev_vx = state.vx;
    prev_vy = state.vy;

    return state;
}

void Tracker::update(TrackerState state) {
    m_time_since_update = 0;
    m_hits += 1;
    m_hit_streak += 1;

    cv::Mat measurement = cv::Mat::zeros(MEASURE_NUM, 1, CV_32F);
    measurement.at<float>(0, 0) = state.centerX;
    measurement.at<float>(1, 0) = state.centerY;
    measurement.at<float>(2, 0) = state.area;
    measurement.at<float>(3, 0) = state.aspectRatio;

    kf.correct(measurement);
}

TrackerState Tracker::getState(void) {
    TrackerState state;
    state.fromMat(kf.statePost);
    return state;
}