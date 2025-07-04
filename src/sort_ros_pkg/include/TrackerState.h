#ifndef TRACKER_STATE_H
#define TRACKER_STATE_H

#include <opencv2/opencv.hpp>

#define MEASURE_NUM 4 // centerX, centerY, area, aspectRatio
#define STATE_NUM 8   // centerX, centerY, area, aspectRatio, vx, vy, d, vd

struct TrackerState {
    float centerX;
    float centerY;
    float area;
    float aspectRatio;
    float vx;
    float vy;
    float d;  // 카메라로부터의 거리
    float vd; // 거리 방향 속도

    cv::Mat toMat(void);
    void fromMat(cv::Mat mat);
};

#endif