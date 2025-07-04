#ifndef SORT_RECT_H
#define SORT_RECT_H

#include "TrackerState.h"

struct SortRect {
    int id;
    float centerX;
    float centerY;
    float width;
    float height;
    float vx;
    float vy;
    float d;  // 거리
    float vd; // 거리 속도

    TrackerState toTrackerState(void);
    void fromTrackerState(TrackerState state);
};

#endif