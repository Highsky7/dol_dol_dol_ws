#ifndef SORT_RECT_H
#define SORT_RECT_H


#include "TrackerState.h"


struct SortRect {

    int id;
    float centerX;
    float centerY;
    float width;
    float height;


    // 추가
    float vx;
    float vy;

    
    TrackerState toTrackerState(void);
    void fromTrackerState(TrackerState state);
};


#endif