#include "SortRect.h"
#include <cmath>

TrackerState SortRect::toTrackerState(void) {
    TrackerState state;
    state.centerX = centerX;
    state.centerY = centerY;
    state.area = width * height;
    state.aspectRatio = width / height;
    state.vx = vx;
    state.vy = vy;
    state.d = d;
    state.vd = vd;
    return state;
}

void SortRect::fromTrackerState(TrackerState state) {
    centerX = state.centerX;
    centerY = state.centerY;
    vx = state.vx;
    vy = state.vy;
    d = std::sqrt(centerX * centerX + centerY * centerY); // 거리 계산
    vd = state.vd;

    if (state.area > 0) {
        width = std::sqrt(state.area * state.aspectRatio);
        height = state.area / width;
    } else {
        width = 0;
        height = 0;
    }
}