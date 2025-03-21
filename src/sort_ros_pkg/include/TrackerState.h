#ifndef TRACKER_STATE_H
#define TRACKER_STATE_H

#define	MEASURE_NUM	4

#include "cv_bridge/cv_bridge.h"


struct TrackerState {

	float centerX;
	float centerY;
	float area;
	float aspectRatio;

	//추가
	float vx;  // x축 속도
	float vy;  // y축 속도

	cv::Mat toMat(void);
	void fromMat(cv::Mat mat);

};


#endif