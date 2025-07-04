#ifndef TRACKER_STATE_H
#define TRACKER_STATE_H

#define	MEASURE_NUM	4

#include "cv_bridge/cv_bridge.h"


struct TrackerState {

	float centerX;
	float centerY;
	float area;
	float aspectRatio;
	float vx; 
	float vy; 

	cv::Mat toMat(void);
	void fromMat(cv::Mat mat);

};


#endif