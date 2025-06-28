#ifndef SORT_ROS_H
#define SORT_ROS_H

#include "ros/ros.h"
#include "visualization_msgs/MarkerArray.h"
#include <dynamic_static_pkg/TrackedObjects.h>  // 추가: 메시지 타입 헤더

#include "Sort.h"
#ifndef NULL
#define NULL 0
#endif

class SortRos {

private:
    static SortRos* instance;

    SortRos(void) {};
    SortRos(const SortRos& other);
    ~SortRos() {};

public:
    static SortRos* GetInstance() {
        if(instance == NULL) instance = new SortRos();
        return instance;
    }

private:
    ros::NodeHandle nh;
    static ros::Publisher pub;
    static ros::Subscriber sub;
    static ros::Publisher pub_text;  // for /tracked_vx_vy 
    static void rectArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& markerArray);    // for /tracked_3D_Box
    static ros::Publisher pub_pred;  // for /predicted trajectory 
    static ros::Publisher pub_pred_endpoint;  // for /predicted trajectory endpoint
    static ros::Publisher  pub_tracks;         // MODIFIED: /tracked_objects
        
private:
    static Sort *s;

public:
    void setup(void);
};

#endif
