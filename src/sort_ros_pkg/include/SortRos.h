#ifndef SORT_ROS_H
#define SORT_ROS_H

#include "ros/ros.h"
#include "visualization_msgs/MarkerArray.h"

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
        
private:
    static Sort *s;

public:
    void setup(void);
};

#endif
