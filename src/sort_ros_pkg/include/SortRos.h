#ifndef SORT_ROS_H
#define SORT_ROS_H

#include "ros/ros.h"
#include "visualization_msgs/MarkerArray.h"

#include "Sort.h"
// cpp은 NULL 정의가 안되어있다는 이유로 오류발생해서 설정
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
    static ros::Publisher speed_pub;
    static ros::Publisher dynamic_obstacle_pub; // 추가: dynamic_on 토픽 publisher
    

    static void rectArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& markerArray);



private:

    static Sort *s;

public:

    void setup(void);

};


#endif