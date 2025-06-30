#include "ros/ros.h"
#include "sensor_msgs/NavSatFix.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Float32.h"

class UbloxFixProcessor {
private:
    ros::NodeHandle nh_;
    ros::Subscriber fix_sub_;
    ros::Publisher status_pub_;
    ros::Publisher cov_pub_;

public:
    UbloxFixProcessor() {
        // 퍼블리셔 초기화
        status_pub_ = nh_.advertise<std_msgs::Int32>("/ublox_status", 10);
        cov_pub_ = nh_.advertise<std_msgs::Float32>("/ublox_position_covariance", 10);

        // 구독자 초기화
        fix_sub_ = nh_.subscribe("/ublox_gps/fix", 10, &UbloxFixProcessor::fixCallback, this);
    }

    void fixCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
        // status.status를 새로운 토픽으로 발행
        std_msgs::Int32 status_msg;
        status_msg.data = msg->status.status;
        status_pub_.publish(status_msg);

        // position_covariance[0]을 새로운 토픽으로 발행
        std_msgs::Float32 cov_msg;
        cov_msg.data = msg->position_covariance[0];
        cov_pub_.publish(cov_msg);

        ROS_INFO("Published: status=%d, covariance[0]=%f", status_msg.data, cov_msg.data);
    }

    void run() {
        ros::spin();
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "ublox_fix_processor");
    UbloxFixProcessor processor;
    processor.run();
    return 0;
}