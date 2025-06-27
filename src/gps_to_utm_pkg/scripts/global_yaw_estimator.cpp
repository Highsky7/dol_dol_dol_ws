#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Float32.h>
#include <cmath>

double wrap_angle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

class GlobalYawEstimator {
public:
    GlobalYawEstimator() : prev_x_(0.0), prev_y_(0.0), has_prev_(false), filtered_yaw_(0.0), filtered_init_(false) {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        raw_yaw_pub_ = nh.advertise<std_msgs::Float32>("/raw_global_yaw", 10);
        filtered_yaw_pub_ = nh.advertise<std_msgs::Float32>("/global_yaw", 10);
        pnh.param("alpha", alpha_, 0.2);
        ROS_INFO("GlobalYawEstimator 노드 시작 (구독: /local_xy)");
        sub_ = nh.subscribe("/local_xy", 10, &GlobalYawEstimator::localXYCallback, this);
    }

    void localXYCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        double x = msg->point.x;
        double y = msg->point.y;
        if (has_prev_) {
            double dx = x - prev_x_;
            double dy = y - prev_y_;
            if (std::hypot(dx, dy) > 0.1) {
                double raw_yaw = std::atan2(dy, dx);
                raw_yaw = wrap_angle(raw_yaw);
                std_msgs::Float32 raw_msg;
                raw_msg.data = raw_yaw;
                raw_yaw_pub_.publish(raw_msg);
                if (!filtered_init_) {
                    filtered_yaw_ = raw_yaw;
                    filtered_init_ = true;
                } else {
                    double diff = wrap_angle(raw_yaw - filtered_yaw_);
                    filtered_yaw_ = wrap_angle(filtered_yaw_ + alpha_ * diff);
                }
                std_msgs::Float32 filt_msg;
                filt_msg.data = filtered_yaw_;
                filtered_yaw_pub_.publish(filt_msg);
                ROS_INFO("Raw Yaw: %.2f deg, Filtered Yaw: %.2f deg",
                         raw_yaw * 180.0 / M_PI, filtered_yaw_ * 180.0 / M_PI);
            }
        }
        prev_x_ = x;
        prev_y_ = y;
        has_prev_ = true;
    }

private:
    ros::Subscriber sub_;
    ros::Publisher raw_yaw_pub_, filtered_yaw_pub_;
    double prev_x_, prev_y_;
    bool has_prev_;
    double filtered_yaw_;
    bool filtered_init_;
    double alpha_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "global_yaw_estimator");
    GlobalYawEstimator gye;
    ros::spin();
    return 0;
}
