#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Float32.h>
#include <cmath>

class RefAntennaVelodyneTFBroadcaster {
public:
    RefAntennaVelodyneTFBroadcaster() : antenna_x_(0.0), antenna_y_(0.0), antenna_z_(0.0), global_yaw_(0.0) {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        pnh.getParam("velodyne_offset_x", offset_x_);
        pnh.getParam("velodyne_offset_y", offset_y_);
        pnh.getParam("velodyne_offset_z", offset_z_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>();
        sub_local_xy_ = nh.subscribe("local_xy", 10, &RefAntennaVelodyneTFBroadcaster::localXYCallback, this);
        sub_yaw_ = nh.subscribe("global_yaw", 10, &RefAntennaVelodyneTFBroadcaster::yawCallback, this);
        timer_ = nh.createTimer(ros::Duration(0.05), &RefAntennaVelodyneTFBroadcaster::timerCallback, this);
        ROS_INFO("ref_ant_vel_tf_broadcaster 노드 시작");
    }

    void localXYCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        antenna_x_ = msg->point.x;
        antenna_y_ = msg->point.y;
        antenna_z_ = msg->point.z;
    }

    void yawCallback(const std_msgs::Float32::ConstPtr& msg) {
        global_yaw_ = msg->data;
    }

    void timerCallback(const ros::TimerEvent&) {
        ros::Time now = ros::Time::now();

        // 1) reference -> antenna
        geometry_msgs::TransformStamped antenna_tf;
        antenna_tf.header.stamp = now;
        antenna_tf.header.frame_id = "reference";
        antenna_tf.child_frame_id = "antenna";
        antenna_tf.transform.translation.x = antenna_x_;
        antenna_tf.transform.translation.y = antenna_y_;
        antenna_tf.transform.translation.z = antenna_z_;
        tf2::Quaternion q;
        q.setRPY(0, 0, global_yaw_);
        antenna_tf.transform.rotation.x = q.x();
        antenna_tf.transform.rotation.y = q.y();
        antenna_tf.transform.rotation.z = q.z();
        antenna_tf.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(antenna_tf);

        // 2) antenna -> velodyne (고정 오프셋)
        geometry_msgs::TransformStamped velodyne_tf;
        velodyne_tf.header.stamp = now;
        velodyne_tf.header.frame_id = "antenna";
        velodyne_tf.child_frame_id = "velodyne";
        velodyne_tf.transform.translation.x = offset_x_;
        velodyne_tf.transform.translation.y = offset_y_;
        velodyne_tf.transform.translation.z = offset_z_;
        velodyne_tf.transform.rotation.x = 0.0;
        velodyne_tf.transform.rotation.y = 0.0;
        velodyne_tf.transform.rotation.z = 0.0;
        velodyne_tf.transform.rotation.w = 1.0;
        tf_broadcaster_->sendTransform(velodyne_tf);

        // ROS_INFO_THROTTLE(0.1,
        //     "[Ref->Ant]->[Ant->Vel] TF: antenna=(%.2f, %.2f), yaw=%.2f deg; offset=(%.2f, %.2f)",
        //     antenna_x_, antenna_y_, global_yaw_ * 180.0 / M_PI,
        //     offset_x_, offset_y_);
    }

private:
    ros::Subscriber sub_local_xy_, sub_yaw_;
    ros::Timer timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    double antenna_x_, antenna_y_, antenna_z_;
    double global_yaw_;
    double offset_x_, offset_y_, offset_z_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ref_ant_vel_tf_broadcaster");
    RefAntennaVelodyneTFBroadcaster node;
    ros::spin();
    return 0;
}
