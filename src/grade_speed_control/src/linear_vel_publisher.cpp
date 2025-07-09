#include <ros/ros.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <std_msgs/Float64.h>
#include <cmath>

class LinearVelocityPublisher
{
public:
  LinearVelocityPublisher()
  {
    ros::NodeHandle pnh("~");

    // 파라미터(토픽명) 설정: 노드 런치에서 변경 가능
    std::string input_topic, output_topic;
    pnh.param("input_topic",  input_topic,  std::string("/ublox_gps/fix_velocity"));
    pnh.param("output_topic", output_topic, std::string("/linear_velocity"));

    pub_ = nh_.advertise<std_msgs::Float64>(output_topic, 10);
    sub_ = nh_.subscribe(input_topic, 10,
                         &LinearVelocityPublisher::callback, this);
  }

private:
  void callback(const geometry_msgs::TwistWithCovarianceStamped::ConstPtr& msg)
  {
    const double vx = msg->twist.twist.linear.x;  // m/s
    const double vy = msg->twist.twist.linear.y;  // m/s

    std_msgs::Float64 vel_msg;
    vel_msg.data = std::hypot(vx, vy);            // √(vx² + vy²)
    pub_.publish(vel_msg);
  }

  ros::NodeHandle nh_;
  ros::Publisher  pub_;
  ros::Subscriber sub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "linear_vel_publisher");
  LinearVelocityPublisher node;
  ros::spin();
  return 0;
}