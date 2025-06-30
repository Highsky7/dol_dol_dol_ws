// src/global_yaw_correction_node.cpp

#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <cmath>

class YawCorrector
{
public:
  YawCorrector()
  : first_msg_(true)
  {
    // Degree-corrected 퍼블리셔
    pub_deg_ = nh_.advertise<std_msgs::Float64>(
      "legoloam_yaw_corrected_deg", 10);
    // Radian-corrected 퍼블리셔
    pub_rad_ = nh_.advertise<std_msgs::Float64>(
      "legoloam_yaw_corrected_rad", 10);
    sub_ = nh_.subscribe<std_msgs::Float64>(
      "legoloam_yaw", 10,
      &YawCorrector::callback, this);
  }

private:
  void callback(const std_msgs::Float64::ConstPtr& msg)
  {
    double raw = msg->data;
    double raw_for_delta = raw;

    if (first_msg_) {
      // 첫 메시지는 보정 없이 그대로 사용
      corrected_ = raw;
      prev_raw_  = raw;
      first_msg_ = false;
    }
    else {
      // 1) raw가 -40°~+40° 사이면 보정 없이 그대로
      if (raw > -40.0 && raw < 40.0) {
        raw_for_delta = raw;
      }
      else {
        // 2) 그렇지 않으면 |raw + prev_raw_| ≤ 30° 면 부호 반전
        if (std::abs(raw + prev_raw_) <= 30.0) {
          raw_for_delta = -raw;
        }
        else {
          raw_for_delta = raw;
        }
      }

      // 3) Δ 계산 및 wrap‐around 보정 (±180°)
      double delta = raw_for_delta - prev_raw_;
      if (delta >  180.0) delta -= 360.0;
      if (delta < -180.0) delta += 360.0;

      // 4) 누적 언랩
      corrected_ += delta;

      // 5) prev_raw_ 갱신
      prev_raw_ = raw_for_delta;
    }

    // 6a) Degree 값 퍼블리시
    std_msgs::Float64 out_deg;
    out_deg.data = corrected_;
    pub_deg_.publish(out_deg);

    // 6b) Radian 값 변환 및 퍼블리시
    std_msgs::Float64 out_rad;
    out_rad.data = corrected_ * M_PI / 180.0;
    pub_rad_.publish(out_rad);
  }

  ros::NodeHandle nh_;
  ros::Publisher    pub_deg_;
  ros::Publisher    pub_rad_;
  ros::Subscriber   sub_;

  bool  first_msg_;
  double prev_raw_;
  double corrected_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "legoloam_yaw_corrected_deg");
  YawCorrector yc;
  ros::spin();
  return 0;
}
