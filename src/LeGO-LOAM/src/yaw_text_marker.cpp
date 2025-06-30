#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_listener.h>   // 굳이 TF를 쓰진 않지만 헤더 구분용

class YawTextMarker {
public:
  YawTextMarker() {
    ros::NodeHandle nh, pnh("~");

    // 파라미터: 표기 소수점 자리, 색깔
    pnh.param("precision", precision_, 2);
    pnh.param("color_r", color_r_, 1.0);
    pnh.param("color_g", color_g_, 1.0);
    pnh.param("color_b", color_b_, 0.0);

    // 구독자 추가
    sub_legoloam_ = nh.subscribe("/legoloam_yaw_corrected_deg", 1,
                                 &YawTextMarker::yawLegoloamCb, this);

    sub_yaw_ = nh.subscribe("/global_yaw", 1,
                            &YawTextMarker::yawCb, this);

    // 퍼블리셔 추가
    pub_legoloam_ = nh.advertise<visualization_msgs::Marker>("legoloam_yaw_corrected_deg_marker", 1);
    pub_yaw_ = nh.advertise<visualization_msgs::Marker>("global_yaw_deg_marker", 1);
  }

private:
  void yawLegoloamCb(const std_msgs::Float64& msg) {
    // legoloam_yaw 텍스트 마커
    visualization_msgs::Marker m;
    m.header.frame_id = "velodyne";        // velodyne 프레임 원점
    m.header.stamp    = ros::Time::now();
    m.ns   = "yaw_text";
    m.id   = 0;
    m.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    m.action = visualization_msgs::Marker::ADD;

    // 텍스트
    m.pose.position.x = 1.5;   // 예: x = 1.0 m
    m.pose.position.y = 0.0;   // 예: y = 2.0 m
    m.pose.position.z = 3.5;   // 예: z = 0.5 m (지면 위 0.5 m)
    m.pose.orientation.w = 1.0;            // 방향 무시
    m.scale.z = 4.0;                       // 글씨 크기(m 단위)
    m.color.a = 1.0;
    m.color.r = color_r_;
    m.color.g = color_g_;
    m.color.b = color_b_;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision_) << "legoloam_yaw_corrected_deg " << msg.data << "°";
    m.text = ss.str();

    pub_legoloam_.publish(m);
  }

  void yawCb(const std_msgs::Float64& msg) {
    // global_yaw 텍스트 마커 (빨간색)
    visualization_msgs::Marker m;
    m.header.frame_id = "velodyne";        // velodyne 프레임 원점
    m.header.stamp    = ros::Time::now();
    m.ns   = "yaw_text";
    m.id   = 1;  // 다른 id로 구분
    m.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    m.action = visualization_msgs::Marker::ADD;

    // 텍스트
    m.pose.position.x = 1.5;   // 예: x = 1.0 m
    m.pose.position.y = 0.0;   // 예: y = 2.0 m
    m.pose.position.z = 0.5;   // 예: z = 0.5 m (지면 위 0.5 m)
    m.pose.orientation.w = 1.0;            // 방향 무시
    m.scale.z = 2.0;                       // 글씨 크기(m 단위)
    m.color.a = 1.0;
    m.color.r = 1.0;  // 빨간색
    m.color.g = 0.0;
    m.color.b = 0.0;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision_) << "global_yaw_deg " << msg.data << "°";
    m.text = ss.str();

    pub_yaw_.publish(m);
  }

  ros::Subscriber sub_legoloam_;
  ros::Subscriber sub_yaw_;
  ros::Publisher pub_legoloam_;
  ros::Publisher pub_yaw_;

  int precision_;
  double color_r_, color_g_, color_b_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "yaw_text_marker");
  YawTextMarker ytm;
  ros::spin();
  return 0;
}
