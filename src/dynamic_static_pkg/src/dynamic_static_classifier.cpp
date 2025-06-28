#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/cache.h>  
#include <dynamic_static_pkg/TrackedObjects.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <cmath>

using namespace message_filters;

class DynamicStaticClassifier
{
public:
  DynamicStaticClassifier()
    : v_thresh_(0.3)
  {
    ros::NodeHandle pnh("~");
    pnh.param("v_thresh", v_thresh_, v_thresh_);
    ROS_INFO("v_thresh set to %.3f m/s", v_thresh_);

    // ① /tracked_objects 구독
    tracked_sub_.subscribe(nh_, "/tracked_objects", 10);

    // ② /utm/vel 구독 + Cache 설정 (큐 크기 50, 데이터 보관 기간 제한 없음)
    vel_sub_.subscribe(nh_, "/utm/vel", 50);
    vel_cache_.reset(new Cache<geometry_msgs::TwistStamped>(vel_sub_, 50));

    // ③ tracked_objects 콜백만 등록
    tracked_sub_.registerCallback(
      boost::bind(&DynamicStaticClassifier::trackedCallback, this, _1)
    );

    // 퍼블리셔 설정
    pub_static_  = nh_.advertise<visualization_msgs::MarkerArray>("/static_objects",  1);
    pub_dynamic_ = nh_.advertise<visualization_msgs::MarkerArray>("/dynamic_objects", 1);
  }

private:
  void trackedCallback(const dynamic_static_pkg::TrackedObjectsConstPtr& tracks)
  {
    // 1) 해당 시점 근처 속도 메시지들 검색 (±5초 범위)
    ros::Time t = tracks->header.stamp;
    auto vel_msgs = vel_cache_->getInterval(t - ros::Duration(5.0),
                                            t + ros::Duration(5.0));

    // 2) 가장 최근 메시지 선택 (없으면 zero)
    geometry_msgs::TwistStamped vel;
    if (!vel_msgs.empty()) {
      vel = *vel_msgs.back();
    }

    visualization_msgs::MarkerArray static_markers, dynamic_markers;

    for (size_t i = 0; i < tracks->id.size(); ++i)
    {
      // (예시) 객체 속력 계산
      float obj_speed = std::hypot(tracks->vx[i], tracks->vy[i]);
      // 속도 메시지의 평면 속도 (필요 시 활용)
      float utm_vx = vel.twist.linear.x;
      float utm_vy = vel.twist.linear.y;

      // Marker 생성
      visualization_msgs::Marker m;
      m.header = tracks->header;
      m.ns     = (obj_speed > v_thresh_) ? "dynamic" : "static";
      m.id     = tracks->id[i];
      m.type   = visualization_msgs::Marker::CUBE;
      m.pose.position = tracks->center[i];
      m.scale.x = m.scale.y = m.scale.z = 1.0;
      m.color.a = 0.8;
      if (obj_speed > v_thresh_) {
        m.color.r = 1.0;  // 동적: 빨강
      } else {
        m.color.g = 1.0;  // 정적: 초록
      }

      if (obj_speed > v_thresh_)
        dynamic_markers.markers.push_back(m);
      else
        static_markers.markers.push_back(m);
    }

    // 퍼블리시
    pub_static_.publish(static_markers);
    pub_dynamic_.publish(dynamic_markers);
  }

  ros::NodeHandle nh_;
  double v_thresh_;

  // message_filters subscribers & cache
  Subscriber<dynamic_static_pkg::TrackedObjects> tracked_sub_;
  Subscriber<geometry_msgs::TwistStamped>        vel_sub_;
  boost::shared_ptr<Cache<geometry_msgs::TwistStamped>> vel_cache_;

  // publishers
  ros::Publisher pub_static_, pub_dynamic_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dynamic_static_classifier");
  DynamicStaticClassifier node;
  ros::spin();
  return 0;
}
