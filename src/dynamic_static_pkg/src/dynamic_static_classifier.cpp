#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <dynamic_static_pkg/TrackedObjects.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float32.h> // 추가: global_yaw를 위한 헤더
#include <cmath>
#include <set>

using namespace message_filters;

class DynamicStaticClassifier
{
public:
  DynamicStaticClassifier()
    : v_thresh_(0.3),
      have_vel_(false),
      global_yaw_(0.0) // 추가: global_yaw_ 초기화
  {
    ros::NodeHandle pnh("~");
    pnh.param("v_thresh", v_thresh_, v_thresh_);
    ROS_INFO("v_thresh set to %.3f m/s", v_thresh_);

    // ① /tracked_objects를 message_filters로 구독
    tracked_sub_.subscribe(nh_, "/tracked_objects", 10);

    // ② /utm/vel을 일반 Subscriber로 구독
    raw_vel_sub_ = nh_.subscribe(
      "/utm/vel", 10,
      &DynamicStaticClassifier::velRawCallback, this
    );

    // 추가: /global_yaw를 일반 Subscriber로 구독
    yaw_sub_ = nh_.subscribe(
      "/global_yaw", 10,
      &DynamicStaticClassifier::yawCallback, this
    );

    // ③ tracked_objects 콜백만 등록
    tracked_sub_.registerCallback(
      boost::bind(&DynamicStaticClassifier::trackedCallback, this, _1)
    );

    // 퍼블리셔 설정
    pub_static_  = nh_.advertise<visualization_msgs::MarkerArray>("/static_objects",  1);
    pub_dynamic_ = nh_.advertise<visualization_msgs::MarkerArray>("/dynamic_objects", 1);

    ROS_INFO("Node initialized successfully"); // 디버깅용 메시지 (선택사항)
  }

private:
  // 추가: /global_yaw 콜백
  void yawCallback(const std_msgs::Float32::ConstPtr& msg)
  {
    global_yaw_ = msg->data; // yaw 값 저장
    ROS_DEBUG("Received global_yaw: %.2f deg", global_yaw_ * 180.0 / M_PI);
  }

  // ——————————————
  // 추가된 부분: /utm/vel 콜백
  void velRawCallback(const geometry_msgs::TwistStamped::ConstPtr& msg)
  {
    last_vel_ = *msg;      // 가장 최근 속도 저장
    have_vel_ = true;      // 수신 플래그 설정
  }

  // tracked_objects 콜백
  void trackedCallback(const dynamic_static_pkg::TrackedObjectsConstPtr& tracks)
  {
    std::set<int> current_ids(tracks->id.begin(), tracks->id.end());
    visualization_msgs::MarkerArray static_markers, dynamic_markers;

    // (0) 삭제 지시: 이전 프레임에만 있던 ID들
    for (int lost_id : last_ids_) {
      if (!current_ids.count(lost_id)) {
        visualization_msgs::Marker del;
        del.header = tracks->header;
        del.ns     = "static";  // (혹은 "dynamic")
        del.id     = lost_id;
        del.action = visualization_msgs::Marker::DELETE;
        static_markers.markers.push_back(del);
      }
    }

    // (1) GNSS 보정 by car's velocity 로그
    if (!have_vel_) {
      ROS_WARN_THROTTLE(1.0, "/utm/vel [Off] -> No correction by subtracting car's velocity");
    } else {
      ROS_INFO_THROTTLE(1.0, "/utm/vel [On]: vx=%.3f, vy=%.3f -> corrected by subtracting car's velocity",
        last_vel_.twist.linear.x,
        last_vel_.twist.linear.y
      );
    }

    // (2) 객체별 상대속도 & 분류
    for (size_t i = 0; i < tracks->id.size(); ++i)
    {
      float obj_vx = tracks->vx[i];
      float obj_vy = tracks->vy[i];

      // GNSS 보정: Velodyne 프레임으로 변환 후 절대 속도 계산
      float rel_vx, rel_vy;
      if (have_vel_) {
        // UTM 속도를 Velodyne 프레임으로 변환
        double cos_yaw = std::cos(-global_yaw_);
        double sin_yaw = std::sin(-global_yaw_);
        float v_vehicle_velo_x = cos_yaw * last_vel_.twist.linear.x + sin_yaw * last_vel_.twist.linear.y;
        float v_vehicle_velo_y = -sin_yaw * last_vel_.twist.linear.x + cos_yaw * last_vel_.twist.linear.y;

        // 물체의 절대 속도 계산 (Velodyne 프레임)
        rel_vx = obj_vx + v_vehicle_velo_x;
        rel_vy = obj_vy + v_vehicle_velo_y;
      } else {
        rel_vx = obj_vx;
        rel_vy = obj_vy;
      }
      float obj_speed = std::hypot(rel_vx, rel_vy);

      visualization_msgs::Marker m;
      m.header    = tracks->header;
      m.ns        = (obj_speed > v_thresh_) ? "dynamic" : "static";
      m.id        = tracks->id[i];
      m.type      = visualization_msgs::Marker::CUBE;
      m.pose.position = tracks->center[i];  // x, y, z를 position에 할당
      m.pose.orientation.w = 1.0;           // 회전 정보는 기본값으로 설정 (회전하지 않음)

      m.scale.x   = m.scale.y = m.scale.z = 0.7;
      m.color.a   = 0.5;
      m.lifetime  = ros::Duration(0.5);
      if (obj_speed > v_thresh_) m.color.r = 1.0;
      else                       m.color.g = 1.0;

      if (obj_speed > v_thresh_)
        dynamic_markers.markers.push_back(m);
      else
        static_markers.markers.push_back(m);
    }

    // (3) 퍼블리시 & ID 저장
    pub_static_.publish(static_markers);
    pub_dynamic_.publish(dynamic_markers);
    last_ids_.swap(current_ids);
  }

  // — 멤버 변수들 —
  ros::NodeHandle nh_;
  double v_thresh_;

  // 트래킹 토픽 (message_filters)
  Subscriber<dynamic_static_pkg::TrackedObjects> tracked_sub_;

  // 보정 속도 토픽 (일반 subscriber)
  ros::Subscriber raw_vel_sub_;
  geometry_msgs::TwistStamped last_vel_;
  bool have_vel_;

  // 추가: yaw 구독 및 저장
  ros::Subscriber yaw_sub_;
  double global_yaw_;

  // 퍼블리셔
  ros::Publisher pub_static_, pub_dynamic_;

  // 삭제 지시용 ID 저장
  std::set<int> last_ids_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dynamic_static_classifier");
  DynamicStaticClassifier node;
  ros::spin();
  return 0;
}