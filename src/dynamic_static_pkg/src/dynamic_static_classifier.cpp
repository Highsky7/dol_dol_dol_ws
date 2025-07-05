#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <dynamic_static_pkg/TrackedObjects.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float32.h>
#include <cmath>
#include <set>
#include <map>
#include <iomanip>       
#include <boost/bind.hpp>
#include <std_msgs/Bool.h>

using namespace message_filters;

class DynamicStaticClassifier
{
public:
  DynamicStaticClassifier()
    : v_thresh_(0.9),
      have_vel_(false),
      global_yaw_(0.0),
      alpha_(0.05), // EMA 필터 강도 강화
      prev_v_vehicle_x_(0.0),
      prev_v_vehicle_y_(0.0),
      prev_global_yaw_(0.0),
      prev_auto_throttle_(0.0),      // 추가: 이전 오토쓰로틀 값
      have_auto_throttle_(false),     // 추가: 오토쓰로틀 수신 여부
      weight_x_(0.5), // 추가: x축 속도 가중치
      weight_y_(30.0),  // 추가: y축 속도 가중치
      roi_x_min_(0.0),  roi_x_max_(5.0),
      roi_y_min_(-2.5), roi_y_max_(2.5)   // NEW
  {
    ros::NodeHandle pnh("~");
    pnh.param("v_thresh", v_thresh_, v_thresh_);
    pnh.param("alpha", alpha_, alpha_);
    ROS_INFO("v_thresh set to %.3f m/s, alpha set to %.3f", v_thresh_, alpha_);

    tracked_sub_.subscribe(nh_, "/tracked_objects", 10);
    raw_vel_sub_ = nh_.subscribe("/utm/vel", 10, &DynamicStaticClassifier::velRawCallback, this);
    yaw_sub_ = nh_.subscribe("/global_yaw", 10, &DynamicStaticClassifier::yawCallback, this);
    throttle_sub_ = nh_.subscribe("/auto_throttle", 10, &DynamicStaticClassifier::throttleCallback, this); // 추가: 구독 설정
    tracked_sub_.registerCallback(boost::bind(&DynamicStaticClassifier::trackedCallback, this, _1));

    pub_static_ = nh_.advertise<visualization_msgs::MarkerArray>("/static_objects", 1);
    pub_dynamic_ = nh_.advertise<visualization_msgs::MarkerArray>("/dynamic_objects", 1);
    pub_absolute_vx_vy_ = nh_.advertise<visualization_msgs::MarkerArray>("/absolute_vx_vy", 1);
    pub_dyn_flag_ = nh_.advertise<std_msgs::Bool>("/dynamic_obstacle", 1, /*latched=*/true); // NEW

    ROS_INFO("Node initialized successfully");
  }

private:
  ros::Publisher pub_dyn_flag_;
  void yawCallback(const std_msgs::Float32::ConstPtr& msg)
  {
    global_yaw_ = alpha_ * msg->data + (1.0 - alpha_) * prev_global_yaw_;
    prev_global_yaw_ = global_yaw_;
    ROS_DEBUG("Filtered global_yaw: %.2f deg", global_yaw_ * 180.0 / M_PI);
  }

  void velRawCallback(const geometry_msgs::TwistStamped::ConstPtr& msg)
  {
    last_vel_ = *msg;
    prev_v_vehicle_x_ = alpha_ * last_vel_.twist.linear.x + (1.0 - alpha_) * prev_v_vehicle_x_;
    prev_v_vehicle_y_ = alpha_ * last_vel_.twist.linear.y + (1.0 - alpha_) * prev_v_vehicle_y_;
    have_vel_ = true;
    ROS_DEBUG("Filtered vehicle vel: vx=%.3f, vy=%.3f", prev_v_vehicle_x_, prev_v_vehicle_y_);
  }

  void throttleCallback(const std_msgs::Float32::ConstPtr& msg)
  {
    auto_throttle_ = alpha_ * msg->data + (1.0 - alpha_) * prev_auto_throttle_;
    prev_auto_throttle_ = auto_throttle_;
    have_auto_throttle_ = true;
    ROS_DEBUG("Filtered auto_throttle: %.3f m/s", auto_throttle_);
  }

  void trackedCallback(const dynamic_static_pkg::TrackedObjectsConstPtr& tracks)
  {
    std::set<int> current_ids(tracks->id.begin(), tracks->id.end());
    visualization_msgs::MarkerArray static_markers, dynamic_markers, absolute_vx_vy_markers;
    bool dynamic_present = false;

    for (int lost_id : last_ids_) {
      if (!current_ids.count(lost_id)) {
        visualization_msgs::Marker del;
        del.header = tracks->header;
        del.ns = "static";
        del.id = lost_id;
        del.action = visualization_msgs::Marker::DELETE;
        static_markers.markers.push_back(del);
        dynamic_markers.markers.push_back(del);

        visualization_msgs::Marker del_text;
        del_text.header = tracks->header;
        del_text.ns = "absolute_vx_vy";
        del_text.id = lost_id;
        del_text.action = visualization_msgs::Marker::DELETE;
        absolute_vx_vy_markers.markers.push_back(del_text);
      }
    }

    if (!have_vel_) {
      ROS_WARN_THROTTLE(1.0, "/utm/vel [Off] -> No correction by subtracting car's velocity");
    } else {
      ROS_INFO_THROTTLE(1.0, "/utm/vel [On]: vx=%.3f, vy=%.3f -> absolute by subtracting car's velocity",
                        prev_v_vehicle_x_, prev_v_vehicle_y_);
    }

    for (size_t i = 0; i < tracks->id.size(); ++i)
    {
      const auto& center = tracks->center[i];

      bool in_roi =
          center.x >= roi_x_min_ && center.x <= roi_x_max_ &&
          center.y >= roi_y_min_ && center.y <= roi_y_max_;   // 수정
      if (!in_roi) {
        ROS_DEBUG("ID %d skipped (out of ROI)", tracks->id[i]);
        continue;
      }
      float obj_vx = tracks->vx[i];
      float obj_vy = tracks->vy[i];

      float abs_vx, abs_vy;
      if (have_vel_) {
        double cos_yaw = std::cos(-global_yaw_);
        double sin_yaw = std::sin(-global_yaw_);
        // float v_vehicle_velo_x = cos_yaw * prev_v_vehicle_x_ + sin_yaw * prev_v_vehicle_y_;
        // float v_vehicle_velo_y = -sin_yaw * prev_v_vehicle_x_ + cos_yaw * prev_v_vehicle_y_;
        // abs_vx = obj_vx - v_vehicle_velo_x; // 차량 속도 빼기
        // 만약 /auto_throttle 토픽에서 데이터를 받은 적이 없으면(have_auto_throttle_ == false), auto_throttle_을 빼지 않고 obj_vx를 그대로 사용합니다. (0.0을 빼므로 결과는 동일).
        // abs_vy = obj_vy - v_vehicle_velo_y;
        abs_vx = obj_vx - (have_auto_throttle_ ? auto_throttle_ : 0.0); // 수정: auto_throttle_ 빼기
        abs_vy = obj_vy;
        // ROS_DEBUG("Vehicle vel (Velodyne): vx=%.3f, vy=%.3f", v_vehicle_velo_x, v_vehicle_velo_y);
      } else {
        abs_vx = obj_vx;
        abs_vy = obj_vy;
      }
      // float obj_speed = std::hypot(abs_vx, abs_vy);
      // std::hypot(x, y)는 두 값의 유클리드 거리(즉, $\sqrt{x^2 + y^2}$)를 계산하는 C++ 표준 라이브러리 함수
      float obj_speed = std::sqrt(weight_x_ * abs_vx * abs_vx + weight_y_ * abs_vy * abs_vy); // 수정: 가중치 적용

      bool is_static_by_position = false;
      if (last_centers_.find(tracks->id[i]) != last_centers_.end()) {
        float dx = tracks->center[i].x - last_centers_[tracks->id[i]].x;
        float dy = tracks->center[i].y - last_centers_[tracks->id[i]].y;
        if (std::hypot(dx, dy) < 0.1) {
          is_static_by_position = true;
        }
      }
      last_centers_[tracks->id[i]] = tracks->center[i];

      bool is_static = (obj_speed <= v_thresh_) || is_static_by_position;
      if (!is_static)
        dynamic_present = true;            // <── ROI 안 동적 발견!

      visualization_msgs::Marker m;
      m.header = tracks->header;
      m.ns = is_static ? "static" : "dynamic";
      m.id = tracks->id[i];
      m.type = visualization_msgs::Marker::CUBE;
      m.pose.position = tracks->center[i];
      m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = m.scale.z = 0.7;
      m.color.a = 0.5;
      m.lifetime = ros::Duration(0.5);
      if (is_static) {
        m.color.g = 1.0;
      } else {
        m.color.r = 1.0;
      }
      if (is_static) {
        static_markers.markers.push_back(m);
      } else {
        dynamic_markers.markers.push_back(m);
      }

      visualization_msgs::Marker text_marker;
      text_marker.header = tracks->header;
      text_marker.ns = "absolute_vx_vy";
      text_marker.id = tracks->id[i];
      text_marker.action = visualization_msgs::Marker::ADD;
      text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text_marker.pose.position.x = tracks->center[i].x - 0.35;
      text_marker.pose.position.y = tracks->center[i].y + 1.2;
      text_marker.pose.position.z = tracks->center[i].z + 0.5;
      text_marker.pose.orientation.w = 1.0;
      std::stringstream ss;
      ss << "abs_vx: " << std::fixed << std::setprecision(3) << abs_vx
         << "\nabs_vy: " << std::fixed << std::setprecision(3) << abs_vy;
      text_marker.text = ss.str();
      text_marker.color.a = 1.0;
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 0.0;
      text_marker.scale.z = 0.3;
      text_marker.lifetime = ros::Duration(0.5);
      absolute_vx_vy_markers.markers.push_back(text_marker);

      ROS_DEBUG("Object %d: rel_vx=%.3f, rel_vy=%.3f, abs_vx=%.3f, abs_vy=%.3f, speed=%.3f, class=%s",
                tracks->id[i], obj_vx, obj_vy, abs_vx, abs_vy, obj_speed, is_static ? "static" : "dynamic");
    }

    pub_static_.publish(static_markers);
    pub_dynamic_.publish(dynamic_markers);
    pub_absolute_vx_vy_.publish(absolute_vx_vy_markers);
    std_msgs::Bool flag;        // NEW
    flag.data = dynamic_present; // NEW
    pub_dyn_flag_.publish(flag);  // NEW
    last_ids_.swap(current_ids);
  }

  ros::NodeHandle nh_;
  double v_thresh_;
  double alpha_;
  Subscriber<dynamic_static_pkg::TrackedObjects> tracked_sub_;
  ros::Subscriber raw_vel_sub_;
  geometry_msgs::TwistStamped last_vel_;
  bool have_vel_;
  ros::Subscriber yaw_sub_;
  double global_yaw_;
  double prev_v_vehicle_x_, prev_v_vehicle_y_;
  double prev_global_yaw_;
  ros::Publisher pub_static_, pub_dynamic_, pub_absolute_vx_vy_;
  std::set<int> last_ids_;
  std::map<int, geometry_msgs::Point> last_centers_;
  ros::Subscriber throttle_sub_;     // 추가
  float auto_throttle_;              // 추가
  float prev_auto_throttle_;         // 추가
  bool have_auto_throttle_;          // 추가
  float weight_x_;                   // 추가: x축 속도 가중치
  float weight_y_;                   // 추가: y축 속도 가중치
  double roi_x_min_, roi_x_max_;
  double roi_y_min_, roi_y_max_;   // NEW
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dynamic_static_classifier");
  DynamicStaticClassifier node;
  ros::spin();
  return 0;
}