#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <dynamic_static_pkg/TrackedObjects.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float32.h>
#include <cmath>
#include <set>
#include <map>

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
      prev_global_yaw_(0.0)
  {
    ros::NodeHandle pnh("~");
    pnh.param("v_thresh", v_thresh_, v_thresh_);
    pnh.param("alpha", alpha_, alpha_);
    ROS_INFO("v_thresh set to %.3f m/s, alpha set to %.3f", v_thresh_, alpha_);

    tracked_sub_.subscribe(nh_, "/tracked_objects", 10);
    raw_vel_sub_ = nh_.subscribe("/utm/vel", 10, &DynamicStaticClassifier::velRawCallback, this);
    yaw_sub_ = nh_.subscribe("/global_yaw", 10, &DynamicStaticClassifier::yawCallback, this);
    tracked_sub_.registerCallback(boost::bind(&DynamicStaticClassifier::trackedCallback, this, _1));

    pub_static_ = nh_.advertise<visualization_msgs::MarkerArray>("/static_objects", 1);
    pub_dynamic_ = nh_.advertise<visualization_msgs::MarkerArray>("/dynamic_objects", 1);
    pub_absolute_vx_vy_ = nh_.advertise<visualization_msgs::MarkerArray>("/absolute_vx_vy", 1);

    ROS_INFO("Node initialized successfully");
  }

private:
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

  void trackedCallback(const dynamic_static_pkg::TrackedObjectsConstPtr& tracks)
  {
    std::set<int> current_ids(tracks->id.begin(), tracks->id.end());
    visualization_msgs::MarkerArray static_markers, dynamic_markers, absolute_vx_vy_markers;

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
      float obj_vx = tracks->vx[i];
      float obj_vy = tracks->vy[i];

      float abs_vx, abs_vy;
      if (have_vel_) {
        double cos_yaw = std::cos(-global_yaw_);
        double sin_yaw = std::sin(-global_yaw_);
        float v_vehicle_velo_x = cos_yaw * prev_v_vehicle_x_ + sin_yaw * prev_v_vehicle_y_;
        float v_vehicle_velo_y = -sin_yaw * prev_v_vehicle_x_ + cos_yaw * prev_v_vehicle_y_;
        abs_vx = obj_vx - v_vehicle_velo_x; // 차량 속도 빼기
        abs_vy = obj_vy - v_vehicle_velo_y;
        ROS_DEBUG("Vehicle vel (Velodyne): vx=%.3f, vy=%.3f", v_vehicle_velo_x, v_vehicle_velo_y);
      } else {
        abs_vx = obj_vx;
        abs_vy = obj_vy;
      }
      float obj_speed = std::hypot(abs_vx, abs_vy);

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
      text_marker.pose.position.x = tracks->center[i].x - 0.4 ;
      text_marker.pose.position.y = tracks->center[i].y + 1.2 ;
      text_marker.pose.position.z = tracks->center[i].z + 0.5 ;
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
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dynamic_static_classifier");
  DynamicStaticClassifier node;
  ros::spin();
  return 0;
}