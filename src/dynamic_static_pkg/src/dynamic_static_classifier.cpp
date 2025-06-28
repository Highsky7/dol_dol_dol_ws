// src/dynamic_static_classifier.cpp

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <dynamic_static_pkg/TrackedObjects.h>
#include <geometry_msgs/TwistStamped.h>
#include <visualization_msgs/MarkerArray.h>

// alias the policy & synchronizer
using SyncPolicy   = message_filters::sync_policies::ApproximateTime<
                        dynamic_static_pkg::TrackedObjects,
                        geometry_msgs::TwistStamped
                    >;
using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

class DynamicStaticClassifier {
public:
  DynamicStaticClassifier()
    : tf_listener_(tf_buffer_)
  {
    // subscribe the two streams
    tracks_sub_.subscribe(nh_, "/tracked_objects", 10);
    vel_sub_   .subscribe(nh_, "/utm/vel",           10);

    // use the "queue size" constructor to select the right overload:
    sync_.reset(new Synchronizer(50, tracks_sub_, vel_sub_));
    sync_->registerCallback(
      boost::bind(&DynamicStaticClassifier::callback, this, _1, _2));

    pub_static_  = nh_.advertise<visualization_msgs::MarkerArray>("/static_objects", 1);
    pub_dynamic_ = nh_.advertise<visualization_msgs::MarkerArray>("/dynamic_objects",1);

    nh_.param("v_thresh", v_thresh_, 1.2);
  }

private:
  void callback(const dynamic_static_pkg::TrackedObjectsConstPtr& tracks,
                const geometry_msgs::TwistStampedConstPtr& gps_vel)
  {

    ROS_INFO("dynamic_static_classifier CALLBACK entered: #tracks=%zu, utm_vel=(%.2f,%.2f)",
            tracks->id.size(),
            gps_vel->twist.linear.x,
            gps_vel->twist.linear.y);
    // ego–motion compensation
    geometry_msgs::Vector3Stamped ego_utm, ego_lidar;
    ego_utm.header = gps_vel->header;
    ego_utm.vector = gps_vel->twist.linear;
    try {
      tf_buffer_.transform(ego_utm, ego_lidar, "velodyne", ros::Duration(0.05));
    } catch (tf2::TransformException &ex) {
      ROS_WARN("TF failed: %s", ex.what());
      return;
    }

    visualization_msgs::MarkerArray statics, dynamics;
    for (size_t i = 0; i < tracks->id.size(); ++i) {
      int id       = tracks->id[i];
      auto center  = tracks->center[i];
      float vx_obj = tracks->vx[i];
      float vy_obj = tracks->vy[i];

      float rvx = vx_obj - ego_lidar.vector.x;
      float rvy = vy_obj - ego_lidar.vector.y;
      float res_speed = std::hypot(rvx, rvy);

      visualization_msgs::Marker m;
      m.header        = tracks->header;
      m.ns            = "dyn_stat";
      m.id            = id;
      m.type          = visualization_msgs::Marker::CUBE;
      m.action        = visualization_msgs::Marker::ADD;
      m.pose.position = center;
      m.scale.x = 0.5; m.scale.y = 0.5; m.scale.z = 1.0;
      m.color.a = 0.6;
      if (res_speed < v_thresh_) {
        m.color.r = 0; m.color.g = 1; m.color.b = 0;
        statics.markers.push_back(m);
      } else {
        m.color.r = 1; m.color.g = 0; m.color.b = 0;
        dynamics.markers.push_back(m);
      }
    }
    pub_static_.publish(statics);
    pub_dynamic_.publish(dynamics);
  }

  ros::NodeHandle nh_;
  tf2_ros::Buffer           tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  message_filters::Subscriber<dynamic_static_pkg::TrackedObjects> tracks_sub_;
  message_filters::Subscriber<geometry_msgs::TwistStamped>        vel_sub_;
  boost::shared_ptr<Synchronizer> sync_;

  ros::Publisher pub_static_, pub_dynamic_;
  double v_thresh_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "dynamic_static_classifier");
  DynamicStaticClassifier node;
  ros::spin();
  return 0;
}
