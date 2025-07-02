#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <vehicle_msgs/Track.h>
#include <vehicle_msgs/TrackCone.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/MarkerArray.h>

#include <sstream>
#include <iomanip>

class ConeFrenetProjector
{
public:
    ConeFrenetProjector()
    : nh_(), pnh_("~"), tf_buffer_(), tf_listener_(tf_buffer_)
    {
        /* ── 파라미터 ───────────────────────── */
        pnh_.param<std::string>("source_frame",  source_frame_,  "velodyne");
        pnh_.param<std::string>("target_frame",  target_frame_,  "reference");
        pnh_.param<std::string>("track_topic",   track_topic_,   "/track");
        pnh_.param<std::string>("out_topic",     out_topic_,     "/track_ref");
        int q; pnh_.param<int>("queue_size", q, 10);

        /* ── Pub / Sub ──────────────────────── */
        pub_track_ref_ = nh_.advertise<vehicle_msgs::Track>(out_topic_, q);
        pub_marker_    = nh_.advertise<visualization_msgs::MarkerArray>(
                           "/track_ref_marker", q);

        sub_track_ = nh_.subscribe(track_topic_, q,
                     &ConeFrenetProjector::trackCallback, this);
    }

private:
    void trackCallback(const vehicle_msgs::Track::ConstPtr& msg_in)
    {
        geometry_msgs::TransformStamped tf;
        try {
            tf = tf_buffer_.lookupTransform(
                    target_frame_, source_frame_,
                    ros::Time(0), ros::Duration(0.05));
        } catch (const tf2::TransformException& e) {
            ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", e.what());
            return;
        }

        vehicle_msgs::Track             msg_out;
        visualization_msgs::MarkerArray m_arr;

        int id = 0;
        for (const auto& cone : msg_in->cones)
        {
            /* 1) velodyne → reference 변환 */
            geometry_msgs::PointStamped pt_src, pt_ref;
            pt_src.header.frame_id = source_frame_;
            pt_src.header.stamp    = ros::Time(0);
            pt_src.point.x = cone.x;  pt_src.point.y = cone.y;  pt_src.point.z = 0.0;
            tf2::doTransform(pt_src, pt_ref, tf);

            /* 2) Track 메시지 */
            vehicle_msgs::TrackCone c;
            c.x = pt_ref.point.x;  c.y = pt_ref.point.y;  c.type = cone.type;
            msg_out.cones.push_back(c);

            /* 3-A) 텍스트(노란색) */
            visualization_msgs::Marker txt;
            txt.header = pt_ref.header;
            txt.ns     = "cone_label";
            txt.id     = id;
            txt.type   = visualization_msgs::Marker::TEXT_VIEW_FACING;
            txt.action = visualization_msgs::Marker::ADD;
            txt.pose.position = pt_ref.point;
            txt.pose.position.z += 0.3;
            txt.scale.z = 0.35;
            txt.color.r = txt.color.g = txt.color.b = 1.0;  // 노란 = R+G
            txt.color.a = 1.0;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2)
               << "(" << pt_ref.point.x << ", " << pt_ref.point.y << ")";
            txt.text = ss.str();
            txt.lifetime = ros::Duration(0.2);
            m_arr.markers.push_back(txt);

            /* 3-B) 포인트(흰색) ★ */
            visualization_msgs::Marker dot;
            dot.header = pt_ref.header;
            dot.ns     = "cone_point";
            dot.id     = id;
            dot.type   = visualization_msgs::Marker::SPHERE;   // 구로 표시
            dot.action = visualization_msgs::Marker::ADD;
            dot.pose.position = pt_ref.point;
            dot.scale.x = dot.scale.y = dot.scale.z = 0.15;    // 점 크기
            dot.color.r = dot.color.g = dot.color.b = 1.0;     // 흰색
            dot.color.a = 1.0;
            dot.lifetime = ros::Duration(0.2);
            m_arr.markers.push_back(dot);

            ++id;
        }

        pub_track_ref_.publish(msg_out);
        pub_marker_.publish(m_arr);
    }

    /* ── 멤버 ───────────────────────────── */
    ros::NodeHandle nh_, pnh_;
    ros::Subscriber sub_track_;
    ros::Publisher  pub_track_ref_, pub_marker_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string source_frame_, target_frame_;
    std::string track_topic_,  out_topic_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cone_frenet_projector");
    ConeFrenetProjector node;
    ros::spin();
    return 0;
}
