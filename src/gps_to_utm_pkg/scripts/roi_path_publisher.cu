// scripts/roi_path_publisher.cu
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <cmath>
#include <vector>

// 2D tuple type
using T2 = thrust::tuple<float,float>;

// Unary functor for reference->velodyne transform
struct TransformFunctor {
    float cos_t, sin_t, tx, ty;
    __host__ __device__
    T2 operator()(const T2 &in) const {
        float x = thrust::get<0>(in), y = thrust::get<1>(in);
        return thrust::make_tuple(
            cos_t * x - sin_t * y + tx,
            sin_t * x + cos_t * y + ty
        );
    }
};

// Binary functor for distance between two T2 points
struct DistanceFunctor {
    __host__ __device__
    float operator()(const T2 &curr, const T2 &prev) const {
        float x1 = thrust::get<0>(curr), y1 = thrust::get<1>(curr);
        float x0 = thrust::get<0>(prev), y0 = thrust::get<1>(prev);
        float dx = x1 - x0, dy = y1 - y0;
        return sqrtf(dx*dx + dy*dy);
    }
};

class ROIPathPublisher {
public:
    ROIPathPublisher()
      : nh_(), pnh_("~"), tf_listener_(tf_buffer_), N_(0)
    {
        pnh_.param("roi_arc_length", roi_arc_length_, 3.8);
        sub_path_   = nh_.subscribe("resampled_path", 1, &ROIPathPublisher::pathCallback, this);
        marker_pub_ = nh_.advertise<visualization_msgs::Marker>("roi_path_marker", 1);
        rrt_pub_    = nh_.advertise<geometry_msgs::PointStamped>("rrt_target", 1);
        timer_      = nh_.createTimer(ros::Duration(0.01), &ROIPathPublisher::timerCallback, this);
        ROS_INFO("GPU ROIPathPublisher started (arc=%.2f)", roi_arc_length_);
    }

private:
    void pathCallback(const nav_msgs::Path::ConstPtr &msg) {
        size_t N = msg->poses.size();
        if (N == 0) return;

        // 1) lookup TF
        geometry_msgs::TransformStamped tfst;
        try {
            tfst = tf_buffer_.lookupTransform("velodyne","reference",
                                              ros::Time(0), ros::Duration(0.1));
        } catch (const tf2::TransformException &e) {
            ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", e.what());
            return;
        }

        // extract yaw from quaternion
        auto &q = tfst.transform.rotation;
        float siny_cosp = 2*(q.w*q.z + q.x*q.y);
        float cosy_cosp = 1 - 2*(q.y*q.y + q.z*q.z);
        float yaw = std::atan2(siny_cosp, cosy_cosp);

        float tx = tfst.transform.translation.x;
        float ty = tfst.transform.translation.y;

        // 2) host -> prepare input tuples
        host_in_.resize(N);
        for (size_t i=0; i<N; ++i) {
            host_in_[i] = thrust::make_tuple(
                msg->poses[i].pose.position.x,
                msg->poses[i].pose.position.y
            );
        }

        // 3) copy to device
        thrust::device_vector<T2> d_in  = host_in_;
        thrust::device_vector<T2> d_out(N);
        thrust::device_vector<float> d_arc(N);

        // 4) unary transform
        thrust::transform(
            d_in.begin(), d_in.end(),
            d_out.begin(),
            TransformFunctor{cosf(yaw), sinf(yaw), tx, ty}
        );

        // 5) binary distance
        if (N > 0) {
            d_arc[0] = 0.0f;
            thrust::transform(
                d_out.begin()+1, d_out.end(),
                d_out.begin(),
                d_arc.begin()+1,
                DistanceFunctor()
            );
        }

        // 6) inclusive scan (prefix-sum)
        thrust::inclusive_scan(d_arc.begin(), d_arc.end(), d_arc.begin());

        // 7) copy back to host
        host_out_.resize(N);
        host_arc_.resize(N);
        thrust::copy(d_out.begin(), d_out.end(), host_out_.begin());
        thrust::copy(d_arc.begin(), d_arc.end(), host_arc_.begin());
        N_ = N;
    }

    void timerCallback(const ros::TimerEvent&) {
        if (N_ == 0) return;

        // find closest point to (0,0)
        size_t min_idx = 0;
        float min_d = std::numeric_limits<float>::infinity();
        for (size_t i=0; i<N_; ++i) {
            float x = thrust::get<0>(host_out_[i]);
            float y = thrust::get<1>(host_out_[i]);
            float d = std::hypot(x,y);
            if (d < min_d) { min_d = d; min_idx = i; }
        }

        // determine end index for ROI
        size_t end_idx = min_idx;
        while (end_idx < N_ && host_arc_[end_idx] - host_arc_[min_idx] <= roi_arc_length_) {
            ++end_idx;
        }
        if (end_idx <= min_idx) return;

        // publish marker
        visualization_msgs::Marker m;
        m.header.frame_id = "velodyne";
        m.header.stamp    = ros::Time::now();
        m.ns = "roi_path"; m.id = 0;
        m.type = visualization_msgs::Marker::LINE_STRIP;
        m.scale.x = 0.1;
        m.color.r = 1; m.color.g = 0; m.color.b = 1; m.color.a = 0.7;
        m.points.reserve(end_idx - min_idx);
        for (size_t i=min_idx; i<end_idx; ++i) {
            geometry_msgs::Point pt;
            pt.x = thrust::get<0>(host_out_[i]);
            pt.y = thrust::get<1>(host_out_[i]);
            pt.z = 0.0;
            m.points.push_back(pt);
        }
        marker_pub_.publish(m);

        // publish RRT target
        geometry_msgs::PointStamped tgt;
        tgt.header = m.header;
        tgt.point.x = thrust::get<0>(host_out_[end_idx-1]);
        tgt.point.y = thrust::get<1>(host_out_[end_idx-1]);
        tgt.point.z = 0.0;
        rrt_pub_.publish(tgt);
    }

    ros::NodeHandle nh_, pnh_;
    ros::Subscriber sub_path_;
    ros::Publisher marker_pub_, rrt_pub_;
    ros::Timer timer_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    std::vector<T2> host_in_, host_out_;
    std::vector<float> host_arc_;
    size_t N_;
    double roi_arc_length_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "roi_path_publisher");
    ROIPathPublisher node;
    ros::spin();
    return 0;
}
