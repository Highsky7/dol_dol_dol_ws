#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/PointStamped.h>
#include <cmath>

// 지구 반경 (미터)
static constexpr double R_EARTH = 6378137.0;

void latlon_to_local(double lat, double lon, double ref_lat, double ref_lon, double& x, double& y) {
    double lat_rad = lat * M_PI / 180.0;
    double lon_rad = lon * M_PI / 180.0;
    double ref_lat_rad = ref_lat * M_PI / 180.0;
    double ref_lon_rad = ref_lon * M_PI / 180.0;
    x = (lon_rad - ref_lon_rad) * std::cos(ref_lat_rad) * R_EARTH;
    y = (lat_rad - ref_lat_rad) * R_EARTH;
}

class GPSToLocalCartesian {
public:
    GPSToLocalCartesian() {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        pnh.param("ref_lat", ref_lat_, 37.541647);
        pnh.param("ref_lon", ref_lon_, 127.078786);
        pub_ = nh.advertise<geometry_msgs::PointStamped>("local_xy", 10);
        sub_ = nh.subscribe("ublox_gps/fix", 10, &GPSToLocalCartesian::gpsCallback, this);
        ROS_INFO("gps_to_local_cartesian 노드 시작됨 (기준: %.3f, %.3f)", ref_lat_, ref_lon_);
    }

    void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
        double lat = msg->latitude;
        double lon = msg->longitude;
        double x, y;
        latlon_to_local(lat, lon, ref_lat_, ref_lon_, x, y);
        geometry_msgs::PointStamped local_msg;
        local_msg.header = msg->header;
        local_msg.point.x = x;
        local_msg.point.y = y;
        local_msg.point.z = 0.0;
        pub_.publish(local_msg);
        ROS_INFO("Published local_xy: x=%.2f, y=%.2f", x, y);
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;
    double ref_lat_, ref_lon_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gps_to_local_cartesian");
    GPSToLocalCartesian node;
    ros::spin();
    return 0;
}
