#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/PointStamped.h>
#include <GeographicLib/UTMUPS.hpp>
#include <fstream>
#include <iomanip>

ros::Publisher pub;

void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
    double lat = msg->latitude;
    double lon = msg->longitude;

    // UTM 변환 결과를 받을 변수들
    int  zone;        // UTM zone 번호
    bool northp;      // 북반구 여부 (true = 북반구, false = 남반구)
    double easting, northing;

    // UTM 변환: Forward(lat, lon, zone, northp, x, y)
    GeographicLib::UTMUPS::Forward(lat, lon, zone, northp, easting, northing);

    // 필요 시 북반구/남반구를 문자로 변환
    char zone_letter = northp ? 'N' : 'S';

    // PointStamped 메시지로 발행
    geometry_msgs::PointStamped utm_point;
    utm_point.header = msg->header;
    utm_point.point.x = easting;
    utm_point.point.y = northing;
    utm_point.point.z = 0.0;
    pub.publish(utm_point);

    // 로깅: zone 번호와 문자, easting/northing 값
    ROS_INFO("Published UTM: zone=%d%c, easting=%.3f, northing=%.3f",
             zone, zone_letter, easting, northing);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "gps_to_utm_node");
    ros::NodeHandle nh;

    // CSV 파일 경로 설정
    std::string pkg_path = ros::package::getPath("gps_to_utm_pkg");
    std::string csv_filename = pkg_path + "/data/utm_coordinates.csv";

    // 패키지 내 data 폴더에 utm_coordinates.csv가 없으면 생성
    std::ifstream infile(csv_filename);
    if (!infile.good()) {
        std::ofstream file(csv_filename);
        file << "timestamp,easting,northing,zone_number,zone_letter\n";
        file.close();
        ROS_INFO("CSV 파일이 생성되었습니다: %s", csv_filename.c_str());
    } else {
        ROS_INFO("CSV 파일이 이미 존재합니다. (새로운 데이터는 기록되지 않습니다)");
    }

    // utm_xy 토픽으로 PointStamped 발행
    pub = nh.advertise<geometry_msgs::PointStamped>("utm_xy", 10);

    // ublox_gps/fix 토픽 구독하여 gpsCallback 호출
    ros::Subscriber sub = nh.subscribe("ublox_gps/fix", 10, gpsCallback);

    ROS_INFO("GPS to UTM 변환 노드가 시작되었습니다.");
    ros::spin();
    return 0;
}
