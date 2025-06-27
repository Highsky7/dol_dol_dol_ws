#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

static constexpr double R_EARTH = 6378137.0;

// 위도/경도를 기준 좌표(ref_lat, ref_lon)의 로컬 좌표 (x, y)로 변환 (미터 단위)
void latlon_to_local(double lat, double lon, double ref_lat, double ref_lon, double& x, double& y) {
    double lat_rad = lat * M_PI / 180.0;
    double lon_rad = lon * M_PI / 180.0;
    double ref_lat_rad = ref_lat * M_PI / 180.0;
    double ref_lon_rad = ref_lon * M_PI / 180.0;
    x = (lon_rad - ref_lon_rad) * std::cos(ref_lat_rad) * R_EARTH;
    y = (lat_rad - ref_lat_rad) * R_EARTH;
}

// CSV 파일에서 경로 데이터를 읽어와 로컬 좌표 목록을 반환
std::vector<std::pair<double,double>> load_course_data(const std::string& csv_filename, double ref_lat, double ref_lon) {
    std::vector<std::pair<double,double>> points;
    std::ifstream file(csv_filename.c_str());
    if (!file.is_open()) {
        ROS_ERROR("CSV 파일이 존재하지 않습니다: %s", csv_filename.c_str());
        return points;
    }
    std::string line;
    std::getline(file, line); // 헤더 건너뛰기
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> row;
        while (std::getline(ss, item, ',')) {
            row.push_back(item);
        }
        if (row.size() < 3) {
            ROS_WARN("잘못된 CSV 데이터 형식: %s", line.c_str());
            continue;
        }
        try {
            double lon = std::stod(row[1]);
            double lat = std::stod(row[2]);
            double lx, ly;
            latlon_to_local(lat, lon, ref_lat, ref_lon, lx, ly);
            points.emplace_back(lx, ly);
        } catch (const std::exception& e) {
            ROS_WARN("잘못된 CSV 데이터 형식: %s", line.c_str());
            continue;
        }
    }
    file.close();
    return points;
}

// 인접 점 필터링 (두 점 간 거리가 min_distance 이상인 점만 남김)
std::vector<std::pair<double,double>> filter_points_by_distance(const std::vector<std::pair<double,double>>& points, double min_distance) {
    if (points.empty()) return points;
    std::vector<std::pair<double,double>> filtered;
    filtered.push_back(points[0]);
    for (size_t i = 1; i < points.size(); ++i) {
        double dx = points[i].first - filtered.back().first;
        double dy = points[i].second - filtered.back().second;
        if (std::hypot(dx, dy) >= min_distance) {
            filtered.push_back(points[i]);
        }
    }
    return filtered;
}

// 자연(자연경계) 1D 큐빅 스플라인 보간
std::vector<std::pair<double,double>> compute_cubic_spline(const std::vector<std::pair<double,double>>& points, double target_spacing) {
    int n = points.size();
    if (n < 2) {
        ROS_ERROR("보간할 점이 충분하지 않습니다.");
        return {};
    }
    // 누적 길이 s 계산
    std::vector<double> s(n, 0.0);
    for (int i = 1; i < n; ++i) {
        double dx = points[i].first - points[i-1].first;
        double dy = points[i].second - points[i-1].second;
        s[i] = s[i-1] + std::hypot(dx, dy);
    }
    double total_length = s.back();
    if (total_length == 0.0) {
        ROS_ERROR("전체 길이가 0입니다.");
        return {};
    }
    // 좌표 분리
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = points[i].first;
        y[i] = points[i].second;
    }
    // 두 번째 미분(y2) 계산 (자연 경계 조건)
    std::vector<double> y2x(n,0.0), y2y(n,0.0), u(n,0.0);
    for (int i = 1; i < n-1; ++i) {
        double sig = (s[i] - s[i-1]) / (s[i+1] - s[i-1]);
        double p = sig * y2x[i-1] + 2.0;
        y2x[i] = (sig - 1.0) / p;
        double ddx = (x[i+1]-x[i])/(s[i+1]-s[i]) - (x[i]-x[i-1])/(s[i]-s[i-1]);
        u[i] = (6.0 * ddx / (s[i+1]-s[i-1]) - sig * u[i-1]) / p;
    }
    y2x[n-1] = 0.0;
    for (int k = n-2; k >= 0; --k) {
        y2x[k] = y2x[k] * y2x[k+1] + u[k];
    }
    // y좌표에 대해서도 동일 계산
    u.assign(n, 0.0);
    for (int i = 1; i < n-1; ++i) {
        double sig = (s[i] - s[i-1]) / (s[i+1] - s[i-1]);
        double p = sig * y2y[i-1] + 2.0;
        y2y[i] = (sig - 1.0) / p;
        double ddy = (y[i+1]-y[i])/(s[i+1]-s[i]) - (y[i]-y[i-1])/(s[i]-s[i-1]);
        u[i] = (6.0 * ddy / (s[i+1]-s[i-1]) - sig * u[i-1]) / p;
    }
    y2y[n-1] = 0.0;
    for (int k = n-2; k >= 0; --k) {
        y2y[k] = y2y[k] * y2y[k+1] + u[k];
    }
    // 재샘플링
    std::vector<std::pair<double,double>> resampled;
    for (double si = 0.0; si <= total_length; si += target_spacing) {
        if (si > total_length) si = total_length;
        // 구간 찾기
        int k = 0;
        while (k < n-1 && s[k+1] < si) ++k;
        if (k >= n-1) k = n-2;
        double h = s[k+1] - s[k];
        double a = (s[k+1] - si) / h;
        double b = (si - s[k]) / h;
        double xi = a*x[k] + b*x[k+1] + ((a*a*a - a)*y2x[k] + (b*b*b - b)*y2x[k+1]) * (h*h) / 6.0;
        double yi = a*y[k] + b*y[k+1] + ((a*a*a - a)*y2y[k] + (b*b*b - b)*y2y[k+1]) * (h*h) / 6.0;
        resampled.emplace_back(xi, yi);
        if (si == total_length) break;
    }
    // 마지막 점 추가 (누락된 경우)
    if (resampled.empty() || resampled.back().first != points.back().first || resampled.back().second != points.back().second) {
        resampled.push_back(points.back());
    }
    return resampled;
}

class LocalPathPublisher {
public:
    LocalPathPublisher() {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        // 기본 CSV 파일 경로 설정
        std::string pkg_path = ros::package::getPath("gps_to_utm_pkg");
        std::string default_csv = pkg_path + "/data/science.csv";
        pnh.param("csv_filename", csv_filename_, default_csv);
        pnh.param("target_spacing", target_spacing_, 0.2);
        pnh.param("min_distance", min_distance_, 0.3);
        pnh.param("ref_lat", ref_lat_, 37.541647);
        pnh.param("ref_lon", ref_lon_, 127.078786);
        sub_path_ = nh.subscribe("resampled_path", 1, &LocalPathPublisher::dummyPathCallback, this); // 실제 사용되지 않음
        path_pub_ = nh.advertise<nav_msgs::Path>("resampled_path", 1);
        local_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("local_xy", 10);
        // 경로 데이터 로드 및 보간
        auto raw_points = load_course_data(csv_filename_, ref_lat_, ref_lon_);
        if (raw_points.empty()) {
            ROS_ERROR("Course 데이터를 찾을 수 없습니다.");
            ros::shutdown();
            return;
        }
        auto filtered_points = filter_points_by_distance(raw_points, min_distance_);
        resampled_points_ = compute_cubic_spline(filtered_points, target_spacing_);
        ROS_INFO("Resampled local path has %lu points", resampled_points_.size());
        timer_ = nh.createTimer(ros::Duration(0.1), &LocalPathPublisher::timerCallback, this);
    }

    void dummyPathCallback(const nav_msgs::Path::ConstPtr&) {
        // 사용되지 않음
    }

    void timerCallback(const ros::TimerEvent&) {
        ros::Time current_time = ros::Time::now();
        nav_msgs::Path path_msg;
        path_msg.header.frame_id = "reference";
        path_msg.header.stamp = current_time;
        for (auto& pt : resampled_points_) {
            geometry_msgs::PoseStamped pose;
            pose.header.frame_id = "reference";
            pose.header.stamp = current_time;
            pose.pose.position.x = pt.first;
            pose.pose.position.y = pt.second;
            pose.pose.position.z = 0.0;
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);
        }
        path_pub_.publish(path_msg);
        ROS_INFO("Published resampled path with %lu points", resampled_points_.size());
        // 마지막 점 publish (/local_xy)
        if (!resampled_points_.empty()) {
            auto last_pt = resampled_points_.back();
            geometry_msgs::PoseStamped local_pose;
            local_pose.header.frame_id = "reference";
            local_pose.header.stamp = current_time;
            local_pose.pose.position.x = last_pt.first;
            local_pose.pose.position.y = last_pt.second;
            local_pose.pose.position.z = 0.0;
            local_pose.pose.orientation.w = 1.0;
            local_pose_pub_.publish(local_pose);
        }
    }

private:
    ros::Subscriber sub_path_;
    ros::Publisher path_pub_, local_pose_pub_;
    ros::Timer timer_;
    std::string csv_filename_;
    double target_spacing_, min_distance_, ref_lat_, ref_lon_;
    std::vector<std::pair<double,double>> resampled_points_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "local_cartesian_path_publisher");
    LocalPathPublisher node;
    ros::spin();
    return 0;
}
