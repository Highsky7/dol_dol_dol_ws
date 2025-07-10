#include <ros/ros.h>
#include <ros/package.h>

#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>          // ★

#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

static constexpr double R_EARTH = 6378137.0;

struct PointCov                       // ★ 좌표 + Cov
{
    double x;
    double y;
    double cov;
};

/* 위도/경도를 기준 좌표(ref_lat, ref_lon)의 로컬 좌표 (x, y)로 변환 (미터 단위) */
void latlon_to_local(double lat, double lon,
                     double ref_lat, double ref_lon,
                     double& x, double& y)
{
    const double lat_rad = lat * M_PI / 180.0;
    const double lon_rad = lon * M_PI / 180.0;
    const double ref_lat_rad = ref_lat * M_PI / 180.0;
    const double ref_lon_rad = ref_lon * M_PI / 180.0;

    x = (lon_rad - ref_lon_rad) * std::cos(ref_lat_rad) * R_EARTH;
    y = (lat_rad - ref_lat_rad) * R_EARTH;
}

/* CSV 파일 → PointCov 벡터 */
std::vector<PointCov>
load_course_data(const std::string& csv_filename,
                 double ref_lat, double ref_lon)
{
    std::vector<PointCov> pts;
    std::ifstream file(csv_filename.c_str());
    if (!file.is_open()) {
        ROS_ERROR("CSV 파일이 존재하지 않습니다: %s", csv_filename.c_str());
        return pts;
    }

    std::string line;
    std::getline(file, line);      // 헤더 skip

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item; std::vector<std::string> row;
        while (std::getline(ss, item, ',')) row.push_back(item);

        if (row.size() < 6) {      // index,Long,Lat,UTM_X,UTM_Y,Cov_00
            ROS_WARN("잘못된 CSV 형식: %s", line.c_str());
            continue;
        }
        try {
            const double lon  = std::stod(row[1]);
            const double lat  = std::stod(row[2]);
            const double cov0 = std::stod(row[5]);
            double lx, ly;
            latlon_to_local(lat, lon, ref_lat, ref_lon, lx, ly);
            pts.push_back({lx, ly, cov0});
        } catch (const std::exception&) {
            ROS_WARN("숫자 변환 실패: %s", line.c_str());
            continue;
        }
    }
    return pts;
}

/* 인접 점 필터링 (거리 ≥ min_distance) */
std::vector<PointCov>
filter_points_by_distance(const std::vector<PointCov>& pts, double min_d)
{
    if (pts.empty()) return pts;
    std::vector<PointCov> filtered; filtered.push_back(pts.front());
    for (size_t i = 1; i < pts.size(); ++i) {
        const double dx = pts[i].x - filtered.back().x;
        const double dy = pts[i].y - filtered.back().y;
        if (std::hypot(dx, dy) >= min_d) filtered.push_back(pts[i]);
    }
    return filtered;
}

/* 자연 1D 큐빅 스플라인 보간 (cov까지 선형 보간) */
std::vector<PointCov>
compute_cubic_spline(const std::vector<PointCov>& pts,
                     double target_spacing)
{
    const int n = pts.size();
    if (n < 2) {
        ROS_ERROR("보간할 점이 부족합니다.");
        return {};
    }

    /* 누적 거리 s */
    std::vector<double> s(n, 0.0);
    for (int i = 1; i < n; ++i)
        s[i] = s[i-1] +
               std::hypot(pts[i].x - pts[i-1].x,
                          pts[i].y - pts[i-1].y);

    const double total_len = s.back();
    if (total_len == 0.0) {
        ROS_ERROR("전체 길이가 0입니다.");
        return {};
    }

    /* 좌표 및 2차 미분 계산 (x, y 각각) ---------------------------- */
    auto spline_second = [&](const std::vector<double>& v)
    {
        std::vector<double> y2(n, 0.0), u(n, 0.0);
        for (int i = 1; i < n-1; ++i) {
            double sig = (s[i] - s[i-1]) / (s[i+1] - s[i-1]);
            double p   = sig * y2[i-1] + 2.0;
            y2[i]      = (sig - 1.0) / p;
            double dv  = (v[i+1]-v[i])/(s[i+1]-s[i])
                       - (v[i]-v[i-1])/(s[i]-s[i-1]);
            u[i] = (6.0 * dv / (s[i+1]-s[i-1]) - sig * u[i-1]) / p;
        }
        y2[n-1] = 0.0;
        for (int k = n-2; k >= 0; --k)
            y2[k] = y2[k]*y2[k+1] + u[k];
        return y2;
    };

    std::vector<double> xs(n), ys(n), covs(n);
    for (int i = 0; i < n; ++i) {
        xs[i]   = pts[i].x;
        ys[i]   = pts[i].y;
        covs[i] = pts[i].cov;
    }
    const auto y2x = spline_second(xs);
    const auto y2y = spline_second(ys);

    /* 재샘플링 ---------------------------------------------- */
    std::vector<PointCov> out;
    for (double si = 0.0; si <= total_len; si += target_spacing) {
        if (si > total_len) si = total_len;
        int k = 0;
        while (k < n-1 && s[k+1] < si) ++k;
        if (k >= n-1) k = n-2;
        const double h = s[k+1] - s[k];
        const double a = (s[k+1] - si) / h;
        const double b = (si - s[k]) / h;

        const double xi = a*xs[k] + b*xs[k+1]
                        + ((a*a*a - a)*y2x[k] + (b*b*b - b)*y2x[k+1]) * h*h/6.0;
        const double yi = a*ys[k] + b*ys[k+1]
                        + ((a*a*a - a)*y2y[k] + (b*b*b - b)*y2y[k+1]) * h*h/6.0;
        const double ci = a*covs[k] + b*covs[k+1];     // cov 선형 보간 ★
        out.push_back({xi, yi, ci});
        if (si == total_len) break;
    }
    if (out.empty() ||
        out.back().x != pts.back().x ||
        out.back().y != pts.back().y)
        out.push_back(pts.back());

    return out;
}

/* ---------------------------------------------------------------------- */

class LocalPathPublisher
{
public:
    LocalPathPublisher()
    {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");

        const std::string pkg_path = ros::package::getPath("gps_to_utm_pkg");
        const std::string def_csv  = pkg_path + "/data/science.csv";

        pnh.param("csv_filename",   csv_filename_,   def_csv);
        pnh.param("target_spacing", target_spacing_, 0.2);
        pnh.param("min_distance",   min_distance_,   0.3);
        pnh.param("ref_lat",        ref_lat_,        37.541647);
        pnh.param("ref_lon",        ref_lon_,        127.078786);
        pnh.param("cov_threshold",  cov_thresh_,     0.01);    // ★ 파라미터

        path_pub_    = nh.advertise<nav_msgs::Path>           ("resampled_path", 1);
        marker_pub_  = nh.advertise<visualization_msgs::MarkerArray>("cov_markers", 1); // ★

        /* CSV → 보간 경로 준비 */
        auto raw_pts      = load_course_data(csv_filename_, ref_lat_, ref_lon_);
        if (raw_pts.empty()) { ros::shutdown(); return; }

        auto filtered_pts = filter_points_by_distance(raw_pts, min_distance_);
        resampled_pts_    = compute_cubic_spline(filtered_pts, target_spacing_);

        ROS_INFO("Resampled %lu points (threshold %.4f)", resampled_pts_.size(), cov_thresh_);
        timer_ = nh.createTimer(ros::Duration(0.1),
                                &LocalPathPublisher::timerCallback, this);
    }

private:
    /* 타이머 콜백: Path + MarkerArray 퍼블리시 */
    void timerCallback(const ros::TimerEvent&)
    {
        const ros::Time now = ros::Time::now();

        /* ------- nav_msgs/Path -------- */
        nav_msgs::Path path_msg;
        path_msg.header.frame_id = "reference";
        path_msg.header.stamp    = now;
        for (const auto& p : resampled_pts_) {
            geometry_msgs::PoseStamped pose;
            pose.header = path_msg.header;
            pose.pose.position.x = p.x;
            pose.pose.position.y = p.y;
            pose.pose.orientation.w = 1.0;
            path_msg.poses.push_back(pose);
        }
        path_pub_.publish(path_msg);

        /* ------- MarkerArray ---------- */
        visualization_msgs::Marker mk;
        mk.header = path_msg.header;
        mk.ns     = "path_cov";
        mk.id     = 0;
        mk.type   = visualization_msgs::Marker::LINE_STRIP;
        mk.action = visualization_msgs::Marker::ADD;
        mk.scale.x = 0.1;               // 선 굵기
        mk.pose.orientation.w = 1.0;

        for (const auto& p : resampled_pts_) {
            geometry_msgs::Point pt;
            pt.x = p.x;  pt.y = p.y;  pt.z = 0.0;
            mk.points.push_back(pt);

            std_msgs::ColorRGBA c;
            if (p.cov >= cov_thresh_) {      // orange
                c.r = 1.0; c.g = 0.0; c.b = 0.0; c.a = 1.0;
            } else {                         // 초록색
                c.r = 0.0; c.g = 1.0; c.b = 0.0; c.a = 0.4;
            }
            mk.colors.push_back(c);          // Vertex별 색 지정 ★
        }

        visualization_msgs::MarkerArray ma;
        ma.markers.push_back(mk);
        marker_pub_.publish(ma);
    }

    /* -------------------------------- */
    ros::Publisher path_pub_, marker_pub_;
    ros::Timer     timer_;

    std::string csv_filename_;
    double target_spacing_, min_distance_, ref_lat_, ref_lon_;
    double cov_thresh_;                           // ★

    std::vector<PointCov> resampled_pts_;
};

/* ----------------------------- main ------------------------------ */
int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_cartesian_path_publisher");
    LocalPathPublisher node;
    ros::spin();
    return 0;
}
