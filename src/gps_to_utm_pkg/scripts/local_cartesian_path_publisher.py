#!/usr/bin/env python3
import rospy
import csv
import os
import math
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import CubicSpline
import rospkg

# rospkg를 이용하여 패키지 gps_to_utm_pkg의 경로를 가져옵니다.
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("gps_to_utm_pkg")

# CSV 파일 경로 (ilgam_lake.csv; 대회측 제공 CSV)
# 만약 다른 csv 파일을 사용하고 싶다면 주석 처리된 smart_factory1.csv 줄을 사용할 수 있습니다.
default_csv = os.path.join(pkg_path, "data", "ilgam_lake.csv")
# default_csv = os.path.join(pkg_path, "data", "smart_factory1.csv")

# rosparam으로 설정 가능한 파라미터; 기본값은 위에서 설정한 default_csv입니다.
csv_filename = rospy.get_param("~csv_filename", default_csv)

# rosparam으로 설정 가능한 파라미터들
TARGET_SPACING = rospy.get_param("~target_spacing", 0.2)  # 보간 후 resampling 간격 (미터)
MIN_DISTANCE = rospy.get_param("~min_distance", 0.3)        # 인접 점 필터링 최소 거리 (미터)
# 와우도 : 37.540085 127.076543
# 스마트팩토리 주차장 : 37.540603 127.079843
REF_LAT = rospy.get_param("~ref_lat", 37.540085)               # 기준 위도
REF_LON = rospy.get_param("~ref_lon", 127.076543)              # 기준 경도
R_EARTH = 6378137.0

def latlon_to_local(lat, lon, ref_lat, ref_lon):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    x = (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad) * R_EARTH
    y = (lat_rad - ref_lat_rad) * R_EARTH
    return x, y

def load_course_data():
    points = []
    if not os.path.exists(csv_filename):
        rospy.logerr("CSV 파일이 존재하지 않습니다: %s", csv_filename)
        return points
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                try:
                    lon = float(row[1])
                    lat = float(row[2])
                    local_x, local_y = latlon_to_local(lat, lon, REF_LAT, REF_LON)
                    points.append((local_x, local_y))
                except ValueError:
                    rospy.logwarn("잘못된 CSV 데이터 형식: %s", row)
    except Exception as e:
        rospy.logerr("CSV 파일 읽기 오류: %s", str(e))
    return points

def filter_points_by_distance(points, min_distance=MIN_DISTANCE):
    if not points:
        return points
    filtered = [points[0]]
    for pt in points[1:]:
        dx = pt[0] - filtered[-1][0]
        dy = pt[1] - filtered[-1][1]
        if math.hypot(dx, dy) >= min_distance:
            filtered.append(pt)
    return filtered

def compute_cubic_spline(points, target_spacing=TARGET_SPACING):
    if len(points) < 2:
        rospy.logerr("보간할 점이 충분하지 않습니다.")
        return []
    s = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        s.append(s[-1] + math.hypot(dx, dy))
    total_length = s[-1]
    if total_length == 0:
        rospy.logerr("전체 길이가 0입니다.")
        return []
    s = np.array(s)
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    cs_x = CubicSpline(s, x_coords)
    cs_y = CubicSpline(s, y_coords)
    s_new = np.arange(0, total_length, target_spacing)
    if s_new[-1] < total_length:
        s_new = np.append(s_new, total_length)
    x_new = cs_x(s_new)
    y_new = cs_y(s_new)
    return list(zip(x_new, y_new))

class LocalPathPublisher:
    def __init__(self):
        rospy.init_node("local_cartesian_path_publisher", anonymous=True)
        self.path_pub = rospy.Publisher("resampled_path", Path, queue_size=1)
        self.local_pose_pub = rospy.Publisher("local_xy", PoseStamped, queue_size=10)
        raw_points = load_course_data()
        if not raw_points:
            rospy.logerr("Course 데이터를 찾을 수 없습니다.")
            rospy.signal_shutdown("No data")
        filtered_points = filter_points_by_distance(raw_points, MIN_DISTANCE)
        self.resampled_points = compute_cubic_spline(filtered_points, TARGET_SPACING)
        rospy.loginfo("Resampled local path has %d points", len(self.resampled_points))
        rospy.Timer(rospy.Duration(5.0), self.timer_callback)

    def timer_callback(self, event):
        current_time = rospy.Time.now()
        path_msg = Path()
        path_msg.header.frame_id = "reference"
        path_msg.header.stamp = current_time
        for (x, y) in self.resampled_points:
            pose = PoseStamped()
            pose.header.frame_id = "reference"
            pose.header.stamp = current_time
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        rospy.loginfo("Published resampled path with %d points", len(self.resampled_points))
        # 마지막 점 publish (/local_xy)
        if self.resampled_points:
            last_x, last_y = self.resampled_points[-1]
            local_pose = PoseStamped()
            local_pose.header.frame_id = "reference"
            local_pose.header.stamp = current_time
            local_pose.pose.position.x = last_x
            local_pose.pose.position.y = last_y
            local_pose.pose.position.z = 0.0
            local_pose.pose.orientation.w = 1.0
            self.local_pose_pub.publish(local_pose)

if __name__ == "__main__":
    try:
        LocalPathPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
