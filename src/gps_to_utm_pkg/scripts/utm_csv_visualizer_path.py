#!/usr/bin/env python3
import rospy
import csv
import os
import math
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.interpolate import CubicSpline

# CSV 파일 경로 (utm_coordinates.csv; 대회측 제공 CSV)
csv_filename = os.path.expanduser("~/git/gnss/src/gps_to_utm_pkg/data/utm_coordinates.csv")

# 파라미터
TARGET_SPACING = 0.2      # 10cm 간격으로 resampling
MIN_DISTANCE = 0.3        # 너무 가까운 점은 제거 (예: 0.3m)

def load_raw_path_data():
    """
    CSV 파일에서 UTM 좌표(raw data)를 읽어 (easting, northing) 리스트로 반환합니다.
    CSV 파일은 헤더가 ["timestamp", "easting", "northing", ...] 형태로 기록되어 있다고 가정합니다.
    """
    points = []
    if not os.path.exists(csv_filename):
        rospy.logerr("CSV 파일이 존재하지 않습니다: %s", csv_filename)
        return points
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # 헤더 스킵
            for row in reader:
                try:
                    # CSV 파일 구조: timestamp, easting, northing, ...
                    easting = float(row[1])
                    northing = float(row[2])
                    points.append((easting, northing))
                except ValueError:
                    rospy.logwarn("잘못된 CSV 데이터 형식: %s", row)
    except Exception as e:
        rospy.logerr("CSV 파일 읽기 오류: %s", str(e))
    return points

def filter_points_by_distance(points, min_distance=MIN_DISTANCE):
    """
    연속된 점들 사이의 거리가 min_distance보다 작으면 제거합니다.
    """
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
    """
    주어진 점들에 대해 누적 길이를 파라미터로 사용하여 CubicSpline을 적용한 후,
    target_spacing 간격으로 resampling한 점들을 반환합니다.
    """
    if len(points) < 2:
        rospy.logerr("보간할 점이 충분하지 않습니다.")
        return []
    # 누적 거리 계산
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
    # CubicSpline 보간 (각각 x(s)와 y(s))
    cs_x = CubicSpline(s, x_coords)
    cs_y = CubicSpline(s, y_coords)
    # target_spacing 간격의 s 값을 생성
    s_new = np.arange(0, total_length, target_spacing)
    if s_new[-1] < total_length:
        s_new = np.append(s_new, total_length)
    x_new = cs_x(s_new)
    y_new = cs_y(s_new)
    return list(zip(x_new, y_new))

class PathPublisher:
    def __init__(self):
        rospy.init_node("utm_csv_visualizer_path", anonymous=True)
        self.path_pub = rospy.Publisher("resampled_path", Path, queue_size=1)
        # CSV에서 데이터를 한 번 읽어 처리합니다.
        raw_points = load_raw_path_data()
        if not raw_points:
            rospy.logerr("Raw path 데이터를 찾을 수 없습니다.")
            rospy.signal_shutdown("No data")
        filtered_points = filter_points_by_distance(raw_points, MIN_DISTANCE)
        self.resampled_points = compute_cubic_spline(filtered_points, TARGET_SPACING)
        rospy.loginfo("Resampled path has %d points", len(self.resampled_points))
        # 5초마다 publish하는 타이머를 설정합니다.
        rospy.Timer(rospy.Duration(5.0), self.timer_callback)

    def timer_callback(self, event):
        current_time = rospy.Time.now()  # 모든 PoseStamped에 동일한 타임스탬프 사용
        path_msg = Path()
        path_msg.header.frame_id = "world"
        path_msg.header.stamp = current_time
        for (x, y) in self.resampled_points:
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.header.stamp = current_time
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        rospy.loginfo("Published resampled path with %d points", len(self.resampled_points))

if __name__ == "__main__":
    try:
        PathPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
