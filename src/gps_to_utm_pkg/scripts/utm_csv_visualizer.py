#!/usr/bin/env python3
import rospy
import csv
import os
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# CSV 파일 경로 (여기서는 CSV가 git/gnss/src/utm_coordinates.csv에 있다고 가정)
csv_filename = "/home/jecs/git/gnss/src/utm_coordinates.csv"

def load_utm_data():
    """
    CSV 파일에서 UTM 좌표를 읽고,
    첫 번째 좌표를 원점(0,0)으로 하여 모든 좌표를 상대 좌표로 변환합니다.
    """
    utm_points = []
    if not os.path.exists(csv_filename):
        rospy.logerr("CSV 파일이 존재하지 않습니다: %s", csv_filename)
        return []

    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 헤더 스킵
            first_point = None
            for row in reader:
                easting = float(row[1])    # CSV: [timestamp, easting, northing, ...]
                northing = float(row[2])
                if first_point is None:
                    first_point = (easting, northing)
                    rospy.loginfo("기준 좌표(원점): (%.3f, %.3f)", first_point[0], first_point[1])
                # 상대 좌표 계산: 기준 좌표를 뺌
                local_easting = easting - first_point[0]
                local_northing = northing - first_point[1]
                utm_points.append((local_easting, local_northing))
    except Exception as e:
        rospy.logerr("CSV 파일을 읽는 중 오류 발생: %s", str(e))
    
    return utm_points

def publish_marker(utm_points):
    """
    변환된 로컬 좌표들을 RViz에 Marker 메시지로 발행합니다.
    """
    marker_pub = rospy.Publisher("utm_visualization", Marker, queue_size=1)
    rospy.init_node("utm_csv_visualizer", anonymous=True)
    rate = rospy.Rate(1)  # 1Hz

    while not rospy.is_shutdown():
        marker = Marker()
        marker.header.frame_id = "world"  # CSV 데이터가 "world" frame에 있다고 가정
        marker.header.stamp = rospy.Time.now()
        marker.ns = "utm_points"
        marker.id = 0
        marker.type = Marker.POINTS  # 여러 점을 표현하는 타입
        marker.action = Marker.ADD
        marker.scale.x = 0.5  # 점 크기 (X)
        marker.scale.y = 0.5  # 점 크기 (Y)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # 로컬 좌표를 Marker 메시지에 추가
        for easting, northing in utm_points:
            point = Point()
            point.x = easting
            point.y = northing
            point.z = 0.0
            marker.points.append(point)

        marker_pub.publish(marker)
        rospy.loginfo("로컬 좌표 %d개 발행", len(utm_points))
        rate.sleep()

if __name__ == "__main__":
    utm_points = load_utm_data()
    if utm_points:
        publish_marker(utm_points)
    else:
        rospy.logerr("CSV 파일에서 좌표를 찾을 수 없습니다.")
