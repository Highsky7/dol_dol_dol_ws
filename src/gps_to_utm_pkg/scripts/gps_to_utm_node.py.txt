#!/usr/bin/env python
import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped
import utm
import csv
import os

# CSV 파일 경로 (데이터 폴더 내)
csv_filename = os.path.expanduser("~/git/gnss/src/gps_to_utm_pkg/data/utm_coordinates.csv")

# 노드 시작 시 CSV 파일이 없을 때만 생성 및 헤더 작성 (즉, 정적 CSV 파일로 사용)
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "easting", "northing", "zone_number", "zone_letter"])
    rospy.loginfo("CSV 파일이 생성되었습니다: %s", csv_filename)
else:
    rospy.loginfo("CSV 파일이 이미 존재합니다. (새로운 데이터는 기록되지 않습니다)")

def gps_callback(gps_msg):
    # 위도와 경도 값 추출
    lat = gps_msg.latitude
    lon = gps_msg.longitude

    # UTM 변환
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)

    # PointStamped 메시지에 변환된 좌표 설정
    utm_point = PointStamped()
    utm_point.header = gps_msg.header
    utm_point.point.x = easting
    utm_point.point.y = northing
    utm_point.point.z = 0.0  # 높이는 0

    # 변환된 좌표 발행
    pub.publish(utm_point)

    # CSV 파일에 새로운 데이터를 기록하지 않습니다.
    # (파일이 이미 존재하면 더 이상 업데이트하지 않습니다.)
    # 만약 처음 한 번만 기록하고 싶다면, 이 부분을 주석 처리합니다.
    # with open(csv_filename, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([gps_msg.header.stamp.to_sec(), easting, northing, zone_number, zone_letter])

    rospy.loginfo("Published UTM: easting=%.3f, northing=%.3f", easting, northing)

if __name__ == '__main__':
    rospy.init_node('gps_to_utm_node', anonymous=True)
    pub = rospy.Publisher('utm_xy', PointStamped, queue_size=10)
    rospy.Subscriber('ublox_gps/fix', NavSatFix, gps_callback)
    rospy.loginfo("GPS to UTM 변환 노드가 시작되었습니다.")
    rospy.spin()
