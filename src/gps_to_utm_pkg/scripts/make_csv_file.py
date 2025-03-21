#!/usr/bin/env python3
import rospy
import csv
from sensor_msgs.msg import NavSatFix

# CSV 파일 생성
csv_file = "gps_data.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["latitude", "longitude"])  # 헤더 추가

# 콜백 함수
def callback(data):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data.latitude, data.longitude])
    rospy.loginfo(f"Saved: {data.latitude}, {data.longitude}")

# ROS 노드 실행
rospy.init_node('gps_logger', anonymous=True)
rospy.Subscriber("/ublox_gps/fix", NavSatFix, callback)
rospy.spin()
