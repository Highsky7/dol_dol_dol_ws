#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped

# 기준 좌표 (건국대학교 서울캠퍼스 근처; rosparam으로 조정 가능)
# 와우도 : 37.540085 127.076543
# 스마트팩토리 주차장 : 37.540603 127.079843
REF_LAT = rospy.get_param("~ref_lat", 37.540085)
REF_LON = rospy.get_param("~ref_lon", 127.076543)
R_EARTH = 6378137.0  # 지구 반경 (미터)

def latlon_to_local(lat, lon, ref_lat, ref_lon):
    """ 위도/경도를 기준 좌표(ref_lat, ref_lon)에 대해 로컬 카르테시안 좌표 (x, y)로 변환 (미터 단위) """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    x = (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad) * R_EARTH
    y = (lat_rad - ref_lat_rad) * R_EARTH
    return x, y

class GPSToLocalCartesian:
    def __init__(self):
        rospy.init_node("gps_to_local_cartesian", anonymous=True)
        self.pub = rospy.Publisher("local_xy", PointStamped, queue_size=10)
        rospy.Subscriber("ublox_gps/fix", NavSatFix, self.gps_callback)
        rospy.loginfo("gps_to_local_cartesian 노드 시작됨 (기준: %.3f, %.3f)", REF_LAT, REF_LON)
        rospy.spin()

    def gps_callback(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        x, y = latlon_to_local(lat, lon, REF_LAT, REF_LON)
        local_msg = PointStamped()
        local_msg.header = msg.header
        local_msg.point.x = x
        local_msg.point.y = y
        local_msg.point.z = 0.0
        self.pub.publish(local_msg)
        rospy.loginfo("Published local_xy: x=%.2f, y=%.2f", x, y)

if __name__ == '__main__':
    try:
        GPSToLocalCartesian()
    except rospy.ROSInterruptException:
        pass
