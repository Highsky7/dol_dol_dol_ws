#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math

class CarVisualNode:
    def __init__(self):
        rospy.init_node('car_visual_node', anonymous=True)

        # 파라미터 설정
        self.frame_id = rospy.get_param('~frame_id', 'velodyne')
        self.x_min = rospy.get_param('~x_min', -0.1)
        self.x_max = rospy.get_param('~x_max', 1.1)
        self.y_min = rospy.get_param('~y_min', -0.35)
        self.y_max = rospy.get_param('~y_max', 0.35)
        self.publish_frequency = rospy.get_param('~publish_frequency', 10.0)
        self.corner_radius = rospy.get_param('~corner_radius', 0.2)
        self.corner_segments = rospy.get_param('~corner_segments', 8)

        self.marker_pub = rospy.Publisher('/car_visual', Marker, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.publish_frequency), self.publish_marker)

        rospy.loginfo(
            "Initialized CarVisualNode: frame=%s, x_min=%.2f, x_max=%.2f, y_min=%.2f, y_max=%.2f, radius=%.2f, seg=%d",
            self.frame_id, self.x_min, self.x_max, self.y_min, self.y_max,
            self.corner_radius, self.corner_segments
        )

    def publish_marker(self, event):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "car_visual"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = marker.color.g = marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0

        r = self.corner_radius
        n = self.corner_segments
        pts = []

        # 1) 좌하 모서리 arc: 180° -> 270°
        cx_bl = self.x_min + r
        cy_bl = self.y_min + r
        for i in range(n + 1):
            theta = math.pi + (math.pi/2) * (i / float(n))
            pts.append(Point(cx_bl + r * math.cos(theta), cy_bl + r * math.sin(theta), 0.0))

        # 2) 하단 직선: (x_min+r, y_min) -> (x_max-r, y_min)
        pts.append(Point(self.x_max - r, self.y_min, 0.0))

        # 3) 우하 모서리 arc: 270° -> 360°
        cx_br = self.x_max - r
        cy_br = self.y_min + r
        for i in range(n + 1):
            theta = 3*math.pi/2 + (math.pi/2) * (i / float(n))
            pts.append(Point(cx_br + r * math.cos(theta), cy_br + r * math.sin(theta), 0.0))

        # 4) 우측 직선: (x_max, y_min+r) -> (x_max, y_max-r)
        pts.append(Point(self.x_max, self.y_max - r, 0.0))

        # 5) 우상 모서리 arc: 0° -> 90°
        cx_tr = self.x_max - r
        cy_tr = self.y_max - r
        for i in range(n + 1):
            theta = 0 + (math.pi/2) * (i / float(n))
            pts.append(Point(cx_tr + r * math.cos(theta), cy_tr + r * math.sin(theta), 0.0))

        # 6) 상단 직선: (x_max-r, y_max) -> (x_min+r, y_max)
        pts.append(Point(self.x_min + r, self.y_max, 0.0))

        # 7) 좌상 모서리 arc: 90° -> 180°
        cx_tl = self.x_min + r
        cy_tl = self.y_max - r
        for i in range(n + 1):
            theta = math.pi/2 + (math.pi/2) * (i / float(n))
            pts.append(Point(cx_tl + r * math.cos(theta), cy_tl + r * math.sin(theta), 0.0))

        # 8) 좌측 직선: (x_min, y_max-r) -> 첫 점 (cx_bl + r*cos(pi), cy_bl + r*sin(pi)) closes loop
        pts.append(Point(self.x_min, self.y_min + r, 0.0))
        # 마지막으로 시작점 재삽입
        pts.append(pts[0])

        marker.points = pts
        self.marker_pub.publish(marker)
        rospy.logdebug("Published car_visual marker with all corners rounded")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        CarVisualNode().run()
    except rospy.ROSInterruptException:
        rospy.loginfo("CarVisualNode terminated")
