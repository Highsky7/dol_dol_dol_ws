#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class CarVisualNode:
    def __init__(self):
        # 노드 초기화
        rospy.init_node('car_visual_node', anonymous=True)

        # 파라미터 읽기
        self.frame_id = rospy.get_param('~frame_id', 'velodyne')
        self.x_min = rospy.get_param('~x_min', -0.1)  # 차량 중심 기준 왼쪽 끝
        self.x_max = rospy.get_param('~x_max', 1.1)   # 차량 중심 기준 오른쪽 끝
        self.y_min = rospy.get_param('~y_min', -0.35) # 차량 중심 기준 아래쪽 끝
        self.y_max = rospy.get_param('~y_max', 0.35)  # 차량 중심 기준 위쪽 끝
        self.publish_frequency = rospy.get_param('~publish_frequency', 10.0)  # 10Hz

        # 퍼블리셔 설정
        self.marker_pub = rospy.Publisher('/car_visual', Marker, queue_size=1)

        # 타이머 설정
        rospy.Timer(rospy.Duration(1.0 / self.publish_frequency), self.publish_marker)

        rospy.loginfo("CarVisualNode initialized with frame_id=%s, x_min=%f, x_max=%f, y_min=%f, y_max=%f",
                      self.frame_id, self.x_min, self.x_max, self.y_min, self.y_max)

    def publish_marker(self, event):
        # 마커 메시지 생성
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "car_visual"
        marker.id = 0
        marker.type = Marker.LINE_STRIP  # 테두리만 그리기 위해 LINE_STRIP 사용
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # 선 두께 (5cm)
        marker.color.r = 1.0   # 하얀색 (R=1, G=1, B=1)
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0   # 불투명
        marker.pose.orientation.w = 1.0  # 기본 방향

        # 사각형 꼭짓점 정의 (시계방향: 좌하 -> 우하 -> 우상 -> 좌상 -> 좌하)
        points = [
            Point(self.x_min, self.y_min, 0.0),  # 좌하
            Point(self.x_max, self.y_min, 0.0),  # 우하
            Point(self.x_max, self.y_max, 0.0),  # 우상
            Point(self.x_min, self.y_max, 0.0),  # 좌상
            Point(self.x_min, self.y_min, 0.0)   # 다시 좌하로 닫기
        ]
        marker.points = points

        # 마커 발행
        self.marker_pub.publish(marker)
        rospy.logdebug("Published car_visual marker")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = CarVisualNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("CarVisualNode terminated")