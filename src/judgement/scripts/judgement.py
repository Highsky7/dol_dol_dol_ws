#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32, Bool

class SteeringJudgement:
    def __init__(self):
        # 구독자 설정: ROS 토픽에서 데이터를 수신
        self.sub_lane = rospy.Subscriber('auto_steer_angle_lane', Float32, self.lane_callback)
        self.sub_cone = rospy.Subscriber('auto_steer_angle_cone', Float32, self.cone_callback)
        self.sub_tunnel = rospy.Subscriber('auto_steer_angle_tunnel', Float32, self.tunnel_callback)
        self.sub_lane_status = rospy.Subscriber('lane_detection_status', Bool, self.lane_status_callback)
        
        # 퍼블리셔 설정: 판단된 조향각을 발행
        self.pub_steering = rospy.Publisher("steering_angle", Float32, queue_size=10)
        
        # 변수 초기화
        self.lane_angle = None  # 차선 기반 조향각
        self.cone_angle = None  # 콘 기반 조향각
        self.tunnel_angle = None  # 터널 기반 조향각
        self.lane_detected = False  # 차선 감지 상태
        
        # 주기적 퍼블리시를 위한 타이머 (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_steering)

    def lane_callback(self, msg):
        """차선 조향각 콜백 함수"""
        self.lane_angle = msg.data

    def cone_callback(self, msg):
        """콘 조향각 콜백 함수"""
        self.cone_angle = msg.data

    def tunnel_callback(self, msg):
        """터널 조향각 콜백 함수"""
        self.tunnel_angle = msg.data

    def lane_status_callback(self, msg):
        """차선 감지 상태 콜백 함수"""
        self.lane_detected = msg.data

    def publish_steering(self, event):
        """조향각 판단 및 퍼블리시 로직"""
        steering_angle = None

        # 1. lane_detected가 True이면 lane_angle 사용
        if self.lane_detected and self.lane_angle is not None:
            steering_angle = self.lane_angle
            rospy.loginfo("Using lane angle: %.2f", steering_angle)
        # 2. lane_detected가 False이고 cone_angle이 있으면 cone_angle 사용
        elif self.cone_angle is not None:
            steering_angle = self.cone_angle
            rospy.loginfo("Using cone angle: %.2f", steering_angle)
        # 3. 위 조건이 모두 안 되면 tunnel_angle 사용 (원래 조건 복구)
        elif self.tunnel_angle is not None:
            steering_angle = self.tunnel_angle
            rospy.loginfo("Using tunnel angle: %.2f", steering_angle)
        else:
            rospy.logwarn("No valid steering angle available")

        # 유효한 조향각이 있으면 퍼블리시
        if steering_angle is not None:
            self.pub_steering.publish(Float32(data=steering_angle))

if __name__ == '__main__':
    rospy.init_node("steering_judgement_node")
    SteeringJudgement()
    rospy.spin()