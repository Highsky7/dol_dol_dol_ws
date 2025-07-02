#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32, Bool

class Judgement:
    def __init__(self):
        # --- 구독자 설정: ROS 토픽에서 데이터를 수신 ---
        # 1. 차선 기반 조향각
        self.sub_lane = rospy.Subscriber('auto_steer_angle_lane', Float32, self.lane_callback)
        self.sub_lane_status = rospy.Subscriber('lane_detection_status', Bool, self.lane_status_callback)
        
        # 2. 장애물 유무 및 경로 계획 기반 조향각 (수정된 부분)
        self.sub_obstacle_existence = rospy.Subscriber('/obstacle_existence', Bool, self.obstacle_callback)
        self.sub_rrt = rospy.Subscriber('/auto_steer_angle_rrt', Float32, self.rrt_callback)
        self.sub_gps = rospy.Subscriber('/auto_steer_angle_gps', Float32, self.gps_callback)
        
        # 3. 터널 기반 조향각 (필요시 주석 해제)
        # self.sub_tunnel = rospy.Subscriber('auto_steer_angle_tunnel', Float32, self.tunnel_callback)

        # --- 퍼블리셔 설정 ---
        # 1. 최종 조향각 발행
        self.pub_steering = rospy.Publisher("steering_angle", Float32, queue_size=10)
        # 2. 고정 쓰로틀값 발행
        self.pub_throttle = rospy.Publisher("auto_throttle", Float32, queue_size=10)

        # --- 변수 초기화 ---
        self.lane_angle = None      # 차선 기반 조향각
        self.lane_detected = False  # 차선 감지 상태
        self.obstacle_exists = False # 장애물 존재 여부
        self.rrt_angle = None       # RRT 알고리즘 기반 조향각
        self.gps_angle = None       # GPS 기반 조향각
        self.tunnel_angle = None    # 터널 기반 조향각

        # 주기적 실행을 위한 타이머 (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    # --- 콜백 함수 정의 ---
    def lane_callback(self, msg):
        """차선 조향각 수신"""
        self.lane_angle = msg.data

    def lane_status_callback(self, msg):
        """차선 감지 상태 수신"""
        self.lane_detected = msg.data

    def obstacle_callback(self, msg):
        """장애물 존재 여부 수신"""
        self.obstacle_exists = msg.data

    def rrt_callback(self, msg):
        """RRT 조향각 수신"""
        self.rrt_angle = msg.data

    def gps_callback(self, msg):
        """GPS 조향각 수신"""
        self.gps_angle = msg.data
        
    def tunnel_callback(self, msg):
        """터널 조향각 수신"""
        self.tunnel_angle = msg.data

    # --- 데이터 발행 로직 ---
    def publish_steering(self, event):
        """조향각 판단 및 발행 로직"""
        steering_angle = None

        # 우선순위 1: 차선이 감지되면 차선 조향각 사용
        if self.lane_detected is True and self.lane_angle is not None:
            steering_angle = self.lane_angle
            rospy.loginfo("Using lane angle: %.2f", steering_angle)
        
        # 우선순위 2: 장애물이 있으면 RRT 조향각 사용
        elif self.obstacle_exists is True and self.rrt_angle is not None:
            steering_angle = self.rrt_angle
            rospy.loginfo("Obstacle detected. Using RRT angle: %.2f", steering_angle)

        # 우선순위 3: 장애물이 없으면 GPS 조향각 사용
        elif self.obstacle_exists is False and self.gps_angle is not None:
            steering_angle = self.gps_angle
            rospy.loginfo("No obstacle. Using GPS angle: %.2f", steering_angle)
            
        # 우선순위 4: (필요시) 터널 조향각 사용
        # elif self.tunnel_angle is not None:
        #     steering_angle = self.tunnel_angle
        #     rospy.loginfo("Using tunnel angle: %.2f", steering_angle)

        # 유효한 조향각이 있으면 발행
        if steering_angle is not None:
            self.pub_steering.publish(Float32(data=steering_angle))

    def publish_throttle(self, event):
        """쓰로틀 발행 로직 (고정값)"""
        throttle = 0.4
        self.pub_throttle.publish(Float32(data=throttle))
        rospy.loginfo("Current throttle: %.2f", throttle)

    def timer_callback(self, event):
        """타이머에 맞춰 조향각과 쓰로틀을 주기적으로 발행"""
        self.publish_steering(event)
        self.publish_throttle(event)

if __name__ == '__main__':
    rospy.init_node("judgement_node")
    Judgement()
    rospy.spin()