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
        # 2. 계산된 쓰로틀값 발행
        self.pub_throttle = rospy.Publisher("auto_throttle", Float32, queue_size=10)

        # --- 변수 초기화 ---
        self.lane_angle = None          # 차선 기반 조향각
        self.lane_detected = False      # 차선 감지 상태
        self.obstacle_exists = False     # 장애물 존재 여부
        self.rrt_angle = None           # RRT 알고리즘 기반 조향각
        self.gps_angle = None           # GPS 기반 조향각
        self.tunnel_angle = None        # 터널 기반 조향각

        self.current_steering_angle = 0.0 # 현재 조향각 (스로틀 계산에 사용)
        self.steering_source_valid = False # 유효한 조향각 소스가 있는지 여부

        # --- 속도 계획 파라미터 ---
        self.max_throttle = rospy.get_param("~max_throttle", 0.5)
        self.min_throttle = rospy.get_param("~min_throttle", 0.3)
        self.steering_throttle_reduction_factor = rospy.get_param("~steering_throttle_reduction_factor", 0.01)

        rospy.loginfo("Throttle parameters initialized: max_throttle=%.2f, min_throttle=%.2f, reduction_factor=%.4f",
                      self.max_throttle, self.min_throttle, self.steering_throttle_reduction_factor)

        # 주기적 실행을 위한 타이머 (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    # --- 콜백 함수 정의 ---
    def lane_callback(self, msg):
        self.lane_angle = msg.data

    def lane_status_callback(self, msg):
        self.lane_detected = msg.data

    def obstacle_callback(self, msg):
        self.obstacle_exists = msg.data

    def rrt_callback(self, msg):
        self.rrt_angle = msg.data

    def gps_callback(self, msg):
        self.gps_angle = msg.data

    def tunnel_callback(self, msg):
        self.tunnel_angle = msg.data

    # --- 데이터 발행 로직 ---
    def publish_steering(self, event):
        steering_angle_output = None
        chosen_source = "None"

        # 우선순위 1: 차선이 감지되면 차선 조향각 사용
        if self.lane_detected is True and self.lane_angle is not None:
            steering_angle_output = self.lane_angle
            chosen_source = "lane"
            
        # 우선순위 2: 장애물이 있으면 RRT 조향각 사용
        elif self.obstacle_exists is True and self.rrt_angle is not None:
            steering_angle_output = self.rrt_angle
            chosen_source = "rrt"

        # 우선순위 3: 장애물이 없으면 GPS 조향각 사용
        elif self.obstacle_exists is False and self.gps_angle is not None:
            steering_angle_output = self.gps_angle
            chosen_source = "gps"

        # 우선순위 4: (필요시) 터널 조향각 사용
        # elif self.tunnel_angle is not None:
        #     steering_angle_output = self.tunnel_angle
        #     chosen_source = "tunnel"

        # 최종 판단된 조향각에 따라 상태 업데이트 및 발행
        if steering_angle_output is not None:
            self.pub_steering.publish(Float32(data=steering_angle_output))
            self.current_steering_angle = steering_angle_output
            self.steering_source_valid = True
            rospy.loginfo("Published steering: %.2f deg (Source: %s)", steering_angle_output, chosen_source)
        else:
            self.steering_source_valid = False
            rospy.logwarn("No valid steering source found. Steering angle will not be published.")

    def publish_throttle(self, event):
        final_throttle = 0.0

        if not self.steering_source_valid:
            # 유효한 조향각 소스가 없으면 최소 스로틀 사용
            final_throttle = self.min_throttle
            rospy.logwarn("No valid steering source, setting throttle to min_throttle: %.2f", final_throttle)
        else:
            # 조향각에 따라 스로틀 계산
            abs_steering = abs(self.current_steering_angle)
            throttle_reduction = abs_steering * self.steering_throttle_reduction_factor
            calculated_throttle = self.max_throttle - throttle_reduction
            # 최종 스로틀 값을 min/max 범위 내로 제한 (Clamping)
            final_throttle = max(self.min_throttle, min(self.max_throttle, calculated_throttle))
            rospy.loginfo("Current steering: %.2f deg, Throttle reduction: %.2f, Published throttle: %.2f",
                          self.current_steering_angle, throttle_reduction, final_throttle)

        self.pub_throttle.publish(Float32(data=final_throttle))

    def timer_callback(self, event):
        """타이머에 맞춰 조향각과 쓰로틀을 주기적으로 발행"""
        self.publish_steering(event)  # 먼저 조향각 결정
        self.publish_throttle(event)  # 결정된 조향각 상태에 따라 쓰로틀 계산

if __name__ == '__main__':
    rospy.init_node("judgement_node")
    judgement_system = Judgement()
    rospy.spin()