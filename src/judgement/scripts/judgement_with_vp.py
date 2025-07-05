#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32, Bool
from collections import deque # 최근 데이터를 저장하기 위해 deque를 임포트합니다.

class Judgement:
    def __init__(self):
        rospy.loginfo("Initializing Judgement Node as Geoffrey Hinton would...")

        # --- 구독자 설정: ROS 토픽에서 데이터를 수신 ---
        self.sub_lane = rospy.Subscriber('auto_steer_angle_lane', Float32, self.lane_callback)
        self.sub_lane_status = rospy.Subscriber('lane_detection_status', Bool, self.lane_status_callback)
        self.sub_obstacle_existence = rospy.Subscriber('/obstacle_existence', Bool, self.obstacle_callback)
        self.sub_rrt = rospy.Subscriber('/auto_steer_angle_rrt', Float32, self.rrt_callback)
        self.sub_gps = rospy.Subscriber('/auto_steer_angle_gps', Float32, self.gps_callback)
        self.sub_dynamic_obstacle = rospy.Subscriber('dynamic_obstacle', Bool, self.dynamic_obstacle_callback)

        # --- 퍼블리셔 설정 ---
        self.pub_steering = rospy.Publisher("steering_angle", Float32, queue_size=10)
        self.pub_throttle = rospy.Publisher("auto_throttle", Float32, queue_size=10)

        # --- 변수 초기화 ---
        self.lane_angle = None
        self.lane_detected = False
        self.obstacle_exists = False
        self.rrt_angle = None
        self.gps_angle = None
        self.current_steering_angle = 0.0
        self.steering_source_valid = False

        # --- 동적 장애물 대응 로직을 위한 변수 ---
        # 최근 5개의 dynamic_obstacle 메시지를 저장할 deque
        self.dynamic_obstacle_history = deque(maxlen=5)
        # 긴급 정지 상태 플래그
        self.is_emergency_stopping = False
        # 긴급 정지 시작 시간을 기록할 변수
        self.emergency_stop_start_time = None
        # 긴급 정지 지속 시간 (5초)
        self.EMERGENCY_STOP_DURATION = rospy.Duration(5.0)

        # --- 속도 계획 파라미터 ---
        self.max_throttle = rospy.get_param("~max_throttle", 0.5)
        self.min_throttle = rospy.get_param("~min_throttle", 0.3)
        self.steering_throttle_reduction_factor = rospy.get_param("~steering_throttle_reduction_factor", 0.01)
        self.CAUTION_THROTTLE = 0.2 # 주의 감속 스로틀
        self.EMERGENCY_STOP_THROTTLE = 0.0 # 긴급 정지 스로틀

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

    def dynamic_obstacle_callback(self, msg):
        """동적 장애물 데이터를 수신하고, 긴급 정지 조건을 확인하는 콜백 함수"""
        # 수신된 데이터를 deque의 오른쪽에 추가 (오래된 데이터는 자동으로 왼쪽에서 제거됨)
        self.dynamic_obstacle_history.append(msg.data)
        
        # 현재 긴급 정지 상태가 아닐 때만 새로운 긴급 정지 조건을 검사
        if not self.is_emergency_stopping:
            # deque에 저장된 True의 개수를 셈
            true_count = self.dynamic_obstacle_history.count(True)
            
            # 5번의 메시지 중 2번 이상 True가 감지되면 긴급 정지 상태로 전환
            if true_count >= 2:
                self.is_emergency_stopping = True
                self.emergency_stop_start_time = rospy.Time.now()
                rospy.logerr("!!! EMERGENCY STOP TRIGGERED !!! Obstacle detected %d/5 times. Stopping for %.1f seconds.",
                             true_count, self.EMERGENCY_STOP_DURATION.to_sec())

    # --- 데이터 발행 로직 ---
    def publish_steering(self, event):
        steering_angle_output = None
        chosen_source = "None"

        if self.lane_detected and self.lane_angle is not None:
            steering_angle_output = self.lane_angle
            chosen_source = "lane"
        elif self.obstacle_exists and self.rrt_angle is not None:
            steering_angle_output = self.rrt_angle
            chosen_source = "rrt"
        elif not self.obstacle_exists and self.gps_angle is not None:
            steering_angle_output = self.gps_angle
            chosen_source = "gps"

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

        # 우선순위 1: 긴급 정지 상태 확인
        if self.is_emergency_stopping:
            # 정지 시작 후 5초가 지났는지 확인
            if rospy.Time.now() - self.emergency_stop_start_time < self.EMERGENCY_STOP_DURATION:
                # 5초가 지나지 않았으면 스로틀을 0으로 유지
                final_throttle = self.EMERGENCY_STOP_THROTTLE
                rospy.logwarn("!!! EMERGENCY STOPPING !!! Throttle forced to %.2f.", final_throttle)
            else:
                # 5초가 지났으면 긴급 정지 상태 해제 및 관련 변수 초기화
                rospy.loginfo("Emergency stop duration finished. Resuming normal operation.")
                self.is_emergency_stopping = False
                self.emergency_stop_start_time = None
                self.dynamic_obstacle_history.clear() # 상태 초기화하여 즉시 재진입 방지
                # 상태 해제 후에는 아래의 일반 로직을 따름
                final_throttle = self.min_throttle # 안전을 위해 최소 스로틀로 시작
        
        # 우선순위 2: 주의 감속 상태 확인 (최근 동적 장애물이 한 번이라도 감지된 경우)
        elif True in self.dynamic_obstacle_history:
            final_throttle = self.CAUTION_THROTTLE
            rospy.logwarn("!! Dynamic Obstacle detected. Applying CAUTION throttle: %.2f", final_throttle)

        # 우선순위 3: 일반 주행 상태
        else:
            if not self.steering_source_valid:
                final_throttle = self.min_throttle
                rospy.logwarn("No valid steering source, setting throttle to min_throttle: %.2f", final_throttle)
            else:
                abs_steering = abs(self.current_steering_angle)
                throttle_reduction = abs_steering * self.steering_throttle_reduction_factor
                calculated_throttle = self.max_throttle - throttle_reduction
                final_throttle = max(self.min_throttle, min(self.max_throttle, calculated_throttle))
                rospy.loginfo("Current steering: %.2f deg, Throttle reduction: %.2f, Published throttle: %.2f",
                              self.current_steering_angle, throttle_reduction, final_throttle)

        self.pub_throttle.publish(Float32(data=final_throttle))

    def timer_callback(self, event):
        """타이머에 맞춰 조향각과 쓰로틀을 주기적으로 발행"""
        self.publish_steering(event)
        self.publish_throttle(event)

if __name__ == '__main__':
    try:
        rospy.init_node("judgement_node")
        judgement_system = Judgement()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Judgement node shut down.")