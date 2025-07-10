#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32, Float64, Bool, ColorRGBA
from collections import deque
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class Judgement:
    def __init__(self):
        rospy.loginfo("Initializing Judgement Node as Geoffrey Hinton would...")

        # --- 구독자 설정 ---
        self.sub_lane = rospy.Subscriber('auto_steer_angle_lane', Float32, self.lane_callback)
        self.sub_lane_status = rospy.Subscriber('lane_detection_status', Bool, self.lane_status_callback)
        self.sub_obstacle_existence = rospy.Subscriber('/obstacle_existence', Bool, self.obstacle_callback)
        self.sub_rrt = rospy.Subscriber('/auto_steer_angle_rrt', Float32, self.rrt_callback)
        self.sub_gps = rospy.Subscriber('/auto_steer_angle_gps', Float32, self.gps_callback)
        self.sub_dynamic_obstacle = rospy.Subscriber('dynamic_obstacle', Bool, self.dynamic_obstacle_callback)
        self.sub_linear_vel = rospy.Subscriber('/linear_velocity', Float64, self.linear_vel_callback)
        self.sub_forced_rrt = rospy.Subscriber('/forced_rrt', Bool, self.forced_rrt_callback)  # ← NEW: RRT 강제 활성화 토픽 구독

        # --- 퍼블리셔 설정 ---
        self.pub_steering = rospy.Publisher("steering_angle", Float32, queue_size=10)
        self.pub_throttle = rospy.Publisher("auto_throttle", Float32, queue_size=10)
        self.pub_marker = rospy.Publisher("steering_angle_marker", Marker, queue_size=10)

        # --- 변수 초기화 ---
        self.lane_angle = None
        self.lane_detected = False
        self.obstacle_exists = False
        self.rrt_angle = None
        self.gps_angle = None
        self.force_rrt_active = False  # ← NEW: RRT 강제 활성화 상태 플래그
        self.current_steering_angle = 0.0
        self.steering_source_valid = False
        self.chosen_source = "None"

        # 선속도 관련
        self.linear_velocity = 0.0
        self.MAX_LINEAR_VEL = rospy.get_param("~max_linear_velocity", 2.1)  # [m/s]
        rospy.loginfo("Max linear velocity threshold set to %.2f m/s", self.MAX_LINEAR_VEL)

        # --- RViz 마커 색상 정의 ---
        self.color_lane = ColorRGBA(0.0, 1.0, 0.0, 1.0)        # Green
        self.color_rrt = ColorRGBA(0.0, 1.0, 1.0, 1.0)        # Cyan
        self.color_gps = ColorRGBA(1.0, 0.0, 1.0, 1.0)        # Magenta
        self.color_forced_rrt = ColorRGBA(1.0, 0.5, 0.0, 1.0) # Orange ← NEW
        self.color_none = ColorRGBA(1.0, 1.0, 1.0, 1.0)       # White

        # --- 동적 장애물 대응 로직 ---
        self.dynamic_obstacle_history = deque(maxlen=5)
        self.is_emergency_stopping = False
        self.emergency_stop_start_time = None
        self.EMERGENCY_STOP_DURATION = rospy.Duration(2.0)

        # --- 속도 계획 파라미터 ---
        self.max_throttle = rospy.get_param("~max_throttle", 0.6)
        self.min_throttle = rospy.get_param("~min_throttle", 0.4)
        self.steering_throttle_reduction_factor = rospy.get_param("~steering_throttle_reduction_factor", 0.01)
        self.CAUTION_THROTTLE = 0.2
        self.EMERGENCY_STOP_THROTTLE = 0.0

        rospy.loginfo(
            "Throttle parameters initialized: max_throttle=%.2f, min_throttle=%.2f, reduction_factor=%.4f",
            self.max_throttle, self.min_throttle, self.steering_throttle_reduction_factor
        )

        # 주기적 실행을 위한 타이머 (10 Hz)
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

    def forced_rrt_callback(self, msg):  # ← NEW: RRT 강제 활성화 콜백 함수
        if self.force_rrt_active != msg.data:
            self.force_rrt_active = msg.data
            rospy.logwarn("Forced RRT Mode Updated: %s", "ACTIVATED" if self.force_rrt_active else "DEACTIVATED")

    def dynamic_obstacle_callback(self, msg):
        self.dynamic_obstacle_history.append(msg.data)
        if not self.is_emergency_stopping:
            true_count = self.dynamic_obstacle_history.count(True)
            if true_count >= 2:
                self.is_emergency_stopping = True
                self.emergency_stop_start_time = rospy.Time.now()
                rospy.logerr(
                    "!!! EMERGENCY STOP TRIGGERED !!! Obstacle detected %d/5 times. "
                    "Stopping for %.1f seconds.",
                    true_count, self.EMERGENCY_STOP_DURATION.to_sec()
                )

    def linear_vel_callback(self, msg):
        self.linear_velocity = msg.data

    # --- 데이터 발행 로직 ---
    def publish_steering(self, event): # ← MODIFIED: 조향각 결정 로직 수정
        steering_angle_output = None
        self.chosen_source = "None"

        # 1순위: RRT 강제 활성화 모드
        if self.force_rrt_active and self.rrt_angle is not None:
            steering_angle_output = self.rrt_angle
            self.chosen_source = "forced_rrt"
        # 2순위: 차선 감지 주행
        elif self.lane_detected and self.lane_angle is not None:
            steering_angle_output = self.lane_angle
            self.chosen_source = "lane"
        # 3순위: 정적 장애물 기반 RRT 주행
        elif self.obstacle_exists and self.rrt_angle is not None:
            steering_angle_output = self.rrt_angle
            self.chosen_source = "rrt"
        # 4순위: GPS 기반 주행
        elif not self.obstacle_exists and self.gps_angle is not None:
            steering_angle_output = self.gps_angle
            self.chosen_source = "gps"

        if steering_angle_output is not None:
            self.pub_steering.publish(Float32(data=steering_angle_output))
            self.current_steering_angle = steering_angle_output
            self.steering_source_valid = True
            rospy.loginfo("Published steering: %.2f deg (Source: %s)",
                          steering_angle_output, self.chosen_source.upper())
        else:
            self.steering_source_valid = False
            self.current_steering_angle = 0.0
            rospy.logwarn("No valid steering source found. Steering angle will not be published.")

    def publish_throttle(self, event):
        final_throttle = 0.0

        if self.is_emergency_stopping:
            if rospy.Time.now() - self.emergency_stop_start_time < self.EMERGENCY_STOP_DURATION:
                final_throttle = self.EMERGENCY_STOP_THROTTLE
                rospy.logwarn("!!! EMERGENCY STOPPING !!! Throttle forced to %.2f.", final_throttle)
            else:
                self.is_emergency_stopping = False
                self.emergency_stop_start_time = None
                self.dynamic_obstacle_history.clear()
                final_throttle = self.min_throttle
        elif True in self.dynamic_obstacle_history:
            final_throttle = self.CAUTION_THROTTLE
            rospy.logwarn("!! Dynamic Obstacle detected. Applying CAUTION throttle: %.2f", final_throttle)
        elif self.linear_velocity > self.MAX_LINEAR_VEL:
            final_throttle = 0.2
            rospy.logwarn(
                "Velocity %.2f m/s > %.2f m/s, limiting throttle to: %.2f",
                self.linear_velocity, self.MAX_LINEAR_VEL, final_throttle
            )
        else:
            if not self.steering_source_valid:
                final_throttle = self.min_throttle
                rospy.logwarn("No valid steering source, setting throttle to min_throttle: %.2f", final_throttle)
            else:
                abs_steering = abs(self.current_steering_angle)
                throttle_reduction = abs_steering * self.steering_throttle_reduction_factor
                calculated_throttle = self.max_throttle - throttle_reduction
                final_throttle = max(self.min_throttle, min(self.max_throttle, calculated_throttle))
                rospy.loginfo(
                    "Current steering: %.2f deg, Throttle reduction: %.2f, Published throttle: %.2f",
                    self.current_steering_angle, throttle_reduction, final_throttle
                )

        self.pub_throttle.publish(Float32(data=final_throttle))

    # ---------------- RViz 마커 관련 함수 ----------------
    def create_debug_marker(self, marker_id, text, position, color):
        marker = Marker()
        marker.header.frame_id = "velodyne"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "steering_sources_debug"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.8
        marker.color = color
        marker.text = text
        marker.lifetime = rospy.Duration(0.5)
        self.pub_marker.publish(marker)

    def publish_main_marker(self, event): # ← MODIFIED: 메인 마커 색상 로직 추가
        marker = Marker()
        marker.header.frame_id = "velodyne"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "chosen_steering_info"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = -8.0
        marker.pose.position.y = 8.0
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 1.6
        marker.text = f"Source: {self.chosen_source.upper()}\nAngle: {self.current_steering_angle:.2f}"

        if self.chosen_source == "lane":
            marker.color = self.color_lane
        elif self.chosen_source == "rrt":
            marker.color = self.color_rrt
        elif self.chosen_source == "forced_rrt": # ← NEW
            marker.color = self.color_forced_rrt
        elif self.chosen_source == "gps":
            marker.color = self.color_gps
        else:
            marker.color = self.color_none
            marker.text = "Source: NONE\nAngle: ---"

        marker.lifetime = rospy.Duration(0.5)
        self.pub_marker.publish(marker)

    def publish_debug_markers(self, event): # ← MODIFIED: 디버그 마커 로직 안정화
        ordered_sources = ['lane', 'rrt', 'gps']
        active_source = self.chosen_source

        # 'forced_rrt' 상태일 때 'rrt'를 활성 소스로 간주하여 시각화 순서 정렬
        display_source_for_ordering = 'rrt' if active_source == 'forced_rrt' else active_source

        if display_source_for_ordering in ordered_sources:
            ordered_sources.remove(display_source_for_ordering)
            ordered_sources.insert(0, display_source_for_ordering)

        # 'forced_rrt' 활성화 시 RRT 디버그 마커 색상 변경
        rrt_color = self.color_forced_rrt if self.chosen_source == 'forced_rrt' else self.color_rrt

        source_data_map = {
            'lane': {'angle': self.lane_angle, 'id': 1, 'color': self.color_lane},
            'rrt': {'angle': self.rrt_angle, 'id': 2, 'color': rrt_color},
            'gps': {'angle': self.gps_angle, 'id': 3, 'color': self.color_gps}
        }

        positions_z = [1.5, 0.5, -0.5]

        for i, source_name in enumerate(ordered_sources):
            data = source_data_map.get(source_name)
            if data:
                angle = data.get('angle')
                if angle is not None:
                    position = Point(x=-4.0, y=8.0, z=positions_z[i])
                    text = f"{source_name.upper()}\n{angle:.2f}"
                    self.create_debug_marker(data['id'], text, position, data['color'])

    # ---------------- Timer ----------------
    def timer_callback(self, event):
        self.publish_steering(event)
        self.publish_throttle(event)
        self.publish_main_marker(event)
        self.publish_debug_markers(event)


if __name__ == '__main__':
    try:
        rospy.init_node("judgement_node")
        judgement_system = Judgement()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Judgement node shut down.")