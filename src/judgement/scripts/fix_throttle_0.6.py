#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32, Bool, ColorRGBA
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
        self.current_steering_angle = 0.0
        self.steering_source_valid = False
        self.chosen_source = "None"

        # --- RViz 마커 색상 정의 ---
        self.color_lane = ColorRGBA(0.0, 1.0, 0.0, 1.0)
        self.color_rrt = ColorRGBA(0.0, 1.0, 1.0, 1.0)
        self.color_gps = ColorRGBA(1.0, 0.0, 1.0, 1.0)
        self.color_none = ColorRGBA(1.0, 1.0, 1.0, 1.0)

        # --- 동적 장애물 대응 로직 (스로틀과 무관하게 유지) ---
        self.dynamic_obstacle_history = deque(maxlen=5)
        self.is_emergency_stopping = False
        self.emergency_stop_start_time = None
        self.EMERGENCY_STOP_DURATION = rospy.Duration(5.0)

        # --- 속도 계획 파라미터 (더 이상 사용되지 않음) ---
        # self.max_throttle = rospy.get_param("~max_throttle", 0.6)
        # self.min_throttle = rospy.get_param("~min_throttle", 0.4)
        # self.steering_throttle_reduction_factor = rospy.get_param("~steering_throttle_reduction_factor", 0.02)
        # self.CAUTION_THROTTLE = 0.2
        # self.EMERGENCY_STOP_THROTTLE = 0.0

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
        # 이 로직은 스로틀에는 영향을 주지 않지만, 다른 잠재적 용도를 위해 유지합니다.
        self.dynamic_obstacle_history.append(msg.data)
        if not self.is_emergency_stopping:
            true_count = self.dynamic_obstacle_history.count(True)
            if true_count >= 2:
                # 긴급 정지 상태는 활성화되지만, publish_throttle 함수에서 이를 사용하지 않습니다.
                self.is_emergency_stopping = True
                self.emergency_stop_start_time = rospy.Time.now()
                rospy.logerr("!!! EMERGENCY STOP TRIGGERED !!! Obstacle detected %d/5 times. Stopping for %.1f seconds.",
                             true_count, self.EMERGENCY_STOP_DURATION.to_sec())

    # --- 데이터 발행 로직 ---
    def publish_steering(self, event):
        steering_angle_output = None
        self.chosen_source = "None"

        if self.lane_detected and self.lane_angle is not None:
            steering_angle_output = self.lane_angle
            self.chosen_source = "lane"
        elif self.obstacle_exists and self.rrt_angle is not None:
            steering_angle_output = self.rrt_angle
            self.chosen_source = "rrt"
        elif not self.obstacle_exists and self.gps_angle is not None:
            steering_angle_output = self.gps_angle
            self.chosen_source = "gps"

        if steering_angle_output is not None:
            self.pub_steering.publish(Float32(data=steering_angle_output))
            self.current_steering_angle = steering_angle_output
            self.steering_source_valid = True
            rospy.loginfo("Published steering: %.2f deg (Source: %s)", steering_angle_output, self.chosen_source)
        else:
            self.steering_source_valid = False
            self.current_steering_angle = 0.0
            rospy.logwarn("No valid steering source found. Steering angle will not be published.")

    # <--- [수정된 부분] ---
    def publish_throttle(self, event):
        """
        요청에 따라 항상 0.6의 고정된 스로틀 값을 발행합니다.
        """
        fixed_throttle = 0.6
        self.pub_throttle.publish(Float32(data=fixed_throttle))
        # rospy.loginfo("Publishing fixed throttle: %.2f", fixed_throttle) # 필요시 주석 해제하여 로그 확인

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

    def publish_main_marker(self, event):
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
        elif self.chosen_source == "gps":
            marker.color = self.color_gps
        else:
            marker.color = self.color_none
            marker.text = "Source: NONE\nAngle: ---"
        
        marker.lifetime = rospy.Duration(0.5)
        self.pub_marker.publish(marker)

    def publish_debug_markers(self, event):
        ordered_sources = ['lane', 'rrt', 'gps']
        
        active_source = self.chosen_source
        if active_source in ordered_sources:
            ordered_sources.remove(active_source)
            ordered_sources.insert(0, active_source)

        source_data_map = {
            'lane': {'angle': self.lane_angle, 'id': 1, 'color': self.color_lane},
            'rrt': {'angle': self.rrt_angle, 'id': 2, 'color': self.color_rrt},
            'gps': {'angle': self.gps_angle, 'id': 3, 'color': self.color_gps}
        }
        
        positions_z = [1.5, 0.5, -0.5] 

        for i, source_name in enumerate(ordered_sources):
            data = source_data_map.get(source_name)
            angle = data.get('angle')
            
            if angle is not None:
                position = Point(x=-4.0, y=8.0, z=positions_z[i])
                text = f"{source_name.upper()}\n{angle:.2f}"
                self.create_debug_marker(data['id'], text, position, data['color'])

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