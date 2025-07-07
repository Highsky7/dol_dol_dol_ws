#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaRRT Pure Pursuit Node (Dynamic Lookahead Ver.)

- /waypoints 토픽(웨이포인트 배열)을 구독하여 B-spline 보간으로 등간격 제어점을 생성합니다.
- /odometry 토픽을 구독하여 차량의 현재 위치(라이다, 후륜축 중심)와 heading을 업데이트합니다.
- /auto_throttle 토픽을 구독하여 현재 스로틀 값에 따라 전방주시거리(Ld)를 동적으로 조절합니다.
- velodyne 좌표계상의 제어점을 차량 좌표계로 변환한 후, pure pursuit 제어를 수행합니다.
- std_msgs/Float32 메시지로 조향각(°)을 publish합니다.
- RViz를 위해 최종 보간 경로(/final_waypoints)와 lookahead point(/lookahead_marker)를 publish합니다.
"""

import rospy
import math
import numpy as np
from vehicle_msgs.msg import Waypoint, WaypointsArray
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Point
from scipy.interpolate import splprep, splev
from tf.transformations import euler_from_quaternion

class MaRRTPurePursuit:
    def __init__(self):
        rospy.init_node('ma_rrt_purepursuit', anonymous=False)
        
        # --- 파라미터 ---
        self.wheelbase = rospy.get_param('~wheelbase', 0.75)
        
        # ======================= [핵심 수정 1: 동적 전방주시거리 파라미터] =======================
        # 고정된 Ld 파라미터를 제거하고, 동적 Ld 계산을 위한 파라미터를 추가합니다.
        # self.Ld = rospy.get_param('~lookahead_distance', 2.5) # 기존 고정값 Ld
        self.THROTTLE_MIN = rospy.get_param('~throttle_min', 0.4)
        self.THROTTLE_MAX = rospy.get_param('~throttle_max', 0.6)
        self.MIN_LOOKAHEAD_DISTANCE = rospy.get_param('~min_lookahead_distance', 3.5)
        self.MAX_LOOKAHEAD_DISTANCE = rospy.get_param('~max_lookahead_distance', 3.5)
        
        # 현재 스로틀 값을 저장할 변수. 안전을 위해 최소값으로 초기화
        self.current_throttle = self.THROTTLE_MIN
        # ========================================================================================

        # 차량의 현재 위치 (velodyne 좌표계, 라이다/후륜축 중심)
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_yaw = 0.0  # 라디안
        
        # B-spline 보간을 통해 생성된 등간격 제어점
        self.control_points = []  
        
        # --- 퍼블리셔 ---
        self.cmd_pub = rospy.Publisher("/auto_steer_angle_rrt", Float32, queue_size=10)
        self.lookahead_rrt_pub = rospy.Publisher("/lookahead_rrt_marker", Marker, queue_size=10)
        self.final_waypoints_pub = rospy.Publisher("/final_waypoints", Path, queue_size=10)
        
        # --- 구독자 ---
        self.waypoints_sub = rospy.Subscriber("/waypoints", WaypointsArray, self.waypoints_callback)
        self.odom_sub = rospy.Subscriber("/odometry", Odometry, self.odometry_callback)
        
        # ======================= [핵심 수정 2: 스로틀 토픽 구독자 추가] =======================
        self.throttle_sub = rospy.Subscriber("/auto_throttle", Float32, self.throttle_callback)
        # ========================================================================================

        rospy.loginfo("MaRRT Pure Pursuit (Dynamic Ld) 노드가 초기화되었습니다.")
    
    # ========================== [핵심 수정 3: 스로틀 콜백 함수 추가] ==========================
    def throttle_callback(self, msg):
        """ /auto_throttle 토픽을 구독하여 self.current_throttle 값을 업데이트합니다. """
        self.current_throttle = np.clip(msg.data, self.THROTTLE_MIN, self.THROTTLE_MAX)
    # ========================================================================================

    def waypoints_callback(self, msg):
        waypoints = []
        for wp in msg.waypoints:
            waypoints.append((wp.x, wp.y))
        if len(waypoints) < 2:
            return
        
        waypoints_np = np.array(waypoints)
        x, y = waypoints_np[:, 0], waypoints_np[:, 1]
        
        try:
            tck, u = splprep([x, y], s=5.0, k=min(3, len(waypoints)-1))
        except Exception as e:
            rospy.logerr("splprep 실행 오류: {}".format(e))
            return
        
        u_fine = np.linspace(0, 1, 100)
        x_fine, y_fine = splev(u_fine, tck)
        
        arc_lengths = [0]
        for i in range(1, len(x_fine)):
            arc_lengths.append(arc_lengths[-1] + math.hypot(x_fine[i] - x_fine[i-1], y_fine[i] - y_fine[i-1]))
        
        total_length = arc_lengths[-1]
        sampling_distance = 0.1
        desired_lengths = np.arange(0, total_length, sampling_distance)
        if desired_lengths.size == 0 or desired_lengths[-1] < total_length:
            desired_lengths = np.append(desired_lengths, total_length)
        
        u_equally = np.interp(desired_lengths, arc_lengths, u_fine)
        x_eq, y_eq = splev(u_equally, tck)
        self.control_points = list(zip(x_eq, y_eq))
        
        final_path = Path()
        final_path.header.frame_id = msg.header.frame_id
        final_path.header.stamp = rospy.Time.now()
        for pt in self.control_points:
            ps = PoseStamped()
            ps.header = final_path.header
            ps.pose.position.x, ps.pose.position.y = pt[0], pt[1]
            ps.pose.orientation.w = 1.0
            final_path.poses.append(ps)
        self.final_waypoints_pub.publish(final_path)
        
        self.do_pure_pursuit()
    
    def odometry_callback(self, msg):
        self.vehicle_x, self.vehicle_y = 0.0, 0.0
        q = msg.pose.pose.orientation
        (_, _, self.vehicle_yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        self.do_pure_pursuit()
    
    def do_pure_pursuit(self):
        if not self.control_points:
            return
        
        # ======================= [핵심 수정 4: 동적 전방주시거리(Ld) 계산] =======================
        throttle_range = self.THROTTLE_MAX - self.THROTTLE_MIN
        if throttle_range <= 0:
            normalized_throttle = 0.0
        else:
            normalized_throttle = (self.current_throttle - self.THROTTLE_MIN) / throttle_range
        
        Ld = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE) * normalized_throttle
        # ========================================================================================

        transformed_points = []
        for pt in self.control_points:
            dx, dy = pt[0] - self.vehicle_x, pt[1] - self.vehicle_y
            x_rel = math.cos(self.vehicle_yaw) * dx + math.sin(self.vehicle_yaw) * dy
            y_rel = -math.sin(self.vehicle_yaw) * dx + math.cos(self.vehicle_yaw) * dy
            transformed_points.append((x_rel, y_rel))
        
        best_diff, lookahead_point = float('inf'), None
        for x_rel, y_rel in transformed_points:
            if x_rel < 0:
                continue
            dist = math.hypot(x_rel, y_rel)
            diff = abs(dist - Ld)
            if diff < best_diff:
                best_diff, lookahead_point = diff, (x_rel, y_rel)
        
        if lookahead_point is None:
            rospy.logwarn("유효한 lookahead point가 없습니다. 정지합니다.")
            self.publish_stop_cmd()
            return
        
        last_pt = transformed_points[-1]
        if math.hypot(last_pt[0], last_pt[1]) < 1.0:
            rospy.loginfo("경로의 끝에 도달했습니다. 정지합니다.")
            self.publish_stop_cmd()
            return
        
        alpha = math.atan2(lookahead_point[1], lookahead_point[0])
        steer_rad = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
        steer_deg = -math.degrees(steer_rad)
        
        rospy.loginfo("Pure Pursuit: Ld={:.2f}m, Lookahead=({:.2f}, {:.2f}), α={:.2f}°, 조향각={:.2f}°".format(
            Ld, lookahead_point[0], lookahead_point[1], math.degrees(alpha), steer_deg))
        
        angle_msg = Float32(data=steer_deg)
        self.cmd_pub.publish(angle_msg)
        self.publish_lookahead_marker(lookahead_point)
    
    def publish_stop_cmd(self):
        self.cmd_pub.publish(Float32(data=0.0))
    
    def publish_lookahead_marker(self, lookahead_point):
        x_rel, y_rel = lookahead_point
        x_lookahead = self.vehicle_x + math.cos(self.vehicle_yaw)*x_rel - math.sin(self.vehicle_yaw)*y_rel
        y_lookahead = self.vehicle_y + math.sin(self.vehicle_yaw)*x_rel + math.cos(self.vehicle_yaw)*y_rel
        
        marker = Marker()
        marker.header.frame_id = "velodyne"
        marker.header.stamp = rospy.Time.now()
        marker.ns, marker.id, marker.type, marker.action = "lookahead", 0, Marker.CUBE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y = x_lookahead, y_lookahead
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = 0.5, 0.5, 0.5
        marker.color.g, marker.color.b, marker.color.a = 1.0, 1.0, 1.0
        marker.lifetime = rospy.Duration(0.3)
        self.lookahead_rrt_pub.publish(marker)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = MaRRTPurePursuit()
        node.run()
    except rospy.ROSInterruptException:
        pass