#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

def euclidean_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

class ROIPathPublisher:
    def __init__(self):
        rospy.init_node("roi_path_publisher", anonymous=True)
        
        # ROI arc length threshold (미터, 기본 5m; rosparam으로 조정 가능)
        self.roi_arc_length = rospy.get_param("~roi_arc_length", 5.0)
        
        # 구독: 전체 경로 (/resampled_path, reference frame) 및 차량 위치 (/local_xy, reference frame)
        rospy.Subscriber("resampled_path", Path, self.path_callback)
        rospy.Subscriber("local_xy", PoseStamped, self.vehicle_pose_callback)
        
        # TF 변환: reference → vehicle
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # ROI Marker를 발행 (vehicle frame)
        self.marker_pub = rospy.Publisher("roi_path_marker", Marker, queue_size=1)
        
        self.full_path = None    # 전체 경로 (Path, reference)
        self.vehicle_pose = None  # 차량 위치 (PoseStamped, reference)
        
        # 0.5초마다 ROI 업데이트
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        rospy.loginfo("ROIPathPublisher 노드 시작: 차량 전방에서 누적 arc length %.1f m 이내 ROI", self.roi_arc_length)
        rospy.spin()
    
    def path_callback(self, msg):
        self.full_path = msg
    
    def vehicle_pose_callback(self, msg):
        self.vehicle_pose = msg
    
    def timer_callback(self, event):
        if self.full_path is None or self.vehicle_pose is None:
            rospy.logwarn("전체 경로 또는 차량 위치 정보를 아직 받지 못했습니다.")
            return
        
        # TF 변환: reference → vehicle
        try:
            transform = self.tf_buffer.lookup_transform("vehicle", "reference", rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("TF lookup 실패: %s", e)
            return
        
        # 전체 경로의 각 점을 vehicle frame으로 변환
        transformed_points = []
        for pose in self.full_path.poses:
            pt_stamped = PointStamped()
            pt_stamped.header = self.full_path.header
            pt_stamped.point = pose.pose.position
            try:
                pt_vehicle = tf2_geometry_msgs.do_transform_point(pt_stamped, transform).point
                transformed_points.append(pt_vehicle)
            except Exception as e:
                rospy.logwarn("Point transform 실패: %s", e)
                continue
        
        if not transformed_points:
            rospy.logwarn("TF 변환된 경로 점이 없습니다.")
            return
        
        # 차량의 vehicle frame에서의 위치는 원점 (0,0)
        # 전체 경로에서 원점과의 유클리드 거리가 가장 짧은 점의 인덱스를 찾습니다.
        min_idx = 0
        min_dist = float('inf')
        for i, pt in enumerate(transformed_points):
            d = math.hypot(pt.x, pt.y)
            if d < min_dist:
                min_dist = d
                min_idx = i
        
        # min_idx부터 순서대로 누적 arc length 계산 (전방 조건: x > 0)
        roi_points = []
        cumulative_length = 0.0
        # 시작점이 전방인지 확인
        start_pt = transformed_points[min_idx]
        if start_pt.x <= 0:
            rospy.logwarn("가장 가까운 점이 차량 전방에 있지 않습니다. (x=%.2f)", start_pt.x)
            return
        
        roi_points.append(start_pt)
        prev_pt = start_pt
        for pt in transformed_points[min_idx+1:]:
            if pt.x <= 0:  # 전방 조건
                break
            seg_length = math.hypot(pt.x - prev_pt.x, pt.y - prev_pt.y)
            cumulative_length += seg_length
            if cumulative_length <= self.roi_arc_length:
                roi_points.append(pt)
            else:
                break
            prev_pt = pt
        
        if not roi_points:
            rospy.logwarn("ROI에 포함되는 점이 없습니다.")
            return
        
        # Marker 생성 및 publish (vehicle frame)
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id="vehicle")
        marker.ns = "roi_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = roi_points
        
        self.marker_pub.publish(marker)
        rospy.loginfo("Published ROI path with %d points, cumulative arc length: %.2f m", len(roi_points), cumulative_length)

if __name__ == '__main__':
    try:
        ROIPathPublisher()
    except rospy.ROSInterruptException:
        pass
