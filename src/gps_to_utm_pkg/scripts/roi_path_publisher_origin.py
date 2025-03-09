#!/usr/bin/env python3
import rospy
import math
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

class ROIPathPublisher:
    def __init__(self):
        rospy.init_node("roi_path_publisher", anonymous=True)
        
        # ROI 설정: 차량 기준 반원 ROI, 기본 반지름 20m (rosparam으로 조정 가능)
        self.roi_radius = rospy.get_param("~roi_radius", 5.0)
        
        # /resampled_path (전체 경로, reference frame) 구독
        rospy.Subscriber("resampled_path", Path, self.full_path_callback)
        
        # TF 변환을 위한 buffer 및 listener 생성
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # ROI 경로를 Marker로 publish
        self.marker_pub = rospy.Publisher("roi_path_marker", Marker, queue_size=1)
        
        self.full_path = None  # 전체 경로 (Path 메시지, reference frame)
        
        rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        rospy.loginfo("ROIPathPublisher 노드 시작 (TF 사용).")
        rospy.spin()

    def full_path_callback(self, msg):
        self.full_path = msg

    def timer_callback(self, event):
        if self.full_path is None:
            rospy.logwarn("전체 경로 메시지를 아직 받지 못했습니다.")
            return
        
        try:
            # "vehicle" frame에서 "reference" frame으로의 변환을 구함
            # lookup_transform(target_frame, source_frame, time)
            transform = self.tf_buffer.lookup_transform("vehicle", "reference", rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("TF lookup 실패: %s", e)
            return
        
        roi_points = []
        # 전체 경로의 각 Pose를 vehicle 좌표계로 변환하고 ROI 조건 적용
        for pose in self.full_path.poses:
            pt_stamped = PointStamped()
            pt_stamped.header = pose.header
            pt_stamped.point = pose.pose.position
            try:
                pt_vehicle = tf2_geometry_msgs.do_transform_point(pt_stamped, transform)
                x_v = pt_vehicle.point.x
                y_v = pt_vehicle.point.y
                dist = math.hypot(x_v, y_v)
                # ROI 조건: 전방 (x_v > 0) 및 차량 기준 거리 ≤ roi_radius
                if x_v > 0 and dist <= self.roi_radius:
                    roi_points.append(Point(x=x_v, y=y_v, z=pt_vehicle.point.z))
            except Exception as e:
                rospy.logwarn("Point transform 실패: %s", e)
                continue
        
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id="vehicle")
        marker.ns = "roi_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2  # 선 두께
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = roi_points
        
        self.marker_pub.publish(marker)
        rospy.loginfo("Published ROI path with %d points", len(roi_points))

if __name__ == "__main__":
    try:
        ROIPathPublisher()
    except rospy.ROSInterruptException:
        pass
