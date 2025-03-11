#!/usr/bin/env python3
import rospy
import math
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

class ROIPathPublisher:
    def __init__(self):
        rospy.init_node("roi_path_publisher", anonymous=True)
        
        # ROI 파라미터 (기본값: 반경 20m)
        self.roi_radius = rospy.get_param("~roi_radius", 10.0)
        
        # /resampled_path (전체 경로, reference frame) 구독
        rospy.Subscriber("resampled_path", Path, self.path_callback)
        
        # TF 변환을 위한 buffer 및 listener (reference -> vehicle)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 결과를 Marker로 발행
        self.marker_pub = rospy.Publisher("roi_path_marker", Marker, queue_size=1)
        
        self.full_path = None
        
        # 1초마다 timer 콜백
        rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        rospy.loginfo("ROIPathPublisher 노드 시작 (경로 진행 방향 기반).")
        rospy.spin()

    def path_callback(self, msg):
        self.full_path = msg

    def timer_callback(self, event):
        if self.full_path is None:
            rospy.logwarn("전체 경로를 아직 받지 못했습니다.")
            return
        
        # reference -> vehicle 변환을 lookup
        try:
            transform = self.tf_buffer.lookup_transform(
                "vehicle",    # target_frame
                "reference",  # source_frame
                rospy.Time(0),
                rospy.Duration(1.0)
            )
        except Exception as e:
            rospy.logwarn("TF lookup 실패: %s", e)
            return
        
        # 결과를 LINE_STRIP Marker로 생성
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
        
        # 구간별로 점들을 변환하고, ROI 조건(전방 + 반경) + 진행 방향 체크
        path_poses = self.full_path.poses
        for i in range(len(path_poses) - 1):
            p_i = path_poses[i].pose.position
            p_i1 = path_poses[i+1].pose.position
            
            # PointStamped로 변환
            pt_i_stamped = PointStamped()
            pt_i_stamped.header = path_poses[i].header
            pt_i_stamped.point = p_i
            
            pt_i1_stamped = PointStamped()
            pt_i1_stamped.header = path_poses[i+1].header
            pt_i1_stamped.point = p_i1
            
            try:
                # reference -> vehicle 변환
                pt_i_v = tf2_geometry_msgs.do_transform_point(pt_i_stamped, transform).point
                pt_i1_v = tf2_geometry_msgs.do_transform_point(pt_i1_stamped, transform).point
            except Exception as e:
                rospy.logwarn("Point transform 실패: %s", e)
                continue
            
            # 구간 벡터 (vehicle frame)
            dx = pt_i1_v.x - pt_i_v.x
            dy = pt_i1_v.y - pt_i_v.y
            
            # 전방인지: dx > 0 (즉, 구간 진행 방향이 차량 진행 방향과 대체로 같음)
            # 추가로, 구간 중간점이 차량에서 roi_radius 이내인지 확인
            mid_x = (pt_i_v.x + pt_i1_v.x) / 2.0
            mid_y = (pt_i_v.y + pt_i1_v.y) / 2.0
            dist_mid = math.hypot(mid_x, mid_y)
            
            if dx > 0 and dist_mid <= self.roi_radius:
                # 이 구간을 ROI로 인정 -> LINE_STRIP에 추가
                # (연속 표시 위해 i, i+1 점을 모두 추가)
                marker.points.append(Point(x=pt_i_v.x, y=pt_i_v.y, z=pt_i_v.z))
                marker.points.append(Point(x=pt_i1_v.x, y=pt_i1_v.y, z=pt_i1_v.z))
        
        self.marker_pub.publish(marker)
        rospy.loginfo("Published ROI path with %d points", len(marker.points))

if __name__ == "__main__":
    try:
        ROIPathPublisher()
    except rospy.ROSInterruptException:
        pass
