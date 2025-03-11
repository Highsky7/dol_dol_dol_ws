#!/usr/bin/env python3
import rospy
import math
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler

class PurePursuit:
    def __init__(self):
        rospy.init_node("purepursuit", anonymous=True)
        
        # 파라미터: lookahead distance와 차량 휠베이스 (단위: 미터)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 2.0)
        self.wheelbase = rospy.get_param("~wheelbase", 0.75)
        
        # ROI Marker를 구독하여 최신 데이터를 저장
        rospy.Subscriber("roi_path_marker", Marker, self.roi_callback)
        
        # 조향각을 publish할 토픽 (Float32, 단위: degree)
        self.steer_pub = rospy.Publisher("/auto_steer_angle_cone", Float32, queue_size=10)
        
        # Lookahead point Marker를 publish할 토픽
        self.lookahead_pub = rospy.Publisher("lookahead_marker", Marker, queue_size=10)
        
        self.latest_roi_marker = None
        
        # 타이머 콜백: 10Hz (0.1초 간격)
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        rospy.loginfo("PurePursuit 노드 시작 (lookahead_distance: %.2f m, wheelbase: %.2f m)", 
                      self.lookahead_distance, self.wheelbase)
        rospy.spin()
    
    def roi_callback(self, marker_msg):
        # 최신 ROI Marker 데이터 저장
        self.latest_roi_marker = marker_msg
        
    def timer_callback(self, event):
        if self.latest_roi_marker is None or not self.latest_roi_marker.points:
            rospy.logwarn_throttle(5, "ROI Marker 데이터가 없습니다.")
            return
        
        # 차량 좌표계에서, x > 0인 점들만 필터링 (전방)
        valid_points = [pt for pt in self.latest_roi_marker.points if pt.x > 0]
        if not valid_points:
            rospy.logwarn_throttle(5, "전방 ROI 점이 없습니다.")
            return
        
        # lookahead_distance 이상인 첫 번째 점을 선택 (거리 기준 오름차순 정렬)
        valid_points.sort(key=lambda pt: math.hypot(pt.x, pt.y))
        lookahead_pt = None
        for pt in valid_points:
            if math.hypot(pt.x, pt.y) >= self.lookahead_distance:
                lookahead_pt = pt
                break
        if lookahead_pt is None:
            # 조건 만족하는 점이 없으면 가장 먼 점을 사용
            lookahead_pt = valid_points[-1]
        dist = math.hypot(lookahead_pt.x, lookahead_pt.y)
        
        # 차량 좌표계에서는 전방이 x축이므로, alpha = arctan2(y, x)
        alpha = math.atan2(lookahead_pt.y, lookahead_pt.x)
        # Pure pursuit 조향각 공식: δ = arctan( 2L sin(α) / d )
        steer_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), dist)
        
        # 라디안으로 계산된 조향각을 degree로 변환
        steer_angle_deg = math.degrees(steer_angle)
        
        # 좌회전이면 양수, 우회전이면 음수로 degree 단위로 publish
        self.steer_pub.publish(Float32(data=steer_angle_deg))
        rospy.loginfo_throttle(1, "Lookahead: (%.2f, %.2f), dist: %.2f m, α: %.2f deg, steer: %.2f deg",
                                lookahead_pt.x, lookahead_pt.y, dist,
                                math.degrees(alpha), steer_angle_deg)
        
        # Lookahead point Marker (SPHERE)
        lookahead_marker = Marker()
        lookahead_marker.header = Header(stamp=rospy.Time.now(), frame_id="vehicle")
        lookahead_marker.ns = "purepursuit"
        lookahead_marker.id = 0
        lookahead_marker.type = Marker.SPHERE
        lookahead_marker.action = Marker.ADD
        lookahead_marker.scale.x = 0.5
        lookahead_marker.scale.y = 0.5
        lookahead_marker.scale.z = 0.5
        lookahead_marker.color.r = 0.0
        lookahead_marker.color.g = 1.0
        lookahead_marker.color.b = 1.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position = lookahead_pt
        lookahead_marker.pose.orientation.w = 1.0
        self.lookahead_pub.publish(lookahead_marker)

if __name__ == '__main__':
    try:
        PurePursuit()
    except rospy.ROSInterruptException:
        pass
