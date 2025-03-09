#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

class GlobalYawEstimator:
    def __init__(self):
        rospy.init_node('global_yaw_estimator', anonymous=True)
        self.prev_x = None
        self.prev_y = None
        
        self.raw_yaw_pub = rospy.Publisher('/raw_global_yaw', Float32, queue_size=10)
        self.filtered_yaw_pub = rospy.Publisher('/global_yaw', Float32, queue_size=10)
        
        self.alpha = rospy.get_param("~alpha", 0.2)
        self.filtered_yaw = None
        
        # 실시간 로컬 좌표 /local_xy를 구독 (PointStamped)
        rospy.Subscriber('/local_xy', PointStamped, self.local_xy_callback)
        rospy.loginfo("GlobalYawEstimator 노드 시작 (구독: /local_xy)")
        rospy.spin()

    def local_xy_callback(self, msg):
        x = msg.point.x
        y = msg.point.y
        if self.prev_x is not None and self.prev_y is not None:
            dx = x - self.prev_x
            dy = y - self.prev_y
            if math.hypot(dx, dy) > 0.01:
                raw_yaw = math.atan2(dy, dx)
                raw_yaw = wrap_angle(raw_yaw)
                self.raw_yaw_pub.publish(Float32(data=raw_yaw))
                if self.filtered_yaw is None:
                    self.filtered_yaw = raw_yaw
                else:
                    diff = wrap_angle(raw_yaw - self.filtered_yaw)
                    self.filtered_yaw = wrap_angle(self.filtered_yaw + self.alpha * diff)
                self.filtered_yaw_pub.publish(Float32(data=self.filtered_yaw))
                rospy.loginfo("Raw Yaw: %.2f deg, Filtered Yaw: %.2f deg",
                              math.degrees(raw_yaw), math.degrees(self.filtered_yaw))
        self.prev_x = x
        self.prev_y = y

if __name__ == '__main__':
    try:
        GlobalYawEstimator()
    except rospy.ROSInterruptException:
        pass
