#!/usr/bin/env python3
import rospy
import tf2_ros
import math
from geometry_msgs.msg import TransformStamped, PointStamped
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float32

class VehicleTFBroadcaster:
    def __init__(self):
        rospy.init_node('vehicle_tf_broadcaster', anonymous=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_z = 0.0
        self.global_yaw = 0.0  # global yaw, 업데이트는 global_yaw_estimator에서 받음

        rospy.Subscriber("utm_xy", PointStamped, self.utm_callback)
        rospy.Subscriber("global_yaw", Float32, self.yaw_callback)
        # 타이머를 사용하여 10Hz로 tf 발행
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        rospy.spin()

    def utm_callback(self, msg):
        self.vehicle_x = msg.point.x
        self.vehicle_y = msg.point.y
        self.vehicle_z = msg.point.z

    def yaw_callback(self, msg):
        self.global_yaw = msg.data

    def timer_callback(self, event):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "vehicle"
        t.transform.translation.x = self.vehicle_x
        t.transform.translation.y = self.vehicle_y
        t.transform.translation.z = self.vehicle_z
        q = quaternion_from_euler(0, 0, self.global_yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)
        rospy.loginfo_throttle(1, "Broadcasting vehicle tf: x=%.2f, y=%.2f, yaw=%.2f deg",
                                 self.vehicle_x, self.vehicle_y, math.degrees(self.global_yaw))

if __name__ == '__main__':
    try:
        VehicleTFBroadcaster()
    except rospy.ROSInterruptException:
        pass
