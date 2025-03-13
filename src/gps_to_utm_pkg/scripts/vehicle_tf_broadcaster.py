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
        self.global_yaw = 0.0
        
        rospy.Subscriber("local_xy", PointStamped, self.local_xy_callback)
        rospy.Subscriber("global_yaw", Float32, self.yaw_callback)
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        rospy.spin()

    def local_xy_callback(self, msg):
        """/local_xy 토픽으로부터 받은 PointStamped 메시지에서 차량의 위치 정보를 추출합니다."""
        self.vehicle_x = msg.point.x
        self.vehicle_y = msg.point.y
        self.vehicle_z = msg.point.z

    def yaw_callback(self, msg):
        """/global_yaw 토픽으로부터 차량의 전역 요(heading) 정보를 업데이트합니다."""
        self.global_yaw = msg.data

    def timer_callback(self, event):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "reference"
        t.child_frame_id = "velodyne"
        t.transform.translation.x = self.vehicle_x
        t.transform.translation.y = self.vehicle_y
        t.transform.translation.z = self.vehicle_z
        q = quaternion_from_euler(0, 0, self.global_yaw)
        """quaternion_from_euler(0, 0, self.global_yaw)를 사용해 오일러 각(roll=0, pitch=0, yaw=self.global_yaw)을 쿼터니언으로 변환합니다."""
        
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)
        rospy.loginfo_throttle(0.1, "Broadcasting vehicle tf: x=%.2f, y=%.2f, yaw=%.2f deg",
                                 self.vehicle_x, self.vehicle_y, math.degrees(self.global_yaw))
        """각도는 radian 단위인 global_yaw를 도(degree) 단위로 변환하여 출력합니다."""

if __name__ == '__main__':
    try:
        VehicleTFBroadcaster()
    except rospy.ROSInterruptException:
        pass

"""
/local_xy 토픽에서 위치 데이터를 받아 저장하고,
/global_yaw 토픽에서 차량의 전역 요 값을 받아 저장합니다.
주기적으로 타이머 콜백이 실행되면서, 최신 위치와 방향 정보를 "reference" 프레임을 기준으로 "velodyne"이라는 자식 프레임에 대해 TF 메시지로 브로드캐스트합니다.
"""