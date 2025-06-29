#!/usr/bin/env python3
import rospy
import tf2_ros
import math
from geometry_msgs.msg import TransformStamped, PointStamped
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float32

class RefAntennaVelodyneTFBroadcaster:
    """reference -> antenna -> velodyne TF 브로드캐스터"""
    def __init__(self):
        rospy.init_node('ref_ant_vel_tf_broadcaster', anonymous=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # GPS 안테나로부터 얻은 위치 및 자세
        self.antenna_x = 0.0
        self.antenna_y = 0.0
        self.antenna_z = 0.0
        self.global_yaw = 0.0

        # antenna -> velodyne 오프셋 (미터 단위)
        # GPS 안테나 프레임(antenna frame) 기준에서 Velodyne 센서 프레임(velodyne frame)이 얼마만큼 떨어져 있는지를 나타내는 고정 오프셋(translation) 값
        # 하드 코딩 or local_cartesian.launch 에서 오프셋 파라미터 설정
        self.offset_x = rospy.get_param("~velodyne_offset_x", -36.0)
        self.offset_y = rospy.get_param("~velodyne_offset_y", 12.0)
        self.offset_z = rospy.get_param("~velodyne_offset_z", 0.0)
        # self.offset_x = rospy.get_param("~velodyne_offset_x")
        # self.offset_y = rospy.get_param("~velodyne_offset_y")
        # self.offset_z = rospy.get_param("~velodyne_offset_z")

        # 토픽 구독 및 타이머 설정
        rospy.Subscriber("local_xy", PointStamped, self.local_xy_callback)
        rospy.Subscriber("global_yaw", Float32, self.yaw_callback)
        rospy.Timer(rospy.Duration(0.05), self.timer_callback)
        rospy.spin()

    def local_xy_callback(self, msg):
        """/local_xy 토픽으로부터 GPS 안테나 위치 업데이트"""
        self.antenna_x = msg.point.x
        self.antenna_y = msg.point.y
        self.antenna_z = msg.point.z

    def yaw_callback(self, msg):
        """/global_yaw 토픽으로부터 전역 요 업데이트"""
        self.global_yaw = msg.data

    def timer_callback(self, event):
        now = rospy.Time.now()

        # 1) reference -> antenna
        antenna_tf = TransformStamped()
        antenna_tf.header.stamp = now
        antenna_tf.header.frame_id = "reference"
        antenna_tf.child_frame_id = "antenna"
        # /local_xy 토픽으로 들어온 GPS 안테나 좌표를 그대로 reference 프레임의 원점에서 평행이동시킵니다.
        antenna_tf.transform.translation.x = self.antenna_x
        antenna_tf.transform.translation.y = self.antenna_y
        antenna_tf.transform.translation.z = self.antenna_z
        # Z축(heading)으로 global_yaw만큼 회전시켜, 안테나 프레임의 방향을 전역 yaw와 일치시킵니다.
        q = quaternion_from_euler(0, 0, self.global_yaw)
        antenna_tf.transform.rotation.x = q[0]
        antenna_tf.transform.rotation.y = q[1]
        antenna_tf.transform.rotation.z = q[2]
        antenna_tf.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(antenna_tf)

        # 2) antenna -> velodyne (고정 오프셋)
        velodyne_tf = TransformStamped()
        velodyne_tf.header.stamp = now
        velodyne_tf.header.frame_id = "antenna"
        velodyne_tf.child_frame_id = "velodyne"
        # GPS 안테나, Velodyne이 떨어진 위치에 있도록 평행이동을 정의합니다.
        velodyne_tf.transform.translation.x = self.offset_x
        velodyne_tf.transform.translation.y = self.offset_y
        velodyne_tf.transform.translation.z = self.offset_z
        # 로컬(anten나) 기준으로는 추가 회전 없이 고정된 방향을 유지합니다.
        velodyne_tf.transform.rotation.x = 0.0
        velodyne_tf.transform.rotation.y = 0.0
        velodyne_tf.transform.rotation.z = 0.0
        velodyne_tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(velodyne_tf)

        rospy.loginfo_throttle(
            0.1,
            "[Ref->Ant]->[Ant->Vel] TF: antenna=(%.2f, %.2f), yaw=%.2f deg; offset=(%.2f, %.2f)",
            self.antenna_x, self.antenna_y, math.degrees(self.global_yaw),
            self.offset_x, self.offset_y
        )

"""
reference 프레임에서 GPS 안테나 위치로 이동 → yaw 회전

다시 안테나 프레임에서 Velodyne 오프셋만큼 이동
이 두 단계를 거쳐 Velodyne 좌표계(translation + yaw 회전)가 완성됩니다.
"""


if __name__ == '__main__':
    try:
        RefAntennaVelodyneTFBroadcaster()
    except rospy.ROSInterruptException:
        pass
