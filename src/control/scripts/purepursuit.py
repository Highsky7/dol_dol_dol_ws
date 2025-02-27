#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Point
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32
from scipy.interpolate import splprep, splev
from nav_msgs.msg import Path

# 글로벌 변수: local path 저장용
local_waypoints = []
local_path_timestamp = 0.0
LOCAL_PATH_TIMEOUT = 2.0  # local path가 2초 이상 갱신되지 않으면 무효

class PurePursuit:
    def __init__(self):
        rospy.init_node("pure_pursuit_node", anonymous=True)

        # 파라미터 설정
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 5.0)
        self.desired_speed = rospy.get_param("~desired_speed", 5.0)
        self.wheel_base = rospy.get_param("~wheel_base", 0.75)
        self.smoothing = rospy.get_param("~smoothing", 0.5)
        self.sampling_interval = rospy.get_param("~sampling_interval", 1.0)

        # CSV 경로 설정 (global path)
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('path_planning')
        csv_path_rddf = package_path + "/data/example1.csv"

        # 글로벌 경로 로드 & B-Spline
        self.x_global, self.y_global = self.load_and_process_global_path(csv_path_rddf)

        # ROS 통신 설정
        rospy.Subscriber("vehicle_pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("local_path", Path, self.local_path_callback)

        # ✅ 퍼블리셔 추가
        self.cmd_pub = rospy.Publisher("ackermann_cmd", AckermannDriveStamped, queue_size=10)
        self.lookahead_pub = rospy.Publisher("lookahead_point", Point, queue_size=10)
        self.steer_angle_pub = rospy.Publisher("auto_steer_angle_cone", Float32, queue_size=10)  # 조향각만 퍼블리시

        # 차량 현재 상태
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        rospy.loginfo("PurePursuit node is ready.")

    def load_and_process_global_path(self, csv_path):
        try:
            data_rddf = pd.read_csv(csv_path).dropna()
        except Exception as e:
            rospy.logerr("Failed to read global path CSV: {}".format(e))
            return None, None
        
        x_rddf = data_rddf['x'].to_numpy()
        y_rddf = data_rddf['y'].to_numpy()

        tck, u = splprep([x_rddf, y_rddf], s=self.smoothing)
        u_fine = np.linspace(0, 1, 1000)
        x_fine, y_fine = splev(u_fine, tck)

        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        ds = np.sqrt(dx**2 + dy**2)
        s_fine = np.insert(np.cumsum(ds), 0, 0)
        total_length = s_fine[-1]
        num_samples = int(total_length / self.sampling_interval) + 1
        s_new = np.linspace(0, total_length, num_samples)
        u_new = np.interp(s_new, s_fine, u_fine)
        x_new, y_new = splev(u_new, tck)
        rospy.loginfo("Global path loaded with {} points.".format(len(x_new)))
        return x_new, y_new

    def local_path_callback(self, msg):
        global local_waypoints, local_path_timestamp
        waypoints = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        local_waypoints = waypoints
        local_path_timestamp = rospy.Time.now().to_sec()

    def pose_callback(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        self.current_yaw = self.quaternion_to_yaw(qx, qy, qz, qw)

        steering_angle, speed = self.pure_pursuit_control()
        self.publish_ackermann_cmd(steering_angle, speed)
        self.publish_steering_angle(steering_angle)  # ✅ 추가된 퍼블리시

    def pure_pursuit_control(self):
        if local_waypoints and (rospy.Time.now().to_sec() - local_path_timestamp < LOCAL_PATH_TIMEOUT):
            selected_path = local_waypoints
        else:
            selected_path = list(zip(self.x_global, self.y_global))
        
        lookahead_point = self.find_lookahead_point(selected_path)
        if lookahead_point is None:
            return 0.0, 0.0

        dx = lookahead_point[0] - self.current_x
        dy = lookahead_point[1] - self.current_y
        local_x = dx * math.cos(-self.current_yaw) - dy * math.sin(-self.current_yaw)
        local_y = dx * math.sin(-self.current_yaw) + dy * math.cos(-self.current_yaw)
        alpha = math.atan2(local_y, local_x)

        steering_angle = math.atan2(2.0 * self.wheel_base * math.sin(alpha),
                                    self.lookahead_distance)
        speed = self.desired_speed

        self.publish_lookahead_point(lookahead_point[0], lookahead_point[1])
        return steering_angle, speed

    def find_lookahead_point(self, path):
        min_idx, min_dist = None, float('inf')
        for i, (px, py) in enumerate(path):
            dist = math.sqrt((px - self.current_x)**2 + (py - self.current_y)**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx is None:
            return None
        for pt in path[min_idx:]:
            dist = math.sqrt((pt[0] - self.current_x)**2 + (pt[1] - self.current_y)**2)
            if dist >= self.lookahead_distance:
                return pt
        return None

    def publish_ackermann_cmd(self, steering_angle, speed):
        cmd_msg = AckermannDriveStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.header.frame_id = "map"
        cmd_msg.drive.speed = speed
        cmd_msg.drive.steering_angle = steering_angle
        self.cmd_pub.publish(cmd_msg)

    def publish_steering_angle(self, steering_angle):
        steer_msg = Float32()
        steer_msg.data = steering_angle
        self.steer_angle_pub.publish(steer_msg)

    def publish_lookahead_point(self, x, y):
        point_msg = Point()
        point_msg.x = x
        point_msg.y = y
        point_msg.z = 0.0
        self.lookahead_pub.publish(point_msg)

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        return math.atan2(siny_cosp, cosy_cosp)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    pp = PurePursuit()
    pp.run()
