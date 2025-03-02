#!/home/highsky/lidar_env/bin/python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Path
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from scipy.interpolate import BSpline
import math
from typing import List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor

# 상수 정의
GOAL_DISTANCE_THRESHOLD = 0.3  # 목표에 도달한 것으로 간주하는 거리 (미터)
OBSTACLE_CLEARANCE = 0.7      # cone 주변 안전 마진 (미터, 동적 조정 가능)
MAX_STEERING_ANGLE = 0.4      # 최대 조향 각도 (라디안, 차량 특성에 따라 조정)
MAP_SIZE_X = 30.0             # x축 맵 크기 (미터, 트랙 크기에 맞춤)
MAP_SIZE_Y = 30.0             # y축 맵 크기 (미터, 트랙 크기에 맞춤)
MAX_ITERATIONS = 5000         # RRT* 최대 반복 횟수
GOAL_BIAS = 0.1              # RRT*에서 목표로 향할 확률 (0~1)

class AdvancedPathPlanner:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('advanced_path_planner_node', anonymous=True)
        rospy.loginfo("고급 경로 계획 노드 초기화 완료")

        # 퍼블리셔
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.path_marker_pub = rospy.Publisher('/planned_path_marker', Marker, queue_size=10)

        # 구독자
        self.center_markers_sub = rospy.Subscriber('/center_markers', MarkerArray, self.marker_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # 상태 변수
        self.cone_positions: List[np.ndarray] = []  # cone 위치 [x, y]
        self.robot_pose = None  # [x, y, theta]
        self.goal_pose = [15.0, 15.0]  # 기본 목표 위치 (트랙 끝부분, 사용자 조정 가능)

        # RRT* 및 경로 스무딩을 위한 변수
        self.min_distance = 0.2  # 샘플링 최소 거리 (미터)
        self.max_distance = 1.0  # 샘플링 최대 거리 (미터)
        self.neighbor_radius = 1.5  # 이웃 노드 탐색 반경 (미터)

    def odom_callback(self, msg):
        """로봇의 현재 위치 및 방향을 odometry에서 업데이트"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        _, _, theta = self.quaternion_to_euler(quaternion)
        self.robot_pose = [x, y, theta]
        rospy.logdebug(f"로봇 위치 업데이트: [{x}, {y}, {theta}]")

    def quaternion_to_euler(self, quaternion: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """쿼터니언을 오일러 각도로 변환 (roll, pitch, yaw)"""
        x, y, z, w = quaternion
        t0 = 2.0 * (w * z + x * y)
        t1 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t0, t1)
        return 0.0, 0.0, yaw

    def marker_callback(self, msg):
        """cone 위치를 /center_markers 토픽에서 추출"""
        self.cone_positions = []
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            self.cone_positions.append(np.array([x, y]))
        rospy.loginfo(f"감지된 건설 cone 수: {len(self.cone_positions)}")

    def is_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """두 점 사이 경로가 cone과 충돌하지 않는지 확인"""
        dist = np.linalg.norm(p2 - p1)
        steps = max(2, int(dist / self.min_distance))
        for t in np.linspace(0, 1, steps):
            point = p1 + t * (p2 - p1)
            for cone in self.cone_positions:
                if np.linalg.norm(point - cone) < OBSTACLE_CLEARANCE:
                    return False
        return True

    def find_nearest_node(self, nodes: List[np.ndarray], point: np.ndarray) -> Tuple[int, float]:
        """가장 가까운 노드를 찾음"""
        min_dist = float('inf')
        nearest_idx = -1
        for i, node in enumerate(nodes):
            dist = np.linalg.norm(node[:2] - point[:2])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx, min_dist

    def rrt_star(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """RRT* 알고리즘으로 최적 경로 계산"""
        nodes = [start]  # [x, y, theta]
        parents = [-1] * MAX_ITERATIONS

        start_time = time.time()
        with ThreadPoolExecutor() as executor:
            for _ in range(MAX_ITERATIONS):
                if time.time() - start_time > 2.0:  # 2초 제한
                    break

                # 랜덤 샘플링 또는 목표로 편향
                if np.random.random() < GOAL_BIAS:
                    sample = np.array([goal[0], goal[1], 0.0])  # theta는 임시로 0
                else:
                    sample = np.array([
                        np.random.uniform(-MAP_SIZE_X / 2, MAP_SIZE_X / 2),
                        np.random.uniform(-MAP_SIZE_Y / 2, MAP_SIZE_Y / 2),
                        np.random.uniform(-math.pi, math.pi)
                    ])

                # 가장 가까운 노드 찾기
                nearest_idx, _ = self.find_nearest_node(nodes, sample)
                nearest = nodes[nearest_idx]

                # 새로운 노드 생성 (차량 운동학 고려)
                new_node = self.steer(nearest, sample)
                if not self.is_collision_free(nearest[:2], new_node[:2]):
                    continue

                # 이웃 노드 찾기 및 비용 최적화
                neighbors = []
                for i, node in enumerate(nodes):
                    if np.linalg.norm(node[:2] - new_node[:2]) < self.neighbor_radius:
                        neighbors.append((i, node))
                
                min_cost = float('inf')
                best_parent = nearest_idx
                for neighbor_idx, neighbor in neighbors:
                    cost = np.linalg.norm(neighbor[:2] - new_node[:2]) + \
                           (g_score.get(str(neighbor_idx), float('inf')) if neighbor_idx != -1 else 0)
                    if cost < min_cost and self.is_collision_free(neighbor[:2], new_node[:2]):
                        min_cost = cost
                        best_parent = neighbor_idx

                nodes.append(new_node)
                parents.append(best_parent)

                # 목표에 가까운지 확인
                if np.linalg.norm(new_node[:2] - goal) < GOAL_DISTANCE_THRESHOLD:
                    path = self.reconstruct_path(nodes, parents, len(nodes) - 1)
                    return self.smooth_path(path)

        rospy.logwarn("RRT*로 경로를 찾지 못했습니다.")
        return None

    def steer(self, from_node: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """차량 운동학을 고려하여 새로운 노드 생성"""
        dx = to_point[0] - from_node[0]
        dy = to_point[1] - from_node[1]
        dist = np.linalg.norm([dx, dy])
        if dist < self.min_distance:
            return from_node.copy()

        angle = math.atan2(dy, dx)
        theta_diff = angle - from_node[2]
        theta_diff = (theta_diff + math.pi) % (2 * math.pi) - math.pi  # 정규화
        if abs(theta_diff) > MAX_STEERING_ANGLE:
            theta_diff = np.sign(theta_diff) * MAX_STEERING_ANGLE

        new_theta = from_node[2] + theta_diff
        new_x = from_node[0] + min(dist, self.max_distance) * math.cos(new_theta)
        new_y = from_node[1] + min(dist, self.max_distance) * math.sin(new_theta)
        return np.array([new_x, new_y, new_theta])

    def reconstruct_path(self, nodes: List[np.ndarray], parents: List[int], goal_idx: int) -> List[np.ndarray]:
        """경로 재구성"""
        path = []
        current = goal_idx
        while current != -1:
            path.append(nodes[current])
            current = parents[current]
        return path[::-1]  # 시작에서 목표로

    def smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """B-스플라인을 사용한 경로 스무딩"""
        if len(path) < 2:
            return path

        # x, y 좌표 추출
        points = np.array([[p[0], p[1]] for p in path])
        t = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, max(100, len(points) * 2))  # 더 부드러운 포인트 수

        # B-스플라인 피팅
        spl_x = BSpline.fit(t, points[:, 0], k=3, degree=3)
        spl_y = BSpline.fit(t, points[:, 1], k=3, degree=3)

        smooth_x = spl_x(t_new)
        smooth_y = spl_y(t_new)

        # 부드러운 경로 생성 (theta는 방향 기반으로 계산)
        smooth_path = []
        for i in range(len(smooth_x)):
            x, y = smooth_x[i], smooth_y[i]
            if i > 0:
                prev_x, prev_y = smooth_x[i-1], smooth_y[i-1]
                theta = math.atan2(y - prev_y, x - prev_x)
            else:
                theta = path[0][2]  # 시작 theta 사용
            smooth_path.append([x, y, theta])

        return smooth_path

    def publish_path(self, path: List[np.ndarray]):
        """계획된 경로를 ROS Path 및 Marker로 게시"""
        if not path:
            return

        # nav_msgs/Path로 게시
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.frame_id = "velodyne"
        path_msg.header.stamp = rospy.Time.now()

        for point in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # 회전 없음
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

        # Marker로 초록색 선 시각화
        marker = Marker()
        marker.header = path_msg.header
        marker.ns = "planned_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # 선 두께
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0  # 초록색
        marker.color.b = 0.0

        for point in path:
            p = PoseStamped()
            p.pose.position.x = point[0]
            p.pose.position.y = point[1]
            p.pose.position.z = 0.0
            marker.points.append(p.pose.position)

        self.path_marker_pub.publish(marker)

    def run(self):
        """메인 루프: 실시간 경로 계획 및 게시"""
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            if self.robot_pose and self.goal_pose and self.cone_positions:
                start = np.array([self.robot_pose[0], self.robot_pose[1], self.robot_pose[2]])
                path = self.rrt_star(start, self.goal_pose)
                if path:
                    self.publish_path(path)
                    rospy.loginfo(f"경로 생성 완료, 포인트 수: {len(path)}")
            rate.sleep()

if __name__ == '__main__':
    try:
        planner = AdvancedPathPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("경로 계획 노드 종료")