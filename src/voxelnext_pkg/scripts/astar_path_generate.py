#!/home/highsky/lidar_env/bin/python3
# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math
import heapq
import time

class ConePathPlanner:
    def __init__(self):
        """
        ROS 노드 초기화 및 멤버 변수 설정
        """
        rospy.init_node("cone_path_planner", anonymous=True)
        rospy.loginfo("===== Cone Path Planner Node Initialized =====")

        # 1) 파라미터 설정 (필요시 동적으로 조정 가능)
        self.grid_min_x = -20.0  # 맵 최소 X
        self.grid_max_x =  20.0  # 맵 최대 X
        self.grid_min_y = -20.0  # 맵 최소 Y
        self.grid_max_y =  20.0  # 맵 최대 Y
        self.resolution = 0.1    # 격자 해상도 (m)
        self.robot_radius = 0.3  # 로봇(차량) 안전 반경 (장애물에 얼마나 떨어져야 하는지)
        # 시작점과 목표점 (원하는 대로 수정 가능)
        self.start_x = 0.0
        self.start_y = 0.0
        self.goal_x  = 10.0
        self.goal_y  = 0.0

        # 2) 내부 맵 크기 계산
        self.width  = int((self.grid_max_x - self.grid_min_x) / self.resolution)
        self.height = int((self.grid_max_y - self.grid_min_y) / self.resolution)

        # occupancy grid (장애물 여부를 저장할 2D 배열)
        # 0: free, 1: obstacle
        self.occupancy_grid = [[0 for _ in range(self.height)] for _ in range(self.width)]

        # 3) /center_markers 토픽 구독 (교통콘 MarkerArray)
        self.cone_sub = rospy.Subscriber(
            "/center_markers",
            MarkerArray,
            self.center_markers_callback,
            queue_size=1
        )

        # 4) 경로 퍼블리셔 (nav_msgs/Path 형태로 발행)
        self.path_pub = rospy.Publisher(
            "/planned_path",
            Path,
            queue_size=1
        )

        # 5) 주기적으로 경로계획 수행(또는 토픽 수신 시마다)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.plan_and_publish_path)

        # 교통콘 좌표 리스트
        self.cone_positions = []
        rospy.loginfo("Path Planner setup complete. Waiting for cones...")

    def center_markers_callback(self, msg):
        """
        /center_markers에서 들어오는 MarkerArray를 받아
        교통콘 좌표를 추출하고 occupancy grid 갱신
        """
        # 1) 기존 occupancy grid 초기화
        self.clear_occupancy_grid()

        # 2) MarkerArray에서 모든 Marker의 (x,y) 좌표 추출
        cone_list = []
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            cone_list.append((x, y))

        self.cone_positions = cone_list

        # 3) occupancy grid 갱신 (콘 주변을 장애물로 표시)
        #    - 콘 중심 + 안전거리(로봇 반경)
        #    - 간단히 주변 셀을 obstacle=1로 마킹
        for (cx, cy) in self.cone_positions:
            self.set_obstacle_in_grid(cx, cy, self.robot_radius)

    def clear_occupancy_grid(self):
        """
        occupancy grid를 전부 0(Free)로 초기화
        """
        for i in range(self.width):
            for j in range(self.height):
                self.occupancy_grid[i][j] = 0

    def set_obstacle_in_grid(self, obs_x, obs_y, inflation_radius):
        """
        (obs_x, obs_y)를 중심으로 하는 원형 영역을 obstacle로 설정
        """
        # 장애물 범위 파악
        min_x = obs_x - inflation_radius
        max_x = obs_x + inflation_radius
        min_y = obs_y - inflation_radius
        max_y = obs_y + inflation_radius

        # 해당 범위를 격자로 환산하여 순회
        grid_min_i, grid_min_j = self.world2grid(min_x, min_y)
        grid_max_i, grid_max_j = self.world2grid(max_x, max_y)

        # grid 좌표 순회하면서 실제로 원 안에 들어가면 obstacle 표시
        for i in range(grid_min_i, grid_max_i+1):
            for j in range(grid_min_j, grid_max_j+1):
                if not self.is_valid_grid(i, j):
                    continue
                # 중심 obs_x, obs_y와의 실제 거리 계산
                wx, wy = self.grid2world(i, j)
                dist = math.hypot(wx - obs_x, wy - obs_y)
                if dist <= inflation_radius:
                    self.occupancy_grid[i][j] = 1

    def plan_and_publish_path(self, event):
        """
        주기적으로 불려서 A* 경로를 계산하고,
        nav_msgs/Path로 퍼블리시
        """
        # A* 경로계획 실행
        path = self.a_star_planning(
            sx=self.start_x,
            sy=self.start_y,
            gx=self.goal_x,
            gy=self.goal_y
        )

        # 경로가 성공적으로 계산되면 Path 메시지로 퍼블리시
        if path is not None and len(path) > 0:
            path_msg = Path()
            path_msg.header.frame_id = "velodyne"  # RViz에서 확인할 프레임 (LiDAR 프레임과 맞춤)
            path_msg.header.stamp = rospy.Time.now()

            for (x, y) in path:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                path_msg.poses.append(pose)

            self.path_pub.publish(path_msg)
            rospy.loginfo("Path published. (length=%d)" % len(path))
        else:
            rospy.logwarn("No valid path found.")

    def a_star_planning(self, sx, sy, gx, gy):
        """
        A* 알고리즘으로 (sx, sy) -> (gx, gy) 경로를 찾는다.
        - 격자 지도(occupancy_grid) 사용
        - 8방향 이동 허용
        """
        start_i, start_j = self.world2grid(sx, sy)
        goal_i,  goal_j  = self.world2grid(gx, gy)

        if not self.is_valid_grid(start_i, start_j):
            rospy.logwarn("Start position is out of grid or in obstacle.")
            return None
        if not self.is_valid_grid(goal_i, goal_j):
            rospy.logwarn("Goal position is out of grid or in obstacle.")
            return None
        if self.occupancy_grid[start_i][start_j] == 1:
            rospy.logwarn("Start position is in obstacle.")
            return None
        if self.occupancy_grid[goal_i][goal_j] == 1:
            rospy.logwarn("Goal position is in obstacle.")
            return None

        # 오픈리스트(우선순위큐)와 닫힌리스트(방문처리) 준비
        open_list = []
        closed_set = set()

        # 노드: (F=g+h, g, (i, j), parent_i, parent_j)
        start_node = (0.0, 0.0, (start_i, start_j), None, None)
        heapq.heappush(open_list, start_node)

        # 부모 추적용 딕셔너리: key=(i,j), value=(parent_i, parent_j)
        parent_dict = {}

        # 방향 정의 (8방향)
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        while open_list:
            f, g, (ci, cj), pi, pj = heapq.heappop(open_list)

            # 이미 방문했던 노드면 스킵
            if (ci, cj) in closed_set:
                continue
            # 부모정보 저장
            parent_dict[(ci, cj)] = (pi, pj)
            # 현재 노드를 방문 처리
            closed_set.add((ci, cj))

            # 목표 지점 도달 체크
            if (ci == goal_i) and (cj == goal_j):
                # 경로 복원
                return self.reconstruct_path(parent_dict, ci, cj)

            # 인접 노드 탐색
            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if not self.is_valid_grid(ni, nj):
                    continue
                if self.occupancy_grid[ni][nj] == 1:  # 장애물
                    continue
                if (ni, nj) in closed_set:
                    continue

                # 비용 계산
                new_g = g + math.hypot(di, dj)  # 이동 거리(가중치)
                h = self.heuristic(ni, nj, goal_i, goal_j)
                new_f = new_g + h

                node = (new_f, new_g, (ni, nj), ci, cj)
                heapq.heappush(open_list, node)

        # 경로를 찾지 못한 경우
        return None

    def reconstruct_path(self, parent_dict, ci, cj):
        path = []
        # (ci, cj)가 (None, None)이 되면 루프 중단
        while (ci, cj) is not None and (ci, cj) != (None, None):
            wx, wy = self.grid2world(ci, cj)
            path.append((wx, wy))
            (ci, cj) = parent_dict[(ci, cj)]
        path.reverse()
        return path


    def heuristic(self, i1, j1, i2, j2):
        """
        휴리스틱(유클리드 거리)
        """
        (x1, y1) = self.grid2world(i1, j1)
        (x2, y2) = self.grid2world(i2, j2)
        return math.hypot(x2 - x1, y2 - y1)

    def world2grid(self, x, y):
        """
        월드좌표(x, y)를 grid 인덱스(i, j)로 변환
        """
        i = int((x - self.grid_min_x) / self.resolution)
        j = int((y - self.grid_min_y) / self.resolution)
        return (i, j)

    def grid2world(self, i, j):
        """
        grid 인덱스(i, j)를 월드좌표(x, y)로 변환
        """
        x = (i * self.resolution) + self.grid_min_x + self.resolution/2.0
        y = (j * self.resolution) + self.grid_min_y + self.resolution/2.0
        return (x, y)

    def is_valid_grid(self, i, j):
        """
        grid 인덱스(i, j)가 맵 범위 내에 있는지, 그리고 obstacle이 아닌지 확인
        """
        if i < 0 or i >= self.width:
            return False
        if j < 0 or j >= self.height:
            return False
        return True

    def run(self):
        """
        ROS spin
        """
        rospy.spin()

if __name__ == "__main__":
    planner = ConePathPlanner()
    planner.run()
