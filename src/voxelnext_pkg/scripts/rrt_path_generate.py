#!/home/highsky/lidar_env/bin/python3
# -*- coding: utf-8 -*-

import rospy
import math
import random
import numpy as np
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.interpolate import splprep, splev

# ---------------------------------------------------------
# RRT* 알고리즘 구현 (기존과 동일)
# ---------------------------------------------------------
class RRTStarPlanner:
    def __init__(self, start, goal, obstacle_list,
                 x_range=(-10, 10), y_range=(-10, 10),
                 expand_dis=0.5, goal_sample_rate=5,
                 max_iter=2000, connect_circle_dist=2.0):
        """
        RRT* 파라미터 설정:
          - start: (x, y) 시작점
          - goal:  (x, y) 목표점
          - obstacle_list: [(ox, oy), ...] 장애물(꼬깔) 좌표 목록
          - x_range, y_range: 샘플링 영역
          - expand_dis: 한 번 확장할 거리
          - goal_sample_rate: 목표 직접 샘플링 확률(%)
          - max_iter: 최대 반복 횟수
          - connect_circle_dist: 리와이어 시 인근 탐색 거리
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.x_range = x_range
        self.y_range = y_range
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist

        self.node_list = [self.start]
        # 장애물과의 최소 허용 간격 (꼬깔 주변 0.3m 이내 충돌로 판단)
        self.obstacle_clearance = 0.3

    def planning(self):
        """
        RRT* 메인 함수: 경로(노드 리스트)를 반환 (실패 시 None)
        """
        for i in range(self.max_iter):
            rnd_node = self.sample_free()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            new_node = self.steer(self.node_list[nearest_ind], rnd_node, self.expand_dis)

            if not self.check_collision(new_node):
                continue

            near_indices = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_indices)
            self.node_list.append(new_node)
            self.rewire(new_node, near_indices)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)
        return None

    def sample_free(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(
                random.uniform(self.x_range[0], self.x_range[1]),
                random.uniform(self.y_range[0], self.y_range[1])
            )
        else:
            rnd = Node(self.goal.x, self.goal.y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        if extend_length > d:
            extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        new_node.parent = from_node
        new_node.cost = from_node.cost + extend_length
        return new_node

    def check_collision(self, node):
        if node is None:
            return False
        if not (self.x_range[0] <= node.x <= self.x_range[1]):
            return False
        if not (self.y_range[0] <= node.y <= self.y_range[1]):
            return False
        for (ox, oy) in self.obstacle_list:
            if math.hypot(node.x - ox, node.y - oy) <= self.obstacle_clearance:
                return False
        return True

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        if r < self.expand_dis:
            r = self.expand_dis
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_indices = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_indices

    def choose_parent(self, new_node, near_indices):
        if not near_indices:
            return new_node
        costs = []
        for i in near_indices:
            near_node = self.node_list[i]
            tmp_node = self.steer(near_node, new_node, self.expand_dis)
            if self.check_collision(tmp_node):
                costs.append(near_node.cost + self.calc_distance(near_node, tmp_node))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        min_ind = near_indices[costs.index(min_cost)]
        if min_cost == float("inf"):
            return new_node
        new_node = self.steer(self.node_list[min_ind], new_node, self.expand_dis)
        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]
        return new_node

    def rewire(self, new_node, near_indices):
        for i in near_indices:
            near_node = self.node_list[i]
            tmp_node = self.steer(new_node, near_node, self.expand_dis)
            if not self.check_collision(tmp_node):
                continue
            cost = new_node.cost + self.calc_distance(new_node, near_node)
            if cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = cost

    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.goal.x, y - self.goal.y)

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    @staticmethod
    def calc_distance(node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

# ---------------------------------------------------------
# Node 클래스 (RRT*용)
# ---------------------------------------------------------
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None

# ---------------------------------------------------------
# Sub-goal 방식 RRT* 경로 생성 및 스무딩 노드
# ---------------------------------------------------------
class LidarSubGoalRRTStarNode:
    def __init__(self):
        self.lidar_frame = "velodyne"
        # 샘플링 영역 (환경에 맞게 조정)
        self.x_range = (-10.0, 10.0)
        self.y_range = (-10.0, 10.0)
        # RRT* 파라미터
        self.expand_dis = 0.5
        self.goal_sample_rate = 10
        self.max_iter = 2000
        self.connect_circle_dist = 2.0
        # 생성할 최대 sub-goal 개수 (이 값을 조정하여 경로 길이를 늘리거나 줄일 수 있음)
        self.max_sub_goals = rospy.get_param("~max_sub_goals", 2)

        self.path_pub = rospy.Publisher('/lidar_rrtstar_path', Path, queue_size=10)
        rospy.Subscriber('/center_markers', MarkerArray, self.marker_callback, queue_size=10)
        rospy.loginfo("LidarSubGoalRRTStarNode 초기화 완료. 대기 중...")

    def marker_callback(self, marker_array):
        """
        1) /center_markers로부터 꼬깔 좌표(장애물)를 수집  
        2) 현재 위치(원점)에서 시작하여, x>0인 꼬깔 중 가장 가까운 2개를 선택해 중점을 sub-goal로 산출  
           (생성할 sub-goal 개수는 self.max_sub_goals로 조절)
        3) 각 구간(현재위치→p₁, p₁→p₂, …)에 대해 RRT* 경로를 생성하고 연결  
        4) 전체 경로를 spline 보간으로 스무딩한 후 퍼블리시
        """
        # (1) 꼬깔 좌표 수집 (2D: z 무시)
        obstacle_list = []
        for marker in marker_array.markers:
            ox = marker.pose.position.x
            oy = marker.pose.position.y
            obstacle_list.append((ox, oy))
        if not obstacle_list:
            rospy.logwarn("받은 트래픽 콘이 없습니다. 경로 생성 불가.")
            return

        # (2) Sub-goal 생성: 현재 위치에서 시작 (0,0)
        start = (0.0, 0.0)
        sub_goals = self.generate_sub_goals(obstacle_list, start, self.max_sub_goals)
        if not sub_goals:
            rospy.logwarn("유효한 sub-goal이 생성되지 않았습니다.")
            return

        rospy.loginfo("생성된 sub-goal: %s", sub_goals)

        # (3) 각 sub-goal 구간에 대해 RRT* 경로 생성 및 연결
        full_path = []
        current_start = start
        for goal in sub_goals:
            rospy.loginfo("구간 경로 생성: 시작점 %s → 목표점 %s", current_start, goal)
            rrt_star = RRTStarPlanner(
                start=current_start,
                goal=goal,
                obstacle_list=obstacle_list,
                x_range=self.x_range,
                y_range=self.y_range,
                expand_dis=self.expand_dis,
                goal_sample_rate=self.goal_sample_rate,
                max_iter=self.max_iter,
                connect_circle_dist=self.connect_circle_dist
            )
            segment = rrt_star.planning()
            if segment is None:
                rospy.logwarn("구간 경로 생성 실패: 시작점 %s → 목표점 %s", current_start, goal)
                return
            if not full_path:
                full_path.extend(segment)
            else:
                full_path.extend(segment[1:])  # 중복 제거
            current_start = goal

        rospy.loginfo("원시 경로 점 개수: %d", len(full_path))
        # (4) 스무딩: spline 보간을 사용하여 경로 스무딩
        smoothed_path = self.smooth_path(full_path, num_points=150, s=1.0)
        rospy.loginfo("스무딩 후 경로 점 개수: %d", len(smoothed_path))

        # (5) 최종 경로 퍼블리시 (smoothed path)
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.lidar_frame
        for (px, py) in smoothed_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = px
            pose.pose.position.y = py
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        rospy.loginfo("전체 스무딩 경로 (%d 점) 퍼블리시 완료.", len(smoothed_path))

    def generate_sub_goals(self, obstacle_list, start, max_sub_goals):
        """
        현재 위치(start)에서 시작하여,  
        x>0인 꼬깔들 중에서 현재 위치로부터 가장 가까운 2개를 선택하고 그 중점을 sub-goal로 산출한 후,  
        새로운 기준점으로 갱신하여 반복한다.
          - 최대 max_sub_goals 개까지 생성 (조건 미충족 시 조기 종료)
        """
        sub_goals = []
        current_pos = start
        iter_count = 0

        while iter_count < max_sub_goals:
            # 현재 기준(current_pos)보다 x좌표가 큰 꼬깔만 고려 (진행 방향 유지)
            candidates = [ (ox, oy) for (ox, oy) in obstacle_list if ox > current_pos[0] and ox > 0 ]
            if len(candidates) < 2:
                break
            # 현재 위치와의 거리 기준 오름차순 정렬
            candidates.sort(key=lambda pt: math.hypot(pt[0]-current_pos[0], pt[1]-current_pos[1]))
            cone1, cone2 = candidates[0], candidates[1]
            mid_x = (cone1[0] + cone2[0]) / 2.0
            mid_y = (cone1[1] + cone2[1]) / 2.0
            new_goal = (mid_x, mid_y)
            if math.hypot(new_goal[0]-current_pos[0], new_goal[1]-current_pos[1]) < 0.1:
                break
            sub_goals.append(new_goal)
            current_pos = new_goal
            iter_count += 1
        return sub_goals

    def smooth_path(self, path, num_points=100, s=0.0):
        """
        경로 보간을 통해 스무딩한다.
          - path: [[x,y], ...] 형태의 원시 경로
          - num_points: 스무딩 후 생성할 포인트 수
          - s: smoothing factor (0이면 interpolation, 양수면 smoothing)
        """
        if len(path) < 3:
            return path  # 포인트가 너무 적으면 스무딩 없이 반환
        path = np.array(path)
        x = path[:,0]
        y = path[:,1]
        tck, u = splprep([x, y], s=s, per=False)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        smoothed = list(zip(x_new, y_new))
        return smoothed

def main():
    rospy.init_node('lidar_rrtstar_node', anonymous=True)
    LidarSubGoalRRTStarNode()
    rospy.loginfo("lidar_rrtstar_node 노드가 시작되었습니다.")
    rospy.spin()

if __name__ == '__main__':
    main()
