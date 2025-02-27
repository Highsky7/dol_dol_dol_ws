#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import numpy as np
import math
import cv2

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# === 전역 변수 (차량의 현재 위치/자세) ===
car_x = 0.0
car_y = 0.0
car_yaw = 0.0  # rad

# 시각화 파라미터
IMG_SIZE = 800
SCALE = 30.0       # 1m 당 30픽셀 (값이 작을수록 축소됨)
CENTER_X = IMG_SIZE // 2
CENTER_Y = int(IMG_SIZE * 0.8)  # 차량을 창 아래쪽(80%)에 배치

# ROI 범위 (로컬 좌표, m)
ROI_MIN = -54.0
ROI_MAX =  54.0

# 회피/로컬 경로 생성 파라미터
AVOIDANCE_OFFSET = 2.0    # 장애물 회피를 위한 lateral offset (m)
LOOKAHEAD_DIST = 15.0     # 목표 복귀점까지의 전방 거리 (m)
MID_POINT_DIST = 5.0      # 회피 제어점까지의 전방 거리 (m)
LOCAL_PATH_RESOLUTION = 50  # Bezier curve 샘플링 점 수

# 클러스터링 파라미터
CLUSTER_DISTANCE_THRESHOLD = 1.0  # 같은 클러스터로 묶일 최대 거리 (m)
DENSE_CLUSTER_MIN_SIZE = 5        # 클러스터 내 콘 개수가 이 값 이상이면 밀집 클러스터로 간주

# Publisher for local path
local_path_pub = None

#######################################################
# Helper Functions: 좌표 변환 및 클러스터링 관련
#######################################################

def global_to_local(gx, gy, cx, cy, yaw):
    dx = gx - cx
    dy = gy - cy
    lx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    ly = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    return lx, ly

def local_to_pixel(lx, ly):
    px = int(CENTER_X - SCALE * ly)
    py = int(CENTER_Y - SCALE * lx)
    return px, py

def local_to_global(lx, ly, cx, cy, yaw):
    gx = cx + lx * math.cos(yaw) - ly * math.sin(yaw)
    gy = cy + lx * math.sin(yaw) + ly * math.cos(yaw)
    return gx, gy

def euclidean_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def cluster_cones(cones):
    visited = [False] * len(cones)
    clusters = []
    for i in range(len(cones)):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            cluster = []
            while queue:
                idx = queue.pop(0)
                cluster.append(cones[idx])
                for j in range(len(cones)):
                    if not visited[j] and euclidean_distance(cones[idx], cones[j]) < CLUSTER_DISTANCE_THRESHOLD:
                        visited[j] = True
                        queue.append(j)
            clusters.append(cluster)
    return clusters

#######################################################
# 결정 함수: 밀집 클러스터를 이용하여 회피 방향 결정
#######################################################

def decide_avoidance_direction_from_clusters(obstacles):
    """
    ROI 내 장애물들을 클러스터링한 후, 밀집 클러스터(크기가 DENSE_CLUSTER_MIN_SIZE 이상인)
    가 있으면, 가장 가까운(평균 x 값이 가장 작은) 밀집 클러스터의 평균 y 값에 따라 회피 방향을 결정.
    평균 y > 0  → 장애물이 좌측에 몰림 → 회피는 오른쪽 (음수 offset)
    평균 y < 0  → 장애물이 우측에 몰림 → 회피는 왼쪽 (양수 offset)
    밀집 클러스터가 없으면 회피 offset은 0.0 (즉, 글로벌 경로를 그대로 따름)
    """
    clusters = cluster_cones(obstacles)
    dense_clusters = [cluster for cluster in clusters if len(cluster) >= DENSE_CLUSTER_MIN_SIZE]
    if not dense_clusters:
        return 0.0  # 밀집 장애물이 없으므로 회피 없음
    closest_cluster = min(dense_clusters, key=lambda clust: np.mean([pt[0] for pt in clust]))
    avg_y = np.mean([pt[1] for pt in closest_cluster])
    if avg_y > 0:
        return -AVOIDANCE_OFFSET  # 장애물이 좌측 → 회피 오른쪽
    else:
        return AVOIDANCE_OFFSET   # 장애물이 우측 → 회피 왼쪽

#######################################################
# Local Path 생성: Bezier Curve
#######################################################

def generate_bezier_path(P0, P1, P2):
    t_vals = np.linspace(0, 1, LOCAL_PATH_RESOLUTION)
    path_x = (1 - t_vals)**2 * P0[0] + 2*(1 - t_vals)*t_vals * P1[0] + t_vals**2 * P2[0]
    path_y = (1 - t_vals)**2 * P0[1] + 2*(1 - t_vals)*t_vals * P1[1] + t_vals**2 * P2[1]
    return path_x, path_y

def draw_local_path(img, path_x, path_y, color=(0, 255, 255)):
    pts = []
    for x, y in zip(path_x, path_y):
        pts.append(local_to_pixel(x, y))
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2)

def publish_local_path(path_x, path_y, car_x, car_y, car_yaw):
    global local_path_pub
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "map"
    for lx, ly in zip(path_x, path_y):
        gx, gy = local_to_global(lx, ly, car_x, car_y, car_yaw)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position.x = gx
        pose.pose.position.y = gy
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        path_msg.poses.append(pose)
    local_path_pub.publish(path_msg)

#######################################################
# Main Loop: 장애물 데이터 수집, 회피 경로 생성 및 시각화
#######################################################

def vehicle_pose_callback(msg):
    global car_x, car_y, car_yaw
    car_x = msg.pose.position.x
    car_y = msg.pose.position.y
    qx = msg.pose.orientation.x
    qy = msg.pose.orientation.y
    qz = msg.pose.orientation.z
    qw = msg.pose.orientation.w
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    car_yaw = math.atan2(siny_cosp, cosy_cosp)

def main():
    global local_path_pub
    rospy.init_node("obstacle_avoiding_node", anonymous=True)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("control")
    csv_path = pkg_path + "/data/cones_with_obstacles2.csv"

    try:
        cones_data = pd.read_csv(csv_path).dropna()
        cones_x = cones_data['x'].to_numpy()
        cones_y = cones_data['y'].to_numpy()
        rospy.loginfo("Loaded {} cones from CSV.".format(len(cones_x)))
    except Exception as e:
        rospy.logerr("Failed to load cones CSV: {}".format(e))
        cones_x = np.array([])
        cones_y = np.array([])

    rospy.Subscriber("vehicle_pose", PoseStamped, vehicle_pose_callback)
    local_path_pub = rospy.Publisher("local_path", Path, queue_size=10)

    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

        # 1) 글로벌 CSV의 콘들을 차량의 로컬 좌표로 변환 및 ROI 필터링
        obstacles_local = []
        for gx, gy in zip(cones_x, cones_y):
            lx, ly = global_to_local(gx, gy, car_x, car_y, car_yaw)
            if ROI_MIN <= lx <= ROI_MAX and ROI_MIN <= ly <= ROI_MAX:
                obstacles_local.append((lx, ly))
                px, py = local_to_pixel(lx, ly)
                cv2.circle(img, (px, py), 5, (0, 165, 255), -1)
                cv2.putText(img, f"({lx:.1f},{ly:.1f})", (px+5, py-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # 2) 클러스터링 후, 밀집된 정적 장애물만 회피 대상으로 결정
        clusters = cluster_cones(obstacles_local)
        dense_clusters = [cluster for cluster in clusters if len(cluster) >= DENSE_CLUSTER_MIN_SIZE]

        if dense_clusters:
            # 밀집 장애물이 인식된 경우에만 회피 경로 생성
            closest_cluster = min(dense_clusters, key=lambda clust: np.mean([pt[0] for pt in clust]))
            avg_y = np.mean([pt[1] for pt in closest_cluster])
            if avg_y > 0:
                avoidance_offset = -AVOIDANCE_OFFSET  # 장애물이 좌측 → 회피 오른쪽
            else:
                avoidance_offset = AVOIDANCE_OFFSET   # 장애물이 우측 → 회피 왼쪽

            # 3) 로컬 회피 경로 생성 (Bezier Curve) - 장애물이 인식될 때만 생성
            P0 = (0.0, 0.0)                                # 현재 차량 위치 (로컬 좌표)
            P1 = (MID_POINT_DIST, avoidance_offset)        # 회피 제어점
            P2 = (LOOKAHEAD_DIST, 0.0)                       # 목표 복귀점 (중앙)
            path_x, path_y = generate_bezier_path(P0, P1, P2)
            draw_local_path(img, path_x, path_y)
            for pt, color in zip([P0, P1, P2], [(255,0,0), (0,255,0), (0,0,255)]):
                cv2.circle(img, local_to_pixel(pt[0], pt[1]), 5, color, -1)
            cv2.putText(img, f"Avoidance Offset: {avoidance_offset:.1f} m", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            # 로컬 경로를 글로벌 좌표로 변환 후 발행
            publish_local_path(path_x, path_y, car_x, car_y, car_yaw)
        else:
            # 밀집 장애물이 없으면 회피 없음 → local path 발행 안함
            cv2.putText(img, "No static obstacles detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        cv2.imshow("Local Path Planning", img)
        if cv2.waitKey(1) == 27:
            break

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
