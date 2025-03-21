#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import numpy as np
import math
import cv2

from geometry_msgs.msg import PoseStamped

# === 전역 변수 (차량의 현재 위치/자세) ===
car_x = 0.0
car_y = 0.0
car_yaw = 0.0  # 라디안(rad)

# 시각화 파라미터
IMG_SIZE = 800
SCALE = 30.0  # 1m 당 30픽셀 (값이 작을수록 축소됨)
CENTER_X = IMG_SIZE // 2
CENTER_Y = int(IMG_SIZE * 0.8)  # 차량을 화면 아래쪽(80%)에 배치

# ROI 범위 설정 (라이다 좌표계 기준)
ROI_MIN = -54.0
ROI_MAX =  54.0

# === [추가] 클러스터링 파라미터 ===
CLUSTER_DISTANCE_THRESHOLD = 1.0  # 같은 클러스터로 묶일 최대 거리(단위: m)
DENSE_CLUSTER_MIN_SIZE = 5       # 콘 개수가 이 값 이상이면 밀집 클러스터로 간주

def vehicle_pose_callback(msg):
    """
    only_gps_model.py에서 발행되는 차량 Pose를 구독.
    """
    global car_x, car_y, car_yaw
    # 위치
    car_x = msg.pose.position.x
    car_y = msg.pose.position.y

    # 쿼터니언 -> yaw 변환
    qx = msg.pose.orientation.x
    qy = msg.pose.orientation.y
    qz = msg.pose.orientation.z
    qw = msg.pose.orientation.w
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    car_yaw = math.atan2(siny_cosp, cosy_cosp)

def global_to_local(gx, gy, cx, cy, yaw):
    """
    글로벌 좌표(gx, gy)를 차량 로컬 좌표계로 변환.
    +x: 차량 전방, +y: 차량 좌측
    """
    dx = gx - cx
    dy = gy - cy
    # -yaw 회전
    lx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    ly = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    return lx, ly

def local_to_pixel(lx, ly):
    """
    로컬 좌표(lx, ly)를 OpenCV 이미지 좌표(px, py)로 변환.
    +x(전방)는 이미지 상단(-y 방향),
    +y(좌측)는 이미지 왼쪽(-x 방향).
    """
    px = int(CENTER_X - SCALE * ly)
    py = int(CENTER_Y - SCALE * lx)
    return px, py

def filter_obstacles_by_roi(cones_x, cones_y):
    """
    ROI (사각형 영역) 내에 있는 콘만 필터링.
    LiDAR(로컬) 좌표에서 x, y가 [-54, 54] 범위 내에 있는지 확인.
    """
    roi_cones = []

    for gx, gy in zip(cones_x, cones_y):
        lx, ly = global_to_local(gx, gy, car_x, car_y, car_yaw)
        if ROI_MIN <= lx <= ROI_MAX and ROI_MIN <= ly <= ROI_MAX:
            roi_cones.append((lx, ly))

    return roi_cones

def draw_axes(img):
    """
    +x, +y 축 표시 (차량 중심)
    +x → 위쪽, +y → 왼쪽
    """
    # +x 축 (위쪽 방향)
    cv2.arrowedLine(
        img,
        (CENTER_X, CENTER_Y),
        (CENTER_X, CENTER_Y - 50),
        (0, 0, 255), 2
    )
    cv2.putText(img, "+x", (CENTER_X + 5, CENTER_Y - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # +y 축 (왼쪽 방향)
    cv2.arrowedLine(
        img,
        (CENTER_X, CENTER_Y),
        (CENTER_X - 50, CENTER_Y),
        (0, 0, 255), 2
    )
    cv2.putText(img, "+y", (CENTER_X - 60, CENTER_Y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

def draw_car(img):
    """
    차량을 간단한 사각형(또는 폴리곤)으로 표현.
    차량 로컬 좌표계에서 전방 +x, 좌측 +y
    """
    corners_local = [
        ( 0.5,  0.2),  # front-left
        ( 0.5, -0.2),  # front-right
        (-0.5, -0.2),  # rear-right
        (-0.5,  0.2)   # rear-left
    ]
    pts = []
    for (lx, ly) in corners_local:
        px, py = local_to_pixel(lx, ly)
        pts.append((px, py))

    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255))  # 빨간색으로 차량 표시

def draw_cones(img, cones):
    """
    ROI 내 콘들만 그려줌.
    cones: [(lx, ly), (lx, ly), ...] (이미 로컬 좌표로 변환된 상태)
    """
    for (lx, ly) in cones:
        px, py = local_to_pixel(lx, ly)
        # 콘 그리기
        cv2.circle(img, (px, py), 5, (0, 165, 255), -1)  # 주황색(콘)
        # 텍스트 표시
        text = "({:.2f}, {:.2f})".format(lx, ly)
        cv2.putText(img, text, (px+5, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

# === [추가] 클러스터링 & 바운딩 박스 로직 ===
def euclidean_distance(a, b):
    """
    두 점(a, b) 사이의 유클리드 거리 계산
    a, b: (x, y)
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def cluster_cones(cones):
    """
    간단한 BFS로 cones를 클러스터링.
    cones: [(lx1, ly1), (lx2, ly2), ...]
    return: [ [cluster1], [cluster2], ... ]
    """
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

                # 근접한 점 탐색
                for j in range(len(cones)):
                    if not visited[j]:
                        dist = euclidean_distance(cones[idx], cones[j])
                        if dist < CLUSTER_DISTANCE_THRESHOLD:
                            visited[j] = True
                            queue.append(j)

            clusters.append(cluster)
    return clusters

def draw_cluster_bounding_boxes(img, clusters):
    """
    각 클러스터에 대해 바운딩 박스 그려줌.
    클러스터 내 콘 개수가 DENSE_CLUSTER_MIN_SIZE 이상이면 (밀집) → 빨간 박스
    아니면 파란 박스
    """
    for cluster in clusters:
        if not cluster:
            continue

        # 클러스터 내 min/max
        lx_vals = [p[0] for p in cluster]
        ly_vals = [p[1] for p in cluster]

        min_lx, max_lx = min(lx_vals), max(lx_vals)
        min_ly, max_ly = min(ly_vals), max(ly_vals)

        # 픽셀 좌표로 변환
        tl = local_to_pixel(min_lx, max_ly)   # top-left
        br = local_to_pixel(max_lx, min_ly)   # bottom-right

        cluster_size = len(cluster)
        if cluster_size >= DENSE_CLUSTER_MIN_SIZE:
            color = (0, 0, 255)  # 빨강 (밀집)
        else:
            color = (255, 0, 0)  # 파랑 (일반)

        # 바운딩 박스
        cv2.rectangle(img, tl, br, color, 2)
        # 클러스터 크기 표시
        cv2.putText(img, f"Cluster:{cluster_size}",
                    (tl[0], tl[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    rospy.init_node("perception_node", anonymous=True)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("control")  # 패키지명에 맞게 수정
    csv_path_cones = pkg_path + "/data/cones_with_obstacles2.csv"

    try:
        cones_data = pd.read_csv(csv_path_cones).dropna()
        cones_x = cones_data['x'].to_numpy()
        cones_y = cones_data['y'].to_numpy()
        rospy.loginfo("Loaded {} cones from CSV.".format(len(cones_x)))
    except Exception as e:
        rospy.logerr("Failed to read cones CSV: {}".format(e))
        cones_x = []
        cones_y = []

    rospy.Subscriber("vehicle_pose", PoseStamped, vehicle_pose_callback)

    rate = rospy.Rate(10)  # 10Hz

    while not rospy.is_shutdown():
        # 흰색 배경 이미지
        img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

        # 1) ROI 필터링
        filtered_cones = filter_obstacles_by_roi(cones_x, cones_y)

        # 2) 축, 차량 그리기 (기존)
        draw_axes(img)
        draw_car(img)

        # 3) 콘 시각화 (기존)
        draw_cones(img, filtered_cones)

        # 4) 콘 클러스터링
        clusters = cluster_cones(filtered_cones)

        # 5) 바운딩 박스 표시 (밀집 클러스터 판별)
        draw_cluster_bounding_boxes(img, clusters)

        # 화면 표시
        cv2.imshow("Perception with Clustering", img)
        if cv2.waitKey(1) == 27:  # ESC 키
            break

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
