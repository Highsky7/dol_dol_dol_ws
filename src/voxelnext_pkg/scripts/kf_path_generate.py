#!/home/highsky/lidar_env/bin/python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header

# --------------------------
# 전역 변수
# --------------------------
pub_path = None   # 경로 퍼블리셔

# EKF 상태
kf_inited = False
x_state  = None   # 상태벡터 [a0, a1, a2, a3]
P_state  = None   # 공분산(4x4)

# --------------------------
# 1) Frenet 변환 (heading=0 가정)
#    s=x, n=y 로 단순 적용
# --------------------------
def xy_to_frenet(x, y):
    return x, y

def frenet_to_xy(s, n):
    return s, n

# --------------------------
# 2) 좌우 콘 분리 + 후방 제거
#    - s>0만 사용
#    - n>0 => 왼쪽, n<0 => 오른쪽
# --------------------------
def filter_and_separate_cones(cones_xy):
    left_cones = []
    right_cones = []
    for (x, y) in cones_xy:
        s, n = xy_to_frenet(x, y)
        if s > 0:  # 후방 제거
            if n >= 0:
                left_cones.append([s, n])
            else:
                right_cones.append([s, n])
    return np.array(left_cones), np.array(right_cones)

# --------------------------
# 3) 좌/우 콘 s 오름차순 정렬 후 중간점(midpoints) 계산
# --------------------------
def generate_midpoints(left_sn, right_sn):
    if len(left_sn) == 0 or len(right_sn) == 0:
        return np.array([])
    left_sorted  = left_sn[np.argsort(left_sn[:,0])]
    right_sorted = right_sn[np.argsort(right_sn[:,0])]
    min_len = min(len(left_sorted), len(right_sorted))
    mids = []
    for i in range(min_len):
        sL, nL = left_sorted[i]
        sR, nR = right_sorted[i]
        s_mid = (sL + sR)/2.0
        n_mid = (nL + nR)/2.0
        mids.append([s_mid, n_mid])
    mids = np.array(mids)
    if len(mids)>1:
        # s 오름차순 정렬
        mids = mids[np.argsort(mids[:,0])]
    return mids

# --------------------------
# 4) EKF 초기화
#    - midpoints로부터 3차 다항식 근사 -> 상태 x=[a0,a1,a2,a3]
#    - 혹은 점이 너무 적으면 차수 낮춰서 fit한 뒤, 4계수로 패딩
# --------------------------
def init_ekf(midpoints):
    # 전역 사용
    global kf_inited, x_state, P_state

    # 만약 midpoints가 충분하면 3차 다항식
    # 부족하면 가능한 차수로 fit 후 남는 계수는 0으로
    M = len(midpoints)
    if M>=4:
        deg = 3
    elif M==3:
        deg = 2
    elif M==2:
        deg = 1
    else:
        deg = 0  # 한 점 이하라면...

    s_vals = midpoints[:,0]
    n_vals = midpoints[:,1]
    if M>=1:
        coefs = np.polyfit(s_vals, n_vals, deg)  # [a_deg,...,a_0]
    else:
        # midpoints가 전혀 없으면 그냥 0차원
        coefs = np.array([0.0])

    # polyfit 결과를 길이 4로 만들기(3차)
    # np.polyfit은 [a_deg ... a_0] 순서
    # a3, a2, a1, a0
    a0,a1,a2,a3 = 0,0,0,0
    for i in range(len(coefs)):
        # coefs[0] = a_deg
        # coefs[-1] = a_0
        a_idx = deg - i  # 거꾸로 인덱스
        val = coefs[i]
        if a_idx==3: a3 = val
        elif a_idx==2: a2 = val
        elif a_idx==1: a1 = val
        elif a_idx==0: a0 = val

    x_state = np.array([a0,a1,a2,a3], dtype=float)
    P_state = np.eye(4)*10.0  # 초기 공분산 크게

    kf_inited = True
    rospy.loginfo("EKF initialized with polynomial: a0=%.3f, a1=%.3f, a2=%.3f, a3=%.3f" % (a0,a1,a2,a3))

# --------------------------
# 5) EKF 예측 단계 (단순: x=그대로, P=P+Q)
# --------------------------
def ekf_predict(x, P, Q):
    x_pred = x  # 별도 동역학이 없으므로 그대로
    P_pred = P + Q
    return x_pred, P_pred

# --------------------------
# 6) EKF 측정 업데이트 (확장 or 선형)
#    여기서는 midpoints (s_i, n_i)를 측정으로 사용
#    측정함수: z_i = n_i
#    예측: h_i(x) = a0 + a1 s_i + a2 s_i^2 + a3 s_i^3
#    실제로는 선형이지만, EKF 형식으로 구현
# --------------------------
def ekf_update(x_pred, P_pred, midpoints, R_std=0.1):
    # midpoints: (M x 2), s_i, n_i
    M = len(midpoints)
    if M==0:
        # 측정 없음 -> 예측만
        return x_pred, P_pred

    s_vals = midpoints[:,0]
    n_meas = midpoints[:,1]

    # h(x_pred), H (Jacobian) 계산
    # h: (M,) = [h_1, ..., h_M]
    # H: (M x 4)
    h = np.zeros(M)
    H = np.zeros((M,4))
    a0,a1,a2,a3 = x_pred
    for i in range(M):
        s = s_vals[i]
        # h_i
        h[i] = a0 + a1*s + a2*(s**2) + a3*(s**3)
        # Jacobian wrt [a0,a1,a2,a3]
        H[i,0] = 1.0
        H[i,1] = s
        H[i,2] = s**2
        H[i,3] = s**3

    # 잔차
    z = n_meas  # (M,)
    y = z - h   # (M,)

    # 측정 잡음 공분산 R (M x M)
    # 여기서는 각 측정 독립 가정 -> 대각행렬
    R = np.eye(M)*(R_std**2)

    # S = H P_pred H^T + R  (MxM)
    S = H @ P_pred @ H.T + R

    # K = P_pred H^T S^-1  (4xM)
    K = P_pred @ H.T @ np.linalg.inv(S)

    # x_new = x_pred + K y
    x_new = x_pred + K @ y

    # P_new = (I - K H) P_pred
    I = np.eye(4)
    P_new = (I - K @ H) @ P_pred

    return x_new, P_new

# --------------------------
# 7) 상태 x=[a0,a1,a2,a3]로부터
#    s_min ~ s_max 구간에서 (x,y) 경로 생성
# --------------------------
def generate_path(x, s_min, s_max, num_points=50):
    # s 범위 제한
    if s_min < 0:
        s_min = 0.0
    s_max = min(s_max, 3.0)  # 최대 30m 정도로 제한
    if s_min >= s_max:
        s_max = s_min + 1.0

    a0,a1,a2,a3 = x
    s_vals = np.linspace(s_min, s_max, num_points)
    path_xy = []
    for s in s_vals:
        n = a0 + a1*s + a2*(s**2) + a3*(s**3)
        xw, yw = frenet_to_xy(s, n)
        path_xy.append([xw, yw])
    return np.array(path_xy)

# --------------------------
# 8) Marker 발행
# --------------------------
def publish_path(path_xy):
    global pub_path
    marker = Marker()
    marker.header = Header(frame_id="velodyne", stamp=rospy.Time.now())
    marker.ns = "frenet_ekf_path"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.15
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0

    pts = []
    for (xx, yy) in path_xy:
        p = Point()
        p.x = xx
        p.y = yy
        p.z = 0.0
        pts.append(p)
    marker.points = pts
    marker.lifetime = rospy.Duration(0)
    pub_path.publish(marker)

# --------------------------
# 9) /center_markers 콜백
#    - 좌우 콘 -> midpoints -> EKF predict+update -> 경로 생성
# --------------------------
def center_markers_callback(msg):
    global kf_inited, x_state, P_state

    # 1) 콘 좌표 수집
    cones_xy = []
    for mk in msg.markers:
        cx = mk.pose.position.x
        cy = mk.pose.position.y
        cones_xy.append([cx, cy])
    cones_xy = np.array(cones_xy)

    # 콘이 거의 없으면 -> 이전 경로만 유지
    if len(cones_xy) < 2:
        rospy.logwarn("Not enough cones. Reuse old path if possible.")
        if kf_inited:
            # 예측만 수행
            Q = np.eye(4)*0.001
            x_pred, P_pred = ekf_predict(x_state, P_state, Q)
            x_state, P_state = x_pred, P_pred
            # 이전 상태로 경로 생성
            path_xy = generate_path(x_state, 0.0, 10.0, 50)
            publish_path(path_xy)
        return

    # 2) 좌우 분리
    left_sn, right_sn = filter_and_separate_cones(cones_xy)
    if len(left_sn)==0 or len(right_sn)==0:
        rospy.logwarn("No valid left/right cones. Reuse old path if possible.")
        if kf_inited:
            Q = np.eye(4)*0.001
            x_pred, P_pred = ekf_predict(x_state, P_state, Q)
            x_state, P_state = x_pred, P_pred
            path_xy = generate_path(x_state, 0.0, 10.0, 50)
            publish_path(path_xy)
        return

    # 3) midpoints
    mids = generate_midpoints(left_sn, right_sn)
    if len(mids)<1:
        rospy.logwarn("No midpoints. Reuse old path if possible.")
        if kf_inited:
            Q = np.eye(4)*0.001
            x_pred, P_pred = ekf_predict(x_state, P_state, Q)
            x_state, P_state = x_pred, P_pred
            path_xy = generate_path(x_state, 0.0, 10.0, 50)
            publish_path(path_xy)
        return

    # 4) EKF 초기화(최초 1회)
    if not kf_inited:
        init_ekf(mids)  # x_state, P_state 세팅

    # 5) EKF predict
    Q = np.eye(4)*0.001  # 프로세스 잡음(임의값)
    x_pred, P_pred = ekf_predict(x_state, P_state, Q)

    # 6) EKF update (확장칼만필터 형식이지만 실제로는 선형)
    #    콘 개수 적으면 R 크게 하여 이전 추세 더 신뢰
    M = len(mids)
    if M<3:
        r_std = 0.5
    else:
        r_std = 0.1

    x_upd, P_upd = ekf_update(x_pred, P_pred, mids, R_std=r_std)

    # 상태 갱신
    x_state, P_state = x_upd, P_upd

    # 7) 경로 생성
    s_min = np.min(mids[:,0])
    s_max = np.max(mids[:,0]) + 2.0
    path_xy = generate_path(x_state, s_min, s_max, 50)

    # 8) 발행
    publish_path(path_xy)

# --------------------------
# (옵션) 시작/목표점 콜백
# --------------------------
def update_start_goal(msg):
    # 필요하다면 구현
    pass

# --------------------------
# 메인
# --------------------------
def main():
    global pub_path
    rospy.init_node('frenet_ekf_path_node', anonymous=True)
    rospy.loginfo("Frenet EKF path node started.")

    pub_path = rospy.Publisher('/path', Marker, queue_size=10)
    rospy.Subscriber('/center_markers', MarkerArray, center_markers_callback)
    rospy.Subscriber('/start_goal', Marker, update_start_goal)

    rospy.spin()

if __name__ == '__main__':
    main()
