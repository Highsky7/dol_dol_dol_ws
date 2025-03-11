#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import argparse
import time
import cv2
import torch
import numpy as np
import cv2.ximgproc as ximgproc
import torch.backends.cudnn as cudnn
import threading
import queue
from math import atan2, degrees
from pathlib import Path
import matplotlib.pyplot as plt

# ROS 메시지
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32  # 조향각 퍼블리시용

# ========== (프로젝트 내 다른 유틸) ==========
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    lane_line_mask,  # YOLOPv2 세그멘테이션 결과 기반 이진화
    AverageMeter,
    LoadCamera,
    LoadImages,
    letterbox,
)

# ===================================================
# argparse 설정 (추가 인자: lookahead, wheelbase, debug)
# ---------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='model.pt 경로')
    parser.add_argument('--source', type=str,
                        # default='/home/highsky/Videos/Webcam/직선.mp4',
                        default='0',
                        help='source: 0(webcam) 또는 영상/이미지 파일 경로')
    parser.add_argument('--img-size', type=int, default=640,
                        help='YOLO 추론 해상도')
    parser.add_argument('--device', default='0',
                        help='cuda device: 0 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.5,
                        help='차선 세그 임계값 (0.0~1.0)')
    parser.add_argument('--nosave', action='store_false',
                        help='저장하지 않으려면 사용')
    parser.add_argument('--project', default='runs/detect',
                        help='결과 저장 폴더')
    parser.add_argument('--name', default='exp',
                        help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_false',
                        help='기존 폴더 사용 허용')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='프레임 건너뛰기 (0이면 건너뛰지 않음)')
    parser.add_argument('--param-file', type=str,
                        default='/home/highsky/dol_dol_dol_ws/bev_params.npz',
                        help='BEV 파라미터 (src_points, dst_points, warp_w, warp_h)')
    # 디버그 옵션: matplotlib을 이용해 차량 좌표계에서 차선 시각화
    parser.add_argument('--debug', action='store_true',
                        help='Matplotlib을 사용하여 차량 좌표계에서 차선 시각화 (Pure Pursuit 디버깅)')

    return parser

# ===================================================
# Extended Kalman Filter 클래스 (급격한 곡률 변화를 고려)
# ---------------------------------------------------
class LaneExtendedKalmanFilter:
    """
    2차 다항식 계수 [a, b, c]와 그 변화율 [da, db, dc]를 추적하기 위한 확장 칼만 필터.
    상태 벡터: x = [a, b, c, da, db, dc]^T
    관측: z = [a, b, c]
    
    상태 전이 모델 (비선형):
      a_{k+1} = a_k + da_k*dt + 0.5*sin(a_k)*dt^2
      b_{k+1} = b_k + db_k*dt
      c_{k+1} = c_k + dc_k*dt
      da_{k+1} = da_k + sin(a_k)*dt
      db_{k+1} = db_k
      dc_{k+1} = dc_k
    
    관측 모델:
      z = [a, b, c] (직접 관측)
    """
    def __init__(self, dt=0.033):
        self.dt = dt
        self.dim_x = 6  # [a, b, c, da, db, dc]
        self.dim_z = 3  # measurement: [a, b, c]
        
        # 초기 상태 벡터
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        
        # 상태 오차 공분산 초기값
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        
        # 프로세스 노이즈 공분산 (필요에 따라 조정)
        self.Q = np.eye(self.dim_x, dtype=np.float32) * 0.1
        
        # 측정 노이즈 공분산
        self.R = np.eye(self.dim_z, dtype=np.float32) * 5.0
        
        self.initialized = False

    def reset(self):
        self.x[:] = 0
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.initialized = False

    def predict(self):
        dt = self.dt
        # 상태 벡터: [a, b, c, da, db, dc]
        a = self.x[0,0]
        b = self.x[1,0]
        c = self.x[2,0]
        da = self.x[3,0]
        db = self.x[4,0]
        dc = self.x[5,0]
        
        # 비선형 상태 전이 함수 f(x)
        a_pred = a + da*dt + 0.5 * np.sin(a) * (dt**2)
        b_pred = b + db*dt
        c_pred = c + dc*dt
        da_pred = da + np.sin(a) * dt
        db_pred = db  # 선형으로 가정
        dc_pred = dc
        
        x_pred = np.array([[a_pred],
                           [b_pred],
                           [c_pred],
                           [da_pred],
                           [db_pred],
                           [dc_pred]], dtype=np.float32)
        self.x = x_pred
        
        # 상태 전이 함수의 Jacobian F = df/dx
        F = np.eye(self.dim_x, dtype=np.float32)
        # a_pred = a + da*dt + 0.5*sin(a)*dt^2
        F[0,0] = 1 + 0.5 * np.cos(a) * (dt**2)  # ∂a_pred/∂a
        F[0,3] = dt                           # ∂a_pred/∂da
        
        # da_pred = da + sin(a)*dt
        F[3,0] = np.cos(a) * dt               # ∂da_pred/∂a
        F[3,3] = 1                            # ∂da_pred/∂da
        
        # b_pred = b + db*dt
        F[1,1] = 1
        F[1,4] = dt
        
        # c_pred = c + dc*dt
        F[2,2] = 1
        F[2,5] = dt
        
        # db_pred, dc_pred는 상수이므로 F[4,4]와 F[5,5]는 이미 1
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        관측: z = [a, b, c]
        측정 모델: h(x) = [a, b, c]
        """
        z = np.array(z, dtype=np.float32).reshape(self.dim_z, 1)
        if not self.initialized:
            # 초기 관측으로 상태를 초기화, 속도 성분은 0으로 설정
            self.x[0,0] = z[0,0]
            self.x[1,0] = z[1,0]
            self.x[2,0] = z[2,0]
            self.x[3,0] = 0.0
            self.x[4,0] = 0.0
            self.x[5,0] = 0.0
            self.initialized = True
            return
        
        # 측정 모델: h(x) = [a, b, c]
        h = np.array([[self.x[0,0]],
                      [self.x[1,0]],
                      [self.x[2,0]]], dtype=np.float32)
        # 잔차: y = z - h(x)
        y = z - h
        
        # 측정 모델의 Jacobian H = dh/dx, 3x6 행렬
        H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        H[0,0] = 1
        H[1,1] = 1
        H[2,2] = 1
        
        S = H @ self.P @ H.T + self.R  # 잔차 공분산
        K = self.P @ H.T @ np.linalg.inv(S)  # 칼만 이득
        
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

# ===================================================
# 폴리피팅 유틸
# ---------------------------------------------------
def polyfit_lane(points_y, points_x, order=2):
    """
    points_y: 세로 방향(전방), points_x: 가로 방향(차폭)
    2차 다항식 x = a*(y^2) + b*y + c 형태로 피팅 (OpenCV 좌표계상 y가 아래로 증가한다고 가정)
    """
    if len(points_y) < 5:
        return None
    fit_coeff = np.polyfit(points_y, points_x, order)
    return fit_coeff  # [a, b, c] (2차)

def eval_poly(coeff, y_vals):
    """ coeff = [a, b, c], y_vals는 배열 → x = a*y^2 + b*y + c """
    if coeff is None:
        return None
    x_vals = np.polyval(coeff, y_vals)
    return x_vals

def extract_lane_points(bev_mask):
    """
    전체 차선 마스크에서 픽셀 좌표 (y, x)를 추출
    (시연용으로 단순히 모든 픽셀을 통째로 피팅; 실제로는 좌/우 차선 분리 가능)
    """
    ys, xs = np.where(bev_mask > 0)
    return ys, xs

# ===================================================
# 모폴로지 및 연결요소 기반 필터
# ---------------------------------------------------
def morph_open(binary_mask, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 2:
        return binary_mask
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    comps.sort(key=lambda x: x[1], reverse=True)
    keep_indices = [i for i, area in comps[:2] if area >= min_area]
    cleaned = np.zeros_like(binary_mask)
    for idx in keep_indices:
        cleaned[labels == idx] = 255
    return cleaned

# def line_fit_filter(binary_mask, max_line_fit_error=2.0, min_angle_deg=70.0, max_angle_deg=110.0):
#     h, w = binary_mask.shape
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
#     out_mask = np.zeros_like(binary_mask)
#     for i in range(1, num_labels):
#         comp_mask = (labels == i).astype(np.uint8)
#         if stats[i, cv2.CC_STAT_AREA] < 5:
#             continue
#         ys, xs = np.where(comp_mask > 0)
#         pts = np.column_stack((xs, ys)).astype(np.float32)
#         if len(pts) < 2:
#             continue
#         line_param = cv2.fitLine(pts, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
#         vx, vy, x0, y0 = line_param.flatten()
#         angle_deg = abs(degrees(atan2(vy, vx)))
#         if angle_deg > 180:
#             angle_deg -= 180
#         if not (min_angle_deg <= angle_deg <= max_angle_deg):
#             continue
#         norm_len = (vx**2 + vy**2)**0.5
#         if norm_len < 1e-12:
#             continue
#         vx_n, vy_n = vx/norm_len, vy/norm_len
#         dist_sum = sum(abs((xx - x0)*(-vy_n) + (yy - y0)*vx_n) for xx, yy in pts)
#         if (dist_sum / len(pts)) <= max_line_fit_error:
#             out_mask[labels == i] = 255
#     return out_mask

def final_filter(bev_mask):
    # f1 = morph_open(bev_mask, ksize=3)
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=300)
    f4 = keep_top2_components(f3, min_area=300)
    # f5 = line_fit_filter(f4, max_line_fit_error=2.0, min_angle_deg=10.0, max_angle_deg=170.0)
    return f4

# ===================================================
# BEV 변환 함수
# ---------------------------------------------------
def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev = cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)
    return bev

# ===================================================
# 폴리라인 점들을 계산하는 함수
# ---------------------------------------------------
def compute_polyline_points(coeff, image_shape, step=4):
    """ 주어진 다항식 coeff에 대해, image_shape 내에서 (x,y) 점들을 리스트로 계산 """
    h, w = image_shape[:2]
    points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w:
            points.append((int(x), int(y)))
    return points

# ===================================================
# 수정된 overlay_polyline: translation 인자 추가
# ---------------------------------------------------
def overlay_polyline(image, coeff, color=(0, 0, 255), step=4, translation=(0,0)):
    """
    2차 다항식 coeff에 맞춰 y좌표를 순회하며 x좌표를 구한 후,
    translation 벡터를 적용하여 image 위에 폴리라인을 그려줌.
    """
    if coeff is None:
        return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w:
            # translation 적용
            draw_points.append((int(x + translation[0]), int(y + translation[1])))
    if len(draw_points) > 1:
        for i in range(len(draw_points) - 1):
            cv2.line(image, draw_points[i], draw_points[i+1], color, 2)
    return image

# Pure Pursuit 결과를 Matplotlib으로 디버깅 시각화
def debug_plot_lane(shifted_poly_points, image_to_vehicle, goal_point=None):
    lane_vehicle = [image_to_vehicle(pt) for pt in shifted_poly_points]
    if len(lane_vehicle) > 0:
        lane_vehicle = np.array(lane_vehicle)

        plt.figure("Lane in Vehicle Coordinates", figsize=(6, 6))
        plt.clf()
        plt.plot(lane_vehicle[:, 1], lane_vehicle[:, 0], 'r-', label="Lane")
        if goal_point is not None:
            plt.scatter(goal_point[1], goal_point[0], color='green', s=100, label="Goal Point")  # 목표점
        # 축 설정
        plt.xlabel("Lateral (m)")
        plt.ylabel("Forward (m)")
        plt.title("Lane Line in Vehicle Coordinates")
        plt.legend()
        
        # x축 반전: 좌측이 양수로 보이도록 설정
        plt.gca().invert_xaxis()
        plt.xlim(1.0, -1.0)
        plt.ylim(0.0, 3.0)
        
        plt.grid(True)
        plt.show(block=False)  # Matplotlib 창 띄우기
        plt.pause(0.001)  # 창 업데이트


# ===================================================
# 메인 처리 함수: detect_and_publish (pub_steering 추가)
# ---------------------------------------------------
def detect_and_publish(opt, pub_mask, pub_steering):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
    cudnn.benchmark = True

    bridge = CvBridge()
    source, weights = opt.source, opt.weights
    imgsz = opt.img_size
    lane_threshold = opt.lane_thres
    bev_param_file = opt.param_file

    # 결과 저장 폴더 생성
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    vid_path = None
    vid_writer = None
    current_save_size = None

    # 모델 로드 및 GPU 최적화
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # Extended Kalman Filter (2차 계수 및 변화율 추적) 인스턴스
    kf = LaneExtendedKalmanFilter(dt=0.033)

    # 입력 소스 결정 (웹캠 vs. 파일)
    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치=%s) 열기", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
    else:
        rospy.loginfo("[INFO] 파일(영상/이미지): %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # ──────────────────────────
    # 함수: 매 프레임 처리 로직
    # ──────────────────────────
    def process_frame(im0s):
        """
        im0s: BGR 원본 이미지
        1) YOLOPv2 추론
        2) lane_line_mask → thinning/필터
        3) BEV 변환 후 final_filter
        4) 연결 요소 기반으로 가장 큰 차선 영역만 선택하여 폴리피팅 수행
        5) 차선이 인식된 경우에만 Extended Kalman Filter 업데이트, 평행이동 및 Pure Pursuit 기반 조향각 계산 후 ROS 퍼블리시
        6) 최종 디스플레이 및 각 창 영상 반환
        """
        
        # 전처리: CLAHE + letterbox
        net_input_img, ratio, pad = letterbox(im0s, (imgsz, imgsz), stride=stride)
        net_input_img = net_input_img[:, :, ::-1].transpose(2, 0, 1)
        net_input_img = np.ascontiguousarray(net_input_img)

        # 모델 추론 준비
        img_t = torch.from_numpy(net_input_img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        # 추론
        t1 = time_synchronized()
        with torch.no_grad():
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        # 차선 세그멘테이션 → 이진화
        binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='otsu')
        #binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='fixed')
        
        # thinning (2D 라인화)
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if thin_mask is None or thin_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → binary_mask 사용")
            thin_mask = binary_mask

        # BEV 변환
        bev_mask = do_bev_transform(thin_mask, bev_param_file)
        
        # 추가 필터링
        bevfilter_mask = final_filter(bev_mask)
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if final_mask is None or final_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → bevfilter_mask 사용")
            final_mask = bevfilter_mask

        # ───────────────────────────────────────────────
        # 연결 요소 기반으로 차선 영역 선택:
        # 카메라에 보이는 차선들 중 배경(라벨 0)을 제외하고 가장 큰 영역 선택
        # ───────────────────────────────────────────────
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        if num_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            lane_mask = np.zeros_like(final_mask)
            lane_mask[labels == largest_label] = 255
            ys, xs = np.where(lane_mask > 0)
        else:
            ys, xs = np.where(final_mask > 0)

        coeff = polyfit_lane(ys, xs, order=2)  # [a, b, c]

        # BEV 변환 영상 (컬러)
        bev_im = do_bev_transform(im0s, bev_param_file)

        # Pure Pursuit 및 조향각 계산은 차선이 인식된 경우에만 수행
        if coeff is not None:
            # 1. 원래 폴리라인 점들 계산 (이미지 좌표계)
            poly_points = compute_polyline_points(coeff, bev_im.shape, step=4)
            if len(poly_points) > 0:
                # 폴리라인의 가장 아래 점(전형적으로 차량 위치에 가까운 점)
                bottom_point = poly_points[-1]  # (x, y)
                desired_start = (320, 640)  # 결과 영상에서 후륜축 중심
                # translation 벡터 계산: bottom_point가 desired_start에 오도록
                translation = (desired_start[0] - bottom_point[0], desired_start[1] - bottom_point[1])
            else:
                translation = (0, 0)

            # 2. 평행이동된 폴리라인 그리기
            bev_im_color = overlay_polyline(bev_im.copy(), coeff, color=(0, 0, 255), step=4, translation=translation)

            # 3. 평행이동된 폴리라인 점들 (리스트)
            shifted_poly_points = [(pt[0] + translation[0], pt[1] + translation[1]) for pt in poly_points]

            # 4. 이미지 좌표 -> 차량 좌표 변환 (새로운 좌표계)
            def image_to_vehicle(pt):
                u, v = pt  # 이미지 좌표 (u,v)
                x_vehicle = (640 - v) * 0.00234375 + 1.4
                y_vehicle = (320 - u) * 0.003125
                return x_vehicle, y_vehicle

            # 5. Pure Pursuit: 전방주시거리 (lookahead) 이상 떨어진 목표점 선택
    
            lookahead = 210.0 # cm 단위
            lookahead_m = lookahead/100.0
            wheelbase = 75.0  # cm 단위
            wheelbase_m = wheelbase/100.0
            goal_point = None
            min_error = float('inf') #초기값을 무한대로 설정
            for pt in shifted_poly_points:
                X_v, Y_v = image_to_vehicle(pt)
                d = np.sqrt(X_v**2 + Y_v**2)
                error = abs(d- lookahead_m)
                if error < min_error:
                    min_error = error
                    goal_point = (X_v, Y_v)
            if goal_point is None and len(shifted_poly_points) > 0:
                goal_point = image_to_vehicle(shifted_poly_points[-1])

            # 6. Pure Pursuit에 따른 조향각 계산 및 ROS 퍼블리시
            if goal_point is not None:
                X_v, Y_v = goal_point
                d = np.sqrt(X_v**2 + Y_v**2) #lookahead distance
                if d < 1e-6:
                    steering_angle = 0.0
                else:
                    alpha = np.arctan2(Y_v, X_v)  # 목표점과 전방 사이의 각도
                    steering_angle = np.arctan((2 * wheelbase_m * np.sin(alpha)) / d)
                
                # 라디안 값을 degree로 변환 후 퍼블리시
                steering_angle_deg = -np.degrees(steering_angle)
                pub_steering.publish(Float32(data=steering_angle_deg))
                rospy.loginfo("[INFO] Published auto_steer_angle_lane: %.2f deg", steering_angle_deg)

                # 시각화를 위해 목표점(차량 좌표계에서 이미지 좌표로 역변환) 표시
                goal_x_img = int(320 - goal_point[1] / 0.003125)
                goal_y_img = int(640 - (goal_point[0] - 1.4) / 0.00234375)
                cv2.circle(bev_im_color, (goal_x_img, goal_y_img), 5, (0, 255, 0), -1)
                cv2.putText(bev_im_color, f"Steering: {steering_angle_deg:.2f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            else:
                bev_im_color = overlay_polyline(bev_im.copy(), coeff, color=(0, 0, 255), step=4, translation=translation)

            # ───────────────────────────────
            # 디버그: matplotlib로 차선 플롯 확인
            opt.debug = True
            if opt.debug:
                debug_plot_lane(shifted_poly_points, image_to_vehicle, goal_point)
        else:
            bev_im_color = bev_im.copy()

        # 결과 디스플레이
        cv2.imshow("BEV + Polyfit", bev_im_color)
        cv2.imshow("final+mask", final_mask)

        # process_frame는 세 영상 모두 반환 (Thin Mask는 이진 영상이므로 BGR 변환하여 기록)
        return bev_im, bev_im_color, final_mask

    # ──────────────────────────
    # 웹캠 vs. 동영상/이미지 처리 분기
    # ──────────────────────────
    start_time = None

    if dataset.mode == 'stream':
        # 웹캠(스트림) 비동기 처리
        frame_skip = opt.frame_skip
        frame_counter = 0
        frame_queue = queue.Queue(maxsize=5)

        def frame_producer():
            for item in dataset:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(item)
            frame_queue.put(None)

        producer_thread = threading.Thread(target=frame_producer)
        producer_thread.daemon = True
        producer_thread.start()

        rospy.loginfo("[DEBUG] 웹캠 비동기 프레임 생산 시작")
        t0 = time.time()

        while not rospy.is_shutdown():
            try:
                frame_data = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame_data is None:
                rospy.loginfo("[INFO] 프레임 종료 신호 수신")
                break
            path_item, net_input_img, im0s, vid_cap = frame_data

            if frame_skip > 0:
                if frame_counter % (frame_skip + 1) != 0:
                    frame_counter += 1
                    continue
                frame_counter += 1

            if start_time is None:
                start_time = time.time()

            bev_im, bev_im_color, thin_mask = process_frame(im0s)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] q 입력 → 종료")
                break
  
    else:
        # 동기 처리 (영상/이미지) 천천히 보여줌
        rospy.loginfo("[DEBUG] 저장된 영상/이미지 동기 처리 시작")
        delay = 30
        if dataset.mode == 'video' and dataset.cap is not None:
            fps = dataset.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                delay = int(1000 / fps)
        start_time = time.time()

        for frame_data in dataset:
            path_item, net_input_img, im0s, vid_cap = frame_data
            bev_im, bev_im_color, thin_mask = process_frame(im0s)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] q 입력 → 종료")
                break
          
        rospy.loginfo("[INFO] 동기 처리 추론 평균 시간: %.4fs/frame, 전체: %.3fs", inf_time.avg, time.time()-start_time)

    cv2.destroyAllWindows()
    rospy.loginfo("[INFO] 추론 완료.")

# ===================================================
# 메인 ros_main
# ---------------------------------------------------
def ros_main():
    rospy.init_node('bev_lane_thinning_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    # 디버그 모드에서 matplotlib interactive 모드 활성화
    if opt.debug:
        plt.ion()
    pub_mask = rospy.Publisher('camera_lane_segmentation/lane_mask', Image, queue_size=1)
    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    detect_and_publish(opt, pub_mask, pub_steering)
    rospy.loginfo("[INFO] bev_lane_thinning_node 종료, spin() 호출")
    rospy.spin()

# ===================================================
# 메인 실행
# ---------------------------------------------------
if __name__=='__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
