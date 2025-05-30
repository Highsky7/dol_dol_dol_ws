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
from std_msgs.msg import Bool    # 차선 검출 상태 퍼블리시용

# 프로젝트 내 유틸
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    lane_line_mask,
    AverageMeter,
    LoadCamera,
    LoadImages,
    letterbox,
)

# argparse 설정
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolopv2.pt', help='model.pt 경로')
    parser.add_argument('--source', type=str,
                        default='2',
                        # default='/home/highsky/Videos/Webcam/좌회전X.mp4',
                        help='source: 0(webcam) 또는 영상/이미지 파일 경로')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO 추론 해상도')
    parser.add_argument('--device', default='0', help='cuda device: 0 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.5, help='차선 세그 임계값 (0.0~1.0)')
    parser.add_argument('--nosave', action='store_false', help='저장하지 않으려면 사용')
    parser.add_argument('--project', default='runs/detect', help='결과 저장 폴더')
    parser.add_argument('--name', default='exp', help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_false', help='기존 폴더 사용 허용')
    parser.add_argument('--frame-skip', type=int, default=0, help='프레임 건너뛰기 (0이면 건너뛰지 않음)')
    parser.add_argument('--param-file', type=str, default='./bev_params.npz', help='BEV 파라미터')
    parser.add_argument('--debug', action='store_true', help='Matplotlib을 사용하여 차선 시각화')
    return parser

# 폴리피팅 유틸
def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: # 충분한 포인트가 없으면 피팅 불가
        return None
    try:
        return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError): # 피팅 중 에러 발생 시
        return None


def compute_polyline_points(coeff, image_shape, step=4):
    h, w = image_shape[:2]
    points = []
    if coeff is None:
        return points
    for y in range(0, h, step): # y를 위에서 아래로 스캔
        x = np.polyval(coeff, y)
        if 0 <= x < w:
            points.append((int(x), int(y)))
    return points

def overlay_polyline(image, coeff, color=(0, 0, 255), step=4, translation=(0,0)):
    if coeff is None:
        return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w: # x 좌표가 이미지 너비 내에 있는지 확인
            draw_points.append((int(x + translation[0]), int(y + translation[1])))
    if len(draw_points) > 1:
        cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, 2)
    return image

# 모폴로지 및 필터링 함수
def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels): # 0번 레이블은 배경
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: # 배경만 있거나 컴포넌트가 없는 경우
        return np.zeros_like(binary_mask)
    
    comps = []
    for i in range(1, num_labels): # 배경 제외
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            comps.append((i, stats[i, cv2.CC_STAT_AREA]))
            
    comps.sort(key=lambda x: x[1], reverse=True) # 면적 기준으로 내림차순 정렬
    
    cleaned = np.zeros_like(binary_mask)
    for i in range(min(len(comps), 2)): # 상위 최대 2개 컴포넌트만 유지
        idx = comps[i][0]
        cleaned[labels == idx] = 255
    return cleaned


def final_filter(bev_mask):
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=300) # 1차 작은 노이즈 제거
    f4 = keep_top2_components(f3, min_area=300)   # 상위 2개 주요 차선 후보만 남김
    return f4

# BEV 변환 함수
def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

# 디버깅 시각화
def debug_plot_lane(shifted_poly_points, image_to_vehicle_func, goal_point=None):
    if not shifted_poly_points: # 리스트가 비어있으면 플롯하지 않음
        return
        
    lane_vehicle = [image_to_vehicle_func(pt) for pt in shifted_poly_points]
    if len(lane_vehicle) > 0:
        lane_vehicle = np.array(lane_vehicle)
        plt.figure("Lane in Vehicle Coordinates", figsize=(6, 6))
        plt.clf()
        plt.plot(lane_vehicle[:, 1], lane_vehicle[:, 0], 'r-', label="Processed Lane") # Y, X 순서로 플롯 (Y: Lateral, X: Forward)
        if goal_point is not None:
            plt.scatter(goal_point[1], goal_point[0], color='green', s=100, label="Goal Point")
        plt.xlabel("Lateral (m)")
        plt.ylabel("Forward (m)")
        plt.title("Lane Line in Vehicle Coordinates")
        plt.legend()
        plt.gca().invert_xaxis() # 차량 오른쪽이 양수 Y가 되도록 X축 반전
        plt.xlim(1.5, -1.5)  # 측방 범위 조정 (예: -1.5m ~ 1.5m)
        plt.ylim(0.0, 3.5)  # 전방 범위 조정 (예: 0m ~ 3.5m)
        plt.grid(True)
        plt.axis('equal')
        plt.show(block=False)
        plt.pause(0.001)

# 메인 처리 함수
def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0) # 멀티스레딩 비활성화 (OpenCV 내부)
    cudnn.benchmark = True

    bridge = CvBridge()
    source, weights = opt.source, opt.weights
    imgsz = opt.img_size
    lane_threshold = opt.lane_thres
    bev_param_file = opt.param_file

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # kf = LaneExtendedKalmanFilter(dt=0.033) # EKF 현재 미사용

    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치=%s) 열기", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=32)
    else:
        rospy.loginfo("[INFO] 파일(영상/이미지): %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=32)

    # BEV 이미지 파라미터 로드 (image_to_vehicle 및 역변환에 사용될 수 있음)
    bev_params = np.load(bev_param_file)
    bev_h_expected = int(bev_params['warp_h']) # 예: 640
    bev_w_expected = int(bev_params['warp_w']) # 예: 640

    # image_to_vehicle 변환 함수 (BEV 이미지 크기에 대한 의존성 명시)
    # 이 값들은 사용자의 BEV 설정에 따라 달라질 수 있으므로 주의.
    # 현재는 하드코딩된 값 사용 (기존 코드와 동일)
    m_per_pixel_y = 0.00234375 # (H_bev - v_bev) * mpp_y + y_offset_m
    y_offset_m = 1.4
    m_per_pixel_x = 0.003125    # (W_bev/2 - u_bev) * mpp_x
    
    def image_to_vehicle(pt_bev): # pt_bev = (u, v) 또는 (x_img, y_img)
        u, v = pt_bev
        # BEV 이미지의 원점은 좌상단. y는 아래로 증가, x는 오른쪽으로 증가.
        # 차량 좌표계: x축은 전방, y축은 차량 중심 기준 좌측이 양수 (또는 우측이 양수, 여기선 우측 양수로 가정하고 플롯에서 반전)
        # X_vehicle: 전방 거리 (m)
        # Y_vehicle: 측방 거리 (m)
        # v (이미지의 y좌표)가 클수록 차량에 가까움. bev_h_expected는 이미지의 전체 높이.
        x_vehicle = (bev_h_expected - v) * m_per_pixel_y + y_offset_m
        y_vehicle = (bev_w_expected / 2 - u) * m_per_pixel_x # u가 W/2보다 작으면 양수 (차량 좌측), 크면 음수 (차량 우측)
        return x_vehicle, y_vehicle


    frame_count = 0
    def process_frame(im0s):
        nonlocal frame_count
        frame_count += 1
        net_input_img, ratio, pad = letterbox(im0s, (imgsz, imgsz), stride=32)
        net_input_img = net_input_img[:, :, ::-1].transpose(2, 0, 1)
        net_input_img = np.ascontiguousarray(net_input_img)
        img_t = torch.from_numpy(net_input_img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            [_, _], seg, ll = model(img_t) # ll은 차선 세그멘테이션 결과
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='otsu')
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_GUOHALL)
        if thin_mask is None or thin_mask.size == 0 or np.sum(thin_mask) == 0:
            rospy.logwarn_throttle(1.0, "[WARNING] Thinning 결과 비어 있음 (GUOHALL) → binary_mask 사용 시도")
            thin_mask = binary_mask # 원본 이진 마스크 사용

        bev_mask_orig = do_bev_transform(thin_mask, bev_param_file)
        bevfilter_mask = final_filter(bev_mask_orig) # 주요 차선 후보 마스크 (최대 2개)
        
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_GUOHALL)
        if final_mask is None or final_mask.size == 0 or np.sum(final_mask) == 0:
            rospy.logwarn_throttle(1.0,"[WARNING] 최종 Thinning 결과 비어 있음 → bevfilter_mask 사용")
            final_mask = bevfilter_mask

        # === 경로 시작점 및 차선 처리 로직 수정 시작 ===
        bev_im_for_shape = do_bev_transform(im0s, bev_param_file) # 원본 BEV (컬러)
        h_bev, w_bev = bev_im_for_shape.shape[:2]
        bev_im_color = bev_im_for_shape.copy() # 시각화용 BEV 이미지

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        
        lane_components = []
        if num_labels > 1: # 배경 외 컴포넌트 존재
            sorted_components_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            
            for i in range(min(len(sorted_components_indices), 2)): # 상위 최대 2개
                label_idx = sorted_components_indices[i]
                if stats[label_idx, cv2.CC_STAT_AREA] >= 100: # 최소 픽셀 수 (final_filter의 min_area보다 작을 수 있음)
                    component_mask = np.zeros_like(final_mask)
                    component_mask[labels == label_idx] = 255
                    ys, xs = np.where(component_mask > 0)
                    if len(ys) > 4 : # polyfit을 위해 최소 5점 필요
                        coeff = polyfit_lane(ys, xs, order=2)
                        if coeff is not None:
                            lane_components.append({'ys': ys, 'xs': xs, 'coeff': coeff, 'label_idx': label_idx})
        
        # 각 원본 검출 차선 시각화 (초록색)
        for comp in lane_components:
            bev_im_color = overlay_polyline(bev_im_color, comp['coeff'], color=(0, 255, 0), step=4)

        lane_detected_bool = len(lane_components) > 0
        pub_lane_status.publish(Bool(data=lane_detected_bool))
        
        main_lane_poly_points = []
        path_start_point_bev = (w_bev // 2, h_bev -1) # 기본값: BEV 이미지 하단 중앙 (y는 맨 아래)
                                                      # h_bev-1 로 하여 이미지 범위 내에 있도록 함.

        if lane_detected_bool:
            active_coeff_for_steering = None # 퓨어퍼슛에 사용할 대표 계수 (또는 폴리라인)

            if len(lane_components) == 1:
                comp = lane_components[0]
                active_coeff_for_steering = comp['coeff']
                
                # 차선의 최대 y 좌표를 가지는 점 (BEV 이미지 하단에 가장 가까운 점)
                max_y_val = np.max(comp['ys'])
                # 해당 y값에서 x값 계산 (polyval 사용)
                # 주의: polyval은 y값을 입력으로 받음. comp['ys']는 정렬되어 있지 않을 수 있음.
                # coeff는 y에 대한 x의 함수 x = f(y)
                lane_bottom_x_at_max_y = int(np.polyval(comp['coeff'], max_y_val))
                lane_bottom_point = (lane_bottom_x_at_max_y, int(max_y_val))
                cv2.circle(bev_im_color, lane_bottom_point, 7, (255, 255, 0), -1) # 차선 하단점 (하늘색)

                # 기울기 추정 (dx/dy). coeff[1]이 1차항 계수 (y에 대한 x의 변화율)
                # 2차 다항식: ax^2 + bx + c. polyfit(y,x) -> ay^2 + by + c = x
                slope_estimate = comp['coeff'][1] if len(comp['coeff']) > 1 else 0 # 1차항 계수
                # 좀 더 안정적인 기울기: 차선 전체 y 범위에 걸쳐 x 변화량 계산
                min_y_for_slope, max_y_for_slope = np.min(comp['ys']), np.max(comp['ys'])
                if (max_y_for_slope - min_y_for_slope) > 10 : # y범위가 충분히 클때만
                     x_at_min_y = np.polyval(comp['coeff'], min_y_for_slope)
                     x_at_max_y = np.polyval(comp['coeff'], max_y_for_slope)
                     slope_estimate = (x_at_max_y - x_at_min_y) / (max_y_for_slope - min_y_for_slope + 1e-6)

                ref_point_name = "Center"
                if slope_estimate > 0.15: # 오른쪽으로 기움 (차량이 오른쪽으로 가야함) -> 경로는 왼쪽 기준선과 연결
                    ref_point = (0, h_bev -1) # BEV 좌하단
                    ref_point_name = "Left_Ref"
                elif slope_estimate < -0.15: # 왼쪽으로 기움 (차량이 왼쪽으로 가야함) -> 경로는 오른쪽 기준선과 연결
                    ref_point = (w_bev -1, h_bev -1) # BEV 우하단
                    ref_point_name = "Right_Ref"
                else: # 거의 수직
                    ref_point = (w_bev // 2, h_bev -1) # BEV 중앙 하단

                path_start_point_bev = ((lane_bottom_point[0] + ref_point[0]) // 2,
                                        (lane_bottom_point[1] + ref_point[1]) // 2)
                # cv2.putText(bev_im_color, f"1L Start: {ref_point_name} S:{slope_estimate:.2f}", (path_start_point_bev[0]-100, path_start_point_bev[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)


            elif len(lane_components) == 2:
                comp1, comp2 = lane_components[0], lane_components[1] # 면적 순 정렬됨
                
                max_y1_val = np.max(comp1['ys'])
                lane1_bottom_x = int(np.polyval(comp1['coeff'], max_y1_val))
                lane1_bottom_point = (lane1_bottom_x, int(max_y1_val))

                max_y2_val = np.max(comp2['ys'])
                lane2_bottom_x = int(np.polyval(comp2['coeff'], max_y2_val))
                lane2_bottom_point = (lane2_bottom_x, int(max_y2_val))
                
                cv2.circle(bev_im_color, lane1_bottom_point, 7, (255, 128, 0), -1)
                cv2.circle(bev_im_color, lane2_bottom_point, 7, (255, 128, 0), -1)

                path_start_point_bev = ( (lane1_bottom_point[0] + lane2_bottom_point[0]) // 2,
                                         (lane1_bottom_point[1] + lane2_bottom_point[1]) // 2 )
                
                # 두 차선의 계수를 평균내어 가상의 중간 차선 coeff 생성
                avg_coeff = (comp1['coeff'] + comp2['coeff']) / 2.0
                active_coeff_for_steering = avg_coeff
                # cv2.putText(bev_im_color, "2L Start: Avg", (path_start_point_bev[0]-50, path_start_point_bev[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)


            # --- `main_lane_poly_points` 생성 ---
            if active_coeff_for_steering is not None:
                # active_coeff_for_steering을 사용하여 폴리라인 생성
                poly_points_from_active_coeff = compute_polyline_points(active_coeff_for_steering, bev_im_color.shape, step=4)

                if len(poly_points_from_active_coeff) > 0:
                    # 이 폴리라인의 하단점 (y값이 가장 큰 점, 즉 리스트의 마지막 점)
                    # compute_polyline_points는 y를 0부터 h까지 증가시키므로, 마지막 점이 하단점.
                    current_polyline_bottom_point = poly_points_from_active_coeff[-1]
                    
                    # translation 계산: 폴리라인의 하단점을 path_start_point_bev로 이동
                    translation = (path_start_point_bev[0] - current_polyline_bottom_point[0],
                                   path_start_point_bev[1] - current_polyline_bottom_point[1])
                    
                    main_lane_poly_points = [(pt[0] + translation[0], pt[1] + translation[1]) for pt in poly_points_from_active_coeff]
                    
                    # 이동된 경로 시각화 (보라색)
                    if len(main_lane_poly_points) > 1:
                        cv2.polylines(bev_im_color, [np.array(main_lane_poly_points, dtype=np.int32)], False, (255, 0, 255), 2)
            
            cv2.circle(bev_im_color, path_start_point_bev, 10, (0, 0, 255), -1) # 최종 경로 시작점 (빨간색)


            # --- 퓨어퍼슛 알고리즘 적용 (main_lane_poly_points 사용) ---
            if len(main_lane_poly_points) > 0:
                lookahead_m = 2.10
                wheelbase_m = 0.75 # 차량 축거 (m)
                goal_point_vehicle_coords = None # (X_v, Y_v)
                min_dist_error = float('inf')

                # main_lane_poly_points는 이미지 좌표계 (u,v)
                # y가 작은 위쪽 점부터 y가 큰 아래쪽 점 순서로 정렬되어 있음.
                # 차량 좌표계에서는 y가 작은 점이 더 먼 점 (X_v가 큰 값)
                for pt_img in main_lane_poly_points:
                    X_v, Y_v = image_to_vehicle(pt_img) # 이미지좌표 -> 차량좌표
                    current_dist_from_vehicle = np.sqrt(X_v**2 + Y_v**2)
                    error = abs(current_dist_from_vehicle - lookahead_m)
                    
                    if error < min_dist_error:
                        min_dist_error = error
                        goal_point_vehicle_coords = (X_v, Y_v)
                
                # 만약 적절한 lookahead 지점을 못 찾았고 경로가 존재하면, 경로의 가장 먼 점을 목표로 설정
                if goal_point_vehicle_coords is None and main_lane_poly_points:
                    # main_lane_poly_points[0] 이 가장 y가 작은 점 -> 차량에서 가장 먼 점
                    goal_point_vehicle_coords = image_to_vehicle(main_lane_poly_points[0])


                if goal_point_vehicle_coords is not None:
                    X_v_goal, Y_v_goal = goal_point_vehicle_coords
                    
                    # 스탠리 방법과 유사한 각도 계산 (또는 순수 추종)
                    # alpha: 차량 전방축과 목표점 사이의 각도
                    # Y_v_goal: 측방 오차 (목표점의 y 좌표)
                    # X_v_goal: 전방 거리 (목표점의 x 좌표)
                    alpha = np.arctan2(Y_v_goal, X_v_goal) # arctan2(y,x)
                    
                    # 순수 추종 (Pure Pursuit) 조향각 공식
                    # L: 차량 축거 (wheelbase_m)
                    # ld: 목표점까지의 직선 거리
                    ld = np.sqrt(X_v_goal**2 + Y_v_goal**2)
                    
                    if ld < 1e-5: # 목표점이 매우 가까우면 조향각 0
                        steering_angle_rad = 0.0
                    else:
                        steering_angle_rad = np.arctan((2 * wheelbase_m * np.sin(alpha)) / ld)
                    
                    steering_angle_deg = np.degrees(steering_angle_rad)
                    
                    # 조향각 범위 제한 (옵션)
                    # steering_angle_deg = np.clip(steering_angle_deg, -25.0, 25.0) 

                    pub_steering.publish(Float32(data=steering_angle_deg))
                    rospy.loginfo_throttle(0.2, "[INFO] Steer: %.2f deg (Alpha: %.2f deg, Goal: X:%.2fm Y:%.2fm)", 
                                           steering_angle_deg, np.degrees(alpha), X_v_goal, Y_v_goal)

                    # 목표점(차량좌표)을 다시 BEV 이미지 좌표로 변환하여 시각화
                    # x_vehicle = (H_bev - v_bev) * mpp_y + y_offset_m  => v_bev = H_bev - (x_vehicle - y_offset_m) / mpp_y
                    # y_vehicle = (W_bev/2 - u_bev) * mpp_x           => u_bev = W_bev/2 - y_vehicle / mpp_x
                    goal_v_img = int(h_bev - (X_v_goal - y_offset_m) / m_per_pixel_y)
                    goal_u_img = int(w_bev/2 - Y_v_goal / m_per_pixel_x)
                    
                    if 0 <= goal_u_img < w_bev and 0 <= goal_v_img < h_bev:
                         cv2.circle(bev_im_color, (goal_u_img, goal_v_img), 8, (0, 255, 0), -1) # 초록색 (퓨어퍼슛 목표점)
                    cv2.putText(bev_im_color, f"Steer: {steering_angle_deg:.1f} deg", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                if opt.debug:
                    debug_plot_lane(main_lane_poly_points, image_to_vehicle, goal_point_vehicle_coords)
            
            else: # main_lane_poly_points 생성 실패 (차선은 검출됐으나 경로화 실패)
                if lane_detected_bool:
                     rospy.logwarn_throttle(1.0,"[WARNING] Lane component(s) found, but path polyline could not be generated. Skipping steering.")
        
        # else: # lane_detected_bool is False (차선 아예 미검출)
            # rospy.logwarn_throttle(1.0,"[WARNING] Lane not detected, skipping steering.")
        
        # publish bev_mask (Image msg)
        if pub_mask.get_num_connections() > 0:
            try:
                # final_mask는 0 또는 255 값을 가지는 단일 채널 마스크
                # ROS Image 메시지로 보내기 위해 3채널로 만들거나, encoding을 mono8로 지정
                # 여기서는 mono8로 보냄
                mask_msg = bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
                pub_mask.publish(mask_msg)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")

        # 항상 최신 BEV 이미지와 마스크를 보여줌
        cv2.imshow("BEV (Processed Path)", bev_im_color)
        # cv2.imshow("Original Thin Mask", thin_mask) # 디버깅용
        # cv2.imshow("BEV Original Thin Mask", bev_mask_orig) # 디버깅용
        cv2.imshow("Final Lane Mask (for Polyfit)", final_mask)
        
        return bev_im_color, final_mask # 반환값은 현재 사용되지 않음 (save_img 등에서 사용될 수 있음)

    # --- 웹캠/비디오 스트림 처리 루프 ---
    if dataset.mode == 'stream': # 웹캠 또는 RTSP 등
        frame_skip = opt.frame_skip
        frame_counter = 0
        # 비동기 프레임 처리를 위한 큐 사용 (기존 코드 유지)
        frame_queue = queue.Queue(maxsize=5) 

        def frame_producer():
            try:
                for item in dataset: # path, img, im0s, vid_cap
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait() # 오래된 프레임 버림
                        except queue.Empty:
                            pass
                    frame_queue.put(item)
            except Exception as e:
                rospy.logerr(f"Frame producer error: {e}")
            finally:
                frame_queue.put(None) # 종료 신호

        producer_thread = threading.Thread(target=frame_producer)
        producer_thread.daemon = True # 메인 스레드 종료 시 함께 종료
        producer_thread.start()
        rospy.loginfo("[DEBUG] 웹캠 비동기 프레임 생산 시작")

        while not rospy.is_shutdown():
            try:
                frame_data = frame_queue.get(timeout=1.0) # 타임아웃 추가
            except queue.Empty:
                rospy.logwarn_throttle(5.0, "Frame queue empty for 1 sec. Producer might have issues or stream ended.")
                # 웹캠이 갑자기 꺼지거나 연결 끊겼을 때 무한 대기 방지
                if not producer_thread.is_alive() and frame_queue.empty():
                    rospy.loginfo("Producer thread stopped and queue is empty. Exiting consumer loop.")
                    break
                continue
            
            if frame_data is None: # 종료 신호 받으면 루프 탈출
                rospy.loginfo("Received None from producer, ending processing.")
                break
            
            path_item, net_input_img, im0s, vid_cap = frame_data
            
            if frame_skip > 0 and frame_counter % (frame_skip + 1) != 0:
                frame_counter += 1
                continue
            frame_counter += 1
            
            process_frame(im0s)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("User pressed 'q', exiting.")
                break
            elif key == ord('p'): # 일시정지/재생 토글 (디버깅용)
                rospy.loginfo("Paused. Press 'p' again to resume.")
                while True:
                    if cv2.waitKey(0) & 0xFF == ord('p'):
                        rospy.loginfo("Resumed.")
                        break
    
    else: # 저장된 영상/이미지 파일 처리 (동기)
        rospy.loginfo("[DEBUG] 저장된 영상/이미지 동기 처리 시작")
        delay = 1 # 이미지 파일일 경우 거의 즉시 다음 프레임
        if dataset.mode == 'video' and hasattr(dataset, 'cap') and dataset.cap is not None:
            fps = dataset.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                delay = int(1000 / fps) # 비디오 FPS에 맞춘 딜레이
        
        for frame_idx, frame_data in enumerate(dataset):
            if rospy.is_shutdown():
                break
            path_item, net_input_img, im0s, vid_cap = frame_data
            
            processing_start_time = time.time()
            process_frame(im0s)
            processing_end_time = time.time()
            
            # 프레임 처리 시간 고려한 delay 조정 (실시간 재생에 가깝게)
            elapsed_processing_ms = (processing_end_time - processing_start_time) * 1000
            current_delay = max(1, delay - int(elapsed_processing_ms))

            if cv2.waitKey(current_delay) & 0xFF == ord('q'):
                rospy.loginfo("User pressed 'q', exiting.")
                break
        rospy.loginfo("[INFO] 동기 처리 추론 평균 시간: %.4fs/frame", inf_time.avg)

    if hasattr(dataset, 'release'): # LoadCamera or LoadImages
        dataset.release()
    cv2.destroyAllWindows()
    if opt.debug:
        plt.ioff() # 대화형 모드 종료
        plt.close('all') # 모든 matplotlib 창 닫기
    rospy.loginfo("[INFO] 처리 완료 및 종료.")

# 메인 함수
def ros_main():
    rospy.init_node('bev_lane_follower_node', anonymous=True) # 노드 이름 변경 가능
    parser = make_parser()
    opt, _ = parser.parse_known_args()

    if opt.debug:
        plt.ion() # Matplotlib 대화형 모드 활성화

    # ROS 퍼블리셔 정의
    pub_mask = rospy.Publisher('camera_bev_lane_mask', Image, queue_size=1) # BEV 마스크
    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
    
    rospy.loginfo("BEV Lane Follower Node 시작")
    rospy.loginfo(f"옵션: {opt}")

    try:
        detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)
    except Exception as e:
        rospy.logerr(f"detect_and_publish 함수에서 예외 발생: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        rospy.loginfo("ROS Main 함수 종료 중...")
        # rospy.signal_shutdown("Processing finished or error occurred.") # 명시적 종료 시그널

if __name__ == '__main__':
    try:
        # torch.no_grad() 컨텍스트는 모델 추론 부분에서 이미 관리됨
        ros_main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS 노드 종료됨 (ROSInterruptException)")
    except Exception as e:
        rospy.logfatal(f"알 수 없는 심각한 오류로 프로그램 종료: {e}")
        import traceback
        rospy.logfatal(traceback.format_exc())
    finally:
        cv2.destroyAllWindows() # 만약을 위해 한번 더 호출
        if plt.get_fignums(): # 열려있는 matplotlib 창이 있다면
            plt.close('all')
        rospy.loginfo("최종 프로그램 종료.")