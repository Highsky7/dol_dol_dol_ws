#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import argparse
import time
import cv2
import numpy as np
import threading
import queue
from math import atan2, degrees
from pathlib import Path
import matplotlib.pyplot as plt
import cv2.ximgproc as ximgproc # 세선화를 위해 추가

# ROS 메시지
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from std_msgs.msg import Bool

# 프로젝트 내 유틸
from utils.utils import (
    time_synchronized,
    increment_path,
    AverageMeter,
    LoadCamera,
    LoadImages,
)

# =============================================
# 차선 인식 파라미터
# =============================================
WHITE_L_CHANNEL_THRESHOLD = 240 # HLS L 채널 임계값
MORPH_CLOSE_KSIZE = 5         # 모폴로지 닫기 연산 커널 크기
MIN_COMPONENT_AREA = 10000      # 최소 차선 컴포넌트 면적
MIN_POLYFIT_POINTS = 5        # Polyfit에 필요한 최소 포인트 수

# argparse 설정
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='2',
                        # default='/home/highsky/Videos/Webcam/right_bev_params_1.mp4',
                        # default='/home/highsky/Videos/Webcam/right_bev_params_2.mp4',
                        help='source')
    parser.add_argument('--img-size', type=int, default=640, help='이미지 처리 해상도')
    parser.add_argument('--nosave', action='store_true', help='저장하지 않으려면 사용')
    parser.add_argument('--project', default='runs/detect_final_polyfit', help='결과 저장 폴더')
    parser.add_argument('--name', default='exp', help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_true', help='기존 폴더 사용 허용')
    parser.add_argument('--frame-skip', type=int, default=0, help='프레임 건너뛰기')
    parser.add_argument('--param-file', type=str,
                        default='./bev_params_3.npz',
                        # default='./bev_params_2.npz',
                        help='BEV 파라미터 파일')
    parser.add_argument('--debug', action='store_true', help='Matplotlib 시각화 사용')
    return parser

# ==============================================================================
# === 함수 정의: 필터링, Polyfit, 경로 생성 및 기타 유틸리티 ===
# ==============================================================================

def get_white_mask(bev_image):
    """HLS 색상 공간의 L 채널을 사용하여 흰색 차선 마스크를 추출합니다."""
    hls = cv2.cvtColor(bev_image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    _, white_mask = cv2.threshold(l_channel, WHITE_L_CHANNEL_THRESHOLD, 255, cv2.THRESH_BINARY)
    return white_mask

def morph_close(binary_mask, ksize=5):
    """모폴로지 닫기 연산을 수행하여 마스크의 작은 구멍들을 메웁니다."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size):
    """작은 노이즈(컴포넌트)를 면적 기준으로 제거합니다."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area):
    """면적이 가장 큰 상위 2개의 컴포넌트만 남깁니다."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_mask)
    
    comps = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            comps.append((i, stats[i, cv2.CC_STAT_AREA]))
            
    comps.sort(key=lambda x: x[1], reverse=True)
    
    cleaned = np.zeros_like(binary_mask)
    for i in range(min(len(comps), 2)):
        idx = comps[i][0]
        cleaned[labels == idx] = 255
    return cleaned

def final_filter(bev_mask):
    """일련의 필터링 과정을 통합하여 최종 차선 후보 마스크를 생성합니다."""
    f2 = morph_close(bev_mask, ksize=MORPH_CLOSE_KSIZE)
    f3 = remove_small_components(f2, min_size=MIN_COMPONENT_AREA)
    f4 = keep_top2_components(f3, min_area=MIN_COMPONENT_AREA)
    return f4

def polyfit_lane(points_y, points_x, order=2):
    """주어진 점들에 대해 다항식 피팅을 수행합니다."""
    if len(points_y) < MIN_POLYFIT_POINTS:
        return None
    try:
        return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError):
        return None

# ==================== 수정된 부분: 경로 생성 함수 ====================
def compute_polyline_points(coeff, y_max, y_min, image_width, step=4):
    """다항식 계수로부터 '주어진 Y 범위 내에서만' 폴리라인 점들을 계산합니다."""
    points = []
    if coeff is None:
        return points
    # y좌표를 아래(y_max)에서 위(y_min)로 스캔하며 경로 생성
    for y in range(int(y_max), int(y_min), -step):
        x = np.polyval(coeff, y)
        if 0 <= x < image_width:
            points.append((int(x), int(y)))
    return points
# =================================================================

def overlay_polyline(image, points, color=(255, 0, 255), thickness=2):
    """이미지 위에 폴리라인을 그립니다."""
    if len(points) > 1:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, thickness)
    return image

def do_bev_transform(image, bev_param_file):
    """입력 이미지에 대해 BEV(Bird's-Eye View) 변환을 수행합니다."""
    params = np.load(bev_param_file)
    src_points, dst_points = params['src_points'], params['dst_points']
    warp_w, warp_h = int(params['warp_w']), int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

# 메인 처리 함수
def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):
    source = opt.source
    dataset = LoadImages(source, img_size=opt.img_size, stride=32) if not source.isdigit() else LoadCamera(source, img_size=opt.img_size, stride=32)

    bev_params = np.load(opt.param_file)
    bev_h, bev_w = int(bev_params['warp_h']), int(bev_params['warp_w'])
    # m_per_pixel_y, y_offset_m, m_per_pixel_x = 0.003015625, 1.8, 0.002734375 # for bev_params_1.npz
    # m_per_pixel_y, y_offset_m, m_per_pixel_x = 0.004015625, 1.83, 0.00278125 # for bev_params_2.npz
    m_per_pixel_y, y_offset_m, m_per_pixel_x = 0.002, 2.88, 0.003390625 # for bev_params_1.npz


    def image_to_vehicle(pt_bev):
        u, v = pt_bev
        x_v = (bev_h - v) * m_per_pixel_y + y_offset_m
        y_v = (bev_w / 2 - u) * m_per_pixel_x
        return x_v, y_v

    def process_frame(im0s):
        bev_im_color = do_bev_transform(im0s, opt.param_file)
        bev_im_for_vis = bev_im_color.copy()

        white_lane_mask = get_white_mask(bev_im_color)
        bevfilter_mask = final_filter(white_lane_mask)
        
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_GUOHALL)
        if final_mask is None or np.sum(final_mask) == 0:
            final_mask = bevfilter_mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        
        lane_components = []
        if num_labels > 1:
            sorted_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            for i in range(min(len(sorted_indices), 2)):
                label_idx = sorted_indices[i]
                if stats[label_idx, cv2.CC_STAT_AREA] >= 100:
                    component_mask = np.zeros_like(final_mask)
                    component_mask[labels == label_idx] = 255
                    ys, xs = np.where(component_mask > 0)
                    if len(ys) > MIN_POLYFIT_POINTS:
                        coeff = polyfit_lane(ys, xs, order=2)
                        if coeff is not None:
                            lane_components.append({'ys': ys, 'xs': xs, 'coeff': coeff})

        for comp in lane_components:
            # 원본 검출 차선 시각화 시에도 y범위를 제한하여 그림
            y_max_comp = np.max(comp['ys'])
            y_min_comp = np.min(comp['ys'])
            orig_points = compute_polyline_points(comp['coeff'], y_max_comp, y_min_comp, bev_w)
            overlay_polyline(bev_im_for_vis, orig_points, color=(0, 255, 0), thickness=1)

        lane_detected_bool = len(lane_components) > 0
        pub_lane_status.publish(Bool(data=lane_detected_bool))

        main_lane_poly_points = []
        path_start_point_bev = (bev_w // 2, bev_h - 1)
        active_coeff_for_steering = None

        if lane_detected_bool:
            # ==================== 수정된 부분: Y축 범위 계산 ====================
            min_y_overall = bev_h
            max_y_overall = 0
            for comp in lane_components:
                min_y_overall = min(min_y_overall, np.min(comp['ys']))
                max_y_overall = max(max_y_overall, np.max(comp['ys']))
            # =================================================================

            if len(lane_components) == 1:
                active_coeff_for_steering = lane_components[0]['coeff']
            elif len(lane_components) == 2:
                active_coeff_for_steering = (lane_components[0]['coeff'] + lane_components[1]['coeff']) / 2.0

            if active_coeff_for_steering is not None:
                # 수정된 함수 호출: y축 범위를 명시적으로 전달
                poly_points_from_coeff = compute_polyline_points(active_coeff_for_steering, max_y_overall, min_y_overall, bev_w)

                if len(poly_points_from_coeff) > 0:
                    # poly_points_from_coeff의 첫번째 점이 가장 아래쪽(y_max) 점임
                    current_polyline_bottom_point = poly_points_from_coeff[0]
                    translation = (path_start_point_bev[0] - current_polyline_bottom_point[0],
                                   path_start_point_bev[1] - current_polyline_bottom_point[1])
                    
                    main_lane_poly_points = [(pt[0] + translation[0], pt[1] + translation[1]) for pt in poly_points_from_coeff]
                    
                    overlay_polyline(bev_im_for_vis, main_lane_poly_points, color=(255, 0, 255), thickness=3)

            cv2.circle(bev_im_for_vis, path_start_point_bev, 10, (0, 0, 255), -1)

            if len(main_lane_poly_points) > 0:
                lookahead_m, wheelbase_m = 3.73, 0.75 # for bev_params_1.npz
                # lookahead_m, wheelbase_m = 3.115, 0.75  # for bev_params_2.npz
                goal_point = None
                min_err = float('inf')

                for pt in main_lane_poly_points:
                    X_v, Y_v = image_to_vehicle(pt)
                    err = abs(np.sqrt(X_v**2 + Y_v**2) - lookahead_m)
                    if err < min_err:
                        min_err, goal_point = err, (X_v, Y_v)

                if goal_point:
                    X_g, Y_g = goal_point
                    alpha = np.arctan2(Y_g, X_g)
                    ld = np.sqrt(X_g**2 + Y_g**2)
                    steer_rad = np.arctan((2 * wheelbase_m * np.sin(alpha)) / ld) if ld > 1e-5 else 0.0
                    pub_steering.publish(Float32(data=np.degrees(steer_rad)))
                    
                    u_g = int(bev_w / 2 - Y_g / m_per_pixel_x)
                    v_g = int(bev_h - (X_g - y_offset_m) / m_per_pixel_y)
                    if 0 <= u_g < bev_w and 0 <= v_g < bev_h:
                        cv2.circle(bev_im_for_vis, (u_g, v_g), 8, (0, 255, 0), -1)
                    cv2.putText(bev_im_for_vis, f"Steer: {np.degrees(steer_rad):.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(bev_im_for_vis, "Steer: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(bev_im_for_vis, "Steer: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("BEV (Processed Path)", bev_im_for_vis)
        cv2.imshow("Final Thin Mask", final_mask)
        cv2.imshow("Original White Mask", white_lane_mask)


    if dataset.mode == 'stream':
        frame_queue = queue.Queue(maxsize=5)
        producer_thread = threading.Thread(target=lambda: (
            [frame_queue.put(item) for item in dataset], frame_queue.put(None)))
        producer_thread.daemon = True
        producer_thread.start()
        while not rospy.is_shutdown():
            try:
                frame_data = frame_queue.get(timeout=1.0)
                if frame_data is None: break
                process_frame(frame_data[2])
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            except queue.Empty:
                if not producer_thread.is_alive() and frame_queue.empty(): break
    else:
        delay = 10
        if dataset.mode == 'video' and hasattr(dataset, 'cap') and dataset.cap.get(cv2.CAP_PROP_FPS) > 0:
            delay = int(1000 / dataset.cap.get(cv2.CAP_PROP_FPS))
        for _, _, im0s, _ in dataset:
            if rospy.is_shutdown(): break
            process_frame(im0s)
            if cv2.waitKey(delay) & 0xFF == ord('q'): break
            
    if hasattr(dataset, 'release'):
        dataset.release()
    cv2.destroyAllWindows()

def ros_main():
    rospy.init_node('bev_lane_polyfit_follower_node', anonymous=True)
    parser = make_parser()
    opt = parser.parse_args()
    pub_mask = rospy.Publisher('camera_bev_lane_mask', Image, queue_size=1)
    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
    rospy.loginfo("Polyfit-based BEV Lane Follower Node (Path Length Limited) 시작")
    try:
        detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)
    except Exception as e:
        rospy.logerr(f"Main loop exception: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())

if __name__ == '__main__':
    try:
        ros_main()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("프로그램 종료.")