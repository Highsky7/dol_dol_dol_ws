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
from math import atan, atan2, degrees
from pathlib import Path
import matplotlib.pyplot as plt

# --- NEW --- : Ultralytics 라이브러리 임포트
from ultralytics import YOLO

# ROS 메시지
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from std_msgs.msg import Bool

# 기존 프로젝트 유틸 (일부만 사용)
# utils.py 파일이 같은 디렉토리에 있다고 가정합니다.
from utils.utils import (
    time_synchronized,
    increment_path,
    AverageMeter,
    LoadCamera,
    LoadImages,
)

# argparse 설정
def make_parser():
    parser = argparse.ArgumentParser(description="Roboflow Instance Segmentation with BEV Transformation")
    parser.add_argument('--weights', type=str, default='./weights.pt', help='path to your roboflow model.pt file')
    parser.add_argument('--source', type=str,
                        # default='2',
                        default='/home/highsky/Videos/Webcam/right_bev_params_2.mp4',
                        help='source: 0(webcam) or video/image file path')
    parser.add_argument('--img-size', type=int, default=640, help='inference resolution')
    parser.add_argument('--device', default='0', help='cuda device: 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--project', default='runs/detect_roboflow', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--frame-skip', type=int, default=0, help='frame skipping (0 to disable)')
    parser.add_argument('--param-file', type=str, default='./bev_params_2.npz', help='BEV parameter file (.npz)')
    parser.add_argument('--debug', action='store_true', help='Visualize lane in vehicle coordinates using Matplotlib')
    return parser

# --- 유틸리티 함수들 ---

def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: return None
    try:
        return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError): return None

def compute_polyline_points(coeff, image_shape, step=4):
    h, w = image_shape[:2]
    points = []
    if coeff is None: return points
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w:
            points.append((int(x), int(y)))
    return points

def overlay_polyline(image, coeff, color=(0, 0, 255), step=4, thickness=2, translation=(0,0)):
    if coeff is None: return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w:
            draw_points.append((int(x + translation[0]), int(y + translation[1])))
    if len(draw_points) > 1:
        cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, thickness)
    return image

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return np.zeros_like(binary_mask)
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
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=10000)
    f4 = keep_top2_components(f3, min_area=300)
    return f4

def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points, dst_points = params['src_points'], params['dst_points']
    warp_w, warp_h = int(params['warp_w']), int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

def debug_plot_lane(shifted_poly_points, image_to_vehicle_func, goal_point=None):
    if not shifted_poly_points: return
    lane_vehicle = [image_to_vehicle_func(pt) for pt in shifted_poly_points]
    if len(lane_vehicle) > 0:
        lane_vehicle = np.array(lane_vehicle)
        plt.figure("Lane in Vehicle Coordinates", figsize=(6, 6))
        plt.clf()
        plt.plot(lane_vehicle[:, 1], lane_vehicle[:, 0], 'r-', label="Processed Lane")
        if goal_point is not None:
            plt.scatter(goal_point[1], goal_point[0], color='green', s=100, label="Goal Point")
        plt.xlabel("Lateral (m)"); plt.ylabel("Forward (m)")
        plt.title("Lane Line in Vehicle Coordinates"); plt.legend()
        plt.gca().invert_xaxis(); plt.xlim(1.5, -1.5)
        plt.ylim(0.0, 3.5); plt.grid(True); plt.axis('equal')
        plt.show(block=False); plt.pause(0.001)


def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):
    cudnn.benchmark = True
    bridge = CvBridge()
    source, weights = opt.source, opt.weights
    bev_param_file = opt.param_file

    inf_time = AverageMeter()

    device_str = opt.device.lower()
    if device_str.isdigit():
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_str}')
        else:
            rospy.logwarn(f"CUDA device '{device_str}' requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    elif device_str == 'cpu':
        device = torch.device('cpu')
    else:
        rospy.logwarn(f"Invalid device string '{device_str}'. Defaulting to CPU.")
        device = torch.device('cpu')
    
    rospy.loginfo(f"[INFO] Using device: {device}")

    rospy.loginfo(f"[INFO] Loading model from {weights}...")
    model = YOLO(weights)
    model.to(device)
    rospy.loginfo("[INFO] Model loaded successfully.")

    if source.isdigit():
        rospy.loginfo(f"[INFO] Opening webcam (device={source})")
        dataset = LoadCamera(source, img_size=opt.img_size)
    else:
        rospy.loginfo(f"[INFO] Opening video/image file: {source}")
        dataset = LoadImages(source, img_size=opt.img_size)

    bev_params = np.load(bev_param_file)
    bev_h_expected, bev_w_expected = int(bev_params['warp_h']), int(bev_params['warp_w'])
    
    # m_per_pixel_y, y_offset_m, m_per_pixel_x = 0.003015625, 1.8, 0.002734375 # for bev_params_1.npz
    m_per_pixel_y, y_offset_m, m_per_pixel_x = 0.004015625, 1.83, 0.00278125 # for bev_params_2.npz
    # m_per_pixel_y, y_offset_m, m_per_pixel_X = 0.002, 1.28, 0.003390625 # for bev_params_3.npz
    
    # --- HINTON'S FIX 1: 차선 추적을 위한 '메모리' 변수 선언 ---
    tracked_lanes = {
        'left': {'coeff': None, 'age': 0},
        'right': {'coeff': None, 'age': 0}
    }
    # --- HINTON'S FIX 5: 주행 경로 추적을 위한 '메모리' 변수 추가 ---
    tracked_center_path = {'coeff': None}

    SMOOTHING_ALPHA = 0.6
    MAX_LANE_AGE = 10

    def image_to_vehicle(pt_bev):
        u, v = pt_bev
        x_vehicle = (bev_h_expected - v) * m_per_pixel_y + y_offset_m
        y_vehicle = (bev_w_expected / 2 - u) * m_per_pixel_x
        return x_vehicle, y_vehicle

    def process_frame(im0s, tracked_lanes, tracked_center_path):
        # 1-4단계: BEV 변환 및 마스크 필터링
        bev_image_input = do_bev_transform(im0s, bev_param_file)
        results = model(bev_image_input, imgsz=opt.img_size, conf=opt.conf_thres, iou=opt.iou_thres, device=device, verbose=False)
        result = results[0]
        combined_mask_bev = np.zeros(result.orig_shape, dtype=np.uint8)
        if result.masks is not None:
            for mask_tensor in result.masks.data:
                mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                if mask_np.shape != result.orig_shape:
                     mask_np = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]))
                combined_mask_bev = np.maximum(combined_mask_bev, mask_np)
        final_mask = final_filter(combined_mask_bev)
        
        # 5단계: 안정적인 주행 경로 생성 및 조향각 계산
        bev_im_for_drawing = bev_image_input.copy()
        h_bev, w_bev = bev_im_for_drawing.shape[:2]

        # 5.1: 현재 프레임에서 차선 후보군 추출
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        current_detections = []
        if num_labels > 1:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 100:
                    ys, xs = np.where(labels == i)
                    coeff = polyfit_lane(ys, xs, order=2)
                    if coeff is not None:
                        x_at_bottom = np.polyval(coeff, h_bev - 1)
                        current_detections.append({'coeff': coeff, 'x_bottom': x_at_bottom})
            current_detections.sort(key=lambda c: c['x_bottom'])

        # --- 차선 추적 및 연관 로직 (이전과 동일) ---
        left_lane_tracked = tracked_lanes['left']
        right_lane_tracked = tracked_lanes['right']
        current_left = None
        current_right = None

        if len(current_detections) == 2:
            current_left, current_right = current_detections[0], current_detections[1]
        elif len(current_detections) == 1:
            detected_lane = current_detections[0]
            dist_to_left = abs(detected_lane['x_bottom'] - np.polyval(left_lane_tracked['coeff'], h_bev - 1)) if left_lane_tracked['coeff'] is not None else float('inf')
            dist_to_right = abs(detected_lane['x_bottom'] - np.polyval(right_lane_tracked['coeff'], h_bev - 1)) if right_lane_tracked['coeff'] is not None else float('inf')
            if dist_to_left < dist_to_right and left_lane_tracked['coeff'] is not None:
                current_left = detected_lane
            elif dist_to_right < dist_to_left and right_lane_tracked['coeff'] is not None:
                current_right = detected_lane
            else:
                if detected_lane['x_bottom'] < w_bev / 2:
                    current_left = detected_lane
                else:
                    current_right = detected_lane
        
        if current_left:
            if left_lane_tracked['coeff'] is None: left_lane_tracked['coeff'] = current_left['coeff']
            else: left_lane_tracked['coeff'] = (SMOOTHING_ALPHA * current_left['coeff'] + (1 - SMOOTHING_ALPHA) * left_lane_tracked['coeff'])
            left_lane_tracked['age'] = 0
        else: left_lane_tracked['age'] += 1

        if current_right:
            if right_lane_tracked['coeff'] is None: right_lane_tracked['coeff'] = current_right['coeff']
            else: right_lane_tracked['coeff'] = (SMOOTHING_ALPHA * current_right['coeff'] + (1 - SMOOTHING_ALPHA) * right_lane_tracked['coeff'])
            right_lane_tracked['age'] = 0
        else: right_lane_tracked['age'] += 1

        if left_lane_tracked['age'] > MAX_LANE_AGE: left_lane_tracked['coeff'] = None
        if right_lane_tracked['age'] > MAX_LANE_AGE: right_lane_tracked['coeff'] = None
        
        final_left_coeff = left_lane_tracked['coeff']
        final_right_coeff = right_lane_tracked['coeff']

        lane_detected_bool = (final_left_coeff is not None) or (final_right_coeff is not None)
        pub_lane_status.publish(Bool(data=lane_detected_bool))
        
        steering_angle_deg = None
        
        # --- HINTON'S FIX 6: 주행 경로의 시간적 안정성 확보 ---
        target_center_lane_coeff = None
        if lane_detected_bool:
            # 5.2: 현재 프레임 기준의 '이상적인' 중앙 주행 경로 계산
            center_points = []
            LANE_WIDTH_M = 1.5
            lane_width_pixels = LANE_WIDTH_M / m_per_pixel_x
            
            for y in range(h_bev - 1, h_bev // 2, -1):
                x_center = None
                if final_left_coeff is not None and final_right_coeff is not None:
                    x_left = np.polyval(final_left_coeff, y)
                    x_right = np.polyval(final_right_coeff, y)
                    x_center = (x_left + x_right) / 2
                elif final_left_coeff is not None:
                    x_center = np.polyval(final_left_coeff, y) + lane_width_pixels / 2
                elif final_right_coeff is not None:
                    x_center = np.polyval(final_right_coeff, y) - lane_width_pixels / 2
                
                if x_center is not None:
                    center_points.append([x_center, y])
            
            if len(center_points) > 10:
                center_points = np.array(center_points)
                target_center_lane_coeff = polyfit_lane(center_points[:, 1], center_points[:, 0], order=2)

        # 5.3: '기억된' 주행 경로를 부드럽게 갱신
        if target_center_lane_coeff is not None:
            if tracked_center_path['coeff'] is None:
                tracked_center_path['coeff'] = target_center_lane_coeff
            else:
                # 경로가 갑자기 튀지 않도록 점진적으로 갱신
                tracked_center_path['coeff'] = (SMOOTHING_ALPHA * target_center_lane_coeff + \
                                               (1 - SMOOTHING_ALPHA) * tracked_center_path['coeff'])
        
        # 5.4: 최종적으로 '기억된' 주행 경로를 사용하여 조향각 계산
        if tracked_center_path['coeff'] is not None:
            final_center_coeff = tracked_center_path['coeff']
            k_crosstrack, k_heading = 0.3, 0.7
            vehicle_x, vehicle_y = w_bev / 2, h_bev - 1
            path_x_at_vehicle = np.polyval(final_center_coeff, vehicle_y)
            crosstrack_error_m = (vehicle_x - path_x_at_vehicle) * m_per_pixel_x
            
            lookahead_y = vehicle_y - 20
            path_slope = 2 * final_center_coeff[0] * lookahead_y + final_center_coeff[1]
            heading_error_rad = atan(path_slope)
            
            crosstrack_steer_rad = atan((k_crosstrack * crosstrack_error_m) / 1.0)
            
            steering_angle_rad = k_heading * heading_error_rad + crosstrack_steer_rad
            steering_angle_deg = np.degrees(steering_angle_rad)

            # --- HINTON'S FIX 7: 조향각 부호 문제 해결 ---
            # 계산된 조향각의 부호를 반전시켜 (우회전: 음수, 좌회전: 양수) 규약을 맞춥니다.
            # steering_angle_deg = -steering_angle_deg

            steering_angle_deg = np.clip(steering_angle_deg, -25.0, 25.0)
            pub_steering.publish(Float32(data=steering_angle_deg))
        
        # 6단계: 최종 결과 시각화
        annotated_frame = result.plot()
        
        overlay_polyline(bev_im_for_drawing, final_left_coeff, color=(255, 0, 0), step=2, thickness=2)
        overlay_polyline(bev_im_for_drawing, final_right_coeff, color=(0, 0, 255), step=2, thickness=2)
        
        # 최종적으로 추적되고 있는 안정화된 중앙 경로를 시각화합니다.
        if tracked_center_path['coeff'] is not None:
            overlay_polyline(bev_im_for_drawing, tracked_center_path['coeff'], color=(0, 255, 0), step=2, thickness=3)

        if steering_angle_deg is not None:
            steer_text = f"Steer: {steering_angle_deg:.1f} deg"
        else:
            steer_text = "Steer: N/A"
            
        cv2.putText(bev_im_for_drawing, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(bev_im_for_drawing, f"Lane Detected: {lane_detected_bool}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Original Camera View", im0s)
        cv2.imshow("Roboflow Detections (on BEV)", annotated_frame)
        cv2.imshow("Final Path (on BEV)", bev_im_for_drawing)


    frame_count = 0
    for path, img, im0s, vid_cap in dataset:
        if rospy.is_shutdown():
            break
        
        if opt.frame_skip > 0 and frame_count % (opt.frame_skip + 1) != 0:
            frame_count += 1
            continue
        frame_count += 1

        # `tracked_lanes`와 `tracked_center_path` 메모리 객체를 process_frame에 전달
        process_frame(im0s, tracked_lanes, tracked_center_path)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.loginfo("User pressed 'q', exiting.")
            break

    return dataset


def ros_main():
    rospy.init_node('roboflow_bev_follower_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    if opt.debug: plt.ion()
    
    pub_mask = rospy.Publisher('camera_bev_lane_mask', Image, queue_size=1)
    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
    
    rospy.loginfo("Roboflow BEV Follower Node Started")
    rospy.loginfo(f"OPTIONS: {opt}")

    dataset_to_clean = None
    try:
        dataset_to_clean = detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted by user (Ctrl+C).")
    except Exception as e:
        rospy.logerr(f"An exception occurred in main execution: {e}", exc_info=True)
    finally:
        rospy.loginfo("Starting cleanup process...")
        if dataset_to_clean is not None:
            rospy.loginfo("Releasing dataset resource (camera or video file)...")
            del dataset_to_clean
        
        rospy.loginfo("Destroying all OpenCV windows...")
        cv2.destroyAllWindows()
        
        if opt.debug:
            plt.close('all')
            
        rospy.loginfo("Cleanup finished. Shutting down ROS Main function...")

if __name__ == '__main__':
    try:
        ros_main()
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in __main__: {e}", exc_info=True)
    finally:
        rospy.loginfo("Program terminated.")