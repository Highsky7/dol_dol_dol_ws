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
from math import atan2, degrees
from pathlib import Path
import matplotlib.pyplot as plt

from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from std_msgs.msg import Bool
# utils.py의 LoadImages는 파일 처리를 위해 그대로 사용합니다.
from utils.utils import time_synchronized, increment_path, AverageMeter, LoadImages

# argparse 설정 (변경 없음)
def make_parser():
    parser = argparse.ArgumentParser(description="Roboflow Instance Segmentation with BEV on Mask")
    parser.add_argument('--weights', type=str, default='./weights.pt', help='path to your BEV-trained model.pt file')
    parser.add_argument('--source', type=str, default='0', help='source: 0(webcam) or video/image file path')
    parser.add_argument('--img-size', type=int, default=640, help='inference resolution for the model')
    parser.add_argument('--device', default='0', help='cuda device: 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--project', default='runs/detect_roboflow_bev', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--nosave', action='store_false', help='저장하지 않으려면 사용')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--frame-skip', type=int, default=0, help='frame skipping (0 to disable)')
    parser.add_argument('--param-file', type=str, default='./bev_params_3.npz', help='BEV parameter file (.npz)')
    parser.add_argument('--debug', action='store_true', help='Visualize lane in vehicle coordinates using Matplotlib')
    return parser

# 유틸리티 함수들 (변경 없음)
def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: return None
    try: return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError): return None
def compute_polyline_points(coeff, image_shape, step=4):
    h, w = image_shape[:2]
    points = []
    if coeff is None: return points
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w: points.append((int(x), int(y)))
    return points
def overlay_polyline(image, coeff, color=(0, 0, 255), step=4, translation=(0,0)):
    if coeff is None: return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w: draw_points.append((int(x + translation[0]), int(y + translation[1])))
    if len(draw_points) > 1: cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, 2)
    return image
def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
def remove_small_components(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size: cleaned[labels == i] = 255
    return cleaned
def keep_top2_components(binary_mask, min_area=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return np.zeros_like(binary_mask)
    comps = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: comps.append((i, stats[i, cv2.CC_STAT_AREA]))
    comps.sort(key=lambda x: x[1], reverse=True)
    cleaned = np.zeros_like(binary_mask)
    for i in range(min(len(comps), 2)):
        idx = comps[i][0]
        cleaned[labels == idx] = 255
    return cleaned
def final_filter(bev_mask):
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=300)
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
        if goal_point is not None: plt.scatter(goal_point[1], goal_point[0], color='green', s=100, label="Goal Point")
        plt.xlabel("Lateral (m)"); plt.ylabel("Forward (m)")
        plt.title("Lane Line in Vehicle Coordinates"); plt.legend()
        plt.gca().invert_xaxis(); plt.xlim(1.5, -1.5)
        plt.ylim(0.0, 3.5); plt.grid(True); plt.axis('equal')
        plt.show(block=False); plt.pause(0.001)

def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):
    cudnn.benchmark = True
    source, weights = opt.source, opt.weights
    bev_param_file = opt.param_file
    
    device_str = opt.device.lower()
    if device_str.isdigit() and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_str}')
    else:
        device = torch.device('cpu')
    rospy.loginfo(f"[INFO] Using device: {device}")

    rospy.loginfo(f"[INFO] Loading model from {weights}...")
    model = YOLO(weights)
    model.to(device)
    rospy.loginfo("[INFO] Model loaded successfully.")

    bev_params = np.load(bev_param_file)
    bev_h_expected, bev_w_expected = int(bev_params['warp_h']), int(bev_params['warp_w'])
    m_per_pixel_y, y_offset_m, m_per_pixel_x = 0.004015625, 1.83, 0.00278125

    def image_to_vehicle(pt_bev):
        u, v = pt_bev
        x_vehicle = (bev_h_expected - v) * m_per_pixel_y + y_offset_m
        y_vehicle = (bev_w_expected / 2 - u) * m_per_pixel_x
        return x_vehicle, y_vehicle

    # --- 핵심 로직 변경: process_frame 함수 ---
    def process_frame(im0s):
        # 1. 원본 프레임(im0s)에서 객체 탐지 (Detect-first)
        results = model(im0s, imgsz=opt.img_size, conf=opt.conf_thres, iou=opt.iou_thres, device=device, verbose=False)
        result = results[0]
        
        # 2. 탐지된 결과에서 마스크 추출 및 하나로 합치기
        combined_mask = np.zeros(result.orig_shape, dtype=np.uint8)
        if result.masks is not None:
            for mask_tensor in result.masks.data:
                # 마스크를 원본 이미지 크기로 리사이즈
                mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                resized_mask = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]))
                combined_mask = np.maximum(combined_mask, resized_mask)

        # 3. 합쳐진 마스크를 BEV로 변환
        bev_mask_orig = do_bev_transform(combined_mask, bev_param_file)
        
        # 4. BEV 마스크 후처리 및 골격선 추출
        bevfilter_mask = final_filter(bev_mask_orig)
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_GUOHALL)
        if final_mask is None or np.sum(final_mask) == 0:
            final_mask = bevfilter_mask # thinning 실패 시 필터링된 마스크 사용

        # 5. 경로 생성 및 조향각 계산 (기존 로직과 거의 동일)
        # 경로를 시각화하기 위해 원본 이미지를 BEV로 변환한 배경 이미지를 생성
        bev_im_for_drawing = do_bev_transform(im0s, bev_param_file)
        h_bev, w_bev = bev_im_for_drawing.shape[:2]
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        lane_components = []
        if num_labels > 1:
            sorted_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            for i in range(min(len(sorted_indices), 2)):
                label_idx = sorted_indices[i]
                if stats[label_idx, cv2.CC_STAT_AREA] >= 100:
                    ys, xs = np.where(labels == label_idx)
                    coeff = polyfit_lane(ys, xs, order=2)
                    if coeff is not None:
                        lane_components.append({'ys': ys, 'xs': xs, 'coeff': coeff})
        
        for comp in lane_components:
            overlay_polyline(bev_im_for_drawing, comp['coeff'], color=(0, 255, 0), step=4)
        
        lane_detected_bool = len(lane_components) > 0
        pub_lane_status.publish(Bool(data=lane_detected_bool))
        
        if lane_detected_bool:
            # 이 아래의 조향각 계산 로직은 기존 코드와 동일하게 유지됩니다.
            main_lane_poly_points = []
            path_start_point_bev = (w_bev // 2, h_bev - 1)
            active_coeff_for_steering = None
            if len(lane_components) == 1:
                comp = lane_components[0]
                active_coeff_for_steering = comp['coeff']
                max_y = np.max(comp['ys']); lane_bottom_x = int(np.polyval(comp['coeff'], max_y)); lane_bottom_point = (lane_bottom_x, int(max_y))
                min_y_s, max_y_s = np.min(comp['ys']), np.max(comp['ys']); slope = 0
                if (max_y_s - min_y_s) > 10:
                    x_at_min_y, x_at_max_y = np.polyval(comp['coeff'], min_y_s), np.polyval(comp['coeff'], max_y_s)
                    slope = (x_at_max_y - x_at_min_y) / (max_y_s - min_y_s + 1e-6)
                ref_point = (w_bev // 2, h_bev - 1)
                if slope > 0.15: ref_point = (0, h_bev - 1)
                elif slope < -0.15: ref_point = (w_bev - 1, h_bev - 1)
                path_start_point_bev = ((lane_bottom_point[0] + ref_point[0]) // 2, (lane_bottom_point[1] + ref_point[1]) // 2)
            elif len(lane_components) == 2:
                comp1, comp2 = lane_components[0], lane_components[1]
                max_y1 = np.max(comp1['ys']); x1 = int(np.polyval(comp1['coeff'], max_y1))
                max_y2 = np.max(comp2['ys']); x2 = int(np.polyval(comp2['coeff'], max_y2))
                path_start_point_bev = ((x1 + x2) // 2, (int(max_y1) + int(max_y2)) // 2)
                active_coeff_for_steering = (comp1['coeff'] + comp2['coeff']) / 2.0
            if active_coeff_for_steering is not None:
                poly_points = compute_polyline_points(active_coeff_for_steering, bev_im_for_drawing.shape, step=4)
                if len(poly_points) > 0:
                    bottom_pt = poly_points[-1]
                    translation = (path_start_point_bev[0] - bottom_pt[0], path_start_point_bev[1] - bottom_pt[1])
                    main_lane_poly_points = [(pt[0] + translation[0], pt[1] + translation[1]) for pt in poly_points]
                    if len(main_lane_poly_points) > 1:
                        cv2.polylines(bev_im_for_drawing, [np.array(main_lane_poly_points, dtype=np.int32)], False, (255, 0, 255), 2)
            cv2.circle(bev_im_for_drawing, path_start_point_bev, 10, (0, 0, 255), -1)
            if len(main_lane_poly_points) > 0:
                lookahead_m, wheelbase_m = 2.10, 0.75
                goal_point_vehicle_coords, min_dist_error = None, float('inf')
                for pt_img in main_lane_poly_points:
                    X_v, Y_v = image_to_vehicle(pt_img); dist = np.sqrt(X_v**2 + Y_v**2); error = abs(dist - lookahead_m)
                    if error < min_dist_error: min_dist_error, goal_point_vehicle_coords = error, (X_v, Y_v)
                if goal_point_vehicle_coords is None and main_lane_poly_points: goal_point_vehicle_coords = image_to_vehicle(main_lane_poly_points[0])
                if goal_point_vehicle_coords is not None:
                    X_v_goal, Y_v_goal = goal_point_vehicle_coords
                    alpha = np.arctan2(Y_v_goal, X_v_goal); ld = np.sqrt(X_v_goal**2 + Y_v_goal**2)
                    steering_angle_rad = np.arctan((2 * wheelbase_m * np.sin(alpha)) / ld) if ld > 1e-5 else 0.0
                    steering_angle_deg = np.degrees(steering_angle_rad)
                    pub_steering.publish(Float32(data=steering_angle_deg))
                    goal_v_img = int(h_bev - (X_v_goal - y_offset_m) / m_per_pixel_y); goal_u_img = int(w_bev/2 - Y_v_goal / m_per_pixel_x)
                    if 0 <= goal_u_img < w_bev and 0 <= goal_v_img < h_bev:
                        cv2.circle(bev_im_for_drawing, (goal_u_img, goal_v_img), 8, (0, 255, 0), -1)
        
        # 6. 시각화
        annotated_frame = result.plot() # 원본 영상에 탐지 결과 표시
        cv2.imshow("Original with Detections", annotated_frame)
        cv2.imshow("Front View Mask", combined_mask)
        cv2.imshow("Final BEV Mask", final_mask)
        cv2.imshow("Final Path (on BEV)", bev_im_for_drawing)


    # --- 메인 루프: 기존의 안정적인 VideoCapture 방식 유지 ---
    cap = None
    if source.isdigit():
        is_webcam = True
        cam_idx = int(source)
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            rospy.logwarn(f"Camera {cam_idx} with V4L2 backend failed. Trying default backend.")
            cap = cv2.VideoCapture(cam_idx)
    else:
        is_webcam = False
        cap = cv2.VideoCapture(source)

    if not cap or not cap.isOpened():
        rospy.logerr(f"Fatal: Failed to open video source: {source}")
        return

    try:
        frame_count = 0
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.loginfo("End of video stream or cannot grab frame.")
                break
            
            if opt.frame_skip > 0 and frame_count % (opt.frame_skip + 1) != 0:
                frame_count += 1
                continue
            frame_count += 1

            process_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("User pressed 'q', exiting loop.")
                break
    finally:
        if cap is not None:
            rospy.loginfo("Releasing video capture resource...")
            cap.release()

def ros_main():
    rospy.init_node('roboflow_bev_follower_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    if opt.debug: plt.ion()
    
    pub_mask = rospy.Publisher('camera_bev_lane_mask', Image, queue_size=1)
    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
    
    # 노드 시작 메시지 변경
    rospy.loginfo("Roboflow BEV Follower Node Started with Detect-first logic")
    rospy.loginfo(f"OPTIONS: {opt}")

    try:
        detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted by user (Ctrl+C).")
    except Exception as e:
        rospy.logerr(f"An exception occurred in main execution: {e}", exc_info=True)
    finally:
        rospy.loginfo("Cleanup process: Destroying all OpenCV windows...")
        cv2.destroyAllWindows()
        if opt.debug:
            plt.close('all')
        rospy.loginfo("Cleanup finished. Shutting down ROS Node...")

if __name__ == '__main__':
    try:
        ros_main()
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in __main__: {e}", exc_info=True)
    finally:
        rospy.loginfo("Program terminated.")