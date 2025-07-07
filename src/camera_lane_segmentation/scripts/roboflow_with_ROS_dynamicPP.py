#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import argparse
import cv2
import torch
import numpy as np
from math import atan, atan2, degrees, sqrt
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32, Bool

# --- 유틸리티 함수 (필터링 등) ---
def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: return None
    try: return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError): return None

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
    f3 = remove_small_components(f2, min_size=10000) # Need to be tuned for noise in real environment
    f4 = keep_top2_components(f3, min_area=300)
    return f4

def overlay_polyline(image, coeff, color=(0, 0, 255), step=4, thickness=2):
    if coeff is None: return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w: draw_points.append((int(x), int(y)))
    if len(draw_points) > 1: cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, thickness)
    return image

# --- 메인 로직 클래스 ---
class LaneFollowerNode:
    def __init__(self, opt):
        self.opt = opt
        self.bridge = CvBridge()

        device_str = self.opt.device.lower()
        if device_str.isdigit() and torch.cuda.is_available(): self.device = torch.device(f'cuda:{device_str}')
        else: self.device = torch.device('cpu')
        rospy.loginfo(f"[Lane Follower] Using device: {self.device}")

        rospy.loginfo(f"[Lane Follower] Loading model from {self.opt.weights}...")
        self.model = YOLO(self.opt.weights)
        self.model.to(self.device)
        rospy.loginfo("[Lane Follower] Model loaded successfully.")

        self.bev_params = np.load(self.opt.param_file)
        self.bev_h, self.bev_w = int(self.bev_params['warp_h']), int(self.bev_params['warp_w'])
        
        self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.0025, 1.25, 0.003578125 # for bev_params_y_5.npz

        # --- 차선 추적 파라미터 ---
        self.tracked_lanes = {'left': {'coeff': None, 'age': 0}, 'right': {'coeff': None, 'age': 0}}
        self.tracked_center_path = {'coeff': None}
        self.SMOOTHING_ALPHA = 0.6 
        self.MAX_LANE_AGE = 7 

        # --- PURE PURSUIT 파라미터 ---
        self.L = 0.73  # 차량 축거 (Wheelbase) [m]
        
        # ======================= [핵심 수정: 동적 전방주시거리 파라미터] =======================
        # throttle 입력 범위 0.3 ~ 0.5에 맞춰 파라미터를 설정합니다. (주행하며 튜닝 권장)
        self.THROTTLE_MIN = 0.4
        self.THROTTLE_MAX = 0.6
        self.MIN_LOOKAHEAD_DISTANCE = 1.75 # 최소 전방주시거리 (throttle=0.3일 때) [m]
        self.MAX_LOOKAHEAD_DISTANCE = 2.35 # 최대 전방주시거리 (throttle=0.5일 때) [m]
        self.current_throttle = self.THROTTLE_MIN # 초기 throttle 값, 최소값으로 안전하게 시작
        # ========================================================================================

        self.pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
        self.pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        # ======================= [핵심: Throttle 토픽 구독자] =======================
        self.throttle_sub = rospy.Subscriber('auto_throttle', Float32, self.throttle_callback, queue_size=1)
        # ========================================================================================

        rospy.loginfo("[Lane Follower] Node initialized and waiting for images...")

    # ======================= [핵심: Throttle 콜백 함수] =======================
    def throttle_callback(self, msg):
        """ 'throttle' 토픽을 수신하여 self.current_throttle 값을 업데이트하는 콜백 함수 """
        # 수신된 throttle 값의 범위를 THOTTLE_MIN ~ THROTTLE_MAX 사이로 제한합니다.
        self.current_throttle = np.clip(msg.data, self.THROTTLE_MIN, self.THROTTLE_MAX)
    # ====================================================================================

    def do_bev_transform(self, image):
        M = cv2.getPerspectiveTransform(self.bev_params['src_points'], self.bev_params['dst_points'])
        return cv2.warpPerspective(image, M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        
    def image_to_vehicle(self, pt_bev):
        u, v = pt_bev
        x_vehicle = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        y_vehicle = (self.bev_w / 2 - u) * self.m_per_pixel_x
        return x_vehicle, y_vehicle

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        self.process_image(cv_image)

    def process_image(self, im0s):
        # 1. BEV 변환 및 추론
        bev_image_input = self.do_bev_transform(im0s)
        results = self.model(bev_image_input, imgsz=self.opt.img_size, conf=self.opt.conf_thres, iou=self.opt.iou_thres, device=self.device, verbose=False)
        result = results[0]
        
        # 2. 마스크 처리 및 필터링
        combined_mask_bev = np.zeros(result.orig_shape, dtype=np.uint8)
        if result.masks is not None:
            confidences = result.boxes.conf
            for i, mask_tensor in enumerate(result.masks.data):
                if confidences[i] >= 0.5:
                    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                    if mask_np.shape != result.orig_shape:
                        mask_np = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]))
                    combined_mask_bev = np.maximum(combined_mask_bev, mask_np)
        final_mask = final_filter(combined_mask_bev)
        bev_im_for_drawing = bev_image_input.copy()

        # 3. 필터링된 마스크에서 차선 후보 추출
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        current_detections = []
        if num_labels > 1:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 100:
                    ys, xs = np.where(labels == i)
                    coeff = polyfit_lane(ys, xs, order=2)
                    if coeff is not None:
                        x_at_bottom = np.polyval(coeff, self.bev_h - 1)
                        current_detections.append({'coeff': coeff, 'x_bottom': x_at_bottom})
            current_detections.sort(key=lambda c: c['x_bottom'])

        # 4. 차선 추적 및 스무딩
        left_lane_tracked, right_lane_tracked = self.tracked_lanes['left'], self.tracked_lanes['right']
        current_left, current_right = None, None
        if len(current_detections) == 2: current_left, current_right = current_detections[0], current_detections[1]
        elif len(current_detections) == 1:
            detected_lane = current_detections[0]
            dist_to_left = abs(detected_lane['x_bottom'] - np.polyval(left_lane_tracked['coeff'], self.bev_h - 1)) if left_lane_tracked['coeff'] is not None else float('inf')
            dist_to_right = abs(detected_lane['x_bottom'] - np.polyval(right_lane_tracked['coeff'], self.bev_h - 1)) if right_lane_tracked['coeff'] is not None else float('inf')
            if dist_to_left < dist_to_right and left_lane_tracked['coeff'] is not None: current_left = detected_lane
            elif dist_to_right < dist_to_left and right_lane_tracked['coeff'] is not None: current_right = detected_lane
            else:
                if detected_lane['x_bottom'] < self.bev_w / 2: current_left = detected_lane
                else: current_right = detected_lane
        if current_left:
            if left_lane_tracked['coeff'] is None: left_lane_tracked['coeff'] = current_left['coeff']
            else: left_lane_tracked['coeff'] = (self.SMOOTHING_ALPHA * current_left['coeff'] + (1 - self.SMOOTHING_ALPHA) * left_lane_tracked['coeff'])
            left_lane_tracked['age'] = 0
        else: left_lane_tracked['age'] += 1
        if current_right:
            if right_lane_tracked['coeff'] is None: right_lane_tracked['coeff'] = current_right['coeff']
            else: right_lane_tracked['coeff'] = (self.SMOOTHING_ALPHA * current_right['coeff'] + (1 - self.SMOOTHING_ALPHA) * right_lane_tracked['coeff'])
            right_lane_tracked['age'] = 0
        else: right_lane_tracked['age'] += 1
        if left_lane_tracked['age'] > self.MAX_LANE_AGE: left_lane_tracked['coeff'] = None
        if right_lane_tracked['age'] > self.MAX_LANE_AGE: right_lane_tracked['coeff'] = None

        final_left_coeff, final_right_coeff = left_lane_tracked['coeff'], right_lane_tracked['coeff']
        lane_detected_bool = (final_left_coeff is not None) or (final_right_coeff is not None)
        self.pub_lane_status.publish(Bool(data=lane_detected_bool))
        
        # 5. Pure Pursuit 조향 제어
        steering_angle_deg = None
        goal_point_bev = None 
        dynamic_lookahead_distance = self.MIN_LOOKAHEAD_DISTANCE # 기본값 설정
        
        if lane_detected_bool:
            center_points = []
            LANE_WIDTH_M = 1.5
            lane_width_pixels = LANE_WIDTH_M / self.m_per_pixel_x
            
            for y in range(self.bev_h - 1, self.bev_h // 2, -1):
                x_center = None
                if final_left_coeff is not None and final_right_coeff is not None:
                    x_center = (np.polyval(final_left_coeff, y) + np.polyval(final_right_coeff, y)) / 2
                elif final_left_coeff is not None:
                    x_center = np.polyval(final_left_coeff, y) + lane_width_pixels / 2
                elif final_right_coeff is not None:
                    x_center = np.polyval(final_right_coeff, y) - lane_width_pixels / 2
                if x_center is not None: center_points.append([x_center, y])

            target_center_lane_coeff = None
            if len(center_points) > 10:
                target_center_lane_coeff = polyfit_lane(np.array(center_points)[:, 1], np.array(center_points)[:, 0], order=2)

            if target_center_lane_coeff is not None:
                if self.tracked_center_path['coeff'] is None: self.tracked_center_path['coeff'] = target_center_lane_coeff
                else: self.tracked_center_path['coeff'] = (self.SMOOTHING_ALPHA * target_center_lane_coeff + (1 - self.SMOOTHING_ALPHA) * self.tracked_center_path['coeff'])
            
            if self.tracked_center_path['coeff'] is not None:
                final_center_coeff = self.tracked_center_path['coeff']

                # ======================= [핵심: 정규화 기반 동적 전방주시거리 계산] =======================
                throttle_range = self.THROTTLE_MAX - self.THROTTLE_MIN
                if throttle_range <= 0: # 분모가 0이 되는 오류 방지
                    normalized_throttle = 0.0
                else:
                    normalized_throttle = (self.current_throttle - self.THROTTLE_MIN) / throttle_range
                
                dynamic_lookahead_distance = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE) * normalized_throttle
                # ========================================================================================
                
                goal_point_vehicle = None
                for y_bev in range(self.bev_h - 1, -1, -1):
                    x_bev = np.polyval(final_center_coeff, y_bev)
                    x_veh, y_veh_right = self.image_to_vehicle((x_bev, y_bev))
                    dist = sqrt(x_veh**2 + y_veh_right**2)

                    if dist >= dynamic_lookahead_distance:
                        goal_point_vehicle = (x_veh, y_veh_right)
                        goal_point_bev = (int(x_bev), int(y_bev)) 
                        break
                
                if goal_point_vehicle is not None:
                    x_goal, y_goal = goal_point_vehicle
                    steering_angle_rad = atan2(2.0 * self.L * y_goal, x_goal**2 + y_goal**2)
                    steering_angle_deg = -np.degrees(steering_angle_rad)
                    steering_angle_deg = np.clip(steering_angle_deg, -25.0, 25.0)
                    self.pub_steering.publish(Float32(data=steering_angle_deg))
        
        # 6. 시각화
        annotated_frame = result.plot()
        
        overlay_polyline(bev_im_for_drawing, final_left_coeff, color=(255, 0, 0), step=2, thickness=2)
        overlay_polyline(bev_im_for_drawing, final_right_coeff, color=(0, 0, 255), step=2, thickness=2)
        if self.tracked_center_path['coeff'] is not None:
            overlay_polyline(bev_im_for_drawing, self.tracked_center_path['coeff'], color=(0, 255, 0), step=2, thickness=3)

        if goal_point_bev is not None:
            cv2.circle(bev_im_for_drawing, goal_point_bev, 10, (0, 255, 255), -1) 

        # 시각화 정보 갱신
        steer_text = f"Steer: {steering_angle_deg:.1f} deg" if steering_angle_deg is not None else "Steer: N/A"
        cv2.putText(bev_im_for_drawing, steer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(bev_im_for_drawing, f"Lane Detected: {lane_detected_bool}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 차선 감지 여부와 상관없이 현재 throttle에 따른 lookahead 값을 항상 표시
        throttle_range = self.THROTTLE_MAX - self.THROTTLE_MIN
        if throttle_range <= 0: normalized_throttle = 0.0
        else: normalized_throttle = (self.current_throttle - self.THROTTLE_MIN) / throttle_range
        viz_lookahead = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE) * normalized_throttle
        cv2.putText(bev_im_for_drawing, f"Lookahead: {viz_lookahead:.2f}m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(bev_im_for_drawing, f"Throttle: {self.current_throttle:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        cv2.imshow("Original Camera View", im0s)
        cv2.imshow("Roboflow Detections (on BEV)", annotated_frame)
        cv2.imshow("Final Path (on BEV)", bev_im_for_drawing)
        cv2.waitKey(1)

def main():
    rospy.init_node('lane_follower_node', anonymous=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights2.pt')
    parser.add_argument('--device', default='0')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.6)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--param-file', type=str, default='./bev_params_y_5.npz')
    opt, _ = parser.parse_known_args()

    node = LaneFollowerNode(opt)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down node.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()