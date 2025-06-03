#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import argparse
import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import threading
import queue
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32, Bool

from utils.utils import (
    time_synchronized, select_device, increment_path, lane_line_mask,
    AverageMeter, LoadCamera, LoadImages, letterbox, apply_clahe,
)

class LaneState:
    def __init__(self, lane_type, confidence_increase=5, confidence_decrease=2, max_confidence=100, lost_threshold_frames=15):
        self.type = lane_type # 'left' or 'right' - 이 정체성은 불변
        self.coeff_orig = None
        self.coeff_bev = None
        self.ys_bev = None
        self.xs_bev = None
        self.last_mean_u_bev = None # 추적을 위한 마지막 BEV x 평균 위치
        self.confidence = 0
        self.last_seen_frame = 0
        self.is_detected_this_frame = False # 현재 프레임에서 업데이트(감지)되었는지 여부
        self.CONF_INC = confidence_increase
        self.CONF_DEC = confidence_decrease
        self.MAX_CONF = max_confidence
        self.LOST_THRESH_FRAMES = lost_threshold_frames

    def update(self, detected_lane_info, current_frame):
        self.is_detected_this_frame = True
        self.last_seen_frame = current_frame
        self.confidence = min(self.MAX_CONF, self.confidence + self.CONF_INC)
        
        alpha = 0.3 
        if detected_lane_info['coeff_orig'] is not None:
            if self.coeff_orig is None or self.confidence < 10 : self.coeff_orig = detected_lane_info['coeff_orig']
            else: self.coeff_orig = alpha * detected_lane_info['coeff_orig'] + (1 - alpha) * self.coeff_orig
        
        if detected_lane_info['coeff_bev'] is not None:
            if self.coeff_bev is None or self.confidence < 10: self.coeff_bev = detected_lane_info['coeff_bev']
            else: self.coeff_bev = alpha * detected_lane_info['coeff_bev'] + (1 - alpha) * self.coeff_bev
            
            self.ys_bev = detected_lane_info.get('ys_bev_pts')
            self.xs_bev = detected_lane_info.get('xs_bev_pts')
            
            if self.xs_bev is not None and len(self.xs_bev) > 0:
                self.last_mean_u_bev = np.mean(self.xs_bev)

    def mark_as_undetected(self, current_frame):
        self.is_detected_this_frame = False
        if current_frame - self.last_seen_frame > self.LOST_THRESH_FRAMES:
            self.confidence = max(0, self.confidence - self.CONF_DEC * 3) 
        else:
            self.confidence = max(0, self.confidence - self.CONF_DEC)
            
        if self.confidence == 0: # 신뢰도 0: 차선 완전 소실로 간주, 재탐색(최초 할당) 준비
             self.coeff_orig = None; self.coeff_bev = None
             self.xs_bev = None; self.ys_bev = None
             self.last_mean_u_bev = None # 이전 위치 정보도 초기화

    def get_bev_range(self):
        if self.ys_bev is not None and len(self.ys_bev) > 0:
            try: return (np.min(self.ys_bev), np.max(self.ys_bev))
            except ValueError: return (0,0)
        return (0, 0)

    def get_bev_points(self):
        if self.xs_bev is not None and self.ys_bev is not None and \
           len(self.xs_bev) > 0 and len(self.ys_bev) > 0:
            return self.xs_bev, self.ys_bev
        return None, None

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolopv2.pt', help='model.pt 경로')
    parser.add_argument('--source', type=str, default='/home/highsky/Videos/Webcam/좌회전.mp4', help='source: 0(webcam) 또는 영상/이미지 파일 경로')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO 추론 해상도')
    parser.add_argument('--device', default='0', help='cuda device: 0 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.4, help='차선 세그 임계값')
    parser.add_argument('--save', action='store_false', help='저장하지 않으려면 사용')
    parser.add_argument('--project', default='runs/detect', help='결과 저장 폴더')
    parser.add_argument('--name', default='exp', help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_true', help='기존 폴더 사용 허용')
    parser.add_argument('--frame-skip', type=int, default=0, help='프레임 건너뛰기')
    parser.add_argument('--param-file', type=str, default='./bev_params.npz', help='BEV 파라미터')
    parser.add_argument('--debug', action='store_true', help='Matplotlib 시각화 사용')
    parser.add_argument('--ransac-residual-threshold', type=float, default=5.0, help='RANSAC 임계값')
    parser.add_argument('--ransac-min-samples-ratio', type=float, default=0.05, help='RANSAC 최소 샘플 비율')
    parser.add_argument('--ransac-min-component-pixels', type=int, default=80, help='RANSAC 최소 픽셀 수')
    parser.add_argument('--orig-slope-threshold', type=float, default=0.1, help='원본 기울기 임계값 (최초 할당 시 참고용)') 
    parser.add_argument('--bev-min-valid-points', type=int, default=20, help='BEV 최소 점 개수')
    parser.add_argument('--bev-min-y-range', type=int, default=50, help='BEV 최소 Y 길이')
    parser.add_argument('--steering-alpha', type=float, default=0.3, help='조향각 EMA 알파')
    parser.add_argument('--actual-lane-width-m', type=float, default=1.5, help='실제 차선 폭 (미터)')
    parser.add_argument('--tracking-max-u-distance', type=float, default=100.0, help='추적 시 최대 BEV u(x) 거리 차이 (픽셀)')
    # HINTON (정체성 고정): 최초 차선 할당 시 필요한 최소 크기 (BEV 픽셀 수)
    parser.add_argument('--min-bev-size-for-initial-assignment', type=int, default=50, help='최초 차선 할당을 위한 최소 BEV 픽셀 수')


    return parser

def polyfit_lane(points_y, points_x, order=2):
    if points_y is None or points_x is None or len(points_y) < order + 1: return None
    try: return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError, ValueError): return None

def compute_polyline_points(coeff, y_start, y_end, image_width, step=4):
    points = []
    if coeff is None: return points
    y_start, y_end = int(y_start), int(y_end)
    if y_start >= y_end: return points
    for y_val in range(y_start, y_end, step):
        try:
            x_val = np.polyval(coeff, y_val)
            if 0 <= x_val < image_width: points.append((int(x_val), int(y_val)))
        except Exception: pass
    return points

def overlay_polyline_generic(image, points, color=(0,0,255), thickness=2):
    if points is not None and len(points) > 1:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, thickness)
    return image

def morph_close_internal(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components_internal(binary_mask, min_size=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def debug_plot_lane(shifted_poly_points, image_to_vehicle_func, goal_point=None):
    if not shifted_poly_points: return
    lane_vehicle = [image_to_vehicle_func(pt) for pt in shifted_poly_points]
    if len(lane_vehicle) > 0:
        lane_vehicle = np.array(lane_vehicle)
        plt.figure("Lane in Vehicle Coordinates", figsize=(6,6)); plt.clf()
        plt.plot(lane_vehicle[:,1], lane_vehicle[:,0], 'r-', label="Processed Lane")
        if goal_point is not None: plt.scatter(goal_point[1], goal_point[0], color='g', s=100, label="Goal")
        plt.xlabel("Lateral (m)"); plt.ylabel("Forward (m)"); plt.title("Lane in Vehicle Coords")
        plt.legend(); plt.gca().invert_xaxis(); plt.xlim(1.5,-1.5); plt.ylim(0.0,3.5)
        plt.grid(True); plt.axis('equal'); plt.show(block=False); plt.pause(0.001)

def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):
    cv2.setUseOptimized(True); cudnn.benchmark = True
    bridge = CvBridge(); source = opt.source; weights = opt.weights
    imgsz = opt.img_size; lane_threshold = opt.lane_thres; bev_param_file = opt.param_file
    save_img = not opt.save

    save_dir = Path(increment_path(Path(opt.project)/opt.name, exist_ok=opt.exist_ok))
    if save_img: save_dir.mkdir(parents=True, exist_ok=True); rospy.loginfo(f"[INFO] Saving to {save_dir}")
    else: rospy.loginfo("[INFO] Not saving results.")

    inf_time = AverageMeter(); model = torch.jit.load(weights)
    device = select_device(opt.device); half = (device.type != 'cpu')
    model = model.to(device);
    if half: model.half()
    model.eval()

    bev_params_np = np.load(bev_param_file)
    src_points_from_file = bev_params_np['src_points']

    try:
        roi_trapezoid_points = np.array([
            src_points_from_file[2], src_points_from_file[3],
            src_points_from_file[1], src_points_from_file[0]
        ], dtype=np.int32)
        rospy.loginfo(f"[INFO] ROI trapezoid points reordered and set: {roi_trapezoid_points.tolist()}")
    except IndexError:
        rospy.logfatal(f"[FATAL] 'src_points' in {bev_param_file} does not have 4 points. Please check the file.")
        rospy.signal_shutdown("Error processing bev_params.npz"); return
    except KeyError:
        rospy.logfatal(f"[FATAL] 'src_points' key not found in {bev_param_file}. Please check the file.")
        rospy.signal_shutdown("Error processing bev_params.npz"); return

    dst_points_for_bev_transform = bev_params_np['dst_points']
    bev_w_expected = int(bev_params_np['warp_w'])
    bev_h_expected = int(bev_params_np['warp_h'])
    M_bev = cv2.getPerspectiveTransform(src_points_from_file.astype(np.float32), dst_points_for_bev_transform.astype(np.float32))

    if source.isdigit(): dataset = LoadCamera(source, img_size=imgsz, stride=32)
    else: dataset = LoadImages(source, img_size=imgsz, stride=32)

    m_per_pixel_y = float(bev_params_np.get('m_per_pixel_y', 0.00234375))
    m_per_pixel_x = float(bev_params_np.get('m_per_pixel_x', 0.003125))
    y_offset_m = float(bev_params_np.get('y_offset_m', 1.4))
    actual_lane_width_m = opt.actual_lane_width_m
    if m_per_pixel_x == 0:
        rospy.logwarn_once("[WARN] m_per_pixel_x is 0. Using default lane_width_pixels. Check bev_params.npz.")
        lane_width_pixels_calculated = bev_w_expected * 0.4
    else:
        lane_width_pixels_calculated = actual_lane_width_m / m_per_pixel_x
    rospy.loginfo(f"[INFO] Calculated lane width in BEV pixels: {lane_width_pixels_calculated:.2f} (from {actual_lane_width_m}m)")

    tracking_max_u_distance = opt.tracking_max_u_distance
    min_bev_size_for_initial_assignment = opt.min_bev_size_for_initial_assignment # HINTON (정체성 고정)

    def image_to_vehicle(pt_bev):
        u_bev, v_bev = pt_bev
        x_vehicle = (bev_h_expected - v_bev) * m_per_pixel_y + y_offset_m
        y_vehicle = (bev_w_expected / 2.0 - u_bev) * m_per_pixel_x
        return x_vehicle, y_vehicle

    frame_count = 0
    ransac_residual_thresh = opt.ransac_residual_threshold
    ransac_min_samples_ratio = opt.ransac_min_samples_ratio
    ransac_min_comp_pixels = opt.ransac_min_component_pixels
    orig_slope_classification_thresh = opt.orig_slope_threshold
    bev_min_valid_points_thresh = opt.bev_min_valid_points
    bev_min_y_range_thresh = opt.bev_min_y_range

    roi_mask_orig_img = None
    left_lane = LaneState('left')
    right_lane = LaneState('right')
    filtered_steering_angle = 0.0
    steering_alpha = opt.steering_alpha

    def process_frame(im0s_orig, current_frame_path="frame"):
        nonlocal frame_count, roi_mask_orig_img, left_lane, right_lane, filtered_steering_angle
        frame_count += 1
        im0s_h, im0s_w = im0s_orig.shape[:2]

        if roi_mask_orig_img is None or roi_mask_orig_img.shape[0] != im0s_h or roi_mask_orig_img.shape[1] != im0s_w:
            roi_mask_orig_img = np.zeros((im0s_h, im0s_w), dtype=np.uint8)
            cv2.fillPoly(roi_mask_orig_img, [roi_trapezoid_points], 255)
            rospy.loginfo_once("ROI mask for original image created/updated.")

        im0s_clahe = apply_clahe(im0s_orig.copy())
        net_input_img, _, _ = letterbox(im0s_clahe, (imgsz, imgsz), stride=32)
        net_input_img = net_input_img[:, :, ::-1].transpose(2, 0, 1)
        net_input_img = np.ascontiguousarray(net_input_img)
        img_t = torch.from_numpy(net_input_img).to(device).half() if half else torch.from_numpy(net_input_img).to(device).float()
        img_t /= 255.0
        if img_t.ndimension() == 3: img_t = img_t.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            outputs = model(img_t)
            ll_seg_out = outputs[2] if isinstance(outputs, (list, tuple)) and len(outputs) > 2 else outputs
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        initial_binary_mask_full_res = lane_line_mask(ll_seg_out, threshold=lane_threshold, method='otsu')
        if initial_binary_mask_full_res.shape[0] != im0s_h or initial_binary_mask_full_res.shape[1] != im0s_w:
            initial_binary_mask_full_res = cv2.resize(initial_binary_mask_full_res, (im0s_w, im0s_h), interpolation=cv2.INTER_NEAREST)

        initial_binary_mask_roi = cv2.bitwise_and(initial_binary_mask_full_res, initial_binary_mask_full_res, mask=roi_mask_orig_img)
        
        mask_h_roi, mask_w_roi = initial_binary_mask_roi.shape
        mask_for_ransac_orig = morph_close_internal(initial_binary_mask_roi, ksize=3)
        mask_for_ransac_orig = remove_small_components_internal(mask_for_ransac_orig, min_size=ransac_min_comp_pixels // 2)

        num_labels_orig, labels_orig, stats_orig, _ = cv2.connectedComponentsWithStats(mask_for_ransac_orig, connectivity=8)
        
        all_raw_lane_candidates_this_frame = []
        if num_labels_orig > 1:
            sorted_indices_orig = sorted(range(1, num_labels_orig), key=lambda i: stats_orig[i, cv2.CC_STAT_AREA], reverse=True)
            for i in range(min(len(sorted_indices_orig), 4)): 
                label_idx = sorted_indices_orig[i]
                if stats_orig[label_idx, cv2.CC_STAT_AREA] >= ransac_min_comp_pixels:
                    component_mask = (labels_orig == label_idx).astype(np.uint8) * 255
                    vs_img, us_img = np.where(component_mask > 0)
                    min_pts_for_ransac = max(15, int(len(vs_img) * ransac_min_samples_ratio))
                    if len(vs_img) >= min_pts_for_ransac:
                        X_fit_orig = vs_img.reshape(-1, 1); y_target_fit_orig = us_img
                        poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
                        X_poly_transformed_orig = poly_transformer.fit_transform(X_fit_orig)
                        ransac = RANSACRegressor(LinearRegression(), min_samples=min_pts_for_ransac,
                                                 residual_threshold=ransac_residual_thresh, max_trials=100)
                        coeff_orig = None; mean_u_orig = np.mean(us_img); slope_orig = 0.0
                        try:
                            ransac.fit(X_poly_transformed_orig, y_target_fit_orig)
                            if ransac.estimator_ and hasattr(ransac.estimator_, 'coef_') and len(ransac.estimator_.coef_) == 2:
                                a, b, c = ransac.estimator_.coef_[1], ransac.estimator_.coef_[0], ransac.estimator_.intercept_
                                coeff_orig = np.array([a, b, c])
                                slope_orig = 2 * a * np.max(vs_img) + b 
                                all_raw_lane_candidates_this_frame.append({
                                    'coeff_orig': coeff_orig, 'v_min': np.min(vs_img), 'v_max': np.max(vs_img), 
                                    'mean_u_orig': mean_u_orig, 'points_uv_orig': (us_img, vs_img), 
                                    'size_orig': len(vs_img), 'slope_orig': slope_orig
                                })
                        except Exception as e: rospy.logwarn_throttle(1.0, f"[RANSAC Orig] Fit failed: {e}")
        
        all_bev_candidates = []
        for raw_cand_info in all_raw_lane_candidates_this_frame:
            coeff_o = raw_cand_info['coeff_orig']
            points_uv = compute_polyline_points(coeff_o, raw_cand_info['v_min'], raw_cand_info['v_max'], mask_w_roi, step=2)
            if len(points_uv) > 1:
                transformed_pts_bev = cv2.perspectiveTransform(np.array([points_uv], dtype=np.float32), M_bev)
                if transformed_pts_bev is not None:
                    valid_pts_bev = [(x,y) for x,y in transformed_pts_bev[0] if 0<=x<bev_w_expected and 0<=y<bev_h_expected]
                    if len(valid_pts_bev) >= bev_min_valid_points_thresh:
                        valid_pts_bev_np = np.array(valid_pts_bev)
                        xs_b, ys_b = valid_pts_bev_np[:, 0], valid_pts_bev_np[:, 1]
                        y_range_ok = (ys_b is not None and len(ys_b) > 0 and (np.max(ys_b) - np.min(ys_b) >= bev_min_y_range_thresh)) or \
                                     (ys_b is not None and len(ys_b) == 1 and bev_min_y_range_thresh == 0)
                        if y_range_ok:
                            coeff_b = polyfit_lane(ys_b, xs_b, order=2)
                            if coeff_b is not None:
                                bev_cand_info = raw_cand_info.copy() 
                                bev_cand_info['coeff_bev'] = coeff_b
                                bev_cand_info['xs_bev_pts'] = xs_b
                                bev_cand_info['ys_bev_pts'] = ys_b
                                bev_cand_info['mean_u_bev'] = np.mean(xs_b)
                                bev_cand_info['size_bev'] = len(xs_b) 
                                all_bev_candidates.append(bev_cand_info)
        
        left_lane.is_detected_this_frame = False
        right_lane.is_detected_this_frame = False
        available_candidates = list(all_bev_candidates) 

        # HINTON (정체성 고정): 왼쪽 차선 처리 (추적 또는 최초 할당)
        selected_candidate_for_left = None
        best_match_idx_left = -1
        if left_lane.coeff_bev is not None and left_lane.confidence > 0 : # 이미 '왼쪽 차선' 정체성으로 추적 중
            best_match_score_left = -float('inf') 
            last_known_u_left = left_lane.last_mean_u_bev if left_lane.last_mean_u_bev is not None else bev_w_expected * 0.25 

            for i, cand in enumerate(available_candidates):
                distance_from_last = abs(cand['mean_u_bev'] - last_known_u_left)
                if distance_from_last < tracking_max_u_distance : 
                    # 유사도 점수: 크기가 클수록, 이전 위치와 가까울수록 높음
                    score = cand['size_bev'] * 1.0 - distance_from_last * 0.5 # 가중치 조절 가능
                    if score > best_match_score_left:
                        best_match_score_left = score
                        selected_candidate_for_left = cand
                        best_match_idx_left = i
            
            if selected_candidate_for_left:
                left_lane.update(selected_candidate_for_left, frame_count)
                if best_match_idx_left != -1: available_candidates.pop(best_match_idx_left)
        else: # 왼쪽 차선에 대한 최초 할당 (또는 완전 소실 후 재탐색)
            best_acq_score_left = -1; 
            # HINTON: pop을 위해 idx 추적하나, 이 로직에서는 left가 먼저 선택하므로 pop(0) 형태로 단순화도 가능
            # 단, 그렇게 하려면 아래 right 처리 시 available_candidates를 다시 원본 all_bev_candidates로 해야 함.
            # 현재는 pop(idx)를 위해 idx 유지
            best_acq_idx_left = -1 
            
            # 최초 할당 시에는 공간적 단서와 차선 강도를 함께 사용
            temp_left_candidates = []
            for i, cand in enumerate(available_candidates):
                if cand['mean_u_bev'] < bev_w_expected * 0.48 and cand['size_bev'] >= min_bev_size_for_initial_assignment:
                     # (선택적) 초기 할당 시 원본 이미지 기울기도 참고 가능:
                     # if cand.get('slope_orig', 0) < -orig_slope_classification_thresh * 0.3:
                    temp_left_candidates.append({'candidate': cand, 'index': i, 'score': cand['size_bev']})
            
            if temp_left_candidates:
                temp_left_candidates.sort(key=lambda x: x['score'], reverse=True) # 가장 강한 후보
                selected_candidate_for_left = temp_left_candidates[0]['candidate']
                best_acq_idx_left = temp_left_candidates[0]['index']
                left_lane.update(selected_candidate_for_left, frame_count)
                if best_acq_idx_left != -1: available_candidates.pop(best_acq_idx_left)


        # HINTON (정체성 고정): 오른쪽 차선 처리 (남은 후보 중에서, 유사 로직)
        selected_candidate_for_right = None
        best_match_idx_right = -1
        if right_lane.coeff_bev is not None and right_lane.confidence > 0: # 추적 모드
            best_match_score_right = -float('inf')
            last_known_u_right = right_lane.last_mean_u_bev if right_lane.last_mean_u_bev is not None else bev_w_expected * 0.75

            for i, cand in enumerate(available_candidates): # 남은 후보들 중에서 탐색
                distance_from_last = abs(cand['mean_u_bev'] - last_known_u_right)
                if distance_from_last < tracking_max_u_distance :
                    score = cand['size_bev'] * 1.0 - distance_from_last * 0.5
                    if score > best_match_score_right:
                        best_match_score_right = score
                        selected_candidate_for_right = cand
                        best_match_idx_right = i
            if selected_candidate_for_right:
                right_lane.update(selected_candidate_for_right, frame_count)
                if best_match_idx_right != -1: available_candidates.pop(best_match_idx_right)
        else: # 최초 할당 (오른쪽)
            best_acq_score_right = -1
            best_acq_idx_right = -1
            temp_right_candidates = []
            for i, cand in enumerate(available_candidates): # 남은 후보들 중에서
                if cand['mean_u_bev'] > bev_w_expected * 0.52 and cand['size_bev'] >= min_bev_size_for_initial_assignment:
                    # if cand.get('slope_orig', 0) > orig_slope_classification_thresh * 0.3:
                    temp_right_candidates.append({'candidate': cand, 'index': i, 'score': cand['size_bev']})

            if temp_right_candidates:
                temp_right_candidates.sort(key=lambda x: x['score'], reverse=True)
                selected_candidate_for_right = temp_right_candidates[0]['candidate']
                best_acq_idx_right = temp_right_candidates[0]['index']
                right_lane.update(selected_candidate_for_right, frame_count)
                # Pop은 이미 left에서 인덱스가 바뀌었을 수 있으므로 주의. 
                # 여기서는 available_candidates가 계속 수정되므로, pop(idx)는 이전 인덱스 기준.
                # 더 안전하게 하려면, 선택된 객체를 직접 remove 하거나, 인덱스를 다시 찾아 pop.
                # 여기서는 left에서 이미 pop 했으므로, right는 남은 것 중 최고를 고르고 pop.
                # 하지만 left가 pop한 후 right의 반복문 인덱스는 다시 0부터 시작하므로,
                # pop(best_acq_idx_right)은 available_candidates의 현재 인덱스를 사용해야 함.
                # 가장 간단한 방법은, right도 left처럼 처리 후, 선택된 candidate 객체를 available_candidates.remove() 하는 것.
                # 일단 현재 로직은 left가 pop한 후의 available_candidates를 right가 사용함.
                if best_acq_idx_right != -1:
                    # available_candidates 리스트는 left 처리 후 줄어들었을 수 있으므로,
                    # best_acq_idx_right는 원래 all_bev_candidates 기준 인덱스.
                    # selected_candidate_for_right 객체를 직접 제거하는 것이 안전.
                    if selected_candidate_for_right in available_candidates:
                        available_candidates.remove(selected_candidate_for_right)


        if not left_lane.is_detected_this_frame: left_lane.mark_as_undetected(frame_count)
        if not right_lane.is_detected_this_frame: right_lane.mark_as_undetected(frame_count)
        
        im0s_vis = im0s_orig.copy() # HINTON: 시각화 전에 원본 이미지 다시 복사 (이전 RANSAC 시각화 덮어쓰기 방지)
        cv2.polylines(im0s_vis, [roi_trapezoid_points], isClosed=True, color=(0,255,255), thickness=1) # ROI는 계속 그림
        bev_im_color_visualization = cv2.warpPerspective(im0s_orig, M_bev, (bev_w_expected, bev_h_expected))
        
        active_coeff_for_steering = None
        path_start_point_bev = (bev_w_expected // 2, bev_h_expected -1)
        current_lane_width_pixels = lane_width_pixels_calculated
        confidence_threshold_viz = 5 
        confidence_threshold_path = 20 

        if left_lane.confidence > confidence_threshold_viz and left_lane.coeff_bev is not None:
            left_bev_y_range = left_lane.get_bev_range()
            bev_lane_pts = compute_polyline_points(left_lane.coeff_bev, left_bev_y_range[0], left_bev_y_range[1], bev_w_expected, step=5)
            bev_im_color_visualization = overlay_polyline_generic(bev_im_color_visualization, bev_lane_pts, color=(0,0,255), thickness=2)
            cv2.putText(bev_im_color_visualization, f"L:{left_lane.confidence}", (10, bev_h_expected - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            if left_lane.coeff_orig is not None: # 원본 이미지에도 그림
                 lane_pts_orig_vis = compute_polyline_points(left_lane.coeff_orig, 0, mask_h_roi, mask_w_roi, step=5)
                 im0s_vis = overlay_polyline_generic(im0s_vis, lane_pts_orig_vis, color=(0,0,255), thickness=2)


        if right_lane.confidence > confidence_threshold_viz and right_lane.coeff_bev is not None:
            right_bev_y_range = right_lane.get_bev_range()
            bev_lane_pts = compute_polyline_points(right_lane.coeff_bev, right_bev_y_range[0], right_bev_y_range[1], bev_w_expected, step=5)
            bev_im_color_visualization = overlay_polyline_generic(bev_im_color_visualization, bev_lane_pts, color=(255,255,0), thickness=2)
            cv2.putText(bev_im_color_visualization, f"R:{right_lane.confidence}", (bev_w_expected - 70, bev_h_expected - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            if right_lane.coeff_orig is not None: # 원본 이미지에도 그림
                 lane_pts_orig_vis = compute_polyline_points(right_lane.coeff_orig, 0, mask_h_roi, mask_w_roi, step=5)
                 im0s_vis = overlay_polyline_generic(im0s_vis, lane_pts_orig_vis, color=(255,255,0), thickness=2)

        # ... (나머지 경로 생성, 조향각 계산, ROS 퍼블리시, 화면 출력, 저장 로직은 이전과 동일)
        if left_lane.confidence > confidence_threshold_path and right_lane.confidence > confidence_threshold_path and \
           left_lane.coeff_bev is not None and right_lane.coeff_bev is not None:
            active_coeff_for_steering = (left_lane.coeff_bev + right_lane.coeff_bev) / 2.0
            xs_l, ys_l = left_lane.get_bev_points(); xs_r, ys_r = right_lane.get_bev_points()
            if ys_l is not None and ys_r is not None:
                y_b = max(np.max(ys_l), np.max(ys_r))
                x_l = np.polyval(left_lane.coeff_bev, y_b); x_r = np.polyval(right_lane.coeff_bev, y_b)
                path_start_point_bev = (int((x_l + x_r) / 2), int(y_b))
        elif left_lane.confidence > confidence_threshold_path and left_lane.coeff_bev is not None:
            active_coeff_for_steering = left_lane.coeff_bev.copy(); active_coeff_for_steering[2] += current_lane_width_pixels / 2.0
            xs_l, ys_l = left_lane.get_bev_points()
            if ys_l is not None:
                y_b = np.max(ys_l); x_l = np.polyval(left_lane.coeff_bev, y_b)
                path_start_point_bev = (int(x_l + current_lane_width_pixels / 2.0), int(y_b))
        elif right_lane.confidence > confidence_threshold_path and right_lane.coeff_bev is not None:
            active_coeff_for_steering = right_lane.coeff_bev.copy(); active_coeff_for_steering[2] -= current_lane_width_pixels / 2.0
            xs_r, ys_r = right_lane.get_bev_points()
            if ys_r is not None:
                y_b = np.max(ys_r); x_r = np.polyval(right_lane.coeff_bev, y_b)
                path_start_point_bev = (int(x_r - current_lane_width_pixels / 2.0), int(y_b))
        else:
            rospy.logwarn_throttle(1.0, "[WARN] Both lanes uncertain or missing for path generation.")

        main_lane_poly_points = []
        if active_coeff_for_steering is not None:
            y_start_for_path = int(path_start_point_bev[1])
            x_on_coeff_at_start_y = np.polyval(active_coeff_for_steering, y_start_for_path)
            x_offset = path_start_point_bev[0] - x_on_coeff_at_start_y
            for y_val in range(y_start_for_path, 0, -4):
                x_val_on_coeff = np.polyval(active_coeff_for_steering, y_val)
                x_val_translated = int(x_val_on_coeff + x_offset)
                if 0 <= x_val_translated < bev_w_expected:
                     main_lane_poly_points.append((x_val_translated, y_val))
            if main_lane_poly_points:
                bev_im_color_visualization = overlay_polyline_generic(bev_im_color_visualization, main_lane_poly_points, color=(255,0,255), thickness=3)
        cv2.circle(bev_im_color_visualization, path_start_point_bev, 10, (0,0,255), -1)

        lane_detected_bool = left_lane.confidence > confidence_threshold_path or right_lane.confidence > confidence_threshold_path
        pub_lane_status.publish(Bool(data=lane_detected_bool))

        if len(main_lane_poly_points) > 1:
            lookahead_m=2.0; wheelbase_m=0.75; goal_pt_veh=None; min_err=float('inf')
            path_veh = [image_to_vehicle(pt) for pt in main_lane_poly_points]
            for Xv, Yv in path_veh:
                if Xv < 0.1 : continue
                err = abs(Xv - lookahead_m)
                if err < min_err: min_err=err; goal_pt_veh=(Xv,Yv)
            if goal_pt_veh is None and path_veh:
                for Xv_path, Yv_path in reversed(path_veh):
                    if Xv_path > 0.5 : goal_pt_veh = (Xv_path, Yv_path); break
                if goal_pt_veh is None: goal_pt_veh = path_veh[0]
            if goal_pt_veh:
                Xg,Yg=goal_pt_veh; alpha_rad=np.arctan2(Yg,Xg); ld=np.sqrt(Xg**2+Yg**2)
                steer_rad=np.arctan2(2*wheelbase_m*np.sin(alpha_rad),ld) if ld > 1e-3 else 0.0
                steer_deg=np.degrees(steer_rad); steer_deg_clip=np.clip(steer_deg,-30.,30.)
                filtered_steering_angle = steering_alpha * steer_deg_clip + (1 - steering_alpha) * filtered_steering_angle
                pub_steering.publish(Float32(data=filtered_steering_angle))
                rospy.loginfo_throttle(0.2,"[INFO] SteerRaw:%.1f Clip:%.1f Filt:%.1f (G:X%.2f Y%.2f Alp:%.1f ld:%.2f)",
                                       steer_deg, steer_deg_clip, filtered_steering_angle, Xg, Yg, np.degrees(alpha_rad), ld)
                gv_img=int(bev_h_expected-(Xg-y_offset_m)/m_per_pixel_y); gu_img=int(bev_w_expected/2.0-Yg/m_per_pixel_x)
                if 0<=gu_img<bev_w_expected and 0<=gv_img<bev_h_expected: cv2.circle(bev_im_color_visualization,(gu_img,gv_img),8,(0,255,0),-1)
                cv2.putText(bev_im_color_visualization,f"Steer:{filtered_steering_angle:.1f}deg",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            if opt.debug and main_lane_poly_points: debug_plot_lane(main_lane_poly_points,image_to_vehicle,goal_pt_veh)
        else:
            rospy.logwarn_throttle(1.0, "[WARN] No path generated, holding steering angle.")
            pub_steering.publish(Float32(data=filtered_steering_angle))
        
        bev_mask_roi_display = cv2.warpPerspective(initial_binary_mask_roi, M_bev, (bev_w_expected, bev_h_expected))
        if pub_mask.get_num_connections() > 0:
            try: pub_mask.publish(bridge.cv2_to_imgmsg(bev_mask_roi_display, encoding="mono8"))
            except CvBridgeError as e: rospy.logerr(f"CvBridge Error (mask pub): {e}")

        cv2.imshow("BEV Path (Identity Tracking)", bev_im_color_visualization) # 창 이름 변경
        cv2.imshow("BEV Mask from ROI", bev_mask_roi_display)
        cv2.imshow("Original Image (Identity Tracking)", im0s_vis) # 창 이름 변경

        if save_img:
            save_path_base = save_dir / Path(current_frame_path).name.replace('.','_')
            cv2.imwrite(str(save_path_base) + "_0_orig_identity_track.jpg", im0s_vis) # 파일 이름 변경
            cv2.imwrite(str(save_path_base) + "_1_bev_identity_track.jpg", bev_im_color_visualization) # 파일 이름 변경
        return bev_im_color_visualization, bev_mask_roi_display

    # --- 웹캠/비디오 스트림 처리 루프 (이전과 동일) ---
    if dataset.mode == 'stream':
        frame_skip = opt.frame_skip; frame_counter_raw = 0
        frame_queue = queue.Queue(maxsize=5); producer_stop_event = threading.Event()
        def frame_producer():
            try:
                for item_idx, item_data in enumerate(dataset):
                    if producer_stop_event.is_set(): break
                    if frame_queue.full():
                        try: frame_queue.get_nowait()
                        except queue.Empty: pass
                    path_item, _, im0s, _ = item_data
                    current_path = path_item if path_item else f"webcam_frame_{item_idx}"
                    frame_queue.put((current_path, im0s))
            except StopIteration: rospy.loginfo("Frame producer: End of stream.")
            except Exception as e: rospy.logerr(f"Frame producer error: {e}")
            finally: frame_queue.put(None)
        producer_thread = threading.Thread(target=frame_producer); producer_thread.daemon = True; producer_thread.start()
        rospy.loginfo("[INFO] Async frame producer (Identity Tracking) started.")
        while not rospy.is_shutdown():
            try: queued_item = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not producer_thread.is_alive() and frame_queue.empty(): rospy.loginfo("Producer stopped, queue empty. Exiting."); break
                continue
            if queued_item is None: rospy.loginfo("End signal from producer."); break
            current_frame_path, im0s_from_queue = queued_item
            if frame_skip > 0 and frame_counter_raw % (frame_skip + 1) != 0: frame_counter_raw += 1; continue
            frame_counter_raw += 1
            process_frame(im0s_from_queue, current_frame_path)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): rospy.loginfo("User 'q' pressed."); producer_stop_event.set(); break
            elif key == ord('s') and save_img: cv2.imwrite(str(save_dir / f"manual_save_frame_{frame_count}.jpg"), im0s_from_queue); rospy.loginfo("Frame saved.")
            elif key == ord('p'):
                rospy.loginfo("Paused. Press 'p' to resume.")
                while not rospy.is_shutdown():
                    if cv2.waitKey(0) & 0xFF == ord('p'): rospy.loginfo("Resumed."); break
                    if rospy.is_shutdown(): break
                if rospy.is_shutdown(): break
        producer_stop_event.set()
        if producer_thread.is_alive(): producer_thread.join(timeout=2.0)
    else: # 저장된 영상/이미지
        rospy.loginfo("[INFO] Starting sync processing (Identity Tracking).")
        delay = 1
        for frame_idx, frame_data in enumerate(dataset):
            if rospy.is_shutdown(): break
            current_frame_path, _, im0s_from_file, _ = frame_data
            if dataset.mode == 'video' and dataset.cap is not None:
                fps_video = dataset.cap.get(cv2.CAP_PROP_FPS)
                delay = int(1000 / fps_video) if fps_video > 0 else 33
            start_t = time.time()
            process_frame(im0s_from_file, current_frame_path)
            elapsed_ms = (time.time() - start_t) * 1000
            current_delay = max(1, delay - int(elapsed_ms))
            key = cv2.waitKey(current_delay) & 0xFF
            if key == ord('q'): rospy.loginfo("User 'q' pressed."); break
            elif key == ord('s') and save_img: cv2.imwrite(str(save_dir / f"manual_save_f_{Path(current_frame_path).stem}_{frame_idx}.jpg"), im0s_from_file); rospy.loginfo("Frame saved.")
        rospy.loginfo(f"[INFO] Sync (Identity Tracking) processing avg inference: {inf_time.avg:.4f}s/frame")

    if hasattr(dataset, 'release'): dataset.release()
    cv2.destroyAllWindows()
    if opt.debug: plt.ioff(); plt.close('all')
    rospy.loginfo("[INFO] (Identity Tracking) processing completed.")

def ros_main():
    rospy.init_node('bev_lane_follower_identity_tracking_node', anonymous=True) # 노드 이름 변경
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    if opt.debug: plt.ion()
    pub_mask = rospy.Publisher('camera_bev_lane_mask_identity_track', Image, queue_size=2) # 토픽 이름 변경
    pub_steering = rospy.Publisher('auto_steer_angle_lane_identity_track', Float32, queue_size=2) # 토픽 이름 변경
    pub_lane_status = rospy.Publisher('lane_detection_status_identity_track', Bool, queue_size=2) # 토픽 이름 변경
    rospy.loginfo("BEV Lane Follower (Identity Tracking) Node initialized.")
    rospy.loginfo(f"Run options: {vars(opt)}")
    try:
        detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)
    except Exception as e:
        rospy.logfatal(f"Critical error in detect_and_publish (Identity Tracking): {e}", exc_info=True)
    finally:
        rospy.loginfo("ROS Main (Identity Tracking) shutting down...")

if __name__ == '__main__':
    try:
        ros_main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node (Identity Tracking) interrupted.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in main (Identity Tracking): {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()
        if plt.get_fignums(): plt.close('all')
        rospy.loginfo("Program (Identity Tracking) terminated.")