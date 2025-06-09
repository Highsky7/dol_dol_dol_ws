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
from math import atan2, degrees
from pathlib import Path
import matplotlib.pyplot as plt

# --- 추가된 임포트 ---
try:
    from skimage.morphology import skeletonize
    from skimage.util import img_as_ubyte
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    rospy.logwarn("[WARN] scikit-image not found. MAT will be skipped.")
    rospy.logwarn("[WARN] Try: pip install scikit-image")
# --------------------

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

# argparse 설정 (기존과 동일)
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolopv2.pt', help='model.pt 경로')
    parser.add_argument('--source', type=str,
                        default='2',
                        # default='/home/highsky/Videos/Webcam/좌회전.mp4',
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
    parser.add_argument('--prune-iter', type=int, default=5, help='Number of iterations for pruning spurs') # 가지치기 반복 횟수 인자 추가
    return parser

# --- 기존 함수들 (polyfit_lane, compute_polyline_points, overlay_polyline, morph_close, remove_small_components, keep_top2_components, final_filter, do_bev_transform, debug_plot_lane) ---
# (이전 코드와 동일, 생략. 아래 코드에 포함되어 있음)
# 폴리피팅 유틸
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
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
    return cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

def debug_plot_lane(shifted_poly_points, image_to_vehicle_func, goal_point=None):
    if not shifted_poly_points: return
    lane_vehicle = [image_to_vehicle_func(pt) for pt in shifted_poly_points]
    if len(lane_vehicle) > 0:
        lane_vehicle = np.array(lane_vehicle)
        plt.figure("Lane in Vehicle Coordinates", figsize=(6, 6)); plt.clf()
        plt.plot(lane_vehicle[:, 1], lane_vehicle[:, 0], 'r-', label="Processed Lane")
        if goal_point is not None: plt.scatter(goal_point[1], goal_point[0], color='green', s=100, label="Goal Point")
        plt.xlabel("Lateral (m)"); plt.ylabel("Forward (m)"); plt.title("Lane Line in Vehicle Coordinates")
        plt.legend(); plt.gca().invert_xaxis(); plt.xlim(1.5, -1.5); plt.ylim(0.0, 3.5)
        plt.grid(True); plt.axis('equal'); plt.show(block=False); plt.pause(0.001)
# -------------------------------------------------------------------------------

# --- MAT 및 가지치기 함수 추가 ---
def mat_and_prune(binary_mask, prune_iterations=5):
    """
    Applies Medial Axis Transform (Skeletonization) and Pruning.
    Args:
        binary_mask (np.ndarray): Input binary image (0 or 255).
        prune_iterations (int): Number of times to remove endpoints.
    Returns:
        np.ndarray: Pruned skeleton image (0 or 255).
    """
    if not SKIMAGE_AVAILABLE or np.sum(binary_mask) == 0:
        rospy.logwarn_throttle(5.0, "[WARN] Skipping MAT+Pruning (Skimage not found or empty mask).")
        # 스키마 이미지가 없거나 마스크가 비어 있으면 원본 반환 (또는 ximgproc 사용)
        try:
            # 대체: ximgproc 사용 시도
            import cv2.ximgproc as ximgproc
            return ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_GUOHALL)
        except:
             return binary_mask


    # 1. Skeletonize (scikit-image 사용)
    # skeletonize는 bool 또는 0/1 입력을 기대합니다.
    skeleton = skeletonize(binary_mask > 0)
    # 결과를 uint8 (0/255)로 변환
    skeleton_img = img_as_ubyte(skeleton)

    # 2. Pruning (가지치기)
    pruned_skeleton = skeleton_img.copy()
    
    # 이웃 픽셀 수를 계산하기 위한 커널
    kernel_neighbors = np.array([[1, 1, 1],
                                 [1, 10, 1], # 중앙 픽셀 가중치를 높여 자신은 세지 않도록 함
                                 [1, 1, 1]], dtype=np.uint8)

    for i in range(prune_iterations):
        # 현재 스켈레톤에서만 이웃 수 계산 (0/1 이미지로 변환 후)
        skel_01 = (pruned_skeleton // 255).astype(np.uint8)
        neighbors_count = cv2.filter2D(skel_01, -1, kernel_neighbors, borderType=cv2.BORDER_CONSTANT)

        # 끝점 찾기: 스켈레톤 픽셀이면서 이웃 수가 1개인 픽셀
        # neighbors_count 값은 (자신(10) + 이웃 수) 이므로, 이웃이 1개면 값은 11.
        endpoints = cv2.bitwise_and(pruned_skeleton, ((neighbors_count == 11) * 255).astype(np.uint8))

        # 제거할 끝점이 없으면 종료
        if np.sum(endpoints) == 0:
            rospy.loginfo(f"[DEBUG] Pruning finished after {i} iterations.")
            break
            
        # 끝점 제거
        pruned_skeleton = cv2.subtract(pruned_skeleton, endpoints)

    rospy.loginfo_throttle(1.0, f"[DEBUG] MAT+Pruning applied ({prune_iterations} iter).")
    return pruned_skeleton
# ---------------------------------


# 메인 처리 함수
def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):
    # ... (기존 설정 코드 - bridge, source, weights, imgsz, lane_threshold, bev_param_file 등) ...
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
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
    if half: model.half()
    model.eval()
    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치=%s) 열기", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=32)
    else:
        rospy.loginfo("[INFO] 파일(영상/이미지): %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=32)
    bev_params = np.load(bev_param_file)
    bev_h_expected = int(bev_params['warp_h'])
    bev_w_expected = int(bev_params['warp_w'])
    m_per_pixel_y = 0.00234375
    y_offset_m = 1.4
    m_per_pixel_x = 0.003125
    def image_to_vehicle(pt_bev):
        u, v = pt_bev
        x_vehicle = (bev_h_expected - v) * m_per_pixel_y + y_offset_m
        y_vehicle = (bev_w_expected / 2 - u) * m_per_pixel_x
        return x_vehicle, y_vehicle
    # ---------------------------------------------------------------------------------------

    frame_count = 0
    def process_frame(im0s):
        nonlocal frame_count
        frame_count += 1
        net_input_img, _, _ = letterbox(im0s, (imgsz, imgsz), stride=32) # 수정: ratio, pad 받기
        net_input_img = net_input_img[:, :, ::-1].transpose(2, 0, 1)
        net_input_img = np.ascontiguousarray(net_input_img)
        img_t = torch.from_numpy(net_input_img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            # YOLOPv2 모델 출력이 [da_seg_out, ll_seg_out] 또는 [det_out, da_seg_out, ll_seg_out] 일 수 있음
            # 제공된 코드에서는 [_, _], seg, ll = model(img_t) 형태 사용. 이는 ll이 3번째 출력임을 가정
            # 모델 구조에 따라 인덱스 조정 필요
            outputs = model(img_t)
            if len(outputs) == 3:
                 _, seg, ll = outputs # 예: [det, da, ll]
            elif len(outputs) == 2:
                 seg, ll = outputs # 예: [da, ll]
            else:
                 rospy.logerr("모델 출력 형식 불일치!")
                 return None, None # 처리 불가

        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='otsu') # 원본은 otsu 사용
        
        # --- thinning 부분을 MAT+Pruning으로 대체 ---
        # 원본의 첫 번째 thinning은 제거하거나 유지할 수 있음. 여기서는 제거하고 BEV에서만 처리.
        # thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_GUOHALL)

        bev_mask_orig = do_bev_transform(binary_mask, bev_param_file) # Thinning 전 마스크를 BEV로 변환
        bevfilter_mask = final_filter(bev_mask_orig) # 주요 차선 후보 마스크 (최대 2개)
        
        # --- bevfilter_mask에 MAT+Pruning 적용 ---
        final_mask = mat_and_prune(bevfilter_mask, prune_iterations=opt.prune_iter)
        # ----------------------------------------

        # ... (이하 경로 시작점 및 차선 처리 로직은 이전 코드와 거의 동일) ...
        # (이전 코드에서 차선 처리, 퓨어퍼슛, 시각화 부분 복사)
        bev_im_for_shape = do_bev_transform(im0s, bev_param_file) # 원본 BEV (컬러)
        h_bev, w_bev = bev_im_for_shape.shape[:2]
        bev_im_color = bev_im_for_shape.copy() # 시각화용 BEV 이미지

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        
        lane_components = []
        if num_labels > 1:
            sorted_components_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            for i in range(min(len(sorted_components_indices), 2)):
                label_idx = sorted_components_indices[i]
                if stats[label_idx, cv2.CC_STAT_AREA] >= 50: # MAT 결과는 픽셀 수가 적으므로 임계값 조정
                    component_mask = np.zeros_like(final_mask)
                    component_mask[labels == label_idx] = 255
                    ys, xs = np.where(component_mask > 0)
                    if len(ys) > 4 :
                        coeff = polyfit_lane(ys, xs, order=2)
                        if coeff is not None:
                            lane_components.append({'ys': ys, 'xs': xs, 'coeff': coeff, 'label_idx': label_idx})
        
        for comp in lane_components:
            bev_im_color = overlay_polyline(bev_im_color, comp['coeff'], color=(0, 255, 0), step=4)

        lane_detected_bool = len(lane_components) > 0
        pub_lane_status.publish(Bool(data=lane_detected_bool))
        
        main_lane_poly_points = []
        path_start_point_bev = (w_bev // 2, h_bev -1)

        if lane_detected_bool:
            active_coeff_for_steering = None

            if len(lane_components) == 1:
                comp = lane_components[0]
                active_coeff_for_steering = comp['coeff']
                max_y_val = np.max(comp['ys'])
                lane_bottom_x_at_max_y = int(np.polyval(comp['coeff'], max_y_val))
                lane_bottom_point = (lane_bottom_x_at_max_y, int(max_y_val))
                cv2.circle(bev_im_color, lane_bottom_point, 7, (255, 255, 0), -1)
                min_y_for_slope, max_y_for_slope = np.min(comp['ys']), np.max(comp['ys'])
                slope_estimate = 0
                if (max_y_for_slope - min_y_for_slope) > 10 :
                     x_at_min_y = np.polyval(comp['coeff'], min_y_for_slope)
                     x_at_max_y = np.polyval(comp['coeff'], max_y_for_slope)
                     slope_estimate = (x_at_max_y - x_at_min_y) / (max_y_for_slope - min_y_for_slope + 1e-6)
                if slope_estimate > 0.15: ref_point = (0, h_bev -1)
                elif slope_estimate < -0.15: ref_point = (w_bev -1, h_bev -1)
                else: ref_point = (w_bev // 2, h_bev -1)
                path_start_point_bev = ((lane_bottom_point[0] + ref_point[0]) // 2, (lane_bottom_point[1] + ref_point[1]) // 2)

            elif len(lane_components) == 2:
                comp1, comp2 = lane_components[0], lane_components[1]
                max_y1_val = np.max(comp1['ys']); lane1_bottom_x = int(np.polyval(comp1['coeff'], max_y1_val)); lane1_bottom_point = (lane1_bottom_x, int(max_y1_val))
                max_y2_val = np.max(comp2['ys']); lane2_bottom_x = int(np.polyval(comp2['coeff'], max_y2_val)); lane2_bottom_point = (lane2_bottom_x, int(max_y2_val))
                cv2.circle(bev_im_color, lane1_bottom_point, 7, (255, 128, 0), -1); cv2.circle(bev_im_color, lane2_bottom_point, 7, (255, 128, 0), -1)
                path_start_point_bev = ( (lane1_bottom_point[0] + lane2_bottom_point[0]) // 2, (lane1_bottom_point[1] + lane2_bottom_point[1]) // 2 )
                active_coeff_for_steering = (comp1['coeff'] + comp2['coeff']) / 2.0

            if active_coeff_for_steering is not None:
                poly_points_from_active_coeff = compute_polyline_points(active_coeff_for_steering, bev_im_color.shape, step=4)
                if len(poly_points_from_active_coeff) > 0:
                    current_polyline_bottom_point = poly_points_from_active_coeff[-1]
                    translation = (path_start_point_bev[0] - current_polyline_bottom_point[0], path_start_point_bev[1] - current_polyline_bottom_point[1])
                    main_lane_poly_points = [(pt[0] + translation[0], pt[1] + translation[1]) for pt in poly_points_from_active_coeff]
                    if len(main_lane_poly_points) > 1:
                        cv2.polylines(bev_im_color, [np.array(main_lane_poly_points, dtype=np.int32)], False, (255, 0, 255), 2)
            cv2.circle(bev_im_color, path_start_point_bev, 10, (0, 0, 255), -1)

            if len(main_lane_poly_points) > 0:
                lookahead_m = 2.10
                wheelbase_m = 0.75
                goal_point_vehicle_coords = None
                min_dist_error = float('inf')
                for pt_img in main_lane_poly_points:
                    X_v, Y_v = image_to_vehicle(pt_img)
                    current_dist_from_vehicle = np.sqrt(X_v**2 + Y_v**2)
                    error = abs(current_dist_from_vehicle - lookahead_m)
                    if error < min_dist_error:
                        min_dist_error = error
                        goal_point_vehicle_coords = (X_v, Y_v)
                if goal_point_vehicle_coords is None and main_lane_poly_points:
                    goal_point_vehicle_coords = image_to_vehicle(main_lane_poly_points[0])
                if goal_point_vehicle_coords is not None:
                    X_v_goal, Y_v_goal = goal_point_vehicle_coords
                    alpha = np.arctan2(Y_v_goal, X_v_goal)
                    ld = np.sqrt(X_v_goal**2 + Y_v_goal**2)
                    steering_angle_rad = np.arctan((2 * wheelbase_m * np.sin(alpha)) / ld) if ld > 1e-5 else 0.0
                    steering_angle_deg = np.degrees(steering_angle_rad)
                    pub_steering.publish(Float32(data=steering_angle_deg))
                    rospy.loginfo_throttle(0.2, "[INFO] Steer: %.2f deg (Alpha: %.2f deg, Goal: X:%.2fm Y:%.2fm)", steering_angle_deg, np.degrees(alpha), X_v_goal, Y_v_goal)
                    goal_v_img = int(h_bev - (X_v_goal - y_offset_m) / m_per_pixel_y)
                    goal_u_img = int(w_bev/2 - Y_v_goal / m_per_pixel_x)
                    if 0 <= goal_u_img < w_bev and 0 <= goal_v_img < h_bev:
                         cv2.circle(bev_im_color, (goal_u_img, goal_v_img), 8, (0, 255, 0), -1)
                    cv2.putText(bev_im_color, f"Steer: {steering_angle_deg:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                if opt.debug: debug_plot_lane(main_lane_poly_points, image_to_vehicle, goal_point_vehicle_coords)
            else:
                if lane_detected_bool: rospy.logwarn_throttle(1.0,"[WARNING] Lane found, but path polyline could not be generated.")
        
        if pub_mask.get_num_connections() > 0:
            try: pub_mask.publish(bridge.cv2_to_imgmsg(final_mask, encoding="mono8"))
            except CvBridgeError as e: rospy.logerr(f"CvBridge Error: {e}")

        cv2.imshow("BEV (Processed Path)", bev_im_color)
        cv2.imshow("Final Lane Mask (MAT+Pruned)", final_mask) # 창 이름 변경
        
        return bev_im_color, final_mask

    # --- 웹캠/비디오 스트림 처리 루프 (기존과 동일) ---
    # (이전 코드와 동일, 생략. 아래 코드에 포함되어 있음)
    if dataset.mode == 'stream':
        frame_skip = opt.frame_skip
        frame_counter = 0
        frame_queue = queue.Queue(maxsize=5) 
        def frame_producer():
            try:
                for item_idx, item_data in enumerate(dataset):
                    path_item, net_input_img_data, im0s_data, vid_cap_data = item_data # Unpack
                    if frame_queue.full():
                        try: frame_queue.get_nowait()
                        except queue.Empty: pass
                    frame_queue.put((path_item, im0s_data)) # Put only necessary im0s
            except Exception as e: rospy.logerr(f"Frame producer error: {e}")
            finally: frame_queue.put(None)

        producer_thread = threading.Thread(target=frame_producer); producer_thread.daemon = True; producer_thread.start()
        rospy.loginfo("[DEBUG] 웹캠 비동기 프레임 생산 시작")

        while not rospy.is_shutdown():
            try: frame_data_tuple = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not producer_thread.is_alive() and frame_queue.empty(): break
                continue
            if frame_data_tuple is None: break
            
            path_item, im0s = frame_data_tuple # Unpack im0s

            if frame_skip > 0 and frame_counter % (frame_skip + 1) != 0: frame_counter += 1; continue
            frame_counter += 1
            
            process_frame(im0s)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): rospy.loginfo("User pressed 'q', exiting."); break
            elif key == ord('p'):
                rospy.loginfo("Paused. Press 'p' again to resume.")
                while True:
                    if cv2.waitKey(0) & 0xFF == ord('p'): rospy.loginfo("Resumed."); break
    else:
        rospy.loginfo("[DEBUG] 저장된 영상/이미지 동기 처리 시작")
        delay = 1
        if dataset.mode == 'video' and hasattr(dataset, 'cap') and dataset.cap is not None:
            fps = dataset.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0: delay = int(1000 / fps)
        
        for frame_idx, (path_item, _, im0s, _) in enumerate(dataset): # Unpack correctly
            if rospy.is_shutdown(): break
            processing_start_time = time.time()
            process_frame(im0s)
            processing_end_time = time.time()
            elapsed_processing_ms = (processing_end_time - processing_start_time) * 1000
            current_delay = max(1, delay - int(elapsed_processing_ms))
            if cv2.waitKey(current_delay) & 0xFF == ord('q'): rospy.loginfo("User pressed 'q', exiting."); break
        rospy.loginfo("[INFO] 동기 처리 추론 평균 시간: %.4fs/frame", inf_time.avg)
    # ---------------------------------------------------------------------------------------

    if hasattr(dataset, 'release'): dataset.release()
    cv2.destroyAllWindows()
    if opt.debug: plt.ioff(); plt.close('all')
    rospy.loginfo("[INFO] 처리 완료 및 종료.")


# 메인 함수 (기존과 동일)
def ros_main():
    rospy.init_node('bev_lane_follower_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    if opt.debug: plt.ion()
    pub_mask = rospy.Publisher('camera_bev_lane_mask', Image, queue_size=1)
    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
    rospy.loginfo("BEV Lane Follower Node 시작")
    rospy.loginfo(f"옵션: {opt}")
    try: detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)
    except Exception as e: rospy.logerr(f"detect_and_publish 함수에서 예외 발생: {e}"); import traceback; rospy.logerr(traceback.format_exc())
    finally: rospy.loginfo("ROS Main 함수 종료 중...")

if __name__ == '__main__':
    try: ros_main()
    except rospy.ROSInterruptException: rospy.loginfo("ROS 노드 종료됨 (ROSInterruptException)")
    except Exception as e: rospy.logfatal(f"알 수 없는 심각한 오류로 프로그램 종료: {e}"); import traceback; rospy.logfatal(traceback.format_exc())
    finally: cv2.destroyAllWindows();
    if plt.get_fignums(): plt.close('all');
    rospy.loginfo("최종 프로그램 종료.")