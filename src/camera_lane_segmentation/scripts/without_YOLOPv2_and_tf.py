#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import argparse
import time
import cv2
import numpy as np
import cv2.ximgproc as ximgproc # 세선화를 위해 추가
import threading
import queue
from math import atan2, degrees
from pathlib import Path

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

# ===================================================
# 확장 칼만 필터 클래스 (이전과 동일)
# ===================================================
class LaneExtendedKalmanFilter:
    def __init__(self, dt=0.033):
        self.dt = dt
        self.dim_x = 6
        self.dim_z = 3
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.Q = np.eye(self.dim_x, dtype=np.float32) * 0.01
        self.R = np.eye(self.dim_z, dtype=np.float32) * 100.0
        self.initialized = False

    def reset(self):
        self.x[:] = 0
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.initialized = False

    def predict(self):
        dt = self.dt
        a, b, c, da, db, dc = self.x.flatten()
        a_pred, b_pred, c_pred = a + da * dt, b + db * dt, c + dc * dt
        da_pred, db_pred, dc_pred = da, db, dc
        self.x = np.array([[a_pred], [b_pred], [c_pred], [da_pred], [db_pred], [dc_pred]], dtype=np.float32)
        F = np.eye(self.dim_x, dtype=np.float32)
        F[0, 3], F[1, 4], F[2, 5] = dt, dt, dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape(self.dim_z, 1)
        if not self.initialized:
            self.x[0:3], self.x[3:6] = z, 0.0
            self.initialized = True
            return
        H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        H[0, 0], H[1, 1], H[2, 2] = 1, 1, 1
        h = H @ self.x
        y = z - h
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

# ===================================================
# 전처리 및 유틸리티 함수 (이전과 동일)
# ===================================================
def get_white_mask(bev_image, l_thresh=240):
    hls = cv2.cvtColor(bev_image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    _, white_mask = cv2.threshold(l_channel, l_thresh, 255, cv2.THRESH_BINARY)
    return white_mask

def final_filter(bev_mask):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    f2 = cv2.morphologyEx(bev_mask, cv2.MORPH_CLOSE, kernel_close)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(f2, connectivity=8)
    
    if num_labels <= 1: return np.zeros_like(f2)

    cleaned = np.zeros_like(f2)
    min_area = 500
    
    # 상위 2개 컴포넌트만 남기는 로직으로 통합
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    comps.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(min(len(comps), 2)):
        idx = comps[i][0]
        cleaned[labels == idx] = 255
        
    return cleaned

def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: return None
    try:
        return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError):
        return None

def compute_polyline_points(coeff, image_shape, step=5):
    h, w = image_shape[:2]
    points = []
    if coeff is None: return points
    for y in range(h - 1, 0, -step):
        x = np.polyval(coeff, y)
        if 0 <= x < w:
            points.append((int(x), int(y)))
    return points

def overlay_polyline(image, coeff, color=(0, 0, 255), thickness=2, step=5, translation=(0,0)):
    if coeff is None: return image
    points = []
    h, w = image.shape[:2]
    y_coords = range(h, 0, -step)
    x_coords = np.polyval(coeff, y_coords)
    for x, y in zip(x_coords, y_coords):
        if 0 <= x < w:
            points.append((int(x + translation[0]), int(y + translation[1])))
    if len(points) > 1:
        cv2.polylines(image, [np.array(points)], isClosed=False, color=color, thickness=thickness)
    return image

def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points, dst_points = params['src_points'], params['dst_points']
    warp_w, warp_h = int(params['warp_w']), int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='2',
                        # default='/home/highsky/Videos/Webcam/right_bev_params_1.mp4',
                        help='카메라 장치 번호 또는 영상 파일 경로')
    parser.add_argument('--img-size', type=int, default=640, help='이미지 처리 해상도')
    parser.add_argument('--param-file', type=str, default='./bev_params_3.npz', help='BEV 파라미터 파일 경로')
    parser.add_argument('--lookahead', type=float, default=600.0, help='Pure Pursuit 전방주시거리 (픽셀 단위)')
    parser.add_argument('--wheelbase', type=float, default=187.0, help='차량 축거 (픽셀 단위)')
    parser.add_argument('--nosave', action='store_true', help='결과 영상 저장 안 함')
    parser.add_argument('--project', default='runs/detect_hybrid_logic', help='결과 저장 폴더')
    parser.add_argument('--name', default='exp', help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_true', help='기존 폴더 덮어쓰기 허용')
    return parser

# ===================================================
# 메인 처리 함수
# ===================================================
def detect_and_publish(opt, pub_steering, pub_lane_status, pub_mask_vis):
    bridge = CvBridge()
    kf = LaneExtendedKalmanFilter(dt=1/30.0)
    dataset = LoadCamera(opt.source, img_size=opt.img_size, stride=32) if opt.source.isdigit() else LoadImages(opt.source, img_size=opt.img_size, stride=32)

    def process_frame(im0s):
        bev_im = do_bev_transform(im0s, opt.param_file)
        bev_im_for_vis = bev_im.copy()
        
        white_lane_mask = get_white_mask(bev_im)
        bevfilter_mask = final_filter(white_lane_mask)
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_GUOHALL)
        if final_mask is None or np.sum(final_mask) == 0:
            final_mask = bevfilter_mask

        # === [핵심 수정] EKF 예측을 항상 먼저 수행하고, 그 결과를 바탕으로 측정값 융합 ===
        
        # [1] EKF 예측: 현재 상태를 바탕으로 다음 스텝의 경로를 예측.
        #     차선 검출에 실패하더라도 이 예측값으로 주행을 이어갈 수 있음.
        kf.predict()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        lane_components = []
        if num_labels > 1:
            sorted_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            for i in range(min(len(sorted_indices), 2)):
                label_idx = sorted_indices[i]
                if stats[label_idx, cv2.CC_STAT_AREA] >= 50:
                    component_mask = (labels == label_idx).astype(np.uint8) * 255
                    ys, xs = np.where(component_mask > 0)
                    if len(ys) > 5:
                        comp_coeff = polyfit_lane(ys, xs, order=2)
                        if comp_coeff is not None:
                            lane_components.append({'coeff': comp_coeff})
        
        # [2] 측정값 생성: 검출된 차선 수에 따라 EKF에 제공할 측정값(coeff)을 생성.
        coeff = None
        if len(lane_components) == 2:
            # 두 차선 검출 (가장 이상적): 두 차선의 평균을 측정값으로 사용.
            rospy.loginfo_throttle(1, "Status: 2 Lanes Detected. Using average path.")
            coeff = (lane_components[0]['coeff'] + lane_components[1]['coeff']) / 2.0
        elif len(lane_components) == 1:
            # 단일 차선 검출: '예측된 위치'와 '측정된 형태'를 융합한 하이브리드 측정값 생성.
            rospy.loginfo_throttle(1, "Status: 1 Lane Detected. Fusing predicted position with measured shape.")
            measured_coeff = lane_components[0]['coeff']
            predicted_coeff = kf.x[0:3].flatten() # EKF가 예측한 경로 계수 [a,b,c]

            # a(곡률), b(기울기)는 측정값을, c(위치)는 예측값을 사용.
            hybrid_coeff = np.array([
                measured_coeff[0],    # 형태(Shape)는 실시간 측정값을 반영
                measured_coeff[1],    # 형태(Shape)는 실시간 측정값을 반영
                predicted_coeff[2]    # 위치(Position)는 예측값을 유지하여 안정성 확보
            ])
            coeff = hybrid_coeff
        else:
            # 차선 미검출: 측정값 없음. EKF는 예측 단계의 결과만으로 주행.
            rospy.loginfo_throttle(1, "Status: 0 Lanes Detected. Following predicted path.")

        # [3] EKF 업데이트: 생성된 측정값(coeff)의 유효성을 검사하고 EKF를 업데이트.
        is_measurement_good = coeff is not None
        if is_measurement_good:
            curvature_threshold = 0.001
            if abs(coeff[0]) > curvature_threshold:
                is_measurement_good = False
        
        pub_lane_status.publish(Bool(data=is_measurement_good))

        if is_measurement_good:
            # 유효한 측정값이 있을 때만 EKF 업데이트 수행
            kf.update(coeff)
        
        # 최종 평활화된 경로는 항상 EKF의 현재 상태.
        smoothed_coeff = kf.x[0:3].flatten() if kf.initialized else None
        
        # ==============================================================================

        # Pure Pursuit 및 시각화 로직 (이하 동일)
        if smoothed_coeff is not None:
            poly_points = compute_polyline_points(smoothed_coeff, bev_im.shape, step=5)
            translation = (0, 0)
            if len(poly_points) > 0:
                bottom_point = poly_points[0]
                desired_start = (bev_im.shape[1] // 2, bev_im.shape[0])
                translation = (desired_start[0] - bottom_point[0], desired_start[1] - bottom_point[1])

            bev_im_for_vis = overlay_polyline(bev_im_for_vis, smoothed_coeff, color=(0, 255, 255), thickness=3, translation=translation)
            shifted_poly_points = [(pt[0] + translation[0], pt[1] + translation[1]) for pt in poly_points]

            def image_to_vehicle(pt, bev_shape):
                x_img, y_img = pt
                X_vehicle = bev_shape[0] - y_img
                Y_vehicle = x_img - (bev_shape[1] // 2)
                return X_vehicle, Y_vehicle

            lookahead, wheelbase = opt.lookahead, opt.wheelbase
            goal_point_vehicle = None
            
            for pt_img in shifted_poly_points:
                X_v, Y_v = image_to_vehicle(pt_img, bev_im.shape)
                d = np.sqrt(X_v**2 + Y_v**2)
                if d >= lookahead:
                    goal_point_vehicle = (X_v, Y_v)
                    break
            
            if goal_point_vehicle is None and len(shifted_poly_points) > 0:
                goal_point_vehicle = image_to_vehicle(shifted_poly_points[-1], bev_im.shape)

            if goal_point_vehicle is not None:
                X_v, Y_v = goal_point_vehicle
                d = np.sqrt(X_v**2 + Y_v**2)
                steering_angle_deg = 0.0
                if d > 1e-6:
                    alpha = np.arctan2(Y_v, X_v)
                    steering_angle_rad = np.arctan((2 * wheelbase * np.sin(-alpha)) / d)
                    steering_angle_deg = np.degrees(steering_angle_rad)
                
                pub_steering.publish(Float32(data=steering_angle_deg))
                
                goal_x_img = int((bev_im.shape[1] // 2) + goal_point_vehicle[1])
                goal_y_img = int(bev_im.shape[0] - goal_point_vehicle[0])
                cv2.circle(bev_im_for_vis, (goal_x_img, goal_y_img), 8, (0, 255, 0), -1)
                cv2.putText(bev_im_for_vis, f"Steering: {steering_angle_deg:.2f} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        try:
            mask_msg = bridge.cv2_to_imgmsg(final_mask, "mono8")
            pub_mask_vis.publish(mask_msg)
        except Exception as e:
            rospy.logwarn(f"Could not publish mask image: {e}")

        cv2.imshow("BEV (Lane Following)", bev_im_for_vis)
        cv2.imshow("Final Thin Mask", final_mask)
        cv2.imshow("Original White Mask", white_lane_mask)

    for frame_data in dataset:
        if rospy.is_shutdown(): break
        _, _, im0s, _ = frame_data
        process_frame(im0s)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    if hasattr(dataset, 'release'):
        dataset.release()
    cv2.destroyAllWindows()
    rospy.loginfo("[INFO] Processing finished.")


def ros_main():
    rospy.init_node('merged_lane_follower_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()

    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
    pub_mask_vis = rospy.Publisher('lane_mask_visualization', Image, queue_size=1)
    
    rospy.loginfo("Hinton's Merged Lane Follower Node (Hybrid Fusion Logic) has started.")
    rospy.loginfo(f"Source: {opt.source}, BEV Params: {opt.param_file}")

    try:
        detect_and_publish(opt, pub_steering, pub_lane_status, pub_mask_vis)
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        rospy.loginfo("Shutting down node.")
        cv2.destroyAllWindows()


if __name__=='__main__':
    try:
        ros_main()
    except rospy.ROSInterruptException:
        pass