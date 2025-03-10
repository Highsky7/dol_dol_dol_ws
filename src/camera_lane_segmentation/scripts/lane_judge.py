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
    parser.add_argument('--weights', type=str, default='/home/highsky/yolopv2.pt', help='model.pt 경로')
    parser.add_argument('--source', type=str, default='2', help='source: 0(webcam) 또는 영상/이미지 파일 경로')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO 추론 해상도')
    parser.add_argument('--device', default='0', help='cuda device: 0 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.5, help='차선 세그 임계값 (0.0~1.0)')
    parser.add_argument('--nosave', action='store_false', help='저장하지 않으려면 사용')
    parser.add_argument('--project', default='runs/detect', help='결과 저장 폴더')
    parser.add_argument('--name', default='exp', help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_false', help='기존 폴더 사용 허용')
    parser.add_argument('--frame-skip', type=int, default=0, help='프레임 건너뛰기 (0이면 건너뛰지 않음)')
    parser.add_argument('--param-file', type=str, default='/home/highsky/dol_dol_dol_ws/bev_params.npz', help='BEV 파라미터')
    parser.add_argument('--debug', action='store_true', help='Matplotlib을 사용하여 차선 시각화')
    return parser

# Extended Kalman Filter 클래스
class LaneExtendedKalmanFilter:
    def __init__(self, dt=0.033):
        self.dt = dt
        self.dim_x = 6  # [a, b, c, da, db, dc]
        self.dim_z = 3  # measurement: [a, b, c]
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.Q = np.eye(self.dim_x, dtype=np.float32) * 0.1
        self.R = np.eye(self.dim_z, dtype=np.float32) * 5.0
        self.initialized = False

    def reset(self):
        self.x[:] = 0
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.initialized = False

    def predict(self):
        dt = self.dt
        a, b, c, da, db, dc = self.x.flatten()
        a_pred = a + da*dt + 0.5 * np.sin(a) * (dt**2)
        b_pred = b + db*dt
        c_pred = c + dc*dt
        da_pred = da + np.sin(a) * dt
        db_pred = db
        dc_pred = dc
        self.x = np.array([[a_pred], [b_pred], [c_pred], [da_pred], [db_pred], [dc_pred]], dtype=np.float32)
        F = np.eye(self.dim_x, dtype=np.float32)
        F[0,0] = 1 + 0.5 * np.cos(a) * (dt**2)
        F[0,3] = dt
        F[3,0] = np.cos(a) * dt
        F[3,3] = 1
        F[1,1] = 1
        F[1,4] = dt
        F[2,2] = 1
        F[2,5] = dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape(self.dim_z, 1)
        if not self.initialized:
            self.x[0:3] = z
            self.initialized = True
            return
        h = self.x[0:3]
        y = z - h
        H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        H[0,0] = H[1,1] = H[2,2] = 1
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

# 폴리피팅 유틸
def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5:
        return None
    return np.polyfit(points_y, points_x, order)

def compute_polyline_points(coeff, image_shape, step=4):
    h, w = image_shape[:2]
    points = []
    for y in range(0, h, step):
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
        if 0 <= x < w:
            draw_points.append((int(x + translation[0]), int(y + translation[1])))
    if len(draw_points) > 1:
        for i in range(len(draw_points) - 1):
            cv2.line(image, draw_points[i], draw_points[i+1], color, 2)
    return image

# 모폴로지 및 필터링 함수
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
    if num_labels <= 2:
        return binary_mask
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    comps.sort(key=lambda x: x[1], reverse=True)
    keep_indices = [i for i, area in comps[:2] if area >= min_area]
    cleaned = np.zeros_like(binary_mask)
    for idx in keep_indices:
        cleaned[labels == idx] = 255
    return cleaned

def final_filter(bev_mask):
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=300)
    f4 = keep_top2_components(f3, min_area=300)
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
def debug_plot_lane(shifted_poly_points, image_to_vehicle, goal_point=None):
    lane_vehicle = [image_to_vehicle(pt) for pt in shifted_poly_points]
    if len(lane_vehicle) > 0:
        lane_vehicle = np.array(lane_vehicle)
        plt.figure("Lane in Vehicle Coordinates", figsize=(6, 6))
        plt.clf()
        plt.plot(lane_vehicle[:, 1], lane_vehicle[:, 0], 'r-', label="Lane")
        if goal_point is not None:
            plt.scatter(goal_point[1], goal_point[0], color='green', s=100, label="Goal Point")
        plt.xlabel("Lateral (m)")
        plt.ylabel("Forward (m)")
        plt.title("Lane Line in Vehicle Coordinates")
        plt.legend()
        plt.gca().invert_xaxis()
        plt.xlim(1.0, -1.0)
        plt.ylim(0.0, 3.0)
        plt.grid(True)
        plt.show(block=False)
        plt.pause(0.001)

# 메인 처리 함수
def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):
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
    if half:
        model.half()
    model.eval()

    kf = LaneExtendedKalmanFilter(dt=0.033)

    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치=%s) 열기", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=32)
    else:
        rospy.loginfo("[INFO] 파일(영상/이미지): %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=32)

    def process_frame(im0s):
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
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='otsu')
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if thin_mask is None or thin_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → binary_mask 사용")
            thin_mask = binary_mask

        bev_mask = do_bev_transform(thin_mask, bev_param_file)
        bevfilter_mask = final_filter(bev_mask)
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if final_mask is None or final_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → bevfilter_mask 사용")
            final_mask = bevfilter_mask

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        if num_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            lane_mask = np.zeros_like(final_mask)
            lane_mask[labels == largest_label] = 255
            ys, xs = np.where(lane_mask > 0)
        else:
            ys, xs = np.where(final_mask > 0)

        coeff = polyfit_lane(ys, xs, order=2)
        lane_detected = coeff is not None and len(ys) > 0
        pub_lane_status.publish(Bool(data=lane_detected))

        bev_im = do_bev_transform(im0s, bev_param_file)
        bev_im_color = bev_im.copy()

        if lane_detected:
            poly_points = compute_polyline_points(coeff, bev_im.shape, step=4)
            if len(poly_points) > 0:
                bottom_point = poly_points[-1]
                desired_start = (320, 640)
                translation = (desired_start[0] - bottom_point[0], desired_start[1] - bottom_point[1])
            else:
                translation = (0, 0)

            bev_im_color = overlay_polyline(bev_im.copy(), coeff, color=(0, 0, 255), step=4, translation=translation)
            shifted_poly_points = [(pt[0] + translation[0], pt[1] + translation[1]) for pt in poly_points]

            def image_to_vehicle(pt):
                u, v = pt
                x_vehicle = (640 - v) * 0.00234375 + 1.4
                y_vehicle = (320 - u) * 0.003125
                return x_vehicle, y_vehicle

            lookahead_m = 2.10  # 210 cm
            wheelbase_m = 0.75  # 75 cm
            goal_point = None
            min_error = float('inf')
            for pt in shifted_poly_points:
                X_v, Y_v = image_to_vehicle(pt)
                d = np.sqrt(X_v**2 + Y_v**2)
                error = abs(d - lookahead_m)
                if error < min_error:
                    min_error = error
                    goal_point = (X_v, Y_v)
            if goal_point is None and len(shifted_poly_points) > 0:
                goal_point = image_to_vehicle(shifted_poly_points[-1])

            if goal_point is not None:
                X_v, Y_v = goal_point
                d = np.sqrt(X_v**2 + Y_v**2)
                if d < 1e-6:
                    steering_angle = 0.0
                else:
                    alpha = np.arctan2(Y_v, X_v)
                    steering_angle = np.arctan((2 * wheelbase_m * np.sin(alpha)) / d)
                steering_angle_deg = -np.degrees(steering_angle)
                pub_steering.publish(Float32(data=steering_angle_deg))
                rospy.loginfo("[INFO] Published auto_steer_angle_lane: %.2f deg", steering_angle_deg)

                goal_x_img = int(320 - goal_point[1] / 0.003125)
                goal_y_img = int(640 - (goal_point[0] - 1.4) / 0.00234375)
                cv2.circle(bev_im_color, (goal_x_img, goal_y_img), 5, (0, 255, 0), -1)
                cv2.putText(bev_im_color, f"Steering: {steering_angle_deg:.2f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            if opt.debug:
                debug_plot_lane(shifted_poly_points, image_to_vehicle, goal_point)
        # else:
        #     rospy.logwarn("[WARNING] Lane not detected, skipping auto_steer_angle_lane publish")

        cv2.imshow("BEV + Polyfit", bev_im_color)
        cv2.imshow("final+mask", final_mask)
        return bev_im, bev_im_color, final_mask

    if dataset.mode == 'stream':
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
        while not rospy.is_shutdown():
            try:
                frame_data = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame_data is None:
                break
            path_item, net_input_img, im0s, vid_cap = frame_data
            if frame_skip > 0 and frame_counter % (frame_skip + 1) != 0:
                frame_counter += 1
                continue
            frame_counter += 1
            process_frame(im0s)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        rospy.loginfo("[DEBUG] 저장된 영상/이미지 동기 처리 시작")
        delay = 30
        if dataset.mode == 'video' and dataset.cap is not None:
            fps = dataset.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                delay = int(1000 / fps)
        start_time = time.time()
        for frame_data in dataset:
            path_item, net_input_img, im0s, vid_cap = frame_data
            process_frame(im0s)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        rospy.loginfo("[INFO] 동기 처리 추론 평균 시간: %.4fs/frame", inf_time.avg)

    cv2.destroyAllWindows()
    rospy.loginfo("[INFO] 추론 완료.")

# 메인 함수
def ros_main():
    rospy.init_node('bev_lane_thinning_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    if opt.debug:
        plt.ion()
    pub_mask = rospy.Publisher('camera_lane_segmentation/lane_mask', Image, queue_size=1)
    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
    detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)
    rospy.spin()

if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass