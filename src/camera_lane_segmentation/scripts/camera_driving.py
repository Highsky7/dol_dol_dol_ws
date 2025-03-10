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
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Bool  # Bool 추가
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from utils.utils import (
    time_synchronized, select_device, increment_path, lane_line_mask,
    AverageMeter, LoadCamera, LoadImages, letterbox,
)

# ROS 퍼블리셔 정의 (lane_detection_status 추가)
pub_lane_marker = rospy.Publisher('lane_data_marker', MarkerArray, queue_size=1)
pub_path_marker = rospy.Publisher('lane_path_marker', MarkerArray, queue_size=1)
pub_goal_marker = rospy.Publisher('goal_point_marker', Marker, queue_size=1)
pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
pub_mask = rospy.Publisher('camera_lane_segmentation/lane_mask', Image, queue_size=1)
pub_binary = rospy.Publisher('camera_lane_segmentation/binary_mask', Image, queue_size=1)
pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)  # 추가

# argparse 설정
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/highsky/yolopv2.pt', help='model.pt 경로')
    parser.add_argument('--source', type=str, default='0', help='source: 0(webcam) 또는 파일 경로')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO 추론 해상도')
    parser.add_argument('--device', default='0', help='cuda device: 0 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.5, help='차선 세그 임계값')
    parser.add_argument('--project', default='runs/detect', help='결과 저장 폴더')
    parser.add_argument('--name', default='exp', help='결과 저장 폴더 이름')
    parser.add_argument('--frame-skip', type=int, default=0, help='프레임 건너뛰기')
    parser.add_argument('--param-file', type=str, default='/home/highsky/dol_dol_dol_ws/bev_params.npz', help='BEV 파라미터')
    parser.add_argument('--debug', action='store_true', help='차량 좌표계 시각화')
    return parser

# 좌표 변환 함수
def image_to_vehicle(pt):
    u, v = pt
    x_vehicle = (640 - v) * 0.00234375 + 1.4
    y_vehicle = (320 - u) * 0.003125
    return x_vehicle, y_vehicle

def vehicle_to_image(point):
    x_vehicle, y_vehicle = point
    v = 640 - (x_vehicle - 1.4) / 0.00234375
    u = 320 - y_vehicle / 0.003125
    return int(round(u)), int(round(v))

# 차선 함수 추출
def extract_lane_functions(binary_image, poly_degree=3):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    lane_data = []
    for label in range(1, num_labels):
        rows, cols = np.where(labels == label)
        vehicle_coords = np.array([image_to_vehicle((col, row)) for row, col in zip(rows, cols)])
        x_vehicle = vehicle_coords[:, 0]
        y_vehicle = vehicle_coords[:, 1]
        poly_coeff = np.polyfit(x_vehicle, y_vehicle, poly_degree)
        lane_data.append(poly_coeff)
    return lane_data

# 다항식 도함수 계산
def compute_derivative(poly_coeff, x_value):
    derivative = 0
    n = len(poly_coeff) - 1
    for i, coeff in enumerate(poly_coeff[:-1]):
        derivative += coeff * (n - i) * (x_value ** (n - i - 1))
    return derivative

# 양의 x절편 계산
def find_positive_x_intercept(poly_coeff):
    roots = np.roots(poly_coeff)
    real_roots = roots[np.isreal(roots)].real
    positive_roots = real_roots[real_roots > 0]
    return positive_roots[0] if len(positive_roots) > 0 else 2.0

# 경로 함수 생성
def create_path_function(lane_coeffs, offset_right=0.75, offset_left=-0.75):
    if not lane_coeffs:
        return None
    if len(lane_coeffs) == 2:
        path_coeff = np.mean(np.array(lane_coeffs), axis=0)
    elif len(lane_coeffs) == 1:
        poly_coeff = lane_coeffs[0]
        x_intercept = find_positive_x_intercept(poly_coeff)
        slope = compute_derivative(poly_coeff, x_intercept)
        path_coeff = poly_coeff.copy()
        path_coeff[-1] += offset_right if slope > 0 else offset_left
    else:
        return None
    return path_coeff

# 경로 점 샘플링
def sample_path_points(poly_coeff, x_start=1.4, x_end=2.9, num_points=50):
    xs = np.linspace(x_start, x_end, num_points)
    ys = np.polyval(poly_coeff, xs)
    return list(zip(xs, ys))

# BEV 이미지에 경로 오버레이
def overlay_polyline(bev_image, path_points, color=(0, 0, 255), thickness=2):
    image_points = [vehicle_to_image(pt) for pt in path_points]
    pts = np.array(image_points, dtype=np.int32).reshape((-1, 1, 2))
    overlayed_image = bev_image.copy()
    cv2.polylines(overlayed_image, [pts], isClosed=False, color=color, thickness=thickness)
    return overlayed_image

# BEV 변환
def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

# 필터링
def final_filter(bev_mask):
    f1 = cv2.morphologyEx(bev_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(f1, connectivity=8)
    cleaned = np.zeros_like(f1)
    if num_labels > 2:
        comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        comps.sort(key=lambda x: x[1], reverse=True)
        for idx in [i for i, area in comps[:2] if area >= 300]:
            cleaned[labels == idx] = 255
    else:
        cleaned = f1
    return cleaned

# RViz 마커 생성
def create_lane_marker(lane_coeffs, frame_id="velodyne", x_start=1.4, x_end=2.9, num_points=50):
    marker_array = MarkerArray()
    for i, coeff in enumerate(lane_coeffs):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "lane_data"
        marker.id = i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        xs = np.linspace(x_start, x_end, num_points)
        ys = np.polyval(coeff, xs)
        marker.points = [Point(x=x, y=y, z=0) for x, y in zip(xs, ys)]
        marker_array.markers.append(marker)
    return marker_array

def create_path_marker(path_points, frame_id="velodyne"):
    marker_array = MarkerArray()
    for i, (x, y) in enumerate(path_points):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "lane_path"
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker_array.markers.append(marker)
    return marker_array

def create_goal_marker(goal_point, frame_id="velodyne"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "goal_point"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.pose.position.x = goal_point[0]
    marker.pose.position.y = goal_point[1]
    marker.pose.position.z = 0
    return marker

# 디버깅 시각화
def debug_plot_lane(path_points, goal_point=None):
    if len(path_points) > 0:
        lane_vehicle = np.array(path_points)
        plt.figure("Lane in Vehicle Coordinates", figsize=(6, 6))
        plt.clf()
        plt.plot(lane_vehicle[:, 1], lane_vehicle[:, 0], 'r-', label="Lane")
        if goal_point:
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

# 메인 처리 함수 (lane_detection_status 퍼블리시 추가)
def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):  # 인자 추가
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
    cudnn.benchmark = True

    bridge = CvBridge()
    source, weights = opt.source, opt.weights
    imgsz, lane_threshold, bev_param_file = opt.img_size, opt.lane_thres, opt.param_file

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # 입력 소스
    dataset = LoadCamera(source, img_size=imgsz, stride=32) if source.isdigit() else LoadImages(source, img_size=imgsz, stride=32)

    def process_frame(im0s):
        net_input_img, _, _ = letterbox(im0s, (imgsz, imgsz), stride=32)
        net_input_img = net_input_img[:, :, ::-1].transpose(2, 0, 1)
        net_input_img = np.ascontiguousarray(net_input_img)
        img_t = torch.from_numpy(net_input_img).to(device).float() / 255.0
        if half:
            img_t = img_t.half()
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        with torch.no_grad():
            [_, _], _, ll = model(img_t)

        binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='otsu')
        thin_mask = ximgproc.thinning(binary_mask)
        if thin_mask is None or thin_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → binary_mask 사용")
            thin_mask = binary_mask
        bev_mask = do_bev_transform(thin_mask, bev_param_file)
        bevfilter_mask = final_filter(bev_mask)
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if final_mask is None or final_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → bevfilter_mask 사용")
            final_mask = bevfilter_mask

        lane_data = extract_lane_functions(final_mask, poly_degree=3)
        path_coeff = create_path_function(lane_data, offset_right=0.75, offset_left=-0.75)
        path_points = sample_path_points(path_coeff) if path_coeff is not None else []

        # 차선 검출 여부 판단 및 퍼블리시
        lane_detected = path_coeff is not None and len(path_points) > 0
        pub_lane_status.publish(Bool(data=lane_detected))

        bev_im = do_bev_transform(im0s, bev_param_file)
        bev_im_color = overlay_polyline(bev_im, path_points)

        if path_coeff is not None:
            lookahead_m, wheelbase_m = 2.1, 0.75
            goal_point = None
            min_error = float('inf')
            for pt in path_points:
                X_v, Y_v = pt
                d = np.sqrt(X_v**2 + Y_v**2)
                error = abs(d - lookahead_m)
                if error < min_error:
                    min_error = error
                    goal_point = (X_v, Y_v)
            if not goal_point and path_points:
                goal_point = path_points[-1]

            if goal_point:
                X_v, Y_v = goal_point
                d = np.sqrt(X_v**2 + Y_v**2)
                alpha = np.arctan2(Y_v, X_v)
                steering_angle = np.arctan((2 * wheelbase_m * np.sin(alpha)) / d) if d > 1e-6 else 0.0
                steering_angle_deg = -np.degrees(steering_angle)
                pub_steering.publish(Float32(data=steering_angle_deg))
                rospy.loginfo("[INFO] Published auto_steer_angle_lane: %.2f deg", steering_angle_deg)
                goal_x_img, goal_y_img = vehicle_to_image(goal_point)
                cv2.circle(bev_im_color, (goal_x_img, goal_y_img), 5, (0, 255, 0), -1)
                cv2.putText(bev_im_color, f"Steering: {steering_angle_deg:.2f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                lane_marker = create_lane_marker(lane_data)
                path_marker = create_path_marker(path_points)
                goal_marker = create_goal_marker(goal_point)
                pub_lane_marker.publish(lane_marker)
                pub_path_marker.publish(path_marker)
                pub_goal_marker.publish(goal_marker)

            if opt.debug:
                debug_plot_lane(path_points, goal_point)
        else:
            delete_marker = Marker()
            delete_marker.header.frame_id = "velodyne"
            delete_marker.header.stamp = rospy.Time.now()
            delete_marker.action = Marker.DELETEALL
            pub_lane_marker.publish(MarkerArray(markers=[delete_marker]))
            pub_goal_marker.publish(delete_marker)
            pub_path_marker.publish(MarkerArray())

        # ROS 퍼블리시
        pub_mask.publish(bridge.cv2_to_imgmsg(bev_im_color, "bgr8"))
        pub_binary.publish(bridge.cv2_to_imgmsg(final_mask, "mono8"))

        if 'DISPLAY' in os.environ:
            cv2.imshow("BEV + Polyfit", bev_im_color)
            cv2.imshow("Final Mask", final_mask)
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
            _, _, im0s, _ = frame_data

            if frame_skip > 0 and frame_counter % (frame_skip + 1) != 0:
                frame_counter += 1
                continue
            frame_counter += 1

            process_frame(im0s)
            if 'DISPLAY' in os.environ and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        for frame_idx, _, im0s, _ in dataset:
            rospy.loginfo(f"Processing frame {frame_idx}/401")
            process_frame(im0s)
            if 'DISPLAY' in os.environ and cv2.waitKey(30) & 0xFF == ord('q'):
                break

    rospy.loginfo("Video processing completed.")
    cv2.destroyAllWindows()

def ros_main():
    rospy.init_node('bev_lane_thinning_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    if opt.debug:
        plt.ion()
    detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)  # 인자 추가

if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass