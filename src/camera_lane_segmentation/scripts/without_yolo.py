#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import argparse
import time
import cv2
import numpy as np
import cv2.ximgproc as ximgproc
import threading
import queue
from math import atan2, degrees, pi
from pathlib import Path
import matplotlib.pyplot as plt
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Bool  # Bool 추가
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# 기존 코드의 일부 유틸 함수(imported functions) – 필요에 따라 사용합니다.
from utils.utils import (
    time_synchronized, select_device, increment_path, lane_line_mask,
    AverageMeter, LoadCamera, LoadImages, letterbox,
)

##############################################
# ROS 퍼블리셔 정의
##############################################
pub_lane_marker = rospy.Publisher('lane_data_marker', MarkerArray, queue_size=1)
pub_path_marker = rospy.Publisher('lane_path_marker', MarkerArray, queue_size=1)
pub_goal_marker = rospy.Publisher('goal_point_marker', Marker, queue_size=1)
pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
pub_mask = rospy.Publisher('camera_lane_segmentation/lane_mask', Image, queue_size=1)
pub_binary = rospy.Publisher('camera_lane_segmentation/binary_mask', Image, queue_size=1)
pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
pub_slope = rospy.Publisher("path_slope", Float32, queue_size=1)

##############################################
# argparse 설정
##############################################
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        #default='2',
                        default='/home/yoo/source/test_video1.mp4',
                        help='source: 0(webcam) 또는 파일 경로')
    parser.add_argument('--img-size', type=int, default=640, help='추론 해상도')
    parser.add_argument('--lane-thres', type=float, default=0.5, help='차선 세그 임계값')
    parser.add_argument('--project', default='runs/detect', help='결과 저장 폴더')
    parser.add_argument('--name', default='exp', help='결과 저장 폴더 이름')
    parser.add_argument('--frame-skip', type=int, default=0, help='프레임 건너뛰기')
    parser.add_argument('--param-file', type=str, default='./bev_params.npz', help='BEV 파라미터')
    parser.add_argument('--debug', action='store_true', help='차량 좌표계 시각화')
    return parser

##############################################
# 좌표 변환 함수들
##############################################
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

##############################################
# 차선 함수 추출 및 경로 관련 함수들
##############################################
def extract_lane_functions(binary_image, poly_degree=3):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    lane_data = []
    for label in range(1, num_labels):
        rows, cols = np.where(labels == label)
        vehicle_coords = np.array([image_to_vehicle((col, row)) for row, col in zip(rows, cols)])
        x_vehicle = vehicle_coords[:, 0]
        y_vehicle = vehicle_coords[:, 1]
        if len(x_vehicle) < poly_degree + 1:
            continue
        poly_coeff = np.polyfit(x_vehicle, y_vehicle, poly_degree)
        lane_data.append(poly_coeff)
    return lane_data

def compute_derivative(poly_coeff, x_value):
    derivative = 0
    n = len(poly_coeff) - 1
    for i, coeff in enumerate(poly_coeff[:-1]):
        derivative += coeff * (n - i) * (x_value ** (n - i - 1))
    return derivative

def find_positive_x_intercept(poly_coeff):
    roots = np.roots(poly_coeff)                    # 모든 근 구하기
    real_roots = roots[np.isreal(roots)].real       # 실근만 추출
    positive_roots = real_roots[real_roots > 0]       # 양의 실근만 추출
    return positive_roots[0] if len(positive_roots) > 0 else 2.0

def shift_poly(poly_coeff, dx, dy):
    """
    다항식 f(x)=poly_coeff (numpy.poly1d 형식)을 f_new(x)= f(x-dx)+dy로 평행이동하여
    새로운 계수를 반환.
    """
    p = np.poly1d(poly_coeff)
    shifted_poly = p(np.poly1d([1, -dx])) + dy
    return shifted_poly.coeffs

def create_path_function(lane_coeffs, d=0.75 , lookahead=2.1):
    """
    유효한 차선 함수만 선택한 후,
      - 두 개의 차선이면 단순 평균하여 중앙 경로를 생성하고,
      - 단일 차선인 경우 lookahead(2.1m) 지점에서의 기울기를 이용해 법선 방향으로 평행이동하여 경로 함수를 생성한다.
    """
    valid_lane_coeffs = []
    for poly_coeff in lane_coeffs:
        x_intercept = find_positive_x_intercept(poly_coeff)
        slope = compute_derivative(poly_coeff, x_intercept)
        if -10.0 <= slope <= 10.0:
            valid_lane_coeffs.append(poly_coeff)
        else:
            rospy.logwarn("[WARNING] Discarding lane function with invalid slope: %.2f", slope)
    
    if not valid_lane_coeffs:
        return None

    if len(valid_lane_coeffs) == 2:
        path_coeff = np.mean(np.array(valid_lane_coeffs), axis=0)
    elif len(valid_lane_coeffs) == 1:
        poly_coeff = valid_lane_coeffs[0]
        x_intercept = find_positive_x_intercept(poly_coeff)
        slope = compute_derivative(poly_coeff, x_intercept)
        m = compute_derivative(poly_coeff, lookahead)
        a = np.arctan(-1/m) if m != 0 else -pi/2
        if slope > 0:  # 우측 차선
            dx = - d * abs(np.cos(a))
            dy = d * abs(np.sin(a))
        else:          # 좌측 차선
            dx = - d * abs(np.cos(a))
            dy = - d * abs(np.sin(a))
        path_coeff = shift_poly(poly_coeff, dx, dy)
        pub_slope.publish(Float32(data=slope))
    else:
        return None

    return path_coeff

# 차선 경로 점 샘플링
def sample_path_points(poly_coeff, x_start=1.4, x_end=2.9, num_points=50):
    xs = np.linspace(x_start, x_end, num_points)
    ys = np.polyval(poly_coeff, xs)
    return list(zip(xs, ys))

def overlay_polyline(bev_image, path_points, color=(0, 0, 255), thickness=2):
    image_points = [vehicle_to_image(pt) for pt in path_points]
    pts = np.array(image_points, dtype=np.int32).reshape((-1, 1, 2))
    overlayed_image = bev_image.copy()
    cv2.polylines(overlayed_image, [pts], isClosed=False, color=color, thickness=thickness)
    return overlayed_image

##############################################
# BEV 변환 및 필터링 함수
##############################################
def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=200):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=50):
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

##############################################
# RViz 마커 생성 함수들
##############################################
def create_lane_marker(lane_coeffs, frame_id="velodyne", x_start=1.4, x_end=2.9, num_points=50):
    marker_array = MarkerArray()
    valid_lane_coeffs = []
    for coeff in lane_coeffs:
        x_int = find_positive_x_intercept(coeff)
        slope = compute_derivative(coeff, x_int)
        if -10.0 <= slope <= 10.0:
            valid_lane_coeffs.append(coeff)
        else:
            rospy.logwarn("[WARNING] Discarding lane function with invalid slope for RViz: %.2f", slope)

    for i, coeff in enumerate(valid_lane_coeffs):
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
        marker.scale.z = 0.5
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

##############################################
# 디버깅 시각화 함수
##############################################
def debug_plot_lane(path_points, lane_data=None, goal_point=None):
    plt.figure("Lane in Vehicle Coordinates", figsize=(6, 6))
    plt.clf()
    
    if len(path_points) > 0:
        path_vehicle = np.array(path_points)
        plt.plot(path_vehicle[:, 1], path_vehicle[:, 0], 'r-', label="Path Function")
    
    valid_lane_data = []
    if lane_data is not None:
        for coeff in lane_data:
            x_int = find_positive_x_intercept(coeff)
            slope = compute_derivative(coeff, x_int)
            if -10.0 <= slope <= 10.0:
                valid_lane_data.append(coeff)
            else:
                rospy.logwarn("[DEBUG] Discarding lane function for plotting with invalid slope: %.2f", slope)
        
        for i, coeff in enumerate(valid_lane_data):
            xs = np.linspace(1.4, 2.9, 50)
            ys = np.polyval(coeff, xs)
            plt.plot(ys, xs, 'b--', label=f"Lane Function {i+1}" if i == 0 else None)
            
            x_int = find_positive_x_intercept(coeff)
            slope = compute_derivative(coeff, x_int)
            plt.scatter(0, x_int, marker='o', color='magenta', s=100,
                        label="X-intercept" if i == 0 else None)
            plt.text(0, x_int, f"{x_int:.2f}\n(slope: {slope:.2f})", color='magenta', fontsize=10,
                     verticalalignment='bottom', horizontalalignment='right')
    
    if goal_point:
        plt.scatter(goal_point[1], goal_point[0], color='green', s=100, label="Goal Point")

    plt.xlabel("Lateral (m)")
    plt.ylabel("Forward (m)")
    plt.title("Lane & Path Functions in Vehicle Coordinates")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.xlim(1.0, -1.0)
    plt.ylim(0.0, 3.0)
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.001)

##############################################
# 메인 처리 함수
##############################################
def detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    bridge = CvBridge()
    source = opt.source
    imgsz = opt.img_size
    lane_threshold = opt.lane_thres
    bev_param_file = opt.param_file

    dataset = LoadCamera(source, img_size=imgsz, stride=32) if source.isdigit() else LoadImages(source, img_size=imgsz, stride=32)

    def process_frame(im0s):
        bev_im = do_bev_transform(im0s, bev_param_file)

        # (1) BEV 이미지를 HSV 색공간으로 변환
        hsv = cv2.cvtColor(bev_im, cv2.COLOR_BGR2HSV)

        # (2) 흰색 범위를 지정하여 마스크 생성
        # 차선이 완전 흰색에 가깝다면, 아래 범위를 좀 더 타이트하게 조정할 수 있습니다.
        # 필요에 따라 lower_white, upper_white 값을 조절하세요.
        lower_white = np.array([0, 0, 200], dtype=np.uint8)  # 예시: 밝은 회색~흰색
        upper_white = np.array([179, 40, 255], dtype=np.uint8)
        binary_mask = cv2.inRange(hsv, lower_white, upper_white)

        bevfilter_mask = final_filter(binary_mask)
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if final_mask is None or final_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → bevfilter_mask 사용")
            final_mask = bevfilter_mask

        lane_data = extract_lane_functions(final_mask, poly_degree=3)
        measured_path_coeff = create_path_function(lane_data, d=0.75, lookahead=2.1)
        
        filtered_path_coeff = measured_path_coeff if measured_path_coeff is not None else None
        path_points = sample_path_points(filtered_path_coeff) if filtered_path_coeff is not None else []

        lane_detected = (filtered_path_coeff is not None) and (len(path_points) > 0)
        pub_lane_status.publish(Bool(data=lane_detected))

        bev_im_color = bev_im.copy()
        
        if lane_detected:
            bev_im_color = overlay_polyline(bev_im_color, path_points)
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
            if goal_point is None and path_points:
                goal_point = path_points[-1]
            if goal_point is not None:
                X_v, Y_v = goal_point
                d = np.sqrt(X_v**2 + Y_v**2)
                alpha = np.arctan2(Y_v, X_v)
                steering_angle = np.arctan((2 * wheelbase_m * np.sin(alpha)) / d) if d > 1e-6 else 0.0
                steering_angle_deg = np.degrees(steering_angle)
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
                    debug_plot_lane(path_points, lane_data, goal_point)
        else:
            bev_im_color = bev_im.copy()

        pub_mask.publish(bridge.cv2_to_imgmsg(bev_im_color, "bgr8"))
        pub_binary.publish(bridge.cv2_to_imgmsg(final_mask, "mono8"))

        if 'DISPLAY' in os.environ:
            cv2.imshow("binary_img", bevfilter_mask)
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
            rospy.loginfo(f"Processing frame {frame_idx}")
            process_frame(im0s)
            if 'DISPLAY' in os.environ and cv2.waitKey(60) & 0xFF == ord('q'):
                break

    rospy.loginfo("Video processing completed.")
    cv2.destroyAllWindows()

##############################################
# ros_main 및 엔트리 포인트
##############################################
def ros_main():
    rospy.init_node('bev_lane_thinning_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    if opt.debug:
        plt.ion()
    detect_and_publish(opt, pub_mask, pub_steering, pub_lane_status)
    rospy.loginfo("[INFO] bev_lane_thinning_node 종료, spin() 호출")
    rospy.spin()

if __name__ == '__main__':
    try:
        ros_main()
    except rospy.ROSInterruptException:
        pass
