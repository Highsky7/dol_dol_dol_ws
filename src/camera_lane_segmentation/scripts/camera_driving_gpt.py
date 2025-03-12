#!/home/yoo/doldol/bin/python3
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

# ========== (프로젝트 내 다른 유틸) ==========
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    lane_line_mask,  # YOLOPv2 세그멘테이션 결과 기반 이진화
    AverageMeter,
    LoadCamera,
    LoadImages,
    letterbox,
)

# ===================================================
# argparse 설정 (추가 인자: lookahead, wheelbase, debug)
# ---------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='/home/yoo/yoo_camera_ws/src/YOLOPv2/weights/yolopv2.pt',
                        help='model.pt 경로')
    parser.add_argument('--source', type=str,
                        default='/home/yoo/source/test_video3.mp4',
                        help='source: 0(webcam) 또는 영상/이미지 파일 경로')
    parser.add_argument('--img-size', type=int, default=640,
                        help='YOLO 추론 해상도')
    parser.add_argument('--device', default='0',
                        help='cuda device: 0 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.5,
                        help='차선 세그 임계값 (0.0~1.0)')
    parser.add_argument('--nosave', action='store_false',
                        help='저장하지 않으려면 사용')
    parser.add_argument('--project', default='runs/detect',
                        help='결과 저장 폴더')
    parser.add_argument('--name', default='exp',
                        help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_false',
                        help='기존 폴더 사용 허용')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='프레임 건너뛰기 (0이면 건너뛰지 않음)')
    parser.add_argument('--param-file', type=str,
                        default='/home/yoo/dol_dol_dol_ws/bev_params.npz',
                        help='BEV 파라미터 (src_points, dst_points, warp_w, warp_h)')
    # 디버그 옵션: matplotlib을 이용해 차량 좌표계에서 차선 시각화
    parser.add_argument('--debug', action='store_true',
                        help='Matplotlib을 사용하여 차량 좌표계에서 차선 시각화 (Pure Pursuit 디버깅)')
    return parser

# ===================================================
# Extended Kalman Filter 클래스 (급격한 곡률 변화를 고려)
# ---------------------------------------------------
class LaneExtendedKalmanFilter:
    """
    2차 다항식 계수 [a, b, c]와 그 변화율 [da, db, dc]를 추적하기 위한 확장 칼만 필터.
    상태 벡터: x = [a, b, c, da, db, dc]^T
    관측: z = [a, b, c]
    """
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
        x_pred = np.array([[a_pred],
                           [b_pred],
                           [c_pred],
                           [da_pred],
                           [db_pred],
                           [dc_pred]], dtype=np.float32)
        self.x = x_pred
        F = np.eye(self.dim_x, dtype=np.float32)
        F[0,0] = 1 + 0.5 * np.cos(a) * (dt**2)
        F[0,3] = dt
        F[3,0] = np.cos(a) * dt
        F[3,3] = 1
        F[1,1] = 1; F[1,4] = dt
        F[2,2] = 1; F[2,5] = dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape(self.dim_z, 1)
        if not self.initialized:
            self.x[0,0], self.x[1,0], self.x[2,0] = z.flatten()
            self.x[3,0] = self.x[4,0] = self.x[5,0] = 0.0
            self.initialized = True
            return
        h = self.x[:3]
        y = z - h
        H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        H[0,0], H[1,1], H[2,2] = 1, 1, 1
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

# ===================================================
# 차선함수 유틸
# ---------------------------------------------------
def image_to_vehicle(pt):
    """
    이미지 좌표 (u,v)를 차량 좌표 (x_vehicle, y_vehicle)로 변환.
    차량 전방: +x, 좌측: +y
    변환식:
      x_vehicle = (640 - v) * 0.00234375 + 1.4
      y_vehicle = (320 - u) * 0.003125
    """
    u, v = pt
    x_vehicle = (640 - v) * 0.00234375 + 1.4
    y_vehicle = (320 - u) * 0.003125
    return x_vehicle, y_vehicle

def extract_lane_functions(binary_image, poly_degree=3):
    """
    이진 이미지에서 cv2.connectedComponentsWithStats로 연결요소 추출 후,
    각 영역에 대해 차량 좌표계로 변환한 뒤 3차 다항식(poly_degree=3)으로 피팅하여
    차선 함수의 계수 리스트를 반환.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    lane_data = []
    for label in range(1, num_labels):
        rows, cols = np.where(labels == label)
        if len(rows) < 50:  # 너무 작은 영역은 무시
            continue
        vehicle_coords = np.array([image_to_vehicle((col, row)) for row, col in zip(rows, cols)])
        x_vehicle = vehicle_coords[:, 0]
        y_vehicle = vehicle_coords[:, 1]
        poly_coeff = np.polyfit(x_vehicle, y_vehicle, poly_degree)
        lane_data.append(poly_coeff)
    return lane_data

def compute_derivative(poly_coeff, x_value):
    """
    다항식 계수가 내림차순(a, b, c, ...)일 때, x_value에서의 도함수 값을 계산.
    """
    derivative = 0
    n = len(poly_coeff) - 1
    for i, coeff in enumerate(poly_coeff[:-1]):
        derivative += coeff * (n - i) * (x_value ** (n - i - 1))
    return derivative

def create_path_function(lane_coeffs, x_front, offset_right=0.75, offset_left=-0.75):
    """
    lane_coeffs: 검출된 차선 함수 계수 리스트 (각각 3차 다항식 계수)
    x_front: 차량 전방에서의 기준 x값 (양의 값)
    offset_right, offset_left: 각각 오른쪽 혹은 왼쪽 차선일 때 y 방향 평행이동 (경로함수에 적용)
    
    - 두 개의 차선이 검출되면 각 계수를 평균하여 경로함수를 생성.
    - 한 개만 검출되면 x_front에서의 도함수를 통해 좌/우 판별하고, 해당 오프셋을 상수항에 적용.
    """
    if len(lane_coeffs) == 2:
        lane_coeffs = np.array(lane_coeffs)
        path_coeff = np.mean(lane_coeffs, axis=0)
    elif len(lane_coeffs) == 1:
        poly_coeff = lane_coeffs[0]
        slope = compute_derivative(poly_coeff, x_front)
        # 양의 x절편에서의 기울기가 양수면 right lane, 음수면 left lane으로 판단
        if slope > 0:
            # right lane: 여기서는 +y offset 적용 (좌표계에 맞게 상수항 조정)
            path_coeff = poly_coeff.copy()
            path_coeff[-1] += offset_right
        else:
            # left lane: -y offset 적용
            path_coeff = poly_coeff.copy()
            path_coeff[-1] += offset_left
    else:
        raise ValueError("유효한 차선 함수가 없습니다.")
    return path_coeff

def sample_path_points(poly_coeff, x_start=1.4, x_end=2.9, num_points=50):
    """
    경로 함수(다항식 계수)를 일정 x 범위에서 평가하여 (x, y) 차량 좌표계상의 점들을 생성.
    """
    xs = np.linspace(x_start, x_end, num_points)
    ys = np.polyval(poly_coeff, xs)
    path_points = list(zip(xs, ys))
    return path_points

# ===================================================
# 이미지 상 차량 좌표계를 이미지 좌표로 변환 (BEV overlay)
# ---------------------------------------------------
def vehicle_to_image(point):
    """
    차량 좌표 (x_vehicle, y_vehicle)를 이미지 좌표 (u, v)로 역변환.
    역변환식:
      v = 640 - (x_vehicle - 1.4) / 0.00234375
      u = 320 - y_vehicle / 0.003125
    """
    x_vehicle, y_vehicle = point
    v = 640 - (x_vehicle - 1.4) / 0.00234375
    u = 320 - y_vehicle / 0.003125
    return int(round(u)), int(round(v))

def overlay_polyline(bev_image, path_points, color=(0, 0, 255), thickness=2):
    """
    BEV 이미지 위에 경로 함수의 샘플링된 차량 좌표계상의 점들을 polyline으로 오버레이.
    """
    image_points = [vehicle_to_image(pt) for pt in path_points]
    pts = np.array(image_points, dtype=np.int32).reshape((-1, 1, 2))
    overlayed_image = bev_image.copy()
    cv2.polylines(overlayed_image, [pts], isClosed=False, color=color, thickness=thickness)
    return overlayed_image

# ===================================================
# 모폴로지 및 연결요소 기반 필터
# ---------------------------------------------------
def morph_open(binary_mask, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=100):
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

# ===================================================
# BEV 변환 함수
# ---------------------------------------------------
def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev = cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)
    return bev

# Pure Pursuit 결과를 Matplotlib으로 디버깅 시각화
def debug_plot_lane(path_points, goal_point=None):
    if len(path_points) > 0:
        lane_vehicle = np.array(path_points)
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

# ===================================================
# 메인 처리 함수: detect_and_publish (pub_steering 추가)
# ---------------------------------------------------
def detect_and_publish(opt, pub_mask, pub_steering):
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
    vid_path = None
    vid_writer = None

    stride = 32
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
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
    else:
        rospy.loginfo("[INFO] 파일(영상/이미지): %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    def process_frame(im0s):
        net_input_img, ratio, pad = letterbox(im0s, (imgsz, imgsz), stride=stride)
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

        # 여기서 final_mask(차선 영역)를 이용해 차선 함수(3차 다항식) 피팅
        lane_data = extract_lane_functions(final_mask, poly_degree=3)
        # 차량 전방에서의 기준 x 값 (예: 2m)에서 도함수를 계산해 좌우 판별
        try:
            path_coeff = create_path_function(lane_data, x_front=2, offset_right=0.75, offset_left=-0.75)
        except ValueError as e:
            rospy.logwarn(str(e))
            path_coeff = None

        if path_coeff is not None:
            path_points = sample_path_points(path_coeff, x_start=1.4, x_end=2.9, num_points=50)
        else:
            path_points = []

        bev_im = do_bev_transform(im0s, bev_param_file)

        if path_coeff is not None and len(path_points) > 0:
            bev_im_color = overlay_polyline(bev_im.copy(), path_points, color=(0, 0, 255), thickness=2)

            # Pure Pursuit: 전방주시거리(lookahead) 이상 떨어진 목표점 선택
            lookahead = 210.0  # cm 단위
            lookahead_m = lookahead/100.0
            wheelbase = 75.0   # cm 단위
            wheelbase_m = wheelbase/100.0
            goal_point = None
            min_error = float('inf')
            for pt in path_points:
                X_v, Y_v = pt
                d = np.sqrt(X_v**2 + Y_v**2)
                error = abs(d - lookahead_m)
                if error < min_error:
                    min_error = error
                    goal_point = (X_v, Y_v)
            if goal_point is None and len(path_points) > 0:
                goal_point = path_points[-1]

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

                # 목표점을 BEV 이미지 상에 표시 (역변환)
                goal_x_img, goal_y_img = vehicle_to_image(goal_point)
                cv2.circle(bev_im_color, (goal_x_img, goal_y_img), 5, (0, 255, 0), -1)
                cv2.putText(bev_im_color, f"Steering: {steering_angle_deg:.2f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            else:
                bev_im_color = overlay_polyline(bev_im.copy(), path_points, color=(0, 0, 255), thickness=2)

            if opt.debug:
                debug_plot_lane(path_points, goal_point)
        else:
            bev_im_color = bev_im.copy()

        cv2.imshow("final+mask", thin_mask)
        cv2.imshow("BEV + Polyfit", bev_im_color)
        cv2.imshow("final+mask", final_mask)
        return bev_im, bev_im_color, final_mask

    start_time = None

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
                rospy.loginfo("[INFO] 프레임 종료 신호 수신")
                break
            path_item, net_input_img, im0s, vid_cap = frame_data
            if frame_skip > 0:
                if frame_counter % (frame_skip + 1) != 0:
                    frame_counter += 1
                    continue
                frame_counter += 1
            if start_time is None:
                start_time = time.time()

            bev_im, bev_im_color, thin_mask = process_frame(im0s)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] q 입력 → 종료")
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
            bev_im, bev_im_color, thin_mask = process_frame(im0s)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] q 입력 → 종료")
                break
        rospy.loginfo("[INFO] 동기 처리 추론 평균 시간: %.4fs/frame, 전체: %.3fs", inf_time.avg, time.time()-start_time)

    cv2.destroyAllWindows()
    rospy.loginfo("[INFO] 추론 완료.")
    
    rospy.loginfo("[INFO] detect_and_publish 종료.")

# ===================================================
# 메인 ros_main
# ---------------------------------------------------
def ros_main():
    rospy.init_node('bev_lane_thinning_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    if opt.debug:
        plt.ion()
    pub_mask = rospy.Publisher('camera_lane_segmentation/lane_mask', Image, queue_size=1)
    pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
    detect_and_publish(opt, pub_mask, pub_steering)
    rospy.loginfo("[INFO] bev_lane_thinning_node 종료, spin() 호출")
    rospy.spin()

# ===================================================
# 메인 실행
# ---------------------------------------------------
if __name__=='__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
