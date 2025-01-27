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
import math
from collections import deque

# (중요) 충돌 방지: pathlib의 Path -> PathlibPath, ROS msg의 Path -> RosPath
from pathlib import Path as PathlibPath
from nav_msgs.msg import Path as RosPath
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64

# ===== utils.py (별도 파일) =====
# 질문에서 이미 존재한다고 하신 utils.py의 함수들 불러오기
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    AverageMeter,
    LoadCamera,
    LoadImages,
    lane_line_mask
)


def make_parser():
    """
    커맨드라인 인자 파서를 생성합니다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='YOLOPv2 모델(.pt) 파일 경로')
    parser.add_argument('--source', type=str,
                        default='0',
                        help='비디오/이미지 경로 or 웹캠 소스(예: 2=/dev/video2)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='추론 시 입력 사이즈 (pixels)')
    parser.add_argument('--device', default='0',
                        help='cuda device (예: 0,1) 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.8,
                        help='차선 세그멘테이션 스코어 임계값 (0~1)')
    parser.add_argument('--nosave', action='store_false',
                        help='결과 영상을 저장하지 않음')
    parser.add_argument('--project', default='/home/highsky/My_project_work_ws/runs/detect',
                        help='결과 저장 기본 디렉터리')
    parser.add_argument('--name', default='exp',
                        help='프로젝트/이름 => runs/detect/exp')
    parser.add_argument('--exist-ok', action='store_true',
                        help='이미 디렉터리가 존재해도 덮어쓸지 여부')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='N 프레임을 건너뛸 때 사용(현재 미사용)')
    return parser


# ------------------------------------------------
# MiDaS(Depth) 클래스
# ------------------------------------------------
class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"[INFO] Loading MiDaS model ({model_type})...")

        # PyTorch Hub에서 MiDaS 모델 로드
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.midas.to(self.device)
        self.midas.eval()

        # MiDaS 전처리용 transform
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if "DPT" in model_type:
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

        rospy.loginfo("[INFO] MiDaS model loaded successfully.")

    def run(self, cv2_image):
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            depth_map = torch.squeeze(prediction).cpu().numpy()

        # 0~1로 정규화
        min_val, max_val = depth_map.min(), depth_map.max()
        denom = (max_val - min_val) if (max_val > min_val) else 1e-8
        depth_map = (depth_map - min_val) / denom
        return depth_map


# ------------------------------------------------
# IPM(인버스 퍼스펙티브 맵) 변환 함수
# ------------------------------------------------
def get_birds_eye_view(frame, src_points, dst_size=(400, 600)):
    """
    frame: 원본 BGR 이미지
    src_points: 원본 영상에서 사다리꼴 형태의 네 꼭짓점 (좌하->우하->우상->좌상) 등의 순서
    dst_size: 출력 톱뷰 이미지 크기 (w,h)
    return: (birds_eye, M, Minv)
        - birds_eye: 톱뷰로 변환된 이미지
        - M: 원본 -> 톱뷰 3x3 행렬
        - Minv: 톱뷰 -> 원본 3x3 행렬
    """
    w, h = dst_size
    dst_points = np.array([
        [0,   h-1],   # 왼쪽 아래
        [w-1, h-1],   # 오른쪽 아래
        [w-1, 0],     # 오른쪽 위
        [0,   0]      # 왼쪽 위
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points.astype(np.float32))
    birds_eye = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    return birds_eye, M, Minv


# ------------------------------------------------
# 톱뷰 상에서 차선을 검출하고, 스플라인(또는 다항식) 보정
# ------------------------------------------------
def lane_spline_fitting_in_birdeye(lane_mask_bev):
    """
    lane_mask_bev: 톱뷰(binary)에서의 차선 영역 (H,W)
    return: (x_new, y_new) 곡선 좌표 리스트(픽셀 단위, 톱뷰 좌표계)
    """
    indices = np.argwhere(lane_mask_bev > 127)  # shape: (N,2) = (y,x)
    if len(indices) < 10:
        return []

    # y 기준 오름차순 정렬
    indices = indices[indices[:, 0].argsort()]

    # y 간격으로 x 평균 구하기
    step = 5
    height = lane_mask_bev.shape[0]
    y_vals = []
    x_vals = []

    for y in range(0, height, step):
        y_bin = indices[(indices[:, 0] >= y) & (indices[:, 0] < y+step)]
        if len(y_bin) > 0:
            mean_x = np.mean(y_bin[:, 1])
            y_center = y + (step/2)
            y_vals.append(y_center)
            x_vals.append(mean_x)

    if len(y_vals) < 3:
        return []

    y_vals = np.array(y_vals, dtype=np.float32)
    x_vals = np.array(x_vals, dtype=np.float32)

    # 2차 다항식 피팅 (np.polyfit)
    poly_coefs = np.polyfit(y_vals, x_vals, deg=2)
    # 보간을 위해 표본점을 좀 더 촘촘하게
    y_new = np.linspace(y_vals[0], y_vals[-1], num=50)
    x_new = poly_coefs[0]*y_new**2 + poly_coefs[1]*y_new + poly_coefs[2]

    curve_xy = list(zip(x_new, y_new))
    return curve_xy  # [(x_bev, y_bev), ...]


# ------------------------------------------------
# 곡선(톱뷰) -> 원본 영상 좌표계로 역투영
# ------------------------------------------------
def project_lane_back(curve_xy, Minv):
    """
    curve_xy: [(x_bev, y_bev), ...] (톱뷰 좌표계)
    Minv: IPM 역행렬
    return: [(u,v), ...] (원본 영상 픽셀좌표)
    """
    if not curve_xy:
        return []

    ones = np.ones((len(curve_xy), 1), dtype=np.float32)
    # curve_xy는 (x,y) = (col,row) 형태이므로, stack 시 (col, row)
    pts_bev = np.hstack([np.array(curve_xy, dtype=np.float32), ones])  # shape: (N, 3)
    pts_bev = pts_bev.reshape(-1, 3).T  # shape: (3, N)

    # 역투영
    pts_src = np.dot(Minv, pts_bev)  # shape: (3, N)
    # normalize
    pts_src[0, :] /= (pts_src[2, :] + 1e-8)
    pts_src[1, :] /= (pts_src[2, :] + 1e-8)

    pts_src = pts_src[:2, :].T  # shape: (N, 2)
    return [(float(p[0]), float(p[1])) for p in pts_src]


# ------------------------------------------------
# 경로(Path)를 이미지에 2D로 그려주는 함수
# ------------------------------------------------
def project_path_to_image(disp_bgr, path_pts, fx, fy, cx, cy):
    """
    path_pts: (x_3d, z_3d) 목록 (y=0 가정)
    disp_bgr: 시각화용 BGR 이미지
    fx, fy, cx, cy: 간단한 투영 파라미터
    """
    h, w = disp_bgr.shape[:2]
    color = (0, 0, 255)  # 빨간색
    thickness = 2

    prev_px, prev_py = None, None
    for (x_3d, z_3d) in path_pts:
        if z_3d <= 0.01:
            continue
        px = (x_3d * fx / z_3d) + cx
        py = (0.0 * fy / z_3d) + cy  # y=0 가정

        if 0 <= px < w and 0 <= py < h:
            cv2.circle(disp_bgr, (int(px), int(py)), 4, color, -1)
            if prev_px is not None and prev_py is not None:
                cv2.line(disp_bgr, (int(prev_px), int(prev_py)), (int(px), int(py)), color, thickness)
            prev_px, prev_py = px, py
        else:
            prev_px, prev_py = None, None


# ------------------------------------------------
# 차량의 현재 위치를 RViz에 마커로 퍼블리시
# ------------------------------------------------
def publish_vehicle_marker(pub_vehicle_marker):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "map"  # 실제 TF에 맞춰 수정
    marker.ns = "vehicle"
    marker.id = 1
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.5  # 길이
    marker.scale.y = 0.2  # 너비
    marker.scale.z = 0.2  # 높이
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0  # 파란색
    marker.color.a = 1.0

    # 차량의 현재 위치 (0,0,0) 및 방향 (예: 전방을 향한 방향)
    marker.points = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0)  # 전방을 향한 방향
    ]

    pub_vehicle_marker.publish(marker)


# ------------------------------------------------
# 3D 차선 포인트를 RViz marker로 퍼블리시
# ------------------------------------------------
def publish_lanes_3d(points_3d, pub_lane_marker):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "map"  # 실제 TF에 맞춰 수정
    marker.ns = "lane"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0  # 녹색
    marker.color.a = 1.0

    for (x_3d, y_3d, z_3d) in points_3d:
        pt = Point()
        # map 프레임에서 x=전방, y=좌우, z=상하 라고 가정
        pt.x = z_3d
        pt.y = x_3d
        pt.z = y_3d
        marker.points.append(pt)

    pub_lane_marker.publish(marker)


# ------------------------------------------------
# 조향각(steering angle) 계산(단순)
# ------------------------------------------------
def compute_steering_angle(path_pts):
    """
    path_pts: [(x_3d, z_3d), ...]
    - 맨 뒤(가장 먼 점)를 활용하여 angle = atan2(x,z)
    - deg(도) 단위로 반환
    """
    if len(path_pts) == 0:
        return 0.0
    x_target, z_target = path_pts[0]
    if abs(z_target) < 1e-3:
        return 0.0
    angle = math.atan2(x_target, z_target) * (180 / math.pi)
    return angle


# ------------------------------------------------
# 경로의 이동 평균을 계산하는 함수
# ------------------------------------------------
def smooth_path(path_buffer):
    """
    path_buffer: deque of path_3d lists
    return: smoothed_path_3d list
    """
    all_x = []
    all_z = []
    for path in path_buffer:
        for (x, y, z) in path:
            all_x.append(x)
            all_z.append(z)
    if len(all_x) < 3:
        return []

    # 2차 다항식 피팅
    poly_coefs = np.polyfit(all_z, all_x, deg=2)
    z_new = np.linspace(min(all_z), max(all_z), num=50)
    x_new = poly_coefs[0]*z_new**2 + poly_coefs[1]*z_new + poly_coefs[2]

    smoothed_path = [(x, 0.0, z) for x, z in zip(x_new, z_new)]
    return smoothed_path


# ------------------------------------------------
# (수정) 부드러운 경로 생성 (IPM + 곡선 피팅 + 역투영 + 3D 변환 + Temporal Smoothing)
# ------------------------------------------------
def build_robust_path_ipm(thin_mask, depth_map, depth_scale,
                          fx, fy, cx, cy,
                          trap_points):
    """
    1) thin_mask(원본) -> IPM(톱뷰) 변환
    2) 톱뷰 상에서 차선 스플라인 보정
    3) 곡선을 원본 픽셀로 역투영
    4) depth_map 이용 (x,z) 3D 좌표 계산
    5) 반환: [(x_3d, y_3d, z_3d), ...]  (현재 y_3d=0 가정)
    """
    h, w = thin_mask.shape[:2]

    # 1) IPM 변환
    mask_3ch = cv2.cvtColor(thin_mask, cv2.COLOR_GRAY2BGR)
    bev_mask, M_bev, Minv_bev = get_birds_eye_view(mask_3ch,
                                                   src_points=trap_points.astype(np.float32),
                                                   dst_size=(400, 600))
    bev_gray = cv2.cvtColor(bev_mask, cv2.COLOR_BGR2GRAY)

    # 2) 톱뷰 곡선 검출
    curve_xy = lane_spline_fitting_in_birdeye(bev_gray)  # [(x_bev, y_bev), ...]

    # 3) 원본 픽셀로 역투영
    curve_uv = project_lane_back(curve_xy, Minv_bev)  # [(u,v), ...]

    # 4) (u,v) + depth_map -> (x_3d, z_3d)
    path_pts_3d = []
    for (u, v) in curve_uv:
        # 범위 체크
        if not (0 <= int(u) < w and 0 <= int(v) < h):
            continue
        d_val = depth_map[int(v), int(u)]  # 0~1
        z_val = depth_scale / (d_val + 1e-3)
        x_3d = (u - cx) * z_val / fx
        # y_3d=0으로 가정 (평면)
        path_pts_3d.append((x_3d, 0.0, z_val))

    # z 오름차순으로 정렬
    path_pts_3d.sort(key=lambda p: p[2])

    return path_pts_3d


# ------------------------------------------------
# OpenCV 영상을 파일로 저장(웹캠 모드)
# ------------------------------------------------
def make_webcam_video(record_frames, save_dir: PathlibPath, stem_name: str, real_duration: float):
    if len(record_frames) == 0:
        rospy.loginfo("[INFO] 저장할 웹캠 프레임이 없습니다.")
        return
    num_frames = len(record_frames)
    if real_duration <= 0:
        real_duration = 1e-6
    real_fps = num_frames / real_duration
    rospy.loginfo("[INFO] 웹캠 녹화: 총 %d프레임, 소요 %.2f초 => FPS ~ %.2f",
                  num_frames, real_duration, real_fps)

    save_path = str(save_dir / f"{stem_name}_webcam.mp4")
    h, w = record_frames[0].shape[:2]
    out = cv2.VideoWriter(save_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          real_fps,
                          (w, h))
    for f in record_frames:
        out.write(f)
    out.release()
    rospy.loginfo("[INFO] 웹캠 결과 영상 저장 완료: %s", save_path)


# ------------------------------------------------
# 메인 함수: YOLO + MiDaS -> 차선 + Depth -> IPM -> Path -> ROS 퍼블리시
# ------------------------------------------------
def detect_and_publish(opt, pub_mask, pub_path, pub_lane_marker, pub_vehicle_marker, pub_steer):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
    cudnn.benchmark = True
    bridge = CvBridge()

    source = opt.source
    weights = opt.weights
    lane_thr = opt.lane_thres
    imgsz = opt.img_size
    save_img = not opt.nosave and (isinstance(source, str) and not source.endswith('.txt'))

    # 결과 저장 디렉토리
    save_dir = PathlibPath(increment_path(PathlibPath(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    vid_path, vid_writer = None, None

    # 1) YOLOPv2 모델 로드
    rospy.loginfo("[INFO] Loading YOLOv2 model: %s", weights)
    device = select_device(opt.device)
    model = torch.jit.load(weights, map_location=device)
    half = (device.type != 'cpu')
    model = model.half() if half else model.float()
    model.eval()

    # 2) MiDaS Depth Estimator 준비
    depth_est = DepthEstimator(model_type="MiDaS_small")

    # 3) 데이터 로드
    if source.isdigit():
        rospy.loginfo(f"[INFO] 웹캠(장치 ID={source})로부터 영상 입력...")
        dataset = LoadCamera(source, img_size=imgsz, stride=32)
    else:
        rospy.loginfo(f"[INFO] 파일(이미지/영상) {source} 로드...")
        dataset = LoadImages(source, img_size=imgsz, stride=32)

    record_frames = []
    start_time = None

    # GPU 워밍업
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # (예시) 카메라 파라미터
    fx = 663.13788013*2
    fy = 663.85577581*2
    cx = 327.61915049*2
    cy = 249.78533605*2
    depth_scale = 0.5724  # MiDaS depth -> 실제 거리로 가정 (조정 필요)

    # 사다리꼴 ROI 좌표 (1280x720 가정)
    # [좌하, 우하, 우상, 좌상] 순
    trap_points = np.array([
        [0,  720],
        [980, 720],
        [980,  0],
        [0,  0]
    ], dtype=np.int32)

    # 경로의 이동 평균을 위한 버퍼 (최근 10개의 경로)
    path_buffer = deque(maxlen=10)

    while True:
        try:
            path_item, img, im0s, vid_cap = next(dataset)
        except StopIteration:
            rospy.loginfo("[INFO] 데이터스트림 종료.")
            break

        if dataset.mode == 'stream' and start_time is None:
            start_time = time.time()

        # img -> torch tensor
        img_t = torch.from_numpy(img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        # 4) YOLOPv2 차선 추론
        t1 = time_synchronized()
        with torch.no_grad():
            [_, _], seg, ll = model(img_t)  # YOLOPv2 출력
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        # 차선 세그멘테이션 -> 이진 마스크
        ll_seg = lane_line_mask(ll, threshold=lane_thr)
        binary_mask = (ll_seg * 255).astype(np.uint8)

        # 스켈레톤
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)

        # 사다리꼴 ROI 적용
        roi_mask = np.zeros_like(thin_mask, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [trap_points], 255)
        thin_mask = cv2.bitwise_and(thin_mask, roi_mask)

        # 5) MiDaS Depth 추정
        depth_map = depth_est.run(im0s)
        if depth_map.shape[:2] != (im0s.shape[0], im0s.shape[1]):
            depth_map = cv2.resize(depth_map, (im0s.shape[1], im0s.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 6) IPM + 곡선 보정 -> 3D Path
        path_3d = build_robust_path_ipm(thin_mask, depth_map, depth_scale,
                                        fx, fy, cx, cy,
                                        trap_points)

        # 7) 경로 버퍼에 추가 및 스무딩
        if path_3d:
            path_buffer.append(path_3d)
            smoothed_path_3d = smooth_path(path_buffer)
        else:
            smoothed_path_3d = []

        # 8) ROS Path 퍼블리시
        path_msg = RosPath()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        for (x_3d, y_3d, z_3d) in smoothed_path_3d:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(z_3d)
            pose.pose.position.y = float(x_3d)
            pose.pose.position.z = float(y_3d)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        pub_path.publish(path_msg)

        # 9) 조향각 계산 & 발행
        path_for_angle = [(pt[0], pt[2]) for pt in smoothed_path_3d]  # (x,z)
        steer_angle = compute_steering_angle(path_for_angle)
        angle_msg = Float64()
        angle_msg.data = steer_angle
        pub_steer.publish(angle_msg)

        # 10) 차선 마스크 ROS 퍼블리시
        try:
            ros_mask = bridge.cv2_to_imgmsg(thin_mask, encoding="mono8")
            pub_mask.publish(ros_mask)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))

        # 11) 차량의 현재 위치 RViz에 마커로 퍼블리시
        publish_vehicle_marker(pub_vehicle_marker)

        # 12) 3D 차선 포인트 -> RViz Marker 퍼블리시
        publish_lanes_3d(smoothed_path_3d, pub_lane_marker)

        # 13) OpenCV 시각화 (Depth + Lane + Path)
        disp_depth = (depth_map * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_depth, cv2.COLORMAP_JET)

        # 차선(흰색) 오버레이
        lane_indices = np.where(thin_mask == 255)
        disp_color[lane_indices] = (255, 255, 255)

        # Path(빨간색) 표시
        path_for_draw = [(pt[0], pt[2]) for pt in smoothed_path_3d]  # (x,z)
        project_path_to_image(disp_color, path_for_draw, fx, fy, cx, cy)

        # 사다리꼴 ROI 폴리라인(노란색)
        cv2.polylines(disp_color, [trap_points], isClosed=True, color=(0, 255, 255), thickness=2)

        # 조향각 표시
        cv2.putText(disp_color, f"SteerAngle(deg)={steer_angle:.3f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Depth + Lane + Path", disp_color)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.loginfo("[INFO] 'q' 키 입력 -> 종료")
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()
            if hasattr(dataset, 'cap') and dataset.cap:
                dataset.cap.release()
            cv2.destroyAllWindows()

            # 웹캠 녹화본 저장
            if dataset.mode == 'stream' and save_img:
                end_time = time.time()
                real_duration = end_time - start_time if start_time else 0
                stem_name = PathlibPath(path_item).stem if path_item else 'webcam0'
                make_webcam_video(record_frames, save_dir, stem_name, real_duration)
            return

        # 14) 결과 저장 (이미지/비디오/웹캠)
        if save_img:
            if dataset.mode == 'image':
                save_path = str(save_dir / PathlibPath(path_item).name)
                sp = PathlibPath(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')
                cv2.imwrite(str(sp), disp_color)

            elif dataset.mode == 'video':
                save_path = str(save_dir / PathlibPath(path_item).stem) + '.mp4'
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) or 30
                    else:
                        fps = 30
                    wv, hv = disp_color.shape[1], disp_color.shape[0]
                    rospy.loginfo(f"[INFO] 비디오 저장 시작: {vid_path} (FPS={fps}, size=({wv},{hv}))")
                    vid_writer = cv2.VideoWriter(
                        vid_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (wv, hv)
                    )
                vid_writer.write(disp_color)
            else:
                # 웹캠(stream)
                record_frames.append(disp_color.copy())


# ------------------------------------------------
# ros_main() : 노드 초기화 및 메인 함수 실행
# ------------------------------------------------
def ros_main():
    rospy.init_node('yolopv2_laneline_node', anonymous=True)

    parser = make_parser()
    opt, _ = parser.parse_known_args()

    # ROS 퍼블리셔 생성
    pub_mask = rospy.Publisher('yolopv2/lane_mask', Image, queue_size=1)
    pub_path = rospy.Publisher('yolopv2/driving_path', RosPath, queue_size=1)
    pub_lane_marker = rospy.Publisher('yolopv2/lane_marker', Marker, queue_size=1)
    pub_vehicle_marker = rospy.Publisher('yolopv2/vehicle_marker', Marker, queue_size=1)  # 차량 위치 마커
    pub_steer = rospy.Publisher('steering_angle', Float64, queue_size=1)

    detect_and_publish(opt, pub_mask, pub_path, pub_lane_marker, pub_vehicle_marker, pub_steer)

    rospy.loginfo("[INFO] Pseudo-Lidar Lane node finished. spin() for keepalive.")
    rospy.spin()


if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
