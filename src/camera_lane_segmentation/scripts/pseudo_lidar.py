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

# (중요) 충돌 방지: pathlib의 Path -> PathlibPath, ROS msg의 Path -> RosPath
from pathlib import Path as PathlibPath
from nav_msgs.msg import Path as RosPath
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64

# ===== utils.py (별도 파일) =====
# 이미 질문에 주어진 utils.py를 import합니다.
# "from utils.utils import ..." 형태로 필요한 함수들을 불러옵니다.
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    AverageMeter,
    LoadCamera,
    LoadImages,
    lane_line_mask
)

"""
이 노드는 YOLOPv2 + MiDaS를 이용하여
1) 차선 검출
2) 깊이 추정
3) 3D 공간상에서 경로(Path) 생성 (개선된 알고리즘)
4) 조향각 계산
5) ROS 토픽 발행 (차선 mask, Path, steering angle 등)
6) OpenCV 윈도우로 실시간 시각화(Depth + Lane + Path)
"""

def make_parser():
    """
    커맨드라인 인자 파서를 생성합니다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='YOLOPv2 모델(.pt) 파일 경로')
    parser.add_argument('--source', type=str,
                        default='2',
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
# **MiDaS** (Depth Estimator) 클래스
# ------------------------------------------------
class DepthEstimator:
    """
    Intel-ISL의 MiDaS 모델을 사용해 단일 이미지에서 깊이 정보를 추정합니다.
    """
    def __init__(self, model_type="MiDaS_small"):
        # 현재 가능한 디바이스: GPU or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"[INFO] Loading MiDaS model ({model_type})...")

        # PyTorch Hub에서 MiDaS 모델 로드 (trust_repo=True: 리포 신뢰)
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
        """
        입력: OpenCV BGR 이미지
        출력: 0~1 범위로 정규화된 depth map (np.array, shape=(H,W))
        """
        # BGR -> RGB
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # MiDaS가 권장하는 transform 적용
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            depth_map = torch.squeeze(prediction).cpu().numpy()

        # min/max 정규화 => 0~1 사이로
        min_val, max_val = depth_map.min(), depth_map.max()
        denom = (max_val - min_val) if (max_val > min_val) else 1e-8
        depth_map = (depth_map - min_val) / denom
        return depth_map

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
        py = (0.0 * fy / z_3d) + cy  # y_3d=0 가정

        # 화면 범위 내인지 확인
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(disp_bgr, (int(px), int(py)), 4, color, -1)
            if prev_px is not None and prev_py is not None:
                cv2.line(disp_bgr, (int(prev_px), int(prev_py)), (int(px), int(py)), color, thickness)
            prev_px, prev_py = px, py
        else:
            prev_px, prev_py = None, None

# ------------------------------------------------
# 3D 차선 포인트를 RViz marker로 퍼블리시
# (현재는 RViz 없이도 보지만, marker만 따로 사용 가능)
# ------------------------------------------------
def publish_lanes_3d(points_3d, pub_marker):
    """
    points_3d: [(x_3d, y_3d, z_3d), ...]
    """
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "map"  # 실제 TF에 맞춰 수정 가능
    marker.ns = "lane"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0

    for (x_3d, y_3d, z_3d) in points_3d:
        pt = Point()
        # map 프레임에서 x=전방, y=좌우, z=상하 가정
        pt.x = z_3d
        pt.y = x_3d
        pt.z = y_3d
        marker.points.append(pt)

    pub_marker.publish(marker)

# ------------------------------------------------
# 조향각(steering angle) 계산(단순)
# ------------------------------------------------
def compute_steering_angle(path_pts):
    """
    path_pts: [(x_3d, z_3d), ...]
    - 가장 먼 점(마지막 점)을 활용하여 angle = atan2(x,z)
    - 여기서는 deg(도) 단위로 반환
    """
    if len(path_pts) == 0:
        return 0.0
    x_target, z_target = path_pts[-1]
    if abs(z_target) < 1e-3:
        return 0.0
    angle = math.atan2(x_target, z_target) * (180 / math.pi)
    return angle

# ------------------------------------------------
# (개선) 경로 생성 알고리즘 
#  - 기존 단순 binning → 
#  - 좌/우 차선 분할 후 중앙선(center line) 생성 → 
#  - (선택) polynomial fit 등 추가 보정 가능
# ------------------------------------------------
def build_robust_path(points_3d, z_bin_size=2.0):
    """
    (1) 3D 차선 포인트들(points_3d)을 z(전방) 기준으로 정렬
    (2) z 구간마다 왼쪽/오른쪽 포인트를 구분하여 평균, 중앙값 계산
    (3) (선택) 전체 (z, x) 쌍에 대해 polynomial fit 수행
    (4) Path 좌표 (x, z) 목록 반환

    points_3d: [(x, y, z), ...] 형태 (y는 여기서는 사용 X)
    z_bin_size: z축을 일정 간격으로 binning할 크기
    """
    if len(points_3d) < 10:
        return []

    # 1) z 기준 정렬
    sorted_pts = sorted(points_3d, key=lambda p: p[2])

    robust_path = []
    current_bin_start = 0.0
    left_points, right_points = [], []

    # 최초 z
    current_bin_start = sorted_pts[0][2]

    for (x_3d, y_3d, z_3d) in sorted_pts:
        if z_3d < current_bin_start + z_bin_size:
            # bin 내부
            if x_3d < 0:
                left_points.append((x_3d, z_3d))
            else:
                right_points.append((x_3d, z_3d))
        else:
            # bin 종료 => 왼/오 포인트의 평균(중앙)을 path로 사용
            if len(left_points) > 0 and len(right_points) > 0:
                avg_x_left = np.mean([p[0] for p in left_points])
                avg_x_right = np.mean([p[0] for p in right_points])
                center_x = (avg_x_left + avg_x_right) / 2.0
                center_z = (np.mean([p[1] for p in left_points]) + np.mean([p[1] for p in right_points])) / 2.0
                robust_path.append((center_x, center_z))
            elif len(left_points) > 0:
                # 오른쪽 차선이 없을 때 => 단독으로 왼쪽 차선만 평균
                avg_x_left = np.mean([p[0] for p in left_points])
                avg_z_left = np.mean([p[1] for p in left_points])
                robust_path.append((avg_x_left, avg_z_left))
            elif len(right_points) > 0:
                # 왼쪽 차선이 없을 때 => 단독으로 오른쪽 차선만 평균
                avg_x_right = np.mean([p[0] for p in right_points])
                avg_z_right = np.mean([p[1] for p in right_points])
                robust_path.append((avg_x_right, avg_z_right))

            # 다음 bin으로 이동
            left_points = []
            right_points = []
            current_bin_start = z_3d

            # 이번 점도 현재 bin에 포함
            if x_3d < 0:
                left_points.append((x_3d, z_3d))
            else:
                right_points.append((x_3d, z_3d))

    # 루프 종료 후 남은 bin 처리
    if len(left_points) > 0 and len(right_points) > 0:
        avg_x_left = np.mean([p[0] for p in left_points])
        avg_x_right = np.mean([p[0] for p in right_points])
        center_x = (avg_x_left + avg_x_right) / 2.0
        center_z = (np.mean([p[1] for p in left_points]) + np.mean([p[1] for p in right_points])) / 2.0
        robust_path.append((center_x, center_z))
    elif len(left_points) > 0:
        avg_x_left = np.mean([p[0] for p in left_points])
        avg_z_left = np.mean([p[1] for p in left_points])
        robust_path.append((avg_x_left, avg_z_left))
    elif len(right_points) > 0:
        avg_x_right = np.mean([p[0] for p in right_points])
        avg_z_right = np.mean([p[1] for p in right_points])
        robust_path.append((avg_x_right, avg_z_right))

    # (선택) 추가 폴리노멀 피팅(예: np.polyfit)으로 곡선을 부드럽게 할 수도 있음
    # 예시: x = f(z) 로 2차 회귀
    if len(robust_path) > 5:
        zs = [pz for (_, pz) in robust_path]
        xs = [px for (px, _) in robust_path]
        # 2차 다항식 계수 추정
        poly_coefs = np.polyfit(zs, xs, deg=2)
        # 일정 샘플링으로 path 재구성(더 부드러운 곡선)
        final_path = []
        min_z = min(zs)
        max_z = max(zs)
        num_samples = 20
        for i in range(num_samples):
            z_smpl = min_z + (max_z - min_z)*i/(num_samples-1)
            x_smpl = poly_coefs[0]*z_smpl**2 + poly_coefs[1]*z_smpl + poly_coefs[2]
            final_path.append((x_smpl, z_smpl))
        return final_path
    else:
        return robust_path

# ------------------------------------------------
# OpenCV 영상을 파일로 저장(웹캠 모드일 때)
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
# 메인 함수: YOLO로 차선 탐지 -> 깊이 -> 경로 -> ROS 퍼블리시
# ------------------------------------------------
def detect_and_publish(opt, pub_mask, pub_path, pub_lane_marker, pub_steer):
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

    # --------------------------------------------
    # 1) YOLOPv2 모델 로드 (TorchScript)
    # --------------------------------------------
    rospy.loginfo("[INFO] Loading YOLOv2 model: %s", weights)
    device = select_device(opt.device)
    model = torch.jit.load(weights, map_location=device)
    half = (device.type != 'cpu')
    model = model.half() if half else model.float()
    model.eval()

    # --------------------------------------------
    # 2) MiDaS Depth Estimator 준비
    # --------------------------------------------
    depth_est = DepthEstimator(model_type="MiDaS_small")

    # 데이터 로드 (카메라 스트림 or 이미지/비디오)
    if source.isdigit():
        # 웹캠/카메라
        rospy.loginfo(f"[INFO] 웹캠(장치 ID={source})로부터 영상 입력 받는 중...")
        dataset = LoadCamera(source, img_size=imgsz, stride=32)
    else:
        # 파일(이미지/비디오)
        rospy.loginfo(f"[INFO] 파일(이미지/영상) {source} 로드 중...")
        dataset = LoadImages(source, img_size=imgsz, stride=32)

    record_frames = []
    start_time = None

    # GPU 워밍업
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # (가정) 카메라 파라미터(임의)
    fx = 700.0
    fy = 700.0
    cx = 640.0
    cy = 360.0
    depth_scale = 10.0  # depth_map을 실제 z로 변환할 때 임의 스케일

    # 사다리꼴 ROI 좌표 (1280x720 가정; 필요 시 조정 가능)
    trap_points = np.array([
        [100, 720],   # 왼쪽 아래
        [1180, 720],  # 오른쪽 아래
        [880,  400],  # 오른쪽 위
        [400,  400]   # 왼쪽 위
    ], dtype=np.int32)

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

        # --------------------------------------------
        # 3) YOLO 추론 (차선 세그멘테이션만 사용)
        # --------------------------------------------
        t1 = time_synchronized()
        with torch.no_grad():
            # YOLOPv2 출력은 [det, da_seg], seg, ll 등이라 가정
            # 여기서 우리는 seg, ll(차선) 만 관심이 있음
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        # 차선 세그멘테이션 -> 이진 마스크
        ll_seg = lane_line_mask(ll, threshold=lane_thr)
        binary_mask = (ll_seg * 255).astype(np.uint8)
        # ximgproc.thinning => 얇은 스켈레톤
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)

        # 사다리꼴 ROI 적용
        roi_mask = np.zeros_like(thin_mask, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [trap_points], 255)
        thin_mask = cv2.bitwise_and(thin_mask, roi_mask)

        # --------------------------------------------
        # 4) Depth 추정 (MiDaS)
        # --------------------------------------------
        depth_map = depth_est.run(im0s)

        # (Depth map 크기 -> 원본 크기와 동일하게 맞춤)
        if depth_map.shape[:2] != (im0s.shape[0], im0s.shape[1]):
            depth_map = cv2.resize(depth_map, (im0s.shape[1], im0s.shape[0]), interpolation=cv2.INTER_LINEAR)

        # --------------------------------------------
        # 5) (차선 + depth_map) -> 3D 포인트
        # --------------------------------------------
        points_3d = []
        mask_indices = np.argwhere(thin_mask > 127)
        for (py, px) in mask_indices:
            d_val = depth_map[py, px]  # 0~1
            z_val = depth_scale / (d_val + 1e-3)  # 임의 스케일
            x_3d = (px - cx) * z_val / fx
            y_3d = (py - cy) * z_val / fy
            points_3d.append((x_3d, y_3d, z_val))

        # --------------------------------------------
        # 6) (개선) 경로 생성
        # --------------------------------------------
        path_pts = build_robust_path(points_3d, z_bin_size=2.0)

        # --------------------------------------------
        # 7) ROS 메시지로 경로 발행
        # --------------------------------------------
        path_msg = RosPath()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        for (x_3d, z_3d) in path_pts:
            pose = PoseStamped()
            pose.header = path_msg.header
            # x=전방, y=좌우, z=상하
            pose.pose.position.x = float(z_3d)
            pose.pose.position.y = float(x_3d)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        pub_path.publish(path_msg)

        # --------------------------------------------
        # 8) 조향각 계산 & 발행
        # --------------------------------------------
        steer_angle = compute_steering_angle(path_pts)  # deg
        angle_msg = Float64()
        angle_msg.data = steer_angle
        pub_steer.publish(angle_msg)

        # --------------------------------------------
        # 9) 차선 마스크 (mono8) ROS 이미지 퍼블리시
        # --------------------------------------------
        try:
            ros_mask = bridge.cv2_to_imgmsg(thin_mask, encoding="mono8")
            pub_mask.publish(ros_mask)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))

        # 3D 차선 포인트 -> Marker 퍼블리시 (RViz 사용 시)
        publish_lanes_3d(points_3d, pub_lane_marker)

        # --------------------------------------------
        # 10) OpenCV를 통한 실시간 시각화
        #     - Depth map에 color map 적용
        #     - Lane + Path 오버레이
        # --------------------------------------------
        disp_depth = (depth_map * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_depth, cv2.COLORMAP_JET)

        # (A) 차선(흰색) 오버레이
        # thin_mask가 255인 픽셀만 흰색으로 표시
        lane_overlay_color = (255, 255, 255)  # 흰색
        lane_indices = np.where(thin_mask == 255)
        disp_color[lane_indices] = lane_overlay_color

        # (B) Path를 빨간색으로 표시
        project_path_to_image(disp_color, path_pts, fx, fy, cx, cy)

        # (C) 사다리꼴 ROI 폴리라인(노란색)
        cv2.polylines(disp_color, [trap_points], isClosed=True, color=(0, 255, 255), thickness=2)

        # (D) 조향각 정보 표시
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

            # 웹캠인 경우 저장
            if dataset.mode == 'stream' and save_img:
                end_time = time.time()
                real_duration = end_time - start_time if start_time else 0
                stem_name = PathlibPath(path_item).stem if path_item else 'webcam0'
                make_webcam_video(record_frames, save_dir, stem_name, real_duration)
            return

        # --------------------------------------------
        # 11) 결과 저장 로직
        # --------------------------------------------
        if save_img:
            # (A) 이미지
            if dataset.mode == 'image':
                save_path = str(save_dir / PathlibPath(path_item).name)
                sp = PathlibPath(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')
                cv2.imwrite(str(sp), disp_color)

            # (B) 비디오
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

            # (C) 웹캠 스트리밍
            else:
                record_frames.append(disp_color.copy())

    # 모든 데이터 처리 후 리소스 정리
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    if hasattr(dataset, 'cap') and dataset.cap:
        dataset.cap.release()
    cv2.destroyAllWindows()

    # 웹캠 녹화본 저장
    if dataset.mode == 'stream' and save_img:
        end_time = time.time()
        real_duration = end_time - start_time if start_time else 0
        make_webcam_video(record_frames, save_dir, 'webcam0', real_duration)

    rospy.loginfo("inference time : (%.4fs/frame)", inf_time.avg)
    rospy.loginfo("Done. (%.3fs total)", (time.time() - start_time if start_time else 0))

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
    pub_steer = rospy.Publisher('steering_angle', Float64, queue_size=1)

    # 메인 로직 호출
    detect_and_publish(opt, pub_mask, pub_path, pub_lane_marker, pub_steer)

    rospy.loginfo("[INFO] Pseudo-Lidar Lane node finished. spin() for keepalive.")
    rospy.spin()

if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
