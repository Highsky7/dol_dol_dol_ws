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
from math import atan2, degrees
from pathlib import Path

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# == utils.py (동일 워크스페이스 내에 있는 utils/utils.py) ==
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    lane_line_mask,
    AverageMeter,
    LoadCamera,
    LoadImages
)

"""
ROS 노드: Line Fitting + Orientation 기반 필터를 추가한 강화판
 1) YOLOPv2 차선 세그멘테이션 -> 이진화
 2) 모폴로지 (Open/Close) + 작은 라벨 제거 + 상위 2개 라인 유지
 3) shape_filter(Aspect Ratio) → (옵션)
 4) [추가] line_fit_filter: 실제 '직선'에 잘 부합하는지 (fitLine으로 검사)
    + '가로선' 각도 배제
 5) 세선화(Thinning) 후, 원하는 방향의 2개 차선만 최종 유지
 6) ROS 퍼블리시(mono8), 화면 표시, 파일 저장
"""

# ----------------------------------------------------------------------------
# 1) 기본 모폴로지 + 연결 요소 함수
# ----------------------------------------------------------------------------
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
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 2:
        return binary_mask

    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comps.append((i, area))
    comps.sort(key=lambda x: x[1], reverse=True)

    keep_indices = []
    for i, area in comps[:2]:
        if area >= min_area:
            keep_indices.append(i)

    cleaned = np.zeros_like(binary_mask)
    for idx in keep_indices:
        cleaned[labels == idx] = 255

    return cleaned

# ----------------------------------------------------------------------------
# 2) [새로 추가] line_fit_filter:
#    연결 요소마다 cv2.fitLine() 적용 -> 직선 잔차 검증 + 각도 제한(가로선 제거)
# ----------------------------------------------------------------------------
def line_fit_filter(binary_mask,
                    max_line_fit_error=2.0,
                    min_angle_deg=70.0,
                    max_angle_deg=110.0):
    """
    - max_line_fit_error: fitLine으로 얻은 선에 대한 평균 잔차가 이 값 이하일 때만 '진짜 라인'이라고 판단.
      (값이 너무 크면, 곡선이나 기이한 형태)
    - min_angle_deg, max_angle_deg: 이 범위 안에 드는 선 각도만 유지.
      예: 70~110도 => 수직(90도) 근처의 차선만 남김.
      (가로선(0°/180° 근처)은 제거)
    """
    h, w = binary_mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    out_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        comp_mask = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 5:
            continue

        # 픽셀 좌표 수집
        ys, xs = np.where(comp_mask > 0)
        # (x, y) float32로 변환
        pts = np.column_stack((xs, ys)).astype(np.float32)

        if len(pts) < 2:
            continue

        # 1) 선 피팅
        line_param = cv2.fitLine(pts, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        vx, vy, x0, y0 = line_param.flatten()

        # 2) 각도 계산 (atan2(y, x))
        angle_deg = abs(degrees(atan2(vy, vx)))  # 0~180
        if angle_deg > 180:
            angle_deg -= 180
        # angle_deg in [0,180], 수직에 가까운 90°가 차선 방향이라고 가정
        # => 70~110도 등 원하는 범위를 설정
        if not (min_angle_deg <= angle_deg <= max_angle_deg):
            continue  # 각도 조건 불만족 → 제거

        # 3) 잔차(Residual) 계산
        #   직선 방정식: 점 p가 line에 얼마나 근접?
        #   (vx, vy)는 단위벡터 아님 → fitLine 반환은 단위벡터가 맞지만, 혹시 몰라 normalize
        norm_len = np.sqrt(vx*vx + vy*vy)
        if norm_len < 1e-12:
            continue
        vx_n, vy_n = vx / norm_len, vy / norm_len

        # 직선 기준점 (x0, y0)
        # 각 점 p=(xs[j], ys[j])에서 line까지의 수직거리 = |(p - p0) x d|
        # (x-product) = dot((p-p0), dir_perp)
        # 여기서는 간단히 => 거리 = |(dx, dy) dot (-vy_n, vx_n)|
        #   dx = x - x0, dy = y - y0
        #   direction perpendicular to line: (-vy_n, vx_n)
        # 평균 거리 계산
        dist_sum = 0.0
        for (xx, yy) in pts:
            dx, dy = xx - x0, yy - y0
            perp_dist = abs(dx * (-vy_n) + dy * vx_n)
            dist_sum += perp_dist
        mean_dist = dist_sum / len(pts)

        # mean_dist가 max_line_fit_error 이하인 경우만 남긴다
        if mean_dist <= max_line_fit_error:
            # 통과 → out_mask에 추가
            out_mask[labels == i] = 255

    return out_mask

# ----------------------------------------------------------------------------
# 3) 전반적 필터 파이프라인(모폴로지+연결요소) & 세선화 이후
# ----------------------------------------------------------------------------
def advanced_filter_pipeline(binary_mask):
    """
    1) morph_open -> morph_close
    2) remove_small_components
    3) keep_top2_components
    4) line_fit_filter(직선 형태 + 수직 방향만 유지)
    """
    step1 = morph_open(binary_mask, ksize=3)
    step2 = morph_close(step1, ksize=5)
    step3 = remove_small_components(step2, min_size=200)
    step4 = keep_top2_components(step3, min_area=150)

    # 추가: line_fit_filter
    #  가로선(각도 70~110도 밖)은 제거, 잔차 2 이하
    step5 = line_fit_filter(step4, max_line_fit_error=5.0, min_angle_deg=10.0, max_angle_deg=170.0)

    return step5

def post_thinning_filter(thin_mask):
    """
    세선화 후 추가 필터:
      1) remove_small_components
      2) keep_top2_components
      3) line_fit_filter (한 번 더)
    """
    s1 = remove_small_components(thin_mask, min_size=50)
    s2 = keep_top2_components(s1, min_area=50)
    # 세선화가 된 후라도, 수평/비직선 부분이 섞여 있을 수 있으므로 다시 한 번 fit
    # s3 = line_fit_filter(s2, max_line_fit_error=5.0, min_angle_deg=30.0, max_angle_deg=150.0)
    return s2

# ----------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='TorchScript YOLOPv2 모델 경로')
    parser.add_argument('--source', type=str,
                        default='0',#'2',
                        help='영상 파일 경로 or 0(웹캠)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='추론 이미지 크기')
    parser.add_argument('--device', default='0',
                        help='cuda device(예:0) 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.5,
                        help='차선 세그멘테이션 임계값(0~1)')
    parser.add_argument('--nosave', action='store_false',
                        help='true면 결과영상 저장 안함, 기본 false=저장O')
    parser.add_argument('--project', default='/home/highsky/My_project_work_ws/runs/detect',
                        help='결과 저장 폴더')
    parser.add_argument('--name', default='exp',
                        help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_false',
                        help='기존 폴더 있어도 ok, 덮어쓰기')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='프레임 스킵(성능 문제 시 사용)')
    return parser

def make_webcam_video(record_frames, save_dir: Path, stem_name: str, real_duration: float):
    if len(record_frames) == 0:
        rospy.loginfo("[INFO] 저장할 스트림 프레임 없음.")
        return

    num_frames = len(record_frames)
    if real_duration <= 0:
        real_duration = 1e-6
    real_fps = num_frames / real_duration

    save_path = str(save_dir / f"{stem_name}_webcam.mp4")
    h, w = record_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, real_fps, (w, h))
    if not out.isOpened():
        rospy.logerr("[ERROR] 비디오 라이터 열 수 없음: %s", save_path)
        return

    for f in record_frames:
        out.write(f)
    out.release()
    rospy.loginfo("[INFO] 웹캠 결과영상 저장 완료: %s", save_path)

def detect_and_publish(opt, pub_mask):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
    cudnn.benchmark = True

    bridge = CvBridge()

    source, weights = opt.source, opt.weights
    imgsz = opt.img_size
    lane_threshold = opt.lane_thres
    save_img = not opt.nosave

    # 결과 저장 폴더
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    vid_path = None
    vid_writer = None
    current_save_size = None

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # 데이터셋 로드
    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치ID=%s) 열기", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
    else:
        rospy.loginfo("[INFO] 파일 로딩: %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    record_frames = []
    start_time = None

    # GPU warm-up
    if device.type != 'cpu':
        _ = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    frame_skip = opt.frame_skip
    frame_counter = 0

    for path_item, img, im0s, vid_cap in dataset:
        if frame_skip > 0:
            if frame_counter % (frame_skip + 1) != 0:
                frame_counter += 1
                continue
            frame_counter += 1

        if dataset.mode == 'stream' and start_time is None:
            start_time = time.time()

        # 텐서 변환
        img_t = torch.from_numpy(im0s).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        # 추론
        t1 = time_synchronized()
        with torch.no_grad():
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()

        # (슬라이싱) 필요시:
        # ll = ll[:, :, 12:372, :]

        # 1) 이진화
        ll_seg_mask = lane_line_mask(ll, threshold=lane_threshold)
        binary_mask = (ll_seg_mask > 0).astype(np.uint8) * 255

        # 2) 고급 필터 파이프라인(모폴로지 + line_fit)
        # filtered_mask = advanced_filter_pipeline(binary_mask)

        # 3) 세선화(Thinning)
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        # if thin_mask is None or thin_mask.size == 0:
        #     rospy.logwarn("[WARNING] 세선화 결과 없음 -> filtered_mask 대체")
        #     thin_mask = filtered_mask

        # 4) 세선화 후 추가 필터(라인 피팅)
        # final_mask_result = post_thinning_filter(thin_mask)

        inf_time.update(t2 - t1, img_t.size(0))

        # ROS 퍼블리시
        try:
            ros_img = bridge.cv2_to_imgmsg(thin_mask, encoding="mono8")
            pub_mask.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

        # 화면 표시
        cv2.imshow('Line-Orientation Filter + Thinning', thin_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.loginfo("[INFO] q -> 종료")
            break

        # 결과 저장
        if save_img:
            color_mask = cv2.cvtColor(thin_mask, cv2.COLOR_GRAY2BGR)
            if dataset.mode == 'image':
                save_path = str(save_dir / Path(path_item).name)
                sp = Path(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')
                cv2.imwrite(str(sp), color_mask)
                rospy.loginfo(f"[INFO] 이미지 저장: {sp}")

            elif dataset.mode == 'video':
                save_path = str(save_dir / Path(path_item).stem) + '.mp4'
                if vid_path != save_path:
                    vid_path = save_path
                    if vid_writer is not None:
                        vid_writer.release()

                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30
                    else:
                        fps = 30

                    wv, hv = color_mask.shape[1], color_mask.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (wv, hv))
                    if not vid_writer.isOpened():
                        rospy.logerr("[ERROR] 비디오 라이터 열 수 없음: %s", save_path)
                        vid_writer = None
                    current_save_size = (wv, hv)

                if vid_writer is not None:
                    if (color_mask.shape[1], color_mask.shape[0]) != current_save_size:
                        rospy.logwarn("[WARNING] 프레임 크기 불일치 -> 리사이즈")
                        try:
                            color_mask = cv2.resize(color_mask, current_save_size, interpolation=cv2.INTER_LINEAR)
                        except cv2.error as e:
                            rospy.logerr("[ERROR] 리사이즈 실패: %s", e)
                            continue
                    vid_writer.write(color_mask)

            elif dataset.mode == 'stream':
                record_frames.append(color_mask.copy())

    # end for

    if vid_writer is not None:
        vid_writer.release()
        rospy.loginfo("[INFO] 비디오 저장 완료: %s", vid_path)

    if hasattr(dataset, 'cap') and dataset.cap:
        dataset.cap.release()
    cv2.destroyAllWindows()

    if dataset.mode == 'stream' and save_img and len(record_frames) > 0:
        end_time = time.time()
        real_duration = end_time - start_time if start_time else 0
        make_webcam_video(record_frames, save_dir, 'webcam0', real_duration)

    rospy.loginfo("[INFO] 추론 시간: 평균 %.4fs/frame", inf_time.avg)
    rospy.loginfo("[INFO] Done. (%.3fs)", (time.time() - t0))

def ros_main():
    rospy.init_node('line_orientation_filter_thinning_node', anonymous=True)

    parser = make_parser()
    opt, _ = parser.parse_known_args()

    pub_mask = rospy.Publisher('yolopv2/lane_mask', Image, queue_size=1)
    detect_and_publish(opt, pub_mask)

    rospy.loginfo("[INFO] line_orientation_filter_thinning_node finished. spin()")
    rospy.spin()

if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
