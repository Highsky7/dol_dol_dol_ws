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
# --------------------------------------------------------------
#  내부에 lane_line_mask, LoadCamera, LoadImages 등 함수/클래스가 정의되어 있다고 가정.
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
ROS 노드 (BEV 변환 + Thinning):
 1) 카메라(또는 비디오) 프레임을 먼저 BEV(Birds-Eye View)로 변환
 2) YOLOPv2 차선 세그멘테이션 -> 이진화 (lane_line_mask)
 3) 세선화(Thinning)로 얇은 라인 추출
 4) 결과를 ROS 토픽(mono8)으로 퍼블리시, 또한 화면 표시 및 (옵션) 파일 저장

 - 본 예시 코드에서는 line_fit_filter, morph_close 등 추가 필터를 주석으로 남겨둠.
 - 원하시면 주석 해제 후 적절한 파라미터로 사용하실 수 있습니다.
 - 아래 BEV 예시는 단순히 직사영역을 warp하는 예시이므로, 실제 환경에 맞춰 src/dst 좌표를 수정해야 합니다.
"""

# ---------------------------------------------------------------------------
# (옵션) 모폴로지/라인 필터 예시 함수들 -- 필요 시 주석 해제 사용
# ---------------------------------------------------------------------------
def morph_open(binary_mask, ksize=3):
    """
    모폴로지 Opening을 통해 작은 노이즈 제거
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

def morph_close(binary_mask, ksize=5):
    """
    모폴로지 Closing을 통해 끊긴 라인을 이어붙이기
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=100):
    """
    연결 요소 분석으로, 영역 크기가 min_size 미만인 노이즈 제거
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=50):
    """
    연결 요소 중 면적이 큰 2개만 남겨 (좌/우 차선만 유지하려는 용도)
    """
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

def line_fit_filter(binary_mask,
                    max_line_fit_error=2.0,
                    min_angle_deg=70.0,
                    max_angle_deg=110.0):
    """
    연결 요소 각각에 대해 cv2.fitLine 적용:
     - 평균 잔차가 일정 이하인 '직선 형태'만 남김
     - 또한 각도가 70~110도 내에 있지 않으면 제거(수평선 등 제외)
    """
    h, w = binary_mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    out_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        comp_mask = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 5:
            continue

        ys, xs = np.where(comp_mask > 0)
        pts = np.column_stack((xs, ys)).astype(np.float32)
        if len(pts) < 2:
            continue

        line_param = cv2.fitLine(pts, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        vx, vy, x0, y0 = line_param.flatten()

        angle_deg = abs(degrees(atan2(vy, vx)))
        if angle_deg > 180:
            angle_deg -= 180

        # 각도 조건
        if not (min_angle_deg <= angle_deg <= max_angle_deg):
            continue

        # 평균 거리(잔차)
        norm_len = np.sqrt(vx*vx + vy*vy)
        if norm_len < 1e-12:
            continue
        vx_n, vy_n = vx / norm_len, vy / norm_len

        dist_sum = 0.0
        for (xx, yy) in pts:
            dx, dy = xx - x0, yy - y0
            perp_dist = abs(dx * (-vy_n) + dy * vx_n)
            dist_sum += perp_dist
        mean_dist = dist_sum / len(pts)

        # 조건 만족 시 라벨을 out_mask에 남김
        if mean_dist <= max_line_fit_error:
            out_mask[labels == i] = 255

    return out_mask

# ---------------------------------------------------------------------------
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
    #  가로선(각도 10~170도 내) + 잔차 5 이하
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
# ---------------------------------------------------------------------------
# BEV 변환 예시 함수
# ---------------------------------------------------------------------------
def do_bev_transform(image, param_file):
    """
    단순 Birds-Eye View 변환 (Homography)
    - src_points, dst_points는 예시값입니다.
    - 실제 환경에 맞춰서 조정해야 합니다.
    """

    # 예시 source 좌표 (왼 하단, 오른 하단, 왼 상단, 오른 상단)
    # 주행 영상에 맞춰 알맞게 지정해야 합니다.
    # 파라미터 로드
    params = np.load(param_file)
    src_points = params['src_points']   # (4,2)
    dst_points = params['dst_points']   # (4,2)
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])

    # Homography
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev = cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

    return bev

# ---------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='TorchScript YOLOPv2 모델 경로')
    parser.add_argument('--source', type=str,
                        default='0',#'/home/highsky/Videos/Webcam/bev변환용 영상.mp4',
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
    # 추가: BEV 파라미터 파일 (default=~/bev_params_3.npz)
    parser.add_argument('--param-file', type=str,
                        default='/home/highsky/My_project_work_ws/bev_params.npz',
                        help='bev_utils.py에서 저장한 npz 파라미터 파일 경로')
    return parser

def make_webcam_video(record_frames, save_dir: Path, stem_name: str, real_duration: float):
    """
    웹캠 모드 등 스트림 영상을 mp4로 저장
    """
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
    """
    메인 파이프라인:
     1) 프레임 획득 -> BEV 변환
     2) YOLOPv2 차선 세그멘테이션 -> 이진화
     3) 세선화(Thinning)
     4) ROS 퍼블리시 + 화면 표시 + (옵션) 결과 저장
    """
    # OpenCV 최적화 설정
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    # PyTorch cudnn 설정
    cudnn.benchmark = True

    # 파라미터 로드
    bev_param_file = opt.param_file

    bridge = CvBridge()

    source, weights = opt.source, opt.weights
    imgsz = opt.img_size
    lane_threshold = opt.lane_thres

    # --nosave 옵션에 따라 저장 여부 결정
    save_img = not opt.nosave

    # 결과 저장 폴더 구성
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    vid_path = None
    vid_writer = None
    current_save_size = None

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)   # TorchScript 모델 로드
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # 데이터셋 로드 (카메라 or 파일)
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

    # ---------------------------
    # 메인 루프
    # ---------------------------
    for path_item, img, im0s, vid_cap in dataset:
        # 프레임 스킵 로직
        if frame_skip > 0:
            if frame_counter % (frame_skip + 1) != 0:
                frame_counter += 1
                continue
            frame_counter += 1

        # 스트림 모드(웹캠)라면 시작시간 기록
        if dataset.mode == 'stream' and start_time is None:
            start_time = time.time()

        # ---------------------------
        # (A) BEV 변환 (원본 프레임 im0s -> bev_im)
        # ---------------------------
        # im0s: 원본 BGR 프레임
        bev_im = do_bev_transform(im0s, bev_param_file)
        # 이 bev_im을 YOLOPv2에 넣기 위해서 letterbox 등 전처리 필요

        # 1) letterbox+RGB+transpose -> pytorch용 img
        #    여기서는 dataset이 이미 letterbox 처리를 img로 해놨지만,
        #    원본 im0s가 아니라 bev_im을 사용하려면 직접 처리해야 함.
        #    (간단히 img와 동일한 파이프라인 적용)

        # letterbox는 LoadImages/LoadCamera에서 이미 했지만,
        # BEV후 해상도가 바뀌지 않아서, 일관성을 위해 다음과 같이 수행:
        # (반드시 dataset이 하는 letterbox와 동일한 로직이어야 함)
        # 그냥 "img" 대체용으로 bev_im을 사용하겠다 → 아래처럼 교체:
        #
        # "img"는 numpy(C,H,W,RGB) 형태이므로, bev_im(BGR,H,W) → letterbox → (C,H,W)
        #
        # 편의상, dataset이 준 "img" 대신, 아래 과정을 수동 수행:

        # size를 (imgsz, imgsz) 등으로 맞추고, BGR->RGB, transpose
        bev_resized = cv2.resize(bev_im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        bev_rgb = bev_resized[:, :, ::-1]  # BGR->RGB
        bev_tensor = bev_rgb.transpose(2, 0, 1)  # (H,W,C)->(C,H,W)
        bev_tensor = np.ascontiguousarray(bev_tensor)

        # ---------------------------
        # (B) YOLOPv2 추론
        # ---------------------------
        img_t = torch.from_numpy(bev_tensor).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            # YOLOPv2는 [det_out, seg_out], ll_out(차선) 이렇게 3개를 반환
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()

        # ---------------------------
        # (C) 차선 세그멘테이션 -> 이진화 -> 세선화
        # ---------------------------
        ll_seg_mask = lane_line_mask(ll, threshold=opt.lane_thres)
        binary_mask = (ll_seg_mask > 0).astype(np.uint8) * 255
        # filtered_mask = advanced_filter_pipeline(binary_mask)
        # 세선화(Thinning)
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        # if thin_mask is None or thin_mask.size == 0:
        #     rospy.logwarn("[WARNING] 세선화 결과 없음 -> binary_mask 사용")
        #     thin_mask = binary_mask

        # ---------------------------
        # (참고) 더 강화된 필터링이 필요하면:
        # final_mask_result = post_thinning_filter(thin_mask)
        # thin_mask = ximgproc.thinning(filtered_mask, ...)
        # post_thinning = line_fit_filter(thin_mask, ...)
        # ...
        # ---------------------------

        # ---------------------------
        # (D) 추론 시간 기록
        # ---------------------------
        inf_time.update(t2 - t1, img_t.size(0))

        # ---------------------------
        # (E) ROS 퍼블리시
        # ---------------------------
        try:
            ros_img = bridge.cv2_to_imgmsg(thin_mask, encoding="mono8")
            pub_mask.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr("[ERROR] CvBridge 변환 실패: %s", e)

        # ---------------------------
        # (F) 화면 표시
        # ---------------------------
        cv2.imshow("BEV Thinning Result", thin_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.loginfo("[INFO] q -> 종료")
            break

        # ---------------------------
        # (G) 결과 저장
        # ---------------------------
        if save_img:
            thin_bgr = cv2.cvtColor(thin_mask, cv2.COLOR_GRAY2BGR)
            if dataset.mode == 'image':
                # 이미지 모드인 경우
                save_path = str(save_dir / Path(path_item).name)
                sp = Path(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')
                cv2.imwrite(str(sp), thin_bgr)
                rospy.loginfo(f"[INFO] 이미지 저장: {sp}")

            elif dataset.mode == 'video':
                # 비디오 모드
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

                    wv, hv = thin_bgr.shape[1], thin_bgr.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (wv, hv))
                    if not vid_writer.isOpened():
                        rospy.logerr("[ERROR] 비디오 라이터 열 수 없음: %s", save_path)
                        vid_writer = None
                    current_save_size = (wv, hv)

                if vid_writer is not None:
                    if (thin_bgr.shape[1], thin_bgr.shape[0]) != current_save_size:
                        rospy.logwarn("[WARNING] 비디오 프레임 크기 불일치 -> 리사이즈")
                        try:
                            thin_bgr = cv2.resize(thin_bgr, current_save_size, interpolation=cv2.INTER_LINEAR)
                        except cv2.error as e:
                            rospy.logerr("[ERROR] 리사이즈 실패: %s", e)
                            continue
                    vid_writer.write(thin_bgr)

            elif dataset.mode == 'stream':
                # 스트리밍 모드 (웹캠)
                record_frames.append(thin_bgr.copy())

    # end for

    # ---------------------------
    # 자원 정리
    # ---------------------------
    if vid_writer is not None:
        vid_writer.release()
        rospy.loginfo(f"[INFO] 비디오 저장 완료: {vid_path}")

    if hasattr(dataset, 'cap') and dataset.cap:
        dataset.cap.release()
    cv2.destroyAllWindows()

    # 웹캠 모드 녹화 저장
    if dataset.mode == 'stream' and save_img and len(record_frames) > 0:
        end_time = time.time()
        real_duration = end_time - start_time if start_time else 0
        make_webcam_video(record_frames, save_dir, 'webcam0', real_duration)

    rospy.loginfo("[INFO] 추론 시간: 평균 %.4fs/frame", inf_time.avg)
    rospy.loginfo("[INFO] Done. (%.3fs)", (time.time() - t0))


def ros_main():
    """
    ROS 노드 초기화 및 실행
    """
    rospy.init_node('bev_line_thinning_node', anonymous=True)

    parser = make_parser()
    opt, _ = parser.parse_known_args()

    # 퍼블리셔: 최종 thin_mask
    pub_mask = rospy.Publisher('yolopv2/lane_mask', Image, queue_size=1)

    detect_and_publish(opt, pub_mask)

    rospy.loginfo("[INFO] bev_line_thinning_node finished. spin()")
    rospy.spin()


if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
