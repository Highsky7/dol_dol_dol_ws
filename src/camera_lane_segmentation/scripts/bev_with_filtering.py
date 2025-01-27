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

# utils.py
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    lane_line_mask,  # YOLOPv2 (공식 슬라이스+업샘플)
    AverageMeter,
    LoadCamera,
    LoadImages,
    letterbox
)

"""
[수정 버전]
1) 원본 이미지 -> YOLOPv2 세그멘테이션 -> 이진화(binary_mask)
2) binary_mask를 (bev_params.npz) 기반으로 BEV 변환
3) 변환된 마스크에 대해 Thinning & 추가 알고리즘
4) 실시간 창 + 저장 + ROS 퍼블리시
(원본 컬러 영상도 BEV 변환하여 같이 확인 가능)

주의:
 - lane_line_mask()는 원래 (ch=2) seg출력 → [12:372] 슬라이스 → 2×업샘플 이므로,
   최종 마스크가 (720, 1280) 등일 수 있음
 - 따라서 warp 전에는 마스크 해상도에 맞춰서 transform 해야 하므로
   (원본 프레임 크기와 동일하다고 가정해야 한다)
 - 두 스크립트(이 코드 & bev_utils.py) 모두 해상도/리사이즈/Letterbox가
   일관되게 유지되어야, BEV 파라미터가 올바르게 적용됩니다.
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
    step3 = remove_small_components(step2, min_size=100)
    step4 = keep_top2_components(step3, min_area=150)

    # 추가: line_fit_filter
    #  가로선(각도 10~170도 내) + 잔차 5 이하
    step5 = line_fit_filter(step4, max_line_fit_error=5.0, min_angle_deg=20.0, max_angle_deg=160.0)

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
def final_filter(bev_mask):
    """
    1) morph_open -> morph_close
    2) remove_small_components
    3) keep_top2_components
    4) line_fit_filter(직선 형태 + 수직 방향만 유지)
    """
    f1 = morph_open(bev_mask, ksize=2)
    f2 = morph_close(f1, ksize=6)
    f3 = remove_small_components(f2, min_size=100)
    f4 = keep_top2_components(f3, min_area=50)

    # 추가: line_fit_filter
    #  가로선(각도 10~170도 내) + 잔차 5 이하
    f5 = line_fit_filter(f4, max_line_fit_error=5, min_angle_deg=15.0, max_angle_deg=165.0)

    return f5
# ---------------------------------------------------------------------------
# BEV 변환 예시 함수
# ---------------------------------------------------------------------------
def do_bev_transform(image, bev_param_file):
    """
    bev_params.npz (src_points, dst_points, warp_w, warp_h) 사용.
    """
    params = np.load(bev_param_file)
    src_points = params['src_points']   # (4,2)
    dst_points = params['dst_points']   # (4,2)
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev = cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)
    return bev

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='0',#,'/home/highsky/Videos/Webcam/차선직진영상.mp4'
                        help='source: 0(webcam) or path to video/image')
    parser.add_argument('--img-size', type=int, default=640,
                        help='YOLO 추론 해상도')
    parser.add_argument('--device', default='0',
                        help='cuda device: 0 or cpu')
    parser.add_argument('--lane-thres', type=float, default=0.5,
                        help='Threshold for lane segmentation (0.0~1.0)')
    parser.add_argument('--nosave', action='store_false',
                        help='if true => do NOT save images/videos')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_false',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='Skip N frames (0 for no skip)')
    parser.add_argument('--param-file', type=str,
                        default='bev_params.npz',
                        help='BEV 파라미터 (src_points, dst_points, warp_w, warp_h)')
    return parser

def make_webcam_video(record_frames, save_dir: Path, stem_name: str, real_duration: float):
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
    if not out.isOpened():
        rospy.logerr(f"[ERROR] 비디오 라이터를 열 수 없습니다: {save_path}")
        return

    for f in record_frames:
        out.write(f)

    out.release()
    rospy.loginfo("[INFO] 웹캠 결과 영상 저장 완료: %s", save_path)

def detect_and_publish(opt, pub_mask):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
    cudnn.benchmark = True

    bridge = CvBridge()

    source, weights = opt.source, opt.weights
    imgsz = opt.img_size
    lane_threshold = opt.lane_thres
    save_img = not opt.nosave
    bev_param_file = opt.param_file

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    vid_path = None
    vid_writer = None
    current_save_size = None

    # YOLO 모델 로드
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
        rospy.loginfo("[INFO] 웹캠(장치=%s) 열기", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
    else:
        rospy.loginfo("[INFO] 파일(이미지/동영상): %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    record_frames = []
    start_time = None

    # GPU warm-up
    if device.type != 'cpu':
        _ = model(torch.zeros(1,3,imgsz,imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    frame_skip = opt.frame_skip
    frame_counter = 0

    for path_item, net_input_img, im0s, vid_cap in dataset:
        # (A) 프레임 스킵
        if frame_skip > 0:
            if frame_counter % (frame_skip + 1) != 0:
                frame_counter += 1
                continue
            frame_counter += 1

        if dataset.mode == 'stream' and start_time is None:
            start_time = time.time()

        # (B) YOLO 추론: 원래 코드에서 bev_transform 전이었지만, 이제는 먼저 세그멘테이션
        # net_input_img: letterbox(...)된 (C,H,W) RGB
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

        # (C) 차선 세그멘테이션 -> 이진화
        ll_seg_mask = lane_line_mask(ll, threshold=lane_threshold)
        binary_mask = (ll_seg_mask > 0).astype(np.uint8) * 255
        # filtered_mask = advanced_filter_pipeline(binary_mask)
        # # 이 시점에서 binary_mask는 "원근 시점" (예: 720x1280 등) 크기
        # (D) 세선화(Thinning) + 추가 알고리즘
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if thin_mask is None or thin_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 -> bev_mask 사용")
            thin_mask = binary_mask
        # (E) 이제 이 binary_mask를 BEV 변환
        #     다만 warpPerspective()는 3채널도 1채널도 모두 처리 가능
        bev_mask = do_bev_transform(thin_mask, bev_param_file)
        bevfilter_mask = final_filter(bev_mask)
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if final_mask is None or final_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 -> bevfilter_mask 사용")
            final_mask = bevfilter_mask
        # bev_mask: (640x640) 가정
        # (참고) 더 강화된 필터링이 필요하면:





        # (F) 필요하다면 컬러영상도 BEV로 보고 싶다면
        #     do_bev_transform(im0s, bev_param_file)
        #     아래처럼 주석 해제:
        bev_im = do_bev_transform(im0s, bev_param_file)

        # (G) ROS 퍼블리시 (thin_mask는 mono8)
        try:
            ros_mask = bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
            pub_mask.publish(ros_mask)
        except CvBridgeError as e:
            rospy.logerr("[ERROR] CvBridge 변환 실패: %s", str(e))

        # (H) 화면 표시: 2개 창
        #  - [BEV Image]: 컬러
        #  - [Thin Mask]: 세선화된 마스크
        cv2.imshow("BEV Image", bev_im)
        cv2.imshow("Thin Mask", final_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.loginfo("[INFO] q -> 종료")
            break

        # (I) 결과 저장
        if save_img:
            thin_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            if dataset.mode == 'image':
                # 이미지
                save_path = str(save_dir / Path(path_item).name)
                sp = Path(save_path)
                if sp.suffix.lower() not in ['.jpg','.jpeg','.png','.bmp']:
                    sp = sp.with_suffix('.jpg')
                cv2.imwrite(str(sp), thin_bgr)
                rospy.loginfo(f"[INFO] 이미지 저장: {sp}")

            elif dataset.mode == 'video':
                # 비디오
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

                    wv,hv = thin_bgr.shape[1], thin_bgr.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (wv,hv))
                    if not vid_writer.isOpened():
                        rospy.logerr(f"[ERROR] 비디오 라이터 열 수 없음: {save_path}")
                        vid_writer = None
                    current_save_size = (wv,hv)
                    rospy.loginfo(f"[INFO] 비디오 저장 시작: {vid_path} (fps={fps}, size=({wv},{hv}))")

                if vid_writer is not None:
                    if (thin_bgr.shape[1], thin_bgr.shape[0]) != current_save_size:
                        rospy.logwarn("[WARNING] 비디오 프레임 크기 불일치 -> 리사이즈")
                        try:
                            thin_bgr = cv2.resize(thin_bgr, current_save_size, interpolation=cv2.INTER_LINEAR)
                        except cv2.error as e:
                            rospy.logerr(f"[ERROR] 리사이즈 실패: {e}")
                            continue
                    vid_writer.write(thin_bgr)

            elif dataset.mode == 'stream':
                record_frames.append(thin_bgr.copy())

    # end for

    # (마무리)
    if vid_writer is not None:
        vid_writer.release()
        rospy.loginfo(f"[INFO] 비디오 저장 완료: {vid_path}")

    if hasattr(dataset, 'cap') and dataset.cap:
        dataset.cap.release()

    cv2.destroyAllWindows()

    if dataset.mode == 'stream' and save_img and len(record_frames) > 0:
        end_time = time.time()
        real_duration = end_time - start_time if start_time else 0
        make_webcam_video(record_frames, save_dir, 'webcam0', real_duration)

    rospy.loginfo(f"[INFO] 추론 시간: 평균 {inf_time.avg:.4f}s/frame")
    rospy.loginfo(f"[INFO] Done. (%.3fs)", (time.time()-t0))

def ros_main():
    rospy.init_node('bev_lane_thinning_node', anonymous=True)

    parser = make_parser()
    opt, _ = parser.parse_known_args()

    pub_mask = rospy.Publisher('yolopv2/lane_mask', Image, queue_size=1)
    detect_and_publish(opt, pub_mask)

    rospy.loginfo("[INFO] bev_lane_thinning_node finished. spin()")
    rospy.spin()

if __name__=='__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
