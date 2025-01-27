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
from pathlib import Path

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# YOLOPv2 유틸
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
ROS 노드:
 1) Lane Line 세그멘테이션 (전체 이미지)
 2) 이진화 마스크 + Thinning (두꺼운 선 -> 얇은 선)
 3) 결과를 ROS 토픽 (mono8) 퍼블리시
 4) 실시간 창 및 저장된 영상에서도 얇은 차선을 확인
"""

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='0',#'/home/highsky/Videos/Webcam/차선직진영상.mp4',
                        help='source: 0(webcam) or path to video/image')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or cpu')
    # 보수적 세그멘테이션을 위한 Threshold
    parser.add_argument('--lane-thres', type=float, default=0.8,
                        help='Threshold for lane segmentation mask (0.0 ~ 1.0). Higher => more conservative')

    # Detection용이지만, 우선은 남겨둠
    parser.add_argument('--conf-thres', type=float, default=0.3, help='(unused)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='(unused)')

    # --nosave: true면 저장하지 않음, false면 저장
    parser.add_argument('--nosave', action='store_false',
                        help='if true => do NOT save images/videos, else => save')

    parser.add_argument('--project', default='/home/highsky/My_project_work_ws/runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_false',
                        help='existing project/name ok, do not increment')

    # 프레임 스킵
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='Skip N frames per read (0 for no skip)')

    return parser

def make_webcam_video(record_frames, save_dir: Path, stem_name: str, real_duration: float):
    """웹캠 모드에서 누적된 프레임(thin_mask)을 mp4로 저장."""
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
                          (w, h))  # frame size=(w,h)

    if not out.isOpened():
        rospy.logerr(f"[ERROR] 비디오 라이터를 열 수 없습니다: {save_path}")
        return

    for f in record_frames:
        out.write(f)

    out.release()
    rospy.loginfo("[INFO] 웹캠 결과 영상 저장 완료: %s", save_path)

def detect_and_publish(opt, pub_mask):
    """
    Lane Line 세그멘테이션 + 얇게(Thinning) -> ROS 퍼블리시 + 영상 저장
    """
    # OpenCV 최적화
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    # PyTorch cuDNN 최적화
    cudnn.benchmark = True

    bridge = CvBridge()

    source, weights = opt.source, opt.weights
    imgsz = opt.img_size

    # --nosave: true -> 저장하지 않음
    # not opt.nosave == 저장해야 함
    save_img = not opt.nosave

    # 결과 저장 경로
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()

    vid_path = None
    vid_writer = None
    current_save_size = None  # 추가: 현재 저장 중인 비디오의 프레임 크기

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # 데이터 로드
    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치 ID=%s)에서 영상을 받습니다.", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
    else:
        rospy.loginfo("[INFO] 파일(이미지/동영상): %s 를 불러옵니다.", source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    record_frames = []
    start_time = None

    # GPU warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    frame_skip = opt.frame_skip
    frame_counter = 0

    # 메인 루프
    for path_item, img, im0s, vid_cap in dataset:
        # 프레임 스킵
        if frame_skip > 0:
            if frame_counter % (frame_skip + 1) != 0:
                frame_counter += 1
                continue
            frame_counter += 1

        if dataset.mode == 'stream' and start_time is None:
            start_time = time.time()

        # 텐서 변환
        img_t = torch.from_numpy(img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()

        # Lane Segmentation -> binary_mask
        ll_seg_mask = lane_line_mask(ll, threshold=opt.lane_thres)  # 보수적 threshold 적용
        binary_mask = (ll_seg_mask > 0).astype(np.uint8) * 255

        # 스켈레톤 알고리즘(Skeletonization) Thinning
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)

        # Thinning 결과가 유효한지 확인
        if thin_mask is None or thin_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과가 유효하지 않습니다.")
            thin_mask = binary_mask  # 대체

        # Thinning 결과 로그
        rospy.loginfo(f"[INFO] thin_mask shape: {thin_mask.shape}, dtype: {thin_mask.dtype}, unique values: {np.unique(thin_mask)}")

        inf_time.update(t2 - t1, img_t.size(0))

        # ROS 퍼블리시: (단일 채널)
        try:
            ros_mask = bridge.cv2_to_imgmsg(thin_mask, encoding="mono8")
            pub_mask.publish(ros_mask)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))

        # 실시간 화면 표시: 얇아진 차선
        cv2.imshow('YOLOPv2 Thin Mask (ROS)', thin_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.loginfo("[INFO] q 키 입력 -> 종료")
            break  # 루프 종료

        # ----------------------
        # 저장 로직 (얇은 선 영상)
        # ----------------------
        if save_img:
            # 3채널 변환
            thin_bgr = cv2.cvtColor(thin_mask, cv2.COLOR_GRAY2BGR)

            # 저장 경로 결정
            if dataset.mode == 'image':
                save_path = str(save_dir / Path(path_item).name)
            else:
                save_path = str(save_dir / Path(path_item).stem) + '.mp4'

            if dataset.mode == 'image':
                sp = Path(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')

                # 이진화 + Thinning 결과를 이미지로 저장
                cv2.imwrite(str(sp), thin_bgr)
                rospy.loginfo(f"[INFO] 이미지 저장 완료: {sp}")

            elif dataset.mode == 'video':
                # VideoWriter 초기화
                if vid_path != save_path:
                    vid_path = save_path
                    if vid_writer is not None:
                        vid_writer.release()

                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        if fps == 0:
                            fps = 30  # 기본 FPS
                    else:
                        fps = 30

                    wv, hv = thin_bgr.shape[1], thin_bgr.shape[0]
                    rospy.loginfo(f"[INFO] 비디오 저장 시작: {vid_path} (FPS={fps}, size=({wv},{hv}))")
                    vid_writer = cv2.VideoWriter(
                        vid_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (wv, hv)
                    )

                    if not vid_writer.isOpened():
                        rospy.logerr(f"[ERROR] 비디오 라이터를 열 수 없습니다: {vid_path}")
                        vid_writer = None  # 재시도 방지

                    current_save_size = (wv, hv)

                # 얇아진 마스크 영상 저장
                if vid_writer is not None:
                    if (thin_bgr.shape[1], thin_bgr.shape[0]) != current_save_size:
                        rospy.logwarn("[WARNING] 프레임 크기가 비디오 라이터와 일치하지 않습니다. 프레임을 조정합니다.")
                        if all([s > 0 for s in current_save_size]):
                            try:
                                thin_bgr = cv2.resize(thin_bgr, current_save_size, interpolation=cv2.INTER_LINEAR)
                            except cv2.error as e:
                                rospy.logerr(f"[ERROR] 프레임 리사이즈 실패: {e}")
                                continue
                        else:
                            rospy.logerr("[ERROR] 유효하지 않은 프레임 크기입니다. 저장을 건너뜁니다.")
                            continue
                    vid_writer.write(thin_bgr)
                else:
                    rospy.logerr("[ERROR] 비디오 라이터가 열려 있지 않습니다. 프레임을 저장할 수 없습니다.")

            elif dataset.mode == 'stream':
                # 웹캠 스트림 저장용 프레임 누적
                record_frames.append(thin_bgr.copy())

    # end for

    # 자원 정리
    if vid_writer is not None:
        vid_writer.release()
        rospy.loginfo(f"[INFO] 비디오 저장 완료: {vid_path}")
    if hasattr(dataset, 'cap') and dataset.cap:
        dataset.cap.release()
    cv2.destroyAllWindows()

    # 웹캠 모드 + 저장 옵션이면 mp4로 저장
    if dataset.mode == 'stream' and save_img and len(record_frames) > 0:
        end_time = time.time()
        real_duration = end_time - start_time if start_time else 0
        stem_name = 'webcam0'  # 고정
        make_webcam_video(record_frames, save_dir, stem_name, real_duration)

    rospy.loginfo("inf : (%.4fs/frame)", inf_time.avg)
    rospy.loginfo("Done. (%.3fs)", (time.time() - t0))

def ros_main():
    rospy.init_node('yolopv2_laneline_node', anonymous=True)

    parser = make_parser()
    opt, _ = parser.parse_known_args()

    # 새 퍼블리셔: 이진화 + Thinning 마스크
    pub_mask = rospy.Publisher('yolopv2/lane_mask', Image, queue_size=1)

    detect_and_publish(opt, pub_mask)

    rospy.loginfo("[INFO] YOLOPv2 LaneLine node finished. spin() for keepalive.")
    rospy.spin()

if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
