#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

# cv2.ximgproc 모듈 임포트 시도 (세선화용)
try:
    import cv2.ximgproc
    THINNING_AVAILABLE = True
except ImportError:
    THINNING_AVAILABLE = False
    print("[WARN] cv2.ximgproc module not found. Thinning will be skipped.")
    print("[WARN] Try: pip install opencv-contrib-python (after uninstalling opencv-python if needed)")

# utils.utils에서 필요한 함수를 임포트합니다.
from utils.utils import (
    time_synchronized, select_device, lane_line_mask,
    AverageMeter, LoadCamera, LoadImages, letterbox, apply_clahe,
)

def make_parser():
    parser = argparse.ArgumentParser(description="YOLOPv2 Lane Segmentation with Canny Edge & Thinning Visualization")
    parser.add_argument('--weights', type=str, default='./yolopv2.pt', help='Path to model.pt file')
    parser.add_argument('--source', type=str,
                        # default='0',
                        default='/home/highsky/Videos/Webcam/좌회전.mp4', # 사용자의 기본 경로 유지
                        help='Source: "0" for webcam, or path to video/image file')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO inference resolution (pixels)')
    parser.add_argument('--device', default='0', help='CUDA device, e.g., "0" or "cpu"')
    parser.add_argument('--lane-thres', type=float, default=0.3, help='Lane segmentation threshold')
    parser.add_argument('--frame-skip', type=int, default=0, help='Process 1 frame out of N+1 (0 means no skip)')
    parser.add_argument('--canny-low', type=int, default=100, help='Canny edge low threshold')
    parser.add_argument('--canny-high', type=int, default=200, help='Canny edge high threshold')
    return parser

def run_yolop_visualization(opt):
    cv2.setUseOptimized(True)
    cudnn.benchmark = True

    source = opt.source
    weights = opt.weights
    imgsz = opt.img_size
    lane_threshold = opt.lane_thres
    canny_low_thresh = opt.canny_low
    canny_high_thresh = opt.canny_high
    
    print(f"[INFO] Loading YOLOPv2 model from: {weights}")
    inf_time = AverageMeter()
    try:
        model = torch.jit.load(weights)
    except Exception as e:
        print(f"[ERROR] Failed to load model {weights}. Error: {e}")
        return

    device = select_device(opt.device)
    half = (device.type != 'cpu')

    model = model.to(device)
    if half:
        model.half()
    model.eval()
    print(f"[INFO] Model loaded successfully. Device: {device}, Half precision: {half}")

    print(f"[INFO] Setting up data loader for source: {source}")
    if source.isdigit():
        dataset = LoadCamera(source, img_size=imgsz, stride=32)
        print(f"[INFO] Using webcam: {source}")
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=32)
        print(f"[INFO] Using source file: {source}")

    frame_count_processed = 0
    frame_count_raw = 0

    print("[INFO] Starting YOLOPv2 lane segmentation visualization loop...")
    for path, img_resized_letterbox, im0s_orig, vid_cap in dataset:
        frame_count_raw += 1
        if opt.frame_skip > 0 and (frame_count_raw -1) % (opt.frame_skip + 1) != 0:
            continue
        
        frame_count_processed += 1
        im0s_h, im0s_w = im0s_orig.shape[:2]

        # --- 모델 입력 형식 변환 ---
        net_input_img = img_resized_letterbox
        net_input_img = np.ascontiguousarray(net_input_img)
        img_t = torch.from_numpy(net_input_img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        # --- YOLOPv2 추론 ---
        t1 = time_synchronized()
        with torch.no_grad():
            outputs = model(img_t)
            if isinstance(outputs, (list, tuple)):
                if len(outputs) > 2: ll_seg_out = outputs[2] 
                elif len(outputs) == 2: ll_seg_out = outputs[1]
                elif len(outputs) == 1: ll_seg_out = outputs[0]
                else: print("[ERROR] Unexpected model output structure."); continue
            else: ll_seg_out = outputs
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        # --- 차선 마스크 생성 및 리사이즈 ---
        binary_lane_mask_full_res = lane_line_mask(ll_seg_out, lane_threshold)
        if binary_lane_mask_full_res.shape[0] != im0s_h or binary_lane_mask_full_res.shape[1] != im0s_w:
            binary_lane_mask_full_res = cv2.resize(binary_lane_mask_full_res, (im0s_w, im0s_h), interpolation=cv2.INTER_NEAREST)

        # --- Canny Edge 검출 적용 ---
        lane_edges = cv2.Canny(binary_lane_mask_full_res, canny_low_thresh, canny_high_thresh)
        
        # --- 세선화(Thinning) 적용 ---
        thinned_lane_mask = None
        if THINNING_AVAILABLE:
            # cv2.ximgproc.thinning은 입력으로 0과 255 사이의 값을 가지는 uint8 타입의 이진 이미지를 기대합니다.
            # binary_lane_mask_full_res는 이미 이 형태입니다.
            # THINNING_GUOHALL 또는 THINNING_ZHANGSUEN 사용 가능
            try:
                thinned_lane_mask = cv2.ximgproc.thinning(binary_lane_mask_full_res, thinningType=cv2.ximgproc.THINNING_GUOHALL)
            except Exception as e:
                print(f"[ERROR] Error during thinning: {e}")
                # 에러 발생 시 다음 프레임 처리를 위해 thinned_lane_mask를 None으로 유지
                thinned_lane_mask = np.zeros_like(binary_lane_mask_full_res) # 또는 빈 이미지 표시


        # --- 시각화 ---
        cv2.imshow("Original Image", im0s_orig)
        cv2.imshow("YOLOPv2 Lane Segmentation Mask", binary_lane_mask_full_res)
        cv2.imshow("Canny Edges on Lane Mask", lane_edges) 
        
        if thinned_lane_mask is not None:
            cv2.imshow("Thinned Lane Mask (Guo-Hall)", thinned_lane_mask)
        
        lane_mask_colored_bgr = cv2.cvtColor(binary_lane_mask_full_res, cv2.COLOR_GRAY2BGR)
        lane_mask_colored_bgr[np.where((lane_mask_colored_bgr == [255,255,255]).all(axis=2))] = [0,255,0]
        overlay_image = cv2.addWeighted(im0s_orig, 0.7, lane_mask_colored_bgr, 0.3, 0)
        cv2.imshow("YOLOPv2 Lane Overlay", overlay_image)
        
        # --- 키 입력 처리 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User pressed 'q', exiting...")
            break
        elif key == ord('p'):
            print("Paused. Press 'p' again to resume or 'q' to quit.")
            while True:
                key_pause = cv2.waitKey(0) & 0xFF
                if key_pause == ord('p'): print("Resumed."); break
                elif key_pause == ord('q'):
                    print("User pressed 'q' during pause, exiting...")
                    cv2.destroyAllWindows()
                    if hasattr(dataset, 'release'): dataset.release()
                    print(f"[INFO] Visualization stopped by user. Average inference time: {inf_time.avg:.4f}s/frame")
                    return

    # --- 정리 ---
    if hasattr(dataset, 'release'): dataset.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Visualization completed. Processed {frame_count_processed} frames.")
    if inf_time.avg > 0: print(f"Average inference time: {inf_time.avg:.4f}s/frame ({1/inf_time.avg:.2f} FPS)")
    else: print(f"Average inference time: {inf_time.avg:.4f}s/frame (FPS not calculable)")

if __name__ == '__main__':
    parser = make_parser()
    opt = parser.parse_args()
    
    print(f"Run options: {vars(opt)}")
    
    try:
        run_yolop_visualization(opt)
    except Exception as e:
        print(f"[FATAL ERROR] An unhandled exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("Program terminated.")