#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

# =================================================================
# ★★★ 핵심 변경사항 1: 원본 utils 파일에서 LoadImages 클래스를 직접 임포트 ★★★
# =================================================================
from utils.utils import LoadImages
# =================================================================


def do_bev_transform(image, bev_param_file):
    """
    입력된 이미지에 대해 BEV(Bird's-Eye-View) 변환을 수행합니다.
    (이 함수는 변경되지 않았습니다.)
    """
    if not Path(bev_param_file).exists():
        print(f"[오류] BEV 파라미터 파일을 찾을 수 없습니다: {bev_param_file}")
        sys.exit(1)

    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_image = cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)
    
    return bev_image

def make_parser():
    """
    스크립트 실행을 위한 인자(argument)를 파싱하는 함수입니다.
    (img-size 인자를 추가하여 원본 코드와 통일)
    """
    parser = argparse.ArgumentParser(description="영상 파일을 BEV(Bird's-Eye-View) 영상으로 변환하고 저장하는 스크립트")
    parser.add_argument('--source', type=str,
                        default='/home/highsky/Downloads/2025-07-09-161352.mp4',
                        help='변환할 원본 영상 파일 경로. 예: /path/to/video.mp4')
    parser.add_argument('--img-size', type=int, default=640, help='처리할 이미지 해상도 (LoadImages 클래스에 전달)')
    parser.add_argument('--param-file', type=str, default='./bev_params_y_5.npz', help='BEV 파라미터 파일 경로. 예: ./bev_params_1.npz')
    parser.add_argument('--nosave', action='store_true', help='결과 영상 저장 안 함')
    parser.add_argument('--output-dir', type=str, default='runs/bev_output', help='결과 영상이 저장될 폴더')
    return parser

def bev_transform_and_save(opt):
    """
    메인 로직: LoadImages를 사용하여 비디오를 읽고, 각 프레임을 변환 후 저장합니다.
    """
    source_path = Path(opt.source)
    if not source_path.is_file():
        print(f"[오류] 영상 파일을 찾을 수 없습니다: {source_path}")
        return

    # =================================================================
    # ★★★ 핵심 변경사항 2: cv2.VideoCapture 대신 LoadImages 클래스 사용 ★★★
    # =================================================================
    dataset = LoadImages(opt.source, img_size=opt.img_size)
    # =================================================================

    writer = None
    output_path = "" # writer 스코프 문제 해결을 위해 외부 선언

    # --nosave 옵션이 아닐 경우, 저장 경로 설정
    if not opt.nosave:
        output_dir = Path(opt.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"bev_{source_path.name}"
        print("=====================================================")
        print(f"  영상 변환을 시작합니다 (저장 모드)")
        print(f"  - 원본 영상: {opt.source}")
        print(f"  - 처리 해상도: {opt.img_size}x{opt.img_size} (비율 유지, LoadImages 사용)")
        print(f"  - BEV 파라미터: {opt.param_file}")
        print(f"  - 저장될 경로: {output_path}")
        print("=====================================================")
    else:
        print("=====================================================")
        print(f"  영상 변환을 시작합니다 (실시간 보기 모드, 저장 안 함)")
        print("=====================================================")

    params = np.load(opt.param_file)
    output_w = int(params['warp_w'])
    output_h = int(params['warp_h'])

    # =================================================================
    # ★★★ 핵심 변경사항 3: LoadImages에 최적화된 루프 구조 사용 ★★★
    # =================================================================
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        
        # 첫 프레임에서 VideoWriter 초기화
        if not opt.nosave and writer is None:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            # =================================================================
            # ★★★ H.264 코덱으로 변경 ★★★
            # 'mp4v' 대신 'avc1' 또는 'h264' 사용. 'avc1'이 mp4와 호환성이 더 좋음.
            # =================================================================
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 코덱
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
        
        # `im0s`가 이미 비율 유지 리사이즈가 완료된 프레임임
        input_frame = im0s

        # BEV 변환 수행
        bev_frame = do_bev_transform(input_frame, opt.param_file)

        if writer is not None:
            writer.write(bev_frame)

        # 실시간 결과 보기
        cv2.imshow('Input (from LoadImages)', input_frame)
        cv2.imshow('BEV Transformed Video', bev_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 진행률 표시
        progress = ((frame_idx + 1) / len(dataset)) * 100
        print(f"\r  진행률: {frame_idx + 1}/{len(dataset)} ({progress:.2f}%)", end="")
    
    # 자원 해제
    if vid_cap:
        vid_cap.release()
    if writer is not None:
        writer.release()
        print("\n\n[완료] H.264 코덱으로 BEV 영상 변환 및 저장이 완료되었습니다.")
        print(f"결과물은 '{output_path}'에서 확인하실 수 있습니다.")
    else:
        print("\n\n[완료] 영상 처리가 종료되었습니다.")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    
    bev_transform_and_save(args)