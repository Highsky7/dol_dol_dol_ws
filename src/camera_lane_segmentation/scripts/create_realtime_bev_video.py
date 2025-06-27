#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import time

# =================================================================
# (주의: 이 스크립트가 작동하려면 utils.utils 모듈에 LoadCamera 클래스가 정의되어 있어야 합니다.)
# =================================================================
from utils.utils import LoadCamera
# =================================================================


def do_bev_transform(image, bev_param_file):
    """
    입력된 이미지에 대해 BEV(Bird's-Eye-View) 변환을 수행합니다.
    (이 함수는 원본 코드와 동일하며 변경되지 않았습니다.)
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
    """
    parser = argparse.ArgumentParser(description="실시간 카메라 영상을 BEV(Bird's-Eye-View)로 변환하고 H.264 코덱으로 녹화하는 스크립트")
    parser.add_argument('--source', type=str,
                        default='2',
                        help='카메라 인덱스. 일반적으로 내장 카메라는 "0", 외부 카메라는 "1" 등. 예: 0')
    parser.add_argument('--img-size', type=int, default=640, help='처리할 이미지 해상도 (LoadCamera 클래스에 전달)')
    parser.add_argument('--param-file', type=str, default='/home/highsky/dol_dol_dol_ws/bev_params_3.npz', help='BEV 파라미터 파일 경로. 예: ./bev_params_1.npz')
    parser.add_argument('--output-dir', type=str, default='runs/bev_output', help='결과 영상이 저장될 폴더')
    return parser

def bev_transform_and_save_realtime(opt):
    """
    메인 로직: LoadCamera를 사용하여 실시간 영상을 읽고, 각 프레임을 변환 후 H.264 코덱으로 녹화합니다.
    'q' 키를 누르면 녹화가 중단되고 프로그램이 종료됩니다.
    """
    try:
        dataset = LoadCamera(opt.source, img_size=opt.img_size)
    except Exception as e:
        print(f"[오류] 카메라를 열 수 없습니다: {opt.source}")
        print(f"  - 세부 정보: {e}")
        return

    writer = None
    
    # 저장 경로 설정
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"realtime_bev_output_{timestamp}_h264.mp4"
    
    print("=====================================================")
    print(f"  실시간 BEV 변환 및 H.264 녹화를 시작합니다")
    print(f"  - 입력 카메라: {opt.source}")
    print(f"  - 처리 해상도: {opt.img_size} (LoadCamera 사용)")
    print(f"  - BEV 파라미터: {opt.param_file}")
    print(f"  - 저장될 경로: {output_path}")
    print("\n  결과 창에서 'q' 키를 누르면 녹화가 중단되고 종료됩니다.")
    print("=====================================================")

    params = np.load(opt.param_file)
    output_w = int(params['warp_w'])
    output_h = int(params['warp_h'])

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        
        # 첫 프레임에서 VideoWriter 초기화
        if writer is None:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                print("[경고] 카메라에서 FPS를 얻을 수 없어 30으로 설정합니다.")
                fps = 30
            
            # =================================================================
            # ★★★ 핵심 변경사항: 'mp4v' 대신 H.264 코덱('H264' 또는 'avc1') 사용 ★★★
            # =================================================================
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 기존 코드
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            # 또는 fourcc = cv2.VideoWriter_fourcc(*'avc1')
            # =================================================================
            
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))

            # ★★★ 추가된 부분: VideoWriter가 H.264 코덱으로 성공적으로 열렸는지 확인 ★★★
            if not writer.isOpened():
                print("\n[오류] H.264 코덱으로 비디오 파일을 생성할 수 없습니다.")
                print("  - 시스템에 H.264 코덱이 설치되어 있는지 확인해주세요.")
                print("  - 또는 다른 FourCC 코드(예: 'avc1', 'X264', 'mp4v')를 시도해보세요.")
                break # 루프 중단

        input_frame = im0s
        bev_frame = do_bev_transform(input_frame, opt.param_file)

        if writer is not None and writer.isOpened():
            writer.write(bev_frame)

        cv2.imshow('Real-time Input (from LoadCamera)', input_frame)
        cv2.imshow('BEV Transformed Video (Recording with H.264...)', bev_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' 키가 입력되어 녹화를 중단합니다.")
            break
    
    # 자원 해제
    if isinstance(dataset, LoadCamera) and dataset.cap:
        dataset.cap.release()
    if writer is not None and writer.isOpened():
        writer.release()
        print("\n[완료] BEV 영상 녹화가 완료되었습니다.")
        print(f"결과물은 '{output_path}'에서 확인하실 수 있습니다.")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    
    bev_transform_and_save_realtime(args)