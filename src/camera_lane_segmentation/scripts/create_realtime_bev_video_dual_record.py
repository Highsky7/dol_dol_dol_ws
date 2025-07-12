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
    parser = argparse.ArgumentParser(description="실시간 카메라 영상을 원본과 BEV(Bird's-Eye-View)로 변환하여 각각 H.264 코덱으로 녹화하는 스크립트")
    parser.add_argument('--source', type=str,
                        default='2',
                        help='카메라 인덱스. 일반적으로 내장 카메라는 "0", 외부 카메라는 "1" 등. 예: 0')
    parser.add_argument('--img-size', type=int, default=640, help='처리할 이미지 해상도 (LoadCamera 클래스에 전달)')
    parser.add_argument('--param-file', type=str, default='/home/highsky/dol_dol_dol_ws/bev_params_y_5.npz', help='BEV 파라미터 파일 경로. 예: ./bev_params_1.npz')
    parser.add_argument('--output-dir', type=str, default='runs/bev_output', help='결과 영상이 저장될 폴더')
    return parser

def bev_transform_and_save_realtime(opt):
    """
    메인 로직: LoadCamera를 사용하여 실시간 영상을 읽고, 원본과 BEV 변환 프레임을 각각 H.264 코덱으로 녹화합니다.
    'q' 키를 누르면 녹화가 중단되고 프로그램이 종료됩니다.
    """
    try:
        dataset = LoadCamera(opt.source, img_size=opt.img_size)
    except Exception as e:
        print(f"[오류] 카메라를 열 수 없습니다: {opt.source}")
        print(f"  - 세부 정보: {e}")
        return

    ### 변경된 부분: 2개의 비디오 녹화를 위해 Writer 객체를 2개 선언합니다. ###
    writer_original = None
    writer_bev = None
    
    # 저장 경로 설정
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    ### 변경된 부분: 원본 영상과 BEV 영상의 저장 경로를 각각 생성합니다. ###
    output_path_original = output_dir / f"original_output_{timestamp}_h264.mp4"
    output_path_bev = output_dir / f"bev_output_{timestamp}_h264.mp4"
    
    print("=====================================================")
    ### 변경된 부분: 안내 메시지를 수정하여 2개의 파일이 저장됨을 알립니다. ###
    print(f"  실시간 원본 및 BEV 변환 영상 H.264 동시 녹화를 시작합니다")
    print(f"  - 입력 카메라: {opt.source}")
    print(f"  - 처리 해상도: {opt.img_size} (LoadCamera 사용)")
    print(f"  - BEV 파라미터: {opt.param_file}")
    print(f"  - 원본 영상 저장 경로: {output_path_original}")
    print(f"  - BEV 영상 저장 경로: {output_path_bev}")
    print("\n  결과 창에서 'q' 키를 누르면 녹화가 중단되고 종료됩니다.")
    print("=====================================================")

    params = np.load(opt.param_file)
    output_w = int(params['warp_w'])
    output_h = int(params['warp_h'])

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        
        # 첫 프레임에서 VideoWriter 초기화
        ### 변경된 부분: 두 Writer가 모두 초기화되지 않았을 때 실행합니다. ###
        if writer_original is None and writer_bev is None:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                print("[경고] 카메라에서 FPS를 얻을 수 없어 30으로 설정합니다.")
                fps = 30
            
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 코덱 사용
            
            ### 추가된 부분: 원본 영상의 가로, 세로 크기를 가져옵니다. ###
            h, w, _ = im0s.shape
            
            ### 추가된 부분: 원본 영상용 VideoWriter를 초기화합니다. ###
            writer_original = cv2.VideoWriter(str(output_path_original), fourcc, fps, (w, h))
            
            ### 변경된 부분: BEV 영상용 VideoWriter를 초기화합니다. ###
            writer_bev = cv2.VideoWriter(str(output_path_bev), fourcc, fps, (output_w, output_h))

            ### 변경된 부분: 두 Writer가 모두 성공적으로 열렸는지 확인합니다. ###
            if not writer_original.isOpened() or not writer_bev.isOpened():
                print("\n[오류] H.264 코덱으로 비디오 파일을 생성할 수 없습니다.")
                print("  - 시스템에 H.264 코덱이 설치되어 있는지 확인해주세요.")
                print("  - 또는 다른 FourCC 코드(예: 'avc1', 'X264', 'mp4v')를 시도해보세요.")
                break # 루프 중단

        input_frame = im0s
        bev_frame = do_bev_transform(input_frame, opt.param_file)

        ### 추가된 부분: 원본 프레임을 녹화합니다. ###
        if writer_original is not None and writer_original.isOpened():
            writer_original.write(input_frame)
            
        ### 변경된 부분: BEV 프레임을 녹화합니다. ###
        if writer_bev is not None and writer_bev.isOpened():
            writer_bev.write(bev_frame)

        cv2.imshow('Real-time Input (Recording...)', input_frame)
        cv2.imshow('BEV Transformed Video (Recording...)', bev_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' 키가 입력되어 녹화를 중단합니다.")
            break
    
    # 자원 해제
    if isinstance(dataset, LoadCamera) and dataset.cap:
        dataset.cap.release()
        
    ### 추가된 부분: 원본 영상 Writer를 해제합니다. ###
    if writer_original is not None and writer_original.isOpened():
        writer_original.release()
        
    ### 변경된 부분: BEV 영상 Writer를 해제합니다. ###
    if writer_bev is not None and writer_bev.isOpened():
        writer_bev.release()

    ### 변경된 부분: 완료 메시지를 수정합니다. ###
    print("\n[완료] 원본 및 BEV 영상 녹화가 완료되었습니다.")
    print(f"결과물은 아래 경로에서 확인하실 수 있습니다:")
    print(f"  - 원본: '{output_path_original}'")
    print(f"  - BEV: '{output_path_bev}'")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    
    bev_transform_and_save_realtime(args)