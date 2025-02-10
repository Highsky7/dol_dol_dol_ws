#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
유틸리티 스크립트: BEV(Birds-Eye View) 파라미터 설정
---------------------------------------------------
(수정본) 1280×720 해상도 기준 웹캠, 저장된 영상, 이미지 모두 지원
    - 영상/카메라 소스 지정 (--source)
    - 이미지 파일일 경우 단일 이미지를 이용하여 BEV 파라미터 지정
    - 영상 파일/웹캠인 경우 프레임이 계속 갱신되며 설정 가능

설정 후 's' 키를 누르면 BEV 파라미터가 npz 파일로 저장됩니다.
"""

import cv2
import numpy as np
import argparse
import os

# 전역 변수: 원본 영상에서 선택한 4점 좌표
src_points = []
max_points = 4

def parse_args():
    parser = argparse.ArgumentParser(description="BEV 파라미터 설정 유틸리티")
    parser.add_argument('--source', type=str, default='/home/highsky/Videos/Webcam/직선.mp4',
                        help='영상/카메라 소스. 숫자 (예: 0,1,...)는 웹캠, 파일 경로는 영상 또는 이미지')
    parser.add_argument('--warp-width', type=int, default=640,
                        help='BEV 결과 영상 너비 (기본 640)')
    parser.add_argument('--warp-height', type=int, default=640,
                        help='BEV 결과 영상 높이 (기본 640)')
    parser.add_argument('--out', type=str, default='bev_params.npz',
                        help='저장할 파라미터 파일 이름')
    return parser.parse_args()

def mouse_callback(event, x, y, flags, param):
    global src_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < max_points:
            src_points.append((x, y))
            print(f"[INFO] 좌표 추가: ({x}, {y}) (총 {len(src_points)}/4)")
        else:
            print("[WARNING] 이미 4점 모두 등록됨. 'r' 키로 초기화하세요.")

def main():
    global src_points  # 전역 변수 src_points 사용 선언
    args = parse_args()
    source = args.source
    warp_w, warp_h = args.warp_width, args.warp_height

    is_image = False   # 이미지 모드 여부
    cap = None
    static_img = None

    # 소스가 숫자인 경우 -> 웹캠
    if source.isdigit():
        cap_idx = int(source)
        cap = cv2.VideoCapture(cap_idx)
        if not cap.isOpened():
            print(f"[ERROR] 카메라({cap_idx})를 열 수 없습니다.")
            return
        # 웹캠 해상도 고정: 1280x720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("[INFO] 웹캠을 통한 실시간 영상 모드")
    else:
        # 파일 경로인 경우, 확장자를 확인하여 이미지/영상 구분
        ext = os.path.splitext(source)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if ext in image_extensions:
            static_img = cv2.imread(source)
            if static_img is None:
                print(f"[ERROR] 이미지 파일({source})을 열 수 없습니다.")
                return
            is_image = True
            print(f"[INFO] 이미지 파일({source})를 통한 단일 이미지 모드")
        else:
            # 영상 파일인 경우
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] 비디오 파일({source})을 열 수 없습니다.")
                return
            print(f"[INFO] 비디오 파일({source})를 통한 영상 모드")

    # BEV 목적 좌표 설정 (왼하단, 오른하단, 왼상단, 오른상단 순)
    dst_points_default = np.float32([
        [0,       warp_h],    # 왼 하단
        [warp_w,  warp_h],    # 오른 하단
        [0,       0],         # 왼 상단
        [warp_w,  0]          # 오른 상단
    ])

    # 창 생성 및 마우스 콜백 등록
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original", mouse_callback)

    print("[INFO] 왼쪽 마우스 클릭으로 원본 영상에서 4점을 선택하세요 (권장 순서: 왼하단 → 오른하단 → 왼상단 → 오른상단)")
    print("      'r' 키: 리셋(4점 좌표 다시 찍기)")
    print("      's' 키: BEV 파라미터 저장 후 종료")
    print("      'q' 키: 종료 (저장 안 함)")

    while True:
        # 이미지 모드: static_img를 사용, 영상 모드: 프레임 읽기
        if is_image:
            frame = static_img.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] 프레임 읽기 실패 또는 영상 종료 -> 종료")
                break

        # 원본 영상에 이미 선택된 점들을 표시
        disp = frame.copy()
        for i, pt in enumerate(src_points):
            cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            cv2.putText(disp, f"{i+1}", (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Original", disp)

        # 4점이 모두 선택된 경우 BEV 변환 결과 표시
        bev_result = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)
        if len(src_points) == 4:
            src_np = np.float32(src_points)
            M = cv2.getPerspectiveTransform(src_np, dst_points_default)
            bev_result = cv2.warpPerspective(frame, M, (warp_w, warp_h))
        cv2.imshow("BEV", bev_result)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' 키 입력 -> 종료 (저장 안 함)")
            break
        elif key == ord('r'):
            print("[INFO] 'r' 키 입력 -> 4점 좌표 초기화")
            src_points = []  # 전역 변수 src_points를 초기화
        elif key == ord('s'):
            if len(src_points) < 4:
                print("[WARNING] 4점 미선택. 4점을 모두 찍은 후 다시 시도하세요.")
            else:
                print("[INFO] 's' 키 입력 -> BEV 파라미터 저장 후 종료")
                out_file = args.out
                src_arr = np.float32(src_points)
                dst_arr = dst_points_default
                np.savez(out_file,
                         src_points=src_arr,
                         dst_points=dst_arr,
                         warp_w=warp_w,
                         warp_h=warp_h)
                print(f"[INFO] '{out_file}' 파일에 BEV 파라미터 저장 완료.")
                break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] bev_utils.py 종료.")

if __name__ == '__main__':
    main()
