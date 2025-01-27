#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
유틸리티 스크립트: BEV(Birds-Eye View) 파라미터 설정
---------------------------------------------------
(수정본) 1280×720로 카메라 해상도 고정 + Letterbox 미사용
=> 이 상태로 4점을 찍은 뒤, (640×640) BEV 파라미터를 npz로 저장.
"""

import cv2
import numpy as np
import argparse

src_points = []
max_points = 4

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0',
                        help='영상/카메라 소스. 0,1,.. 또는 video.mp4')
    parser.add_argument('--warp-width', type=int, default=640,
                        help='BEV 결과 영상 너비(기본 640)')
    parser.add_argument('--warp-height', type=int, default=640,
                        help='BEV 결과 영상 높이(기본 640)')
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
    args = parse_args()

    # 카메라(또는 영상) 열기
    source = args.source
    cap = None
    if source.isdigit():
        cap_idx = int(source)
        cap = cv2.VideoCapture(cap_idx)
        if not cap.isOpened():
            print(f"[ERROR] 카메라({cap_idx})를 열 수 없습니다.")
            return
        # 1280x720 고정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] 비디오 파일({source})을 열 수 없습니다.")
            return

    warp_w, warp_h = args.warp_width, args.warp_height

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)

    cv2.setMouseCallback("Original", mouse_callback)

    print("[INFO] 왼쪽 마우스 클릭으로 원본 영상 4점을 선택 (왼하단→오른하단→왼상단→오른상단 권장)")
    print("      'r' 키: 리셋(4점 좌표 다시 찍기)")
    print("      's' 키: BEV 파라미터 저장 후 종료")
    print("      'q' 키: 종료(저장 안 함)")

    # 목적 좌표 ([0, warp_h], [warp_w, warp_h], [0, 0], [warp_w, 0]) => 640×640
    dst_points_default = np.float32([
        [0,       warp_h],    # 왼 하단
        [warp_w,  warp_h],    # 오른 하단
        [0,       0],         # 왼 상단
        [warp_w,  0]          # 오른 상단
    ])

    global src_points

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 프레임 읽기 실패 -> 종료")
            break

        # frame: (1280x720) or 소스 원본
        # 여기선 추가로 Letterbox or resize하지 않고, '있는 그대로' 표시
        disp = frame.copy()

        for i, pt in enumerate(src_points):
            cv2.circle(disp, pt, 5, (0,255,0), -1)
            cv2.putText(disp, f"{i+1}", (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Original", disp)

        # BEV 계산 (4점이 모두 선택된 경우에만)
        bev_result = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)
        if len(src_points) == 4:
            src_np = np.float32(src_points)
            M = cv2.getPerspectiveTransform(src_np, dst_points_default)
            bev_result = cv2.warpPerspective(frame, M, (warp_w, warp_h))

        cv2.imshow("BEV", bev_result)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' -> 종료 (저장 안 함)")
            break
        elif key == ord('r'):
            print("[INFO] 'r' -> 4점 좌표 초기화")
            src_points = []
        elif key == ord('s'):
            if len(src_points) < 4:
                print("[WARNING] 4점 미선택. 4점을 모두 찍은 후 다시 시도.")
            else:
                print("[INFO] 's' -> BEV 파라미터 저장 후 종료")
                out_file = args.out
                src_arr = np.float32(src_points)
                dst_arr = dst_points_default
                np.savez(out_file,
                         src_points=src_arr,
                         dst_points=dst_arr,
                         warp_w=warp_w,
                         warp_h=warp_h)
                print(f"[INFO] '{out_file}' 에 BEV 파라미터 저장 완료.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] bev_utils.py 종료.")

if __name__ == '__main__':
    main()
