#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
유틸리티 스크립트: BEV(Birds-Eye View) 파라미터 설정
---------------------------------------------------
(수정본) 1280×720 해상도 기준 웹캠, 저장된 영상, 이미지 모두 지원
    - 영상/카메라 소스 지정 (--source)
    - 이미지 파일일 경우 단일 이미지를 이용하여 BEV 파라미터 지정
    - 영상 파일/웹캠인 경우 프레임이 계속 갱신되며 설정 가능

수정 사항:
- 왼쪽 점 선택 시 x축 대칭으로 오른쪽 점 자동 생성
- 선택된 4개 원본 좌표(src_points)를 txt 파일로 저장 기능 추가

설정 후 's' 키를 누르면 BEV 파라미터가 npz 파일 및 txt 파일로 저장됩니다.
"""

import cv2
import numpy as np
import argparse
import os

# 전역 변수: 원본 영상에서 선택한 4점 좌표
src_points = []
max_points = 4
current_frame_width = 0 # 현재 프레임 너비를 저장할 전역 변수

def parse_args():
    parser = argparse.ArgumentParser(description="BEV 파라미터 설정 유틸리티")
    parser.add_argument('--source', type=str, default='2',
                        help='영상/카메라 소스. 숫자 (예: 0,1,...)는 웹캠, 파일 경로는 영상 또는 이미지')
    parser.add_argument('--warp-width', type=int, default=640,
                        help='BEV 결과 영상 너비 (기본 640)')
    parser.add_argument('--warp-height', type=int, default=640,
                        help='BEV 결과 영상 높이 (기본 640)')
    parser.add_argument('--out-npz', type=str, default='bev_params_3.npz',
                        help='저장할 NPZ 파라미터 파일 이름')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_3.txt',
                        help='저장할 TXT 좌표 파일 이름')
    return parser.parse_args()

def mouse_callback(event, x, y, flags, param):
    global src_points, current_frame_width
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_frame_width == 0:
            print("[WARNING] 프레임 너비가 아직 설정되지 않았습니다. 프레임이 표시될 때까지 기다려주세요.")
            return

        if len(src_points) < max_points:
            if len(src_points) == 0: # 첫 번째 클릭 (왼쪽 아래점)
                src_points.append((x, y))
                # 대칭점 (오른쪽 아래점) 자동 추가
                symmetric_x = current_frame_width - 1 - x
                src_points.append((symmetric_x, y))
                print(f"[INFO] 왼쪽 아래점 추가: ({x}, {y})")
                print(f"[INFO] 오른쪽 아래점 자동 추가: ({symmetric_x}, {y}) (총 {len(src_points)}/4)")
            elif len(src_points) == 2: # 세 번째 클릭 (왼쪽 위점)
                src_points.append((x, y))
                # 대칭점 (오른쪽 위점) 자동 추가
                symmetric_x = current_frame_width - 1 - x
                src_points.append((symmetric_x, y))
                print(f"[INFO] 왼쪽 위점 추가: ({x}, {y})")
                print(f"[INFO] 오른쪽 위점 자동 추가: ({symmetric_x}, {y}) (총 {len(src_points)}/4)")
                print("[INFO] 4점 모두 선택 완료. 's'로 저장 또는 'r'로 리셋.")
            else:
                # 이 경우는 두 번째 또는 네 번째 점을 수동으로 찍으려 할 때 발생 (현재 로직에서는 발생 안 함)
                print(f"[INFO] 현재 {len(src_points)}개 점 선택됨. 다음 점을 선택하세요.")

        else:
            print("[WARNING] 이미 4점 모두 등록됨. 'r' 키로 초기화하거나 's' 키로 저장하세요.")

def main():
    global src_points, current_frame_width  # 전역 변수 사용 선언
    args = parse_args()
    source = args.source
    warp_w, warp_h = args.warp_width, args.warp_height

    is_image = False
    cap = None
    static_img = None

    if source.isdigit():
        cap_idx = int(source)
        cap = cv2.VideoCapture(cap_idx, cv2.CAP_V4L2)  # V4L2 사용하여 카메라 열기
        if not cap.isOpened():
            print(f"[ERROR] 카메라({cap_idx})를 열 수 없습니다.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        frame_width_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        current_frame_width = frame_width_cam # 초기 프레임 너비 설정
        print(f"[INFO] 웹캠 ({frame_width_cam}x{frame_height_cam})을 통한 실시간 영상 모드")
    else:
        ext = os.path.splitext(source)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if ext in image_extensions:
            static_img = cv2.imread(source)
            if static_img is None:
                print(f"[ERROR] 이미지 파일({source})을 열 수 없습니다.")
                return
            is_image = True
            current_frame_width = static_img.shape[1] # 이미지 너비 설정
            print(f"[INFO] 이미지 파일({source}, {static_img.shape[1]}x{static_img.shape[0]})을 통한 단일 이미지 모드")
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] 비디오 파일({source})을 열 수 없습니다.")
                return
            frame_width_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            current_frame_width = frame_width_vid # 초기 프레임 너비 설정
            print(f"[INFO] 비디오 파일({source}, {frame_width_vid}x{frame_height_vid})을 통한 영상 모드")

    dst_points_default = np.float32([
        [0,       warp_h],    # 왼 하단
        [warp_w,  warp_h],    # 오른 하단
        [0,       0],         # 왼 상단
        [warp_w,  0]          # 오른 상단
    ])

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original", mouse_callback)

    print("[INFO] 왼쪽 마우스 클릭으로 원본 영상에서 점을 선택하세요.")
    print("      1. 왼쪽 아래점 선택 -> 오른쪽 아래점 자동 생성")
    print("      2. 왼쪽 위점 선택 -> 오른쪽 위점 자동 생성")
    print("      'r' 키: 리셋(4점 좌표 다시 찍기)")
    print("      's' 키: BEV 파라미터 저장 후 종료")
    print("      'q' 키: 종료 (저장 안 함)")

    while True:
        if is_image:
            frame = static_img.copy()
            # 이미지 모드에서는 current_frame_width가 이미 설정됨
        else:
            ret, frame = cap.read()
            if not ret:
                if cap and cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("[INFO] 비디오의 끝에 도달했습니다. 마지막 프레임을 사용합니다.")
                    # 비디오의 끝에 도달하면 루프를 빠져나가거나 마지막 프레임을 계속 사용할 수 있습니다.
                    # 여기서는 루프를 유지하여 사용자가 's' 또는 'q'를 누를 수 있도록 합니다.
                    # 새로운 프레임을 읽으려고 시도하지 않도록 플래그 설정 또는 프레임 복사본 사용
                    if frame is None: # 마지막 프레임조차 없으면 종료
                         print("[ERROR] 마지막 프레임도 가져올 수 없습니다. 종료합니다.")
                         break
                else:
                    print("[WARNING] 프레임 읽기 실패 또는 영상 종료 -> 종료")
                    break
            if frame is not None and current_frame_width != frame.shape[1] : # 혹시 모를 동적 해상도 변경 대비
                current_frame_width = frame.shape[1]


        disp = frame.copy()
        # 점 선택 순서에 맞게 번호 표시 (왼쪽아래-1, 오른쪽아래-2, 왼쪽위-3, 오른쪽위-4)
        point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]
        for i, pt in enumerate(src_points):
            cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            label = point_labels[i] if i < len(point_labels) else f"{i+1}"
            cv2.putText(disp, label, (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(src_points) == 4: # 4개 점이 모두 선택되면 폴리곤으로 연결하여 표시
             cv2.polylines(disp, [np.array(src_points, dtype=np.int32)], isClosed=True, color=(0,0,255), thickness=2)

        cv2.imshow("Original", disp)

        bev_result = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)
        if len(src_points) == 4:
            # src_points 순서: 사용자 클릭(왼쪽아래), 자동생성(오른쪽아래), 사용자클릭(왼쪽위), 자동생성(오른쪽위)
            # getPerspectiveTransform을 위한 src_np의 순서는 왼하, 우하, 왼상, 우상 이어야 함.
            # 현재 src_points의 저장 순서가 이와 동일 (0: 왼아래, 1:오른아래, 2:왼위, 3:오른위)
            src_np = np.float32(src_points)
            M = cv2.getPerspectiveTransform(src_np, dst_points_default)
            bev_result = cv2.warpPerspective(frame, M, (warp_w, warp_h))
        cv2.imshow("BEV", bev_result)

        key = cv2.waitKey(30) & 0xFF # 영상 재생 속도 고려하여 waitKey 값 조정 (예: 30ms)
        if key == ord('q'):
            print("[INFO] 'q' 키 입력 -> 종료 (저장 안 함)")
            break
        elif key == ord('r'):
            print("[INFO] 'r' 키 입력 -> 4점 좌표 초기화")
            src_points = []
        elif key == ord('s'):
            if len(src_points) < 4:
                print("[WARNING] 4점 미선택. 2개의 점을 클릭하여 4점을 모두 설정 후 다시 시도하세요.")
            else:
                print("[INFO] 's' 키 입력 -> BEV 파라미터 저장 후 종료")
                out_file_npz = args.out_npz
                out_file_txt = args.out_txt

                src_arr = np.float32(src_points)
                dst_arr = dst_points_default

                # NPZ 파일 저장 (기존 로직 유지)
                np.savez(out_file_npz,
                         src_points=src_arr,
                         dst_points=dst_arr,
                         warp_w=warp_w,
                         warp_h=warp_h)
                print(f"[INFO] '{out_file_npz}' 파일에 BEV 파라미터 저장 완료.")

                # TXT 파일 저장
                try:
                    with open(out_file_txt, 'w') as f:
                        f.write("# Selected BEV Source Points (x, y) for original image\n")
                        f.write("# Order: Left-Bottom, Right-Bottom, Left-Top, Right-Top\n")
                        for i, point in enumerate(src_points):
                            f.write(f"{point[0]}, {point[1]} # {point_labels[i]}\n")
                    print(f"[INFO] '{out_file_txt}' 파일에 선택된 좌표 저장 완료.")
                except Exception as e:
                    print(f"[ERROR] TXT 파일 저장 중 오류 발생: {e}")
                break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] bev_utils.py 종료.")

if __name__ == '__main__':
    main()