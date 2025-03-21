#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np

"""
미다스(MiDaS) Depth만 별도로 테스트하여,
특정 픽셀에서의 Depth 값을 클릭으로 확인하기 위한 코드.

Usage:
  python midas_depth.py --model-type DPT_Large --source 0
     - model-type: ["DPT_Large", "DPT_Hybrid", "MiDaS_small"] 등
     - source: 웹캠 ID(0,1,2...) or 영상/이미지 파일 경로
  - 실행 후 화면에서 마우스 클릭하면, 해당 픽셀의 MiDaS 정규화된 depth 값(0~1)을 콘솔에 출력

이를 통해 실제 물체까지의 거리(미터)와 MiDaS 값의 관계를 측정 → depth_scale 파라미터를 잡는 데 활용.
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='DPT_Hybrid',
                        help='MiDaS model type: DPT_Large, DPT_Hybrid, MiDaS_small')
    parser.add_argument('--source', type=str, default='2',
                        help='0 for webcam, or path to video/image file')
    return parser.parse_args()


class MiDaSDepth:
    def __init__(self, model_type="DPT_Hybrid"):
        """
        MiDaS 모델 로드:
          - pip install timm
          - torch.hub.load("intel-isl/MiDaS", model_type)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Loading MiDaS model: {model_type} on {self.device}...")

        # untrusted repo warning을 피하기 위해 trust_repo=True
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.midas.to(self.device)
        self.midas.eval()

        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if "DPT" in model_type:
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

        print("[INFO] MiDaS model loaded successfully.")

    def run_depth(self, cv2_bgr):
        """
        cv2_bgr -> normalized depth map (0~1)
        """
        img_rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            depth_map = torch.squeeze(prediction).cpu().numpy()

        # normalize to 0~1
        mn, mx = depth_map.min(), depth_map.max()
        denom = mx - mn if mx > mn else 1e-8
        depth_map = (depth_map - mn) / denom
        return depth_map


def create_color_map(depth_map):
    """
    depth_map: float ndarray (0~1)
    return: color-coded depth (BGR)
    """
    cm = (depth_map*255).astype(np.uint8)
    cm_color = cv2.applyColorMap(cm, cv2.COLORMAP_INFERNO)
    return cm_color


class DepthTester:
    def __init__(self, model_type, source):
        self.model_type = model_type
        self.source = source
        self.midas = MiDaSDepth(model_type)

        # 마우스 이벤트용
        self.depth_map = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.depth_map is not None:
                # depth_map shape == current frame shape
                if 0 <= y < self.depth_map.shape[0] and 0 <= x < self.depth_map.shape[1]:
                    d_val = self.depth_map[y, x]
                    print(f"[INFO] Click at ({x},{y}), MiDaS depth value={d_val:.3f}")
                    # 실제 거리(예: 3.0m)와 비교하여 scale 보정 가능

    def run(self):
        # source -> webcam or file
        if self.source.isdigit():
            # webcam
            cap_index = int(self.source)
            cap = cv2.VideoCapture(cap_index)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open webcam index {cap_index}")
            print(f"[INFO] Opened webcam index: {cap_index}")
        else:
            # file
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open file: {self.source}")
            print(f"[INFO] Opened file: {self.source}")

        cv2.namedWindow("MiDaS Depth")
        cv2.setMouseCallback("MiDaS Depth", self.mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] No more frames or cannot read.")
                break

            # run midas
            depth_map = self.midas.run_depth(frame)
            self.depth_map = depth_map  # for mouse callback

            # color map for display
            cm_depth = create_color_map(depth_map)

            # side-by-side for reference
            h, w = frame.shape[:2]
            vis = np.zeros((h, w*2, 3), dtype=np.uint8)
            # left: original
            frame_resized = cv2.resize(frame, (w, h))
            vis[:, :w, :] = frame_resized
            # right: depth colormap
            cm_resized = cv2.resize(cm_depth, (w, h))
            vis[:, w:, :] = cm_resized

            cv2.imshow("MiDaS Depth", vis)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("[INFO] Quit.")
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    dt = DepthTester(args.model_type, args.source)
    dt.run()


if __name__ == "__main__":
    main()
