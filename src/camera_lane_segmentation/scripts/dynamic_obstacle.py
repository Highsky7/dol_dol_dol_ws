#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import argparse
import cv2
import torch
import numpy as np
from math import sqrt
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool, Float32

# =============================================================================
# 칼만 필터 및 다중 객체 추적기 클래스 (이전과 동일한 부분은 설명 생략)
# =============================================================================
class KalmanFilter:
    def __init__(self, dt=1.0):
        self.F = np.array([[1,0,dt,0,0,0],[0,1,0,dt,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.float32)
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        self.R = np.eye(4, dtype=np.float32) * 1.0
        self.x = np.zeros((6, 1), dtype=np.float32); self.P = np.eye(6, dtype=np.float32) * 10.
    def predict(self):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S); self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
    def initialize_state(self, box):
        w=box[2]-box[0]; h=box[3]-box[1]; cx=box[0]+w/2; cy=box[1]+h/2
        self.x = np.array([cx, cy, 0, 0, w, h]).reshape(6, 1)

class Tracker:
    def __init__(self, iou_threshold=0.3, max_age=10, motion_threshold=10.0, vx_weight=2.0, dynamic_confirm=3, static_confirm=10):
        self.tracks = []; self.next_id = 0
        self.iou_threshold=iou_threshold; self.max_age=max_age; self.motion_threshold=motion_threshold
        self.vx_weight=vx_weight; self.dynamic_confirmation_threshold=dynamic_confirm; self.static_confirmation_threshold=static_confirm
    def _iou(self, b1, b2):
        x1,y1,x2,y2=max(b1[0],b2[0]),max(b1[1],b2[1]),min(b1[2],b2[2]),min(b1[3],b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / union if union > 0 else 0
    def update(self, detections):
        if self.tracks:
            for t in self.tracks: t['kf'].predict(); t['age'] += 1
        matched = []
        if self.tracks and detections:
            predicted_boxes = [(t['kf'].x[0,0]-t['kf'].x[4,0]/2, t['kf'].x[1,0]-t['kf'].x[5,0]/2, t['kf'].x[0,0]+t['kf'].x[4,0]/2, t['kf'].x[1,0]+t['kf'].x[5,0]/2) for t in self.tracks]
            iou_matrix = np.array([[self._iou(p, d) for d in detections] for p in predicted_boxes])
            iou_matrix[iou_matrix < self.iou_threshold] = 0
            r, c = linear_sum_assignment(1.0 - iou_matrix)
            matched = [(r, c) for r, c in zip(r, c) if iou_matrix[r, c] > 0]
        for r, c in matched:
            t, d = self.tracks[r], detections[c]
            z = np.array([d[0]+(d[2]-d[0])/2, d[1]+(d[3]-d[1])/2, d[2]-d[0], d[3]-d[1]]).reshape(4,1)
            t['kf'].update(z); t['age'] = 0; t['box'] = d
        unmatched_dets = [d for i, d in enumerate(detections) if i not in [c for _, c in matched]]
        for d in unmatched_dets:
            kf = KalmanFilter(); kf.initialize_state(d)
            self.tracks.append({'id':self.next_id, 'kf':kf, 'age':0, 'box':d, 'status':'STATIC', 'dynamic_frames':0, 'static_frames':0}); self.next_id+=1
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        output = []
        for t in self.tracks:
            vx, vy = t['kf'].x[2:4].flatten()
            weighted_speed = sqrt((vx * self.vx_weight)**2 + vy**2)
            if weighted_speed > self.motion_threshold:
                t['static_frames'] = 0; t['dynamic_frames'] += 1
            else:
                t['dynamic_frames'] = 0; t['static_frames'] += 1
            if t['dynamic_frames'] >= self.dynamic_confirmation_threshold: t['status'] = 'DYNAMIC'
            elif t['static_frames'] >= self.static_confirmation_threshold: t['status'] = 'STATIC'
            
            # [수정] 반환 딕셔너리에 'speed' 키와 값을 추가합니다.
            output.append({'id':t['id'], 'box':t['box'], 'status':t['status'], 'speed': weighted_speed})
        return output

# =============================================================================
# 메인 ROS 노드 클래스
# =============================================================================
class DynamicObstacleDetectorNode:
    def __init__(self, opt):
        self.opt = opt
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"[Obstacle Detector] Using device: {self.device}")

        rospy.loginfo(f"[Obstacle Detector] Loading cone model from {self.opt.weights}...")
        self.model = YOLO(self.opt.weights).to(self.device)
        rospy.loginfo("[Obstacle Detector] Model loaded successfully.")

        self.tracker = Tracker(
            motion_threshold=self.opt.motion_thres, vx_weight=self.opt.vx_weight,
            dynamic_confirm=self.opt.dynamic_confirm, static_confirm=self.opt.static_confirm
        )

        self.current_steering_angle = 0.0
        self.steering_threshold = self.opt.steer_thres

        self.pub_image = rospy.Publisher('/obstacle_detector/image_annotated', Image, queue_size=1)
        self.pub_status = rospy.Publisher('/dynamic_obstacle', Bool, queue_size=1)

        self.sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.sub_steering = rospy.Subscriber('/steering_angle', Float32, self.steering_callback, queue_size=1)
        
        rospy.loginfo("[Obstacle Detector] Node initialized. Waiting for images and steering angle...")

    def steering_callback(self, msg):
        self.current_steering_angle = msg.data

    def image_callback(self, msg):
        try: cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e: rospy.logerr(e); return

        results = self.model(cv_image, conf=self.opt.conf_thres, verbose=False)
        detections = [box.xyxy[0].cpu().numpy().astype(int) for r in results for box in r.boxes if int(box.cls) == 0]
        
        tracked_obstacles = self.tracker.update(detections)

        dynamic_obstacle_found = False
        for obj in tracked_obstacles:
            if obj['status'] == 'DYNAMIC':
                dynamic_obstacle_found = True
                break

        final_publish_status = False
        if abs(self.current_steering_angle) > self.steering_threshold:
            final_publish_status = False
        else:
            final_publish_status = dynamic_obstacle_found

        self.pub_status.publish(Bool(data=final_publish_status))

        # --- 시각화 부분 ---
        annotated_frame = cv_image.copy()
        
        # [수정] 추적된 모든 객체에 대해 루프를 돌며 시각화 정보를 추가합니다.
        for obj in tracked_obstacles:
            # [수정] 'speed' 값을 딕셔너리에서 가져옵니다.
            box, obj_id, status, speed = obj['box'], obj['id'], obj['status'], obj['speed']
            
            color = (0, 0, 255) if status == "DYNAMIC" else (0, 255, 0)
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # 객체 정보 텍스트 (ID, 상태)
            info_text = f"ID:{obj_id} {status}"
            cv2.putText(annotated_frame, info_text, (box[0], box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # [수정] 증폭된 속도 값을 시각화합니다.
            speed_text = f"Speed: {speed:.1f}"
            cv2.putText(annotated_frame, speed_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 현재 조향각 및 최종 발행 상태 시각화
        steer_info_text = f"Steer Angle: {self.current_steering_angle:.1f} deg"
        # [수정] 최종 발행 상태에 따라 텍스트 색상을 다르게 표시하여 직관성을 높입니다.
        publish_color = (0, 0, 255) if final_publish_status else (0, 255, 255) # True일 때 빨강, False일 때 노랑
        final_status_text = f"Publishing '/dynamic_obstacle': {final_publish_status}"
        cv2.putText(annotated_frame, steer_info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated_frame, final_status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, publish_color, 2)

        try: self.pub_image.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))
        except CvBridgeError as e: rospy.logerr(e)

def main():
    rospy.init_node('dynamic_obstacle_detector_node', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./best.pt', help="YOLO model path")
    parser.add_argument('--conf-thres', type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument('--motion-thres', type=float, default=8.0, help="Weighted motion threshold")
    parser.add_argument('--vx-weight', type=float, default=2.5, help="Weight for horizontal velocity")
    parser.add_argument('--dynamic-confirm', type=int, default=3, help="Frames to confirm DYNAMIC state")
    parser.add_argument('--static-confirm', type=int, default=10, help="Frames to confirm STATIC state")
    parser.add_argument('--steer-thres', type=float, default=15.0, help="Steering angle threshold (degrees) to suppress dynamic detection")
    opt, _ = parser.parse_known_args()

    node = DynamicObstacleDetectorNode(opt)
    try: rospy.spin()
    except KeyboardInterrupt: rospy.loginfo("Shutting down node.")

if __name__ == '__main__':
    main()