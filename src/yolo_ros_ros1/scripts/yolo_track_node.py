#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Float32
from cv_bridge import CvBridge
from ultralytics import YOLO

# Import custom messages from this package
from yolo_msgs.msg import BoundingBox2D, Detection, DetectionArray

class DynamicObstacleDetectorNode:
    def __init__(self):
        rospy.init_node("yolo_track_node", anonymous=True)

        # Parameters
        self.threshold = rospy.get_param("~threshold", 0.5)
        self.iou = rospy.get_param("~iou", 0.5)
        self.max_det = rospy.get_param("~max_det", 100)
        self.imgsz_height = rospy.get_param("~imgsz_height", 360)
        self.imgsz_width = rospy.get_param("~imgsz_width", 640)
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")

        # 동적 객체로 판단할 바운딩 박스 넓이 임계값
        self.bbox_area_threshold = rospy.get_param("~bbox_area_threshold", 4500)
        # 동적 객체로 판단할 픽셀 이동 거리 임계값 (ego-motion 보상 후)
        self.MOTION_THRESHOLD_PX = 10.0

        model_path = rospy.get_param("~model_path", "./best.pt")
        self.steering_angle = 0.0

        rospy.loginfo(f"loading YOLO Model: {model_path}")
        self.model = YOLO(model_path)

        # Publishers
        self._tracking_pub = rospy.Publisher("tracking", DetectionArray, queue_size=10)
        self._dbg_pub = rospy.Publisher("dbg_image", Image, queue_size=10)
        self._info_pub = rospy.Publisher("cone_info", String, queue_size=10)
        self._motion_error_pub = rospy.Publisher("motion_error_info", String, queue_size=10)
        self._dynamic_obstacle_pub = rospy.Publisher("/dynamic_obstacle", Bool, queue_size=10)

        # Subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.steering_sub = rospy.Subscriber("/steering_angle", Float32, self.steering_angle_cb, queue_size=10)
        rospy.loginfo(f"image topic subscription start: {self.image_topic}")

        self.cv_bridge = CvBridge()

        self.old_gray = None
        self.prev_bg_features = None
        self.tracked_cones = {}
        
        self.lk_params = dict(winSize=(31, 31), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)

        rospy.loginfo("Node setting complete")
    
    def steering_angle_cb(self, msg):
        self.steering_angle = msg.data

    def image_cb(self, msg: Image):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"image translate fail: {e}")
            return

        cv_image = cv2.resize(cv_image, (self.imgsz_width, self.imgsz_height))
        height, width, _ = cv_image.shape
        frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        debug_image = cv_image.copy()

        try:
            results = self.model.track(cv_image, persist=True, conf=self.threshold, iou=self.iou)[0]
        except Exception as e:
            rospy.logerr(f"YOLO track error: {e}")
            return

        # 1. 배경 마스크 생성 (YOLO로 검출된 콘 영역을 제외)
        background_mask = np.ones(frame_gray.shape, dtype=np.uint8) * 255
        if results.boxes:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box
                cv2.rectangle(background_mask, (x1, y1), (x2, y2), 0, -1)

        # 2. 배경 마스크를 사용하여 배경 특징점 추출
        ego_motion_matrix = None
        if self.old_gray is not None and self.prev_bg_features is not None and len(self.prev_bg_features) > 4:
            # Optical Flow 계산
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.prev_bg_features, None, **self.lk_params)
            
            # 추적에 성공한 특징점만 필터링
            good_old = self.prev_bg_features[st == 1]
            good_new = p1[st == 1]

            if len(good_new) > 4:
                # Homography 행렬을 계산하여 ego-motion 보상/ 카메라 움직임을 모델링
                ego_motion_matrix, h_mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)

        tracked_detections = DetectionArray()
        tracked_detections.header = msg.header
        cone_info_list = []
        current_tracked_cones = {}
        is_dynamic_obstacle_detected = False

        # 3. 추적된 각 콘에 대해 동적/정적 판단 및 시각화
        if results.boxes and results.boxes.id is not None:
            boxes_xywh = results.boxes.xywh.cpu()
            confs = results.boxes.conf.cpu()
            clss = results.boxes.cls.cpu()
            track_ids = results.boxes.id.int().cpu().tolist()

            for i, track_id in enumerate(track_ids):
                x_center, y_center, w, h = boxes_xywh[i].numpy().astype(int)
                class_idx = int(clss[i].numpy())
                yolo_label = results.names[class_idx]
                
                is_dynamic = False
                motion_error = 0.0

                # Ego-motion 행렬이 있고, 이전 프레임에서 추적되던 콘일 경우
                if ego_motion_matrix is not None and track_id in self.tracked_cones:
                    bbox_area = w * h
                    is_in_central_roi = (width // 3 < x_center < 2 * width // 3)

                    if bbox_area > self.bbox_area_threshold and is_in_central_roi:
                        # 이전 프레임의 중심점 위치
                        prev_pos = np.array([[self.tracked_cones[track_id]['position']]], dtype=np.float32)
                        predicted_pos = cv2.perspectiveTransform(prev_pos, ego_motion_matrix)[0][0]
                        actual_pos = np.array([x_center, y_center])
                        motion_error = np.linalg.norm(actual_pos - predicted_pos)

                        motion_error_info_msg = String()
                        motion_error_info_msg.data = f"ID {track_id}: Motion Error {motion_error:.2f} px"
                        self._motion_error_pub.publish(motion_error_info_msg)

                        # 시각화 : 예측 위치와 실제 위치를 연결하는 선
                        cv2.circle(debug_image, (int(predicted_pos[0]), int(predicted_pos[1])), 5, (0, 0, 255), -1)
                        cv2.line(debug_image, (int(predicted_pos[0]), int(predicted_pos[1])), (int(actual_pos[0]), int(actual_pos[1])), (0, 255, 255), 2)
                        
                        # --- 코드 수정 부분 시작 ---
                        # 객체의 중심이 중앙 1/3 영역에 있는지 확인
                        if motion_error > self.MOTION_THRESHOLD_PX and -15 < self.steering_angle < 15 and is_in_central_roi:
                            is_dynamic = True
                            is_dynamic_obstacle_detected = True
                            
                            rospy.logwarn(f'DYNAMIC OBSTACLE! ID: {track_id}, Motion Error: {motion_error:.2f} px')
                        # --- 코드 수정 부분 끝 ---

                        # if motion_error > self.MOTION_THRESHOLD_PX and -5 < self.steering_angle < 5:
                        #     is_dynamic = True
                        #     is_dynamic_obstacle_detected = True
                            
                        #     rospy.logwarn(f'DYNAMIC OBSTACLE! ID: {track_id}, Motion Error: {motion_error:.2f} px')

                final_label = yolo_label
                if is_dynamic:
                    final_label += " (Dynamic)"

                current_tracked_cones[track_id] = {'position': (x_center, y_center), 'class_name': yolo_label}

                detection = Detection()
                detection.id = str(track_id)
                detection.class_id = class_idx
                detection.class_name = final_label
                detection.score = float(confs[i].numpy())
                
                bbox = BoundingBox2D()
                bbox.center.position.x = float(x_center)
                bbox.center.position.y = float(y_center)
                bbox.size.x = float(w)
                bbox.size.y = float(h)
                detection.bbox = bbox
                tracked_detections.detections.append(detection)

                cone_info_list.append(f"ID {track_id}: {final_label} ({x_center}, {y_center})")

                x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                color = (0, 255, 255) if is_dynamic else (0, 0, 0)
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                
                # 바운딩 박스 넓이 / error 표시
                bbox_area = w * h
                if is_dynamic:
                    cv2.putText(debug_image, f"{motion_error:.1f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(debug_image, f"{bbox_area}", (x1, y1 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 10)
                else:
                    cv2.putText(debug_image, f"{bbox_area}", (x1, y1 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 4. 다음 프레임을 위한 상태 업데이트
        self.old_gray = frame_gray.copy()
        self.prev_bg_features = cv2.goodFeaturesToTrack(self.old_gray, mask=background_mask, **self.feature_params)
        self.tracked_cones = current_tracked_cones

        # 시각화: 현재 추적 중인 배경 특징점 그리기
        # if self.prev_bg_features is not None:
        #     for point in self.prev_bg_features:
        #         x, y = point.ravel()
        #         cv2.circle(debug_image, (int(x), int(y)), 3, (0, 255, 0), -1)

        if is_dynamic_obstacle_detected:
            cv2.putText(debug_image, "Dynamic", (200, self.imgsz_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 0, 255), 5)

        self._tracking_pub.publish(tracked_detections)
        info_msg = String()
        info_msg.data = "; ".join(cone_info_list)
        self._info_pub.publish(info_msg)

        dynamic_obstacle_msg = Bool()
        dynamic_obstacle_msg.data = is_dynamic_obstacle_detected
        self._dynamic_obstacle_pub.publish(dynamic_obstacle_msg)

        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = msg.header
        self._dbg_pub.publish(debug_msg)

def main():
    node = DynamicObstacleDetectorNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("노드 종료 중...")

if __name__ == "__main__":
    main()
