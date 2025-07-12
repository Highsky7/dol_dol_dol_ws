#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, argparse, cv2, torch, numpy as np
from math import atan, atan2, degrees, sqrt
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path
import tf2_ros
import tf2_geometry_msgs        # 실질 사용은 없지만 의존성 유지를 위해 포함
from vehicle_msgs.msg import Track, TrackCone  # /track 메시지 처리를 위해 추가 (사용자 환경에 맞게 설치 필요)


# ────────── 유틸리티 함수 ────────── #
def polyfit_lane(points_y, points_x, order=2):
    if len(points_y) < 5: return None
    try: return np.polyfit(points_y, points_x, order)
    except (np.linalg.LinAlgError, TypeError): return None

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size: cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1: return np.zeros_like(binary_mask)
    comps = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: comps.append((i, stats[i, cv2.CC_STAT_AREA]))
    comps.sort(key=lambda x: x[1], reverse=True)
    cleaned = np.zeros_like(binary_mask)
    for i in range(min(len(comps), 2)):
        idx = comps[i][0]
        cleaned[labels == idx] = 255
    return cleaned

def final_filter(bev_mask):
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=10000) # 실제 환경 노이즈에 따라 튜닝 필요
    f4 = keep_top2_components(f3, min_area=300)
    return f4

def overlay_polyline(image, coeff, color=(0, 0, 255), step=4, thickness=2):
    if coeff is None: return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w: draw_points.append((int(x), int(y)))
    if len(draw_points) > 1: cv2.polylines(image, [np.array(draw_points, dtype=np.int32)], False, color, thickness)
    return image

# ────────── 메인 로직 클래스 ────────── #
class LaneFollowerNode:
    def __init__(self, opt):
        self.opt = opt
        self.bridge = CvBridge()

        # 디바이스 설정
        device_str = self.opt.device.lower()
        if device_str.isdigit() and torch.cuda.is_available(): self.device = torch.device(f'cuda:{device_str}')
        else: self.device = torch.device('cpu')
        rospy.loginfo(f"[Lane Follower] Using device: {self.device}")

        # YOLO 모델 로딩
        rospy.loginfo(f"[Lane Follower] Loading model from {self.opt.weights}...")
        self.model = YOLO(self.opt.weights).to(self.device)
        rospy.loginfo("[Lane Follower] Model loaded successfully.")

        # BEV 파라미터 로딩
        self.bev_params = np.load(self.opt.param_file)
        self.bev_h, self.bev_w = int(self.bev_params['warp_h']), int(self.bev_params['warp_w'])
        
        # 좌표 변환 계수 (사용하는 bev_params 파일에 맞게 튜닝 필요)
        self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = 0.0038125, 1.41, 0.00240625

        # 차선 추적 파라미터
        self.tracked_lanes = {'left': {'coeff': None, 'age': 0}, 'right': {'coeff': None, 'age': 0}}
        self.tracked_center_path = {'coeff': None}
        self.SMOOTHING_ALPHA = 0.6
        self.MAX_LANE_AGE = 10  # 최대 차선 추적 나이 (프레임 수)

        # Pure Pursuit 및 동적 전방주시거리 파라미터
        self.L = 0.73  # 차량 축거 (Wheelbase) [m]
        self.THROTTLE_MIN = 0.4
        self.THROTTLE_MAX = 0.6
        self.MIN_LOOKAHEAD_DISTANCE = 1.8
        self.MAX_LOOKAHEAD_DISTANCE = 2.5
        self.current_throttle = self.THROTTLE_MIN # 초기 throttle 값

        # 교통 콘 위치 저장을 위한 변수
        self.cones = []

        # 퍼블리셔 설정
        self.pub_steering = rospy.Publisher('auto_steer_angle_lane', Float32, queue_size=1)
        self.pub_lane_status = rospy.Publisher('lane_detection_status', Bool, queue_size=1)
        self.pub_markers = rospy.Publisher("lane_markers", MarkerArray, queue_size=1)
        self.pub_path = rospy.Publisher("center_path", Path, queue_size=1)
        self.pub_look = rospy.Publisher("lookahead_point", Marker, queue_size=1)
        self.pub_forced_rrt = rospy.Publisher("/forced_rrt", Bool, queue_size=1)

        # 서브스크라이버 설정
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.throttle_sub = rospy.Subscriber('auto_throttle', Float32, self.throttle_callback, queue_size=1)
        rospy.Subscriber("/track", Track, self.track_cb, queue_size=1)

        # TF 설정
        self.static_br = tf2_ros.StaticTransformBroadcaster()
        self.pub_static_tf()
        rospy.loginfo("[Lane Follower] Node initialized and ready.")

    # --------- TF 설정 (roboflow_final.py에서 가져옴) --------- #
    def pub_static_tf(self):
        tx = rospy.get_param("~camera_offset_x", 0.0)
        ty = rospy.get_param("~camera_offset_y", 0.0)
        tz = rospy.get_param("~camera_offset_z", 0.0)
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "velodyne"
        t.child_frame_id = "camera"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = tx, ty, tz
        t.transform.rotation.w = 1.0
        self.static_br.sendTransform(t)
        rospy.loginfo(f"[TF] velodyne→camera ({tx:.2f},{ty:.2f},{tz:.2f}) published.")

    # --------- 콜백 함수 --------- #
    def throttle_callback(self, msg):
        self.current_throttle = np.clip(msg.data, self.THROTTLE_MIN, self.THROTTLE_MAX)

    def track_cb(self, msg):
        """ /track 토픽 콜백: 교통 콘 위치 저장 (roboflow_final.py에서 가져옴) """
        self.cones = []
        for cone in msg.cones:
            # 여기서는 모든 타입의 콘을 장애물로 간주, 필요시 "traffic_cone" 등 특정 타입만 필터링
            self.cones.append((cone.x, cone.y))

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e); return
        self.process_image(cv_image)

    # --------- 좌표 변환 --------- #
    def do_bev_transform(self, image):
        M = cv2.getPerspectiveTransform(self.bev_params['src_points'], self.bev_params['dst_points'])
        return cv2.warpPerspective(image, M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        
    def image_to_vehicle(self, pt_bev):
        u, v = pt_bev
        x_vehicle = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        y_vehicle = (self.bev_w / 2 - u) * self.m_per_pixel_x
        return x_vehicle, y_vehicle

    # --------- 메인 처리 함수 --------- #
    def process_image(self, im0s):
        # 1. BEV 변환 및 추론
        bev_image_input = self.do_bev_transform(im0s)
        results = self.model(bev_image_input, imgsz=self.opt.img_size, conf=self.opt.conf_thres, iou=self.opt.iou_thres, device=self.device, verbose=False)
        result = results[0]
        
        # 2. 마스크 처리 및 필터링
        combined_mask_bev = np.zeros(result.orig_shape, dtype=np.uint8)
        if result.masks is not None:
            for conf, mask_tensor in zip(result.boxes.conf, result.masks.data):
                if conf >= 0.5:
                    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                    mask_np = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]))
                    combined_mask_bev = np.maximum(combined_mask_bev, mask_np)
        final_mask = final_filter(combined_mask_bev)

        # 3. 필터링된 마스크에서 차선 후보 추출 및 추적
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        current_detections = []
        if num_labels > 1:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 100:
                    ys, xs = np.where(labels == i)
                    coeff = polyfit_lane(ys, xs, order=2)
                    if coeff is not None:
                        x_at_bottom = np.polyval(coeff, self.bev_h - 1)
                        current_detections.append({'coeff': coeff, 'x_bottom': x_at_bottom})
            current_detections.sort(key=lambda c: c['x_bottom'])
        
        left_lane_tracked, right_lane_tracked = self.tracked_lanes['left'], self.tracked_lanes['right']
        current_left, current_right = None, None
        if len(current_detections) == 2: current_left, current_right = current_detections[0], current_detections[1]
        elif len(current_detections) == 1:
            d = current_detections[0]
            dl=abs(d['x_bottom']-np.polyval(left_lane_tracked['coeff'],self.bev_h-1)) if left_lane_tracked['coeff'] is not None else 1e9
            dr=abs(d['x_bottom']-np.polyval(right_lane_tracked['coeff'],self.bev_h-1)) if right_lane_tracked['coeff'] is not None else 1e9
            if dl < dr: current_left = d
            else: current_right = d
        
        def update_lane(trk, det):
            if det:
                trk['coeff'] = det['coeff'] if trk['coeff'] is None else self.SMOOTHING_ALPHA * det['coeff'] + (1 - self.SMOOTHING_ALPHA) * trk['coeff']
                trk['age'] = 0
            else:
                trk['age'] += 1
                if trk['age'] > self.MAX_LANE_AGE: trk['coeff'] = None
        update_lane(left_lane_tracked, current_left)
        update_lane(right_lane_tracked, current_right)

        final_left_coeff, final_right_coeff = left_lane_tracked['coeff'], right_lane_tracked['coeff']
        lane_detected_bool = (final_left_coeff is not None) or (final_right_coeff is not None)
        self.pub_lane_status.publish(Bool(data=lane_detected_bool))
        
        # 4. 중앙 경로 계산 및 Pure Pursuit 조향 제어
        steering_angle_deg, goal_point_vehicle, final_center_coeff = None, None, None
        if lane_detected_bool:
            center_points = []
            LANE_WIDTH_M = 1.5
            lane_width_pixels = LANE_WIDTH_M / self.m_per_pixel_x
            
            for y in range(self.bev_h - 1, self.bev_h // 2, -1):
                x_center = None
                if final_left_coeff is not None and final_right_coeff is not None:
                    x_center = (np.polyval(final_left_coeff, y) + np.polyval(final_right_coeff, y)) / 2
                elif final_left_coeff is not None:
                    x_center = np.polyval(final_left_coeff, y) + lane_width_pixels / 2
                elif final_right_coeff is not None:
                    x_center = np.polyval(final_right_coeff, y) - lane_width_pixels / 2
                if x_center is not None: center_points.append([x_center, y])

            target_center_lane_coeff = None
            if len(center_points) > 10:
                p = np.array(center_points)
                target_center_lane_coeff = polyfit_lane(p[:, 1], p[:, 0], order=2)

            if target_center_lane_coeff is not None:
                if self.tracked_center_path['coeff'] is None: self.tracked_center_path['coeff'] = target_center_lane_coeff
                else: self.tracked_center_path['coeff'] = (self.SMOOTHING_ALPHA * target_center_lane_coeff + (1 - self.SMOOTHING_ALPHA) * self.tracked_center_path['coeff'])
            final_center_coeff = self.tracked_center_path['coeff']
            
            if final_center_coeff is not None:
                # 동적 전방주시거리 계산
                throttle_range = self.THROTTLE_MAX - self.THROTTLE_MIN
                normalized_throttle = (self.current_throttle - self.THROTTLE_MIN) / throttle_range if throttle_range > 0 else 0
                dynamic_lookahead_distance = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE) * normalized_throttle
                
                # 목표 지점 탐색
                for y_bev in range(self.bev_h - 1, -1, -1):
                    x_bev = np.polyval(final_center_coeff, y_bev)
                    x_veh, y_veh_right = self.image_to_vehicle((x_bev, y_bev))
                    dist = sqrt(x_veh**2 + y_veh_right**2)
                    if dist >= dynamic_lookahead_distance:
                        goal_point_vehicle = (x_veh, y_veh_right)
                        break
                
                if goal_point_vehicle is not None:
                    x_goal, y_goal = goal_point_vehicle
                    steering_angle_rad = atan2(2.0 * self.L * y_goal, x_goal**2 + y_goal**2)
                    steering_angle_deg = -np.degrees(steering_angle_rad)
                    steering_angle_deg = np.clip(steering_angle_deg, -25.0, 25.0)
                    self.pub_steering.publish(Float32(data=steering_angle_deg))

        # 5. Forced RRT 로직 (roboflow_final.py에서 가져옴)
        # BEV ROI (관심 영역)는 차량 좌표계 기준 (단위: 미터)
        x_min, x_max = 1.41, 3.5
        y_min, y_max = -1.22, 1.22
        cone_in_bev = any(x_min <= x <= x_max and y_min <= y <= y_max for x, y in self.cones)
        forced_rrt = lane_detected_bool and cone_in_bev
        self.pub_forced_rrt.publish(Bool(data=forced_rrt))
        
        # 6. RViz 시각화 데이터 발행 (roboflow_final.py에서 가져옴)
        self.publish_viz(final_left_coeff, final_right_coeff, final_center_coeff, goal_point_vehicle)

        # 7. 화면 디버깅 및 시각화
        bev_im_for_drawing = bev_image_input.copy()
        overlay_polyline(bev_im_for_drawing, final_left_coeff, color=(255, 0, 0), step=2, thickness=2)  # 왼쪽 차선 파란색
        overlay_polyline(bev_im_for_drawing, final_right_coeff, color=(0, 0, 255), step=2, thickness=2)  # 오른쪽 차선 빨간색
        if final_center_coeff is not None:
            overlay_polyline(bev_im_for_drawing, final_center_coeff, color=(0, 255, 0), step=2, thickness=3)

        if goal_point_vehicle is not None:
            # 차량 좌표계의 목표점을 다시 BEV 이미지 좌표로 변환하여 표시
            x_g, y_g = goal_point_vehicle
            u_g = self.bev_w / 2 - y_g / self.m_per_pixel_x
            v_g = self.bev_h - (x_g - self.y_offset_m) / self.m_per_pixel_y
            cv2.circle(bev_im_for_drawing, (int(u_g), int(v_g)), 10, (0, 255, 255), -1)

        txt = lambda s, y: cv2.putText(bev_im_for_drawing, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        txt(f"Steer: {steering_angle_deg:.1f}" if steering_angle_deg is not None else "Steer: N/A", 30)
        txt(f"Lane Detected: {lane_detected_bool}", 60)
        lookahead = self.MIN_LOOKAHEAD_DISTANCE + (self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE) * ((self.current_throttle - self.THROTTLE_MIN) / (self.THROTTLE_MAX - self.THROTTLE_MIN) if (self.THROTTLE_MAX - self.THROTTLE_MIN) > 0 else 0)
        txt(f"Lookahead: {lookahead:.2f}m", 90)
        txt(f"Throttle: {self.current_throttle:.2f}", 120)
        txt(f"Cones in BEV: {cone_in_bev}", 150)
        txt(f"Forced RRT: {forced_rrt}", 180)

        cv2.imshow("Original Camera View", im0s)
        cv2.imshow("Final Path (on BEV)", bev_im_for_drawing)
        cv2.waitKey(1)

    # --------- RViz 시각화 함수 (roboflow_final.py에서 가져와 수정) --------- #
    def publish_viz(self, cL, cR, cC, goal):
        now = rospy.Time.now()
        marr = MarkerArray()

        def lane_m(coeff, idx, color):
            m = Marker()
            m.header.stamp, m.header.frame_id = now, "camera"
            m.ns, m.id = "lane", idx
            if coeff is None:
                m.action = Marker.DELETE; return m
            m.action = Marker.ADD; m.type = Marker.LINE_STRIP; m.scale.x = 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = *color, 1.0; m.pose.orientation.w = 1.0
            for y in range(self.bev_h - 1, self.bev_h // 2, -8):
                x = np.polyval(coeff, y)
                xv, yv = self.image_to_vehicle((x, y))
                m.points.append(Point(x=xv, y=yv, z=0.0))
            return m
        marr.markers.extend([lane_m(cL, 0, (1, 0, 0)), lane_m(cR, 1, (0, 0, 1))])

        bev_area_marker = Marker()
        bev_area_marker.header.stamp, bev_area_marker.header.frame_id = now, "camera"
        bev_area_marker.ns, bev_area_marker.id = "bev_roi_area", 100
        bev_area_marker.type, bev_area_marker.action = Marker.LINE_STRIP, Marker.ADD
        bev_area_marker.pose.orientation.w = 1.0
        bev_area_marker.scale.x = 0.03
        bev_area_marker.color.r, bev_area_marker.color.g, bev_area_marker.color.b, bev_area_marker.color.a = 0.0, 1.0, 1.0, 1.0
        bev_area_marker.points = [
            Point(x=2.95, y=-1.22, z=0.0), Point(x=2.95, y=1.22, z=0.0),
            Point(x=1.41, y=1.22, z=0.0), Point(x=1.41, y=-1.22, z=0.0),
            Point(x=2.95, y=-1.22, z=0.0)
        ]
        marr.markers.append(bev_area_marker)
        self.pub_markers.publish(marr)

        p = Path(); p.header.stamp, p.header.frame_id = now, "camera"
        if cC is not None:
            for y in range(self.bev_h - 1, self.bev_h // 2, -6):
                x = np.polyval(cC, y)
                xv, yv = self.image_to_vehicle((x, y))
                ps = PoseStamped(); ps.header = p.header; ps.pose.position.x, ps.pose.position.y = xv, yv; ps.pose.orientation.w = 1.0
                p.poses.append(ps)
        self.pub_path.publish(p)

        mk = Marker(); mk.header.stamp, mk.header.frame_id = now, "camera"; mk.ns, mk.id = "lookahead", 0
        if goal is None:
            mk.action = Marker.DELETE
        else:
            mk.action = Marker.ADD; mk.type = Marker.SPHERE
            mk.scale.x = mk.scale.y = mk.scale.z = 0.25
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1, 1, 0, 1
            mk.pose.position.x, mk.pose.position.y = goal; mk.pose.orientation.w = 1.0
        self.pub_look.publish(mk)

# ────────── 메인 함수 ────────── #
def main():
    rospy.init_node('lane_follower_node', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights3.pt')
    parser.add_argument('--device', default='0')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.6)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--param-file', type=str, default='./bev_params_y_5.npz')
    opt, _ = parser.parse_known_args()

    node = LaneFollowerNode(opt)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down node.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()