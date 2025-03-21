#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import numpy as np
import cv2
import math
from scipy.interpolate import splprep, splev
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, Point

# 전역 변수: 차량 상태
current_x = 0.0
current_y = 0.0
current_yaw = 0.0
current_speed = 0.0
steering_angle = 0.0

# >>> lookahead_point 좌표 저장용 (None이면 미수신 상태)
lookahead_x = None
lookahead_y = None

# Publisher
vehicle_pose_pub = None

# 시뮬레이션용 파라미터
MAP_SCALE = 6.0
IMG_SIZE = 800
OFFSET_X = 0
OFFSET_Y = 0

WHEEL_BASE = 2.5

def ackermann_cmd_callback(msg):
    global current_speed, steering_angle
    current_speed = msg.drive.speed
    steering_angle = msg.drive.steering_angle

def yaw_callback(msg):
    global current_yaw
    current_yaw = msg.data

# >>> lookahead_point 콜백
def lookahead_point_callback(msg):
    global lookahead_x, lookahead_y
    lookahead_x = msg.x
    lookahead_y = msg.y

def create_bspline_waypoints(x_coords, y_coords, sampling_interval, smoothing=0.5):
    tck, u = splprep([x_coords, y_coords], s=smoothing)
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine = splev(u_fine, tck)
    dx = np.diff(x_fine)
    dy = np.diff(y_fine)
    ds = np.sqrt(dx**2 + dy**2)
    s_fine = np.insert(np.cumsum(ds), 0, 0)
    total_length = s_fine[-1]
    num_samples = int(total_length / sampling_interval) + 1
    s_new = np.linspace(0, total_length, num_samples)
    u_new = np.interp(s_new, s_fine, u_fine)
    x_new, y_new = splev(u_new, tck)
    return x_new, y_new

def world_to_pixel(wx, wy):
    px = int(wx * MAP_SCALE + OFFSET_X)
    py = int(-wy * MAP_SCALE + OFFSET_Y)
    return px, py

def draw_points(img, x_list, y_list, color=(0,255,0), radius=4):
    for i in range(len(x_list)):
        px, py = world_to_pixel(x_list[i], y_list[i])
        cv2.circle(img, (px, py), radius, color, -1)

def draw_vehicle(img, x=0, y=0, yaw=0):
    Lr = 1.0
    Lf = 2.5
    vehicle_width = 1.8

    corners = [
        [-Lr, -vehicle_width/2],
        [-Lr,  vehicle_width/2],
        [ Lf,  vehicle_width/2],
        [ Lf, -vehicle_width/2]
    ]

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotated_points = []
    for cx, cy in corners:
        rx = cx*cos_yaw - cy*sin_yaw + x
        ry = cx*sin_yaw + cy*cos_yaw + y
        px, py = world_to_pixel(rx, ry)
        rotated_points.append((px, py))
    
    pts = np.array(rotated_points, dtype=np.int32)
    cv2.polylines(img, [pts], True, (192,192,192), 2)
    cv2.fillPoly(img, [pts], (128,128,128))

    # 전방 화살표
    fx_local = Lf
    fy_local = 0
    fx_global = fx_local*cos_yaw - fy_local*sin_yaw + x
    fy_global = fx_local*sin_yaw + fy_local*cos_yaw + y
    fx_pix, fy_pix = world_to_pixel(fx_global, fy_global)

    cx_pix, cy_pix = world_to_pixel(x, y)
    cv2.arrowedLine(img, (cx_pix, cy_pix), (fx_pix, fy_pix),
                    (0,0,255), 2, tipLength=0.3)
    cv2.circle(img, (cx_pix, cy_pix), 3, (255,0,0), -1)

def simulation_loop():
    global current_x, current_y, current_yaw, vehicle_pose_pub
    global OFFSET_X, OFFSET_Y

    rospy.init_node("only_gps_model_node", anonymous=True)
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('path_planning')
    
    csv_path_rddf = package_path + "/data/example1.csv"  # 가정
    csv_path_cone = package_path + "/data/cones_with_obstacles2.csv"
    
    try:
        data_rddf = pd.read_csv(csv_path_rddf).dropna()
        cones_data = pd.read_csv(csv_path_cone).dropna()
    except Exception as e:
        rospy.logerr("Failed to read CSV files: {}".format(e))
        return

    rddf_x = data_rddf['x'].to_numpy()
    rddf_y = data_rddf['y'].to_numpy()

    sampling_interval = rospy.get_param("~sampling_interval", 1.0)
    spline_x, spline_y = create_bspline_waypoints(
        rddf_x, rddf_y, sampling_interval, smoothing=0.5
    )

    min_x = min(np.min(rddf_x), np.min(spline_x))
    max_x = max(np.max(rddf_x), np.max(spline_x))
    min_y = min(np.min(rddf_y), np.min(spline_y))
    max_y = max(np.max(rddf_y), np.max(spline_y))

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    OFFSET_X = IMG_SIZE // 2 - int(center_x * MAP_SCALE)
    OFFSET_Y = IMG_SIZE // 2 + int(center_y * MAP_SCALE)

    current_x = spline_x[0]
    current_y = spline_y[0]
    current_yaw = math.radians(rospy.get_param("~initial_yaw", 90.0))
    
    # 기존 Subscriber들
    rospy.Subscriber("ackermann_cmd", AckermannDriveStamped, ackermann_cmd_callback)
    rospy.Subscriber("vehicle_yaw", Float32, yaw_callback)

    # >>> lookahead_point 구독 (PurePursuit에서 발행)
    rospy.Subscriber("lookahead_point", Point, lookahead_point_callback)

    vehicle_pose_pub = rospy.Publisher("vehicle_pose", PoseStamped, queue_size=10)
    
    rate = rospy.Rate(10)
    dt = 0.1

    while not rospy.is_shutdown():
        yaw_rate = (current_speed / WHEEL_BASE) * math.tan(steering_angle)
        current_yaw += yaw_rate * dt
        current_x += current_speed * math.cos(current_yaw) * dt
        current_y += current_speed * math.sin(current_yaw) * dt
        
        # PoseStamped 발행
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = current_x
        pose_msg.pose.position.y = current_y
        half_yaw = current_yaw / 2.0
        quat = [0, 0, math.sin(half_yaw), math.cos(half_yaw)]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        vehicle_pose_pub.publish(pose_msg)

        img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

        # 원본 RDDF 점(파란색)
        draw_points(img, rddf_x, rddf_y, color=(255,0,0), radius=5)

        # B-Spline 점(초록색)
        draw_points(img, spline_x, spline_y, color=(0,255,0), radius=4)

        # 콘(빨간색)
        draw_points(img,
                    cones_data['x'].to_numpy(),
                    cones_data['y'].to_numpy(),
                    color=(0,0,255),
                    radius=5)

        # 차량
        draw_vehicle(img, current_x, current_y, current_yaw)

        # 스티어링/속도/헤딩 텍스트
        steering_deg = math.degrees(steering_angle)
        yaw_deg = math.degrees(current_yaw)
        cv2.putText(img, "Speed: {:.2f} m/s".format(current_speed),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
        cv2.putText(img, "Steering: {:.2f} deg".format(steering_deg),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
        cv2.putText(img, "Yaw: {:.2f} deg".format(yaw_deg),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)

        # >>> lookahead_point 시각화 (마젠타색 원 등)
        if lookahead_x is not None and lookahead_y is not None:
            lx, ly = world_to_pixel(lookahead_x, lookahead_y)
            cv2.circle(img, (lx, ly), 6, (255,0,255), -1)  # 분홍색

        cv2.imshow("OpenCV Simulation", img)
        cv2.waitKey(1)

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        simulation_loop()
    except rospy.ROSInterruptException:
        pass
