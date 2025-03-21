#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
from scipy.interpolate import splprep, splev

def bspline_interpolation_distance(x, y, sampling_interval, smoothing=0.5):
    """
    B-Spline 보간 후, 곡선 상의 누적 거리를 기준으로 등간격 샘플링하는 함수.
    
    수행 단계:
      1. 원본 (x, y) 데이터를 splprep를 통해 B-Spline 곡선(tck)으로 피팅.
         - smoothing 값이 0이면 원본 데이터를 정확히 통과하며, 값이 클수록 곡선이 부드러워짐.
      2. u 파라미터를 0~1 범위에서 1000개 점으로 샘플링하여 곡선상의 (x, y) 좌표 계산.
      3. 인접 좌표 간 유클리드 거리를 계산해 누적 거리(s_fine)를 생성.
      4. 전체 곡선 길이(total_length)를 기준으로, sampling_interval 간격의 거리 값 생성.
      5. np.interp를 통해 생성한 등간격 누적 거리 값에 대응하는 u 값을 계산,
         그리고 이를 통해 최종 보간 좌표 (x_new, y_new)를 산출.
    
    Parameters:
      - x, y: 원본 GPS 좌표 배열 (numpy arrays)
      - sampling_interval: 등간격 샘플링할 거리 간격 (미터 단위), 예: 2.0
      - smoothing: B-Spline 보간 시 smoothing factor (0이면 원본 그대로, 값이 클수록 부드러워짐)
    
    Returns:
      - x_new, y_new: 보간 및 등간격 샘플링된 좌표 배열
    """
    # 1. B-Spline 곡선 피팅
    tck, u = splprep([x, y], s=smoothing)
    
    # 2. u를 fine하게 샘플링 (0~1 구간을 1000개 점으로 분할)
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine = splev(u_fine, tck)
    
    # 3. 인접 점 사이의 유클리드 거리 계산 및 누적 거리 생성
    dx = np.diff(x_fine)
    dy = np.diff(y_fine)
    ds = np.sqrt(dx**2 + dy**2)
    s_fine = np.insert(np.cumsum(ds), 0, 0)
    total_length = s_fine[-1]
    
    # 4. 전체 길이(total_length)를 기준으로 등간격 거리 값 생성
    num_samples = int(total_length / sampling_interval) + 1
    s_new = np.linspace(0, total_length, num_samples)
    
    # 5. s_new에 대응하는 u 값을 선형 보간으로 계산 후 보간 좌표 생성
    u_new = np.interp(s_new, s_fine, u_fine)
    x_new, y_new = splev(u_new, tck)
    return x_new, y_new

def draw_vehicle(ax, x=0, y=0, yaw=0):
    """
    현실적인 2D 차량 모델을 그리는 함수 (후륜축 중심 기준)
      - x, y: 차량의 후륜축 중심의 global 좌표
      - yaw: 헤딩 각도 (rad). yaw=0이면 차량의 전방(+x, LiDAR 기준)이 동쪽을 향함.
    
    실제 차량 환경:
      - 후륜축 중심 z 선상에 VLP-16 3D 라이다와 그 위에 GPS 안테나가 설치됨.
      - GPS만으로는 동서남북 구분이 불가능하므로, 라이다의 좌표계 
        (전방이 +x, 좌측이 +y)를 따릅니다.
      - 하드웨어적으로 라이다 전방(+x)이 동쪽에 위치하도록 정렬하여,
        소프트웨어에서의 yaw=0이 동쪽을 의미하게 됩니다.
    """
    # 차량 치수 (예: 승용차)
    Lr = 1.0    # 후륜축에서 후방 범퍼까지 거리
    Lf = 2.5    # 후륜축에서 전방 범퍼까지 거리
    vehicle_length = Lr + Lf  # 총 차량 길이 (예: 3.5m)
    vehicle_width = 1.8       # 차량 폭
    
    # 차량 본체: 로컬 좌표계(후륜축 중심 기준)에서 다각형으로 표현
    car_coords = [
        [-Lr, -vehicle_width/2],
        [-Lr,  vehicle_width/2],
        [ Lf,  vehicle_width/2],
        [ Lf, -vehicle_width/2]
    ]
    car_body = patches.Polygon(car_coords, closed=True, edgecolor='black', facecolor='silver', zorder=2)
    
    # 로컬 차량을 yaw만큼 회전시킨 후 (x, y)로 평행이동
    t = transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
    car_body.set_transform(t)
    ax.add_patch(car_body)
    
    # 바퀴 표현 (직사각형)
    wheel_length = 0.7
    wheel_width = 0.3
    wheel_offsets = [
        [ Lf - wheel_length, -vehicle_width/2 - wheel_width],  # 전륜 좌측
        [ Lf - wheel_length,  vehicle_width/2],                 # 전륜 우측
        [-Lr, -vehicle_width/2 - wheel_width],                  # 후륜 좌측
        [-Lr,  vehicle_width/2]                                  # 후륜 우측
    ]
    for wx, wy in wheel_offsets:
        wheel_rect = patches.Rectangle((wx, wy), wheel_length, wheel_width,
                                       edgecolor='black', facecolor='black', zorder=3)
        wheel_rect.set_transform(t)
        ax.add_patch(wheel_rect)
    
    # 차량 기준점(후륜축 중심)을 파란 원으로 표시
    ax.plot(x, y, 'bo', markersize=6, zorder=4)
    
    # 헤딩 방향 화살표: 전방(+x 방향, Lf 길이) 표시
    arrow_length = Lf
    ax.arrow(x, y, arrow_length * np.cos(yaw), arrow_length * np.sin(yaw),
             head_width=0.3, head_length=0.3, fc='red', ec='red', zorder=5)
    
    # 전방에 'F' 텍스트 표시
    front_x = x + (Lf) * np.cos(yaw)
    front_y = y + (Lf) * np.sin(yaw)
    ax.text(front_x, front_y, "F", fontsize=12, color='red', zorder=6)

def plot_map():
    """
    대회측 제공 global 경로 (example1.csv)를 B-Spline 보간 후, 
    cones.csv와 함께 플롯하고, global 경로 상의 시작 위치에 차량을 정적으로 표시하는 함수.
    
    주요 기능:
      - GPS RDDF 파일에서 global 경로 데이터를 읽어들임.
      - B-Spline 보간 및 거리 기반 등간격 샘플링 수행 (ROS 파라미터 "~sampling_interval"으로 간격 조정, 기본값: 2.0m)
      - Global 경로의 각 GPS 좌표를 녹색 점으로 표시함.
      - 시작 위치에 차량을 표시하는데, 초기 헤딩(yaw)은 ROS 파라미터 "~initial_yaw"를 사용하여 설정.
        * 여기서 yaw=0은 차량의 전방(+x, 라이다 기준)이 동쪽(양의 x축)을 의미함.
        * 이는 하드웨어적으로 라이다의 전방이 동쪽에 정렬되어 있기 때문입니다.
    """
    rospy.init_node("plot_map_node", anonymous=True)
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('path_planning')
    
    # CSV 파일 경로 설정
    csv_path_example = package_path + "/data/example1.csv"  # Global path (GPS RDDF)
    csv_path_cone    = package_path + "/data/cones.csv"       # Cones 데이터 (건들지 않음)
    
    try:
        # CSV 파일 읽기 및 NaN 값 제거
        data_example = pd.read_csv(csv_path_example).dropna()
        data_cone = pd.read_csv(csv_path_cone).dropna()
        
        # Global 경로 좌표 추출
        x_example = data_example['x'].to_numpy()
        y_example = data_example['y'].to_numpy()
        
        # B-Spline 보간 및 거리 기반 등간격 샘플링 수행
        sampling_interval = rospy.get_param("~sampling_interval", 2.0)  # (미터 단위)
        x_global, y_global = bspline_interpolation_distance(x_example, y_example, sampling_interval, smoothing=1.0)
        
        # 시작 위치: global 경로의 첫 번째 샘플 포인트
        start_x, start_y = x_global[3], y_global[3] # 실제로는 첫 gps 좌표 위치
        
        # 초기 헤딩은 GPS만으로는 구할 수 없으므로, 외부(라이다/LOAM 기반)에서 얻은 값을 사용.
        # ROS 파라미터 "~initial_yaw"로 설정 (기본값 0.0; yaw=0이면 차량 전방(+x)이 동쪽을 의미)
        yaw = rospy.get_param("~initial_yaw", 0.0)
        
        # Plot 설정
        plt.figure(figsize=(12, 8))
        # Global path: 각 GPS 좌표를 녹색 점으로 표시
        plt.plot(x_global, y_global, 'go', label="Global Path", zorder=1)
        # Cones: 빨간 점으로 표시 (cones.csv 데이터)
        plt.plot(data_cone['x'].to_numpy(), data_cone['y'].to_numpy(), 'ro', label="Cones", zorder=1)
        
        plt.xlabel("X 좌표 (m)")
        plt.ylabel("Y 좌표 (m)")
        plt.title("Map with Vehicle at Start Position")
        plt.grid(True)
        plt.legend()
        
        # 시작 위치에 차량 표시
        draw_vehicle(plt.gca(), start_x, start_y, yaw)
        
        plt.show()
    
    except Exception as e:
        rospy.logerr(f"Failed to load or plot map: {e}")

if __name__ == "__main__":
    plot_map()
