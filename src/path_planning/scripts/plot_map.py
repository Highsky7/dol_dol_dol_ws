#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

def bspline_interpolation_distance(x, y, sampling_interval, smoothing=0.5):
    """
    B-Spline 보간 후, 곡선 상의 거리를 기준으로 등간격 샘플링하는 함수.
    :param x: 원본 x 좌표 배열
    :param y: 원본 y 좌표 배열
    :param sampling_interval: 경로 상의 샘플링 간격 (예: 2m)
    :param smoothing: B-Spline smoothing factor (값이 클수록 더 부드럽게)
    :return: 등간격 샘플링된 x, y 좌표 배열
    """
    # B-Spline 곡선 피팅
    tck, u = splprep([x, y], s=smoothing)
    
    # u를 fine하게 샘플링하여 곡선의 arc length (누적 거리)를 근사 계산
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine = splev(u_fine, tck)
    
    # 각 세그먼트의 길이 및 누적 거리 계산
    dx = np.diff(x_fine)
    dy = np.diff(y_fine)
    ds = np.sqrt(dx**2 + dy**2)
    s_fine = np.insert(np.cumsum(ds), 0, 0)  # 누적 거리, 시작점을 0으로 추가
    total_length = s_fine[-1]
    
    # 총 길이를 기준으로 sampling_interval 간격의 거리를 생성
    num_samples = int(total_length / sampling_interval) + 1
    s_new = np.linspace(0, total_length, num_samples)
    
    # 등간격 거리에 대응하는 u 값을 선형 보간법으로 찾음
    u_new = np.interp(s_new, s_fine, u_fine)
    
    # 새로운 u 값에 따른 보간 좌표 생성
    x_new, y_new = splev(u_new, tck)
    return x_new, y_new

def plot_csv():
    rospy.init_node("plot_csv_node", anonymous=True)

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('path_planning')

    # CSV 파일 경로
    csv_path_example = package_path + "/data/example1.csv"  # 대회측 제공 GPS RDDF 파일 (global path)
    csv_path_cone    = package_path + "/data/cones.csv"       # cones.csv는 건들지 않음

    try:
        # CSV 파일 읽기 및 NaN 값 제거
        data_example = pd.read_csv(csv_path_example).dropna()
        data_cone = pd.read_csv(csv_path_cone).dropna()

        # X, Y 데이터 추출
        x_example = data_example['x'].to_numpy()
        y_example = data_example['y'].to_numpy()

        x_cone = data_cone['x'].to_numpy()
        y_cone = data_cone['y'].to_numpy()

        # ROS 파라미터로부터 샘플링 간격(기본값: 2m) 읽기
        sampling_interval = rospy.get_param("~sampling_interval", 2.0)
        rospy.loginfo(f"Using sampling interval: {sampling_interval} meters")
        
        # B-Spline 보간 후, 거리 기반 등간격 샘플링 수행 (global path)
        x_bspline, y_bspline = bspline_interpolation_distance(x_example, y_example, sampling_interval, smoothing=3.0)

        # 플롯 크기 설정
        plt.figure(figsize=(12, 8))

        # --- 원본 global path 플롯 (비교용) ---
        # 아래 코드는 필요시 주석 해제하여 원본 경로와 보간 경로를 함께 비교할 수 있습니다.
        # plt.plot(x_example, y_example, marker='o', linestyle='-', markersize=4, color='blue', label="Original Global Path")

        # B-Spline 보간된 global path 플롯 (실행 시 표시)
        plt.plot(x_bspline, y_bspline, marker='o', linestyle='-', markersize=4, color='green', label="B-Spline Global Path")

        # cones.csv 데이터 플롯 (건들지 않음)
        plt.plot(x_cone, y_cone, marker='o', linestyle='None', markersize=4, color='red', label="Cones")

        # 데이터 범위에 따라 축 조정
        min_x = min(np.min(x_example), np.min(x_cone))
        max_x = max(np.max(x_example), np.max(x_cone))
        min_y = min(np.min(y_example), np.min(y_cone))
        max_y = max(np.max(y_example), np.max(y_cone))
        margin = 10
        plt.xlim(min_x - margin, max_x + margin)
        plt.ylim(min_y - margin, max_y + margin)

        plt.xlabel("X 좌표 (m)")
        plt.ylabel("Y 좌표 (m)")
        plt.title("B-Spline Interpolated Global Path and Cones")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        rospy.logerr(f"Failed to load or plot CSV: {e}")

if __name__ == "__main__":
    plot_csv()
