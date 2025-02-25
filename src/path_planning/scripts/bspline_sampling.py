#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def load_rddf_data():
    """RDDF 데이터를 로드하는 함수"""
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('path_planning')
    csv_path = package_path + "/data/example1.csv"

    try:
        data = pd.read_csv(csv_path)
        x = data['x'].to_numpy()
        y = data['y'].to_numpy()
        return x, y
    except Exception as e:
        rospy.logerr(f"Failed to load CSV: {e}")
        return None, None

def bspline_interpolation(x, y, sampling_interval, s=10.0):
    """B-Spline을 통해 등간격 샘플링을 수행하는 함수 (s 값을 인자로 추가)"""
    # B-Spline 보간을 위한 곡선 피팅: s 값을 증가시켜 보간 결과를 부드럽게 함
    tck, u = splprep([x, y], s=s)
    
    total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))  # 전체 경로 길이
    num_samples = int(total_length / sampling_interval)
    u_new = np.linspace(0, 1, num_samples)
    
    x_new, y_new = splev(u_new, tck)
    return x_new, y_new


def plot_path(original_x, original_y, sampled_x, sampled_y):
    """원본 경로와 샘플링된 경로를 시각화하는 함수"""
    plt.figure(figsize=(8, 6))
    plt.plot(original_x, original_y, 'r--', label="Original Path", alpha=0.6)
    plt.plot(sampled_x, sampled_y, 'bo', markersize=4, label="B-Spline Sampled Path")

    plt.xlabel("X 좌표 (m)")
    plt.ylabel("Y 좌표 (m)")
    plt.title("B-Spline Based Path Sampling")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    rospy.init_node("bspline_sampling_node", anonymous=True)

    # 샘플링 간격을 ROS 파라미터로 설정 (기본값: 2m)
    sampling_interval = rospy.get_param("~sampling_interval", 2.0)
    rospy.loginfo(f"Using sampling interval: {sampling_interval} meters")

    # RDDF 데이터 로드
    x, y = load_rddf_data()
    if x is None or y is None:
        return

    # B-Spline 보간 및 샘플링
    sampled_x, sampled_y = bspline_interpolation(x, y, sampling_interval)

    # 플로팅
    plot_path(x, y, sampled_x, sampled_y)

if __name__ == "__main__":
    main()