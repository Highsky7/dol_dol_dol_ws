#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv():
    rospy.init_node("plot_csv_node", anonymous=True)

    # ROS 패키지 경로 가져오기
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('path_planning')
    csv_path = package_path + "/data/example1.csv"

    try:
        # CSV 파일 불러오기
        data = pd.read_csv(csv_path)

        # X, Y 좌표 추출 (오류 수정: .values → .to_numpy())
        x = data['x'].to_numpy()
        y = data['y'].to_numpy()

        # 플로팅
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='None', markersize=4, label="Waypoints")

        # 그래프 설정
        plt.xlabel("X 좌표 (m)")
        plt.ylabel("Y 좌표 (m)")
        plt.title("Waypoints Plot")
        plt.legend()
        plt.grid(True)

        # 그래프 표시
        plt.show()
    
    except Exception as e:
        rospy.logerr(f"Failed to load or plot CSV: {e}")

if __name__ == "__main__":
    plot_csv()
