#!/usr/bin/env python3
import rospy
import rospkg
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

def add_cones_to_csv():
    rospy.init_node("add_cones_to_csv_node", anonymous=True)

    # ROS 패키지 경로 가져오기
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('path_planning')
    csv_path = os.path.join(package_path, "data/example1.csv")  # RDDF 경로 파일
    new_csv_path = os.path.join(package_path, "data/cones_with_obstacles2.csv")  # 저장할 콘 위치 파일

    try:
        # CSV 파일 불러오기
        data = pd.read_csv(csv_path)

        # X, Y 좌표 읽기 (plot_csv.py와 동일한 방식)
        if "x" in data.columns and "y" in data.columns:
            x_coords = data["x"].to_numpy()
            y_coords = data["y"].to_numpy()
        else:
            x_coords = data.iloc[:, 0].values  # X 좌표 (첫 번째 열)
            y_coords = data.iloc[:, 1].values  # Y 좌표 (두 번째 열)

        # 추가할 콘 좌표 리스트
        cone_positions = []

        # 마우스 클릭 이벤트 처리 함수
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                x, y = event.xdata, event.ydata
                cone_positions.append((x, y))
                print(f"콘 추가: ({x:.2f}, {y:.2f})")

                # 그래프 위에 점 추가 (버튼 위가 아닌 그래프에 표시)
                ax.scatter(x, y, color='r', marker='o', label="Cone Position" if len(cone_positions) == 1 else "")
                ax.figure.canvas.draw()  # 그래프 업데이트

        # CSV 저장 함수 (example1.csv와 동일한 구조 유지)
        def save_csv():
            if len(cone_positions) == 0:
                print("추가된 콘 위치가 없습니다.")
                return

            # 새로운 CSV 파일로 저장
            with open(new_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])  # 헤더 추가
                for pos in cone_positions:
                    writer.writerow([pos[0], pos[1]])

            print(f"{len(cone_positions)}개의 콘 위치가 {new_csv_path} 에 저장되었습니다.")

        # Matplotlib 윈도우 생성
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_coords, y_coords, marker='o', linestyle='-', markersize=4, label="Waypoints (Global Path)")
        ax.set_xlabel("X 좌표 (m)")
        ax.set_ylabel("Y 좌표 (m)")
        ax.set_title("Click to Add Cone Positions")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")  # ✅ plot_csv.py와 동일한 축 스케일 적용

        # 클릭 이벤트 연결
        fig.canvas.mpl_connect('button_press_event', onclick)

        # 버튼을 오른쪽 아래에 배치
        plt.subplots_adjust(right=0.8)  # 버튼이 그래프를 가리지 않도록 조정
        ax_save = plt.axes([0.85, 0.05, 0.1, 0.05])
        button_save = plt.Button(ax_save, 'Save CSV')
        button_save.on_clicked(lambda event: save_csv())

        # 그래프 표시
        plt.show()

    except Exception as e:
        rospy.logerr(f"Failed to load or plot CSV: {e}")

if __name__ == "__main__":
    add_cones_to_csv()
