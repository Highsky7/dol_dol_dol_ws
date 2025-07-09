#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import matplotlib.pyplot as plt
import math
import numpy as np # 이동 평균 계산을 위해 numpy 추가

# --- 사용자 설정 변수 ---
# 여기에 분석할 rosbag 파일의 전체 경로를 입력하세요.
BAG_FILE_PATH = '/home/highsky/my_data_2025-07-09-15-55-00.bag'
# 속도 토픽 이름
VELOCITY_TOPIC = '/ublox_gps/fix_velocity'
# Throttle 토픽 이름
THROTTLE_TOPIC = '/auto_throttle'
# 이동 평균 윈도우 크기 (이 값을 조절하여 부드러운 정도를 변경할 수 있습니다)
MOVING_AVG_WINDOW = 20
# -------------------------

def moving_average(data, window_size):
    """주어진 데이터에 대해 이동 평균을 계산합니다."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def plot_velocity_and_throttle(bag_file, vel_topic, throttle_topic, window_size):
    """
    rosbag 파일에서 데이터를 추출하고, '원본 데이터'와 '이동 평균' 그래프를 각각 생성합니다.
    """
    # 데이터 저장을 위한 리스트 초기화
    velocity_times = []
    linear_velocities = []
    throttle_times = []
    throttle_values = []

    print(f"'{bag_file}' 파일을 읽는 중입니다...")

    try:
        bag = rosbag.Bag(bag_file, 'r')
        target_topics = [vel_topic, throttle_topic]

        for topic, msg, t in bag.read_messages(topics=target_topics):
            if topic == vel_topic:
                linear_vel = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
                velocity_times.append(t.to_sec())
                linear_velocities.append(linear_vel)
            elif topic == throttle_topic:
                throttle_values.append(msg.data)
                throttle_times.append(t.to_sec())

    except Exception as e:
        print(f"오류 발생: {e}")
        return
    finally:
        if 'bag' in locals() and bag:
            bag.close()
            print("Bag 파일 처리가 완료되었습니다.")

    if not velocity_times or not throttle_times:
        print("그래프를 그릴 데이터가 충분하지 않습니다. 토픽 이름이나 bag 파일을 확인해주세요.")
        return

    # --- 1. 원본 데이터 그래프 그리기 ---
    fig1, ax1_raw = plt.subplots(figsize=(15, 7))
    ax1_raw.set_title('Raw Data Comparison: Velocity vs. Throttle', fontsize=16)

    # 첫 번째 Y축 (원본 선속도)
    color = 'tab:blue'
    ax1_raw.set_xlabel('Time (s)')
    ax1_raw.set_ylabel('Linear Velocity (m/s)', color=color)
    ax1_raw.plot(velocity_times, linear_velocities, color=color, alpha=0.8, label='Raw Velocity')
    ax1_raw.tick_params(axis='y', labelcolor=color)
    ax1_raw.grid(True)

    # 두 번째 Y축 (원본 Throttle)
    ax2_raw = ax1_raw.twinx()
    color = 'tab:red'
    ax2_raw.set_ylabel('Throttle Value', color=color)
    ax2_raw.plot(throttle_times, throttle_values, color=color, alpha=0.7, linestyle='--', label='Raw Throttle')
    ax2_raw.tick_params(axis='y', labelcolor=color)
    
    # 범례 설정
    lines, labels = ax1_raw.get_legend_handles_labels()
    lines2, labels2 = ax2_raw.get_legend_handles_labels()
    ax2_raw.legend(lines + lines2, labels + labels2, loc='upper left')


    # --- 2. 이동 평균 추세선 그래프 그리기 ---
    
    # 이동 평균 계산
    vel_ma, vel_times_ma, throttle_ma, throttle_times_ma = None, None, None, None
    if len(linear_velocities) > window_size:
        vel_ma = moving_average(linear_velocities, window_size)
        vel_times_ma = velocity_times[window_size-1:]
    else:
        print("속도 데이터가 부족하여 이동 평균 그래프를 그릴 수 없습니다.")

    if len(throttle_values) > window_size:
        throttle_ma = moving_average(throttle_values, window_size)
        throttle_times_ma = throttle_times[window_size-1:]
    else:
        print("Throttle 데이터가 부족하여 이동 평균 그래프를 그릴 수 없습니다.")
        
    # 이동 평균 데이터가 있을 경우에만 그래프 생성
    if vel_ma is not None and throttle_ma is not None:
        fig2, ax1_ma = plt.subplots(figsize=(15, 7))
        ax1_ma.set_title(f'Moving Average (Window={window_size}) Trend Comparison', fontsize=16)

        # 첫 번째 Y축 (이동 평균 선속도)
        color = 'tab:blue'
        ax1_ma.set_xlabel('Time (s)')
        ax1_ma.set_ylabel('Linear Velocity (m/s)', color=color)
        ax1_ma.plot(vel_times_ma, vel_ma, color=color, linewidth=2.5, label=f'Velocity MA')
        ax1_ma.tick_params(axis='y', labelcolor=color)
        ax1_ma.grid(True)

        # 두 번째 Y축 (이동 평균 Throttle)
        ax2_ma = ax1_ma.twinx()
        color = 'tab:red'
        ax2_ma.set_ylabel('Throttle Value', color=color)
        ax2_ma.plot(throttle_times_ma, throttle_ma, color=color, linewidth=2.5, linestyle='--', label=f'Throttle MA')
        ax2_ma.tick_params(axis='y', labelcolor=color)

        # 범례 설정
        lines, labels = ax1_ma.get_legend_handles_labels()
        lines2, labels2 = ax2_ma.get_legend_handles_labels()
        ax2_ma.legend(lines + lines2, labels + labels2, loc='upper left')

    # 모든 그래프 창을 화면에 보여주기
    plt.show()


if __name__ == '__main__':
    plot_velocity_and_throttle(BAG_FILE_PATH, VELOCITY_TOPIC, THROTTLE_TOPIC, MOVING_AVG_WINDOW)