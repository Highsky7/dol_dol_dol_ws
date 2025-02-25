#!/usr/bin/env python3
import rospy
import sys, select, termios, tty
import math
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

# 초기 터미널 설정 저장
settings = termios.tcgetattr(sys.stdin)

def getKey(timeout):
    """키 입력을 non-blocking으로 읽습니다."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def print_instructions():
    print("Teleop Keyboard Control:")
    print("------------------------")
    print("w : accelerate (increase speed by 0.1 m/s)")
    print("s : decelerate/brake (decrease speed by 0.1 m/s)")
    print("a : steer left (increase steering angle by 0.1 rad)")
    print("d : steer right (decrease steering angle by 0.1 rad)")
    print("q : quit")
    print("------------------------")
    print("Initial speed is 0 m/s, and initial yaw is set via ROS parameter '~initial_yaw' (default 90°).\n")

def teleop():
    rospy.init_node("teleop_keyboard_node", anonymous=True)
    
    drive_pub = rospy.Publisher("ackermann_cmd", AckermannDriveStamped, queue_size=10)
    yaw_pub = rospy.Publisher("vehicle_yaw", Float32, queue_size=10)
    
    steering_angle = 0.0    # radians
    speed = rospy.get_param("~initial_speed", 0.0)  # m/s, 초기 속도 0.0
    yaw = rospy.get_param("~initial_yaw", 90.0)       # 초기 헤딩, default 90° (north) → rad 변환 후 사용
    yaw = math.radians(yaw)
    wheelbase = 3.5         # 휠베이스
    
    rate = rospy.Rate(10)
    dt = 0.1
    
    print_instructions()
    
    try:
        while not rospy.is_shutdown():
            key = getKey(0.1)
            
            if key == 'w':        # 가속
                speed += 0.1
            elif key == 's':      # 감속/브레이크
                speed = max(0.0, speed - 0.1)
            elif key == 'a':      # 좌회전: steering angle 증가 (양수)
                steering_angle += 0.1
            elif key == 'd':      # 우회전: steering angle 감소 (음수)
                steering_angle -= 0.1
            elif key == 'q':      # 종료
                break
            
            # 간단한 bicycle 모델을 통한 yaw 업데이트
            yaw_rate = speed / wheelbase * math.tan(steering_angle)
            yaw += yaw_rate * dt
            
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.drive.steering_angle = steering_angle
            drive_msg.drive.speed = speed
            
            drive_pub.publish(drive_msg)
            yaw_pub.publish(Float32(data=yaw))
            
            rospy.loginfo("Steering Angle: {:.2f} rad, Speed: {:.2f} m/s, Yaw: {:.2f} rad".format(
                steering_angle, speed, yaw))
            
            rate.sleep()
    
    except Exception as e:
        rospy.logerr("Exception in teleop: {}".format(e))
    
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == "__main__":
    teleop()
