#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32, Bool

class Judgement:
    def __init__(self):
        # 구독자 설정: ROS 토픽에서 데이터를 수신
        self.sub_lane = rospy.Subscriber('auto_steer_angle_lane', Float32, self.lane_callback)
        self.sub_cone = rospy.Subscriber('auto_steer_angle_cone', Float32, self.cone_callback)
        self.sub_tunnel = rospy.Subscriber('auto_steer_angle_tunnel', Float32, self.tunnel_callback)
        self.sub_lane_status = rospy.Subscriber('lane_detection_status', Bool, self.lane_status_callback)
        # self.sub_dynamic_obstacle = rospy.Subscriber("dynamic_obstacle", Bool, self.obstacle_callback)

        # 퍼블리셔 설정: 판단된 조향각을 발행
        self.pub_steering = rospy.Publisher("steering_angle", Float32, queue_size=10)
        # 퍼블리셔 설정: 계산된 쓰로틀값을 발행
        self.pub_throttle = rospy.Publisher("auto_throttle", Float32, queue_size=10)

        # 변수 초기화
        self.lane_angle = None  # 차선 기반 조향각
        self.cone_angle = None  # 콘 기반 조향각
        self.tunnel_angle = None  # 터널 기반 조향각
        self.lane_detected = False  # 차선 감지 상태
        self.current_steering_angle = 0.0  # 현재 조향각 (스로틀 계산에 사용), 초기값은 0 (직진)
        self.steering_source_valid = False # 유효한 조향각 소스가 있는지 여부를 나타내는 플래그

        # 속도 플래닝 파라미터
        self.max_throttle = rospy.get_param("~max_throttle", 0.5)  # 최대 스로틀 값 (직진 시 또는 유효 조향각 있을 시)
        self.min_throttle = rospy.get_param("~min_throttle", 0.3)  # 최소 스로틀 값 (최대 조향 시 또는 유효 조향각 없을 시)
        # 조향각이 스로틀 감소에 미치는 영향 계수.
        # 아래 주석은 current_steering_angle이 '도(degree)' 단위라고 가정합니다.
        # 예: 조향각이 최대 30도 정도에서 min_throttle에 도달하게 하려면,
        # (max_throttle - min_throttle) / max_expected_abs_steering_at_min_throttle
        # (0.6 - 0.4) / 30 = 0.2 / 30 = 약 0.0067 (아래 기본값 0.01과 유사)
        self.steering_throttle_reduction_factor = rospy.get_param("~steering_throttle_reduction_factor", 0.01)

        rospy.loginfo("Throttle parameters initialized: max_throttle=%.2f, min_throttle=%.2f, reduction_factor=%.4f",
                      self.max_throttle, self.min_throttle, self.steering_throttle_reduction_factor)

        # 주기적 퍼블리시를 위한 타이머 (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def lane_callback(self, msg):
        self.lane_angle = msg.data

    def cone_callback(self, msg):
        self.cone_angle = msg.data

    def tunnel_callback(self, msg):
        self.tunnel_angle = msg.data

    def lane_status_callback(self, msg):
        self.lane_detected = msg.data

    def publish_steering(self, event):
        steering_angle_output = None
        chosen_source = "None"

        if self.lane_detected is True and self.lane_angle is not None:
            steering_angle_output = self.lane_angle
            chosen_source = "lane"
        elif self.cone_angle is not None:
            steering_angle_output = self.cone_angle
            chosen_source = "cone"
        # elif self.tunnel_angle is not None: # 필요시 터널 로직 활성화
        #     steering_angle_output = self.tunnel_angle
        #     chosen_source = "tunnel"

        if steering_angle_output is not None:
            self.pub_steering.publish(Float32(data=steering_angle_output))
            self.current_steering_angle = steering_angle_output
            self.steering_source_valid = True
            rospy.loginfo("Published steering: %.2f deg (Source: %s)", steering_angle_output, chosen_source)
        else:
            self.steering_source_valid = False
            rospy.logwarn("No valid steering source found. Steering angle will not be published.")
            # self.current_steering_angle은 이전 값을 유지하거나 초기값(0.0)을 사용합니다.
            # 이는 self.steering_source_valid가 True일 때의 스로틀 계산에만 사용됩니다.

    def publish_throttle(self, event):
        final_throttle = 0.0

        if not self.steering_source_valid:
            # 유효한 조향각 소스가 없는 경우, min_throttle 사용
            final_throttle = self.min_throttle
            rospy.logwarn("No valid steering source, setting throttle to min_throttle: %.2f", final_throttle)
        else:
            # 유효한 조향각 소스가 있는 경우, 조향각에 따라 스로틀 계산
            # self.current_steering_angle이 '도(degree)' 단위라고 가정합니다.
            abs_steering = abs(self.current_steering_angle)
            throttle_reduction = abs_steering * self.steering_throttle_reduction_factor
            calculated_throttle = self.max_throttle - throttle_reduction
            final_throttle = max(self.min_throttle, min(self.max_throttle, calculated_throttle)) # Clamping
            rospy.loginfo("Current steering: %.2f deg, Throttle reduction: %.2f, Published throttle: %.2f",
                          self.current_steering_angle, throttle_reduction, final_throttle)

        self.pub_throttle.publish(Float32(data=final_throttle))

    def timer_callback(self, event):
        self.publish_steering(event)  # 먼저 조향각을 결정하고 steering_source_valid 업데이트
        self.publish_throttle(event)  # 업데이트된 상태를 바탕으로 스로틀 계산

if __name__ == '__main__':
    rospy.init_node("judgement_node")
    judgement_system = Judgement()
    rospy.spin()