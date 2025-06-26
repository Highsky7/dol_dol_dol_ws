#include <ros.h>
#include <std_msgs/Float32.h>

ros::NodeHandle nh;

// ROS 구독 변수
float auto_steer_angle = 0.0;
float auto_throttle = 0.0;

// 콜백 함수
void steeringCallback(const std_msgs::Float32& msg) {
  auto_steer_angle = msg.data;
}
void throttleCallback(const std_msgs::Float32& msg) {
  auto_throttle = msg.data;
}

// 토픽 이름 수정: 퍼블리셔와 일치시키세요
ros::Subscriber<std_msgs::Float32> sub_steering("/steering_angle", steeringCallback);
ros::Subscriber<std_msgs::Float32> sub_throttle("/auto_throttle", throttleCallback);

// 핀 설정
#define STEER_PIN    2   // 조향 PWM 입력 인터럽트
#define ACCEL_PIN    3   // 가속 PWM 입력 인터럽트
#define BRAKE_PIN   18   // 브레이크 PWM 입력 인터럽트
#define MANUAL_PIN  20   // 수동 모드 PWM 입력 인터럽트
#define AUTO_PIN    21   // 자동 모드 PWM 입력 인터럽트

#define DIR1        4
#define PWM1        5
#define DIR2        6
#define PWM2        7
#define DIR3        8    // 조향 모터 제어 방향 핀
#define PWM3        9    // 조향 모터 제어 PWM 핀

#define POT_PIN     A0
#define POT_MIN     167
#define POT_MAX     822
#define MAX_STEER_ANGLE 18.0f
#define STEER_DEAD_BAND 1.0f  // 도 단위 데드밴드
#define PULSE_MIN   800
#define PULSE_MAX   2200

// PID 계수
const float KP = 0.08;
const float KI = 0.00002;
const float KD = 0;

volatile unsigned long last_edge[5];
volatile unsigned long pulse_width[5];
enum {IDX_STEER, IDX_ACCEL, IDX_BRAKE, IDX_MANUAL, IDX_AUTO};

void pulseISR(int idx) {
  unsigned long now = micros();
  pulse_width[idx] = now - last_edge[idx];
  last_edge[idx] = now;
}

float mapf(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

double PID(double ref, double sense, double dt_us) {
  static double prev_err = 0;
  static double integral = 0;
  double err = ref - sense;
  double dt = dt_us * 1e-6;
  integral += err * dt;
  double P = KP * err;
  double I = KI * integral;
  double D = KD * ((err - prev_err) / dt);
  prev_err = err;
  return P + I + D;
}

void setup() {
  // Serial 초기화 (디버그용)
  Serial.begin(57600);

  // 인터럽트 변수 초기화
  for (int i = 0; i < 5; i++) {
    last_edge[i] = micros();
    pulse_width[i] = 0;
  }

  // ROS 시리얼 설정 후 구독
  nh.getHardware()->setBaud(57600);
  nh.initNode();
  nh.subscribe(sub_steering);
  nh.subscribe(sub_throttle);

  // 인터럽트 핀 설정
  pinMode(STEER_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(STEER_PIN), [](){ pulseISR(IDX_STEER); }, CHANGE);
  pinMode(ACCEL_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ACCEL_PIN), [](){ pulseISR(IDX_ACCEL); }, CHANGE);
  pinMode(BRAKE_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BRAKE_PIN), [](){ pulseISR(IDX_BRAKE); }, CHANGE);
  pinMode(MANUAL_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(MANUAL_PIN), [](){ pulseISR(IDX_MANUAL); }, CHANGE);
  pinMode(AUTO_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(AUTO_PIN), [](){ pulseISR(IDX_AUTO); }, CHANGE);

  // 모터 제어 핀
  pinMode(DIR1, OUTPUT); pinMode(PWM1, OUTPUT);
  pinMode(DIR2, OUTPUT); pinMode(PWM2, OUTPUT);
  pinMode(DIR3, OUTPUT); pinMode(PWM3, OUTPUT);

  // 포텐셔미터
  pinMode(POT_PIN, INPUT);
}

void loop() {
  nh.spinOnce();

  // 제어 주기 계산
  static unsigned long prev_time = micros();
  unsigned long now = micros();
  unsigned long dt = now - prev_time;
  prev_time = now;

  // 입력값 초기화
  float accel_input = 0;
  float steer_input = 0;
  int mode = 0;

  // 펄스 폭 매핑
  if (pulse_width[IDX_ACCEL] > PULSE_MIN && pulse_width[IDX_ACCEL] < PULSE_MAX) {
    accel_input = mapf(pulse_width[IDX_ACCEL], 984, 1972, -1.0, 1.0);
  }
  if (pulse_width[IDX_STEER] > PULSE_MIN && pulse_width[IDX_STEER] < PULSE_MAX) {
    steer_input = mapf(pulse_width[IDX_STEER], 992.2, 1964, -1.0, 1.0);
  }
  if (pulse_width[IDX_BRAKE] > 1500)      mode = 0;
  else if (pulse_width[IDX_MANUAL] > 1500) mode = 1;
  else if (pulse_width[IDX_AUTO] > 1500)   mode = 2;

  // 포텐셔미터 판독 -> 각도로 변환
  int potRaw = analogRead(POT_PIN);
  float curr_angle = mapf(potRaw, POT_MIN, POT_MAX, MAX_STEER_ANGLE, -MAX_STEER_ANGLE);

  // 목표 각도 결정
  float target_angle;
  if (mode == 1)        target_angle = mapf(steer_input, -1.0, 1.0, MAX_STEER_ANGLE, -MAX_STEER_ANGLE);
  else if (mode == 2)   target_angle = constrain(auto_steer_angle, -MAX_STEER_ANGLE, MAX_STEER_ANGLE);
  else                  target_angle = curr_angle;

  // PID 계산
  double pid_val = PID(target_angle, curr_angle, dt);

  // 데드밴드 처리
  bool steerDead = fabs(target_angle - curr_angle) <= STEER_DEAD_BAND;

  // 드라이브 & 조향 제어
  if (mode == 0) {
    // 정지
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);
    analogWrite(PWM3, 0);
  } else {
    // 스로틀
    float drive_th = (mode == 2) ? constrain(auto_throttle, -1.0, 1.0) : accel_input;
    if (drive_th >= 0) {
      digitalWrite(DIR1, LOW); digitalWrite(DIR2, LOW);
      analogWrite(PWM1, int(drive_th * 255)); analogWrite(PWM2, int(drive_th * 255));
    } else {
      digitalWrite(DIR1, HIGH); digitalWrite(DIR2, HIGH);
      analogWrite(PWM1, int(-drive_th * 255)); analogWrite(PWM2, int(-drive_th * 255));
    }
    // 조향
    if (steerDead) {
      analogWrite(PWM3, 0);
    } else if (pid_val > 0) {
      digitalWrite(DIR3, HIGH);
      analogWrite(PWM3, int(pid_val * 127));
    } else {
      digitalWrite(DIR3, LOW);
      analogWrite(PWM3, int(-pid_val * 127));
    }
  }

  /*
  // 디버그 출력
  Serial.print("Mode="); Serial.print(mode);
  Serial.print(" Tgt="); Serial.print(target_angle);
  Serial.print(" Cur="); Serial.print(curr_angle);
  Serial.print(" SteerPulse="); Serial.print(pulse_width[IDX_STEER]);
  Serial.print(" PID="); Serial.println(pid_val);
  */
  Serial.print("Potval= "); Serial.println(potRaw);
}
