#include <ros.h>
#include <std_msgs/Float32.h>

ros::NodeHandle nh;

// ROS 구독 변수
float auto_steer_angle = 0.0;
float auto_throttle    = 0.0;

// 콜백 함수
void steeringCallback(const std_msgs::Float32& msg) {
  auto_steer_angle = msg.data;
}
void throttleCallback(const std_msgs::Float32& msg) {
  auto_throttle = msg.data;
}

ros::Subscriber<std_msgs::Float32> sub_steering("/steering_angle",  steeringCallback);
ros::Subscriber<std_msgs::Float32> sub_throttle("/auto_throttle",   throttleCallback);

// PWM 측정용
#define STEER_PIN    2
#define ACCEL_PIN    3
#define BRAKE_PIN   18
#define MANUAL_PIN  20
#define AUTO_PIN    21

#define DIR1        4
#define PWM1        5
#define DIR2        6
#define PWM2        7
#define DIR3        8
#define PWM3        9

#define POT_PIN     A0
#define POT_MIN     167
#define POT_MAX     822
#define MAX_STEER_ANGLE 18.0f
#define STEER_DEAD_BAND 1.0f
#define PULSE_MIN   800
#define PULSE_MAX   2200

const float KP = 0.08, KI = 0.00002, KD = 0;

volatile unsigned long last_edge[5];
volatile unsigned long pulse_width[5];
enum {IDX_STEER, IDX_ACCEL, IDX_BRAKE, IDX_MANUAL, IDX_AUTO};

void pulseISR(int idx) {
  unsigned long now = micros();
  pulse_width[idx] = now - last_edge[idx];
  last_edge[idx] = now;
}

float mapf(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min)*(out_max - out_min)/(in_max - in_min) + out_min;
}

double PID(double ref, double sense, double dt_us) {
  static double prev_err = 0, integral = 0;
  double err = ref - sense;
  double dt = dt_us*1e-6;
  integral += err * dt;
  double P = KP * err;
  double I = KI * integral;
  double D = KD * ((err - prev_err) / dt);
  prev_err = err;
  return P + I + D;
}

void setup() {
  Serial.begin(57600);
  for(int i=0;i<5;i++){
    last_edge[i] = micros();
    pulse_width[i] = 0;
  }
  nh.getHardware()->setBaud(57600);
  nh.initNode();
  nh.subscribe(sub_steering);
  nh.subscribe(sub_throttle);

  pinMode(STEER_PIN,    INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(STEER_PIN),  [](){ pulseISR(IDX_STEER); }, CHANGE);
  pinMode(ACCEL_PIN,    INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ACCEL_PIN),  [](){ pulseISR(IDX_ACCEL); }, CHANGE);
  pinMode(BRAKE_PIN,    INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BRAKE_PIN),  [](){ pulseISR(IDX_BRAKE); }, CHANGE);
  pinMode(MANUAL_PIN,   INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(MANUAL_PIN), [](){ pulseISR(IDX_MANUAL); }, CHANGE);
  pinMode(AUTO_PIN,     INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(AUTO_PIN),   [](){ pulseISR(IDX_AUTO); }, CHANGE);

  pinMode(DIR1, OUTPUT); pinMode(PWM1, OUTPUT);
  pinMode(DIR2, OUTPUT); pinMode(PWM2, OUTPUT);
  pinMode(DIR3, OUTPUT); pinMode(PWM3, OUTPUT);
  pinMode(POT_PIN, INPUT);
}

void loop() {
  nh.spinOnce();
  static unsigned long prev_time = micros();
  unsigned long now = micros(), dt = now - prev_time;
  prev_time = now;

  float accel_input = 0, steer_input = 0;
  int mode = 0;

  if (pulse_width[IDX_ACCEL]>PULSE_MIN && pulse_width[IDX_ACCEL]<PULSE_MAX)
    accel_input = mapf(pulse_width[IDX_ACCEL], 984, 1972, -1.0, 1.0);
  if (pulse_width[IDX_STEER]>PULSE_MIN && pulse_width[IDX_STEER]<PULSE_MAX)
    steer_input = mapf(pulse_width[IDX_STEER], 992.2, 1964, -1.0, 1.0);

  if      (pulse_width[IDX_BRAKE] > 1500) mode = 0;
  else if (pulse_width[IDX_MANUAL]> 1500) mode = 1;
  else if (pulse_width[IDX_AUTO]  > 1500) mode = 2;

  int potRaw = analogRead(POT_PIN);
  float curr_angle = mapf(potRaw, POT_MIN, POT_MAX, MAX_STEER_ANGLE, -MAX_STEER_ANGLE);

  float target_angle;
  if      (mode==1)      target_angle = mapf(steer_input, -1,1, MAX_STEER_ANGLE, -MAX_STEER_ANGLE);
  else if (mode==2)      target_angle = constrain(auto_steer_angle, -MAX_STEER_ANGLE, MAX_STEER_ANGLE);
  else                   target_angle = curr_angle;

  double pid_val = PID(target_angle, curr_angle, dt);

  // ← 좌우 조향 부호 반전
  pid_val = -pid_val;

  bool steerDead = fabs(target_angle - curr_angle) <= STEER_DEAD_BAND;

  if (mode==0) {
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);
    analogWrite(PWM3, 0);
  } else {
    float drive_th = (mode==2)? constrain(auto_throttle, -1.0,1.0) : accel_input;
    if (drive_th>=0) {
      digitalWrite(DIR1, LOW);
      digitalWrite(DIR2, LOW);
      analogWrite(PWM1, int(drive_th*255));
      analogWrite(PWM2, int(drive_th*255));
    } else {
      digitalWrite(DIR1, HIGH);
      digitalWrite(DIR2, HIGH);
      analogWrite(PWM1, int(-drive_th*255));
      analogWrite(PWM2, int(-drive_th*255));
    }

    if (steerDead) {
      analogWrite(PWM3, 0);
    } else if (pid_val > 0) {
      digitalWrite(DIR3, HIGH);
      analogWrite(PWM3, int(min(pid_val,1.0)*127));
    } else {
      digitalWrite(DIR3, LOW);
      analogWrite(PWM3, int(min(-pid_val,1.0)*127));
    }
  }

  Serial.print("Pot= "); Serial.print(potRaw);
  Serial.print(" Tgt= "); Serial.print(target_angle);
  Serial.print(" Cur= "); Serial.print(curr_angle);
  Serial.print(" PID= "); Serial.println(pid_val);
}
