#include <ros.h>
#include <std_msgs/Float32.h>

ros::NodeHandle nh;

// 기존: 여러 조향각 변수와 플래그 제거, 단일 조향각만 유지
float auto_steer_angle = 0.0;
float auto_throttle = 0.0;

// 새로운 콜백 함수: steering_angle 토픽에서 직접 auto_steer_angle 업데이트
void steeringCallback(const std_msgs::Float32& msg) {
  auto_steer_angle = msg.data;
}

void throttleCallback(const std_msgs::Float32& msg) {
  auto_throttle = msg.data;
}

// 새로운 구독자: steering_angle 토픽 구독
ros::Subscriber<std_msgs::Float32> sub_steering("steering_angle", steeringCallback);
ros::Subscriber<std_msgs::Float32> sub_throttle("auto_throttle", throttleCallback);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define PULSE_MAX               2200
#define PULSE_MIN               800
#define DETECTION_ERR           -1

#define STEERING_PULSE_PIN      2
#define ACCEL_PULSE_PIN         3

#define BREAK_MODE_PIN          18
#define SPEED_MODE_PIN          19
#define MANUAL_MODE_PIN         20
#define AUTO_MODE_PIN           21

#define BREAK_MODE              200
#define ONESTEP_MODE            500
#define TWOSTEP_MODE            800
#define THREESTEP_MODE          1100
#define MANUAL_MODE             1400
#define AUTO_MODE               1700

#define TORQUE_MIN              -1
#define TORQUE_MAX              1

#define SERVO_MIN               -1
#define SERVO_MAX               1

#define SIGNAL_THRESHOLD        0.1

#define POT_MAX                 908
#define POT_MIN                 264
#define MAX_STEER_TIRE_DEG      18

#define KP                      0.08
#define KI                      0.00002
#define KD                      0

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

volatile long Steering_Edge_now_us = DETECTION_ERR;
volatile long Steering_Edge_before_us = DETECTION_ERR;
volatile long Steering_us = DETECTION_ERR;

volatile long Accel_Edge_now_us = DETECTION_ERR;
volatile long Accel_Edge_before_us = DETECTION_ERR;
volatile long Accel_us = DETECTION_ERR;

volatile long Break_Edge_now_us = DETECTION_ERR;
volatile long Break_Edge_before_us = DETECTION_ERR;
volatile long Break_us = DETECTION_ERR;

volatile long Speed_Edge_now_us = DETECTION_ERR;
volatile long Speed_Edge_before_us = DETECTION_ERR;
volatile long Speed_us = DETECTION_ERR;

volatile long Manual_Edge_now_us = DETECTION_ERR;
volatile long Manual_Edge_before_us = DETECTION_ERR;
volatile long Manual_us = DETECTION_ERR;

volatile long Auto_Edge_now_us = DETECTION_ERR;
volatile long Auto_Edge_before_us = DETECTION_ERR;
volatile long Auto_us = DETECTION_ERR;

int DIR1 = 4;
int PWM1 = 5;
int DIR2 = 6;
int PWM2 = 7;
int DIR3 = 8;
int PWM3 = 9;

int POTval = 0;
int POTPin = A0;

unsigned long t_us = 0;
unsigned long prev_t_us = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float Mapping(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SteeringPulseInt() {
  Steering_Edge_now_us = micros();
  Steering_us = Steering_Edge_now_us - Steering_Edge_before_us;
  Steering_Edge_before_us = Steering_Edge_now_us;
}

void AccelPulseInt() {
  Accel_Edge_now_us = micros();
  Accel_us = Accel_Edge_now_us - Accel_Edge_before_us;
  Accel_Edge_before_us = Accel_Edge_now_us;
}

void BreakPulseInt() {
  Break_Edge_now_us = micros();
  Break_us = Break_Edge_now_us - Break_Edge_before_us;
  Break_Edge_before_us = Break_Edge_now_us;
}

void SpeedPulseInt() {
  Speed_Edge_now_us = micros();
  Speed_us = Speed_Edge_now_us - Speed_Edge_before_us;
  Speed_Edge_before_us = Speed_Edge_now_us;
}

void ManualPulseInt() {
  Manual_Edge_now_us = micros();
  Manual_us = Manual_Edge_now_us - Manual_Edge_before_us;
  Manual_Edge_before_us = Manual_Edge_now_us;
}

void AutoPulseInt() {
  Auto_Edge_now_us = micros();
  Auto_us = Auto_Edge_now_us - Auto_Edge_before_us;
  Auto_Edge_before_us = Auto_Edge_now_us;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double PID(double ref, double sense, double dt_us) {
  static double prev_err = 0.0;
  static double integral = 0.0;
  double err = ref - sense;
  double dt_s = dt_us * 1.0e-6;
  integral += err * dt_s;
  double P = err * KP;
  double I = integral * KI;
  double D = ((err - prev_err) / dt_us) * KD;
  prev_err = err;
  return P + I + D;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void StopMotor() {
  digitalWrite(DIR1, HIGH);
  analogWrite(PWM1, 0);
  digitalWrite(DIR2, HIGH);
  analogWrite(PWM2, 0);
  digitalWrite(DIR3, HIGH);
  analogWrite(PWM3, 0);
}

void MoveForward(double throttle) {
  if (throttle > 1.0) throttle = 1.0;
  else if (throttle < 0.0) throttle = 0.0;
  int in = (int)(Mapping(throttle, 0.0, 1.0, 0.0, 255.0));
  if (abs(throttle) < SIGNAL_THRESHOLD) in = 0;
  digitalWrite(DIR1, LOW);
  analogWrite(PWM1, in);
  digitalWrite(DIR2, LOW);
  analogWrite(PWM2, in);
}

void MoveBackward(double throttle) {
  if (throttle > 1.0) throttle = 1.0;
  else if (throttle < 0.0) throttle = 0.0;
  int in = (int)(Mapping(throttle, 0.0, 1.0, 0.0, 255.0));
  if (abs(throttle) < SIGNAL_THRESHOLD) in = 0;
  digitalWrite(DIR1, HIGH);
  analogWrite(PWM1, in);
  digitalWrite(DIR2, HIGH);
  analogWrite(PWM2, in);
}

void Steer(double throttle) {
  if (throttle > 1.0) throttle = 1.0;
  else if (throttle < -1.0) throttle = -1.0;
  int in = (int)(Mapping(throttle, -1.0, 1.0, -255.0, 255.0));
  if (abs(throttle) < SIGNAL_THRESHOLD) in = 0;
  if (in > 0.0) {
    digitalWrite(DIR3, HIGH);
    analogWrite(PWM3, in/2);
  }
  else {
    in *= -1.0;
    digitalWrite(DIR3, LOW);
    analogWrite(PWM3, in/2);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void setup() {
  Serial.begin(57600);
  nh.initNode();
  nh.subscribe(sub_steering);
  nh.subscribe(sub_throttle);

  pinMode(STEERING_PULSE_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(STEERING_PULSE_PIN), SteeringPulseInt, CHANGE);
  pinMode(ACCEL_PULSE_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ACCEL_PULSE_PIN), AccelPulseInt, CHANGE);
  pinMode(BREAK_MODE_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BREAK_MODE_PIN), BreakPulseInt, CHANGE);
  pinMode(SPEED_MODE_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(SPEED_MODE_PIN), SpeedPulseInt, CHANGE);
  pinMode(MANUAL_MODE_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(MANUAL_MODE_PIN), ManualPulseInt, CHANGE);
  pinMode(AUTO_MODE_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(AUTO_MODE_PIN), AutoPulseInt, CHANGE);

  pinMode(POTPin, INPUT_PULLUP);
  
  pinMode(DIR1, OUTPUT);
  pinMode(PWM1, OUTPUT);
  pinMode(DIR2, OUTPUT);
  pinMode(PWM2, OUTPUT);
  pinMode(DIR3, OUTPUT);
  pinMode(PWM3, OUTPUT);
  pinMode(POTPin, INPUT);
}

void loop() {
  nh.spinOnce();
  /*
  unsigned long current_time = millis();
  if (lane_received && (current_time - last_lane_time > 500)) {
    lane_received = false;
  }
  if (cone_received && (current_time - last_cone_time > 500)) {
    cone_received = false;
  }
  if (tunnel_received && (current_time - last_tunnel_time > 500)) {
    tunnel_received = false;
  }
  */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  static int prev_t_us = 0;
  int t_us = micros();

  int Steering_val;
  int Accel_val;
  int Mode_val;
  int Speed_val;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  cli();
  if ((Steering_us > PULSE_MIN) && (Steering_us < PULSE_MAX)) {
    Steering_val = Steering_us;
  }
  sei();

  cli();
  if ((Accel_us > PULSE_MIN) && (Accel_us < PULSE_MAX)) {
    Accel_val = Accel_us;
  }
  sei();

  cli();
  if ((Break_us > PULSE_MIN) && (Break_us < PULSE_MAX)) {
    if (Break_us >= 1900) {
      Mode_val = BREAK_MODE;
    }
    else if ((Manual_us > 1900 && Auto_us <= 1100)) {
      Mode_val = MANUAL_MODE;
    }
    else if ((Auto_us >= 1900 && Manual_us <= 1100)){
      Mode_val = AUTO_MODE;
      if (Speed_us <= 1100 && Speed_us >= 900 && Break_us <= 1100){
        Speed_val = ONESTEP_MODE;
      }
      else if (Speed_us <= 1600 && Speed_us >= 1400 && Break_us <= 1100){
        Speed_val = TWOSTEP_MODE;
      }
      else if (Speed_us >= 1900 && Break_us <= 1100){
        Speed_val = THREESTEP_MODE;
      }
    }
  }
  else {
    Mode_val = BREAK_MODE;
  }
  sei();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  float Throttle_input = Mapping(Accel_val, 984, 1972.0, -1.0, 1.0);
  if (Accel_val >= 1470 && Accel_val <= 1480) {
    Throttle_input = 0;
  }
  float Steer_input = Mapping(Steering_val, 992.2, 1964.0, -1.0, 1.0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  double ref_steer_deg = Mapping(Steer_input, -1.0, 1.0, MAX_STEER_TIRE_DEG, -MAX_STEER_TIRE_DEG);

  POTval = analogRead(POTPin);
  
  double deg = Mapping(POTval, POT_MIN, POT_MAX, -MAX_STEER_TIRE_DEG, MAX_STEER_TIRE_DEG);
  
  int dt = t_us - prev_t_us;
  
  double pid_return = PID(ref_steer_deg, deg, dt);
  if (pid_return > 1.0){
    pid_return = 1.0;
  }
  else if (pid_return < -1.0) {
    pid_return = -1.0;
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /*
  if (lane_received == true) {
    auto_steer_angle = auto_steer_angle_lane;
  }
  else if (lane_received == false && cone_received == true) {
    auto_steer_angle = auto_steer_angle_cone;
  }
  else if (lane_received == false && cone_received == false && tunnel_received == true) {
    auto_steer_angle = auto_steer_angle_tunnel;
  }
  */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (Mode_val == BREAK_MODE){
    StopMotor();
  }
  else if (Mode_val == MANUAL_MODE) {
    if (Throttle_input > 0.0) {
      MoveForward(Throttle_input);
    }
    else {
      Throttle_input *= -1.0;
      MoveBackward(Throttle_input);
    }
    Steer(pid_return);
  }
  else if (Mode_val == AUTO_MODE) {
    
    float ref_steer_deg = auto_steer_angle;
    double pid_return = PID(ref_steer_deg, deg, dt);

    if (Speed_val == ONESTEP_MODE){
      MoveForward(0.3);
      Steer(pid_return);
    }
    if (Speed_val == TWOSTEP_MODE){
      MoveForward(0.5);
      Steer(pid_return);
    }
    if (Speed_val == THREESTEP_MODE){
      MoveForward(auto_throttle);
      Steer(pid_return);
    }
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /*
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 200) {
    Serial.print("ref: "); Serial.println(ref_steer_deg);
    Serial.print("deg: "); Serial.println(deg);
    Serial.print("Steering_us: "); Serial.println(Steering_us);
    Serial.print("Steer_input: "); Serial.println(Steer_input);
    Serial.print("Accel_us: "); Serial.println(Accel_us);
    Serial.print("Accel_val: "); Serial.println(Accel_val);
    Serial.print("Mode_val: "); Serial.println(Mode_val);
    Serial.print("PotPin: "); Serial.println(POTval);
    Serial.print("PID return: "); Serial.println(pid_return);
    Serial.print("err: "); Serial.println(ref_steer_deg - deg);
    Serial.println("////////////////////////////////////////");
    lastPrint = millis();
  }
  */

  prev_t_us = t_us;
}
