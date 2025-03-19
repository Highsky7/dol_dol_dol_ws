#pragma once
#include <vector>
#include <geometry_msgs/Point.h>

class KalmanFilter {
public:
    KalmanFilter();
    // (x, y, vx, vy)를 초기 상태로 설정
    void init(double x, double y, double vx, double vy);
    // dt 시간 간격 동안 constant velocity 모델로 예측 (한 스텝)
    void predict();
    // 새로운 (x, y, vx, vy)로 상태 업데이트
    void update(double x, double y, double vx, double vy);
    double getX() const;
    double getY() const;
    double getVx() const;
    double getVy() const;
    
    // 예측 단계 수 설정 및 반환
    void setPredictionSteps(int steps);
    int getPredictionSteps() const;
    // 내부 파라미터(predictionSteps_)를 이용하여 예측 궤적을 계산해 반환
    std::vector<geometry_msgs::Point> predictTrajectory();
    
private:
    double x_, y_;    // 위치
    double vx_, vy_;  // 속도
    double dt_;       // 시간 간격 (여기서는 1.0초)
    int predictionSteps_; // 예측 단계 수 (기본값 5)
};
