#include "KalmanFilter.h"

KalmanFilter::KalmanFilter() : x_(0.0), y_(0.0), vx_(0.0), vy_(0.0), dt_(1.0), predictionSteps_(5) {}

void KalmanFilter::init(double x, double y, double vx, double vy) {
    x_ = x;
    y_ = y;
    vx_ = vx;
    vy_ = vy;
}

void KalmanFilter::predict() {
    // 단순 constant velocity 모델: x = x + vx*dt, y = y + vy*dt
    x_ = x_ + vx_ * dt_;
    y_ = y_ + vy_ * dt_;
}

void KalmanFilter::update(double x, double y, double vx, double vy) {
    x_ = x;
    y_ = y;
    vx_ = vx;
    vy_ = vy;
}

double KalmanFilter::getX() const {
    return x_;
}

double KalmanFilter::getY() const {
    return y_;
}

double KalmanFilter::getVx() const {
    return vx_;
}

double KalmanFilter::getVy() const {
    return vy_;
}

void KalmanFilter::setPredictionSteps(int steps) {
    predictionSteps_ = steps;
}

int KalmanFilter::getPredictionSteps() const {
    return predictionSteps_;
}

std::vector<geometry_msgs::Point> KalmanFilter::predictTrajectory() {
    std::vector<geometry_msgs::Point> pts;
    KalmanFilter temp = *this;
    for (int i = 0; i < predictionSteps_; i++) {
        temp.predict();
        geometry_msgs::Point pt;
        pt.x = temp.getX();
        pt.y = temp.getY();
        pt.z = 0.0;
        pts.push_back(pt);
    }
    return pts;
}
