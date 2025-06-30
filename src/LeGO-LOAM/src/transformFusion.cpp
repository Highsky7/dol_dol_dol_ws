#include "utility.h"
#include <tf/transform_datatypes.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Float64.h>
#include <deque>
#include <cstdlib>  // atof

#ifndef NULL
#define NULL 0
#endif

// 1초 전의 오도메트리 데이터를 저장하기 위한 구조체
struct OdomData {
  ros::Time stamp;
  double x;
  double y;
};

class TransformFusion {

private:
    ros::NodeHandle nh;

    // /odometry 토픽 퍼블리셔 (동일한 결과로 발행)
    ros::Publisher pubOdom;

    // 이동벡터 토픽 퍼블리셔 (geometry_msgs::Vector3 사용)
    ros::Publisher pubMovingVector;

    // global yaw 퍼블리셔 (이제 이름이 /global_yaw_legoloam)
    ros::Publisher pubGlobalYaw;

    ros::Subscriber subLaserOdometry;
    // ros::Subscriber subOdomAftMapped;
  
    nav_msgs::Odometry laserOdometry2;
    tf::StampedTransform laserOdometryTrans2;
    tf::TransformBroadcaster tfBroadcaster2;

    float transformSum[6];
    float transformIncre[6];
    float transformMapped[6];
    float transformBefMapped[6];
    float transformAftMapped[6];

    std_msgs::Header currentHeader;

    // 오도메트리 데이터를 저장하는 버퍼 (1초 전 데이터를 위한)
    std::deque<OdomData> odomBuffer;

    // yaw 초기값 (단위: degree)
    double yaw_init_;

public:
    // 생성자에서 yaw_init 값을 전달받습니다.
    TransformFusion(double yaw_init) : yaw_init_(yaw_init) {
        pubOdom = nh.advertise<nav_msgs::Odometry>("/odometry", 5);
        pubMovingVector = nh.advertise<geometry_msgs::Vector3>("/moving_vector", 5);
        pubGlobalYaw = nh.advertise<std_msgs::Float64>("/global_yaw_legoloam", 5);

        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &TransformFusion::laserOdometryHandler, this);
        // subOdomAftMapped = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 5, &TransformFusion::odomAftMappedHandler, this);

        laserOdometry2.header.frame_id = "velodyne_init";
        laserOdometry2.child_frame_id = "velodyne";

        laserOdometryTrans2.frame_id_ = "velodyne_init";
        laserOdometryTrans2.child_frame_id_ = "velodyne";

        for (int i = 0; i < 6; ++i)
        {
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }
    }

    void transformAssociateToMap()
    {
        // 기존 오도메트리와 관련된 변환 계산
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3])
                 - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3])
                 + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

        float x2 = x1;
        float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
        float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

        transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
        transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
        transformIncre[5] = z2;

        float sbcx = sin(transformSum[0]);
        float cbcx = cos(transformSum[0]);
        float sbcy = sin(transformSum[1]);
        float cbcy = cos(transformSum[1]);
        float sbcz = sin(transformSum[2]);
        float cbcz = cos(transformSum[2]);

        float sblx = sin(transformBefMapped[0]);
        float cblx = cos(transformBefMapped[0]);
        float sbly = sin(transformBefMapped[1]);
        float cbly = cos(transformBefMapped[1]);
        float sblz = sin(transformBefMapped[2]);
        float cblz = cos(transformBefMapped[2]);

        float salx = sin(transformAftMapped[0]);
        float calx = cos(transformAftMapped[0]);
        float saly = sin(transformAftMapped[1]);
        float caly = cos(transformAftMapped[1]);
        float salz = sin(transformAftMapped[2]);
        float calz = cos(transformAftMapped[2]);

        float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                  - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                  - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                  - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                  - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
        transformMapped[0] = -asin(srx);

        float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                     - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                     - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                     + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                     + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
        float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                     - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                     + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                     + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                     - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                     + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
        transformMapped[1] = atan2(srycrx / cos(transformMapped[0]), crycrx / cos(transformMapped[0]));

        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz+ calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz+ calx*calz*cblx*cblz);
        transformMapped[2] = atan2(srzcrx / cos(transformMapped[0]), crzcrx / cos(transformMapped[0]));

        // x1, y1, z1, x2, y2, z2 재사용
        x1 = cos(transformMapped[2]) * transformIncre[3] - sin(transformMapped[2]) * transformIncre[4];
        y1 = sin(transformMapped[2]) * transformIncre[3] + cos(transformMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];
        
        x2 = x1;
        y2 = cos(transformMapped[0]) * y1 - sin(transformMapped[0]) * z1;
        z2 = sin(transformMapped[0]) * y1 + cos(transformMapped[0]) * z1;
        
        transformMapped[3] = transformAftMapped[3] - (cos(transformMapped[1]) * x2 + sin(transformMapped[1]) * z2);
        transformMapped[4] = transformAftMapped[4] - y2;
        transformMapped[5] = transformAftMapped[5] - (-sin(transformMapped[1]) * x2 + cos(transformMapped[1]) * z2);
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry)
    {
        currentHeader = laserOdometry->header;

        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

        transformSum[0] = roll;
        transformSum[1] = pitch;
        transformSum[2] = yaw;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;

        transformAssociateToMap();
        geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformMapped[0], transformMapped[2], transformMapped[1]);

        laserOdometry2.header.stamp = ros::Time(currentHeader.stamp.toSec());
        laserOdometry2.header.frame_id = "velodyne_init";
        laserOdometry2.child_frame_id = "velodyne";

        laserOdometry2.pose.pose.orientation = geoQuat;
        laserOdometry2.pose.pose.position.x = transformMapped[5];
        laserOdometry2.pose.pose.position.y = transformMapped[3];
        laserOdometry2.pose.pose.position.z = transformMapped[4];

        {
            tf::Quaternion q_tf(laserOdometry2.pose.pose.orientation.x,
                                laserOdometry2.pose.pose.orientation.y,
                                laserOdometry2.pose.pose.orientation.z,
                                laserOdometry2.pose.pose.orientation.w);
            double r, p, y;
            tf::Matrix3x3(q_tf).getRPY(r, p, y);
            double r_deg = r * 180.0 / M_PI;
            double p_deg = p * 180.0 / M_PI;
            double y_deg = y * 180.0 / M_PI;
            ROS_INFO("----------------------------------");
            ROS_INFO("----------------------------------");
            ROS_INFO("X (pos): %.10f", laserOdometry2.pose.pose.position.x);
            ROS_INFO("Y (pos): %.10f", laserOdometry2.pose.pose.position.y);
            ROS_INFO("Z (pos): %.10f", laserOdometry2.pose.pose.position.z);
            ROS_INFO("Roll (deg): %.10f", r_deg);
            ROS_INFO("Pitch (deg): %.10f", p_deg);
            ROS_INFO("Yaw (deg): %.10f", y_deg);

            // global yaw 계산: 측정된 yaw(도)에 yaw_init_을 더합니다.
            double global_yaw_deg = y_deg + yaw_init_;
            // ROS_INFO에서는 도 단위로 출력
            ROS_INFO("Global Yaw (deg): %.10f", global_yaw_deg);
            // 실제로 발행되는 값은 라디안으로 변환
            double global_yaw_rad = global_yaw_deg * M_PI / 180.0;
            std_msgs::Float64 yaw_msg;
            yaw_msg.data = global_yaw_rad;
            pubGlobalYaw.publish(yaw_msg);
        }

        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = laserOdometry->header.stamp;
        odom_msg.header.frame_id = "velodyne_init";
        odom_msg.child_frame_id = "velodyne";
        odom_msg.pose.pose = laserOdometry2.pose.pose;
        odom_msg.twist.twist.linear.x = 0.0;
        odom_msg.twist.twist.linear.y = 0.0;
        odom_msg.twist.twist.linear.z = 0.0;
        odom_msg.twist.twist.angular.x = 0.0;
        odom_msg.twist.twist.angular.y = 0.0;
        odom_msg.twist.twist.angular.z = 0.0;

        pubOdom.publish(odom_msg);

        laserOdometryTrans2.stamp_ = laserOdometry->header.stamp;
        laserOdometryTrans2.setRotation(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w));
        laserOdometryTrans2.setOrigin(tf::Vector3(transformMapped[5], transformMapped[3], transformMapped[4]));
        tfBroadcaster2.sendTransform(laserOdometryTrans2);

        // 최신 오도메트리 데이터를 버퍼에 저장 (실시간 이동벡터 계산을 위해)
        OdomData currData;
        currData.stamp = laserOdometry2.header.stamp;
        currData.x = laserOdometry2.pose.pose.position.x;
        currData.y = laserOdometry2.pose.pose.position.y;
        odomBuffer.push_back(currData);

        // 버퍼에서 현재 시각보다 1.5초 이상 지난 데이터 제거
        ros::Time currentTime = laserOdometry2.header.stamp;
        while (!odomBuffer.empty() && (currentTime - odomBuffer.front().stamp).toSec() > 1.5) {
            odomBuffer.pop_front();
        }

        // 실시간 이동벡터 계산 및 발행 (현재 시각과 1초 전 위치 차분)
        publishMovingVectorRealTime();
    }

    void odomAftMappedHandler(const nav_msgs::Odometry::ConstPtr& odomAftMapped)
    {
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = odomAftMapped->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

        transformAftMapped[0] = -pitch;
        transformAftMapped[1] = -yaw;
        transformAftMapped[2] = roll;
        transformAftMapped[3] = odomAftMapped->pose.pose.position.x;
        transformAftMapped[4] = odomAftMapped->pose.pose.position.y;
        transformAftMapped[5] = odomAftMapped->pose.pose.position.z;

        transformBefMapped[0] = odomAftMapped->twist.twist.angular.x;
        transformBefMapped[1] = odomAftMapped->twist.twist.angular.y;
        transformBefMapped[2] = odomAftMapped->twist.twist.angular.z;
        transformBefMapped[3] = odomAftMapped->twist.twist.linear.x;
        transformBefMapped[4] = odomAftMapped->twist.twist.linear.y;
        transformBefMapped[5] = odomAftMapped->twist.twist.linear.z;
    }

    void publishMovingVectorRealTime()
    {
        if (odomBuffer.empty()) return;

        ros::Time currentTime = laserOdometry2.header.stamp;
        ros::Time targetTime = currentTime - ros::Duration(1.0);

        OdomData older;
        bool found = false;
        for (auto it = odomBuffer.begin(); it != odomBuffer.end(); ++it) {
            if (it->stamp >= targetTime) {
                if (it == odomBuffer.begin()) {
                    older = *it;
                } else {
                    auto prevIt = std::prev(it);
                    double dt = (it->stamp - prevIt->stamp).toSec();
                    if (dt < 1e-6) {
                        older = *it;
                    } else {
                        double ratio = (targetTime - prevIt->stamp).toSec() / dt;
                        older.x = prevIt->x + ratio * (it->x - prevIt->x);
                        older.y = prevIt->y + ratio * (it->y - prevIt->y);
                    }
                }
                found = true;
                break;
            }
        }
        if (!found) {
            older = odomBuffer.front();
        }

        double curr_x = laserOdometry2.pose.pose.position.x;
        double curr_y = laserOdometry2.pose.pose.position.y;
        double dx = curr_x - older.x;
        double dy = curr_y - older.y;

        geometry_msgs::Vector3 vec;
        vec.x = dx;
        vec.y = dy;
        vec.z = 0.0;
        pubMovingVector.publish(vec);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "transformFusion");

    // roslaunch를 사용할 경우, private 파라미터에서 yaw_init 값을 읽어옵니다.
    ros::NodeHandle nh("~");
    double yaw_init = 0.0;
    nh.param("yaw_init", yaw_init, 0.0);
    ROS_INFO("Yaw init set to: %.3f deg", yaw_init);

    TransformFusion TFusion(yaw_init);
    ROS_INFO("\033[1;32m---->\033[0m Transform Fusion Started.");
    ros::spin();
    return 0;
}