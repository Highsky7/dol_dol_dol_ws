// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#include "utility.h"
#include <tf/transform_datatypes.h>
#include <nav_msgs/Odometry.h>

// ROS_INFO 사용할 때 cpp은 NULL 정의가 안되어있다는 이유로 오류발생해서 설정
#ifndef NULL
#define NULL 0
#endif

class TransformFusion{

private:

    ros::NodeHandle nh;

    ros::Publisher pubLaserOdometry2;

    // 새 /odometry 토픽 발행 (ma_rrt_ws 요구)
    ros::Publisher pubOdom;

    ros::Subscriber subLaserOdometry;
    ros::Subscriber subOdomAftMapped;
  

    nav_msgs::Odometry laserOdometry2;
    tf::StampedTransform laserOdometryTrans2;
    tf::TransformBroadcaster tfBroadcaster2;


    // 기존에는 map, camera, base_link 프레임을 사용했지만,
    // 여기서는 라이다 전용 프레임만 사용.
    // 따라서, map_2_camera_init_Trans, camera_2_base_link_Trans 등은 제거하거나 주석 처리.
    // tf::StampedTransform map_2_camera_init_Trans;
    // tf::TransformBroadcaster tfBroadcasterMap2CameraInit;

    // tf::StampedTransform camera_2_base_link_Trans;
    // tf::TransformBroadcaster tfBroadcasterCamera2Baselink;

    float transformSum[6];
    float transformIncre[6];
    float transformMapped[6];
    float transformBefMapped[6];
    float transformAftMapped[6];

    std_msgs::Header currentHeader;

public:

    TransformFusion(){

        pubLaserOdometry2 = nh.advertise<nav_msgs::Odometry> ("/6axis", 5);
        pubOdom = nh.advertise<nav_msgs::Odometry> ("/odometry", 5);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &TransformFusion::laserOdometryHandler, this);
        subOdomAftMapped = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 5, &TransformFusion::odomAftMappedHandler, this);

        // 변경: 라이다 센서의 기준 프레임으로 "velodyne_init", 실시간 센서 좌표는 "velodyne"
        // laserOdometry2.header.frame_id = "camera_init";
        // laserOdometry2.child_frame_id = "camera";

        // laserOdometryTrans2.frame_id_ = "camera_init";
        // laserOdometryTrans2.child_frame_id_ = "camera";
        laserOdometry2.header.frame_id = "velodyne_init";
        laserOdometry2.child_frame_id = "velodyne";

        laserOdometryTrans2.frame_id_ = "velodyne_init";
        laserOdometryTrans2.child_frame_id_ = "velodyne";

        // map_2_camera_init_Trans.frame_id_ = "map";
        // map_2_camera_init_Trans.child_frame_id_ = "camera_init";

        // camera_2_base_link_Trans.frame_id_ = "camera";
        // camera_2_base_link_Trans.child_frame_id_ = "base_link";

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
        transformMapped[1] = atan2(srycrx / cos(transformMapped[0]), 
                                   crycrx / cos(transformMapped[0]));
        
        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz+ calx*calz*cblx*cblz);
        transformMapped[2] = atan2(srzcrx / cos(transformMapped[0]), 
                                   crzcrx / cos(transformMapped[0]));

        x1 = cos(transformMapped[2]) * transformIncre[3] - sin(transformMapped[2]) * transformIncre[4];
        y1 = sin(transformMapped[2]) * transformIncre[3] + cos(transformMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformMapped[0]) * y1 - sin(transformMapped[0]) * z1;
        z2 = sin(transformMapped[0]) * y1 + cos(transformMapped[0]) * z1;

        transformMapped[3] = transformAftMapped[3] 
                           - (cos(transformMapped[1]) * x2 + sin(transformMapped[1]) * z2);
        transformMapped[4] = transformAftMapped[4] - y2;
        transformMapped[5] = transformAftMapped[5] 
                           - (-sin(transformMapped[1]) * x2 + cos(transformMapped[1]) * z2);
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry)
    {
        currentHeader = laserOdometry->header;

        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;

        
        // 수정: 올바른 순서 (x, y, z, w)를 사용하여 RPY 계산
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
        // /6axis 메시지: 유지하되, 프레임을 "velodyne_init" -> "velodyne"로 변경
        laserOdometry2.header.frame_id = "velodyne_init";
        laserOdometry2.child_frame_id = "velodyne";


        laserOdometry2.pose.pose.orientation = geoQuat;


        laserOdometry2.pose.pose.position.x = transformMapped[5];
        laserOdometry2.pose.pose.position.y = transformMapped[3];
        laserOdometry2.pose.pose.position.z = transformMapped[4];



        // -------------------------------
        // 추가: 퍼블리시하기 직전에 RPY(도 단위)로 출력
        // -------------------------------
        {
            // geometry_msgs::Quaternion -> tf::Quaternion
            tf::Quaternion q_tf(laserOdometry2.pose.pose.orientation.x,
                                laserOdometry2.pose.pose.orientation.y,
                                laserOdometry2.pose.pose.orientation.z,
                                laserOdometry2.pose.pose.orientation.w);
            // RPY 추출
            double r, p, y;
            tf::Matrix3x3(q_tf).getRPY(r, p, y);

            // 라디안을 각도(deg)로 변환
            double r_deg = r * 180.0 / M_PI;
            double p_deg = p * 180.0 / M_PI;
            double y_deg = y * 180.0 / M_PI;

            ROS_INFO("----------------------------------");
            ROS_INFO("----------------------------------");

            // x, y, z도 함께 출력
            double x_pos = laserOdometry2.pose.pose.position.x;
            double y_pos = laserOdometry2.pose.pose.position.y;
            double z_pos = laserOdometry2.pose.pose.position.z;
            ROS_INFO("X (pos): %.10f", x_pos);
            ROS_INFO("Y (pos): %.10f", y_pos);
            ROS_INFO("Z (pos): %.10f", z_pos);
            ROS_INFO("Roll (deg): %.10f", r_deg);
            ROS_INFO("Pitch (deg): %.10f", p_deg);
            ROS_INFO("Yaw (deg): %.10f", y_deg);    
        }
        // -------------------------------

        pubLaserOdometry2.publish(laserOdometry2);

        laserOdometryTrans2.stamp_ = laserOdometry->header.stamp;
        
        laserOdometryTrans2.setRotation(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w));

        laserOdometryTrans2.setOrigin(tf::Vector3(transformMapped[5], transformMapped[3], transformMapped[4]));
        tfBroadcaster2.sendTransform(laserOdometryTrans2);


        // /odometry 토픽 발행: ma_rrt_ws가 요구하는 구조로 (velodyne_init -> velodyne)
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
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "transformFusion");
    
    TransformFusion TFusion;

    ROS_INFO("\033[1;32m---->\033[0m Transform Fusion Started.");
    
    ros::spin();

    return 0;
}
