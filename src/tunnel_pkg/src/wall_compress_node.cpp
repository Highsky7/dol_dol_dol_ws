#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

ros::Publisher wall_pub;  // Publisher for /compressed_wall

// Callback function to process incoming point clouds
void wallCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud) {
    // Create a copy of the input cloud to modify
    sensor_msgs::PointCloud2 output_cloud = *input_cloud;
    // Iterate over all points and set their z coordinate to 0.0
    sensor_msgs::PointCloud2Iterator<float> iter_z(output_cloud, "z");
    for (; iter_z != iter_z.end(); ++iter_z) {
        *iter_z = 0.0f;
    }
    // Publish the modified point cloud
    wall_pub.publish(output_cloud);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "wall_compress_node");       // Initialize ROS node
    ros::NodeHandle nh;                                // Node handle for creating pub/sub
    wall_pub = nh.advertise<sensor_msgs::PointCloud2>("/compressed_wall", 1);
    ros::Subscriber wall_sub = nh.subscribe("/detected_wall", 1, wallCallback);
    ROS_INFO("wall_compress_node started.");
    ros::spin();  // Spin to process callbacks indefinitely
    return 0;
}
