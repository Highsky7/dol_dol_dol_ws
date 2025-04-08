#!/usr/bin/env python3
import sys
import os
import rospy
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2  # Using sensor_msgs.point_cloud2 instead of ros_numpy
from voxelnext_load import load_voxelnext_model



# -------------------------
# Define colors for each class
# -------------------------
color_map = {
    'car': [1, 0.5, 0.5], # Light Red
    'truck': [0, 1, 0], # Green
    'construction_vehicle': [0, 0, 1], # Blue
    'bus': [1, 1, 0], # Yellow
    'trailer': [1, 0, 1], # Magenta
    'barrier': [0, 1, 1], # Cyan
    'motorcycle': [0.5, 0.5, 0.5], # Gray
    'bicycle': [1, 0.5, 0], # Orange
    'pedestrian': [0.5, 0, 0.5], # Purple
    'traffic_cone': [1, 0, 0],  # 빨강
}
default_color = [0, 0, 0]   # Default: Black


# -------------------------
# Define class number for each class
# -------------------------
nuscenes_class_names = [
    'car',                   # 1
    'truck',                 # 2
    'construction_vehicle',  # 3
    'bus',                   # 4
    'trailer',               # 5
    'barrier',               # 6
    'motorcycle',            # 7
    'bicycle',               # 8
    'pedestrian',            # 9
    'traffic_cone'           # 10
]

# -------------------------
# Function: Convert PointCloud2 message to a NumPy array in (N, 5) format
# -------------------------
def pointcloud2_to_numpy(msg):
    """
    Convert a ROS PointCloud2 message to a NumPy array with shape (N, 5).
    - Format: [x, y, z, intensity, timestamp]
    - Extracts only points from the region of interest (ROI).
    """
    points = np.array(
        list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))),
        dtype=np.float32)
    # Add timestamp column (fixed at 0.0 in this case)
    timestamp = np.full((points.shape[0], 1), 0.0, dtype=np.float32)
    points_with_timestamp = np.hstack((points, timestamp))
    
    return points_with_timestamp



# -------------------------
# Function: Detect objects (preprocessing, voxel conversion, and model inference)
# -------------------------
def detect_objects(points, voxelnext_model, lidar_dataset):
    rospy.loginfo("📡 Converting LiDAR data...")
    data_dict = {"points": points}
    
    # Perform point feature encoding
    data_dict = lidar_dataset.point_feature_encoder.forward(data_dict)
    
    # Process data using each processor in the dataset configuration
    for processor in lidar_dataset.dataset_cfg.DATA_PROCESSOR:
        if processor["NAME"] == "transform_points_to_voxels":
            voxels, coords, num_points_per_voxel = lidar_dataset.voxel_generator.generate(data_dict["points"])
            data_dict["voxels"] = voxels
            data_dict["voxel_coords"] = coords
            data_dict["voxel_num_points"] = num_points_per_voxel
    rospy.logdebug(f"🔍 voxel_coords: {data_dict['voxel_coords'].shape}")
    rospy.logdebug(f"🔍 voxels: {data_dict['voxels'].shape}")
    rospy.logdebug(f"🔍 num_points_per_voxel: {data_dict['voxel_num_points'].shape}")

    # Prepare tensors for model inference
    device = next(voxelnext_model.parameters()).device
    voxel_coords_tensor = torch.from_numpy(data_dict["voxel_coords"]).int().to(device)
    rospy.logdebug(f"🔍 voxel_coords_tensor.shape: {voxel_coords_tensor.shape}")

    with torch.no_grad():
        batch_dict = {
            "batch_size": 1,
            "points": torch.from_numpy(data_dict["points"]).to(device),
            "voxels": torch.from_numpy(data_dict["voxels"]).to(device),
            "voxel_coords": voxel_coords_tensor,
            "voxel_num_points": torch.from_numpy(data_dict["voxel_num_points"]).to(device),
        }
        rospy.logdebug("📦 Batch Dict ready")
        output_dicts, _ = voxelnext_model(batch_dict)
    return output_dicts



# -------------------------
# Function: Publish detected objects as Markers (bounding boxes)
# -------------------------
def publish_markers(output_dicts, pub_detected_objects, class_names):
    rospy.loginfo("📡 publishing detected objects..")

    marker_array = MarkerArray()
    for i, output in enumerate(output_dicts):
        for j, (box, label, score) in enumerate(zip(output["pred_boxes"], output["pred_labels"], output["pred_scores"])):
            label = label.cpu().item()
            score = score.cpu().item()
            
            # Extract x, y and z coordinates (the first, second and third values of the box)
            x_coord = box[0].cpu().item()
            y_coord = box[1].cpu().item()
            z_coord = box[2].cpu().item()
            class_name = class_names[label - 1] # Adjust the class label index (assuming class indices start from 1)
            color = color_map.get(class_name, default_color)
            rospy.loginfo(f"🔍 Object number: {j+1},  Class: {class_name},  Score: {score:.2f},  Position: ({x_coord:.2f}, {y_coord:.2f}, {z_coord:.2f})")

            # Create a marker for the detected object
            marker = Marker()
            marker.header = Header()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "velodyne"
            marker.ns = "detected_objects"
            marker.id = i * 1000 + j
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = box[0].cpu().item()
            marker.pose.position.y = box[1].cpu().item()
            marker.pose.position.z = box[2].cpu().item()
            marker.scale.x = box[3].cpu().item()
            marker.scale.y = box[4].cpu().item()
            marker.scale.z = box[5].cpu().item()
            marker.color.a = 0.5
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(marker)
            
        # Publish the marker array for all detected objects in the current output    
        pub_detected_objects.publish(marker_array)
        



# -------------------------
# ROS Callback function: Process incoming LiDAR data
# -------------------------
def lidar_callback(msg, args):
    
    # Unpack the arguments passed to the callback
    voxelnext_model, lidar_dataset, pub_detected_objects = args
    rospy.loginfo("📡 Receiving LiDAR data...")

    try:
        points = pointcloud2_to_numpy(msg)
    except Exception as e:
        rospy.logerr(f"❌ Error converting PointCloud2: {e}")
        return
    
    if points.shape[1] != 5:
        rospy.logwarn(f"❌ Incorrect point format! Expected (N,5), got: {points.shape}")
        return
    
    try:
        output_dicts = detect_objects(points, voxelnext_model, lidar_dataset)
        publish_markers(output_dicts, pub_detected_objects, voxelnext_model.class_names)
    except Exception as e:
        rospy.logerr(f"❌ Error during object detection/publishing: {e}")


# -------------------------
# Main function: Initialize ROS node, load model, set up publishers/subscribers, and keep the node running
# -------------------------
def main():
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Define the project root directory (assumed to be one level up from the current script)
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Change the working directory to the project root (important for resolving relative paths)
    os.chdir(project_dir)
    
    # Define absolute paths for the configuration file and the model checkpoint
    config_path = os.path.join(project_dir, 'tools', 'cfgs', 'nuscenes_models', 'cbgs_voxel0075_voxelnext.yaml')
    model_checkpoint = os.path.join(project_dir, 'checkpoints', 'voxelnext_nuscenes_kernel1.pth')
    
    rospy.loginfo(f"Config Path: {config_path}")
    rospy.loginfo(f"Model Checkpoint Path: {model_checkpoint}")
    rospy.loginfo(f"Config file exists: {os.path.exists(config_path)}")
    rospy.loginfo(f"Model checkpoint exists: {os.path.exists(model_checkpoint)}")

    # Exit if configuration or model checkpoint is not found
    if not os.path.exists(config_path):
        rospy.logerr(f"Config file not found: {config_path}")
        sys.exit(1)
    if not os.path.exists(model_checkpoint):
        rospy.logerr(f"Model checkpoint not found: {model_checkpoint}")
        sys.exit(1)

    # Initialize the ROS node
    rospy.init_node('lidar_voxelnext_node', anonymous=True)
    rospy.loginfo("ROS Node initialized")
    
    # Load the VoxelNeXt model and associated lidar dataset
    voxelnext_model, lidar_dataset = load_voxelnext_model(config_path, model_checkpoint)
    voxelnext_model.eval() # Set the model to evaluation mode
    rospy.loginfo("VoxelNeXt model load completed")
    
    # Create a ROS publisher for detected objects (bounding box markers)
    pub_detected_objects = rospy.Publisher('/detected_objects', MarkerArray, queue_size=10)
    rospy.loginfo("Publisher '/detected_objects' topic generated")

    # Create a ROS subscriber to receive PointCloud2 messages from the LiDAR sensor
    rospy.Subscriber(
        '/velodyne_points',
        PointCloud2,
        lidar_callback,
        callback_args=(voxelnext_model, lidar_dataset, pub_detected_objects),
        queue_size=1,
        buff_size=2**24
    )
    rospy.loginfo("Subscriber for '/velodyne_points' created")

    rospy.spin()
    rospy.loginfo("ROS node is running...")

if __name__ == '__main__':
    main()
