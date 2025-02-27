#!/home/highsky/lidar_env/bin/python3

import sys
import os
import rospy
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from voxelnext_pkg.voxelnext_load import load_voxelnext_model
import open3d as o3d

color_map = {
    'traffic_cone': [1, 0, 0],
    'car': [1, 0.5, 0.5],
    'truck': [0, 1, 0],
    'construction_vehicle': [0, 0, 1],
    'bus': [1, 1, 0],
    'trailer': [1, 0, 1],
    'barrier': [0, 1, 1],
    'motorcycle': [0.5, 0.5, 0.5],
    'bicycle': [1, 0.5, 0],
    'pedestrian': [0.5, 0, 0.5],
}
default_color = [0, 0, 0]

def pointcloud2_to_numpy(msg):
    points = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))), dtype=np.float32)
    timestamp = np.full((points.shape[0], 1), 0.0, dtype=np.float32)
    return np.hstack((points, timestamp))

def create_bounding_box(box):
    x, y, z, w, l, h, theta, vx, vy = box
    x = x.cpu().item()
    y = y.cpu().item()
    z = z.cpu().item()
    w = w.cpu().item()
    l = l.cpu().item()
    h = h.cpu().item()
    theta = theta.cpu().item()
    box_obb = o3d.geometry.OrientedBoundingBox()
    box_obb.center = [x, y, z]
    box_obb.extent = [w, l, h]
    R = box_obb.get_rotation_matrix_from_xyz((0, 0, theta))
    box_obb.R = R
    return box_obb

def detect_objects(points, voxelnext_model, lidar_dataset):
    data_dict = {"points": points}
    data_dict = lidar_dataset.point_feature_encoder.forward(data_dict)
    for processor in lidar_dataset.dataset_cfg.DATA_PROCESSOR:
        if processor["NAME"] == "transform_points_to_voxels":
            voxels, coords, num_points_per_voxel = lidar_dataset.voxel_generator.generate(data_dict["points"])
            data_dict["voxels"] = voxels
            data_dict["voxel_coords"] = coords
            data_dict["voxel_num_points"] = num_points_per_voxel
    device = next(voxelnext_model.parameters()).device
    voxel_coords_tensor = torch.from_numpy(data_dict["voxel_coords"]).int().to(device)
    with torch.no_grad():
        batch_dict = {
            "batch_size": 1,
            "points": torch.from_numpy(data_dict["points"]).to(device),
            "voxels": torch.from_numpy(data_dict["voxels"]).to(device),
            "voxel_coords": voxel_coords_tensor,
            "voxel_num_points": torch.from_numpy(data_dict["voxel_num_points"]).to(device),
        }
        output_dicts, _ = voxelnext_model(batch_dict)
    return output_dicts

def publish_markers(output_dicts, pub_detected_objects, class_names):
    marker_array = MarkerArray()
    for i, output in enumerate(output_dicts):
        for j, (box, label, score) in enumerate(zip(output["pred_boxes"], output["pred_labels"], output["pred_scores"])):
            label = label.cpu().item()
            score = score.cpu().item()
            class_name = class_names[label - 1]
            if class_name != 'traffic_cone':
                continue
            color = color_map.get(class_name, default_color)
            if color == default_color:
                continue
            marker = Marker()
            marker.header = Header()
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
            marker.lifetime = rospy.Duration(0.3)
            marker_array.markers.append(marker)
    if len(marker_array.markers) > 0:
        pub_detected_objects.publish(marker_array)

def publish_center_markers(output_dicts, pub_center_markers, class_names):
    marker_array = MarkerArray()
    marker_id = 0
    for i, output in enumerate(output_dicts):
        for j, (box, label) in enumerate(zip(output["pred_boxes"], output["pred_labels"])):
            label = label.cpu().item()
            class_name = class_names[label - 1]
            if class_name != 'traffic_cone':
                continue
            center_x = box[0].cpu().item()
            center_y = box[1].cpu().item()
            center_z = 0
            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "velodyne"
            marker.ns = "center_markers"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = center_z
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(marker)
    if len(marker_array.markers) > 0:
        pub_center_markers.publish(marker_array)

def lidar_callback(msg, args):
    voxelnext_model, lidar_dataset, pub_detected_objects, pub_center_markers = args
    try:
        points = pointcloud2_to_numpy(msg)
    except Exception as e:
        rospy.logerr(f"Error converting PointCloud2: {e}")
        return
    if points.shape[1] != 5:
        rospy.logwarn(f"Point format error! Expected (N,5), got {points.shape}")
        return
    try:
        output_dicts = detect_objects(points, voxelnext_model, lidar_dataset)
        publish_markers(output_dicts, pub_detected_objects, voxelnext_model.class_names)
        publish_center_markers(output_dicts, pub_center_markers, voxelnext_model.class_names)
    except Exception as e:
        rospy.logerr(f"Error in object detection/publishing: {e}")

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    os.chdir(project_dir)
    config_path = os.path.join(project_dir, 'tools', 'cfgs', 'nuscenes_models', 'cbgs_voxel0075_voxelnext.yaml')
    model_checkpoint = os.path.join(project_dir, 'checkpoints', 'voxelnext_nuscenes_kernel1.pth')
    rospy.loginfo(f"Config Path: {config_path}")
    rospy.loginfo(f"Model Checkpoint Path: {model_checkpoint}")
    if not os.path.exists(config_path) or not os.path.exists(model_checkpoint):
        rospy.logerr("Config or model checkpoint not found")
        sys.exit(1)
    rospy.init_node('cone_detector_node', anonymous=True)
    rospy.loginfo("ROS Node initialized")
    voxelnext_model, lidar_dataset = load_voxelnext_model(config_path, model_checkpoint)
    voxelnext_model.eval()
    rospy.loginfo("VoxelNeXt model loaded")
    pub_detected_objects = rospy.Publisher('/detected_objects', MarkerArray, queue_size=10)
    pub_center_markers = rospy.Publisher('/center_markers', MarkerArray, queue_size=10)
    rospy.loginfo("Publishers created")
    rospy.Subscriber(
        '/velodyne_points',
        PointCloud2,
        lidar_callback,
        callback_args=(voxelnext_model, lidar_dataset, pub_detected_objects, pub_center_markers),
        queue_size=1,
        buff_size=2**24
    )
    rospy.loginfo("Subscriber created")
    rospy.spin()
    rospy.loginfo("ROS node running")

if __name__ == '__main__':
    main()