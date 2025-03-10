#!/home/highsky/lidar_env/bin/python3

import sys
import os
import rospy
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2  # ros_numpy 대신 사용
from voxelnext_load import load_voxelnext_model
from vehicle_msgs.msg import Track, TrackCone  # Track과 TrackCone 임포트



# -------------------------
# 클래스별 색상 정의
# -------------------------
color_map = {
    'traffic_cone': [1, 0, 0],  # 빨강
    'car': [1, 0.5, 0.5], # Light Red
    'truck': [0, 1, 0], # Green
    'construction_vehicle': [0, 0, 1], # Blue
    'bus': [1, 1, 0], # Yellow
    'trailer': [1, 0, 1], # Magenta
    'barrier': [0, 1, 1], # Cyan
    'motorcycle': [0.5, 0.5, 0.5], # Gray
    'bicycle': [1, 0.5, 0], # Orange
    'pedestrian': [0.5, 0, 0.5], # Purple
}
default_color = [0, 0, 0]  # 기본: 검정 (표시 생략할 때 사용)



# -------------------------
# 함수: PointCloud2 메시지를 NumPy 배열 (N,5)로 변환
# -------------------------
def pointcloud2_to_numpy(msg):
    """
    ROS PointCloud2 메시지를 NumPy 배열 (N,5) 형식으로 변환
    - 구성: [x, y, z, intensity, timestamp]
    - ROI 범위 내의 포인트만 추출
    """
    points = np.array(
        list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))),
        dtype=np.float32)

    
    # 타임스탬프 열 추가 (여기서는 0.0 고정)
    timestamp = np.full((points.shape[0], 1), 0.0, dtype=np.float32)
    

    
    
    points_with_timestamp = np.hstack((points, timestamp))
    
    return points_with_timestamp



# -------------------------
# 함수: 객체 탐지 (전처리, Voxel 변환, 모델 추론)
# -------------------------
def detect_objects(points, voxelnext_model, lidar_dataset):
    rospy.loginfo("📡 LiDAR 데이터 변환 중...")
    data_dict = {"points": points}
    data_dict = lidar_dataset.point_feature_encoder.forward(data_dict)
    for processor in lidar_dataset.dataset_cfg.DATA_PROCESSOR:
        if processor["NAME"] == "transform_points_to_voxels":
            voxels, coords, num_points_per_voxel = lidar_dataset.voxel_generator.generate(data_dict["points"])
            data_dict["voxels"] = voxels
            data_dict["voxel_coords"] = coords
            data_dict["voxel_num_points"] = num_points_per_voxel
    rospy.logdebug(f"🔍 voxel_coords: {data_dict['voxel_coords'].shape}")
    rospy.logdebug(f"🔍 voxels: {data_dict['voxels'].shape}")
    rospy.logdebug(f"🔍 num_points_per_voxel: {data_dict['voxel_num_points'].shape}")

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
        rospy.logdebug("📦 Batch Dict 준비 완료")
        output_dicts, _ = voxelnext_model(batch_dict)
    return output_dicts



# -------------------------
# 함수: 트래픽 콘만 Marker로 발행 (바운딩박스)
# -------------------------
def publish_markers(output_dicts, pub_detected_objects, pub_2d_detected_objects, class_names):
    rospy.loginfo("📡 탐지된 객체를 퍼블리시 중...")
    
    marker_array = MarkerArray()
    marker_array_2d = MarkerArray()
    
    
    
    for i, output in enumerate(output_dicts):
        for j, (box, label, score) in enumerate(zip(output["pred_boxes"], output["pred_labels"], output["pred_scores"])):
            label = label.cpu().item()
            score = score.cpu().item()
            
            # x, y 좌표 추출 (box의 첫 번째와 두 번째 값)
            x_coord = box[0].cpu().item()
            y_coord = box[1].cpu().item()
            rospy.loginfo(f"🔍 객체 {j+1}: 클래스 {label}, 점수 {score:.2f}, 좌표 ({x_coord:.2f}, {y_coord:.2f})")
            class_name = class_names[label - 1] # 객체 종류의 레이블이 1부터 시작해서 보정
            
            # traffic_cone만 표시하기 위해
            # 예외처리 1
            if class_name != 'traffic_cone':
                continue
            # 예외처리 2
            color = color_map.get(class_name, default_color)
            if color == default_color:
                continue
            
            # traffic_cone만 표시
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
        pub_detected_objects.publish(marker_array)
        
        
        
        # -------------------------
        # 추가: SORT-ROS 패키지 요구형태 (/markers_detected)
        # -------------------------
        # 원본 MarkerArray 를 복제하여 2D 형식으로 변환
        # 원본 MarkerArray 는 예외처리 1, 2로 인해서 traffic_cone만 보유
        for marker in marker_array.markers:
            new_marker = Marker()
            new_marker.header = marker.header
            new_marker.header.stamp = rospy.Time.now()
            new_marker.ns = marker.ns
            new_marker.id = marker.id
            new_marker.type = marker.type
            new_marker.action = marker.action
            # 2D 평면용: z 좌표를 0으로 강제
            new_marker.pose.position.x = marker.pose.position.x
            new_marker.pose.position.y = marker.pose.position.y
            new_marker.pose.position.z = 0.0
            new_marker.pose.orientation = marker.pose.orientation
            new_marker.scale.x = marker.scale.x
            new_marker.scale.y = marker.scale.y
            
            # scale.z를 0.0 (평면) (또는 SORT-ROS에서 요구하는 고정값)으로 설정
            new_marker.scale.z = 0.0
            new_marker.color.a = 0.5
            new_marker.color.r = color[0]
            new_marker.color.g = color[0]
            new_marker.color.b = color[1]
            new_marker.lifetime = rospy.Duration(0.1)
            
            marker_array_2d.markers.append(new_marker)
        pub_2d_detected_objects.publish(marker_array_2d)
        
        
        


# -------------------------
# 함수: 바운딩박스 중심 좌표를 초록색 점으로 Marker 토픽에 발행
# -------------------------
def publish_center_markers(output_dicts, pub_detected_objects_center, class_names):
    """
    탐지된 객체 중 'traffic_cone'의 바운딩박스 중심 좌표를 초록색 점(SPHERE)으로 발행
    """
    marker_array = MarkerArray()
    marker_id = 0
    for i, output in enumerate(output_dicts):
        for j, (box, label) in enumerate(zip(output["pred_boxes"], output["pred_labels"])):
            label = label.cpu().item()
            class_name = class_names[label - 1]
            
            if class_name != 'traffic_cone':
                continue
            
            # 중심 좌표는 box의 처음 3개 값 [x, y, z]
            center_x = box[0].cpu().item()
            center_y = box[1].cpu().item()
            # center_z = box[2].cpu().item()
            center_z = 0
            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "velodyne"
            marker.ns = "detected_objects_center"
            marker.id = marker_id
            marker_id += 1  
            
            # # SPHERE 시각화
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = center_z
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            # 크기 (필요에 따라 조정)
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            # 초록색
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
                        
            marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(marker)
            
        pub_detected_objects_center.publish(marker_array)
        
        
        
def publish_track_message(output_dicts, pub_track, class_names):
    """
    탐지된 객체 중 traffic_cone에 해당하는 정보를 기반으로 Track 메시지를 생성 후 발행합니다.
    """
    track_msg = Track()  # Track 메시지 생성

    # 각 검출 결과에서 traffic_cone만 처리
    for output in output_dicts:
        for box, label, score in zip(output["pred_boxes"], output["pred_labels"], output["pred_scores"]):
            label_val = label.cpu().item()
            obj_class = class_names[label_val - 1]  # 1부터 시작하는 경우 보정
            if obj_class != 'traffic_cone':
                continue

            cone = TrackCone()
            cone.x = box[0].cpu().item()  # x 좌표
            cone.y = box[1].cpu().item()  # y 좌표
            cone.type = obj_class         # 예: "traffic_cone"
            track_msg.cones.append(cone)

    pub_track.publish(track_msg)
    rospy.loginfo("📡 /track 메시지 발행 완료")

        
        
        
        


# -------------------------
# ROS 콜백 함수: LiDAR 데이터 수신 후 처리
# -------------------------
def lidar_callback(msg, args):
    """
    LiDAR 콜백 함수:
      1. PointCloud2 메시지를 NumPy 배열 (ROI 적용)로 변환
      2. 객체 탐지 (detect_objects)
      3. 트래픽 콘만 Marker로 발행 (publish_markers)
      4. 바운딩박스 중심 좌표를 초록색 점으로 발행 (detected_objects_center)
    """
    voxelnext_model, lidar_dataset, pub_detected_objects, pub_2d_detected_objects, pub_detected_objects_center, pub_track = args
    rospy.loginfo("📡 LiDAR 데이터 수신 중...")

    try:
        points = pointcloud2_to_numpy(msg)
    except Exception as e:
        rospy.logerr(f"❌ PointCloud2 변환 오류: {e}")
        return

    if points.shape[1] != 5:
        rospy.logwarn(f"❌ 포인트 형식 오류! (N,5) 필요, 현재: {points.shape}")
        return

    try:
        output_dicts = detect_objects(points, voxelnext_model, lidar_dataset)
        publish_markers(output_dicts, pub_detected_objects, pub_2d_detected_objects, voxelnext_model.class_names)
        publish_center_markers(output_dicts, pub_detected_objects_center, voxelnext_model.class_names)
        
        # 추가: /track 메시지 발행
        publish_track_message(output_dicts, pub_track, voxelnext_model.class_names)
    except Exception as e:
        rospy.logerr(f"❌ 객체 탐지/퍼블리싱 오류: {e}")


# -------------------------
# main 함수: ROS 노드 초기화, 모델 로드, 퍼블리셔/서브스크라이버 설정 및 노드 실행 유지
# -------------------------
def main():
    # 현재 스크립트의 디렉토리 경로 가져오기
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # 프로젝트 루트 디렉토리 설정 (voxelnext_pkg 디렉토리)
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    # 작업 디렉토리 변경: 상대 경로 해석을 프로젝트 루트를 기준으로 함
    os.chdir(project_dir)

    # config_path 및 model_checkpoint의 절대 경로 설정
    config_path = os.path.join(project_dir, 'tools', 'cfgs', 'nuscenes_models', 'cbgs_voxel0075_voxelnext.yaml')
    
    
    
    model_checkpoint = os.path.join(project_dir, 'checkpoints', 'voxelnext_nuscenes_kernel1.pth')

    # rospy.loginfo(f"Config Path: {config_path}")
    # rospy.loginfo(f"Model Checkpoint Path: {model_checkpoint}")
    # rospy.loginfo(f"Config file exists: {os.path.exists(config_path)}")
    # rospy.loginfo(f"Model checkpoint exists: {os.path.exists(model_checkpoint)}")

    if not os.path.exists(config_path):
        rospy.logerr(f"Config file not found: {config_path}")
        sys.exit(1)
    if not os.path.exists(model_checkpoint):
        rospy.logerr(f"Model checkpoint not found: {model_checkpoint}")
        sys.exit(1)

    # ROS 노드 초기화
    rospy.init_node('lidar_voxelnext_node', anonymous=True)
    rospy.loginfo("ROS Node initialized")

    # VoxelNeXt 모델 로드 및 eval 모드 설정
    voxelnext_model, lidar_dataset = load_voxelnext_model(config_path, model_checkpoint)
    voxelnext_model.eval()
    rospy.loginfo("VoxelNeXt 모델 로드 완료")

    # ROS 퍼블리셔 생성 (기존: 바운딩박스 MarkerArray)
    pub_detected_objects = rospy.Publisher('/detected_objects', MarkerArray, queue_size=10)
    rospy.loginfo("Publisher '/detected_objects' 생성 완료")
    
    
     # 새 ROS 퍼블리셔 생성: SORT-ROS 패키지 요구 형태용 (/markers_detected, 2D 정보)
    pub_2d_detected_objects = rospy.Publisher('/markers_detected', MarkerArray, queue_size=10)
    rospy.loginfo("Publisher '/markers_detected' 생성 완료")
    

    # 새 ROS 퍼블리셔 생성: 바운딩박스 중심 좌표를 위한 MarkerArray (초록색 점)
    pub_detected_objects_center = rospy.Publisher('/detected_objects_center', MarkerArray, queue_size=10)
    rospy.loginfo("Publisher '/detected_objects_center' 생성 완료")
    
    
    # 새 ROS 퍼블리셔 생성: /track 메시지 Publisher 생성
    pub_track = rospy.Publisher('/track', Track, queue_size=10)
    rospy.loginfo("Publisher '/track' 생성 완료")
    
    
    

    # ROS 구독자 생성 (PointCloud2 메시지 수신)
    rospy.Subscriber(
        '/velodyne_points',
        PointCloud2,
        lidar_callback,
        callback_args=(voxelnext_model, lidar_dataset, pub_detected_objects, pub_2d_detected_objects, pub_detected_objects_center, pub_track),
        queue_size=1,
        buff_size=2**24
    )
    rospy.loginfo("Subscriber '/velodyne_points' 생성 완료")

    rospy.spin()
    rospy.loginfo("ROS 노드 실행 중...")

if __name__ == '__main__':
    main()
