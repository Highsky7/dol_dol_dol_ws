<<<<<<< Updated upstream
#!/home/hannibal/anaconda3/envs/voxelnext/bin/python3
=======
#!/usr/bin/env python3
>>>>>>> Stashed changes

import torch
import numpy as np
import open3d as o3d
from voxelnext_load import load_voxelnext_model  # 기존의 모델 로드 함수 재사용
from pcdet.datasets.live_lidar_dataset import LiveLidarDataset
import os

# ✅ 1. 모델 로드

base_dir = os.path.dirname(os.path.abspath(__file__))
# ✅ 절대경로 -> 상대경로 변경
config_path = os.path.join(base_dir, "tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml")
model_checkpoint = os.path.join(base_dir, "checkpoints/voxelnext_nuscenes_kernel1.pth")

voxelnext_model, lidar_dataset = load_voxelnext_model(config_path, model_checkpoint)
voxelnext_model.eval()  # 추론 모드 설정

# ✅ 2. PCD 파일 로드
def load_pcd(file_path):
    """ PCD 파일을 로드하여 (N, 5) 형식의 NumPy 배열로 변환 """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)

    # ✅ intensity 필드가 없는 경우, 기본값 설정 (여기서는 1.0으로 설정)
    intensity = np.ones((points.shape[0], 1), dtype=np.float32)

    # ✅ 타임스탬프 필드 추가 (현재 시간을 사용하거나, 고정된 값으로 설정)
    timestamp = np.full((points.shape[0], 1), 0.0, dtype=np.float32)  # 여기서는 0.0으로 설정

    # ✅ 최종 (N, 5) 배열 생성 (x, y, z, intensity, timestamp)
    points_with_intensity = np.hstack((points, intensity, timestamp))
    return points_with_intensity

# ✅ 3. PCD 파일 경로 설정
pcd_file_path = "/home/hannibal/Downloads/test.pcd"

if not os.path.exists(pcd_file_path):
    print(f"❌ PCD 파일을 찾을 수 없습니다: {pcd_file_path}")
    exit(1)

point_cloud = load_pcd(pcd_file_path)

print(f"📌 로드된 포인트 클라우드의 형태: {point_cloud.shape}")  # 예: (N, 5)

# ✅ 4. 데이터 변환 (LiveLidarDataset 활용)
data_dict = {"points": point_cloud}
data_dict = lidar_dataset.point_feature_encoder.forward(data_dict)

# ✅ 추가: Voxel 변환을 수행하여 `batch_dict`에 `voxels` 추가
for processor in lidar_dataset.dataset_cfg.DATA_PROCESSOR:
    if processor["NAME"] == "transform_points_to_voxels":
        voxels, coordinates, num_points_per_voxel = lidar_dataset.voxel_generator.generate(data_dict["points"])
        data_dict["voxels"] = voxels
        data_dict["voxel_coords"] = coordinates
        data_dict["voxel_num_points"] = num_points_per_voxel

# ✅ 5. 모델에 입력 후 추론
with torch.no_grad():
    batch_dict = {
        "batch_size": 1,
        "points": torch.from_numpy(data_dict["points"]).cuda(non_blocking=True),
        "voxels": torch.from_numpy(data_dict["voxels"]).cuda(non_blocking=True),
        "voxel_coords": torch.from_numpy(data_dict["voxel_coords"]).cuda(non_blocking=True),
        "voxel_num_points": torch.from_numpy(data_dict["voxel_num_points"]).cuda(non_blocking=True),
    }

    output_dicts, _ = voxelnext_model(batch_dict)

# ✅ 6. 결과 확인
print(f"📌 모델이 학습한 클래스 목록: {voxelnext_model.class_names}")

# ✅ 7. 모든 클래스 필터링 및 결과 출력
print("📌 예측 결과 (모든 객체 출력):")
for i, output in enumerate(output_dicts):
    print(f"🔹 샘플 {i+1}:")
    print(f"   - 예측 박스 개수: {len(output['pred_boxes'])}")
    print(f"   - 예측 점수: {output['pred_scores']}")
    print(f"   - 예측 클래스: {output['pred_labels']}")
    print(f"   - 예측 박스 좌표: [x, y, z, w, l, h, θ, vx, vy]")
    print(output['pred_boxes'])

print("✅ 모든 객체 필터링 완료!")
print("✅ VoxelNeXt 테스트 완료!")

# ✅ 8. 바운딩 박스를 PCD에 시각화하기
def create_bounding_box(box):
    """
    VoxelNeXt 박스 정보를 Open3D BoundingBox 형식으로 변환
    box: [x, y, z, w, l, h, θ, vx, vy]
    """
    # x, y, z: 박스 중심
    # w, l, h: 박스 크기 (width, length, height)
    # θ: 회전 각도 (라디안)

    x, y, z, w, l, h, theta, vx, vy = box

    # Create a box geometry centered at origin
    box_obb = o3d.geometry.OrientedBoundingBox()
    
    # Convert tensors to floats
    x = x.item()
    y = y.item()
    z = z.item()
    w = w.item()
    l = l.item()
    h = h.item()
    theta = theta.item()

    box_obb.center = [x, y, z]
    box_obb.extent = [w, l, h]

    # Compute rotation matrix around Z-axis
    R = box_obb.get_rotation_matrix_from_xyz((0, 0, theta))
    box_obb.R = R

    return box_obb

def visualize_with_boxes(pcd_path, output_dicts, class_names):
    """
    포인트 클라우드와 예측된 바운딩 박스를 시각화
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)

    # Create list of bounding boxes
    bounding_boxes = []
    colors = []

    # Define colors for each class
    color_map = {
        'car': [1, 0, 0],               # Red
        'truck': [0, 1, 0],             # Green
        'construction_vehicle': [0, 0, 1],  # Blue
        'bus': [1, 1, 0],               # Yellow
        'trailer': [1, 0, 1],           # Magenta
        'barrier': [0, 1, 1],           # Cyan
        'motorcycle': [0.5, 0.5, 0.5],  # Gray
        'bicycle': [1, 0.5, 0],         # Orange
        'pedestrian': [0.5, 0, 0.5],    # Purple
        'traffic_cone': [1, 0.5, 0.5],  # Light Red
    }

    # Generate a default color for unknown classes
    default_color = [0, 0, 0]  # Black

    for output in output_dicts:
        for box, label in zip(output['pred_boxes'], output['pred_labels']):
            label = label.cpu().item()  # Convert tensor to int

            if 1 <= label <= len(class_names):
                class_name = class_names[label - 1]
            else:
                class_name = "unknown"

            obb = create_bounding_box(box)
            bounding_boxes.append(obb)

            # Assign color based on class
            colors.append(color_map.get(class_name, default_color))  # Default to black if class not found

    # Assign colors to bounding boxes
    for obb, color in zip(bounding_boxes, colors):
        obb.color = color

    # Visualize
    if bounding_boxes:
        o3d.visualization.draw_geometries([pcd] + bounding_boxes)
    else:
        print("No objects detected to visualize.")

# ✅ 9. 시각화 실행
visualize_with_boxes(pcd_file_path, output_dicts, voxelnext_model.class_names)
