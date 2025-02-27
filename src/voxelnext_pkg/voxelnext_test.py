#!/home/highsky/lidar_env/bin/python3
import torch
import numpy as np
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

# ✅ 2. 랜덤 샘플 포인트 클라우드 생성 (N, 5) 형식 (x, y, z, intensity, timestamp)
num_points = 500  # 예제 포인트 개수
point_cloud = np.random.uniform(-5, 5, size=(num_points, 5)).astype(np.float32)  # 범위: [-50, 50]

# ✅ 3. 데이터 변환 (LiveLidarDataset 활용)
data_dict = {"points": point_cloud}
data_dict = lidar_dataset.point_feature_encoder.forward(data_dict)



# ✅ 추가: Voxel 변환을 수행하여 `batch_dict`에 `voxels` 추가
for processor in lidar_dataset.dataset_cfg.DATA_PROCESSOR:
    if processor["NAME"] == "transform_points_to_voxels":
        voxel_generator = lidar_dataset.voxel_generator
        #voxels, coordinates, num_points_per_voxel = voxel_generator.generate_voxels(data_dict["points"])
        
        # 수정 코드
        voxels, coordinates, num_points_per_voxel = voxel_generator.generate(data_dict["points"])

        data_dict["voxels"] = voxels
        data_dict["voxel_coords"] = coordinates
        data_dict["voxel_num_points"] = num_points_per_voxel




# ✅ 4. 모델에 입력 후 추론
# ✅ batch_dict 최적화 (한 번에 GPU로 올리기)
with torch.no_grad():
    batch_dict = {
        "batch_size": 1,
        "points": torch.from_numpy(data_dict["points"]).cuda(non_blocking=True),
        "voxels": torch.from_numpy(data_dict["voxels"]).cuda(non_blocking=True),
        "voxel_coords": torch.from_numpy(data_dict["voxel_coords"]).cuda(non_blocking=True),
        "voxel_num_points": torch.from_numpy(data_dict["voxel_num_points"]).cuda(non_blocking=True),
    }

    output_dicts, _ = voxelnext_model(batch_dict)


    
    
    

# ✅ 5. 결과 확인
print(f"📌 모델이 학습한 클래스 목록: {voxelnext_model.class_names}")
# print("📌 예측 결과:")
# for i, output in enumerate(output_dicts):
#     print(f"🔹 샘플 {i+1}:")
#     print(f"   - 예측 박스 개수: {len(output['pred_boxes'])}")
#     print(f"   - 예측 점수: {output['pred_scores']}")
#     print(f"   - 예측 클래스: {output['pred_labels']}")
#     print(f"   - 예측 박스 좌표: {output['pred_boxes']}")



# ✅ 6. Traffic Cone만 필터링
print("📌 예측 결과 (Traffic Cone만 출력):")
for i, output in enumerate(output_dicts):
    # ✅ traffic_cone 클래스 ID 가져오기 (10)
    traffic_cone_id = 10

    # ✅ traffic_cone 클래스만 필터링
    mask = output["pred_labels"] == traffic_cone_id

    # ✅ 결과 출력
    if len(output["pred_boxes"][mask]) > 0:
        print(f"🔹 샘플 {i+1}:")
        print(f"   - 예측 박스 개수 (Traffic Cone): {len(output['pred_boxes'][mask])}")
        print(f"   - 예측 점수: {output['pred_scores'][mask]}")
        
        print(f"   - 예측 박스 좌표: [x, y, z, w, l, h, θ, vx, vy]")
        print(output['pred_boxes'][mask])
        
    else:
        print(f"🔹 샘플 {i+1}: No traffic cones detected.")

print("✅ Traffic Cone 필터링 완료!")



print("✅ VoxelNeXt 테스트 완료!")

