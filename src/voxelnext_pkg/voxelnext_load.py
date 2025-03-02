#!/home/hannibal/anaconda3/envs/voxelnext/bin/python3


import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.detectors.voxelnext import VoxelNeXt
from pcdet.datasets.live_lidar_dataset import LiveLidarDataset  # 커스텀 데이터셋 불러오기
import logging
import os

def load_voxelnext_model(config_path, model_checkpoint):
    """
    - VoxelNeXt 모델을 로드하고 실시간 LiDAR 처리를 위한 데이터셋을 생성하는 함수
    """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 기준 경로 설정
    
    # ✅ 경로 수정
    config_path = os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path
    model_checkpoint = os.path.join(base_dir, model_checkpoint) if not os.path.isabs(model_checkpoint) else model_checkpoint

    cfg_from_yaml_file(config_path, cfg)
    

    # ✅ 실시간 LiDAR 데이터를 위한 커스텀 데이터셋 사용
    dataset = LiveLidarDataset(
        dataset_cfg=cfg.DATA_CONFIG,  # 🔹 모델 설정이 아닌 데이터셋 설정만 전달
        class_names=cfg.CLASS_NAMES,
        training=False
    )

    # VoxelNeXt 모델 생성
    model = VoxelNeXt(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset  # NuScenesDataset 대신 LiveLidarDataset 사용
    )

    # 기본 로그 설정 추가
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("VoxelNeXt")

    # 모델 로드 시 `logger` 추가
    model.load_params_from_file(model_checkpoint, logger=logger, to_cpu=False)
    
    
    model.cuda()
    model.eval()
    
    
    
    # print(cfg.DATA_CONFIG)
    # print("POINT_FEATURE_ENCODING:", cfg.DATA_CONFIG.get('POINT_FEATURE_ENCODING', "NOT FOUND"))


    return model, dataset  # 모델과 데이터셋을 반환
    print("✅ 실시간 LiDAR 처리를 위한 VoxelNeXt 모델이 로드되었습니다!")
    
# # 실행 코드
# if __name__ == "__main__":
    
#     config_path = "tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
#     model_checkpoint = "checkpoints/voxelnext_nuscenes_kernel1.pth"

#     # 모델 및 데이터셋 로드
#     voxelnext_model, lidar_dataset = load_voxelnext_model(config_path, model_checkpoint)

    
