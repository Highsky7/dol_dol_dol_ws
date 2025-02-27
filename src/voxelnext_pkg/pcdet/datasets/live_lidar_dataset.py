import numpy as np
import torch
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper


class LiveLidarDataset:
    def __init__(self, dataset_cfg, class_names, training):
        self.dataset_cfg = dataset_cfg  # 🔹 추가 (dataset_cfg 저장)
        self.class_names = class_names
        self.training = training
        
        
        
        # ✅ 포인트 특징 인코더 추가
        self.point_feature_encoding = self.dataset_cfg.get("POINT_FEATURE_ENCODING", None)
        if self.point_feature_encoding is None:
            raise ValueError("❌ POINT_FEATURE_ENCODING is missing in dataset configuration.")
        
        # 🔹 point_feature_encoder 추가
        self.point_feature_encoder = PointFeatureEncoder(self.point_feature_encoding)
        
        
        # ✅ Voxel 변환기 추가 (VoxelGeneratorWrapper 사용)
        self.voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=self.dataset_cfg.DATA_PROCESSOR[2]['VOXEL_SIZE'],  # Voxel 크기
            coors_range_xyz=self.dataset_cfg.POINT_CLOUD_RANGE,          # 포인트 클라우드 범위
            num_point_features=len(self.point_feature_encoding['used_feature_list']),
            max_num_points_per_voxel=self.dataset_cfg.DATA_PROCESSOR[2]['MAX_POINTS_PER_VOXEL'],  # 한 Voxel당 최대 포인트 수
            max_num_voxels=self.dataset_cfg.DATA_PROCESSOR[2]['MAX_NUMBER_OF_VOXELS']['train']  # 최대 Voxel 수
        )
        
        # ✅ 추가된 속성 출력
        print(f"✅ LiveLidarDataset initialized with attributes: {self.__dict__.keys()}")
        
        
        # ✅
        # 🔹 자동으로 데이터셋 설정에서 참조
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.voxel_size = np.array(self.dataset_cfg.DATA_PROCESSOR[2]["VOXEL_SIZE"], dtype=np.float32)
        self.grid_size = self.calculate_grid_size()

        # 🔹 `Detector3DTemplate`이 필요로 하는 속성들을 자동으로 참조
        required_attributes = [
            "depth_downsample_factor",
            "num_point_features",
            "num_rawpoint_features"
        ]

        for attr in required_attributes:
            setattr(self, attr, getattr(self.dataset_cfg, attr, 1))  # 기본값 1
        
        
        
        
        
        # 🔹 디버그 로그 추가
        self.debug_dataset_attributes()
        

    def calculate_grid_size(self):
        """
        실시간 라이더 데이터를 위한 grid_size 계산.
        """
        grid_size = ((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size).astype(np.int32)
        print(f"✅ LiveLidarDataset - grid_size: {grid_size}, shape: {grid_size.shape}")  # 추가된 부분
        return grid_size
        

    def load_lidar_data(self, point_cloud):
        """
        - point_cloud: LiDAR 포인트 클라우드 (N x 4) -> [x, y, z, intensity]
        - 모델이 요구하는 형태로 변환하여 반환
        """
        # 포인트 클라우드 데이터 NumPy → Torch Tensor 변환
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32).cuda()
        
        # 모델에 입력하기 위한 batch_dict 생성
        batch_dict = {
            "points": point_cloud.unsqueeze(0),  # (1, N, 4) 형태로 맞춤
            "batch_size": 1
        }
        return batch_dict

    def debug_dataset_attributes(self):
            """
            현재 `LiveLidarDataset`이 필요한 모든 속성을 가지고 있는지 확인하는 함수
            """
            expected_attributes = [
            "point_feature_encoding",
            "point_feature_encoder",
            "point_cloud_range",
            "voxel_size",
            "grid_size",
            "depth_downsample_factor",
            "num_point_features",
            "num_rawpoint_features"
        ]
            
            for attr in expected_attributes:
                if not hasattr(self, attr):
                    print(f"⚠️ LiveLidarDataset is missing attribute: {attr}")
            
            print(f"✅ LiveLidarDataset initialized with attributes: {self.__dict__.keys()}")
