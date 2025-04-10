import numpy as np
import torch
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper


class LiveLidarDataset:
    def __init__(self, dataset_cfg, class_names, training):
        
        # Save dataset configuration
        self.dataset_cfg = dataset_cfg  
        self.class_names = class_names
        self.training = training
        
        # Add point feature encoder
        self.point_feature_encoding = self.dataset_cfg.get("POINT_FEATURE_ENCODING", None)
        if self.point_feature_encoding is None:
            raise ValueError("❌ POINT_FEATURE_ENCODING is missing in dataset configuration.")
        
        # Add point_feature_encoder
        self.point_feature_encoder = PointFeatureEncoder(self.point_feature_encoding)
        
        # Add voxel generator using VoxelGeneratorWrapper
        self.voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=self.dataset_cfg.DATA_PROCESSOR[2]['VOXEL_SIZE'],  # Voxel size
            coors_range_xyz=self.dataset_cfg.POINT_CLOUD_RANGE,          # Point cloud range
            num_point_features=len(self.point_feature_encoding['used_feature_list']),
            max_num_points_per_voxel=self.dataset_cfg.DATA_PROCESSOR[2]['MAX_POINTS_PER_VOXEL'],  # Max points per voxel
            max_num_voxels=self.dataset_cfg.DATA_PROCESSOR[2]['MAX_NUMBER_OF_VOXELS']['train']  # Max number of voxels
        )
        
        # Print the added attributes
        print(f"✅ LiveLidarDataset initialized with attributes: {self.__dict__.keys()}")
        
        # Automatically reference point cloud range, voxel size and calculate grid size from dataset configuration
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.voxel_size = np.array(self.dataset_cfg.DATA_PROCESSOR[2]["VOXEL_SIZE"], dtype=np.float32)
        self.grid_size = self.calculate_grid_size()

        # Automatically reference required attributes for Detector3DTemplate
        required_attributes = [
            "depth_downsample_factor",
            "num_point_features",
            "num_rawpoint_features"
        ]

        for attr in required_attributes:
            setattr(self, attr, getattr(self.dataset_cfg, attr, 1))  # Default to 1
        
        # Add debug log
        self.debug_dataset_attributes()
        
        
    # -------------------------
    # Function to Calculate grid_size for live LiDAR data.
    # -------------------------
    def calculate_grid_size(self):
        grid_size = ((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size).astype(np.int32)
        print(f"✅ LiveLidarDataset - grid_size: {grid_size}, shape: {grid_size.shape}")
        return grid_size
    
    
    # -------------------------
    # point_cloud: LiDAR point cloud (N x 4) -> [x, y, z, intensity]
    # Convert and return in the format required by the model.
    # -------------------------    
    def load_lidar_data(self, point_cloud):

        # Convert point cloud data from NumPy to Torch Tensor and move to GPU
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32).cuda()
        
        # Create batch_dict for model input
        batch_dict = {
            "points": point_cloud.unsqueeze(0),  # Reshape to (1, N, 4)
            "batch_size": 1
        }
        return batch_dict


    # -------------------------
    # Function to verify that LiveLidarDataset contains all the necessary attributes.
    # -------------------------    
    def debug_dataset_attributes(self):
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
