#!/usr/bin/env python3

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.detectors.voxelnext import VoxelNeXt
from pcdet.datasets.live_lidar_dataset import LiveLidarDataset  # 커스텀 데이터셋 불러오기
import logging
import os


# -------------------------
# Function to load the VoxelNeXt model and create a dataset for live LiDAR processing.
# -------------------------
def load_voxelnext_model(config_path, model_checkpoint):
    
    # Determine the absolute path for configuration and checkpoint files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path) if not os.path.isabs(config_path) else config_path
    model_checkpoint = os.path.join(base_dir, model_checkpoint) if not os.path.isabs(model_checkpoint) else model_checkpoint
    
    # Load configuration from the YAML file into the cfg object
    cfg_from_yaml_file(config_path, cfg)
    

    # Use custom dataset for live LiDAR data
    dataset = LiveLidarDataset(
        dataset_cfg=cfg.DATA_CONFIG,  # Pass only dataset configuration (not model configuration)
        class_names=cfg.CLASS_NAMES,
        training=False
    )

    # Create VoxelNeXt model
    model = VoxelNeXt(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset  # Use LiveLidarDataset instead of NuScenesDataset
    )

    # Set up basic logging configuration
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("VoxelNeXt")

    # Load model parameters from checkpoint with logger, then move model to GPU and set to evaluation mode
    model.load_params_from_file(model_checkpoint, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    
    # Return both the model and the dataset
    return model, dataset

    
# ## Execution code example
# if __name__ == "__main__":
    
#     # Define paths for the configuration file and model checkpoint
#     config_path = "tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
#     model_checkpoint = "checkpoints/voxelnext_nuscenes_kernel1.pth"

#     # Load the model and dataset
#     voxelnext_model, lidar_dataset = load_voxelnext_model(config_path, model_checkpoint)

    
