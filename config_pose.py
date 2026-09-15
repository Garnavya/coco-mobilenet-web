import os
from pathlib import Path

class PoseConfig:
    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    COCO_ROOT: Path = BASE_DIR / "data" / "coco"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    
    # Dataset Paths (Adjust to your exact folder structure if needed)
    TRAIN_IMAGES_DIR: Path = COCO_ROOT / "train2017" / "train2017"
    VAL_IMAGES_DIR: Path = COCO_ROOT / "val2017" / "val2017"
    
    # Keypoint Annotations
    TRAIN_KEYPOINTS_FILE: Path = COCO_ROOT / "annotations_trainval2017" / "annotations" / "person_keypoints_train2017.json"
    VAL_KEYPOINTS_FILE: Path = COCO_ROOT / "annotations_trainval2017" / "annotations" / "person_keypoints_val2017.json"
    
    # Hardware
    GPU_INDEX: int = 0
    
    # Pose Experiment Info
    MODEL_NAME: str = "MobileNetV3-Pose"
    DATASET_NAME: str = "COCO-Keypoints"
    BATCH_SIZE: int = 64  # You can increase this since 256x256 crops use less VRAM
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 5
    IMAGE_RESOLUTION: tuple = (256, 256) # Pipeline Model 2 crops to this size
    
    MONITOR_INTERVAL_SECONDS: int = 5

config_pose = PoseConfig()