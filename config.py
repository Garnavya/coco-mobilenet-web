import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Monitor config
    MONITOR_INTERVAL_SECONDS: float = 10.0
    GPU_INDEX: int = 0
    
    # Battery & Power fail-safes
    BATTERY_ABORT_THRESHOLD: int = 40  
    
    # Experiment info
    MODEL_NAME: str = "MobileNetV3-Large"
    DATASET_NAME: str = "COCO"
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 5  # Stop if no improvement after 5 epochs
    IMAGE_RESOLUTION: tuple = (320, 320)
    
    # Base Paths
    BASE_DIR: Path = Path(os.path.abspath(os.path.dirname(__file__)))
    REPORTS_DIR: Path = BASE_DIR / "reports"
    
    # COCO Dataset Paths (Updated for official extraction structure)
    COCO_ROOT: Path = BASE_DIR / "data" / "coco"
    
    TRAIN_IMAGES_DIR: Path = COCO_ROOT / "train2017" / "train2017"
    VAL_IMAGES_DIR: Path = COCO_ROOT / "val2017" / "val2017"
    
    ANNOTATIONS_DIR: Path = COCO_ROOT / "annotations_trainval2017" / "annotations"
    TRAIN_ANNOTATION_FILE: Path = ANNOTATIONS_DIR / "instances_train2017.json"
    VAL_ANNOTATION_FILE: Path = ANNOTATIONS_DIR / "instances_val2017.json"

config = Config()
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)