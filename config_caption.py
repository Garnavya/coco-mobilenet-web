import os
from pathlib import Path

class CaptionConfig:
    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    COCO_ROOT: Path = BASE_DIR / "data" / "coco"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    
    # Dataset Paths
    TRAIN_IMAGES_DIR: Path = COCO_ROOT / "train2017" / "train2017"
    VAL_IMAGES_DIR: Path = COCO_ROOT / "val2017" / "val2017"
    
    # Caption Annotations
    TRAIN_CAPTIONS_FILE: Path = COCO_ROOT / "annotations_trainval2017" / "annotations" / "captions_train2017.json"
    VAL_CAPTIONS_FILE: Path = COCO_ROOT / "annotations_trainval2017" / "annotations" / "captions_val2017.json"
    
    # Hardware
    GPU_INDEX: int = 0
    
    # Caption Experiment Info
    MODEL_NAME: str = "MobileNetV3-Caption"
    DATASET_NAME: str = "COCO-Captions"
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 5
    IMAGE_RESOLUTION: tuple = (256, 256)
    
    # NLP Specifics
    MAX_SEQ_LENGTH: int = 20 # Maximum words in a caption
    VOCAB_FREQ_THRESHOLD: int = 5 # Words must appear 5 times to be kept
    EMBEDDING_DIM: int = 256
    HIDDEN_DIM: int = 512
    
    MONITOR_INTERVAL_SECONDS: int = 5

config_cap = CaptionConfig()