import time
import json
import csv
import threading
import traceback
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from pycocotools.coco import COCO

from config_pose import config_pose as config
from system_info import save_system_info
from monitor import SystemMonitor
from report_generator import generate_final_report

# 1. THE POSE ESTIMATOR MODEL
class MobileNetV3Pose(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        
        # Replace the classification head with a 51-value regression head
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_keypoints * 3),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

# 2. ROBUST DATASET WRAPPER
class CocoTopDownKeypoints(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file):
        self.coco = COCO(str(ann_file))
        self.img_dir = img_dir
        self.anns = [ann for ann in self.coco.dataset['annotations'] 
                     if 'keypoints' in ann and ann['num_keypoints'] > 0]

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        
        try:
            img = Image.open(self.img_dir / img_info['file_name']).convert("RGB")
            
            x, y, w, h = [int(v) for v in ann['bbox']]
            if w <= 0 or h <= 0: return None
            
            cropped_img = img.crop((x, y, x + w, y + h))
            cropped_img = cropped_img.resize((256, 256))
            tensor_img = F.to_tensor(cropped_img)
            
            kpts = torch.tensor(ann['keypoints'], dtype=torch.float32).view(-1, 3)
            kpts[:, 0] = (kpts[:, 0] - x) / w
            kpts[:, 1] = (kpts[:, 1] - y) / h
            kpts[:, 2] = kpts[:, 2] / 2.0  
            
            return tensor_img, kpts.flatten()
            
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None
        except Exception:
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return [], []
    return tuple(zip(*batch))

# 3. MASKED LOSS FUNCTION
def masked_keypoint_loss(preds, targets):
    """Calculates Mean Squared Error only on visible keypoints."""
    preds = preds.view(-1, 17, 3)
    targets = targets.view(-1, 17, 3)
    
    # Check visibility flag (targets > 0)
    mask = targets[:, :, 2] > 0
    
    # We only care about training X and Y coordinates (indices 0 and 1)
    pred_xy = preds[:, :, :2]
    target_xy = targets[:, :, :2]
    
    # Calculate MSE loss only where the keypoint actually exists
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
        
    return nn.functional.mse_loss(pred_xy[mask], target_xy[mask])

# 4. TRAINING LOOP
def run_training():
    DRY_RUN = False  # Toggle for full training

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Prefix folder with 'run_pose_' to differentiate from object detector
    run_dir = config.REPORTS_DIR / f"run_pose_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    sys_info_path = run_dir / "system_info.json"
    metrics_csv = run_dir / "system_metrics.csv"
    train_metrics_csv = run_dir / "training_metrics.csv"
    metadata_json = run_dir / "run_metadata.json"
    final_report = run_dir / "final_report.json"
    best_model_path = run_dir / "best_model_pose.pth"

    pause_event = threading.Event()
    abort_event = threading.Event()
    
    metadata = {
        "run_id": f"run_pose_{timestamp_str}",
        "model": "MobileNetV3-Pose",
        "dataset": config.DATASET_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_index": config.GPU_INDEX,
        "status": "running"
    }
    
    start_timer = time.time()
    save_system_info(sys_info_path)
    monitor = SystemMonitor(metrics_csv, config.MONITOR_INTERVAL_SECONDS, config.GPU_INDEX, pause_event, abort_event)
    monitor.start()

    with open(train_metrics_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "loss", "epoch_duration_sec"])

    try:
        print(f"Starting Pose run {metadata['run_id']}...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MobileNetV3Pose().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        best_loss = float('inf')
        epochs_without_improvement = 0

        print(f"[INFO] Loading Pose Dataset...")
        train_dataset = CocoTopDownKeypoints(config.TRAIN_IMAGES_DIR, config.TRAIN_KEYPOINTS_FILE)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
            num_workers=0, collate_fn=collate_fn
        )

        for epoch in range(config.EPOCHS):
            epoch_start = time.time()
            epoch_loss = 0.0
            model.train() 

            total_batches = len(train_loader)
            for batch_idx, (images, targets) in enumerate(train_loader): 
                if len(images) == 0: continue
                
                if abort_event.is_set(): break 
                
                # Stack tuples into single tensors and move to GPU
                images = torch.stack(images).to(device)
                targets = torch.stack(targets).to(device)

                outputs = model(images)
                loss = masked_keypoint_loss(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                current = batch_idx + 1
                bar_len = 30
                filled_len = int(bar_len * current / total_batches)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                print(f"\rEpoch {epoch+1} [{bar}] {current}/{total_batches} | MSE Loss: {loss.item():.4f}", end="", flush=True)

                if DRY_RUN and batch_idx == 1:
                    print(f"\n[DRY RUN SUCCESS] Peak VRAM: {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB")
                    break 

            print()
            if abort_event.is_set() or DRY_RUN: break 

            avg_loss = epoch_loss / total_batches
            epoch_dur = round(time.time() - epoch_start, 2)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
                print(f"Epoch {epoch+1} Summary | MSE Loss: {avg_loss:.4f} -> [BEST MODEL SAVED]")
            else:
                epochs_without_improvement += 1
                print(f"Epoch {epoch+1} Summary | MSE Loss: {avg_loss:.4f} -> [No improvement for {epochs_without_improvement} epoch(s)]")
            
            with open(train_metrics_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch + 1, round(avg_loss, 4), epoch_dur])
                
            if epochs_without_improvement >= config.PATIENCE:
                print("\n[INFO] Early stopping triggered!")
                metadata["status"] = "completed_early_stopping"
                break

        if not abort_event.is_set() and not DRY_RUN and metadata["status"] == "running":
            metadata["status"] = "completed"

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        metadata["status"] = "interrupted_by_user"
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        metadata["status"] = "failed"
        metadata["error_message"] = str(e)
        metadata["traceback"] = traceback.format_exc()
        raise  
    finally:
        print("\n[INFO] Shutting down monitor and saving reports...")
        monitor.stop()
        metadata["training_duration_seconds"] = round(time.time() - start_timer, 2)
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        generate_final_report(metrics_csv, final_report, metadata)
        print(f"[SUCCESS] Run saved to: {run_dir}")

if __name__ == "__main__":
    run_training()