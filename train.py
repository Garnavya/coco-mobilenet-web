import time
import json
import csv
import threading
import traceback
import torch
import torchvision
import torchvision.transforms.functional as F
from datetime import datetime, timezone
from pathlib import Path
from PIL import UnidentifiedImageError

from config import config
from system_info import save_system_info
from monitor import SystemMonitor
from report_generator import generate_final_report

# 1. FAULT-TOLERANT DATASET WRAPPER
class RobustCocoDetection(torchvision.datasets.CocoDetection):
    """Wraps the COCO dataset to catch missing or corrupted images safely."""
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            # If the file is missing or corrupted, quietly ignore it and return None
            return None
        except Exception as e:
            print(f"[WARNING] Unexpected error loading image index {index}: {e}")
            return None

# Transform to convert COCO images and bounding boxes to PyTorch tensors
class ToTensorTargetMap:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        boxes = [obj['bbox'] for obj in target]
        # COCO bbox is [x, y, w, h] -> PyTorch expects [x1, y1, x2, y2]
        boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in boxes if b[2] > 0 and b[3] > 0] 
        labels = [obj['category_id'] for obj in target if obj['bbox'][2] > 0 and obj['bbox'][3] > 0]
        
        # Handle images with no valid bounding boxes
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        formatted_target = {"boxes": boxes, "labels": labels}
        return image, formatted_target

# 2. FAULT-TOLERANT BATCH COLLATOR
def collate_fn(batch):
    # Remove any None values (missing/corrupted images) from this specific batch
    batch = [item for item in batch if item is not None]
    
    # If the entire batch was corrupted (very rare), return empty lists
    if len(batch) == 0:
        return [], []
        
    return tuple(zip(*batch))

def run_training():
    # --- TOGGLE THIS TO FALSE AFTER YOUR TEST ---
    DRY_RUN = False  
    # --------------------------------------------

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.REPORTS_DIR / f"run_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    sys_info_path = run_dir / "system_info.json"
    metrics_csv = run_dir / "system_metrics.csv"
    train_metrics_csv = run_dir / "training_metrics.csv"
    metadata_json = run_dir / "run_metadata.json"
    final_report = run_dir / "final_report.json"
    best_model_path = run_dir / "best_model.pth"

    pause_event = threading.Event()
    abort_event = threading.Event()
    
    metadata = {
        "run_id": f"run_{timestamp_str}",
        "model": config.MODEL_NAME,
        "dataset": config.DATASET_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_index": config.GPU_INDEX,
        "monitoring_interval_seconds": config.MONITOR_INTERVAL_SECONDS,
        "training_start": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "running"
    }
    start_timer = time.time()

    save_system_info(sys_info_path)
    monitor = SystemMonitor(metrics_csv, config.MONITOR_INTERVAL_SECONDS, config.GPU_INDEX, pause_event, abort_event)
    monitor.start()

    with open(train_metrics_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "loss", "epoch_duration_sec"])

    try:
        print(f"Starting run {metadata['run_id']}...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        best_loss = float('inf')

        print(f"[INFO] Loading COCO from {config.TRAIN_ANNOTATION_FILE}...")
        
        # 3. USE THE ROBUST DATASET
        train_dataset = RobustCocoDetection(
            root=str(config.TRAIN_IMAGES_DIR),
            annFile=str(config.TRAIN_ANNOTATION_FILE),
            transforms=ToTensorTargetMap()
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0, 
            collate_fn=collate_fn
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        best_loss = float('inf')
        epochs_without_improvement = 0  # NEW: Tracker for early stopping

        print(f"[INFO] Loading COCO from {config.TRAIN_ANNOTATION_FILE}...")

        for epoch in range(config.EPOCHS):
            epoch_start = time.time()
            epoch_loss = 0.0
            
            model.train() 

            for batch_idx, (images, targets) in enumerate(train_loader): 
                
                # 4. SKIP EMPTY BATCHES
                # If all images in this batch were corrupted, images will be empty. Skip to the next batch.
                if len(images) == 0:
                    continue
                
                if pause_event.is_set():
                    print("\n[WARNING] AC power lost. Pausing...")
                    while pause_event.is_set() and not abort_event.is_set(): time.sleep(2)
                
                if abort_event.is_set():
                    print("\n[CRITICAL] Battery below 40%. Aborting.")
                    metadata["status"] = "interrupted_battery_low"
                    break 
                
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                
                # --- NEW PROGRESS BAR ---
                total_batches = len(train_loader)
                current = batch_idx + 1
                bar_len = 30
                filled_len = int(bar_len * current / total_batches)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                
                print(f"\rEpoch {epoch+1} [{bar}] {current}/{total_batches} | Loss: {losses.item():.4f}", end="", flush=True)
                # ------------------------

            # Print a newline when the epoch finishes or is aborted so it doesn't overwrite the final bar
            print() 

            if abort_event.is_set() or DRY_RUN:
                break

            # --- NEW EARLY STOPPING & SAVING LOGIC ---
            avg_loss = epoch_loss / len(train_loader)
            epoch_dur = round(time.time() - epoch_start, 2)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
                print(f"Epoch {epoch+1} Summary | Loss: {avg_loss:.4f} | Time: {epoch_dur}s -> [BEST MODEL SAVED]")
            else:
                epochs_without_improvement += 1
                print(f"Epoch {epoch+1} Summary | Loss: {avg_loss:.4f} | Time: {epoch_dur}s -> [No improvement for {epochs_without_improvement} epoch(s)]")
            
            # Write to CSV
            with open(train_metrics_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch + 1, round(avg_loss, 4), epoch_dur])
                f.flush()

            # Trigger Early Stopping
            if epochs_without_improvement >= config.PATIENCE:
                print(f"\n[INFO] Early stopping triggered! Loss hasn't improved in {config.PATIENCE} epochs.")
                metadata["status"] = "completed_early_stopping"
                break
            # -----------------------------------------

        if not abort_event.is_set() and not DRY_RUN and metadata["status"] == "running":
            metadata["status"] = "completed"

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")
        metadata["status"] = "interrupted_by_user"
    except Exception as e:
        print(f"\n[ERROR] Exception during training: {e}")
        metadata["status"] = "failed"
        metadata["error_type"] = type(e).__name__
        metadata["error_message"] = str(e)
        metadata["traceback"] = traceback.format_exc()
        raise  
    finally:
        print("\n[INFO] Shutting down monitor and saving reports...")
        monitor.stop()
        metadata["training_end"] = datetime.now(timezone.utc).astimezone().isoformat()
        metadata["training_duration_seconds"] = round(time.time() - start_timer, 2)
        
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        generate_final_report(metrics_csv, final_report, metadata)
        print(f"[SUCCESS] Run saved to: {run_dir}")

if __name__ == "__main__":
    run_training()