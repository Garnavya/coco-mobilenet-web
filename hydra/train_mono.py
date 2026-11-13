import sys
import time
import json
import csv
import threading
import torch
import torch.nn as nn
from pathlib import Path

# Go up one directory to access the main project folders
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Local Track A imports
from model_mono import MonolithicMobileNetV3
from dataset_mono import HydraDataset, collate_hydra

# Telemetry imports
from monitor import SystemMonitor
from system_info import save_system_info
from report_generator import generate_final_report

def run_mono_training():
    DRY_RUN = False  # Toggle this to False after testing
    
    # Setup paths
    img_dir = BASE_DIR / "data/coco/train2017/train2017"
    inst_file = BASE_DIR / "data/coco/annotations_trainval2017/annotations/person_keypoints_train2017.json"
    cap_file = BASE_DIR / "data/coco/annotations_trainval2017/annotations/captions_train2017.json"
    vocab_file = BASE_DIR / "vocab.json"
    
    run_id = f"run_mono_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = BASE_DIR / "reports" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_csv = run_dir / "mono_metrics.csv"
    sys_info_path = run_dir / "system_info.json"
    system_metrics_csv = run_dir / "system_metrics.csv"
    final_report = run_dir / "final_report.json"
    metadata_json = run_dir / "run_metadata.json"
    
    with open(metrics_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "total_loss", "box_loss", "pose_loss", "cap_loss"])
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(vocab_file, 'r') as f:
        vocab_size = len(json.load(f)['word2idx'])
        
    print("[INFO] Initializing Monolithic Track A Model...")
    model = MonolithicMobileNetV3(vocab_size=vocab_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define the math
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
    
    # Loss Multipliers to balance gradients
    W_BOX = 50.0
    W_POSE = 50.0
    W_CAP = 1.0
    
    dataset = HydraDataset(img_dir, inst_file, cap_file, vocab_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_hydra, num_workers=0)
    
    # Setup Telemetry & Fail-safes
    pause_event = threading.Event()
    abort_event = threading.Event()
    battery_threshold = 40.0
    
    metadata = {
        "run_id": run_id,
        "track": "Track A (Monolithic Multi-Task)",
        "model": "MonolithicMobileNetV3",
        "dataset": "COCO-Hydra-Triple",
        "device": str(device),
        "vocab_size": vocab_size,
        "status": "running"
    }
    
    save_system_info(sys_info_path)
    monitor = SystemMonitor(
        output_csv=system_metrics_csv,
        interval_seconds=10.0,
        gpu_index=0,
        pause_event=pause_event,
        abort_event=abort_event,
        battery_abort_threshold=battery_threshold
    )
    monitor.start()
    
    best_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    epochs = 50
    start_time = time.time()
    
    try:
        print("[INFO] Starting Hydra Multi-Task Training with System Telemetry...")
        for epoch in range(epochs):
            model.train()
            epoch_total = 0.0
            epoch_box = 0.0
            epoch_pose = 0.0
            epoch_cap = 0.0
            
            total_batches = len(dataloader)
            
            for batch_idx, (imgs, boxes_gt, poses_gt, caps_gt) in enumerate(dataloader):
                if pause_event.is_set():
                    print("\n[WARNING] AC power lost. Pausing Track A training...")
                    while pause_event.is_set() and not abort_event.is_set():
                        time.sleep(2)
                        
                if abort_event.is_set():
                    print(f"\n[CRITICAL] Battery below {battery_threshold}%. Aborting Track A training.")
                    metadata["status"] = "interrupted_battery_low"
                    break
                
                imgs = imgs.to(device)
                boxes_gt = boxes_gt.to(device)
                poses_gt = poses_gt.to(device)
                caps_gt = caps_gt.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass yields all three predictions
                box_pred, pose_pred, cap_pred = model(imgs, caps_gt)
                
                # 1. Calculate individual losses
                loss_box = criterion_mse(box_pred, boxes_gt)
                loss_pose = criterion_mse(pose_pred, poses_gt)
                loss_cap = criterion_ce(cap_pred.view(-1, vocab_size), caps_gt.reshape(-1))
                
                # 2. Apply static weighting to balance gradients
                total_loss = (loss_box * W_BOX) + (loss_pose * W_POSE) + (loss_cap * W_CAP)
                
                # 3. Backpropagate the unified error
                total_loss.backward()
                optimizer.step()
                
                epoch_total += total_loss.item()
                epoch_box += loss_box.item()
                epoch_pose += loss_pose.item()
                epoch_cap += loss_cap.item()
                
                current = batch_idx + 1
                bar_len = 20
                filled = int(bar_len * current / total_batches)
                bar = '█' * filled + '-' * (bar_len - filled)
                print(f"\rEp {epoch+1} [{bar}] | Tot: {total_loss.item():.2f} | B: {loss_box.item():.4f} P: {loss_pose.item():.4f} C: {loss_cap.item():.2f}", end="", flush=True)
                
                if DRY_RUN and batch_idx == 1:
                    print(f"\n[DRY RUN SUCCESS] VRAM: {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB" if torch.cuda.is_available() else "")
                    break
                    
            if DRY_RUN or abort_event.is_set():
                break
            
            avg_tot = epoch_total / total_batches
            avg_b = epoch_box / total_batches
            avg_p = epoch_pose / total_batches
            avg_c = epoch_cap / total_batches
            
            print(f"\nEpoch {epoch+1} Summary | Tot: {avg_tot:.4f} | Box: {avg_b:.4f} | Pose: {avg_p:.4f} | Cap: {avg_c:.4f}")
            
            with open(metrics_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch+1, round(avg_tot,4), round(avg_b,4), round(avg_p,4), round(avg_c,4)])
            
            if avg_tot < best_loss:
                best_loss = avg_tot
                torch.save(model.state_dict(), run_dir / "best_mono.pth")
                epochs_no_improve = 0
                print("-> [BEST MODEL SAVED]")
            else:
                epochs_no_improve += 1
                print(f"-> [No improvement for {epochs_no_improve} epoch(s)]")
                
            if epochs_no_improve >= patience:
                print("\n[INFO] Early stopping triggered.")
                break
                
        if metadata["status"] == "running":
            metadata["status"] = "completed"
            
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        metadata["status"] = "interrupted_user"
    except Exception as e:
        print(f"\n[ERROR] Training crashed: {e}")
        metadata["status"] = "failed"
    finally:
        metadata["total_time_seconds"] = round(time.time() - start_time, 2)
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        
        monitor.stop()
        generate_final_report(system_metrics_csv, final_report, metadata)
        print(f"[SUCCESS] Run logs and hardware telemetry saved to: {run_dir}")

if __name__ == "__main__":
    run_mono_training()
