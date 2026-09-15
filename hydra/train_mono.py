import time
import json
import csv
import torch
import torch.nn as nn
from pathlib import Path

# Go up one directory to access the main project folders
BASE_DIR = Path(__file__).resolve().parent.parent

# Local Track A imports
from model_mono import MonolithicMobileNetV3
from dataset_mono import HydraDataset, collate_hydra

def run_mono_training():
    DRY_RUN = False  # Toggle this to False after testing
    
    # Setup paths
    img_dir = BASE_DIR / "data/coco/train2017/train2017"
    inst_file = BASE_DIR / "data/coco/annotations_trainval2017/annotations/person_keypoints_train2017.json"
    cap_file = BASE_DIR / "data/coco/annotations_trainval2017/annotations/captions_train2017.json"
    vocab_file = BASE_DIR / "vocab.json"
    
    run_dir = BASE_DIR / "reports" / f"run_mono_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = run_dir / "mono_metrics.csv"
    
    with open(metrics_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "total_loss", "box_loss", "pose_loss", "cap_loss"])
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(vocab_file, 'r') as f:
        vocab_size = len(json.load(f)['word2idx'])
        
    print("[INFO] Initializing Monolithic Track A Model...")
    model = MonolithicMobileNetV3(vocab_size=vocab_size).to(device)
    
    # Unlike Track B, we are training the ENTIRE network simultaneously
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define the math
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
    
    # Loss Multipliers to prevent Gradient Domination
    W_BOX = 50.0
    W_POSE = 50.0
    W_CAP = 1.0
    
    dataset = HydraDataset(img_dir, inst_file, cap_file, vocab_file)
    # Reduced batch size to 32 because multi-task gradients consume more VRAM
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_hydra, num_workers=0)
    
    best_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    epochs = 50
    
    try:
        print("[INFO] Starting Hydra Multi-Task Training...")
        for epoch in range(epochs):
            model.train()
            epoch_total = 0.0
            epoch_box = 0.0
            epoch_pose = 0.0
            epoch_cap = 0.0
            
            total_batches = len(dataloader)
            
            for batch_idx, (imgs, boxes_gt, poses_gt, caps_gt) in enumerate(dataloader):
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
                # Flatten caption outputs for CrossEntropy
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
                    print(f"\n[DRY RUN SUCCESS] VRAM: {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB")
                    break
                    
            if DRY_RUN: break
            
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
                
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    finally:
        print(f"[SUCCESS] Run logs saved to: {run_dir}")

if __name__ == "__main__":
    run_mono_training()