import time
import json
import csv
import threading
import traceback
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from datetime import datetime
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from pycocotools.coco import COCO

from config_caption import config_cap as config
from system_info import save_system_info
from monitor import SystemMonitor
from report_generator import generate_final_report

# 1. THE ENCODER (Eyes)
class EncoderCNN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(960, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)
        
    def forward(self, images):
        features = self.features(images)
        features = self.pool(features).view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# 2. THE DECODER (Mouth)
class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, features, captions):
        # Drop the last token to align inputs/targets, then embed the text
        embeddings = self.embed(captions[:, :-1])
        # Tack the visual summary onto the front of the sentence
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# 3. NLP DATASET WRAPPER
class CocoCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, vocab_file):
        self.coco = COCO(str(ann_file))
        self.img_dir = img_dir
        self.anns = list(self.coco.anns.values())
        
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
            self.word2idx = vocab_data['word2idx']
        
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_id = ann['image_id']
        caption = str(ann['caption']).lower()
        
        img_info = self.coco.loadImgs(img_id)[0]
        
        try:
            img = Image.open(self.img_dir / img_info['file_name']).convert("RGB")
            img = img.resize(config.IMAGE_RESOLUTION)
            tensor_img = F.to_tensor(img)
            
            # Tokenize the caption
            import re
            tokens = re.findall(r'\w+', caption)
            
            # Convert to integer IDs with START and END tokens
            caption_indices = [self.word2idx["<START>"]]
            for word in tokens:
                caption_indices.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
            caption_indices.append(self.word2idx["<END>"])
            
            # Pad or truncate to MAX_SEQ_LENGTH
            if len(caption_indices) < config.MAX_SEQ_LENGTH:
                padding = [self.word2idx["<PAD>"]] * (config.MAX_SEQ_LENGTH - len(caption_indices))
                caption_indices.extend(padding)
            else:
                caption_indices = caption_indices[:config.MAX_SEQ_LENGTH-1] + [self.word2idx["<END>"]]
                
            return tensor_img, torch.tensor(caption_indices, dtype=torch.long)
            
        except Exception:
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return [], []
    images, captions = zip(*batch)
    return torch.stack(images), torch.stack(captions)

# 4. TRAINING LOOP
def run_training():
    DRY_RUN = False  # Toggle for full training
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.REPORTS_DIR / f"run_caption_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    sys_info_path = run_dir / "system_info.json"
    metrics_csv = run_dir / "system_metrics.csv"
    train_metrics_csv = run_dir / "training_metrics.csv"
    metadata_json = run_dir / "run_metadata.json"
    final_report = run_dir / "final_report.json"
    
    pause_event = threading.Event()
    abort_event = threading.Event()
    
    vocab_path = config.BASE_DIR / "vocab.json"
    with open(vocab_path, 'r') as f:
        vocab_size = len(json.load(f)['word2idx'])

    metadata = {
        "run_id": f"run_caption_{timestamp_str}",
        "model": "MobileNetV3-LSTM",
        "vocab_size": vocab_size,
        "status": "running"
    }
    
    start_timer = time.time()
    save_system_info(sys_info_path)
    monitor = SystemMonitor(metrics_csv, config.MONITOR_INTERVAL_SECONDS, config.GPU_INDEX, pause_event, abort_event)
    monitor.start()

    with open(train_metrics_csv, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "loss", "epoch_duration_sec"])

    try:
        print(f"Starting Caption run {metadata['run_id']}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        encoder = EncoderCNN(config.EMBEDDING_DIM).to(device)
        decoder = DecoderRNN(config.EMBEDDING_DIM, config.HIDDEN_DIM, vocab_size).to(device)
        
        # Use CrossEntropyLoss, but tell it to mathematically ignore <PAD> tokens
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        params = list(encoder.linear.parameters()) + list(encoder.bn.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE)
        
        print(f"[INFO] Loading Caption Dataset...")
        train_dataset = CocoCaptionDataset(config.TRAIN_IMAGES_DIR, config.TRAIN_CAPTIONS_FILE, vocab_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)

        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(config.EPOCHS):
            epoch_start = time.time()
            epoch_loss = 0.0
            encoder.train()
            decoder.train()

            total_batches = len(train_loader)
            for batch_idx, (images, captions) in enumerate(train_loader):
                if abort_event.is_set(): break
                
                images = images.to(device)
                captions = captions.to(device)
                
                # Target is the caption sequence predicting the NEXT word
                targets = captions
                
                features = encoder(images)
                outputs = decoder(features, captions)
                
                # Flatten the outputs and targets for CrossEntropy math
                loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                current = batch_idx + 1
                bar_len = 30
                filled_len = int(bar_len * current / total_batches)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                print(f"\rEpoch {epoch+1} [{bar}] {current}/{total_batches} | CE Loss: {loss.item():.4f}", end="", flush=True)

                if DRY_RUN and batch_idx == 1:
                    print(f"\n[DRY RUN SUCCESS] VRAM Allocated: {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB")
                    break

            print()
            if abort_event.is_set() or DRY_RUN: break

            avg_loss = epoch_loss / total_batches
            epoch_dur = round(time.time() - epoch_start, 2)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(encoder.state_dict(), run_dir / "best_encoder.pth")
                torch.save(decoder.state_dict(), run_dir / "best_decoder.pth")
                epochs_without_improvement = 0
                print(f"Epoch {epoch+1} Summary | CE Loss: {avg_loss:.4f} -> [BEST MODEL SAVED]")
            else:
                epochs_without_improvement += 1
                print(f"Epoch {epoch+1} Summary | CE Loss: {avg_loss:.4f} -> [No improvement for {epochs_without_improvement} epoch(s)]")
            
            with open(train_metrics_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch + 1, round(avg_loss, 4), epoch_dur])
                
            if epochs_without_improvement >= config.PATIENCE:
                print("\n[INFO] Early stopping triggered!")
                break

    except KeyboardInterrupt:
        metadata["status"] = "interrupted_by_user"
    except Exception as e:
        metadata["status"] = "failed"
        print(f"\n[ERROR] Exception: {e}")
        raise
    finally:
        print("\n[INFO] Saving reports...")
        monitor.stop()
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        generate_final_report(metrics_csv, final_report, metadata)
        print(f"[SUCCESS] Run saved to: {run_dir}")

if __name__ == "__main__":
    run_training()