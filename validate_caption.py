import json
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO

try:
    from pycocoevalcap.eval import COCOEvalCap
except ImportError:
    print("[ERROR] Please run: pip install pycocoevalcap")
    exit()

from config_caption import config_cap as config

# 1. REBUILD ARCHITECTURE FOR INFERENCE
class EncoderCNN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(960, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)
        
    def forward(self, images):
        features = self.features(images)
        features = self.pool(features).view(features.size(0), -1)
        return self.bn(self.linear(features))

class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

# 2. INFERENCE ALGORITHM (Greedy Search)
def generate_caption(image_tensor, encoder, decoder, word2idx, idx2word, device):
    with torch.no_grad():
        features = encoder(image_tensor)
        
        # Prime the LSTM with the visual summary
        inputs = features.unsqueeze(1)
        _, states = decoder.lstm(inputs)
        
        # Start generating words
        word_id = word2idx["<START>"]
        sampled_ids = []
        
        for _ in range(config.MAX_SEQ_LENGTH):
            inputs = decoder.embed(torch.tensor([[word_id]]).to(device))
            hiddens, states = decoder.lstm(inputs, states)
            outputs = decoder.linear(hiddens.squeeze(1))
            
            predicted = outputs.argmax(1).item()
            
            if predicted == word2idx["<END>"]:
                break
                
            sampled_ids.append(predicted)
            word_id = predicted
            
        # Translate IDs back to English
        caption = [idx2word[idx] for idx in sampled_ids if idx not in [word2idx["<PAD>"], word2idx["<UNK>"]]]
        return " ".join(caption)

def validate_caption():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find the latest caption run
    runs = sorted(list(config.REPORTS_DIR.glob("run_caption_*")))
    if not runs:
        print("[ERROR] No caption runs found.")
        return
        
    run_dir = runs[-1]
    results_path = run_dir / "val_caption_predictions.json"
    
    # Load Vocabulary
    vocab_path = config.BASE_DIR / "vocab.json"
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        word2idx = vocab_data['word2idx']
        # JSON keys are strings, convert back to ints for fast lookup
        idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        
    vocab_size = len(word2idx)
    
    # Load Models
    print(f"[INFO] Loading models from: {run_dir.name}")
    encoder = EncoderCNN(config.EMBEDDING_DIM).to(device)
    decoder = DecoderRNN(config.EMBEDDING_DIM, config.HIDDEN_DIM, vocab_size).to(device)
    
    encoder.load_state_dict(torch.load(run_dir / "best_encoder.pth", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(run_dir / "best_decoder.pth", map_location=device, weights_only=True))
    
    encoder.eval()
    decoder.eval()

    # Load Dataset
    print(f"[INFO] Loading COCO Captions Validation...")
    coco_gt = COCO(str(config.VAL_CAPTIONS_FILE))
    img_ids = list(coco_gt.imgs.keys())
    
    coco_results = []
    print(f"[INFO] Running inference on {len(img_ids)} images...")
    start_time = time.time()
    
    for i, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = config.VAL_IMAGES_DIR / img_info['file_name']
        
        if not img_path.exists(): continue
        
        img = Image.open(img_path).convert("RGB").resize(config.IMAGE_RESOLUTION)
        tensor_img = F.to_tensor(img).unsqueeze(0).to(device)
        
        caption = generate_caption(tensor_img, encoder, decoder, word2idx, idx2word, device)
        
        coco_results.append({
            "image_id": img_id,
            "caption": caption
        })
        
        # Print the first 5 images so you can see it working
        if i < 5:
            print(f"Image {img_info['file_name']} -> \"{caption}\"")
            
        if i % 500 == 0 or i == len(img_ids)-1:
            print(f"\rProcessing image {i+1}/{len(img_ids)}", end="", flush=True)

    print(f"\n[INFO] Inference completed in {time.time() - start_time:.2f} seconds.")
    
    with open(results_path, 'w') as f:
        json.dump(coco_results, f)
        
    print("\n" + "="*40)
    print("    OFFICIAL COCO NLP METRICS")
    print("="*40)
    coco_dt = coco_gt.loadRes(str(results_path))
    coco_eval = COCOEvalCap(coco_gt, coco_dt)
    coco_eval.params['image_id'] = coco_dt.getImgIds()
    coco_eval.evaluate()
    
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

if __name__ == "__main__":
    validate_caption()