import json
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path

# ==========================================
# 1. TRACK B MICRO-MODEL ARCHITECTURES
# ==========================================
class MobileNetV3Pose(torch.nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        self.features = torchvision.models.mobilenet_v3_large(weights=None).features
        self.regressor = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(960, 512),
            torch.nn.Hardswish(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(512, num_keypoints * 3),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.regressor(self.features(x))

class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.features = torchvision.models.mobilenet_v3_large(weights=None).features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Linear(960, embed_dim)
        self.bn = torch.nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, x):
        features = self.features(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return self.bn(self.linear(features))

class DecoderRNN(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)

# ==========================================
# 2. THE UNIFIED INFERENCE ENGINE
# ==========================================
class EdgeVisionEngine:
    """
    A plug-and-play modular pipeline for object detection, pose estimation, 
    and image captioning optimized for edge inference.
    """
    def __init__(self, vocab_path, pose_weights, enc_weights, dec_weights, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Initializing EdgeVisionEngine on {self.device}...")

        # 1. Load Vocabulary
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            self.word2idx = vocab['word2idx']
            self.idx2word = {int(k): v for k, v in vocab['idx2word'].items()}
            self.vocab_size = len(self.word2idx)

        # 2. Load Detector (Pre-trained SSDLite320)
        self.detector = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        ).to(self.device).eval()

        # 3. Load Pose Model
        self.pose_model = MobileNetV3Pose().to(self.device).eval()
        self.pose_model.load_state_dict(torch.load(pose_weights, map_location=self.device, weights_only=True))

        # 4. Load Captioner (Encoder-Decoder)
        self.encoder = EncoderCNN(256).to(self.device).eval()
        self.encoder.load_state_dict(torch.load(enc_weights, map_location=self.device, weights_only=True))
        
        self.decoder = DecoderRNN(256, 512, self.vocab_size).to(self.device).eval()
        self.decoder.load_state_dict(torch.load(dec_weights, map_location=self.device, weights_only=True))

    def _generate_caption(self, image_tensor):
        """Autoregressive caption generation."""
        features = self.encoder(image_tensor).unsqueeze(1)
        _, states = self.decoder.lstm(features)
        
        start_idx = self.word2idx.get("<START>", 1)
        end_idx = self.word2idx.get("<END>", 2)
        pad_idx = self.word2idx.get("<PAD>", 0)
        unk_idx = self.word2idx.get("<UNK>", 3)

        word_id = start_idx
        sampled_ids = []
        
        for _ in range(20): # Max sequence length
            inputs = self.decoder.embed(torch.tensor([[word_id]]).to(self.device))
            hiddens, states = self.decoder.lstm(inputs, states)
            predicted = self.decoder.linear(hiddens.squeeze(1)).argmax(1).item()
            
            if predicted == end_idx:
                break
                
            sampled_ids.append(predicted)
            word_id = predicted
            
        caption = " ".join([self.idx2word[idx] for idx in sampled_ids if idx not in [pad_idx, unk_idx, start_idx, end_idx]])
        return caption

    def process(self, image_path_or_pil):
        """
        Executes the modular pipeline sequentially.
        Returns a structured dictionary of the scene graph.
        """
        if isinstance(image_path_or_pil, str) or isinstance(image_path_or_pil, Path):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")

        global_tensor = F.to_tensor(img).unsqueeze(0).to(self.device)
        
        results = {
            "detections": [],
            "global_caption": ""
        }

        with torch.no_grad():
            # Step 1: Object Detection
            det_out = self.detector(global_tensor)[0]
            
            # Filter for high-confidence persons (Label 1 in COCO)
            for box, label, score in zip(det_out['boxes'], det_out['labels'], det_out['scores']):
                if label == 1 and score > 0.5:
                    x1, y1, x2, y2 = box.cpu().tolist()
                    w, h = x2 - x1, y2 - y1
                    
                    # Step 2: Pose Estimation on the CROP
                    crop = img.crop((x1, y1, x2, y2)).resize((256, 256))
                    crop_tensor = F.to_tensor(crop).unsqueeze(0).to(self.device)
                    
                    # Get local keypoints (0 to 1 scale relative to the crop)
                    local_kpts = self.pose_model(crop_tensor)[0].view(17, 3).cpu().numpy()
                    
                    # Map local keypoints back to global image coordinates
                    global_kpts = []
                    for k in local_kpts:
                        kx = x1 + (k[0] * w)
                        ky = y1 + (k[1] * h)
                        visibility = 2 if k[2] > 0.5 else 0
                        global_kpts.append([round(kx, 2), round(ky, 2), visibility])
                    
                    results["detections"].append({
                        "bounding_box": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                        "confidence": round(score.item(), 3),
                        "keypoints": global_kpts
                    })

            # Step 3: Global Scene Captioning
            # Resize full image to 256x256 for the caption encoder
            cap_tensor = F.to_tensor(img.resize((256, 256))).unsqueeze(0).to(self.device)
            results["global_caption"] = self._generate_caption(cap_tensor)

        return results

# ==========================================
# 3. EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    import pprint
    
    base_dir = Path(__file__).resolve().parent
    vocab_file = base_dir / "vocab.json"
    
    # Auto-resolve latest weights if available
    pose_runs = sorted(list((base_dir / "reports").glob("run_pose_*")))
    cap_runs = sorted(list((base_dir / "reports").glob("run_caption_*")))
    
    pose_weights = pose_runs[-1] / "best_model_pose.pth" if pose_runs else base_dir / "models" / "best_model_pose.pth"
    enc_weights = cap_runs[-1] / "best_encoder.pth" if cap_runs else base_dir / "models" / "best_encoder.pth"
    dec_weights = cap_runs[-1] / "best_decoder.pth" if cap_runs else base_dir / "models" / "best_decoder.pth"
    
    test_image = base_dir / "track_b_sample.jpg"
    
    if pose_weights.exists() and enc_weights.exists() and dec_weights.exists() and test_image.exists():
        engine = EdgeVisionEngine(
            vocab_path=str(vocab_file),
            pose_weights=str(pose_weights),
            enc_weights=str(enc_weights),
            dec_weights=str(dec_weights)
        )
        output = engine.process(test_image)
        print("\n[SUCCESS] Pipeline Execution Complete. Result:")
        pprint.pprint(output)
    else:
        print("[INFO] EdgeVisionEngine loaded successfully. Test skipped as sample image or checkpoints are pending.")