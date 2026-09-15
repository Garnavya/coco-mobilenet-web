import json
import torch
import torchvision
import gradio as gr
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from pathlib import Path

# Import Track A
from hydra.model_mono import MonolithicMobileNetV3

BASE_DIR = Path(__file__).resolve().parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. TRACK B ARCHITECTURE DEFINITIONS
# ==========================================
class MobileNetV3Pose(torch.nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        self.features = torchvision.models.mobilenet_v3_large(weights=None).features
        self.regressor = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(960, 512),
            torch.nn.Hardswish(), torch.nn.Dropout(p=0.2), torch.nn.Linear(512, num_keypoints * 3),
            torch.nn.Sigmoid()
        )
    def forward(self, x): return self.regressor(self.features(x))

class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.features = torchvision.models.mobilenet_v3_large(weights=None).features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Linear(960, embed_dim)
        self.bn = torch.nn.BatchNorm1d(embed_dim, momentum=0.01)
    def forward(self, x): return self.bn(self.linear(self.pool(self.features(x)).view(x.size(0), -1)))

class DecoderRNN(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)

# ==========================================
# 2. LOAD ALL MODELS
# ==========================================
with open(BASE_DIR / "vocab.json", 'r') as f:
    vocab = json.load(f)
    word2idx = vocab['word2idx']
    idx2word = {int(k): v for k, v in vocab['idx2word'].items()}
    vocab_size = len(word2idx)

print("[INFO] Loading Track A (Monolithic)...")
mono_run = sorted(list((BASE_DIR / "reports").glob("run_mono_*")))[-1] 
model_a = MonolithicMobileNetV3(vocab_size=vocab_size).to(device).eval()
model_a.load_state_dict(torch.load(mono_run / "best_mono.pth", map_location=device, weights_only=True))

print("[INFO] Loading Track B (Modular)...")
detector_b = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).to(device).eval()

pose_b = MobileNetV3Pose().to(device).eval()
pose_run = sorted(list((BASE_DIR / "reports").glob("run_pose_*")))[-1]
pose_b.load_state_dict(torch.load(pose_run / "best_model_pose.pth", map_location=device, weights_only=True))

encoder_b = EncoderCNN(256).to(device).eval()
decoder_b = DecoderRNN(256, 512, vocab_size).to(device).eval()
cap_run = sorted(list((BASE_DIR / "reports").glob("run_caption_*")))[-1]
encoder_b.load_state_dict(torch.load(cap_run / "best_encoder.pth", map_location=device, weights_only=True))
decoder_b.load_state_dict(torch.load(cap_run / "best_decoder.pth", map_location=device, weights_only=True))

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def draw_predictions(image, box, keypoints, color):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    if box:
        x1, y1, w, h = box
        draw.rectangle([x1, y1, x1+w, y1+h], outline=color, width=4)
    for i in range(0, len(keypoints), 3):
        kx, ky, kv = keypoints[i], keypoints[i+1], keypoints[i+2]
        if kv > 0: draw.ellipse([kx-4, ky-4, kx+4, ky+4], fill="blue", outline="white")
    return img_draw

def generate_caption(features, lstm_model, linear_model, embed_model):
    inputs = features.unsqueeze(1)
    _, states = lstm_model(inputs)
    word_id = word2idx["<START>"]
    sampled_ids = []
    for _ in range(20):
        inputs = embed_model(torch.tensor([[word_id]]).to(device))
        hiddens, states = lstm_model(inputs, states)
        predicted = linear_model(hiddens.squeeze(1)).argmax(1).item()
        if predicted == word2idx["<END>"]: break
        sampled_ids.append(predicted)
        word_id = predicted
    return " ".join([idx2word[idx] for idx in sampled_ids if idx not in [word2idx["<PAD>"], word2idx["<UNK>"]]])

# ==========================================
# 4. INFERENCE ENGINE
# ==========================================
def run_comparison(input_image):
    orig_w, orig_h = input_image.size
    
    # --- TRACK A INFERENCE ---
    img_a = input_image.resize((256, 256))
    tensor_a = F.to_tensor(img_a).unsqueeze(0).to(device)
    
    with torch.no_grad():
        box_pred, pose_pred, cap_feat_a = model_a(tensor_a)
        b = box_pred[0].cpu().tolist()
        x1_a, y1_a = b[0] * orig_w, b[1] * orig_h
        w_a, h_a = (b[2] * orig_w) - x1_a, (b[3] * orig_h) - y1_a
        
        kpts_a = pose_pred[0].view(17, 3).cpu().numpy()
        fmt_kpts_a = []
        for k in kpts_a: fmt_kpts_a.extend([k[0] * orig_w, k[1] * orig_h, 2 if k[2] > 0.5 else 0])
        
        cap_a = generate_caption(cap_feat_a, model_a.cap_lstm, model_a.cap_linear, model_a.cap_embed)
        
    annotated_a = draw_predictions(input_image, [x1_a, y1_a, w_a, h_a], fmt_kpts_a, color="red")

    # --- TRACK B INFERENCE ---
    with torch.no_grad():
        # 1. Detector
        det_out = detector_b(F.to_tensor(input_image).unsqueeze(0).to(device))[0]
        box_b = None
        fmt_kpts_b = []
        
        for box, label, score in zip(det_out['boxes'], det_out['labels'], det_out['scores']):
            if label == 1 and score > 0.5: # Person found
                bx1, by1, bx2, by2 = box.cpu().tolist()
                bw, bh = bx2 - bx1, by2 - by1
                box_b = [bx1, by1, bw, bh]
                
                # 2. Pose on CROP
                crop = input_image.crop((bx1, by1, bx2, by2)).resize((256, 256))
                kpts_b = pose_b(F.to_tensor(crop).unsqueeze(0).to(device))[0].view(17, 3).cpu().numpy()
                for k in kpts_b:
                    fmt_kpts_b.extend([bx1 + (k[0] * bw), by1 + (k[1] * bh), 2 if k[2] > 0.5 else 0])
                break # Just process the most confident person
        
        # 3. Caption
        cap_feat_b = encoder_b(F.to_tensor(input_image.resize((256, 256))).unsqueeze(0).to(device))
        cap_b = generate_caption(cap_feat_b, decoder_b.lstm, decoder_b.linear, decoder_b.embed)
        
    annotated_b = draw_predictions(input_image, box_b, fmt_kpts_b, color="green")
    
    return annotated_b, cap_b, annotated_a, cap_a

# ==========================================
# 5. GRADIO UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Capacity Bottleneck: Modular vs. Monolithic Edge AI")
    gr.Markdown("Upload an image to visually compare how a dedicated pipeline (Track B) performs against a bottlenecked shared backbone (Track A).")
    
    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Test Image")
        
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✅ Track B: Modular Pipeline (Green)")
            out_img_b = gr.Image(label="Track B Visuals")
            out_cap_b = gr.Textbox(label="Track B Caption", lines=2)
            
        with gr.Column():
            gr.Markdown("### ❌ Track A: Monolithic Hydra (Red)")
            out_img_a = gr.Image(label="Track A Visuals")
            out_cap_a = gr.Textbox(label="Track A Caption", lines=2)
            
    submit_btn = gr.Button("Run Comparative Inference", variant="primary")
    submit_btn.click(fn=run_comparison, inputs=input_img, outputs=[out_img_b, out_cap_b, out_img_a, out_cap_a])

if __name__ == "__main__":
    demo.launch()