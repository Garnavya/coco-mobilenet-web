import gradio as gr
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import json
from pathlib import Path

# 1. ARCHITECTURE DEFINITIONS (Redefined here to avoid import conflicts)
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

# 2. LOAD MODELS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("[INFO] Loading Track B Pipeline Models...")

# Load SSD Detector (COCO pre-trained fallback if you didn't save a custom weights file for Model 1)
# Using default weights here to guarantee the pipeline runs if the custom checkpoint isn't found
detector = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
detector.to(device).eval()

# Load Pose Estimator
pose_runs = sorted(list(Path("reports").glob("run_pose_*")))
pose_model = MobileNetV3Pose().to(device)
if pose_runs: pose_model.load_state_dict(torch.load(pose_runs[-1] / "best_model_pose.pth", map_location=device, weights_only=True))
pose_model.eval()

# Load Captioner
cap_runs = sorted(list(Path("reports").glob("run_caption_*")))
with open("vocab.json", 'r') as f:
    vocab = json.load(f)
    word2idx = vocab['word2idx']
    idx2word = {int(k): v for k, v in vocab['idx2word'].items()}
    
encoder = EncoderCNN(256).to(device)
decoder = DecoderRNN(256, 512, len(word2idx)).to(device)
if cap_runs:
    encoder.load_state_dict(torch.load(cap_runs[-1] / "best_encoder.pth", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(cap_runs[-1] / "best_decoder.pth", map_location=device, weights_only=True))
encoder.eval()
decoder.eval()

# 3. PIPELINE LOGIC
def run_pipeline(img):
    # Convert Gradio image to PIL and Tensor
    pil_img = Image.fromarray(img).convert("RGB")
    tensor_img = F.to_tensor(pil_img).unsqueeze(0).to(device)
    
    # Step A: Captioning
    caption_text = []
    with torch.no_grad():
        features = encoder(F.to_tensor(pil_img.resize((256, 256))).unsqueeze(0).to(device))
        inputs = features.unsqueeze(1)
        _, states = decoder.lstm(inputs)
        word_id = word2idx["<START>"]
        for _ in range(20):
            inputs = decoder.embed(torch.tensor([[word_id]]).to(device))
            hiddens, states = decoder.lstm(inputs, states)
            predicted = decoder.linear(hiddens.squeeze(1)).argmax(1).item()
            if predicted == word2idx["<END>"]: break
            caption_text.append(idx2word[predicted])
            word_id = predicted
    final_caption = " ".join([w for w in caption_text if w not in ["<PAD>", "<UNK>"]])
    
    # Step B: Detection
    draw = ImageDraw.Draw(pil_img)
    with torch.no_grad():
        det_out = detector(tensor_img)[0]
        
    # Find the highest confidence person (class 1 in COCO)
    best_person_box = None
    best_score = 0.5 # Minimum threshold
    for box, label, score in zip(det_out['boxes'], det_out['labels'], det_out['scores']):
        if label == 1 and score > best_score:
            best_person_box = box.tolist()
            best_score = score
            break
            
    # Step C: Pose Estimation (Top-Down Crop)
    if best_person_box:
        x1, y1, x2, y2 = best_person_box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Crop to the person and predict
        crop = pil_img.crop((x1, y1, x2, y2)).resize((256, 256))
        with torch.no_grad():
            kpts = pose_model(F.to_tensor(crop).unsqueeze(0).to(device))[0].view(17, 3).cpu().numpy()
            
        w, h = x2 - x1, y2 - y1
        for kpt in kpts:
            # kpt is [x_percent, y_percent, visibility_percent]
            if kpt[2] > 0.5: # Only draw visible joints
                abs_x = kpt[0] * w + x1
                abs_y = kpt[1] * h + y1
                draw.ellipse([abs_x-3, abs_y-3, abs_x+3, abs_y+3], fill="cyan", outline="blue")

    return pil_img, final_caption.capitalize() + "."

# 4. GRADIO INTERFACE
demo = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Image(type="numpy", label="Upload or Capture Image"),
    outputs=[gr.Image(type="pil", label="Visual Pipeline Result"), gr.Textbox(label="Generated Caption")],
    title="Track B: Modular Pipeline Demo",
    description="This app runs the image through the Captioning model, uses the SSD Detector to locate the main subject, crops them, and passes the crop to the Pose Regressor."
)

if __name__ == "__main__":
    demo.launch()