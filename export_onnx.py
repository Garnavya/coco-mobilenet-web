import json
import torch
import torchvision
from pathlib import Path

# 1. ARCHITECTURE DEFINITIONS
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

# Wrapper to export the LSTM for step-by-step word generation
class ONNXDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
    def forward(self, word_id, hidden_state, cell_state):
        embeds = self.decoder.embed(word_id)
        lstm_out, (h_new, c_new) = self.decoder.lstm(embeds, (hidden_state, cell_state))
        logits = self.decoder.linear(lstm_out.squeeze(1))
        return logits, h_new, c_new

def export_models():
    device = torch.device('cpu') # Exporting is safer and cleaner on CPU
    out_dir = Path("onnx_models")
    out_dir.mkdir(exist_ok=True)
    opset = 12 # Highly compatible with web browsers

    #print("[INFO] Exporting Model 1: SSD Detector...")
    #detector = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    #detector.eval()
    #dummy_img_320 = torch.randn(1, 3, 320, 320)
    #torch.onnx.export(detector, dummy_img_320, out_dir / "detector.onnx", 
    #                 opset_version=opset, input_names=['image'], output_names=['boxes', 'labels', 'scores'])

    print("[INFO] Exporting Model 2: Pose Regressor...")
    pose_runs = sorted(list(Path("reports").glob("run_pose_*")))
    pose_model = MobileNetV3Pose()
    if pose_runs: pose_model.load_state_dict(torch.load(pose_runs[-1] / "best_model_pose.pth", map_location=device, weights_only=True))
    pose_model.eval()
    dummy_img_256 = torch.randn(1, 3, 256, 256)
    torch.onnx.export(pose_model, dummy_img_256, out_dir / "pose.onnx", 
                      opset_version=opset, input_names=['image'], output_names=['keypoints'])

    print("[INFO] Loading Vocabulary for NLP Models...")
    with open("vocab.json", 'r') as f:
        vocab_size = len(json.load(f)['word2idx'])

    print("[INFO] Exporting Model 3a: Caption Encoder...")
    cap_runs = sorted(list(Path("reports").glob("run_caption_*")))
    encoder = EncoderCNN(256)
    decoder = DecoderRNN(256, 512, vocab_size)
    if cap_runs:
        encoder.load_state_dict(torch.load(cap_runs[-1] / "best_encoder.pth", map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load(cap_runs[-1] / "best_decoder.pth", map_location=device, weights_only=True))
    encoder.eval()
    decoder.eval()
    
    torch.onnx.export(encoder, dummy_img_256, out_dir / "caption_encoder.onnx", 
                      opset_version=opset, input_names=['image'], output_names=['visual_embedding'])

    print("[INFO] Exporting Model 3b: Caption Decoder...")
    decoder_wrapper = ONNXDecoderWrapper(decoder).eval()
    dummy_word = torch.tensor([[1]]) # <START> token ID
    dummy_h = torch.randn(1, 1, 512)
    dummy_c = torch.randn(1, 1, 512)
    
    torch.onnx.export(decoder_wrapper, (dummy_word, dummy_h, dummy_c), out_dir / "caption_decoder.onnx",
                      opset_version=opset, 
                      input_names=['word_id', 'hidden_in', 'cell_in'], 
                      output_names=['logits', 'hidden_out', 'cell_out'])

    print(f"[SUCCESS] All models exported to ONNX format in the '{out_dir.name}' folder!")

if __name__ == "__main__":
    export_models()