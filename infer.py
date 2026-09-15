import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import json
import sys
from pathlib import Path
from config import config

# COCO 91-class index to name mapping (subset of common classes)
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 
    20: 'cow', 63: 'laptop', 73: 'laptop', 74: 'mouse', 75: 'remote', 
    76: 'keyboard', 77: 'cell phone', 84: 'book', 88: 'teddy bear'
    # (You can load the full COCO 91-class map if needed, these are the core ones)
}

def run_inference(image_path: str, model_path: str, confidence_threshold: float = 0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 1. Load the trained model architecture
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91)
    
    # 2. Load your saved weights
    if not Path(model_path).exists():
        print(f"[ERROR] Model weights not found at: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode

    # 3. Load and preprocess the image
    if not Path(image_path).exists():
        print(f"[ERROR] Image not found at: {image_path}")
        return
        
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device) # Add batch dimension

    # 4. Run Inference
    print("[INFO] Running inference...")
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # 5. Format outputs into clean JSON
    results = []
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            class_id = int(label)
            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            
            results.append({
                "class": class_name,
                "confidence": round(float(score), 4),
                "box": [round(float(coord), 2) for coord in box] # [x1, y1, x2, y2]
            })

    # Output strictly as simple JSON
    output_json = json.dumps(results, indent=4)
    print("\n--- JSON OUTPUT ---")
    print(output_json)
    print("-------------------")

if __name__ == "__main__":
    # Example usage: Pass the path to your saved model and any test image
    # Replace these paths with your actual run's best_model.pth and a test image path
    MODEL_FILE = "reports/run_YYYYMMDD_HHMMSS/best_model.pth" 
    TEST_IMAGE = "data/coco/val2017/val2017/000000000139.jpg"
    
    # If you run this from terminal: python infer.py path/to/image.jpg
    if len(sys.argv) > 1:
        TEST_IMAGE = sys.argv[1]
        
    # Find the most recent best_model.pth automatically if path is default
    if "YYYYMMDD" in MODEL_FILE:
        runs = sorted(list(config.REPORTS_DIR.glob("run_*")))
        if runs:
            MODEL_FILE = runs[-1] / "best_model.pth"

    run_inference(TEST_IMAGE, str(MODEL_FILE))