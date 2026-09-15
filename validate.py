import json
import time
import torch
import torchvision
import torchvision.transforms.functional as F
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import config

def validate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Find your trained model
    runs = sorted(list(config.REPORTS_DIR.glob("run_*")))
    if not runs:
        print("[ERROR] No training runs found.")
        return
        
    model_path = runs[-1] / "best_model.pth"
    results_path = runs[-1] / "val_predictions.json"
    
    print(f"[INFO] Loading weights from: {model_path}")
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=91)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load the Validation Dataset Ground Truth
    print(f"\n[INFO] Loading validation annotations...")
    coco_gt = COCO(str(config.VAL_ANNOTATION_FILE))
    img_ids = coco_gt.getImgIds()
    
    coco_results = []
    print(f"[INFO] Running inference on {len(img_ids)} images (This will take a few minutes)...")
    start_time = time.time()
    
    # 3. Run Inference Loop
    with torch.no_grad():
        for i, img_id in enumerate(img_ids):
            # Real-time progress bar
            if i % 10 == 0 or i == len(img_ids)-1:
                print(f"\rProcessing image {i+1}/{len(img_ids)}", end="", flush=True)
                
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = config.VAL_IMAGES_DIR / img_info['file_name']
            
            if not img_path.exists():
                continue
                
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
            
            outputs = model(img_tensor)[0]
            
            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            
            # 4. Format predictions for COCO evaluation
            for box, score, label in zip(boxes, scores, labels):
                if score < 0.05: # Optimization: Ignore extremely low confidence guesses
                    continue
                    
                # PyTorch outputs [x1, y1, x2, y2]. COCO format demands [x, y, width, height]
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                coco_results.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })
                
    print(f"\n[INFO] Inference completed in {time.time() - start_time:.2f} seconds.")
    
    # 5. Save and trigger the official grading algorithm
    if not coco_results:
        print("[WARNING] The model made zero predictions.")
        return
        
    print("[INFO] Saving predictions to JSON and evaluating...\n")
    with open(results_path, 'w') as f:
        json.dump(coco_results, f)
        
    print("="*40)
    print("       OFFICIAL COCO METRICS")
    print("="*40)
    coco_dt = coco_gt.loadRes(str(results_path))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    validate_model()