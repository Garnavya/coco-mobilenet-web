import json
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config_pose import config_pose as config

# Re-define model here for standalone execution
class MobileNetV3Pose(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(960, 512),
            nn.Hardswish(), nn.Dropout(p=0.2), nn.Linear(512, num_keypoints * 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

def validate_pose():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Find the best pose model
    runs = sorted(list(config.REPORTS_DIR.glob("run_pose_*")))
    if not runs:
        print("[ERROR] No pose training runs found.")
        return
        
    model_path = runs[-1] / "best_model_pose.pth"
    results_path = runs[-1] / "val_pose_predictions.json"
    
    print(f"[INFO] Loading weights from: {model_path}")
    model = MobileNetV3Pose().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load Validation Set
    print(f"[INFO] Loading COCO Keypoints Validation...")
    coco_gt = COCO(str(config.VAL_KEYPOINTS_FILE))
    
    # Get only annotations that are people with valid keypoints
    anns = [ann for ann in coco_gt.dataset['annotations'] if ann.get('num_keypoints', 0) > 0]
    
    coco_results = []
    print(f"[INFO] Running inference on {len(anns)} cropped humans...")
    start_time = time.time()
    
    # 3. Top-Down Inference Loop
    with torch.no_grad():
        for i, ann in enumerate(anns):
            if i % 100 == 0 or i == len(anns)-1:
                print(f"\rProcessing person {i+1}/{len(anns)}", end="", flush=True)
                
            img_id = ann['image_id']
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = config.VAL_IMAGES_DIR / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            # Get the box to crop the human
            x, y, w, h = [float(v) for v in ann['bbox']]
            if w <= 0 or h <= 0: continue
            
            img = Image.open(img_path).convert("RGB")
            cropped_img = img.crop((x, y, x + w, y + h)).resize(config.IMAGE_RESOLUTION)
            tensor_img = F.to_tensor(cropped_img).unsqueeze(0).to(device)
            
            # Predict
            output = model(tensor_img)[0].view(17, 3).cpu().numpy()
            
            # Un-normalize coordinates back to absolute image space
            keypoints_out = []
            for kpt in output:
                norm_x, norm_y, norm_v = kpt
                abs_x = norm_x * w + x
                abs_y = norm_y * h + y
                # Convert our 0.0-1.0 visibility back to COCO's 0, 1, 2 format
                abs_v = 2 if norm_v > 0.5 else 1 
                
                keypoints_out.extend([float(abs_x), float(abs_y), int(abs_v)])
            
            coco_results.append({
                "image_id": img_id,
                "category_id": 1, # Person
                "keypoints": keypoints_out,
                "score": float(output[:, 2].mean()) # Use average visibility as confidence score
            })
            
    print(f"\n[INFO] Inference completed in {time.time() - start_time:.2f} seconds.")
    
    # 4. Save and Score
    with open(results_path, 'w') as f:
        json.dump(coco_results, f)
        
    print("="*40)
    print("    OFFICIAL COCO KEYPOINT METRICS")
    print("="*40)
    coco_dt = coco_gt.loadRes(str(results_path))
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    validate_pose()