import json
import time
import torch
import torchvision.transforms.functional as F
from pathlib import Path
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocoevalcap.eval import COCOEvalCap

# Go up one directory to access the main project folders
BASE_DIR = Path(__file__).resolve().parent.parent
from model_mono import MonolithicMobileNetV3

def run_mono_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Paths
    val_img_dir = BASE_DIR / "data/coco/val2017/val2017"
    val_inst_file = BASE_DIR / "data/coco/annotations_trainval2017/annotations/person_keypoints_val2017.json"
    val_cap_file = BASE_DIR / "data/coco/annotations_trainval2017/annotations/captions_val2017.json"
    vocab_file = BASE_DIR / "vocab.json"
    
    # Find the latest Hydra run
    runs = sorted(list((BASE_DIR / "reports").glob("run_mono_*")))
    if not runs:
        print("[ERROR] No Hydra runs found.")
        return
    run_dir = runs[-1]
    
    # 2. Load Vocab
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
        word2idx = vocab['word2idx']
        idx2word = {int(k): v for k, v in vocab['idx2word'].items()}
        vocab_size = len(word2idx)
        
    # 3. Load Model
    print(f"[INFO] Loading Monolithic Model from {run_dir.name}...")
    model = MonolithicMobileNetV3(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(run_dir / "best_mono.pth", map_location=device, weights_only=True))
    model.eval()
    
    # 4. Prepare COCO Ground Truth
    print("[INFO] Loading COCO Validation sets...")
    coco_inst = COCO(val_inst_file)
    coco_cap = COCO(val_cap_file)
    
    # Filter for Multi-Task validation images (Humans + Captions)
    person_cat_id = coco_inst.getCatIds(catNms=['person'])[0]
    inst_img_ids = set(coco_inst.getImgIds(catIds=[person_cat_id]))
    cap_img_ids = set(coco_cap.getImgIds())
    valid_img_ids = list(inst_img_ids.intersection(cap_img_ids))[:2000] # Validate on subset for speed
    
    box_results = []
    pose_results = []
    cap_results = []
    
    print(f"[INFO] Running Monolithic Inference on {len(valid_img_ids)} images...")
    start_time = time.time()
    
    for i, img_id in enumerate(valid_img_ids):
        file_name = coco_inst.loadImgs(img_id)[0]['file_name']
        img_path = val_img_dir / file_name
        if not img_path.exists(): continue
            
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        tensor_img = F.to_tensor(img.resize((256, 256))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            box_pred, pose_pred, cap_features = model(tensor_img)
            
            # --- PROCESS BOX ---
            # Model outputs percentages [x1, y1, x2, y2]
            b = box_pred[0].cpu().tolist()
            x1, y1, x2, y2 = b[0]*orig_w, b[1]*orig_h, b[2]*orig_w, b[3]*orig_h
            w, h = x2 - x1, y2 - y1
            box_results.append({
                "image_id": img_id, "category_id": person_cat_id,
                "bbox": [x1, y1, w, h], "score": 0.95
            })
            
            # --- PROCESS POSE ---
            # Model outputs 17 keypoints [x_pct, y_pct, v_pct]
            kpts = pose_pred[0].view(17, 3).cpu().numpy()
            formatted_kpts = []
            for kpt in kpts:
                formatted_kpts.extend([float(kpt[0]*orig_w), float(kpt[1]*orig_h), 2 if kpt[2] > 0.5 else 0])
            pose_results.append({
                "image_id": img_id, "category_id": person_cat_id,
                "keypoints": formatted_kpts, "score": 0.95
            })
            
            # --- PROCESS CAPTION (Autoregressive LSTM) ---
            inputs = cap_features.unsqueeze(1)
            _, states = model.cap_lstm(inputs)
            word_id = word2idx["<START>"]
            sampled_ids = []
            
            for _ in range(20):
                inputs = model.cap_embed(torch.tensor([[word_id]]).to(device))
                hiddens, states = model.cap_lstm(inputs, states)
                predicted = model.cap_linear(hiddens.squeeze(1)).argmax(1).item()
                if predicted == word2idx["<END>"]: break
                sampled_ids.append(predicted)
                word_id = predicted
                
            caption = " ".join([idx2word[idx] for idx in sampled_ids if idx not in [word2idx["<PAD>"], word2idx["<UNK>"]]])
            cap_results.append({"image_id": img_id, "caption": caption})
            
        if i % 100 == 0 or i == len(valid_img_ids)-1:
            print(f"\rProcessed {i+1}/{len(valid_img_ids)} images", end="", flush=True)
            
    print(f"\n[INFO] Inference completed in {time.time() - start_time:.2f} seconds.")
    
    # Save results
    box_json = run_dir / "mono_box_results.json"
    pose_json = run_dir / "mono_pose_results.json"
    cap_json = run_dir / "mono_cap_results.json"
    
    with open(box_json, 'w') as f: json.dump(box_results, f)
    with open(pose_json, 'w') as f: json.dump(pose_results, f)
    with open(cap_json, 'w') as f: json.dump(cap_results, f)
        
    print("\n" + "="*40 + "\n        TRACK A: MONOLITHIC METRICS\n" + "="*40)
    
    # 1. Evaluate Box
    print("\n--- 1. OBJECT DETECTION (Box) ---")
    coco_dt_box = coco_inst.loadRes(str(box_json))
    cocoEval = COCOeval(coco_inst, coco_dt_box, 'bbox')
    cocoEval.params.imgIds = valid_img_ids
    cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
    
    # 2. Evaluate Pose
    print("\n--- 2. POSE ESTIMATION (Keypoints) ---")
    coco_dt_pose = coco_inst.loadRes(str(pose_json))
    cocoEvalPose = COCOeval(coco_inst, coco_dt_pose, 'keypoints')
    cocoEvalPose.params.imgIds = valid_img_ids
    cocoEvalPose.evaluate(); cocoEvalPose.accumulate(); cocoEvalPose.summarize()
    
    # 3. Evaluate Captioning
    print("\n--- 3. IMAGE CAPTIONING (NLP) ---")
    coco_dt_cap = coco_cap.loadRes(str(cap_json))
    cocoEvalCap = COCOEvalCap(coco_cap, coco_dt_cap)
    cocoEvalCap.params['image_id'] = coco_dt_cap.getImgIds()
    cocoEvalCap.evaluate()
    for metric, score in cocoEvalCap.eval.items():
        if metric in ['Bleu_4', 'CIDEr']:
            print(f"{metric}: {score:.3f}")

if __name__ == "__main__":
    run_mono_validation()