import json
import torch
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
import re

class HydraDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, inst_file, cap_file, vocab_file, resolution=(256, 256)):
        print("[INFO] Initializing Hydra Dataset...")
        self.img_dir = Path(img_dir)
        self.res = resolution
        
        with open(vocab_file, 'r') as f:
            self.word2idx = json.load(f)['word2idx']
            
        self.coco_inst = COCO(inst_file)
        self.coco_cap = COCO(cap_file)
        
        # Cross-reference: We only want images with Humans AND Captions
        person_cat_id = self.coco_inst.getCatIds(catNms=['person'])[0]
        inst_img_ids = set(self.coco_inst.getImgIds(catIds=[person_cat_id]))
        cap_img_ids = set(self.coco_cap.getImgIds())
        
        valid_img_ids = list(inst_img_ids.intersection(cap_img_ids))
        
        self.data = []
        for img_id in valid_img_ids:
            ann_ids = self.coco_inst.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=False)
            anns = self.coco_inst.loadAnns(ann_ids)
            
            # Find the most prominent person with valid keypoints
            valid_person = None
            max_area = 0
            for ann in anns:
                if ann['num_keypoints'] > 0 and ann['area'] > max_area:
                    valid_person = ann
                    max_area = ann['area']
                    
            if not valid_person: continue
                
            cap_ids = self.coco_cap.getAnnIds(imgIds=img_id)
            if not cap_ids: continue
            caption = self.coco_cap.loadAnns(cap_ids[0])[0]['caption']
            
            self.data.append({
                'img_id': img_id,
                'file_name': self.coco_inst.loadImgs(img_id)[0]['file_name'],
                'box': valid_person['bbox'], # [x, y, width, height]
                'keypoints': valid_person['keypoints'],
                'caption': str(caption).lower()
            })
            
        print(f"[SUCCESS] Filtered Hydra Dataset: {len(self.data)} valid multi-task images.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            img = Image.open(self.img_dir / item['file_name']).convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize(self.res)
            tensor_img = F.to_tensor(img)
        except Exception:
            return None
            
        # Normalize Box to percentages
        x, y, w, h = item['box']
        box_tensor = torch.tensor([x/orig_w, y/orig_h, (x+w)/orig_w, (y+h)/orig_h], dtype=torch.float32)
        
        # Normalize Pose Keypoints to percentages
        kpts = torch.tensor(item['keypoints'], dtype=torch.float32).view(-1, 3)
        kpts[:, 0] /= orig_w
        kpts[:, 1] /= orig_h
        kpts[:, 2] = (kpts[:, 2] > 0).float()
        pose_tensor = kpts.flatten()
        
        # Tokenize Caption
        tokens = re.findall(r'\w+', item['caption'])
        cap_indices = [self.word2idx["<START>"]] + [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens] + [self.word2idx["<END>"]]
        
        if len(cap_indices) < 20:
            cap_indices.extend([self.word2idx["<PAD>"]] * (20 - len(cap_indices)))
        else:
            cap_indices = cap_indices[:19] + [self.word2idx["<END>"]]
            
        cap_tensor = torch.tensor(cap_indices, dtype=torch.long)
        
        return tensor_img, box_tensor, pose_tensor, cap_tensor

def collate_hydra(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return [], [], [], []
    imgs, boxes, poses, caps = zip(*batch)
    return torch.stack(imgs), torch.stack(boxes), torch.stack(poses), torch.stack(caps)

if __name__ == "__main__":
    # Go up one level since we are inside the 'hydra' folder
    base_dir = Path(__file__).resolve().parent.parent 
    
    img_dir = base_dir / "data/coco/train2017/train2017"
    inst_file = base_dir / "data/coco/annotations_trainval2017/annotations/person_keypoints_train2017.json"
    cap_file = base_dir / "data/coco/annotations_trainval2017/annotations/captions_train2017.json"
    vocab_file = base_dir / "vocab.json"
    
    dataset = HydraDataset(img_dir, inst_file, cap_file, vocab_file)
    
    img, box, pose, cap = dataset[0]
    print(f"\nSample outputs from the Hydra loader:")
    print(f"Box Shape: {box.shape}")
    print(f"Pose Shape: {pose.shape}")
    print(f"Caption Shape: {cap.shape}")