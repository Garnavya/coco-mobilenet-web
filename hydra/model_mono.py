import torch
import torch.nn as nn
import torchvision

class MonolithicMobileNetV3(nn.Module):
    def __init__(self, vocab_size, num_keypoints=17, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        # ==========================================
        # 1. THE SHARED BACKBONE (The Bottleneck)
        # ==========================================
        # This single 960-channel feature extractor must learn to recognize 
        # bounding boxes, human joints, and English semantics simultaneously.
        backbone = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        
        # Shared pooling layer to flatten the features for all heads
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # ==========================================
        # 2. HEAD A: Object Detection (Box Regressor)
        # ==========================================
        self.det_head = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid() # Outputs percentages for [x_min, y_min, x_max, y_max]
        )
        
        # ==========================================
        # 3. HEAD B: Pose Estimation (Joint Regressor)
        # ==========================================
        self.pose_head = nn.Sequential(
            nn.Linear(960, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_keypoints * 3),
            nn.Sigmoid() # Outputs percentages for 17 joints (x, y, visibility)
        )
        
        # ==========================================
        # 4. HEAD C: Image Captioning (Encoder + Decoder)
        # ==========================================
        self.cap_encoder_linear = nn.Linear(960, embed_dim)
        self.cap_encoder_bn = nn.BatchNorm1d(embed_dim, momentum=0.01)
        
        self.cap_embed = nn.Embedding(vocab_size, embed_dim)
        self.cap_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.cap_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions=None):
        # Pass the image through the shared bottleneck
        shared_features = self.features(images)
        shared_flattened = self.pool(shared_features).view(shared_features.size(0), -1)
        
        # Branch 1: Predict Box
        box_preds = self.det_head(shared_flattened)
        
        # Branch 2: Predict Pose
        pose_preds = self.pose_head(shared_flattened)
        
        # Branch 3: Predict Caption
        cap_features = self.cap_encoder_bn(self.cap_encoder_linear(shared_flattened))
        
        if captions is not None:
            # Training mode: Teacher forcing with actual captions
            embeddings = self.cap_embed(captions[:, :-1])
            embeddings = torch.cat((cap_features.unsqueeze(1), embeddings), dim=1)
            hiddens, _ = self.cap_lstm(embeddings)
            cap_preds = self.cap_linear(hiddens)
            return box_preds, pose_preds, cap_preds
        else:
            # Inference mode: We'll handle step-by-step generation in the evaluation script
            return box_preds, pose_preds, cap_features