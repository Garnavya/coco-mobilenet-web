# The Capacity Bottleneck: Monolithic vs. Modular Multi-Task Learning on Edge Hardware

**Author:** Garnavya Rawal (Meridian Dynamics / University of Lucknow)  
**Hardware Context:** RTX 5050 Laptop GPU (8GB VRAM)  

![Edge AI](https://img.shields.io/badge/AI-Edge_Computing-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch_2.0-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

As artificial intelligence pushes toward edge computing, hardware constraints necessitate optimized architectures. This repository investigates the **Capacity Bottleneck**—the architectural and mathematical threshold where forcing radically different computational tasks (spatial geometry + natural language generation) into a single lightweight backbone causes catastrophic accuracy degradation.

This project benchmarks a **Homogeneous Monolithic Architecture (Track A)** against a **Heterogeneous Modular Pipeline (Track B)** for simultaneous object detection, pose estimation, and image captioning on a strictly constrained 5.4-million parameter MobileNetV3 backbone.

---

## 📊 Key Findings & Results

Testing on consumer-grade edge hardware (8GB VRAM max) demonstrated that the monolithic architecture suffered a near-total collapse in spatial geometry tasks and semantic generation due to spatial feature vectorization and gradient domination.

| Evaluation Metric | Domain | Track A (Monolithic) | Track B (Modular) | Capacity Delta | Root Architectural Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Object Detection (mAP @ 0.50:0.95)** | Spatial | 0.037 (3.7%) | 0.220 (22.0%) | **-83.1%** | Single global vector box regressor vs. multi-scale anchor grids + NMS. |
| **Pose Estimation (OKS AP)** | Skeletal | 0.001 (0.1%) | 0.450 (45.0%) | **-99.7%** | Global average pooling destroys 2D coordinates; joints cluster at centroid. |
| **Image Captioning (CIDEr)** | Semantics | 0.462 | 0.765 | **-39.6%** | End-to-end backbone thrashing reduces specific vocabulary richness. |
| **Image Captioning (BLEU-4)** | Grammar | 0.186 | 0.242 | **-23.1%** | Autoregressive LSTM syntax decoding remains robust to visual noise. |
| **Inference Latency (RTX 5050)** | Edge Speed | **12.4 ms** | 48.7 ms | **+74.5% Faster** | Track A single-pass forward execution vs 3-stage sequential execution. |
| **Peak VRAM Allocated** | Memory | **1.84 GB** | 4.12 GB | **-55.3% Memory** | Unified feature buffer reuse vs three decoupled networks. |

---

## ⚙️ Architecture Details

### Track A (Monolithic "Hydra"):
* **Backbone:** A unified `MobileNetV3-Large` feature extractor (5.4M parameters) operating at $256 \times 256$ input resolution.
* **Heads:**
  * **Detection Head:** Dense linear layer predicting normalized box coordinates $[x_1, y_1, x_2, y_2]$.
  * **Pose Head:** Linear regression predicting 17 keypoint coordinates $(x, y, v)$.
  * **Captioning Head:** Linear projection + BatchNorm feeding a 512-hidden-dim LSTM with an embedding dimension of 256.
* **Loss Formula:** $(50 \times L_{\text{box}}) + (50 \times L_{\text{pose}}) + (1 \times L_{\text{caption}})$.

### Track B (Modular Pipeline - Recommended):
* **Stage 1 (Global Detection):** `SSDLite320_MobileNet_V3_Large` operating at native $320 \times 320$ resolution to locate subjects with multi-scale anchors.
* **Stage 2 (Localized Pose):** Crops detected person bounding boxes, resizes to $256 \times 256$, and passes them to a top-down `MobileNetV3Pose` regressor.
* **Stage 3 (Global Captioning):** Full-image encoder CNN + LSTM decoder generating autoregressive scene descriptions.

---

## 🔬 Scientific Dissection: Why Did Track A Fall Short?

1. **The Global Pooling Spatial Collapse:**
   In Track A, features pass through `AdaptiveAvgPool2d(1)`, which reduces the $8 \times 8 \times 960$ feature map to a 1D vector of 960 scalar numbers. All 2D spatial coordinate information is averaged out. Expecting linear dense layers to reconstruct 17 individual anatomical joint coordinates across the canvas from a non-spatial summary is mathematically underdetermined, causing joints to collapse around the centroid of mass ($OKS \approx 0.001$).
2. **Oracle Cropping in Track B vs. Full-Image Guessing in Track A:**
   Track B evaluates keypoints on isolated, centered person crops. Track A is forced to infer keypoints across a cluttered full-image canvas without spatial attention or localized cropping.
3. **Gradient Domination & Static Loss Weighting:**
   Cross-Entropy loss across 10,192 dictionary tokens naturally produces scalar values around 9.16 at epoch 1, producing massive backpropagation gradients compared to coordinate MSE losses ($\approx 0.05$). Static weights fail to maintain dynamic equilibrium as tasks converge at non-linear rates.
4. **Backbone Fine-Tuning Thrashing:**
   Updating the shared backbone with three conflicting gradient objectives at $lr=0.001$ degrades pre-trained ImageNet visual tokens, reducing caption specificity (CIDEr score).

---

## 🚀 Installation & Usage

### 1. Setup the Environment
```bash
git clone https://github.com/Garnavya/coco-mobilenet-web.git
cd coco-mobilenet-web
pip install -r requirements.txt
```

### 2. Launch the Unified Web Application

Start the lightweight Python server:
```bash
python server.py
# Opens on http://localhost:8000
```
Or open `web/index.html` directly in any web browser for standalone benchmark inspection.

The unified web application combines:
1. **Model Duel (Live Inference Sandbox):** Drag-and-drop test image upload with real-time canvas rendering of Track A (centroid collapse) vs Track B (SSDLite + top-down articulated pose).
2. **Empirical Benchmark Matrix:** Multi-task metrics (mAP, OKS, CIDEr, BLEU-4) and edge latency/memory trade-offs.
3. **Interactive Analytics:** Chart.js bar comparisons, capability radar profiles, and 20-epoch loss convergence curves.
4. **Root-Cause Architectural Dissection:** Detailed breakdown of the vector pooling spatial trap, cropping disparities, and gradient domination.
5. **Academic Metrics Encyclopedia:** Formal mathematical definitions and failure signatures.
6. **Edge Hardware Telemetry:** Real-time RTX 5050 power, thermal, and VRAM waterfall.

### 3. Programmatic Modular Pipeline Inference
```python
from engine import EdgeVisionEngine

engine = EdgeVisionEngine(
    vocab_path="vocab.json",
    pose_weights="reports/run_pose_20260905_214138/best_model_pose.pth",
    enc_weights="reports/run_caption_20260906_105321/best_encoder.pth",
    dec_weights="reports/run_caption_20260906_105321/best_decoder.pth"
)

result = engine.process("web/assets/track_b_sample.jpg")
print(result)
```

---

## 📁 Repository Structure

```text
coco-mobilenet-web/
├── server.py                # Unified REST API & static server for the web app
├── web/                     # Dedicated professional web application
│   ├── index.html           # Main semantic single-page application
│   ├── css/
│   │   └── styles.css       # Custom styling, dark glassmorphism, responsive UI
│   ├── js/
│   │   ├── app.js           # Master controller, tab routing, API client, presets
│   │   ├── charts.js        # Chart.js visualization engine (bars, radar, loss)
│   │   └── canvas.js        # HTML5 Canvas bounding box & skeleton drawing engine
│   └── assets/              # Benchmark sample assets & diagnostic images
├── build_vocab.py           # Tokenizer & vocabulary builder for COCO captions
├── config.py                # Detection experiment configuration
├── config_caption.py        # Captioning experiment configuration
├── config_pose.py           # Pose experiment configuration
├── engine.py                # Unified modular edge inference engine
├── export_onnx.py           # ONNX model export utility
├── infer.py                 # Standalone detection CLI inference script
├── monitor.py               # Background GPU/CPU telemetry daemon (NVML + psutil)
├── report_generator.py      # Energy (Joules) and telemetry aggregation
├── requirements.txt         # Clean, validated project dependencies
├── system_info.py           # Static hardware profiling logger
├── track_a_sample.jpg       # Visual diagnostic sample (Track A collapse)
├── track_b_sample.jpg       # Visual diagnostic sample (Track B precision)
├── train.py                 # SSDLite detection training script with AMP
├── train_caption.py         # Captioning encoder-decoder training with AMP
├── train_pose.py            # Keypoint regression training with visibility loss & AMP
├── validate.py              # COCO detection evaluator (mAP)
├── validate_caption.py      # Official COCO caption evaluation (CIDEr/BLEU)
├── validate_pose.py         # COCO keypoint evaluation (OKS AP)
├── vocab.json               # 10,192-token dictionary mapping
├── hydra/                   # Track A (Monolithic) implementation
│   ├── dataset_mono.py      # Filtered multi-task triple-overlap dataset
│   ├── model_mono.py        # Monolithic MobileNetV3 multi-head model
│   ├── train_mono.py        # Unified multi-task training loop
│   └── validate_mono.py     # Triple-modality validation script
└── reports/                 # Empirical training runs, telemetry CSVs & weights
```

---

## 📜 License
MIT License. Created by Garnavya Rawal.
