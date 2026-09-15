# The Capacity Bottleneck: Monolithic vs. Modular Multi-Task Learning on Edge Hardware

**Author:** Garnavya Rawal (Meridian Dynamics / University of Lucknow)  
**Hardware Context:** RTX 5050 (8GB VRAM)  

![Edge AI](https://img.shields.io/badge/AI-Edge_Computing-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview
As artificial intelligence pushes toward edge computing, hardware constraints necessitate optimized architectures. This repository investigates the **Capacity Bottleneck**—the mathematical threshold where forcing radically different computational tasks (spatial geometry + NLP) into a single lightweight backbone causes catastrophic accuracy degradation.

This project benchmarks a **Homogeneous Monolithic Architecture (Track A)** against a **Heterogeneous Modular Pipeline (Track B)** for simultaneous object detection, pose estimation, and image captioning on a strictly constrained 5.4-million parameter MobileNetV3 backbone.

## 📊 Key Findings & Results

Testing on consumer-grade hardware (8GB VRAM max) demonstrated that the monolithic architecture suffered a near-total collapse in spatial geometry tasks and semantic generation due to gradient domination and parameter exhaustion. 

| Evaluation Metric | Track A (Monolithic) | Track B (Modular) | Performance Degradation |
| :--- | :--- | :--- | :--- |
| **Object Detection (mAP @ 0.50:0.95)** | 0.037 | ~0.220 | **-83.1%** |
| **Pose Estimation (OKS AP)** | 0.001 | ~0.450 | **-99.7%** |
| **Image Captioning (CIDEr)** | 0.462 | 0.765 | **-39.6%** |
| **Image Captioning (BLEU-4)** | 0.186 | 0.242 | **-23.1%** |


## ⚙️ Architecture details

*   **Track A (Monolithic):** A unified MobileNetV3 architecture utilizing a homogeneous 256x256 input resolution. Trained with a static loss weighting formula: `(50 × Box) + (50 × Pose) + (1 × Caption)` to combat Cross-Entropy gradient domination.
*   **Track B (Modular - Recommended):** A decoupled pipeline utilizing heterogeneous input resolutions optimized per task (SSDLite320 global detector \(\rightarrow\) 256x256 cropped pose regressor + standalone captioning encoder).

## 🚀 Installation & Usage

### 1. Setup the Environment
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/Garnavya/coco-mobilenet-web.git](https://github.com/Garnavya/coco-mobilenet-web.git)
cd coco-mobilenet-web
pip install -r requirements.txt