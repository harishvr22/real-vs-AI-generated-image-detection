  # Deepfake vs Real Image Detection - Fullstack Project

# Real vs AI Image Detection

## Overview
This project implements a deep learning–based image classification system to identify whether an input image is real or AI-generated. The solution addresses the growing challenge of synthetic image misuse by applying a Convolutional Neural Network trained on labeled data.

The model is designed for binary classification and can be reused or extended for research, academic, or early-stage production use cases.

---

## Key Features
- Binary image classification: Real vs AI-generated
- CNN-based architecture using ResNet18
- Standardized image preprocessing pipeline
- Model evaluation using industry-recognized metrics
- Reusable trained model checkpoint

---

## Technical Stack
- Language: Python  
- Framework: PyTorch  
- Model: ResNet18 (CNN)  
- Dataset: KaggleHub (`saurabhbagchi/deepfake-image-detection`)  
- Input Size: 128 × 128  
- Environment: Visual Studio Code  

---

## Architecture Summary
1. Load and label image dataset
2. Resize and normalize input images
3. Train CNN model on training set
4. Evaluate model using test set
5. Predict class for unseen images

---

## Dataset
- Source: KaggleHub
- Classes:
  - Real
  - AI-generated
- Preprocessing:
  - Resize to 128 × 128
  - Normalize pixel values
  - Train–test split

---

## Project Structure
Real-vs-AI-Image-Detection/
├── dataset/
│ ├── real/
│ └── ai/
├── model/
│ └── ai_vs_real_model.pth
├── train.py
├── test.py
├── predict.py
├── requirements.txt
└── README.md

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/harishvr22/Real-vs-AI-Image-Detection.git
cd Real-vs-AI-Image-Detection
pip install -r requirements.txt

