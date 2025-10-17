# 🧠 Data Augmentation for Object Detection (Pascal VOC – “Cow” Class)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Albumentations](https://img.shields.io/badge/Albumentations-OK-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

A compact but complete project comparing **data augmentation techniques** for **object detection** using **Faster R-CNN (MobileNetV3-Large 320 FPN)** on the **Pascal VOC 2012** dataset.  
All experiments focus on a single class — **cow** — to simulate low-data scenarios.

📄 Full technical report → [report.pdf](./report.pdf)

---

## 🚀 Overview

This project studies the effect of various augmentation strategies on object detection accuracy under limited data conditions.

### 🧩 Models Compared
| Model | Transform | Description |
|--------|------------|-------------|
| **Baseline** | Resize + Normalize | Minimal preprocessing |
| **Flip** | + HorizontalFlip | Simple geometric flip |
| **Crop** | + RandomSizedBBoxSafeCrop | Object-partial exposure |
| **Rotate** | + ShiftScaleRotate | Small-angle rotation |
| **Mixed** | Combo | Flip + Color Jitter + Rotate + Crop |

### 📊 Metrics Evaluated
- Average training loss per epoch  
- Precision @ IoU ≥ 0.5  
- Average IoU  

---

## ⚙️ Installation

1️⃣ Clone this repository  
`git clone https://github.com/<your-username>/<repo-name>.git`  
`cd <repo-name>`

2️⃣ Create and activate a virtual environment  
`python -m venv .venv`  
`source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)

3️⃣ Install dependencies  
`pip install --upgrade pip`  
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`  
`pip install albumentations opencv-python matplotlib pandas numpy tqdm`

⚠️ Note: To quickly test results, open the notebook in Google Colab.
Pretrained models are included under the model/ directory — just load them using the provided load_trained_model() function before running the evaluation cells.

---

## 📦 Dataset Setup

1. Download **Pascal VOC 2012** from [the official site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).  
2. Extract it into the following structure:

data/  
└── VOCdevkit/  
  └── VOC2012/  
   ├── JPEGImages/  
   ├── Annotations/  
   └── ImageSets/  
    └── Main/  

3. The notebook automatically filters data to include only the **cow** class.

---

## ▶️ Running the Notebook

Run locally:  
`jupyter notebook notebooks/NN_PROJECT_CODE.ipynb`

Or open it in **Google Colab** (a Drive-mount cell is already included).

---

## 📊 Results Summary

| Model | Precision | Avg IoU | Avg Loss |
|--------|-----------|---------|-----------|
| **Baseline** | 0.433 | 0.381 | 0.407 |
| **Flip** | **0.446** | **0.406** | **0.399** |
| **Crop** | **0.435** | **0.406** | **0.399** |
| **Rotate** | 0.442 | 0.397 | 0.506 |
| **Mixed** | 0.416 | 0.380 | 0.504 |

### 🔍 Key Insights
- Simple augmentations like **Flip** and **Crop** improved IoU by ≈ 6–7% over baseline.  
- Overly complex “Mixed” augmentation showed lower performance because the number of training epochs remained the same. Since the mixed augmentation effectively increases the dataset size, more epochs would be needed to achieve full convergence. 
- Effectiveness depends on dataset size and diversity.

See detailed plots and comparisons in [report.pdf](./report.pdf).

---

## 🧠 How It Works

- **Dataset Loader:** Custom `VOCDataset` parses Pascal VOC XML annotations (filtered to “cow”).  
- **Transforms:** Albumentations pipelines — flip, crop, rotate, mixed.  
- **Model:** `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn` with custom head.  
- **Training:** SGD optimizer, checkpointing, IoU / precision computation.  
- **Evaluation:** Loss curves and IoU comparisons between training & testing.

---

## 🧪 Reproducing Experiments

Inside the notebook, run:

`train_and_evaluate(model_name="flip", transform_fn=get_flip_transforms, num_epochs=30, target_class="cow", num_classes=2)`

Repeat for each augmentation type (`baseline`, `flip`, `crop`, `rotate`, `mix`) to reproduce all experiments.

---

## 💡 Future Improvements

- Extend to multi-class detection  
- Integrate **AutoAugment / RandAugment**  
- Report **mAP @ 0.5:0.95** metrics  
- Convert notebook into CLI script: `python src/train.py --exp flip`

---

## 🪪 License

This project is released under the **MIT License**.  
Pascal VOC 2012 dataset follows its original license.

---

## 🙌 Acknowledgments

- **Albumentations** – Augmentation library  
- **PyTorch / torchvision** – Detection backbone  
- **Pascal VOC 2012** – Dataset  
- Developed as part of the *Neural Networks* course project (July 2025)

---

## 📖 Citation

*Data Augmentation for Object Detection: A Comparative Study of Transformation Techniques.*  
Neural Networks Course Project, July 2025.  

---

⭐ **If you find this project helpful, please give it a star!**
