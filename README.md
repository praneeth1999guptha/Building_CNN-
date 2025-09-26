# Building_CNN-
In this part, you will build and evaluate a custom CNN model for image classification. You will work with a multi-class dataset to practice key aspects of building, training, and evaluating CNNs in PyTorch. Expected accuracy on the test dataset: > 85%.
This repository contains the notebook **`notebooks/Building a CNN.ipynb`**, which trains a Convolutional Neural Network for image classification using **PyTorch + torchvision**.

## ✨ Overview
- **Dataset:** Custom dataset via ImageFolder
- **Input size:** 28×28 with 1 channel(s)
- **Conv layers:** 6 · **Dense layers:** 4
- **Optimizer:** SGD (momentum) · **LR:** 0.001 · **Batch size:** 64 · **Epochs:** 10
- **LR Scheduler:** StepLR
- **Loss:** CrossEntropyLoss
- **Monitoring:** - Confusion matrix & classification report
- **Result:** **Accuracy:** ~92.06%

> The notebook includes data loading, augmentations/transforms, model definition, training loop, evaluation, and plots.

---

## 📦 Data

If you use a builtin dataset (e.g., **Custom dataset via ImageFolder**), it will be downloaded automatically.  
For a **custom dataset** with `ImageFolder`, use this layout:

```
data/
└── train/
    ├── class_0/
    └── class_1/
└── val/
    ├── class_0/
    └── class_1/
```

Update paths in the notebook where the dataset is created.

---

## 🛠️ Setup

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run
```bash
jupyter notebook notebooks/Building\ a\ CNN.ipynb
```
Run cells in order. If GPU is available, PyTorch/TensorFlow will auto-detect it.

---





---

## ⚙️ Reproducibility / Tips
- Set random seeds for `torch`/`tensorflow`, `numpy`, and loader workers.
- Lower `batch_size` if you hit OOM; consider mixed precision (`torch.cuda.amp` or `tf.keras.mixed_precision`).
- Try stronger augmentations (e.g., `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`).
- Adjust model depth/width depending on dataset difficulty.
- Early stopping and LR schedules can stabilize training.

---


