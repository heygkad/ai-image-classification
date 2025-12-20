# Facial Emotion Recognition

Real-time facial emotion recognition using transfer learning on various Vision Transformers originally trained for object classification. Detects 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral.

# IMPORTANT:
## All of our trained models must be downloaded from this google drive for testing: [Drive Folder Link](https://drive.google.com/drive/folders/1jRLWKNgA3CNP6lV7YG9vYRan0H0rKTqZ?usp=sharing)
## After downloading, unzip, and drag the model folders contained (ViT-Base, DeIT-Small, etc.) into the `models/` folder
### This is due to GitHub not allowing file sizes larger than 100MB, and using GitHub LFS seemed to break our models' safetensor files when pulling from GitHub.

## First-time setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# If you're on a Mac
source .venv/bin/activate

# If you're on a Windows
.venv\Scripts\activate
```

3. Install **specific dependencies (This is very important, otherwise code that ran in my virtual environment might not run on someone else's)**:
```bash
pip install -r requirements.txt
```

---

## Main Files

### `train.py`

Fine-tunes a specified transformer model on the RAF-DB 7-class facial emotion dataset.

**What it does:**
- Loads images from the RAF-DB dataset: `DATASET/train` and `DATASET/test`
- Splits training data 80/20 into train/validation
- Fine-tunes user-specified `model_name` for 15 epochs
- Saves the best model (by validation accuracy) to `fine_tuned_emotion_model/`

**To train our models we used the following specifications:**
- GPU: GTX 3070ti
- CPU: Intel i7-12700h
- RAM: 16GB DDR4

**How to use:**
```bash
python train.py
```

---

### `webcam_emotion.py`

Real-time emotion detection using webcam, run this to see live emotion detection.

**What it does:**
- Opens webcam and detects faces using Haar cascades
- Runs emotion prediction on detected faces
- Displays emotion label with color-coded bounding box
- Allows switching between different trained models via dropdown on the top-left

**How to use:**
```bash
python webcam_emotion.py
```

**Controls:**
- Click the dropdown (top-left) to switch models
- Click STOP button or press Q to exit

---

### `main.py`

Evaluates a trained model on the RAF-DB dataset.

**What it does:**
- Loads all test images from `DATASET/test`
- Runs predictions using the fine-tuned model
- Prints overall accuracy

**How to use:**
```bash
python main.py
```

## Backend Files

### `tuned_model_inference.py`

Contains the `EmotionModelTuned` class for running fine_tuned models, utilized by main.py and webcam_emotion.py

**What it does:**
- Loads a fine-tuned model from a local folder
- Provides `predict(image)` returning (label_string, confidence)
- Provides `evaluate(images, labels)` to compute accuracy

---

### `preprocessing.py`

Utility functions for loading the dataset, utilized by train.py and tuned_model_inference.py

**What it does:**
- `load_split(split_root)` - loads images/labels from a train or test folder
- `load_train_test(dataset_root)` - loads both train and test splits

---

## Dataset Structure

Images are organized by emotion folder:

```
DATASET/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

---

## Trained Models

Our trained models are in the `models/` folder:
- `models/ConvNext/`
- `models/DeIT-Small/`
- `models/DeIT-Small-Undersampled/`
- `models/ViT-Base/`
- `models/ViT-Base-Undersampled/`

These correspond to the folders in the google drive:
- `ConvNext`
- `DeIT-Small`
- `DeIT-Small-Undersampled`
- `ViT-Base`
- `ViT-Base-Undersampled`

---

## Hardware

Automatically uses:
- **MPS** on Apple Silicon Macs
- **CUDA** on NVIDIA GPUs
- **CPU** as fallback
