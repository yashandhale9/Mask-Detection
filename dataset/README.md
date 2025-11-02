# 🗂️ Dataset Folder — Mask Detection Project

This folder contains the image dataset used to train and validate the **Real-Time Face Mask Detection** model.

**Note:** Dataset images are not included in this repository due to size and privacy restrictions. Please add your dataset manually inside the /dataset folder before training.

---

## 📦 Structure
The dataset is divided into three main subfolders representing the target classes:

| Folder Name | Description | Example Image Type |
|--------------|--------------|--------------------|
| **with_mask/** | People correctly wearing face masks. | ✅ Properly covered nose & mouth |
| **without_mask/** | People not wearing any mask. | ❌ No mask on face |
| **improper_mask/** | People wearing masks incorrectly. | ⚠️ Mask below nose / chin |

Each subfolder contains multiple images of different individuals with varying lighting, pose, and background to ensure robust training.

---

## 📸 Dataset Details
- **Total Classes:** 3  
- **Image Format:** `.jpg` / `.png`  
- **Recommended Image Size:** 224 × 224 pixels  
- **Split Ratio:**  
  - 80% → Training  
  - 20% → Validation (automatically handled in `face_mask_train.py`)  

---

## 🧠 Usage
The dataset is automatically loaded and preprocessed in the training script using:

ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

Make sure this folder is placed inside your project root directory before running:
```
python face_mask_train.py
```
---

### ⚙️ **Notes**

Ensure that all three class folders (with_mask, without_mask, improper_mask) exist before training.

You can add more images to improve accuracy.

Avoid blurry or duplicate images to maintain dataset quality.

If you use a custom dataset, keep the same folder names for compatibility.

---

### **Prepared by:**
👨‍💻 Yash Gorakshnath Andhale
    01/11/2025

---
