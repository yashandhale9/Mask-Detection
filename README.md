# 😷 Real-Time Face Mask Detection using CNN & MobileNetV2

---

## 📘 Project Overview

The **Real-Time Face Mask Detection System** is a deep learning–based application designed to identify whether a person is wearing a mask correctly, wearing it improperly, or not wearing one at all.

This project was developed considering the need for **public safety and infection prevention**, especially during pandemics such as COVID-19, where mask-wearing became crucial in minimizing the spread of airborne diseases.

The system leverages a **Convolutional Neural Network (CNN)** architecture using **MobileNetV2** — a lightweight and efficient deep learning model suitable for real-time applications. The network is fine-tuned on a custom dataset containing three categories:
1. **With Mask**
2. **Without Mask**
3. **Improper Mask**

The model achieves high accuracy while maintaining low computational requirements, making it suitable for deployment on **CPU-only laptops** and **edge devices** (like Raspberry Pi or Android).

The system performs **live face detection** using OpenCV’s Haar Cascade classifier, and each detected face is analyzed by the trained model to determine mask status. Based on the prediction:
- A 🟩 **green box** appears for “With Mask,”  
- A 🟨 **yellow box** for “Improper Mask,”  
- A 🟥 **red box** for “Without Mask.”  

To enhance user awareness, an **audio beep alert** is triggered for the last two cases, helping ensure compliance in real-time environments such as public spaces, offices, or educational institutions.

The overall pipeline includes:
- Data preprocessing and augmentation  
- Transfer learning using MobileNetV2  
- Fine-tuning of the top layers  
- Model evaluation and visualization  
- Real-time inference with visual and audio feedback  

This project demonstrates the application of **AI and Computer Vision** in enhancing safety measures and serves as an excellent example of how lightweight deep learning models can perform effectively in real-world scenarios.

---

## 🎯 Objectives
- Develop an accurate and lightweight mask detection model  
- Detect mask status in real-time via webcam  
- Reduce model size using **TensorFlow Lite** for deployment on low-end devices  
- Provide visual (colored boxes) and audio alerts to users  

---

## 🧠 Technologies Used
| Category | Tools/Frameworks |
|-----------|------------------|
| Programming Language | Python 3 |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Visualization | Matplotlib, Seaborn |
| Model | MobileNetV2 (Transfer Learning) |
| Alerts | winsound (beep) |
| Dataset | Custom 3-class dataset: `with_mask`, `without_mask`, `improper_mask` |

---

## 📂 Folder Structure

```
MaskDetection/
│
├── artifacts/
│ ├── best_model.h5
│ ├── mask_detector_model.h5
│ ├── mask_detector_model.tflite
│ ├── confusion_matrix.png
│ ├── training_curves.png
│ ├── augmentation_samples.png
│
├── dataset/
│ ├── with_mask/
│ ├── without_mask/
│ └── improper_mask/
│
├── face_mask_train.py
├── face_mask_detection_realtime.py
└── README.md

```
---


---

## 🧰 Installation & Setup

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
```

### 3️⃣ (Optional) Update pip if required
```
python -m pip install --upgrade pip
```

---

### 🧮 **Model Training**

To train and evaluate the model:
```
python face_mask_train.py
```

This will:

Train MobileNetV2 with data augmentation

Generate accuracy/loss plots

Save confusion matrix

Export both .h5 and .tflite models inside /artifacts

---

### 🎥 **Real-Time Detection**

To run the webcam-based detection:
```
python face_mask_detection_realtime.py
```

Press **Q** to exit webcam feed.

**Color Legend:**

🟩 Green → Mask Detected

🟨 Yellow → Improper Mask

🟥 Red → No Mask

🔊 **Beep alerts:**

Beep every 3 seconds for Improper or No Mask

---

### 📊 **Results**

| Metric              | Value                       |
| ------------------- | --------------------------- |
| Validation Accuracy | ~98%                        |
| Loss                | Very Low                    |
| Model Size (.h5)    | ~14 MB                      |
| TFLite Size         | ~4 MB                       |
| FPS                 | 15–20 (on Intel i5, no GPU) |

Visual Results:

✅ artifacts/training_curves.png – Accuracy & loss curves

✅ artifacts/confusion_matrix.png – Class performance

✅ artifacts/augmentation_samples.png – Augmented samples

---

### ⚙️ **Improvements & Future Scope**

Integrate voice-based alerts (Hindi + English)

Deploy model on mobile using TensorFlow Lite / Android Studio

Add face tracking to improve multi-person detection

Optimize for GPU / Jetson Nano for higher FPS

---

### 🧾 **References**

TensorFlow Documentation: https://www.tensorflow.org

MobileNetV2 Paper: https://arxiv.org/abs/1801.04381

OpenCV Haar Cascades: https://opencv.org

---

### 🏁 **Conclusion**

This project demonstrates how deep learning and computer vision can be effectively used to ensure public safety by detecting face mask usage in real-time.
The model achieves high accuracy, low latency, and can be easily deployed on any system.

---

### **Developed by:**
👨‍💻 Yash Gorakshnath Andhale

    01/11/2025

---
