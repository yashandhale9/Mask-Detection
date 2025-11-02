####################################################################
# File: face_mask_detection_realtime.py
# Project: Real-Time Face Mask Detection
# Author: Yash Gorakshnath Andhale
# Date: 01/11/2025
# Description:
#   Detects human faces via webcam and classifies them as:
#     → With Mask (Green Box)
#     → Improper Mask (Yellow Box)
#     → Without Mask (Red Box)
#
#   Features:
#     - Real-time classification using MobileNetV2-based model
#     - 3-second buffered beep alert for improper/no mask
#     - Lightweight inference and smooth performance
####################################################################

import cv2
import numpy as np
import tensorflow as tf
import time
import winsound  # Simple built-in beep alerts (Windows)

####################################################################
# Section : Load the Pre-Trained Model
# Description :
#   Loads the trained CNN model (.h5) for inference.
#   Optionally supports TensorFlow Lite model for low-resource devices.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

model = tf.keras.models.load_model("artifacts/mask_detector_model.h5")

# If you wish to use TensorFlow Lite model instead of .h5, uncomment below:
"""
interpreter = tf.lite.Interpreter(model_path="artifacts/mask_detector_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
is_tflite = True
"""
is_tflite = False  # Keep False to use .h5 model by default

####################################################################
# Section : Define Class Labels and Box Colors
# Description :
#   Maps each class (mask status) to a specific color for visualization.
#   - Green  → With Mask
#   - Yellow → Improper Mask
#   - Red    → Without Mask
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

labels = ["improper_mask", "with_mask", "without_mask"]

color_map = {
    "with_mask": (0, 255, 0),          # Green
    "improper_mask": (0, 255, 255),    # Yellow
    "without_mask": (0, 0, 255)        # Red
}

####################################################################
# Section : Load Haar Cascade for Face Detection
# Description :
#   Utilizes OpenCV's Haar Cascade classifier for real-time
#   human face detection using webcam frames.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

####################################################################
# Section : Initialize Webcam
# Description :
#   Opens system webcam with 640x480 resolution for smooth FPS.
#   Terminates if webcam cannot be accessed.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
print("\nWebcam started. Press 'Q' to quit.\n")

####################################################################
# Section : Beep Alert Configuration
# Description :
#   Uses a time buffer (3 seconds) between beeps to avoid
#   constant alert noise when multiple detections occur.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

last_alert_time = 0
alert_delay = 3  # seconds buffer between alerts

####################################################################
# Section : Main Real-Time Detection Loop
# Description :
#   Processes each frame to detect faces and classify mask status.
#   Draws bounding boxes and labels with color codes.
#   Plays beep alert for "No Mask" and "Improper Mask" only.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for faster face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in current frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue  # Skip invalid frames

        # Preprocess cropped face image for model input
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Predict mask status
        if not is_tflite:
            preds = model.predict(img_array, verbose=0)
        else:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])

        # Extract prediction result
        pred_index = np.argmax(preds)
        label = labels[pred_index]
        color = color_map[label]

        # Draw detection bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, label.replace("_", " ").title(),
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)

        ####################################################################
        # Sub-Section : Beep Alert (No Mask / Improper Mask Only)
        # Description :
        #   Plays short beep sound every 3 seconds if user is detected
        #   without a mask or wearing it improperly.
        #   No alert for correct mask usage.
        # Author : Yash Gorakshnath Andhale
        # Date : 01/11/2025
        ####################################################################

        current_time = time.time()
        if label in ["without_mask", "improper_mask"]:
            if current_time - last_alert_time > alert_delay:
                winsound.Beep(1200, 400)  # frequency=1200Hz, duration=400ms
                last_alert_time = current_time

    # Display annotated video frame
    cv2.imshow("Real-Time Face Mask Detection (Press Q to Quit)", frame)

    # Exit condition: Press 'Q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

####################################################################
# Section : Cleanup
# Description :
#   Releases camera resources and closes all display windows
#   after user exits detection loop.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

cap.release()
cv2.destroyAllWindows()
print("\nDetection ended. Webcam closed successfully.")
