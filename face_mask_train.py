####################################################################
# File: face_mask_train.py
# Author: Yash Gorakshnath Andhale
# Date: 01/11/2025
# Description:
#   Trains a real-time face mask detection model using MobileNetV2.
#   Includes data augmentation, transfer learning, fine-tuning,
#   evaluation (confusion matrix, curves), and model export (.h5/.tflite)
####################################################################

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

####################################################################
# Section : Setup Configuration
# Description :
#   Defines dataset path, image size, training parameters,
#   and ensures artifact directory for saving outputs.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

dataset_dir = "dataset"
img_size = 224
batch_size = 32
epochs = 10
os.makedirs("artifacts", exist_ok=True)

####################################################################
# Section : Data Preparation
# Description :
#   Uses ImageDataGenerator for preprocessing, augmentation,
#   and splitting dataset into training and validation sets.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset="training"
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset="validation",
    shuffle=False
)

print("\nDetected Classes:", train_gen.class_indices)

####################################################################
# Section : Data Augmentation Visualization
# Description :
#   Displays and saves a sample grid of augmented training images.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

x, y = next(train_gen)
plt.figure(figsize=(10,5))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow((x[i] * 127.5 + 127.5).astype("uint8"))
    plt.axis("off")
plt.suptitle("Augmented Training Samples")
plt.savefig("artifacts/augmentation_samples.png")
plt.close()

####################################################################
# Section : Model Creation (MobileNetV2 Transfer Learning)
# Description :
#   Loads MobileNetV2 base (ImageNet weights),
#   adds custom dense layers, compiles for training.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base layers for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

####################################################################
# Section : Callback Configuration
# Description :
#   Uses EarlyStopping to prevent overfitting and
#   ModelCheckpoint to save the best validation accuracy model.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint("artifacts/best_model.h5", save_best_only=True, monitor='val_accuracy')
]

####################################################################
# Section : Training Phase 1 (Feature Extraction)
# Description :
#   Trains only top layers while keeping base model frozen.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks
)

####################################################################
# Section : Training Phase 2 (Fine-Tuning)
# Description :
#   Unfreezes last 30 layers of base model and retrains with low LR
#   to slightly improve performance.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3
)

####################################################################
# Section : Evaluation and Visualization
# Description :
#   Generates accuracy/loss curves and confusion matrix heatmap.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

# --- Accuracy and Loss Curves ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'] + fine_tune.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'] + fine_tune.history['val_accuracy'], label='Val')
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'] + fine_tune.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + fine_tune.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/training_curves.png")
plt.close()

# --- Confusion Matrix ---
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(val_gen.classes, y_pred)
target_names = list(train_gen.class_indices.keys())

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("artifacts/confusion_matrix.png")
plt.close()

print("\nClassification Report:\n")
print(classification_report(val_gen.classes, y_pred, target_names=target_names))

####################################################################
# Section : Model Saving
# Description :
#   Saves the final trained model (.h5) and exports a lightweight
#   TensorFlow Lite version (.tflite) for real-time deployment.
# Author : Yash Gorakshnath Andhale
# Date : 01/11/2025
####################################################################

# Save full model
model.save("artifacts/mask_detector_model.h5")
print("Saved: artifacts/mask_detector_model.h5")

# Save lightweight model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("artifacts/mask_detector_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved TFLite model (for lightweight inference).")

# Display final validation accuracy
print("\nFinal Validation Accuracy:", round(fine_tune.history['val_accuracy'][-1]*100, 2), "%")
print("All artifacts saved in /artifacts folder.")
