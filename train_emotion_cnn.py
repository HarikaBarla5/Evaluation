"""
Train CNN for Facial Emotion Recognition on FER2013 dataset
Output: models/emotion_cnn.h5

Dataset: fer2013.csv with columns -> emotion, pixels, Usage
emotion labels: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
CSV_PATH = "datasets/fer2013.csv"
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 30
MODEL_OUT = "backend/models/emotion_cnn.keras"

EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

# ---------------- LOAD DATA ----------------
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

def parse_pixels(pixel_str):
    return np.array(pixel_str.split(), dtype="float32").reshape(IMG_SIZE, IMG_SIZE, 1)

X = np.stack(df["pixels"].apply(parse_pixels).values)
X = X / 255.0  # normalize
y = to_categorical(df["emotion"].values, NUM_CLASSES)

# Split using the Usage column if present, else manual split
if "Usage" in df.columns:
    train_mask = df["Usage"] == "Training"
    val_mask = df["Usage"] != "Training"
    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_val, y_val = X[val_mask.values], y[val_mask.values]
else:
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=df["emotion"]
    )

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ---------------- DATA AUGMENTATION ----------------
train_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
train_gen.fit(X_train)

# ---------------- MODEL ----------------
model = Sequential([
    Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

# ---------------- TRAIN ----------------
history = model.fit(
    train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------- SAVE FINAL ----------------
model.save("cnn_model")

print(f"Model saved to {MODEL_OUT}")

# ---------------- EVALUATE ----------------
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Final Validation Accuracy: {val_acc:.4f}")
