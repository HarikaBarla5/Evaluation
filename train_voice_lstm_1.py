"""
Train LSTM for Voice Confidence Scoring on RAVDESS Dataset
OPTIMIZED VERSION — replaces slow librosa.pyin() with fast yin()
~10-20x faster feature extraction than original script.

Dataset: RAVDESS — 24 actors, 2880 .wav files
Zip: archive__4_.zip -> Actor_01/ ... Actor_24/

RAVDESS filename format:
  03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav
  Position 3 = emotion: 01=neutral,02=calm,03=happy,04=sad,05=angry,06=fearful,07=disgust,08=surprised
  Position 4 = intensity: 01=normal, 02=strong

Interview confidence score mapping (0-100):
  calm     -> 85
  neutral  -> 75
  happy    -> 80
  surprised-> 60
  sad      -> 35
  angry    -> 30
  fearful  -> 25
  disgust  -> 20
  strong intensity adds +5

Output: backend/models/voice_lstm.h5
        backend/models/voice_feature_mean.npy
        backend/models/voice_feature_std.npy
"""

import os, zipfile, time
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ================= CONFIG =================
ZIP_PATH    = "datasets/archive__4_.zip"
EXTRACT_DIR = "datasets/voice_ravdess"
MODEL_OUT   = "backend/models/voice_lstm.h5"
MEAN_OUT    = "backend/models/voice_feature_mean.npy"
STD_OUT     = "backend/models/voice_feature_std.npy"

SAMPLE_RATE = 22050
DURATION    = 3
MAX_FRAMES  = 130
N_MFCC      = 13
EPOCHS      = 60
BATCH_SIZE  = 32

EMOTION_SCORE_MAP = {
    "01": 75, "02": 85, "03": 80, "04": 35,
    "05": 30, "06": 25, "07": 20, "08": 60,
}
INTENSITY_BONUS = {"01": 0, "02": 5}

# ================= STEP 1: EXTRACT =================
if not os.path.exists(EXTRACT_DIR):
    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Done.")
else:
    print(f"Already extracted at {EXTRACT_DIR}")


# ================= STEP 2: FILENAME -> SCORE =================
def parse_ravdess_score(filename):
    name  = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("-")
    if len(parts) != 7:
        return None
    emotion   = parts[2]
    intensity = parts[3]
    if emotion not in EMOTION_SCORE_MAP:
        return None
    return float(min(EMOTION_SCORE_MAP[emotion] + INTENSITY_BONUS.get(intensity, 0), 100))


# ================= STEP 3: FAST FEATURE EXTRACTION =================
def extract_features(file_path):
    """
    FAST version — uses librosa.yin() instead of pyin().
    yin() is ~10-15x faster; accuracy is sufficient for regression scoring.
    Features: 13 MFCC + 13 delta-MFCC + 1 pitch + 1 RMS = 28 per frame
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    target_len = SAMPLE_RATE * DURATION
    y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]

    # MFCCs + delta
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)   # (13, T)
    mfcc_delta = librosa.feature.delta(mfcc)                         # (13, T)

    # FAST pitch via yin (not pyin)
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
    )                                                                 # (T_yin,)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]                                # (T_rms,)

    n_frames = mfcc.shape[1]
    f0  = np.resize(f0,  n_frames)
    rms = np.resize(rms, n_frames)

    features = np.vstack([mfcc, mfcc_delta, f0, rms]).T              # (T, 28)

    # pad / truncate to MAX_FRAMES
    if features.shape[0] < MAX_FRAMES:
        features = np.pad(features, ((0, MAX_FRAMES - features.shape[0]), (0, 0)))
    else:
        features = features[:MAX_FRAMES]

    return features


# ================= STEP 4: LOAD DATASET =================
print("Extracting features (fast mode — using yin instead of pyin)...")
t0 = time.time()

X_list, y_list = [], []
skipped = 0
total   = 0

wav_files = []
for root, _, files in os.walk(EXTRACT_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_files.append(os.path.join(root, f))

print(f"Found {len(wav_files)} .wav files")

for i, fpath in enumerate(wav_files):
    score = parse_ravdess_score(fpath)
    if score is None:
        skipped += 1
        continue
    try:
        feats = extract_features(fpath)
        X_list.append(feats)
        y_list.append(score)
        total += 1
        # progress every 100 files
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(wav_files) - i - 1) / rate
            print(f"  [{i+1}/{len(wav_files)}] loaded={total} | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s")
    except Exception as e:
        print(f"  Skipping {os.path.basename(fpath)}: {e}")
        skipped += 1

elapsed_total = time.time() - t0
print(f"\nDone: {total} samples loaded, {skipped} skipped | {elapsed_total:.1f}s total")

X = np.array(X_list, dtype="float32")          # (N, 130, 28)
y = np.array(y_list, dtype="float32") / 100.0  # normalize to 0-1
print(f"X shape: {X.shape} | y range: [{y.min():.2f}, {y.max():.2f}]")

# ================= STEP 5: NORMALISE =================
mean = X.mean(axis=(0, 1), keepdims=True)
std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
X    = (X - mean) / std

np.save(MEAN_OUT, mean)
np.save(STD_OUT,  std)
print(f"Saved normalisation stats -> {MEAN_OUT}, {STD_OUT}")

# ================= STEP 6: SPLIT =================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

# ================= STEP 7: MODEL =================
model = Sequential([
    Masking(mask_value=0.0, input_shape=(MAX_FRAMES, 28)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1,  activation="sigmoid"),
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ================= STEP 8: TRAIN =================
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# ================= STEP 9: SAVE + EVAL =================
model.save(MODEL_OUT)
_, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"\nModel saved -> {MODEL_OUT}")
print(f"Val MAE: {val_mae:.4f} (~{val_mae * 100:.1f} pts on 0-100 scale)")