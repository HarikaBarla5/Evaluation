"""
Voice confidence/pace inference — uses RAVDESS-trained LSTM
Matches exactly the feature extraction used in train_voice_lstm.py:
  28 features/frame (13 MFCC + 13 delta-MFCC + f0 + RMS), MAX_FRAMES=130
"""

import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = "backend/models/voice_lstm.h5"
MEAN_PATH  = "backend/models/voice_feature_mean.npy"
STD_PATH   = "backend/models/voice_feature_std.npy"

SAMPLE_RATE = 22050
DURATION    = 3
MAX_FRAMES  = 130
N_MFCC      = 13

_model = _mean = _std = None


def _load():
    global _model, _mean, _std
    if _model is None:
        _model = load_model(MODEL_PATH)
        _mean  = np.load(MEAN_PATH)
        _std   = np.load(STD_PATH)


def _extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    target_len = SAMPLE_RATE * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)

    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                fmax=librosa.note_to_hz("C7"), sr=sr)
    f0  = np.nan_to_num(f0)
    rms = librosa.feature.rms(y=y)[0]

    n_frames = mfcc.shape[1]
    f0  = np.resize(f0,  n_frames)
    rms = np.resize(rms, n_frames)

    features = np.vstack([mfcc, mfcc_delta, f0, rms]).T  # (T, 28)

    if features.shape[0] < MAX_FRAMES:
        features = np.pad(features, ((0, MAX_FRAMES - features.shape[0]), (0, 0)))
    else:
        features = features[:MAX_FRAMES]

    return features, y, sr


def _estimate_pace(y, sr):
    """
    Syllable-rate proxy via onset detection -> pace score 0-100.
    ~4 syllables/sec = 100 (normal conversational pace).
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets     = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration_s = len(y) / sr
    rate       = len(onsets) / duration_s if duration_s > 0 else 0
    pace_score = min(100.0, (rate / 4.0) * 100)
    return round(pace_score, 2), round(rate, 2)


def predict_voice_score(file_path):
    """
    Returns:
        {
            "confidence_score": float (0-100),
            "pace_score":       float (0-100),
            "syllable_rate":    float (syllables/sec)
        }
    """
    _load()
    features, y, sr = _extract_features(file_path)

    norm = (features - _mean[0]) / _std[0]
    pred = _model.predict(norm[np.newaxis], verbose=0)[0][0]
    confidence_score = float(pred) * 100

    pace_score, syllable_rate = _estimate_pace(y, sr)

    return {
        "confidence_score": round(confidence_score, 2),
        "pace_score":       pace_score,
        "syllable_rate":    syllable_rate,
    }