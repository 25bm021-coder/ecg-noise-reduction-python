
import wfdb
import numpy as np
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import joblib

# 1. Load ECG recording and expert annotations
record = wfdb.rdrecord("data/100")
annotations = wfdb.rdann("data/100", "atr")

ecg = record.p_signal[:, 0]
fs = record.fs

print("ECG loaded:", len(ecg), "samples")
print("Sampling frequency:", fs, "Hz")

# 2. Define broad experimental labels
normal_symbols = {"N", "L", "R", "e", "j"}
abnormal_symbols = {"A", "a", "J", "S", "V", "E", "F", "f"}

# Extract a heartbeat window around each annotation
before = int(0.20 * fs)
after = int(0.40 * fs)

X = []
y = []

for sample, symbol in zip(
    annotations.sample, annotations.symbol
):
    if symbol in normal_symbols:
        label = 0
    elif symbol in abnormal_symbols:
        label = 1
    else:
        continue

    start = sample - before
    end = sample + after

    # Skip beats too close to recording boundaries
    if start < 0 or end > len(ecg):
        continue

    beat = ecg[start:end]

    if len(beat) != before + after:
        continue

    # Features: heartbeat waveform samples
    X.append(beat)
    y.append(label)

X = np.asarray(X)
y = np.asarray(y)

print("\nHeartbeat dataset shape:", X.shape)
print("Class counts:", Counter(y))
print("0 = Normal, 1 = Potentially abnormal")

# 3. Verify that both classes have enough samples
if len(np.unique(y)) < 2:
    raise ValueError(
        "Only one class was found. More labelled ECG "
        "recordings are needed to train this classifier."
    )

counts = Counter(y)
if min(counts.values()) < 2:
    raise ValueError(
        "Too few examples in one class. Download and "
        "combine additional labelled ECG recordings."
    )

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5. Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# 6. Evaluate on the held-out beats
predictions = model.predict(X_test)

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Potentially abnormal"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    y_test, predictions, labels=[0, 1]
))

# 7. Save the trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/ecg_rf_model.joblib")

print("\nModel saved to models/ecg_rf_model.joblib")
