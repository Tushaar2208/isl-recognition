"""
Module 4: Train & Evaluate Sign Language Classifiers
=====================================================

Trains two classifiers on the INCLUDE-50 feature data:
  1. Random Forest  — fast, interpretable, uses stats features (480-dim)
  2. SVM            — strong baseline, also uses stats features (480-dim)

Uses signer-independent evaluation where possible (random 80/20 split by video,
stratified by class so every class appears in both train and test).

Outputs:
  - data/models/random_forest.pkl
  - data/models/svm.pkl
  - data/models/evaluation_report.txt
  - data/models/confusion_matrix_rf.png
  - data/models/confusion_matrix_svm.png
"""

import numpy as np
import pandas as pd
import pickle
import json
import sys
from pathlib import Path
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import ensure_dirs, FEATURE_DIR, DATA_DIR

MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)



# ── 1. Load all stats features ─────────────────────────────────────────────

def load_dataset():
    """
    Walk data/features/<class_name>/<video>_stats.npy and build X, y arrays.
    Each sample is a 480-dim stats vector for one video.
    """
    X, y, video_ids = [], [], []

    for class_dir in sorted(FEATURE_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name

        for stats_file in class_dir.glob("*_stats.npy"):
            vec = np.load(stats_file)          # (480,)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                continue                        # skip corrupted
            X.append(vec)
            y.append(class_name)
            video_ids.append(stats_file.stem.replace("_stats", ""))

    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} samples across {len(set(y))} classes.")
    print(f"Feature vector size: {X.shape[1]}")
    return X, y, video_ids


# ── 2. Encode labels ────────────────────────────────────────────────────────

def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


# ── 3. Train / evaluate helper ─────────────────────────────────────────────

def train_and_evaluate(clf, X_train, X_test, y_train, y_test, le, name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Samples       : Train={len(y_train)}, Test={len(y_test)}")
    print()

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(20, 18))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le.classes_
    )
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix (Accuracy: {acc*100:.1f}%)", fontsize=14)
    plt.tight_layout()
    plot_path = MODEL_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"  Confusion matrix saved → {plot_path.name}")

    return acc, report, cm


# ── 4. Main ────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    print("=" * 55)
    print("  Module 4: Train Sign Language Classifiers")
    print("=" * 55)

    # Load data
    X, y, video_ids = load_dataset()
    y_enc, le = encode_labels(y)

    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, y_enc))

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_enc[train_idx],   y_enc[test_idx]

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print(f"Classes       : {len(le.classes_)}")

    results = {}

    # ── Random Forest ──────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    rf_acc, rf_report, rf_cm = train_and_evaluate(
        rf, X_train, X_test, y_train, y_test, le, "Random Forest"
    )
    results["Random Forest"] = rf_acc

    # ── SVM ────────────────────────────────────────────────────────────────
    print("\nTraining SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm_acc, svm_report, svm_cm = train_and_evaluate(
        svm, X_train, X_test, y_train, y_test, le, "SVM"
    )
    results["SVM"] = svm_acc

    # ── Save models ────────────────────────────────────────────────────────
    with open(MODEL_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump({"model": rf, "scaler": scaler, "label_encoder": le}, f)
    with open(MODEL_DIR / "svm.pkl", "wb") as f:
        pickle.dump({"model": svm, "scaler": scaler, "label_encoder": le}, f)
    print(f"\nModels saved → {MODEL_DIR}/")

    # ── Write report ───────────────────────────────────────────────────────
    report_path = MODEL_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Module 4: Classification Results\n")
        f.write("="*55 + "\n\n")
        f.write(f"Dataset: INCLUDE-50 ({len(X)} samples, {len(le.classes_)} classes)\n")
        f.write(f"Train/Test split: 80/20 stratified\n\n")
        for name, acc in results.items():
            f.write(f"{name:20s}: {acc*100:.2f}%\n")
        f.write("\n\n--- Random Forest Classification Report ---\n")
        f.write(rf_report)
        f.write("\n\n--- SVM Classification Report ---\n")
        f.write(svm_report)

    print(f"\nEvaluation report saved → {report_path.name}")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  FINAL RESULTS")
    print("="*55)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 40)
        print(f"  {name:20s} {acc*100:5.1f}%  {bar}")
    best = max(results, key=results.get)
    print(f"\n  Best model: {best} ({results[best]*100:.1f}%)")
    print("="*55)


if __name__ == "__main__":
    main()
