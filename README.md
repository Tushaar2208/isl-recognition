# Real-Time Indian Sign Language Recognition

A webcam-based system that recognizes Indian Sign Language (ISL) signs in real time and converts them into readable sentences with optional text-to-speech output.

**B.Tech CSE Final Year Project — AKTU, India**

---

## System Pipeline

```
Webcam → Hand Detection → Landmark Extraction → Normalization
   → Sign Classification → Sentence Construction → Text Output → TTS
```

## How It Works

1. **Capture** — Webcam reads live video frames
2. **Detect** — MediaPipe locates hands and extracts 21 skeletal joint coordinates per hand
3. **Normalize** — Landmarks are centered on the wrist and scaled to be distance/position invariant
4. **Classify** — *(upcoming)* Trained models recognize which ISL sign is being performed
5. **Communicate** — *(upcoming)* Recognized signs are assembled into sentences and spoken aloud

## Key Design Decisions

- **Landmark-based recognition** — Uses compact 3D joint coordinates instead of raw video frames, enabling real-time performance on a standard laptop CPU without GPU
- **Isolated sign recognition** — Each sign is bounded by a brief pause; continuous sign-stream recognition is out of scope
- **INCLUDE-50 vocabulary** — Standard 50-word benchmark subset of the AI4Bharat INCLUDE dataset
- **Signer-independent evaluation** — Train and test sets use disjoint sets of signers
- **Rule-based sentence construction** — Templated grammar, not a trained translation model

## Project Status

| Module | Description | Status |
|---|---|---|
| 1. Sign Capture & Preprocessing | Webcam capture, hand detection, landmark normalization | ✅ Complete |
| 2. Dataset Preparation | INCLUDE-50 benchmark (958 videos, 50 classes) & batch landmark extraction | ✅ Complete |
| 3. Feature Engineering | Kinematic feature extraction (angles, distances, ratios, dynamics) | ✅ Complete |
| 4. Supervised Classifier | Multi-class classification (Random Forest @ 80.2% accuracy, SVM @ 75.0%) | ✅ Complete |
| 5. Dynamic Sign Classifier | Sequence modeling with LSTM / GRU for temporal signs | 🔲 In Progress |
| 6. Real-Time Recognition | Live webcam inference with rolling buffer | 🔲 Upcoming |
| 7. Sentence Construction & TTS | Rule-based sentence assembly and voice output | 🔲 Upcoming |
| 8. Integration & Deployment | Lightweight interactive Web/Desktop UI | 🔲 Upcoming |


## Module 1 — Sign Capture & Preprocessing

The completed module captures live webcam video, detects hands using MediaPipe, extracts 21 landmark points per hand, and normalizes them to be scale and position invariant.

### Demo Screenshots

| One Hand Detected | Two Hands Detected |
|---|---|
| ![One hand](docs/screenshots/snapshot_1788868360.png) | ![Two hands](docs/screenshots/snapshot_1788868774.png) |

### Normalization

Each hand's 21 landmarks are:
1. **Translated** so the wrist (landmark 0) becomes the origin `[0, 0, 0]`
2. **Scaled** by the wrist-to-middle-finger-MCP distance so landmark 9 has unit length

This makes the output invariant to hand size and camera distance.

### Output Format

Normalized landmarks are logged to `landmark_log.jsonl` (one JSON object per line):

```json
{
  "frame": 15,
  "timestamp": 1788866759.125,
  "num_hands": 1,
  "hands": [
    {
      "handedness": "Right",
      "confidence": 0.944,
      "landmarks": [[0.0, 0.0, 0.0], [0.587, -0.244, -0.135], "...21 total..."]
    }
  ]
}
```

> Full technical report with math, verification results, and complete source code: [docs/module1_report.md](docs/module1_report.md)

## Module 2 — Dataset Preparation & Landmark Extraction

Module 2 downloads the official **AI4Bharat INCLUDE-50** dataset (a 50-word standard benchmark subset of Indian Sign Language) and extracts normalized 3D hand landmarks across all 958 videos.

- **Dataset Size:** 958 videos, 50 ISL classes across 15 parent categories.
- **Hand Slotting:** Dual-hand representation `(T, 2, 21, 3)` where Slot 0 = Right Hand, Slot 1 = Left Hand.
- **Normalization:** Identical to Module 1 (wrist-centered, MCP9-scaled) ensuring training data and live inference occupy the exact same coordinate space.
- **Output:** 957 `.npy` sequence tensors and companion `_meta.json` files storing FPS, dimensions, and detection ratios.

```bash
# Extract landmarks from dataset videos
python module2_extract_landmarks.py
```

---

## Module 3 — Feature Engineering

Raw $(X, Y, Z)$ coordinates vary by signer body size and distance from the camera. Module 3 computes **scale-invariant and position-invariant geometric features**:

1. **15 Joint Angles:** Finger bend angles $[0, \pi]$ using 3D dot products across adjacent bones.
2. **10 Fingertip Distances:** Pairwise Euclidean distances between all 5 fingertips.
3. **5 Palm Distances:** Distance from each fingertip to the wrist origin.
4. **5 Extension Ratios:** Fingertip-to-wrist distance relative to MCP-to-wrist distance (distinguishes curled vs open fingers).
5. **2 Hand Orientation Angles:** Elevation and azimuth angles of the normal vector to the palm plane.
6. **Temporal Deltas:** Frame-to-frame velocity vectors capturing sign movement dynamics.

- **Per-Frame Feature Vector:** 120 compact features per frame.
- **Per-Video Temporal Stats:** 480-dimensional summary vector (mean, std, min, max across time).

```bash
# Generate feature tensors
python module3_feature_engineering.py
```

---

## Module 4 — Classifier Training & Benchmark Results

Module 4 trains supervised machine learning models on the 480-dimensional feature summaries across all 50 sign classes using an 80/20 stratified train/test split (765 training samples, 192 test samples).

### Benchmark Performance

| Classifier | Test Accuracy (50 Classes) | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Random Forest (300 Trees)** | **80.21%** | **0.82** | **0.80** | **0.80** |
| **SVM (RBF Kernel, C=10)** | **75.00%** | **0.77** | **0.75** | **0.75** |

### Confusion Matrix (Random Forest — 80.2% Accuracy)

![Random Forest Confusion Matrix](data/models/confusion_matrix_random_forest.png)

> The prominent diagonal confirms robust discriminative power across all 50 vocabulary words running entirely on laptop CPU.

```bash
# Train classifiers and output evaluation plots
python module4_train_classifiers.py
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download the hand landmarker model (one-time, ~7.5 MB)
curl -L -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Run Module 1
python module1_sign_capture.py
```

### Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | 5-second countdown, then save a snapshot with skeleton overlay |

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Computer Vision | OpenCV |
| Hand Detection | MediaPipe (Tasks API) |
| Numerical Computing | NumPy |

> No GPU required. Runs in real time on a standard laptop CPU.
