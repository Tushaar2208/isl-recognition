"""
Module 1 — Real-Time Sign Capture and Preprocessing
Real-Time Isolated Indian Sign Language Recognition Project

Pipeline:
    Webcam --> Frame Capture --> MediaPipe Hand Detection -->
    Landmark Extraction --> Normalization --> Output (live overlay + logged JSON)

Run:
    python module1_sign_capture.py

Controls:
    q       - quit
    s       - save a snapshot of the current frame (for documentation/screenshots)

Output:
    landmark_log.jsonl - one JSON line per processed frame with normalized landmarks
    snapshot_*.png      - saved frames (only when 's' is pressed)

Note:
    This version uses the MediaPipe Tasks API (mediapipe >= 1.0.0) because the
    legacy mp.solutions API was removed in newer releases.  It requires the
    hand_landmarker.task model file in the same directory as the script.

    Download the model once:
        curl -L -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ── Configuration ───────────────────────────────────────────────────────────
CAMERA_INDEX = 0
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5
LOG_EVERY_N_FRAMES = 5
SCRIPT_DIR = Path(__file__).parent.resolve()
LOG_PATH = SCRIPT_DIR / "landmark_log.jsonl"
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"

# ── Hand-connection list (for drawing the skeleton overlay) ─────────────────
# Same connections that mp.solutions.hands.HAND_CONNECTIONS used to provide.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle   (5,9) added below
    (0, 13), (13, 14), (14, 15), (15, 16), # ring     (9,13) added below
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky    (13,17) added below
    (5, 9), (9, 13), (13, 17),             # palm cross-connections
]

# Landmark and connection drawing colours / sizes
_LM_COLOR = (0, 0, 255)     # red dots
_LM_RADIUS = 4
_CONN_COLOR = (0, 255, 0)   # green lines
_CONN_THICKNESS = 2


# ── Normalization (identical to original) ───────────────────────────────────

def normalize_landmarks(landmarks) -> list[list[float]]:
    """
    Convert 21 raw (x, y, z) landmarks into a representation that is invariant
    to the signer's distance from the camera and position within the frame:

      1. Translate so landmark 0 (the wrist) is the origin.
      2. Scale by the distance from the wrist to the middle-finger MCP
         (landmark 9), a stable reference length that scales with hand size.

    *landmarks* is a list of mediapipe NormalizedLandmark objects (from the
    Tasks API), each with .x, .y, .z attributes — same as the legacy API.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    wrist = pts[0]
    centered = pts - wrist

    ref_length = np.linalg.norm(centered[9])
    if ref_length < 1e-6:
        ref_length = 1e-6

    normalized = centered / ref_length
    return normalized.round(5).tolist()


# ── Drawing helper ──────────────────────────────────────────────────────────

def draw_hand_landmarks(frame, landmarks, w, h):
    """Draw landmarks and connections on *frame* given normalized landmarks."""
    # Convert normalized coords to pixel positions
    pts_px = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        pts_px.append((px, py))

    # Draw connections
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, pts_px[i], pts_px[j], _CONN_COLOR, _CONN_THICKNESS)

    # Draw landmark dots on top of lines
    for px, py in pts_px:
        cv2.circle(frame, (px, py), _LM_RADIUS, _LM_COLOR, -1)


# ── Main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    # Verify model file is present
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {MODEL_PATH.resolve()}.\n"
            "Download it with:\n"
            "  curl -L -O https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Check that a webcam is connected and not in use by another app."
        )

    # Create the HandLandmarker (IMAGE mode — we feed one frame at a time)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=TRACKING_CONFIDENCE,
    )
    landmarker = HandLandmarker.create_from_options(options)

    frame_count = 0
    log_file = LOG_PATH.open("a", encoding="utf-8")

    print("Module 1 — Real-Time Sign Capture and Preprocessing")
    print(f"MediaPipe version: {mp.__version__} (Tasks API)")
    print(f"Logging normalized landmarks to: {LOG_PATH.resolve()}")
    print("Press 'q' to quit, 's' to save a snapshot.\n")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed — exiting.")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe Tasks expects an mp.Image wrapping an RGB numpy array
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = landmarker.detect(mp_image)

            hands_payload = []

            if results.hand_landmarks:
                for idx, hand_lms in enumerate(results.hand_landmarks):
                    # Draw the skeleton overlay
                    draw_hand_landmarks(frame, hand_lms, w, h)

                    # Normalize
                    normalized = normalize_landmarks(hand_lms)

                    # Determine handedness label
                    if results.handedness and idx < len(results.handedness):
                        h_label = results.handedness[idx][0].category_name
                        h_score = round(results.handedness[idx][0].score, 3)
                    else:
                        h_label = "Unknown"
                        h_score = 0.0

                    hands_payload.append(
                        {
                            "handedness": h_label,
                            "confidence": h_score,
                            "landmarks": normalized,
                        }
                    )

            # Log every N-th frame when hands are detected
            if hands_payload and frame_count % LOG_EVERY_N_FRAMES == 0:
                record = {
                    "frame": frame_count,
                    "timestamp": time.time(),
                    "num_hands": len(hands_payload),
                    "hands": hands_payload,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

            # HUD text
            status = f"Hands detected: {len(hands_payload)}"
            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, "q: quit   s: save snapshot", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            cv2.imshow("Module 1 - Sign Capture and Preprocessing", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # 5-second countdown before capturing the snapshot
                for sec in range(5, 0, -1):
                    ok2, cframe = cap.read()
                    if ok2:
                        cframe = cv2.flip(cframe, 1)
                        ch, cw = cframe.shape[:2]
                        rgb2 = cv2.cvtColor(cframe, cv2.COLOR_BGR2RGB)
                        mp2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2)
                        res2 = landmarker.detect(mp2)
                        if res2.hand_landmarks:
                            for hand_lms in res2.hand_landmarks:
                                draw_hand_landmarks(cframe, hand_lms, cw, ch)
                        cv2.putText(
                            cframe, f"Snapshot in {sec}...", (cw // 2 - 140, ch // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                        )
                        cv2.imshow("Module 1 - Sign Capture and Preprocessing", cframe)
                    cv2.waitKey(1000)

                # Capture the actual snapshot frame
                ok3, snap_frame = cap.read()
                if ok3:
                    snap_frame = cv2.flip(snap_frame, 1)
                    sh, sw = snap_frame.shape[:2]
                    rgb3 = cv2.cvtColor(snap_frame, cv2.COLOR_BGR2RGB)
                    mp3 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb3)
                    res3 = landmarker.detect(mp3)
                    if res3.hand_landmarks:
                        for hand_lms in res3.hand_landmarks:
                            draw_hand_landmarks(snap_frame, hand_lms, sw, sh)
                    snap_path = SCRIPT_DIR / f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(str(snap_path), snap_frame)
                    print(f"Saved {snap_path}")

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"\nDone. Landmark log written to {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
