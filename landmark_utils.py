"""
Landmark Utilities — Shared MediaPipe + Normalization Helpers
==============================================================

This module extracts the core normalization and detection logic from Module 1
so that Modules 2 and 3 can reuse the exact same pipeline.

Functions:
    normalize_landmarks     — wrist-centered, MCP9-scaled normalization
    create_hand_landmarker  — create a reusable MediaPipe HandLandmarker
    extract_landmarks_from_frame — detect + normalize one RGB frame

Usage:
    from landmark_utils import (
        normalize_landmarks,
        create_hand_landmarker,
        extract_landmarks_from_frame,
    )
"""

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from config import MODEL_PATH, MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE


# ── Normalization (identical to Module 1) ──────────────────────────────────

def normalize_landmarks(landmarks) -> list[list[float]]:
    """
    Convert 21 raw (x, y, z) landmarks into a representation that is invariant
    to the signer's distance from the camera and position within the frame.

    Steps:
        1. Translate so landmark 0 (the wrist) is the origin.
        2. Scale by the Euclidean distance from the wrist to landmark 9
           (middle-finger MCP), a stable reference length proportional to
           hand size.

    Parameters
    ----------
    landmarks : list of mediapipe NormalizedLandmark objects
        Each landmark has .x, .y, .z attributes.

    Returns
    -------
    list[list[float]]
        21 normalized [x, y, z] coordinates, rounded to 5 decimal places.
        Landmark 0 is always [0, 0, 0].
        Landmark 9 has unit distance from origin.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Step 1: Wrist-centering
    wrist = pts[0]
    centered = pts - wrist

    # Step 2: Scale by wrist-to-MCP9 distance
    ref_length = np.linalg.norm(centered[9])
    if ref_length < 1e-6:
        ref_length = 1e-6  # Guard against degenerate detections

    normalized = centered / ref_length
    return normalized.round(5).tolist()


def normalize_landmarks_array(landmarks) -> np.ndarray:
    """
    Same normalization as normalize_landmarks(), but accepts and returns
    numpy arrays directly.  Used for batch processing in Modules 2 & 3.

    Parameters
    ----------
    landmarks : list of mediapipe NormalizedLandmark objects

    Returns
    -------
    np.ndarray
        Shape (21, 3), float64, rounded to 5 decimal places.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    wrist = pts[0]
    centered = pts - wrist

    ref_length = np.linalg.norm(centered[9])
    if ref_length < 1e-6:
        ref_length = 1e-6

    normalized = centered / ref_length
    return normalized.round(5)


# ── MediaPipe HandLandmarker Factory ───────────────────────────────────────

def create_hand_landmarker(model_path=None) -> HandLandmarker:
    """
    Create a MediaPipe HandLandmarker instance using the Tasks API.

    Uses the same settings as Module 1:
      - IMAGE running mode (process one frame at a time)
      - CPU delegate (no GPU required)
      - Detection/tracking confidence thresholds from config

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the hand_landmarker.task model file.
        Defaults to MODEL_PATH from config.py.

    Returns
    -------
    HandLandmarker
        Ready-to-use landmarker instance.  Caller must close it when done
        (use as a context manager or call .close()).

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if model_path is None:
        model_path = MODEL_PATH

    model_path_str = str(model_path)

    from pathlib import Path
    if not Path(model_path_str).exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path_str}.\n"
            "Download it with:\n"
            "  curl -L -O https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/"
            "hand_landmarker.task"
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path_str,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=TRACKING_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)


# ── Single-Frame Extraction ───────────────────────────────────────────────

def extract_landmarks_from_frame(landmarker, rgb_frame):
    """
    Run hand detection + normalization on a single RGB frame.

    Parameters
    ----------
    landmarker : HandLandmarker
        A MediaPipe HandLandmarker created via create_hand_landmarker().
    rgb_frame : np.ndarray
        RGB image as a numpy array (H, W, 3), dtype uint8.

    Returns
    -------
    list[dict]
        One dict per detected hand, each with keys:
          - "handedness" : str ("Right" or "Left")
          - "confidence" : float (detection score, 3 decimals)
          - "landmarks"  : list[list[float]] (21 × [x, y, z], normalized)
          - "landmarks_array" : np.ndarray (21, 3) (same data as array)
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(mp_image)

    hands = []
    if results.hand_landmarks:
        for idx, hand_lms in enumerate(results.hand_landmarks):
            # Determine handedness
            if results.handedness and idx < len(results.handedness):
                h_label = results.handedness[idx][0].category_name
                h_score = round(results.handedness[idx][0].score, 3)
            else:
                h_label = "Unknown"
                h_score = 0.0

            # Normalize
            normalized_list = normalize_landmarks(hand_lms)
            normalized_array = np.array(normalized_list)

            hands.append({
                "handedness": h_label,
                "confidence": h_score,
                "landmarks": normalized_list,
                "landmarks_array": normalized_array,
            })

    return hands
